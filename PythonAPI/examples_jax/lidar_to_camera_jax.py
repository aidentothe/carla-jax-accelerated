#!/usr/bin/env python

"""JAX-accelerated LiDAR to camera projection for CARLA.

This example demonstrates how to use JAX for high-performance 3D to 2D
projection with JIT compilation, vectorized operations, and efficient
geometric transformations for real-time sensor fusion applications.
"""

import glob
import os
import sys
import time
import argparse
from queue import Queue, Empty
from typing import Tuple, Dict, Optional

try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

try:
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, lax
    from jax.typing import ArrayLike
    import chex
    from functools import partial
except ImportError:
    raise RuntimeError('JAX not found. Install with: pip install jax jaxlib chex')

try:
    from PIL import Image
    import matplotlib.pyplot as plt
    from matplotlib import cm
except ImportError:
    raise RuntimeError('PIL or matplotlib not found. Install with: pip install Pillow matplotlib')


# ==============================================================================
# -- JAX Geometric Transformation Functions ----------------------------------
# ==============================================================================

@jit
def create_projection_matrix(focal_length: float, 
                           image_width: int, 
                           image_height: int) -> jnp.ndarray:
    """Create camera projection matrix K using JAX."""
    K = jnp.eye(3)
    K = K.at[0, 0].set(focal_length)
    K = K.at[1, 1].set(focal_length)
    K = K.at[0, 2].set(image_width / 2.0)
    K = K.at[1, 2].set(image_height / 2.0)
    return K


@jit
def homogeneous_transform(points_3d: jnp.ndarray) -> jnp.ndarray:
    """Convert 3D points to homogeneous coordinates."""
    # points_3d shape: (N, 3) -> output shape: (N, 4)
    ones = jnp.ones((points_3d.shape[0], 1))
    return jnp.concatenate([points_3d, ones], axis=1)


@jit
def apply_transform_matrix(points_homogeneous: jnp.ndarray, 
                          transform_matrix: jnp.ndarray) -> jnp.ndarray:
    """Apply 4x4 transformation matrix to homogeneous points."""
    # points_homogeneous shape: (N, 4), transform_matrix: (4, 4)
    # Result: (N, 4)
    return jnp.dot(points_homogeneous, transform_matrix.T)


@jit
def coordinate_system_conversion(sensor_points: jnp.ndarray) -> jnp.ndarray:
    """Convert from UE4 coordinate system to standard camera coordinates.
    
    UE4: x=forward, y=right, z=up
    Camera: x=right, y=down, z=forward
    Conversion: (x, y, z) -> (y, -z, x)
    """
    # sensor_points shape: (N, 3 or 4)
    converted = jnp.zeros_like(sensor_points)
    converted = converted.at[:, 0].set(sensor_points[:, 1])  # y -> x
    converted = converted.at[:, 1].set(-sensor_points[:, 2])  # -z -> y
    converted = converted.at[:, 2].set(sensor_points[:, 0])   # x -> z
    
    if sensor_points.shape[1] == 4:  # Preserve homogeneous coordinate
        converted = converted.at[:, 3].set(sensor_points[:, 3])
    
    return converted


@jit
def project_3d_to_2d(points_3d_camera: jnp.ndarray, 
                    K_matrix: jnp.ndarray) -> jnp.ndarray:
    """Project 3D points in camera coordinates to 2D image coordinates."""
    # points_3d_camera shape: (N, 3), K_matrix: (3, 3)
    # Result: (N, 3) with [u, v, depth]
    
    # Project points: K @ points_3d_camera.T -> (3, N)
    projected_homogeneous = jnp.dot(K_matrix, points_3d_camera.T)
    
    # Normalize by depth (z-coordinate)
    depth = projected_homogeneous[2, :]
    
    # Avoid division by zero
    safe_depth = jnp.where(jnp.abs(depth) < 1e-6, 1e-6, depth)
    
    u = projected_homogeneous[0, :] / safe_depth
    v = projected_homogeneous[1, :] / safe_depth
    
    # Return as (N, 3) array: [u, v, depth]
    return jnp.stack([u, v, depth], axis=1)


@jit
def filter_points_in_image(points_2d: jnp.ndarray,
                          image_width: int,
                          image_height: int,
                          min_depth: float = 0.1) -> jnp.ndarray:
    """Filter points that are within image boundaries and in front of camera."""
    u = points_2d[:, 0]
    v = points_2d[:, 1]
    depth = points_2d[:, 2]
    
    # Create mask for valid points
    in_image_mask = (
        (u >= 0) & (u < image_width) &
        (v >= 0) & (v < image_height) &
        (depth > min_depth)
    )
    
    return in_image_mask


@jit
def apply_colormap_viridis(values: jnp.ndarray, 
                          vmin: float = 0.0, 
                          vmax: float = 1.0) -> jnp.ndarray:
    """Apply viridis colormap to values using JAX."""
    # Normalize values to [0, 1]
    normalized = (values - vmin) / (vmax - vmin)
    normalized = jnp.clip(normalized, 0.0, 1.0)
    
    # Simplified viridis colormap approximation
    # These coefficients approximate the viridis colormap
    r = jnp.where(normalized < 0.5,
                  0.267 * normalized + 0.004,
                  0.973 - 0.267 * (1 - normalized))
    
    g = jnp.where(normalized < 0.25,
                  normalized * 0.282,
                  jnp.where(normalized < 0.75,
                           0.282 + (normalized - 0.25) * 1.472,
                           0.650 + (normalized - 0.75) * 0.348))
    
    b = jnp.where(normalized < 0.7,
                  0.146 + normalized * 0.572,
                  0.545 + (normalized - 0.7) * 0.455)
    
    # Convert to 0-255 range
    rgb = jnp.stack([r, g, b], axis=1) * 255.0
    return rgb.astype(jnp.uint8)


# ==============================================================================
# -- JAX Complete Projection Pipeline -----------------------------------------
# ==============================================================================

@jit
def complete_lidar_to_camera_projection(
    lidar_points: jnp.ndarray,           # (N, 3) LiDAR points in local coordinates
    lidar_intensity: jnp.ndarray,        # (N,) intensity values
    lidar_to_world_matrix: jnp.ndarray,  # (4, 4) transformation matrix
    world_to_camera_matrix: jnp.ndarray, # (4, 4) transformation matrix
    camera_K_matrix: jnp.ndarray,        # (3, 3) intrinsic matrix
    image_width: int,
    image_height: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Complete JAX-compiled LiDAR to camera projection pipeline."""
    
    # Convert to homogeneous coordinates
    lidar_homogeneous = homogeneous_transform(lidar_points)
    
    # Transform to world coordinates
    world_points = apply_transform_matrix(lidar_homogeneous, lidar_to_world_matrix)
    
    # Transform to camera coordinates
    camera_points = apply_transform_matrix(world_points, world_to_camera_matrix)
    
    # Convert coordinate systems (UE4 -> standard camera)
    camera_coords_standard = coordinate_system_conversion(camera_points[:, :3])
    
    # Project to 2D
    points_2d = project_3d_to_2d(camera_coords_standard, camera_K_matrix)
    
    # Filter points in image
    valid_mask = filter_points_in_image(points_2d, image_width, image_height)
    
    # Apply mask to get valid points
    valid_points_2d = points_2d[valid_mask]
    valid_intensity = lidar_intensity[valid_mask]
    
    # Get integer pixel coordinates
    u_coords = valid_points_2d[:, 0].astype(jnp.int32)
    v_coords = valid_points_2d[:, 1].astype(jnp.int32)
    
    return valid_points_2d, valid_intensity, u_coords, v_coords


# ==============================================================================
# -- Vectorized Batch Processing ----------------------------------------------
# ==============================================================================

@jit
def batch_project_multiple_frames(
    lidar_points_batch: jnp.ndarray,      # (B, N, 3) multiple frames
    lidar_intensity_batch: jnp.ndarray,   # (B, N) intensity for each frame
    lidar_transforms_batch: jnp.ndarray,  # (B, 4, 4) transform matrices
    camera_transforms_batch: jnp.ndarray, # (B, 4, 4) camera matrices
    camera_K: jnp.ndarray,                # (3, 3) camera intrinsics
    image_width: int,
    image_height: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Vectorized projection for multiple frames using vmap."""
    
    # Create vectorized version of the projection function
    vectorized_projection = vmap(
        complete_lidar_to_camera_projection,
        in_axes=(0, 0, 0, 0, None, None, None)
    )
    
    return vectorized_projection(
        lidar_points_batch,
        lidar_intensity_batch,
        lidar_transforms_batch,
        camera_transforms_batch,
        camera_K,
        image_width,
        image_height
    )


# ==============================================================================
# -- JAX Sensor Fusion Manager ------------------------------------------------
# ==============================================================================

class JAXLidarCameraFusion:
    """JAX-accelerated LiDAR-camera fusion with real-time processing."""
    
    def __init__(self, image_width: int, image_height: int, camera_fov: float):
        self.image_width = image_width
        self.image_height = image_height
        self.camera_fov = camera_fov
        
        # Pre-compute camera intrinsics
        focal_length = image_width / (2.0 * np.tan(np.radians(camera_fov) / 2.0))
        self.K_matrix = create_projection_matrix(focal_length, image_width, image_height)
        
        # Pre-compile JAX functions
        self.project_lidar = jit(complete_lidar_to_camera_projection)
        self.apply_colormap = jit(apply_colormap_viridis)
        
        # Performance monitoring
        self.processing_times = []
        
        print(f"JAX LiDAR-Camera Fusion initialized:")
        print(f"  Image size: {image_width}x{image_height}")
        print(f"  Camera FOV: {camera_fov}°")
        print(f"  Focal length: {focal_length:.1f}")
        print(f"  JAX devices: {jax.devices()}")
    
    def process_frame(self,
                     lidar_data: carla.LidarMeasurement,
                     camera_data: carla.Image,
                     lidar_transform: carla.Transform,
                     camera_transform: carla.Transform) -> Tuple[np.ndarray, Dict[str, float]]:
        """Process a single frame with JAX acceleration."""
        
        start_time = time.time()
        
        # Convert CARLA data to JAX arrays
        lidar_points, lidar_intensity = self._carla_lidar_to_jax(lidar_data)
        camera_image = self._carla_image_to_jax(camera_data)
        
        # Convert transforms to JAX matrices
        lidar_to_world = jnp.array(lidar_transform.get_matrix())
        world_to_camera = jnp.array(camera_transform.get_inverse_matrix())
        
        conversion_time = time.time() - start_time
        
        # Perform projection with JAX
        projection_start = time.time()
        valid_points_2d, valid_intensity, u_coords, v_coords = self.project_lidar(
            lidar_points, lidar_intensity, lidar_to_world, world_to_camera,
            self.K_matrix, self.image_width, self.image_height
        )
        projection_time = time.time() - projection_start
        
        # Create overlay image
        overlay_start = time.time()
        overlay_image = self._create_overlay_image(
            camera_image, valid_intensity, u_coords, v_coords
        )
        overlay_time = time.time() - overlay_start
        
        total_time = time.time() - start_time
        
        # Performance stats
        stats = {
            'total_time_ms': total_time * 1000,
            'conversion_time_ms': conversion_time * 1000,
            'projection_time_ms': projection_time * 1000,
            'overlay_time_ms': overlay_time * 1000,
            'num_lidar_points': len(lidar_points),
            'num_valid_points': len(valid_points_2d),
            'projection_ratio': len(valid_points_2d) / len(lidar_points) if len(lidar_points) > 0 else 0
        }
        
        self.processing_times.append(total_time)
        
        return np.array(overlay_image), stats
    
    def _carla_lidar_to_jax(self, lidar_data: carla.LidarMeasurement) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Convert CARLA LiDAR data to JAX arrays."""
        # Convert raw data to numpy
        lidar_np = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
        lidar_np = lidar_np.reshape((-1, 4))
        
        # Extract points and intensity
        points = jnp.array(lidar_np[:, :3])
        intensity = jnp.array(lidar_np[:, 3])
        
        return points, intensity
    
    def _carla_image_to_jax(self, camera_data: carla.Image) -> jnp.ndarray:
        """Convert CARLA camera image to JAX array."""
        # Convert to numpy array
        image_np = np.frombuffer(camera_data.raw_data, dtype=np.uint8)
        image_np = image_np.reshape((camera_data.height, camera_data.width, 4))
        image_rgb = image_np[:, :, :3][:, :, ::-1]  # BGRA to RGB
        
        return jnp.array(image_rgb)
    
    def _create_overlay_image(self,
                            camera_image: jnp.ndarray,
                            intensity: jnp.ndarray,
                            u_coords: jnp.ndarray,
                            v_coords: jnp.ndarray,
                            dot_size: int = 2) -> jnp.ndarray:
        """Create overlay image with LiDAR points projected onto camera image."""
        
        # Start with camera image
        overlay = camera_image.copy()
        
        if len(intensity) == 0:
            return overlay
        
        # Normalize intensity for colormap
        intensity_normalized = (intensity - jnp.min(intensity)) / (jnp.max(intensity) - jnp.min(intensity) + 1e-6)
        
        # Apply colormap
        colors = self.apply_colormap(intensity_normalized)
        
        # Draw points on image (simplified - could be optimized further)
        # For now, we'll use numpy for the final drawing step
        overlay_np = np.array(overlay)
        u_coords_np = np.array(u_coords)
        v_coords_np = np.array(v_coords)
        colors_np = np.array(colors)
        
        # Ensure coordinates are within bounds
        valid_indices = (
            (u_coords_np >= dot_size) & (u_coords_np < self.image_width - dot_size) &
            (v_coords_np >= dot_size) & (v_coords_np < self.image_height - dot_size)
        )
        
        u_valid = u_coords_np[valid_indices]
        v_valid = v_coords_np[valid_indices]
        colors_valid = colors_np[valid_indices]
        
        # Draw points with specified dot size
        for i in range(len(u_valid)):
            u, v = u_valid[i], v_valid[i]
            color = colors_valid[i]
            
            overlay_np[v-dot_size:v+dot_size+1, u-dot_size:u+dot_size+1] = color
        
        return jnp.array(overlay_np)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get processing performance statistics."""
        if not self.processing_times:
            return {}
        
        times = np.array(self.processing_times)
        return {
            'avg_time_ms': np.mean(times) * 1000,
            'max_time_ms': np.max(times) * 1000,
            'min_time_ms': np.min(times) * 1000,
            'std_time_ms': np.std(times) * 1000,
            'fps': 1.0 / np.mean(times) if np.mean(times) > 0 else 0
        }


# ==============================================================================
# -- CARLA Integration and Main Function --------------------------------------
# ==============================================================================

def sensor_callback(data, queue):
    """Thread-safe sensor callback."""
    queue.put(data)


def jax_tutorial(args):
    """JAX-accelerated LiDAR to camera projection tutorial."""
    
    print("Starting JAX-accelerated LiDAR-Camera projection...")
    print(f"JAX version: {jax.__version__}")
    print(f"Available JAX devices: {jax.devices()}")
    
    # Initialize JAX fusion processor
    fusion_processor = JAXLidarCameraFusion(
        image_width=args.width,
        image_height=args.height,
        camera_fov=90.0  # Default FOV
    )
    
    # Connect to CARLA
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()
    
    # Setup synchronous mode
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1  # 10 FPS
    world.apply_settings(settings)
    
    vehicle = None
    camera = None
    lidar = None
    
    try:
        # Create output directory
        if not os.path.isdir('_out_jax'):
            os.mkdir('_out_jax')
        
        # Get blueprints
        vehicle_bp = bp_lib.filter("vehicle.lincoln.mkz_2017")[0]
        camera_bp = bp_lib.filter("sensor.camera.rgb")[0]
        lidar_bp = bp_lib.filter("sensor.lidar.ray_cast")[0]
        
        # Configure camera
        camera_bp.set_attribute("image_size_x", str(args.width))
        camera_bp.set_attribute("image_size_y", str(args.height))
        camera_bp.set_attribute("fov", "90")
        
        # Configure LiDAR
        if args.no_noise:
            lidar_bp.set_attribute('dropoff_general_rate', '0.0')
            lidar_bp.set_attribute('dropoff_intensity_limit', '1.0')
            lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')
        
        lidar_bp.set_attribute('upper_fov', str(args.upper_fov))
        lidar_bp.set_attribute('lower_fov', str(args.lower_fov))
        lidar_bp.set_attribute('channels', str(args.channels))
        lidar_bp.set_attribute('range', str(args.range))
        lidar_bp.set_attribute('points_per_second', str(args.points_per_second))
        
        # Spawn actors
        spawn_points = world.get_map().get_spawn_points()
        vehicle = world.spawn_actor(blueprint=vehicle_bp, transform=spawn_points[0])
        vehicle.set_autopilot(True)
        
        camera = world.spawn_actor(
            blueprint=camera_bp,
            transform=carla.Transform(carla.Location(x=1.6, z=1.6)),
            attach_to=vehicle
        )
        
        lidar = world.spawn_actor(
            blueprint=lidar_bp,
            transform=carla.Transform(carla.Location(x=1.0, z=1.8)),
            attach_to=vehicle
        )
        
        # Setup sensor queues
        image_queue = Queue()
        lidar_queue = Queue()
        
        camera.listen(lambda data: sensor_callback(data, image_queue))
        lidar.listen(lambda data: sensor_callback(data, lidar_queue))
        
        print(f"Processing {args.frames} frames with JAX acceleration...")
        
        all_stats = []
        
        for frame in range(args.frames):
            world.tick()
            world_frame = world.get_snapshot().frame
            
            try:
                # Get synchronized sensor data
                image_data = image_queue.get(True, 2.0)
                lidar_data = lidar_queue.get(True, 2.0)
            except Empty:
                print(f"[Warning] Frame {frame}: Some sensor data missed")
                continue
            
            # Verify synchronization
            if image_data.frame != lidar_data.frame or image_data.frame != world_frame:
                print(f"[Warning] Frame {frame}: Sensor synchronization mismatch")
                continue
            
            # Process frame with JAX
            overlay_image, stats = fusion_processor.process_frame(
                lidar_data, image_data,
                lidar.get_transform(), camera.get_transform()
            )
            
            # Save result
            image = Image.fromarray(overlay_image)
            image.save(f"_out_jax/{image_data.frame:08d}_jax.png")
            
            all_stats.append(stats)
            
            # Progress and performance reporting
            sys.stdout.write(f"\r({frame+1}/{args.frames}) Frame: {world_frame}, "
                           f"JAX time: {stats['total_time_ms']:.2f}ms, "
                           f"Points: {stats['num_valid_points']}/{stats['num_lidar_points']}")
            sys.stdout.flush()
        
        print("\n")
        
        # Final performance summary
        if all_stats:
            avg_total = np.mean([s['total_time_ms'] for s in all_stats])
            avg_projection = np.mean([s['projection_time_ms'] for s in all_stats])
            avg_points = np.mean([s['num_valid_points'] for s in all_stats])
            avg_ratio = np.mean([s['projection_ratio'] for s in all_stats])
            
            print("JAX Processing Performance Summary:")
            print(f"  Average total time: {avg_total:.2f}ms")
            print(f"  Average projection time: {avg_projection:.2f}ms")
            print(f"  Average valid points: {avg_points:.1f}")
            print(f"  Average projection ratio: {avg_ratio:.3f}")
            print(f"  Effective FPS: {1000.0/avg_total:.1f}")
            
            final_stats = fusion_processor.get_performance_stats()
            print(f"  Overall performance: {final_stats['fps']:.1f} FPS")
    
    finally:
        # Cleanup
        world.apply_settings(original_settings)
        
        if camera and camera.is_alive:
            camera.destroy()
        if lidar and lidar.is_alive:
            lidar.destroy()
        if vehicle and vehicle.is_alive:
            vehicle.destroy()
        
        print("Cleanup completed.")


def main():
    """Main function with argument parsing."""
    
    argparser = argparse.ArgumentParser(
        description='JAX-accelerated CARLA LiDAR to Camera Projection')
    
    argparser.add_argument('--host', default='127.0.0.1', help='IP of CARLA server')
    argparser.add_argument('-p', '--port', default=2000, type=int, help='TCP port')
    argparser.add_argument('--width', default=800, type=int, help='Image width')
    argparser.add_argument('--height', default=600, type=int, help='Image height')
    argparser.add_argument('--frames', default=100, type=int, help='Number of frames')
    
    # LiDAR configuration
    argparser.add_argument('--no-noise', action='store_true', help='Disable LiDAR noise')
    argparser.add_argument('--upper-fov', default=15.0, type=float, help='Upper FOV')
    argparser.add_argument('--lower-fov', default=-25.0, type=float, help='Lower FOV')
    argparser.add_argument('--channels', default=32, type=int, help='LiDAR channels')
    argparser.add_argument('--range', default=50.0, type=float, help='LiDAR range')
    argparser.add_argument('--points-per-second', default=100000, type=int, 
                          help='Points per second')
    
    # Visualization
    argparser.add_argument('--dot-extent', default=2, type=int, 
                          help='Dot size for LiDAR points')
    
    args = argparser.parse_args()
    
    try:
        jax_tutorial(args)
    except KeyboardInterrupt:
        print('\n - Exited by user.')
    except Exception as e:
        print(f'\n - Error: {e}')
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()