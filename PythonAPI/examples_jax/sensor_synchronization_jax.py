#!/usr/bin/env python

"""JAX-accelerated sensor synchronization for CARLA.

This example demonstrates how to use JAX for high-performance multi-sensor
data processing with batched operations, JIT compilation, and efficient
memory management for real-time autonomous driving applications.
"""

import glob
import os
import sys
import time
import collections
from queue import Queue, Empty
from typing import Dict, List, Tuple, Optional

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
    import chex
    from typing import NamedTuple
except ImportError:
    raise RuntimeError('JAX not found. Install with: pip install jax jaxlib chex')


# ==============================================================================
# -- JAX Data Structures ------------------------------------------------------
# ==============================================================================

class SensorData(NamedTuple):
    """JAX-compatible sensor data structure."""
    frame: int
    timestamp: float
    data: jnp.ndarray
    transform: jnp.ndarray  # [x, y, z, roll, pitch, yaw]
    sensor_type: int  # 0=camera, 1=lidar, 2=radar


class SensorFusion(NamedTuple):
    """Multi-sensor fusion state."""
    camera_data: jnp.ndarray
    lidar_points: jnp.ndarray
    radar_detections: jnp.ndarray
    transforms: jnp.ndarray
    timestamps: jnp.ndarray


# ==============================================================================
# -- JAX Sensor Processing Functions ------------------------------------------
# ==============================================================================

@jit
def process_camera_data(rgb_array: jnp.ndarray) -> jnp.ndarray:
    """JAX-compiled camera image processing."""
    # Convert to float and normalize
    normalized = rgb_array.astype(jnp.float32) / 255.0
    
    # Apply gamma correction
    gamma_corrected = jnp.power(normalized, 1.0 / 2.2)
    
    # Extract features (simplified edge detection)
    sobel_x = jnp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = jnp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    # Apply convolution to grayscale
    gray = jnp.mean(gamma_corrected, axis=-1)
    edges_x = lax.conv_general_dilated(
        gray[None, None, :, :], sobel_x[None, None, :, :],
        window_strides=[1, 1], padding='SAME'
    )[0, 0]
    edges_y = lax.conv_general_dilated(
        gray[None, None, :, :], sobel_y[None, None, :, :],
        window_strides=[1, 1], padding='SAME'
    )[0, 0]
    
    edge_magnitude = jnp.sqrt(edges_x**2 + edges_y**2)
    
    return edge_magnitude


@jit
def process_lidar_points(points: jnp.ndarray, transform: jnp.ndarray) -> jnp.ndarray:
    """JAX-compiled LiDAR point cloud processing."""
    # points shape: (N, 3) - x, y, z coordinates
    # transform shape: (6,) - x, y, z, roll, pitch, yaw
    
    # Extract translation and rotation
    translation = transform[:3]
    rotation = transform[3:]
    
    # Create rotation matrix from Euler angles
    cos_r, sin_r = jnp.cos(rotation[0]), jnp.sin(rotation[0])
    cos_p, sin_p = jnp.cos(rotation[1]), jnp.sin(rotation[1])
    cos_y, sin_y = jnp.cos(rotation[2]), jnp.sin(rotation[2])
    
    # Roll matrix
    R_x = jnp.array([[1, 0, 0],
                     [0, cos_r, -sin_r],
                     [0, sin_r, cos_r]])
    
    # Pitch matrix
    R_y = jnp.array([[cos_p, 0, sin_p],
                     [0, 1, 0],
                     [-sin_p, 0, cos_p]])
    
    # Yaw matrix
    R_z = jnp.array([[cos_y, -sin_y, 0],
                     [sin_y, cos_y, 0],
                     [0, 0, 1]])
    
    # Combined rotation matrix
    R = R_z @ R_y @ R_x
    
    # Transform points to world coordinates
    transformed_points = (R @ points.T).T + translation
    
    # Filter points by distance and height
    distances = jnp.linalg.norm(transformed_points[:, :2], axis=1)
    height_filter = (transformed_points[:, 2] > -2.0) & (transformed_points[:, 2] < 10.0)
    distance_filter = (distances > 1.0) & (distances < 100.0)
    
    valid_points = transformed_points[height_filter & distance_filter]
    
    return valid_points


@jit
def process_radar_detections(detections: jnp.ndarray) -> jnp.ndarray:
    """JAX-compiled radar detection processing."""
    # detections shape: (N, 4) - [velocity, azimuth, altitude, depth]
    
    # Convert polar to Cartesian coordinates
    azimuth = detections[:, 1]
    altitude = detections[:, 2]
    depth = detections[:, 3]
    
    x = depth * jnp.cos(altitude) * jnp.cos(azimuth)
    y = depth * jnp.cos(altitude) * jnp.sin(azimuth)
    z = depth * jnp.sin(altitude)
    
    cartesian_points = jnp.stack([x, y, z], axis=1)
    
    # Filter by distance and velocity
    distances = jnp.linalg.norm(cartesian_points, axis=1)
    velocities = detections[:, 0]
    
    distance_filter = distances < 100.0
    velocity_filter = jnp.abs(velocities) > 0.1  # Moving objects
    
    valid_detections = cartesian_points[distance_filter & velocity_filter]
    
    return valid_detections


# ==============================================================================
# -- JAX Sensor Fusion --------------------------------------------------------
# ==============================================================================

@jit
def fuse_sensor_data(camera_features: jnp.ndarray,
                    lidar_points: jnp.ndarray,
                    radar_detections: jnp.ndarray,
                    camera_transform: jnp.ndarray,
                    timestamp: float) -> jnp.ndarray:
    """JAX-compiled multi-sensor fusion algorithm."""
    
    # Project LiDAR points to camera image plane
    # Simplified projection (assumes calibrated camera)
    focal_length = 500.0
    image_width, image_height = 800, 600
    
    # Transform LiDAR points to camera frame
    relative_transform = camera_transform[:3]  # Simplified
    camera_points = lidar_points - relative_transform
    
    # Project to image plane
    valid_mask = camera_points[:, 2] > 0.1  # Points in front of camera
    valid_points = camera_points[valid_mask]
    
    if len(valid_points) > 0:
        projected_x = (valid_points[:, 0] / valid_points[:, 2]) * focal_length + image_width / 2
        projected_y = (valid_points[:, 1] / valid_points[:, 2]) * focal_length + image_height / 2
        
        # Check bounds
        in_bounds = ((projected_x >= 0) & (projected_x < image_width) &
                    (projected_y >= 0) & (projected_y < image_height))
        
        # Create depth map from LiDAR
        depth_map = jnp.zeros((image_height, image_width))
        if jnp.sum(in_bounds) > 0:
            valid_proj_x = projected_x[in_bounds].astype(jnp.int32)
            valid_proj_y = projected_y[in_bounds].astype(jnp.int32)
            valid_depths = valid_points[in_bounds, 2]
            
            # Simple depth assignment (could be improved with proper aggregation)
            depth_map = depth_map.at[valid_proj_y, valid_proj_x].set(valid_depths)
    else:
        depth_map = jnp.zeros((image_height, image_width))
    
    # Combine camera features with depth information
    if camera_features.shape[:2] == depth_map.shape:
        fused_features = jnp.stack([camera_features, depth_map], axis=-1)
    else:
        # Resize depth map to match camera features
        scale_y = camera_features.shape[0] / depth_map.shape[0]
        scale_x = camera_features.shape[1] / depth_map.shape[1]
        # Simplified resize (for full implementation, use proper interpolation)
        resized_depth = jnp.mean(depth_map) * jnp.ones_like(camera_features)
        fused_features = jnp.stack([camera_features, resized_depth], axis=-1)
    
    # Add radar information as global context (simplified)
    num_radar_detections = len(radar_detections)
    radar_density = num_radar_detections / 1000.0  # Normalize
    
    # Create output feature vector
    fusion_result = jnp.concatenate([
        fused_features.flatten(),
        jnp.array([radar_density, timestamp])
    ])
    
    return fusion_result


# ==============================================================================
# -- Vectorized Batch Processing ----------------------------------------------
# ==============================================================================

@jit
def batch_process_cameras(camera_batch: jnp.ndarray) -> jnp.ndarray:
    """Process multiple camera frames in parallel using vmap."""
    return vmap(process_camera_data)(camera_batch)


@jit
def batch_process_lidars(lidar_batch: jnp.ndarray, 
                        transform_batch: jnp.ndarray) -> jnp.ndarray:
    """Process multiple LiDAR scans in parallel using vmap."""
    return vmap(process_lidar_points)(lidar_batch, transform_batch)


# ==============================================================================
# -- JAX Sensor Manager -------------------------------------------------------
# ==============================================================================

class JAXSensorManager:
    """JAX-accelerated sensor data manager with batched processing."""
    
    def __init__(self, max_batch_size: int = 10):
        self.max_batch_size = max_batch_size
        self.sensor_data_buffer = collections.defaultdict(list)
        self.processing_times = collections.deque(maxlen=100)
        
        # Pre-compile functions
        self.process_camera = jit(process_camera_data)
        self.process_lidar = jit(process_lidar_points)
        self.process_radar = jit(process_radar_detections)
        self.fuse_sensors = jit(fuse_sensor_data)
        
        # Batch processing functions
        self.batch_cameras = jit(batch_process_cameras)
        self.batch_lidars = jit(batch_process_lidars)
        
        print(f"JAX Sensor Manager initialized. Devices: {jax.devices()}")
    
    def add_sensor_data(self, sensor_name: str, frame: int, data: np.ndarray, 
                       transform: carla.Transform, sensor_type: str):
        """Add sensor data to processing buffer."""
        
        # Convert transform to JAX array
        transform_array = jnp.array([
            transform.location.x, transform.location.y, transform.location.z,
            np.radians(transform.rotation.roll),
            np.radians(transform.rotation.pitch), 
            np.radians(transform.rotation.yaw)
        ])
        
        # Convert data to JAX array
        jax_data = jnp.array(data)
        
        # Determine sensor type ID
        type_id = {'camera': 0, 'lidar': 1, 'radar': 2}.get(sensor_type, 0)
        
        sensor_data = SensorData(
            frame=frame,
            timestamp=time.time(),
            data=jax_data,
            transform=transform_array,
            sensor_type=type_id
        )
        
        self.sensor_data_buffer[sensor_name].append(sensor_data)
        
        # Limit buffer size
        if len(self.sensor_data_buffer[sensor_name]) > self.max_batch_size:
            self.sensor_data_buffer[sensor_name].pop(0)
    
    def process_frame(self, frame: int) -> Optional[Dict[str, jnp.ndarray]]:
        """Process all sensor data for a specific frame with JAX acceleration."""
        
        start_time = time.time()
        results = {}
        
        for sensor_name, data_list in self.sensor_data_buffer.items():
            # Find data for this frame
            frame_data = [d for d in data_list if d.frame == frame]
            
            if not frame_data:
                continue
            
            sensor_data = frame_data[0]  # Take the latest for this frame
            
            # Process based on sensor type
            if sensor_data.sensor_type == 0:  # Camera
                if len(sensor_data.data.shape) == 3:  # RGB image
                    processed = self.process_camera(sensor_data.data)
                    results[f"{sensor_name}_features"] = processed
            
            elif sensor_data.sensor_type == 1:  # LiDAR
                if len(sensor_data.data.shape) == 2:  # Point cloud
                    processed = self.process_lidar(sensor_data.data, sensor_data.transform)
                    results[f"{sensor_name}_points"] = processed
            
            elif sensor_data.sensor_type == 2:  # Radar
                if len(sensor_data.data.shape) == 2:  # Detections
                    processed = self.process_radar(sensor_data.data)
                    results[f"{sensor_name}_detections"] = processed
        
        # Perform sensor fusion if we have multiple sensor types
        if len(results) >= 2:
            camera_features = None
            lidar_points = None
            radar_detections = None
            camera_transform = None
            
            for key, value in results.items():
                if 'features' in key:
                    camera_features = value
                    camera_transform = list(self.sensor_data_buffer.values())[0][0].transform
                elif 'points' in key:
                    lidar_points = value
                elif 'detections' in key:
                    radar_detections = value
            
            if (camera_features is not None and lidar_points is not None 
                and radar_detections is not None):
                fused = self.fuse_sensors(
                    camera_features, lidar_points, radar_detections,
                    camera_transform, time.time()
                )
                results['fused_features'] = fused
        
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        return results if results else None
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get processing performance statistics."""
        if not self.processing_times:
            return {}
        
        times = list(self.processing_times)
        return {
            'avg_processing_time_ms': np.mean(times) * 1000,
            'max_processing_time_ms': np.max(times) * 1000,
            'min_processing_time_ms': np.min(times) * 1000,
            'processing_fps': 1.0 / np.mean(times) if np.mean(times) > 0 else 0
        }


# ==============================================================================
# -- JAX-Enhanced Sensor Callback ---------------------------------------------
# ==============================================================================

def jax_sensor_callback(sensor_data, sensor_queue, sensor_name, sensor_manager, sensor_type):
    """Enhanced sensor callback with JAX processing."""
    
    # Convert CARLA sensor data to numpy
    if sensor_type == 'camera':
        array = np.frombuffer(sensor_data.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (sensor_data.height, sensor_data.width, 4))
        array = array[:, :, :3]  # RGB only
    elif sensor_type == 'lidar':
        points = np.frombuffer(sensor_data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        array = points[:, :3]  # x, y, z only
    elif sensor_type == 'radar':
        points = np.frombuffer(sensor_data.raw_data, dtype=np.dtype('f4'))
        array = np.reshape(points, (int(points.shape[0] / 4), 4))
    else:
        array = np.array([])
    
    # Add to JAX sensor manager
    sensor_manager.add_sensor_data(
        sensor_name, sensor_data.frame, array, 
        sensor_data.transform, sensor_type
    )
    
    # Add to traditional queue for synchronization
    sensor_queue.put((sensor_data.frame, sensor_name))


# ==============================================================================
# -- Main Function ------------------------------------------------------------
# ==============================================================================

def main():
    """JAX-enhanced sensor synchronization example."""
    
    print("Starting JAX-enhanced sensor synchronization...")
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    
    # Initialize JAX sensor manager
    sensor_manager = JAXSensorManager(max_batch_size=5)
    
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    try:
        # Save original settings
        original_settings = world.get_settings()
        settings = world.get_settings()
        
        # Set synchronous mode
        settings.fixed_delta_seconds = 0.1  # 10 FPS for processing
        settings.synchronous_mode = True
        world.apply_settings(settings)
        
        # Create sensor queue
        sensor_queue = Queue()
        
        # Get sensor blueprints
        blueprint_library = world.get_blueprint_library()
        cam_bp = blueprint_library.find('sensor.camera.rgb')
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        radar_bp = blueprint_library.find('sensor.other.radar')
        
        # Configure camera
        cam_bp.set_attribute('image_size_x', '800')
        cam_bp.set_attribute('image_size_y', '600')
        cam_bp.set_attribute('fov', '90')
        
        # Configure LiDAR
        lidar_bp.set_attribute('points_per_second', '100000')
        lidar_bp.set_attribute('rotation_frequency', '10')
        lidar_bp.set_attribute('range', '100')
        
        # Create and spawn sensors
        sensor_list = []
        
        # Camera sensor
        cam_transform = carla.Transform(carla.Location(x=2.0, z=1.5))
        camera = world.spawn_actor(cam_bp, cam_transform)
        camera.listen(lambda data: jax_sensor_callback(
            data, sensor_queue, "camera01", sensor_manager, "camera"))
        sensor_list.append(camera)
        
        # LiDAR sensor
        lidar_transform = carla.Transform(carla.Location(z=2.0))
        lidar = world.spawn_actor(lidar_bp, lidar_transform)
        lidar.listen(lambda data: jax_sensor_callback(
            data, sensor_queue, "lidar01", sensor_manager, "lidar"))
        sensor_list.append(lidar)
        
        # Radar sensor
        radar_transform = carla.Transform(carla.Location(x=2.0, z=1.0))
        radar = world.spawn_actor(radar_bp, radar_transform)
        radar.listen(lambda data: jax_sensor_callback(
            data, sensor_queue, "radar01", sensor_manager, "radar"))
        sensor_list.append(radar)
        
        print(f"Created {len(sensor_list)} sensors with JAX processing")
        
        frame_count = 0
        max_frames = 100  # Run for 100 frames
        
        # Main processing loop
        while frame_count < max_frames:
            # Tick the server
            world.tick()
            w_frame = world.get_snapshot().frame
            frame_count += 1
            
            print(f"\n--- Frame {w_frame} ({frame_count}/{max_frames}) ---")
            
            # Wait for sensor data
            received_sensors = []
            try:
                for _ in range(len(sensor_list)):
                    s_frame, sensor_name = sensor_queue.get(True, 2.0)
                    received_sensors.append((s_frame, sensor_name))
                    print(f"  Received: {sensor_name} (frame {s_frame})")
                
                # Process with JAX
                jax_results = sensor_manager.process_frame(w_frame)
                
                if jax_results:
                    print(f"  JAX processed {len(jax_results)} sensor streams:")
                    for key, value in jax_results.items():
                        print(f"    {key}: shape {value.shape}")
                    
                    # Print performance stats every 10 frames
                    if frame_count % 10 == 0:
                        stats = sensor_manager.get_performance_stats()
                        if stats:
                            print(f"  Performance: {stats['avg_processing_time_ms']:.2f}ms avg, "
                                  f"{stats['processing_fps']:.1f} FPS")
                
            except Empty:
                print("  Timeout: Some sensor data missed")
        
        print(f"\nCompleted {frame_count} frames")
        final_stats = sensor_manager.get_performance_stats()
        if final_stats:
            print(f"Final performance stats:")
            print(f"  Average processing time: {final_stats['avg_processing_time_ms']:.2f}ms")
            print(f"  Processing FPS: {final_stats['processing_fps']:.1f}")
    
    finally:
        # Cleanup
        world.apply_settings(original_settings)
        for sensor in sensor_list:
            if sensor.is_alive:
                sensor.destroy()
        print("Sensors destroyed and settings restored")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('\n - Exited by user.')
    except Exception as e:
        print(f'\n - Error: {e}')
        import traceback
        traceback.print_exc()