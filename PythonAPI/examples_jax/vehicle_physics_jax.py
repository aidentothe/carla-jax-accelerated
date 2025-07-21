#!/usr/bin/env python

"""JAX-accelerated vehicle physics simulation with automatic differentiation.

This example demonstrates how to use JAX for physics-based vehicle modeling
with automatic differentiation for parameter optimization, gradient-based
control, and real-time physics simulation acceleration.
"""

import argparse
import glob
import math
import os
import sys
import time
from typing import NamedTuple, Tuple, Dict, Any

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
    from jax import jit, grad, vmap, lax
    import optax
    import chex
    from functools import partial
except ImportError:
    raise RuntimeError('JAX not found. Install with: pip install jax jaxlib optax chex')

try:
    import pygame
    from pygame.locals import KMOD_CTRL, K_ESCAPE, K_q, K_r, K_SPACE
except ImportError:
    raise RuntimeError('pygame not found. Install with: pip install pygame')


# ==============================================================================
# -- JAX Vehicle State and Parameters -----------------------------------------
# ==============================================================================

class VehicleState(NamedTuple):
    """JAX-compatible vehicle state representation."""
    position: jnp.ndarray      # [x, y, z]
    velocity: jnp.ndarray      # [vx, vy, vz]
    orientation: jnp.ndarray   # [roll, pitch, yaw]
    angular_velocity: jnp.ndarray  # [wx, wy, wz]
    wheel_speeds: jnp.ndarray  # [fl, fr, rl, rr] front-left, front-right, etc.
    steering_angle: float
    throttle: float
    brake: float


class VehicleParameters(NamedTuple):
    """Physics parameters for vehicle simulation."""
    mass: float               # kg
    inertia_xx: float        # kg*m^2
    inertia_yy: float        # kg*m^2  
    inertia_zz: float        # kg*m^2
    wheelbase: float         # m (distance between front and rear axles)
    track_width: float       # m (distance between left and right wheels)
    center_of_mass_height: float  # m
    drag_coefficient: float  # aerodynamic drag
    rolling_resistance: float
    tire_stiffness_front: float
    tire_stiffness_rear: float
    max_steering_angle: float  # radians


# ==============================================================================
# -- JAX Vehicle Physics Model ------------------------------------------------
# ==============================================================================

@jit
def bicycle_model_dynamics(state: VehicleState, 
                          control: jnp.ndarray,  # [throttle, brake, steering]
                          params: VehicleParameters,
                          dt: float) -> VehicleState:
    """JAX-compiled bicycle model vehicle dynamics."""
    
    # Extract state variables
    x, y, z = state.position
    vx, vy, vz = state.velocity
    roll, pitch, yaw = state.orientation
    
    # Extract control inputs
    throttle = jnp.clip(control[0], 0.0, 1.0)
    brake = jnp.clip(control[1], 0.0, 1.0)
    steering = jnp.clip(control[2], -1.0, 1.0) * params.max_steering_angle
    
    # Current speed and direction
    speed = jnp.sqrt(vx**2 + vy**2)
    
    # Tire forces calculation
    # Front and rear tire slip angles
    if speed > 0.1:  # Avoid division by zero
        alpha_f = steering - jnp.arctan2(vy + params.wheelbase/2 * state.angular_velocity[2], vx)
        alpha_r = -jnp.arctan2(vy - params.wheelbase/2 * state.angular_velocity[2], vx)
    else:
        alpha_f = steering
        alpha_r = 0.0
    
    # Lateral tire forces (simplified linear tire model)
    F_yf = params.tire_stiffness_front * alpha_f
    F_yr = params.tire_stiffness_rear * alpha_r
    
    # Longitudinal forces
    # Engine force (simplified)
    F_engine = throttle * 5000.0  # N (simplified engine map)
    
    # Braking force
    F_brake = brake * 8000.0  # N
    
    # Total longitudinal force
    F_x = F_engine - F_brake
    
    # Aerodynamic drag
    F_drag = 0.5 * params.drag_coefficient * 1.225 * speed**2  # air density = 1.225 kg/m^3
    F_x -= F_drag
    
    # Rolling resistance
    F_rolling = params.rolling_resistance * params.mass * 9.81
    F_x -= F_rolling * jnp.sign(vx)
    
    # Accelerations in vehicle frame
    ax_vehicle = F_x / params.mass
    ay_vehicle = (F_yf + F_yr) / params.mass
    
    # Transform accelerations to global frame
    cos_yaw = jnp.cos(yaw)
    sin_yaw = jnp.sin(yaw)
    
    ax_global = ax_vehicle * cos_yaw - ay_vehicle * sin_yaw
    ay_global = ax_vehicle * sin_yaw + ay_vehicle * cos_yaw
    az_global = -9.81  # gravity
    
    # Angular acceleration (yaw only for bicycle model)
    yaw_moment = F_yf * params.wheelbase / 2
    angular_acc_z = yaw_moment / params.inertia_zz
    
    # Integrate using explicit Euler (could be improved with RK4)
    new_position = state.position + state.velocity * dt
    new_velocity = state.velocity + jnp.array([ax_global, ay_global, az_global]) * dt
    new_angular_velocity = state.angular_velocity.at[2].add(angular_acc_z * dt)
    new_orientation = state.orientation.at[2].add(state.angular_velocity[2] * dt)
    
    # Update wheel speeds (simplified)
    wheel_speed = speed / 0.35  # assuming 0.35m wheel radius
    new_wheel_speeds = jnp.array([wheel_speed, wheel_speed, wheel_speed, wheel_speed])
    
    return VehicleState(
        position=new_position,
        velocity=new_velocity,
        orientation=new_orientation,
        angular_velocity=new_angular_velocity,
        wheel_speeds=new_wheel_speeds,
        steering_angle=steering,
        throttle=throttle,
        brake=brake
    )


@jit 
def simulate_trajectory(initial_state: VehicleState,
                       control_sequence: jnp.ndarray,  # shape: (T, 3)
                       params: VehicleParameters,
                       dt: float) -> jnp.ndarray:
    """Simulate vehicle trajectory over multiple time steps."""
    
    def step_fn(state, control):
        new_state = bicycle_model_dynamics(state, control, params, dt)
        return new_state, new_state.position
    
    _, trajectory = lax.scan(step_fn, initial_state, control_sequence)
    return trajectory


# ==============================================================================
# -- JAX Trajectory Optimization ----------------------------------------------
# ==============================================================================

@jit
def trajectory_cost(control_sequence: jnp.ndarray,
                   initial_state: VehicleState,
                   target_positions: jnp.ndarray,
                   params: VehicleParameters,
                   dt: float) -> float:
    """Cost function for trajectory optimization."""
    
    # Simulate trajectory
    trajectory = simulate_trajectory(initial_state, control_sequence, params, dt)
    
    # Position tracking cost
    position_error = trajectory - target_positions
    tracking_cost = jnp.sum(jnp.linalg.norm(position_error, axis=1)**2)
    
    # Control effort cost
    control_cost = 0.1 * jnp.sum(control_sequence**2)
    
    # Smoothness cost (penalize rapid control changes)
    control_diff = jnp.diff(control_sequence, axis=0)
    smoothness_cost = 0.01 * jnp.sum(control_diff**2)
    
    # Speed limit cost (penalize excessive speed)
    speeds = jnp.linalg.norm(jnp.diff(trajectory, axis=0), axis=1) / dt
    speed_penalty = jnp.sum(jnp.maximum(0, speeds - 30.0)**2)  # 30 m/s speed limit
    
    total_cost = tracking_cost + control_cost + smoothness_cost + 0.1 * speed_penalty
    return total_cost


class JAXTrajectoryOptimizer:
    """JAX-based trajectory optimizer using automatic differentiation."""
    
    def __init__(self, vehicle_params: VehicleParameters, dt: float = 0.1):
        self.params = vehicle_params
        self.dt = dt
        
        # Initialize optimizer
        self.optimizer = optax.adam(learning_rate=0.01)
        
        # Compile cost function and its gradient
        self.cost_fn = jit(partial(trajectory_cost, 
                                  params=self.params, 
                                  dt=self.dt))
        self.grad_fn = jit(grad(self.cost_fn, argnums=0))
        
    def optimize_trajectory(self, 
                          initial_state: VehicleState,
                          target_positions: jnp.ndarray,
                          horizon: int = 50,
                          max_iterations: int = 100) -> Tuple[jnp.ndarray, float]:
        """Optimize control sequence to follow target trajectory."""
        
        # Initialize control sequence
        control_sequence = jnp.zeros((horizon, 3))
        
        # Initialize optimizer state
        opt_state = self.optimizer.init(control_sequence)
        
        best_cost = float('inf')
        best_controls = control_sequence
        
        for i in range(max_iterations):
            # Compute gradients
            grads = self.grad_fn(control_sequence, initial_state, target_positions)
            
            # Update parameters
            updates, opt_state = self.optimizer.update(grads, opt_state)
            control_sequence = optax.apply_updates(control_sequence, updates)
            
            # Clip controls to valid ranges
            control_sequence = jnp.clip(control_sequence, 
                                      jnp.array([0.0, 0.0, -1.0]), 
                                      jnp.array([1.0, 1.0, 1.0]))
            
            # Evaluate cost
            current_cost = self.cost_fn(control_sequence, initial_state, target_positions)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_controls = control_sequence
            
            # Early stopping
            if i > 10 and current_cost < 1e-3:
                break
        
        return best_controls, best_cost


# ==============================================================================
# -- JAX Parameter Identification ---------------------------------------------
# ==============================================================================

@jit
def parameter_identification_loss(params_vector: jnp.ndarray,
                                 measurements: jnp.ndarray,
                                 control_inputs: jnp.ndarray,
                                 initial_state: VehicleState,
                                 dt: float) -> float:
    """Loss function for vehicle parameter identification."""
    
    # Unpack parameters
    mass, drag_coeff, tire_stiffness_f, tire_stiffness_r = params_vector
    
    # Create parameter struct
    params = VehicleParameters(
        mass=mass,
        inertia_xx=1500.0,  # Fixed for simplicity
        inertia_yy=3000.0,
        inertia_zz=3000.0,
        wheelbase=2.5,
        track_width=1.8,
        center_of_mass_height=0.5,
        drag_coefficient=drag_coeff,
        rolling_resistance=0.015,
        tire_stiffness_front=tire_stiffness_f,
        tire_stiffness_rear=tire_stiffness_r,
        max_steering_angle=jnp.pi/4
    )
    
    # Simulate with these parameters
    predicted_trajectory = simulate_trajectory(initial_state, control_inputs, params, dt)
    
    # Compare with measurements
    error = predicted_trajectory - measurements
    loss = jnp.sum(error**2)
    
    return loss


class JAXParameterIdentifier:
    """Parameter identification using JAX autodiff."""
    
    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.optimizer = optax.adam(learning_rate=0.001)
        
        # Compile loss and gradient functions
        self.loss_fn = jit(partial(parameter_identification_loss, dt=self.dt))
        self.grad_fn = jit(grad(self.loss_fn, argnums=0))
    
    def identify_parameters(self,
                          measurements: jnp.ndarray,
                          control_inputs: jnp.ndarray,
                          initial_state: VehicleState,
                          max_iterations: int = 500) -> Dict[str, float]:
        """Identify vehicle parameters from measurement data."""
        
        # Initial parameter guess
        params_vector = jnp.array([1500.0, 0.3, 80000.0, 80000.0])  # mass, drag, tire_f, tire_r
        
        # Initialize optimizer
        opt_state = self.optimizer.init(params_vector)
        
        losses = []
        
        for i in range(max_iterations):
            # Compute gradients
            grads = self.grad_fn(params_vector, measurements, control_inputs, initial_state)
            
            # Update parameters
            updates, opt_state = self.optimizer.update(grads, opt_state)
            params_vector = optax.apply_updates(params_vector, updates)
            
            # Constrain parameters to reasonable ranges
            params_vector = jnp.clip(params_vector,
                                   jnp.array([800.0, 0.1, 10000.0, 10000.0]),
                                   jnp.array([3000.0, 1.0, 200000.0, 200000.0]))
            
            # Track loss
            current_loss = self.loss_fn(params_vector, measurements, control_inputs, initial_state)
            losses.append(float(current_loss))
            
            if i % 50 == 0:
                print(f"Iteration {i}, Loss: {current_loss:.6f}")
        
        return {
            'mass': float(params_vector[0]),
            'drag_coefficient': float(params_vector[1]),
            'tire_stiffness_front': float(params_vector[2]),
            'tire_stiffness_rear': float(params_vector[3]),
            'final_loss': losses[-1],
            'loss_history': losses
        }


# ==============================================================================
# -- CARLA Integration --------------------------------------------------------
# ==============================================================================

class JAXPhysicsController:
    """JAX-enhanced physics-based vehicle controller."""
    
    def __init__(self, vehicle, vehicle_params: VehicleParameters):
        self.vehicle = vehicle
        self.params = vehicle_params
        self.optimizer = JAXTrajectoryOptimizer(vehicle_params)
        
        # State tracking
        self.state_history = []
        self.control_history = []
        
    def carla_to_jax_state(self) -> VehicleState:
        """Convert CARLA vehicle state to JAX state representation."""
        
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        angular_velocity = self.vehicle.get_angular_velocity()
        
        position = jnp.array([
            transform.location.x,
            transform.location.y, 
            transform.location.z
        ])
        
        velocity_vec = jnp.array([
            velocity.x,
            velocity.y,
            velocity.z
        ])
        
        orientation = jnp.array([
            math.radians(transform.rotation.roll),
            math.radians(transform.rotation.pitch),
            math.radians(transform.rotation.yaw)
        ])
        
        angular_vel = jnp.array([
            math.radians(angular_velocity.x),
            math.radians(angular_velocity.y),
            math.radians(angular_velocity.z)
        ])
        
        # Get wheel physics (simplified)
        wheel_speeds = jnp.array([0.0, 0.0, 0.0, 0.0])  # Could be extracted from CARLA
        
        return VehicleState(
            position=position,
            velocity=velocity_vec,
            orientation=orientation,
            angular_velocity=angular_vel,
            wheel_speeds=wheel_speeds,
            steering_angle=0.0,
            throttle=0.0,
            brake=0.0
        )
    
    def compute_optimal_control(self, target_waypoints: np.ndarray) -> carla.VehicleControl:
        """Compute optimal control using JAX trajectory optimization."""
        
        current_state = self.carla_to_jax_state()
        target_positions = jnp.array(target_waypoints)
        
        # Optimize trajectory
        optimal_controls, cost = self.optimizer.optimize_trajectory(
            current_state, target_positions, horizon=min(50, len(target_waypoints))
        )
        
        # Extract first control action
        next_control = optimal_controls[0]
        
        # Convert to CARLA control
        control = carla.VehicleControl()
        control.throttle = float(next_control[0])
        control.brake = float(next_control[1])
        control.steer = float(next_control[2])
        
        # Store for analysis
        self.state_history.append(current_state)
        self.control_history.append(next_control)
        
        return control


# ==============================================================================
# -- Main Demonstration -------------------------------------------------------
# ==============================================================================

def main():
    """Demonstrate JAX vehicle physics with CARLA integration."""
    
    argparser = argparse.ArgumentParser(description='JAX Vehicle Physics Demo')
    argparser.add_argument('--host', default='127.0.0.1', help='IP of CARLA server')
    argparser.add_argument('-p', '--port', default=2000, type=int, help='TCP port')
    argparser.add_argument('--sync', action='store_true', help='Synchronous mode')
    argparser.add_argument('--demo-mode', default='physics', 
                          choices=['physics', 'optimization', 'identification'],
                          help='Demo mode to run')
    
    args = argparser.parse_args()
    
    print("JAX Vehicle Physics Demonstration")
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Demo mode: {args.demo_mode}")
    
    if args.demo_mode == 'physics':
        # Demonstrate real-time physics simulation with CARLA
        run_physics_demo(args)
    elif args.demo_mode == 'optimization':
        # Demonstrate trajectory optimization
        run_optimization_demo()
    elif args.demo_mode == 'identification':
        # Demonstrate parameter identification
        run_identification_demo()


def run_physics_demo(args):
    """Run real-time physics simulation demo with CARLA."""
    
    pygame.init()
    
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    
    try:
        # Setup synchronous mode
        if args.sync:
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.1
            world.apply_settings(settings)
        
        # Spawn vehicle
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        spawn_points = world.get_map().get_spawn_points()
        vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
        
        # Create JAX controller with realistic parameters
        params = VehicleParameters(
            mass=1500.0,
            inertia_xx=1200.0,
            inertia_yy=3000.0,
            inertia_zz=3000.0,
            wheelbase=2.875,
            track_width=1.849,
            center_of_mass_height=0.5,
            drag_coefficient=0.24,
            rolling_resistance=0.015,
            tire_stiffness_front=80000.0,
            tire_stiffness_rear=80000.0,
            max_steering_angle=math.pi/4
        )
        
        controller = JAXPhysicsController(vehicle, params)
        
        # Create target waypoints (simple circular path)
        center = spawn_points[0].location
        radius = 50.0
        num_waypoints = 100
        
        waypoints = []
        for i in range(num_waypoints):
            angle = 2 * math.pi * i / num_waypoints
            x = center.x + radius * math.cos(angle)
            y = center.y + radius * math.sin(angle)
            z = center.z
            waypoints.append([x, y, z])
        
        waypoints = np.array(waypoints)
        
        print("Starting physics simulation. Press ESC to exit.")
        
        clock = pygame.time.Clock()
        frame_count = 0
        jax_times = []
        
        while frame_count < 1000:  # Run for 1000 frames
            clock.tick(10)  # 10 FPS
            
            if args.sync:
                world.tick()
            
            # Get next waypoints
            start_idx = (frame_count * 2) % len(waypoints)
            end_idx = min(start_idx + 20, len(waypoints))
            target_waypoints = waypoints[start_idx:end_idx]
            
            # Compute optimal control with JAX
            start_time = time.time()
            control = controller.compute_optimal_control(target_waypoints)
            jax_time = time.time() - start_time
            jax_times.append(jax_time)
            
            # Apply control
            vehicle.apply_control(control)
            
            frame_count += 1
            
            if frame_count % 50 == 0:
                avg_time = np.mean(jax_times[-50:]) * 1000
                print(f"Frame {frame_count}: JAX optimization time: {avg_time:.2f}ms")
        
        avg_jax_time = np.mean(jax_times) * 1000
        print(f"Demo completed. Average JAX processing time: {avg_jax_time:.2f}ms")
        
    finally:
        if 'vehicle' in locals():
            vehicle.destroy()
        
        pygame.quit()


def run_optimization_demo():
    """Demonstrate standalone trajectory optimization."""
    
    print("Running trajectory optimization demo...")
    
    # Create vehicle parameters
    params = VehicleParameters(
        mass=1500.0, inertia_xx=1200.0, inertia_yy=3000.0, inertia_zz=3000.0,
        wheelbase=2.5, track_width=1.8, center_of_mass_height=0.5,
        drag_coefficient=0.3, rolling_resistance=0.015,
        tire_stiffness_front=80000.0, tire_stiffness_rear=80000.0,
        max_steering_angle=math.pi/4
    )
    
    # Initial state
    initial_state = VehicleState(
        position=jnp.array([0.0, 0.0, 0.0]),
        velocity=jnp.array([0.0, 0.0, 0.0]),
        orientation=jnp.array([0.0, 0.0, 0.0]),
        angular_velocity=jnp.array([0.0, 0.0, 0.0]),
        wheel_speeds=jnp.array([0.0, 0.0, 0.0, 0.0]),
        steering_angle=0.0,
        throttle=0.0,
        brake=0.0
    )
    
    # Target trajectory (S-curve)
    t = jnp.linspace(0, 10, 50)
    target_x = 2 * t
    target_y = 5 * jnp.sin(0.5 * t)
    target_z = jnp.zeros_like(t)
    target_positions = jnp.stack([target_x, target_y, target_z], axis=1)
    
    # Optimize trajectory
    optimizer = JAXTrajectoryOptimizer(params)
    
    start_time = time.time()
    optimal_controls, final_cost = optimizer.optimize_trajectory(
        initial_state, target_positions, horizon=50, max_iterations=200
    )
    optimization_time = time.time() - start_time
    
    print(f"Optimization completed in {optimization_time:.3f}s")
    print(f"Final cost: {final_cost:.6f}")
    
    # Simulate optimized trajectory
    optimized_trajectory = simulate_trajectory(initial_state, optimal_controls, params, 0.2)
    
    print(f"Trajectory optimization successful!")
    print(f"Final position error: {jnp.linalg.norm(optimized_trajectory[-1] - target_positions[-1]):.3f}m")


def run_identification_demo():
    """Demonstrate parameter identification."""
    
    print("Running parameter identification demo...")
    
    # True parameters (what we're trying to identify)
    true_params = VehicleParameters(
        mass=1600.0, inertia_xx=1200.0, inertia_yy=3000.0, inertia_zz=3000.0,
        wheelbase=2.5, track_width=1.8, center_of_mass_height=0.5,
        drag_coefficient=0.35, rolling_resistance=0.015,
        tire_stiffness_front=85000.0, tire_stiffness_rear=75000.0,
        max_steering_angle=math.pi/4
    )
    
    # Generate synthetic measurement data
    initial_state = VehicleState(
        position=jnp.array([0.0, 0.0, 0.0]),
        velocity=jnp.array([10.0, 0.0, 0.0]),
        orientation=jnp.array([0.0, 0.0, 0.0]),
        angular_velocity=jnp.array([0.0, 0.0, 0.0]),
        wheel_speeds=jnp.array([0.0, 0.0, 0.0, 0.0]),
        steering_angle=0.0,
        throttle=0.0,
        brake=0.0
    )
    
    # Random control inputs
    key = jax.random.PRNGKey(42)
    control_inputs = jax.random.uniform(key, (100, 3), minval=-0.5, maxval=0.5)
    control_inputs = control_inputs.at[:, :2].set(jnp.abs(control_inputs[:, :2]))  # positive throttle/brake
    
    # Generate "measurements" using true parameters
    measurements = simulate_trajectory(initial_state, control_inputs, true_params, 0.1)
    
    # Add some noise
    noise_key = jax.random.PRNGKey(123)
    noise = jax.random.normal(noise_key, measurements.shape) * 0.1
    measurements += noise
    
    # Identify parameters
    identifier = JAXParameterIdentifier()
    identified_params = identifier.identify_parameters(
        measurements, control_inputs, initial_state, max_iterations=300
    )
    
    print("Parameter Identification Results:")
    print(f"True mass: {true_params.mass:.1f} kg, Identified: {identified_params['mass']:.1f} kg")
    print(f"True drag coeff: {true_params.drag_coefficient:.3f}, Identified: {identified_params['drag_coefficient']:.3f}")
    print(f"True tire stiffness front: {true_params.tire_stiffness_front:.0f} N/rad, Identified: {identified_params['tire_stiffness_front']:.0f} N/rad")
    print(f"True tire stiffness rear: {true_params.tire_stiffness_rear:.0f} N/rad, Identified: {identified_params['tire_stiffness_rear']:.0f} N/rad")
    print(f"Final identification loss: {identified_params['final_loss']:.6f}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n - Exited by user.')
    except Exception as e:
        print(f'\n - Error: {e}')
        import traceback
        traceback.print_exc()