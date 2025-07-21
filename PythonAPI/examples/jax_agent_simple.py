#!/usr/bin/env python

"""
Simple JAX-based Neural Network Agent for CARLA Autonomous Driving
Uses pure JAX without Flax dependency for maximum compatibility
"""

import glob
import os
import sys
import numpy as np
from typing import Optional, Dict, Any, Tuple
import pickle

import jax
import jax.numpy as jnp
from jax import random

# Add CARLA to path
try:
    # Try current Python version first
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    try:
        # Fall back to Python 3.7 egg if available
        sys.path.append(glob.glob('../carla/dist/carla-*3.7-%s.egg' % (
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    except IndexError:
        pass

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla


def init_network_params(key, input_dim: int, hidden_dim: int, output_dim: int):
    """Initialize neural network parameters using pure JAX"""
    keys = random.split(key, 6)
    
    # Xavier initialization
    w1_scale = jnp.sqrt(2.0 / input_dim)
    w2_scale = jnp.sqrt(2.0 / hidden_dim)
    w3_scale = jnp.sqrt(2.0 / hidden_dim)
    
    params = {
        'w1': random.normal(keys[0], (input_dim, hidden_dim)) * w1_scale,
        'b1': jnp.zeros((hidden_dim,)),
        'w2': random.normal(keys[1], (hidden_dim, hidden_dim)) * w2_scale,
        'b2': jnp.zeros((hidden_dim,)),
        'w3': random.normal(keys[2], (hidden_dim, output_dim)) * w3_scale,
        'b3': jnp.zeros((output_dim,))
    }
    return params


@jax.jit
def policy_forward(params, x):
    """Forward pass for policy network"""
    # Layer 1
    x = jnp.dot(x, params['w1']) + params['b1']
    x = jax.nn.relu(x)
    
    # Layer 2
    x = jnp.dot(x, params['w2']) + params['b2']
    x = jax.nn.relu(x)
    
    # Output layer with tanh activation
    x = jnp.dot(x, params['w3']) + params['b3']
    return jax.nn.tanh(x)


@jax.jit
def value_forward(params, x):
    """Forward pass for value network"""
    # Layer 1
    x = jnp.dot(x, params['w1']) + params['b1']
    x = jax.nn.relu(x)
    
    # Layer 2
    x = jnp.dot(x, params['w2']) + params['b2']
    x = jax.nn.relu(x)
    
    # Output layer (single value)
    x = jnp.dot(x, params['w3']) + params['b3']
    return x.squeeze()


class SimpleJAXAgent:
    """Simple JAX-based neural network agent for CARLA"""
    
    def __init__(self, vehicle, target_speed: float = 30.0, model_path: Optional[str] = None):
        self.vehicle = vehicle
        self.target_speed = target_speed
        self._target_waypoint = None
        
        # Network dimensions
        self.obs_dim = 12  # Simplified observation space
        self.action_dim = 2  # [steering, throttle/brake]
        self.hidden_dim = 128
        
        # Initialize random key
        self.key = random.PRNGKey(42)
        
        # Initialize network parameters
        policy_key, value_key = random.split(self.key, 2)
        
        self.policy_params = init_network_params(
            policy_key, self.obs_dim, self.hidden_dim, self.action_dim
        )
        
        self.value_params = init_network_params(
            value_key, self.obs_dim, self.hidden_dim, 1
        )
        
        # JIT compile prediction functions
        self._predict_action = jax.jit(self._predict_action_fn)
        self._predict_value = jax.jit(self._predict_value_fn)
        
        # Load pre-trained weights if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def _predict_action_fn(self, params, obs):
        """JIT-compiled action prediction"""
        return policy_forward(params, obs)
    
    def _predict_value_fn(self, params, obs):
        """JIT-compiled value prediction"""
        return value_forward(params, obs)
    
    def set_destination(self, location: carla.Location):
        """Set the destination for the agent"""
        self._target_waypoint = self.vehicle.get_world().get_map().get_waypoint(location)
    
    def get_observations(self) -> np.ndarray:
        """Extract simplified observation vector from CARLA environment"""
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        location = transform.location
        rotation = transform.rotation
        
        # Get map information
        world = self.vehicle.get_world()
        map = world.get_map()
        waypoint = map.get_waypoint(location)
        
        # Basic vehicle state
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        forward_vec = transform.get_forward_vector()
        right_vec = transform.get_right_vector()
        
        # Lateral velocity (sliding)
        lateral_speed = np.dot([velocity.x, velocity.y, velocity.z],
                              [right_vec.x, right_vec.y, right_vec.z])
        
        # Get next waypoint
        next_waypoints = waypoint.next(5.0)  # 5 meters ahead
        if next_waypoints:
            next_wp = next_waypoints[0]
            # Relative position to next waypoint
            wp_location = next_wp.transform.location
            wp_relative = wp_location - location
            
            # Transform to vehicle frame
            yaw_rad = np.radians(rotation.yaw)
            local_x = wp_relative.x * np.cos(-yaw_rad) - wp_relative.y * np.sin(-yaw_rad)
            local_y = wp_relative.x * np.sin(-yaw_rad) + wp_relative.y * np.cos(-yaw_rad)
            
            # Lane direction error
            lane_yaw = next_wp.transform.rotation.yaw
            yaw_diff = rotation.yaw - lane_yaw
            while yaw_diff > 180:
                yaw_diff -= 360
            while yaw_diff < -180:
                yaw_diff += 360
        else:
            local_x = local_y = yaw_diff = 0.0
        
        # Distance to lane center
        lane_center = waypoint.transform.location
        distance_to_center = np.sqrt((location.x - lane_center.x)**2 + 
                                   (location.y - lane_center.y)**2)
        
        # Target speed difference
        target_speed_ms = self.target_speed / 3.6  # Convert km/h to m/s
        speed_diff = target_speed_ms - speed
        
        # Compile simplified observation vector (12 features)
        obs = np.array([
            speed / 20.0,  # Normalized speed (max ~20 m/s)
            lateral_speed / 5.0,  # Normalized lateral speed
            distance_to_center / 3.0,  # Normalized distance to lane center
            np.sin(np.radians(yaw_diff)),  # Sin of yaw difference
            np.cos(np.radians(yaw_diff)),  # Cos of yaw difference
            speed_diff / 10.0,  # Normalized speed difference
            local_x / 10.0,  # Normalized waypoint x
            local_y / 10.0,  # Normalized waypoint y
            rotation.pitch / 45.0,  # Normalized pitch
            rotation.roll / 45.0,  # Normalized roll
            velocity.z / 5.0,  # Normalized vertical velocity
            np.tanh(transform.location.z / 100.0)  # Normalized height
        ], dtype=np.float32)
        
        return obs
    
    def run_step(self) -> carla.VehicleControl:
        """Execute one step of control"""
        # Get observations
        obs = self.get_observations()
        
        # Convert to JAX array and add batch dimension
        obs_jax = jnp.expand_dims(jnp.array(obs), axis=0)
        
        # Get action from policy network (JIT compiled)
        action = self._predict_action(self.policy_params, obs_jax)
        action = jnp.squeeze(action, axis=0)  # Remove batch dimension
        
        # Convert JAX array to Python floats for CARLA
        steer = float(action[0])
        throttle_brake = float(action[1])
        
        # Create CARLA control command
        control = carla.VehicleControl()
        control.steer = np.clip(steer, -1.0, 1.0)
        
        if throttle_brake >= 0:
            control.throttle = np.clip(throttle_brake, 0.0, 1.0)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = np.clip(-throttle_brake, 0.0, 1.0)
        
        # Always use automatic transmission
        control.manual_gear_shift = False
        
        return control
    
    def done(self) -> bool:
        """Check if the agent has reached its destination"""
        if self._target_waypoint is None:
            return False
        
        vehicle_location = self.vehicle.get_location()
        target_location = self._target_waypoint.transform.location
        
        distance = vehicle_location.distance(target_location)
        return distance < 2.0  # Within 2 meters of target
    
    def save_model(self, path: str):
        """Save model parameters"""
        params = {
            'policy': self.policy_params,
            'value': self.value_params
        }
        with open(path, 'wb') as f:
            pickle.dump(params, f)
    
    def load_model(self, path: str):
        """Load model parameters"""
        with open(path, 'rb') as f:
            params = pickle.load(f)
        self.policy_params = params['policy']
        self.value_params = params['value']
    
    def get_value_estimate(self, obs: Optional[np.ndarray] = None) -> float:
        """Get value estimate for current or given observation"""
        if obs is None:
            obs = self.get_observations()
        
        obs_jax = jnp.expand_dims(jnp.array(obs), axis=0)
        value = self._predict_value(self.value_params, obs_jax)
        return float(value)


# Alias for compatibility
JAXCARLAAgent = SimpleJAXAgent