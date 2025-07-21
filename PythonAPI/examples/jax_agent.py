#!/usr/bin/env python

"""
JAX-based Neural Network Agent for CARLA Autonomous Driving
Implements a policy gradient agent using JAX and Flax for high-performance inference
"""

import glob
import os
import sys
import numpy as np
from typing import Optional, Dict, Any, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from flax.core import FrozenDict
import optax

# Add CARLA to path
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla


class PolicyNetwork(nn.Module):
    """Actor network for continuous control in autonomous driving"""
    action_dim: int = 2  # [steering, throttle/brake]
    hidden_dim: int = 256
    
    @nn.compact
    def __call__(self, x):
        # Feature extraction layers
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.LayerNorm()(x)
        
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.LayerNorm()(x)
        
        x = nn.Dense(self.hidden_dim // 2)(x)
        x = nn.relu(x)
        
        # Output layer with tanh activation for bounded actions
        x = nn.Dense(self.action_dim)(x)
        return nn.tanh(x)  # Output in [-1, 1]


class ValueNetwork(nn.Module):
    """Critic network for value estimation"""
    hidden_dim: int = 256
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.LayerNorm()(x)
        
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.LayerNorm()(x)
        
        x = nn.Dense(1)(x)
        return x


class JAXCARLAAgent:
    """JAX-based neural network agent for CARLA autonomous driving"""
    
    def __init__(self, vehicle, target_speed: float = 30.0, model_path: Optional[str] = None):
        self.vehicle = vehicle
        self.target_speed = target_speed
        self._target_waypoint = None
        self._route_waypoints = []
        
        # Initialize random key
        self.key = jax.random.PRNGKey(42)
        
        # Initialize networks
        self.policy_net = PolicyNetwork()
        self.value_net = ValueNetwork()
        
        # Define observation dimension based on features
        self.obs_dim = 18  # Adjust based on get_observations implementation
        
        # Initialize model parameters
        self.key, policy_key, value_key = jax.random.split(self.key, 3)
        dummy_obs = jnp.zeros((1, self.obs_dim))
        
        self.policy_params = self.policy_net.init(policy_key, dummy_obs)
        self.value_params = self.value_net.init(value_key, dummy_obs)
        
        # JIT compile prediction functions for speed
        self._predict_action_jit = jax.jit(self._predict_action)
        self._predict_value_jit = jax.jit(self._predict_value)
        
        # Load pre-trained weights if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def _predict_action(self, params: FrozenDict, obs: jnp.ndarray) -> jnp.ndarray:
        """JIT-compatible action prediction"""
        return self.policy_net.apply(params, obs)
    
    def _predict_value(self, params: FrozenDict, obs: jnp.ndarray) -> jnp.ndarray:
        """JIT-compatible value prediction"""
        return self.value_net.apply(params, obs)
    
    def set_destination(self, location: carla.Location):
        """Set the destination for the agent"""
        self._target_waypoint = self.vehicle.get_world().get_map().get_waypoint(location)
        # In a full implementation, compute route here
        # For now, we'll use simple waypoint following
    
    def get_observations(self) -> np.ndarray:
        """Extract observation vector from CARLA environment"""
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        location = transform.location
        rotation = transform.rotation
        
        # Get map and waypoint information
        world = self.vehicle.get_world()
        map = world.get_map()
        waypoint = map.get_waypoint(location)
        
        # Forward vector
        forward_vec = transform.get_forward_vector()
        
        # Get next waypoints for trajectory
        next_waypoints = []
        current_waypoint = waypoint
        for _ in range(3):  # Look ahead 3 waypoints
            next_wp = current_waypoint.next(2.0)  # 2 meters ahead
            if next_wp:
                current_waypoint = next_wp[0]
                next_waypoints.append(current_waypoint)
        
        # Calculate relative positions of waypoints
        waypoint_features = []
        for wp in next_waypoints:
            wp_loc = wp.transform.location
            # Transform to vehicle's local coordinate system
            relative_pos = wp_loc - location
            # Simplified 2D rotation (ignoring pitch/roll)
            yaw_rad = np.radians(rotation.yaw)
            local_x = relative_pos.x * np.cos(-yaw_rad) - relative_pos.y * np.sin(-yaw_rad)
            local_y = relative_pos.x * np.sin(-yaw_rad) + relative_pos.y * np.cos(-yaw_rad)
            waypoint_features.extend([local_x, local_y])
        
        # Pad if we have fewer waypoints
        while len(waypoint_features) < 6:  # 3 waypoints * 2 coordinates
            waypoint_features.extend([0.0, 0.0])
        
        # Speed in vehicle's forward direction
        speed = np.dot([velocity.x, velocity.y, velocity.z], 
                      [forward_vec.x, forward_vec.y, forward_vec.z])
        
        # Lateral velocity (for detecting sliding)
        right_vec = transform.get_right_vector()
        lateral_speed = np.dot([velocity.x, velocity.y, velocity.z],
                              [right_vec.x, right_vec.y, right_vec.z])
        
        # Distance to center of lane
        lane_center = waypoint.transform.location
        distance_to_center = np.sqrt((location.x - lane_center.x)**2 + 
                                   (location.y - lane_center.y)**2)
        
        # Angle difference with lane direction
        lane_yaw = waypoint.transform.rotation.yaw
        yaw_diff = rotation.yaw - lane_yaw
        # Normalize to [-180, 180]
        while yaw_diff > 180:
            yaw_diff -= 360
        while yaw_diff < -180:
            yaw_diff += 360
        
        # Target speed difference
        speed_diff = self.target_speed / 3.6 - speed  # Convert km/h to m/s
        
        # Compile observation vector
        obs = np.array([
            speed / 30.0,  # Normalized speed
            lateral_speed / 10.0,  # Normalized lateral speed
            distance_to_center / 5.0,  # Normalized distance to lane center
            np.sin(np.radians(yaw_diff)),  # Sin of yaw difference
            np.cos(np.radians(yaw_diff)),  # Cos of yaw difference
            speed_diff / 30.0,  # Normalized speed difference
            rotation.pitch / 90.0,  # Normalized pitch
            rotation.roll / 90.0,  # Normalized roll
            velocity.z / 10.0,  # Normalized vertical velocity
            *waypoint_features[:6],  # Next 3 waypoint positions (normalized)
            # Total: 18 features
        ], dtype=np.float32)
        
        return obs
    
    def run_step(self) -> carla.VehicleControl:
        """Execute one step of control"""
        # Get observations
        obs = self.get_observations()
        
        # Convert to JAX array and add batch dimension
        obs_jax = jnp.expand_dims(jnp.array(obs), axis=0)
        
        # Get action from policy network (JIT compiled for speed)
        action = self._predict_action_jit(self.policy_params, obs_jax)
        action = jnp.squeeze(action, axis=0)  # Remove batch dimension
        
        # Convert JAX array to Python floats for CARLA
        steer = float(action[0])
        throttle_brake = float(action[1])
        
        # Create CARLA control command
        control = carla.VehicleControl()
        control.steer = steer
        
        if throttle_brake >= 0:
            control.throttle = throttle_brake
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = -throttle_brake
        
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
        import pickle
        params = {
            'policy': self.policy_params,
            'value': self.value_params
        }
        with open(path, 'wb') as f:
            pickle.dump(params, f)
    
    def load_model(self, path: str):
        """Load model parameters"""
        import pickle
        with open(path, 'rb') as f:
            params = pickle.load(f)
        self.policy_params = params['policy']
        self.value_params = params['value']
    
    def get_value_estimate(self, obs: Optional[np.ndarray] = None) -> float:
        """Get value estimate for current or given observation"""
        if obs is None:
            obs = self.get_observations()
        
        obs_jax = jnp.expand_dims(jnp.array(obs), axis=0)
        value = self._predict_value_jit(self.value_params, obs_jax)
        return float(jnp.squeeze(value))