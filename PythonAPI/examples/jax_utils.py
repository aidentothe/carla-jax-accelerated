#!/usr/bin/env python

"""
Utility functions for JAX-based CARLA agents
Includes data processing, visualization, and performance utilities
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Callable
import matplotlib.pyplot as plt
from functools import partial
import time


# Waymax-inspired utilities for trajectory processing
@jax.jit
def normalize_angles(angles: jnp.ndarray) -> jnp.ndarray:
    """Normalize angles to [-pi, pi]"""
    return jnp.arctan2(jnp.sin(angles), jnp.cos(angles))


@jax.jit
def compute_relative_positions(ego_pos: jnp.ndarray, ego_yaw: float, 
                              other_positions: jnp.ndarray) -> jnp.ndarray:
    """Transform positions to ego vehicle's coordinate frame"""
    # Translate to ego origin
    relative_pos = other_positions - ego_pos
    
    # Rotate to ego frame
    cos_yaw = jnp.cos(-ego_yaw)
    sin_yaw = jnp.sin(-ego_yaw)
    
    x_rot = relative_pos[..., 0] * cos_yaw - relative_pos[..., 1] * sin_yaw
    y_rot = relative_pos[..., 0] * sin_yaw + relative_pos[..., 1] * cos_yaw
    
    return jnp.stack([x_rot, y_rot], axis=-1)


@jax.jit
def compute_trajectory_features(positions: jnp.ndarray, 
                               velocities: jnp.ndarray,
                               yaws: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """Extract features from trajectory data"""
    # Compute speeds
    speeds = jnp.linalg.norm(velocities, axis=-1)
    
    # Compute accelerations (finite differences)
    accelerations = jnp.diff(velocities, axis=0, prepend=velocities[0:1])
    
    # Compute yaw rates
    yaw_rates = jnp.diff(yaws, axis=0, prepend=yaws[0:1])
    yaw_rates = normalize_angles(yaw_rates)
    
    # Compute curvature
    curvature = yaw_rates / (speeds + 1e-6)
    
    return {
        'speeds': speeds,
        'accelerations': accelerations,
        'yaw_rates': yaw_rates,
        'curvature': curvature
    }


# Batch processing utilities using vmap
@partial(jax.vmap, in_axes=(0, None))
def batch_normalize_observations(obs: jnp.ndarray, stats: Dict) -> jnp.ndarray:
    """Normalize observations using pre-computed statistics"""
    return (obs - stats['mean']) / (stats['std'] + 1e-8)


@partial(jax.vmap, in_axes=(0, 0, None))
def batch_compute_rewards(states: jnp.ndarray, actions: jnp.ndarray, 
                         reward_config: Dict) -> jnp.ndarray:
    """Compute rewards for a batch of state-action pairs"""
    # Speed reward
    speed = states[..., 0] * 30.0  # Denormalize speed
    target_speed = reward_config['target_speed'] / 3.6  # km/h to m/s
    speed_error = jnp.abs(speed - target_speed)
    speed_reward = jnp.exp(-0.1 * speed_error)
    
    # Lane keeping reward
    lane_distance = jnp.abs(states[..., 2]) * 5.0  # Denormalize
    lane_reward = jnp.exp(-0.5 * lane_distance)
    
    # Smooth control reward
    steer_penalty = jnp.abs(actions[..., 0])
    throttle_penalty = jnp.abs(jnp.diff(actions[..., 1], prepend=actions[0, 1]))
    control_reward = jnp.exp(-0.2 * (steer_penalty + throttle_penalty))
    
    # Combine rewards
    total_reward = (reward_config['speed_weight'] * speed_reward +
                   reward_config['lane_weight'] * lane_reward +
                   reward_config['control_weight'] * control_reward)
    
    return total_reward


# Performance monitoring utilities
class PerformanceMonitor:
    """Monitor JAX computation performance"""
    
    def __init__(self):
        self.timings = {}
        self.counts = {}
    
    @staticmethod
    def timed_function(func: Callable, name: str = None) -> Callable:
        """Decorator to time JAX functions"""
        if name is None:
            name = func.__name__
        
        def wrapper(*args, **kwargs):
            # Block until all JAX operations complete
            jax.block_until_ready(func(*args, **kwargs))
            
            start_time = time.time()
            result = func(*args, **kwargs)
            jax.block_until_ready(result)
            end_time = time.time()
            
            elapsed = end_time - start_time
            print(f"{name} took {elapsed:.4f} seconds")
            
            return result
        
        return wrapper
    
    def start_timer(self, name: str):
        """Start timing a section"""
        self.timings[name] = time.time()
    
    def end_timer(self, name: str):
        """End timing and record result"""
        if name in self.timings:
            elapsed = time.time() - self.timings[name]
            if name not in self.counts:
                self.counts[name] = []
            self.counts[name].append(elapsed)
            del self.timings[name]
    
    def get_average_time(self, name: str) -> float:
        """Get average time for a named section"""
        if name in self.counts:
            return np.mean(self.counts[name])
        return 0.0
    
    def print_summary(self):
        """Print performance summary"""
        print("\n=== Performance Summary ===")
        for name, times in self.counts.items():
            avg_time = np.mean(times)
            std_time = np.std(times)
            print(f"{name}: {avg_time:.4f} ± {std_time:.4f} seconds "
                  f"(n={len(times)})")


# Visualization utilities
def plot_trajectory(positions: np.ndarray, velocities: np.ndarray,
                   actions: np.ndarray, rewards: np.ndarray,
                   save_path: Optional[str] = None):
    """Visualize agent trajectory and performance"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Trajectory plot
    ax = axes[0, 0]
    ax.plot(positions[:, 0], positions[:, 1], 'b-', label='Trajectory')
    ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start')
    ax.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=10, label='End')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Vehicle Trajectory')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    
    # Speed profile
    ax = axes[0, 1]
    speeds = np.linalg.norm(velocities, axis=-1) * 3.6  # m/s to km/h
    ax.plot(speeds, 'b-')
    ax.axhline(y=30, color='r', linestyle='--', label='Target Speed')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Speed (km/h)')
    ax.set_title('Speed Profile')
    ax.legend()
    ax.grid(True)
    
    # Control actions
    ax = axes[1, 0]
    ax.plot(actions[:, 0], 'b-', label='Steering')
    ax.plot(actions[:, 1], 'r-', label='Throttle/Brake')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Control Value')
    ax.set_title('Control Actions')
    ax.legend()
    ax.grid(True)
    ax.set_ylim(-1.1, 1.1)
    
    # Rewards
    ax = axes[1, 1]
    ax.plot(rewards, 'g-')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Reward')
    ax.set_title('Reward Over Time')
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    
    plt.close()


# Data collection utilities
class ExperienceCollector:
    """Collect and preprocess experiences for training"""
    
    def __init__(self, max_episodes: int = 1000):
        self.episodes = []
        self.max_episodes = max_episodes
    
    def add_episode(self, trajectory: Dict):
        """Add a complete episode"""
        if len(self.episodes) >= self.max_episodes:
            self.episodes.pop(0)  # Remove oldest
        self.episodes.append(trajectory)
    
    def get_statistics(self) -> Dict:
        """Compute statistics over collected episodes"""
        all_states = []
        all_rewards = []
        episode_lengths = []
        
        for episode in self.episodes:
            all_states.extend(episode['states'])
            all_rewards.extend(episode['rewards'])
            episode_lengths.append(len(episode['states']))
        
        states_array = np.array(all_states)
        rewards_array = np.array(all_rewards)
        
        return {
            'state_mean': np.mean(states_array, axis=0),
            'state_std': np.std(states_array, axis=0),
            'mean_reward': np.mean(rewards_array),
            'mean_episode_length': np.mean(episode_lengths),
            'total_episodes': len(self.episodes)
        }
    
    def sample_batch(self, batch_size: int) -> Dict:
        """Sample a batch of experiences"""
        # Sample episodes
        episode_indices = np.random.choice(len(self.episodes), 
                                         size=min(batch_size, len(self.episodes)),
                                         replace=True)
        
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []
        
        for idx in episode_indices:
            episode = self.episodes[idx]
            # Sample a transition from the episode
            t = np.random.randint(len(episode['states']) - 1)
            
            batch_states.append(episode['states'][t])
            batch_actions.append(episode['actions'][t])
            batch_rewards.append(episode['rewards'][t])
            batch_next_states.append(episode['states'][t + 1])
            batch_dones.append(episode['dones'][t])
        
        return {
            'states': np.array(batch_states),
            'actions': np.array(batch_actions),
            'rewards': np.array(batch_rewards),
            'next_states': np.array(batch_next_states),
            'dones': np.array(batch_dones)
        }


# Model evaluation utilities
@jax.jit
def evaluate_policy(policy_params, value_params, policy_fn, value_fn, 
                   test_states: jnp.ndarray) -> Dict:
    """Evaluate policy on test states"""
    # Get policy predictions
    actions = policy_fn(policy_params, test_states)
    
    # Get value predictions
    values = value_fn(value_params, test_states)
    
    # Compute action statistics
    action_mean = jnp.mean(actions, axis=0)
    action_std = jnp.std(actions, axis=0)
    
    # Compute value statistics
    value_mean = jnp.mean(values)
    value_std = jnp.std(values)
    
    return {
        'action_mean': action_mean,
        'action_std': action_std,
        'value_mean': value_mean,
        'value_std': value_std
    }


# Safety checking utilities
def check_action_safety(action: np.ndarray, current_speed: float,
                       safety_config: Dict) -> Tuple[np.ndarray, bool]:
    """Check and potentially modify actions for safety"""
    safe_action = action.copy()
    is_safe = True
    
    # Limit steering based on speed
    max_steer = safety_config['max_steer_low_speed']
    if current_speed > safety_config['speed_threshold']:
        # Reduce max steering at high speed
        speed_factor = safety_config['speed_threshold'] / current_speed
        max_steer *= speed_factor
    
    if abs(safe_action[0]) > max_steer:
        safe_action[0] = np.clip(safe_action[0], -max_steer, max_steer)
        is_safe = False
    
    # Limit acceleration/deceleration
    if safe_action[1] > safety_config['max_throttle']:
        safe_action[1] = safety_config['max_throttle']
        is_safe = False
    elif safe_action[1] < -safety_config['max_brake']:
        safe_action[1] = -safety_config['max_brake']
        is_safe = False
    
    return safe_action, is_safe


# Default configurations
DEFAULT_REWARD_CONFIG = {
    'target_speed': 30.0,  # km/h
    'speed_weight': 0.4,
    'lane_weight': 0.4,
    'control_weight': 0.2
}

DEFAULT_SAFETY_CONFIG = {
    'max_steer_low_speed': 0.8,
    'speed_threshold': 50.0,  # km/h
    'max_throttle': 0.8,
    'max_brake': 1.0
}