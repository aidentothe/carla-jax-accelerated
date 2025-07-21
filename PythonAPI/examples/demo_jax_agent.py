#!/usr/bin/env python

"""
Demo of JAX-based Neural Network Agent for CARLA
Demonstrates JAX neural network without requiring CARLA to be running
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from typing import Dict, Tuple

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


class JAXAgentDemo:
    """Demo JAX-based agent for autonomous driving"""
    
    def __init__(self):
        # Network dimensions
        self.obs_dim = 12  # Speed, position, waypoints, etc.
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
        
        print("✅ JAX Agent initialized successfully!")
        print(f"   - Observation dimension: {self.obs_dim}")
        print(f"   - Action dimension: {self.action_dim}")
        print(f"   - Hidden dimension: {self.hidden_dim}")
        try:
            # JAX v0.6.0+ uses jax.tree.leaves
            policy_size = sum(p.size for p in jax.tree.leaves(self.policy_params))
            value_size = sum(p.size for p in jax.tree.leaves(self.value_params))
        except AttributeError:
            # Fallback for older JAX versions
            policy_size = sum(p.size for p in jax.tree_util.tree_leaves(self.policy_params))
            value_size = sum(p.size for p in jax.tree_util.tree_leaves(self.value_params))
        
        print(f"   - Policy parameters: {policy_size:,}")
        print(f"   - Value parameters: {value_size:,}")
    
    def _predict_action_fn(self, params, obs):
        """JIT-compiled action prediction"""
        return policy_forward(params, obs)
    
    def _predict_value_fn(self, params, obs):
        """JIT-compiled value prediction"""
        return value_forward(params, obs)
    
    def create_mock_observation(self) -> np.ndarray:
        """Create a mock observation vector"""
        # Simulate realistic driving observations
        obs = np.array([
            0.5,   # Normalized speed (50% of max)
            0.0,   # Lateral velocity (no sliding)
            0.1,   # Distance to lane center (slightly off-center)
            0.98,  # Sin(yaw_diff) - mostly aligned with lane
            0.2,   # Cos(yaw_diff)
            -0.1,  # Speed difference (slightly below target)
            2.0,   # Next waypoint x (2m ahead)
            0.5,   # Next waypoint y (slight curve)
            0.0,   # Pitch (flat road)
            0.0,   # Roll (no banking)
            0.0,   # Vertical velocity
            0.5    # Normalized height
        ], dtype=np.float32)
        
        return obs
    
    def run_demo_step(self, obs: np.ndarray) -> Tuple[np.ndarray, float]:
        """Run one prediction step"""
        # Convert to JAX array and add batch dimension
        obs_jax = jnp.expand_dims(jnp.array(obs), axis=0)
        
        # Get action from policy network (JIT compiled)
        action = self._predict_action(self.policy_params, obs_jax)
        action = jnp.squeeze(action, axis=0)  # Remove batch dimension
        
        # Get value estimate
        value = self._predict_value(self.value_params, obs_jax)
        
        return np.array(action), float(value)
    
    def demonstrate_batch_processing(self, batch_size: int = 32):
        """Demonstrate JAX's batch processing capabilities"""
        print(f"\n🚀 Demonstrating batch processing with {batch_size} observations...")
        
        # Create batch of random observations
        batch_obs = jnp.array(np.random.randn(batch_size, self.obs_dim).astype(np.float32))
        
        # Time the batch prediction
        import time
        
        # Warm up JIT
        _ = self._predict_action(self.policy_params, batch_obs[:1])
        
        start_time = time.time()
        batch_actions = self._predict_action(self.policy_params, batch_obs)
        jax.block_until_ready(batch_actions)  # Ensure computation completes
        end_time = time.time()
        
        batch_time = end_time - start_time
        per_sample_time = batch_time / batch_size * 1000  # Convert to ms
        
        print(f"   - Batch processing time: {batch_time:.4f} seconds")
        print(f"   - Per sample time: {per_sample_time:.4f} ms")
        print(f"   - Effective throughput: {batch_size / batch_time:.0f} samples/second")
        
        return batch_actions
    
    def demonstrate_vmap(self):
        """Demonstrate JAX's vmap for vectorized operations"""
        print(f"\n📊 Demonstrating vmap for vectorized reward computation...")
        
        # Create batch of states and actions
        batch_size = 100
        states = jnp.array(np.random.randn(batch_size, self.obs_dim).astype(np.float32))
        actions = jnp.array(np.random.randn(batch_size, self.action_dim).astype(np.float32))
        
        # Define a simple reward function
        def compute_reward(state, action):
            speed = state[0]  # Normalized speed
            steering = action[0]  # Steering action
            
            # Reward for maintaining good speed and smooth steering
            speed_reward = jnp.exp(-jnp.abs(speed - 0.5))  # Target 50% speed
            smooth_reward = jnp.exp(-jnp.abs(steering))     # Penalty for large steering
            
            return 0.7 * speed_reward + 0.3 * smooth_reward
        
        # Vectorize using vmap
        batch_compute_reward = jax.vmap(compute_reward, in_axes=(0, 0))
        
        # Time the computation
        import time
        start_time = time.time()
        rewards = batch_compute_reward(states, actions)
        jax.block_until_ready(rewards)
        end_time = time.time()
        
        print(f"   - Computed {batch_size} rewards in {(end_time - start_time) * 1000:.2f} ms")
        print(f"   - Average reward: {jnp.mean(rewards):.4f}")
        print(f"   - Reward std: {jnp.std(rewards):.4f}")


def main():
    """Main demo function"""
    print("=" * 60)
    print("🏎️  JAX-based Autonomous Driving Agent Demo")
    print("=" * 60)
    
    # Initialize agent
    agent = JAXAgentDemo()
    
    # Create mock observation
    print("\n📊 Creating mock driving observation...")
    obs = agent.create_mock_observation()
    print(f"   Observation shape: {obs.shape}")
    print(f"   Observation values: {obs}")
    
    # Run prediction
    print("\n🧠 Running neural network prediction...")
    action, value = agent.run_demo_step(obs)
    
    print(f"   Predicted action: {action}")
    print(f"   - Steering: {action[0]:.4f} (range: [-1, 1])")
    print(f"   - Throttle/Brake: {action[1]:.4f} (+ = throttle, - = brake)")
    print(f"   Value estimate: {value:.4f}")
    
    # Interpret the action
    if action[1] > 0:
        throttle_brake = f"Throttle: {action[1]:.2f}"
    else:
        throttle_brake = f"Brake: {-action[1]:.2f}"
    
    if action[0] > 0.1:
        steering = "Turn RIGHT"
    elif action[0] < -0.1:
        steering = "Turn LEFT"
    else:
        steering = "Go STRAIGHT"
    
    print(f"\n🚗 Driving command: {steering}, {throttle_brake}")
    
    # Demonstrate batch processing
    agent.demonstrate_batch_processing(batch_size=64)
    
    # Demonstrate vmap
    agent.demonstrate_vmap()
    
    print(f"\n✨ Demo completed successfully!")
    print(f"   - JAX version: {jax.__version__}")
    print(f"   - Available devices: {jax.devices()}")
    print(f"   - Using device: {jax.devices()[0].device_kind}")
    
    print("\n🎯 This demonstrates key JAX features for autonomous driving:")
    print("   ✅ JIT compilation for fast inference")
    print("   ✅ Batch processing for multiple scenarios")
    print("   ✅ Vectorized operations with vmap")
    print("   ✅ Hardware acceleration ready (GPU/TPU)")
    print("   ✅ Pure functional programming for reproducibility")


if __name__ == '__main__':
    main()