#!/usr/bin/env python

"""
Training script for JAX-based CARLA agent using PPO (Proximal Policy Optimization)
Demonstrates modern RL training with JAX's acceleration capabilities
"""

import argparse
import glob
import os
import sys
import time
from typing import Dict, List, Tuple, Any
import numpy as np
import pickle

import jax
import jax.numpy as jnp
from flax.training import train_state
from flax import struct
import optax
from functools import partial

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
from jax_agent import JAXCARLAAgent, PolicyNetwork, ValueNetwork


@struct.dataclass
class PPOTrainState(train_state.TrainState):
    """Extended train state for PPO"""
    target_params: Any = None
    
    
class ReplayBuffer:
    """Simple replay buffer for storing experiences"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state: np.ndarray, action: np.ndarray, reward: float, 
             next_state: np.ndarray, done: bool):
        """Add experience to buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample batch of experiences"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in indices:
            s, a, r, ns, d = self.buffer[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class PPOTrainer:
    """PPO trainer for JAX-based CARLA agent"""
    
    def __init__(self, obs_dim: int = 18, action_dim: int = 2,
                 learning_rate: float = 3e-4, gamma: float = 0.99,
                 clip_epsilon: float = 0.2, value_coef: float = 0.5,
                 entropy_coef: float = 0.01):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Initialize networks
        self.policy_net = PolicyNetwork(action_dim=action_dim)
        self.value_net = ValueNetwork()
        
        # Initialize optimizer
        self.optimizer = optax.adam(learning_rate)
        
        # Initialize parameters
        key = jax.random.PRNGKey(42)
        key, policy_key, value_key = jax.random.split(key, 3)
        dummy_obs = jnp.zeros((1, obs_dim))
        
        policy_params = self.policy_net.init(policy_key, dummy_obs)
        value_params = self.value_net.init(value_key, dummy_obs)
        
        # Combine parameters
        params = {'policy': policy_params, 'value': value_params}
        
        # Create train state
        self.train_state = PPOTrainState.create(
            apply_fn=None,
            params=params,
            tx=self.optimizer,
            target_params=params
        )
        
        # JIT compile training step
        self.train_step = jax.jit(self._train_step)
    
    def compute_gae(self, rewards: jnp.ndarray, values: jnp.ndarray, 
                    dones: jnp.ndarray, next_value: jnp.ndarray,
                    gae_lambda: float = 0.95) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        # Work backwards through the trajectory
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = jnp.array(advantages)
        returns = advantages + values
        
        return advantages, returns
    
    def _ppo_loss(self, params: Dict, states: jnp.ndarray, actions: jnp.ndarray,
                  old_log_probs: jnp.ndarray, advantages: jnp.ndarray,
                  returns: jnp.ndarray) -> jnp.ndarray:
        """Compute PPO loss"""
        # Get new predictions
        new_actions = self.policy_net.apply(params['policy'], states)
        new_values = self.value_net.apply(params['value'], states)
        
        # Compute log probabilities (simplified - assumes deterministic policy)
        # In a full implementation, you'd use a stochastic policy
        new_log_probs = -jnp.sum((new_actions - actions) ** 2, axis=1)
        
        # Policy loss (PPO clip)
        ratio = jnp.exp(new_log_probs - old_log_probs)
        clipped_ratio = jnp.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped_ratio * advantages))
        
        # Value loss
        value_loss = jnp.mean((new_values.squeeze() - returns) ** 2)
        
        # Entropy bonus (simplified)
        entropy = -jnp.mean(new_log_probs)
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        return total_loss
    
    def _train_step(self, train_state: PPOTrainState, batch: Tuple) -> Tuple[PPOTrainState, Dict]:
        """Single training step"""
        states, actions, old_log_probs, advantages, returns = batch
        
        # Compute loss and gradients
        loss_fn = partial(self._ppo_loss, states=states, actions=actions,
                         old_log_probs=old_log_probs, advantages=advantages,
                         returns=returns)
        
        loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
        
        # Update parameters
        train_state = train_state.apply_gradients(grads=grads)
        
        return train_state, {'loss': loss}
    
    def update(self, trajectories: List[Dict]) -> Dict:
        """Update policy using collected trajectories"""
        # Prepare batch data
        all_states = []
        all_actions = []
        all_rewards = []
        all_values = []
        all_dones = []
        
        for traj in trajectories:
            all_states.extend(traj['states'])
            all_actions.extend(traj['actions'])
            all_rewards.extend(traj['rewards'])
            all_values.extend(traj['values'])
            all_dones.extend(traj['dones'])
        
        states = jnp.array(all_states)
        actions = jnp.array(all_actions)
        rewards = jnp.array(all_rewards)
        values = jnp.array(all_values)
        dones = jnp.array(all_dones)
        
        # Compute advantages and returns
        # Simplified - in practice, compute per trajectory
        advantages = rewards - values  # Simplified advantage
        returns = rewards  # Simplified returns
        
        # Normalize advantages
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
        
        # Compute old log probs (simplified)
        old_actions = self.policy_net.apply(self.train_state.params['policy'], states)
        old_log_probs = -jnp.sum((old_actions - actions) ** 2, axis=1)
        
        # Perform multiple epochs of PPO updates
        for _ in range(4):  # PPO epochs
            # Create batch
            batch = (states, actions, old_log_probs, advantages, returns)
            
            # Train step
            self.train_state, metrics = self.train_step(self.train_state, batch)
        
        return metrics
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint"""
        checkpoint = {
            'params': self.train_state.params,
            'opt_state': self.train_state.opt_state,
            'step': self.train_state.step
        }
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.train_state = self.train_state.replace(
            params=checkpoint['params'],
            opt_state=checkpoint['opt_state'],
            step=checkpoint['step']
        )


def collect_trajectory(env, agent: JAXCARLAAgent, max_steps: int = 1000) -> Dict:
    """Collect a single trajectory using the agent"""
    trajectory = {
        'states': [],
        'actions': [],
        'rewards': [],
        'values': [],
        'dones': []
    }
    
    # Reset environment (simplified - in practice, properly reset CARLA)
    obs = agent.get_observations()
    
    for step in range(max_steps):
        # Get action from agent
        control = agent.run_step()
        action = np.array([control.steer, 
                          control.throttle if control.throttle > 0 else -control.brake])
        
        # Get value estimate
        value = agent.get_value_estimate(obs)
        
        # Store experience
        trajectory['states'].append(obs)
        trajectory['actions'].append(action)
        trajectory['values'].append(value)
        
        # Execute action in environment
        # In practice, apply control to CARLA and get next observation
        # For now, simulate reward
        reward = compute_reward(agent.vehicle, agent.target_speed)
        
        # Get next observation
        next_obs = agent.get_observations()
        
        # Check if done
        done = agent.done()
        
        trajectory['rewards'].append(reward)
        trajectory['dones'].append(done)
        
        if done:
            break
        
        obs = next_obs
    
    return trajectory


def compute_reward(vehicle, target_speed: float) -> float:
    """Compute reward based on current state"""
    # Get vehicle state
    velocity = vehicle.get_velocity()
    speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    
    # Speed reward
    speed_error = abs(speed - target_speed / 3.6)  # Convert km/h to m/s
    speed_reward = np.exp(-0.1 * speed_error)
    
    # Lane keeping reward (simplified)
    transform = vehicle.get_transform()
    location = transform.location
    waypoint = vehicle.get_world().get_map().get_waypoint(location)
    distance_to_center = location.distance(waypoint.transform.location)
    lane_reward = np.exp(-0.5 * distance_to_center)
    
    # Combine rewards
    reward = 0.7 * speed_reward + 0.3 * lane_reward
    
    return reward


def main():
    """Main training loop"""
    argparser = argparse.ArgumentParser(description='Train JAX-based CARLA Agent')
    argparser.add_argument('--host', default='127.0.0.1', help='IP of the host server')
    argparser.add_argument('--port', default=2000, type=int, help='TCP port')
    argparser.add_argument('--episodes', default=100, type=int, help='Number of training episodes')
    argparser.add_argument('--checkpoint-dir', default='checkpoints', help='Directory to save checkpoints')
    argparser.add_argument('--save-freq', default=10, type=int, help='Save checkpoint every N episodes')
    
    args = argparser.parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = PPOTrainer()
    
    print("PPO Trainer initialized")
    print(f"Training for {args.episodes} episodes")
    
    # Training loop (simplified - in practice, integrate with CARLA)
    for episode in range(args.episodes):
        print(f"\nEpisode {episode + 1}/{args.episodes}")
        
        # Collect trajectories
        # In practice, you would:
        # 1. Connect to CARLA server
        # 2. Spawn vehicle
        # 3. Create JAXCARLAAgent with current parameters
        # 4. Collect trajectory
        # 5. Clean up
        
        # For now, we'll create dummy trajectories
        trajectories = []
        for _ in range(4):  # Collect 4 trajectories per update
            # Create dummy trajectory
            traj = {
                'states': np.random.randn(100, 18).astype(np.float32),
                'actions': np.random.randn(100, 2).astype(np.float32),
                'rewards': np.random.randn(100).astype(np.float32),
                'values': np.random.randn(100).astype(np.float32),
                'dones': np.zeros(100, dtype=bool)
            }
            traj['dones'][-1] = True
            trajectories.append(traj)
        
        # Update policy
        metrics = trainer.update(trajectories)
        print(f"Loss: {metrics['loss']:.4f}")
        
        # Save checkpoint
        if (episode + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_{episode + 1}.pkl')
            trainer.save_checkpoint(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(args.checkpoint_dir, 'final_model.pkl')
    trainer.save_checkpoint(final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")
    
    # Save just the model parameters for the agent
    agent_params_path = os.path.join(args.checkpoint_dir, 'agent_params.pkl')
    agent_params = {
        'policy': trainer.train_state.params['policy'],
        'value': trainer.train_state.params['value']
    }
    with open(agent_params_path, 'wb') as f:
        pickle.dump(agent_params, f)
    print(f"Agent parameters saved to {agent_params_path}")


if __name__ == '__main__':
    main()