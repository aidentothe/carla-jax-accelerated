# JAX-based Autonomous Driving Agent for CARLA

## 🎯 Overview

This project demonstrates a modern JAX-based neural network agent for autonomous driving in CARLA, showcasing the same technology used by companies like Waymo in their Waymax simulator.

## ✅ What's Working

### 1. **Pure JAX Neural Network Agent** (`demo_jax_agent.py`)
- ✅ **JIT-compiled inference**: Lightning-fast neural network predictions
- ✅ **Batch processing**: Process multiple scenarios simultaneously using `vmap`
- ✅ **Modern architecture**: Policy and value networks for continuous control
- ✅ **Hardware acceleration ready**: Works on CPU, GPU, and TPU
- ✅ **18,434 policy parameters** and **18,305 value parameters**

**Performance Metrics:**
- **707 samples/second** throughput on CPU
- **1.4ms per sample** inference time
- **JIT compilation** for production-speed inference

### 2. **JAX Agent Implementation** (`jax_agent_simple.py`)
- ✅ Pure JAX implementation (no Flax dependencies)
- ✅ Compatible with CARLA's control interface
- ✅ 12-dimensional observation space (speed, position, waypoints, etc.)
- ✅ 2-dimensional action space (steering, throttle/brake)
- ✅ Xavier weight initialization
- ✅ ReLU activations with Tanh output

### 3. **Training Infrastructure** (`train_jax_agent.py`)
- ✅ PPO (Proximal Policy Optimization) implementation
- ✅ Generalized Advantage Estimation (GAE)
- ✅ JAX-native automatic differentiation
- ✅ Checkpoint saving/loading system

### 4. **Utility Functions** (`jax_utils.py`)
- ✅ Waymax-inspired trajectory processing
- ✅ Performance monitoring tools
- ✅ Batch reward computation using `vmap`
- ✅ Safety checking for actions

## ⚠️ Current Issue: CARLA Integration

The full CARLA integration (`automatic_control.py --agent JAX`) isn't working due to:

**Python Version Mismatch:**
- System has Python 3.13
- CARLA only provides eggs for Python 2.7 and 3.7
- Using Python 3.7 egg with 3.13 causes segmentation faults

## 🚀 Demo the JAX Agent

Run the working demo:

```bash
python3 demo_jax_agent.py
```

This demonstrates:
- **Neural network initialization** with proper weight initialization
- **Mock observation processing** (speed, lane position, waypoints)
- **Action prediction** (steering + throttle/brake commands)
- **Batch processing** with 64 samples at 707 samples/second
- **Vectorized reward computation** for 100 state-action pairs

## 🧠 Technical Highlights

### JAX Features Demonstrated

1. **JIT Compilation**
   ```python
   self._predict_action = jax.jit(self._predict_action_fn)
   ```

2. **Vectorized Operations**
   ```python
   batch_compute_reward = jax.vmap(compute_reward, in_axes=(0, 0))
   ```

3. **Automatic Differentiation**
   ```python
   loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
   ```

### Network Architecture

- **Input**: 12D observation (speed, position, lane info, waypoints)
- **Hidden**: 128 units with ReLU activation
- **Output**: 2D action (steering ∈ [-1,1], throttle/brake ∈ [-1,1])
- **Activation**: Tanh output for bounded actions

### Observation Space

```python
obs = [
    speed / 20.0,           # Normalized speed
    lateral_speed / 5.0,    # Lateral velocity (sliding)
    distance_to_center / 3.0, # Distance to lane center
    sin(yaw_diff),          # Lane alignment (sin)
    cos(yaw_diff),          # Lane alignment (cos)
    speed_diff / 10.0,      # Target speed difference
    waypoint_x / 10.0,      # Next waypoint x
    waypoint_y / 10.0,      # Next waypoint y
    pitch / 45.0,           # Vehicle pitch
    roll / 45.0,            # Vehicle roll
    velocity_z / 5.0,       # Vertical velocity
    height / 100.0          # Normalized height
]
```

## 🏆 Why This Matters for Waymo

This implementation demonstrates understanding of:

1. **Modern RL Architecture**: Policy-value networks like those used in autonomous driving
2. **JAX Ecosystem**: Same technology stack as Waymo's Waymax simulator
3. **Performance Optimization**: JIT compilation and vectorization for production systems
4. **Scalability**: Batch processing for training multiple scenarios
5. **Industry Best Practices**: Proper observation design and action spaces

## 📈 Performance Results

```
🏎️ JAX Agent Performance:
✅ 707 samples/second throughput
✅ 1.4ms per sample inference time  
✅ 18,434 trainable parameters
✅ Hardware acceleration ready
✅ Batch processing with vmap
```

## 🔧 Next Steps

To fully integrate with CARLA:

1. **Install compatible CARLA version** with Python 3.13 support
2. **Connect trained model** to CARLA's vehicle control interface
3. **Implement data collection** pipeline for training
4. **Deploy on GPU/TPU** for maximum performance

The core JAX implementation is ready - only the CARLA interface integration remains!