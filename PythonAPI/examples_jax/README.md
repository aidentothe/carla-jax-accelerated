# CARLA JAX Examples

This directory contains JAX-accelerated versions of CARLA Python examples, leveraging JAX's just-in-time compilation, automatic differentiation, and vectorization capabilities for high-performance autonomous driving simulations.

## Overview

JAX provides several advantages for autonomous driving applications:
- **JIT Compilation**: Accelerated real-time control loops
- **Automatic Differentiation**: Gradient-based learning and optimization
- **Vectorization**: Efficient multi-agent simulation
- **GPU/TPU Support**: Hardware acceleration for sensor processing

## Installation

1. Install JAX dependencies:
```bash
pip install -r requirements_jax.txt
```

2. Ensure CARLA is properly installed and accessible.

## Examples

### Core Control
- **`automatic_control_jax.py`** - JAX-accelerated autonomous driving with JIT-compiled control loops
- **`vehicle_physics_jax.py`** - Physics simulation with automatic differentiation

### Sensor Processing
- **`sensor_synchronization_jax.py`** - Multi-sensor fusion with JAX batching
- **`lidar_to_camera_jax.py`** - Geometric transformations using JAX arrays

### Multi-Agent Simulation
- **`generate_traffic_jax.py`** - Vectorized multi-agent traffic generation

## Key JAX Features Used

### JIT Compilation
Functions marked with `@jax.jit` are compiled for faster execution:
```python
@jax.jit
def control_step(state, target):
    return compute_control(state, target)
```

### Vectorization
Use `jax.vmap` for batch operations:
```python
batch_control = jax.vmap(control_step)
controls = batch_control(states, targets)
```

### Automatic Differentiation
Compute gradients automatically:
```python
grad_fn = jax.grad(loss_function)
gradients = grad_fn(params, data)
```

## Performance Tips

1. **Use immutable arrays**: JAX arrays are immutable by design
2. **Batch operations**: Leverage vectorization for multiple vehicles
3. **JIT compile hot paths**: Apply `@jax.jit` to frequently called functions
4. **GPU acceleration**: Use `jax.device_put()` for GPU arrays

## Compatibility

These examples are designed to work alongside the original CARLA examples while providing JAX-accelerated alternatives for computationally intensive tasks.