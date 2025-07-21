"""JAX-accelerated examples for CARLA autonomous driving simulation.

This package provides high-performance implementations of CARLA examples
using JAX for just-in-time compilation, automatic differentiation, and
vectorized operations.

Key Features:
- JIT-compiled control algorithms for real-time performance
- Vectorized multi-agent simulation using vmap
- Automatic differentiation for learning and optimization
- GPU/TPU acceleration support
- Efficient sensor processing and fusion

Examples:
- automatic_control_jax.py: JAX-accelerated vehicle control
- sensor_synchronization_jax.py: Multi-sensor fusion with batching
- vehicle_physics_jax.py: Physics simulation with autodiff
- lidar_to_camera_jax.py: Geometric transformations
- generate_traffic_jax.py: Vectorized traffic simulation

Requirements:
- JAX >= 0.4.20
- JAXlib >= 0.4.20
- CARLA >= 0.9.14
- NumPy >= 1.21.0
"""

__version__ = "1.0.0"
__author__ = "CARLA JAX Examples"

# Import core utilities
from .jax_utils import (
    ensure_jax_device,
    profile_jax_function,
    JAXProfiler,
    convert_carla_transform,
    convert_carla_vector,
    batch_process_with_jax,
    safe_jit_compile
)

__all__ = [
    "ensure_jax_device",
    "profile_jax_function", 
    "JAXProfiler",
    "convert_carla_transform",
    "convert_carla_vector",
    "batch_process_with_jax",
    "safe_jit_compile"
]