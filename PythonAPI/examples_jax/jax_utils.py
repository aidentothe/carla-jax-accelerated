"""Utility functions for JAX integration with CARLA.

This module provides common utilities for working with JAX in CARLA environments,
including device management, profiling, data conversion, and performance optimization.
"""

import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from contextlib import contextmanager
import functools

try:
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax import jit, device_put, devices
    import chex
except ImportError:
    raise RuntimeError('JAX not found. Install with: pip install jax jaxlib chex')

try:
    import carla
except ImportError:
    warnings.warn("CARLA not available. Some functions may not work.")
    carla = None


# ==============================================================================
# -- Device Management --------------------------------------------------------
# ==============================================================================

def ensure_jax_device(preferred_device: str = 'gpu') -> str:
    """Ensure JAX is running on the preferred device.
    
    Args:
        preferred_device: 'gpu', 'tpu', or 'cpu'
        
    Returns:
        str: The actual device being used
    """
    available_devices = devices()
    
    if preferred_device == 'gpu' and any('gpu' in str(d) for d in available_devices):
        print(f"JAX using GPU: {[d for d in available_devices if 'gpu' in str(d)]}")
        return 'gpu'
    elif preferred_device == 'tpu' and any('tpu' in str(d) for d in available_devices):
        print(f"JAX using TPU: {[d for d in available_devices if 'tpu' in str(d)]}")
        return 'tpu'
    else:
        print(f"JAX using CPU: {[d for d in available_devices if 'cpu' in str(d)]}")
        return 'cpu'


def get_device_info() -> Dict[str, Any]:
    """Get information about available JAX devices."""
    device_info = {
        'devices': devices(),
        'device_count': len(devices()),
        'default_backend': jax.default_backend(),
        'available_backends': jax.lib.xla_bridge.get_backend().platform
    }
    
    # Check for GPU memory if available
    try:
        if any('gpu' in str(d) for d in devices()):
            device_info['gpu_memory'] = jax.lib.xla_bridge.get_backend('gpu').get_memory_info()
    except:
        pass
        
    return device_info


# ==============================================================================
# -- Performance Profiling ----------------------------------------------------
# ==============================================================================

class JAXProfiler:
    """Simple profiler for JAX functions."""
    
    def __init__(self):
        self.timings = {}
        self.call_counts = {}
        
    def profile_function(self, func_name: str):
        """Decorator to profile a JAX function."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                
                # Block until computation is complete (important for JAX async execution)
                if hasattr(result, 'block_until_ready'):
                    result.block_until_ready()
                elif isinstance(result, (tuple, list)):
                    for r in result:
                        if hasattr(r, 'block_until_ready'):
                            r.block_until_ready()
                
                elapsed = time.time() - start_time
                
                if func_name not in self.timings:
                    self.timings[func_name] = []
                    self.call_counts[func_name] = 0
                
                self.timings[func_name].append(elapsed)
                self.call_counts[func_name] += 1
                
                return result
            return wrapper
        return decorator
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get profiling statistics."""
        stats = {}
        for func_name, times in self.timings.items():
            stats[func_name] = {
                'mean_time_ms': np.mean(times) * 1000,
                'std_time_ms': np.std(times) * 1000,
                'min_time_ms': np.min(times) * 1000,
                'max_time_ms': np.max(times) * 1000,
                'total_time_s': np.sum(times),
                'call_count': self.call_counts[func_name],
                'calls_per_second': self.call_counts[func_name] / np.sum(times) if np.sum(times) > 0 else 0
            }
        return stats
    
    def print_stats(self):
        """Print profiling statistics."""
        stats = self.get_stats()
        print("JAX Function Profiling Statistics:")
        print("=" * 60)
        
        for func_name, func_stats in stats.items():
            print(f"\n{func_name}:")
            print(f"  Mean time: {func_stats['mean_time_ms']:.2f}ms")
            print(f"  Std time:  {func_stats['std_time_ms']:.2f}ms")
            print(f"  Min time:  {func_stats['min_time_ms']:.2f}ms")
            print(f"  Max time:  {func_stats['max_time_ms']:.2f}ms")
            print(f"  Total time: {func_stats['total_time_s']:.2f}s")
            print(f"  Call count: {func_stats['call_count']}")
            print(f"  Calls/sec: {func_stats['calls_per_second']:.1f}")
    
    def reset(self):
        """Reset all profiling data."""
        self.timings.clear()
        self.call_counts.clear()


def profile_jax_function(func_name: str, profiler: Optional[JAXProfiler] = None):
    """Decorator to profile a JAX function.
    
    Args:
        func_name: Name to use in profiling results
        profiler: JAXProfiler instance to use (creates new one if None)
    """
    if profiler is None:
        profiler = JAXProfiler()
    
    return profiler.profile_function(func_name)


@contextmanager
def jax_profiling_context():
    """Context manager for JAX profiling."""
    # Enable JAX profiling if available
    try:
        with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
            yield
    except AttributeError:
        # Fallback for older JAX versions
        yield


# ==============================================================================
# -- CARLA Data Conversion ----------------------------------------------------
# ==============================================================================

def convert_carla_transform(transform: 'carla.Transform') -> jnp.ndarray:
    """Convert CARLA Transform to JAX array.
    
    Args:
        transform: CARLA Transform object
        
    Returns:
        JAX array with shape (6,) containing [x, y, z, roll, pitch, yaw] in radians
    """
    if carla is None:
        raise RuntimeError("CARLA not available")
        
    return jnp.array([
        transform.location.x,
        transform.location.y,
        transform.location.z,
        np.radians(transform.rotation.roll),
        np.radians(transform.rotation.pitch),
        np.radians(transform.rotation.yaw)
    ])


def convert_carla_vector(vector: 'carla.Vector3D') -> jnp.ndarray:
    """Convert CARLA Vector3D to JAX array.
    
    Args:
        vector: CARLA Vector3D object
        
    Returns:
        JAX array with shape (3,) containing [x, y, z]
    """
    if carla is None:
        raise RuntimeError("CARLA not available")
        
    return jnp.array([vector.x, vector.y, vector.z])


def convert_carla_rotation(rotation: 'carla.Rotation') -> jnp.ndarray:
    """Convert CARLA Rotation to JAX array in radians.
    
    Args:
        rotation: CARLA Rotation object
        
    Returns:
        JAX array with shape (3,) containing [roll, pitch, yaw] in radians
    """
    if carla is None:
        raise RuntimeError("CARLA not available")
        
    return jnp.array([
        np.radians(rotation.roll),
        np.radians(rotation.pitch),
        np.radians(rotation.yaw)
    ])


def jax_to_carla_transform(jax_transform: jnp.ndarray) -> 'carla.Transform':
    """Convert JAX array to CARLA Transform.
    
    Args:
        jax_transform: JAX array with shape (6,) containing [x, y, z, roll, pitch, yaw]
        
    Returns:
        CARLA Transform object
    """
    if carla is None:
        raise RuntimeError("CARLA not available")
        
    location = carla.Location(
        x=float(jax_transform[0]),
        y=float(jax_transform[1]),
        z=float(jax_transform[2])
    )
    
    rotation = carla.Rotation(
        roll=float(np.degrees(jax_transform[3])),
        pitch=float(np.degrees(jax_transform[4])),
        yaw=float(np.degrees(jax_transform[5]))
    )
    
    return carla.Transform(location, rotation)


def jax_to_carla_vector(jax_vector: jnp.ndarray) -> 'carla.Vector3D':
    """Convert JAX array to CARLA Vector3D.
    
    Args:
        jax_vector: JAX array with shape (3,) containing [x, y, z]
        
    Returns:
        CARLA Vector3D object
    """
    if carla is None:
        raise RuntimeError("CARLA not available")
        
    return carla.Vector3D(
        x=float(jax_vector[0]),
        y=float(jax_vector[1]),
        z=float(jax_vector[2])
    )


# ==============================================================================
# -- Batch Processing Utilities -----------------------------------------------
# ==============================================================================

def batch_process_with_jax(data: List[Any], 
                          process_func: Callable, 
                          batch_size: int = 32,
                          progress_callback: Optional[Callable] = None) -> List[Any]:
    """Process data in batches using JAX vectorization.
    
    Args:
        data: List of data items to process
        process_func: JAX function to apply to each batch
        batch_size: Size of each batch
        progress_callback: Optional callback for progress reporting
        
    Returns:
        List of processed results
    """
    results = []
    num_batches = (len(data) + batch_size - 1) // batch_size
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        
        # Convert to JAX arrays if needed
        if isinstance(batch[0], np.ndarray):
            batch_array = jnp.stack(batch)
        else:
            batch_array = jnp.array(batch)
        
        # Process batch
        batch_result = process_func(batch_array)
        
        # Convert back to list
        if isinstance(batch_result, jnp.ndarray):
            results.extend(list(batch_result))
        else:
            results.append(batch_result)
        
        # Progress callback
        if progress_callback:
            progress_callback(i // batch_size + 1, num_batches)
    
    return results


def safe_jit_compile(func: Callable, 
                    static_argnums: Optional[Tuple[int, ...]] = None,
                    device: Optional[str] = None) -> Callable:
    """Safely compile a function with JAX JIT, with error handling.
    
    Args:
        func: Function to compile
        static_argnums: Arguments to treat as static
        device: Target device for compilation
        
    Returns:
        JIT-compiled function or original function if compilation fails
    """
    try:
        jit_func = jit(func, static_argnums=static_argnums, device=device)
        
        # Test compilation with dummy inputs (optional)
        # This would require knowing the function signature
        
        print(f"Successfully JIT-compiled function: {func.__name__}")
        return jit_func
        
    except Exception as e:
        warnings.warn(f"JIT compilation failed for {func.__name__}: {e}. "
                     f"Using original function.")
        return func


# ==============================================================================
# -- Array Utilities ----------------------------------------------------------
# ==============================================================================

def ensure_jax_array(data: Union[np.ndarray, jnp.ndarray, List, Tuple], 
                    dtype: Optional[jnp.dtype] = None) -> jnp.ndarray:
    """Ensure data is a JAX array with optional dtype conversion.
    
    Args:
        data: Input data
        dtype: Target dtype
        
    Returns:
        JAX array
    """
    if not isinstance(data, jnp.ndarray):
        data = jnp.array(data)
    
    if dtype is not None and data.dtype != dtype:
        data = data.astype(dtype)
    
    return data


def batch_convert_arrays(arrays: List[Union[np.ndarray, List]], 
                        target_shape: Optional[Tuple[int, ...]] = None) -> jnp.ndarray:
    """Convert a list of arrays to a batched JAX array.
    
    Args:
        arrays: List of arrays or lists
        target_shape: Optional target shape for each array
        
    Returns:
        Batched JAX array with shape (batch_size, *array_shape)
    """
    # Convert all to numpy first
    np_arrays = []
    for arr in arrays:
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)
        
        if target_shape is not None:
            arr = arr.reshape(target_shape)
        
        np_arrays.append(arr)
    
    # Stack and convert to JAX
    stacked = np.stack(np_arrays)
    return jnp.array(stacked)


def split_array_for_devices(array: jnp.ndarray, 
                           num_devices: Optional[int] = None) -> List[jnp.ndarray]:
    """Split array across multiple JAX devices.
    
    Args:
        array: Array to split
        num_devices: Number of devices (uses all available if None)
        
    Returns:
        List of array shards, one per device
    """
    if num_devices is None:
        num_devices = len(devices())
    
    if len(array) < num_devices:
        # If array is smaller than number of devices, pad with zeros
        padding_needed = num_devices - len(array)
        padded = jnp.concatenate([array, jnp.zeros((padding_needed,) + array.shape[1:])])
        return jnp.array_split(padded, num_devices)
    else:
        return jnp.array_split(array, num_devices)


# ==============================================================================
# -- Debugging and Validation -------------------------------------------------
# ==============================================================================

def validate_jax_function(func: Callable, 
                         test_inputs: List[Any],
                         expected_output_shape: Optional[Tuple[int, ...]] = None,
                         check_gradients: bool = False) -> Dict[str, Any]:
    """Validate a JAX function with test inputs.
    
    Args:
        func: JAX function to validate
        test_inputs: List of test inputs
        expected_output_shape: Expected output shape
        check_gradients: Whether to check if gradients can be computed
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'success': True,
        'errors': [],
        'warnings': [],
        'performance': {}
    }
    
    try:
        # Test basic execution
        start_time = time.time()
        output = func(*test_inputs)
        execution_time = time.time() - start_time
        
        results['performance']['execution_time_ms'] = execution_time * 1000
        
        # Check output shape
        if expected_output_shape is not None:
            if hasattr(output, 'shape') and output.shape != expected_output_shape:
                results['warnings'].append(
                    f"Output shape {output.shape} != expected {expected_output_shape}"
                )
        
        # Test JIT compilation
        try:
            start_time = time.time()
            jit_func = jit(func)
            jit_output = jit_func(*test_inputs)
            jit_time = time.time() - start_time
            
            results['performance']['jit_compilation_time_ms'] = jit_time * 1000
            
            # Check JIT output matches
            if hasattr(output, 'shape') and hasattr(jit_output, 'shape'):
                if not jnp.allclose(output, jit_output, rtol=1e-5):
                    results['warnings'].append("JIT output differs from regular output")
        
        except Exception as e:
            results['errors'].append(f"JIT compilation failed: {e}")
        
        # Test gradients if requested
        if check_gradients:
            try:
                from jax import grad
                grad_func = grad(func)
                grad_output = grad_func(*test_inputs)
                results['performance']['gradient_computation'] = 'success'
            except Exception as e:
                results['errors'].append(f"Gradient computation failed: {e}")
    
    except Exception as e:
        results['success'] = False
        results['errors'].append(f"Function execution failed: {e}")
    
    return results


def debug_jax_array(array: jnp.ndarray, name: str = "array") -> None:
    """Print debug information about a JAX array.
    
    Args:
        array: JAX array to debug
        name: Name for the array in output
    """
    print(f"Debug info for {name}:")
    print(f"  Shape: {array.shape}")
    print(f"  Dtype: {array.dtype}")
    print(f"  Device: {array.device()}")
    print(f"  Min/Max: {jnp.min(array):.6f} / {jnp.max(array):.6f}")
    print(f"  Mean/Std: {jnp.mean(array):.6f} / {jnp.std(array):.6f}")
    
    if jnp.any(jnp.isnan(array)):
        print(f"  WARNING: Contains {jnp.sum(jnp.isnan(array))} NaN values")
    
    if jnp.any(jnp.isinf(array)):
        print(f"  WARNING: Contains {jnp.sum(jnp.isinf(array))} infinite values")


# ==============================================================================
# -- Configuration and Settings -----------------------------------------------
# ==============================================================================

def configure_jax_for_carla(enable_x64: bool = False,
                           enable_jit: bool = True,
                           device: str = 'auto') -> Dict[str, Any]:
    """Configure JAX settings optimized for CARLA usage.
    
    Args:
        enable_x64: Enable 64-bit precision (slower but more accurate)
        enable_jit: Enable JIT compilation
        device: Target device ('auto', 'cpu', 'gpu')
        
    Returns:
        Dictionary with applied configuration
    """
    config = {}
    
    # Configure precision
    if enable_x64:
        jax.config.update("jax_enable_x64", True)
        config['precision'] = '64-bit'
    else:
        config['precision'] = '32-bit'
    
    # Configure JIT
    if not enable_jit:
        jax.config.update("jax_disable_jit", True)
    config['jit_enabled'] = enable_jit
    
    # Configure device
    if device != 'auto':
        config['requested_device'] = device
    
    config['actual_device'] = ensure_jax_device(device if device != 'auto' else 'gpu')
    config['available_devices'] = [str(d) for d in devices()]
    
    # Memory allocation strategy
    try:
        # Pre-allocate GPU memory if available
        if 'gpu' in config['actual_device']:
            jax.config.update("jax_gpu_memory_fraction", 0.8)
            config['gpu_memory_fraction'] = 0.8
    except:
        pass
    
    print("JAX Configuration for CARLA:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    return config