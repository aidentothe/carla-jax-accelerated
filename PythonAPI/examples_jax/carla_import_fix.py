"""
CARLA Import Fix Utility

This module handles CARLA imports across different installation methods
and Python versions.
"""

import sys
import os
import glob
import warnings
from pathlib import Path

def setup_carla_import():
    """
    Attempt to make CARLA importable through various methods.
    Returns True if successful, False otherwise.
    """
    # First, check if CARLA is already importable
    try:
        import carla
        return True
    except ImportError:
        pass
    
    # Method 1: Check for CARLA egg files relative to this script
    script_dir = Path(__file__).parent.absolute()
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    
    # Common relative paths to check
    egg_patterns = [
        # Standard CARLA installation paths
        f"../../carla/dist/carla-*{python_version}*.egg",
        f"../../../carla/dist/carla-*{python_version}*.egg",
        # Alternate paths
        f"../../carla/PythonAPI/carla/dist/carla-*{python_version}*.egg",
        # Any Python version as fallback
        "../../carla/dist/carla-*.egg",
        "../../../carla/dist/carla-*.egg",
    ]
    
    for pattern in egg_patterns:
        full_pattern = script_dir / pattern
        eggs = glob.glob(str(full_pattern))
        if eggs:
            # Add the first matching egg to path
            sys.path.insert(0, eggs[0])
            try:
                import carla
                return True
            except ImportError:
                # Remove if it didn't work
                sys.path.remove(eggs[0])
    
    # Method 2: Check CARLA installation directory from environment
    carla_root = os.environ.get('CARLA_ROOT')
    if carla_root:
        egg_path = Path(carla_root) / 'PythonAPI' / 'carla' / 'dist'
        if egg_path.exists():
            eggs = list(egg_path.glob(f'carla-*{python_version}*.egg'))
            if not eggs:
                # Try any version
                eggs = list(egg_path.glob('carla-*.egg'))
            
            if eggs:
                sys.path.insert(0, str(eggs[0]))
                try:
                    import carla
                    return True
                except ImportError:
                    sys.path.remove(str(eggs[0]))
    
    # Method 3: Check common CARLA installation locations
    common_paths = [
        Path.home() / 'carla_simulator' / 'CARLA_0.9.15',
        Path.home() / 'carla_simulator' / 'CARLA_0.9.14',
        Path.home() / 'carla_simulator' / 'CARLA_0.9.13',
        Path('/opt/carla-simulator'),
        Path('/opt/carla'),
    ]
    
    for carla_path in common_paths:
        if carla_path.exists():
            egg_path = carla_path / 'PythonAPI' / 'carla' / 'dist'
            if egg_path.exists():
                eggs = list(egg_path.glob(f'carla-*{python_version}*.egg'))
                if not eggs:
                    eggs = list(egg_path.glob('carla-*.egg'))
                
                if eggs:
                    sys.path.insert(0, str(eggs[0]))
                    try:
                        import carla
                        return True
                    except ImportError:
                        sys.path.remove(str(eggs[0]))
    
    # Method 4: Try pip-installed CARLA (should work if installed correctly)
    try:
        import carla
        return True
    except ImportError:
        pass
    
    return False

def get_carla_version():
    """Get CARLA version if available."""
    try:
        import carla
        return carla.__version__
    except:
        return None

def ensure_carla_available():
    """
    Ensure CARLA is available for import.
    Raises ImportError with helpful message if not found.
    """
    if setup_carla_import():
        version = get_carla_version()
        if version:
            print(f"✅ CARLA {version} successfully imported")
        else:
            print("✅ CARLA successfully imported")
        return True
    else:
        error_msg = """
❌ CARLA Python package not found!

To fix this issue, try one of the following:

1. Install CARLA via pip (recommended):
   pip install carla==0.9.13  # or 0.9.14, 0.9.15

2. Set CARLA_ROOT environment variable:
   export CARLA_ROOT=/path/to/CARLA_0.9.15

3. Download CARLA and extract it to:
   ~/carla_simulator/CARLA_0.9.15

4. For JAX-only examples, use --demo-mode or --mode jax-only

Current Python version: {}.{}
""".format(sys.version_info.major, sys.version_info.minor)
        
        raise ImportError(error_msg)

# Convenience function for scripts
def import_carla():
    """Import CARLA with automatic path setup."""
    ensure_carla_available()
    import carla
    return carla