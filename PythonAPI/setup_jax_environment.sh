#!/bin/bash

# JAX Environment Setup Script for CARLA
# This script sets up everything needed to run JAX-accelerated CARLA examples

set -e  # Exit on any error

echo "🚀 Setting up JAX Environment for CARLA..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "📋 Python version: $PYTHON_VERSION"

# Check if we're in the right directory
if [ ! -d "examples_jax" ]; then
    echo "❌ Error: examples_jax directory not found. Please run this script from PythonAPI directory."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv_jax" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv_jax
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv_jax/bin/activate

# Upgrade pip
echo "🔧 Upgrading pip..."
pip install --upgrade pip

# Install JAX dependencies
echo "🔧 Installing JAX and dependencies..."
pip install -r examples_jax/requirements_jax.txt

# Install additional dependencies from examples directory if they exist
if [ -f "examples/requirements.txt" ]; then
    echo "🔧 Installing additional CARLA example dependencies..."
    pip install -r examples/requirements.txt
fi

# Install basic CARLA dependencies
echo "🔧 Installing basic CARLA dependencies..."
pip install pygame numpy matplotlib pillow

# Check JAX installation
echo "🧪 Testing JAX installation..."
python3 -c "
import jax
import jax.numpy as jnp
print(f'✅ JAX version: {jax.__version__}')
print(f'✅ JAX devices: {jax.devices()}')
print(f'✅ JAX backend: {jax.default_backend()}')

# Test basic JAX operations
x = jnp.array([1, 2, 3])
y = jnp.sum(x)
print(f'✅ Basic JAX operation: sum([1,2,3]) = {y}')

# Test JIT compilation
from jax import jit
@jit
def test_jit(x):
    return x ** 2

result = test_jit(5.0)
print(f'✅ JIT compilation: 5^2 = {result}')
"

# Test JAX utilities
echo "🧪 Testing JAX utilities..."
cd examples_jax
python3 -c "
from jax_utils import ensure_jax_device, configure_jax_for_carla
print('✅ JAX utilities loaded successfully')

# Configure JAX for CARLA
config = configure_jax_for_carla()
print(f'✅ JAX configured for CARLA: {config[\"actual_device\"]}')
"
cd ..

# Check GPU availability (optional)
echo "🔍 Checking GPU availability..."
python3 -c "
import jax
gpu_available = len([d for d in jax.devices() if 'gpu' in str(d)]) > 0
if gpu_available:
    print('✅ GPU available for JAX acceleration')
else:
    print('ℹ️  No GPU found, using CPU (still fast with JAX)')
"

# Create run scripts
echo "🔧 Creating convenience run scripts..."

# Script to run JAX examples without CARLA
cat > run_jax_demo.sh << 'EOF'
#!/bin/bash
# Run JAX demonstrations without CARLA dependency

echo "🚀 Running JAX-only demonstrations..."

# Activate environment
source venv_jax/bin/activate

cd examples_jax

echo "1. Testing JAX utilities..."
python3 -c "
from jax_utils import JAXProfiler, configure_jax_for_carla
import jax.numpy as jnp
import jax

# Configure JAX
configure_jax_for_carla()

# Test profiler
profiler = JAXProfiler()

@profiler.profile_function('test_function')
@jax.jit
def test_function(x):
    return jnp.sum(x ** 2)

# Run test
x = jnp.arange(1000)
result = test_function(x)
print(f'✅ Test result: {result}')

profiler.print_stats()
"

echo "2. Running traffic simulation demo (JAX-only mode)..."
python3 generate_traffic_jax.py --mode jax-only --number-of-vehicles 50 --simulation-steps 100

echo "3. Running vehicle physics demo..."
python3 vehicle_physics_jax.py --demo-mode optimization

echo "✅ JAX demonstrations completed!"
EOF

chmod +x run_jax_demo.sh

# Script to run with CARLA (when available)
cat > run_with_carla.sh << 'EOF'
#!/bin/bash
# Run JAX examples with CARLA integration

echo "🚀 Running JAX examples with CARLA..."

# Check if CARLA is running
if ! nc -z localhost 2000; then
    echo "❌ CARLA server not running on localhost:2000"
    echo "Please start CARLA first:"
    echo "  cd /path/to/CARLA"
    echo "  ./CarlaUE4.sh"
    exit 1
fi

# Activate environment
source venv_jax/bin/activate

cd examples_jax

echo "Running CARLA-JAX integration examples..."

echo "1. JAX-accelerated automatic control..."
python3 automatic_control_jax.py --jax-agent --sync --host localhost --port 2000

echo "2. JAX sensor synchronization..."
python3 sensor_synchronization_jax.py --frames 50 --host localhost --port 2000

echo "3. JAX LiDAR-camera projection..."
python3 lidar_to_camera_jax.py --frames 30 --host localhost --port 2000

echo "4. JAX traffic generation with CARLA..."
python3 generate_traffic_jax.py --mode carla-integration --number-of-vehicles 20 --host localhost --port 2000

echo "✅ CARLA-JAX examples completed!"
EOF

chmod +x run_with_carla.sh

# Create environment activation script
cat > activate_jax.sh << 'EOF'
#!/bin/bash
# Activate JAX environment for CARLA

echo "🔧 Activating JAX environment..."
source venv_jax/bin/activate

echo "✅ JAX environment activated!"
echo "Available commands:"
echo "  ./run_jax_demo.sh     - Run JAX demos without CARLA"
echo "  ./run_with_carla.sh   - Run with CARLA integration"
echo "  cd examples_jax       - Go to JAX examples directory"
echo ""
echo "JAX configuration:"
python3 -c "
import jax
print(f'  JAX version: {jax.__version__}')
print(f'  Devices: {jax.devices()}')
print(f'  Backend: {jax.default_backend()}')
"
EOF

chmod +x activate_jax.sh

echo "✅ JAX environment setup completed!"
echo ""
echo "🎯 Quick start:"
echo "  1. Run JAX demos:     ./run_jax_demo.sh"
echo "  2. With CARLA:        ./run_with_carla.sh" 
echo "  3. Activate env:      ./activate_jax.sh"
echo ""
echo "📁 Available examples:"
echo "  - examples_jax/automatic_control_jax.py     (JIT-compiled vehicle control)"
echo "  - examples_jax/sensor_synchronization_jax.py (Multi-sensor fusion)"
echo "  - examples_jax/vehicle_physics_jax.py       (Physics with autodiff)"
echo "  - examples_jax/lidar_to_camera_jax.py       (Geometric transforms)"
echo "  - examples_jax/generate_traffic_jax.py      (Vectorized traffic)"
echo ""
echo "🚀 JAX acceleration ready for CARLA autonomous driving!"