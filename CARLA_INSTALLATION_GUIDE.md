# CARLA Installation Guide for Linux Ubuntu

This guide provides complete instructions for installing CARLA simulator on Ubuntu Linux to work with the JAX-accelerated examples.

## 🎯 Quick Setup Summary

For users who want to get started quickly:

```bash
# 1. Install CARLA client (Python API)
pip install carla==0.9.15

# 2. Download and extract CARLA server
wget https://github.com/carla-simulator/carla/releases/download/0.9.15/CARLA_0.9.15.tar.gz
tar -xzf CARLA_0.9.15.tar.gz
cd CARLA_0.9.15

# 3. Test CARLA server
./CarlaUE4.sh -vulkan

# 4. In another terminal, test Python connection
python -c "import carla; client = carla.Client('localhost', 2000); print('CARLA connected successfully!')"
```

## 📋 System Requirements

### Minimum Requirements
- **OS**: Ubuntu 18.04 or later (Ubuntu 22.04 recommended for latest versions)
- **Python**: 3.7-3.10 (Python 3.8+ recommended)
- **GPU**: 6 GB VRAM minimum (8+ GB recommended)
- **RAM**: 8 GB minimum (16+ GB recommended)
- **Disk Space**: ~130 GB free (31 GB for CARLA + 91 GB for dependencies)
- **CPU**: Intel i7 gen 9th-11th, Intel i9 gen 9th-11th, AMD Ryzen 7, or AMD Ryzen 9

### Graphics Requirements
- **NVIDIA GPU** with CUDA support (recommended)
- **Vulkan support** (for optimal performance)
- Updated NVIDIA drivers

## 🚀 Installation Methods

### Method 1: Recommended Installation (pip + server download)

This is the most reliable method for getting CARLA working with JAX examples.

#### Step 1: Install System Dependencies

For **Ubuntu 18.04/20.04**:
```bash
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    libomp5 \
    libpng16-dev \
    libtiff5-dev \
    libjpeg-dev \
    tzdata \
    sed \
    curl \
    unzip \
    autoconf \
    libtool \
    rsync \
    libxml2-dev \
    git-lfs
```

For **Ubuntu 22.04**:
```bash
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    libomp5 \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    tzdata \
    sed \
    curl \
    unzip \
    autoconf \
    libtool \
    rsync \
    libxml2-dev \
    git-lfs
```

#### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment for CARLA
python3 -m venv venv_carla_jax
source venv_carla_jax/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

#### Step 3: Install CARLA Python Client
```bash
# Install CARLA client library
pip install carla==0.9.15

# Verify installation
python -c "import carla; print(f'CARLA Python API version: {carla.__version__}')"
```

#### Step 4: Download CARLA Server
```bash
# Create CARLA directory
mkdir -p ~/carla_simulator
cd ~/carla_simulator

# Download CARLA 0.9.15 server
wget https://github.com/carla-simulator/carla/releases/download/0.9.15/CARLA_0.9.15.tar.gz

# Extract CARLA server
tar -xzf CARLA_0.9.15.tar.gz
cd CARLA_0.9.15

# Make executable (if needed)
chmod +x CarlaUE4.sh
```

#### Step 5: Download Additional Maps (Optional)
```bash
# Download additional maps
cd ~/carla_simulator
wget https://github.com/carla-simulator/carla/releases/download/0.9.15/AdditionalMaps_0.9.15.tar.gz

# Extract to CARLA directory
tar -xzf AdditionalMaps_0.9.15.tar.gz -C CARLA_0.9.15/
```

### Method 2: APT Installation (Ubuntu 18.04/20.04 only)

```bash
# Add CARLA repository
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 1AF1527DE64CB8D9
sudo add-apt-repository "deb [arch=amd64] http://dist.carla.org/carla $(lsb_release -sc) main"

# Update and install
sudo apt-get update
sudo apt-get install carla-simulator

# CARLA will be installed to /opt/carla-simulator
```

## 🔧 Configuration and Testing

### Start CARLA Server

#### Option 1: Standard Mode
```bash
cd ~/carla_simulator/CARLA_0.9.15
./CarlaUE4.sh
```

#### Option 2: Vulkan Mode (Better Performance)
```bash
cd ~/carla_simulator/CARLA_0.9.15
./CarlaUE4.sh -vulkan
```

#### Option 3: Headless Mode (No Graphics)
```bash
cd ~/carla_simulator/CARLA_0.9.15
./CarlaUE4.sh -RenderOffScreen
```

#### Option 4: Low Quality Mode (For Development)
```bash
cd ~/carla_simulator/CARLA_0.9.15
./CarlaUE4.sh -quality-level=Low -fps=20
```

### Test CARLA Connection

In a new terminal:
```bash
# Activate your virtual environment
source venv_carla_jax/bin/activate

# Test basic connection
python3 -c "
import carla
import time

try:
    # Connect to CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    # Get world
    world = client.get_world()
    print(f'✅ Connected to CARLA world: {world.get_map().name}')
    
    # Get available maps
    maps = client.get_available_maps()
    print(f'✅ Available maps: {len(maps)}')
    
    print('🎉 CARLA installation successful!')
    
except Exception as e:
    print(f'❌ CARLA connection failed: {e}')
    print('Make sure CARLA server is running on localhost:2000')
"
```

## 🔗 Integration with JAX Examples

### Update CARLA Path in JAX Examples

The JAX examples expect CARLA to be in a specific location. You need to update the import paths:

#### Option 1: Set PYTHONPATH
```bash
# Add CARLA to Python path
export PYTHONPATH=$PYTHONPATH:~/carla_simulator/CARLA_0.9.15/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
```

#### Option 2: Modify Example Files
Edit each JAX example file to fix the import path. Replace:
```python
sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
    sys.version_info.major,
    sys.version_info.minor,
    'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
```

With:
```python
try:
    import carla
except ImportError:
    # Try to find CARLA installation
    import glob
    import sys
    import os
    
    # Common CARLA installation paths
    carla_paths = [
        # pip installation (should work by default)
        None,
        # Local installation
        os.path.expanduser('~/carla_simulator/CARLA_0.9.15/PythonAPI/carla'),
        # APT installation
        '/opt/carla-simulator/PythonAPI/carla',
        # Build from source
        '../../carla/dist'
    ]
    
    for path in carla_paths:
        if path is None:
            continue
            
        egg_files = glob.glob(f'{path}/dist/carla-*{sys.version_info.major}.{sys.version_info.minor}-*.egg')
        if egg_files:
            sys.path.append(egg_files[0])
            break
    
    import carla
```

### Run JAX Examples with CARLA

```bash
# Start CARLA server (Terminal 1)
cd ~/carla_simulator/CARLA_0.9.15
./CarlaUE4.sh -vulkan

# Run JAX examples (Terminal 2)
cd /path/to/carla-jax-accelerated/PythonAPI/examples_jax
source venv_carla_jax/bin/activate

# Test individual examples
python automatic_control_jax.py --jax-agent --sync
python sensor_synchronization_jax.py --frames 50
python generate_traffic_jax.py --mode carla-integration --number-of-vehicles 20
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. "ImportError: No module named 'carla'"
```bash
# Check if CARLA is installed
pip list | grep carla

# If not installed:
pip install carla==0.9.15

# Verify installation
python -c "import carla; print('✅ CARLA imported successfully')"
```

#### 2. "RuntimeError: cannot connect to CARLA server"
```bash
# Check if CARLA server is running
ps aux | grep CarlaUE4

# Check if port is open
nc -z localhost 2000

# Restart CARLA server
pkill -f CarlaUE4
cd ~/carla_simulator/CARLA_0.9.15
./CarlaUE4.sh -vulkan
```

#### 3. "GPU/Graphics issues"
```bash
# Check NVIDIA driver
nvidia-smi

# Install NVIDIA drivers if needed
sudo ubuntu-drivers autoinstall

# Check Vulkan support
vulkaninfo | head -20

# Install Vulkan if needed
sudo apt install vulkan-utils vulkan-tools
```

#### 4. "Permission denied: CarlaUE4.sh"
```bash
cd ~/carla_simulator/CARLA_0.9.15
chmod +x CarlaUE4.sh
chmod +x CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping
```

#### 5. "Version mismatch errors"
```bash
# Check versions match
python -c "import carla; print(f'Client: {carla.__version__}')"
# Should match server version (0.9.15)

# If mismatch, reinstall client:
pip uninstall carla
pip install carla==0.9.15
```

### Performance Optimization

#### 1. Graphics Settings
```bash
# Low quality for development
./CarlaUE4.sh -quality-level=Low -fps=20

# Benchmark mode
./CarlaUE4.sh -benchmark -fps=30

# Fixed time step
./CarlaUE4.sh -fixed-dt=0.05
```

#### 2. Memory Management
```bash
# Limit memory usage
export UE4_MEMORY_LIMIT=8G

# Enable garbage collection
export CARLA_GARBAGE_COLLECTION=true
```

#### 3. Network Optimization
```bash
# Use different ports if needed
./CarlaUE4.sh -carla-rpc-port=2002 -carla-streaming-port=2003
```

## 📚 Additional Resources

- **Official CARLA Documentation**: https://carla.readthedocs.io/
- **CARLA GitHub Repository**: https://github.com/carla-simulator/carla
- **CARLA Releases**: https://github.com/carla-simulator/carla/releases
- **CARLA Python API Reference**: https://carla.readthedocs.io/en/latest/python_api/

## 🎯 Next Steps

After successful installation:

1. **Test JAX-only examples** to verify JAX acceleration works
2. **Run CARLA server** and test basic connection
3. **Run JAX+CARLA examples** to see the full integration
4. **Explore different scenarios** and maps
5. **Monitor performance** and optimize settings as needed

## 🆘 Getting Help

If you encounter issues:

1. **Check CARLA logs**: Look in `~/carla_simulator/CARLA_0.9.15/CarlaUE4/Saved/Logs/`
2. **CARLA Forum**: https://forum.carla.org/
3. **GitHub Issues**: https://github.com/carla-simulator/carla/issues
4. **Discord Community**: Join the CARLA Discord server

---

**Ready to accelerate your autonomous driving research with CARLA + JAX!** 🚀