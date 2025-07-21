#!/bin/bash

# CARLA + JAX Installation Script for Ubuntu Linux
# This script automates the installation of CARLA simulator and JAX dependencies

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CARLA_VERSION="0.9.15"
CARLA_INSTALL_DIR="$HOME/carla_simulator"
VENV_NAME="venv_carla_jax"
CARLA_SERVER_URL="https://github.com/carla-simulator/carla/releases/download/0.9.15/CARLA_0.9.15.tar.gz"
CARLA_MAPS_URL="https://github.com/carla-simulator/carla/releases/download/0.9.15/AdditionalMaps_0.9.15.tar.gz"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect Ubuntu version
get_ubuntu_version() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        echo $VERSION_ID
    else
        echo "unknown"
    fi
}

# Function to check system requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check Ubuntu version
    UBUNTU_VERSION=$(get_ubuntu_version)
    print_status "Detected Ubuntu version: $UBUNTU_VERSION"
    
    # Check Python version
    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
        print_status "Python version: $PYTHON_VERSION"
        
        # Check if Python version is supported (3.7-3.10)
        if [[ "$PYTHON_VERSION" < "3.7" ]] || [[ "$PYTHON_VERSION" > "3.10" ]]; then
            print_warning "Python $PYTHON_VERSION may not be fully supported. Recommended: Python 3.8-3.10"
        fi
    else
        print_error "Python 3 is not installed. Please install Python 3.7+ first."
        exit 1
    fi
    
    # Check available disk space (need ~130GB)
    AVAILABLE_SPACE=$(df / | awk 'NR==2 {print $4}')
    REQUIRED_SPACE=$((130 * 1024 * 1024))  # 130GB in KB
    
    if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
        print_warning "Low disk space detected. CARLA needs ~130GB. Available: $(($AVAILABLE_SPACE / 1024 / 1024))GB"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # Check for NVIDIA GPU
    if command_exists nvidia-smi; then
        print_success "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
    else
        print_warning "NVIDIA GPU not detected or drivers not installed"
        print_warning "CARLA will run on CPU only (much slower)"
    fi
}

# Function to install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    sudo apt update
    
    # Common dependencies for all Ubuntu versions
    COMMON_DEPS="build-essential cmake git python3 python3-pip python3-dev python3-venv libomp5 \
                 libjpeg-dev tzdata sed curl unzip autoconf libtool rsync libxml2-dev git-lfs \
                 wget ca-certificates gnupg lsb-release"
    
    # Version-specific dependencies
    if [[ "$UBUNTU_VERSION" == "18.04" ]] || [[ "$UBUNTU_VERSION" == "20.04" ]]; then
        DEPS="$COMMON_DEPS libpng16-dev libtiff5-dev"
    else
        # Ubuntu 22.04+
        DEPS="$COMMON_DEPS libpng-dev libtiff5-dev"
    fi
    
    print_status "Installing: $DEPS"
    sudo apt install -y $DEPS
    
    # Install Vulkan support (optional but recommended)
    print_status "Installing Vulkan support..."
    sudo apt install -y vulkan-utils vulkan-tools mesa-vulkan-drivers
    
    print_success "System dependencies installed"
}

# Function to create virtual environment
setup_virtual_env() {
    print_status "Setting up Python virtual environment..."
    
    # Remove existing environment if it exists
    if [ -d "$VENV_NAME" ]; then
        print_warning "Existing virtual environment found. Removing..."
        rm -rf "$VENV_NAME"
    fi
    
    # Create new virtual environment
    python3 -m venv "$VENV_NAME"
    source "$VENV_NAME/bin/activate"
    
    # Upgrade pip and setuptools
    pip install --upgrade pip setuptools wheel
    
    print_success "Virtual environment created: $VENV_NAME"
}

# Function to install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    # Activate virtual environment
    source "$VENV_NAME/bin/activate"
    
    # Install JAX dependencies (from requirements_jax.txt)
    print_status "Installing JAX and related packages..."
    pip install "jax[cpu]>=0.4.20" jaxlib>=0.4.20 numpy>=1.21.0 optax>=0.1.7 \
                flax>=0.7.0 chex>=0.1.7 dm-haiku>=0.0.9
    
    # Install visualization and data handling
    print_status "Installing visualization packages..."
    pip install matplotlib>=3.5.0 opencv-python>=4.5.0
    
    # Try to install Open3D (may fail on some systems)
    print_status "Installing Open3D (optional)..."
    pip install open3d>=0.15.0 || print_warning "Open3D installation failed (optional package)"
    
    # Install CARLA Python client
    print_status "Installing CARLA Python client..."
    pip install carla==$CARLA_VERSION
    
    # Install optional ML libraries
    print_status "Installing optional ML libraries..."
    pip install tensorflow-probability>=0.19.0 || print_warning "TensorFlow Probability installation failed"
    pip install wandb>=0.13.0 || print_warning "Weights & Biases installation failed"
    
    # Install basic CARLA dependencies
    print_status "Installing additional CARLA dependencies..."
    pip install pygame pillow
    
    print_success "Python dependencies installed"
}

# Function to download and install CARLA server
install_carla_server() {
    print_status "Downloading and installing CARLA server..."
    
    # Create installation directory
    mkdir -p "$CARLA_INSTALL_DIR"
    cd "$CARLA_INSTALL_DIR"
    
    # Download CARLA server if not already present
    if [ ! -f "CARLA_${CARLA_VERSION}.tar.gz" ]; then
        print_status "Downloading CARLA ${CARLA_VERSION} server..."
        wget -O "CARLA_${CARLA_VERSION}.tar.gz" "$CARLA_SERVER_URL"
    else
        print_status "CARLA server archive already exists"
    fi
    
    # Extract CARLA server
    if [ ! -d "CARLA_${CARLA_VERSION}" ]; then
        print_status "Extracting CARLA server..."
        tar -xzf "CARLA_${CARLA_VERSION}.tar.gz"
    else
        print_status "CARLA server already extracted"
    fi
    
    # Make executables
    cd "CARLA_${CARLA_VERSION}"
    chmod +x CarlaUE4.sh
    if [ -f "CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping" ]; then
        chmod +x CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping
    fi
    
    print_success "CARLA server installed to: $CARLA_INSTALL_DIR/CARLA_${CARLA_VERSION}"
}

# Function to download additional maps
install_additional_maps() {
    print_status "Downloading additional maps..."
    
    cd "$CARLA_INSTALL_DIR"
    
    # Download additional maps if not present
    if [ ! -f "AdditionalMaps_${CARLA_VERSION}.tar.gz" ]; then
        print_status "Downloading additional maps..."
        wget -O "AdditionalMaps_${CARLA_VERSION}.tar.gz" "$CARLA_MAPS_URL"
    else
        print_status "Additional maps archive already exists"
    fi
    
    # Extract additional maps
    print_status "Extracting additional maps..."
    tar -xzf "AdditionalMaps_${CARLA_VERSION}.tar.gz" -C "CARLA_${CARLA_VERSION}/"
    
    print_success "Additional maps installed"
}

# Function to test installation
test_installation() {
    print_status "Testing installation..."
    
    # Activate virtual environment
    source "$VENV_NAME/bin/activate"
    
    # Test JAX installation
    print_status "Testing JAX installation..."
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
    
    # Test CARLA Python client
    print_status "Testing CARLA Python client..."
    python3 -c "
import carla
print(f'✅ CARLA Python API version: {carla.__version__}')
print('✅ CARLA client library imported successfully')
"
    
    # Check CARLA server executable
    if [ -f "$CARLA_INSTALL_DIR/CARLA_${CARLA_VERSION}/CarlaUE4.sh" ]; then
        print_success "CARLA server executable found"
    else
        print_error "CARLA server executable not found"
        return 1
    fi
    
    print_success "Installation test completed successfully!"
}

# Function to create startup scripts
create_scripts() {
    print_status "Creating convenience scripts..."
    
    # Create CARLA startup script
    cat > start_carla.sh << EOF
#!/bin/bash
# Start CARLA server

echo "🚀 Starting CARLA server..."
cd "$CARLA_INSTALL_DIR/CARLA_${CARLA_VERSION}"

# Check if server is already running
if pgrep -f "CarlaUE4" > /dev/null; then
    echo "⚠️  CARLA server is already running"
    echo "   Use 'pkill -f CarlaUE4' to stop it first"
    exit 1
fi

# Start server based on options
if [[ "\$1" == "--vulkan" ]]; then
    echo "Starting with Vulkan renderer..."
    ./CarlaUE4.sh -vulkan
elif [[ "\$1" == "--headless" ]]; then
    echo "Starting in headless mode..."
    ./CarlaUE4.sh -RenderOffScreen
elif [[ "\$1" == "--low-quality" ]]; then
    echo "Starting in low quality mode..."
    ./CarlaUE4.sh -quality-level=Low -fps=20
else
    echo "Starting in default mode..."
    echo "Available options:"
    echo "  --vulkan      : Use Vulkan renderer (better performance)"
    echo "  --headless    : No graphics (for servers)"
    echo "  --low-quality : Low quality for development"
    ./CarlaUE4.sh
fi
EOF
    chmod +x start_carla.sh
    
    # Create environment activation script
    cat > activate_carla_jax.sh << EOF
#!/bin/bash
# Activate CARLA + JAX environment

echo "🔧 Activating CARLA + JAX environment..."
source "$VENV_NAME/bin/activate"

echo "✅ Environment activated!"
echo ""
echo "🎯 Available commands:"
echo "  ./start_carla.sh          - Start CARLA server"
echo "  ./start_carla.sh --vulkan - Start with Vulkan (recommended)"
echo "  ./test_carla_connection.sh - Test CARLA connection"
echo ""
echo "📁 Directories:"
echo "  CARLA server: $CARLA_INSTALL_DIR/CARLA_${CARLA_VERSION}"
echo "  JAX examples: PythonAPI/examples_jax/"
echo ""
echo "💡 Environment info:"
python3 -c "
import jax
import carla
print(f'  JAX version: {jax.__version__}')
print(f'  CARLA version: {carla.__version__}')
print(f'  JAX devices: {jax.devices()}')
print(f'  JAX backend: {jax.default_backend()}')
"
EOF
    chmod +x activate_carla_jax.sh
    
    # Create connection test script
    cat > test_carla_connection.sh << EOF
#!/bin/bash
# Test CARLA connection

echo "🧪 Testing CARLA connection..."

# Activate environment
source "$VENV_NAME/bin/activate"

# Check if CARLA server is running
if ! nc -z localhost 2000 2>/dev/null; then
    echo "❌ CARLA server is not running on localhost:2000"
    echo "   Start CARLA server first: ./start_carla.sh"
    exit 1
fi

# Test Python connection
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
    for i, map_name in enumerate(maps[:5]):  # Show first 5 maps
        print(f'   {i+1}. {map_name}')
    if len(maps) > 5:
        print(f'   ... and {len(maps)-5} more')
    
    print('🎉 CARLA connection test successful!')
    
except Exception as e:
    print(f'❌ CARLA connection failed: {e}')
    print('Make sure CARLA server is running: ./start_carla.sh')
    exit 1
"
EOF
    chmod +x test_carla_connection.sh
    
    print_success "Convenience scripts created"
}

# Function to show final instructions
show_final_instructions() {
    print_success "🎉 CARLA + JAX installation completed successfully!"
    echo ""
    echo -e "${GREEN}📋 Quick Start Guide:${NC}"
    echo ""
    echo "1. Activate environment:"
    echo "   ${BLUE}./activate_carla_jax.sh${NC}"
    echo ""
    echo "2. Start CARLA server (in terminal 1):"
    echo "   ${BLUE}./start_carla.sh --vulkan${NC}"
    echo ""
    echo "3. Test connection (in terminal 2):"
    echo "   ${BLUE}./test_carla_connection.sh${NC}"
    echo ""
    echo "4. Run JAX examples:"
    echo "   ${BLUE}cd PythonAPI/examples_jax${NC}"
    echo "   ${BLUE}python generate_traffic_jax.py --mode jax-only --number-of-vehicles 50${NC}"
    echo "   ${BLUE}python generate_traffic_jax.py --mode carla-integration --number-of-vehicles 20${NC}"
    echo ""
    echo -e "${GREEN}📁 Installation Locations:${NC}"
    echo "  • CARLA server: ${CARLA_INSTALL_DIR}/CARLA_${CARLA_VERSION}"
    echo "  • Virtual environment: ${VENV_NAME}/"
    echo "  • JAX examples: PythonAPI/examples_jax/"
    echo ""
    echo -e "${GREEN}🚀 Start accelerating your autonomous driving research!${NC}"
    echo ""
    echo -e "${YELLOW}📚 Documentation:${NC}"
    echo "  • Installation guide: CARLA_INSTALLATION_GUIDE.md"
    echo "  • JAX examples: PythonAPI/examples_jax/README.md"
    echo "  • CARLA docs: https://carla.readthedocs.io/"
}

# Main installation function
main() {
    echo -e "${GREEN}🚀 CARLA + JAX Installation Script${NC}"
    echo -e "${GREEN}===================================${NC}"
    echo ""
    
    # Parse command line arguments
    INSTALL_MAPS=true
    SKIP_TESTS=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-maps)
                INSTALL_MAPS=false
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo ""
                echo "Options:"
                echo "  --no-maps     Skip downloading additional maps"
                echo "  --skip-tests  Skip installation tests"
                echo "  --help        Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Installation steps
    check_requirements
    install_system_deps
    setup_virtual_env
    install_python_deps
    install_carla_server
    
    if [ "$INSTALL_MAPS" = true ]; then
        install_additional_maps
    else
        print_status "Skipping additional maps installation"
    fi
    
    if [ "$SKIP_TESTS" = false ]; then
        test_installation
    else
        print_status "Skipping installation tests"
    fi
    
    create_scripts
    show_final_instructions
}

# Run main function
main "$@"