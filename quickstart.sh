#!/bin/bash
# Quick setup script for CARLA JAX

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🚀 CARLA JAX Quick Setup${NC}"
echo -e "${BLUE}========================${NC}\n"

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if conda is available
if command -v conda &> /dev/null; then
    echo -e "${GREEN}✓ Conda found${NC}"
    
    # Create/update conda environment
    echo -e "\n${BLUE}Setting up conda environment...${NC}"
    
    if conda env list | grep -q "^carla-jax "; then
        echo -e "${YELLOW}Updating existing environment...${NC}"
        conda env update -f "$SCRIPT_DIR/environment.yml"
    else
        echo -e "${YELLOW}Creating new environment...${NC}"
        conda env create -f "$SCRIPT_DIR/environment.yml"
    fi
    
    echo -e "\n${GREEN}✅ Conda environment ready!${NC}"
    echo -e "\n${BLUE}To activate:${NC}"
    echo -e "  ${GREEN}conda activate carla-jax${NC}"
else
    echo -e "${YELLOW}Conda not found. Using pip fallback...${NC}"
    
    # Create virtual environment
    if [ ! -d "venv_carla_jax" ]; then
        python3 -m venv venv_carla_jax
    fi
    
    source venv_carla_jax/bin/activate
    pip install -r PythonAPI/examples_jax/requirements_jax.txt
    pip install carla==0.9.15
fi

# Run diagnostics first
echo -e "\n${BLUE}Running system diagnostics...${NC}"
python3 "$SCRIPT_DIR/diagnose_setup.py"

# Run system-wide installation
echo -e "\n${BLUE}Installing system-wide commands...${NC}"
bash "$SCRIPT_DIR/install_system_wide.sh"

echo -e "\n${GREEN}✅ Setup complete!${NC}"
echo -e "\n${BLUE}You can now run from anywhere:${NC}"
echo -e "  ${GREEN}carla-jax${NC}          # Launch CLI"
echo -e "  ${GREEN}carla-jax --status${NC} # Check status"
echo -e "  ${GREEN}cj${NC}                 # Short alias"