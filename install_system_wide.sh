#!/bin/bash
# System-wide installation script for CARLA JAX CLI

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}🚀 CARLA JAX System-Wide Installation${NC}"
echo -e "${BLUE}=====================================${NC}"

# Function to create shell configuration
create_shell_config() {
    local shell_rc="$1"
    local shell_name="$2"
    
    echo -e "\n${BLUE}Configuring $shell_name...${NC}"
    
    # Check if shell config file exists
    if [ ! -f "$shell_rc" ]; then
        touch "$shell_rc"
    fi
    
    # Check if already configured
    if grep -q "CARLA_JAX_HOME" "$shell_rc"; then
        echo -e "${YELLOW}CARLA JAX already configured in $shell_rc${NC}"
        echo -e "${YELLOW}Updating configuration...${NC}"
        # Remove old configuration
        sed -i '/# CARLA JAX Configuration/,/# End CARLA JAX Configuration/d' "$shell_rc"
    fi
    
    # Add configuration
    cat >> "$shell_rc" << EOF

# CARLA JAX Configuration
export CARLA_JAX_HOME="$SCRIPT_DIR"
alias carla-jax="$SCRIPT_DIR/carla-jax-universal"
alias cj="carla-jax"  # Short alias
alias cj-status="carla-jax --status"
alias cj-start="carla-jax --start-carla"
alias cj-stop="carla-jax --stop-carla"
# End CARLA JAX Configuration
EOF
    
    echo -e "${GREEN}✓ $shell_name configuration added${NC}"
}

# Function to install to system bin (requires sudo)
install_to_system_bin() {
    local install_dir="/usr/local/bin"
    local script_name="carla-jax"
    
    echo -e "\n${BLUE}Installing to system bin (requires sudo)...${NC}"
    
    # Create wrapper script
    local wrapper_content="#!/bin/bash
# CARLA JAX System Wrapper
export CARLA_JAX_HOME=\"$SCRIPT_DIR\"
exec \"$SCRIPT_DIR/carla-jax-universal\" \"\$@\"
"
    
    # Write wrapper to temp file
    local temp_file=$(mktemp)
    echo "$wrapper_content" > "$temp_file"
    chmod +x "$temp_file"
    
    # Install with sudo
    echo -e "${YELLOW}Installing to $install_dir/$script_name${NC}"
    sudo mv "$temp_file" "$install_dir/$script_name"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Installed to $install_dir/$script_name${NC}"
        return 0
    else
        echo -e "${RED}✗ Failed to install to system bin${NC}"
        return 1
    fi
}

# Function to create desktop entry (Linux)
create_desktop_entry() {
    local desktop_dir="$HOME/.local/share/applications"
    local desktop_file="$desktop_dir/carla-jax.desktop"
    
    echo -e "\n${BLUE}Creating desktop entry...${NC}"
    
    # Create directory if it doesn't exist
    mkdir -p "$desktop_dir"
    
    # Create desktop entry
    cat > "$desktop_file" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=CARLA JAX CLI
Comment=Interactive CLI for CARLA JAX-accelerated autonomous driving
Exec=$SCRIPT_DIR/carla-jax-launcher
Icon=$SCRIPT_DIR/assets/icon.png
Terminal=true
Categories=Development;Science;
Keywords=CARLA;JAX;Autonomous;Driving;Simulation;
EOF
    
    # Make it executable
    chmod +x "$desktop_file"
    
    echo -e "${GREEN}✓ Desktop entry created${NC}"
}

# Function to setup conda environment
setup_conda_env() {
    echo -e "\n${BLUE}Setting up conda environment...${NC}"
    
    # Check if conda is available
    if ! command -v conda &> /dev/null; then
        echo -e "${YELLOW}Conda not found. Skipping conda environment setup.${NC}"
        echo -e "${YELLOW}You can install conda from: https://docs.conda.io/en/latest/miniconda.html${NC}"
        return 1
    fi
    
    # Create conda environment if it doesn't exist
    if ! conda env list | grep -q "^carla-jax "; then
        echo -e "${BLUE}Creating conda environment 'carla-jax'...${NC}"
        conda create -n carla-jax python=3.8 -y
        
        # Install dependencies in the environment
        conda run -n carla-jax pip install --upgrade pip
        conda run -n carla-jax pip install jax jaxlib numpy carla==0.9.15
        
        echo -e "${GREEN}✓ Conda environment 'carla-jax' created${NC}"
    else
        echo -e "${GREEN}✓ Conda environment 'carla-jax' already exists${NC}"
    fi
}

# Main installation
main() {
    # Make scripts executable
    echo -e "\n${BLUE}Making scripts executable...${NC}"
    chmod +x "$SCRIPT_DIR/carla-jax-launcher"
    chmod +x "$SCRIPT_DIR/carla-jax"
    chmod +x "$SCRIPT_DIR/carla_jax_cli.py"
    chmod +x "$SCRIPT_DIR/install_carla.sh" 2>/dev/null || true
    
    # Detect shell and configure
    echo -e "\n${BLUE}Detecting shell configuration...${NC}"
    
    # Configure bash
    if [ -f "$HOME/.bashrc" ]; then
        create_shell_config "$HOME/.bashrc" "Bash"
    fi
    
    # Configure zsh
    if [ -f "$HOME/.zshrc" ]; then
        create_shell_config "$HOME/.zshrc" "Zsh"
    fi
    
    # Configure fish
    if [ -d "$HOME/.config/fish" ]; then
        echo -e "\n${BLUE}Configuring Fish shell...${NC}"
        local fish_config="$HOME/.config/fish/conf.d/carla-jax.fish"
        cat > "$fish_config" << EOF
# CARLA JAX Configuration for Fish
set -gx CARLA_JAX_HOME "$SCRIPT_DIR"
alias carla-jax "$SCRIPT_DIR/carla-jax-launcher"
alias cj "carla-jax"
alias cj-status "carla-jax --status"
alias cj-start "carla-jax --start-carla"
alias cj-stop "carla-jax --stop-carla"
EOF
        echo -e "${GREEN}✓ Fish configuration added${NC}"
    fi
    
    # Ask about system-wide installation
    echo -e "\n${YELLOW}Do you want to install system-wide? (requires sudo) (y/N): ${NC}"
    read -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_to_system_bin
    fi
    
    # Create desktop entry on Linux
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        create_desktop_entry
    fi
    
    # Setup conda environment
    echo -e "\n${YELLOW}Do you want to create/update the conda environment? (Y/n): ${NC}"
    read -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        setup_conda_env
    fi
    
    # Create a simple test
    echo -e "\n${BLUE}Testing installation...${NC}"
    if [ -f "/usr/local/bin/carla-jax" ]; then
        echo -e "${GREEN}✓ System-wide command available${NC}"
    fi
    
    # Show completion message
    echo -e "\n${GREEN}✅ Installation completed!${NC}"
    echo -e "\n${BLUE}Available commands:${NC}"
    echo -e "  ${GREEN}carla-jax${NC}         - Launch interactive CLI"
    echo -e "  ${GREEN}cj${NC}                - Short alias for carla-jax"
    echo -e "  ${GREEN}cj-status${NC}         - Show system status"
    echo -e "  ${GREEN}cj-start${NC}          - Start CARLA server"
    echo -e "  ${GREEN}cj-stop${NC}           - Stop CARLA server"
    
    echo -e "\n${BLUE}Usage examples:${NC}"
    echo -e "  # From anywhere in your system:"
    echo -e "  ${GREEN}carla-jax${NC}"
    echo -e "  ${GREEN}carla-jax --status${NC}"
    echo -e "  ${GREEN}carla-jax --no-conda${NC}  # Skip conda activation"
    
    echo -e "\n${YELLOW}Note: You may need to restart your shell or run:${NC}"
    echo -e "  ${GREEN}source ~/.bashrc${NC}  (or ~/.zshrc for zsh)"
    echo -e "\n${BLUE}Environment variable set:${NC}"
    echo -e "  CARLA_JAX_HOME=$SCRIPT_DIR"
}

# Run main installation
main