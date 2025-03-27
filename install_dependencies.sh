#!/usr/bin/env zsh

# TSAP MCP Server - System Dependencies Installer
# This script installs all required system dependencies for the TSAP MCP Server

set -e  # Exit immediately if a command exits with a non-zero status

# Print with colors and emojis for better user experience
print_step() {
  echo "\033[1;34m🔷 $1\033[0m"
}

print_success() {
  echo "\033[1;32m✅ $1\033[0m"
}

print_error() {
  echo "\033[1;31m❌ $1\033[0m"
}

print_warning() {
  echo "\033[1;33m⚠️  $1\033[0m"
}

# Check for sudo privileges
check_sudo() {
  print_step "Checking for sudo privileges..."
  if ! command -v sudo &> /dev/null; then
    print_error "sudo not found. Please install sudo or run this script as root."
    exit 1
  fi
  
  # Test sudo access
  if ! sudo -v; then
    print_error "sudo access is required to install dependencies."
    exit 1
  fi
  print_success "sudo privileges confirmed."
}

# Update package repositories
update_repos() {
  print_step "Updating package repositories..."
  if ! sudo apt-get update; then
    print_error "Failed to update package repositories."
    exit 1
  fi
  print_success "Package repositories updated."
}

# Install core dependencies
install_core_deps() {
  print_step "Installing core dependencies..."
  
  CORE_DEPS=(
    build-essential
    curl
    git
    ripgrep
    jq
    gawk
    sqlite3
    libxml2-dev
    libxslt1-dev
    libffi-dev
    libpoppler-cpp-dev
    poppler-utils
    libssl-dev
    ca-certificates
    python3-pip  # Minimal Python for initial uv setup
  )
  
  if ! sudo apt-get install -y --no-install-recommends ${CORE_DEPS[@]}; then
    print_error "Failed to install core dependencies."
    exit 1
  fi
  print_success "Core dependencies installed."
}

# Install uv
install_uv() {
  print_step "Checking for uv..."
  
  if command -v uv &> /dev/null; then
    print_success "uv is already installed."
    return 0
  fi
  
  print_step "Installing uv..."
  
  # Install uv using the official install script
  if curl -sSf https://astral.sh/uv/install.sh | sh; then
    # Add uv to PATH for current session if it's not already there
    if [[ ! ":$PATH:" == *":$HOME/.cargo/bin:"* ]]; then
      export PATH="$HOME/.cargo/bin:$PATH"
      print_step "Added uv to PATH for current session. You may want to add this to your .zshrc:"
      echo 'export PATH="$HOME/.cargo/bin:$PATH"'
    fi
    print_success "uv installed successfully."
  else
    print_error "Failed to install uv."
    print_warning "You may need to install uv manually: https://github.com/astral-sh/uv"
    return 1
  fi
}

# Install optional visualization dependencies
install_visualization_deps() {
  print_step "Installing visualization dependencies..."
  
  VIZ_DEPS=(
    graphviz
    libgraphviz-dev
    pkg-config
  )
  
  if ! sudo apt-get install -y --no-install-recommends ${VIZ_DEPS[@]}; then
    print_warning "Failed to install some visualization dependencies."
    # Continue anyway as these are optional
  else
    print_success "Visualization dependencies installed."
  fi
}

# Verify installations
verify_installations() {
  print_step "Verifying installations..."
  
  local all_good=true
  
  # Check core tools
  for tool in ripgrep jq gawk sqlite3; do
    if ! command -v $tool &> /dev/null; then
      print_error "$tool is not installed or not in PATH."
      all_good=false
    fi
  done
  
  # Check uv
  if ! command -v uv &> /dev/null; then
    print_warning "uv is not installed or not in PATH."
    all_good=false
  fi
  
  if $all_good; then
    print_success "All core dependencies verified!"
  else
    print_warning "Some dependencies may be missing. Please check the error messages above."
  fi
}

# Main function
main() {
  echo "===================================================================="
  echo "   TSAP MCP Server - System Dependencies Installer"
  echo "===================================================================="
  echo ""
  
  check_sudo
  update_repos
  install_core_deps
  install_uv
  install_visualization_deps
  verify_installations
  
  echo ""
  echo "===================================================================="
  print_success "Installation completed!"
  echo "Next steps:"
  echo "1. Create a Python 3.13 virtual environment: 'uv venv --python=3.13 .venv'"
  echo "2. Activate the virtual environment: 'source .venv/bin/activate'"
  echo "3. Install project dependencies: 'uv pip install -e \".[dev]\"'"
  echo "===================================================================="
}

# Run the main function
main