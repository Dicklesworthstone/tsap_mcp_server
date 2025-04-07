#!/usr/bin/env python3
"""
Add MCP dependency to the project.

This script adds the MCP Python SDK dependency to the project's
pyproject.toml file, ensuring it's available for the MCP-native
implementation.
"""
import os
import sys
import re
import subprocess
from pathlib import Path


def find_pyproject_toml() -> Path:
    """Find the pyproject.toml file in the project.
    
    Searches up the directory tree from the current directory
    to find the pyproject.toml file.
    
    Returns:
        Path to pyproject.toml
    """
    # Start from the script's directory
    current_dir = Path(__file__).parent
    
    # Search up the directory tree
    while current_dir != current_dir.parent:
        pyproject_path = current_dir / "pyproject.toml"
        if pyproject_path.exists():
            return pyproject_path
        current_dir = current_dir.parent
    
    # Search from current working directory if not found
    current_dir = Path.cwd()
    while current_dir != current_dir.parent:
        pyproject_path = current_dir / "pyproject.toml"
        if pyproject_path.exists():
            return pyproject_path
        current_dir = current_dir.parent
    
    raise FileNotFoundError("Could not find pyproject.toml file")


def add_dependency_to_pyproject(pyproject_path: Path, dependency: str, version: str) -> bool:
    """Add a dependency to the pyproject.toml file.
    
    Args:
        pyproject_path: Path to pyproject.toml
        dependency: Dependency name
        version: Dependency version
        
    Returns:
        True if the dependency was added or already exists, False otherwise
    """
    # Read the file
    with open(pyproject_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check if dependency already exists
    dependency_pattern = rf'"{dependency}[>=<~].*?"'
    if re.search(dependency_pattern, content):
        print(f"Dependency {dependency} already exists in pyproject.toml")
        return True
    
    # Find the dependencies section
    dependencies_match = re.search(r'dependencies\s*=\s*\[(.*?)\]', content, re.DOTALL)
    if not dependencies_match:
        print(f"Could not find dependencies section in {pyproject_path}")
        return False
    
    # Extract the dependencies section
    dependencies_section = dependencies_match.group(0)
    
    # Add the new dependency at the end of the section, before the closing bracket
    closing_bracket_pos = dependencies_section.rfind("]")
    if closing_bracket_pos == -1:
        print(f"Invalid dependencies section format in {pyproject_path}")
        return False
    
    new_dependency = f'    "{dependency}>={version}",              # Model Context Protocol SDK\n'
    
    # Split the section and insert the new dependency
    prefix = dependencies_section[:closing_bracket_pos]
    suffix = dependencies_section[closing_bracket_pos:]
    new_section = f"{prefix}\n{new_dependency}{suffix}"
    
    # Replace the old section with the new one
    new_content = content.replace(dependencies_section, new_section)
    
    # Write the updated content back to the file
    with open(pyproject_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print(f"Added dependency {dependency}>={version} to {pyproject_path}")
    return True


def add_optional_dependency_to_pyproject(pyproject_path: Path) -> bool:
    """Add an optional dependency section for MCP.
    
    Args:
        pyproject_path: Path to pyproject.toml
        
    Returns:
        True if the optional dependency was added, False otherwise
    """
    # Read the file
    with open(pyproject_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check if mcp optional dependency already exists
    if "mcp = [" in content:
        print("MCP optional dependency already exists in pyproject.toml")
        return True
    
    # Find the optional dependencies section
    optional_deps_match = re.search(r'\[project\.optional-dependencies\](.*?)(?=\[|\Z)', content, re.DOTALL)
    if not optional_deps_match:
        print(f"Could not find [project.optional-dependencies] section in {pyproject_path}")
        return False
    
    # Extract the dependencies section
    optional_deps_section = optional_deps_match.group(0)
    
    # Create the new section
    mcp_section = """
# MCP-native server dependencies
mcp = [
    "mcp>=0.1.0",                  # Model Context Protocol SDK
]
"""
    
    # Combine the sections
    new_section = f"{optional_deps_section}{mcp_section}"
    
    # Replace the old section with the new one
    new_content = content.replace(optional_deps_section, new_section)
    
    # Write the updated content back to the file
    with open(pyproject_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print(f"Added optional dependency section for MCP to {pyproject_path}")
    return True


def install_dependency():
    """Install the MCP dependency using the current package manager."""
    try:
        # Try uv first (modern Python package manager)
        result = subprocess.run(
            ["uv", "pip", "install", "mcp"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            print("Installed MCP dependency using uv")
            return True
    except FileNotFoundError:
        pass  # uv not installed, try pip
    
    try:
        # Try pip
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "mcp"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            print("Installed MCP dependency using pip")
            return True
    except FileNotFoundError:
        pass  # pip not available
    
    print("Could not install MCP dependency automatically")
    print("Please install manually with: pip install mcp")
    return False


def main():
    """Main entry point."""
    # Find pyproject.toml
    try:
        pyproject_path = find_pyproject_toml()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Add dependency
    success = add_dependency_to_pyproject(pyproject_path, "mcp", "0.1.0")
    if not success:
        print("Failed to add MCP dependency to pyproject.toml")
        sys.exit(1)
    
    # Add optional dependency section
    success = add_optional_dependency_to_pyproject(pyproject_path)
    if not success:
        print("Failed to add MCP optional dependency to pyproject.toml")
        print("You can manually add it by adding the following section:")
        print('[project.optional-dependencies]\nmcp = ["mcp>=0.1.0"]')
    
    # Install dependency
    install_dependency()
    
    print("""
MCP dependency has been added to the project. You can now:

1. Run the MCP-native server: python -m tsap_mcp run
2. Install in Claude Desktop: python -m tsap_mcp install
3. Run in development mode: python -m tsap_mcp dev

See the migration guide for more details: src/scripts/migration_guide.md
""")


if __name__ == "__main__":
    main() 