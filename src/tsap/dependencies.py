"""
Dependency checking and management for TSAP.

This module handles checking and installing dependencies that TSAP requires,
such as external command-line tools like ripgrep, awk, jq, and sqlite.
"""
import sys
import subprocess
import shutil
import platform
from typing import Dict, List, Any, Optional, Tuple
import importlib.metadata

from tsap.utils.logging import logger

# Required CLI tools
REQUIRED_CLI_TOOLS = [
    {
        "name": "ripgrep",
        "commands": ["rg", "ripgrep"],
        "version_args": ["--version"],
        "version_parse": lambda x: x.split()[1] if x and len(x.split()) > 1 else None,
        "install_guide": {
            "linux": "Install using your distribution's package manager: "
                    "apt install ripgrep, yum install ripgrep, etc.",
            "darwin": "Install using Homebrew: brew install ripgrep",
            "windows": "Install using Chocolatey: choco install ripgrep",
        },
    },
    {
        "name": "awk",
        "commands": ["awk", "gawk"],
        "version_args": ["--version"],
        "version_parse": lambda x: x.split()[2] if x and len(x.split()) > 2 else None,
        "install_guide": {
            "linux": "Install using your distribution's package manager: "
                    "apt install gawk, yum install gawk, etc.",
            "darwin": "Install using Homebrew: brew install gawk",
            "windows": "Install using Chocolatey: choco install gawk",
        },
    },
    {
        "name": "jq",
        "commands": ["jq"],
        "version_args": ["--version"],
        "version_parse": lambda x: x.strip(),
        "install_guide": {
            "linux": "Install using your distribution's package manager: "
                    "apt install jq, yum install jq, etc.",
            "darwin": "Install using Homebrew: brew install jq",
            "windows": "Install using Chocolatey: choco install jq",
        },
    },
    {
        "name": "sqlite",
        "commands": ["sqlite3"],
        "version_args": ["--version"],
        "version_parse": lambda x: x.strip(),
        "install_guide": {
            "linux": "Install using your distribution's package manager: "
                    "apt install sqlite3, yum install sqlite, etc.",
            "darwin": "Install using Homebrew: brew install sqlite",
            "windows": "Install using Chocolatey: choco install sqlite",
        },
    },
]

# Required Python packages
REQUIRED_PACKAGES = [
    "fastapi",
    "uvicorn",
    "pydantic",
    "typer",
    "rich",
    "click",
    "pyyaml",
    "httpx",
]


def _run_command(
    command: List[str],
    timeout: int = 5,
    check: bool = False,
    capture_output: bool = True,
) -> Tuple[int, str, str]:
    """Run a command and return exit code, stdout, and stderr.
    
    Args:
        command: Command to run
        timeout: Command timeout in seconds
        check: Whether to raise an exception on non-zero exit code
        capture_output: Whether to capture stdout and stderr
        
    Returns:
        Tuple of (exit_code, stdout, stderr)
        
    Raises:
        subprocess.SubprocessError: If command fails and check=True
    """
    try:
        proc = subprocess.run(
            command,
            timeout=timeout,
            check=check,
            capture_output=capture_output,
            text=True,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.SubprocessError as e:
        if check:
            raise
        return 1, "", str(e)


def _find_command(commands: List[str]) -> Optional[str]:
    """Find the first available command from a list of options.
    
    Args:
        commands: List of command names to check
        
    Returns:
        Path to the command or None if not found
    """
    for cmd in commands:
        path = shutil.which(cmd)
        if path:
            return path
    return None


def check_cli_tool(tool_info: Dict[str, Any]) -> Dict[str, Any]:
    """Check if a CLI tool is installed and get its version.
    
    Args:
        tool_info: Tool information dictionary
        
    Returns:
        Dictionary with installation status and details
    """
    name = tool_info["name"]
    commands = tool_info["commands"]
    
    result = {
        "name": name,
        "installed": False,
        "status": "Not found",
        "path": None,
        "version": None,
        "message": f"{name} is not installed or not in PATH",
        "install_guide": tool_info["install_guide"].get(
            platform.system().lower(),
            f"Please install {name} manually",
        ),
    }
    
    # Find the command
    cmd_path = _find_command(commands)
    if not cmd_path:
        return result
        
    # Command found, get version
    version_args = tool_info.get("version_args", ["--version"])
    version_parse = tool_info.get("version_parse", lambda x: x.strip())
    
    exit_code, stdout, stderr = _run_command([cmd_path] + version_args)
    
    if exit_code != 0:
        result["status"] = "Found but failed to get version"
        result["path"] = cmd_path
        result["message"] = f"{name} was found at {cmd_path} but failed to get version"
        return result
        
    # Try to parse version from output
    version_output = stdout or stderr
    version = version_parse(version_output)
    
    result["installed"] = True
    result["status"] = "Installed"
    result["path"] = cmd_path
    result["version"] = version
    result["message"] = f"{name} is installed at {cmd_path}" + (f" (version {version})" if version else "")
    
    return result


def check_python_package(package_name: str) -> Dict[str, Any]:
    """Check if a Python package is installed and get its version.
    
    Args:
        package_name: Package name
        
    Returns:
        Dictionary with installation status and details
    """
    result = {
        "name": package_name,
        "installed": False,
        "status": "Not found",
        "version": None,
        "message": f"{package_name} is not installed",
    }
    
    try:
        version = importlib.metadata.version(package_name)
        result["installed"] = True
        result["status"] = "Installed"
        result["version"] = version
        result["message"] = f"{package_name} is installed (version {version})"
    except importlib.metadata.PackageNotFoundError:
        pass
        
    return result


def check_dependencies(fix: bool = False) -> List[Dict[str, Any]]:
    """Check if all required dependencies are installed.
    
    Args:
        fix: Whether to attempt to fix missing dependencies
        
    Returns:
        List of dependency check results
    """
    logger.info(
        "Checking dependencies...",
        component="dependencies",
        operation="check"
    )
    
    results = []
    
    # Check CLI tools
    for tool_info in REQUIRED_CLI_TOOLS:
        result = check_cli_tool(tool_info)
        results.append(result)
        
        if not result["installed"] and fix:
            logger.info(
                f"Attempting to fix missing dependency: {tool_info['name']}",
                component="dependencies",
                operation="fix"
            )
            # TODO: Implement dependency installation
            logger.warning(
                f"Automatic installation not implemented for {tool_info['name']}",
                component="dependencies",
                operation="fix",
                details=[result["install_guide"]]
            )
    
    # Check Python packages
    for package_name in REQUIRED_PACKAGES:
        result = check_python_package(package_name)
        results.append(result)
        
        if not result["installed"] and fix:
            logger.info(
                f"Attempting to install Python package: {package_name}",
                component="dependencies",
                operation="fix"
            )
            try:
                with logger.catch_and_log(
                    component="dependencies",
                    operation="install_package",
                    reraise=False
                ):
                    # Use subprocess to avoid affecting the current process
                    exit_code, stdout, stderr = _run_command(
                        [sys.executable, "-m", "pip", "install", package_name],
                        timeout=60,
                    )
                    
                    if exit_code == 0:
                        # Check again to confirm installation
                        result = check_python_package(package_name)
                        # Update the result in our list
                        results[-1] = result
                        
                        if result["installed"]:
                            logger.success(
                                f"Successfully installed {package_name} {result['version']}",
                                component="dependencies",
                                operation="fix"
                            )
                        else:
                            logger.error(
                                f"Failed to install {package_name} (unexpected error)",
                                component="dependencies",
                                operation="fix"
                            )
                    else:
                        logger.error(
                            f"Failed to install {package_name}: {stderr}",
                            component="dependencies",
                            operation="fix"
                        )
            except Exception as e:
                logger.error(
                    f"Failed to install {package_name}: {str(e)}",
                    component="dependencies",
                    operation="fix",
                    exception=e
                )
    
    # Log summary
    installed_count = sum(1 for r in results if r["installed"])
    total_count = len(results)
    
    if installed_count == total_count:
        logger.success(
            f"All dependencies are installed ({installed_count}/{total_count})",
            component="dependencies",
            operation="check"
        )
    else:
        logger.warning(
            f"Some dependencies are missing ({installed_count}/{total_count} installed)",
            component="dependencies",
            operation="check",
            details=[r["message"] for r in results if not r["installed"]]
        )
    
    return results


def install_dependency(name: str) -> bool:
    """Attempt to install a dependency.
    
    Args:
        name: Dependency name
        
    Returns:
        Whether installation was successful
    """
    # This is a placeholder for now
    logger.error(
        f"Automatic installation not implemented for {name}",
        component="dependencies",
        operation="install"
    )
    return False