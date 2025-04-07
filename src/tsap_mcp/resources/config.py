"""
Configuration resources for TSAP MCP Server.

This module provides MCP resource implementations for accessing configuration
and settings.
"""
import os
import json
import sys
from typing import Optional, Dict, Any
from pathlib import Path
from mcp.server.fastmcp import FastMCP


def register_config_resources(mcp: FastMCP) -> None:
    """Register all configuration-related resources with the MCP server.
    
    Args:
        mcp: FastMCP server instance
    """
    
    @mcp.resource("config://app")
    async def get_app_config() -> str:
        """Get application configuration.
        
        This resource provides access to the application's configuration.
        
        Returns:
            Application configuration as JSON string
        """
        # Try to import from original implementation
        try:
            from tsap.config import get_config
            config = get_config()
            
            # Convert to dictionary
            if hasattr(config, "dict"):
                config_dict = config.dict()
            elif hasattr(config, "__dict__"):
                config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith("_")}
            else:
                config_dict = dict(config)
                
            # Remove sensitive information
            _sanitize_config(config_dict)
                
            return json.dumps(config_dict, indent=2, default=str)
        except ImportError:
            # Fallback to basic config if original implementation not available
            config = _get_fallback_app_config()
                
            return json.dumps(config, indent=2)
    
    @mcp.resource("config://app/{section}")
    async def get_app_config_section(section: str) -> str:
        """Get a specific section of the application configuration.
        
        This resource provides access to a specific section of the application's
        configuration.
        
        Args:
            section: Configuration section to return
            
        Returns:
            Section configuration as JSON string
        """
        # Try to import from original implementation
        try:
            from tsap.config import get_config
            config = get_config()
            
            # Convert to dictionary
            if hasattr(config, "dict"):
                config_dict = config.dict()
            elif hasattr(config, "__dict__"):
                config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith("_")}
            else:
                config_dict = dict(config)
                
            # Check if section exists
            if section not in config_dict:
                return json.dumps({"error": f"Section not found: {section}"}, indent=2)
                
            # Get section
            section_dict = config_dict[section]
                
            # Remove sensitive information
            if isinstance(section_dict, dict):
                _sanitize_config(section_dict)
                
            return json.dumps({section: section_dict}, indent=2, default=str)
        except ImportError:
            # Fallback to basic config if original implementation not available
            config = _get_fallback_app_config()
            
            # Check if section exists
            if section not in config:
                return json.dumps({"error": f"Section not found: {section}"}, indent=2)
                
            return json.dumps({section: config[section]}, indent=2)
    
    @mcp.resource("config://user")
    async def get_user_config() -> str:
        """Get user configuration.
        
        This resource provides access to user-specific configuration.
        
        Returns:
            User configuration as JSON string
        """
        # Try to get user config from standard locations
        config = _get_user_config()
            
        return json.dumps(config, indent=2)
    
    @mcp.resource("config://user/{section}")
    async def get_user_config_section(section: str) -> str:
        """Get a specific section of the user configuration.
        
        This resource provides access to a specific section of the
        user-specific configuration.
        
        Args:
            section: Configuration section to return
            
        Returns:
            Section configuration as JSON string
        """
        # Try to get user config from standard locations
        config = _get_user_config()
        
        # Check if section exists
        if section not in config:
            return json.dumps({"error": f"Section not found: {section}"}, indent=2)
            
        return json.dumps({section: config[section]}, indent=2)
    
    @mcp.resource("config://env")
    async def get_env_config() -> str:
        """Get environment configuration.
        
        This resource provides access to environment variables.
        
        Returns:
            Environment configuration as JSON string
        """
        # Get environment variables
        env_vars = {}
        
        for key, value in os.environ.items():
            # Skip internal or sensitive variables
            if key.startswith(("_", "SECRET_", "PASSWORD", "KEY", "TOKEN")) or "SECRET" in key or "PASSWORD" in key or "KEY" in key or "TOKEN" in key:
                continue
                
            env_vars[key] = value
            
        return json.dumps(env_vars, indent=2)
    
    @mcp.resource("config://env/{prefix}")
    async def get_env_config_with_prefix(prefix: str) -> str:
        """Get environment configuration with a specific prefix.
        
        This resource provides access to environment variables
        that start with the specified prefix.
        
        Args:
            prefix: Prefix to filter environment variables
            
        Returns:
            Filtered environment configuration as JSON string
        """
        # Get environment variables
        env_vars = {}
        
        for key, value in os.environ.items():
            # Skip internal or sensitive variables
            if key.startswith(("_", "SECRET_", "PASSWORD", "KEY", "TOKEN")) or "SECRET" in key or "PASSWORD" in key or "KEY" in key or "TOKEN" in key:
                continue
                
            # Filter by prefix
            if prefix and not key.startswith(prefix):
                continue
                
            env_vars[key] = value
            
        return json.dumps(env_vars, indent=2)
    
    @mcp.resource("config://system")
    async def get_system_config() -> str:
        """Get system configuration.
        
        This resource provides information about the system environment,
        including Python version, platform, and other system details.
        
        Returns:
            System configuration as JSON string
        """
        import platform
        
        # Basic system info
        system_info = {
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
                "compiler": platform.python_compiler(),
                "build": platform.python_build(),
                "executable": sys.executable,
            },
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
            "paths": {
                "cwd": str(Path.cwd()),
                "python_path": sys.path,
            }
        }
        
        # Add more system information safely
        try:
            import psutil
            system_info["memory"] = {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent_used": psutil.virtual_memory().percent,
            }
            system_info["cpu"] = {
                "cores_physical": psutil.cpu_count(logical=False),
                "cores_logical": psutil.cpu_count(logical=True),
                "percent_used": psutil.cpu_percent(interval=0.1),
            }
        except ImportError:
            # psutil not available, skip detailed system info
            pass
            
        return json.dumps(system_info, indent=2, default=str)


def _sanitize_config(config: Dict[str, Any]) -> None:
    """Remove sensitive information from configuration.
    
    Args:
        config: Configuration dictionary to sanitize (modified in-place)
    """
    sensitive_keys = ["password", "secret", "token", "key", "credential", "auth"]
    
    for key in list(config.keys()):
        # Check if the key might contain sensitive information
        key_lower = key.lower()
        if any(sensitive in key_lower for sensitive in sensitive_keys):
            if isinstance(config[key], str):
                config[key] = "********"
            elif isinstance(config[key], dict):
                _sanitize_config(config[key])
        elif isinstance(config[key], dict):
            _sanitize_config(config[key])


def _get_fallback_app_config() -> Dict[str, Any]:
    """Get a fallback application configuration when the original is not available.
    
    Returns:
        Fallback configuration dictionary
    """
    # Basic fallback configuration
    config = {
        "app": {
            "name": "TSAP MCP Server",
            "version": "0.1.0",
            "description": "Text Search and Analysis Processing (TSAP) MCP Server",
        },
        "server": {
            "host": os.environ.get("TSAP_HOST", "127.0.0.1"),
            "port": int(os.environ.get("TSAP_PORT", "8000")),
            "debug": os.environ.get("TSAP_DEBUG", "false").lower() == "true",
        },
        "paths": {
            "cwd": str(Path.cwd()),
            "config_dir": str(Path.home() / ".tsap"),
            "data_dir": str(Path.home() / ".tsap" / "data"),
            "cache_dir": str(Path.home() / ".tsap" / "cache"),
        },
    }
    
    return config


def _get_user_config() -> Dict[str, Any]:
    """Get user configuration from standard locations.
    
    Returns:
        User configuration dictionary
    """
    # Standard config locations
    config_paths = [
        Path.cwd() / ".tsaprc",
        Path.cwd() / ".tsap.json",
        Path.cwd() / ".tsap.toml",
        Path.home() / ".tsaprc",
        Path.home() / ".tsap" / "config.json",
        Path.home() / ".config" / "tsap" / "config.json",
    ]
    
    # Initialize empty config
    config = {
        "preferences": {
            "theme": "default",
            "language": "en",
            "history_size": 100,
            "cache_enabled": True,
        },
        "tools": {
            "ripgrep": {
                "max_results": 1000,
                "timeout": 30,
            },
            "jq": {
                "timeout": 10,
            },
            "semantic_search": {
                "model": "default",
                "similarity_threshold": 0.7,
            },
        },
    }
    
    # Try to load config from files
    for config_path in config_paths:
        if not config_path.exists():
            continue
            
        try:
            if config_path.suffix == ".json":
                with open(config_path, "r") as f:
                    user_config = json.load(f)
                    config.update(user_config)
            elif config_path.suffix == ".toml":
                try:
                    import tomli
                    with open(config_path, "rb") as f:
                        user_config = tomli.load(f)
                        config.update(user_config)
                except ImportError:
                    pass
            else:
                # Simple key=value parsing for .tsaprc
                with open(config_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            key, value = line.split("=", 1)
                            keys = key.strip().split(".")
                            current = config
                            for k in keys[:-1]:
                                current.setdefault(k, {})
                                current = current[k]
                            try:
                                # Try to convert to appropriate type
                                if value.strip().lower() in ("true", "false"):
                                    current[keys[-1]] = value.strip().lower() == "true"
                                elif value.strip().isdigit():
                                    current[keys[-1]] = int(value.strip())
                                elif "." in value.strip() and all(p.isdigit() for p in value.strip().split(".", 1)):
                                    current[keys[-1]] = float(value.strip())
                                else:
                                    current[keys[-1]] = value.strip()
                            except Exception:
                                current[keys[-1]] = value.strip()
        except Exception:
            # Skip files with errors
            continue
    
    return config 