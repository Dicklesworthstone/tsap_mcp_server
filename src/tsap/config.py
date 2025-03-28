"""
Configuration management for TSAP MCP Server.

This module handles loading, validation, and access to configuration settings
from various sources (environment variables, config files, command line).
"""
import os
import json
from typing import Dict, Any, Optional, List

import yaml
from pydantic import BaseModel, Field, validator

from tsap.utils.logging import logger

# Default configuration file paths
DEFAULT_CONFIG_PATHS = [
    "./tsap.yaml",
    "./tsap.yml",
    "./tsap.json",
    "~/.config/tsap/config.yaml",
    "~/.tsap.yaml",
]

# Global configuration instance
_config = None


class ServerConfig(BaseModel):
    """Server configuration settings."""
    
    host: str = Field("127.0.0.1", description="Host to bind the server to")
    port: int = Field(8021, description="Port to bind the server to")
    workers: int = Field(1, description="Number of worker processes")
    debug: bool = Field(False, description="Enable debug mode")
    cors_origins: List[str] = Field(["*"], description="CORS allowed origins")
    log_level: str = Field("info", description="Logging level")
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level."""
        allowed = ['debug', 'info', 'warning', 'error', 'critical']
        if v.lower() not in allowed:
            raise ValueError(f"Log level must be one of {allowed}")
        return v.lower()


class ToolsConfig(BaseModel):
    """Configuration for core tools."""
    
    ripgrep_path: Optional[str] = Field(None, description="Path to ripgrep executable")
    awk_path: Optional[str] = Field(None, description="Path to awk executable")
    jq_path: Optional[str] = Field(None, description="Path to jq executable")
    sqlite_path: Optional[str] = Field(None, description="Path to sqlite executable")
    tool_timeout: float = Field(30.0, description="Default timeout for tool execution in seconds")
    max_process_memory: Optional[int] = Field(None, description="Maximum memory per process in MB")


class CacheConfig(BaseModel):
    """Cache configuration settings."""
    
    enabled: bool = Field(True, description="Enable caching")
    directory: str = Field("~/.cache/tsap", description="Cache directory")
    max_size: int = Field(1024, description="Maximum cache size in MB")
    ttl: int = Field(86400, description="Default TTL for cache entries in seconds")
    invalidation_strategy: str = Field("lru", description="Cache invalidation strategy")
    
    @validator('invalidation_strategy')
    def validate_strategy(cls, v):
        """Validate invalidation strategy."""
        allowed = ['lru', 'fifo', 'lifo', 'lfu']
        if v.lower() not in allowed:
            raise ValueError(f"Invalidation strategy must be one of {allowed}")
        return v.lower()


class PerformanceConfig(BaseModel):
    """Performance configuration settings."""
    
    mode: str = Field("standard", description="Performance mode (fast, standard, deep)")
    max_threads: int = Field(4, description="Maximum number of threads")
    batch_size: int = Field(1000, description="Default batch size for operations")
    
    @validator('mode')
    def validate_mode(cls, v):
        """Validate performance mode."""
        allowed = ['fast', 'standard', 'deep']
        if v.lower() not in allowed:
            raise ValueError(f"Performance mode must be one of {allowed}")
        return v.lower()


class TSAPConfig(BaseModel):
    """Main TSAP configuration model."""
    
    server: ServerConfig = Field(default_factory=ServerConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    plugins_directory: str = Field("~/.tsap/plugins", description="Plugins directory")
    templates_directory: str = Field("~/.tsap/templates", description="Templates directory")
    storage_directory: str = Field("~/.tsap/storage", description="Storage directory")
    project_directory: Optional[str] = Field(None, description="Current project directory")
    
    # Custom fields can be added dynamically
    extra: Dict[str, Any] = Field(default_factory=dict, description="Additional configuration")


def expand_path(path: str) -> str:
    """Expand user and variables in path.
    
    Args:
        path: Path to expand
        
    Returns:
        Expanded path
    """
    expanded = os.path.expanduser(path)
    expanded = os.path.expandvars(expanded)
    return expanded


def find_config_file() -> Optional[str]:
    """Find the first available configuration file from default paths.
    
    Returns:
        Path to config file or None if not found
    """
    for path in DEFAULT_CONFIG_PATHS:
        expanded_path = expand_path(path)
        if os.path.isfile(expanded_path):
            return expanded_path
    return None


def load_config_from_file(path: str) -> Dict[str, Any]:
    """Load configuration from a file.
    
    Args:
        path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file format is invalid
    """
    path = expand_path(path)
    
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
        
    logger.debug(f"Loading configuration from {path}", operation="load_config")
    
    with open(path, 'r') as f:
        if path.endswith(('.yaml', '.yml')):
            try:
                return yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML in configuration file: {e}")
        elif path.endswith('.json'):
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in configuration file: {e}")
        else:
            raise ValueError(f"Unsupported configuration file format: {path}")


def load_config_from_env() -> Dict[str, Any]:
    """Load configuration from environment variables.
    
    Environment variables should be prefixed with TSAP_.
    Nested keys are separated by double underscore.
    Example: TSAP_SERVER__PORT=8021
    
    Returns:
        Configuration dictionary
    """
    config = {}
    prefix = "TSAP_"
    
    for key, value in os.environ.items():
        if key.startswith(prefix):
            # Remove prefix and split by double underscore
            key_parts = key[len(prefix):].lower().split('__')
            
            # Convert to appropriate type
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.isdigit():
                value = int(value)
            elif value.replace('.', '', 1).isdigit() and value.count('.') == 1:
                value = float(value)
                
            # Build nested dictionary
            current = config
            for part in key_parts[:-1]:
                current = current.setdefault(part, {})
            current[key_parts[-1]] = value
            
    return config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge configuration dictionaries.
    
    Args:
        base: Base configuration dictionary
        override: Override configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
            
    return result


def load_config(
    config_file: Optional[str] = None,
    env_override: bool = True,
    defaults: Optional[Dict[str, Any]] = None,
) -> TSAPConfig:
    """Load and initialize the configuration.
    
    Args:
        config_file: Optional path to configuration file
        env_override: Whether to allow environment variables to override file config
        defaults: Optional default values
        
    Returns:
        Validated TSAPConfig instance
        
    Raises:
        FileNotFoundError: If specified config file is not found
        ValueError: If configuration validation fails
    """
    global _config
    
    # Start with defaults or empty dict
    config_data = defaults or {}
    
    # Try to find and load config file
    if config_file:
        # Use specified config file
        config_data = merge_configs(config_data, load_config_from_file(config_file))
    else:
        # Try to find a default config file
        default_file = find_config_file()
        if default_file:
            try:
                file_config = load_config_from_file(default_file)
                config_data = merge_configs(config_data, file_config)
                logger.debug(
                    f"Loaded configuration from {default_file}",
                    operation="load_config"
                )
            except (ValueError, FileNotFoundError) as e:
                logger.warning(
                    f"Error loading default config file: {e}",
                    operation="load_config"
                )
    
    # Override with environment variables if requested
    if env_override:
        env_config = load_config_from_env()
        if env_config:
            config_data = merge_configs(config_data, env_config)
            logger.debug(
                "Applied environment variable configuration overrides",
                operation="load_config"
            )
    
    # Create and validate config
    try:
        _config = TSAPConfig(**config_data)
        
        # Set log level from config
        logger.set_level(_config.server.log_level)
        
        # Log configuration load
        logger.success(
            "Configuration loaded successfully",
            operation="load_config",
            context={"mode": _config.performance.mode}
        )
        
        return _config
        
    except Exception as e:
        logger.error(
            "Failed to load configuration",
            operation="load_config",
            exception=e
        )
        raise ValueError(f"Configuration validation failed: {e}")


def get_config() -> TSAPConfig:
    """Get the current configuration.
    
    Returns:
        Current configuration instance
        
    Raises:
        RuntimeError: If configuration is not initialized
    """
    global _config
    
    if _config is None:
        # Auto-initialize with defaults
        return load_config()
        
    return _config


def get_config_as_dict() -> Dict[str, Any]:
    """Get the current configuration as a dictionary.
    
    Returns:
        Configuration as a dictionary
    """
    config = get_config()
    return config.dict()


def save_config(path: str) -> None:
    """Save the current configuration to a file.
    
    Args:
        path: Path to save configuration to
        
    Raises:
        RuntimeError: If configuration is not initialized
    """
    config = get_config_as_dict()
    path = expand_path(path)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    # Save based on file extension
    if path.endswith(('.yaml', '.yml')):
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    elif path.endswith('.json'):
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported file format for saving configuration: {path}")
        
    logger.success(f"Configuration saved to {path}", operation="save_config")