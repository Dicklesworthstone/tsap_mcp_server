"""
Performance mode management for TSAP MCP Server.

This module handles different performance modes that balance speed vs depth
of analysis in TSAP operations.
"""
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum
import os
import threading

from tsap.utils.logging import logger

# Thread-local storage for performance mode
_thread_local = threading.local()


class PerformanceMode(str, Enum):
    """Available performance modes."""
    
    FAST = "fast"
    STANDARD = "standard"
    DEEP = "deep"


# Default mode configurations
DEFAULT_MODE_CONFIGS = {
    PerformanceMode.FAST: {
        "max_matches": 100,                # Maximum matches to return
        "max_depth": 2,                    # Maximum recursion depth
        "parallel_processes": 8,           # Number of parallel processes
        "context_lines": 1,                # Number of context lines around matches
        "timeout": 10,                     # Operation timeout in seconds
        "cache_ttl": 3600,                 # Cache TTL in seconds
        "max_file_size": 1024 * 1024,      # Max file size in bytes (1MB)
        "evolution_generations": 2,        # Evolution generations
        "evolution_population": 10,        # Evolution population size
        "pattern_complexity": "low",       # Pattern complexity
        "sample_ratio": 0.1,               # Sampling ratio for large datasets
        "max_iterations": 2,               # Maximum iterations for recursive operations
    },
    PerformanceMode.STANDARD: {
        "max_matches": 1000,               # Maximum matches to return
        "max_depth": 5,                    # Maximum recursion depth
        "parallel_processes": 4,           # Number of parallel processes
        "context_lines": 2,                # Number of context lines around matches
        "timeout": 30,                     # Operation timeout in seconds
        "cache_ttl": 86400,                # Cache TTL in seconds
        "max_file_size": 10 * 1024 * 1024, # Max file size in bytes (10MB)
        "evolution_generations": 5,        # Evolution generations
        "evolution_population": 20,        # Evolution population size
        "pattern_complexity": "medium",    # Pattern complexity
        "sample_ratio": 0.25,              # Sampling ratio for large datasets
        "max_iterations": 5,               # Maximum iterations for recursive operations
    },
    PerformanceMode.DEEP: {
        "max_matches": 10000,              # Maximum matches to return
        "max_depth": 10,                   # Maximum recursion depth
        "parallel_processes": 2,           # Number of parallel processes (fewer but more thorough)
        "context_lines": 5,                # Number of context lines around matches
        "timeout": 120,                    # Operation timeout in seconds
        "cache_ttl": 604800,               # Cache TTL in seconds (1 week)
        "max_file_size": 100 * 1024 * 1024, # Max file size in bytes (100MB)
        "evolution_generations": 10,       # Evolution generations
        "evolution_population": 50,        # Evolution population size
        "pattern_complexity": "high",      # Pattern complexity
        "sample_ratio": 1.0,               # No sampling (use all data)
        "max_iterations": 10,              # Maximum iterations for recursive operations
    }
}

# Current mode configuration (starts with None, will be populated on first access)
_current_mode_config = None


def get_performance_mode() -> str:
    """Get the current performance mode.
    
    Returns:
        Current performance mode name
    """
    # Check thread-local storage first
    if hasattr(_thread_local, 'mode'):
        return _thread_local.mode
        
    # Otherwise use the global mode from environment or default to STANDARD
    return os.environ.get('TSAP_PERFORMANCE_MODE', PerformanceMode.STANDARD)


def set_performance_mode(mode: str) -> None:
    """Set the performance mode.
    
    Args:
        mode: Performance mode (fast, standard, deep)
        
    Raises:
        ValueError: If mode is invalid
    """
    mode = mode.lower()
    
    # Validate mode
    if mode not in [m.value for m in PerformanceMode]:
        valid_modes = ", ".join([m.value for m in PerformanceMode])
        raise ValueError(f"Invalid performance mode: {mode}. Valid modes: {valid_modes}")
    
    # Store in thread-local storage for thread safety
    _thread_local.mode = mode
    
    # Reset current mode config to force reload
    global _current_mode_config
    _current_mode_config = None
    
    # Log mode change
    logger.info(
        f"Performance mode set to {mode}",
        component="performance",
        operation="set_mode",
        context={"mode": mode}
    )


def get_mode_config() -> Dict[str, Any]:
    """Get the configuration for the current performance mode.
    
    Returns:
        Mode configuration dictionary
    """
    global _current_mode_config
    
    # Use cached config if available
    if _current_mode_config is not None:
        return _current_mode_config
        
    # Get current mode
    mode = get_performance_mode()
    
    # Get default config for this mode
    mode_enum = PerformanceMode(mode)
    config = DEFAULT_MODE_CONFIGS[mode_enum].copy()
    
    # TODO: Apply overrides from user configuration
    # This would be implemented once the config module is available
    
    # Cache the result
    _current_mode_config = config
    
    return config


def get_parameter(name: str, default: Any = None) -> Any:
    """Get a specific parameter for the current performance mode.
    
    Args:
        name: Parameter name
        default: Default value if parameter is not found
        
    Returns:
        Parameter value
    """
    config = get_mode_config()
    return config.get(name, default)


def with_performance_mode(mode: str):
    """Decorator to temporarily set performance mode for a function.
    
    Args:
        mode: Performance mode to use
        
    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Save original mode
            original_mode = get_performance_mode()
            
            try:
                # Set temporary mode
                set_performance_mode(mode)
                
                # Call function
                return func(*args, **kwargs)
            finally:
                # Restore original mode
                set_performance_mode(original_mode)
                
        return wrapper
    return decorator


def describe_current_mode() -> Dict[str, Any]:
    """Get a human-readable description of the current performance mode.
    
    Returns:
        Dictionary with mode description
    """
    mode = get_performance_mode()
    config = get_mode_config()
    
    descriptions = {
        PerformanceMode.FAST: (
            "Fast mode prioritizes speed over thoroughness. "
            "It uses aggressive limits and sampling to provide quick results, "
            "but may miss some matches or patterns."
        ),
        PerformanceMode.STANDARD: (
            "Standard mode balances speed and thoroughness. "
            "It provides good coverage and pattern detection while "
            "maintaining reasonable performance."
        ),
        PerformanceMode.DEEP: (
            "Deep mode prioritizes thoroughness over speed. "
            "It performs exhaustive analysis with minimal limits, "
            "but may take significantly longer to complete."
        )
    }
    
    return {
        "mode": mode,
        "description": descriptions[PerformanceMode(mode)],
        "config": config,
    }