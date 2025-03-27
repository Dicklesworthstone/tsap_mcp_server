"""
Base classes for TSAP analysis tools.

This module provides abstract base classes for all analysis tools in TSAP,
defining common interfaces and shared functionality.
"""
import os
import time
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple, Type
from contextlib import asynccontextmanager

from tsap.utils.logging import logger
from tsap.performance_mode import get_performance_mode, get_parameter


class BaseAnalysisTool(ABC):
    """Abstract base class for all analysis tools."""
    
    def __init__(self, name: str):
        """Initialize the analysis tool.
        
        Args:
            name: Name of the analysis tool
        """
        self.name = name
        self.execution_count = 0
        self.total_execution_time = 0.0
    
    @asynccontextmanager
    async def _measure_execution_time(self):
        """Context manager to measure execution time.
        
        Yields:
            None
        """
        start_time = time.time()
        
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.execution_count += 1
            self.total_execution_time += elapsed
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics for this tool.
        
        Returns:
            Dictionary with execution statistics
        """
        avg_time = 0.0
        if self.execution_count > 0:
            avg_time = self.total_execution_time / self.execution_count
            
        return {
            "tool_name": self.name,
            "execution_count": self.execution_count,
            "total_execution_time": self.total_execution_time,
            "avg_execution_time": avg_time,
        }
    
    def reset_statistics(self) -> None:
        """Reset execution statistics."""
        self.execution_count = 0
        self.total_execution_time = 0.0
    
    async def analyze(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data with this tool.
        
        This method should be overridden by subclasses.
        
        Args:
            params: Analysis parameters
            
        Returns:
            Analysis results
        """
        raise NotImplementedError("Subclasses must implement analyze method")


class AnalysisRegistry:
    """Registry for analysis tools."""
    
    _registry: Dict[str, Type[BaseAnalysisTool]] = {}
    _instances: Dict[str, BaseAnalysisTool] = {}
    
    @classmethod
    def register(cls, name: str, tool_class: Type[BaseAnalysisTool]) -> None:
        """Register an analysis tool class.
        
        Args:
            name: Tool name
            tool_class: Tool class
        """
        cls._registry[name] = tool_class
        logger.debug(
            f"Registered analysis tool: {name}",
            component="analysis",
            operation="register_tool"
        )
    
    @classmethod
    def get_tool_class(cls, name: str) -> Optional[Type[BaseAnalysisTool]]:
        """Get an analysis tool class by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool class or None if not found
        """
        return cls._registry.get(name)
    
    @classmethod
    def get_tool(cls, name: str) -> Optional[BaseAnalysisTool]:
        """Get an analysis tool instance by name.
        
        If the tool instance doesn't exist yet, it will be created.
        
        Args:
            name: Tool name
            
        Returns:
            Tool instance or None if not found
        """
        # Check if instance already exists
        if name in cls._instances:
            return cls._instances[name]
            
        # Get the tool class
        tool_class = cls.get_tool_class(name)
        if not tool_class:
            return None
            
        # Create and store the instance
        instance = tool_class(name)
        cls._instances[name] = instance
        
        return instance
    
    @classmethod
    def list_tools(cls) -> List[str]:
        """List all registered analysis tools.
        
        Returns:
            List of tool names
        """
        return list(cls._registry.keys())


def register_analysis_tool(name: str):
    """Decorator to register an analysis tool class.
    
    Args:
        name: Tool name
        
    Returns:
        Decorator function
    """
    def decorator(cls):
        AnalysisRegistry.register(name, cls)
        return cls
    return decorator


class AnalysisContext:
    """Context for analysis operations.
    
    This class provides a shared context for analysis operations,
    including performance parameters, statistics, and caching.
    """
    
    def __init__(self):
        """Initialize the analysis context."""
        self.performance_mode = get_performance_mode()
        self.start_time = time.time()
        self.statistics = {}
        self.cache = {}
        self.results = {}
    
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get a performance parameter.
        
        Args:
            name: Parameter name
            default: Default value if parameter not found
            
        Returns:
            Parameter value
        """
        return get_parameter(name, default)
    
    def add_statistic(self, key: str, value: Any) -> None:
        """Add a statistic to the context.
        
        Args:
            key: Statistic key
            value: Statistic value
        """
        self.statistics[key] = value
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since context creation.
        
        Returns:
            Elapsed time in seconds
        """
        return time.time() - self.start_time
    
    def cache_result(self, key: str, value: Any) -> None:
        """Cache a result for reuse.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        self.cache[key] = value
    
    def get_cached_result(self, key: str) -> Optional[Any]:
        """Get a cached result.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        return self.cache.get(key)
    
    def add_result(self, key: str, value: Any) -> None:
        """Add a result to the context.
        
        Args:
            key: Result key
            value: Result value
        """
        self.results[key] = value
    
    def get_result(self, key: str) -> Optional[Any]:
        """Get a result from the context.
        
        Args:
            key: Result key
            
        Returns:
            Result value or None if not found
        """
        return self.results.get(key)
    
    def get_all_results(self) -> Dict[str, Any]:
        """Get all results.
        
        Returns:
            Dictionary of all results
        """
        return self.results.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the analysis context.
        
        Returns:
            Summary dictionary
        """
        return {
            "performance_mode": self.performance_mode,
            "elapsed_time": self.get_elapsed_time(),
            "statistics": self.statistics,
            "result_keys": list(self.results.keys()),
        }