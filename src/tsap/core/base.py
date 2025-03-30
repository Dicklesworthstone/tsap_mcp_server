"""
TSAP Composite Operations Base.

This module provides the base classes and utilities for composite operations,
which combine multiple core tools to perform more complex tasks.
"""

import time
import inspect
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Type, TypeVar, Generic
from contextlib import contextmanager, asynccontextmanager

from tsap.utils.logging import logger
from tsap.utils.errors import TSAPError
from tsap.performance_mode import get_parameter
from tsap.cache import cache_result, get_cached_result


class BaseCoreTool(ABC):
    """Base class for core tools that wrap CLI utilities."""
    
    def __init__(self, name: str):
        """Initialize a core tool.
        
        Args:
            name: Tool name
        """
        self.name = name
        self.statistics: Dict[str, Any] = {
            "calls": 0,
            "errors": 0,
            "execution_time": 0.0,
        }
    
    @asynccontextmanager
    async def _measure_execution_time(self):
        """Context manager to measure execution time and update stats."""
        start_time = time.perf_counter()
        self.statistics["calls"] += 1
        try:
            yield
        except Exception as e:
            self.statistics["errors"] += 1
            # Re-raise the exception after incrementing the error count
            raise e
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            self.statistics["execution_time"] += duration

    def get_statistics(self) -> Dict[str, Any]:
        """Get tool statistics.
        
        Returns:
            Dictionary with tool statistics
        """
        return dict(self.statistics)
    
    def reset_statistics(self) -> None:
        """Reset tool statistics."""
        self.statistics = {
            "calls": 0,
            "errors": 0,
            "execution_time": 0.0,
        }


class ToolRegistry:
    """Registry for core tools."""
    
    _tools: Dict[str, Type[BaseCoreTool]] = {}
    _instances: Dict[str, BaseCoreTool] = {}
    
    @classmethod
    def register(cls, name: str, tool_class: Type[BaseCoreTool]) -> None:
        """Register a core tool class.
        
        Args:
            name: Tool name
            tool_class: Tool class
        """
        cls._tools[name] = tool_class
        logger.debug(f"Registered core tool: {name}")
    
    @classmethod
    def get_tool_class(cls, name: str) -> Optional[Type[BaseCoreTool]]:
        """Get a core tool class by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool class or None if not found
        """
        return cls._tools.get(name)
    
    @classmethod
    def get_tool(cls, name: str) -> Optional[BaseCoreTool]:
        """Get a core tool instance by name.
        
        If an instance doesn't exist, it will be created.
        
        Args:
            name: Tool name
            
        Returns:
            Tool instance or None if not found
        """
        # Check if we already have an instance
        if name in cls._instances:
            return cls._instances[name]
        
        # Get the tool class
        tool_class = cls.get_tool_class(name)
        if not tool_class:
            return None
        
        # Create an instance
        instance = tool_class()
        cls._instances[name] = instance
        
        return instance
    
    @classmethod
    def list_tools(cls) -> List[str]:
        """Get a list of all registered tools.
        
        Returns:
            List of tool names
        """
        return list(cls._tools.keys())


def register_tool(name: str) -> Callable[[Type[BaseCoreTool]], Type[BaseCoreTool]]:
    """Decorator to register a core tool.
    
    Args:
        name: Tool name
        
    Returns:
        Decorator function
    """
    def decorator(cls: Type[BaseCoreTool]) -> Type[BaseCoreTool]:
        ToolRegistry.register(name, cls)
        return cls
    
    return decorator


class CompositeError(TSAPError):
    """Error raised when a composite operation fails."""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a composite operation error.
        
        Args:
            message: Error message
            operation: Operation name
            details: Additional error details
        """
        error_details = details or {}
        if operation:
            error_details["operation"] = operation
        
        super().__init__(message, "COMPOSITE_ERROR", error_details)


# Type variables for input and output
I = TypeVar('I')  # noqa: E741
O = TypeVar('O')  # noqa: E741


class CompositeOperation(Generic[I, O], ABC):
    """Base class for composite operations.
    
    Composite operations combine multiple core tools and other operations
    to perform more complex tasks.
    """
    
    def __init__(self, name: str):
        """Initialize a composite operation.
        
        Args:
            name: Operation name
        """
        self.name = name
        self.statistics: Dict[str, Any] = {
            "calls": 0,
            "errors": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "execution_time": 0.0,
        }
    
    @abstractmethod
    async def execute(self, params: I) -> O:
        """Execute the operation.
        
        Args:
            params: Operation parameters
            
        Returns:
            Operation result
            
        Raises:
            CompositeError: If the operation fails
        """
        pass
    
    async def execute_with_stats(self, params: I) -> O:
        """Execute the operation with statistics tracking.
        
        Args:
            params: Operation parameters
            
        Returns:
            Operation result
            
        Raises:
            CompositeError: If the operation fails
        """
        # Increment call count
        self.statistics["calls"] += 1
        
        # Get performance mode parameters
        use_cache = get_parameter("use_cache", True)
        
        # Try to get from cache
        if use_cache:
            cached_result = await get_cached_result(self.name, params)
            if cached_result is not None:
                self.statistics["cache_hits"] += 1
                return cached_result
            else:
                self.statistics["cache_misses"] += 1
        
        # Start timing
        start_time = time.time()
        
        try:
            # Execute the operation
            result = await self.execute(params)
            
            # Cache the result
            if use_cache:
                await cache_result(self.name, params, result)
            
            return result
            
        except Exception as e:
            # Increment error count
            self.statistics["errors"] += 1
            
            # Wrap in CompositeError if needed, preserving more info
            if not isinstance(e, CompositeError):
                error_message = f"Error during {self.name}: {type(e).__name__} - {str(e)}"
                error_details = {
                    "original_error_type": type(e).__name__,
                    "original_error_message": str(e),
                }
                logger.error(
                    f"Wrapping exception in CompositeError: {error_message}", 
                    operation=self.name, 
                    exception=e
                )
                raise CompositeError(
                    error_message,
                    operation=self.name,
                    details=error_details,
                ) from e
            
            # Re-raise if it was already a CompositeError
            # Log it first for visibility
            logger.error(f"Re-raising CompositeError: {e}", operation=self.name, exception=e)
            raise
            
        finally:
            # Update execution time
            execution_time = time.time() - start_time
            self.statistics["execution_time"] += execution_time
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get operation statistics.
        
        Returns:
            Dictionary with operation statistics
        """
        return dict(self.statistics)
    
    def reset_statistics(self) -> None:
        """Reset operation statistics."""
        self.statistics = {
            "calls": 0,
            "errors": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "execution_time": 0.0,
        }


class CompositeRegistry:
    """Registry for composite operations."""
    
    _operations: Dict[str, Type[CompositeOperation]] = {}
    _instances: Dict[str, CompositeOperation] = {}
    
    @classmethod
    def register(cls, name: str, operation_class: Type[CompositeOperation]) -> None:
        """Register a composite operation class.
        
        Args:
            name: Operation name
            operation_class: Operation class
        """
        cls._operations[name] = operation_class
        logger.debug(f"Registered composite operation: {name}")
    
    @classmethod
    def get_operation_class(cls, name: str) -> Optional[Type[CompositeOperation]]:
        """Get a composite operation class by name.
        
        Args:
            name: Operation name
            
        Returns:
            Operation class or None if not found
        """
        return cls._operations.get(name)
    
    @classmethod
    def get_operation(cls, name: str) -> Optional[CompositeOperation]:
        """Get a composite operation instance by name.
        
        If an instance doesn't exist, it will be created.
        
        Args:
            name: Operation name
            
        Returns:
            Operation instance or None if not found
        """
        # Check if we already have an instance
        if name in cls._instances:
            return cls._instances[name]
        
        # Get the operation class
        operation_class = cls.get_operation_class(name)
        if not operation_class:
            return None
        
        # Create an instance
        instance = operation_class(name)
        cls._instances[name] = instance
        
        return instance
    
    @classmethod
    def list_operations(cls) -> List[str]:
        """Get a list of all registered operations.
        
        Returns:
            List of operation names
        """
        return list(cls._operations.keys())


def register_operation(name: str) -> Callable[[Type[CompositeOperation]], Type[CompositeOperation]]:
    """Decorator to register a composite operation.
    
    Args:
        name: Operation name
        
    Returns:
        Decorator function
    """
    def decorator(cls: Type[CompositeOperation]) -> Type[CompositeOperation]:
        CompositeRegistry.register(name, cls)
        return cls
    
    return decorator


def create_operation_function(
    operation_class: Type[CompositeOperation],
    execute_method: Callable,
) -> Callable:
    """Create a function wrapper for a composite operation.
    
    Args:
        operation_class: Operation class
        execute_method: Execute method
        
    Returns:
        Wrapped function
    """
    # Get function signature
    sig = inspect.signature(execute_method)
    
    # Create wrapper function
    async def wrapper(*args, **kwargs):
        # Get the operation instance
        operation = CompositeRegistry.get_operation(operation_class.__name__)
        if not operation:
            # Create an instance if not registered
            operation = operation_class(operation_class.__name__)
        
        # Bind arguments to parameters
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # Extract params
        params = bound_args.arguments.get("params")
        
        # Execute with statistics
        return await operation.execute_with_stats(params)
    
    # Update wrapper metadata
    wrapper.__name__ = execute_method.__name__
    wrapper.__doc__ = execute_method.__doc__
    wrapper.__annotations__ = execute_method.__annotations__
    
    return wrapper


def operation(name: str) -> Callable[[Type[CompositeOperation]], Type[CompositeOperation]]:
    """Decorator to register a composite operation and create a function wrapper.
    
    Args:
        name: Operation name
        
    Returns:
        Decorator function
    """
    def decorator(cls: Type[CompositeOperation]) -> Type[CompositeOperation]:
        # Register the operation
        CompositeRegistry.register(name, cls)
        
        # Get the execute method
        execute_method = cls.execute
        
        # Create a function wrapper
        wrapper = create_operation_function(cls, execute_method)
        
        # Add the wrapper as a class attribute
        setattr(cls, f"{name}_operation", staticmethod(wrapper))
        
        # Register the wrapper in the module namespace
        module = inspect.getmodule(cls)
        if module:
            setattr(module, name, wrapper)
        
        return cls
    
    return decorator


# Utilities for confidence scoring

class ConfidenceLevel:
    """Confidence levels for operation results."""
    
    LOW = 0.3
    MEDIUM = 0.6
    HIGH = 0.9
    
    @staticmethod
    def to_str(confidence: float) -> str:
        """Convert a confidence value to a string representation.
        
        Args:
            confidence: Confidence value
            
        Returns:
            String representation
        """
        if confidence >= ConfidenceLevel.HIGH:
            return "high"
        elif confidence >= ConfidenceLevel.MEDIUM:
            return "medium"
        else:
            return "low"


def calculate_confidence(factors: Dict[str, float]) -> float:
    """Calculate a confidence score based on multiple factors.
    
    Args:
        factors: Dictionary mapping factor names to factor values (0.0 to 1.0)
        
    Returns:
        Overall confidence score (0.0 to 1.0)
    """
    if not factors:
        return 0.0
    
    # Calculate weighted average
    total_value = sum(factors.values())
    total_weight = len(factors)
    
    return total_value / total_weight


# Convenience functions

def get_operation(name: str) -> Optional[CompositeOperation]:
    """Get a composite operation instance by name.
    
    Args:
        name: Operation name
        
    Returns:
        Operation instance or None if not found
    """
    return CompositeRegistry.get_operation(name)


def list_operations() -> List[str]:
    """Get a list of all registered composite operations.
    
    Returns:
        List of operation names
    """
    return CompositeRegistry.list_operations()