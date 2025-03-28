"""
Base classes and utilities for composite operations.

This module defines the core abstractions for composite operations, which combine
multiple core tools to perform more complex tasks. It includes the base classes,
registry, decorators, and confidence scoring mechanisms.
"""

import time
import inspect
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Type, TypeVar, Generic
from tsap.utils.logging import logger
from tsap.utils.errors import TSAPError
from tsap.performance_mode import get_parameter
from tsap.cache import cache_result, get_cached_result

# Type variables for input and output types
I = TypeVar('I')  # Input type  # noqa: E741
O = TypeVar('O')  # Output type  # noqa: E741


class CompositeError(TSAPError):
    """
    Exception raised for errors in composite operations.
    
    Attributes:
        message: Error message
        operation: Name of the operation that caused the error
        details: Additional error details
    """
    def __init__(self, message: str, operation: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, code=f"COMPOSITE_{operation.upper()}_ERROR" if operation else "COMPOSITE_ERROR", details=details)
        self.operation = operation


class CompositeOperation(Generic[I, O], ABC):
    """
    Abstract base class for all composite operations.
    
    A composite operation combines multiple core tools to perform a more complex task.
    Each operation has a unique name, implements an execute method, and tracks execution
    statistics.
    
    Attributes:
        name: Unique identifier for the operation
        _stats: Dictionary of execution statistics
    """
    def __init__(self, name: str) -> None:
        """
        Initialize a new composite operation.
        
        Args:
            name: Unique identifier for the operation
        """
        self.name = name
        self._stats = {
            "executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_execution_time": 0.0,
            "average_execution_time": 0.0,
            "last_execution_time": None,
            "last_executed_at": None,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    @abstractmethod
    async def execute(self, params: I) -> O:
        """
        Execute the composite operation.
        
        This method must be implemented by all concrete composite operations.
        
        Args:
            params: Parameters for the operation
            
        Returns:
            Result of the operation
        
        Raises:
            CompositeError: If the operation fails
        """
        pass
    
    async def execute_with_stats(self, params: I) -> O:
        """
        Execute the operation and track execution statistics.
        
        This method wraps the execute method and tracks execution time and success/failure.
        
        Args:
            params: Parameters for the operation
            
        Returns:
            Result of the operation
            
        Raises:
            CompositeError: If the operation fails
        """
        start_time = time.time()
        self._stats["executions"] += 1
        
        try:
            # Check cache first if enabled
            use_cache = get_parameter("use_cache", True)
            if use_cache:
                cache_key = f"{self.name}:{hash(str(params))}"  # noqa: F841
                cached_result = await get_cached_result(self.name, {"params": params})
                
                if cached_result is not None:
                    self._stats["cache_hits"] += 1
                    return cached_result
                
                self._stats["cache_misses"] += 1
            
            # Execute the operation
            result = await self.execute(params)
            
            # Cache the result if caching is enabled
            if use_cache:
                await cache_result(self.name, {"params": params}, result)
            
            # Update statistics
            execution_time = time.time() - start_time
            self._stats["successful_executions"] += 1
            self._stats["total_execution_time"] += execution_time
            self._stats["average_execution_time"] = (
                self._stats["total_execution_time"] / self._stats["successful_executions"]
            )
            self._stats["last_execution_time"] = execution_time
            self._stats["last_executed_at"] = time.time()
            
            return result
            
        except Exception as e:
            # Update failure statistics
            self._stats["failed_executions"] += 1
            self._stats["last_executed_at"] = time.time()
            
            # Re-raise as CompositeError if it's not already one
            if not isinstance(e, CompositeError):
                raise CompositeError(
                    message=str(e),
                    operation=self.name,
                    details={"original_error": str(e), "error_type": type(e).__name__}
                ) from e
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get execution statistics for this operation.
        
        Returns:
            Dictionary of execution statistics
        """
        return self._stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset all execution statistics."""
        self._stats = {
            "executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_execution_time": 0.0,
            "average_execution_time": 0.0,
            "last_execution_time": None,
            "last_executed_at": None,
            "cache_hits": 0,
            "cache_misses": 0
        }


class CompositeRegistry:
    """
    Registry for composite operations.
    
    Maintains a global registry of all available composite operations.
    """
    _operations: Dict[str, Type[CompositeOperation]] = {}
    _instances: Dict[str, CompositeOperation] = {}
    
    @classmethod
    def register(cls, name: str, operation_class: Type[CompositeOperation]) -> None:
        """
        Register a composite operation class.
        
        Args:
            name: Unique identifier for the operation
            operation_class: Class implementing the operation
        """
        cls._operations[name] = operation_class
        logger.debug(f"Registered composite operation: {name}")
    
    @classmethod
    def get_operation_class(cls, name: str) -> Optional[Type[CompositeOperation]]:
        """
        Get a composite operation class by name.
        
        Args:
            name: Unique identifier for the operation
            
        Returns:
            The operation class, or None if not found
        """
        return cls._operations.get(name)
    
    @classmethod
    def get_operation(cls, name: str) -> Optional[CompositeOperation]:
        """
        Get or create an instance of a composite operation.
        
        Args:
            name: Unique identifier for the operation
            
        Returns:
            An instance of the operation, or None if not found
        """
        # Return existing instance if available
        if name in cls._instances:
            return cls._instances[name]
        
        # Create a new instance if the class is registered
        operation_class = cls.get_operation_class(name)
        if operation_class:
            instance = operation_class(name)
            cls._instances[name] = instance
            return instance
        
        return None
    
    @classmethod
    def list_operations(cls) -> List[str]:
        """
        List all registered operations.
        
        Returns:
            List of operation names
        """
        return list(cls._operations.keys())


def register_operation(name: str) -> Callable[[Type[CompositeOperation]], Type[CompositeOperation]]:
    """
    Decorator to register a composite operation class.
    
    Args:
        name: Unique identifier for the operation
        
    Returns:
        Decorator function
    """
    def decorator(operation_class: Type[CompositeOperation]) -> Type[CompositeOperation]:
        CompositeRegistry.register(name, operation_class)
        return operation_class
    return decorator


def create_operation_function(operation_class: Type[CompositeOperation], execute_method: Callable) -> Callable:
    """
    Create a standalone function that executes an operation.
    
    This is used to create the module-level functions for each operation.
    
    Args:
        operation_class: Class implementing the operation
        execute_method: The execute method of the operation
        
    Returns:
        A function that executes the operation
    """
    async def wrapper(*args, **kwargs):
        # Create an instance of the operation if it doesn't exist
        operation_name = operation_class.__name__.lower()
        operation = CompositeRegistry.get_operation(operation_name)
        if not operation:
            operation = operation_class(operation_name)
            CompositeRegistry._instances[operation_name] = operation
        
        # Determine if the first argument is the params object or if params need to be constructed
        if args and len(args) == 1 and not kwargs:
            # Assume the single argument is the params object
            return await operation.execute_with_stats(args[0])
        else:
            # Construct params from args and kwargs based on the signature
            sig = inspect.signature(execute_method)
            params = {}
            
            # Map positional arguments to parameter names
            positional_params = [p for p in sig.parameters.values() if p.name != 'self']
            for i, arg in enumerate(args):
                if i < len(positional_params):
                    params[positional_params[i].name] = arg
            
            # Add keyword arguments
            params.update(kwargs)
            
            return await operation.execute_with_stats(params)
    
    # Copy metadata from the execute method to the wrapper
    wrapper.__name__ = execute_method.__name__
    wrapper.__doc__ = execute_method.__doc__
    wrapper.__module__ = execute_method.__module__
    
    return wrapper


def operation(name: str) -> Callable[[Type[CompositeOperation]], Type[CompositeOperation]]:
    """
    Decorator to register a composite operation and create a module-level function.
    
    Args:
        name: Unique identifier for the operation
        
    Returns:
        Decorator function
    """
    def decorator(operation_class: Type[CompositeOperation]) -> Type[CompositeOperation]:
        # Register the operation
        CompositeRegistry.register(name, operation_class)
        
        # Create a module-level function with the same name
        execute_method = operation_class.execute
        
        # Add the function to the module where the class is defined
        module = inspect.getmodule(operation_class)
        if module:
            setattr(module, name, create_operation_function(operation_class, execute_method))
        
        return operation_class
    return decorator


class ConfidenceLevel:
    """
    Constants for confidence levels.
    
    These represent the confidence level of a composite operation result.
    """
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95
    CERTAIN = 1.0
    
    @staticmethod
    def to_str(confidence: float) -> str:
        """
        Convert a confidence value to a string representation.
        
        Args:
            confidence: Confidence value (0.0 to 1.0)
            
        Returns:
            String representation of the confidence level
        """
        if confidence >= ConfidenceLevel.CERTAIN:
            return "Certain"
        elif confidence >= ConfidenceLevel.VERY_HIGH:
            return "Very High"
        elif confidence >= ConfidenceLevel.HIGH:
            return "High"
        elif confidence >= ConfidenceLevel.MEDIUM:
            return "Medium"
        elif confidence >= ConfidenceLevel.LOW:
            return "Low"
        else:
            return "Very Low"


def calculate_confidence(factors: Dict[str, float]) -> float:
    """
    Calculate a confidence score based on multiple factors.
    
    Each factor is a key-value pair where the key is a description of the factor
    and the value is a confidence value between 0.0 and 1.0.
    
    Args:
        factors: Dictionary of confidence factors
        
    Returns:
        Overall confidence score (0.0 to 1.0)
    """
    if not factors:
        return 0.0
    
    # Calculate weighted average of factors
    total_weight = sum(1.0 for _ in factors.values())
    weighted_sum = sum(value for value in factors.values())
    
    # Return normalized confidence score
    return min(1.0, max(0.0, weighted_sum / total_weight))


def get_operation(name: str) -> Optional[CompositeOperation]:
    """
    Get a composite operation instance by name.
    
    Args:
        name: Unique identifier for the operation
        
    Returns:
        An instance of the operation, or None if not found
    """
    return CompositeRegistry.get_operation(name)


def list_operations() -> List[str]:
    """
    List all registered composite operations.
    
    Returns:
        List of operation names
    """
    return CompositeRegistry.list_operations()