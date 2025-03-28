"""
Performance profiling tools for TSAP.

This module provides tools for profiling functions, code sections, and memory usage.
It includes components for tracking execution time, memory consumption, and resource
utilization, as well as tools for identifying performance bottlenecks.
"""

import time
import functools
import cProfile
import pstats
import io
import tracemalloc
import threading
import asyncio
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager

from tsap.utils.errors import TSAPError


class ProfilerError(TSAPError):
    """
    Exception raised for errors in profiling operations.
    
    Attributes:
        message: Error message
        details: Additional error details
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, code="PROFILER_ERROR", details=details)


@dataclass
class ProfileResult:
    """
    Result of a profiling operation.
    
    Attributes:
        name: Name of the profiled function or section
        execution_time: Total execution time in seconds
        calls: Number of calls
        timestamp: Time when the profiling was performed
        metadata: Additional metadata
        stats: Detailed statistics
    """
    name: str
    execution_time: float
    calls: int = 1
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "execution_time": self.execution_time,
            "calls": self.calls,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "stats": self.stats
        }


@dataclass
class MemoryProfileResult:
    """
    Result of a memory profiling operation.
    
    Attributes:
        name: Name of the profiled function or section
        memory_usage: Memory usage in bytes
        peak_memory: Peak memory usage in bytes
        timestamp: Time when the profiling was performed
        metadata: Additional metadata
        stats: Detailed statistics
    """
    name: str
    memory_usage: int
    peak_memory: int
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "memory_usage": self.memory_usage,
            "peak_memory": self.peak_memory,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "stats": self.stats
        }


class Profiler:
    """
    Base class for profilers.
    
    Attributes:
        name: Name of the profiler
        results: Dictionary of profile results by name
    """
    def __init__(self, name: str) -> None:
        """
        Initialize the profiler.
        
        Args:
            name: Name of the profiler
        """
        self.name = name
        self.results: Dict[str, Any] = {}
        self._lock = threading.Lock()
    
    def reset(self) -> None:
        """Reset all profile results."""
        with self._lock:
            self.results = {}
    
    def get_result(self, name: str) -> Optional[Any]:
        """
        Get a profile result by name.
        
        Args:
            name: Name of the profile result
            
        Returns:
            Profile result, or None if not found
        """
        with self._lock:
            return self.results.get(name)
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get all profile results.
        
        Returns:
            Dictionary of profile results by name
        """
        with self._lock:
            return self.results.copy()
    
    def add_result(self, result: Any) -> None:
        """
        Add a profile result.
        
        Args:
            result: Profile result to add
        """
        with self._lock:
            self.results[result.name] = result


class FunctionProfiler(Profiler):
    """
    Profiler for measuring function execution time.
    
    Attributes:
        name: Name of the profiler
        results: Dictionary of profile results by name
        use_cprofile: Whether to use cProfile for detailed profiling
    """
    def __init__(self, name: str = "function_profiler", use_cprofile: bool = False) -> None:
        """
        Initialize the function profiler.
        
        Args:
            name: Name of the profiler
            use_cprofile: Whether to use cProfile for detailed profiling
        """
        super().__init__(name)
        self.use_cprofile = use_cprofile
    
    def profile(self, func: Callable) -> Callable:
        """
        Profile a function.
        
        This decorator measures the execution time of the function and records the result.
        
        Args:
            func: Function to profile
            
        Returns:
            Decorated function
        """
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            function_name = func.__name__
            
            # Basic timing
            start_time = time.time()
            
            if self.use_cprofile:
                # Use cProfile for detailed profiling
                profiler = cProfile.Profile()
                profiler.enable()
                
                try:
                    result = func(*args, **kwargs)
                finally:
                    profiler.disable()
                    
                    # Get stats
                    s = io.StringIO()
                    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
                    ps.print_stats(20)  # Print top 20 functions
                    stats_str = s.getvalue()
                    
                    # Extract key metrics
                    stats = {
                        "profile_text": stats_str,
                        "function_calls": ps.total_calls,
                        "primitive_calls": ps.prim_calls,
                    }
                    
            else:
                # Simple timing only
                try:
                    result = func(*args, **kwargs)
                    stats = {}
                finally:
                    pass
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Create profile result
            profile_result = ProfileResult(
                name=function_name,
                execution_time=execution_time,
                calls=1,
                metadata={
                    "function": function_name,
                    "module": func.__module__,
                    "doc": func.__doc__
                },
                stats=stats
            )
            
            # Add result
            existing_result = self.get_result(function_name)
            if existing_result:
                # Update existing result
                existing_result.execution_time += execution_time
                existing_result.calls += 1
                existing_result.timestamp = datetime.now()
                if stats:
                    existing_result.stats = stats  # Use the latest stats
            else:
                # Add new result
                self.add_result(profile_result)
            
            return result
        
        return wrapper
    
    async def profile_async(self, func: Callable) -> Callable:
        """
        Profile an asynchronous function.
        
        This decorator measures the execution time of the async function and records the result.
        
        Args:
            func: Async function to profile
            
        Returns:
            Decorated async function
        """
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            function_name = func.__name__
            
            # Basic timing
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
            finally:
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Create profile result
                profile_result = ProfileResult(
                    name=function_name,
                    execution_time=execution_time,
                    calls=1,
                    metadata={
                        "function": function_name,
                        "module": func.__module__,
                        "doc": func.__doc__,
                        "is_async": True
                    },
                    stats={}
                )
                
                # Add result
                existing_result = self.get_result(function_name)
                if existing_result:
                    # Update existing result
                    existing_result.execution_time += execution_time
                    existing_result.calls += 1
                    existing_result.timestamp = datetime.now()
                else:
                    # Add new result
                    self.add_result(profile_result)
            
            return result
        
        return wrapper
    
    @contextmanager
    def profile_section(self, section_name: str) -> None:
        """
        Profile a section of code.
        
        This context manager measures the execution time of a code section and records the result.
        
        Args:
            section_name: Name of the code section
            
        Yields:
            None
        """
        # Basic timing
        start_time = time.time()
        
        try:
            yield
        finally:
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Create profile result
            profile_result = ProfileResult(
                name=section_name,
                execution_time=execution_time,
                calls=1,
                metadata={
                    "section": section_name,
                    "type": "code_section"
                },
                stats={}
            )
            
            # Add result
            existing_result = self.get_result(section_name)
            if existing_result:
                # Update existing result
                existing_result.execution_time += execution_time
                existing_result.calls += 1
                existing_result.timestamp = datetime.now()
            else:
                # Add new result
                self.add_result(profile_result)


class MemoryProfiler(Profiler):
    """
    Profiler for measuring memory usage.
    
    Attributes:
        name: Name of the profiler
        results: Dictionary of profile results by name
    """
    def __init__(self, name: str = "memory_profiler") -> None:
        """
        Initialize the memory profiler.
        
        Args:
            name: Name of the profiler
        """
        super().__init__(name)
        self._tracemalloc_started = False
    
    def _ensure_tracemalloc_started(self) -> None:
        """Ensure tracemalloc is started."""
        if not self._tracemalloc_started:
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            self._tracemalloc_started = True
    
    def profile(self, func: Callable) -> Callable:
        """
        Profile memory usage of a function.
        
        This decorator measures the memory usage of the function and records the result.
        
        Args:
            func: Function to profile
            
        Returns:
            Decorated function
        """
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            function_name = func.__name__
            
            # Ensure tracemalloc is started
            self._ensure_tracemalloc_started()
            
            # Reset and take snapshot
            tracemalloc.clear_traces()
            start_snapshot = tracemalloc.take_snapshot()
            
            try:
                result = func(*args, **kwargs)
            finally:
                # Take snapshot after execution
                end_snapshot = tracemalloc.take_snapshot()
                
                # Compare snapshots
                stats = end_snapshot.compare_to(start_snapshot, 'lineno')
                
                # Calculate memory usage
                memory_usage = sum(stat.size_diff for stat in stats)
                peak_memory = max(stat.size_diff for stat in stats) if stats else 0
                
                # Get detailed statistics
                top_stats = stats[:10]  # Top 10 memory consumers
                detailed_stats = [
                    {
                        "file": str(stat.traceback.frame.filename),
                        "line": stat.traceback.frame.lineno,
                        "size": stat.size,
                        "size_diff": stat.size_diff
                    }
                    for stat in top_stats
                ]
                
                # Create memory profile result
                profile_result = MemoryProfileResult(
                    name=function_name,
                    memory_usage=memory_usage,
                    peak_memory=peak_memory,
                    metadata={
                        "function": function_name,
                        "module": func.__module__,
                        "doc": func.__doc__
                    },
                    stats={
                        "detailed_stats": detailed_stats,
                        "total_memory_diff": memory_usage
                    }
                )
                
                # Add result
                self.add_result(profile_result)
            
            return result
        
        return wrapper
    
    async def profile_async(self, func: Callable) -> Callable:
        """
        Profile memory usage of an asynchronous function.
        
        This decorator measures the memory usage of the async function and records the result.
        
        Args:
            func: Async function to profile
            
        Returns:
            Decorated async function
        """
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            function_name = func.__name__
            
            # Ensure tracemalloc is started
            self._ensure_tracemalloc_started()
            
            # Reset and take snapshot
            tracemalloc.clear_traces()
            start_snapshot = tracemalloc.take_snapshot()
            
            try:
                result = await func(*args, **kwargs)
            finally:
                # Take snapshot after execution
                end_snapshot = tracemalloc.take_snapshot()
                
                # Compare snapshots
                stats = end_snapshot.compare_to(start_snapshot, 'lineno')
                
                # Calculate memory usage
                memory_usage = sum(stat.size_diff for stat in stats)
                peak_memory = max(stat.size_diff for stat in stats) if stats else 0
                
                # Get detailed statistics
                top_stats = stats[:10]  # Top 10 memory consumers
                detailed_stats = [
                    {
                        "file": str(stat.traceback.frame.filename),
                        "line": stat.traceback.frame.lineno,
                        "size": stat.size,
                        "size_diff": stat.size_diff
                    }
                    for stat in top_stats
                ]
                
                # Create memory profile result
                profile_result = MemoryProfileResult(
                    name=function_name,
                    memory_usage=memory_usage,
                    peak_memory=peak_memory,
                    metadata={
                        "function": function_name,
                        "module": func.__module__,
                        "doc": func.__doc__,
                        "is_async": True
                    },
                    stats={
                        "detailed_stats": detailed_stats,
                        "total_memory_diff": memory_usage
                    }
                )
                
                # Add result
                self.add_result(profile_result)
            
            return result
        
        return wrapper
    
    @contextmanager
    def profile_section(self, section_name: str) -> None:
        """
        Profile memory usage of a code section.
        
        This context manager measures the memory usage of a code section and records the result.
        
        Args:
            section_name: Name of the code section
            
        Yields:
            None
        """
        # Ensure tracemalloc is started
        self._ensure_tracemalloc_started()
        
        # Reset and take snapshot
        tracemalloc.clear_traces()
        start_snapshot = tracemalloc.take_snapshot()
        
        try:
            yield
        finally:
            # Take snapshot after execution
            end_snapshot = tracemalloc.take_snapshot()
            
            # Compare snapshots
            stats = end_snapshot.compare_to(start_snapshot, 'lineno')
            
            # Calculate memory usage
            memory_usage = sum(stat.size_diff for stat in stats)
            peak_memory = max(stat.size_diff for stat in stats) if stats else 0
            
            # Get detailed statistics
            top_stats = stats[:10]  # Top 10 memory consumers
            detailed_stats = [
                {
                    "file": str(stat.traceback.frame.filename),
                    "line": stat.traceback.frame.lineno,
                    "size": stat.size,
                    "size_diff": stat.size_diff
                }
                for stat in top_stats
            ]
            
            # Create memory profile result
            profile_result = MemoryProfileResult(
                name=section_name,
                memory_usage=memory_usage,
                peak_memory=peak_memory,
                metadata={
                    "section": section_name,
                    "type": "code_section"
                },
                stats={
                    "detailed_stats": detailed_stats,
                    "total_memory_diff": memory_usage
                }
            )
            
            # Add result
            self.add_result(profile_result)


# Singleton instances
_function_profiler_instance = None
_memory_profiler_instance = None


def get_function_profiler() -> FunctionProfiler:
    """
    Get the global function profiler instance.
    
    Returns:
        Function profiler instance
    """
    global _function_profiler_instance
    if _function_profiler_instance is None:
        _function_profiler_instance = FunctionProfiler()
    return _function_profiler_instance


def get_memory_profiler() -> MemoryProfiler:
    """
    Get the global memory profiler instance.
    
    Returns:
        Memory profiler instance
    """
    global _memory_profiler_instance
    if _memory_profiler_instance is None:
        _memory_profiler_instance = MemoryProfiler()
    return _memory_profiler_instance


def profile_function(func: Callable) -> Callable:
    """
    Profile a function.
    
    This decorator measures the execution time of the function and records the result.
    
    Args:
        func: Function to profile
        
    Returns:
        Decorated function
    """
    if asyncio.iscoroutinefunction(func):
        return get_function_profiler().profile_async(func)
    else:
        return get_function_profiler().profile(func)


def profile_memory_usage(func: Callable) -> Callable:
    """
    Profile memory usage of a function.
    
    This decorator measures the memory usage of the function and records the result.
    
    Args:
        func: Function to profile
        
    Returns:
        Decorated function
    """
    if asyncio.iscoroutinefunction(func):
        return get_memory_profiler().profile_async(func)
    else:
        return get_memory_profiler().profile(func)


@contextmanager
def profile_code_section(section_name: str, profile_memory: bool = False) -> None:
    """
    Profile a section of code.
    
    This context manager measures the execution time and optionally memory usage
    of a code section and records the result.
    
    Args:
        section_name: Name of the code section
        profile_memory: Whether to profile memory usage
        
    Yields:
        None
    """
    function_profiler = get_function_profiler()
    
    with function_profiler.profile_section(section_name):
        if profile_memory:
            memory_profiler = get_memory_profiler()
            with memory_profiler.profile_section(section_name):
                yield
        else:
            yield


def get_profile_results() -> Dict[str, ProfileResult]:
    """
    Get all function profile results.
    
    Returns:
        Dictionary of profile results by function name
    """
    return get_function_profiler().get_results()


def get_memory_profile_results() -> Dict[str, MemoryProfileResult]:
    """
    Get all memory profile results.
    
    Returns:
        Dictionary of memory profile results by function name
    """
    return get_memory_profiler().get_results()


def reset_profilers() -> None:
    """Reset all profilers."""
    get_function_profiler().reset()
    get_memory_profiler().reset()