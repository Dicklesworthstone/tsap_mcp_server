"""
Optimization utilities for the TSAP MCP Server.

This module provides functions for performance tuning, resource optimization,
algorithm selection, and execution strategy optimization.
"""

import os
import time
import math
import threading
import functools
import itertools
from typing import Dict, List, Any, Optional, Callable, Tuple, TypeVar, Generic, Iterator

import tsap.utils.logging as logging
from tsap.performance_mode import get_parameter
from tsap.utils.errors import TSAPError
from tsap.utils.metrics import measure_time


# Type variables for generic functions
T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')


class OptimizationError(TSAPError):
    """Exception raised for optimization-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.details = details or {}


class ResourceEstimator:
    """Estimates resource requirements for operations."""
    
    def __init__(self):
        self.operation_stats = {}
        self._lock = threading.RLock()
    
    def record_operation(self, operation_type: str, params: Dict, 
                        cpu_time: float, memory_usage: float, 
                        data_size: int) -> None:
        """
        Record statistics for an operation.
        
        Args:
            operation_type: Type of operation
            params: Parameters used for the operation
            cpu_time: CPU time used (seconds)
            memory_usage: Memory used (bytes)
            data_size: Size of data processed (bytes)
        """
        with self._lock:
            if operation_type not in self.operation_stats:
                self.operation_stats[operation_type] = []
            
            self.operation_stats[operation_type].append({
                'cpu_time': cpu_time,
                'memory_usage': memory_usage,
                'data_size': data_size,
                'params': params
            })
            
            # Limit history size
            if len(self.operation_stats[operation_type]) > 100:
                self.operation_stats[operation_type] = self.operation_stats[operation_type][-100:]
    
    def estimate_resources(self, operation_type: str, params: Dict, 
                          data_size: Optional[int] = None) -> Dict[str, float]:
        """
        Estimate resource requirements for an operation.
        
        Args:
            operation_type: Type of operation
            params: Parameters for the operation
            data_size: Size of data to process (bytes)
            
        Returns:
            Estimated resource requirements
        """
        with self._lock:
            if operation_type not in self.operation_stats or not self.operation_stats[operation_type]:
                return {
                    'cpu_time': 0.1,  # Default CPU time (seconds)
                    'memory_usage': 10 * 1024 * 1024,  # Default memory (10 MB)
                    'confidence': 0.0
                }
            
            # Find similar operations to use as a baseline
            stats = self.operation_stats[operation_type]
            
            if data_size is not None:
                # Weight by similarity in data size
                weights = [1.0 / (1.0 + abs(s['data_size'] - data_size) / max(s['data_size'], 1)) 
                          for s in stats]
            else:
                # Equal weights if no data size provided
                weights = [1.0] * len(stats)
            
            total_weight = sum(weights)
            if total_weight == 0:
                return {
                    'cpu_time': 0.1,
                    'memory_usage': 10 * 1024 * 1024,
                    'confidence': 0.0
                }
            
            # Calculate weighted averages
            cpu_time = sum(s['cpu_time'] * w for s, w in zip(stats, weights)) / total_weight
            memory_usage = sum(s['memory_usage'] * w for s, w in zip(stats, weights)) / total_weight
            
            # Calculate confidence (higher with more data and recent operations)
            confidence = min(0.9, (1.0 - 1.0 / (1 + len(stats) / 10.0)))
            
            return {
                'cpu_time': cpu_time,
                'memory_usage': memory_usage,
                'confidence': confidence
            }


# Global instance of ResourceEstimator
_resource_estimator = ResourceEstimator()


def get_resource_estimator() -> ResourceEstimator:
    """
    Get the global ResourceEstimator instance.
    
    Returns:
        The global ResourceEstimator
    """
    return _resource_estimator


def memoize(max_size: int = 128, ttl: Optional[float] = None):
    """
    Decorator for memoization (caching function results).
    
    Args:
        max_size: Maximum number of results to cache
        ttl: Time-to-live for cached results (seconds)
    
    Returns:
        Decorated function
    """
    def decorator(func):
        cache = {}
        lock = threading.RLock()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate a cache key
            key = str(args) + str(sorted(kwargs.items()))
            
            with lock:
                # Check for cached result
                if key in cache:
                    timestamp, result = cache[key]
                    if ttl is None or time.time() - timestamp < ttl:
                        return result
                
                # Call the function
                result = func(*args, **kwargs)
                
                # Cache the result
                cache[key] = (time.time(), result)
                
                # Trim cache if necessary
                if max_size is not None and len(cache) > max_size:
                    # Remove oldest entries
                    oldest_keys = sorted(cache.keys(), key=lambda k: cache[k][0])[:len(cache) - max_size]
                    for old_key in oldest_keys:
                        del cache[old_key]
                
                return result
        
        # Add function to clear the cache
        def clear_cache():
            with lock:
                cache.clear()
        
        wrapper.clear_cache = clear_cache
        return wrapper
    
    return decorator


def optimize_chunk_size(data_size: int, processing_overhead: float = 0.001, 
                       min_chunk_size: int = 1024, max_chunk_size: int = 10 * 1024 * 1024) -> int:
    """
    Calculate an optimal chunk size for processing data.
    
    Args:
        data_size: Total size of data to process
        processing_overhead: Overhead per chunk (seconds)
        min_chunk_size: Minimum allowed chunk size
        max_chunk_size: Maximum allowed chunk size
    
    Returns:
        Optimal chunk size
    """
    # Get the number of available CPU cores
    try:
        cpu_count = os.cpu_count() or 4
    except Exception:
        cpu_count = 4
    
    # Get performance mode factor (1.0 for standard mode)
    performance_factor = get_parameter('chunk_size_factor', 1.0)
    
    # Calculate base chunk size
    if data_size <= min_chunk_size * cpu_count:
        # Data is small, use minimum chunk size
        return min_chunk_size
    
    # Balance between overhead and parallelism
    base_chunk_size = math.sqrt(data_size * processing_overhead) * 1000
    
    # Adjust for performance mode and CPU count
    adjusted_size = base_chunk_size * performance_factor * math.sqrt(cpu_count)
    
    # Ensure within bounds
    return max(min_chunk_size, min(int(adjusted_size), max_chunk_size, data_size))


def optimize_batch_size(item_count: int, processing_time_per_item: float = 0.01,
                       overhead_per_batch: float = 0.1, 
                       min_batch_size: int = 1, max_batch_size: int = 1000) -> int:
    """
    Calculate an optimal batch size for processing items.
    
    Args:
        item_count: Total number of items
        processing_time_per_item: Time to process one item (seconds)
        overhead_per_batch: Overhead per batch (seconds)
        min_batch_size: Minimum allowed batch size
        max_batch_size: Maximum allowed batch size
    
    Returns:
        Optimal batch size
    """
    if item_count <= min_batch_size:
        return item_count
    
    # Calculate optimal batch size to minimize total processing time
    # Optimization formula: sqrt(overhead_per_batch / processing_time_per_item)
    if processing_time_per_item > 0:
        optimal_size = int(math.sqrt(overhead_per_batch / processing_time_per_item))
    else:
        optimal_size = max_batch_size
    
    # Adjust for performance mode
    performance_factor = get_parameter('batch_size_factor', 1.0)
    optimal_size = int(optimal_size * performance_factor)
    
    # Ensure within bounds
    return max(min_batch_size, min(optimal_size, max_batch_size, item_count))


def optimize_concurrency(io_bound: bool = True, cpu_intensive: bool = False, 
                        memory_intensive: bool = False) -> int:
    """
    Calculate optimal concurrency level based on system resources and workload type.
    
    Args:
        io_bound: Whether the workload is I/O bound
        cpu_intensive: Whether the workload is CPU intensive
        memory_intensive: Whether the workload is memory intensive
    
    Returns:
        Optimal concurrency level
    """
    try:
        cpu_count = os.cpu_count() or 4
    except Exception:
        cpu_count = 4
    
    # Start with a base concurrency level
    if io_bound and not cpu_intensive:
        # I/O bound workloads can benefit from higher concurrency
        base_concurrency = cpu_count * 2
    elif cpu_intensive and not io_bound:
        # CPU-bound workloads should match CPU count
        base_concurrency = max(1, cpu_count - 1)  # Leave one CPU for system
    elif memory_intensive:
        # Memory-intensive workloads need lower concurrency
        base_concurrency = max(1, cpu_count // 2)
    else:
        # Mixed workload
        base_concurrency = cpu_count
    
    # Adjust for performance mode
    performance_factor = get_parameter('concurrency_factor', 1.0)
    concurrency = int(base_concurrency * performance_factor)
    
    # Ensure at least 1
    return max(1, concurrency)


def partition_data(data: List[T], partition_count: Optional[int] = None,
                  partition_size: Optional[int] = None) -> List[List[T]]:
    """
    Partition a list into smaller lists.
    
    Args:
        data: The list to partition
        partition_count: Number of partitions (takes precedence if both are provided)
        partition_size: Size of each partition
    
    Returns:
        List of partitioned lists
    """
    if not data:
        return []
    
    data_len = len(data)
    
    # Calculate partition size
    if partition_count is not None and partition_count > 0:
        size = math.ceil(data_len / partition_count)
    elif partition_size is not None and partition_size > 0:
        size = partition_size
    else:
        # Default to system CPU count
        try:
            cpu_count = os.cpu_count() or 4
            size = math.ceil(data_len / cpu_count)
        except Exception:
            size = math.ceil(data_len / 4)
    
    # Ensure size is at least 1
    size = max(1, size)
    
    # Create partitions
    return [data[i:i + size] for i in range(0, data_len, size)]


def lazy_partition(iterable: Iterator[T], partition_size: int) -> Iterator[List[T]]:
    """
    Lazily partition an iterator into chunks.
    
    Args:
        iterable: The iterator to partition
        partition_size: Size of each partition
    
    Returns:
        Iterator of partitioned lists
    """
    iterator = iter(iterable)
    while True:
        chunk = list(itertools.islice(iterator, partition_size))
        if not chunk:
            break
        yield chunk


class LazyMap(Generic[T, U]):
    """
    Memory-efficient map implementation for large datasets.
    
    Instead of loading all data into memory at once, this processes
    items one at a time or in batches.
    """
    
    def __init__(self, func: Callable[[T], U], iterable: Iterator[T], 
                batch_size: Optional[int] = None):
        """
        Initialize LazyMap.
        
        Args:
            func: Function to apply to each item
            iterable: Source iterable
            batch_size: Process items in batches of this size (None for one by one)
        """
        self.func = func
        self.iterable = iter(iterable)
        self.batch_size = batch_size
    
    def __iter__(self) -> Iterator[U]:
        """
        Iterate through mapped values.
        
        Returns:
            Iterator of mapped values
        """
        if self.batch_size is None:
            # Process one by one
            for item in self.iterable:
                yield self.func(item)
        else:
            # Process in batches
            for batch in lazy_partition(self.iterable, self.batch_size):
                yield from map(self.func, batch)


def lazy_map(func: Callable[[T], U], iterable: Iterator[T], 
           batch_size: Optional[int] = None) -> Iterator[U]:
    """
    Apply a function to each item in an iterable lazily.
    
    Args:
        func: Function to apply to each item
        iterable: Source iterable
        batch_size: Process items in batches of this size (None for one by one)
    
    Returns:
        Iterator of mapped values
    """
    return LazyMap(func, iterable, batch_size)


def benchmark_function(func: Callable, *args, repetitions: int = 3, **kwargs) -> Dict[str, Any]:
    """
    Benchmark a function's performance.
    
    Args:
        func: Function to benchmark
        *args: Arguments to pass to the function
        repetitions: Number of times to run the function
        **kwargs: Keyword arguments to pass to the function
    
    Returns:
        Dictionary with benchmark results
    """
    times = []
    results = None
    
    # Run the function multiple times
    for i in range(repetitions):
        start_time = time.time()
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Record time
        elapsed = time.time() - start_time
        times.append(elapsed)
        
        # Save result from first run
        if i == 0:
            results = result
    
    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    # Calculate standard deviation
    if len(times) > 1:
        variance = sum((t - avg_time) ** 2 for t in times) / len(times)
        std_dev = math.sqrt(variance)
    else:
        std_dev = 0
    
    return {
        'function': func.__name__,
        'times': times,
        'min_time': min_time,
        'max_time': max_time,
        'avg_time': avg_time,
        'std_dev': std_dev,
        'repetitions': repetitions,
        'result': results
    }


def compare_functions(funcs: List[Callable], args_list: List[tuple], 
                     kwargs_list: List[Dict] = None, 
                     repetitions: int = 3) -> Dict[str, Any]:
    """
    Compare performance of multiple functions.
    
    Args:
        funcs: List of functions to compare
        args_list: List of argument tuples for each function
        kwargs_list: List of keyword argument dictionaries for each function
        repetitions: Number of times to run each function
    
    Returns:
        Dictionary with comparison results
    """
    if kwargs_list is None:
        kwargs_list = [{} for _ in funcs]
    
    if len(funcs) != len(args_list) or len(funcs) != len(kwargs_list):
        raise ValueError("Number of functions, args_list, and kwargs_list must match")
    
    results = []
    
    # Benchmark each function
    for i, func in enumerate(funcs):
        benchmark = benchmark_function(func, *args_list[i], repetitions=repetitions, **kwargs_list[i])
        results.append(benchmark)
    
    # Find the fastest function
    fastest_index = min(range(len(results)), key=lambda i: results[i]['avg_time'])
    
    return {
        'results': results,
        'fastest': {
            'function': funcs[fastest_index].__name__,
            'index': fastest_index,
            'avg_time': results[fastest_index]['avg_time']
        },
        'comparisons': [
            {
                'slower_function': result['function'],
                'faster_function': results[fastest_index]['function'],
                'times_slower': result['avg_time'] / results[fastest_index]['avg_time']
            }
            for result in results if result['function'] != results[fastest_index]['function']
        ]
    }


def optimize_algorithm(algorithms: Dict[str, Callable], input_data: Any, 
                     evaluate_func: Callable[[Any], float],
                     time_limit: float = 5.0) -> Tuple[str, Any]:
    """
    Select the best algorithm for a specific input based on runtime performance.
    
    Args:
        algorithms: Dictionary mapping names to algorithm functions
        input_data: Input data to test algorithms on
        evaluate_func: Function to evaluate algorithm output (lower is better)
        time_limit: Time limit for testing (seconds)
    
    Returns:
        Tuple of (best algorithm name, algorithm result)
    """
    results = {}
    start_time = time.time()
    
    # Try each algorithm with timeout
    for name, algorithm in algorithms.items():
        # Check if we've exceeded the time limit
        if time.time() - start_time > time_limit:
            logging.debug(f"Time limit reached after testing {len(results)} algorithms", 
                        component="optimization")
            break
        
        # Run the algorithm and measure time
        try:
            with measure_time("optimization", f"algorithm_{name}"):
                result = algorithm(input_data)
            
            # Evaluate the result
            score = evaluate_func(result)
            
            results[name] = {
                'result': result,
                'score': score,
                'time': time.time() - start_time
            }
            
        except Exception as e:
            logging.warning(f"Algorithm {name} failed: {str(e)}", component="optimization")
    
    if not results:
        raise OptimizationError("No algorithms succeeded")
    
    # Find the algorithm with the best score
    best_algorithm = min(results.items(), key=lambda x: x[1]['score'])
    
    return best_algorithm[0], best_algorithm[1]['result']


def create_execution_plan(operations: List[Dict], dependencies: Dict[str, List[str]]) -> List[List[str]]:
    """
    Create an optimized execution plan for operations with dependencies.
    
    Args:
        operations: List of operation dictionaries with 'id' key
        dependencies: Dictionary mapping operation IDs to lists of dependency IDs
    
    Returns:
        List of lists, where each inner list contains operation IDs that can be executed in parallel
    """
    # Create a copy of dependencies to work with
    remaining_deps = {op['id']: set(dependencies.get(op['id'], [])) for op in operations}
    
    # Set of all operation IDs
    all_ops = {op['id'] for op in operations}
    
    # Track processed operations
    processed = set()
    
    # Result: list of lists of operation IDs
    execution_plan = []
    
    while processed != all_ops:
        # Find operations with no remaining dependencies
        ready = [op_id for op_id in all_ops - processed 
                if not remaining_deps[op_id] - processed]
        
        if not ready:
            missing = all_ops - processed
            raise OptimizationError(f"Circular dependency detected in operations: {missing}")
        
        # Add this batch to the execution plan
        execution_plan.append(ready)
        
        # Mark these operations as processed
        processed.update(ready)
    
    return execution_plan


def dynamic_timeout(history: List[float], safety_factor: float = 1.5,
                  min_timeout: float = 1.0, max_timeout: float = 60.0) -> float:
    """
    Calculate a dynamic timeout based on execution history.
    
    Args:
        history: List of previous execution times
        safety_factor: Multiplication factor for safety margin
        min_timeout: Minimum timeout value
        max_timeout: Maximum timeout value
    
    Returns:
        Calculated timeout value
    """
    if not history:
        return min_timeout
    
    # Calculate statistics
    avg_time = sum(history) / len(history)
    
    if len(history) > 1:
        # Calculate standard deviation
        variance = sum((t - avg_time) ** 2 for t in history) / len(history)
        std_dev = math.sqrt(variance)
        
        # Timeout = average + (standard deviation * safety factor)
        timeout = avg_time + (std_dev * safety_factor)
    else:
        # With only one data point, use a larger safety factor
        timeout = avg_time * safety_factor * 2
    
    # Ensure within bounds
    return max(min_timeout, min(timeout, max_timeout))


def adaptive_rate_limit(success_rate: float, current_rate: float, 
                       min_rate: float, max_rate: float) -> float:
    """
    Calculate adaptive rate limit based on success rate.
    
    Args:
        success_rate: Success rate (0.0 to 1.0)
        current_rate: Current rate limit
        min_rate: Minimum allowed rate
        max_rate: Maximum allowed rate
    
    Returns:
        New rate limit
    """
    # Target success rate (sweet spot for throughput)
    target_rate = 0.9
    
    # Calculate adjustment factor
    if success_rate < target_rate:
        # Reduce rate when success rate is low
        adjustment = 0.9  # 10% reduction
    else:
        # Gradually increase rate when success rate is good
        adjustment = 1.05  # 5% increase
    
    # Calculate new rate
    new_rate = current_rate * adjustment
    
    # Ensure within bounds
    return max(min_rate, min(new_rate, max_rate))