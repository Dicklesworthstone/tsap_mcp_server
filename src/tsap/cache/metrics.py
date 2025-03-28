"""
Metrics collection for the caching system.

This module provides functionality for collecting, aggregating, and reporting
metrics about the cache system's performance, such as hit rates, latencies,
memory usage, and eviction patterns.
"""

import time
import threading
from typing import Dict, List, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, field

from tsap.utils.metrics import MetricsCollector, get_collector


@dataclass
class CacheMetricsSnapshot:
    """
    Snapshot of cache metrics at a point in time.
    
    Attributes:
        timestamp: When the snapshot was taken
        cache_name: Name of the cache
        size: Current number of items in the cache
        memory_usage: Estimated memory usage in bytes
        capacity: Maximum capacity of the cache
        hits: Cache hit count
        misses: Cache miss count
        hit_rate: Cache hit rate (hits / (hits + misses))
        sets: Number of set operations
        gets: Number of get operations
        deletes: Number of delete operations
        evictions: Number of cache evictions
        expirations: Number of expired entries
        latency_avg_get: Average latency of get operations in ms
        latency_avg_set: Average latency of set operations in ms
        invalidations: Number of manual invalidations
        custom_metrics: Additional metrics specific to the cache implementation
    """
    timestamp: float = field(default_factory=time.time)
    cache_name: str = ""
    size: int = 0
    memory_usage: int = 0
    capacity: int = 0
    hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0
    sets: int = 0
    gets: int = 0
    deletes: int = 0
    evictions: int = 0
    expirations: int = 0
    latency_avg_get: float = 0.0
    latency_avg_set: float = 0.0
    invalidations: int = 0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the snapshot to a dictionary.
        
        Returns:
            Dictionary representation of the snapshot
        """
        return {
            "timestamp": self.timestamp,
            "cache_name": self.cache_name,
            "size": self.size,
            "memory_usage": self.memory_usage,
            "capacity": self.capacity,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "sets": self.sets,
            "gets": self.gets,
            "deletes": self.deletes,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "latency_avg_get": self.latency_avg_get,
            "latency_avg_set": self.latency_avg_set,
            "invalidations": self.invalidations,
            **{f"custom_{k}": v for k, v in self.custom_metrics.items()}
        }


class CacheMetricsCollector:
    """
    Collects and aggregates metrics from one or more caches.
    
    Provides functionality to monitor cache performance over time, 
    record periodic snapshots, and compute aggregate statistics.
    """
    def __init__(self, max_history: int = 100) -> None:
        """
        Initialize a new cache metrics collector.
        
        Args:
            max_history: Maximum number of snapshots to retain per cache
        """
        self._snapshots: Dict[str, List[CacheMetricsSnapshot]] = defaultdict(list)
        self._aggregates: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._max_history = max_history
        self._lock = threading.Lock()
        
        # Create a metrics collector for the cache system
        self._metrics = get_collector("cache_system")
        if not self._metrics:
            self._metrics = MetricsCollector("cache_system", "Cache system metrics")
            self._metrics.define_metric("total_hits", "Total cache hits across all caches")
            self._metrics.define_metric("total_misses", "Total cache misses across all caches")
            self._metrics.define_metric("total_hit_rate", "Overall cache hit rate", unit="%")
            self._metrics.define_metric("total_memory_usage", "Total memory usage of all caches", unit="bytes")
            self._metrics.define_metric("total_evictions", "Total cache evictions across all caches")
            self._metrics.define_metric("total_expirations", "Total expired entries across all caches")
    
    def add_snapshot(self, snapshot: CacheMetricsSnapshot) -> None:
        """
        Add a metrics snapshot for a cache.
        
        Args:
            snapshot: The metrics snapshot to add
        """
        cache_name = snapshot.cache_name
        
        with self._lock:
            # Add snapshot to history, maintaining max_history limit
            snapshots = self._snapshots[cache_name]
            snapshots.append(snapshot)
            if len(snapshots) > self._max_history:
                snapshots.pop(0)
            
            # Update aggregate metrics
            self._update_aggregates(cache_name)
            
            # Update global metrics
            self._update_global_metrics()
    
    def _update_aggregates(self, cache_name: str) -> None:
        """
        Update aggregate metrics for a specific cache.
        
        Args:
            cache_name: The name of the cache to update aggregates for
        """
        snapshots = self._snapshots[cache_name]
        if not snapshots:
            return
        
        # Calculate aggregates over the entire history
        agg = {}
        
        # Simple metrics (latest value)
        latest = snapshots[-1]
        agg["size"] = latest.size
        agg["memory_usage"] = latest.memory_usage
        agg["capacity"] = latest.capacity
        
        # Cumulative metrics (sum over all snapshots)
        agg["hits"] = sum(s.hits for s in snapshots)
        agg["misses"] = sum(s.misses for s in snapshots)
        agg["sets"] = sum(s.sets for s in snapshots)
        agg["gets"] = sum(s.gets for s in snapshots)
        agg["deletes"] = sum(s.deletes for s in snapshots)
        agg["evictions"] = sum(s.evictions for s in snapshots)
        agg["expirations"] = sum(s.expirations for s in snapshots)
        agg["invalidations"] = sum(s.invalidations for s in snapshots)
        
        # Calculated metrics
        if agg["hits"] + agg["misses"] > 0:
            agg["hit_rate"] = agg["hits"] / (agg["hits"] + agg["misses"])
        else:
            agg["hit_rate"] = 0.0
        
        # Average latencies
        latencies_get = [s.latency_avg_get for s in snapshots if s.latency_avg_get > 0]
        latencies_set = [s.latency_avg_set for s in snapshots if s.latency_avg_set > 0]
        
        agg["latency_avg_get"] = sum(latencies_get) / len(latencies_get) if latencies_get else 0.0
        agg["latency_avg_set"] = sum(latencies_set) / len(latencies_set) if latencies_set else 0.0
        
        # Custom metrics (simple average of latest 5 snapshots)
        recent_snapshots = snapshots[-5:] if len(snapshots) >= 5 else snapshots
        custom_metrics = set()
        for s in recent_snapshots:
            custom_metrics.update(s.custom_metrics.keys())
        
        for metric in custom_metrics:
            values = [s.custom_metrics.get(metric, 0) for s in recent_snapshots if metric in s.custom_metrics]
            if values:
                agg[f"custom_{metric}"] = sum(values) / len(values)
        
        self._aggregates[cache_name] = agg
    
    def _update_global_metrics(self) -> None:
        """Update global metrics across all caches."""
        # Calculate system-wide metrics
        total_hits = sum(agg.get("hits", 0) for agg in self._aggregates.values())
        total_misses = sum(agg.get("misses", 0) for agg in self._aggregates.values())
        total_memory = sum(agg.get("memory_usage", 0) for agg in self._aggregates.values())
        total_evictions = sum(agg.get("evictions", 0) for agg in self._aggregates.values())
        total_expirations = sum(agg.get("expirations", 0) for agg in self._aggregates.values())
        
        # Calculate overall hit rate
        if total_hits + total_misses > 0:
            total_hit_rate = (total_hits / (total_hits + total_misses)) * 100
        else:
            total_hit_rate = 0.0
        
        # Record metrics in the system collector
        self._metrics.record("total_hits", total_hits)
        self._metrics.record("total_misses", total_misses)
        self._metrics.record("total_hit_rate", total_hit_rate)
        self._metrics.record("total_memory_usage", total_memory)
        self._metrics.record("total_evictions", total_evictions)
        self._metrics.record("total_expirations", total_expirations)
    
    def get_snapshot(self, cache_name: str, index: int = -1) -> Optional[CacheMetricsSnapshot]:
        """
        Get a specific snapshot for a cache.
        
        Args:
            cache_name: Name of the cache
            index: Index of the snapshot (-1 for latest)
            
        Returns:
            The requested snapshot, or None if not available
        """
        with self._lock:
            snapshots = self._snapshots.get(cache_name, [])
            if not snapshots or abs(index) > len(snapshots):
                return None
            return snapshots[index]
    
    def get_snapshots(self, cache_name: str, limit: Optional[int] = None) -> List[CacheMetricsSnapshot]:
        """
        Get all snapshots for a cache.
        
        Args:
            cache_name: Name of the cache
            limit: Maximum number of snapshots to return (most recent first)
            
        Returns:
            List of snapshots for the cache
        """
        with self._lock:
            snapshots = self._snapshots.get(cache_name, [])
            if limit is not None and limit > 0:
                return list(reversed(snapshots[-limit:]))
            return list(reversed(snapshots))
    
    def get_aggregates(self, cache_name: str) -> Dict[str, Any]:
        """
        Get aggregate metrics for a cache.
        
        Args:
            cache_name: Name of the cache
            
        Returns:
            Dictionary of aggregate metrics
        """
        with self._lock:
            return self._aggregates.get(cache_name, {}).copy()
    
    def get_all_aggregates(self) -> Dict[str, Dict[str, Any]]:
        """
        Get aggregate metrics for all caches.
        
        Returns:
            Dictionary mapping cache names to aggregate metrics
        """
        with self._lock:
            return {name: aggs.copy() for name, aggs in self._aggregates.items()}
    
    def get_cache_names(self) -> List[str]:
        """
        Get list of cache names being tracked.
        
        Returns:
            List of cache names
        """
        with self._lock:
            return list(self._snapshots.keys())
    
    def get_global_metrics(self) -> Dict[str, Any]:
        """
        Get global metrics across all caches.
        
        Returns:
            Dictionary of global metrics
        """
        return self._metrics.get_metrics()
    
    def clear_history(self, cache_name: Optional[str] = None) -> None:
        """
        Clear metrics history for a specific cache or all caches.
        
        Args:
            cache_name: Name of the cache to clear history for, or None for all caches
        """
        with self._lock:
            if cache_name:
                if cache_name in self._snapshots:
                    self._snapshots[cache_name] = []
                    self._aggregates[cache_name] = {}
            else:
                self._snapshots.clear()
                self._aggregates.clear()
                self._metrics.reset()


# Singleton instance of the metrics collector
_cache_metrics_collector: Optional[CacheMetricsCollector] = None


def get_cache_metrics_collector() -> CacheMetricsCollector:
    """
    Get the global cache metrics collector instance.
    
    Returns:
        The global cache metrics collector
    """
    global _cache_metrics_collector
    if _cache_metrics_collector is None:
        _cache_metrics_collector = CacheMetricsCollector()
    return _cache_metrics_collector


def record_cache_operation(
    cache_name: str, 
    operation: str, 
    success: bool = True, 
    latency: Optional[float] = None,
    key: Optional[str] = None,
    size: Optional[int] = None
) -> None:
    """
    Record a cache operation for metrics.
    
    Args:
        cache_name: Name of the cache
        operation: Operation type ('get', 'set', 'delete', 'evict', 'expire', 'invalidate')
        success: Whether the operation was successful
        latency: Latency of the operation in milliseconds
        key: The cache key involved (for debugging/analysis)
        size: Current size of the cache after the operation
    """
    # This is a helper function that would be called from the cache implementation
    # It could update metrics in real-time or queue them for batch processing
    # In a real implementation, we might want to sample or batch metrics to reduce overhead
    pass


def create_cache_snapshot(
    cache_name: str,
    size: int,
    capacity: int,
    hits: int,
    misses: int,
    **custom_metrics
) -> None:
    """
    Create and record a snapshot of cache metrics.
    
    Args:
        cache_name: Name of the cache
        size: Current number of items in the cache
        capacity: Maximum capacity of the cache
        hits: Cumulative cache hit count
        misses: Cumulative cache miss count
        **custom_metrics: Additional metrics specific to the cache implementation
    """
    # Create a snapshot and add it to the collector
    snapshot = CacheMetricsSnapshot(
        timestamp=time.time(),
        cache_name=cache_name,
        size=size,
        capacity=capacity,
        hits=hits,
        misses=misses,
        hit_rate=hits / (hits + misses) if hits + misses > 0 else 0.0,
        custom_metrics=custom_metrics
    )
    
    # Add to collector
    collector = get_cache_metrics_collector()
    collector.add_snapshot(snapshot)


def get_cache_metrics(cache_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get metrics for a specific cache or global metrics.
    
    Args:
        cache_name: Name of the cache, or None for global metrics
        
    Returns:
        Dictionary of metrics
    """
    collector = get_cache_metrics_collector()
    
    if cache_name:
        return collector.get_aggregates(cache_name)
    else:
        return collector.get_global_metrics()


def get_cache_hit_rate(cache_name: Optional[str] = None) -> float:
    """
    Get the hit rate for a specific cache or the global hit rate.
    
    Args:
        cache_name: Name of the cache, or None for global hit rate
        
    Returns:
        Hit rate as a percentage (0-100)
    """
    metrics = get_cache_metrics(cache_name)
    
    if cache_name:
        hit_rate = metrics.get("hit_rate", 0.0)
        return hit_rate * 100.0
    else:
        return metrics.get("total_hit_rate", 0.0)