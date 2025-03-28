"""
General metrics collection and reporting utilities.

This module provides a centralized system for tracking, collecting, and reporting
various metrics throughout the TSAP system. It's designed to be lightweight, 
thread-safe, and work alongside but separate from the evolution-specific metrics.
"""

import time
import threading
import statistics
from typing import Dict, List, Any, Optional
from contextlib import contextmanager
from dataclasses import dataclass, field

from tsap.utils.logging import debug


class MetricsRegistry:
    """Global registry for metrics collectors."""
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MetricsRegistry, cls).__new__(cls)
                cls._instance._collectors = {}
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        with self._lock:
            if not self._initialized:
                self._collectors = {}
                self._initialized = True
    
    def register_collector(self, name: str, collector: 'MetricsCollector') -> None:
        """Register a metrics collector with the registry."""
        with self._lock:
            if name in self._collectors:
                debug(f"Replacing existing metrics collector: {name}", component="metrics")
            self._collectors[name] = collector
    
    def get_collector(self, name: str) -> Optional['MetricsCollector']:
        """Get a metrics collector by name."""
        with self._lock:
            return self._collectors.get(name)
    
    def list_collectors(self) -> List[str]:
        """List all registered metrics collectors."""
        with self._lock:
            return list(self._collectors.keys())
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics from all collectors."""
        result = {}
        with self._lock:
            for name, collector in self._collectors.items():
                result[name] = collector.get_metrics()
        return result


@dataclass
class MetricValue:
    """Holds values and metadata for a single metric."""
    name: str
    description: str
    unit: Optional[str] = None
    value: Any = None
    values: List[Any] = field(default_factory=list)
    count: int = 0
    sum: float = 0
    min: Optional[float] = None
    max: Optional[float] = None
    last_update: float = field(default_factory=time.time)
    aggregation: str = "last"  # Options: last, sum, min, max, avg, count, stats
    
    def add_value(self, value: Any) -> None:
        """Add a value to this metric."""
        self.value = value
        self.count += 1
        self.last_update = time.time()
        
        # Handle numeric values
        if isinstance(value, (int, float)):
            self.values.append(value)
            self.sum += value
            
            if self.min is None or value < self.min:
                self.min = value
            
            if self.max is None or value > self.max:
                self.max = value
    
    def get_aggregated_value(self) -> Any:
        """Get the value according to the specified aggregation method."""
        if self.aggregation == "last":
            return self.value
        elif self.aggregation == "sum":
            return self.sum
        elif self.aggregation == "min":
            return self.min
        elif self.aggregation == "max":
            return self.max
        elif self.aggregation == "avg":
            return self.sum / self.count if self.count > 0 else None
        elif self.aggregation == "count":
            return self.count
        elif self.aggregation == "stats":
            return self.get_statistics()
        return self.value
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistical information about this metric's values."""
        stats = {
            "count": self.count,
            "min": self.min,
            "max": self.max,
            "last": self.value,
            "sum": self.sum,
        }
        
        if self.count > 0:
            stats["avg"] = self.sum / self.count
            
        if len(self.values) >= 2:
            try:
                stats["median"] = statistics.median(self.values)
                stats["stdev"] = statistics.stdev(self.values)
            except (TypeError, statistics.StatisticsError):
                pass  # Handle non-numeric or insufficient values
                
        return stats
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert this metric to a dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "unit": self.unit,
            "value": self.get_aggregated_value(),
            "count": self.count,
            "last_update": self.last_update,
        }


class MetricsCollector:
    """Collects and manages metrics for a specific component."""
    
    def __init__(self, name: str, description: Optional[str] = None):
        self.name = name
        self.description = description or f"Metrics collector for {name}"
        self._metrics = {}
        self._lock = threading.RLock()
        
        # Auto-register with global registry
        registry = MetricsRegistry()
        registry.register_collector(name, self)
        
    def define_metric(
        self, 
        name: str, 
        description: str, 
        unit: Optional[str] = None,
        aggregation: str = "last"
    ) -> None:
        """Define a new metric."""
        with self._lock:
            if name in self._metrics:
                debug(f"Replacing existing metric definition: {name}", component="metrics")
            
            self._metrics[name] = MetricValue(
                name=name,
                description=description,
                unit=unit,
                aggregation=aggregation
            )
    
    def record(self, name: str, value: Any) -> None:
        """Record a value for a metric."""
        with self._lock:
            if name not in self._metrics:
                # Auto-create metric if it doesn't exist
                self.define_metric(name, f"Auto-created metric: {name}")
            
            self._metrics[name].add_value(value)
    
    def increment(self, name: str, amount: float = 1.0) -> None:
        """Increment a counter metric."""
        with self._lock:
            if name not in self._metrics:
                # Auto-create counter metric
                self.define_metric(
                    name, 
                    f"Auto-created counter: {name}", 
                    aggregation="sum"
                )
                self._metrics[name].value = 0
            
            current = self._metrics[name].value
            if current is None:
                current = 0
            
            # Ensure it's numeric before incrementing
            if not isinstance(current, (int, float)):
                current = 0
                
            self._metrics[name].add_value(current + amount)
    
    def get_metric(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific metric by name."""
        with self._lock:
            metric = self._metrics.get(name)
            return metric.to_dict() if metric else None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics from this collector."""
        result = {
            "name": self.name,
            "description": self.description,
            "metrics": {},
        }
        
        with self._lock:
            for name, metric in self._metrics.items():
                result["metrics"][name] = metric.to_dict()
        
        return result
    
    def reset(self, metric_name: Optional[str] = None) -> None:
        """Reset metrics in this collector."""
        with self._lock:
            if metric_name:
                if metric_name in self._metrics:
                    # Reset a specific metric
                    old_metric = self._metrics[metric_name]
                    self._metrics[metric_name] = MetricValue(
                        name=old_metric.name,
                        description=old_metric.description,
                        unit=old_metric.unit,
                        aggregation=old_metric.aggregation
                    )
            else:
                # Reset all metrics while preserving definitions
                metrics_copy = {}
                for name, metric in self._metrics.items():
                    metrics_copy[name] = MetricValue(
                        name=metric.name,
                        description=metric.description,
                        unit=metric.unit,
                        aggregation=metric.aggregation
                    )
                self._metrics = metrics_copy
    
    @contextmanager
    def measure_time(self, metric_name: str, description: Optional[str] = None):
        """Context manager to measure execution time of a block."""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            if description and metric_name not in self._metrics:
                self.define_metric(
                    metric_name, 
                    description,
                    unit="seconds",
                    aggregation="stats"
                )
            self.record(metric_name, elapsed)


# Global helper functions
def get_metrics_registry() -> MetricsRegistry:
    """Get the global metrics registry."""
    return MetricsRegistry()

def get_collector(name: str) -> Optional[MetricsCollector]:
    """Get a metrics collector by name."""
    return get_metrics_registry().get_collector(name)

def create_collector(name: str, description: Optional[str] = None) -> MetricsCollector:
    """Create and register a new metrics collector."""
    collector = MetricsCollector(name, description)
    return collector

def record_metric(collector_name: str, metric_name: str, value: Any) -> None:
    """Record a metric value in the specified collector."""
    collector = get_collector(collector_name)
    if not collector:
        collector = create_collector(collector_name)
    collector.record(metric_name, value)

def increment_metric(collector_name: str, metric_name: str, amount: float = 1.0) -> None:
    """Increment a counter metric in the specified collector."""
    collector = get_collector(collector_name)
    if not collector:
        collector = create_collector(collector_name)
    collector.increment(metric_name, amount)

@contextmanager
def measure_time(collector_name: str, metric_name: str, description: Optional[str] = None):
    """Context manager to measure execution time and record it to a collector."""
    collector = get_collector(collector_name)
    if not collector:
        collector = create_collector(collector_name)
    
    with collector.measure_time(metric_name, description):
        yield

def get_all_metrics() -> Dict[str, Dict[str, Any]]:
    """Get metrics from all collectors."""
    return get_metrics_registry().get_all_metrics()

def reset_metrics(collector_name: Optional[str] = None, metric_name: Optional[str] = None) -> None:
    """Reset metrics in specified collector(s)."""
    if collector_name:
        # Reset a specific collector
        collector = get_collector(collector_name)
        if collector:
            collector.reset(metric_name)
    else:
        # Reset all collectors
        registry = get_metrics_registry()
        for name in registry.list_collectors():
            collector = registry.get_collector(name)
            if collector:
                collector.reset(metric_name)