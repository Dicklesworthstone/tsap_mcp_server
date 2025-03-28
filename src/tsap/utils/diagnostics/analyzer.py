"""
Diagnostics analyzer for TSAP.

This module provides tools for analyzing system behavior, detecting anomalies,
and checking system health. It includes components for analyzing performance metrics,
identifying bottlenecks, and providing recommendations for system optimization.
"""

import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from tsap.utils.logging import logger
from tsap.utils.errors import TSAPError


class DiagnosticError(TSAPError):
    """
    Exception raised for errors in diagnostics operations.
    
    Attributes:
        message: Error message
        details: Additional error details
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, code="DIAGNOSTIC_ERROR", details=details)


class HealthStatus(str, Enum):
    """Enum for system health status levels."""
    CRITICAL = "critical"
    WARNING = "warning"
    HEALTHY = "healthy"
    UNKNOWN = "unknown"


@dataclass
class AnalysisResult:
    """
    Result of a diagnostic analysis.
    
    Attributes:
        status: Health status of the analyzed component
        component: Name of the analyzed component
        details: Detailed analysis information
        timestamp: Time when the analysis was performed
        recommendations: List of recommendations based on the analysis
        metrics: Metrics collected during the analysis
    """
    status: HealthStatus
    component: str
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    recommendations: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "status": self.status,
            "component": self.component,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "recommendations": self.recommendations,
            "metrics": self.metrics
        }


@dataclass
class SystemHealthCheck:
    """
    System health check configuration.
    
    Attributes:
        component: Name of the component to check
        check_function: Function to perform the health check
        threshold_warning: Warning threshold
        threshold_critical: Critical threshold
        interval: Check interval in seconds
        enabled: Whether the check is enabled
        description: Description of the health check
    """
    component: str
    check_function: Callable[[], Tuple[HealthStatus, Dict[str, Any]]]
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    interval: float = 60.0
    enabled: bool = True
    description: Optional[str] = None
    last_execution: Optional[datetime] = None
    last_status: HealthStatus = HealthStatus.UNKNOWN
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "component": self.component,
            "threshold_warning": self.threshold_warning,
            "threshold_critical": self.threshold_critical,
            "interval": self.interval,
            "enabled": self.enabled,
            "description": self.description,
            "last_execution": self.last_execution.isoformat() if self.last_execution else None,
            "last_status": self.last_status
        }


class DiagnosticAnalyzer:
    """
    Analyzer for system diagnostics.
    
    Provides tools for analyzing system behavior, detecting anomalies,
    and checking system health.
    
    Attributes:
        health_checks: Dictionary of health checks by component name
        anomaly_detectors: Dictionary of anomaly detectors by component name
        performance_analyzers: Dictionary of performance analyzers by component name
    """
    def __init__(self) -> None:
        """Initialize the diagnostic analyzer."""
        self.health_checks: Dict[str, SystemHealthCheck] = {}
        self.anomaly_detectors: Dict[str, Any] = {}
        self.performance_analyzers: Dict[str, Any] = {}
        self._running = False
        self._health_check_thread = None
        self._lock = threading.Lock()
    
    def register_health_check(self, health_check: SystemHealthCheck) -> None:
        """
        Register a health check.
        
        Args:
            health_check: Health check to register
        """
        with self._lock:
            self.health_checks[health_check.component] = health_check
    
    def unregister_health_check(self, component: str) -> bool:
        """
        Unregister a health check.
        
        Args:
            component: Name of the component to unregister
            
        Returns:
            True if the health check was unregistered, False if not found
        """
        with self._lock:
            if component in self.health_checks:
                del self.health_checks[component]
                return True
            return False
    
    def register_anomaly_detector(self, component: str, detector: Any) -> None:
        """
        Register an anomaly detector.
        
        Args:
            component: Name of the component to monitor
            detector: Anomaly detector to register
        """
        with self._lock:
            self.anomaly_detectors[component] = detector
    
    def register_performance_analyzer(self, component: str, analyzer: Any) -> None:
        """
        Register a performance analyzer.
        
        Args:
            component: Name of the component to analyze
            analyzer: Performance analyzer to register
        """
        with self._lock:
            self.performance_analyzers[component] = analyzer
    
    def start_monitoring(self) -> None:
        """Start monitoring system health."""
        if self._running:
            return
        
        self._running = True
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self._health_check_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop monitoring system health."""
        self._running = False
        if self._health_check_thread:
            self._health_check_thread.join(timeout=1.0)
            self._health_check_thread = None
    
    def _health_check_loop(self) -> None:
        """Background thread for periodic health checks."""
        while self._running:
            try:
                self.run_health_checks()
            except Exception as e:
                logger.error(f"Error in health check loop: {str(e)}")
            
            # Sleep for a short interval
            time.sleep(1.0)
    
    def run_health_checks(self) -> Dict[str, AnalysisResult]:
        """
        Run all registered health checks.
        
        Returns:
            Dictionary of analysis results by component name
        """
        results = {}
        current_time = datetime.now()
        
        with self._lock:
            # Make a copy to avoid modification during iteration
            checks = list(self.health_checks.values())
        
        for check in checks:
            # Skip if not enabled
            if not check.enabled:
                continue
            
            # Skip if not due yet
            if check.last_execution and (current_time - check.last_execution).total_seconds() < check.interval:
                continue
            
            try:
                # Run the check
                status, details = check.check_function()
                
                # Update check status
                check.last_execution = current_time
                check.last_status = status
                
                # Create analysis result
                result = AnalysisResult(
                    status=status,
                    component=check.component,
                    details=details,
                    timestamp=current_time,
                    recommendations=[]  # Could add automatic recommendations based on status
                )
                
                results[check.component] = result
                
                # Log if status is warning or critical
                if status != HealthStatus.HEALTHY:
                    logger.warning(f"Health check for {check.component} is {status}: {details}")
                
            except Exception as e:
                logger.error(f"Error running health check for {check.component}: {str(e)}")
                
                # Create error result
                result = AnalysisResult(
                    status=HealthStatus.UNKNOWN,
                    component=check.component,
                    details={"error": str(e)},
                    timestamp=current_time,
                    recommendations=["Fix the health check function"]
                )
                
                results[check.component] = result
        
        return results
    
    def check_component_health(self, component: str) -> Optional[AnalysisResult]:
        """
        Check the health of a specific component.
        
        Args:
            component: Name of the component to check
            
        Returns:
            Analysis result for the component, or None if not found
        """
        with self._lock:
            if component not in self.health_checks:
                return None
            
            check = self.health_checks[component]
        
        try:
            # Run the check
            status, details = check.check_function()
            
            # Update check status
            check.last_execution = datetime.now()
            check.last_status = status
            
            # Create analysis result
            result = AnalysisResult(
                status=status,
                component=component,
                details=details,
                timestamp=check.last_execution,
                recommendations=[]  # Could add automatic recommendations based on status
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking health of component {component}: {str(e)}")
            
            # Create error result
            result = AnalysisResult(
                status=HealthStatus.UNKNOWN,
                component=component,
                details={"error": str(e)},
                timestamp=datetime.now(),
                recommendations=["Fix the health check function"]
            )
            
            return result
    
    def analyze_system_health(self) -> Dict[str, AnalysisResult]:
        """
        Analyze the health of the entire system.
        
        Returns:
            Dictionary of analysis results by component name
        """
        # Run health checks
        results = self.run_health_checks()
        
        # Add system-level checks
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_status = HealthStatus.HEALTHY
            if cpu_percent > 90:
                cpu_status = HealthStatus.CRITICAL
            elif cpu_percent > 75:
                cpu_status = HealthStatus.WARNING
            
            results["system_cpu"] = AnalysisResult(
                status=cpu_status,
                component="system_cpu",
                details={"cpu_percent": cpu_percent},
                recommendations=["Reduce CPU-intensive operations"] if cpu_percent > 75 else []
            )
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_status = HealthStatus.HEALTHY
            if memory_percent > 90:
                memory_status = HealthStatus.CRITICAL
            elif memory_percent > 75:
                memory_status = HealthStatus.WARNING
            
            results["system_memory"] = AnalysisResult(
                status=memory_status,
                component="system_memory",
                details={
                    "memory_percent": memory_percent,
                    "available_mb": memory.available / (1024 * 1024),
                    "total_mb": memory.total / (1024 * 1024)
                },
                recommendations=["Free up memory"] if memory_percent > 75 else []
            )
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_status = HealthStatus.HEALTHY
            if disk_percent > 90:
                disk_status = HealthStatus.CRITICAL
            elif disk_percent > 75:
                disk_status = HealthStatus.WARNING
            
            results["system_disk"] = AnalysisResult(
                status=disk_status,
                component="system_disk",
                details={
                    "disk_percent": disk_percent,
                    "free_gb": disk.free / (1024 * 1024 * 1024),
                    "total_gb": disk.total / (1024 * 1024 * 1024)
                },
                recommendations=["Free up disk space"] if disk_percent > 75 else []
            )
            
        except Exception as e:
            logger.error(f"Error analyzing system health: {str(e)}")
        
        return results
    
    def analyze_performance(self, component: str) -> Dict[str, Any]:
        """
        Analyze the performance of a component.
        
        Args:
            component: Name of the component to analyze
            
        Returns:
            Performance analysis results
            
        Raises:
            DiagnosticError: If the component has no registered performance analyzer
        """
        with self._lock:
            if component not in self.performance_analyzers:
                raise DiagnosticError(f"No performance analyzer registered for component: {component}")
            
            analyzer = self.performance_analyzers[component]  # noqa: F841
        
        # This is a placeholder - actual implementation would depend on the analyzer
        return {"component": component, "status": "Not implemented"}
    
    def detect_anomalies(self, component: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect anomalies in component data.
        
        Args:
            component: Name of the component to analyze
            data: Data to analyze
            
        Returns:
            Anomaly detection results
            
        Raises:
            DiagnosticError: If the component has no registered anomaly detector
        """
        with self._lock:
            if component not in self.anomaly_detectors:
                raise DiagnosticError(f"No anomaly detector registered for component: {component}")
            
            detector = self.anomaly_detectors[component]  # noqa: F841
        
        # This is a placeholder - actual implementation would depend on the detector
        return {"component": component, "anomalies_detected": False}


# Singleton instance
_analyzer_instance = None


def get_analyzer() -> DiagnosticAnalyzer:
    """
    Get the global diagnostic analyzer instance.
    
    Returns:
        Diagnostic analyzer instance
    """
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = DiagnosticAnalyzer()
    return _analyzer_instance


def register_health_check(component: str, check_function: Callable, interval: float = 60.0) -> None:
    """
    Register a health check function.
    
    Args:
        component: Name of the component to check
        check_function: Function to perform the health check
        interval: Check interval in seconds
    """
    health_check = SystemHealthCheck(
        component=component,
        check_function=check_function,
        interval=interval
    )
    get_analyzer().register_health_check(health_check)


def analyze_system_health() -> Dict[str, AnalysisResult]:
    """
    Analyze the health of the entire system.
    
    Returns:
        Dictionary of analysis results by component name
    """
    return get_analyzer().analyze_system_health()


def analyze_performance(component: str) -> Dict[str, Any]:
    """
    Analyze the performance of a component.
    
    Args:
        component: Name of the component to analyze
        
    Returns:
        Performance analysis results
    """
    return get_analyzer().analyze_performance(component)


def detect_anomalies(component: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect anomalies in component data.
    
    Args:
        component: Name of the component to analyze
        data: Data to analyze
        
    Returns:
        Anomaly detection results
    """
    return get_analyzer().detect_anomalies(component, data)


# Register default health checks
def _register_default_health_checks() -> None:
    """Register default system health checks."""
    analyzer = get_analyzer()
    
    # CPU health check
    def check_cpu_health() -> Tuple[HealthStatus, Dict[str, Any]]:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        status = HealthStatus.HEALTHY
        if cpu_percent > 90:
            status = HealthStatus.CRITICAL
        elif cpu_percent > 75:
            status = HealthStatus.WARNING
        return status, {"cpu_percent": cpu_percent}
    
    analyzer.register_health_check(SystemHealthCheck(
        component="system_cpu",
        check_function=check_cpu_health,
        threshold_warning=75.0,
        threshold_critical=90.0,
        interval=5.0,
        description="System CPU usage"
    ))
    
    # Memory health check
    def check_memory_health() -> Tuple[HealthStatus, Dict[str, Any]]:
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        status = HealthStatus.HEALTHY
        if memory_percent > 90:
            status = HealthStatus.CRITICAL
        elif memory_percent > 75:
            status = HealthStatus.WARNING
        return status, {
            "memory_percent": memory_percent,
            "available_mb": memory.available / (1024 * 1024),
            "total_mb": memory.total / (1024 * 1024)
        }
    
    analyzer.register_health_check(SystemHealthCheck(
        component="system_memory",
        check_function=check_memory_health,
        threshold_warning=75.0,
        threshold_critical=90.0,
        interval=10.0,
        description="System memory usage"
    ))
    
    # Disk health check
    def check_disk_health() -> Tuple[HealthStatus, Dict[str, Any]]:
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        status = HealthStatus.HEALTHY
        if disk_percent > 90:
            status = HealthStatus.CRITICAL
        elif disk_percent > 75:
            status = HealthStatus.WARNING
        return status, {
            "disk_percent": disk_percent,
            "free_gb": disk.free / (1024 * 1024 * 1024),
            "total_gb": disk.total / (1024 * 1024 * 1024)
        }
    
    analyzer.register_health_check(SystemHealthCheck(
        component="system_disk",
        check_function=check_disk_health,
        threshold_warning=75.0,
        threshold_critical=90.0,
        interval=60.0,
        description="System disk usage"
    ))


# Register default health checks
_register_default_health_checks()