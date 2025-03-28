"""
Diagnostics package for TSAP.

This package provides tools for analyzing system behavior, performance profiling,
generating diagnostic reports, and visualizing diagnostic data. It includes components
for real-time monitoring, anomaly detection, and system health checking.
"""

from tsap.utils.diagnostics.analyzer import (
    DiagnosticAnalyzer, SystemHealthCheck, AnalysisResult,
    analyze_system_health, analyze_performance, detect_anomalies
)
from tsap.utils.diagnostics.profiler import (
    Profiler, FunctionProfiler, ProfileResult, MemoryProfiler,
    profile_function, profile_memory_usage, profile_code_section
)
from tsap.utils.diagnostics.reporter import (
    DiagnosticReporter, SystemHealthReport, PerformanceReport,
    generate_health_report, generate_performance_report, generate_anomaly_report
)
from tsap.utils.diagnostics.visualizer import (
    DiagnosticVisualizer, PerformanceChart, TimelineChart,
    generate_performance_chart, generate_system_health_chart, generate_timeline_chart
)


__all__ = [
    # Analyzer
    'DiagnosticAnalyzer', 'SystemHealthCheck', 'AnalysisResult',
    'analyze_system_health', 'analyze_performance', 'detect_anomalies',
    
    # Profiler
    'Profiler', 'FunctionProfiler', 'ProfileResult', 'MemoryProfiler',
    'profile_function', 'profile_memory_usage', 'profile_code_section',
    
    # Reporter
    'DiagnosticReporter', 'SystemHealthReport', 'PerformanceReport',
    'generate_health_report', 'generate_performance_report', 'generate_anomaly_report',
    
    # Visualizer
    'DiagnosticVisualizer', 'PerformanceChart', 'TimelineChart',
    'generate_performance_chart', 'generate_system_health_chart', 'generate_timeline_chart'
]


def initialize_diagnostics() -> None:
    """
    Initialize the diagnostics system.
    
    This function sets up the necessary components for diagnostics,
    including analyzers, profilers, reporters, and visualizers.
    """
    from tsap.utils.logging import logger
    logger.debug("Initializing diagnostics system...")
    
    # Initialize components as needed
    # This is a placeholder for future implementation