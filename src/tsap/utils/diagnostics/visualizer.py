"""
Diagnostic data visualization tools for TSAP.

This module provides tools for visualizing diagnostic data, including performance metrics,
system health information, and historical trends. It can generate various chart types
and visual representations to help understand system behavior and performance.
"""

import os
import base64
import json
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from tsap.utils.logging import logger
from tsap.utils.errors import TSAPError
from tsap.utils.diagnostics.analyzer import AnalysisResult, HealthStatus
from tsap.utils.diagnostics.profiler import ProfileResult, MemoryProfileResult


class VisualizerError(TSAPError):
    """
    Exception raised for errors in visualization operations.
    
    Attributes:
        message: Error message
        details: Additional error details
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, code="VISUALIZER_ERROR", details=details)


class ChartType(str, Enum):
    """Enum for supported chart types."""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    AREA = "area"
    HEATMAP = "heatmap"
    TIMELINE = "timeline"
    GAUGE = "gauge"


class OutputFormat(str, Enum):
    """Enum for supported output formats."""
    PNG = "png"
    SVG = "svg"
    HTML = "html"
    JSON = "json"
    BASE64 = "base64"


@dataclass
class ChartOptions:
    """
    Options for chart generation.
    
    Attributes:
        title: Chart title
        width: Chart width in pixels
        height: Chart height in pixels
        x_label: X-axis label
        y_label: Y-axis label
        colors: List of colors to use
        show_legend: Whether to show the legend
        show_grid: Whether to show grid lines
        theme: Chart theme
    """
    title: str
    width: int = 800
    height: int = 500
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    colors: List[str] = field(default_factory=lambda: [
        "#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6",
        "#1abc9c", "#d35400", "#34495e", "#7f8c8d", "#2c3e50"
    ])
    show_legend: bool = True
    show_grid: bool = True
    theme: str = "default"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "title": self.title,
            "width": self.width,
            "height": self.height,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "colors": self.colors,
            "show_legend": self.show_legend,
            "show_grid": self.show_grid,
            "theme": self.theme
        }


@dataclass
class Chart:
    """
    Base class for chart objects.
    
    Attributes:
        chart_type: Type of chart
        data: Chart data
        options: Chart options
    """
    chart_type: ChartType
    data: Dict[str, Any]
    options: ChartOptions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "chart_type": self.chart_type,
            "data": self.data,
            "options": self.options.to_dict()
        }


@dataclass
class PerformanceChart(Chart):
    """
    Chart for performance data.
    
    Attributes:
        profile_results: Dictionary of profile results by name
        metric_type: Type of metric to visualize (execution_time, calls, etc.)
        top_n: Number of top functions to include
    """
    profile_results: Dict[str, ProfileResult]
    metric_type: str = "execution_time"
    top_n: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = super().to_dict()
        result["profile_results"] = {
            name: result.to_dict()
            for name, result in self.profile_results.items()
        }
        result["metric_type"] = self.metric_type
        result["top_n"] = self.top_n
        return result


@dataclass
class HealthChart(Chart):
    """
    Chart for system health data.
    
    Attributes:
        health_results: Dictionary of health analysis results by component name
    """
    health_results: Dict[str, AnalysisResult]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = super().to_dict()
        result["health_results"] = {
            component: result.to_dict()
            for component, result in self.health_results.items()
        }
        return result


@dataclass
class TimelineChart(Chart):
    """
    Chart for time-series data.
    
    Attributes:
        timeline_data: List of data points with timestamps
        timeline_start: Start time of the timeline
        timeline_end: End time of the timeline
    """
    timeline_data: List[Dict[str, Any]]
    timeline_start: datetime
    timeline_end: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = super().to_dict()
        result["timeline_data"] = self.timeline_data
        result["timeline_start"] = self.timeline_start.isoformat()
        result["timeline_end"] = self.timeline_end.isoformat()
        return result


class DiagnosticVisualizer:
    """
    Visualizer for diagnostic data.
    
    Provides tools for generating charts and visualizations based on diagnostic data
    such as performance metrics and system health information.
    """
    def __init__(self) -> None:
        """Initialize the diagnostic visualizer."""
        self.charts: Dict[str, Chart] = {}
        self.output_directory: Optional[str] = None
        self._check_dependencies()
    
    def _check_dependencies(self) -> None:
        """
        Check if required dependencies are available.
        
        This method checks for matplotlib and bokeh, which are optional dependencies
        needed for generating charts.
        """
        self.has_matplotlib = False
        self.has_bokeh = False
        
        try:
            self.has_matplotlib = True
        except ImportError:
            logger.warning("matplotlib not available. Some chart types will be limited to HTML output.")
        
        try:
            self.has_bokeh = True
        except ImportError:
            logger.warning("bokeh not available. Interactive charts will not be available.")
    
    def set_output_directory(self, directory: str) -> None:
        """
        Set the directory for saving charts.
        
        Args:
            directory: Directory path
        """
        os.makedirs(directory, exist_ok=True)
        self.output_directory = directory
    
    def add_chart(self, chart: Chart) -> str:
        """
        Add a chart.
        
        Args:
            chart: Chart to add
            
        Returns:
            Chart ID
        """
        chart_id = f"{chart.options.title.lower().replace(' ', '_')}_{int(time.time())}"
        self.charts[chart_id] = chart
        return chart_id
    
    def get_chart(self, chart_id: str) -> Optional[Chart]:
        """
        Get a chart by ID.
        
        Args:
            chart_id: Chart ID
            
        Returns:
            Chart, or None if not found
        """
        return self.charts.get(chart_id)
    
    def generate_performance_chart(self, profile_results: Dict[str, ProfileResult], metric_type: str = "execution_time", chart_type: ChartType = ChartType.BAR, options: Optional[ChartOptions] = None) -> PerformanceChart:
        """
        Generate a chart for performance data.
        
        Args:
            profile_results: Dictionary of profile results by name
            metric_type: Type of metric to visualize (execution_time, calls, etc.)
            chart_type: Type of chart to generate
            options: Chart options
            
        Returns:
            Performance chart
            
        Raises:
            VisualizerError: If there are no profile results
        """
        if not profile_results:
            raise VisualizerError("No profile results to visualize")
        
        if options is None:
            title = f"Function {metric_type.replace('_', ' ').title()}"
            options = ChartOptions(title=title)
            
            if metric_type == "execution_time":
                options.y_label = "Time (seconds)"
            elif metric_type == "calls":
                options.y_label = "Number of calls"
        
        # Sort functions by the specified metric
        if metric_type == "execution_time":
            sorted_functions = sorted(
                profile_results.items(),
                key=lambda item: item[1].execution_time,
                reverse=True
            )
        elif metric_type == "calls":
            sorted_functions = sorted(
                profile_results.items(),
                key=lambda item: item[1].calls,
                reverse=True
            )
        elif metric_type == "total_time":
            sorted_functions = sorted(
                profile_results.items(),
                key=lambda item: item[1].execution_time * item[1].calls,
                reverse=True
            )
        else:
            sorted_functions = sorted(
                profile_results.items(),
                key=lambda item: getattr(item[1], metric_type, 0),
                reverse=True
            )
        
        # Take top 10 functions
        top_functions = sorted_functions[:10]
        
        # Prepare data based on chart type
        if chart_type in [ChartType.BAR, ChartType.LINE, ChartType.AREA]:
            labels = [func_name for func_name, _ in top_functions]
            
            if metric_type == "execution_time":
                values = [result.execution_time for _, result in top_functions]
            elif metric_type == "calls":
                values = [result.calls for _, result in top_functions]
            elif metric_type == "total_time":
                values = [result.execution_time * result.calls for _, result in top_functions]
            else:
                values = [getattr(result, metric_type, 0) for _, result in top_functions]
            
            data = {
                "labels": labels,
                "datasets": [
                    {
                        "label": metric_type.replace("_", " ").title(),
                        "data": values,
                        "backgroundColor": options.colors[0]
                    }
                ]
            }
        
        elif chart_type == ChartType.PIE:
            labels = [func_name for func_name, _ in top_functions]
            
            if metric_type == "execution_time":
                values = [result.execution_time for _, result in top_functions]
            elif metric_type == "calls":
                values = [result.calls for _, result in top_functions]
            elif metric_type == "total_time":
                values = [result.execution_time * result.calls for _, result in top_functions]
            else:
                values = [getattr(result, metric_type, 0) for _, result in top_functions]
            
            data = {
                "labels": labels,
                "datasets": [
                    {
                        "data": values,
                        "backgroundColor": options.colors[:len(values)]
                    }
                ]
            }
        
        else:
            # Default to bar chart for unsupported types
            chart_type = ChartType.BAR
            labels = [func_name for func_name, _ in top_functions]
            
            if metric_type == "execution_time":
                values = [result.execution_time for _, result in top_functions]
            elif metric_type == "calls":
                values = [result.calls for _, result in top_functions]
            elif metric_type == "total_time":
                values = [result.execution_time * result.calls for _, result in top_functions]
            else:
                values = [getattr(result, metric_type, 0) for _, result in top_functions]
            
            data = {
                "labels": labels,
                "datasets": [
                    {
                        "label": metric_type.replace("_", " ").title(),
                        "data": values,
                        "backgroundColor": options.colors[0]
                    }
                ]
            }
        
        # Create chart
        chart = PerformanceChart(
            chart_type=chart_type,
            data=data,
            options=options,
            profile_results={name: result for name, result in top_functions},
            metric_type=metric_type,
            top_n=10
        )
        
        # Add the chart to our collection
        self.add_chart(chart)
        
        return chart
    
    def generate_memory_chart(self, memory_results: Dict[str, MemoryProfileResult], metric_type: str = "memory_usage", chart_type: ChartType = ChartType.BAR, options: Optional[ChartOptions] = None) -> Chart:
        """
        Generate a chart for memory usage data.
        
        Args:
            memory_results: Dictionary of memory profile results by name
            metric_type: Type of metric to visualize (memory_usage, peak_memory)
            chart_type: Type of chart to generate
            options: Chart options
            
        Returns:
            Chart object
            
        Raises:
            VisualizerError: If there are no memory results
        """
        if not memory_results:
            raise VisualizerError("No memory profile results to visualize")
        
        if options is None:
            title = f"Function {metric_type.replace('_', ' ').title()}"
            options = ChartOptions(title=title)
            
            if metric_type in ["memory_usage", "peak_memory"]:
                options.y_label = "Memory (bytes)"
        
        # Sort functions by the specified metric
        sorted_functions = sorted(
            memory_results.items(),
            key=lambda item: getattr(item[1], metric_type, 0),
            reverse=True
        )
        
        # Take top 10 functions
        top_functions = sorted_functions[:10]
        
        # Prepare data based on chart type
        if chart_type in [ChartType.BAR, ChartType.LINE, ChartType.AREA]:
            labels = [func_name for func_name, _ in top_functions]
            values = [getattr(result, metric_type, 0) for _, result in top_functions]
            
            data = {
                "labels": labels,
                "datasets": [
                    {
                        "label": metric_type.replace("_", " ").title(),
                        "data": values,
                        "backgroundColor": options.colors[0]
                    }
                ]
            }
        
        elif chart_type == ChartType.PIE:
            labels = [func_name for func_name, _ in top_functions]
            values = [getattr(result, metric_type, 0) for _, result in top_functions]
            
            data = {
                "labels": labels,
                "datasets": [
                    {
                        "data": values,
                        "backgroundColor": options.colors[:len(values)]
                    }
                ]
            }
        
        else:
            # Default to bar chart for unsupported types
            chart_type = ChartType.BAR
            labels = [func_name for func_name, _ in top_functions]
            values = [getattr(result, metric_type, 0) for _, result in top_functions]
            
            data = {
                "labels": labels,
                "datasets": [
                    {
                        "label": metric_type.replace("_", " ").title(),
                        "data": values,
                        "backgroundColor": options.colors[0]
                    }
                ]
            }
        
        # Create chart
        chart = Chart(
            chart_type=chart_type,
            data=data,
            options=options
        )
        
        # Add the chart to our collection
        self.add_chart(chart)
        
        return chart
    
    def generate_health_chart(self, health_results: Dict[str, AnalysisResult], chart_type: ChartType = ChartType.PIE, options: Optional[ChartOptions] = None) -> HealthChart:
        """
        Generate a chart for system health data.
        
        Args:
            health_results: Dictionary of health analysis results by component name
            chart_type: Type of chart to generate
            options: Chart options
            
        Returns:
            Health chart
            
        Raises:
            VisualizerError: If there are no health results
        """
        if not health_results:
            raise VisualizerError("No health results to visualize")
        
        if options is None:
            options = ChartOptions(
                title="System Health Status",
                colors=["#2ecc71", "#f39c12", "#e74c3c", "#7f8c8d"]  # Green, Yellow, Red, Gray
            )
        
        # Count components by status
        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.WARNING: 0,
            HealthStatus.CRITICAL: 0,
            HealthStatus.UNKNOWN: 0
        }
        
        for result in health_results.values():
            status_counts[result.status] = status_counts.get(result.status, 0) + 1
        
        # Prepare data based on chart type
        if chart_type == ChartType.PIE:
            labels = [status.value.title() for status in status_counts.keys()]
            values = list(status_counts.values())
            
            data = {
                "labels": labels,
                "datasets": [
                    {
                        "data": values,
                        "backgroundColor": options.colors[:len(values)]
                    }
                ]
            }
        
        elif chart_type == ChartType.BAR:
            labels = [status.value.title() for status in status_counts.keys()]
            values = list(status_counts.values())
            
            data = {
                "labels": labels,
                "datasets": [
                    {
                        "label": "Component Count",
                        "data": values,
                        "backgroundColor": options.colors[:len(values)]
                    }
                ]
            }
        
        elif chart_type == ChartType.GAUGE:
            # Calculate health score (0-100)
            total_components = sum(status_counts.values())
            if total_components == 0:
                health_score = 100
            else:
                weighted_sum = (
                    status_counts[HealthStatus.HEALTHY] * 100 +
                    status_counts[HealthStatus.WARNING] * 50 +
                    status_counts[HealthStatus.CRITICAL] * 0 +
                    status_counts[HealthStatus.UNKNOWN] * 25
                )
                health_score = weighted_sum / total_components
            
            data = {
                "value": health_score,
                "min": 0,
                "max": 100,
                "thresholds": {
                    "0": {"color": options.colors[2], "label": "Critical"},
                    "40": {"color": options.colors[1], "label": "Warning"},
                    "75": {"color": options.colors[0], "label": "Healthy"}
                }
            }
        
        else:
            # Default to pie chart for unsupported types
            chart_type = ChartType.PIE
            labels = [status.value.title() for status in status_counts.keys()]
            values = list(status_counts.values())
            
            data = {
                "labels": labels,
                "datasets": [
                    {
                        "data": values,
                        "backgroundColor": options.colors[:len(values)]
                    }
                ]
            }
        
        # Create chart
        chart = HealthChart(
            chart_type=chart_type,
            data=data,
            options=options,
            health_results=health_results
        )
        
        # Add the chart to our collection
        self.add_chart(chart)
        
        return chart
    
    def generate_timeline_chart(self, timeline_data: List[Dict[str, Any]], timestamp_key: str = "timestamp", value_key: str = "value", series_key: Optional[str] = None, chart_type: ChartType = ChartType.LINE, options: Optional[ChartOptions] = None) -> TimelineChart:
        """
        Generate a chart for time-series data.
        
        Args:
            timeline_data: List of data points with timestamps
            timestamp_key: Key for timestamp values in data points
            value_key: Key for metric values in data points
            series_key: Key for series name in data points (for multiple series)
            chart_type: Type of chart to generate
            options: Chart options
            
        Returns:
            Timeline chart
            
        Raises:
            VisualizerError: If there are no timeline data points
        """
        if not timeline_data:
            raise VisualizerError("No timeline data to visualize")
        
        if options is None:
            options = ChartOptions(
                title="Timeline Chart",
                x_label="Time",
                y_label="Value"
            )
        
        # Convert string timestamps to datetime objects if needed
        for point in timeline_data:
            if timestamp_key in point and isinstance(point[timestamp_key], str):
                try:
                    point[timestamp_key] = datetime.fromisoformat(point[timestamp_key])
                except ValueError:
                    # Try with different format
                    try:
                        point[timestamp_key] = datetime.strptime(point[timestamp_key], "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        logger.warning(f"Could not parse timestamp: {point[timestamp_key]}")
        
        # Sort data by timestamp
        sorted_data = sorted(timeline_data, key=lambda x: x.get(timestamp_key, datetime.min))
        
        # Get start and end times
        if sorted_data:
            timeline_start = sorted_data[0].get(timestamp_key, datetime.now())
            timeline_end = sorted_data[-1].get(timestamp_key, datetime.now())
        else:
            timeline_start = datetime.now() - timedelta(hours=1)
            timeline_end = datetime.now()
        
        # Prepare data based on chart type and series
        if series_key and any(series_key in point for point in sorted_data):
            # Multiple series
            series_data = {}
            for point in sorted_data:
                series = point.get(series_key, "Unknown")
                if series not in series_data:
                    series_data[series] = []
                
                if timestamp_key in point and value_key in point:
                    series_data[series].append({
                        "x": point[timestamp_key].isoformat() if isinstance(point[timestamp_key], datetime) else point[timestamp_key],
                        "y": point[value_key]
                    })
            
            if chart_type in [ChartType.LINE, ChartType.AREA, ChartType.BAR]:
                datasets = []
                for i, (series, points) in enumerate(series_data.items()):
                    color = options.colors[i % len(options.colors)]
                    datasets.append({
                        "label": series,
                        "data": points,
                        "backgroundColor": color,
                        "borderColor": color,
                        "fill": chart_type == ChartType.AREA
                    })
                
                data = {"datasets": datasets}
            
            elif chart_type == ChartType.SCATTER:
                datasets = []
                for i, (series, points) in enumerate(series_data.items()):
                    color = options.colors[i % len(options.colors)]
                    datasets.append({
                        "label": series,
                        "data": points,
                        "backgroundColor": color
                    })
                
                data = {"datasets": datasets}
            
            else:
                # Default to line chart for unsupported types
                chart_type = ChartType.LINE
                datasets = []
                for i, (series, points) in enumerate(series_data.items()):
                    color = options.colors[i % len(options.colors)]
                    datasets.append({
                        "label": series,
                        "data": points,
                        "backgroundColor": color,
                        "borderColor": color,
                        "fill": False
                    })
                
                data = {"datasets": datasets}
            
        else:
            # Single series
            if chart_type in [ChartType.LINE, ChartType.AREA, ChartType.BAR, ChartType.SCATTER]:
                points = []
                for point in sorted_data:
                    if timestamp_key in point and value_key in point:
                        points.append({
                            "x": point[timestamp_key].isoformat() if isinstance(point[timestamp_key], datetime) else point[timestamp_key],
                            "y": point[value_key]
                        })
                
                data = {
                    "datasets": [
                        {
                            "label": value_key.replace("_", " ").title(),
                            "data": points,
                            "backgroundColor": options.colors[0],
                            "borderColor": options.colors[0],
                            "fill": chart_type == ChartType.AREA
                        }
                    ]
                }
            
            else:
                # Default to line chart for unsupported types
                chart_type = ChartType.LINE
                points = []
                for point in sorted_data:
                    if timestamp_key in point and value_key in point:
                        points.append({
                            "x": point[timestamp_key].isoformat() if isinstance(point[timestamp_key], datetime) else point[timestamp_key],
                            "y": point[value_key]
                        })
                
                data = {
                    "datasets": [
                        {
                            "label": value_key.replace("_", " ").title(),
                            "data": points,
                            "backgroundColor": options.colors[0],
                            "borderColor": options.colors[0],
                            "fill": False
                        }
                    ]
                }
        
        # Create chart
        chart = TimelineChart(
            chart_type=chart_type,
            data=data,
            options=options,
            timeline_data=timeline_data,
            timeline_start=timeline_start,
            timeline_end=timeline_end
        )
        
        # Add the chart to our collection
        self.add_chart(chart)
        
        return chart
    
    def render_chart(self, chart: Chart, output_format: OutputFormat = OutputFormat.HTML) -> Union[str, bytes]:
        """
        Render a chart to the specified format.
        
        Args:
            chart: Chart to render
            output_format: Output format
            
        Returns:
            Rendered chart as string (HTML, SVG, JSON) or bytes (PNG)
            
        Raises:
            VisualizerError: If the format is not supported or rendering fails
        """
        try:
            if output_format == OutputFormat.HTML:
                return self._render_chart_html(chart)
            elif output_format == OutputFormat.SVG:
                return self._render_chart_svg(chart)
            elif output_format == OutputFormat.PNG:
                return self._render_chart_png(chart)
            elif output_format == OutputFormat.JSON:
                return json.dumps(chart.to_dict(), indent=2)
            elif output_format == OutputFormat.BASE64:
                png_data = self._render_chart_png(chart)
                return base64.b64encode(png_data).decode("utf-8")
            else:
                raise VisualizerError(f"Unsupported output format: {output_format}")
            
        except Exception as e:
            raise VisualizerError(f"Error rendering chart: {str(e)}")
    
    def _render_chart_html(self, chart: Chart) -> str:
        """
        Render a chart as HTML.
        
        Args:
            chart: Chart to render
            
        Returns:
            HTML string
        """
        # Basic HTML template with Chart.js
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .chart-container {{ width: {width}px; height: {height}px; margin: 0 auto; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <div class="chart-container">
                <canvas id="chart"></canvas>
            </div>
            <script>
                const ctx = document.getElementById('chart').getContext('2d');
                const data = {data};
                const options = {options};
                
                new Chart(ctx, {{
                    type: '{chart_type}',
                    data: data,
                    options: options
                }});
            </script>
        </body>
        </html>
        """
        
        # Prepare chart options
        chart_options = {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {
                "title": {
                    "display": True,
                    "text": chart.options.title
                },
                "legend": {
                    "display": chart.options.show_legend
                }
            },
            "scales": {}
        }
        
        # Add axis labels if provided
        if chart.options.x_label:
            chart_options["scales"]["x"] = {
                "title": {
                    "display": True,
                    "text": chart.options.x_label
                }
            }
        
        if chart.options.y_label:
            chart_options["scales"]["y"] = {
                "title": {
                    "display": True,
                    "text": chart.options.y_label
                }
            }
        
        # Render the HTML
        return html.format(
            title=chart.options.title,
            width=chart.options.width,
            height=chart.options.height,
            chart_type=chart.chart_type,
            data=json.dumps(chart.data),
            options=json.dumps(chart_options)
        )
    
    def _render_chart_svg(self, chart: Chart) -> str:
        """
        Render a chart as SVG.
        
        Args:
            chart: Chart to render
            
        Returns:
            SVG string
            
        Raises:
            VisualizerError: If matplotlib is not available
        """
        if not self.has_matplotlib:
            raise VisualizerError("matplotlib is required for SVG output")
        
        # Import matplotlib here to avoid dependency issues
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from io import StringIO
        
        # Create figure
        fig = Figure(figsize=(chart.options.width / 100, chart.options.height / 100), dpi=100)
        ax = fig.add_subplot(111)
        
        # Render based on chart type
        if chart.chart_type == ChartType.BAR:
            if "labels" in chart.data and "datasets" in chart.data and chart.data["datasets"]:
                labels = chart.data["labels"]
                dataset = chart.data["datasets"][0]
                ax.bar(labels, dataset["data"], color=dataset.get("backgroundColor", "blue"))
        
        elif chart.chart_type == ChartType.LINE:
            if "datasets" in chart.data:
                for dataset in chart.data["datasets"]:
                    if "data" in dataset and isinstance(dataset["data"], list):
                        # Check if data is in x/y format or just values
                        if dataset["data"] and isinstance(dataset["data"][0], dict) and "x" in dataset["data"][0] and "y" in dataset["data"][0]:
                            x = [point["x"] for point in dataset["data"]]
                            y = [point["y"] for point in dataset["data"]]
                        else:
                            x = range(len(dataset["data"]))
                            y = dataset["data"]
                        
                        ax.plot(x, y, label=dataset.get("label"), color=dataset.get("borderColor", "blue"))
        
        elif chart.chart_type == ChartType.PIE:
            if "labels" in chart.data and "datasets" in chart.data and chart.data["datasets"]:
                labels = chart.data["labels"]
                dataset = chart.data["datasets"][0]
                ax.pie(dataset["data"], labels=labels, colors=dataset.get("backgroundColor", ["blue"]), autopct='%1.1f%%')
        
        elif chart.chart_type == ChartType.SCATTER:
            if "datasets" in chart.data:
                for dataset in chart.data["datasets"]:
                    if "data" in dataset and isinstance(dataset["data"], list):
                        if dataset["data"] and isinstance(dataset["data"][0], dict) and "x" in dataset["data"][0] and "y" in dataset["data"][0]:
                            x = [point["x"] for point in dataset["data"]]
                            y = [point["y"] for point in dataset["data"]]
                            ax.scatter(x, y, label=dataset.get("label"), color=dataset.get("backgroundColor", "blue"))
        
        else:
            # Default to bar chart for unsupported types
            if "labels" in chart.data and "datasets" in chart.data and chart.data["datasets"]:
                labels = chart.data["labels"]
                dataset = chart.data["datasets"][0]
                ax.bar(labels, dataset["data"], color=dataset.get("backgroundColor", "blue"))
        
        # Set labels and title
        ax.set_title(chart.options.title)
        if chart.options.x_label:
            ax.set_xlabel(chart.options.x_label)
        if chart.options.y_label:
            ax.set_ylabel(chart.options.y_label)
        
        # Add legend if needed
        if chart.options.show_legend and chart.chart_type != ChartType.PIE:
            ax.legend()
        
        # Add grid if needed
        if chart.options.show_grid:
            ax.grid(True)
        
        # Tight layout
        fig.tight_layout()
        
        # Save SVG to string
        svg_io = StringIO()
        fig.savefig(svg_io, format="svg")
        plt.close(fig)
        
        return svg_io.getvalue()
    
    def _render_chart_png(self, chart: Chart) -> bytes:
        """
        Render a chart as PNG.
        
        Args:
            chart: Chart to render
            
        Returns:
            PNG data as bytes
            
        Raises:
            VisualizerError: If matplotlib is not available
        """
        if not self.has_matplotlib:
            raise VisualizerError("matplotlib is required for PNG output")
        
        # Import matplotlib here to avoid dependency issues
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from io import BytesIO
        
        # Create figure
        fig = Figure(figsize=(chart.options.width / 100, chart.options.height / 100), dpi=100)
        ax = fig.add_subplot(111)
        
        # Render based on chart type
        if chart.chart_type == ChartType.BAR:
            if "labels" in chart.data and "datasets" in chart.data and chart.data["datasets"]:
                labels = chart.data["labels"]
                dataset = chart.data["datasets"][0]
                ax.bar(labels, dataset["data"], color=dataset.get("backgroundColor", "blue"))
        
        elif chart.chart_type == ChartType.LINE:
            if "datasets" in chart.data:
                for dataset in chart.data["datasets"]:
                    if "data" in dataset and isinstance(dataset["data"], list):
                        # Check if data is in x/y format or just values
                        if dataset["data"] and isinstance(dataset["data"][0], dict) and "x" in dataset["data"][0] and "y" in dataset["data"][0]:
                            x = [point["x"] for point in dataset["data"]]
                            y = [point["y"] for point in dataset["data"]]
                        else:
                            x = range(len(dataset["data"]))
                            y = dataset["data"]
                        
                        ax.plot(x, y, label=dataset.get("label"), color=dataset.get("borderColor", "blue"))
        
        elif chart.chart_type == ChartType.PIE:
            if "labels" in chart.data and "datasets" in chart.data and chart.data["datasets"]:
                labels = chart.data["labels"]
                dataset = chart.data["datasets"][0]
                ax.pie(dataset["data"], labels=labels, colors=dataset.get("backgroundColor", ["blue"]), autopct='%1.1f%%')
        
        elif chart.chart_type == ChartType.SCATTER:
            if "datasets" in chart.data:
                for dataset in chart.data["datasets"]:
                    if "data" in dataset and isinstance(dataset["data"], list):
                        if dataset["data"] and isinstance(dataset["data"][0], dict) and "x" in dataset["data"][0] and "y" in dataset["data"][0]:
                            x = [point["x"] for point in dataset["data"]]
                            y = [point["y"] for point in dataset["data"]]
                            ax.scatter(x, y, label=dataset.get("label"), color=dataset.get("backgroundColor", "blue"))
        
        else:
            # Default to bar chart for unsupported types
            if "labels" in chart.data and "datasets" in chart.data and chart.data["datasets"]:
                labels = chart.data["labels"]
                dataset = chart.data["datasets"][0]
                ax.bar(labels, dataset["data"], color=dataset.get("backgroundColor", "blue"))
        
        # Set labels and title
        ax.set_title(chart.options.title)
        if chart.options.x_label:
            ax.set_xlabel(chart.options.x_label)
        if chart.options.y_label:
            ax.set_ylabel(chart.options.y_label)
        
        # Add legend if needed
        if chart.options.show_legend and chart.chart_type != ChartType.PIE:
            ax.legend()
        
        # Add grid if needed
        if chart.options.show_grid:
            ax.grid(True)
        
        # Tight layout
        fig.tight_layout()
        
        # Save PNG to bytes
        png_io = BytesIO()
        fig.savefig(png_io, format="png")
        plt.close(fig)
        
        return png_io.getvalue()
    
    def save_chart(self, chart: Chart, filepath: str, output_format: Optional[OutputFormat] = None) -> str:
        """
        Save a chart to a file.
        
        Args:
            chart: Chart to save
            filepath: Path to save the chart to
            output_format: Output format (inferred from filepath if not specified)
            
        Returns:
            Path to the saved chart
            
        Raises:
            VisualizerError: If the format is not supported or saving fails
        """
        try:
            # Determine output format from filepath if not specified
            if output_format is None:
                ext = os.path.splitext(filepath)[1].lower()
                if ext == ".html":
                    output_format = OutputFormat.HTML
                elif ext == ".svg":
                    output_format = OutputFormat.SVG
                elif ext == ".png":
                    output_format = OutputFormat.PNG
                elif ext == ".json":
                    output_format = OutputFormat.JSON
                else:
                    # Default to HTML
                    output_format = OutputFormat.HTML
            
            # Render the chart
            rendered_chart = self.render_chart(chart, output_format)
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            # Write the chart to the file
            if output_format in [OutputFormat.HTML, OutputFormat.SVG, OutputFormat.JSON]:
                # Write as text
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(rendered_chart)
            else:
                # Write as binary
                with open(filepath, "wb") as f:
                    f.write(rendered_chart)
            
            logger.debug(f"Saved chart to {filepath}")
            
            return filepath
            
        except Exception as e:
            raise VisualizerError(f"Error saving chart: {str(e)}")
    
    def export_chart(self, chart_id: str, output_format: OutputFormat, directory: Optional[str] = None) -> str:
        """
        Export a chart to a file.
        
        Args:
            chart_id: ID of the chart to export
            output_format: Output format
            directory: Directory to save the chart to (defaults to output_directory)
            
        Returns:
            Path to the exported chart
            
        Raises:
            VisualizerError: If the chart is not found or cannot be exported
        """
        # Get the chart
        chart = self.get_chart(chart_id)
        if not chart:
            raise VisualizerError(f"Chart not found: {chart_id}")
        
        # Determine the directory
        if directory is None:
            if self.output_directory is None:
                raise VisualizerError("No output directory specified")
            directory = self.output_directory
        
        # Create a filename based on the chart title and timestamp
        timestamp = int(time.time())
        filename = f"{chart.options.title.lower().replace(' ', '_')}_{timestamp}.{output_format.value}"
        filepath = os.path.join(directory, filename)
        
        # Save the chart
        return self.save_chart(chart, filepath, output_format)
    
    def create_dashboard(self, charts: List[Chart], title: str = "Diagnostic Dashboard", width: int = 1200, height: int = 800) -> str:
        """
        Create an HTML dashboard with multiple charts.
        
        Args:
            charts: List of charts to include
            title: Dashboard title
            width: Dashboard width in pixels
            height: Dashboard height in pixels
            
        Returns:
            HTML dashboard
            
        Raises:
            VisualizerError: If there are no charts or rendering fails
        """
        if not charts:
            raise VisualizerError("No charts to include in dashboard")
        
        try:
            # Basic HTML template with Chart.js
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>{title}</title>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .dashboard {{ display: flex; flex-wrap: wrap; justify-content: center; }}
                    .chart-container {{ margin: 10px; border: 1px solid #ddd; border-radius: 5px; padding: 10px; }}
                </style>
            </head>
            <body>
                <h1>{title}</h1>
                <div class="dashboard">
                    {chart_divs}
                </div>
                <script>
                    {chart_scripts}
                </script>
            </body>
            </html>
            """
            
            # Prepare chart divs and scripts
            chart_divs = []
            chart_scripts = []
            
            for i, chart in enumerate(charts):
                # Calculate chart dimensions
                chart_width = min(width // 2 - 40, chart.options.width)
                chart_height = min(height // 2 - 40, chart.options.height)
                
                # Create chart div
                chart_divs.append(f"""
                <div class="chart-container" style="width: {chart_width}px; height: {chart_height}px;">
                    <h2>{chart.options.title}</h2>
                    <canvas id="chart{i}"></canvas>
                </div>
                """)
                
                # Prepare chart options
                chart_options = {
                    "responsive": True,
                    "maintainAspectRatio": False,
                    "plugins": {
                        "title": {
                            "display": False  # Title is in the div
                        },
                        "legend": {
                            "display": chart.options.show_legend
                        }
                    },
                    "scales": {}
                }
                
                # Add axis labels if provided
                if chart.options.x_label:
                    chart_options["scales"]["x"] = {
                        "title": {
                            "display": True,
                            "text": chart.options.x_label
                        }
                    }
                
                if chart.options.y_label:
                    chart_options["scales"]["y"] = {
                        "title": {
                            "display": True,
                            "text": chart.options.y_label
                        }
                    }
                
                # Create chart script
                chart_scripts.append(f"""
                const ctx{i} = document.getElementById('chart{i}').getContext('2d');
                const data{i} = {json.dumps(chart.data)};
                const options{i} = {json.dumps(chart_options)};
                
                new Chart(ctx{i}, {{
                    type: '{chart.chart_type}',
                    data: data{i},
                    options: options{i}
                }});
                """)
            
            # Render the HTML
            return html.format(
                title=title,
                chart_divs="\n".join(chart_divs),
                chart_scripts="\n".join(chart_scripts)
            )
            
        except Exception as e:
            raise VisualizerError(f"Error creating dashboard: {str(e)}")


# Singleton instance
_visualizer_instance = None


def get_visualizer() -> DiagnosticVisualizer:
    """
    Get the global diagnostic visualizer instance.
    
    Returns:
        Diagnostic visualizer instance
    """
    global _visualizer_instance
    if _visualizer_instance is None:
        _visualizer_instance = DiagnosticVisualizer()
    return _visualizer_instance


def generate_performance_chart(profile_results: Dict[str, ProfileResult], metric_type: str = "execution_time", chart_type: ChartType = ChartType.BAR, options: Optional[ChartOptions] = None) -> PerformanceChart:
    """
    Generate a chart for performance data.
    
    Args:
        profile_results: Dictionary of profile results by name
        metric_type: Type of metric to visualize (execution_time, calls, etc.)
        chart_type: Type of chart to generate
        options: Chart options
        
    Returns:
        Performance chart
    """
    return get_visualizer().generate_performance_chart(profile_results, metric_type, chart_type, options)


def generate_system_health_chart(health_results: Dict[str, AnalysisResult], chart_type: ChartType = ChartType.PIE, options: Optional[ChartOptions] = None) -> HealthChart:
    """
    Generate a chart for system health data.
    
    Args:
        health_results: Dictionary of health analysis results by component name
        chart_type: Type of chart to generate
        options: Chart options
        
    Returns:
        Health chart
    """
    return get_visualizer().generate_health_chart(health_results, chart_type, options)


def generate_timeline_chart(timeline_data: List[Dict[str, Any]], timestamp_key: str = "timestamp", value_key: str = "value", series_key: Optional[str] = None, chart_type: ChartType = ChartType.LINE, options: Optional[ChartOptions] = None) -> TimelineChart:
    """
    Generate a chart for time-series data.
    
    Args:
        timeline_data: List of data points with timestamps
        timestamp_key: Key for timestamp values in data points
        value_key: Key for metric values in data points
        series_key: Key for series name in data points (for multiple series)
        chart_type: Type of chart to generate
        options: Chart options
        
    Returns:
        Timeline chart
    """
    return get_visualizer().generate_timeline_chart(timeline_data, timestamp_key, value_key, series_key, chart_type, options)


def export_chart(chart: Chart, filepath: str, output_format: Optional[OutputFormat] = None) -> str:
    """
    Export a chart to a file.
    
    Args:
        chart: Chart to export
        filepath: Path to save the chart to
        output_format: Output format (inferred from filepath if not specified)
        
    Returns:
        Path to the exported chart
    """
    return get_visualizer().save_chart(chart, filepath, output_format)


def create_dashboard(charts: List[Chart], title: str = "Diagnostic Dashboard", output_path: Optional[str] = None) -> str:
    """
    Create an HTML dashboard with multiple charts.
    
    Args:
        charts: List of charts to include
        title: Dashboard title
        output_path: Path to save the dashboard to (optional)
        
    Returns:
        HTML dashboard or path to saved dashboard
    """
    visualizer = get_visualizer()
    dashboard = visualizer.create_dashboard(charts, title)
    
    if output_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Write the dashboard to the file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(dashboard)
        
        return output_path
    
    return dashboard