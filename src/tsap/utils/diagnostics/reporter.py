"""
Diagnostic report generation tools for TSAP.

This module provides tools for generating diagnostic reports based on analysis results,
performance profiling data, and system health checks. It includes components for
formatting, summarizing, and presenting diagnostic information in various formats.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from tsap.utils.logging import logger
from tsap.utils.errors import TSAPError
from tsap.utils.diagnostics.analyzer import AnalysisResult, HealthStatus
from tsap.utils.diagnostics.profiler import ProfileResult, MemoryProfileResult


class ReportError(TSAPError):
    """
    Exception raised for errors in report generation.
    
    Attributes:
        message: Error message
        details: Additional error details
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, code="REPORT_ERROR", details=details)


class ReportFormat(str, Enum):
    """Enum for supported report formats."""
    TEXT = "text"
    HTML = "html"
    JSON = "json"
    MARKDOWN = "markdown"
    CSV = "csv"


@dataclass
class ReportSection:
    """
    Section of a diagnostic report.
    
    Attributes:
        title: Section title
        content: Section content (string, dict, or list)
        subsections: List of subsections
    """
    title: str
    content: Optional[Union[str, Dict[str, Any], List[Any]]] = None
    subsections: List['ReportSection'] = field(default_factory=list)
    
    def add_subsection(self, section: 'ReportSection') -> 'ReportSection':
        """
        Add a subsection.
        
        Args:
            section: Subsection to add
            
        Returns:
            The added subsection for chaining
        """
        self.subsections.append(section)
        return section
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "title": self.title,
            "content": self.content
        }
        
        if self.subsections:
            result["subsections"] = [subsection.to_dict() for subsection in self.subsections]
        
        return result


@dataclass
class Report:
    """
    Complete diagnostic report.
    
    Attributes:
        title: Report title
        timestamp: Report generation time
        sections: List of report sections
        metadata: Report metadata
    """
    title: str
    timestamp: datetime = field(default_factory=datetime.now)
    sections: List[ReportSection] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_section(self, section: ReportSection) -> ReportSection:
        """
        Add a section to the report.
        
        Args:
            section: Section to add
            
        Returns:
            The added section for chaining
        """
        self.sections.append(section)
        return section
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "title": self.title,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "sections": [section.to_dict() for section in self.sections]
        }


@dataclass
class SystemHealthReport(Report):
    """
    System health diagnostic report.
    
    Attributes:
        overall_status: Overall system health status
        components: List of component health results
    """
    overall_status: HealthStatus = HealthStatus.UNKNOWN
    components: Dict[str, AnalysisResult] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = super().to_dict()
        result["overall_status"] = self.overall_status
        result["components"] = {
            component: result.to_dict()
            for component, result in self.components.items()
        }
        return result


@dataclass
class PerformanceReport(Report):
    """
    Performance diagnostic report.
    
    Attributes:
        profile_results: Dictionary of profile results by name
        memory_results: Dictionary of memory profile results by name
        metrics: Additional performance metrics
    """
    profile_results: Dict[str, ProfileResult] = field(default_factory=dict)
    memory_results: Dict[str, MemoryProfileResult] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = super().to_dict()
        result["profile_results"] = {
            name: profile.to_dict()
            for name, profile in self.profile_results.items()
        }
        result["memory_results"] = {
            name: profile.to_dict()
            for name, profile in self.memory_results.items()
        }
        result["metrics"] = self.metrics
        return result


class DiagnosticReporter:
    """
    Reporter for generating diagnostic reports.
    
    Provides tools for generating, formatting, and exporting diagnostic reports
    based on system health, performance, and other diagnostic data.
    """
    def __init__(self) -> None:
        """Initialize the diagnostic reporter."""
        self.reports: Dict[str, Report] = {}
        self.report_directory: Optional[str] = None
    
    def set_report_directory(self, directory: str) -> None:
        """
        Set the directory for saving reports.
        
        Args:
            directory: Directory path
        """
        os.makedirs(directory, exist_ok=True)
        self.report_directory = directory
    
    def add_report(self, report: Report) -> None:
        """
        Add a report.
        
        Args:
            report: Report to add
        """
        report_id = f"{report.title.lower().replace(' ', '_')}_{int(time.time())}"
        self.reports[report_id] = report
    
    def get_report(self, report_id: str) -> Optional[Report]:
        """
        Get a report by ID.
        
        Args:
            report_id: Report ID
            
        Returns:
            Report, or None if not found
        """
        return self.reports.get(report_id)
    
    def generate_system_health_report(self, health_results: Dict[str, AnalysisResult]) -> SystemHealthReport:
        """
        Generate a system health report.
        
        Args:
            health_results: Dictionary of health analysis results by component name
            
        Returns:
            System health report
        """
        # Determine overall status
        overall_status = HealthStatus.HEALTHY
        for result in health_results.values():
            if result.status == HealthStatus.CRITICAL:
                overall_status = HealthStatus.CRITICAL
                break
            elif result.status == HealthStatus.WARNING and overall_status != HealthStatus.CRITICAL:
                overall_status = HealthStatus.WARNING
        
        # Create report
        report = SystemHealthReport(
            title="System Health Report",
            overall_status=overall_status,
            components=health_results,
            metadata={
                "report_type": "system_health",
                "component_count": len(health_results)
            }
        )
        
        # Add summary section
        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.WARNING: 0,
            HealthStatus.CRITICAL: 0,
            HealthStatus.UNKNOWN: 0
        }
        
        for result in health_results.values():
            status_counts[result.status] = status_counts.get(result.status, 0) + 1
        
        report.add_section(ReportSection(
            title="Summary",
            content={
                "overall_status": overall_status,
                "timestamp": report.timestamp.isoformat(),
                "components": len(health_results),
                "status_counts": status_counts
            }
        ))
        
        # Add components section
        components_section = report.add_section(ReportSection(
            title="Components"
        ))
        
        # First add critical components
        critical_components = {
            name: result for name, result in health_results.items()
            if result.status == HealthStatus.CRITICAL
        }
        
        if critical_components:
            critical_section = components_section.add_subsection(ReportSection(
                title="Critical Components",
                content=f"Found {len(critical_components)} components in critical state."
            ))
            
            for name, result in critical_components.items():
                critical_section.add_subsection(ReportSection(
                    title=name,
                    content={
                        "status": result.status,
                        "details": result.details,
                        "recommendations": result.recommendations
                    }
                ))
        
        # Then add warning components
        warning_components = {
            name: result for name, result in health_results.items()
            if result.status == HealthStatus.WARNING
        }
        
        if warning_components:
            warning_section = components_section.add_subsection(ReportSection(
                title="Warning Components",
                content=f"Found {len(warning_components)} components in warning state."
            ))
            
            for name, result in warning_components.items():
                warning_section.add_subsection(ReportSection(
                    title=name,
                    content={
                        "status": result.status,
                        "details": result.details,
                        "recommendations": result.recommendations
                    }
                ))
        
        # Then add healthy components
        healthy_components = {
            name: result for name, result in health_results.items()
            if result.status == HealthStatus.HEALTHY
        }
        
        if healthy_components:
            healthy_section = components_section.add_subsection(ReportSection(
                title="Healthy Components",
                content=f"Found {len(healthy_components)} components in healthy state."
            ))
            
            for name, result in healthy_components.items():
                healthy_section.add_subsection(ReportSection(
                    title=name,
                    content={
                        "status": result.status,
                        "details": result.details
                    }
                ))
        
        # Add the report to our collection
        self.add_report(report)
        
        return report
    
    def generate_performance_report(self, profile_results: Dict[str, ProfileResult], memory_results: Dict[str, MemoryProfileResult] = None) -> PerformanceReport:
        """
        Generate a performance report.
        
        Args:
            profile_results: Dictionary of profile results by name
            memory_results: Dictionary of memory profile results by name
            
        Returns:
            Performance report
        """
        memory_results = memory_results or {}
        
        # Create report
        report = PerformanceReport(
            title="Performance Report",
            profile_results=profile_results,
            memory_results=memory_results,
            metadata={
                "report_type": "performance",
                "function_count": len(profile_results),
                "memory_profile_count": len(memory_results)
            }
        )
        
        # Add summary section
        total_execution_time = sum(result.execution_time for result in profile_results.values())
        total_calls = sum(result.calls for result in profile_results.values())
        
        report.add_section(ReportSection(
            title="Summary",
            content={
                "timestamp": report.timestamp.isoformat(),
                "profiled_functions": len(profile_results),
                "memory_profiled_functions": len(memory_results),
                "total_execution_time": total_execution_time,
                "total_calls": total_calls,
                "average_execution_time": total_execution_time / total_calls if total_calls > 0 else 0
            }
        ))
        
        # Add execution time section
        execution_section = report.add_section(ReportSection(
            title="Execution Time"
        ))
        
        # Sort functions by total execution time
        sorted_functions = sorted(
            profile_results.items(),
            key=lambda item: item[1].execution_time * item[1].calls,
            reverse=True
        )
        
        # Add top 10 functions by execution time
        top_execution_section = execution_section.add_subsection(ReportSection(
            title="Top Functions by Execution Time",
            content="The following functions have the highest total execution time."
        ))
        
        for name, result in sorted_functions[:10]:
            top_execution_section.add_subsection(ReportSection(
                title=name,
                content={
                    "execution_time": result.execution_time,
                    "calls": result.calls,
                    "total_time": result.execution_time * result.calls,
                    "average_time": result.execution_time / result.calls if result.calls > 0 else 0,
                    "metadata": result.metadata
                }
            ))
        
        # Add memory usage section if available
        if memory_results:
            memory_section = report.add_section(ReportSection(
                title="Memory Usage"
            ))
            
            # Sort functions by memory usage
            sorted_memory = sorted(
                memory_results.items(),
                key=lambda item: item[1].memory_usage,
                reverse=True
            )
            
            # Add top 10 functions by memory usage
            top_memory_section = memory_section.add_subsection(ReportSection(
                title="Top Functions by Memory Usage",
                content="The following functions have the highest memory usage."
            ))
            
            for name, result in sorted_memory[:10]:
                top_memory_section.add_subsection(ReportSection(
                    title=name,
                    content={
                        "memory_usage": result.memory_usage,
                        "peak_memory": result.peak_memory,
                        "metadata": result.metadata
                    }
                ))
        
        # Add the report to our collection
        self.add_report(report)
        
        return report
    
    def generate_anomaly_report(self, anomalies: List[Dict[str, Any]]) -> Report:
        """
        Generate an anomaly report.
        
        Args:
            anomalies: List of anomaly results
            
        Returns:
            Anomaly report
        """
        # Create report
        report = Report(
            title="Anomaly Report",
            metadata={
                "report_type": "anomaly",
                "anomaly_count": len(anomalies)
            }
        )
        
        # Add summary section
        report.add_section(ReportSection(
            title="Summary",
            content={
                "timestamp": report.timestamp.isoformat(),
                "anomalies_detected": len(anomalies),
            }
        ))
        
        # Add anomalies section
        anomalies_section = report.add_section(ReportSection(
            title="Detected Anomalies",
            content=f"Found {len(anomalies)} anomalies."
        ))
        
        # Add each anomaly
        for i, anomaly in enumerate(anomalies):
            anomalies_section.add_subsection(ReportSection(
                title=f"Anomaly {i+1}: {anomaly.get('type', 'Unknown')}",
                content=anomaly
            ))
        
        # Add the report to our collection
        self.add_report(report)
        
        return report
    
    def format_report(self, report: Report, format: ReportFormat) -> str:
        """
        Format a report as a string.
        
        Args:
            report: Report to format
            format: Output format
            
        Returns:
            Formatted report string
            
        Raises:
            ReportError: If the format is not supported
        """
        if format == ReportFormat.JSON:
            return json.dumps(report.to_dict(), indent=2)
        elif format == ReportFormat.TEXT:
            return self._format_text_report(report)
        elif format == ReportFormat.MARKDOWN:
            return self._format_markdown_report(report)
        elif format == ReportFormat.HTML:
            return self._format_html_report(report)
        elif format == ReportFormat.CSV:
            return self._format_csv_report(report)
        else:
            raise ReportError(f"Unsupported report format: {format}")
    
    def _format_text_report(self, report: Report) -> str:
        """
        Format a report as plain text.
        
        Args:
            report: Report to format
            
        Returns:
            Formatted report string
        """
        lines = []
        
        # Add title
        lines.append(report.title.upper())
        lines.append("=" * len(report.title))
        lines.append("")
        
        # Add timestamp
        lines.append(f"Generated: {report.timestamp.isoformat()}")
        lines.append("")
        
        # Add metadata
        if report.metadata:
            lines.append("Metadata:")
            for key, value in report.metadata.items():
                lines.append(f"  {key}: {value}")
            lines.append("")
        
        # Add sections
        for section in report.sections:
            self._format_text_section(section, lines)
        
        return "\n".join(lines)
    
    def _format_text_section(self, section: ReportSection, lines: List[str], level: int = 0) -> None:
        """
        Format a report section as plain text.
        
        Args:
            section: Section to format
            lines: List of lines to append to
            level: Indentation level
        """
        # Add title
        indent = "  " * level
        lines.append(f"{indent}{section.title}")
        lines.append(f"{indent}{'-' * len(section.title)}")
        
        # Add content
        if section.content is not None:
            if isinstance(section.content, str):
                # String content
                lines.append(f"{indent}{section.content}")
            elif isinstance(section.content, (dict, list)):
                # Dictionary or list content
                content_str = json.dumps(section.content, indent=2)
                for line in content_str.split("\n"):
                    lines.append(f"{indent}  {line}")
            lines.append("")
        
        # Add subsections
        for subsection in section.subsections:
            self._format_text_section(subsection, lines, level + 1)
    
    def _format_markdown_report(self, report: Report) -> str:
        """
        Format a report as Markdown.
        
        Args:
            report: Report to format
            
        Returns:
            Formatted report string
        """
        lines = []
        
        # Add title
        lines.append(f"# {report.title}")
        lines.append("")
        
        # Add timestamp
        lines.append(f"*Generated: {report.timestamp.isoformat()}*")
        lines.append("")
        
        # Add metadata
        if report.metadata:
            lines.append("## Metadata")
            lines.append("")
            for key, value in report.metadata.items():
                lines.append(f"* **{key}:** {value}")
            lines.append("")
        
        # Add sections
        for section in report.sections:
            self._format_markdown_section(section, lines, 2)
        
        return "\n".join(lines)
    
    def _format_markdown_section(self, section: ReportSection, lines: List[str], level: int = 2) -> None:
        """
        Format a report section as Markdown.
        
        Args:
            section: Section to format
            lines: List of lines to append to
            level: Heading level
        """
        # Add title
        lines.append(f"{'#' * level} {section.title}")
        lines.append("")
        
        # Add content
        if section.content is not None:
            if isinstance(section.content, str):
                # String content
                lines.append(section.content)
            elif isinstance(section.content, (dict, list)):
                # Dictionary or list content
                lines.append("```json")
                lines.append(json.dumps(section.content, indent=2))
                lines.append("```")
            lines.append("")
        
        # Add subsections
        for subsection in section.subsections:
            self._format_markdown_section(subsection, lines, level + 1)
    
    def _format_html_report(self, report: Report) -> str:
        """
        Format a report as HTML.
        
        Args:
            report: Report to format
            
        Returns:
            Formatted report string
        """
        lines = []
        
        # Add HTML header
        lines.append("<!DOCTYPE html>")
        lines.append("<html>")
        lines.append("<head>")
        lines.append(f"<title>{report.title}</title>")
        lines.append("<style>")
        lines.append("body { font-family: Arial, sans-serif; margin: 20px; }")
        lines.append("h1 { color: #2c3e50; }")
        lines.append("h2 { color: #3498db; }")
        lines.append("h3 { color: #2980b9; }")
        lines.append("h4 { color: #1abc9c; }")
        lines.append("h5 { color: #16a085; }")
        lines.append("h6 { color: #27ae60; }")
        lines.append("pre { background-color: #f8f9fa; padding: 10px; border-radius: 5px; }")
        lines.append("</style>")
        lines.append("</head>")
        lines.append("<body>")
        
        # Add title
        lines.append(f"<h1>{report.title}</h1>")
        
        # Add timestamp
        lines.append(f"<p><em>Generated: {report.timestamp.isoformat()}</em></p>")
        
        # Add metadata
        if report.metadata:
            lines.append("<h2>Metadata</h2>")
            lines.append("<ul>")
            for key, value in report.metadata.items():
                lines.append(f"<li><strong>{key}:</strong> {value}</li>")
            lines.append("</ul>")
        
        # Add sections
        for section in report.sections:
            self._format_html_section(section, lines, 2)
        
        # Add HTML footer
        lines.append("</body>")
        lines.append("</html>")
        
        return "\n".join(lines)
    
    def _format_html_section(self, section: ReportSection, lines: List[str], level: int = 2) -> None:
        """
        Format a report section as HTML.
        
        Args:
            section: Section to format
            lines: List of lines to append to
            level: Heading level
        """
        # Add title
        lines.append(f"<h{level}>{section.title}</h{level}>")
        
        # Add content
        if section.content is not None:
            if isinstance(section.content, str):
                # String content
                lines.append(f"<p>{section.content}</p>")
            elif isinstance(section.content, (dict, list)):
                # Dictionary or list content
                lines.append("<pre>")
                lines.append(json.dumps(section.content, indent=2))
                lines.append("</pre>")
        
        # Add subsections
        for subsection in section.subsections:
            self._format_html_section(subsection, lines, min(level + 1, 6))
    
    def _format_csv_report(self, report: Report) -> str:
        """
        Format a report as CSV.
        
        Note: This format is limited and only works well for tabular data.
        
        Args:
            report: Report to format
            
        Returns:
            Formatted report string
        """
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header row
        writer.writerow(["Report", report.title])
        writer.writerow(["Generated", report.timestamp.isoformat()])
        writer.writerow([])
        
        # Write metadata
        if report.metadata:
            writer.writerow(["Metadata"])
            for key, value in report.metadata.items():
                writer.writerow([key, value])
            writer.writerow([])
        
        # Write sections (limited support)
        for section in report.sections:
            self._format_csv_section(section, writer)
        
        return output.getvalue()
    
    def _format_csv_section(self, section: ReportSection, writer) -> None:
        """
        Format a report section as CSV.
        
        Args:
            section: Section to format
            writer: CSV writer
        """
        # Write section title
        writer.writerow([section.title])
        
        # Write content (limited support)
        if section.content is not None:
            if isinstance(section.content, str):
                # String content
                writer.writerow([section.content])
            elif isinstance(section.content, dict):
                # Dictionary content
                for key, value in section.content.items():
                    if not isinstance(value, (dict, list)):
                        writer.writerow([key, value])
            elif isinstance(section.content, list) and all(isinstance(item, dict) for item in section.content):
                # List of dictionaries - treat as table
                if section.content:
                    # Write header row
                    writer.writerow(section.content[0].keys())
                    # Write data rows
                    for item in section.content:
                        writer.writerow(item.values())
        
        writer.writerow([])
        
        # Write subsections
        for subsection in section.subsections:
            self._format_csv_section(subsection, writer)
    
    def save_report(self, report: Report, filepath: str, format: ReportFormat) -> None:
        """
        Save a report to a file.
        
        Args:
            report: Report to save
            filepath: Path to save the report to
            format: Output format
            
        Raises:
            ReportError: If the format is not supported or the file cannot be written
        """
        try:
            # Format the report
            formatted_report = self.format_report(report, format)
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            # Write the report to the file
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(formatted_report)
            
            logger.debug(f"Saved report to {filepath}")
            
        except Exception as e:
            raise ReportError(f"Error saving report: {str(e)}")
    
    def export_report(self, report_id: str, format: ReportFormat, directory: Optional[str] = None) -> str:
        """
        Export a report to a file.
        
        Args:
            report_id: ID of the report to export
            format: Output format
            directory: Directory to save the report to (defaults to report_directory)
            
        Returns:
            Path to the exported report
            
        Raises:
            ReportError: If the report is not found or cannot be exported
        """
        # Get the report
        report = self.get_report(report_id)
        if not report:
            raise ReportError(f"Report not found: {report_id}")
        
        # Determine the directory
        if directory is None:
            if self.report_directory is None:
                raise ReportError("No report directory specified")
            directory = self.report_directory
        
        # Create a filename based on the report title and timestamp
        timestamp = int(time.time())
        filename = f"{report.title.lower().replace(' ', '_')}_{timestamp}.{format.value}"
        filepath = os.path.join(directory, filename)
        
        # Save the report
        self.save_report(report, filepath, format)
        
        return filepath


# Singleton instance
_reporter_instance = None


def get_reporter() -> DiagnosticReporter:
    """
    Get the global diagnostic reporter instance.
    
    Returns:
        Diagnostic reporter instance
    """
    global _reporter_instance
    if _reporter_instance is None:
        _reporter_instance = DiagnosticReporter()
    return _reporter_instance


def generate_health_report(health_results: Dict[str, AnalysisResult]) -> SystemHealthReport:
    """
    Generate a system health report.
    
    Args:
        health_results: Dictionary of health analysis results by component name
        
    Returns:
        System health report
    """
    return get_reporter().generate_system_health_report(health_results)


def generate_performance_report(profile_results: Dict[str, ProfileResult], memory_results: Dict[str, MemoryProfileResult] = None) -> PerformanceReport:
    """
    Generate a performance report.
    
    Args:
        profile_results: Dictionary of profile results by name
        memory_results: Dictionary of memory profile results by name
        
    Returns:
        Performance report
    """
    return get_reporter().generate_performance_report(profile_results, memory_results)


def generate_anomaly_report(anomalies: List[Dict[str, Any]]) -> Report:
    """
    Generate an anomaly report.
    
    Args:
        anomalies: List of anomaly results
        
    Returns:
        Anomaly report
    """
    return get_reporter().generate_anomaly_report(anomalies)


def export_report(report: Report, format: ReportFormat, filepath: Optional[str] = None) -> str:
    """
    Export a report to a file.
    
    Args:
        report: Report to export
        format: Output format
        filepath: Path to save the report to (optional)
        
    Returns:
        Path to the exported report
    """
    reporter = get_reporter()
    
    # Add the report to the reporter
    reporter.add_report(report)
    
    # Determine the report ID
    report_id = next(
        (id for id, r in reporter.reports.items() if r == report),
        None
    )
    
    if report_id is None:
        raise ReportError("Failed to get report ID")
    
    # If filepath is specified, use it
    if filepath:
        # Ensure directory exists
        directory = os.path.dirname(os.path.abspath(filepath))
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        # Save the report
        reporter.save_report(report, filepath, format)
        return filepath
    
    # Otherwise export using the reporter
    return reporter.export_report(report_id, format)