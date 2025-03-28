"""
TSAP Reporting Module.

This module provides utilities for generating comprehensive reports from TSAP analysis results.
Reports can be generated in various formats and provide structured, summarized information
about analysis findings.
"""

import io
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Union, TextIO

from tsap.utils.logging import logger


class ReportFormat(str, Enum):
    """Available report formats."""
    
    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    CSV = "csv"


class ReportSection:
    """A section in a report with title, content, and optional subsections."""

    def __init__(
        self,
        title: str,
        content: Optional[Union[str, Dict[str, Any], List[Any]]] = None,
    ):
        """Initialize a report section.
        
        Args:
            title: Section title
            content: Section content (string, dict, or list)
        """
        self.title = title
        self.content = content or ""
        self.subsections: List[ReportSection] = []
        self.metadata: Dict[str, Any] = {}

    def add_subsection(self, section: "ReportSection") -> "ReportSection":
        """Add a subsection to this section.
        
        Args:
            section: Subsection to add
            
        Returns:
            The added subsection (for chaining)
        """
        self.subsections.append(section)
        return section

    def add_content(self, content: Union[str, Dict[str, Any], List[Any]]) -> None:
        """Add or append content to this section.
        
        Args:
            content: Content to add
        """
        if not self.content:
            self.content = content
        elif isinstance(self.content, str) and isinstance(content, str):
            self.content += "\n" + content
        elif isinstance(self.content, list) and isinstance(content, list):
            self.content.extend(content)
        elif isinstance(self.content, dict) and isinstance(content, dict):
            self.content.update(content)
        else:
            # If types don't match, convert both to strings and concatenate
            self.content = str(self.content) + "\n" + str(content)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the section to a dictionary representation.
        
        Returns:
            Dictionary representation of the section
        """
        result = {
            "title": self.title,
            "content": self.content,
        }
        
        if self.metadata:
            result["metadata"] = self.metadata
            
        if self.subsections:
            result["subsections"] = [s.to_dict() for s in self.subsections]
            
        return result


class Report:
    """A comprehensive report with metadata and multiple sections."""

    def __init__(
        self,
        title: str,
        description: Optional[str] = None,
    ):
        """Initialize a report.
        
        Args:
            title: Report title
            description: Report description
        """
        self.title = title
        self.description = description
        self.sections: List[ReportSection] = []
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
        }

    def add_section(self, section: ReportSection) -> ReportSection:
        """Add a section to the report.
        
        Args:
            section: Section to add
            
        Returns:
            The added section (for chaining)
        """
        self.sections.append(section)
        return section

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the report.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert the report to a dictionary representation.
        
        Returns:
            Dictionary representation of the report
        """
        result = {
            "title": self.title,
            "metadata": self.metadata,
            "sections": [s.to_dict() for s in self.sections],
        }
        
        if self.description:
            result["description"] = self.description
            
        return result


class ReportGenerator:
    """Generator for reports in various formats."""

    def __init__(self, report: Report):
        """Initialize a report generator.
        
        Args:
            report: Report to generate
        """
        self.report = report

    def generate(
        self, 
        format: Union[str, ReportFormat],
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """Generate the report in the specified format.
        
        Args:
            format: Output format
            output_path: Path to write the report to (None for string return)
            
        Returns:
            Report content as a string if output_path is None, otherwise None
            
        Raises:
            ValueError: If format is not supported
        """
        if isinstance(format, str):
            try:
                format = ReportFormat(format.lower())
            except ValueError:
                raise ValueError(
                    f"Unsupported format: {format}. "
                    f"Supported formats: {', '.join(f.value for f in ReportFormat)}"
                )
        
        # Generate the report content
        if format == ReportFormat.TEXT:
            content = self._generate_text()
        elif format == ReportFormat.MARKDOWN:
            content = self._generate_markdown()
        elif format == ReportFormat.HTML:
            content = self._generate_html()
        elif format == ReportFormat.JSON:
            content = self._generate_json()
        elif format == ReportFormat.CSV:
            content = self._generate_csv()
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # If output_path is provided, write the content to the file
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Report saved to {output_path}")
            return None
        
        # Otherwise, return the content as a string
        return content

    def _generate_text(self) -> str:
        """Generate the report in plain text format.
        
        Returns:
            Report content as plain text
        """
        output = io.StringIO()
        
        # Write title and description
        output.write(f"{self.report.title.upper()}\n")
        output.write("=" * len(self.report.title) + "\n\n")
        
        if self.report.description:
            output.write(f"{self.report.description}\n\n")
        
        # Write metadata
        output.write("METADATA\n")
        output.write("--------\n")
        for key, value in sorted(self.report.metadata.items()):
            output.write(f"{key}: {value}\n")
        output.write("\n")
        
        # Write sections
        for section in self.report.sections:
            self._write_text_section(section, output)
            output.write("\n")
        
        return output.getvalue()

    def _write_text_section(
        self, section: ReportSection, output: TextIO, level: int = 0
    ) -> None:
        """Write a section in text format.
        
        Args:
            section: Section to write
            output: Output stream
            level: Nesting level (0 for top-level)
        """
        # Write section title with appropriate underline
        if level == 0:
            underline_char = "="
        elif level == 1:
            underline_char = "-"
        else:
            underline_char = "~"
        
        # Apply indentation for nested sections
        indent = "  " * level
        
        # Write title and underline
        output.write(f"{indent}{section.title}\n")
        output.write(f"{indent}{underline_char * len(section.title)}\n\n")
        
        # Write content
        if section.content:
            if isinstance(section.content, str):
                # For string content, write directly with indentation
                for line in section.content.split("\n"):
                    output.write(f"{indent}{line}\n")
            elif isinstance(section.content, dict):
                # For dictionary content, write key-value pairs
                for key, value in sorted(section.content.items()):
                    if isinstance(value, (dict, list)):
                        # For complex values, write key and then nested indented value
                        output.write(f"{indent}{key}:\n")
                        formatted_value = self._format_complex_value(value)
                        for line in formatted_value.split("\n"):
                            output.write(f"{indent}  {line}\n")
                    else:
                        # For simple values, write key: value
                        output.write(f"{indent}{key}: {value}\n")
            elif isinstance(section.content, list):
                # For list content, write items with bullet points
                for item in section.content:
                    if isinstance(item, (dict, list)):
                        # For complex items, format recursively
                        output.write(f"{indent}- ")
                        formatted_item = self._format_complex_value(item)
                        first_line = True
                        for line in formatted_item.split("\n"):
                            if first_line:
                                output.write(f"{line}\n")
                                first_line = False
                            else:
                                output.write(f"{indent}  {line}\n")
                    else:
                        # For simple items, write with bullet
                        output.write(f"{indent}- {item}\n")
            
            output.write("\n")
        
        # Write subsections
        for subsection in section.subsections:
            self._write_text_section(subsection, output, level + 1)

    def _generate_markdown(self) -> str:
        """Generate the report in Markdown format.
        
        Returns:
            Report content as Markdown
        """
        output = io.StringIO()
        
        # Write title and description
        output.write(f"# {self.report.title}\n\n")
        
        if self.report.description:
            output.write(f"{self.report.description}\n\n")
        
        # Write metadata
        output.write("## Metadata\n\n")
        output.write("| Key | Value |\n")
        output.write("|-----|-------|\n")
        for key, value in sorted(self.report.metadata.items()):
            # Escape pipe characters in the value
            safe_value = str(value).replace("|", "\\|")
            output.write(f"| {key} | {safe_value} |\n")
        output.write("\n")
        
        # Write sections
        for section in self.report.sections:
            self._write_markdown_section(section, output, 2)  # Start at level 2 (##)
        
        return output.getvalue()

    def _write_markdown_section(
        self, section: ReportSection, output: TextIO, level: int
    ) -> None:
        """Write a section in Markdown format.
        
        Args:
            section: Section to write
            output: Output stream
            level: Heading level (number of # characters)
        """
        # Write section title
        output.write(f"{'#' * level} {section.title}\n\n")
        
        # Write content
        if section.content:
            if isinstance(section.content, str):
                # For string content, write directly
                output.write(f"{section.content}\n\n")
            elif isinstance(section.content, dict):
                # For dictionary content, create a table
                output.write("| Key | Value |\n")
                output.write("|-----|-------|\n")
                for key, value in sorted(section.content.items()):
                    if isinstance(value, (dict, list)):
                        # For complex values, use code block
                        output.write(f"| {key} | ```\n{self._format_complex_value(value)}\n``` |\n")
                    else:
                        # For simple values, escape pipe characters
                        safe_value = str(value).replace("|", "\\|")
                        output.write(f"| {key} | {safe_value} |\n")
                output.write("\n")
            elif isinstance(section.content, list):
                # For list content, write items with bullet points
                for item in section.content:
                    if isinstance(item, dict) and "title" in item and "content" in item:
                        # Special case for list of sections
                        output.write(f"- **{item['title']}**\n")
                        if isinstance(item["content"], str):
                            output.write(f"  {item['content']}\n")
                        else:
                            output.write(f"  ```\n{self._format_complex_value(item['content'])}\n  ```\n")
                    elif isinstance(item, (dict, list)):
                        # For complex items, use code block
                        output.write(f"- ```\n{self._format_complex_value(item)}\n  ```\n")
                    else:
                        # For simple items, write with bullet
                        output.write(f"- {item}\n")
                output.write("\n")
        
        # Write subsections
        for subsection in section.subsections:
            self._write_markdown_section(subsection, output, level + 1)

    def _generate_html(self) -> str:
        """Generate the report in HTML format.
        
        Returns:
            Report content as HTML
        """
        output = io.StringIO()
        
        # Write HTML header
        output.write("<!DOCTYPE html>\n")
        output.write("<html lang=\"en\">\n")
        output.write("<head>\n")
        output.write("  <meta charset=\"UTF-8\">\n")
        output.write("  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n")
        output.write(f"  <title>{self.report.title}</title>\n")
        output.write("  <style>\n")
        output.write("    body {\n")
        output.write("      font-family: Arial, sans-serif;\n")
        output.write("      line-height: 1.6;\n")
        output.write("      max-width: 1200px;\n")
        output.write("      margin: 0 auto;\n")
        output.write("      padding: 20px;\n")
        output.write("    }\n")
        output.write("    h1, h2, h3, h4, h5, h6 {\n")
        output.write("      color: #2c3e50;\n")
        output.write("    }\n")
        output.write("    table {\n")
        output.write("      border-collapse: collapse;\n")
        output.write("      width: 100%;\n")
        output.write("      margin-bottom: 20px;\n")
        output.write("    }\n")
        output.write("    th, td {\n")
        output.write("      border: 1px solid #ddd;\n")
        output.write("      padding: 8px;\n")
        output.write("      text-align: left;\n")
        output.write("    }\n")
        output.write("    th {\n")
        output.write("      background-color: #f2f2f2;\n")
        output.write("    }\n")
        output.write("    pre {\n")
        output.write("      background-color: #f8f8f8;\n")
        output.write("      border: 1px solid #ddd;\n")
        output.write("      border-radius: 3px;\n")
        output.write("      padding: 10px;\n")
        output.write("      overflow: auto;\n")
        output.write("    }\n")
        output.write("    .metadata {\n")
        output.write("      background-color: #f9f9f9;\n")
        output.write("      border: 1px solid #ddd;\n")
        output.write("      border-radius: 3px;\n")
        output.write("      padding: 10px;\n")
        output.write("      margin-bottom: 20px;\n")
        output.write("    }\n")
        output.write("  </style>\n")
        output.write("</head>\n")
        output.write("<body>\n")
        
        # Write title and description
        output.write(f"  <h1>{self.report.title}</h1>\n")
        
        if self.report.description:
            output.write(f"  <p>{self.report.description}</p>\n")
        
        # Write metadata
        output.write("  <div class=\"metadata\">\n")
        output.write("    <h2>Metadata</h2>\n")
        output.write("    <table>\n")
        output.write("      <tr><th>Key</th><th>Value</th></tr>\n")
        for key, value in sorted(self.report.metadata.items()):
            output.write(f"      <tr><td>{key}</td><td>{value}</td></tr>\n")
        output.write("    </table>\n")
        output.write("  </div>\n")
        
        # Write sections
        for section in self.report.sections:
            self._write_html_section(section, output, 2)  # Start at level 2 (h2)
        
        # Write HTML footer
        output.write("</body>\n")
        output.write("</html>\n")
        
        return output.getvalue()

    def _write_html_section(
        self, section: ReportSection, output: TextIO, level: int
    ) -> None:
        """Write a section in HTML format.
        
        Args:
            section: Section to write
            output: Output stream
            level: Heading level (h1, h2, etc.)
        """
        # Write section title
        output.write(f"  <h{level}>{section.title}</h{level}>\n")
        
        # Write content
        if section.content:
            if isinstance(section.content, str):
                # For string content, write as paragraphs
                paragraphs = section.content.split("\n\n")
                for paragraph in paragraphs:
                    if paragraph.strip():
                        output.write(f"  <p>{paragraph}</p>\n")
            elif isinstance(section.content, dict):
                # For dictionary content, create a table
                output.write("  <table>\n")
                output.write("    <tr><th>Key</th><th>Value</th></tr>\n")
                for key, value in sorted(section.content.items()):
                    if isinstance(value, (dict, list)):
                        # For complex values, use pre tag
                        output.write(f"    <tr><td>{key}</td><td><pre>{self._html_escape(self._format_complex_value(value))}</pre></td></tr>\n")
                    else:
                        # For simple values, write directly
                        output.write(f"    <tr><td>{key}</td><td>{value}</td></tr>\n")
                output.write("  </table>\n")
            elif isinstance(section.content, list):
                # For list content, create an unordered list
                output.write("  <ul>\n")
                for item in section.content:
                    if isinstance(item, dict) and "title" in item and "content" in item:
                        # Special case for list of sections
                        output.write(f"    <li><strong>{item['title']}</strong>: ")
                        if isinstance(item["content"], str):
                            output.write(f"{item['content']}</li>\n")
                        else:
                            output.write(f"<pre>{self._html_escape(self._format_complex_value(item['content']))}</pre></li>\n")
                    elif isinstance(item, (dict, list)):
                        # For complex items, use pre tag
                        output.write(f"    <li><pre>{self._html_escape(self._format_complex_value(item))}</pre></li>\n")
                    else:
                        # For simple items, write directly
                        output.write(f"    <li>{item}</li>\n")
                output.write("  </ul>\n")
        
        # Write subsections
        for subsection in section.subsections:
            self._write_html_section(subsection, output, min(level + 1, 6))  # Cap at h6

    def _generate_json(self) -> str:
        """Generate the report in JSON format.
        
        Returns:
            Report content as JSON
        """
        import json
        return json.dumps(self.report.to_dict(), indent=2)

    def _generate_csv(self) -> str:
        """Generate the report in CSV format.
        
        This is a simplified CSV representation as reports can be complex
        hierarchical structures that don't map directly to tabular format.
        
        Returns:
            Report content as CSV
        """
        import csv
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(["Report", self.report.title])
        writer.writerow(["Section", "Title", "Content"])
        
        # Write metadata
        for key, value in sorted(self.report.metadata.items()):
            writer.writerow(["Metadata", key, value])
        
        # Write sections (flattened)
        def write_section(section, parent_path=""):
            path = f"{parent_path}/{section.title}" if parent_path else section.title
            
            # Write the section
            if isinstance(section.content, str):
                writer.writerow(["Section", path, section.content])
            elif isinstance(section.content, (dict, list)):
                # For complex content, serialize to string
                writer.writerow(["Section", path, self._format_complex_value(section.content)])
            
            # Write subsections
            for subsection in section.subsections:
                write_section(subsection, path)
        
        # Process all sections
        for section in self.report.sections:
            write_section(section)
        
        return output.getvalue()

    def _format_complex_value(self, value: Union[Dict[str, Any], List[Any]]) -> str:
        """Format a complex value (dict or list) as a formatted string.
        
        Args:
            value: Complex value to format
            
        Returns:
            Formatted string representation
        """
        import json
        try:
            return json.dumps(value, indent=2)
        except (TypeError, ValueError):
            # Fallback for non-serializable values
            if isinstance(value, dict):
                lines = []
                for k, v in value.items():
                    lines.append(f"{k}: {v}")
                return "\n".join(lines)
            elif isinstance(value, list):
                return "\n".join(str(item) for item in value)
            return str(value)

    def _html_escape(self, text: str) -> str:
        """Escape special characters for HTML.
        
        Args:
            text: Text to escape
            
        Returns:
            Escaped text
        """
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )


class ReportBuilder:
    """Builder for constructing reports incrementally."""

    def __init__(self, title: str, description: Optional[str] = None):
        """Initialize a report builder.
        
        Args:
            title: Report title
            description: Report description
        """
        self.report = Report(title, description)
        self.current_section = None

    def add_metadata(self, key: str, value: Any) -> "ReportBuilder":
        """Add metadata to the report.
        
        Args:
            key: Metadata key
            value: Metadata value
            
        Returns:
            Self for method chaining
        """
        self.report.add_metadata(key, value)
        return self

    def add_section(self, title: str, content: Optional[Union[str, Dict[str, Any], List[Any]]] = None) -> "ReportBuilder":
        """Add a top-level section to the report.
        
        Args:
            title: Section title
            content: Section content
            
        Returns:
            Self for method chaining
        """
        section = ReportSection(title, content)
        self.report.add_section(section)
        self.current_section = section
        return self

    def add_subsection(self, title: str, content: Optional[Union[str, Dict[str, Any], List[Any]]] = None) -> "ReportBuilder":
        """Add a subsection to the current section.
        
        Args:
            title: Subsection title
            content: Subsection content
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If no current section exists
        """
        if self.current_section is None:
            raise ValueError("No current section to add subsection to")
        
        subsection = ReportSection(title, content)
        self.current_section.add_subsection(subsection)
        self.current_section = subsection
        return self

    def parent_section(self) -> "ReportBuilder":
        """Navigate to the parent of the current section.
        
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If no current section or no parent section
        """
        if self.current_section is None:
            raise ValueError("No current section")
        
        # Find the parent section
        parent = self._find_parent(self.report.sections, self.current_section)
        
        if parent is None:
            # Current section is a top-level section
            self.current_section = None
        else:
            # Current section is a subsection
            self.current_section = parent
        
        return self

    def _find_parent(
        self, sections: List[ReportSection], target: ReportSection
    ) -> Optional[ReportSection]:
        """Find the parent section of the target section.
        
        Args:
            sections: List of sections to search
            target: Target section to find the parent of
            
        Returns:
            Parent section if found, None otherwise
        """
        for section in sections:
            # Check if target is a direct subsection
            if target in section.subsections:
                return section
            
            # Check in subsections recursively
            parent = self._find_parent(section.subsections, target)
            if parent is not None:
                return parent
        
        return None

    def build(self) -> Report:
        """Build and return the final report.
        
        Returns:
            Constructed report
        """
        return self.report

    def generate(
        self, 
        format: Union[str, ReportFormat],
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """Generate the report in the specified format.
        
        Args:
            format: Output format
            output_path: Path to write the report to
            
        Returns:
            Report content if output_path is None, otherwise None
        """
        generator = ReportGenerator(self.report)
        return generator.generate(format, output_path)


# Convenience functions

def create_report_builder(title: str, description: Optional[str] = None) -> ReportBuilder:
    """Create a new report builder.
    
    Args:
        title: Report title
        description: Report description
        
    Returns:
        Initialized ReportBuilder
    """
    return ReportBuilder(title, description)


def generate_analysis_report(
    analysis_results: Dict[str, Any],
    title: Optional[str] = None,
    description: Optional[str] = None,
    format: str = "markdown",
    output_path: Optional[str] = None,
) -> Optional[str]:
    """Generate a report from analysis results.
    
    Args:
        analysis_results: Results from a TSAP analysis
        title: Report title (default: Analysis Report)
        description: Report description
        format: Output format
        output_path: Path to write the report to
        
    Returns:
        Report content if output_path is None, otherwise None
    """
    # Create the builder
    builder = ReportBuilder(
        title or "Analysis Report",
        description or "TSAP Analysis Results",
    )
    
    # Add metadata
    builder.add_metadata("analysis_type", analysis_results.get("analysis_type", "Unknown"))
    builder.add_metadata("timestamp", analysis_results.get("timestamp", datetime.now().isoformat()))
    builder.add_metadata("execution_time", analysis_results.get("execution_time", 0))
    
    # Add summary section
    if "summary" in analysis_results:
        builder.add_section("Summary", analysis_results["summary"])
    
    # Add statistics section if available
    if "statistics" in analysis_results:
        builder.add_section("Statistics", analysis_results["statistics"])
    
    # Add results section
    if "results" in analysis_results:
        results = analysis_results["results"]
        
        if isinstance(results, dict):
            builder.add_section("Results")
            
            # Add subsections for each result category
            for category, category_results in results.items():
                builder.add_subsection(category, category_results)
                builder.parent_section()  # Return to Results section
        else:
            builder.add_section("Results", results)
    
    # Generate the report
    return builder.generate(format, output_path)


def generate_comparison_report(
    baseline_results: Dict[str, Any],
    comparison_results: Dict[str, Any],
    title: Optional[str] = None,
    description: Optional[str] = None,
    format: str = "markdown",
    output_path: Optional[str] = None,
) -> Optional[str]:
    """Generate a report comparing two sets of analysis results.
    
    Args:
        baseline_results: Baseline results
        comparison_results: Results to compare against baseline
        title: Report title (default: Comparison Report)
        description: Report description
        format: Output format
        output_path: Path to write the report to
        
    Returns:
        Report content if output_path is None, otherwise None
    """
    # Create the builder
    builder = ReportBuilder(
        title or "Comparison Report",
        description or "TSAP Analysis Comparison",
    )
    
    # Add metadata
    builder.add_metadata("baseline_timestamp", baseline_results.get("timestamp", "Unknown"))
    builder.add_metadata("comparison_timestamp", comparison_results.get("timestamp", "Unknown"))
    builder.add_metadata("analysis_type", baseline_results.get("analysis_type", "Unknown"))
    
    # Add summary section
    builder.add_section("Summary", "Comparison between baseline and current analysis results.")
    
    # Add baseline results section
    builder.add_section("Baseline Results")
    if "summary" in baseline_results:
        builder.add_subsection("Summary", baseline_results["summary"])
    if "results" in baseline_results:
        builder.add_subsection("Results", baseline_results["results"])
    builder.parent_section()  # Return to root
    
    # Add comparison results section
    builder.add_section("Comparison Results")
    if "summary" in comparison_results:
        builder.add_subsection("Summary", comparison_results["summary"])
    if "results" in comparison_results:
        builder.add_subsection("Results", comparison_results["results"])
    builder.parent_section()  # Return to root
    
    # Add differences section (simplified)
    builder.add_section("Differences")
    
    # Compare statistics if available
    if "statistics" in baseline_results and "statistics" in comparison_results:
        statistics_diff = {}
        for key, value in comparison_results["statistics"].items():
            if key in baseline_results["statistics"]:
                baseline_value = baseline_results["statistics"][key]
                if value != baseline_value:
                    if isinstance(value, (int, float)) and isinstance(baseline_value, (int, float)):
                        diff = value - baseline_value
                        percent = (diff / baseline_value * 100) if baseline_value != 0 else float('inf')
                        statistics_diff[key] = f"{value} (changed by {diff:+g}, {percent:+.2f}%)"
                    else:
                        statistics_diff[key] = f"Changed from '{baseline_value}' to '{value}'"
        
        if statistics_diff:
            builder.add_subsection("Statistics Changes", statistics_diff)
    
    # Generate the report
    return builder.generate(format, output_path)