"""
TSAP Output Formatter.

This module provides the base formatter class and utility functions
for consistent output formatting across different output formats.
"""

import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Union, TextIO

from tsap.utils.logging import logger


class OutputFormatter(ABC):
    """Base class for all output formatters.
    
    OutputFormatters transform TSAP results into specific formats (JSON, CSV, etc.)
    for export or display.
    """

    def __init__(self, pretty: bool = True):
        """Initialize the formatter.
        
        Args:
            pretty: Whether to format the output in a human-readable way
        """
        self.pretty = pretty

    @abstractmethod
    def format(self, data: Any) -> str:
        """Format the data as a string.
        
        Args:
            data: The data to format
            
        Returns:
            Formatted string representation
        """
        pass

    @abstractmethod
    def format_stream(self, data: Any, stream: TextIO) -> None:
        """Write formatted data directly to a stream.
        
        Args:
            data: The data to format
            stream: Output stream to write to
        """
        pass

    def save_to_file(self, data: Any, file_path: str) -> None:
        """Save formatted data to a file.
        
        Args:
            data: The data to format
            file_path: Path to the output file
        """
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                self.format_stream(data, f)
            logger.info(f"Saved formatted output to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save formatted output to {file_path}: {str(e)}")
            raise


class JsonFormatter(OutputFormatter):
    """Formatter for JSON output."""

    def __init__(self, pretty: bool = True, indent: int = 2):
        """Initialize the JSON formatter.
        
        Args:
            pretty: Whether to format the output in a human-readable way
            indent: Number of spaces for indentation when pretty is True
        """
        super().__init__(pretty)
        self.indent = indent if pretty else None

    def format(self, data: Any) -> str:
        """Format the data as a JSON string.
        
        Args:
            data: The data to format
            
        Returns:
            JSON string
        """
        return json.dumps(data, indent=self.indent, sort_keys=self.pretty, ensure_ascii=False)

    def format_stream(self, data: Any, stream: TextIO) -> None:
        """Write JSON directly to a stream.
        
        Args:
            data: The data to format
            stream: Output stream to write to
        """
        json.dump(data, stream, indent=self.indent, sort_keys=self.pretty, ensure_ascii=False)


class TextFormatter(OutputFormatter):
    """Formatter for plain text output."""

    def __init__(self, pretty: bool = True):
        """Initialize the text formatter.
        
        Args:
            pretty: Whether to format the output with additional spacing
        """
        super().__init__(pretty)

    def format(self, data: Any) -> str:
        """Format data as plain text.
        
        For dictionaries and lists, creates a simplified text representation.
        
        Args:
            data: The data to format
            
        Returns:
            Formatted text string
        """
        if isinstance(data, str):
            return data
        elif isinstance(data, (dict, list)):
            return self._format_complex_data(data)
        else:
            return str(data)

    def format_stream(self, data: Any, stream: TextIO) -> None:
        """Write formatted text directly to a stream.
        
        Args:
            data: The data to format
            stream: Output stream to write to
        """
        stream.write(self.format(data))

    def _format_complex_data(self, data: Union[Dict, List]) -> str:
        """Format dictionaries and lists as readable text.
        
        Args:
            data: The complex data to format
            
        Returns:
            Text representation of the data
        """
        if isinstance(data, dict):
            lines = []
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{key}:")
                    # Indent the complex value
                    formatted_value = self._format_complex_data(value)
                    lines.append("\n".join(f"  {line}" for line in formatted_value.split("\n")))
                else:
                    lines.append(f"{key}: {value}")
            return "\n".join(lines)
        elif isinstance(data, list):
            lines = []
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    lines.append(f"[{i}]")
                    # Indent the complex value
                    formatted_item = self._format_complex_data(item)
                    lines.append("\n".join(f"  {line}" for line in formatted_item.split("\n")))
                else:
                    lines.append(f"[{i}] {item}")
            return "\n".join(lines)
        return str(data)


def get_formatter(format_type: str, pretty: bool = True) -> OutputFormatter:
    """Factory function to get the appropriate formatter based on format type.
    
    Args:
        format_type: The desired output format ('json', 'text', etc.)
        pretty: Whether to format the output in a human-readable way
        
    Returns:
        An instance of the appropriate OutputFormatter
        
    Raises:
        ValueError: If the format type is not supported
    """
    format_type = format_type.lower()
    
    if format_type == "json":
        return JsonFormatter(pretty)
    elif format_type == "text":
        return TextFormatter(pretty)
    else:
        raise ValueError(f"Unsupported format type: {format_type}")