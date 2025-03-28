"""
TSAP CSV Output Formatter.

This module provides functionality for formatting TSAP results as CSV data.
"""

import csv
import io
from typing import Dict, List, Any, TextIO, Set

from tsap.output.formatter import OutputFormatter
from tsap.utils.helpers import flatten_dict


class CsvFormatter(OutputFormatter):
    """Formatter for CSV output."""

    def __init__(
        self,
        pretty: bool = True,
        dialect: str = "excel",
        delimiter: str = ",",
        include_headers: bool = True,
    ):
        """Initialize the CSV formatter.
        
        Args:
            pretty: Controls whether to add extra spacing (unused in CSV format)
            dialect: CSV dialect to use
            delimiter: Field delimiter character
            include_headers: Whether to include header row
        """
        super().__init__(pretty)
        self.dialect = dialect
        self.delimiter = delimiter
        self.include_headers = include_headers

    def format(self, data: Any) -> str:
        """Format the data as a CSV string.
        
        Supports lists of dictionaries, lists of lists, and single dictionaries.
        
        Args:
            data: The data to format
            
        Returns:
            CSV formatted string
        """
        output = io.StringIO()
        self.format_stream(data, output)
        return output.getvalue()

    def format_stream(self, data: Any, stream: TextIO) -> None:
        """Write CSV directly to a stream.
        
        Args:
            data: The data to format
            stream: Output stream to write to
        """
        if not data:
            return

        # Configure the CSV writer
        writer = csv.writer(
            stream, dialect=self.dialect, delimiter=self.delimiter
        )
        
        # Handle different data types
        if isinstance(data, list):
            if not data:
                return
                
            if isinstance(data[0], dict):
                self._write_dict_list(writer, data)
            elif isinstance(data[0], list):
                self._write_list_list(writer, data)
            else:
                # Simple list of values
                writer.writerow(data)
        elif isinstance(data, dict):
            self._write_dict(writer, data)
        else:
            # For non-structured data, just write as a single cell
            writer.writerow([str(data)])

    def _write_dict_list(self, writer: csv.writer, data: List[Dict[str, Any]]) -> None:
        """Write a list of dictionaries as CSV.
        
        Args:
            writer: CSV writer
            data: List of dictionaries to write
        """
        # Flatten nested dictionaries for CSV representation
        flat_data = [flatten_dict(item) for item in data]
        
        # Collect all possible field names across all dictionaries
        fieldnames: Set[str] = set()
        for item in flat_data:
            fieldnames.update(item.keys())
        
        fieldnames_list = sorted(list(fieldnames))
        
        # Write header row
        if self.include_headers:
            writer.writerow(fieldnames_list)
        
        # Write data rows
        for item in flat_data:
            row = [item.get(field, "") for field in fieldnames_list]
            writer.writerow(row)

    def _write_list_list(self, writer: csv.writer, data: List[List[Any]]) -> None:
        """Write a list of lists as CSV.
        
        Args:
            writer: CSV writer
            data: List of lists to write
        """
        # Directly write each row
        for row in data:
            writer.writerow(row)

    def _write_dict(self, writer: csv.writer, data: Dict[str, Any]) -> None:
        """Write a single dictionary as CSV.
        
        Writes as two columns: key and value.
        
        Args:
            writer: CSV writer
            data: Dictionary to write
        """
        # Flatten the dictionary
        flat_data = flatten_dict(data)
        
        # Write header if required
        if self.include_headers:
            writer.writerow(["Field", "Value"])
        
        # Write each key-value pair as a row
        for key, value in sorted(flat_data.items()):
            writer.writerow([key, value])


# Update the formatter registry in formatter.py to include CSV
def get_csv_formatter(pretty: bool = True, delimiter: str = ",") -> CsvFormatter:
    """Get a configured CSV formatter.
    
    Args:
        pretty: Whether to format the output in a human-readable way
        delimiter: Field delimiter character
        
    Returns:
        Configured CSV formatter
    """
    return CsvFormatter(pretty=pretty, delimiter=delimiter)