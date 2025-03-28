"""
TSAP Output Package.

This package provides formatting utilities for converting TSAP results
into various output formats (JSON, CSV, plain text, etc.).
"""

from tsap.output.formatter import (
    OutputFormatter, 
    JsonFormatter, 
    TextFormatter,
    get_formatter,
)

from tsap.output.json_output import (
    EnhancedJsonFormatter,
    JsonLinesFormatter, 
    EnhancedJsonEncoder,
    get_enhanced_json_formatter,
    get_jsonl_formatter,
)

from tsap.output.csv_output import (
    CsvFormatter,
    get_csv_formatter,
)

# Dictionary mapping format names to formatter factory functions
FORMATTERS = {
    "json": lambda **kwargs: JsonFormatter(**kwargs),
    "enhanced_json": lambda **kwargs: get_enhanced_json_formatter(**kwargs),
    "jsonl": lambda **kwargs: get_jsonl_formatter(**kwargs),
    "csv": lambda **kwargs: get_csv_formatter(**kwargs),
    "text": lambda **kwargs: TextFormatter(**kwargs),
}


def create_formatter(format_type: str, **kwargs) -> OutputFormatter:
    """Create a formatter of the specified type with the given options.
    
    Args:
        format_type: The desired output format ('json', 'csv', 'text', etc.)
        **kwargs: Format-specific options
        
    Returns:
        An instance of the appropriate OutputFormatter
        
    Raises:
        ValueError: If the format type is not supported
    """
    format_type = format_type.lower()
    
    if format_type in FORMATTERS:
        return FORMATTERS[format_type](**kwargs)
    else:
        raise ValueError(f"Unsupported format type: {format_type}. "
                         f"Supported formats: {', '.join(FORMATTERS.keys())}")


def format_output(data: any, format_type: str, **kwargs) -> str:
    """Format data using the specified formatter.
    
    Args:
        data: The data to format
        format_type: The desired output format
        **kwargs: Format-specific options
        
    Returns:
        Formatted string
    """
    formatter = create_formatter(format_type, **kwargs)
    return formatter.format(data)


def save_output(data: any, file_path: str, format_type: str = None, **kwargs) -> None:
    """Save formatted data to a file.
    
    If format_type is not specified, it will be inferred from the file extension.
    
    Args:
        data: The data to format
        file_path: Path to the output file
        format_type: The desired output format (optional)
        **kwargs: Format-specific options
        
    Raises:
        ValueError: If the format cannot be determined
    """
    # Infer format from file extension if not specified
    if format_type is None:
        ext = file_path.lower().split('.')[-1]
        if ext == 'json':
            format_type = 'json'
        elif ext == 'jsonl':
            format_type = 'jsonl'
        elif ext == 'csv':
            format_type = 'csv'
        elif ext in ['txt', 'text']:
            format_type = 'text'
        else:
            raise ValueError(f"Could not determine format from file extension: {ext}")
    
    formatter = create_formatter(format_type, **kwargs)
    formatter.save_to_file(data, file_path)


__all__ = [
    # Base formatters
    "OutputFormatter",
    "JsonFormatter",
    "TextFormatter",
    "get_formatter",
    
    # Enhanced JSON formatters
    "EnhancedJsonFormatter",
    "JsonLinesFormatter",
    "EnhancedJsonEncoder",
    "get_enhanced_json_formatter",
    "get_jsonl_formatter",
    
    # CSV formatter
    "CsvFormatter",
    "get_csv_formatter",
    
    # Factory functions
    "create_formatter",
    "format_output",
    "save_output",
]