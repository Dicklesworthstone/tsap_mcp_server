"""
TSAP Helper Functions.

This module provides general utility functions used throughout the TSAP system.
"""

import uuid
import datetime
from typing import Dict, List, Any, Optional, TypeVar, Iterator

T = TypeVar('T')


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with an optional prefix.
    
    Args:
        prefix: Optional string prefix for the ID
        
    Returns:
        A unique string ID
    """
    unique_id = str(uuid.uuid4())
    return f"{prefix}{unique_id}" if prefix else unique_id


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate a string to a maximum length, appending a suffix if truncated.
    
    Args:
        text: The string to truncate
        max_length: Maximum length of the returned string including suffix
        suffix: String to append if truncation occurs
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def format_timestamp(timestamp: Optional[float] = None, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format a timestamp as a human-readable string.
    
    Args:
        timestamp: Unix timestamp (seconds since epoch), uses current time if None
        format_str: strftime format string
        
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        dt = datetime.datetime.now()
    else:
        dt = datetime.datetime.fromtimestamp(timestamp)
    return dt.strftime(format_str)


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string (e.g., "2h 30m 45s")
    """
    if seconds < 0:
        return "0s"
    
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    
    return " ".join(parts)


def format_bytes(num_bytes: int) -> str:
    """Format a byte count as a human-readable string with appropriate units.
    
    Args:
        num_bytes: Number of bytes
        
    Returns:
        Formatted string (e.g., "2.5 MB")
    """
    if num_bytes < 0:
        raise ValueError("Byte count cannot be negative")
    
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    unit_index = 0
    size = float(num_bytes)
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    return f"{size:.2f} {units[unit_index]}"


def deep_update(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update a nested dictionary with another dictionary.
    
    Args:
        target: Dictionary to update
        source: Dictionary with updates
        
    Returns:
        Updated dictionary
    """
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            target[key] = deep_update(target[key], value)
        else:
            target[key] = value
    return target


def flatten_dict(nested_dict: Dict[str, Any], separator: str = ".") -> Dict[str, Any]:
    """Flatten a nested dictionary structure into a single-level dictionary.
    
    Args:
        nested_dict: Nested dictionary
        separator: String used to join nested keys
        
    Returns:
        Flattened dictionary
    """
    result = {}
    
    def _flatten(d: Dict[str, Any], parent_key: str = ""):
        for key, value in d.items():
            new_key = f"{parent_key}{separator}{key}" if parent_key else key
            
            if isinstance(value, dict):
                _flatten(value, new_key)
            else:
                result[new_key] = value
    
    _flatten(nested_dict)
    return result


def batch_items(items: List[T], batch_size: int) -> Iterator[List[T]]:
    """Split a list into batches of specified size.
    
    Args:
        items: List of items to batch
        batch_size: Maximum size of each batch
        
    Returns:
        Iterator of batched lists
    """
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def filter_none_values(d: Dict[str, Any]) -> Dict[str, Any]:
    """Remove None values from a dictionary.
    
    Args:
        d: Dictionary to filter
        
    Returns:
        Dictionary with None values removed
    """
    return {k: v for k, v in d.items() if v is not None}


def safe_get(data: Dict[str, Any], path: str, default: Any = None, separator: str = ".") -> Any:
    """Safely get a value from a nested dictionary using a path string.
    
    Args:
        data: Dictionary to retrieve value from
        path: Path to the value (e.g., "user.address.city")
        default: Value to return if path doesn't exist
        separator: Character used to separate keys in the path
        
    Returns:
        Value at path or default if not found
    """
    keys = path.split(separator)
    result = data
    
    for key in keys:
        if isinstance(result, dict) and key in result:
            result = result[key]
        else:
            return default
    
    return result


def chunk_text(text: str, chunk_size: int, overlap: int = 0) -> List[str]:
    """Split text into overlapping chunks of a specified size.
    
    Args:
        text: Text to split into chunks
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")
    if overlap < 0:
        raise ValueError("Overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("Overlap must be less than chunk size")
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    
    return chunks