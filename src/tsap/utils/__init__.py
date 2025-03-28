"""
TSAP Utilities Package.

This package provides various utility functions and helpers used throughout the TSAP
system, including filesystem operations, error handling, logging, and diagnostics.
"""

from tsap.utils.filesystem import (
    is_text_file,
    detect_mime_type,
    hash_file,
    get_file_info,
    find_files,
    read_file_async,
    write_file_async,
    create_temp_dir,
    create_temp_file,
    extract_archive,
)

# Import and expose logging utilities
from tsap.utils.logging import logger

# We'll import more utility functions as they're implemented

__all__ = [
    # Filesystem utilities
    "is_text_file",
    "detect_mime_type",
    "hash_file",
    "get_file_info",
    "find_files",
    "read_file_async",
    "write_file_async",
    "create_temp_dir",
    "create_temp_file",
    "extract_archive",
    # Logging
    "logger",
]