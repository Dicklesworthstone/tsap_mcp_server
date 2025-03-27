"""
TSAP MCP Server - Text Search and Processing Model Context Protocol Server.

A comprehensive implementation of the Model Context Protocol designed to give 
large language models like Claude 3.7 powerful capabilities for searching, 
analyzing, and transforming text data across various domains.
"""

from .version import __version__

# Initialize logging early
from tsap.utils.logging import logger

# Import key components for easier access
from tsap.server import create_server, start_server
from tsap.config import load_config, get_config
from tsap.performance_mode import set_performance_mode, get_performance_mode

# Package metadata
__title__ = "tsap-mcp-server"
__description__ = "Text Search and Processing Model Context Protocol Server"
__author__ = "TSAP Team"
__license__ = "MIT"

# Convenience exports
__all__ = [
    "create_server",
    "start_server",
    "load_config",
    "get_config",
    "set_performance_mode",
    "get_performance_mode",
    "logger",
    "__version__",
]