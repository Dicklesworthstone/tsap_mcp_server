"""
TSAP MCP Server - Text and Source Analysis Platform for MCP.

This module provides a Model Context Protocol (MCP) server implementation for TSAP, 
allowing TSAP tools and functionality to be used with MCP clients.
"""
from importlib.metadata import version, PackageNotFoundError

# Package version
try:
    __version__ = version("tsap_mcp")
except PackageNotFoundError:
    __version__ = "0.1.0"  # Default version if not installed as package

# Import main components for easy access
from tsap_mcp.server import mcp

# Export convenience methods
__all__ = [
    "mcp",
    "__version__",
] 