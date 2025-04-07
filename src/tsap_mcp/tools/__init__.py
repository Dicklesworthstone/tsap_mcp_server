"""
Tools package for TSAP MCP Server.

This package contains all the MCP tool implementations that wrap the 
original TSAP tools. Each module provides different functionality:

- search: Text and source code search tools (ripgrep, regex, semantic)
- processing: Data processing tools (HTML, SQL, PDF, table, text, etc.)
- analysis: Code and data analysis tools (structure, security, etc.)
- visualization: Data visualization tools (charts, graphs, etc.)
- composite: Composite tools that combine multiple core functions
"""
from typing import Optional
from mcp.server.fastmcp import FastMCP

from tsap_mcp.tools.search import register_search_tools
from tsap_mcp.tools.processing import register_processing_tools
from tsap_mcp.tools.analysis import register_analysis_tools
from tsap_mcp.tools.visualization import register_visualization_tools
from tsap_mcp.tools.composite import register_composite_tools


def register_all_tools(mcp: FastMCP) -> None:
    """Register all tools with the MCP server.
    
    This function registers all available tools from all categories
    with the provided MCP server instance.
    
    Args:
        mcp: FastMCP server instance
    """
    # Register tools by category
    register_search_tools(mcp)
    register_processing_tools(mcp)
    register_analysis_tools(mcp)
    register_visualization_tools(mcp)
    register_composite_tools(mcp)
    
    # More categories can be added in the future as needed
    # register_visualization_tools(mcp)
    # etc. 