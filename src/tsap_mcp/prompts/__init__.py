"""
Prompts for TSAP MCP Server.

This package provides MCP prompt implementations for various operations
and use cases.
"""
from typing import Optional
from mcp.server.fastmcp import FastMCP

from tsap_mcp.prompts.search import register_search_prompts
from tsap_mcp.prompts.code_analysis import register_code_analysis_prompts
from tsap_mcp.prompts.visualization import register_visualization_prompts
from tsap_mcp.prompts.processing import register_processing_prompts
from tsap_mcp.prompts.composite import register_composite_prompts


def register_all_prompts(mcp: FastMCP) -> None:
    """Register all prompts with the MCP server.
    
    This function registers all available prompts from all categories
    with the provided MCP server instance.
    
    Args:
        mcp: FastMCP server instance
    """
    # Register prompts by category
    register_search_prompts(mcp)
    register_code_analysis_prompts(mcp)
    register_visualization_prompts(mcp)
    register_processing_prompts(mcp)
    register_composite_prompts(mcp)
    
    # More categories will be added as they are implemented
    # register_documentation_prompts(mcp)
    # register_optimization_prompts(mcp)
    # etc. 