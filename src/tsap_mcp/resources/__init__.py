"""
Resources package for TSAP MCP Server.

This package contains all the MCP resource implementations that expose 
data from the original TSAP functions. Each module provides different resources:

- files: File system resources (read/write files, list directories)
- project: Project-related resources (structure, dependencies)
- config: Configuration-related resources (server settings)
- semantic: Semantic analysis resources (embeddings, knowledge base)
- analysis: Code analysis resources (structure, metrics, security)
- composite: Composite operation resources (document profiling, pattern matching)
- processing: Data processing resources (text, HTML, database operations)
- search: Search-related resources (regex, ripgrep, semantic search)
- visualization: Visualization resources (charts, graphs, code visualization)
"""
from mcp.server.fastmcp import FastMCP

from tsap_mcp.resources.files import register_file_resources
from tsap_mcp.resources.config import register_config_resources
from tsap_mcp.resources.project import register_project_resources
from tsap_mcp.resources.semantic import register_semantic_resources
from tsap_mcp.resources.analysis import register_analysis_resources
from tsap_mcp.resources.composite import register_composite_resources
from tsap_mcp.resources.processing import register_processing_resources
from tsap_mcp.resources.search import register_search_resources
from tsap_mcp.resources.visualization import register_visualization_resources


def register_all_resources(mcp: FastMCP) -> None:
    """Register all resource implementations with the MCP server.
    
    Args:
        mcp: The FastMCP server instance
    """
    register_file_resources(mcp)
    register_project_resources(mcp)
    register_config_resources(mcp)
    register_semantic_resources(mcp)
    register_analysis_resources(mcp)
    register_composite_resources(mcp)
    register_processing_resources(mcp)
    register_search_resources(mcp)
    register_visualization_resources(mcp)
    
    # More categories can be added in the future as needed 