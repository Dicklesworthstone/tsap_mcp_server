"""
TSAP MCP Server implementation.

This module provides the main entry point for the TSAP MCP server,
integrating all tools, resources, and prompts.
"""
import os
import logging
from typing import Any, Dict
from dataclasses import dataclass
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from mcp.server.fastmcp import FastMCP

# Import all component registration functions
from tsap_mcp.tools.search import register_search_tools
from tsap_mcp.tools.processing import register_processing_tools
from tsap_mcp.tools.analysis import register_analysis_tools
from tsap_mcp.tools.visualization import register_visualization_tools
from tsap_mcp.tools.composite import register_composite_tools

from tsap_mcp.resources.files import register_file_resources
from tsap_mcp.resources.project import register_project_resources
from tsap_mcp.resources.config import register_config_resources
from tsap_mcp.resources.semantic import register_semantic_resources

from tsap_mcp.prompts.search import register_search_prompts
from tsap_mcp.prompts.analysis import register_code_analysis_prompts

# Set up logging
logger = logging.getLogger("tsap_mcp")
logger.setLevel(logging.INFO)

@dataclass
class AppContext:
    """Type-safe application context for the TSAP MCP server."""
    config: Dict[str, Any]

@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle.
    
    This function handles startup and shutdown operations for the server.
    
    Args:
        server: FastMCP server instance
        
    Yields:
        Strongly typed context for request handlers
    """
    # Startup operations
    logger.info("Starting TSAP MCP Server...")
    config = {
        "performance_mode": os.environ.get("TSAP_PERFORMANCE_MODE", "balanced"),
        "cache_enabled": os.environ.get("TSAP_CACHE_ENABLED", "true").lower() == "true",
        "debug": os.environ.get("TSAP_DEBUG", "false").lower() == "true",
    }
    
    # Initialize resources, if needed
    try:
        # Register all components
        register_all_components()
        
        # Yield control back to server with strongly typed context
        yield AppContext(config=config)
    finally:
        # Shutdown operations
        logger.info("Shutting down TSAP MCP Server...")

# Create MCP server with lifespan provided directly at initialization
mcp = FastMCP(
    name="TSAP MCP Server",
    description="TSAP MCP Server for text and source code analysis, processing, and visualization.",
    version="0.1.0",
    lifespan=lifespan,
    dependencies=[
        "fastapi",
        "ripgrep",
        "sentence-transformers",
        "matplotlib",
        "networkx",
    ],
)

def register_all_components() -> None:
    """Register all MCP components (tools, resources, prompts)."""
    logger.info("Registering MCP components...")
    
    # Register tools
    register_search_tools(mcp)
    register_processing_tools(mcp)
    register_analysis_tools(mcp)
    register_visualization_tools(mcp)
    register_composite_tools(mcp)
    
    # Register resources
    register_file_resources(mcp)
    register_project_resources(mcp)
    register_config_resources(mcp)
    register_semantic_resources(mcp)
    
    # Register prompts
    register_search_prompts(mcp)
    register_code_analysis_prompts(mcp)
    
    logger.info(f"Registered {len(mcp.list_tool_functions())} tools")

def run_server() -> None:
    """Run the TSAP MCP server."""
    logger.info("Running TSAP MCP Server...")
    mcp.run()

if __name__ == "__main__":
    run_server()

# Function to get the ASGI app for mounting in other applications
def get_mcp_app():
    """Get the ASGI app for the MCP server.
    
    This can be used to mount the MCP server in another ASGI application.
    
    Returns:
        ASGI application
    """
    return mcp.sse_app() 