"""
Lifespan management for TSAP MCP Server.

This module handles application lifecycle, initialization, and context management
using the MCP SDK's lifespan protocol.
"""
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Dict, Any, Optional

from mcp.server.fastmcp import FastMCP


@dataclass
class TSAPContext:
    """Context object for TSAP shared across requests."""
    
    config: Dict[str, Any]
    tools: Dict[str, Any]
    performance_mode: str
    cache_manager: Optional[Any] = None
    semantic_search_engine: Optional[Any] = None
    plugin_manager: Optional[Any] = None


async def initialize_tool(tool_name: str, config: Dict[str, Any]) -> Any:
    """Initialize a specific tool instance.
    
    Args:
        tool_name: Name of the tool to initialize
        config: Configuration dictionary
        
    Returns:
        Initialized tool instance
    """
    # Import from original implementation
    if tool_name == "ripgrep":
        from tsap.core.ripgrep import RipgrepTool
        return RipgrepTool()
    elif tool_name == "jq":
        from tsap.core.jq import JqTool
        return JqTool()
    elif tool_name == "awk":
        from tsap.core.awk import AwkTool
        return AwkTool()
    elif tool_name == "sqlite":
        from tsap.core.sqlite import SqliteTool
        return SqliteTool()
    # Add other tools as needed
    return None


async def initialize_cache(config: Dict[str, Any]) -> Any:
    """Initialize the cache system.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized cache manager
    """
    # Import and initialize from original implementation if available
    # For now, just return a simple dictionary
    return {"enabled": True}


async def initialize_semantic_search(config: Dict[str, Any]) -> Any:
    """Initialize semantic search engine.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized semantic search engine
    """
    # Import from original implementation if available
    return None


async def initialize_plugins(config: Dict[str, Any]) -> Any:
    """Initialize plugin system.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized plugin manager
    """
    # Import from original implementation if available
    return None


async def shutdown_cache(cache_manager: Any) -> None:
    """Shutdown the cache system.
    
    Args:
        cache_manager: Cache manager to shutdown
    """
    # Clean up cache resources if needed
    pass


async def shutdown_semantic_search(engine: Any) -> None:
    """Shutdown the semantic search engine.
    
    Args:
        engine: Semantic search engine to shutdown
    """
    # Clean up semantic search resources if needed
    pass


async def shutdown_plugins(plugin_manager: Any) -> None:
    """Shutdown the plugin system.
    
    Args:
        plugin_manager: Plugin manager to shutdown
    """
    # Clean up plugin resources if needed
    pass


@asynccontextmanager
async def tsap_lifespan(server: FastMCP) -> AsyncIterator[TSAPContext]:
    """Manage the application lifecycle with type-safe context.
    
    This function initializes all necessary subsystems during startup
    and cleans them up during shutdown.
    
    Args:
        server: The FastMCP server instance
        
    Yields:
        TSAPContext with all initialized subsystems
    """
    # Load configuration from original implementation
    from tsap.config import get_config
    config = get_config().dict()  # Convert to dictionary
    
    # Initialize performance mode
    performance_mode = "standard"  # Default
    
    # Initialize cache system
    cache_manager = await initialize_cache(config)
    
    # Initialize tools subsystem
    tools = {}
    for tool_name in ["ripgrep", "jq", "awk", "sqlite"]:
        tools[tool_name] = await initialize_tool(tool_name, config)
    
    # Initialize semantic search
    semantic_search_engine = await initialize_semantic_search(config)
    
    # Initialize plugin system
    plugin_manager = await initialize_plugins(config)
    
    print(f"TSAP MCP server initialized with {len(tools)} tools")
    
    try:
        # Yield the complete context with all subsystems
        yield TSAPContext(
            config=config,
            performance_mode=performance_mode,
            tools=tools,
            cache_manager=cache_manager,
            semantic_search_engine=semantic_search_engine,
            plugin_manager=plugin_manager
        )
    finally:
        # Clean shutdown of all systems
        print("TSAP MCP server shutting down...")
        await shutdown_plugins(plugin_manager)
        await shutdown_semantic_search(semantic_search_engine)
        await shutdown_cache(cache_manager)
        print("TSAP MCP server shutdown complete") 