"""
Context utilities for TSAP MCP Server.

This module provides utilities for working with MCP context objects
and extracting information.
"""
from typing import Any, Dict, Optional
from mcp.server.fastmcp import Context


def extract_tsap_context(ctx: Context) -> Dict[str, Any]:
    """Extract TSAP-specific context from MCP Context.
    
    Args:
        ctx: MCP Context object
        
    Returns:
        TSAP context dictionary
    """
    if not ctx or not hasattr(ctx, "request_context") or not hasattr(ctx.request_context, "lifespan_context"):
        return {}
    
    return ctx.request_context.lifespan_context


def get_tool_from_context(ctx: Context, tool_name: str) -> Optional[Any]:
    """Get a specific tool instance from context.
    
    Args:
        ctx: MCP Context object
        tool_name: Name of the tool to retrieve
        
    Returns:
        Tool instance or None if not found
    """
    tsap_context = extract_tsap_context(ctx)
    tools = tsap_context.get("tools", {})
    return tools.get(tool_name)


def get_config_from_context(ctx: Context) -> Dict[str, Any]:
    """Get configuration from context.
    
    Args:
        ctx: MCP Context object
        
    Returns:
        Configuration dictionary
    """
    tsap_context = extract_tsap_context(ctx)
    return tsap_context.get("config", {})


def get_performance_mode_from_context(ctx: Context) -> str:
    """Get performance mode from context.
    
    Args:
        ctx: MCP Context object
        
    Returns:
        Performance mode string (fast, standard, deep)
    """
    tsap_context = extract_tsap_context(ctx)
    return tsap_context.get("performance_mode", "standard") 