"""
Tool name mapping module for TSAP MCP Server.

This module provides a mapping between the tool names used in examples
and the actual tool names registered in the server.
"""

# Map from example tool names to actual MCP tool names
TOOL_MAP = {
    # AWK tool
    "awk_process": "process_awk",
    
    # Ripgrep search tools
    "ripgrep_search": "search",
    
    # HTML processing
    "html_process": "process_html",
    
    # Other common aliases
    "process_text": "process_text",
    "extract_data": "extract_data",
    "jq_process": "process_jq",
    "sqlite_query": "query_sqlite",
}

def get_mapped_tool_name(tool_name):
    """Get the mapped tool name for a given input tool name.
    
    Args:
        tool_name: Original tool name used in examples
        
    Returns:
        Mapped tool name or original if no mapping exists
    """
    return TOOL_MAP.get(tool_name, tool_name) 