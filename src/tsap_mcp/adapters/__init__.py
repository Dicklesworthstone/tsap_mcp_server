"""
Adapters package for TSAP MCP Server.

This package provides adapters that bridge between the original TSAP
implementations and the MCP API format. Adapters handle parameter
mapping, result formatting, and error handling.
"""

from tsap_mcp.adapters.tool_adapters import (
    adapted_ripgrep_search,
    adapted_jq_query,
    adapted_html_process,
    adapted_document_profiler,
    adapted_regex_generator,
    adapted_pattern_finder,
    adapted_diff_generator,
    adapted_code_structure_analyzer,
    adapted_security_analyzer,
    adapted_chart_generator,
    adapted_graph_visualizer,
)

__all__ = [
    # Search adapters
    "adapted_ripgrep_search",
    
    # Processing adapters
    "adapted_jq_query",
    "adapted_html_process",
    
    # Composite adapters
    "adapted_document_profiler",
    "adapted_regex_generator",
    "adapted_pattern_finder",
    "adapted_diff_generator",
    
    # Analysis adapters
    "adapted_code_structure_analyzer",
    "adapted_security_analyzer",
    
    # Visualization adapters
    "adapted_chart_generator",
    "adapted_graph_visualizer",
] 