"""
Utilities for TSAP MCP Server.

This package contains utility functions for the TSAP MCP Server
implementation, including context handling, data conversion,
and other shared functionality.
"""

from tsap_mcp.utils.context import (
    extract_tsap_context,
    get_tool_from_context,
    get_config_from_context,
    get_performance_mode_from_context,
)

from tsap_mcp.utils.visualization import (
    encode_image_base64,
    decode_image_base64,
    convert_chart_data_for_mcp,
    sanitize_chart_options,
)

from tsap_mcp.utils.search import (
    format_search_results,
    sanitize_regex_pattern,
    build_ripgrep_command,
    normalize_search_query,
    categorize_search_results,
)

from tsap_mcp.utils.processing import (
    convert_csv_to_json,
    convert_json_to_csv,
    clean_text,
    extract_table_from_markdown,
    parse_jq_query,
    generate_awk_command,
    extract_data_by_pattern,
)

from tsap_mcp.utils.analysis import (
    parse_function_signature,
    extract_class_info,
    calculate_code_complexity,
    format_security_findings,
    extract_docstring,
    analyze_imports,
    find_dependencies,
)

from tsap_mcp.utils.composite import (
    generate_diff,
    extract_text_sections,
    compile_regex_from_examples,
    analyze_file_naming_patterns,
    extract_context_around_match,
    merge_results,
)

__all__ = [
    # Context utilities
    "extract_tsap_context",
    "get_tool_from_context",
    "get_config_from_context",
    "get_performance_mode_from_context",
    
    # Visualization utilities
    "encode_image_base64",
    "decode_image_base64",
    "convert_chart_data_for_mcp",
    "sanitize_chart_options",
    
    # Search utilities
    "format_search_results",
    "sanitize_regex_pattern",
    "build_ripgrep_command",
    "normalize_search_query",
    "categorize_search_results",
    
    # Processing utilities
    "convert_csv_to_json",
    "convert_json_to_csv",
    "clean_text",
    "extract_table_from_markdown",
    "parse_jq_query",
    "generate_awk_command",
    "extract_data_by_pattern",
    
    # Analysis utilities
    "parse_function_signature",
    "extract_class_info",
    "calculate_code_complexity",
    "format_security_findings",
    "extract_docstring",
    "analyze_imports",
    "find_dependencies",
    
    # Composite utilities
    "generate_diff",
    "extract_text_sections",
    "compile_regex_from_examples",
    "analyze_file_naming_patterns",
    "extract_context_around_match",
    "merge_results",
] 