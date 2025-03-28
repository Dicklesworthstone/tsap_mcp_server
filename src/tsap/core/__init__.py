"""
TSAP Core Tools Package.

This package provides low-level wrapper tools around high-performance CLI utilities
with standardized interfaces. These tools form the foundation of TSAP's text processing
capabilities.
"""

from tsap.core.base import (
    BaseCoreTool, 
    ToolRegistry, 
    register_tool,
)

# Import core tools
from tsap.core.ripgrep import (
    RipgrepTool,
    get_ripgrep_tool,
    ripgrep_search,
)

from tsap.core.awk import (
    AwkTool,
    get_awk_tool,
    awk_process,
)

from tsap.core.jq import (
    JqTool,
    get_jq_tool,
    jq_query,
)

from tsap.core.sqlite import (
    SqliteTool,
    get_sqlite_tool,
    sqlite_query,
)

from tsap.core.html_processor import (
    HtmlProcessor,
    get_html_processor,
    process_html,
    extract_html_text,
    extract_html_tables,
)

from tsap.core.pdf_extractor import (
    PdfExtractor,
    get_pdf_extractor,
    extract_pdf_text,
    extract_pdf_metadata,
)

from tsap.core.process import (
    run_process,
    run_pipeline,
    ProcessResult,
)

from tsap.core.validation import (
    validate_path,
    validate_regex,
    validate_json,
    validate_type,
    validate_range,
    validate_list,
    validate_dict,
    sanitize_command_args,
    validate_file_content,
    ValidationError,
    ValidationLevel,
)

# Mapping of tool names to their getter functions
CORE_TOOLS = {
    "ripgrep": get_ripgrep_tool,
    "awk": get_awk_tool,
    "jq": get_jq_tool,
    "sqlite": get_sqlite_tool,
    "html_processor": get_html_processor,
    "pdf_extractor": get_pdf_extractor,
}


def get_tool(name: str) -> BaseCoreTool:
    """Get a core tool instance by name.
    
    Args:
        name: Name of the tool to get
        
    Returns:
        Instance of the requested tool
        
    Raises:
        ValueError: If the tool name is not recognized
    """
    if name not in CORE_TOOLS:
        raise ValueError(f"Unknown tool: {name}")
    
    return CORE_TOOLS[name]()


def list_tools() -> list:
    """Get a list of available core tools.
    
    Returns:
        List of tool names
    """
    return list(CORE_TOOLS.keys())


__all__ = [
    # Base classes
    "BaseCoreTool",
    "ToolRegistry",
    "register_tool",
    
    # Tool access
    "get_tool",
    "list_tools",
    
    # Tool classes and getters
    "RipgrepTool",
    "get_ripgrep_tool",
    "AwkTool",
    "get_awk_tool",
    "JqTool",
    "get_jq_tool",
    "SqliteTool",
    "get_sqlite_tool",
    "HtmlProcessor",
    "get_html_processor",
    "PdfExtractor",
    "get_pdf_extractor",
    
    # Direct functions
    "ripgrep_search",
    "awk_process",
    "jq_query",
    "sqlite_query",
    "process_html",
    "extract_html_text",
    "extract_html_tables",
    "extract_pdf_text",
    "extract_pdf_metadata",
    
    # Process utilities
    "run_process",
    "run_pipeline",
    "ProcessResult",
    
    # Validation utilities
    "validate_path",
    "validate_regex",
    "validate_json",
    "validate_type",
    "validate_range",
    "validate_list",
    "validate_dict",
    "sanitize_command_args",
    "validate_file_content",
    "ValidationError",
    "ValidationLevel",
]