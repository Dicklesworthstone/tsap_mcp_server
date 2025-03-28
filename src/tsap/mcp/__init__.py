"""
TSAP Model Context Protocol (MCP) Package.

This package provides the protocol definition, request handling,
and data models for the Model Context Protocol, which is the
standardized interface for AI models to interact with TSAP tools.
"""

from tsap.mcp.protocol import (
    MCPRequest,
    MCPResponse,
    MCPError,
    MCPCommandType,
    create_success_response,
    create_error_response,
)

from tsap.mcp.handler import (
    handle_request,
    register_command_handler,
    get_command_handler,
)

# Export commonly used models
from tsap.mcp.models import (
    # Core tool models
    RipgrepSearchParams,
    RipgrepMatch,
    RipgrepSearchResult,
    AwkProcessParams,
    AwkProcessResult,
    JqQueryParams,
    JqQueryResult,
    SqliteQueryParams,
    SqliteQueryResult,
    HtmlProcessParams,
    HtmlProcessResult,
    PdfExtractParams,
    PdfExtractResult,
    
    # Composite operation models
    SearchPattern,
    ParallelSearchParams,
    ParallelSearchMatch,
    ParallelSearchResult,
    ContextExtractParams,
    ContextExtractResult,
    
    # Analysis models
    CodeAnalyzerParams,
    CodeAnalyzerResult,
    DocumentExplorerParams,
    DocumentExplorerResult,
)

__all__ = [
    # Protocol
    "MCPRequest",
    "MCPResponse",
    "MCPError",
    "MCPCommandType",
    "create_success_response",
    "create_error_response",
    
    # Handler
    "handle_request",
    "register_command_handler",
    "get_command_handler",
    
    # Models (Core)
    "RipgrepSearchParams",
    "RipgrepMatch",
    "RipgrepSearchResult",
    "AwkProcessParams",
    "AwkProcessResult",
    "JqQueryParams",
    "JqQueryResult",
    "SqliteQueryParams",
    "SqliteQueryResult",
    "HtmlProcessParams",
    "HtmlProcessResult",
    "PdfExtractParams",
    "PdfExtractResult",
    
    # Models (Composite)
    "SearchPattern",
    "ParallelSearchParams",
    "ParallelSearchMatch",
    "ParallelSearchResult",
    "ContextExtractParams",
    "ContextExtractResult",
    
    # Models (Analysis)
    "CodeAnalyzerParams",
    "CodeAnalyzerResult",
    "DocumentExplorerParams",
    "DocumentExplorerResult",
]