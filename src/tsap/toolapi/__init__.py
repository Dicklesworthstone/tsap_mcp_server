"""
TSAP Tool API Package.

This package provides the protocol definition, request handling,
and data models for the Tool API, which is the interface for 
AI models to interact with TSAP tools.
"""

from tsap.toolapi.protocol import (
    ToolAPIRequest,
    ToolAPIResponse,
    ToolAPIError,
    ToolAPICommandType,
    create_success_response,
    create_error_response,
)

from tsap.toolapi.handler import (
    handle_request,
    register_command_handler,
    get_command_handler,
)

# Export client implementation
from tsap.toolapi.client import (
    ToolAPIClient,
    DEFAULT_SERVER_URL,
)

# Export commonly used models
from tsap.toolapi.models import (
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
    "ToolAPIRequest",
    "ToolAPIResponse",
    "ToolAPIError",
    "ToolAPICommandType",
    "create_success_response",
    "create_error_response",
    
    # Handler
    "handle_request",
    "register_command_handler",
    "get_command_handler",
    
    # Client
    "ToolAPIClient",
    "DEFAULT_SERVER_URL",
    
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