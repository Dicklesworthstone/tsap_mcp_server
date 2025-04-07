"""
API models package for the TSAP ToolAPI Server API.

This package defines Pydantic models used in the public REST API,
distinct from the ToolAPI protocol models defined in tsap.toolapi.models.
"""

# Common response models
from pydantic import BaseModel, Field
from typing import Any, List, Optional

# Import specific model definitions
# Auth models
from .auth import ApiKeyResponse, ApiKeyRequest, ApiKeyList

# Core models
from .core import (
    AsyncJobResponse, CoreJobStatusResponse,
    RipgrepSearchRequest, RipgrepSearchResponse,
    AwkProcessRequest, AwkProcessResponse,
    JqQueryRequest, JqQueryResponse,
    SqliteQueryRequest, SqliteQueryResponse,
    HtmlProcessRequest, HtmlProcessResponse,
    PdfExtractRequest, PdfExtractResponse,
    TableProcessRequest, TableProcessResponse
)

# Analysis models
from .analysis import (
    CodeAnalysisRequest, CodeAnalysisResponse,
    DocumentExplorationRequest, DocumentExplorationResponse,
    CorpusCartographyRequest, CorpusCartographyResponse,
    CounterfactualAnalysisRequest, CounterfactualAnalysisResponse,
    StrategyCompilationRequest, StrategyCompilationResponse,
    AnalysisJobStatusResponse
)


class ErrorResponse(BaseModel):
    """Error response model for API endpoints."""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Any] = Field(None, description="Additional error details")


class APIResponse(BaseModel):
    """Base response model for API endpoints."""
    status: str = Field(..., description="Response status (success or error)")
    data: Optional[Any] = Field(None, description="Response data")
    error: Optional[ErrorResponse] = Field(None, description="Error information if status is error")
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")


class PaginatedResponse(BaseModel):
    """Paginated response model for API endpoints."""
    items: List[Any] = Field(..., description="Page items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")
    pages: int = Field(..., description="Total number of pages")


# Define exported names for this package
__all__ = [
    # Auth models
    'ApiKeyResponse', 'ApiKeyRequest', 'ApiKeyList',
    
    # Core models
    'AsyncJobResponse', 'CoreJobStatusResponse',
    'RipgrepSearchRequest', 'RipgrepSearchResponse',
    'AwkProcessRequest', 'AwkProcessResponse',
    'JqQueryRequest', 'JqQueryResponse',
    'SqliteQueryRequest', 'SqliteQueryResponse',
    'HtmlProcessRequest', 'HtmlProcessResponse',
    'PdfExtractRequest', 'PdfExtractResponse',
    'TableProcessRequest', 'TableProcessResponse',
    
    # Analysis models
    'CodeAnalysisRequest', 'CodeAnalysisResponse',
    'DocumentExplorationRequest', 'DocumentExplorationResponse',
    'CorpusCartographyRequest', 'CorpusCartographyResponse',
    'CounterfactualAnalysisRequest', 'CounterfactualAnalysisResponse',
    'StrategyCompilationRequest', 'StrategyCompilationResponse',
    'AnalysisJobStatusResponse',
    
    # Models defined in this file
    'ErrorResponse', 'APIResponse', 'PaginatedResponse'
]