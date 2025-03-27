"""
API models package for the TSAP MCP Server API.

This package contains Pydantic models specific to the API layer,
distinct from the MCP protocol models defined in tsap.mcp.models.
"""

# Common response models
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union


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