"""
Authentication models for the TSAP ToolAPI Server API.

This module defines Pydantic models for authentication-related API endpoints.
"""
from pydantic import BaseModel, Field
from typing import Optional, List

class ApiKeyResponse(BaseModel):
    """API key information response model."""
    key: str = Field(..., description="The API key")
    name: Optional[str] = Field(None, description="A friendly name for this API key")
    created_at: str = Field(..., description="When the key was created")
    last_used: Optional[str] = Field(None, description="When the key was last used")

class ApiKeyRequest(BaseModel):
    """API key creation request model."""
    name: Optional[str] = Field(None, description="A friendly name for this API key")
    permissions: Optional[List[str]] = Field(None, description="List of permission scopes")
    expiration: Optional[str] = Field(None, description="Optional expiration date")

class ApiKeyList(BaseModel):
    """List of API keys response model."""
    keys: List[ApiKeyResponse] = Field(..., description="List of API keys")
    count: int = Field(..., description="Total number of keys")

# Additional auth models can be added here as needed 