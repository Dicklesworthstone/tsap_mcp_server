"""
Pydantic models for the core tools API endpoints.

This module defines the request and response models for the core tools API,
extending the MCP models with API-specific fields and validation.
"""

from typing import Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime

from tsap.mcp.models import (
    RipgrepSearchParams, RipgrepSearchResult,
    AwkProcessParams, AwkProcessResult,
    JqQueryParams, JqQueryResult,
    SqliteQueryParams, SqliteQueryResult,
    HtmlProcessParams, HtmlProcessResult,
    PdfExtractParams, PdfExtractResult,
    TableProcessParams, TableProcessResult
)


class RipgrepSearchRequest(BaseModel):
    """Request model for the ripgrep search API endpoint."""
    params: RipgrepSearchParams
    async_execution: bool = Field(default=False, description="Whether to execute the search asynchronously")
    performance_mode: Optional[str] = Field(default=None, description="Performance mode for this specific request")
    
    class Config:
        schema_extra = {
            "example": {
                "params": {
                    "pattern": "function",
                    "paths": ["./src"],
                    "case_sensitive": False,
                    "is_regex": True,
                    "file_patterns": ["*.py"],
                    "exclude_patterns": ["*__pycache__*"],
                    "context_lines": 2,
                    "max_matches": 100
                },
                "async_execution": False,
                "performance_mode": "standard"
            }
        }


class AwkProcessRequest(BaseModel):
    """Request model for the awk process API endpoint."""
    params: AwkProcessParams
    async_execution: bool = Field(default=False, description="Whether to execute the processing asynchronously")
    performance_mode: Optional[str] = Field(default=None, description="Performance mode for this specific request")
    
    class Config:
        schema_extra = {
            "example": {
                "params": {
                    "script": "{ print $1, $3 }",
                    "input_file": "data.txt",
                    "field_separator": ",",
                    "output_file": "output.txt"
                },
                "async_execution": False,
                "performance_mode": "standard"
            }
        }


class JqQueryRequest(BaseModel):
    """Request model for the jq query API endpoint."""
    params: JqQueryParams
    async_execution: bool = Field(default=False, description="Whether to execute the query asynchronously")
    performance_mode: Optional[str] = Field(default=None, description="Performance mode for this specific request")
    
    class Config:
        schema_extra = {
            "example": {
                "params": {
                    "query": ".items[] | select(.active == true)",
                    "input_file": "data.json",
                    "output_file": "filtered.json",
                    "raw_output": False,
                    "compact_output": False
                },
                "async_execution": False,
                "performance_mode": "standard"
            }
        }


class SqliteQueryRequest(BaseModel):
    """Request model for the SQLite query API endpoint."""
    params: SqliteQueryParams
    async_execution: bool = Field(default=False, description="Whether to execute the query asynchronously")
    performance_mode: Optional[str] = Field(default=None, description="Performance mode for this specific request")
    
    class Config:
        schema_extra = {
            "example": {
                "params": {
                    "database": "database.db",
                    "query": "SELECT * FROM users WHERE age > ?",
                    "params": [18],
                    "mode": "dict"
                },
                "async_execution": False,
                "performance_mode": "standard"
            }
        }


class HtmlProcessRequest(BaseModel):
    """Request model for the HTML process API endpoint."""
    params: HtmlProcessParams
    async_execution: bool = Field(default=False, description="Whether to execute the processing asynchronously")
    performance_mode: Optional[str] = Field(default=None, description="Performance mode for this specific request")
    
    class Config:
        schema_extra = {
            "example": {
                "params": {
                    "html_source": "file",
                    "file_path": "page.html",
                    "css_selector": "div.content",
                    "extract_links": True,
                    "extract_tables": True,
                    "extract_text": True,
                    "extract_metadata": True
                },
                "async_execution": False,
                "performance_mode": "standard"
            }
        }


class PdfExtractRequest(BaseModel):
    """Request model for the PDF extraction API endpoint."""
    params: PdfExtractParams
    async_execution: bool = Field(default=False, description="Whether to execute the extraction asynchronously")
    performance_mode: Optional[str] = Field(default=None, description="Performance mode for this specific request")
    
    class Config:
        schema_extra = {
            "example": {
                "params": {
                    "file_path": "document.pdf",
                    "pages": "1-5",
                    "extract_metadata": True,
                    "extract_text": True,
                    "extract_tables": False,
                    "extract_images": False,
                    "password": None
                },
                "async_execution": False,
                "performance_mode": "standard"
            }
        }


class TableProcessRequest(BaseModel):
    """Request model for the table processing API endpoint."""
    params: TableProcessParams
    async_execution: bool = Field(default=False, description="Whether to execute the processing asynchronously")
    performance_mode: Optional[str] = Field(default=None, description="Performance mode for this specific request")
    
    class Config:
        schema_extra = {
            "example": {
                "params": {
                    "file_path": "data.csv",
                    "format": "csv",
                    "operations": [
                        {"type": "filter", "column": "age", "operator": ">", "value": 18},
                        {"type": "sort", "column": "name", "ascending": True}
                    ],
                    "output_format": "csv",
                    "output_file": "filtered.csv"
                },
                "async_execution": False,
                "performance_mode": "standard"
            }
        }


class AsyncJobResponse(BaseModel):
    """Response model for asynchronous job submission."""
    job_id: str = Field(..., description="ID of the submitted job")
    status: str = Field(..., description="Initial status of the job")
    submitted_at: datetime = Field(..., description="Timestamp when the job was submitted")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    
    @validator('status')
    def validate_status(cls, v):
        """Validate that the status is a known value."""
        allowed_statuses = {'pending', 'running', 'completed', 'failed', 'canceled'}
        if v not in allowed_statuses:
            raise ValueError(f"Status must be one of: {', '.join(allowed_statuses)}")
        return v


class RipgrepSearchResponse(BaseModel):
    """Response model for the ripgrep search API endpoint."""
    result: RipgrepSearchResult
    execution_time: float = Field(..., description="Execution time in seconds")
    timestamp: datetime = Field(..., description="Timestamp when the result was generated")
    performance_mode: str = Field(..., description="Performance mode used for the execution")


class AwkProcessResponse(BaseModel):
    """Response model for the awk process API endpoint."""
    result: AwkProcessResult
    execution_time: float = Field(..., description="Execution time in seconds")
    timestamp: datetime = Field(..., description="Timestamp when the result was generated")
    performance_mode: str = Field(..., description="Performance mode used for the execution")


class JqQueryResponse(BaseModel):
    """Response model for the jq query API endpoint."""
    result: JqQueryResult
    execution_time: float = Field(..., description="Execution time in seconds")
    timestamp: datetime = Field(..., description="Timestamp when the result was generated")
    performance_mode: str = Field(..., description="Performance mode used for the execution")


class SqliteQueryResponse(BaseModel):
    """Response model for the SQLite query API endpoint."""
    result: SqliteQueryResult
    execution_time: float = Field(..., description="Execution time in seconds")
    timestamp: datetime = Field(..., description="Timestamp when the result was generated")
    performance_mode: str = Field(..., description="Performance mode used for the execution")


class HtmlProcessResponse(BaseModel):
    """Response model for the HTML process API endpoint."""
    result: HtmlProcessResult
    execution_time: float = Field(..., description="Execution time in seconds")
    timestamp: datetime = Field(..., description="Timestamp when the result was generated")
    performance_mode: str = Field(..., description="Performance mode used for the execution")


class PdfExtractResponse(BaseModel):
    """Response model for the PDF extraction API endpoint."""
    result: PdfExtractResult
    execution_time: float = Field(..., description="Execution time in seconds")
    timestamp: datetime = Field(..., description="Timestamp when the result was generated")
    performance_mode: str = Field(..., description="Performance mode used for the execution")


class TableProcessResponse(BaseModel):
    """Response model for the table processing API endpoint."""
    result: TableProcessResult
    execution_time: float = Field(..., description="Execution time in seconds")
    timestamp: datetime = Field(..., description="Timestamp when the result was generated")
    performance_mode: str = Field(..., description="Performance mode used for the execution")


class CoreJobStatusResponse(BaseModel):
    """Response model for job status API endpoint."""
    job_id: str = Field(..., description="ID of the job")
    status: str = Field(..., description="Current status of the job")
    submitted_at: datetime = Field(..., description="Timestamp when the job was submitted")
    started_at: Optional[datetime] = Field(None, description="Timestamp when the job started execution")
    completed_at: Optional[datetime] = Field(None, description="Timestamp when the job completed")
    progress: Optional[float] = Field(None, description="Job progress as a percentage")
    message: Optional[str] = Field(None, description="Additional status message")
    
    @validator('status')
    def validate_status(cls, v):
        """Validate that the status is a known value."""
        allowed_statuses = {'pending', 'running', 'completed', 'failed', 'canceled'}
        if v not in allowed_statuses:
            raise ValueError(f"Status must be one of: {', '.join(allowed_statuses)}")
        return v