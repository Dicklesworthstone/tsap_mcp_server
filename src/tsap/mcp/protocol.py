"""
MCP (Model Context Protocol) protocol implementation.

This module defines the core protocol structures for communication between
language models and the TSAP server.
"""
import uuid
from enum import Enum
from typing import Dict, List, Any, Optional
from datetime import datetime

from pydantic import BaseModel, Field, validator, model_validator


class MCPCommandType(str, Enum):
    """Types of MCP commands."""
    
    # Core tool commands
    RIPGREP_SEARCH = "ripgrep_search"
    AWK_PROCESS = "awk_process"
    JQ_QUERY = "jq_query"
    SQLITE_QUERY = "sqlite_query"
    HTML_PROCESS = "html_process"
    PDF_EXTRACT = "pdf_extract"
    TABLE_PROCESS = "table_process"
    
    # Composite operations
    PARALLEL_SEARCH = "parallel_search"
    RECURSIVE_REFINE = "recursive_refine"
    CONTEXT_EXTRACT = "context_extract"
    PATTERN_ANALYZE = "pattern_analyze"
    FILENAME_DISCOVER = "filename_discover"
    STRUCTURE_ANALYZE = "structure_analyze"
    STRUCTURE_SEARCH = "structure_search"
    DIFF_GENERATE = "diff_generate"
    REGEX_GENERATE = "regex_generate"
    DOCUMENT_PROFILE = "document_profile"
    
    # Analysis tools
    CODE_ANALYZE = "code_analyze"
    DOCUMENT_EXPLORE = "document_explore"
    METADATA_EXTRACT = "metadata_extract"
    CORPUS_MAP = "corpus_map"
    COUNTERFACTUAL_ANALYZE = "counterfactual_analyze"
    STRATEGY_COMPILE = "strategy_compile"
    SEMANTIC_SEARCH = "semantic_search"
    
    # Meta commands
    INFO = "info"
    STATUS = "status"
    CANCEL = "cancel"
    LIST_TOOLS = "list_tools"
    LIST_STRATEGIES = "list_strategies"
    LIST_TEMPLATES = "list_templates"


class MCPError(BaseModel):
    """Error information for MCP responses."""
    
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Detailed error information")
    
    class Config:
        frozen = True


class MCPRequest(BaseModel):
    """Model Context Protocol request structure."""
    
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier"
    )
    command: str = Field(..., description="Command to execute")
    args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Command arguments"
    )
    timeout: Optional[float] = Field(
        None,
        description="Request timeout in seconds"
    )
    mode: Optional[str] = Field(
        None,
        description="Performance mode (fast, standard, deep)"
    )
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional context data"
    )
    
    @validator("command")
    def validate_command(cls, v):
        """Validate command name."""
        try:
            MCPCommandType(v)
        except ValueError:
            # Allow custom commands for extensibility
            pass
        return v
    
    @validator("mode")
    def validate_mode(cls, v):
        """Validate performance mode."""
        if v is not None:
            allowed = ["fast", "standard", "deep"]
            if v.lower() not in allowed:
                raise ValueError(f"Performance mode must be one of {allowed}")
            return v.lower()
        return v


class MCPResponse(BaseModel):
    """Model Context Protocol response structure."""
    
    request_id: str = Field(..., description="Request ID from the request")
    status: str = Field(..., description="Response status (success, error)")
    command: str = Field(..., description="Command that was executed")
    data: Optional[Any] = Field(None, description="Response data")
    error: Optional[MCPError] = Field(None, description="Error information if status is 'error'")
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages"
    )
    execution_time: Optional[float] = Field(
        None,
        description="Command execution time in seconds"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Response timestamp (ISO 8601)"
    )
    
    @model_validator(mode='after')
    def validate_status_and_data(cls, values):
        """Validate that status, data, and error are consistent."""
        status = values.status
        data = values.data  # noqa: F841
        error = values.error
        
        if status == "success" and error is not None:
            raise ValueError("Error should be None when status is 'success'")
            
        if status == "error" and error is None:
            raise ValueError("Error is required when status is 'error'")
            
        return values


class RipgrepSearchParams(BaseModel):
    """Parameters for ripgrep search command."""
    
    pattern: str = Field(..., description="Search pattern (regular expression)")
    paths: List[str] = Field(..., description="Paths to search")
    case_sensitive: bool = Field(False, description="Case sensitive search")
    file_types: Optional[List[str]] = Field(
        None,
        description="File extensions to include"
    )
    exclude_file_types: Optional[List[str]] = Field(
        None,
        description="File extensions to exclude"
    )
    max_depth: Optional[int] = Field(
        None,
        description="Maximum directory depth"
    )
    max_count: Optional[int] = Field(
        None,
        description="Maximum matches per file"
    )
    context_lines: int = Field(
        2,
        description="Number of context lines before and after match"
    )
    follow_symlinks: bool = Field(
        False,
        description="Follow symbolic links"
    )


class RipgrepSearchResult(BaseModel):
    """Result of a ripgrep search command."""
    
    path: str = Field(..., description="File path")
    line_number: int = Field(..., description="Line number of match")
    match: str = Field(..., description="Matched line")
    context_before: List[str] = Field(
        default_factory=list,
        description="Lines before match"
    )
    context_after: List[str] = Field(
        default_factory=list,
        description="Lines after match"
    )


class ParallelSearchParams(BaseModel):
    """Parameters for parallel search command."""
    
    patterns: List[Dict[str, Any]] = Field(
        ...,
        description="List of search patterns with metadata"
    )
    paths: List[str] = Field(..., description="Paths to search")
    case_sensitive: bool = Field(False, description="Case sensitive search")
    file_types: Optional[List[str]] = Field(
        None,
        description="File extensions to include"
    )
    max_matches: Optional[int] = Field(
        None,
        description="Maximum total matches to return"
    )


class SearchMatch(BaseModel):
    """A single search match with metadata."""
    
    path: str = Field(..., description="File path")
    line_number: int = Field(..., description="Line number of match")
    match: str = Field(..., description="Matched line")
    pattern: str = Field(..., description="Pattern that matched")
    pattern_description: Optional[str] = Field(
        None,
        description="Description of the pattern"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional match context"
    )


class MCPCommandResult(BaseModel):
    """Generic command result structure."""
    
    command: str = Field(..., description="Command that was executed")
    result: Any = Field(..., description="Command result data")
    execution_time: float = Field(..., description="Execution time in seconds")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Result timestamp (ISO 8601)"
    )


def create_success_response(
    request_id: str,
    command: str,
    data: Any,
    execution_time: Optional[float] = None,
    warnings: Optional[List[str]] = None,
) -> MCPResponse:
    """Create a successful MCP response.
    
    Args:
        request_id: Request ID from the request
        command: Command that was executed
        data: Response data
        execution_time: Command execution time in seconds
        warnings: Warning messages
        
    Returns:
        MCPResponse with success status
    """
    return MCPResponse(
        request_id=request_id,
        status="success",
        command=command,
        data=data,
        execution_time=execution_time,
        warnings=warnings or [],
    )


def create_error_response(
    request_id: str,
    command: str,
    error_code: str,
    error_message: str,
    error_details: Optional[str] = None,
    execution_time: Optional[float] = None,
    warnings: Optional[List[str]] = None,
) -> MCPResponse:
    """Create an error MCP response.
    
    Args:
        request_id: Request ID from the request
        command: Command that was executed
        error_code: Error code
        error_message: Error message
        error_details: Detailed error information
        execution_time: Command execution time in seconds
        warnings: Warning messages
        
    Returns:
        MCPResponse with error status
    """
    return MCPResponse(
        request_id=request_id,
        status="error",
        command=command,
        error=MCPError(
            code=error_code,
            message=error_message,
            details=error_details,
        ),
        execution_time=execution_time,
        warnings=warnings or [],
    )