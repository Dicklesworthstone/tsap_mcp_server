"""
API models for composite operation endpoints in the TSAP MCP Server API.
"""

from typing import Dict, List, Any, Optional, Union, Set
from pydantic import BaseModel, Field, validator, root_validator
from datetime import datetime

from tsap.mcp.models import (
    ParallelSearchParams,
    ParallelSearchResult,
    ContextExtractParams,
    ContextExtractResult,
    RegexGeneratorParams,
    RegexGeneratorResult,
    StructureSearchParams,
    StructureSearchResult,
    DiffGeneratorParams,
    DiffGeneratorResult,
    FilenamePatternParams,
    FilenamePatternResult
)


# API Request models - extend the MCP models with additional API-specific parameters

class ParallelSearchRequest(BaseModel):
    """API request model for parallel search."""
    params: ParallelSearchParams = Field(..., description="Parallel search parameters")
    performance_mode: Optional[str] = Field(None, description="Performance mode override")
    async_execution: Optional[bool] = Field(False, description="Execute asynchronously and return a job ID")
    include_file_content: Optional[bool] = Field(False, description="Include file content in results")
    highlight_matches: Optional[bool] = Field(True, description="Highlight matches in file content")
    consolidate_overlapping: Optional[bool] = Field(True, description="Consolidate overlapping matches")
    max_results: Optional[int] = Field(None, description="Maximum results to return")


class ContextExtractionRequest(BaseModel):
    """API request model for context extraction."""
    params: ContextExtractParams = Field(..., description="Context extraction parameters")
    performance_mode: Optional[str] = Field(None, description="Performance mode override")
    highlight_matches: Optional[bool] = Field(True, description="Highlight matches in extracted context")
    include_file_info: Optional[bool] = Field(True, description="Include file metadata")
    max_context_size: Optional[int] = Field(None, description="Maximum size of extracted context in characters")
    format_output: Optional[bool] = Field(True, description="Format output for readability")


class RegexGenerationRequest(BaseModel):
    """API request model for regex generation."""
    params: RegexGeneratorParams = Field(..., description="Regex generator parameters")
    performance_mode: Optional[str] = Field(None, description="Performance mode override")
    num_candidates: Optional[int] = Field(5, description="Number of regex candidates to generate")
    include_explanations: Optional[bool] = Field(True, description="Include explanations of regex patterns")
    test_on_examples: Optional[bool] = Field(True, description="Test generated regex on examples")
    include_visualization: Optional[bool] = Field(False, description="Include regex visualization")


class StructureSearchRequest(BaseModel):
    """API request model for structure search."""
    params: StructureSearchParams = Field(..., description="Structure search parameters")
    performance_mode: Optional[str] = Field(None, description="Performance mode override")
    async_execution: Optional[bool] = Field(False, description="Execute asynchronously and return a job ID")
    include_file_content: Optional[bool] = Field(False, description="Include file content in results")
    highlight_matches: Optional[bool] = Field(True, description="Highlight matches in file content")
    include_parent_context: Optional[bool] = Field(True, description="Include context from parent elements")
    max_results: Optional[int] = Field(None, description="Maximum results to return")


class DiffGenerationRequest(BaseModel):
    """API request model for diff generation."""
    params: DiffGeneratorParams = Field(..., description="Diff generator parameters")
    performance_mode: Optional[str] = Field(None, description="Performance mode override")
    include_file_content: Optional[bool] = Field(False, description="Include file content in results")
    highlight_changes: Optional[bool] = Field(True, description="Highlight changes in diff output")
    format_as_html: Optional[bool] = Field(False, description="Format diff as HTML")
    include_unchanged_sections: Optional[bool] = Field(False, description="Include unchanged sections in output")
    max_diff_size: Optional[int] = Field(None, description="Maximum size of diff output in lines")


class FilenamePatternRequest(BaseModel):
    """API request model for filename pattern discovery."""
    params: FilenamePatternParams = Field(..., description="Filename pattern parameters")
    performance_mode: Optional[str] = Field(None, description="Performance mode override")
    max_patterns: Optional[int] = Field(10, description="Maximum number of patterns to discover")
    min_confidence: Optional[float] = Field(0.5, description="Minimum confidence threshold for patterns")
    include_examples: Optional[bool] = Field(True, description="Include example files for each pattern")
    generate_regex: Optional[bool] = Field(True, description="Generate regex for discovered patterns")


class DocumentProfileRequest(BaseModel):
    """API request model for document profiling."""
    document_path: str = Field(..., description="Path to the document")
    include_content_features: bool = Field(True, description="Include content features in profile")
    reference_profiles: Optional[Dict[str, Any]] = Field(None, description="Reference profiles for comparison")
    performance_mode: Optional[str] = Field(None, description="Performance mode override")
    include_metadata: Optional[bool] = Field(True, description="Include document metadata")
    include_structure: Optional[bool] = Field(True, description="Include document structure")
    include_preview: Optional[bool] = Field(False, description="Include document preview")


class BatchProfileRequest(BaseModel):
    """API request model for batch document profiling."""
    document_paths: List[str] = Field(..., description="Paths to documents")
    include_content_features: bool = Field(True, description="Include content features in profiles")
    compare_documents: bool = Field(True, description="Compare documents with each other")
    performance_mode: Optional[str] = Field(None, description="Performance mode override")
    cluster_documents: Optional[bool] = Field(False, description="Cluster documents by similarity")
    max_documents: Optional[int] = Field(None, description="Maximum number of documents to process")


# API Response models - extend or wrap the MCP result models

class AsyncJobResponse(BaseModel):
    """Response model for asynchronous job submission."""
    job_id: str = Field(..., description="Asynchronous job ID")
    status: str = Field("submitted", description="Job status")
    submitted_at: datetime = Field(default_factory=datetime.now, description="Job submission time")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    status_url: str = Field(..., description="URL to check job status")


class ParallelSearchResponse(BaseModel):
    """API response model for parallel search."""
    result: ParallelSearchResult = Field(..., description="Parallel search result")
    execution_time: float = Field(..., description="Execution time in seconds")
    performance_mode: str = Field(..., description="Performance mode used")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    filtered_results: Optional[bool] = Field(None, description="Whether results were filtered")
    file_content_included: bool = Field(False, description="Whether file content was included")


class ContextExtractionResponse(BaseModel):
    """API response model for context extraction."""
    result: ContextExtractResult = Field(..., description="Context extraction result")
    execution_time: float = Field(..., description="Execution time in seconds")
    performance_mode: str = Field(..., description="Performance mode used")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    contexts_truncated: bool = Field(False, description="Whether contexts were truncated")
    file_info_included: bool = Field(False, description="Whether file info was included")


class RegexGenerationResponse(BaseModel):
    """API response model for regex generation."""
    result: RegexGeneratorResult = Field(..., description="Regex generator result")
    execution_time: float = Field(..., description="Execution time in seconds")
    performance_mode: str = Field(..., description="Performance mode used")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    test_results: Optional[Dict[str, Any]] = Field(None, description="Test results on examples")
    visualization: Optional[Dict[str, Any]] = Field(None, description="Regex visualization data")


class StructureSearchResponse(BaseModel):
    """API response model for structure search."""
    result: StructureSearchResult = Field(..., description="Structure search result")
    execution_time: float = Field(..., description="Execution time in seconds")
    performance_mode: str = Field(..., description="Performance mode used")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    filtered_results: Optional[bool] = Field(None, description="Whether results were filtered")
    file_content_included: bool = Field(False, description="Whether file content was included")
    parent_context_included: bool = Field(False, description="Whether parent context was included")


class DiffGenerationResponse(BaseModel):
    """API response model for diff generation."""
    result: DiffGeneratorResult = Field(..., description="Diff generator result")
    execution_time: float = Field(..., description="Execution time in seconds")
    performance_mode: str = Field(..., description="Performance mode used")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    file_content_included: bool = Field(False, description="Whether file content was included")
    html_formatted: bool = Field(False, description="Whether diff was formatted as HTML")
    diff_truncated: bool = Field(False, description="Whether diff was truncated")


class FilenamePatternResponse(BaseModel):
    """API response model for filename pattern discovery."""
    result: FilenamePatternResult = Field(..., description="Filename pattern result")
    execution_time: float = Field(..., description="Execution time in seconds")
    performance_mode: str = Field(..., description="Performance mode used")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    pattern_count: int = Field(..., description="Number of patterns discovered")
    examples_included: bool = Field(False, description="Whether examples were included")
    regex_generated: bool = Field(False, description="Whether regex was generated for patterns")


class DocumentProfileResponse(BaseModel):
    """API response model for document profiling."""
    document_path: str = Field(..., description="Path to the document")
    profile: Dict[str, Any] = Field(..., description="Document profile")
    comparisons: Dict[str, Any] = Field(default_factory=dict, description="Comparisons to reference profiles")
    execution_time: float = Field(..., description="Execution time in seconds")
    performance_mode: str = Field(..., description="Performance mode used")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    metadata_included: bool = Field(False, description="Whether metadata was included")
    structure_included: bool = Field(False, description="Whether structure was included")
    preview_included: bool = Field(False, description="Whether preview was included")
    error: Optional[str] = Field(None, description="Error message if profiling failed")


class BatchProfileResponse(BaseModel):
    """API response model for batch document profiling."""
    profiles: Dict[str, Any] = Field(..., description="Document profiles")
    comparisons: Optional[Dict[str, Any]] = Field(None, description="Document comparisons")
    clustering: Optional[Dict[str, Any]] = Field(None, description="Document clustering")
    execution_time: float = Field(..., description="Execution time in seconds")
    performance_mode: str = Field(..., description="Performance mode used")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    documents_processed: int = Field(..., description="Number of documents processed")
    error: Optional[str] = Field(None, description="Error message if profiling failed")


class CompositeJobStatusResponse(BaseModel):
    """Response model for checking composite job status."""
    job_id: str = Field(..., description="Asynchronous job ID")
    status: str = Field(..., description="Job status (pending, running, completed, failed)")
    submitted_at: datetime = Field(..., description="Job submission time")
    started_at: Optional[datetime] = Field(None, description="Job start time")
    completed_at: Optional[datetime] = Field(None, description="Job completion time")
    progress: Optional[float] = Field(None, description="Job progress (0-100%)")
    result_url: Optional[str] = Field(None, description="URL to get results when completed")
    error: Optional[str] = Field(None, description="Error message if failed")
    
    @validator("status")
    def validate_status(cls, v):
        """Validate the job status value."""
        valid_statuses = ["pending", "running", "completed", "failed", "cancelled"]
        if v not in valid_statuses:
            raise ValueError(f"Status must be one of: {', '.join(valid_statuses)}")
        return v