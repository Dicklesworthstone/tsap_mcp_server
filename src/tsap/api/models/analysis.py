"""
API models for analysis tool endpoints in the TSAP MCP Server API.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime

from tsap.mcp.models import (
    CodeAnalyzerParams,
    CodeAnalyzerResult,
    DocumentExplorerParams,
    DocumentExplorerResult,
    CorpusCartographerParams,
    CorpusCartographerResult,
    CounterfactualAnalyzerParams,
    CounterfactualAnalyzerResult,
    StrategyCompilerParams,
    StrategyCompilerResult
)


# API Request models - extend the MCP models with additional API-specific parameters

class CodeAnalysisRequest(BaseModel):
    """API request model for code analysis."""
    params: CodeAnalyzerParams = Field(..., description="Code analyzer parameters")
    performance_mode: Optional[str] = Field(None, description="Performance mode override")
    async_execution: Optional[bool] = Field(False, description="Execute asynchronously and return a job ID")
    include_file_content: Optional[bool] = Field(False, description="Include file content in results")
    highlight_matches: Optional[bool] = Field(True, description="Highlight matches in file content")
    max_results_per_category: Optional[int] = Field(None, description="Maximum results to return per category")


class DocumentExplorationRequest(BaseModel):
    """API request model for document exploration."""
    params: DocumentExplorerParams = Field(..., description="Document explorer parameters")
    performance_mode: Optional[str] = Field(None, description="Performance mode override")
    async_execution: Optional[bool] = Field(False, description="Execute asynchronously and return a job ID")
    include_previews: Optional[bool] = Field(True, description="Include document previews in results")
    preview_length: Optional[int] = Field(200, description="Maximum length of document previews")
    generate_thumbnails: Optional[bool] = Field(False, description="Generate thumbnails for documents")


class CorpusCartographyRequest(BaseModel):
    """API request model for corpus cartography."""
    params: CorpusCartographerParams = Field(..., description="Corpus cartographer parameters")
    performance_mode: Optional[str] = Field(None, description="Performance mode override")
    async_execution: Optional[bool] = Field(False, description="Execute asynchronously and return a job ID")
    include_previews: Optional[bool] = Field(False, description="Include document previews in results")
    generate_visualization: Optional[bool] = Field(True, description="Generate a visualization of the corpus map")
    visualization_format: Optional[str] = Field("json", description="Format for the visualization (json, d3, graphviz)")


class CounterfactualAnalysisRequest(BaseModel):
    """API request model for counterfactual analysis."""
    params: CounterfactualAnalyzerParams = Field(..., description="Counterfactual analyzer parameters")
    performance_mode: Optional[str] = Field(None, description="Performance mode override")
    async_execution: Optional[bool] = Field(False, description="Execute asynchronously and return a job ID")
    include_context: Optional[bool] = Field(True, description="Include context for each counterfactual")
    context_lines: Optional[int] = Field(2, description="Number of context lines to include")
    filter_confidence: Optional[float] = Field(0.5, description="Minimum confidence threshold for counterfactuals")


class StrategyCompilationRequest(BaseModel):
    """API request model for strategy compilation."""
    params: StrategyCompilerParams = Field(..., description="Strategy compiler parameters")
    performance_mode: Optional[str] = Field(None, description="Performance mode override")
    include_explanations: Optional[bool] = Field(True, description="Include explanations for each strategy step")
    test_strategy: Optional[bool] = Field(False, description="Test the compiled strategy with sample data")
    optimize_strategy: Optional[bool] = Field(True, description="Optimize the compiled strategy for performance")


# API Response models - extend or wrap the MCP result models

class AsyncJobResponse(BaseModel):
    """Response model for asynchronous job submission."""
    job_id: str = Field(..., description="Asynchronous job ID")
    status: str = Field("submitted", description="Job status")
    submitted_at: datetime = Field(default_factory=datetime.now, description="Job submission time")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    status_url: str = Field(..., description="URL to check job status")


class CodeAnalysisResponse(BaseModel):
    """API response model for code analysis."""
    result: CodeAnalyzerResult = Field(..., description="Code analysis result")
    execution_time: float = Field(..., description="Execution time in seconds")
    performance_mode: str = Field(..., description="Performance mode used")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    filtered_results: Optional[bool] = Field(None, description="Whether results were filtered")


class DocumentExplorationResponse(BaseModel):
    """API response model for document exploration."""
    result: DocumentExplorerResult = Field(..., description="Document exploration result")
    execution_time: float = Field(..., description="Execution time in seconds")
    performance_mode: str = Field(..., description="Performance mode used")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    previews_included: bool = Field(..., description="Whether previews were included")
    thumbnails_included: bool = Field(..., description="Whether thumbnails were included")


class CorpusCartographyResponse(BaseModel):
    """API response model for corpus cartography."""
    result: CorpusCartographerResult = Field(..., description="Corpus cartography result")
    execution_time: float = Field(..., description="Execution time in seconds")
    performance_mode: str = Field(..., description="Performance mode used")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    visualization: Optional[Dict[str, Any]] = Field(None, description="Visualization data if requested")
    visualization_format: Optional[str] = Field(None, description="Format of the visualization")


class CounterfactualAnalysisResponse(BaseModel):
    """API response model for counterfactual analysis."""
    result: CounterfactualAnalyzerResult = Field(..., description="Counterfactual analysis result")
    execution_time: float = Field(..., description="Execution time in seconds")
    performance_mode: str = Field(..., description="Performance mode used")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    filtered_by_confidence: Optional[float] = Field(None, description="Confidence threshold applied")


class StrategyCompilationResponse(BaseModel):
    """API response model for strategy compilation."""
    result: StrategyCompilerResult = Field(..., description="Strategy compilation result")
    execution_time: float = Field(..., description="Execution time in seconds")
    performance_mode: str = Field(..., description="Performance mode used")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    optimized: bool = Field(..., description="Whether the strategy was optimized")
    test_results: Optional[Dict[str, Any]] = Field(None, description="Results of strategy testing")


class AnalysisJobStatusResponse(BaseModel):
    """Response model for checking analysis job status."""
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