"""
API routes for composite operations in the TSAP MCP Server API.
"""

import asyncio
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Path, status

from tsap.utils.logging import logger
from tsap.api.dependencies import api_key_dependency, performance_mode_dependency
from tsap.api.models.composite import (
    ParallelSearchRequest,
    ParallelSearchResponse,
    ContextExtractionRequest,
    ContextExtractionResponse,
    RegexGenerationRequest,
    RegexGenerationResponse,
    StructureSearchRequest,
    StructureSearchResponse,
    DiffGenerationRequest,
    DiffGenerationResponse,
    FilenamePatternRequest,
    FilenamePatternResponse,
    DocumentProfileRequest,
    DocumentProfileResponse,
    BatchProfileRequest,
    BatchProfileResponse,
    AsyncJobResponse,
    CompositeJobStatusResponse
)

# Create the router
router = APIRouter()

# In-memory store for async jobs (would use Redis or similar in production)
_async_jobs: Dict[str, Dict[str, Any]] = {}


# Simulate background task execution (in a real system, this would be done with Celery, etc.)
async def _run_composite_job(job_id: str, operation_type: str, params: Dict[str, Any]):
    """
    Run a composite operation job in the background.
    
    Args:
        job_id: Job ID
        operation_type: Type of operation to run
        params: Operation parameters
    """
    try:
        # Update job status
        _async_jobs[job_id]["status"] = "running"
        _async_jobs[job_id]["started_at"] = datetime.now()
        
        # Simulate processing time
        total_steps = 10
        for i in range(total_steps):
            # Update progress
            _async_jobs[job_id]["progress"] = (i / total_steps) * 100
            
            # Simulate work
            await asyncio.sleep(0.5)
        
        # In a real implementation, this would call the actual composite operation functions
        # For now, we'll just create a placeholder result
        if operation_type == "parallel_search":
            from tsap.composite.parallel import parallel_search
            result = await parallel_search(params)
        elif operation_type == "context_extract":
            from tsap.composite.context import extract_context
            result = await extract_context(params)
        elif operation_type == "regex_generator":
            from tsap.composite.regex_generator import generate_regex
            result = await generate_regex(params)
        elif operation_type == "structure_search":
            from tsap.composite.structure_search import structure_search
            result = await structure_search(params)
        elif operation_type == "diff_generator":
            from tsap.composite.diff_generator import generate_diff
            result = await generate_diff(params)
        elif operation_type == "filename_patterns":
            from tsap.composite.filenames import discover_filename_patterns
            result = await discover_filename_patterns(params)
        elif operation_type == "document_profile":
            from tsap.composite.document_profiler import profile_document
            result = await profile_document(params)
        elif operation_type == "batch_profile":
            from tsap.composite.document_profiler import profile_documents
            result = await profile_documents(params["document_paths"], params["include_content_features"], params["compare_documents"])
        else:
            raise ValueError(f"Unknown operation type: {operation_type}")
        
        # Complete the job
        _async_jobs[job_id]["status"] = "completed"
        _async_jobs[job_id]["completed_at"] = datetime.now()
        _async_jobs[job_id]["progress"] = 100
        _async_jobs[job_id]["result"] = result
        
    except Exception as e:
        # Handle errors
        logger.error(
            f"Error in async composite job {job_id}: {str(e)}",
            component="composite",
            operation="async_job"
        )
        _async_jobs[job_id]["status"] = "failed"
        _async_jobs[job_id]["completed_at"] = datetime.now()
        _async_jobs[job_id]["error"] = str(e)


@router.post("/search", response_model=ParallelSearchResponse, status_code=status.HTTP_200_OK)
async def parallel_search(
    request: ParallelSearchRequest,
    background_tasks: BackgroundTasks,
    api_key: str = api_key_dependency,
    performance_mode: str = Depends(performance_mode_dependency)
):
    """
    Run multiple search patterns simultaneously.
    
    This endpoint enables parallel execution of multiple search patterns
    with optimization strategies and consolidated results.
    """
    # Handle async execution if requested
    if request.async_execution:
        job_id = str(uuid.uuid4())
        
        # Store job information
        _async_jobs[job_id] = {
            "job_id": job_id,
            "type": "parallel_search",
            "status": "pending",
            "submitted_at": datetime.now(),
            "estimated_completion": datetime.now() + timedelta(seconds=10),
            "progress": 0,
            "params": request.params.dict()
        }
        
        # Start background task
        background_tasks.add_task(_run_composite_job, job_id, "parallel_search", request.params.dict())
        
        # Return job information
        return AsyncJobResponse(
            job_id=job_id,
            status="submitted",
            submitted_at=_async_jobs[job_id]["submitted_at"],
            estimated_completion=_async_jobs[job_id]["estimated_completion"],
            status_url=f"/api/composite/jobs/{job_id}"
        )
    
    # Synchronous execution
    try:
        # Set performance mode from request if provided
        if request.performance_mode:
            performance_mode = request.performance_mode
        
        start_time = time.time()
        
        # Call the composite operation
        from tsap.composite.parallel import parallel_search
        result = await parallel_search(request.params)
        
        execution_time = time.time() - start_time
        
        return ParallelSearchResponse(
            result=result,
            execution_time=execution_time,
            performance_mode=performance_mode,
            warnings=[],
            filtered_results=request.max_results is not None,
            file_content_included=request.include_file_content
        )
    except Exception as e:
        logger.error(
            f"Error in parallel search: {str(e)}",
            component="api",
            operation="parallel_search"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Parallel search failed: {str(e)}"
        )


@router.post("/context", response_model=ContextExtractionResponse, status_code=status.HTTP_200_OK)
async def extract_context(
    request: ContextExtractionRequest,
    background_tasks: BackgroundTasks,
    api_key: str = api_key_dependency,
    performance_mode: str = Depends(performance_mode_dependency)
):
    """
    Extract context around matches.
    
    This endpoint extracts meaningful code or text units around search matches,
    with awareness of language structures and contextual boundaries.
    """
    try:
        # Set performance mode from request if provided
        if request.performance_mode:
            performance_mode = request.performance_mode
        
        start_time = time.time()
        
        # Call the composite operation
        from tsap.composite.context import extract_context
        result = await extract_context(request.params)
        
        execution_time = time.time() - start_time
        
        return ContextExtractionResponse(
            result=result,
            execution_time=execution_time,
            performance_mode=performance_mode,
            warnings=[],
            contexts_truncated=request.max_context_size is not None,
            file_info_included=request.include_file_info
        )
    except Exception as e:
        logger.error(
            f"Error in context extraction: {str(e)}",
            component="api",
            operation="context_extract"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Context extraction failed: {str(e)}"
        )


@router.post("/regex", response_model=RegexGenerationResponse, status_code=status.HTTP_200_OK)
async def generate_regex(
    request: RegexGenerationRequest,
    background_tasks: BackgroundTasks,
    api_key: str = api_key_dependency,
    performance_mode: str = Depends(performance_mode_dependency)
):
    """
    Generate regex patterns from examples.
    
    This endpoint automatically generates optimal regular expressions
    for matching patterns based on positive and negative examples.
    """
    try:
        # Set performance mode from request if provided
        if request.performance_mode:
            performance_mode = request.performance_mode
        
        start_time = time.time()
        
        # Call the composite operation
        from tsap.composite.regex_generator import generate_regex
        result = await generate_regex(request.params)
        
        execution_time = time.time() - start_time
        
        # Create test results if requested
        test_results = None
        if request.test_on_examples and request.params.positive_examples:
            import re
            
            # Test best regex on examples
            test_results = {"matches": [], "non_matches": []}
            try:
                regex = re.compile(result.best_regex)
                
                # Test on positive examples
                for example in request.params.positive_examples:
                    match = regex.fullmatch(example)
                    if match:
                        test_results["matches"].append(example)
                    else:
                        test_results["non_matches"].append(example)
                
                # Test on negative examples
                if request.params.negative_examples:
                    for example in request.params.negative_examples:
                        match = regex.fullmatch(example)
                        if match:
                            test_results["false_positives"] = test_results.get("false_positives", []) + [example]
            except re.error:
                test_results["error"] = "Invalid regex pattern"
        
        return RegexGenerationResponse(
            result=result,
            execution_time=execution_time,
            performance_mode=performance_mode,
            warnings=[],
            test_results=test_results,
            visualization=None  # Would generate visualization in a real implementation
        )
    except Exception as e:
        logger.error(
            f"Error in regex generation: {str(e)}",
            component="api",
            operation="regex_generator"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Regex generation failed: {str(e)}"
        )


@router.post("/structure_search", response_model=StructureSearchResponse, status_code=status.HTTP_200_OK)
async def structure_search(
    request: StructureSearchRequest,
    background_tasks: BackgroundTasks,
    api_key: str = api_key_dependency,
    performance_mode: str = Depends(performance_mode_dependency)
):
    """
    Search based on document structure.
    
    This endpoint enables searching based on structural position and context,
    not just content, with awareness of code structures, document sections, etc.
    """
    # Handle async execution if requested
    if request.async_execution:
        job_id = str(uuid.uuid4())
        
        # Store job information
        _async_jobs[job_id] = {
            "job_id": job_id,
            "type": "structure_search",
            "status": "pending",
            "submitted_at": datetime.now(),
            "estimated_completion": datetime.now() + timedelta(seconds=15),
            "progress": 0,
            "params": request.params.dict()
        }
        
        # Start background task
        background_tasks.add_task(_run_composite_job, job_id, "structure_search", request.params.dict())
        
        # Return job information
        return AsyncJobResponse(
            job_id=job_id,
            status="submitted",
            submitted_at=_async_jobs[job_id]["submitted_at"],
            estimated_completion=_async_jobs[job_id]["estimated_completion"],
            status_url=f"/api/composite/jobs/{job_id}"
        )
    
    # Synchronous execution
    try:
        # Set performance mode from request if provided
        if request.performance_mode:
            performance_mode = request.performance_mode
        
        start_time = time.time()
        
        # Call the composite operation
        from tsap.composite.structure_search import structure_search
        result = await structure_search(request.params)
        
        execution_time = time.time() - start_time
        
        return StructureSearchResponse(
            result=result,
            execution_time=execution_time,
            performance_mode=performance_mode,
            warnings=[],
            filtered_results=request.max_results is not None,
            file_content_included=request.include_file_content,
            parent_context_included=request.include_parent_context
        )
    except Exception as e:
        logger.error(
            f"Error in structure search: {str(e)}",
            component="api",
            operation="structure_search"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Structure search failed: {str(e)}"
        )


@router.post("/diff", response_model=DiffGenerationResponse, status_code=status.HTTP_200_OK)
async def generate_diff(
    request: DiffGenerationRequest,
    background_tasks: BackgroundTasks,
    api_key: str = api_key_dependency,
    performance_mode: str = Depends(performance_mode_dependency)
):
    """
    Generate diffs between files or versions.
    
    This endpoint finds meaningful changes between texts or documents,
    with awareness of structural elements and semantic changes.
    """
    try:
        # Set performance mode from request if provided
        if request.performance_mode:
            performance_mode = request.performance_mode
        
        start_time = time.time()
        
        # Set additional parameters from the request
        params_dict = request.params.dict()
        params_dict["highlight_changes"] = request.highlight_changes
        if request.max_diff_size is not None:
            params_dict["max_diff_size"] = request.max_diff_size
        
        # Call the composite operation
        from tsap.composite.diff_generator import generate_diff
        from tsap.mcp.models import DiffGeneratorParams
        result = await generate_diff(DiffGeneratorParams(**params_dict))
        
        execution_time = time.time() - start_time
        
        # Format as HTML if requested
        html_formatted = False
        if request.format_as_html:
            # Placeholder for HTML formatting
            # In a real implementation, this would convert the diff to HTML
            html_formatted = True
        
        # Check if diff was truncated
        diff_truncated = False
        if request.max_diff_size is not None:
            for chunk in result.diff_chunks:
                if len(chunk.lines) == request.max_diff_size:
                    diff_truncated = True
                    break
        
        return DiffGenerationResponse(
            result=result,
            execution_time=execution_time,
            performance_mode=performance_mode,
            warnings=["Diff output was truncated"] if diff_truncated else [],
            file_content_included=request.include_file_content,
            html_formatted=html_formatted,
            diff_truncated=diff_truncated
        )
    except Exception as e:
        logger.error(
            f"Error in diff generation: {str(e)}",
            component="api",
            operation="diff_generator"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Diff generation failed: {str(e)}"
        )


@router.post("/filename_patterns", response_model=FilenamePatternResponse, status_code=status.HTTP_200_OK)
async def discover_filename_patterns(
    request: FilenamePatternRequest,
    background_tasks: BackgroundTasks,
    api_key: str = api_key_dependency,
    performance_mode: str = Depends(performance_mode_dependency)
):
    """
    Discover patterns in filenames.
    
    This endpoint finds patterns and conventions in filenames within a directory
    structure, including prefixes, suffixes, and organization schemes.
    """
    try:
        # Set performance mode from request if provided
        if request.performance_mode:
            performance_mode = request.performance_mode
        
        start_time = time.time()
        
        # Call the composite operation
        from tsap.composite.filenames import discover_filename_patterns
        result = await discover_filename_patterns(request.params)
        
        execution_time = time.time() - start_time
        
        # Count patterns
        pattern_count = 0
        if "naming_patterns" in result.naming_patterns:
            pattern_candidates = result.naming_patterns.get("pattern_candidates", [])
            pattern_count = len(pattern_candidates)
        
        return FilenamePatternResponse(
            result=result,
            execution_time=execution_time,
            performance_mode=performance_mode,
            timestamp=datetime.now(),
            warnings=[],
            pattern_count=pattern_count,
            examples_included=request.include_examples,
            regex_generated=request.generate_regex
        )
    except Exception as e:
        logger.error(
            f"Error in filename pattern discovery: {str(e)}",
            component="api",
            operation="filename_patterns"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Filename pattern discovery failed: {str(e)}"
        )


@router.post("/profile", response_model=DocumentProfileResponse, status_code=status.HTTP_200_OK)
async def profile_document(
    request: DocumentProfileRequest,
    background_tasks: BackgroundTasks,
    api_key: str = api_key_dependency,
    performance_mode: str = Depends(performance_mode_dependency)
):
    """
    Create a profile of a document.
    
    This endpoint creates a structural and content-based profile of a document
    for comparison, classification, or search purposes.
    """
    try:
        # Set performance mode from request if provided
        if request.performance_mode:
            performance_mode = request.performance_mode
        
        start_time = time.time()
        
        # Call the composite operation
        from tsap.composite.document_profiler import profile_document
        from tsap.mcp.models import DocumentProfilerParams
        
        params = DocumentProfilerParams(
            document_path=request.document_path,
            include_content_features=request.include_content_features,
            reference_profiles=request.reference_profiles
        )
        
        result = await profile_document(params)
        
        execution_time = time.time() - start_time
        
        return DocumentProfileResponse(
            document_path=request.document_path,
            profile=result.profile,
            comparisons=result.comparisons,
            execution_time=execution_time,
            performance_mode=performance_mode,
            timestamp=datetime.now(),
            warnings=[],
            metadata_included=request.include_metadata,
            structure_included=request.include_structure,
            preview_included=request.include_preview,
            error=result.error
        )
    except Exception as e:
        logger.error(
            f"Error in document profiling: {str(e)}",
            component="api",
            operation="document_profile"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document profiling failed: {str(e)}"
        )


@router.post("/batch_profile", response_model=BatchProfileResponse, status_code=status.HTTP_200_OK)
async def batch_profile_documents(
    request: BatchProfileRequest,
    background_tasks: BackgroundTasks,
    api_key: str = api_key_dependency,
    performance_mode: str = Depends(performance_mode_dependency)
):
    """
    Profile multiple documents and compare them.
    
    This endpoint creates profiles for multiple documents and optionally
    compares them to identify similarities and differences.
    """
    # Handle large batches
    if request.max_documents is not None and len(request.document_paths) > request.max_documents:
        request.document_paths = request.document_paths[:request.max_documents]
    
    try:
        # Set performance mode from request if provided
        if request.performance_mode:
            performance_mode = request.performance_mode
        
        start_time = time.time()
        
        # Call the composite operation
        from tsap.composite.document_profiler import profile_documents
        
        result = await profile_documents(
            request.document_paths,
            request.include_content_features,
            request.compare_documents
        )
        
        execution_time = time.time() - start_time
        
        # Add clustering if requested
        clustering = None
        if request.cluster_documents and "profiles" in result:
            # Placeholder for clustering implementation
            # In a real implementation, this would cluster documents by similarity
            clustering = {
                "clusters": [
                    {
                        "id": "cluster1",
                        "name": "Cluster 1",
                        "documents": list(result["profiles"].keys())[:len(result["profiles"])//2],
                        "centroid": "Document characteristics"
                    },
                    {
                        "id": "cluster2",
                        "name": "Cluster 2",
                        "documents": list(result["profiles"].keys())[len(result["profiles"])//2:],
                        "centroid": "Document characteristics"
                    }
                ],
                "method": "placeholder",
                "silhouette_score": 0.75
            }
        
        return BatchProfileResponse(
            profiles=result.get("profiles", {}),
            comparisons=result.get("comparisons"),
            clustering=clustering,
            execution_time=execution_time,
            performance_mode=performance_mode,
            timestamp=datetime.now(),
            warnings=[],
            documents_processed=len(request.document_paths),
            error=None
        )
    except Exception as e:
        logger.error(
            f"Error in batch document profiling: {str(e)}",
            component="api",
            operation="batch_profile"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch document profiling failed: {str(e)}"
        )


@router.get("/jobs/{job_id}", response_model=CompositeJobStatusResponse, status_code=status.HTTP_200_OK)
async def get_job_status(
    job_id: str = Path(..., description="Job ID"),
    api_key: str = api_key_dependency
):
    """
    Get the status of an asynchronous composite job.
    
    Args:
        job_id: ID of the job to check
        
    Returns:
        Job status information
    """
    # Check if job exists
    if job_id not in _async_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID {job_id} not found"
        )
    
    # Get job information
    job = _async_jobs[job_id]
    
    # Create response
    response = CompositeJobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        submitted_at=job["submitted_at"],
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        progress=job.get("progress"),
        error=job.get("error")
    )
    
    # Add result URL if completed
    if job["status"] == "completed":
        response.result_url = f"/api/composite/jobs/{job_id}/result"
    
    return response


@router.get("/jobs/{job_id}/result", status_code=status.HTTP_200_OK)
async def get_job_result(
    job_id: str = Path(..., description="Job ID"),
    api_key: str = api_key_dependency
):
    """
    Get the result of a completed asynchronous composite job.
    
    Args:
        job_id: ID of the job to get results for
        
    Returns:
        Job results
    """
    # Check if job exists
    if job_id not in _async_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID {job_id} not found"
        )
    
    # Get job information
    job = _async_jobs[job_id]
    
    # Check if job is completed
    if job["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job with ID {job_id} is not completed (status: {job['status']})"
        )
    
    # Check if result exists
    if "result" not in job:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"No result found for job with ID {job_id}"
        )
    
    # Return the result
    return job["result"]


@router.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_job(
    job_id: str = Path(..., description="Job ID"),
    api_key: str = api_key_dependency
):
    """
    Cancel an asynchronous composite job.
    
    Args:
        job_id: ID of the job to cancel
    """
    # Check if job exists
    if job_id not in _async_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID {job_id} not found"
        )
    
    # Get job information
    job = _async_jobs[job_id]
    
    # Check if job can be cancelled
    if job["status"] not in ["pending", "running"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job with ID {job_id} cannot be cancelled (status: {job['status']})"
        )
    
    # Cancel the job
    job["status"] = "cancelled"
    job["completed_at"] = datetime.now()
    
    # No content response
    return None