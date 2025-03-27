"""
API routes for core tools in the TSAP MCP Server API.
"""

import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status, Body, File, UploadFile

from tsap.utils.logging import logger
from tsap.api.dependencies import api_key_dependency, performance_mode_dependency
from tsap.mcp.models import (
    RipgrepSearchParams,
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
    PdfExtractResult
)

# Create the router
router = APIRouter()


@router.post("/search", response_model=RipgrepSearchResult, status_code=status.HTTP_200_OK)
async def ripgrep_search(
    params: RipgrepSearchParams = Body(...),
    api_key: str = api_key_dependency,
    performance_mode: str = Depends(performance_mode_dependency)
):
    """
    Search for patterns in files using ripgrep.
    
    This endpoint provides high-performance text searching with 
    regex support through ripgrep.
    """
    try:
        start_time = time.time()
        
        # Call the core function
        from tsap.core.ripgrep import ripgrep_search
        result = await ripgrep_search(params)
        
        execution_time = time.time() - start_time
        
        logger.info(
            f"Ripgrep search completed in {execution_time:.3f}s with {len(result.matches)} matches",
            component="api",
            operation="ripgrep_search"
        )
        
        return result
    except Exception as e:
        logger.error(
            f"Error in ripgrep search: {str(e)}",
            component="api",
            operation="ripgrep_search"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.post("/awk", response_model=AwkProcessResult, status_code=status.HTTP_200_OK)
async def awk_process(
    params: AwkProcessParams = Body(...),
    api_key: str = api_key_dependency,
    performance_mode: str = Depends(performance_mode_dependency)
):
    """
    Process text with AWK.
    
    This endpoint provides text transformation and field processing
    through AWK.
    """
    try:
        start_time = time.time()
        
        # Call the core function
        from tsap.core.awk import awk_process
        result = await awk_process(params)
        
        execution_time = time.time() - start_time
        
        logger.info(
            f"AWK process completed in {execution_time:.3f}s",
            component="api",
            operation="awk_process"
        )
        
        return result
    except Exception as e:
        logger.error(
            f"Error in AWK process: {str(e)}",
            component="api",
            operation="awk_process"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AWK process failed: {str(e)}"
        )


@router.post("/jq", response_model=JqQueryResult, status_code=status.HTTP_200_OK)
async def jq_query(
    params: JqQueryParams = Body(...),
    api_key: str = api_key_dependency,
    performance_mode: str = Depends(performance_mode_dependency)
):
    """
    Query JSON data with jq.
    
    This endpoint provides JSON data processing and transformation
    through jq.
    """
    try:
        start_time = time.time()
        
        # Call the core function
        from tsap.core.jq import jq_query
        result = await jq_query(params)
        
        execution_time = time.time() - start_time
        
        logger.info(
            f"jq query completed in {execution_time:.3f}s",
            component="api",
            operation="jq_query"
        )
        
        return result
    except Exception as e:
        logger.error(
            f"Error in jq query: {str(e)}",
            component="api",
            operation="jq_query"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"jq query failed: {str(e)}"
        )


@router.post("/sqlite", response_model=SqliteQueryResult, status_code=status.HTTP_200_OK)
async def sqlite_query(
    params: SqliteQueryParams = Body(...),
    api_key: str = api_key_dependency,
    performance_mode: str = Depends(performance_mode_dependency)
):
    """
    Query SQLite databases.
    
    This endpoint provides relational queries on extracted or transformed data
    through SQLite.
    """
    try:
        start_time = time.time()
        
        # Call the core function
        from tsap.core.sqlite import sqlite_query
        result = await sqlite_query(params)
        
        execution_time = time.time() - start_time
        
        logger.info(
            f"SQLite query completed in {execution_time:.3f}s",
            component="api",
            operation="sqlite_query"
        )
        
        return result
    except Exception as e:
        logger.error(
            f"Error in SQLite query: {str(e)}",
            component="api",
            operation="sqlite_query"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SQLite query failed: {str(e)}"
        )


@router.post("/html", response_model=HtmlProcessResult, status_code=status.HTTP_200_OK)
async def process_html(
    params: HtmlProcessParams = Body(...),
    api_key: str = api_key_dependency,
    performance_mode: str = Depends(performance_mode_dependency)
):
    """
    Process HTML content.
    
    This endpoint provides intelligent parsing and searching of HTML content
    with structure awareness.
    """
    try:
        start_time = time.time()
        
        # Call the core function
        from tsap.core.html_processor import process_html
        result = await process_html(params)
        
        execution_time = time.time() - start_time
        
        logger.info(
            f"HTML processing completed in {execution_time:.3f}s",
            component="api",
            operation="process_html"
        )
        
        return result
    except Exception as e:
        logger.error(
            f"Error in HTML processing: {str(e)}",
            component="api",
            operation="process_html"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"HTML processing failed: {str(e)}"
        )


@router.post("/pdf", response_model=PdfExtractResult, status_code=status.HTTP_200_OK)
async def extract_pdf(
    params: PdfExtractParams = Body(...),
    api_key: str = api_key_dependency,
    performance_mode: str = Depends(performance_mode_dependency)
):
    """
    Extract content from PDF files.
    
    This endpoint provides native PDF text extraction that works with
    complex layouts.
    """
    try:
        start_time = time.time()
        
        # Call the core function
        from tsap.core.pdf_extractor import extract_pdf_text
        result = await extract_pdf_text(params.path, params.pages, params.password)
        
        execution_time = time.time() - start_time
        
        logger.info(
            f"PDF extraction completed in {execution_time:.3f}s",
            component="api",
            operation="extract_pdf"
        )
        
        # Construct result object
        pdf_result = PdfExtractResult(
            text=result if isinstance(result, str) else None,
            pages=result if isinstance(result, dict) else None,
            metadata={},  # Would be populated in a real implementation
            tables=[],    # Would be populated in a real implementation
            images=[]     # Would be populated in a real implementation
        )
        
        return pdf_result
    except Exception as e:
        logger.error(
            f"Error in PDF extraction: {str(e)}",
            component="api",
            operation="extract_pdf"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PDF extraction failed: {str(e)}"
        )


@router.post("/upload", status_code=status.HTTP_200_OK)
async def upload_file(
    file: UploadFile = File(...),
    process: Optional[str] = Query(None, description="Processing to apply (extract, search)"),
    api_key: str = api_key_dependency
):
    """
    Upload a file for processing.
    
    This endpoint allows uploading files to be processed by other tools.
    """
    try:
        import os
        from tempfile import NamedTemporaryFile
        
        # Create a temporary file
        with NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp:
            # Write the uploaded file to the temporary file
            content = await file.read()
            temp.write(content)
            temp_path = temp.name
        
        logger.info(
            f"File uploaded: {file.filename} ({len(content)} bytes)",
            component="api",
            operation="upload_file"
        )
        
        # Process the file if requested
        result = {"filename": file.filename, "size": len(content), "path": temp_path}
        
        if process == "extract":
            # Extract content based on file type
            ext = os.path.splitext(file.filename)[1].lower()
            
            if ext == ".pdf":
                from tsap.core.pdf_extractor import extract_pdf_text
                text = await extract_pdf_text(temp_path, None, None)
                result["content"] = text
                
            elif ext in [".html", ".htm"]:
                from tsap.core.html_processor import extract_html_text
                text = await extract_html_text(None, None, temp_path)
                result["content"] = text
                
            else:
                # Read as text
                with open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
                    result["content"] = f.read()
        
        elif process == "search":
            # Read file content for searching elsewhere
            result["ready_for_search"] = True
        
        return result
    except Exception as e:
        logger.error(
            f"Error in file upload: {str(e)}",
            component="api",
            operation="upload_file"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File upload failed: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if "temp_path" in locals():
            try:
                os.unlink(temp_path)
            except Exception:
                pass