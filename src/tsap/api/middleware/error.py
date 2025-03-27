"""
Error handling middleware for the TSAP MCP Server API.
"""

import traceback
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from typing import Dict, Any, Optional

from tsap.utils.logging import logger
from tsap.constants import (
    ERROR_VALIDATION,
    ERROR_EXECUTION,
    ERROR_TIMEOUT,
    ERROR_PERMISSION,
    ERROR_NOT_FOUND,
    ERROR_UNSUPPORTED
)


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle FastAPI request validation errors.
    
    Args:
        request: FastAPI request object
        exc: Validation exception
        
    Returns:
        JSON response with error details
    """
    # Format error message
    errors = []
    for error in exc.errors():
        loc = " > ".join([str(l) for l in error.get("loc", [])])
        msg = error.get("msg", "")
        errors.append(f"{loc}: {msg}")
    
    error_message = "Validation error"
    if errors:
        error_message += ": " + "; ".join(errors)
    
    # Log the error
    logger.warning(
        f"Request validation error: {error_message}",
        component="api",
        operation="validation",
        context={"errors": exc.errors(), "path": request.url.path}
    )
    
    # Return a consistent error response
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": ERROR_VALIDATION,
                "message": error_message,
                "details": exc.errors()
            }
        }
    )


async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    """
    Handle Pydantic validation errors.
    
    Args:
        request: FastAPI request object
        exc: Validation exception
        
    Returns:
        JSON response with error details
    """
    # Format error message
    errors = []
    for error in exc.errors():
        loc = " > ".join([str(l) for l in error.get("loc", [])])
        msg = error.get("msg", "")
        errors.append(f"{loc}: {msg}")
    
    error_message = "Validation error"
    if errors:
        error_message += ": " + "; ".join(errors)
    
    # Log the error
    logger.warning(
        f"Pydantic validation error: {error_message}",
        component="api",
        operation="validation",
        context={"errors": exc.errors(), "path": request.url.path}
    )
    
    # Return a consistent error response
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": ERROR_VALIDATION,
                "message": error_message,
                "details": exc.errors()
            }
        }
    )


async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle general exceptions.
    
    Args:
        request: FastAPI request object
        exc: Exception
        
    Returns:
        JSON response with error details
    """
    # Get traceback
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    tb_str = "".join(tb)
    
    # Determine error type and status code
    error_code = ERROR_EXECUTION
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    
    # Customize based on exception type
    if isinstance(exc, TimeoutError):
        error_code = ERROR_TIMEOUT
        status_code = status.HTTP_504_GATEWAY_TIMEOUT
    elif isinstance(exc, PermissionError):
        error_code = ERROR_PERMISSION
        status_code = status.HTTP_403_FORBIDDEN
    elif isinstance(exc, FileNotFoundError):
        error_code = ERROR_NOT_FOUND
        status_code = status.HTTP_404_NOT_FOUND
    elif isinstance(exc, NotImplementedError):
        error_code = ERROR_UNSUPPORTED
        status_code = status.HTTP_501_NOT_IMPLEMENTED
    
    # Log the error
    logger.error(
        f"Unhandled exception: {str(exc)}",
        component="api",
        operation="exception",
        context={
            "exception_type": type(exc).__name__,
            "traceback": tb_str,
            "path": request.url.path,
            "method": request.method
        }
    )
    
    # Return a consistent error response
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": error_code,
                "message": str(exc),
                "exception_type": type(exc).__name__
            }
        }
    )


def add_error_handlers(app: FastAPI):
    """
    Add exception handlers to the FastAPI app.
    
    Args:
        app: FastAPI application
    """
    # Add handlers for specific exception types
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(ValidationError, pydantic_validation_exception_handler)
    
    # Add general exception handler
    app.add_exception_handler(Exception, general_exception_handler)
    
    logger.info("API error handlers configured", component="api")