"""
Request/response logging middleware for the TSAP MCP Server API.
"""

import time
import uuid
from typing import Callable
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from tsap.utils.logging import logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging requests and responses.
    
    This middleware logs incoming requests and outgoing responses,
    including timing information and status codes.
    """
    
    def __init__(self, app: ASGIApp):
        """Initialize the middleware."""
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and log information.
        
        Args:
            request: Incoming request
            call_next: Function to call the next middleware
            
        Returns:
            Response from the next middleware
        """
        # Generate request ID if not present
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        
        # Log the incoming request
        logger.info(
            f"Request {request_id}: {request.method} {request.url.path}",
            component="api",
            operation="request",
            context={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query": str(request.query_params),
                "client_host": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "unknown")
            }
        )
        
        # Process the request and measure timing
        start_time = time.time()
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log the response
            logger.info(
                f"Response {request_id}: {response.status_code} ({process_time:.3f}s)",
                component="api",
                operation="response",
                context={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "process_time": process_time
                }
            )
            
            # Add timing header
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-ID"] = request_id
            
            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Error processing request {request_id}: {str(e)}",
                component="api",
                operation="request_error",
                context={
                    "request_id": request_id,
                    "error": str(e),
                    "process_time": process_time
                }
            )
            raise


def add_logging_middleware(app: FastAPI) -> None:
    """
    Add request logging middleware to the FastAPI app.
    
    Args:
        app: FastAPI application
    """
    app.add_middleware(RequestLoggingMiddleware)