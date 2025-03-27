"""
TSAP MCP Server implementation.

This module provides the main server functionality that listens for MCP
requests and dispatches them to the appropriate handlers.
"""
import signal
import asyncio
from typing import Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware

from tsap.utils.logging import logger
from tsap.config import get_config
from tsap.version import __version__, get_version_info
from tsap.performance_mode import get_performance_mode, set_performance_mode
from tsap.mcp.protocol import MCPRequest, MCPResponse, MCPError


# Global server instance
_server_app = None

# Shutdown event
_shutdown_event = asyncio.Event()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Server lifespan context manager.
    
    This handles startup and shutdown events for the server.
    
    Args:
        app: FastAPI application instance
    """
    # Startup
    logger.startup(
        __version__,
        component="server",
        mode=get_performance_mode(),
        context={"api_version": get_version_info()["api_version"]}
    )
    
    try:
        # Register signal handlers for graceful shutdown
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop = asyncio.get_event_loop()
            loop.add_signal_handler(sig, _shutdown_event.set)
            
        # Load all plugins
        # TODO: Implement plugin loading
        
        # Start background tasks if any
        # TODO: Implement background tasks
        
        # Server is ready
        logger.success(
            "TSAP MCP Server started and ready",
            component="server",
            operation="startup"
        )
        
        yield
        
    finally:
        # Shutdown
        logger.info(
            "Shutting down TSAP MCP Server...",
            component="server",
            operation="shutdown"
        )
        
        # Cleanup
        # TODO: Implement cleanup tasks
        
        logger.shutdown(component="server")


def create_server() -> FastAPI:
    """Create and configure the FastAPI server.
    
    Returns:
        Configured FastAPI application
    """
    global _server_app
    
    # Check if server already exists
    if _server_app is not None:
        return _server_app
        
    # Get configuration
    config = get_config()
    
    # Create FastAPI app
    app = FastAPI(
        title="TSAP MCP Server",
        description="Text Search and Processing Model Context Protocol Server",
        version=__version__,
        lifespan=lifespan,
        debug=config.server.debug,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add routers
    app.include_router(_create_api_router(), prefix="/api")
    app.include_router(_create_mcp_router(), prefix="/mcp")
    
    # Add health check route
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "ok",
            "version": __version__,
            "mode": get_performance_mode(),
        }
    
    # Add version info route
    @app.get("/version")
    async def version():
        """Version information endpoint."""
        return get_version_info()
    
    # Store the app instance
    _server_app = app
    
    return app


def _create_api_router() -> APIRouter:
    """Create the API router.
    
    Returns:
        Configured APIRouter for the API
    """
    router = APIRouter(tags=["api"])
    
    @router.get("/")
    async def api_root():
        """API root endpoint."""
        return {
            "name": "TSAP MCP Server API",
            "version": get_version_info()["api_version"],
            "documentation": "/docs",
        }
    
    # TODO: Add other API routes (core, composite, analysis, etc.)
    
    return router


def _create_mcp_router() -> APIRouter:
    """Create the MCP protocol router.
    
    Returns:
        Configured APIRouter for the MCP protocol
    """
    router = APIRouter(tags=["mcp"])
    
    @router.post("/")
    async def mcp_endpoint(request: MCPRequest):
        """Main MCP protocol endpoint.
        
        This receives MCP requests and dispatches them to the appropriate handlers.
        
        Args:
            request: MCP request
            
        Returns:
            MCP response
        """
        logger.info(
            f"Received MCP request: {request.command}",
            component="mcp",
            operation="request",
            context={"request_id": request.request_id}
        )
        
        try:
            # Set performance mode based on request
            if request.mode:
                set_performance_mode(request.mode)
                
            # TODO: Dispatch to appropriate handler based on command
            
            # Temporary placeholder response
            response = MCPResponse(
                request_id=request.request_id,
                status="success",
                command=request.command,
                data={"message": "Not implemented yet"},
            )
            
            logger.success(
                f"Completed MCP request: {request.command}",
                component="mcp",
                operation="response",
                context={"request_id": request.request_id}
            )
            
            return response
            
        except Exception as e:
            logger.error(
                f"Error processing MCP request: {str(e)}",
                component="mcp",
                operation="error",
                exception=e,
                context={"request_id": request.request_id}
            )
            
            return MCPResponse(
                request_id=request.request_id,
                status="error",
                command=request.command,
                error=MCPError(
                    code="internal_error",
                    message=str(e),
                    details=str(e.__class__.__name__),
                )
            )
    
    return router


def start_server(
    host: Optional[str] = None,
    port: Optional[int] = None,
    workers: Optional[int] = None,
    log_level: Optional[str] = None,
    reload: bool = False,
) -> None:
    """Start the TSAP MCP Server.
    
    Args:
        host: Host to bind to (overrides config)
        port: Port to bind to (overrides config)
        workers: Number of worker processes (overrides config)
        log_level: Logging level (overrides config)
        reload: Whether to enable auto-reload
    """
    # Get configuration
    config = get_config()
    
    # Override config with parameters if provided
    server_host = host or config.server.host
    server_port = port or config.server.port
    server_workers = workers or config.server.workers
    server_log_level = log_level or config.server.log_level
    
    # Create server if not already created
    app = create_server()  # noqa: F841
    
    # Log startup info
    logger.info(
        f"Starting TSAP MCP Server on {server_host}:{server_port}",
        component="server",
        operation="start",
        context={
            "host": server_host,
            "port": server_port,
            "workers": server_workers,
            "log_level": server_log_level,
            "reload": reload,
        }
    )
    
    # Start server with Uvicorn
    uvicorn.run(
        "tsap.server:create_server",
        host=server_host,
        port=server_port,
        workers=server_workers,
        log_level=server_log_level.lower(),
        reload=reload,
        factory=True,
    )