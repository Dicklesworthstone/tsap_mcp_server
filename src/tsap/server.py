"""
TSAP MCP Server implementation.

This module provides the main server functionality that listens for MCP
requests and dispatches them to the appropriate handlers.
"""
import signal
import asyncio
from typing import Optional, Any
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from tsap.utils.logging import logger
from tsap.config import get_config
from tsap.version import __version__, get_version_info
from tsap.performance_mode import get_performance_mode
from tsap.mcp.protocol import MCPRequest
from tsap.mcp.handler import handle_request, initialize_handlers


# Global server instance
_server_app = None

# Shutdown event
_shutdown_event = asyncio.Event()


# Updated Helper function for aggressive, recursive truncation
def truncate_repr(obj: Any, max_len: int = 50) -> str:
    """Return a representation of the object, aggressively truncating all long strings."""
    try:
        if isinstance(obj, dict):
            # Recursively process dictionary values
            items_repr = []
            for k, v in obj.items():
                k_repr = repr(k) # Keys are usually short, keep as is
                v_repr = truncate_repr(v, max_len) # Recurse on value
                items_repr.append(f"{k_repr}: {v_repr}")
            return "{" + ", ".join(items_repr) + "}"

        elif isinstance(obj, list):
            # Recursively process list items, showing only first few
            num_items = len(obj)
            if num_items == 0:
                return "[]"

            truncated_list = []
            for i, item in enumerate(obj):
                if i < 3:
                    item_repr = truncate_repr(item, max_len) # Recurse on item
                    truncated_list.append(item_repr)
                elif i == 3:
                    truncated_list.append("...")
                    break # Stop after showing ellipsis
            return f"[{', '.join(truncated_list)}] ({num_items} items total)"

        elif isinstance(obj, str):
            # Truncate string if it exceeds max_len
            if len(obj) > max_len:
                return f"'{obj[:max_len]}... (truncated {len(obj) - max_len} chars)'"
            else:
                return repr(obj) # Use repr to keep quotes for strings

        elif isinstance(obj, bytes):
             # Truncate bytes
             if len(obj) > max_len:
                 return f"{repr(obj[:max_len])[:-1]}... (truncated {len(obj) - max_len} bytes)'"
             else:
                 return repr(obj)

        else:
            # For other types, use standard repr but truncate if too long
            obj_repr = repr(obj)
            if len(obj_repr) > max_len:
                 # Try to get a shorter representation for objects
                 if hasattr(obj, '__class__'):
                     short_repr = f"<{obj.__class__.__name__} object>"
                     if len(short_repr) <= max_len: 
                         return short_repr
                 return f"{obj_repr[:max_len]}... (truncated)"
            return obj_repr

    except Exception as e:
        # Ultimate fallback for problematic objects during recursion/repr
        try:
            return f"<Object of type {type(obj).__name__} - Error during repr: {str(e)[:max_len]}>"
        except Exception:
             return "<Unrepresentable object - Error during repr>"


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
            
        # Make sure MCP handlers are initialized
        initialize_handlers()
            
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
        # Use the truncate_repr helper for logging request details
        print(f"REQUEST RECEIVED: {truncate_repr(request)}")
        print(f"  Command: {request.command}")
        print(f"  Request ID: {request.request_id}")
        # Truncate args separately if needed for more clarity
        print(f"  Args: {truncate_repr(request.args)}")

        logger.info(
            f"Received MCP request: {request.command}",
            component="mcp",
            operation="request",
            context={"request_id": request.request_id}
        )
        
        try:
            # Use the handle_request function from mcp/handler.py
            print("About to call handle_request...")
            response = await handle_request(request)
            # Use the truncate_repr helper for logging response details
            print(f"RESPONSE: {truncate_repr(response)}")
            print(f"  Status: {response.status}")
            if response.data:
                 # Truncate data separately if needed
                print(f"  Data: {truncate_repr(response.data)}")
            if response.error:
                print(f"  Error: {truncate_repr(response.error)}")
            
            logger.success(
                f"Completed MCP request: {request.command}",
                component="mcp",
                operation="response",
                context={"request_id": request.request_id}
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            # Use truncate_repr for logging error response details too
            error_content = {"error": "Failed to process request", "details": str(e)}
            response = JSONResponse(
                content=error_content,
                status_code=400,
            )
            print(f"ERROR RESPONSE: {truncate_repr(error_content)}") # Log truncated error
            return response

        # # The following section seems redundant with the initial logging and truncation logic above.
        # # Commenting it out to avoid double logging/processing. If needed, integrate its logic
        # # more cleanly into the initial print statements.

        # # Log received request (truncate 'texts' if present)
        # log_request_repr = repr(request)
        # try:
        #     # Check if args exist and contain 'texts'
        #     if hasattr(request, 'args') and isinstance(request.args, dict) and 'texts' in request.args:
        #         # Create a copy to modify for logging
        #         log_args = request.args.copy()
        #         texts_value = log_args['texts']
        #         truncated_texts_repr = "[...some texts...]"
        #         num_items = 0

        #         if isinstance(texts_value, list):
        #             num_items = len(texts_value)
        #             truncated_list = []
        #             # Truncate each text item if it's a long string
        #             for i, text_item in enumerate(texts_value):
        #                 if i < 3: # Show first few items truncated
        #                     if isinstance(text_item, str) and len(text_item) > 100:
        #                         truncated_list.append(f"'{text_item[:100]}... (truncated)'")
        #                     else:
        #                         # Keep short strings or non-string items as is (using repr)
        #                         truncated_list.append(repr(text_item))
        #                 elif i == 3:
        #                     truncated_list.append("...") # Indicate more items were truncated
        #                     break # Stop after showing ellipsis
        #             truncated_texts_repr = f"[{', '.join(truncated_list)}] ({num_items} items total)"
        #         elif isinstance(texts_value, str) and len(texts_value) > 200:
        #             num_items = 1
        #             truncated_texts_repr = f"'{texts_value[:200]}... (truncated {len(texts_value) - 200} chars)'"
        #         else:
        #              # Handle other types or short strings by just showing the type/count
        #             num_items = 1 if not isinstance(texts_value, list) else len(texts_value)
        #             truncated_texts_repr = f"<{type(texts_value).__name__} object ({num_items} item(s))>"

        #         log_args['texts'] = truncated_texts_repr # Replace in the copied dict

        #         # Reconstruct a limited repr for logging, showing command, ID, and modified args
        #         log_request_repr = f"MCPRequest(request_id='{request.request_id}', command='{request.command}', args={log_args})"
        # except Exception as log_exc: # Catch potential errors during log formatting
        #     logger.warning(f"Failed to format request args for logging: {log_exc}")
        #     # Fallback to default repr if formatting fails
        #     log_request_repr = repr(request)

        # print(f"REQUEST RECEIVED: {log_request_repr}") # This seems redundant

        # # Process request (This was already done above before the response logging)
        # mcp_response = await handle_request(request) # This line is duplicated

        # logger.success( # This logger call is also duplicated from the try block
        #     f"Completed MCP request: {request.command}",
        #     component="mcp",
        #     operation="response",
        #     context={"request_id": request.request_id}
        # )

        # return mcp_response # This return is duplicated


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