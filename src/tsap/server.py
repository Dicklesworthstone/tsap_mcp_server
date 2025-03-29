"""
TSAP MCP Server implementation.

This module provides the main server functionality that listens for MCP
requests and dispatches them to the appropriate handlers.
"""
import asyncio
from typing import Optional, Any
from contextlib import asynccontextmanager
import os
import logging
import logging.config

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
from tsap.api.app import api_router


# --- Define Logging Configuration Dictionary ---

LOG_FILE_PATH = "logs/tsap.log"

# Ensure log directory exists before config is used
log_dir = os.path.dirname(LOG_FILE_PATH)
if log_dir:
    os.makedirs(log_dir, exist_ok=True)

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False, # Let Uvicorn's loggers pass through if needed
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(levelprefix)s %(message)s",
            "use_colors": None,
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": '%(levelprefix)s %(client_addr)s - \"%(request_line)s\" %(status_code)s',
        },
        "rich": { # Formatter for Rich console output (can be simpler)
             "format": "%(message)s",
             "datefmt": "[%X]",
        },
         "file": { # Formatter for file output
             "format": "%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
             "datefmt": "%Y-%m-%d %H:%M:%S",
         },
    },
    "handlers": {
        "default": { # Default Uvicorn console handler (might be replaced by Rich)
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
        "access": { # Uvicorn access log handler
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "rich_console": { # Our Rich console handler using a factory
             # "class": "tsap.utils.logging.formatter.RichLoggingHandler",
             # "formatter": "rich", # Formatter might be set internally by handler or factory
             # "level": "DEBUG", # Level is set on the logger/root now
             # "console": "ext://tsap.utils.logging.console.get_rich_console",
             # "show_path": False,
             # "markup": True,
             # "rich_tracebacks": True,
             "()": "tsap.utils.logging.formatter.create_rich_console_handler",
             # Pass other necessary args to the factory if needed, e.g.:
             # "level": "DEBUG", # Set level via factory if desired 
             # "show_path": False, 
        },
        "rotating_file": { # Our rotating file handler
             "class": "logging.handlers.RotatingFileHandler",
             "formatter": "file",
             "level": "DEBUG", # Set default level here, will be overridden
             "filename": LOG_FILE_PATH,
             "maxBytes": 2 * 1024 * 1024, # 2 MB
             "backupCount": 5,
             "encoding": "utf-8",
        },
    },
    "loggers": {
        "uvicorn": {"handlers": ["rich_console"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"level": "INFO", "propagate": True}, # Propagate errors to root
        "uvicorn.access": {"handlers": ["access", "rotating_file"], "level": "INFO", "propagate": False},
        "tsap": { # Our application's logger namespace
             "handlers": ["rich_console", "rotating_file"],
             "level": "DEBUG", # Default level, will be overridden by root level typically
             "propagate": False,
         },
    },
     "root": { # Root logger configuration
         "level": "DEBUG", # Default level, will be overridden
         "handlers": ["rich_console", "rotating_file"], # Root catches logs not handled by specific loggers
     },
}

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
    # Startup logging is now handled by the dictConfig
    # logger.startup(...) can still be used if needed for specific messages
    
    try:
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
        # Shutdown logging handled by dictConfig
        logger.info("Server shutdown sequence initiated.") # Example specific message


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
    app.include_router(api_router, prefix="/api")
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


    return router


def start_server(
    host: Optional[str] = None,
    port: Optional[int] = None,
    workers: Optional[int] = None,
    log_level: Optional[str] = None,
    reload: bool = False,
) -> None:
    """Start the TSAP MCP Server using dictConfig for logging."""
    config = get_config()
    server_host = host or config.server.host
    server_port = port or config.server.port
    server_workers = workers or config.server.workers
    # Determine final log level (passed in from __main__ which considers verbose/config)
    final_log_level = (log_level or config.server.log_level).upper()

    # --- Update LOGGING_CONFIG with the final level ---
    LOGGING_CONFIG["root"]["level"] = final_log_level
    # LOGGING_CONFIG["handlers"]["rich_console"]["level"] = final_log_level # Level set via logger/root
    # LOGGING_CONFIG["handlers"]["rotating_file"]["level"] = final_log_level # Level set via logger/root
    LOGGING_CONFIG["loggers"]["tsap"]["level"] = final_log_level
    # Set Uvicorn access level based on final level (e.g., hide access logs if CRITICAL)
    LOGGING_CONFIG["loggers"]["uvicorn.access"]["level"] = final_log_level if final_log_level != "CRITICAL" else "CRITICAL"
    # Ensure Uvicorn base/error logs are at least INFO unless final level is DEBUG
    uvicorn_base_level = "INFO" if final_log_level not in ["DEBUG"] else "DEBUG"
    LOGGING_CONFIG["loggers"]["uvicorn"]["level"] = uvicorn_base_level
    LOGGING_CONFIG["loggers"]["uvicorn.error"]["level"] = uvicorn_base_level

    # Log startup info using the standard logger BEFORE Uvicorn takes over logging
    # This might still use basic config if Uvicorn hasn't processed dictConfig yet
    logging.info(f"Preparing to start Uvicorn on {server_host}:{server_port}...") 

    # Start server with Uvicorn using the dictConfig
    uvicorn.run(
        "tsap.server:create_server", # Use factory pattern
        host=server_host,
        port=server_port,
        workers=server_workers,
        log_config=LOGGING_CONFIG, # Pass the config dict
        reload=reload,
        factory=True,
    )