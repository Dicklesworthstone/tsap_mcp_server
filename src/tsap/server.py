"""
TSAP MCP Server implementation.

This module provides the main server functionality that listens for MCP
requests and dispatches them to the appropriate handlers.
"""
import asyncio
from typing import Optional, Any, Dict, Match, Pattern
from contextlib import asynccontextmanager
import os
import logging
import logging.config
import re
import time
import json

import uvicorn
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text
from rich import box
from rich.syntax import Syntax
from rich.json import JSON

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

# Rich console for fancy output
console = Console()

# Regular expression patterns for log message parsing
LOG_PATTERN = re.compile(r'\[(\d+:\d+:\d+\.\d+)\]\s+([^[]+)\s+\[([A-Z]+)\]\s+\[([^]]+)\]\s+(.*)')
PROGRESS_PATTERN = re.compile(r'(.+):\s+(\d+)%\|(.+)\|\s+(\d+)/(\d+)\s+\[(.+)<(.+),\s+(.+)\]')

def format_log_message(message: str) -> str:
    """Format log messages with Rich styling when they match expected patterns.
    
    Args:
        message: The log message to format
        
    Returns:
        The formatted message or the original if no patterns match
    """
    try:
        # Try to match standard log pattern
        match = LOG_PATTERN.match(message)
        if match:
            timestamp, icon, level, component, content = match.groups()
            level_color = {
                "INFO": "blue",
                "WARNING": "yellow",
                "ERROR": "red",
                "DEBUG": "dim white",
                "SUCCESS": "green",
                "CRITICAL": "red reverse"
            }.get(level, "white")
            
            component_parts = component.split()
            component_str = ""
            if len(component_parts) > 1:
                comp, operation = component_parts
                component_str = f"[magenta]{comp}[/] [cyan]{operation}:[/] "
            else:
                component_str = f"[magenta]{component}:[/] "
                
            formatted = f"[dim]{timestamp}[/] {icon} [{level_color}]{level}[/] {component_str}{content}"
            return formatted
            
        # Try to match progress bar pattern
        match = PROGRESS_PATTERN.match(message)
        if match:
            # For progress bars, just return the original as they're already well-formatted
            return message
            
        # No patterns matched, return original
        return message
        
    except Exception:
        # If any error occurs during formatting, return the original message
        return message

def format_command_execution(command: str, elapsed_time: Optional[float] = None) -> None:
    """Format command execution with elapsed time in a panel.
    
    Args:
        command: The command that was executed
        elapsed_time: Optional execution time in seconds
    """
    try:
        title = "[bold]Command Execution[/]"
        if elapsed_time is not None:
            title = f"[bold]Command Execution ([green]{elapsed_time:.4f}s[/])[/]"
            
        panel_content = Text.from_markup(f"[yellow]{command}[/]")
        console.print(Panel(panel_content, title=title, border_style="blue", box=box.ROUNDED))
    except Exception:
        # Fallback to simple output if panel creation fails
        if elapsed_time is not None:
            console.print(f"Command: {command} (Time: {elapsed_time:.4f}s)")
        else:
            console.print(f"Command: {command}")

# Add a Spinner for async operations
def create_spinner(message: str) -> Progress:
    """Create a spinner with message for long-running operations.
    
    Args:
        message: Message to display with the spinner
        
    Returns:
        A Progress object with spinner
    """
    try:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            console=console
        )
        task_id = progress.add_task(message, total=None)
        return progress, task_id
    except Exception:
        # Return None if spinner creation fails
        return None, None

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

def format_json_data(data: Any, max_length: int = 1000, max_depth: int = 3, max_string: int = 100, max_array: int = 5) -> Any:
    """Format JSON data with rich styling if it's valid JSON, with strict truncation limits.
    
    Args:
        data: Data that might be JSON
        max_length: Maximum length of the JSON string representation
        max_depth: Maximum nesting depth to display
        max_string: Maximum length of string values
        max_array: Maximum number of array items to show
        
    Returns:
        Rich JSON object if valid, otherwise the original data
    """
    # Safety first - if data is None, return None
    if data is None:
        return None
        
    try:
        # Check if data is already a dict/list (Python object)
        if isinstance(data, (dict, list)):
            # Apply depth-based truncation first
            truncated_data = truncate_nested_data(data, max_depth, max_string, max_array)
            
            # Convert to JSON string to ensure it's valid JSON and check length
            json_str = json.dumps(truncated_data, default=str)
            
            # Check length and truncate if needed
            if len(json_str) > max_length:
                # For large objects, use truncated version
                return truncate_repr(data)
            
            # Return Rich JSON object for pretty display
            return JSON.from_data(truncated_data)
            
        # Check if it's a JSON string
        elif isinstance(data, str):
            try:
                json_obj = json.loads(data)
                
                # Apply depth-based truncation
                truncated_obj = truncate_nested_data(json_obj, max_depth, max_string, max_array)
                
                # Check if parsed result needs truncation
                json_str = json.dumps(truncated_obj)
                if len(json_str) > max_length:
                    return f"<JSON string, {len(data)} chars>"
                
                # Return Rich JSON object
                return JSON.from_data(truncated_obj)
            except json.JSONDecodeError:
                # Not valid JSON, return original
                return data
        
        # For other types, return original
        return data
        
    except Exception as e:
        # If any errors, log and return original data
        print(f"Error in format_json_data: {str(e)}")
        return data

def truncate_nested_data(data: Any, max_depth: int = 3, max_string: int = 100, max_array: int = 5, current_depth: int = 0) -> Any:
    """Recursively truncate nested data structures to control output size.
    
    Args:
        data: The data structure to truncate
        max_depth: Maximum nesting depth to allow
        max_string: Maximum string length to show
        max_array: Maximum number of array items to show
        current_depth: Current recursion depth (internal use)
        
    Returns:
        Truncated version of the data structure
    """
    try:
        # Stop at max depth
        if current_depth >= max_depth:
            if isinstance(data, dict) and data:
                return {f"<{len(data)} dict keys>": "..."}
            elif isinstance(data, list) and data:
                return [f"<{len(data)} items>"]
            else:
                return data
                
        # Handle different types
        if isinstance(data, dict):
            result = {}
            # Sort keys for consistent output
            for i, (k, v) in enumerate(sorted(data.items())):
                # Limit number of dict items
                if i >= max_array:
                    result["..."] = f"<{len(data) - max_array} more keys>"
                    break
                    
                # Truncate keys if they're too long
                if isinstance(k, str) and len(k) > max_string:
                    k = k[:max_string] + "..."
                    
                # Recursively truncate values
                result[k] = truncate_nested_data(v, max_depth, max_string, max_array, current_depth + 1)
            return result
            
        elif isinstance(data, list):
            # Truncate lists
            if len(data) > max_array:
                truncated = [truncate_nested_data(x, max_depth, max_string, max_array, current_depth + 1) 
                            for x in data[:max_array]]
                truncated.append(f"<{len(data) - max_array} more items>")
                return truncated
            else:
                return [truncate_nested_data(x, max_depth, max_string, max_array, current_depth + 1) 
                        for x in data]
                        
        elif isinstance(data, str):
            # Truncate long strings
            if len(data) > max_string:
                return data[:max_string] + "..."
                
        # Return other types as is
        return data
        
    except Exception:
        # Safety fallback
        return "<Error during truncation>"

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
        console.print(f"[bold blue]REQUEST RECEIVED:[/] {truncate_repr(request)}", emoji=True)
        console.print(f"  [cyan]Command:[/] [yellow]{request.command}[/] :joystick:", emoji=True)
        console.print(f"  [cyan]Request ID:[/] [green]{request.request_id}[/] :id:", emoji=True)
        
        # Carefully format args as JSON if possible
        try:
            if hasattr(request, 'args') and request.args is not None:
                # Try to format args as pretty JSON
                try:
                    formatted_args = format_json_data(request.args)
                    if isinstance(formatted_args, str) or formatted_args is None:
                        # If not formattable as JSON, use truncated representation
                        console.print(f"  [cyan]Args:[/] {truncate_repr(request.args)} :package:", emoji=True)
                    else:
                        # Display nicely formatted JSON
                        console.print(f"  [cyan]Args:[/] :package:", emoji=True)
                        console.print("    ", end="")
                        console.print(formatted_args)
                except Exception as e:
                    # On any error, fall back to simple representation
                    console.print(f"  [cyan]Args:[/] {truncate_repr(request.args)} :package:", emoji=True)
            else:
                # No args or empty args
                console.print(f"  [cyan]Args:[/] None :package:", emoji=True)
        except Exception as e:
            console.print(f"  [cyan]Args:[/] <Error displaying args: {str(e)}> :package:", emoji=True)

        logger.info(
            f"Received MCP request: {request.command}",
            component="mcp",
            operation="request",
            context={"request_id": request.request_id}
        )
        
        try:
            # Simplified processing - remove spinner for now as it might be causing issues
            console.print("[bold cyan]About to call handle_request...[/] :rocket:", emoji=True)
            start_time = time.time()
            
            response = await handle_request(request)
            elapsed_time = time.time() - start_time
            
            # Use the truncate_repr helper for logging response details
            console.print(f"[bold green]RESPONSE:[/] {truncate_repr(response)} :incoming_envelope:", emoji=True)
            
            # Format status with appropriate color and icon
            status_color = "green" if response.status in ["success", "ok"] else "red"
            status_icon = ":white_check_mark:" if response.status in ["success", "ok"] else ":x:"
            console.print(f"  [cyan]Status:[/] [{status_color}]{response.status}[/] {status_icon}", emoji=True)
            
            # Format timing info
            if elapsed_time:
                console.print(f"  [cyan]Time:[/] [yellow]{elapsed_time:.4f}s[/] :stopwatch:", emoji=True)
                
            # Carefully introduce prettier data display
            if response.data:
                try:
                    # Check if data contains execution_time for enhanced display
                    execution_time = None
                    if isinstance(response.data, dict) and 'execution_time' in response.data:
                        try:
                            execution_time = float(response.data['execution_time'])
                            console.print(f"  [cyan]Execution time:[/] [green]{execution_time:.4f}s[/] :zap:", emoji=True)
                        except (ValueError, TypeError):
                            pass
                    
                    # Safely try JSON formatting
                    try:
                        formatted_data = format_json_data(response.data)
                        if isinstance(formatted_data, str) or formatted_data is None:
                            # Fallback to simple representation
                            console.print(f"  [cyan]Data:[/] {truncate_repr(response.data)} :open_file_folder:", emoji=True)
                        else:
                            # Pretty format the JSON data
                            console.print(f"  [cyan]Data:[/] :open_file_folder:", emoji=True)
                            console.print("    ", end="")
                            console.print(formatted_data)
                            
                            # Special handling for commands
                            if isinstance(response.data, dict) and 'command' in response.data:
                                try:
                                    cmd = response.data['command']
                                    if isinstance(cmd, str):
                                        console.print(f"  [cyan]Command:[/] [yellow]{cmd}[/] :keyboard:", emoji=True)
                                except Exception:
                                    pass
                    except Exception:
                        # If JSON formatting fails, use simple display
                        console.print(f"  [cyan]Data:[/] {truncate_repr(response.data)} :open_file_folder:", emoji=True)
                except Exception as e:
                    console.print(f"  [cyan]Data:[/] <Error displaying data: {str(e)}> :open_file_folder:", emoji=True)
            
            # Enhanced error display with fallback
            if response.error:
                try:
                    error_repr = truncate_repr(response.error)
                    
                    # Try to use a panel for errors, but with fallback
                    try:
                        console.print(Panel(Text.from_markup(f"[bold red]{error_repr}[/]"), 
                                    title="[bold red]Error[/]", 
                                    border_style="red",
                                    box=box.ROUNDED), emoji=True)
                    except Exception:
                        # Fallback to simple format if panel fails
                        console.print(f"  [cyan]Error:[/] [bold red]{error_repr}[/] :warning:", emoji=True)
                except Exception as e:
                    console.print(f"  [cyan]Error:[/] <Error displaying error: {str(e)}> :warning:", emoji=True)
            
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
            
            # Enhanced error display with fallback
            try:
                error_repr = truncate_repr(error_content)
                # Try to use a panel for errors, but with fallback
                try:
                    console.print(Panel(Text.from_markup(f"[bold red]{error_repr}[/]"), 
                                title="[bold red]ERROR RESPONSE[/]", 
                                border_style="red",
                                box=box.ROUNDED), emoji=True)
                except Exception:
                    # Fallback to simple format if panel fails
                    console.print(f"[bold red]ERROR RESPONSE:[/] {error_repr} :rotating_light:", emoji=True)
            except Exception as err:
                console.print(f"[bold red]ERROR RESPONSE:[/] <Error displaying error: {str(err)}> :rotating_light:", emoji=True)
            
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