#!/usr/bin/env python3
"""
MCP StdIO Server Example

This script demonstrates creating a complete MCP server that communicates via
standard input/output, allowing direct communication with MCP clients without
requiring HTTP.
"""
import asyncio
import sys
import logging
import json
from typing import Dict, Any, List
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

# Import the MCP server components
import mcp.server.stdio
from mcp.server.lowlevel import Server
from mcp.server.models import InitializationOptions
from mcp.types import TextContent, ReadResourceResult as ResourceResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("mcp_stdio_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mcp_stdio_server")

# Example in-memory database for demonstration
USERS = {
    "user1": {"name": "Alice", "role": "admin"},
    "user2": {"name": "Bob", "role": "user"},
    "user3": {"name": "Charlie", "role": "user"}
}

FILE_CONTENTS = {
    "readme.txt": "This is a sample readme file.",
    "config.json": '{"debug": true, "version": "1.0.0"}'
}

@asynccontextmanager
async def server_lifespan(server: Server) -> AsyncIterator[Dict[str, Any]]:
    """Manage server startup and shutdown lifecycle."""
    # Initialize resources on startup
    logger.info("Server starting up...")
    try:
        # This context will be available to all request handlers
        yield {
            "start_time": asyncio.get_event_loop().time(),
            "users": USERS,
            "files": FILE_CONTENTS
        }
    finally:
        # Clean up on shutdown
        logger.info("Server shutting down...")

# Create the server instance
server = Server(
    name="Example StdIO Server",
    description="A simple MCP server that communicates via stdin/stdout",
    version="1.0.0",
    lifespan=server_lifespan
)

# Tool implementations
@server.call_tool()
async def echo(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Echo back the provided message."""
    message = arguments.get("message", "")
    logger.info(f"Echo tool called with message: {message}")
    
    return {
        "type": "success",
        "content": [
            {
                "type": "text",
                "text": message
            }
        ]
    }

@server.call_tool()
async def calculator(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Perform basic arithmetic operations."""
    operation = arguments.get("operation", "add")
    a = float(arguments.get("a", 0))
    b = float(arguments.get("b", 0))
    
    result = None
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        if b == 0:
            return {
                "type": "error",
                "error": {
                    "message": "Cannot divide by zero"
                }
            }
        result = a / b
    else:
        return {
            "type": "error",
            "error": {
                "message": f"Unknown operation: {operation}"
            }
        }
    
    logger.info(f"Calculator tool called: {a} {operation} {b} = {result}")
    
    return {
        "type": "success",
        "content": [
            {
                "type": "text",
                "text": str(result)
            }
        ]
    }

@server.call_tool()
async def get_user(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Get user information."""
    user_id = arguments.get("user_id", "")
    ctx = server.request_context
    users = ctx.lifespan_context.get("users", {})
    
    if user_id not in users:
        return {
            "type": "error",
            "error": {
                "message": f"User not found: {user_id}"
            }
        }
    
    user_data = users[user_id]
    logger.info(f"User data requested for {user_id}")
    
    return {
        "type": "success",
        "content": [
            {
                "type": "text",
                "text": json.dumps(user_data)
            }
        ]
    }

@server.call_tool()
async def list_files(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """List available files."""
    ctx = server.request_context
    files = ctx.lifespan_context.get("files", {})
    
    file_list = list(files.keys())
    logger.info(f"File list requested: {file_list}")
    
    return {
        "type": "success",
        "content": [
            {
                "type": "text",
                "text": json.dumps(file_list)
            }
        ]
    }

@server.call_tool()
async def read_file(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Read the contents of a file."""
    filename = arguments.get("filename", "")
    ctx = server.request_context
    files = ctx.lifespan_context.get("files", {})
    
    if filename not in files:
        return {
            "type": "error",
            "error": {
                "message": f"File not found: {filename}"
            }
        }
    
    content = files[filename]
    logger.info(f"File read requested for {filename}")
    
    return {
        "type": "success",
        "content": [
            {
                "type": "text",
                "text": content
            }
        ]
    }

@server.call_tool()
async def info(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Get server information."""
    ctx = server.request_context
    start_time = ctx.lifespan_context.get("start_time", 0)
    current_time = asyncio.get_event_loop().time()
    uptime = current_time - start_time
    
    info_data = {
        "version": "1.0.0",
        "name": "Example StdIO Server",
        "uptime": f"{uptime:.2f} seconds",
        "tools": ["echo", "calculator", "get_user", "list_files", "read_file", "info"],
        "resources": ["user/{user_id}", "file/{filename}"]
    }
    
    logger.info("Server info requested")
    
    return {
        "type": "success",
        "content": [
            {
                "type": "text",
                "text": json.dumps(info_data)
            }
        ]
    }

# Resource implementations
@server.get_resource()
async def get_resource(url: str) -> ResourceResponse:
    """Handle resource requests."""
    logger.info(f"Resource requested: {url}")
    ctx = server.request_context
    
    # Parse the URL to determine what resource to return
    if url.startswith("user/"):
        # User resource: user/{user_id}
        user_id = url[5:]  # Remove "user/" prefix
        users = ctx.lifespan_context.get("users", {})
        
        if user_id not in users:
            return ResourceResponse(
                success=False,
                error=f"User not found: {user_id}"
            )
        
        user_data = users[user_id]
        return ResourceResponse(
            success=True,
            content=[
                TextContent(text=json.dumps(user_data))
            ]
        )
    
    elif url.startswith("file/"):
        # File resource: file/{filename}
        filename = url[5:]  # Remove "file/" prefix
        files = ctx.lifespan_context.get("files", {})
        
        if filename not in files:
            return ResourceResponse(
                success=False,
                error=f"File not found: {filename}"
            )
        
        content = files[filename]
        return ResourceResponse(
            success=True,
            content=[
                TextContent(text=content)
            ]
        )
    
    else:
        return ResourceResponse(
            success=False,
            error=f"Unknown resource: {url}"
        )

# Tool discovery handlers
@server.list_tools()
async def list_tools() -> List[Dict[str, Any]]:
    """Return the list of available tools."""
    tools = [
        {
            "name": "echo",
            "description": "Echo back the provided message",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to echo back"
                    }
                },
                "required": ["message"]
            }
        },
        {
            "name": "calculator",
            "description": "Perform basic arithmetic operations",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "The operation to perform (add, subtract, multiply, divide)",
                        "enum": ["add", "subtract", "multiply", "divide"]
                    },
                    "a": {
                        "type": "number",
                        "description": "First number"
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number"
                    }
                },
                "required": ["operation", "a", "b"]
            }
        },
        {
            "name": "get_user",
            "description": "Get user information",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The user ID to lookup"
                    }
                },
                "required": ["user_id"]
            }
        },
        {
            "name": "list_files",
            "description": "List available files",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "read_file",
            "description": "Read the contents of a file",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The name of the file to read"
                    }
                },
                "required": ["filename"]
            }
        },
        {
            "name": "info",
            "description": "Get server information",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }
    ]
    
    logger.info("Tool list requested")
    return tools

# Custom initialization options
@server.initialize()
async def initialize(options: InitializationOptions) -> Dict[str, Any]:
    """Custom initialization handler."""
    logger.info(f"Client initializing with protocol version: {options.protocolVersion}")
    logger.info(f"Client info: {options.clientInfo.name} {options.clientInfo.version}")
    
    # You can override or extend the default initialization response
    return {
        "protocolVersion": options.protocolVersion,
        "serverInfo": {
            "name": "Example StdIO Server",
            "version": "1.0.0"
        },
        "capabilities": {
            "prompts": {
                "listChanged": False
            },
            "resources": {
                "subscribe": False,
                "listChanged": False
            },
            "tools": {
                "listChanged": True
            },
            "logging": {},
            "completions": {}
        },
        "instructions": "This is a simple MCP server for demonstration purposes."
    }

async def main():
    """Run the StdIO server."""
    logger.info("Starting MCP StdIO server...")
    
    # Create stdio server streams for communication
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        # Start the server with stdio streams
        await server.serve(read_stream, write_stream)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1) 