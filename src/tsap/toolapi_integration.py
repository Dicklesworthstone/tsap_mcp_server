"""
Integration module for ToolAPI server in original TSAP server.

This module provides utilities for integrating the standards-compliant ToolAPI server
with the original TSAP implementation, allowing both to operate side-by-side.
"""
from typing import Dict, Any
from fastapi import APIRouter, FastAPI
from pydantic import BaseModel

from tsap.toolapi_mount import get_toolapi_startup_handler, get_toolapi_shutdown_handler, execute_toolapi_command


# Create a request model for the ToolAPI commands
class ToolAPIRequest(BaseModel):
    """Model for ToolAPI command requests."""
    command: str
    args: Dict[str, Any]
    request_id: str = None


# Create a router for ToolAPI commands
toolapi_router = APIRouter(prefix="/toolapi", tags=["toolapi"])


@toolapi_router.post("/")
async def execute_toolapi_command(request: ToolAPIRequest) -> Dict[str, Any]:
    """Execute a command on the standards-compliant ToolAPI server.
    
    Args:
        request: The ToolAPI request containing command and arguments
        
    Returns:
        Command result
    """
    try:
        # Execute the command on the ToolAPI server
        result = await execute_toolapi_command(request.command, request.args)
        
        # Format the response for compatibility
        return {
            "status": "success",
            "error": None,
            "data": result,
            "request_id": request.request_id,
        }
    except Exception as e:
        # Handle errors
        return {
            "status": "error",
            "error": str(e),
            "data": None,
            "request_id": request.request_id,
        }


# Function to mount the ToolAPI router to the main FastAPI app
def mount_toolapi_server(app: FastAPI) -> None:
    """Mount the ToolAPI server integration to the main FastAPI app.
    
    Args:
        app: The FastAPI application
    """
    # Include the ToolAPI router
    app.include_router(toolapi_router)
    
    # Register startup and shutdown handlers
    app.add_event_handler("startup", get_toolapi_startup_handler())
    app.add_event_handler("shutdown", get_toolapi_shutdown_handler())
    
    print("ToolAPI native server mounted at /toolapi")


# Function to get a list of available ToolAPI tools
async def get_available_toolapi_tools() -> Dict[str, Any]:
    """Get a list of available tools from the ToolAPI server.
    
    Returns:
        Dictionary containing available tools
    """
    from mcp.client import ClientSession
    from mcp.client.stdio import stdio_client
    from mcp import StdioServerParameters
    
    # Create server parameters
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "src.tsap_mcp.__main__", "run"],
    )
    
    # Connect to the server
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            
            # List available tools
            tools = await session.list_tools()
            
            # List available resources
            resources = await session.list_resources()
            
            # List available prompts
            prompts = await session.list_prompts()
            
            return {
                "tools": [
                    {
                        "name": t.name,
                        "description": t.description,
                        "arguments": [
                            {"name": a.name, "description": a.description, "required": a.required}
                            for a in t.arguments
                        ],
                    }
                    for t in tools
                ],
                "resources": [
                    {
                        "name": r.name,
                        "description": r.description,
                    }
                    for r in resources
                ],
                "prompts": [
                    {
                        "name": p.name,
                        "description": p.description,
                        "arguments": [
                            {"name": a.name, "description": a.description, "required": a.required}
                            for a in p.arguments
                        ],
                    }
                    for p in prompts
                ],
            } 