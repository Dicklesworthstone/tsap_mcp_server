"""
Mount point for the ToolAPI Server in the original TSAP implementation.

This module provides utilities to mount the standards-compliant ToolAPI server
inside the original TSAP server, allowing for a gradual migration path.
"""
from typing import Dict, Any, Optional, Callable, Awaitable
import asyncio
import subprocess
import os
import signal
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ToolAPIServerProcess:
    """Manages a subprocess running the ToolAPI server."""
    
    def __init__(self):
        """Initialize the ToolAPI server process manager."""
        self.process: Optional[subprocess.Popen] = None
        self.running = False
    
    async def start(self) -> None:
        """Start the ToolAPI server in a subprocess."""
        if self.running:
            return
        
        # Start the ToolAPI server
        print("Starting ToolAPI server subprocess...")
        self.process = subprocess.Popen(
            ["python", "-m", "src.tsap_mcp.__main__", "run"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,  # Create a new process group
        )
        
        self.running = True
        
        # Give it time to start
        await asyncio.sleep(2)
        
        # Check if it's still running
        if self.process.poll() is not None:
            raise RuntimeError(f"ToolAPI server failed to start: {self.process.stderr.read().decode()}")
        
        print("ToolAPI server subprocess started successfully")
    
    async def stop(self) -> None:
        """Stop the ToolAPI server subprocess."""
        if not self.running or self.process is None:
            return
        
        print("Stopping ToolAPI server subprocess...")
        
        # Stop the process group
        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            
            # Wait for process to terminate
            for _ in range(10):  # Wait up to 5 seconds
                if self.process.poll() is not None:
                    break
                await asyncio.sleep(0.5)
            
            # Force kill if still running
            if self.process.poll() is None:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
        except ProcessLookupError:
            # Process already terminated
            pass
        
        self.running = False
        self.process = None
        print("ToolAPI server subprocess stopped")


class ToolAPIServerProxy:
    """Proxy for the ToolAPI server that handles communication through the client."""
    
    def __init__(self):
        """Initialize the ToolAPI server proxy."""
        self.server_process = ToolAPIServerProcess()
        self._session = None
    
    async def start(self) -> None:
        """Start the ToolAPI server and establish a connection."""
        await self.server_process.start()
    
    async def stop(self) -> None:
        """Stop the ToolAPI server and close the connection."""
        await self.server_process.stop()
    
    async def execute(self, command: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a command on the ToolAPI server.
        
        Args:
            command: Command name
            args: Command arguments
            
        Returns:
            Command result
        """
        # Import here to avoid circular imports
        from mcp.client import ClientSession
        from mcp.client.stdio import stdio_client
        from mcp import StdioServerParameters
        
        # Create server parameters
        server_params = StdioServerParameters(
            command="python",
            args=["-m", "src.tsap_mcp.__main__", "run"],
        )
        
        # Connect to the server and execute the command
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()
                
                # Call the tool
                result = await session.call_tool(command, arguments=args)
                
                return result


# Singleton instance
_toolapi_server = ToolAPIServerProxy()


async def startup() -> None:
    """Start the ToolAPI server on application startup."""
    await _toolapi_server.start()


async def shutdown() -> None:
    """Stop the ToolAPI server on application shutdown."""
    await _toolapi_server.stop()


async def execute_toolapi_command(command: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a command on the ToolAPI server.
    
    This function is used as an adapter between the original API and the ToolAPI server.
    
    Args:
        command: Command name
        args: Command arguments
        
    Returns:
        Command result
    """
    return await _toolapi_server.execute(command, args)


def get_toolapi_startup_handler() -> Callable[[], Awaitable[None]]:
    """Get the ToolAPI server startup handler.
    
    Returns:
        Async function to start the ToolAPI server
    """
    return startup


def get_toolapi_shutdown_handler() -> Callable[[], Awaitable[None]]:
    """Get the ToolAPI server shutdown handler.
    
    Returns:
        Async function to stop the ToolAPI server
    """
    return shutdown 