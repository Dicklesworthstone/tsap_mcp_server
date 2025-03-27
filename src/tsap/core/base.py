"""
Base classes for TSAP core tools.

This module provides the abstract base classes for all core tools
in the TSAP system, defining common interfaces and functionality.
"""
import os
import time
import asyncio
import shutil
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
from contextlib import asynccontextmanager

from tsap.utils.logging import logger


class BaseCoreTool(ABC):
    """Abstract base class for all core tools."""
    
    def __init__(self, tool_name: str):
        """Initialize the core tool.
        
        Args:
            tool_name: Name of the tool
        """
        self.tool_name = tool_name
        self.execution_count = 0
        self.total_execution_time = 0.0
    
    @asynccontextmanager
    async def _measure_execution_time(self):
        """Context manager to measure execution time.
        
        Yields:
            None
        """
        start_time = time.time()
        
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.execution_count += 1
            self.total_execution_time += elapsed
    
    async def _run_process(
        self,
        cmd: List[str],
        timeout: float = 30.0,
        input_data: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        workdir: Optional[str] = None,
    ) -> Tuple[int, str, str]:
        """Run a subprocess with timeout.
        
        Args:
            cmd: Command to run as a list of arguments
            timeout: Timeout in seconds
            input_data: Optional input data to send to process stdin
            env: Optional environment variables
            workdir: Optional working directory
            
        Returns:
            Tuple of (exit_code, stdout, stderr)
            
        Raises:
            asyncio.TimeoutError: If process times out
            subprocess.SubprocessError: For other subprocess errors
        """
        # Create environment with parent process environment plus any overrides
        process_env = os.environ.copy()
        if env:
            process_env.update(env)
        
        # Log the command
        cmd_str = " ".join(cmd)
        logger.debug(
            f"Running command: {cmd_str}",
            component="core",
            operation=f"{self.tool_name}_exec",
            context={"command": cmd_str}
        )
        
        # Create and start the process
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE if input_data else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=process_env,
            cwd=workdir,
        )
        
        try:
            # Communicate with the process (with timeout)
            input_bytes = input_data.encode() if input_data else None
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input_bytes), timeout=timeout
            )
            
            # Decode output
            stdout_str = stdout.decode("utf-8", errors="replace") if stdout else ""
            stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""
            
            return process.returncode, stdout_str, stderr_str
            
        except asyncio.TimeoutError:
            # Process timed out, try to kill it
            try:
                process.kill()
            except Exception:
                pass
                
            logger.warning(
                f"Command timed out after {timeout}s: {cmd_str}",
                component="core",
                operation=f"{self.tool_name}_exec",
                context={"command": cmd_str, "timeout": timeout}
            )
            
            raise
            
        except Exception as e:
            # Process failed for other reasons
            try:
                process.kill()
            except Exception:
                pass
                
            logger.error(
                f"Command failed: {cmd_str}",
                component="core",
                operation=f"{self.tool_name}_exec",
                exception=e,
                context={"command": cmd_str}
            )
            
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics for this tool.
        
        Returns:
            Dictionary with execution statistics
        """
        avg_time = 0.0
        if self.execution_count > 0:
            avg_time = self.total_execution_time / self.execution_count
            
        return {
            "tool_name": self.tool_name,
            "execution_count": self.execution_count,
            "total_execution_time": self.total_execution_time,
            "avg_execution_time": avg_time,
        }
    
    def reset_statistics(self) -> None:
        """Reset execution statistics."""
        self.execution_count = 0
        self.total_execution_time = 0.0


class ToolRegistry:
    """Registry for core tools."""
    
    _registry: Dict[str, Any] = {}
    
    @classmethod
    def register(cls, name: str, tool_class: type) -> None:
        """Register a tool class.
        
        Args:
            name: Tool name
            tool_class: Tool class
        """
        cls._registry[name] = tool_class
        logger.debug(
            f"Registered tool: {name}",
            component="core",
            operation="register_tool"
        )
    
    @classmethod
    def get_tool_class(cls, name: str) -> Optional[type]:
        """Get a tool class by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool class or None if not found
        """
        return cls._registry.get(name)
    
    @classmethod
    def list_tools(cls) -> List[str]:
        """List all registered tools.
        
        Returns:
            List of tool names
        """
        return list(cls._registry.keys())


def register_tool(name: str):
    """Decorator to register a tool class.
    
    Args:
        name: Tool name
        
    Returns:
        Decorator function
    """
    def decorator(cls):
        ToolRegistry.register(name, cls)
        return cls
    return decorator