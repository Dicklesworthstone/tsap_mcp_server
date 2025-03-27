"""
JQ integration for TSAP.

This module provides functionality to query and transform JSON data using the
jq command-line tool, with enhanced features and result processing.
"""
import os
import json
import shutil
import asyncio
import subprocess
from typing import Dict, List, Any, Optional, Union

from tsap.utils.logging import logger
from tsap.config import get_config
from tsap.performance_mode import get_parameter
from tsap.core.base import BaseCoreTool, register_tool
from tsap.mcp.models import JqQueryParams, JqQueryResult


@register_tool("jq")
class JqTool(BaseCoreTool):
    """Interface to the jq command-line tool."""
    
    def __init__(self, executable_path: Optional[str] = None):
        """Initialize the jq tool.
        
        Args:
            executable_path: Optional path to jq executable
        """
        super().__init__("jq")
        
        # Find jq executable
        self.executable = executable_path or self._find_executable()
        
        if not self.executable:
            raise FileNotFoundError(
                "jq executable not found. Please install jq or specify the path."
            )
            
        # Verify the executable works
        self._verify_executable()
    
    def _find_executable(self) -> Optional[str]:
        """Find the jq executable in the system.
        
        Returns:
            Path to jq executable or None if not found
        """
        # Try config first
        config = get_config()
        if config.tools.jq_path:
            if os.path.isfile(config.tools.jq_path) and os.access(config.tools.jq_path, os.X_OK):
                return config.tools.jq_path
        
        # Try common names
        path = shutil.which("jq")
        if path:
            return path
                
        # Not found
        return None
    
    def _verify_executable(self) -> None:
        """Verify that the jq executable works.
        
        Raises:
            RuntimeError: If jq executable doesn't work
        """
        try:
            # Run a simple command to verify
            result = subprocess.run(
                [self.executable, "--version"],
                capture_output=True,
                text=True,
                check=False,
            )
            
            if result.returncode != 0:
                raise RuntimeError(
                    f"jq executable failed: {result.stderr.strip()}"
                )
                
            logger.debug(
                f"jq executable verified: {result.stdout.strip()}",
                component="core",
                operation="verify_jq"
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to verify jq executable: {str(e)}")
    
    def _build_command(self, params: JqQueryParams) -> List[str]:
        """Build the jq command from query parameters.
        
        Args:
            params: Query parameters
            
        Returns:
            List of command arguments
        """
        cmd = [self.executable]
        
        # Add options based on parameters
        if params.raw_output:
            cmd.append("-r")
            
        if params.compact_output:
            cmd.append("-c")
            
        if params.monochrome_output:
            cmd.append("-M")
        
        # Add the query
        cmd.append(params.query)
        
        # Add input files if specified
        if params.input_files:
            cmd.extend(params.input_files)
            
        return cmd
    
    async def _parse_output(
        self, output: str, raw_output: bool
    ) -> Union[str, List[Any], Dict[str, Any]]:
        """Parse the output from jq.
        
        Args:
            output: Output string from jq
            raw_output: Whether raw output was requested
            
        Returns:
            Parsed output as string, list, or dict
        """
        if raw_output:
            # Raw output is already in desired format
            return output.strip()
            
        # Parse as JSON
        try:
            # Handle multiple JSON objects (one per line)
            lines = output.strip().split("\n")
            results = []
            
            for line in lines:
                if line.strip():
                    results.append(json.loads(line))
                    
            # If only one result, return it directly
            if len(results) == 1:
                return results[0]
                
            # Otherwise, return the list of results
            return results
            
        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to parse jq output as JSON: {str(e)}",
                component="core",
                operation="jq_query",
                context={"output": output[:100] + "..." if len(output) > 100 else output}
            )
            
            # If parsing fails, return as string
            return output.strip()
    
    async def query(self, params: JqQueryParams) -> JqQueryResult:
        """Query JSON data using jq.
        
        Args:
            params: Query parameters
            
        Returns:
            Query results
        """
        async with self._measure_execution_time():
            start_time = asyncio.get_event_loop().time()
            
            # Build the command
            cmd = self._build_command(params)
            cmd_str = " ".join(str(arg) for arg in cmd)
            
            # Get timeout from performance mode
            timeout = get_parameter("timeout", 30.0)
            
            # Log the operation
            logger.info(
                f"Executing jq query: {params.query}",
                component="core",
                operation="jq_query",
                context={
                    "query": params.query,
                    "files": params.input_files,
                    "command": cmd_str,
                }
            )
            
            try:
                # Create a temporary file for input if needed
                input_data = None
                tempfile_path = None
                
                if params.input_json and not params.input_files:
                    # Input provided as string, use stdin
                    input_data = params.input_json
                    
                    # Log the input size
                    logger.debug(
                        f"Using input JSON from string ({len(input_data)} bytes)",
                        component="core",
                        operation="jq_query"
                    )
                
                # Execute the command
                returncode, stdout, stderr = await self._run_process(
                    cmd=cmd,
                    timeout=timeout,
                    input_data=input_data,
                )
                
                # Check for errors
                if returncode != 0:
                    raise RuntimeError(
                        f"jq query failed (exit code {returncode}): {stderr}"
                    )
                    
                # Parse the output
                parsed_output = await self._parse_output(stdout, params.raw_output)
                
                # Calculate execution time
                execution_time = asyncio.get_event_loop().time() - start_time
                
                # Determine if output was successfully parsed as JSON
                parsed_as_json = not isinstance(parsed_output, str) or params.raw_output
                
                # Log the result
                logger.success(
                    "jq query completed",
                    component="core",
                    operation="jq_query",
                    context={
                        "execution_time": execution_time,
                        "parsed_as_json": parsed_as_json,
                    }
                )
                
                # Create and return the result
                return JqQueryResult(
                    output=parsed_output,
                    parsed=parsed_as_json,
                    exit_code=returncode,
                    command=cmd_str,
                    execution_time=execution_time,
                )
                
            except asyncio.TimeoutError:
                # Log the timeout
                logger.warning(
                    f"jq query timed out after {timeout}s",
                    component="core",
                    operation="jq_query",
                    context={
                        "query": params.query,
                        "timeout": timeout,
                    }
                )
                
                # Create and return a timeout result
                return JqQueryResult(
                    output="Error: Query timed out",
                    parsed=False,
                    exit_code=124,  # Standard timeout exit code
                    command=cmd_str,
                    execution_time=timeout,
                )
                
            except Exception as e:
                # Log the error
                logger.error(
                    f"jq query failed: {str(e)}",
                    component="core",
                    operation="jq_query",
                    exception=e,
                    context={
                        "query": params.query,
                    }
                )
                
                # Create and return an error result
                return JqQueryResult(
                    output=f"Error: {str(e)}",
                    parsed=False,
                    exit_code=1,
                    command=cmd_str,
                    execution_time=asyncio.get_event_loop().time() - start_time,
                )
                
            finally:
                # Clean up temporary file if created
                if tempfile_path and os.path.exists(tempfile_path):
                    os.unlink(tempfile_path)


# Create a singleton instance
_jq_tool: Optional[JqTool] = None


def get_jq_tool() -> JqTool:
    """Get the singleton JqTool instance.
    
    Returns:
        JqTool instance
    """
    global _jq_tool
    
    if _jq_tool is None:
        try:
            _jq_tool = JqTool()
        except Exception as e:
            logger.error(
                f"Failed to initialize JqTool: {str(e)}",
                component="core",
                operation="init_jq",
                exception=e
            )
            raise
            
    return _jq_tool


async def jq_query(params: JqQueryParams) -> JqQueryResult:
    """Execute a jq query.
    
    This is a convenience function that uses the singleton JqTool instance.
    
    Args:
        params: Query parameters
        
    Returns:
        Query results
    """
    return await get_jq_tool().query(params)