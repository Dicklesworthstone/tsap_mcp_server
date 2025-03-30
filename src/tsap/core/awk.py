"""
AWK integration for TSAP.

This module provides functionality to process text using the AWK command-line
tool, with enhanced features and result processing.
"""
import os
import shutil
import asyncio
import subprocess
from typing import List, Optional, Tuple

from tsap.utils.logging import logger
from tsap.config import get_config
from tsap.performance_mode import get_parameter
from tsap.core.base import BaseCoreTool, register_tool
from tsap.mcp.models import AwkProcessParams, AwkProcessResult


@register_tool("awk")
class AwkTool(BaseCoreTool):
    """Interface to the AWK command-line tool."""
    
    def __init__(self, executable_path: Optional[str] = None):
        """Initialize the AWK tool.
        
        Args:
            executable_path: Optional path to AWK executable
        """
        super().__init__("awk")
        
        # Find AWK executable
        self.executable = executable_path or self._find_executable()
        
        if not self.executable:
            raise FileNotFoundError(
                "AWK executable not found. Please install AWK or specify the path."
            )
            
        # Verify the executable works
        self._verify_executable()
    
    def _find_executable(self) -> Optional[str]:
        """Find the AWK executable in the system.
        
        Returns:
            Path to AWK executable or None if not found
        """
        # Try config first
        config = get_config()
        if config.tools.awk_path:
            if os.path.isfile(config.tools.awk_path) and os.access(config.tools.awk_path, os.X_OK):
                return config.tools.awk_path
        
        # Try common names
        for name in ["awk", "gawk", "mawk"]:
            path = shutil.which(name)
            if path:
                return path
                
        # Not found
        return None
    
    def _verify_executable(self) -> None:
        """Verify that the AWK executable works.
        
        Raises:
            RuntimeError: If AWK executable doesn't work
        """
        try:
            # Run a simple command to verify
            result = subprocess.run(
                [self.executable, "--version"],
                capture_output=True,
                text=True,
                check=False,
            )
            
            # Note: Different AWK implementations might behave differently
            # Some might print version to stdout, some to stderr
            # Some might return non-zero for --version
            
            if result.stdout or result.stderr:
                # Executable produced some output, consider it working
                version_info = result.stdout or result.stderr
                logger.debug(
                    f"AWK executable verified: {version_info.strip()}",
                    component="core",
                    operation="verify_awk"
                )
                return
                
            # No output from version command, try a simple echo
            result = subprocess.run(
                [self.executable, "BEGIN { print \"AWK is working\" }"],
                capture_output=True,
                text=True,
                check=False,
            )
            
            if result.returncode != 0 or "AWK is working" not in result.stdout:
                raise RuntimeError(
                    f"AWK executable failed simple test: {result.stderr.strip()}"
                )
                
        except Exception as e:
            raise RuntimeError(f"Failed to verify AWK executable: {str(e)}")
    
    def _build_command(self, params: AwkProcessParams) -> List[str]:
        """Build the AWK command from process parameters.
        
        Args:
            params: Process parameters
            
        Returns:
            List of command arguments
        """
        cmd = [self.executable]
        
        # Add field separator if specified
        if params.field_separator:
            cmd.extend(["-F", params.field_separator])
            
        # Add output field separator if specified
        if params.output_field_separator:
            cmd.extend(["-v", f"OFS={params.output_field_separator}"])
            
        # Add variables if specified
        if params.variables:
            for name, value in params.variables.items():
                cmd.extend(["-v", f"{name}={value}"])
        
        # Add the script
        cmd.append(params.script)
        
        # Add input files if specified
        if params.input_files:
            cmd.extend(params.input_files)
            
        return cmd
    
    async def _run_process(
        self, cmd: List[str], timeout: float, input_data: Optional[str] = None
    ) -> Tuple[int, str, str]:
        """Run the AWK process.

        Args:
            cmd: The command list to execute.
            timeout: Timeout in seconds.
            input_data: Optional string data to pass to stdin.

        Returns:
            Tuple of (returncode, stdout, stderr)
        """
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=10 * 1024 * 1024,  # 10 MB buffer
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(input=input_data.encode() if input_data else None),
                timeout=timeout,
            )
            returncode = process.returncode
        except asyncio.TimeoutError:
            try:
                process.terminate()
                await asyncio.wait_for(process.wait(), timeout=1.0) # Allow time for termination
            except asyncio.TimeoutError:
                process.kill()
            await process.wait() # Ensure process is cleaned up
            raise asyncio.TimeoutError # Re-raise the original timeout

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        return returncode, stdout, stderr

    async def process(self, params: dict) -> AwkProcessResult:
        """Process text using AWK.
        
        Args:
            params: Process parameters as a dictionary
            
        Returns:
            Process results
        """
        # Convert dict to AwkProcessParams object
        try:
            awk_params = AwkProcessParams(**params)
        except Exception as e:
            logger.error(
                f"Failed to parse AWK parameters: {str(e)}",
                component="core",
                operation="awk_process",
                exception=e,
                context={"raw_params": params}
            )
            # Return an error result if parsing fails
            return AwkProcessResult(
                output=f"Error parsing parameters: {str(e)}",
                exit_code=1,
                command="N/A (Parameter Parsing Error)",
                execution_time=0
            )
            
        start_time = asyncio.get_event_loop().time()
        
        # Build the command using the AwkProcessParams object
        cmd = self._build_command(awk_params)
        cmd_str = " ".join(str(arg) for arg in cmd)
        
        # Get timeout from performance mode
        timeout = get_parameter("timeout", 30.0)
        
        # Log the operation
        logger.info(
            "Executing AWK process",
            component="core",
            operation="awk_process",
            context={
                "script": awk_params.script,
                "files": awk_params.input_files,
                "command": cmd_str,
            }
        )
        
        try:
            # Create a temporary file for input if needed
            input_data = None
            tempfile_path = None
            
            # Use awk_params for checks and access
            if awk_params.input_text and not awk_params.input_files:
                # Input provided as string, use stdin
                input_data = awk_params.input_text
                
                # Log the input size
                logger.debug(
                    f"Using input text from string ({len(input_data)} chars)",
                    component="core",
                    operation="awk_process"
                )
            
            # Execute the command
            returncode, stdout, stderr = await self._run_process(
                cmd=cmd,
                timeout=timeout,
                input_data=input_data,
            )
            
            # Calculate execution time
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Log the result
            if returncode == 0:
                logger.success(
                    "AWK process completed",
                    component="core",
                    operation="awk_process",
                    context={
                        "execution_time": execution_time,
                        "output_size": len(stdout),
                    }
                )
            else:
                logger.error(
                    f"AWK process failed (exit code {returncode}): {stderr}",
                    component="core",
                    operation="awk_process",
                    context={
                        "execution_time": execution_time,
                        "error": stderr,
                    }
                )
            
            # Create and return the result
            return AwkProcessResult(
                output=stdout,
                exit_code=returncode,
                command=cmd_str,
                execution_time=execution_time,
            )
            
        except asyncio.TimeoutError:
            # Log the timeout
            logger.warning(
                f"AWK process timed out after {timeout}s",
                component="core",
                operation="awk_process",
                context={
                    "script": awk_params.script,
                    "timeout": timeout,
                }
            )
            
            # Create and return a timeout result
            return AwkProcessResult(
                output=f"Error: Process timed out after {timeout} seconds",
                exit_code=124,  # Standard timeout exit code
                command=cmd_str,
                execution_time=timeout,
            )
            
        except Exception as e:
            # Log the error
            logger.error(
                f"AWK process failed: {str(e)}",
                component="core",
                operation="awk_process",
                exception=e,
                context={
                    "script": awk_params.script,
                }
            )
            
            # Create and return an error result
            return AwkProcessResult(
                output=f"Error: {str(e)}",
                exit_code=1,
                command=cmd_str,
                execution_time=asyncio.get_event_loop().time() - start_time,
            )
            
        finally:
            # Clean up temporary file if created
            if tempfile_path and os.path.exists(tempfile_path):
                os.unlink(tempfile_path)


# Create a singleton instance
_awk_tool: Optional[AwkTool] = None


def get_awk_tool() -> AwkTool:
    """Get the singleton AwkTool instance.
    
    Returns:
        AwkTool instance
    """
    global _awk_tool
    
    if _awk_tool is None:
        try:
            _awk_tool = AwkTool()
        except Exception as e:
            logger.error(
                f"Failed to initialize AwkTool: {str(e)}",
                component="core",
                operation="init_awk",
                exception=e
            )
            raise
            
    return _awk_tool


async def awk_process(params: dict) -> AwkProcessResult:
    """Process text using AWK.
    
    This is a convenience function that uses the singleton AwkTool instance.
    
    Args:
        params: Process parameters
        
    Returns:
        Process results
    """
    return await get_awk_tool().process(params)