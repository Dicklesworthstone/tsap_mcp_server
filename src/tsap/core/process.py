"""
Process execution and management for TSAP.

This module provides functionality for executing external processes,
managing timeouts, capturing output, and handling errors.
"""
import os
import asyncio
import subprocess
from typing import Dict, List, Optional
import shlex
import time
from dataclasses import dataclass

from tsap.utils.logging import logger
from tsap.performance_mode import get_parameter


@dataclass
class ProcessResult:
    """Result of a process execution."""
    
    command: List[str]
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    timed_out: bool = False
    
    @property
    def succeeded(self) -> bool:
        """Check if the process succeeded (exit code 0).
        
        Returns:
            Whether the process succeeded
        """
        return self.exit_code == 0
    
    @property
    def failed(self) -> bool:
        """Check if the process failed (non-zero exit code).
        
        Returns:
            Whether the process failed
        """
        return not self.succeeded and not self.timed_out
    
    @property
    def command_str(self) -> str:
        """Get the command as a string.
        
        Returns:
            Command string
        """
        return " ".join(shlex.quote(str(arg)) for arg in self.command)


async def run_process(
    command: List[str],
    input_data: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
    timeout: Optional[float] = None,
    log_command: bool = True,
    log_output: bool = True,
    check: bool = False,
    operation: str = "run_process",
    component: str = "core",
) -> ProcessResult:
    """Run a subprocess asynchronously.
    
    Args:
        command: Command to run as a list of arguments
        input_data: Optional input data to pass to process stdin
        env: Optional environment variables
        cwd: Optional working directory
        timeout: Optional timeout in seconds
        log_command: Whether to log the command
        log_output: Whether to log the output
        check: Whether to raise an exception on non-zero exit code
        operation: Operation name for logging
        component: Component name for logging
        
    Returns:
        Process result
        
    Raises:
        asyncio.TimeoutError: If the process times out
        subprocess.SubprocessError: If check=True and the process fails
    """
    # Use default timeout from performance mode if not specified
    if timeout is None:
        timeout = get_parameter("timeout", 30.0)
    
    # Create environment with parent process environment plus any overrides
    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    
    # Log the command
    if log_command:
        cmd_str = " ".join(shlex.quote(str(arg)) for arg in command)
        logger.debug(
            f"Running command: {cmd_str}",
            component=component,
            operation=operation,
            context={"command": cmd_str}
        )
    
    start_time = time.time()
    timed_out = False
    process = None
    
    try:
        # Create and start the process
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE if input_data else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=process_env,
            cwd=cwd,
        )
        
        # Communicate with the process (with timeout)
        input_bytes = input_data.encode() if input_data else None
        stdout, stderr = await asyncio.wait_for(
            process.communicate(input_bytes), timeout=timeout
        )
        
        # Decode output
        stdout_str = stdout.decode("utf-8", errors="replace") if stdout else ""
        stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Log output if requested
        if log_output and stdout_str and not stderr_str:
            logger.debug(
                f"Command output: {len(stdout_str)} chars",
                component=component,
                operation=operation,
                context={"output_length": len(stdout_str)}
            )
        elif log_output and stderr_str:
            logger.warning(
                f"Command stderr: {stderr_str}",
                component=component,
                operation=operation,
                context={"error": stderr_str}
            )
        
        # Check exit code if requested
        if check and process.returncode != 0:
            error_msg = f"Command failed with exit code {process.returncode}: {stderr_str}"
            raise subprocess.SubprocessError(error_msg)
        
        # Create and return result
        return ProcessResult(
            command=command,
            exit_code=process.returncode,
            stdout=stdout_str,
            stderr=stderr_str,
            execution_time=execution_time,
            timed_out=False,
        )
        
    except asyncio.TimeoutError:
        # Process timed out, try to kill it
        if process:
            try:
                process.kill()
                await process.wait()
            except Exception:
                pass
        
        timed_out = True  # noqa: F841
        execution_time = time.time() - start_time
        
        # Log the timeout
        logger.warning(
            f"Command timed out after {timeout}s",
            component=component,
            operation=operation,
            context={"timeout": timeout}
        )
        
        # Create and return timeout result
        return ProcessResult(
            command=command,
            exit_code=124,  # Standard timeout exit code
            stdout="",
            stderr=f"Process timed out after {timeout} seconds",
            execution_time=execution_time,
            timed_out=True,
        )
        
    except Exception as e:
        # Process failed for other reasons
        if process:
            try:
                process.kill()
                await process.wait()
            except Exception:
                pass
        
        execution_time = time.time() - start_time
        
        # Log the error
        logger.error(
            f"Command failed: {str(e)}",
            component=component,
            operation=operation,
            exception=e,
        )
        
        # Reraise if check=True
        if check:
            raise
        
        # Create and return error result
        return ProcessResult(
            command=command,
            exit_code=1,
            stdout="",
            stderr=str(e),
            execution_time=execution_time,
            timed_out=False,
        )


async def run_pipeline(
    commands: List[List[str]],
    input_data: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
    timeout: Optional[float] = None,
    log_commands: bool = True,
    log_output: bool = True,
    check: bool = False,
    operation: str = "run_pipeline",
    component: str = "core",
) -> ProcessResult:
    """Run a pipeline of processes (like shell pipes).
    
    Args:
        commands: List of commands to run in pipeline
        input_data: Optional input data to pass to first process stdin
        env: Optional environment variables
        cwd: Optional working directory
        timeout: Optional timeout in seconds
        log_commands: Whether to log the commands
        log_output: Whether to log the output
        check: Whether to raise an exception on non-zero exit code
        operation: Operation name for logging
        component: Component name for logging
        
    Returns:
        Process result for the last process in the pipeline
        
    Raises:
        asyncio.TimeoutError: If the pipeline times out
        subprocess.SubprocessError: If check=True and any process fails
    """
    # Use default timeout from performance mode if not specified
    if timeout is None:
        timeout = get_parameter("timeout", 30.0)
    
    # Log the pipeline
    if log_commands:
        pipeline_str = " | ".join(
            " ".join(shlex.quote(str(arg)) for arg in cmd)
            for cmd in commands
        )
        logger.debug(
            f"Running pipeline: {pipeline_str}",
            component=component,
            operation=operation,
            context={"pipeline": pipeline_str}
        )
    
    # Use asyncio.create_subprocess_exec with pipes to connect processes
    processes = []
    start_time = time.time()
    
    try:
        # Create all processes in the pipeline
        for i, command in enumerate(commands):
            stdin = None
            stdout = None
            
            if i == 0:
                # First process gets input_data if provided
                stdin = asyncio.subprocess.PIPE if input_data else None
            else:
                # Other processes get stdin from previous process
                stdin = processes[i-1].stdout
            
            if i == len(commands) - 1:
                # Last process captures stdout
                stdout = asyncio.subprocess.PIPE
            else:
                # Other processes pipe stdout to next process
                stdout = asyncio.subprocess.PIPE
            
            # Create the process
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=stdin,
                stdout=stdout,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=cwd,
            )
            
            processes.append(process)
        
        # Send input to first process if provided
        first_process = processes[0]
        last_process = processes[-1]  # noqa: F841
        
        # Set up task for first process stdin
        stdin_task = None
        if input_data and first_process.stdin:
            stdin_task = asyncio.create_task(
                first_process.stdin.write(input_data.encode())
            )
        
        # Wait for all processes with timeout
        async def wait_for_all():
            # Wait for stdin task
            if stdin_task:
                await stdin_task
                first_process.stdin.close()
            
            # Wait for all processes
            results = []
            for process in processes:
                stdout, stderr = await process.communicate()
                results.append((process.returncode, stdout, stderr))
            return results
        
        # Run all processes with timeout
        results = await asyncio.wait_for(wait_for_all(), timeout=timeout)
        
        # Get last process output
        last_returncode, last_stdout, last_stderr = results[-1]
        
        # Decode output
        stdout_str = last_stdout.decode("utf-8", errors="replace") if last_stdout else ""
        stderr_str = last_stderr.decode("utf-8", errors="replace") if last_stderr else ""
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Check exit code if requested
        if check and any(code != 0 for code, _, _ in results):
            # Find the first failing process
            for i, (code, _, stderr) in enumerate(results):
                if code != 0:
                    stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""
                    error_msg = f"Process {i} failed with exit code {code}: {stderr_str}"
                    raise subprocess.SubprocessError(error_msg)
        
        # Create and return result for the last process
        return ProcessResult(
            command=commands[-1],
            exit_code=last_returncode,
            stdout=stdout_str,
            stderr=stderr_str,
            execution_time=execution_time,
            timed_out=False,
        )
        
    except asyncio.TimeoutError:
        # Pipeline timed out, kill all processes
        for process in processes:
            try:
                process.kill()
                await process.wait()
            except Exception:
                pass
        
        execution_time = time.time() - start_time
        
        # Log the timeout
        logger.warning(
            f"Pipeline timed out after {timeout}s",
            component=component,
            operation=operation,
            context={"timeout": timeout}
        )
        
        # Create and return timeout result
        return ProcessResult(
            command=commands[-1] if commands else [],
            exit_code=124,  # Standard timeout exit code
            stdout="",
            stderr=f"Pipeline timed out after {timeout} seconds",
            execution_time=execution_time,
            timed_out=True,
        )
        
    except Exception as e:
        # Pipeline failed for other reasons, kill all processes
        for process in processes:
            try:
                process.kill()
                await process.wait()
            except Exception:
                pass
        
        execution_time = time.time() - start_time
        
        # Log the error
        logger.error(
            f"Pipeline failed: {str(e)}",
            component=component,
            operation=operation,
            exception=e,
        )
        
        # Reraise if check=True
        if check:
            raise
        
        # Create and return error result
        return ProcessResult(
            command=commands[-1] if commands else [],
            exit_code=1,
            stdout="",
            stderr=str(e),
            execution_time=execution_time,
            timed_out=False,
        )