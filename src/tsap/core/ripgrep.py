"""
Ripgrep integration for TSAP.

This module provides functionality to search files using the ripgrep command-line
tool, with enhanced features and result processing.
"""
import os
import re
import json
import asyncio
import shutil
import tempfile
import subprocess
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from pathlib import Path
import shlex

from tsap.utils.logging import logger
from tsap.config import get_config
from tsap.performance_mode import get_parameter
from tsap.core.base import BaseCoreTool
from tsap.mcp.protocol import RipgrepSearchParams
from tsap.mcp.models import RipgrepMatch, RipgrepSearchResult


class RipgrepTool(BaseCoreTool):
    """Interface to the ripgrep command-line tool."""
    
    def __init__(self, executable_path: Optional[str] = None):
        """Initialize the ripgrep tool.
        
        Args:
            executable_path: Optional path to ripgrep executable
        """
        super().__init__("ripgrep")
        
        # Find ripgrep executable
        self.executable = executable_path or self._find_executable()
        
        if not self.executable:
            raise FileNotFoundError(
                "Ripgrep executable not found. Please install ripgrep or specify the path."
            )
            
        # Verify the executable works
        self._verify_executable()
    
    def _find_executable(self) -> Optional[str]:
        """Find the ripgrep executable in the system.
        
        Returns:
            Path to ripgrep executable or None if not found
        """
        # Try config first
        config = get_config()
        if config.tools.ripgrep_path:
            if os.path.isfile(config.tools.ripgrep_path) and os.access(config.tools.ripgrep_path, os.X_OK):
                return config.tools.ripgrep_path
        
        # Try common names
        for name in ["rg", "ripgrep"]:
            path = shutil.which(name)
            if path:
                return path
                
        # Not found
        return None
    
    def _verify_executable(self) -> None:
        """Verify that the ripgrep executable works.
        
        Raises:
            RuntimeError: If ripgrep executable doesn't work
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
                    f"Ripgrep executable failed: {result.stderr.strip()}"
                )
                
            logger.debug(
                f"Ripgrep executable verified: {result.stdout.strip()}",
                component="core",
                operation="verify_ripgrep"
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to verify ripgrep executable: {str(e)}")
    
    def _build_command(self, params: RipgrepSearchParams) -> List[str]:
        """Build the ripgrep command from search parameters.
        
        Args:
            params: Search parameters
            
        Returns:
            List of command arguments
        """
        cmd = [self.executable]
        
        # Add options based on parameters
        if not params.case_sensitive:
            cmd.append("-i")
            
        if params.whole_word:
            cmd.append("-w")
            
        if not params.regex:
            cmd.append("-F")  # Fixed strings (not regex)
            
        if params.invert_match:
            cmd.append("-v")
            
        # Context lines
        if params.before_context is not None:
            cmd.extend(["-B", str(params.before_context)])
        elif params.after_context is not None:
            cmd.extend(["-A", str(params.after_context)])
        elif params.context_lines > 0:
            cmd.extend(["-C", str(params.context_lines)])
            
        # Max count per file
        if params.max_count is not None:
            cmd.extend(["-m", str(params.max_count)])
            
        # Max depth
        if params.max_depth is not None:
            cmd.extend(["--max-depth", str(params.max_depth)])
            
        # Other options
        if params.follow_symlinks:
            cmd.append("--follow")
            
        if params.hidden:
            cmd.append("--hidden")
            
        if params.no_ignore:
            cmd.append("--no-ignore")
            
        # Encoding
        if params.encoding:
            cmd.extend(["--encoding", params.encoding])
            
        # Binary files
        if not params.binary:
            cmd.append("--no-binary")
            
        # File patterns
        if params.file_patterns:
            for pattern in params.file_patterns:
                cmd.extend(["-g", pattern])
                
        # Exclude patterns
        if params.exclude_patterns:
            for pattern in params.exclude_patterns:
                cmd.extend(["-g", f"!{pattern}"])
                
        # Add JSON output for easier parsing
        cmd.append("--json")
        
        # Add pattern and paths
        cmd.append(params.pattern)
        cmd.extend(params.paths)
        
        return cmd

    async def _parse_json_output(
        self, output: str, max_total_matches: Optional[int] = None
    ) -> Tuple[List[RipgrepMatch], Dict[str, Any], bool]:
        """Parse JSON output from ripgrep.
        
        Args:
            output: JSON output from ripgrep
            max_total_matches: Maximum total matches to return
            
        Returns:
            Tuple of (matches, stats, truncated)
        """
        matches = []
        stats = {
            "files_searched": 0,
            "files_with_matches": 0,
            "total_matches": 0,
            "lines_searched": 0,
        }
        truncated = False
        
        # Track files with matches
        files_with_matches = set()
        
        # Process each line as a separate JSON object
        for line in output.splitlines():
            if not line.strip():
                continue
                
            try:
                data = json.loads(line)
                
                if data.get("type") == "match":
                    # Found a match
                    path = data.get("data", {}).get("path", {}).get("text", "")
                    if path:
                        files_with_matches.add(path)
                    
                    # Extract submatches
                    submatches = []
                    for m in data.get("data", {}).get("submatches", []):
                        submatches.append({
                            "match": m.get("match", {}).get("text", ""),
                            "start": m.get("start", 0),
                            "end": m.get("end", 0),
                        })
                    
                    # Create match object
                    match = RipgrepMatch(
                        path=path,
                        line_number=data.get("data", {}).get("line_number", 0),
                        column_number=data.get("data", {}).get("absolute_offset", None),
                        match_text=submatches[0]["match"] if submatches else "",
                        line_text=data.get("data", {}).get("lines", {}).get("text", ""),
                        before_context=[],
                        after_context=[],
                        submatches=submatches,
                    )
                    
                    matches.append(match)
                    stats["total_matches"] += 1
                    
                    # Check if we've reached the maximum total matches
                    if max_total_matches and stats["total_matches"] >= max_total_matches:
                        truncated = True
                        break
                        
                elif data.get("type") == "context":
                    # Process context lines
                    if not matches:
                        continue
                        
                    last_match = matches[-1]
                    context_data = data.get("data", {})
                    context_lines = context_data.get("lines", {}).get("text", "")
                    context_line_number = context_data.get("line_number", 0)
                    
                    if context_line_number < last_match.line_number:
                        # Before context
                        last_match.before_context.append(context_lines)
                    else:
                        # After context
                        last_match.after_context.append(context_lines)
                        
                elif data.get("type") == "summary":
                    # Process summary stats
                    stats.update({
                        "elapsed_total": data.get("data", {}).get("elapsed_total", {}).get("secs", 0),
                        "stats": data.get("data", {}).get("stats", {}),
                    })
                    
            except json.JSONDecodeError:
                logger.warning(
                    f"Failed to parse ripgrep JSON output: {line}",
                    component="core",
                    operation="ripgrep_search"
                )
        
        # Update stats
        stats["files_with_matches"] = len(files_with_matches)
        
        # Sort context lines by line number
        for match in matches:
            match.before_context.reverse()  # Context lines come in reverse order
        
        return matches, stats, truncated
    
    async def search(
        self, params: RipgrepSearchParams
    ) -> RipgrepSearchResult:
        """Search for patterns using ripgrep.
        
        Args:
            params: Search parameters
            
        Returns:
            Search results
        """
        start_time = asyncio.get_event_loop().time()
        
        # Build the command
        cmd = self._build_command(params)
        cmd_str = " ".join(shlex.quote(arg) for arg in cmd)
        
        # Get timeout from performance mode or parameters
        timeout = get_parameter("timeout", 30.0)
        
        # Log the operation
        logger.info(
            f"Executing ripgrep search: {params.pattern}",
            component="core",
            operation="ripgrep_search",
            context={
                "pattern": params.pattern,
                "paths": params.paths,
                "command": cmd_str,
            }
        )
        
        try:
            # Execute the command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=10 * 1024 * 1024,  # 10 MB buffer
            )
            
            # Wait for the process to complete with timeout
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
            
            output = stdout.decode("utf-8", errors="replace")
            error_output = stderr.decode("utf-8", errors="replace")
            
            # Check for errors
            if process.returncode != 0 and process.returncode != 1:
                # Return code 1 means no matches, which is not an error
                raise RuntimeError(
                    f"Ripgrep search failed (exit code {process.returncode}): {error_output}"
                )
                
            # Parse the output
            matches, stats, truncated = await self._parse_json_output(
                output, params.max_total_matches
            )
            
            # Calculate execution time
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Log the result
            logger.success(
                f"Ripgrep search completed: {len(matches)} matches",
                component="core",
                operation="ripgrep_search",
                context={
                    "matches_count": len(matches),
                    "execution_time": execution_time,
                    "truncated": truncated,
                }
            )
            
            # Create and return the result
            return RipgrepSearchResult(
                matches=matches,
                stats=stats,
                truncated=truncated,
                command=cmd_str,
                execution_time=execution_time,
            )
            
        except asyncio.TimeoutError:
            # Log the timeout
            logger.warning(
                f"Ripgrep search timed out after {timeout}s",
                component="core",
                operation="ripgrep_search",
                context={
                    "pattern": params.pattern,
                    "paths": params.paths,
                    "timeout": timeout,
                }
            )
            
            # Create and return a timeout result
            return RipgrepSearchResult(
                matches=[],
                stats={
                    "error": "timeout",
                    "timeout": timeout,
                },
                truncated=True,
                command=cmd_str,
                execution_time=timeout,
            )
            
        except Exception as e:
            # Log the error
            logger.error(
                f"Ripgrep search failed: {str(e)}",
                component="core",
                operation="ripgrep_search",
                exception=e,
                context={
                    "pattern": params.pattern,
                    "paths": params.paths,
                }
            )
            
            # Create and return an error result
            return RipgrepSearchResult(
                matches=[],
                stats={
                    "error": str(e),
                },
                truncated=False,
                command=cmd_str,
                execution_time=asyncio.get_event_loop().time() - start_time,
            )


# Create a singleton instance
_ripgrep_tool: Optional[RipgrepTool] = None


def get_ripgrep_tool() -> RipgrepTool:
    """Get the singleton RipgrepTool instance.
    
    Returns:
        RipgrepTool instance
    """
    global _ripgrep_tool
    
    if _ripgrep_tool is None:
        try:
            _ripgrep_tool = RipgrepTool()
        except Exception as e:
            logger.error(
                f"Failed to initialize RipgrepTool: {str(e)}",
                component="core",
                operation="init_ripgrep",
                exception=e
            )
            raise
            
    return _ripgrep_tool


async def ripgrep_search(params: RipgrepSearchParams) -> RipgrepSearchResult:
    """Perform a ripgrep search.
    
    This is a convenience function that uses the singleton RipgrepTool instance.
    
    Args:
        params: Search parameters
        
    Returns:
        Search results
    """
    return await get_ripgrep_tool().search(params)