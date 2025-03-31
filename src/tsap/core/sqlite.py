"""
SQLite integration for TSAP.

This module provides functionality to query and interact with SQLite databases,
with enhanced features for analysis and result processing.
"""
import os
import sqlite3
import asyncio
import shutil
import json
from typing import List, Any, Optional, Tuple
import traceback

from tsap.utils.logging import logger
from tsap.config import get_config
from tsap.performance_mode import get_parameter
from tsap.core.base import BaseCoreTool, register_tool
from tsap.mcp.models import SqliteQueryParams, SqliteQueryResult


@register_tool("sqlite")
class SqliteTool(BaseCoreTool):
    """Interface to the SQLite command-line and Python API."""
    
    def __init__(self, executable_path: Optional[str] = None):
        """Initialize the SQLite tool.
        
        Args:
            executable_path: Optional path to SQLite executable
        """
        super().__init__("sqlite")
        
        # Find SQLite executable for CLI operations
        self.executable = executable_path or self._find_executable()
        
        if not self.executable:
            logger.warning(
                "SQLite executable not found. Falling back to Python sqlite3 module.",
                component="core",
                operation="init_sqlite"
            )
    
    def _find_executable(self) -> Optional[str]:
        """Find the SQLite executable in the system.
        
        Returns:
            Path to SQLite executable or None if not found
        """
        # Try config first
        config = get_config()
        if config.tools.sqlite_path:
            if os.path.isfile(config.tools.sqlite_path) and os.access(config.tools.sqlite_path, os.X_OK):
                return config.tools.sqlite_path
        
        # Try common names
        for name in ["sqlite3", "sqlite"]:
            path = shutil.which(name)
            if path:
                return path
                
        # Not found
        return None
    
    async def _execute_query_python(
        self,
        database: str,
        query: str,
        params: Optional[List[Any]] = None,
        mode: str = "list",
    ) -> Tuple[List[Any], List[str]]:
        """Execute a query using the Python sqlite3 module.
        
        Args:
            database: Path to SQLite database
            query: SQL query to execute
            params: Query parameters
            mode: Result mode (list, dict, or table)
            
        Returns:
            Tuple of (rows, column_names)
            
        Raises:
            sqlite3.Error: If the query fails
        """
        # Check if this is a multi-statement query
        if ";" in query and not query.strip().endswith(";"):
            # This looks like a multi-statement query
            # For multi-statement queries, we need special handling
            return await self._execute_script_python(database, query, mode)
            
        # Wrap in a threadpool to avoid blocking
        loop = asyncio.get_event_loop()
        
        def run_query():
            conn = None
            try:
                conn = sqlite3.connect(database)
                
                # Configure connection
                conn.row_factory = sqlite3.Row if mode == "dict" else None
                
                # Execute the query
                cursor = conn.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                # Get column names
                column_names = [description[0] for description in cursor.description] if cursor.description else []
                
                # Fetch results
                if mode == "dict":
                    # Convert rows to dictionaries
                    rows = [dict(row) for row in cursor.fetchall()]
                else:
                    # Keep as tuples (list mode)
                    rows = cursor.fetchall()
                
                return rows, column_names
            finally:
                if conn:
                    conn.close()
        
        # Execute in a threadpool
        return await loop.run_in_executor(None, run_query)
        
    async def _execute_script_python(
        self,
        database: str,
        script: str,
        mode: str = "list",
    ) -> Tuple[List[Any], List[str]]:
        """Execute a multi-statement script using the Python sqlite3 module.
        
        Args:
            database: Path to SQLite database
            script: SQL script with multiple statements
            mode: Result mode (list, dict, or table)
            
        Returns:
            Tuple of (rows, column_names) from the last statement that returned rows
            
        Raises:
            sqlite3.Error: If the script fails
        """
        loop = asyncio.get_event_loop()
        
        def run_script():
            conn = None
            last_rows = []
            last_columns = []
            
            try:
                conn = sqlite3.connect(database)
                
                # Configure connection
                conn.row_factory = sqlite3.Row if mode == "dict" else None
                
                # Use executescript for proper multi-statement execution
                cursor = conn.cursor()
                cursor.executescript(script)
                
                # Check if the last operation returned rows
                if cursor.description:
                    last_columns = [desc[0] for desc in cursor.description]
                    
                    if mode == "dict":
                        last_rows = [dict(row) for row in cursor.fetchall()]
                    else:
                        last_rows = cursor.fetchall()
                
                # Commit any changes
                conn.commit()
                
                return last_rows, last_columns
                
            except Exception as e:
                if conn:
                    conn.rollback()
                logger.error(
                    f"Python SQLite script execution failed: {str(e)}\n{traceback.format_exc()}",
                    component="core",
                    operation="sqlite_script"
                )
                raise e
                
            finally:
                if conn:
                    conn.close()
        
        # Execute in a threadpool
        return await loop.run_in_executor(None, run_script)
    
    async def _execute_query_cli(
        self,
        database: str,
        query: str,
        params: Optional[List[Any]] = None,
        mode: str = "list",
    ) -> Tuple[List[Any], List[str]]:
        """Execute a query using the SQLite CLI.
        
        Args:
            database: Path to SQLite database
            query: SQL query to execute
            params: Query parameters
            mode: Result mode (list, dict, or table)
            
        Returns:
            Tuple of (rows, column_names)
            
        Raises:
            RuntimeError: If the CLI command fails
        """
        if not self.executable:
            raise RuntimeError("SQLite executable not found")

        # Prepare the SQL input string
        if params:
            # Replace parameters with literals (simplified)
            # In a real implementation, proper escaping would be needed
            temp_query = query
            for param in params:
                if isinstance(param, str):
                    temp_query = temp_query.replace("?", f"'{param}'", 1)
                else:
                    temp_query = temp_query.replace("?", str(param), 1)
        else:
            temp_query = query

        # Configure output format and add the query with proper newlines
        input_sql = f""".mode json
.headers on
{temp_query}
"""

        try:
            # Execute the command with SQL piped via stdin
            cmd = [self.executable, database]

            returncode, stdout, stderr = await self._run_process(
                cmd=cmd,
                timeout=get_parameter("timeout", 30.0),
                stdin_data=input_sql,  # Pass SQL via stdin
            )

            if returncode != 0:
                # Include stderr in the error message if available
                error_message = f"SQLite query failed (code {returncode})"
                if stderr:
                    error_message += f": {stderr.strip()}"
                raise RuntimeError(error_message)

            # Parse JSON output
            try:
                # Handle potentially empty stdout (e.g., for non-SELECT statements)
                if not stdout.strip():
                    return [], []

                results = json.loads(stdout)

                # Extract column names from the first row
                column_names = list(results[0].keys()) if results else []

                if mode == "dict":
                    # Already in dict format
                    rows = results
                else:
                    # Convert to list format
                    rows = [tuple(row.values()) for row in results]

                return rows, column_names
            except json.JSONDecodeError as e:
                # If JSON parsing fails, return the raw output or raise an error
                logger.warning(f"Failed to parse SQLite CLI JSON output: {e}. Output: {stdout[:100]}...")
                # Consider raising an error or returning a structured error format
                return [(stdout,)], ["output"]
            except IndexError:
                 # Handle cases where results might be empty or not structured as expected
                logger.warning(f"Could not extract column names from CLI output: {stdout[:100]}...")
                return [], []
        except asyncio.TimeoutError:
             logger.error(f"SQLite CLI command timed out: {' '.join(cmd)}")
             raise RuntimeError("SQLite query via CLI timed out.")
        except Exception as e:
             logger.error(f"Error executing SQLite CLI query: {e}")
             raise RuntimeError(f"Failed to execute query via SQLite CLI: {e}")

    async def _run_process(self, cmd: List[str], timeout: float, stdin_data: Optional[str] = None) -> Tuple[int, str, str]:
        """Run a subprocess and capture its output, optionally providing stdin.
        
        Args:
            cmd: Command to run as a list of arguments
            timeout: Timeout in seconds
            stdin_data: Optional string data to pass to the process's stdin
            
        Returns:
            Tuple of (returncode, stdout, stderr)
            
        Raises:
            asyncio.TimeoutError: If the process times out
        """
        # Determine stdin configuration
        stdin_pipe = asyncio.subprocess.PIPE if stdin_data is not None else None

        # Create the process
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=stdin_pipe,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            # Prepare input data if provided
            input_bytes = stdin_data.encode() if stdin_data is not None else None

            # Wait for the process to complete with timeout
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(input=input_bytes), timeout=timeout
            )

            # Return the process results
            return (
                process.returncode,
                stdout_bytes.decode("utf-8", errors="replace"),
                stderr_bytes.decode("utf-8", errors="replace"),
            )
        except asyncio.TimeoutError:
            # Kill the process if it times out
            try:
                logger.warning(f"Process {cmd} timed out after {timeout}s. Killing.")
                process.kill()
                # Attempt to communicate again to clean up resources
                await process.communicate()
            except ProcessLookupError:
                # Process already finished
                pass
            except Exception as kill_exc:
                 logger.error(f"Error during process kill/cleanup: {kill_exc}")

            # Re-raise the timeout error
            raise
        except Exception as e:
            logger.error(f"Error during subprocess execution {' '.join(cmd)}: {e}")
            # Ensure process cleanup if communication failed unexpectedly
            if process.returncode is None:
                 try:
                     process.kill()
                     await process.communicate()
                 except Exception as cleanup_exc:
                     logger.error(f"Error during emergency process cleanup: {cleanup_exc}")
            raise # Re-raise the original error
    
    async def query(self, params: SqliteQueryParams) -> SqliteQueryResult:
        """Execute a SQLite query.
        
        Args:
            params: Query parameters
            
        Returns:
            Query results
        """
        async with self._measure_execution_time():
            start_time = asyncio.get_event_loop().time()
            
            # Log the operation
            logger.info(
                f"Executing SQLite query on database {params.database}",
                component="core",
                operation="sqlite_query",
                context={
                    "database": params.database,
                    "query": params.query[:100] + "..." if len(params.query) > 100 else params.query,
                }
            )
            
            try:
                # Check if database exists
                if not os.path.isfile(params.database):
                    raise FileNotFoundError(f"Database file not found: {params.database}")
                
                # Check if this is a multi-statement query
                is_multi_statement = ";" in params.query and not params.query.strip().endswith(";")
                has_problematic_pragma = "PRAGMA" in params.query.upper()
                
                # If this is clearly a multi-statement script, use execute_script directly
                if is_multi_statement:
                    try:
                        # Use the Python script execution method for multi-statement queries
                        rows, columns = await self._execute_script_python(
                            database=params.database,
                            script=params.query,
                            mode=params.mode,
                        )
                        logger.info(f"SQLite script execution completed: {len(rows)} rows", component="core", operation="sqlite_query")
                    except Exception as script_error:
                        logger.error(f"SQLite script execution failed: {script_error}\n{traceback.format_exc()}", component="core", operation="sqlite_query")
                        if self.executable:
                            try:
                                # Try CLI as a fallback
                                rows, columns = await self._execute_query_cli(
                                    database=params.database,
                                    query=params.query,
                                    params=params.params,
                                    mode=params.mode,
                                )
                                logger.info(f"SQLite CLI fallback completed: {len(rows)} rows", component="core", operation="sqlite_query")
                            except Exception as cli_error:
                                logger.error(f"Both Python script and CLI SQLite implementations failed. Script Error: {script_error}, CLI Error: {cli_error}\n{traceback.format_exc()}", component="core", operation="sqlite_query")
                                raise RuntimeError(f"SQLite execution failed. Script: {script_error}, CLI: {cli_error}") from cli_error
                        else:
                            # No CLI available
                            raise RuntimeError(f"SQLite script execution failed: {script_error}") from script_error
                # For PRAGMA queries that aren't multi-statement, try CLI first (some PRAGMAs work better with CLI)
                elif has_problematic_pragma and self.executable:
                    try:
                        # Try CLI implementation first for problematic queries
                        rows, columns = await self._execute_query_cli(
                            database=params.database,
                            query=params.query,
                            params=params.params,
                            mode=params.mode,
                        )
                        logger.info(f"SQLite CLI query completed successfully: {len(rows)} rows", component="core", operation="sqlite_query")
                    except Exception as cli_error:
                        # Use logger.error here to include traceback for the CLI failure
                        logger.error(f"SQLite CLI execution failed: {cli_error}. Falling back to Python.\n{traceback.format_exc()}", component="core", operation="sqlite_query")
                        # Fallback to Python implementation
                        try:
                            rows, columns = await self._execute_query_python(
                                database=params.database,
                                query=params.query,
                                params=params.params,
                                mode=params.mode,
                            )
                            logger.info(f"SQLite Python fallback completed: {len(rows)} rows", component="core", operation="sqlite_query")
                        except Exception as py_error:
                            # Both methods failed, use logger.error
                            logger.error(f"Both CLI and Python SQLite implementations failed. CLI Error: {cli_error}, Python Error: {py_error}\n{traceback.format_exc()}", component="core", operation="sqlite_query")
                            raise RuntimeError(f"SQLite execution failed. CLI: {cli_error}, Python: {py_error}") from py_error
                else: # Try Python first for standard queries
                    try:
                        # Try Python implementation first for normal queries
                        rows, columns = await self._execute_query_python(
                            database=params.database,
                            query=params.query,
                            params=params.params,
                            mode=params.mode,
                        )
                        logger.info(f"SQLite Python query completed: {len(rows)} rows", component="core", operation="sqlite_query")
                    except sqlite3.Error as py_error:
                        logger.warning(f"Python SQLite implementation failed: {py_error}. Falling back to CLI.", component="core", operation="sqlite_query")
                        if self.executable:
                            try:
                                rows, columns = await self._execute_query_cli(
                                    database=params.database,
                                    query=params.query,
                                    params=params.params,
                                    mode=params.mode,
                                )
                                logger.info(f"SQLite CLI fallback completed: {len(rows)} rows", component="core", operation="sqlite_query")
                            except Exception as cli_error:
                                # Both methods failed, use logger.error with traceback
                                logger.error(f"Both Python and CLI SQLite implementations failed. Python Error: {py_error}, CLI Error: {cli_error}\n{traceback.format_exc()}", component="core", operation="sqlite_query")
                                raise RuntimeError(f"SQLite execution failed. Python: {py_error}, CLI: {cli_error}") from cli_error
                        else:
                            # Python failed and CLI is unavailable, use logger.error with traceback
                            logger.error(f"Python SQLite failed and CLI unavailable. Python Error: {py_error}\n{traceback.format_exc()}", component="core", operation="sqlite_query")
                            raise RuntimeError(f"SQLite execution failed (Python): {py_error}. CLI unavailable.") from py_error
                    except Exception as py_error: # Catch other potential Python errors
                        # Unexpected Python errors, use logger.error with traceback
                        logger.error(f"Unexpected error in Python SQLite execution: {py_error}\n{traceback.format_exc()}", component="core", operation="sqlite_query")
                        raise # Re-raise unexpected errors

                # Calculate execution time
                execution_time = asyncio.get_event_loop().time() - start_time
                
                # Log success
                logger.success(
                    f"SQLite query completed: {len(rows)} rows",
                    component="core",
                    operation="sqlite_query",
                    context={
                        "row_count": len(rows),
                        "execution_time": execution_time,
                    }
                )
                
                # Create and return result
                return SqliteQueryResult(
                    rows=rows,
                    columns=columns if params.headers else None,
                    row_count=len(rows),
                    execution_time=execution_time,
                )
                
            except Exception as e:
                logger.error(
                    f"SQLite query failed: {str(e)}\n{traceback.format_exc()}",
                    component="core",
                    operation="sqlite_query"
                )
                # Remove unused status and error_message variables since they aren't used
                
                # Calculate execution time
                execution_time = asyncio.get_event_loop().time() - start_time
                
                # Log error
                logger.error(
                    f"SQLite query failed after {execution_time:.2f}s: {str(e)}",
                    component="core",
                    operation="sqlite_query",
                    context={
                        "database": params.database,
                        "query": params.query[:100] + "..." if len(params.query) > 100 else params.query,
                    }
                )
                
                # Create and return error result
                return SqliteQueryResult(
                    rows=[],
                    columns=None,
                    row_count=0,
                    execution_time=execution_time,
                )


# Create a singleton instance
_sqlite_tool: Optional[SqliteTool] = None


def get_sqlite_tool() -> SqliteTool:
    """Get the singleton SqliteTool instance.
    
    Returns:
        SqliteTool instance
    """
    global _sqlite_tool
    
    if _sqlite_tool is None:
        try:
            _sqlite_tool = SqliteTool()
        except Exception as e:
            logger.error(
                f"Failed to initialize SqliteTool: {str(e)}",
                component="core",
                operation="init_sqlite",
                exception=e
            )
            raise
            
    return _sqlite_tool


async def sqlite_query(params: SqliteQueryParams) -> SqliteQueryResult:
    """Execute a SQLite query.
    
    This is a convenience function that uses the singleton SqliteTool instance.
    
    Args:
        params: Query parameters
        
    Returns:
        Query results
    """
    return await get_sqlite_tool().query(params)