"""
SQLite integration for TSAP.

This module provides functionality to query and interact with SQLite databases,
with enhanced features for analysis and result processing.
"""
import os
import sqlite3
import asyncio
import tempfile
import shutil
import json
from typing import List, Any, Optional, Tuple

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
            
        # Create a temporary script file for the query
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as script_file:
            script_path = script_file.name
            
            # Write query to script
            if params:
                # Replace parameters with literals (simplified)
                # In a real implementation, proper escaping would be needed
                for param in params:
                    if isinstance(param, str):
                        query = query.replace("?", f"'{param}'", 1)
                    else:
                        query = query.replace("?", str(param), 1)
            
            # Configure output format
            script_file.write(".mode json\n")  # Always use JSON for parsing
            script_file.write(".headers on\n")
            
            # Add the query
            script_file.write(f"{query};\n")
        
        try:
            # Execute the script
            cmd = [self.executable, database, "-init", script_path]
            
            returncode, stdout, stderr = await self._run_process(
                cmd=cmd,
                timeout=get_parameter("timeout", 30.0),
            )
            
            if returncode != 0:
                raise RuntimeError(f"SQLite query failed: {stderr}")
                
            # Parse JSON output
            try:
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
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw output
                return [(stdout,)], ["output"]
        finally:
            # Clean up the temporary script
            if os.path.exists(script_path):
                os.unlink(script_path)
    
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
                "Executing SQLite query",
                component="core",
                operation="sqlite_query",
                context={
                    "database": params.database,
                    "query": params.query,
                }
            )
            
            try:
                # Check if database exists
                if not os.path.isfile(params.database):
                    raise FileNotFoundError(f"Database file not found: {params.database}")
                
                # Determine execution method
                use_cli = self.executable is not None
                
                # Execute the query
                if use_cli:
                    # Use CLI
                    rows, columns = await self._execute_query_cli(
                        database=params.database,
                        query=params.query,
                        params=params.params,
                        mode=params.mode,
                    )
                else:
                    # Use Python module
                    rows, columns = await self._execute_query_python(
                        database=params.database,
                        query=params.query,
                        params=params.params,
                        mode=params.mode,
                    )
                
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
                # Log error
                logger.error(
                    f"SQLite query failed: {str(e)}",
                    component="core",
                    operation="sqlite_query",
                    exception=e,
                    context={
                        "database": params.database,
                        "query": params.query,
                    }
                )
                
                # Calculate execution time
                execution_time = asyncio.get_event_loop().time() - start_time
                
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