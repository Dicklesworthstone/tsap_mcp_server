"""
Project command history management for TSAP.

This module provides functionality for recording, retrieving, and managing the
history of commands and operations executed within a project session.
"""

import time
import uuid
import json
import threading
import sqlite3
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Tuple, Generator

import tsap.utils.logging as logging
from tsap.utils.errors import TSAPError
from tsap.project.context import get_project, ProjectContext


class HistoryError(TSAPError):
    """Exception raised for history-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.details = details or {}


class CommandEntry:
    """Represents a single command in the history."""
    
    def __init__(self, command_id: Optional[str] = None,
                command_type: Optional[str] = None,
                command: Optional[Dict] = None,
                parameters: Optional[Dict] = None,
                result: Optional[Any] = None,
                status: str = "pending",
                tags: Optional[List[str]] = None,
                notes: Optional[str] = None):
        """
        Initialize a command entry.
        
        Args:
            command_id: Unique ID for the command
            command_type: Type of command (e.g., "search", "analyze")
            command: The command itself (structure depends on type)
            parameters: Parameters for the command
            result: Result of the command execution
            status: Status of the command
            tags: Optional tags for organization/filtering
            notes: Optional notes about the command
        """
        self.command_id = command_id or str(uuid.uuid4())
        self.command_type = command_type
        self.command = command or {}
        self.parameters = parameters or {}
        self.result = result
        self.status = status
        self.tags = tags or []
        self.notes = notes
        
        self.created_at = time.time()
        self.updated_at = time.time()
        self.executed_at = None
        self.execution_time = None
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert command entry to a dictionary.
        
        Returns:
            Dictionary representation of the command entry
        """
        return {
            'command_id': self.command_id,
            'command_type': self.command_type,
            'command': self.command,
            'parameters': self.parameters,
            'result': self.result,
            'status': self.status,
            'tags': self.tags,
            'notes': self.notes,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'executed_at': self.executed_at,
            'execution_time': self.execution_time
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CommandEntry':
        """
        Create a command entry from a dictionary.
        
        Args:
            data: Dictionary representation of a command entry
            
        Returns:
            New CommandEntry instance
        """
        entry = cls(
            command_id=data.get('command_id'),
            command_type=data.get('command_type'),
            command=data.get('command'),
            parameters=data.get('parameters'),
            result=data.get('result'),
            status=data.get('status'),
            tags=data.get('tags'),
            notes=data.get('notes')
        )
        
        entry.created_at = data.get('created_at', entry.created_at)
        entry.updated_at = data.get('updated_at', entry.updated_at)
        entry.executed_at = data.get('executed_at')
        entry.execution_time = data.get('execution_time')
        
        return entry
    
    def update_status(self, status: str, result: Optional[Any] = None) -> None:
        """
        Update command status and optionally result.
        
        Args:
            status: New status
            result: Optional result
        """
        self.status = status
        self.updated_at = time.time()
        
        if result is not None:
            self.result = result
        
        if status == "completed" or status == "failed":
            # Set execution time if not already set
            if self.executed_at is None:
                self.executed_at = time.time()
                # Calculate execution time
                self.execution_time = self.executed_at - self.created_at


class CommandHistory:
    """Manages command history for a project."""
    
    def __init__(self, project: ProjectContext):
        """
        Initialize command history for a project.
        
        Args:
            project: Project context
        """
        self.project = project
        self.db_path = None
        self._conn = None
        self._lock = threading.RLock()
        self._initialize_db()
    
    def _initialize_db(self) -> None:
        """Initialize the SQLite database for history storage."""
        with self._lock:
            # Create the database file in the project's output directory
            if self.project.output_directory:
                import os
                
                # Ensure output directory exists
                os.makedirs(self.project.output_directory, exist_ok=True)
                
                # Use a history.db file in the output directory
                self.db_path = os.path.join(
                    self.project.output_directory, 
                    f"{self.project.project_id}_history.db"
                )
                
                # Initialize the database tables
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Create command history table
                    cursor.execute('''
                    CREATE TABLE IF NOT EXISTS command_history (
                        command_id TEXT PRIMARY KEY,
                        command_type TEXT,
                        command TEXT,
                        parameters TEXT,
                        result TEXT,
                        status TEXT,
                        tags TEXT,
                        notes TEXT,
                        created_at REAL,
                        updated_at REAL,
                        executed_at REAL,
                        execution_time REAL
                    )
                    ''')
                    
                    # Create tags table for indexing
                    cursor.execute('''
                    CREATE TABLE IF NOT EXISTS command_tags (
                        command_id TEXT,
                        tag TEXT,
                        PRIMARY KEY (command_id, tag),
                        FOREIGN KEY (command_id) REFERENCES command_history(command_id)
                    )
                    ''')
                    
                    # Create indices
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_command_type ON command_history(command_type)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON command_history(status)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON command_history(created_at)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_tag ON command_tags(tag)')
                    
                    conn.commit()
                    
                    logging.debug(f"Initialized command history database at {self.db_path}", 
                                component="history")
            else:
                logging.warning("No output directory set for project, history will not be persisted", 
                              component="history")
    
    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Get a database connection.
        
        Yields:
            SQLite connection object
        """
        if not self.db_path:
            raise HistoryError("No database path specified")
        
        with self._lock:
            conn = None
            try:
                conn = sqlite3.connect(self.db_path)
                
                # Configure connection
                conn.row_factory = sqlite3.Row
                
                # Enable foreign keys
                conn.execute("PRAGMA foreign_keys = ON")
                
                yield conn
            finally:
                if conn:
                    conn.close()
    
    def _serialize_value(self, value: Any) -> str:
        """
        Serialize a value to JSON.
        
        Args:
            value: Value to serialize
            
        Returns:
            JSON string representation
        """
        try:
            return json.dumps(value)
        except (TypeError, OverflowError) as e:
            # Handle non-serializable objects
            logging.warning(f"Could not serialize value: {str(e)}", component="history")
            return json.dumps(str(value))
    
    def _deserialize_value(self, value: str) -> Any:
        """
        Deserialize a value from JSON.
        
        Args:
            value: JSON string representation
            
        Returns:
            Deserialized value
        """
        if value is None:
            return None
            
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    
    def add_command(self, entry: CommandEntry) -> str:
        """
        Add a command to the history.
        
        Args:
            entry: Command entry to add
            
        Returns:
            Command ID
        """
        if not self.db_path:
            # Just return the command ID without persisting
            return entry.command_id
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Insert the command entry
            cursor.execute('''
            INSERT INTO command_history 
            (command_id, command_type, command, parameters, result, status, 
             tags, notes, created_at, updated_at, executed_at, execution_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry.command_id,
                entry.command_type,
                self._serialize_value(entry.command),
                self._serialize_value(entry.parameters),
                self._serialize_value(entry.result),
                entry.status,
                json.dumps(entry.tags),
                entry.notes,
                entry.created_at,
                entry.updated_at,
                entry.executed_at,
                entry.execution_time
            ))
            
            # Insert tags for indexing
            for tag in entry.tags:
                cursor.execute('''
                INSERT OR IGNORE INTO command_tags (command_id, tag)
                VALUES (?, ?)
                ''', (entry.command_id, tag))
            
            conn.commit()
            
            logging.debug(f"Added command {entry.command_id} to history", component="history")
            return entry.command_id
    
    def get_command(self, command_id: str) -> Optional[CommandEntry]:
        """
        Get a command by ID.
        
        Args:
            command_id: ID of the command
            
        Returns:
            Command entry or None if not found
        """
        if not self.db_path:
            return None
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT * FROM command_history
            WHERE command_id = ?
            ''', (command_id,))
            
            row = cursor.fetchone()
            
            if row:
                entry = CommandEntry(
                    command_id=row['command_id'],
                    command_type=row['command_type'],
                    command=self._deserialize_value(row['command']),
                    parameters=self._deserialize_value(row['parameters']),
                    result=self._deserialize_value(row['result']),
                    status=row['status'],
                    tags=json.loads(row['tags']) if row['tags'] else [],
                    notes=row['notes']
                )
                
                entry.created_at = row['created_at']
                entry.updated_at = row['updated_at']
                entry.executed_at = row['executed_at']
                entry.execution_time = row['execution_time']
                
                return entry
                
            return None
    
    def update_command(self, command_id: str, 
                      status: Optional[str] = None,
                      result: Optional[Any] = None,
                      tags: Optional[List[str]] = None,
                      notes: Optional[str] = None) -> bool:
        """
        Update a command entry.
        
        Args:
            command_id: ID of the command
            status: Optional new status
            result: Optional new result
            tags: Optional new tags
            notes: Optional new notes
            
        Returns:
            True if command was updated, False if not found
        """
        if not self.db_path:
            return False
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # First check if the command exists
            cursor.execute('SELECT 1 FROM command_history WHERE command_id = ?', (command_id,))
            if not cursor.fetchone():
                return False
            
            # Prepare updates
            updates = []
            params = []
            
            updates.append('updated_at = ?')
            params.append(time.time())
            
            if status is not None:
                updates.append('status = ?')
                params.append(status)
                
                if status in ('completed', 'failed'):
                    # Set execution time if not already set
                    cursor.execute(
                        'SELECT created_at, executed_at FROM command_history WHERE command_id = ?', 
                        (command_id,)
                    )
                    row = cursor.fetchone()
                    
                    if row and row['executed_at'] is None:
                        now = time.time()
                        execution_time = now - row['created_at']
                        
                        updates.append('executed_at = ?')
                        params.append(now)
                        
                        updates.append('execution_time = ?')
                        params.append(execution_time)
            
            if result is not None:
                updates.append('result = ?')
                params.append(self._serialize_value(result))
            
            if tags is not None:
                updates.append('tags = ?')
                params.append(json.dumps(tags))
                
                # Update tags table
                cursor.execute('DELETE FROM command_tags WHERE command_id = ?', (command_id,))
                
                for tag in tags:
                    cursor.execute('''
                    INSERT OR IGNORE INTO command_tags (command_id, tag)
                    VALUES (?, ?)
                    ''', (command_id, tag))
            
            if notes is not None:
                updates.append('notes = ?')
                params.append(notes)
            
            # Execute update
            if updates:
                query = f"UPDATE command_history SET {', '.join(updates)} WHERE command_id = ?"
                params.append(command_id)
                
                cursor.execute(query, params)
                conn.commit()
                
                logging.debug(f"Updated command {command_id} in history", component="history")
                return True
            
            return False
    
    def delete_command(self, command_id: str) -> bool:
        """
        Delete a command from history.
        
        Args:
            command_id: ID of the command
            
        Returns:
            True if command was deleted, False if not found
        """
        if not self.db_path:
            return False
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Delete from tags table first (due to foreign key constraint)
            cursor.execute('DELETE FROM command_tags WHERE command_id = ?', (command_id,))
            
            # Delete the command
            cursor.execute('DELETE FROM command_history WHERE command_id = ?', (command_id,))
            
            if cursor.rowcount > 0:
                conn.commit()
                logging.debug(f"Deleted command {command_id} from history", component="history")
                return True
            else:
                return False
    
    def search_commands(self, 
                       command_type: Optional[str] = None,
                       status: Optional[str] = None,
                       tag: Optional[str] = None,
                       query: Optional[str] = None,
                       start_time: Optional[float] = None,
                       end_time: Optional[float] = None,
                       limit: int = 100,
                       offset: int = 0,
                       sort_by: str = 'created_at',
                       sort_order: str = 'desc') -> Tuple[List[CommandEntry], int]:
        """
        Search for commands in history.
        
        Args:
            command_type: Filter by command type
            status: Filter by status
            tag: Filter by tag
            query: Search term for command or parameters
            start_time: Filter by created_at >= start_time
            end_time: Filter by created_at <= end_time
            limit: Maximum number of results
            offset: Offset for pagination
            sort_by: Field to sort by
            sort_order: Sort order ('asc' or 'desc')
            
        Returns:
            Tuple of (list of command entries, total count)
        """
        if not self.db_path:
            return [], 0
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Build the query
            query_parts = ['SELECT * FROM command_history']
            where_clauses = []
            params = []
            
            # Handle tag filter using a subquery
            if tag:
                where_clauses.append('command_id IN (SELECT command_id FROM command_tags WHERE tag = ?)')
                params.append(tag)
            
            if command_type:
                where_clauses.append('command_type = ?')
                params.append(command_type)
            
            if status:
                where_clauses.append('status = ?')
                params.append(status)
            
            if query:
                # Search in command and parameters
                where_clauses.append('(command LIKE ? OR parameters LIKE ? OR notes LIKE ?)')
                search_term = f'%{query}%'
                params.extend([search_term, search_term, search_term])
            
            if start_time is not None:
                where_clauses.append('created_at >= ?')
                params.append(start_time)
            
            if end_time is not None:
                where_clauses.append('created_at <= ?')
                params.append(end_time)
            
            # Add WHERE clause if needed
            if where_clauses:
                query_parts.append('WHERE ' + ' AND '.join(where_clauses))
            
            # First get the total count
            count_query = 'SELECT COUNT(*) as count FROM command_history'
            if where_clauses:
                count_query += ' WHERE ' + ' AND '.join(where_clauses)
            
            cursor.execute(count_query, params)
            total_count = cursor.fetchone()['count']
            
            # Add sorting
            valid_sort_fields = {
                'created_at', 'updated_at', 'executed_at', 
                'execution_time', 'command_type', 'status'
            }
            
            if sort_by not in valid_sort_fields:
                sort_by = 'created_at'
            
            sort_order = 'DESC' if sort_order.lower() == 'desc' else 'ASC'
            query_parts.append(f'ORDER BY {sort_by} {sort_order}')
            
            # Add pagination
            query_parts.append('LIMIT ? OFFSET ?')
            params.append(limit)
            params.append(offset)
            
            # Execute the query
            cursor.execute(' '.join(query_parts), params)
            rows = cursor.fetchall()
            
            # Convert rows to CommandEntry objects
            commands = []
            for row in rows:
                entry = CommandEntry(
                    command_id=row['command_id'],
                    command_type=row['command_type'],
                    command=self._deserialize_value(row['command']),
                    parameters=self._deserialize_value(row['parameters']),
                    result=self._deserialize_value(row['result']),
                    status=row['status'],
                    tags=json.loads(row['tags']) if row['tags'] else [],
                    notes=row['notes']
                )
                
                entry.created_at = row['created_at']
                entry.updated_at = row['updated_at']
                entry.executed_at = row['executed_at']
                entry.execution_time = row['execution_time']
                
                commands.append(entry)
            
            return commands, total_count
    
    def get_recent_commands(self, limit: int = 10) -> List[CommandEntry]:
        """
        Get the most recent commands.
        
        Args:
            limit: Maximum number of commands to return
            
        Returns:
            List of recent command entries
        """
        commands, _ = self.search_commands(
            limit=limit,
            sort_by='created_at',
            sort_order='desc'
        )
        return commands
    
    def get_command_stats(self) -> Dict[str, Any]:
        """
        Get statistics about commands in history.
        
        Returns:
            Dictionary with statistics
        """
        if not self.db_path:
            return {
                'total_count': 0,
                'by_type': {},
                'by_status': {},
                'by_tag': {}
            }
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get total count
            cursor.execute('SELECT COUNT(*) as count FROM command_history')
            total_count = cursor.fetchone()['count']
            
            # Count by type
            cursor.execute('''
            SELECT command_type, COUNT(*) as count 
            FROM command_history 
            GROUP BY command_type
            ''')
            by_type = {row['command_type']: row['count'] for row in cursor.fetchall()}
            
            # Count by status
            cursor.execute('''
            SELECT status, COUNT(*) as count 
            FROM command_history 
            GROUP BY status
            ''')
            by_status = {row['status']: row['count'] for row in cursor.fetchall()}
            
            # Count by tag
            cursor.execute('''
            SELECT tag, COUNT(*) as count 
            FROM command_tags 
            GROUP BY tag
            ''')
            by_tag = {row['tag']: row['count'] for row in cursor.fetchall()}
            
            # Get average execution time
            cursor.execute('''
            SELECT AVG(execution_time) as avg_time 
            FROM command_history 
            WHERE execution_time IS NOT NULL
            ''')
            avg_execution_time = cursor.fetchone()['avg_time']
            
            return {
                'total_count': total_count,
                'by_type': by_type,
                'by_status': by_status,
                'by_tag': by_tag,
                'avg_execution_time': avg_execution_time
            }
    
    def clear_history(self, 
                    before_time: Optional[float] = None,
                    command_type: Optional[str] = None,
                    status: Optional[str] = None) -> int:
        """
        Clear command history.
        
        Args:
            before_time: Clear commands created before this time
            command_type: Clear commands of this type
            status: Clear commands with this status
            
        Returns:
            Number of commands cleared
        """
        if not self.db_path:
            return 0
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Build the query
            where_clauses = []
            params = []
            
            if before_time is not None:
                where_clauses.append('created_at < ?')
                params.append(before_time)
            
            if command_type:
                where_clauses.append('command_type = ?')
                params.append(command_type)
            
            if status:
                where_clauses.append('status = ?')
                params.append(status)
            
            # First, get the affected command IDs
            command_ids_query = 'SELECT command_id FROM command_history'
            if where_clauses:
                command_ids_query += ' WHERE ' + ' AND '.join(where_clauses)
            
            cursor.execute(command_ids_query, params)
            command_ids = [row['command_id'] for row in cursor.fetchall()]
            
            if not command_ids:
                return 0
            
            # Delete from tags table first
            placeholders = ','.join(['?'] * len(command_ids))
            cursor.execute(f'DELETE FROM command_tags WHERE command_id IN ({placeholders})', command_ids)
            
            # Delete the commands
            delete_query = 'DELETE FROM command_history'
            if where_clauses:
                delete_query += ' WHERE ' + ' AND '.join(where_clauses)
            
            cursor.execute(delete_query, params)
            count = cursor.rowcount
            
            conn.commit()
            logging.debug(f"Cleared {count} commands from history", component="history")
            return count
    
    def vacuum_database(self) -> None:
        """Optimize the database by vacuuming."""
        if not self.db_path:
            return
        
        with self._get_connection() as conn:
            conn.execute('VACUUM')
            logging.debug("Vacuumed command history database", component="history")


# Global registry of history managers
_history_managers = {}
_history_lock = threading.RLock()


def get_command_history(project_id: Optional[str] = None) -> CommandHistory:
    """
    Get the command history for a project.
    
    Args:
        project_id: ID of the project (or None for active project)
        
    Returns:
        CommandHistory instance
        
    Raises:
        HistoryError: If no project is found or active
    """
    # Get the project
    project = get_project(project_id)
    
    if not project:
        if project_id:
            raise HistoryError(f"Project not found: {project_id}")
        else:
            raise HistoryError("No active project")
    
    with _history_lock:
        # Check if we already have a history manager for this project
        if project.project_id in _history_managers:
            return _history_managers[project.project_id]
        
        # Create a new history manager
        history = CommandHistory(project)
        _history_managers[project.project_id] = history
        
        return history


def record_command(command_type: str, 
                 command: Dict[str, Any],
                 parameters: Optional[Dict[str, Any]] = None,
                 project_id: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 notes: Optional[str] = None) -> str:
    """
    Record a command in the project history.
    
    Args:
        command_type: Type of command
        command: The command itself
        parameters: Command parameters
        project_id: ID of the project (or None for active project)
        tags: Optional tags
        notes: Optional notes
        
    Returns:
        Command ID
        
    Raises:
        HistoryError: If no project is found or active
    """
    # Get the history manager
    history = get_command_history(project_id)
    
    # Create a command entry
    entry = CommandEntry(
        command_type=command_type,
        command=command,
        parameters=parameters,
        tags=tags,
        notes=notes
    )
    
    # Add to history
    return history.add_command(entry)


def update_command_status(command_id: str, 
                        status: str,
                        result: Optional[Any] = None,
                        project_id: Optional[str] = None) -> bool:
    """
    Update a command's status and optionally its result.
    
    Args:
        command_id: ID of the command
        status: New status
        result: Optional result
        project_id: ID of the project (or None for active project)
        
    Returns:
        True if command was updated, False if not found
        
    Raises:
        HistoryError: If no project is found or active
    """
    # Get the history manager
    history = get_command_history(project_id)
    
    # Update the command
    return history.update_command(command_id, status=status, result=result)


def get_command(command_id: str, 
              project_id: Optional[str] = None) -> Optional[CommandEntry]:
    """
    Get a command by ID.
    
    Args:
        command_id: ID of the command
        project_id: ID of the project (or None for active project)
        
    Returns:
        Command entry or None if not found
        
    Raises:
        HistoryError: If no project is found or active
    """
    # Get the history manager
    history = get_command_history(project_id)
    
    # Get the command
    return history.get_command(command_id)


def get_recent_commands(limit: int = 10, 
                      project_id: Optional[str] = None) -> List[CommandEntry]:
    """
    Get recent commands.
    
    Args:
        limit: Maximum number of commands to return
        project_id: ID of the project (or None for active project)
        
    Returns:
        List of recent command entries
        
    Raises:
        HistoryError: If no project is found or active
    """
    # Get the history manager
    history = get_command_history(project_id)
    
    # Get recent commands
    return history.get_recent_commands(limit=limit)


@contextmanager
def command_context(command_type: str, 
                  command: Dict[str, Any],
                  parameters: Optional[Dict[str, Any]] = None,
                  project_id: Optional[str] = None,
                  tags: Optional[List[str]] = None,
                  notes: Optional[str] = None) -> Generator[CommandEntry, None, None]:
    """
    Context manager for recording a command and its result.
    
    Args:
        command_type: Type of command
        command: The command itself
        parameters: Command parameters
        project_id: ID of the project (or None for active project)
        tags: Optional tags
        notes: Optional notes
        
    Yields:
        Command entry
        
    Raises:
        HistoryError: If no project is found or active
    """
    # Record the command
    command_id = record_command(
        command_type=command_type,
        command=command,
        parameters=parameters,
        project_id=project_id,
        tags=tags,
        notes=notes
    )
    
    # Get the command entry
    entry = get_command(command_id, project_id)
    
    if not entry:
        raise HistoryError(f"Command not found after recording: {command_id}")
    
    try:
        # Set status to running
        update_command_status(command_id, "running", project_id=project_id)
        
        # Yield the command entry
        yield entry
        
        # If no exception, set status to completed
        update_command_status(command_id, "completed", result=entry.result, project_id=project_id)
        
    except Exception as e:
        # On exception, set status to failed
        update_command_status(
            command_id, 
            "failed", 
            result={"error": str(e), "error_type": type(e).__name__},
            project_id=project_id
        )
        
        # Re-raise the exception
        raise