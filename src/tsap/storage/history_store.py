"""
Command history storage for TSAP.

This module provides persistent storage for command history records,
allowing commands and their results to be saved, retrieved, and queried.
"""

import os
import time
import threading
from typing import Dict, List, Any, Optional, Tuple

from tsap.utils.errors import TSAPError
from tsap.storage.database import Database, get_database, create_database


class HistoryStoreError(TSAPError):
    """Exception raised for history storage errors."""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.details = details or {}


class HistoryStore:
    """
    Persistent storage for command history.
    
    This class provides methods for saving, retrieving, and querying
    command history records in a SQLite database.
    """
    
    def __init__(self, db: Database):
        """
        Initialize a history store.
        
        Args:
            db: Database instance
        """
        self.db = db
        self._lock = threading.RLock()
        self._initialize_schema()
    
    def _initialize_schema(self) -> None:
        """Initialize the database schema for history storage."""
        with self._lock:
            with self.db.transaction():
                # Create command history table
                if not self.db.table_exists('command_history'):
                    self.db.create_table(
                        'command_history',
                        {
                            'command_id': 'TEXT NOT NULL',
                            'project_id': 'TEXT NOT NULL',
                            'command_type': 'TEXT',
                            'command': 'TEXT',
                            'parameters': 'TEXT',
                            'result': 'TEXT',
                            'status': 'TEXT NOT NULL',
                            'notes': 'TEXT',
                            'created_at': 'REAL NOT NULL',
                            'updated_at': 'REAL NOT NULL',
                            'executed_at': 'REAL',
                            'execution_time': 'REAL'
                        },
                        primary_key='command_id'
                    )
                    
                    # Create indices
                    self.db.create_index('idx_ch_project_id', 'command_history', 'project_id')
                    self.db.create_index('idx_ch_command_type', 'command_history', 'command_type')
                    self.db.create_index('idx_ch_status', 'command_history', 'status')
                    self.db.create_index('idx_ch_created_at', 'command_history', 'created_at')
                
                # Create command tags table
                if not self.db.table_exists('command_tags'):
                    self.db.create_table(
                        'command_tags',
                        {
                            'command_id': 'TEXT NOT NULL',
                            'tag': 'TEXT NOT NULL'
                        },
                        primary_key=['command_id', 'tag']
                    )
                    
                    # Create index on tag
                    self.db.create_index('idx_ct_tag', 'command_tags', 'tag')
                    
                    # Create foreign key index
                    self.db.create_index('idx_ct_command_id', 'command_tags', 'command_id')
    
    def add_command(self, command_id: str, project_id: str, command_type: str,
                  command: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None,
                  status: str = 'pending', notes: Optional[str] = None,
                  tags: Optional[List[str]] = None) -> None:
        """
        Add a command to history.
        
        Args:
            command_id: Unique ID for the command
            project_id: ID of the project
            command_type: Type of command
            command: The command itself
            parameters: Command parameters
            status: Command status
            notes: Optional notes
            tags: Optional tags
        """
        with self._lock:
            with self.db.transaction():
                # Serialize command and parameters
                command_json = self.db.json_serialize(command)
                parameters_json = self.db.json_serialize(parameters)
                
                # Current time
                current_time = time.time()
                
                # Insert command
                self.db.insert(
                    'command_history',
                    {
                        'command_id': command_id,
                        'project_id': project_id,
                        'command_type': command_type,
                        'command': command_json,
                        'parameters': parameters_json,
                        'result': None,
                        'status': status,
                        'notes': notes,
                        'created_at': current_time,
                        'updated_at': current_time,
                        'executed_at': None,
                        'execution_time': None
                    }
                )
                
                # Add tags if provided
                if tags:
                    for tag in tags:
                        self.db.insert(
                            'command_tags',
                            {
                                'command_id': command_id,
                                'tag': tag
                            }
                        )
    
    def update_command(self, command_id: str, status: Optional[str] = None,
                     result: Optional[Any] = None, notes: Optional[str] = None,
                     executed_at: Optional[float] = None,
                     execution_time: Optional[float] = None) -> bool:
        """
        Update a command in history.
        
        Args:
            command_id: ID of the command
            status: New status
            result: Command result
            notes: Command notes
            executed_at: Execution timestamp
            execution_time: Execution duration
            
        Returns:
            True if command was updated, False if not found
        """
        with self._lock:
            with self.db.transaction():
                # Check if command exists
                command = self.db.query_one(
                    'SELECT command_id, created_at FROM command_history WHERE command_id = ?',
                    (command_id,)
                )
                
                if not command:
                    return False
                
                # Build update data
                update_data = {
                    'updated_at': time.time()
                }
                
                if status is not None:
                    update_data['status'] = status
                    
                    # If status is 'completed' or 'failed', set executed_at if not provided
                    if status in ('completed', 'failed') and executed_at is None and not self.db.query_one(
                        'SELECT executed_at FROM command_history WHERE command_id = ? AND executed_at IS NOT NULL',
                        (command_id,)
                    ):
                        update_data['executed_at'] = time.time()
                        
                        # Calculate execution time if not provided
                        if execution_time is None:
                            update_data['execution_time'] = update_data['executed_at'] - command['created_at']
                
                if result is not None:
                    update_data['result'] = self.db.json_serialize(result)
                
                if notes is not None:
                    update_data['notes'] = notes
                
                if executed_at is not None:
                    update_data['executed_at'] = executed_at
                
                if execution_time is not None:
                    update_data['execution_time'] = execution_time
                
                # Update command
                self.db.update(
                    'command_history',
                    update_data,
                    'command_id = ?',
                    (command_id,)
                )
                
                return True
    
    def get_command(self, command_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a command from history.
        
        Args:
            command_id: ID of the command
            
        Returns:
            Command data or None if not found
        """
        with self._lock:
            # Get command data
            command = self.db.query_one(
                'SELECT * FROM command_history WHERE command_id = ?',
                (command_id,)
            )
            
            if not command:
                return None
            
            # Deserialize command and parameters
            command['command'] = self.db.json_deserialize(command['command'])
            command['parameters'] = self.db.json_deserialize(command['parameters'])
            command['result'] = self.db.json_deserialize(command['result'])
            
            # Get tags
            tags = self.db.query(
                'SELECT tag FROM command_tags WHERE command_id = ?',
                (command_id,)
            )
            
            command['tags'] = [tag['tag'] for tag in tags]
            
            return command
    
    def delete_command(self, command_id: str) -> bool:
        """
        Delete a command from history.
        
        Args:
            command_id: ID of the command
            
        Returns:
            True if command was deleted, False if not found
        """
        with self._lock:
            with self.db.transaction():
                # Delete tags first (due to foreign key constraint)
                self.db.delete('command_tags', 'command_id = ?', (command_id,))
                
                # Delete command
                rows_affected = self.db.delete('command_history', 'command_id = ?', (command_id,))
                
                return rows_affected > 0
    
    def clear_project_history(self, project_id: str) -> int:
        """
        Clear command history for a project.
        
        Args:
            project_id: ID of the project
            
        Returns:
            Number of commands deleted
        """
        with self._lock:
            with self.db.transaction():
                # Get command IDs to delete
                commands = self.db.query(
                    'SELECT command_id FROM command_history WHERE project_id = ?',
                    (project_id,)
                )
                
                command_ids = [command['command_id'] for command in commands]
                
                if not command_ids:
                    return 0
                
                # Delete tags first
                for command_id in command_ids:
                    self.db.delete('command_tags', 'command_id = ?', (command_id,))
                
                # Delete commands
                rows_affected = self.db.delete('command_history', 'project_id = ?', (project_id,))
                
                return rows_affected
    
    def add_tag(self, command_id: str, tag: str) -> bool:
        """
        Add a tag to a command.
        
        Args:
            command_id: ID of the command
            tag: Tag to add
            
        Returns:
            True if tag was added, False if command not found
        """
        with self._lock:
            # Check if command exists
            command = self.db.query_one(
                'SELECT 1 FROM command_history WHERE command_id = ?',
                (command_id,)
            )
            
            if not command:
                return False
            
            # Check if tag already exists
            existing_tag = self.db.query_one(
                'SELECT 1 FROM command_tags WHERE command_id = ? AND tag = ?',
                (command_id, tag)
            )
            
            if existing_tag:
                return True  # Tag already exists
            
            # Add tag
            with self.db.transaction():
                self.db.insert(
                    'command_tags',
                    {
                        'command_id': command_id,
                        'tag': tag
                    }
                )
                
                # Update command updated_at
                self.db.update(
                    'command_history',
                    {'updated_at': time.time()},
                    'command_id = ?',
                    (command_id,)
                )
                
                return True
    
    def remove_tag(self, command_id: str, tag: str) -> bool:
        """
        Remove a tag from a command.
        
        Args:
            command_id: ID of the command
            tag: Tag to remove
            
        Returns:
            True if tag was removed, False if command or tag not found
        """
        with self._lock:
            with self.db.transaction():
                # Delete tag
                rows_affected = self.db.delete(
                    'command_tags',
                    'command_id = ? AND tag = ?',
                    (command_id, tag)
                )
                
                if rows_affected > 0:
                    # Update command updated_at
                    self.db.update(
                        'command_history',
                        {'updated_at': time.time()},
                        'command_id = ?',
                        (command_id,)
                    )
                    
                    return True
                
                return False
    
    def search_commands(self, project_id: Optional[str] = None,
                      command_type: Optional[str] = None,
                      status: Optional[str] = None,
                      tag: Optional[str] = None,
                      text_query: Optional[str] = None,
                      start_time: Optional[float] = None,
                      end_time: Optional[float] = None,
                      limit: int = 100,
                      offset: int = 0,
                      sort_by: str = 'created_at',
                      sort_order: str = 'desc') -> Tuple[List[Dict[str, Any]], int]:
        """
        Search for commands in history.
        
        Args:
            project_id: Filter by project ID
            command_type: Filter by command type
            status: Filter by status
            tag: Filter by tag
            text_query: Text search query
            start_time: Filter by created_at >= start_time
            end_time: Filter by created_at <= end_time
            limit: Maximum number of results to return
            offset: Offset for pagination
            sort_by: Field to sort by
            sort_order: Sort order ('asc' or 'desc')
            
        Returns:
            Tuple of (command list, total count)
        """
        with self._lock:
            # Build the query
            query_parts = ['SELECT ch.* FROM command_history ch']
            where_clauses = []
            params = []
            
            # Join with tags table if filtering by tag
            if tag:
                query_parts[0] = 'SELECT ch.* FROM command_history ch JOIN command_tags ct ON ch.command_id = ct.command_id'
                where_clauses.append('ct.tag = ?')
                params.append(tag)
            
            # Add filter conditions
            if project_id:
                where_clauses.append('ch.project_id = ?')
                params.append(project_id)
            
            if command_type:
                where_clauses.append('ch.command_type = ?')
                params.append(command_type)
            
            if status:
                where_clauses.append('ch.status = ?')
                params.append(status)
            
            if text_query:
                # Search in command, parameters, and notes
                where_clauses.append('(ch.command LIKE ? OR ch.parameters LIKE ? OR ch.notes LIKE ?)')
                search_term = f'%{text_query}%'
                params.extend([search_term, search_term, search_term])
            
            if start_time is not None:
                where_clauses.append('ch.created_at >= ?')
                params.append(start_time)
            
            if end_time is not None:
                where_clauses.append('ch.created_at <= ?')
                params.append(end_time)
            
            # Combine WHERE clauses
            if where_clauses:
                query_parts.append('WHERE ' + ' AND '.join(where_clauses))
            
            # Build the full query for count
            count_query = ' '.join(['SELECT COUNT(*) as count FROM ('] + query_parts + [') as subquery'])
            
            # Get total count
            count_result = self.db.query_one(count_query, params)
            total_count = count_result['count'] if count_result else 0
            
            # Add sorting
            valid_sort_fields = {'created_at', 'updated_at', 'executed_at', 'execution_time', 'command_type', 'status'}
            if sort_by not in valid_sort_fields:
                sort_by = 'created_at'
            
            sort_order = 'DESC' if sort_order.lower() == 'desc' else 'ASC'
            query_parts.append(f'ORDER BY ch.{sort_by} {sort_order}')
            
            # Add pagination
            query_parts.append('LIMIT ? OFFSET ?')
            params.append(limit)
            params.append(offset)
            
            # Execute the query
            query = ' '.join(query_parts)
            commands = self.db.query(query, params)
            
            # Process results
            result = []
            for command in commands:
                # Deserialize command and parameters
                command['command'] = self.db.json_deserialize(command['command'])
                command['parameters'] = self.db.json_deserialize(command['parameters'])
                command['result'] = self.db.json_deserialize(command['result'])
                
                # Get tags
                tags = self.db.query(
                    'SELECT tag FROM command_tags WHERE command_id = ?',
                    (command['command_id'],)
                )
                
                command['tags'] = [tag['tag'] for tag in tags]
                
                result.append(command)
            
            return result, total_count
    
    def get_recent_commands(self, project_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent commands.
        
        Args:
            project_id: Optional project ID filter
            limit: Maximum number of commands to return
            
        Returns:
            List of recent commands
        """
        commands, _ = self.search_commands(
            project_id=project_id,
            limit=limit,
            sort_by='created_at',
            sort_order='desc'
        )
        
        return commands
    
    def get_command_stats(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about commands in history.
        
        Args:
            project_id: Optional project ID filter
            
        Returns:
            Dictionary with statistics
        """
        with self._lock:
            # Base query parts
            base_where = 'WHERE project_id = ?' if project_id else ''
            base_params = (project_id,) if project_id else ()
            
            # Get total count
            total_count_query = f'SELECT COUNT(*) as count FROM command_history {base_where}'
            total_count_result = self.db.query_one(total_count_query, base_params)
            total_count = total_count_result['count'] if total_count_result else 0
            
            # Skip further queries if no commands
            if total_count == 0:
                return {
                    'total_count': 0,
                    'by_type': {},
                    'by_status': {},
                    'by_tag': {},
                    'avg_execution_time': None
                }
            
            # Count by type
            type_query = f'SELECT command_type, COUNT(*) as count FROM command_history {base_where} GROUP BY command_type'
            type_results = self.db.query(type_query, base_params)
            by_type = {r['command_type']: r['count'] for r in type_results}
            
            # Count by status
            status_query = f'SELECT status, COUNT(*) as count FROM command_history {base_where} GROUP BY status'
            status_results = self.db.query(status_query, base_params)
            by_status = {r['status']: r['count'] for r in status_results}
            
            # Count by tag
            tag_join = f'JOIN command_tags ON command_history.command_id = command_tags.command_id {base_where}'
            tag_query = f'SELECT tag, COUNT(*) as count FROM command_history {tag_join} GROUP BY tag'
            tag_results = self.db.query(tag_query, base_params)
            by_tag = {r['tag']: r['count'] for r in tag_results}
            
            # Get average execution time
            exec_time_query = f'SELECT AVG(execution_time) as avg_time FROM command_history {base_where} AND execution_time IS NOT NULL'
            exec_time_result = self.db.query_one(exec_time_query, base_params)
            avg_execution_time = exec_time_result['avg_time'] if exec_time_result else None
            
            return {
                'total_count': total_count,
                'by_type': by_type,
                'by_status': by_status,
                'by_tag': by_tag,
                'avg_execution_time': avg_execution_time
            }
    
    def get_project_history_size(self, project_id: str) -> int:
        """
        Get the number of commands in project history.
        
        Args:
            project_id: Project ID
            
        Returns:
            Number of commands
        """
        with self._lock:
            result = self.db.query_one(
                'SELECT COUNT(*) as count FROM command_history WHERE project_id = ?',
                (project_id,)
            )
            
            return result['count'] if result else 0


# Global registry of history stores
_history_stores = {}
_history_lock = threading.RLock()


def get_history_store(db_name: str = 'history') -> HistoryStore:
    """
    Get a history store.
    
    Args:
        db_name: Database name
        
    Returns:
        HistoryStore instance
    """
    with _history_lock:
        if db_name in _history_stores:
            return _history_stores[db_name]
        
        # Get or create the database
        db = get_database(db_name)
        
        if db is None:
            # Create history database in data directory
            from tsap.config import get_config
            
            config = get_config()
            
            if hasattr(config, 'storage') and hasattr(config.storage, 'data_dir'):
                data_dir = config.storage.data_dir
            else:
                # Use system-dependent user data directory
                import appdirs  # Optional dependency
                data_dir = appdirs.user_data_dir("tsap", "tsap")
            
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, f"{db_name}.db")
            
            db = create_database(db_name, db_path)
        
        # Create history store
        store = HistoryStore(db)
        _history_stores[db_name] = store
        
        return store


def get_project_history_store(project_id: str) -> HistoryStore:
    """
    Get a project-specific history store.
    
    This is a convenience function that uses the global history store
    but applies project-specific filtering.
    
    Args:
        project_id: Project ID
        
    Returns:
        HistoryStore instance
    """
    # Use the global history store
    return get_history_store()


def add_command_to_history(command_id: str, project_id: str, command_type: str,
                        command: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None,
                        status: str = 'pending', notes: Optional[str] = None,
                        tags: Optional[List[str]] = None) -> None:
    """
    Add a command to history.
    
    Args:
        command_id: Unique ID for the command
        project_id: ID of the project
        command_type: Type of command
        command: The command itself
        parameters: Command parameters
        status: Command status
        notes: Optional notes
        tags: Optional tags
    """
    store = get_history_store()
    store.add_command(command_id, project_id, command_type, command, parameters, status, notes, tags)


def update_command_in_history(command_id: str, status: Optional[str] = None,
                           result: Optional[Any] = None, notes: Optional[str] = None,
                           executed_at: Optional[float] = None,
                           execution_time: Optional[float] = None) -> bool:
    """
    Update a command in history.
    
    Args:
        command_id: ID of the command
        status: New status
        result: Command result
        notes: Command notes
        executed_at: Execution timestamp
        execution_time: Execution duration
        
    Returns:
        True if command was updated, False if not found
    """
    store = get_history_store()
    return store.update_command(command_id, status, result, notes, executed_at, execution_time)


def get_command_from_history(command_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a command from history.
    
    Args:
        command_id: ID of the command
        
    Returns:
        Command data or None if not found
    """
    store = get_history_store()
    return store.get_command(command_id)


def search_command_history(project_id: Optional[str] = None,
                        command_type: Optional[str] = None,
                        status: Optional[str] = None,
                        tag: Optional[str] = None,
                        text_query: Optional[str] = None,
                        start_time: Optional[float] = None,
                        end_time: Optional[float] = None,
                        limit: int = 100,
                        offset: int = 0,
                        sort_by: str = 'created_at',
                        sort_order: str = 'desc') -> Tuple[List[Dict[str, Any]], int]:
    """
    Search for commands in history.
    
    Args:
        project_id: Filter by project ID
        command_type: Filter by command type
        status: Filter by status
        tag: Filter by tag
        text_query: Text search query
        start_time: Filter by created_at >= start_time
        end_time: Filter by created_at <= end_time
        limit: Maximum number of results to return
        offset: Offset for pagination
        sort_by: Field to sort by
        sort_order: Sort order ('asc' or 'desc')
        
    Returns:
        Tuple of (command list, total count)
    """
    store = get_history_store()
    return store.search_commands(
        project_id, command_type, status, tag, text_query,
        start_time, end_time, limit, offset, sort_by, sort_order
    )


def get_recent_commands_from_history(project_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get recent commands from history.
    
    Args:
        project_id: Optional project ID filter
        limit: Maximum number of commands to return
        
    Returns:
        List of recent commands
    """
    store = get_history_store()
    return store.get_recent_commands(project_id, limit)