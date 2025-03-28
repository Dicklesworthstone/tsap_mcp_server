"""
Database utilities for the TSAP MCP Server.

This module provides a common interface and utilities for interacting with
databases (primarily SQLite) used throughout the TSAP system.
"""

import os
import json
import time
import sqlite3
import threading
import functools
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Union, Tuple, Iterator, TypeVar

import tsap.utils.logging as logging
from tsap.utils.errors import TSAPError
from tsap.config import get_config


# Type variables for generic functions
T = TypeVar('T')


class DatabaseError(TSAPError):
    """Exception raised for database-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.details = details or {}


class Database:
    """
    Base class for database connections.
    
    This class provides common functionality for working with databases,
    including connection management, query execution, and transaction handling.
    """
    
    def __init__(self, db_path: str, timeout: float = 30.0):
        """
        Initialize a database.
        
        Args:
            db_path: Path to the database file
            timeout: Connection timeout in seconds
        """
        self.db_path = db_path
        self.timeout = timeout
        self._lock = threading.RLock()
        
        # Connection cache (thread-local)
        self._local = threading.local()
        
        # Track active transactions
        self._transaction_count = 0
        self._transaction_lock = threading.RLock()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    
    def _get_connection(self) -> sqlite3.Connection:
        """
        Get a SQLite connection (thread-local).
        
        Returns:
            SQLite connection
        """
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                self.db_path,
                timeout=self.timeout,
                isolation_level=None,  # We'll manage transactions manually
                check_same_thread=False
            )
            
            # Configure connection
            self._local.connection.row_factory = sqlite3.Row
            
            # Enable foreign keys
            self._local.connection.execute("PRAGMA foreign_keys = ON")
        
        return self._local.connection
    
    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            if hasattr(self._local, 'connection') and self._local.connection is not None:
                self._local.connection.close()
                self._local.connection = None
    
    @contextmanager
    def transaction(self) -> Iterator[None]:
        """
        Context manager for database transactions.
        
        This allows nested transactions (using savepoints) and
        ensures proper commit/rollback handling.
        
        Yields:
            None
        """
        conn = self._get_connection()
        
        with self._transaction_lock:
            # Begin transaction or savepoint
            if self._transaction_count == 0:
                conn.execute("BEGIN")
            else:
                savepoint_name = f"savepoint_{self._transaction_count}"
                conn.execute(f"SAVEPOINT {savepoint_name}")
            
            self._transaction_count += 1
        
        try:
            # Yield control to the caller
            yield
            
            # Commit transaction or release savepoint
            with self._transaction_lock:
                self._transaction_count -= 1
                
                if self._transaction_count == 0:
                    conn.execute("COMMIT")
                else:
                    savepoint_name = f"savepoint_{self._transaction_count}"
                    conn.execute(f"RELEASE {savepoint_name}")
        
        except Exception:
            # Rollback transaction or savepoint
            with self._transaction_lock:
                self._transaction_count -= 1
                
                if self._transaction_count == 0:
                    conn.execute("ROLLBACK")
                else:
                    savepoint_name = f"savepoint_{self._transaction_count}"
                    conn.execute(f"ROLLBACK TO {savepoint_name}")
            
            # Re-raise the exception
            raise
    
    def execute(self, query: str, params: Optional[Union[Tuple, Dict]] = None) -> sqlite3.Cursor:
        """
        Execute a SQL query.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            SQLite cursor
        """
        conn = self._get_connection()
        return conn.execute(query, params or ())
    
    def executemany(self, query: str, params_list: List[Union[Tuple, Dict]]) -> sqlite3.Cursor:
        """
        Execute a SQL query with multiple parameter sets.
        
        Args:
            query: SQL query
            params_list: List of parameter sets
            
        Returns:
            SQLite cursor
        """
        conn = self._get_connection()
        return conn.executemany(query, params_list)
    
    def executescript(self, script: str) -> sqlite3.Cursor:
        """
        Execute a SQL script.
        
        Args:
            script: SQL script
            
        Returns:
            SQLite cursor
        """
        conn = self._get_connection()
        return conn.executescript(script)
    
    def query(self, query: str, params: Optional[Union[Tuple, Dict]] = None) -> List[Dict[str, Any]]:
        """
        Execute a query and return results as a list of dictionaries.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            List of result rows as dictionaries
        """
        cursor = self.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def query_one(self, query: str, params: Optional[Union[Tuple, Dict]] = None) -> Optional[Dict[str, Any]]:
        """
        Execute a query and return the first result as a dictionary.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            First result row as dictionary or None if no results
        """
        cursor = self.execute(query, params)
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def insert(self, table: str, data: Dict[str, Any]) -> int:
        """
        Insert data into a table.
        
        Args:
            table: Table name
            data: Data to insert (column -> value)
            
        Returns:
            ID of the inserted row (ROWID/last_insert_rowid())
        """
        # Build the query
        columns = list(data.keys())
        placeholders = ', '.join(['?'] * len(columns))
        
        query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
        
        # Execute the query
        with self.transaction():
            cursor = self.execute(query, list(data.values()))
            return cursor.lastrowid
    
    def update(self, table: str, data: Dict[str, Any], where: str, params: Union[Tuple, Dict]) -> int:
        """
        Update data in a table.
        
        Args:
            table: Table name
            data: Data to update (column -> value)
            where: WHERE clause
            params: Parameters for WHERE clause
            
        Returns:
            Number of rows affected
        """
        # Build the query
        set_clause = ', '.join([f"{column} = ?" for column in data.keys()])
        
        query = f"UPDATE {table} SET {set_clause} WHERE {where}"
        
        # Combine data values and where params
        all_params = list(data.values())
        if isinstance(params, dict):
            # Need to extract values in the same order as placeholders in WHERE clause
            # This is complex, so we'll just append the params as a tuple
            all_params.extend(params.values())
        else:
            all_params.extend(params)
        
        # Execute the query
        with self.transaction():
            cursor = self.execute(query, all_params)
            return cursor.rowcount
    
    def delete(self, table: str, where: str, params: Union[Tuple, Dict]) -> int:
        """
        Delete data from a table.
        
        Args:
            table: Table name
            where: WHERE clause
            params: Parameters for WHERE clause
            
        Returns:
            Number of rows affected
        """
        query = f"DELETE FROM {table} WHERE {where}"
        
        with self.transaction():
            cursor = self.execute(query, params)
            return cursor.rowcount
    
    def table_exists(self, table: str) -> bool:
        """
        Check if a table exists.
        
        Args:
            table: Table name
            
        Returns:
            True if the table exists, False otherwise
        """
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
        cursor = self.execute(query, (table,))
        return cursor.fetchone() is not None
    
    def create_table(self, table: str, columns: Dict[str, str], 
                   primary_key: Optional[Union[str, List[str]]] = None,
                   if_not_exists: bool = True) -> None:
        """
        Create a table.
        
        Args:
            table: Table name
            columns: Column definitions (name -> type definition)
            primary_key: Primary key column(s)
            if_not_exists: Whether to add IF NOT EXISTS clause
        """
        # Build column definitions
        col_defs = []
        
        for column, definition in columns.items():
            if isinstance(primary_key, str) and column == primary_key:
                col_defs.append(f"{column} {definition} PRIMARY KEY")
            else:
                col_defs.append(f"{column} {definition}")
        
        # Add composite primary key if needed
        if isinstance(primary_key, list) and primary_key:
            col_defs.append(f"PRIMARY KEY ({', '.join(primary_key)})")
        
        # Build the query
        exists_clause = "IF NOT EXISTS " if if_not_exists else ""
        query = f"CREATE TABLE {exists_clause}{table} ({', '.join(col_defs)})"
        
        with self.transaction():
            self.execute(query)
    
    def create_index(self, index: str, table: str, columns: Union[str, List[str]], 
                   unique: bool = False, if_not_exists: bool = True) -> None:
        """
        Create an index.
        
        Args:
            index: Index name
            table: Table name
            columns: Column(s) to index
            unique: Whether the index is unique
            if_not_exists: Whether to add IF NOT EXISTS clause
        """
        # Format columns
        if isinstance(columns, list):
            columns_clause = ', '.join(columns)
        else:
            columns_clause = columns
        
        # Build the query
        exists_clause = "IF NOT EXISTS " if if_not_exists else ""
        unique_clause = "UNIQUE " if unique else ""
        
        query = f"CREATE {unique_clause}INDEX {exists_clause}{index} ON {table} ({columns_clause})"
        
        with self.transaction():
            self.execute(query)
    
    def vacuum(self) -> None:
        """Optimize the database by vacuuming."""
        self.execute("VACUUM")
    
    def backup(self, backup_path: str) -> None:
        """
        Create a backup of the database.
        
        Args:
            backup_path: Path for the backup file
        """
        with self._lock:
            # Close existing connections to ensure all changes are written
            self.close()
            
            # Create a new connection for backup
            src_conn = sqlite3.connect(self.db_path)
            
            try:
                # Ensure backup directory exists
                os.makedirs(os.path.dirname(os.path.abspath(backup_path)), exist_ok=True)
                
                # Open backup database
                dst_conn = sqlite3.connect(backup_path)
                
                # Perform backup
                src_conn.backup(dst_conn)
                
                # Close backup connection
                dst_conn.close()
                
                logging.debug(f"Created database backup at {backup_path}", component="database")
                
            finally:
                # Close source connection
                src_conn.close()
    
    def json_serialize(self, value: Any) -> str:
        """
        Serialize a value to JSON for storage.
        
        Args:
            value: Value to serialize
            
        Returns:
            JSON string
        """
        if value is None:
            return None
        return json.dumps(value)
    
    def json_deserialize(self, value: str) -> Any:
        """
        Deserialize a JSON string from storage.
        
        Args:
            value: JSON string
            
        Returns:
            Deserialized value
        """
        if value is None:
            return None
        return json.loads(value)


class DatabaseRegistry:
    """Registry for managing database connections."""
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DatabaseRegistry, cls).__new__(cls)
                cls._instance._databases = {}
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        with self._lock:
            if not self._initialized:
                self._databases = {}
                self._initialized = True
    
    def register(self, name: str, database: Database) -> None:
        """
        Register a database.
        
        Args:
            name: Database name
            database: Database instance
        """
        with self._lock:
            self._databases[name] = database
    
    def get(self, name: str) -> Optional[Database]:
        """
        Get a database by name.
        
        Args:
            name: Database name
            
        Returns:
            Database instance or None if not found
        """
        with self._lock:
            return self._databases.get(name)
    
    def create(self, name: str, db_path: str, timeout: float = 30.0) -> Database:
        """
        Create and register a database.
        
        Args:
            name: Database name
            db_path: Path to the database file
            timeout: Connection timeout in seconds
            
        Returns:
            Database instance
        """
        with self._lock:
            database = Database(db_path, timeout)
            self._databases[name] = database
            return database
    
    def close_all(self) -> None:
        """Close all registered databases."""
        with self._lock:
            for database in self._databases.values():
                database.close()


# Singleton instance
_registry = DatabaseRegistry()


def get_registry() -> DatabaseRegistry:
    """
    Get the database registry.
    
    Returns:
        Database registry instance
    """
    return _registry


def get_database(name: str) -> Optional[Database]:
    """
    Get a database by name.
    
    Args:
        name: Database name
        
    Returns:
        Database instance or None if not found
    """
    return get_registry().get(name)


def create_database(name: str, db_path: str, timeout: float = 30.0) -> Database:
    """
    Create and register a database.
    
    Args:
        name: Database name
        db_path: Path to the database file
        timeout: Connection timeout in seconds
        
    Returns:
        Database instance
    """
    return get_registry().create(name, db_path, timeout)


def get_system_database() -> Database:
    """
    Get the main system database.
    
    Returns:
        System database instance
    """
    db = get_database('system')
    
    if db is None:
        # Create system database
        config = get_config()
        
        if hasattr(config, 'storage') and hasattr(config.storage, 'system_db_path'):
            db_path = config.storage.system_db_path
        else:
            # Default to data directory
            if hasattr(config, 'storage') and hasattr(config.storage, 'data_dir'):
                data_dir = config.storage.data_dir
            else:
                # Use system-dependent user data directory
                import appdirs  # Optional dependency
                data_dir = appdirs.user_data_dir("tsap", "tsap")
            
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, "system.db")
        
        db = create_database('system', db_path)
        
        # Initialize system tables
        _initialize_system_database(db)
    
    return db


def _initialize_system_database(db: Database) -> None:
    """
    Initialize the system database schema.
    
    Args:
        db: Database instance
    """
    with db.transaction():
        # Create version table
        if not db.table_exists('version'):
            db.create_table(
                'version',
                {
                    'version': 'TEXT NOT NULL',
                    'updated_at': 'REAL NOT NULL',
                    'description': 'TEXT'
                },
                primary_key='version'
            )
            
            # Insert initial version
            db.insert(
                'version',
                {
                    'version': '1.0',
                    'updated_at': time.time(),
                    'description': 'Initial schema version'
                }
            )
        
        # Create projects table
        if not db.table_exists('projects'):
            db.create_table(
                'projects',
                {
                    'project_id': 'TEXT NOT NULL',
                    'name': 'TEXT NOT NULL',
                    'description': 'TEXT',
                    'created_at': 'REAL NOT NULL',
                    'updated_at': 'REAL NOT NULL',
                    'root_directory': 'TEXT',
                    'output_directory': 'TEXT',
                    'is_active': 'INTEGER NOT NULL DEFAULT 0'
                },
                primary_key='project_id'
            )
            
            # Create index on name
            db.create_index('idx_projects_name', 'projects', 'name')
        
        # Create plugins table
        if not db.table_exists('plugins'):
            db.create_table(
                'plugins',
                {
                    'plugin_id': 'TEXT NOT NULL',
                    'name': 'TEXT NOT NULL',
                    'version': 'TEXT NOT NULL',
                    'description': 'TEXT',
                    'author': 'TEXT',
                    'installed_at': 'REAL NOT NULL',
                    'updated_at': 'REAL NOT NULL',
                    'path': 'TEXT',
                    'enabled': 'INTEGER NOT NULL DEFAULT 1',
                    'capabilities': 'TEXT'
                },
                primary_key='plugin_id'
            )
            
            # Create unique index on name and version
            db.create_index('idx_plugins_name_version', 'plugins', ['name', 'version'], unique=True)
        
        # Create settings table
        if not db.table_exists('settings'):
            db.create_table(
                'settings',
                {
                    'key': 'TEXT NOT NULL',
                    'value': 'TEXT',
                    'updated_at': 'REAL NOT NULL'
                },
                primary_key='key'
            )
        
        # Create user table (for API keys, etc.)
        if not db.table_exists('users'):
            db.create_table(
                'users',
                {
                    'user_id': 'TEXT NOT NULL',
                    'username': 'TEXT NOT NULL',
                    'password_hash': 'TEXT',
                    'created_at': 'REAL NOT NULL',
                    'updated_at': 'REAL NOT NULL',
                    'last_login': 'REAL',
                    'is_admin': 'INTEGER NOT NULL DEFAULT 0',
                    'is_active': 'INTEGER NOT NULL DEFAULT 1'
                },
                primary_key='user_id'
            )
            
            # Create unique index on username
            db.create_index('idx_users_username', 'users', 'username', unique=True)
        
        # Create API keys table
        if not db.table_exists('api_keys'):
            db.create_table(
                'api_keys',
                {
                    'api_key': 'TEXT NOT NULL',
                    'user_id': 'TEXT NOT NULL',
                    'name': 'TEXT NOT NULL',
                    'created_at': 'REAL NOT NULL',
                    'expires_at': 'REAL',
                    'last_used': 'REAL',
                    'is_active': 'INTEGER NOT NULL DEFAULT 1',
                    'permissions': 'TEXT'
                },
                primary_key='api_key'
            )
            
            # Create index on user_id
            db.create_index('idx_api_keys_user_id', 'api_keys', 'user_id')


def close_all_databases() -> None:
    """Close all registered databases."""
    get_registry().close_all()


# Database access decorators and utilities

def with_database(db_name: str):
    """
    Decorator to provide a database connection to a function.
    
    Args:
        db_name: Database name
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            db = get_database(db_name)
            
            if db is None:
                raise DatabaseError(f"Database not found: {db_name}")
            
            return func(db, *args, **kwargs)
        
        return wrapper
    
    return decorator


def with_transaction(db_name: str):
    """
    Decorator to execute a function within a database transaction.
    
    Args:
        db_name: Database name
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            db = get_database(db_name)
            
            if db is None:
                raise DatabaseError(f"Database not found: {db_name}")
            
            with db.transaction():
                return func(db, *args, **kwargs)
        
        return wrapper
    
    return decorator


def with_system_db(func):
    """
    Decorator to provide the system database to a function.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        db = get_system_database()
        return func(db, *args, **kwargs)
    
    return wrapper


def with_system_transaction(func):
    """
    Decorator to execute a function within a system database transaction.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        db = get_system_database()
        
        with db.transaction():
            return func(db, *args, **kwargs)
    
    return wrapper


# Convenience functions for system database

@with_system_db
def get_setting(db: Database, key: str, default: Any = None) -> Any:
    """
    Get a system setting.
    
    Args:
        db: Database instance
        key: Setting key
        default: Default value if setting is not found
        
    Returns:
        Setting value or default
    """
    result = db.query_one("SELECT value FROM settings WHERE key = ?", (key,))
    
    if result:
        return db.json_deserialize(result['value'])
    
    return default


@with_system_transaction
def set_setting(db: Database, key: str, value: Any) -> None:
    """
    Set a system setting.
    
    Args:
        db: Database instance
        key: Setting key
        value: Setting value
    """
    serialized = db.json_serialize(value)
    
    # Check if setting exists
    exists = db.query_one("SELECT 1 FROM settings WHERE key = ?", (key,))
    
    if exists:
        # Update existing setting
        db.update(
            'settings',
            {'value': serialized, 'updated_at': time.time()},
            'key = ?',
            (key,)
        )
    else:
        # Insert new setting
        db.insert(
            'settings',
            {'key': key, 'value': serialized, 'updated_at': time.time()}
        )


@with_system_transaction
def delete_setting(db: Database, key: str) -> bool:
    """
    Delete a system setting.
    
    Args:
        db: Database instance
        key: Setting key
        
    Returns:
        True if setting was deleted, False if not found
    """
    rowcount = db.delete('settings', 'key = ?', (key,))
    return rowcount > 0


@with_system_db
def list_settings(db: Database) -> Dict[str, Any]:
    """
    List all system settings.
    
    Args:
        db: Database instance
        
    Returns:
        Dictionary of settings (key -> value)
    """
    rows = db.query("SELECT key, value FROM settings")
    
    return {
        row['key']: db.json_deserialize(row['value'])
        for row in rows
    }


@with_system_transaction
def register_project(db: Database, project_id: str, name: str, 
                   root_directory: Optional[str] = None,
                   output_directory: Optional[str] = None) -> None:
    """
    Register a project in the system database.
    
    Args:
        db: Database instance
        project_id: Project ID
        name: Project name
        root_directory: Optional root directory
        output_directory: Optional output directory
    """
    # Check if project exists
    exists = db.query_one("SELECT 1 FROM projects WHERE project_id = ?", (project_id,))
    
    current_time = time.time()
    
    if exists:
        # Update existing project
        db.update(
            'projects',
            {
                'name': name,
                'updated_at': current_time,
                'root_directory': root_directory,
                'output_directory': output_directory
            },
            'project_id = ?',
            (project_id,)
        )
    else:
        # Insert new project
        db.insert(
            'projects',
            {
                'project_id': project_id,
                'name': name,
                'created_at': current_time,
                'updated_at': current_time,
                'root_directory': root_directory,
                'output_directory': output_directory,
                'is_active': 0
            }
        )


@with_system_transaction
def set_active_project(db: Database, project_id: str) -> bool:
    """
    Set the active project in the system database.
    
    Args:
        db: Database instance
        project_id: Project ID
        
    Returns:
        True if active project was set, False if project not found
    """
    # Check if project exists
    exists = db.query_one("SELECT 1 FROM projects WHERE project_id = ?", (project_id,))
    
    if not exists:
        return False
    
    # Update all projects to inactive
    db.update(
        'projects',
        {'is_active': 0},
        '1 = 1',
        ()
    )
    
    # Set the specified project as active
    db.update(
        'projects',
        {'is_active': 1, 'updated_at': time.time()},
        'project_id = ?',
        (project_id,)
    )
    
    return True


@with_system_db
def get_active_project(db: Database) -> Optional[Dict[str, Any]]:
    """
    Get the active project from the system database.
    
    Args:
        db: Database instance
        
    Returns:
        Active project data or None if no active project
    """
    return db.query_one("SELECT * FROM projects WHERE is_active = 1")


@with_system_db
def list_projects(db: Database) -> List[Dict[str, Any]]:
    """
    List all projects in the system database.
    
    Args:
        db: Database instance
        
    Returns:
        List of project data
    """
    return db.query("SELECT * FROM projects ORDER BY updated_at DESC")