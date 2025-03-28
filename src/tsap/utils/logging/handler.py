"""
Custom log handlers for the TSAP logging system.

This module provides specialized log handlers that can be used to route
log messages to various destinations, including files, databases, network
services, and other custom targets. Each handler is designed to integrate
with the Rich formatting system used throughout TSAP.
"""

import os
import logging
import threading
import json
import time
import sqlite3
from typing import Dict, Any, Optional

from rich.console import Console

from tsap.utils.logging.formatter import TSAPLogRecord, DetailedLogFormatter
from tsap.utils.errors import TSAPError
from tsap.config import get_config


class HandlerError(TSAPError):
    """Exception raised for errors in log handlers."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, code="HANDLER_ERROR", details=details)


class RichFileHandler(logging.Handler):
    """
    A logging handler that writes Rich-formatted logs to a file.
    
    This handler captures the Rich console output and writes it to a file,
    preserving colors and formatting if the file format supports it (e.g., HTML).
    
    Attributes:
        file_path: Path to the log file
        mode: File open mode ('a' for append, 'w' for write)
        encoding: File encoding
        formatter: Log formatter
        console: Rich console for rendering
    """
    def __init__(
        self,
        file_path: str,
        mode: str = 'a',
        encoding: str = 'utf-8',
        level: int = logging.NOTSET,
        formatter: Optional[logging.Formatter] = None,
        rich_tracebacks: bool = True,
        markup: bool = False,
        highlight: bool = False
    ) -> None:
        """
        Initialize a new Rich file handler.
        
        Args:
            file_path: Path to the log file
            mode: File open mode ('a' for append, 'w' for write)
            encoding: File encoding
            level: Logging level
            formatter: Log formatter
            rich_tracebacks: Whether to use Rich for formatting tracebacks
            markup: Whether to interpret Rich markup in log messages
            highlight: Whether to syntax highlight log messages
        """
        super().__init__(level)
        self.file_path = file_path
        self.mode = mode
        self.encoding = encoding
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Set up Rich console writing to the file
        self.file = open(file_path, mode, encoding=encoding)
        self.console = Console(
            file=self.file,
            force_terminal=False,
            color_system="standard",
            width=120,
            markup=markup,
            highlight=highlight
        )
        
        # Set formatter
        if formatter is None:
            self.formatter = DetailedLogFormatter(show_time=True, show_level=True, show_component=True)
        else:
            self.formatter = formatter
        
        self.rich_tracebacks = rich_tracebacks
        self._lock = threading.RLock()
    
    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record to the file.
        
        Args:
            record: Log record to emit
        """
        try:
            with self._lock:
                # Convert to TSAP log record
                tsap_record = TSAPLogRecord(record)
                
                # Format the record
                formatted = self.formatter.format_record(tsap_record)
                
                # Write to the file
                self.console.print(formatted)
                self.file.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)
    
    def close(self) -> None:
        """Close the log file."""
        if hasattr(self, 'file') and self.file:
            self.file.close()
        super().close()


class RotatingFileHandler(RichFileHandler):
    """
    A handler that writes Rich-formatted logs to a set of rotating files.
    
    The handler rotates files when they reach a certain size or at specified time intervals.
    
    Attributes:
        maxBytes: Maximum size in bytes before rotating
        backupCount: Number of backup files to keep
        interval: Rotation interval in seconds
        when: When to rotate ('S' for seconds, 'M' for minutes, 'H' for hours, 'D' for days)
    """
    def __init__(
        self,
        file_path: str,
        mode: str = 'a',
        encoding: str = 'utf-8',
        level: int = logging.NOTSET,
        formatter: Optional[logging.Formatter] = None,
        maxBytes: int = 0,
        backupCount: int = 0,
        interval: int = 0,
        when: str = 'D'
    ) -> None:
        """
        Initialize a new rotating file handler.
        
        Args:
            file_path: Path to the log file
            mode: File open mode ('a' for append, 'w' for write)
            encoding: File encoding
            level: Logging level
            formatter: Log formatter
            maxBytes: Maximum size in bytes before rotating
            backupCount: Number of backup files to keep
            interval: Rotation interval in seconds
            when: When to rotate ('S' for seconds, 'M' for minutes, 'H' for hours, 'D' for days)
        """
        super().__init__(file_path, mode, encoding, level, formatter)
        
        self.maxBytes = maxBytes
        self.backupCount = backupCount
        self.interval = interval
        self.when = when
        
        self._next_rollover = self._compute_next_rollover()
    
    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record to the file, rotating if necessary.
        
        Args:
            record: Log record to emit
        """
        try:
            with self._lock:
                # Check if we need to rotate
                if self._should_rollover():
                    self._do_rollover()
                
                # Emit the record
                super().emit(record)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)
    
    def _should_rollover(self) -> bool:
        """
        Check if the log file should be rotated.
        
        Returns:
            True if the file should be rotated, False otherwise
        """
        # Check size-based rotation
        if self.maxBytes > 0:
            self.file.flush()
            if os.path.getsize(self.file_path) >= self.maxBytes:
                return True
        
        # Check time-based rotation
        if self.interval > 0 and time.time() >= self._next_rollover:
            return True
        
        return False
    
    def _do_rollover(self) -> None:
        """Rotate the log files."""
        self.file.close()
        
        if self.backupCount > 0:
            # Shift existing backup files
            for i in range(self.backupCount - 1, 0, -1):
                src = f"{self.file_path}.{i}"
                dst = f"{self.file_path}.{i + 1}"
                
                if os.path.exists(src):
                    if os.path.exists(dst):
                        os.remove(dst)
                    os.rename(src, dst)
            
            # Rename current file to .1
            if os.path.exists(self.file_path):
                dst = f"{self.file_path}.1"
                if os.path.exists(dst):
                    os.remove(dst)
                os.rename(self.file_path, dst)
        
        # Open new file
        self.file = open(self.file_path, self.mode, encoding=self.encoding)
        
        # Update Rich console
        self.console = Console(
            file=self.file,
            force_terminal=False,
            color_system="standard",
            width=120
        )
        
        # Update next rollover time
        if self.interval > 0:
            self._next_rollover = self._compute_next_rollover()
    
    def _compute_next_rollover(self) -> float:
        """
        Compute the timestamp for the next rotation.
        
        Returns:
            Timestamp for the next rotation
        """
        if self.interval <= 0:
            return 0
        
        now = time.time()
        
        # Calculate interval in seconds
        interval_seconds = self.interval
        if self.when == 'M':
            interval_seconds = self.interval * 60
        elif self.when == 'H':
            interval_seconds = self.interval * 60 * 60
        elif self.when == 'D':
            interval_seconds = self.interval * 60 * 60 * 24
        
        # Calculate next rollover time
        next_rollover = now + interval_seconds
        
        return next_rollover


class DatabaseHandler(logging.Handler):
    """
    A logging handler that writes log records to a database.
    
    The handler can write to SQLite, MySQL, PostgreSQL, or other databases
    using appropriate connectors.
    
    Attributes:
        db_path: Path to the database file (for SQLite)
        table_name: Name of the table to write to
        connection: Database connection
    """
    def __init__(
        self,
        db_path: str,
        table_name: str = 'logs',
        level: int = logging.NOTSET
    ) -> None:
        """
        Initialize a new database handler.
        
        Args:
            db_path: Path to the database file (for SQLite)
            table_name: Name of the table to write to
            level: Logging level
        """
        super().__init__(level)
        self.db_path = db_path
        self.table_name = table_name
        self._lock = threading.RLock()
        
        # Create database directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        
        # Initialize database
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the database and create the table if it doesn't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    level TEXT,
                    logger TEXT,
                    component TEXT,
                    message TEXT,
                    exception TEXT,
                    extra TEXT
                )
            ''')
            conn.commit()
    
    def _get_connection(self) -> sqlite3.Connection:
        """
        Get a database connection.
        
        Returns:
            Database connection
        """
        return sqlite3.connect(self.db_path)
    
    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record to the database.
        
        Args:
            record: Log record to emit
        """
        try:
            with self._lock:
                # Convert record to a database row
                timestamp = record.created
                level = record.levelname
                logger_name = record.name
                component = getattr(record, 'component', 'unspecified')
                message = self.format(record)
                exception = record.exc_text if record.exc_text else None
                
                # Convert extra attributes to JSON
                extra = {}
                for key, value in record.__dict__.items():
                    if key not in ('timestamp', 'level', 'name', 'component', 'message', 'exc_text'):
                        try:
                            # Only include serializable values
                            json.dumps({key: value})
                            extra[key] = value
                        except (TypeError, ValueError):
                            pass
                
                extra_json = json.dumps(extra)
                
                # Insert into database
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        f'''
                        INSERT INTO {self.table_name}
                        (timestamp, level, logger, component, message, exception, extra)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''',
                        (timestamp, level, logger_name, component, message, exception, extra_json)
                    )
                    conn.commit()
                
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


class NetworkHandler(logging.Handler):
    """
    A logging handler that sends log records to a network service.
    
    The handler can send logs to various network destinations, such as
    a centralized logging server, a webhook, or a messaging service.
    
    Attributes:
        url: URL of the network service
        method: HTTP method ('GET', 'POST')
        headers: HTTP headers
        timeout: Request timeout in seconds
    """
    def __init__(
        self,
        url: str,
        method: str = 'POST',
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 5.0,
        level: int = logging.NOTSET,
        formatter: Optional[logging.Formatter] = None
    ) -> None:
        """
        Initialize a new network handler.
        
        Args:
            url: URL of the network service
            method: HTTP method ('GET', 'POST')
            headers: HTTP headers
            timeout: Request timeout in seconds
            level: Logging level
            formatter: Log formatter
        """
        super().__init__(level)
        self.url = url
        self.method = method.upper()
        self.headers = headers or {'Content-Type': 'application/json'}
        self.timeout = timeout
        
        if formatter is None:
            self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        else:
            self.formatter = formatter
        
        self._lock = threading.RLock()
        self._queue = []  # Queue for failed messages
        self._retry_thread = None
        self._retry_interval = 60  # Retry every 60 seconds
        self._max_retries = 3  # Maximum number of retries
        self._shutdown = threading.Event()
    
    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record to the network service.
        
        Args:
            record: Log record to emit
        """
        try:
            with self._lock:
                # Format the record
                formatted = self.format(record)
                
                # Send to network (placeholder - would use requests or aiohttp in reality)
                success = self._send_log(record, formatted)
                
                # Queue for retry if failed
                if not success:
                    self._queue.append((record, formatted, 0))  # Record, formatted message, retry count
                    
                    # Start retry thread if not already running
                    if self._retry_thread is None or not self._retry_thread.is_alive():
                        self._start_retry_thread()
        
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)
    
    def _send_log(self, record: logging.LogRecord, formatted: str) -> bool:
        """
        Send a log record to the network service.
        
        Args:
            record: Log record to send
            formatted: Formatted log message
            
        Returns:
            True if the log was sent successfully, False otherwise
        """
        # This is a placeholder - in a real implementation, this would use
        # requests, aiohttp, or another HTTP client to send the log
        
        # Example implementation:
        # try:
        #     import requests
        #     
        #     data = {
        #         'timestamp': record.created,
        #         'level': record.levelname,
        #         'logger': record.name,
        #         'component': getattr(record, 'component', 'unspecified'),
        #         'message': formatted,
        #         'exception': record.exc_text if record.exc_text else None
        #     }
        #     
        #     response = requests.request(
        #         method=self.method,
        #         url=self.url,
        #         headers=self.headers,
        #         json=data,
        #         timeout=self.timeout
        #     )
        #     
        #     return response.status_code >= 200 and response.status_code < 300
        # except Exception as e:
        #     return False
        
        # For now, just simulate success
        return True
    
    def _start_retry_thread(self) -> None:
        """Start a thread to retry failed log sends."""
        self._retry_thread = threading.Thread(target=self._retry_worker, daemon=True)
        self._retry_thread.start()
    
    def _retry_worker(self) -> None:
        """Worker thread for retrying failed log sends."""
        while not self._shutdown.is_set():
            # Wait for retry interval
            if self._shutdown.wait(self._retry_interval):
                break
            
            # Retry failed sends
            with self._lock:
                still_queued = []
                
                for record, formatted, retry_count in self._queue:
                    if retry_count >= self._max_retries:
                        # Exceeded max retries, drop the log
                        continue
                    
                    # Try to send again
                    success = self._send_log(record, formatted)
                    
                    if not success:
                        still_queued.append((record, formatted, retry_count + 1))
                
                self._queue = still_queued
                
                # If queue is empty, stop the thread
                if not self._queue:
                    break
    
    def close(self) -> None:
        """Close the handler and stop the retry thread."""
        self._shutdown.set()
        if self._retry_thread and self._retry_thread.is_alive():
            self._retry_thread.join(timeout=1.0)
        super().close()


def get_rotating_file_handler(
    file_path: Optional[str] = None,
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5
) -> RotatingFileHandler:
    """
    Create a rotating file handler with default settings.
    
    Args:
        file_path: Path to the log file
        level: Logging level
        max_bytes: Maximum size in bytes before rotating
        backup_count: Number of backup files to keep
        
    Returns:
        Rotating file handler
    """
    if file_path is None:
        config = get_config()
        log_dir = config.server.log_directory
        file_path = os.path.join(log_dir, 'tsap.log')
    
    handler = RotatingFileHandler(
        file_path=file_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        level=level
    )
    
    return handler


def get_database_handler(
    db_path: Optional[str] = None,
    level: int = logging.INFO
) -> DatabaseHandler:
    """
    Create a database handler with default settings.
    
    Args:
        db_path: Path to the database file
        level: Logging level
        
    Returns:
        Database handler
    """
    if db_path is None:
        config = get_config()
        log_dir = config.server.log_directory
        db_path = os.path.join(log_dir, 'tsap_logs.db')
    
    handler = DatabaseHandler(
        db_path=db_path,
        level=level
    )
    
    return handler