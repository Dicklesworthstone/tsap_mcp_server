"""
Cache persistence mechanisms beyond simple file storage.

This module provides advanced cache persistence options including:
- Database-backed storage (SQLite)
- Redis-backed storage
- Distributed cache with synchronization
- Optimized binary serialization
- Compression strategies
"""

import os
import json
import time
import pickle
import sqlite3
import gzip
import threading
import zlib
import hashlib
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from contextlib import contextmanager

from tsap.utils.errors import TSAPError
from tsap.config import get_config


class CachePersistenceError(TSAPError):
    """
    Exception raised for errors in cache persistence operations.
    
    Attributes:
        message: Error message
        details: Additional error details
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, code="CACHE_PERSISTENCE_ERROR", details=details)


class CachePersistenceProvider(ABC):
    """
    Abstract base class for cache persistence providers.
    
    A persistence provider handles storing and retrieving cache entries
    in a persistent backend (file, database, external service, etc.).
    """
    
    @abstractmethod
    async def load(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Load a cache entry from the persistence backend.
        
        Args:
            key: Cache entry key
        
        Returns:
            The cache entry (including value and metadata) or None if not found
            
        Raises:
            CachePersistenceError: If the load operation fails
        """
        pass
    
    @abstractmethod
    async def save(self, key: str, entry: Dict[str, Any]) -> None:
        """
        Save a cache entry to the persistence backend.
        
        Args:
            key: Cache entry key
            entry: Cache entry (including value and metadata)
            
        Raises:
            CachePersistenceError: If the save operation fails
        """
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """
        Delete a cache entry from the persistence backend.
        
        Args:
            key: Cache entry key
            
        Returns:
            True if the entry was deleted, False if it didn't exist
            
        Raises:
            CachePersistenceError: If the delete operation fails
        """
        pass
    
    @abstractmethod
    async def contains(self, key: str) -> bool:
        """
        Check if the persistence backend contains a cache entry.
        
        Args:
            key: Cache entry key
            
        Returns:
            True if the entry exists, False otherwise
            
        Raises:
            CachePersistenceError: If the check operation fails
        """
        pass
    
    @abstractmethod
    async def clear(self) -> int:
        """
        Clear all cache entries from the persistence backend.
        
        Returns:
            Number of entries cleared
            
        Raises:
            CachePersistenceError: If the clear operation fails
        """
        pass
    
    @abstractmethod
    async def get_keys(self) -> List[str]:
        """
        Get all cache keys from the persistence backend.
        
        Returns:
            List of cache keys
            
        Raises:
            CachePersistenceError: If the get_keys operation fails
        """
        pass
    
    @abstractmethod
    async def get_size(self) -> int:
        """
        Get the total size of the cache in the persistence backend.
        
        Returns:
            Size in bytes
            
        Raises:
            CachePersistenceError: If the get_size operation fails
        """
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache in the persistence backend.
        
        Returns:
            Dictionary of statistics
            
        Raises:
            CachePersistenceError: If the get_stats operation fails
        """
        pass


class FilePersistenceProvider(CachePersistenceProvider):
    """
    File-based cache persistence provider.
    
    Stores cache entries as files in a directory, with options for
    compression and serialization format.
    """
    
    def __init__(
        self, 
        cache_dir: str,
        compression: bool = True,
        serialization_format: str = "json",
        max_file_size: int = 100 * 1024 * 1024  # 100 MB
    ) -> None:
        """
        Initialize a new file persistence provider.
        
        Args:
            cache_dir: Directory to store cache files in
            compression: Whether to compress cache files
            serialization_format: Format to use for serialization ('json' or 'pickle')
            max_file_size: Maximum size of a cache file in bytes
            
        Raises:
            CachePersistenceError: If the cache directory cannot be created or is not writable
        """
        self.cache_dir = os.path.abspath(cache_dir)
        self.compression = compression
        self.serialization_format = serialization_format
        self.max_file_size = max_file_size
        
        # Create cache directory if it doesn't exist
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except Exception as e:
            raise CachePersistenceError(
                f"Failed to create cache directory {self.cache_dir}: {str(e)}",
                details={"cache_dir": self.cache_dir}
            )
        
        # Check if cache directory is writable
        if not os.access(self.cache_dir, os.W_OK):
            raise CachePersistenceError(
                f"Cache directory {self.cache_dir} is not writable",
                details={"cache_dir": self.cache_dir}
            )
    
    def _get_file_path(self, key: str) -> str:
        """
        Get the file path for a cache key.
        
        Args:
            key: Cache key
            
        Returns:
            Path to the cache file
        """
        # Use a hash of the key as the filename to avoid invalid characters
        key_hash = hashlib.md5(key.encode("utf-8")).hexdigest()
        file_name = f"{key_hash}.cache"
        
        if self.compression:
            file_name += ".gz"
        
        return os.path.join(self.cache_dir, file_name)
    
    def _serialize(self, entry: Dict[str, Any]) -> bytes:
        """
        Serialize a cache entry to bytes.
        
        Args:
            entry: Cache entry to serialize
            
        Returns:
            Serialized entry as bytes
            
        Raises:
            CachePersistenceError: If serialization fails
        """
        try:
            if self.serialization_format == "json":
                # Use a more compact JSON format
                data = json.dumps(entry, separators=(",", ":")).encode("utf-8")
            elif self.serialization_format == "pickle":
                # Use pickle protocol 4 for better performance with large objects
                data = pickle.dumps(entry, protocol=4)
            else:
                raise CachePersistenceError(
                    f"Unsupported serialization format: {self.serialization_format}",
                    details={"format": self.serialization_format}
                )
            
            # Apply compression if enabled
            if self.compression:
                data = gzip.compress(data)
                
            return data
        
        except Exception as e:
            raise CachePersistenceError(
                f"Failed to serialize cache entry: {str(e)}",
                details={"error_type": type(e).__name__}
            )
    
    def _deserialize(self, data: bytes) -> Dict[str, Any]:
        """
        Deserialize a cache entry from bytes.
        
        Args:
            data: Serialized cache entry
            
        Returns:
            Deserialized cache entry
            
        Raises:
            CachePersistenceError: If deserialization fails
        """
        try:
            # Decompress if necessary
            if self.compression:
                try:
                    data = gzip.decompress(data)
                except Exception as e:
                    raise CachePersistenceError(
                        f"Failed to decompress cache entry: {str(e)}",
                        details={"error_type": type(e).__name__}
                    )
            
            # Deserialize based on format
            if self.serialization_format == "json":
                return json.loads(data.decode("utf-8"))
            elif self.serialization_format == "pickle":
                return pickle.loads(data)
            else:
                raise CachePersistenceError(
                    f"Unsupported serialization format: {self.serialization_format}",
                    details={"format": self.serialization_format}
                )
        
        except Exception as e:
            if not isinstance(e, CachePersistenceError):
                raise CachePersistenceError(
                    f"Failed to deserialize cache entry: {str(e)}",
                    details={"error_type": type(e).__name__}
                )
            raise
    
    async def load(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Load a cache entry from a file.
        
        Args:
            key: Cache entry key
        
        Returns:
            The cache entry or None if not found
            
        Raises:
            CachePersistenceError: If the load operation fails
        """
        file_path = self._get_file_path(key)
        
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, "rb") as f:
                data = f.read()
                
            return self._deserialize(data)
            
        except Exception as e:
            if not isinstance(e, CachePersistenceError):
                raise CachePersistenceError(
                    f"Failed to load cache entry from {file_path}: {str(e)}",
                    details={"key": key, "file_path": file_path, "error_type": type(e).__name__}
                )
            raise
    
    async def save(self, key: str, entry: Dict[str, Any]) -> None:
        """
        Save a cache entry to a file.
        
        Args:
            key: Cache entry key
            entry: Cache entry
            
        Raises:
            CachePersistenceError: If the save operation fails
        """
        file_path = self._get_file_path(key)
        
        try:
            # Serialize the entry
            data = self._serialize(entry)
            
            # Check if the data exceeds the maximum file size
            if len(data) > self.max_file_size:
                raise CachePersistenceError(
                    f"Cache entry size ({len(data)} bytes) exceeds maximum file size ({self.max_file_size} bytes)",
                    details={"key": key, "size": len(data), "max_size": self.max_file_size}
                )
            
            # Write to a temporary file first to avoid corruption if the process is interrupted
            temp_file_path = f"{file_path}.tmp"
            with open(temp_file_path, "wb") as f:
                f.write(data)
                
            # Rename the temporary file to the actual file path (atomic operation)
            os.replace(temp_file_path, file_path)
            
        except Exception as e:
            if not isinstance(e, CachePersistenceError):
                raise CachePersistenceError(
                    f"Failed to save cache entry to {file_path}: {str(e)}",
                    details={"key": key, "file_path": file_path, "error_type": type(e).__name__}
                )
            raise
    
    async def delete(self, key: str) -> bool:
        """
        Delete a cache entry file.
        
        Args:
            key: Cache entry key
            
        Returns:
            True if the entry was deleted, False if it didn't exist
            
        Raises:
            CachePersistenceError: If the delete operation fails
        """
        file_path = self._get_file_path(key)
        
        if not os.path.exists(file_path):
            return False
        
        try:
            os.remove(file_path)
            return True
            
        except Exception as e:
            raise CachePersistenceError(
                f"Failed to delete cache entry {file_path}: {str(e)}",
                details={"key": key, "file_path": file_path, "error_type": type(e).__name__}
            )
    
    async def contains(self, key: str) -> bool:
        """
        Check if a cache entry file exists.
        
        Args:
            key: Cache entry key
            
        Returns:
            True if the entry exists, False otherwise
        """
        file_path = self._get_file_path(key)
        return os.path.exists(file_path)
    
    async def clear(self) -> int:
        """
        Clear all cache entry files.
        
        Returns:
            Number of entries cleared
            
        Raises:
            CachePersistenceError: If the clear operation fails
        """
        try:
            count = 0
            
            # Get all cache files
            for file_name in os.listdir(self.cache_dir):
                if file_name.endswith(".cache") or file_name.endswith(".cache.gz"):
                    file_path = os.path.join(self.cache_dir, file_name)
                    
                    # Skip non-files (e.g., subdirectories)
                    if not os.path.isfile(file_path):
                        continue
                    
                    # Delete the file
                    os.remove(file_path)
                    count += 1
            
            return count
            
        except Exception as e:
            raise CachePersistenceError(
                f"Failed to clear cache: {str(e)}",
                details={"cache_dir": self.cache_dir, "error_type": type(e).__name__}
            )
    
    async def get_keys(self) -> List[str]:
        """
        Get all cache keys from file names.
        
        Returns:
            List of cache keys
            
        Raises:
            CachePersistenceError: If the get_keys operation fails
        """
        # This implementation is approximate since we only have file hashes, not original keys
        try:
            keys = []
            
            for file_name in os.listdir(self.cache_dir):
                if file_name.endswith(".cache") or file_name.endswith(".cache.gz"):
                    # Extract the hash from the file name
                    key_hash = file_name.split('.')[0]
                    keys.append(key_hash)
            
            return keys
            
        except Exception as e:
            raise CachePersistenceError(
                f"Failed to get cache keys: {str(e)}",
                details={"cache_dir": self.cache_dir, "error_type": type(e).__name__}
            )
    
    async def get_size(self) -> int:
        """
        Get the total size of all cache files.
        
        Returns:
            Size in bytes
            
        Raises:
            CachePersistenceError: If the get_size operation fails
        """
        try:
            total_size = 0
            
            for file_name in os.listdir(self.cache_dir):
                if file_name.endswith(".cache") or file_name.endswith(".cache.gz"):
                    file_path = os.path.join(self.cache_dir, file_name)
                    
                    # Skip non-files (e.g., subdirectories)
                    if not os.path.isfile(file_path):
                        continue
                    
                    # Add file size
                    total_size += os.path.getsize(file_path)
            
            return total_size
            
        except Exception as e:
            raise CachePersistenceError(
                f"Failed to get cache size: {str(e)}",
                details={"cache_dir": self.cache_dir, "error_type": type(e).__name__}
            )
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache files.
        
        Returns:
            Dictionary of statistics
            
        Raises:
            CachePersistenceError: If the get_stats operation fails
        """
        try:
            # Get file count and size
            file_count = 0
            total_size = 0
            oldest_file_time = None
            newest_file_time = None
            
            for file_name in os.listdir(self.cache_dir):
                if file_name.endswith(".cache") or file_name.endswith(".cache.gz"):
                    file_path = os.path.join(self.cache_dir, file_name)
                    
                    # Skip non-files (e.g., subdirectories)
                    if not os.path.isfile(file_path):
                        continue
                    
                    # Update counters
                    file_count += 1
                    size = os.path.getsize(file_path)
                    total_size += size
                    
                    # Update timestamps
                    mtime = os.path.getmtime(file_path)
                    if oldest_file_time is None or mtime < oldest_file_time:
                        oldest_file_time = mtime
                    if newest_file_time is None or mtime > newest_file_time:
                        newest_file_time = mtime
            
            # Calculate statistics
            avg_size = total_size / file_count if file_count > 0 else 0
            
            return {
                "file_count": file_count,
                "total_size": total_size,
                "avg_size": avg_size,
                "oldest_file_time": oldest_file_time,
                "newest_file_time": newest_file_time,
                "provider_type": "file",
                "compression": self.compression,
                "serialization_format": self.serialization_format
            }
            
        except Exception as e:
            raise CachePersistenceError(
                f"Failed to get cache stats: {str(e)}",
                details={"cache_dir": self.cache_dir, "error_type": type(e).__name__}
            )


class SqlitePersistenceProvider(CachePersistenceProvider):
    """
    SQLite-based cache persistence provider.
    
    Stores cache entries in a SQLite database, with options for
    compression and serialization format.
    """
    
    def __init__(
        self, 
        db_path: str,
        compression: bool = True,
        serialization_format: str = "json",
        vacuum_interval: int = 100,  # Vacuum after this many deletions
        max_entry_size: int = 50 * 1024 * 1024  # 50 MB
    ) -> None:
        """
        Initialize a new SQLite persistence provider.
        
        Args:
            db_path: Path to the SQLite database file
            compression: Whether to compress cache entries
            serialization_format: Format to use for serialization ('json' or 'pickle')
            vacuum_interval: Number of deletions after which to vacuum the database
            max_entry_size: Maximum size of a cache entry in bytes
            
        Raises:
            CachePersistenceError: If the database cannot be created or initialized
        """
        self.db_path = os.path.abspath(db_path)
        self.compression = compression
        self.serialization_format = serialization_format
        self.vacuum_interval = vacuum_interval
        self.max_entry_size = max_entry_size
        
        # Thread-local storage for database connections
        self._local = threading.local()
        
        # Deletion counter for vacuum
        self._deletion_counter = 0
        self._deletion_lock = threading.Lock()
        
        # Create database directory if it doesn't exist
        db_dir = os.path.dirname(self.db_path)
        try:
            os.makedirs(db_dir, exist_ok=True)
        except Exception as e:
            raise CachePersistenceError(
                f"Failed to create database directory {db_dir}: {str(e)}",
                details={"db_dir": db_dir}
            )
        
        # Initialize database
        try:
            with self._get_connection() as conn:
                # Create cache entries table if it doesn't exist
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        key TEXT PRIMARY KEY,
                        value BLOB NOT NULL,
                        created_at REAL NOT NULL,
                        accessed_at REAL NOT NULL,
                        expires_at REAL,
                        size INTEGER NOT NULL
                    )
                """)
                
                # Create index on expires_at for faster TTL expiration
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_cache_entries_expires_at
                    ON cache_entries (expires_at)
                """)
                
                # Create metadata table if it doesn't exist
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache_metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    )
                """)
                
                # Set provider metadata
                conn.execute("""
                    INSERT OR REPLACE INTO cache_metadata (key, value)
                    VALUES (?, ?)
                """, ("provider_type", "sqlite"))
                
                conn.execute("""
                    INSERT OR REPLACE INTO cache_metadata (key, value)
                    VALUES (?, ?)
                """, ("compression", str(self.compression)))
                
                conn.execute("""
                    INSERT OR REPLACE INTO cache_metadata (key, value)
                    VALUES (?, ?)
                """, ("serialization_format", self.serialization_format))
                
                conn.execute("""
                    INSERT OR REPLACE INTO cache_metadata (key, value)
                    VALUES (?, ?)
                """, ("created_at", str(time.time())))
                
                conn.commit()
                
        except Exception as e:
            raise CachePersistenceError(
                f"Failed to initialize SQLite database at {self.db_path}: {str(e)}",
                details={"db_path": self.db_path, "error_type": type(e).__name__}
            )
    
    @contextmanager
    def _get_connection(self) -> sqlite3.Connection:
        """
        Get a SQLite connection from the thread-local pool.
        
        Returns:
            SQLite connection
            
        Raises:
            CachePersistenceError: If the connection cannot be established
        """
        try:
            # Check if connection exists in thread-local storage
            if not hasattr(self._local, "connection"):
                # Create new connection
                self._local.connection = sqlite3.connect(self.db_path, timeout=30.0)
                
                # Enable WAL mode for better concurrency
                self._local.connection.execute("PRAGMA journal_mode=WAL")
                
                # Use write-ahead logging for better performance
                self._local.connection.execute("PRAGMA synchronous=NORMAL")
                
                # Use reasonable cache size
                self._local.connection.execute("PRAGMA cache_size=-4000")  # 4MB cache
            
            # Yield the connection
            conn = self._local.connection
            yield conn
            
            # No need to close the connection - we're reusing it
            
        except Exception as e:
            raise CachePersistenceError(
                f"Failed to get SQLite connection: {str(e)}",
                details={"db_path": self.db_path, "error_type": type(e).__name__}
            )
    
    def _serialize(self, entry: Dict[str, Any]) -> bytes:
        """
        Serialize a cache entry to bytes.
        
        Args:
            entry: Cache entry to serialize
            
        Returns:
            Serialized entry as bytes
            
        Raises:
            CachePersistenceError: If serialization fails
        """
        try:
            if self.serialization_format == "json":
                # Use a more compact JSON format
                data = json.dumps(entry, separators=(",", ":")).encode("utf-8")
            elif self.serialization_format == "pickle":
                # Use pickle protocol 4 for better performance with large objects
                data = pickle.dumps(entry, protocol=4)
            else:
                raise CachePersistenceError(
                    f"Unsupported serialization format: {self.serialization_format}",
                    details={"format": self.serialization_format}
                )
            
            # Apply compression if enabled
            if self.compression:
                data = zlib.compress(data, level=6)  # Level 6 for balance of speed/compression
                
            return data
        
        except Exception as e:
            raise CachePersistenceError(
                f"Failed to serialize cache entry: {str(e)}",
                details={"error_type": type(e).__name__}
            )
    
    def _deserialize(self, data: bytes) -> Dict[str, Any]:
        """
        Deserialize a cache entry from bytes.
        
        Args:
            data: Serialized cache entry
            
        Returns:
            Deserialized cache entry
            
        Raises:
            CachePersistenceError: If deserialization fails
        """
        try:
            # Decompress if necessary
            if self.compression:
                try:
                    data = zlib.decompress(data)
                except Exception as e:
                    raise CachePersistenceError(
                        f"Failed to decompress cache entry: {str(e)}",
                        details={"error_type": type(e).__name__}
                    )
            
            # Deserialize based on format
            if self.serialization_format == "json":
                return json.loads(data.decode("utf-8"))
            elif self.serialization_format == "pickle":
                return pickle.loads(data)
            else:
                raise CachePersistenceError(
                    f"Unsupported serialization format: {self.serialization_format}",
                    details={"format": self.serialization_format}
                )
        
        except Exception as e:
            if not isinstance(e, CachePersistenceError):
                raise CachePersistenceError(
                    f"Failed to deserialize cache entry: {str(e)}",
                    details={"error_type": type(e).__name__}
                )
            raise
    
    async def load(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Load a cache entry from the SQLite database.
        
        Args:
            key: Cache entry key
        
        Returns:
            The cache entry or None if not found
            
        Raises:
            CachePersistenceError: If the load operation fails
        """
        try:
            with self._get_connection() as conn:
                # Get the cache entry
                cursor = conn.execute("""
                    SELECT value, expires_at
                    FROM cache_entries
                    WHERE key = ?
                """, (key,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                value_blob, expires_at = row
                
                # Check if the entry has expired
                if expires_at is not None and expires_at < time.time():
                    # Delete the expired entry
                    conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    conn.commit()
                    return None
                
                # Update the last accessed time
                conn.execute("""
                    UPDATE cache_entries
                    SET accessed_at = ?
                    WHERE key = ?
                """, (time.time(), key))
                
                conn.commit()
                
                # Deserialize the entry
                return self._deserialize(value_blob)
                
        except Exception as e:
            if not isinstance(e, CachePersistenceError):
                raise CachePersistenceError(
                    f"Failed to load cache entry from SQLite: {str(e)}",
                    details={"key": key, "error_type": type(e).__name__}
                )
            raise
    
    async def save(self, key: str, entry: Dict[str, Any]) -> None:
        """
        Save a cache entry to the SQLite database.
        
        Args:
            key: Cache entry key
            entry: Cache entry
            
        Raises:
            CachePersistenceError: If the save operation fails
        """
        try:
            # Serialize the entry
            data = self._serialize(entry)
            
            # Check if the data exceeds the maximum entry size
            if len(data) > self.max_entry_size:
                raise CachePersistenceError(
                    f"Cache entry size ({len(data)} bytes) exceeds maximum entry size ({self.max_entry_size} bytes)",
                    details={"key": key, "size": len(data), "max_size": self.max_entry_size}
                )
            
            # Extract expiration time from entry metadata if available
            expires_at = entry.get("metadata", {}).get("expires_at")
            
            with self._get_connection() as conn:
                # Insert or replace the cache entry
                conn.execute("""
                    INSERT OR REPLACE INTO cache_entries (key, value, created_at, accessed_at, expires_at, size)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (key, data, time.time(), time.time(), expires_at, len(data)))
                
                conn.commit()
                
        except Exception as e:
            if not isinstance(e, CachePersistenceError):
                raise CachePersistenceError(
                    f"Failed to save cache entry to SQLite: {str(e)}",
                    details={"key": key, "error_type": type(e).__name__}
                )
            raise
    
    async def delete(self, key: str) -> bool:
        """
        Delete a cache entry from the SQLite database.
        
        Args:
            key: Cache entry key
            
        Returns:
            True if the entry was deleted, False if it didn't exist
            
        Raises:
            CachePersistenceError: If the delete operation fails
        """
        try:
            with self._get_connection() as conn:
                # Check if the entry exists
                cursor = conn.execute("SELECT 1 FROM cache_entries WHERE key = ?", (key,))
                if not cursor.fetchone():
                    return False
                
                # Delete the entry
                conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                conn.commit()
                
                # Increment deletion counter and vacuum if needed
                with self._deletion_lock:
                    self._deletion_counter += 1
                    if self._deletion_counter >= self.vacuum_interval:
                        conn.execute("VACUUM")
                        self._deletion_counter = 0
                
                return True
                
        except Exception as e:
            raise CachePersistenceError(
                f"Failed to delete cache entry from SQLite: {str(e)}",
                details={"key": key, "error_type": type(e).__name__}
            )
    
    async def contains(self, key: str) -> bool:
        """
        Check if the SQLite database contains a cache entry.
        
        Args:
            key: Cache entry key
            
        Returns:
            True if the entry exists and has not expired, False otherwise
            
        Raises:
            CachePersistenceError: If the check operation fails
        """
        try:
            with self._get_connection() as conn:
                # Check if the entry exists and has not expired
                cursor = conn.execute("""
                    SELECT expires_at
                    FROM cache_entries
                    WHERE key = ?
                """, (key,))
                
                row = cursor.fetchone()
                if not row:
                    return False
                
                expires_at = row[0]
                
                # Check if the entry has expired
                if expires_at is not None and expires_at < time.time():
                    # Delete the expired entry
                    conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    conn.commit()
                    return False
                
                return True
                
        except Exception as e:
            raise CachePersistenceError(
                f"Failed to check if cache entry exists in SQLite: {str(e)}",
                details={"key": key, "error_type": type(e).__name__}
            )
    
    async def clear(self) -> int:
        """
        Clear all cache entries from the SQLite database.
        
        Returns:
            Number of entries cleared
            
        Raises:
            CachePersistenceError: If the clear operation fails
        """
        try:
            with self._get_connection() as conn:
                # Get the count before deleting
                cursor = conn.execute("SELECT COUNT(*) FROM cache_entries")
                count = cursor.fetchone()[0]
                
                # Delete all entries
                conn.execute("DELETE FROM cache_entries")
                conn.commit()
                
                # Vacuum the database
                conn.execute("VACUUM")
                
                # Reset deletion counter
                with self._deletion_lock:
                    self._deletion_counter = 0
                
                return count
                
        except Exception as e:
            raise CachePersistenceError(
                f"Failed to clear cache in SQLite: {str(e)}",
                details={"error_type": type(e).__name__}
            )
    
    async def get_keys(self) -> List[str]:
        """
        Get all cache keys from the SQLite database.
        
        Returns:
            List of cache keys
            
        Raises:
            CachePersistenceError: If the get_keys operation fails
        """
        try:
            now = time.time()
            keys = []
            
            with self._get_connection() as conn:
                # Get all non-expired keys
                cursor = conn.execute("""
                    SELECT key
                    FROM cache_entries
                    WHERE expires_at IS NULL OR expires_at > ?
                """, (now,))
                
                for row in cursor:
                    keys.append(row[0])
                
                return keys
                
        except Exception as e:
            raise CachePersistenceError(
                f"Failed to get cache keys from SQLite: {str(e)}",
                details={"error_type": type(e).__name__}
            )
    
    async def get_size(self) -> int:
        """
        Get the total size of all cache entries in the SQLite database.
        
        Returns:
            Size in bytes
            
        Raises:
            CachePersistenceError: If the get_size operation fails
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT SUM(size) FROM cache_entries")
                result = cursor.fetchone()[0]
                
                return result or 0
                
        except Exception as e:
            raise CachePersistenceError(
                f"Failed to get cache size from SQLite: {str(e)}",
                details={"error_type": type(e).__name__}
            )
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache in the SQLite database.
        
        Returns:
            Dictionary of statistics
            
        Raises:
            CachePersistenceError: If the get_stats operation fails
        """
        try:
            now = time.time()
            stats = {}
            
            with self._get_connection() as conn:
                # Get basic stats
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_entries,
                        SUM(size) as total_size,
                        AVG(size) as avg_size,
                        MIN(created_at) as oldest_entry,
                        MAX(created_at) as newest_entry,
                        COUNT(CASE WHEN expires_at IS NOT NULL AND expires_at > ? THEN 1 END) as ttl_entries,
                        COUNT(CASE WHEN expires_at IS NOT NULL AND expires_at <= ? THEN 1 END) as expired_entries
                    FROM cache_entries
                """, (now, now))
                
                row = cursor.fetchone()
                if row:
                    stats["total_entries"] = row[0] or 0
                    stats["total_size"] = row[1] or 0
                    stats["avg_size"] = row[2] or 0
                    stats["oldest_entry_time"] = row[3]
                    stats["newest_entry_time"] = row[4]
                    stats["ttl_entries"] = row[5] or 0
                    stats["expired_entries"] = row[6] or 0
                
                # Get metadata
                cursor = conn.execute("SELECT key, value FROM cache_metadata")
                for row in cursor:
                    stats[row[0]] = row[1]
                
                # Get database size
                cursor = conn.execute("PRAGMA page_count")
                page_count = cursor.fetchone()[0]
                
                cursor = conn.execute("PRAGMA page_size")
                page_size = cursor.fetchone()[0]
                
                stats["database_size"] = page_count * page_size
                
                return stats
                
        except Exception as e:
            raise CachePersistenceError(
                f"Failed to get cache stats from SQLite: {str(e)}",
                details={"error_type": type(e).__name__}
            )


# Factory function to create a persistence provider based on configuration
def create_persistence_provider(
    provider_type: str = "file",
    config: Optional[Dict[str, Any]] = None
) -> CachePersistenceProvider:
    """
    Create a cache persistence provider.
    
    Args:
        provider_type: Type of persistence provider ('file' or 'sqlite')
        config: Configuration options for the provider
        
    Returns:
        Cache persistence provider
        
    Raises:
        CachePersistenceError: If the provider cannot be created
    """
    if config is None:
        config = {}
    
    cache_dir = config.get("cache_dir", os.path.join(os.getcwd(), "cache"))
    
    if provider_type == "file":
        return FilePersistenceProvider(
            cache_dir=config.get("cache_dir", cache_dir),
            compression=config.get("compression", True),
            serialization_format=config.get("serialization_format", "json"),
            max_file_size=config.get("max_file_size", 100 * 1024 * 1024)
        )
    elif provider_type == "sqlite":
        return SqlitePersistenceProvider(
            db_path=config.get("db_path", os.path.join(cache_dir, "cache.db")),
            compression=config.get("compression", True),
            serialization_format=config.get("serialization_format", "json"),
            vacuum_interval=config.get("vacuum_interval", 100),
            max_entry_size=config.get("max_entry_size", 50 * 1024 * 1024)
        )
    else:
        raise CachePersistenceError(
            f"Unsupported persistence provider type: {provider_type}",
            details={"provider_type": provider_type}
        )


# Get the default persistence provider based on configuration
def get_default_persistence_provider() -> CachePersistenceProvider:
    """
    Get the default cache persistence provider based on configuration.
    
    Returns:
        Cache persistence provider
        
    Raises:
        CachePersistenceError: If the provider cannot be created
    """
    config = get_config()
    
    provider_type = config.cache.persistence_provider
    cache_dir = config.cache.directory
    
    return create_persistence_provider(
        provider_type=provider_type,
        config={
            "cache_dir": cache_dir,
            "compression": config.cache.compression,
            "serialization_format": config.cache.serialization_format
        }
    )