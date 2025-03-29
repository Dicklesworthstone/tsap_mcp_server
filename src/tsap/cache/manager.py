"""
Cache manager for the TSAP MCP Server.

Handles caching operations, invalidation strategies, and persistence.
"""

import os
import json
import time
import hashlib
import asyncio
import functools
from typing import Dict, Any, Optional, Union, Callable, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime

from tsap.utils.logging import logger
from tsap.config import get_config
from tsap.constants import DEFAULT_CACHE_DIR
from pydantic import BaseModel


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for cache serialization."""
    
    def default(self, obj):
        # Handle Pydantic models
        if isinstance(obj, BaseModel):
            if hasattr(obj, 'model_dump'):
                return obj.model_dump()
            elif hasattr(obj, 'dict'):
                return obj.dict()
        
        # Handle datetime objects
        if isinstance(obj, datetime):
            return obj.isoformat()
            
        # Let the base class handle it or raise TypeError
        return super().default(obj)


class CacheManager:
    """
    Manages in-memory and disk-based caching for TSAP operations.
    
    The CacheManager provides caching functionality to avoid redundant
    operations and improve performance. It supports both in-memory
    caching and disk-based persistence with various invalidation strategies.
    """
    
    def __init__(
        self,
        enabled: bool = True,
        cache_dir: Optional[str] = None,
        max_size: int = 1024 * 1024 * 100,  # 100 MB
        ttl: int = 3600,  # 1 hour
        invalidation_strategy: str = "lru"
    ):
        """
        Initialize the cache manager.
        
        Args:
            enabled: Whether caching is enabled
            cache_dir: Directory for persistent cache
            max_size: Maximum cache size in bytes
            ttl: Time-to-live for cache entries in seconds
            invalidation_strategy: Strategy for cache invalidation (lru, ttl, fifo)
        """
        self.enabled = enabled
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), DEFAULT_CACHE_DIR)
        self.max_size = max_size
        self.ttl = ttl
        self.invalidation_strategy = invalidation_strategy
        
        # In-memory cache
        self._cache: Dict[str, Dict[str, Any]] = {}
        
        # Cache metadata
        self._metadata: Dict[str, Dict[str, Any]] = {}
        
        # Access order for LRU strategy
        self._access_order: List[str] = []
        
        # Thread pool for disk operations
        self._executor = ThreadPoolExecutor(max_workers=2)
        
        # Current cache size
        self._current_size = 0
        
        # Init the cache
        self._init_cache()
        
        logger.info(
            f"Cache manager initialized with strategy: {invalidation_strategy}",
            component="cache"
        )
    
    def _init_cache(self) -> None:
        """Initialize the cache and load metadata from disk."""
        if not self.enabled:
            return
        
        # Create cache directory if it doesn't exist
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Load metadata from disk
            metadata_path = os.path.join(self.cache_dir, "metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r") as f:
                        self._metadata = json.load(f)
                    
                    # Rebuild access order
                    self._access_order = sorted(
                        self._metadata.keys(),
                        key=lambda k: self._metadata[k].get("last_access", 0)
                    )
                    
                    # Calculate current size
                    self._current_size = sum(
                        self._metadata[k].get("size", 0) for k in self._metadata
                    )
                    
                    logger.info(
                        f"Loaded cache metadata with {len(self._metadata)} entries",
                        component="cache"
                    )
                except Exception as e:
                    logger.error(
                        f"Error loading cache metadata: {str(e)}",
                        component="cache"
                    )
    
    def _get_cache_key(self, operation: str, params: Dict[str, Any]) -> str:
        """
        Generate a cache key from operation and parameters.
        
        Args:
            operation: Operation name
            params: Operation parameters
            
        Returns:
            Cache key as a string
        """
        # Handle Pydantic models by converting to dict if possible
        if hasattr(params, 'model_dump'):
            params_dict = params.model_dump()
        elif hasattr(params, 'dict'):
            params_dict = params.dict()
        else:
            params_dict = params
            
        # Serialize parameters to a consistent string
        params_str = json.dumps(params_dict, sort_keys=True, cls=CustomJSONEncoder)
        
        # Create a hash of the operation and parameters
        key = hashlib.sha256(f"{operation}:{params_str}".encode()).hexdigest()
        
        return key
    
    async def _save_metadata(self) -> None:
        """Save cache metadata to disk asynchronously."""
        if not self.enabled or not self.cache_dir:
            return
        
        try:
            metadata_path = os.path.join(self.cache_dir, "metadata.json")
            
            # Save asynchronously using thread pool
            await asyncio.get_event_loop().run_in_executor(
                self._executor,
                functools.partial(
                    self._write_json_file,
                    metadata_path,
                    self._metadata
                )
            )
        except Exception as e:
            logger.error(
                f"Error saving cache metadata: {str(e)}",
                component="cache"
            )
    
    def _write_json_file(self, path: str, data: Dict[str, Any]) -> None:
        """
        Write JSON data to a file.
        
        Args:
            path: File path
            data: Data to write
        """
        with open(path, "w") as f:
            json.dump(data, f, cls=CustomJSONEncoder)
    
    def _update_access(self, key: str) -> None:
        """
        Update access time for a cache entry.
        
        Args:
            key: Cache key
        """
        if key in self._metadata:
            # Update last access time
            self._metadata[key]["last_access"] = time.time()
            
            # Update access order for LRU
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
    
    def _is_entry_valid(self, key: str) -> bool:
        """
        Check if a cache entry is still valid (not expired).
        
        Args:
            key: Cache key
            
        Returns:
            True if valid, False otherwise
        """
        if key not in self._metadata:
            return False
        
        # Check TTL if using TTL strategy
        if self.invalidation_strategy == "ttl":
            last_access = self._metadata[key]["creation_time"]
            if time.time() - last_access > self.ttl:
                return False
        
        return True
    
    def _invalidate_entries(self, needed_space: int) -> None:
        """
        Invalidate cache entries to free up space.
        
        Args:
            needed_space: Amount of space needed in bytes
        """
        if not self._access_order:
            return
        
        if self.invalidation_strategy == "lru":
            # Least Recently Used strategy
            freed_space = 0
            while freed_space < needed_space and self._access_order:
                key = self._access_order.pop(0)  # Remove least recently used
                freed_space += self._invalidate_entry(key)
        
        elif self.invalidation_strategy == "fifo":
            # First In, First Out strategy
            freed_space = 0
            keys_by_creation = sorted(
                self._metadata.keys(),
                key=lambda k: self._metadata[k].get("creation_time", 0)
            )
            
            while freed_space < needed_space and keys_by_creation:
                key = keys_by_creation.pop(0)  # Remove oldest
                self._access_order.remove(key)
                freed_space += self._invalidate_entry(key)
    
    def _invalidate_entry(self, key: str) -> int:
        """
        Invalidate a specific cache entry.
        
        Args:
            key: Cache key
            
        Returns:
            Size of freed space in bytes
        """
        if key not in self._metadata:
            return 0
        
        freed_space = self._metadata[key].get("size", 0)
        
        # Remove from memory cache
        if key in self._cache:
            del self._cache[key]
        
        # Remove from disk if exists
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"{key}.json")
            try:
                if os.path.exists(cache_path):
                    os.remove(cache_path)
            except Exception as e:
                logger.warning(
                    f"Error removing cache file {key}: {str(e)}",
                    component="cache"
                )
        
        # Update metadata
        del self._metadata[key]
        self._current_size -= freed_space
        
        return freed_space
    
    async def get(self, operation: str, params: Dict[str, Any]) -> Optional[Any]:
        """
        Get a cached result for an operation.
        
        Args:
            operation: Operation name
            params: Operation parameters
            
        Returns:
            Cached result or None if not found
        """
        if not self.enabled:
            return None
        
        key = self._get_cache_key(operation, params)
        
        # Check if entry is valid
        if not self._is_entry_valid(key):
            return None
        
        # Check in-memory cache first
        if key in self._cache:
            self._update_access(key)
            return self._cache[key]
        
        # If not in memory but in metadata, load from disk
        if key in self._metadata and self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"{key}.json")
            try:
                if os.path.exists(cache_path):
                    # Load asynchronously using thread pool
                    result = await asyncio.get_event_loop().run_in_executor(
                        self._executor,
                        functools.partial(self._read_json_file, cache_path)
                    )
                    
                    # Store in memory cache
                    self._cache[key] = result
                    
                    # Update access time
                    self._update_access(key)
                    
                    return result
            except Exception as e:
                logger.warning(
                    f"Error loading cache entry {key}: {str(e)}",
                    component="cache"
                )
        
        return None
    
    def _read_json_file(self, path: str) -> Any:
        """
        Read JSON data from a file.
        
        Args:
            path: File path
            
        Returns:
            Loaded JSON data
        """
        with open(path, "r") as f:
            return json.load(f)
    
    async def set(self, operation: str, params: Dict[str, Any], result: Any) -> None:
        """
        Set a result in the cache.
        
        Args:
            operation: Operation name
            params: Operation parameters
            result: Operation result
        """
        if not self.enabled:
            return
        
        # Skip if result is None or not JSON serializable
        if result is None:
            return
        
        try:
            # Try to serialize to ensure it's JSON serializable
            serialized = json.dumps(result, cls=CustomJSONEncoder)
            size = len(serialized)
        except (TypeError, OverflowError) as e:
            logger.warning(
                f"Cannot cache non-serializable result for {operation}: {str(e)}",
                component="cache"
            )
            return
        
        # Check if result is too large for cache
        if size > self.max_size:
            logger.warning(
                f"Result too large for cache: {size} bytes > {self.max_size} bytes",
                component="cache"
            )
            return
        
        # Generate cache key
        key = self._get_cache_key(operation, params)
        
        # Check if we need to free up space
        if self._current_size + size > self.max_size:
            self._invalidate_entries(size)
        
        # Store in memory cache
        self._cache[key] = result
        
        # Update metadata
        self._metadata[key] = {
            "operation": operation,
            "size": size,
            "creation_time": time.time(),
            "last_access": time.time()
        }
        
        # Update cache size
        self._current_size += size
        
        # Update access order
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        
        # Store on disk if enabled
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"{key}.json")
            try:
                # Save asynchronously using thread pool
                await asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    functools.partial(
                        self._write_json_file,
                        cache_path,
                        result
                    )
                )
            except Exception as e:
                logger.warning(
                    f"Error saving cache entry {key}: {str(e)}",
                    component="cache"
                )
        
        # Save metadata
        await self._save_metadata()
    
    async def invalidate(self, operation: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> int:
        """
        Invalidate cache entries.
        
        Args:
            operation: Optional operation name to invalidate
            params: Optional parameters to invalidate
            
        Returns:
            Number of invalidated entries
        """
        if not self.enabled:
            return 0
        
        invalidated = 0
        
        if operation and params:
            # Invalidate specific operation+params
            key = self._get_cache_key(operation, params)
            if key in self._metadata:
                self._invalidate_entry(key)
                invalidated = 1
        
        elif operation:
            # Invalidate all entries for an operation
            keys_to_remove = []
            for key, meta in self._metadata.items():
                if meta.get("operation") == operation:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self._invalidate_entry(key)
                if key in self._access_order:
                    self._access_order.remove(key)
            
            invalidated = len(keys_to_remove)
        
        else:
            # Invalidate all entries
            invalidated = len(self._metadata)
            self._cache = {}
            self._metadata = {}
            self._access_order = []
            self._current_size = 0
            
            # Remove all files in cache directory
            if self.cache_dir and os.path.exists(self.cache_dir):
                try:
                    files = [f for f in os.listdir(self.cache_dir) if f.endswith(".json")]
                    for file in files:
                        os.remove(os.path.join(self.cache_dir, file))
                except Exception as e:
                    logger.warning(
                        f"Error clearing cache directory: {str(e)}",
                        component="cache"
                    )
        
        # Save metadata
        await self._save_metadata()
        
        return invalidated
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.enabled:
            return {
                "enabled": False,
                "entries": 0,
                "size": 0,
                "max_size": self.max_size
            }
        
        # Count entries by operation
        operation_counts = {}
        for meta in self._metadata.values():
            op = meta.get("operation", "unknown")
            operation_counts[op] = operation_counts.get(op, 0) + 1
        
        return {
            "enabled": True,
            "entries": len(self._metadata),
            "size": self._current_size,
            "max_size": self.max_size,
            "in_memory_entries": len(self._cache),
            "ttl": self.ttl,
            "invalidation_strategy": self.invalidation_strategy,
            "operations": operation_counts
        }


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """
    Get the global cache manager instance.
    
    Returns:
        Cache manager instance
    """
    global _cache_manager
    
    if _cache_manager is None:
        # Initialize from config
        config = get_config()
        _cache_manager = CacheManager(
            enabled=config.cache.enabled,
            cache_dir=config.cache.directory,
            max_size=config.cache.max_size,
            ttl=config.cache.ttl,
            invalidation_strategy=config.cache.invalidation_strategy
        )
    
    return _cache_manager


async def cache_result(operation: str, params: Dict[str, Any], result: Any) -> None:
    """
    Cache an operation result.
    
    Args:
        operation: Operation name
        params: Operation parameters
        result: Operation result
    """
    cache_manager = get_cache_manager()
    await cache_manager.set(operation, params, result)


async def get_cached_result(operation: str, params: Dict[str, Any]) -> Optional[Any]:
    """
    Get a cached operation result.
    
    Args:
        operation: Operation name
        params: Operation parameters
        
    Returns:
        Cached result or None if not found
    """
    cache_manager = get_cache_manager()
    return await cache_manager.get(operation, params)


async def invalidate_cache_entry(operation: str, params: Optional[Dict[str, Any]] = None) -> int:
    """
    Invalidate cache entries for an operation.
    
    Args:
        operation: Operation name
        params: Optional parameters to invalidate
        
    Returns:
        Number of invalidated entries
    """
    cache_manager = get_cache_manager()
    return await cache_manager.invalidate(operation, params)


async def clear_cache() -> int:
    """
    Clear the entire cache.
    
    Returns:
        Number of invalidated entries
    """
    cache_manager = get_cache_manager()
    return await cache_manager.invalidate()