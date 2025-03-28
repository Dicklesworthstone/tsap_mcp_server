"""
TSAP Caching Utilities.

This module provides decorators and utilities for caching function results
to improve performance by avoiding redundant computations.
"""

import os
import time
import json
import hashlib
import functools
from typing import Dict, Any, Optional, Callable, Union, List, TypeVar, Awaitable
from dataclasses import dataclass, field
from threading import RLock

from tsap.utils.logging import logger
from tsap.config import get_config

# Type variables for function signatures
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])
AsyncF = TypeVar('AsyncF', bound=Callable[..., Awaitable[Any]])


@dataclass
class CacheEntry:
    """Represents a cached function result."""
    
    key: str
    value: Any
    timestamp: float
    ttl: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def age(self) -> float:
        """Get the age of the cache entry in seconds.
        
        Returns:
            Age in seconds
        """
        return time.time() - self.timestamp
    
    @property
    def is_expired(self) -> bool:
        """Check if the cache entry has expired.
        
        Returns:
            True if expired, False otherwise
        """
        if self.ttl is None:
            return False
        return self.age > self.ttl
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the cache entry to a dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "key": self.key,
            "value": self.value,
            "timestamp": self.timestamp,
            "ttl": self.ttl,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create a cache entry from a dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Cache entry
        """
        return cls(
            key=data["key"],
            value=data["value"],
            timestamp=data["timestamp"],
            ttl=data.get("ttl"),
            metadata=data.get("metadata", {}),
        )


class MemoryCache:
    """Simple in-memory cache with optional TTL."""
    
    def __init__(self, max_size: int = 1000, default_ttl: Optional[float] = None):
        """Initialize the memory cache.
        
        Args:
            max_size: Maximum number of entries to keep
            default_ttl: Default time-to-live in seconds (None for no expiration)
        """
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = RLock()
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self.misses += 1
                return None
            
            if entry.is_expired:
                self._cache.pop(key)
                self.misses += 1
                return None
            
            self.hits += 1
            return entry.value
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None for default TTL)
            metadata: Additional metadata to store with the value
        """
        with self._lock:
            # Check if we need to evict entries
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict()
            
            # Use provided TTL or default
            actual_ttl = ttl if ttl is not None else self.default_ttl
            
            # Create and store the entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                ttl=actual_ttl,
                metadata=metadata or {},
            )
            self._cache[key] = entry
    
    def delete(self, key: str) -> bool:
        """Delete an entry from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if the key was found and deleted, False otherwise
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> int:
        """Clear all entries from the cache.
        
        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count
    
    def _evict(self) -> None:
        """Evict entries to make room for new ones.
        
        This implements a simple strategy:
        1. Remove expired entries
        2. If still need space, remove oldest entries
        """
        with self._lock:
            # First, remove expired entries
            expired_keys = [
                key for key, entry in self._cache.items() if entry.is_expired
            ]
            for key in expired_keys:
                del self._cache[key]
            
            # If still need space, remove oldest entries
            if len(self._cache) >= self.max_size:
                # Sort by timestamp (oldest first)
                sorted_keys = sorted(
                    self._cache.keys(),
                    key=lambda k: self._cache[k].timestamp
                )
                
                # Remove oldest entries to get below max_size
                num_to_remove = len(self._cache) - self.max_size + 1
                for key in sorted_keys[:num_to_remove]:
                    del self._cache[key]
    
    def cleanup(self) -> int:
        """Remove expired entries from the cache.
        
        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items() if entry.is_expired
            ]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_ratio": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0,
            }
    
    def get_keys(self) -> List[str]:
        """Get all keys in the cache.
        
        Returns:
            List of cache keys
        """
        with self._lock:
            return list(self._cache.keys())
    
    def get_entries(self) -> Dict[str, CacheEntry]:
        """Get all entries in the cache.
        
        Returns:
            Dictionary mapping keys to cache entries
        """
        with self._lock:
            return self._cache.copy()


class DiskCache:
    """Cache that persists entries to disk."""
    
    def __init__(
        self,
        directory: str,
        max_size: int = 1000,
        default_ttl: Optional[float] = None,
    ):
        """Initialize the disk cache.
        
        Args:
            directory: Directory to store cache files
            max_size: Maximum number of entries to keep
            default_ttl: Default time-to-live in seconds
        """
        self.directory = os.path.expanduser(directory)
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.hits = 0
        self.misses = 0
        self._lock = RLock()
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.directory, exist_ok=True)
        
        # Initialize index
        self._index_path = os.path.join(self.directory, "index.json")
        self._index: Dict[str, Dict[str, Any]] = self._load_index()
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load the cache index from disk.
        
        Returns:
            Cache index
        """
        with self._lock:
            if not os.path.exists(self._index_path):
                return {}
            
            try:
                with open(self._index_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load cache index: {e}")
                return {}
    
    def _save_index(self) -> None:
        """Save the cache index to disk."""
        with self._lock:
            try:
                with open(self._index_path, 'w') as f:
                    json.dump(self._index, f)
            except IOError as e:
                logger.warning(f"Failed to save cache index: {e}")
    
    def _get_path(self, key: str) -> str:
        """Get the file path for a cache key.
        
        Args:
            key: Cache key
            
        Returns:
            Path to the cache file
        """
        # Use a hash of the key for the filename
        filename = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.directory, filename)
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        with self._lock:
            # Check if key exists in index
            if key not in self._index:
                self.misses += 1
                return None
            
            # Get entry metadata
            entry_meta = self._index[key]
            
            # Check if expired
            if entry_meta.get("ttl") is not None:
                age = time.time() - entry_meta["timestamp"]
                if age > entry_meta["ttl"]:
                    # Entry expired, remove it
                    self.delete(key)
                    self.misses += 1
                    return None
            
            # Load value from file
            path = self._get_path(key)
            if not os.path.exists(path):
                # File missing, remove from index
                del self._index[key]
                self._save_index()
                self.misses += 1
                return None
            
            try:
                with open(path, 'r') as f:
                    value = json.load(f)
                    self.hits += 1
                    return value
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load cache value for {key}: {e}")
                self.misses += 1
                return None
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
            metadata: Additional metadata to store with the value
        """
        with self._lock:
            # Check if we need to evict entries
            if len(self._index) >= self.max_size and key not in self._index:
                self._evict()
            
            # Use provided TTL or default
            actual_ttl = ttl if ttl is not None else self.default_ttl
            
            # Create entry metadata
            entry_meta = {
                "timestamp": time.time(),
                "ttl": actual_ttl,
                "metadata": metadata or {},
            }
            
            # Save to index
            self._index[key] = entry_meta
            self._save_index()
            
            # Save value to file
            path = self._get_path(key)
            try:
                with open(path, 'w') as f:
                    json.dump(value, f)
            except IOError as e:
                logger.warning(f"Failed to save cache value for {key}: {e}")
                # Remove from index if file save failed
                del self._index[key]
                self._save_index()
    
    def delete(self, key: str) -> bool:
        """Delete an entry from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if the key was found and deleted, False otherwise
        """
        with self._lock:
            if key not in self._index:
                return False
            
            # Remove from index
            del self._index[key]
            self._save_index()
            
            # Remove file if it exists
            path = self._get_path(key)
            if os.path.exists(path):
                try:
                    os.remove(path)
                except IOError as e:
                    logger.warning(f"Failed to delete cache file for {key}: {e}")
            
            return True
    
    def clear(self) -> int:
        """Clear all entries from the cache.
        
        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._index)
            
            # Clear index
            self._index = {}
            self._save_index()
            
            # Remove all cache files
            for filename in os.listdir(self.directory):
                if filename != "index.json":
                    try:
                        os.remove(os.path.join(self.directory, filename))
                    except IOError as e:
                        logger.warning(f"Failed to delete cache file {filename}: {e}")
            
            return count
    
    def _evict(self) -> None:
        """Evict entries to make room for new ones."""
        with self._lock:
            # First, remove expired entries
            expired_keys = []
            for key, entry_meta in self._index.items():
                if entry_meta.get("ttl") is not None:
                    age = time.time() - entry_meta["timestamp"]
                    if age > entry_meta["ttl"]:
                        expired_keys.append(key)
            
            for key in expired_keys:
                self.delete(key)
            
            # If still need space, remove oldest entries
            if len(self._index) >= self.max_size:
                # Sort by timestamp (oldest first)
                sorted_keys = sorted(
                    self._index.keys(),
                    key=lambda k: self._index[k]["timestamp"]
                )
                
                # Remove oldest entries to get below max_size
                num_to_remove = len(self._index) - self.max_size + 1
                for key in sorted_keys[:num_to_remove]:
                    self.delete(key)
    
    def cleanup(self) -> int:
        """Remove expired entries from the cache.
        
        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = []
            for key, entry_meta in self._index.items():
                if entry_meta.get("ttl") is not None:
                    age = time.time() - entry_meta["timestamp"]
                    if age > entry_meta["ttl"]:
                        expired_keys.append(key)
            
            for key in expired_keys:
                self.delete(key)
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            # Get disk usage
            total_size = 0
            for filename in os.listdir(self.directory):
                path = os.path.join(self.directory, filename)
                if os.path.isfile(path):
                    total_size += os.path.getsize(path)
            
            return {
                "size": len(self._index),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_ratio": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0,
                "disk_usage": total_size,
            }
    
    def get_keys(self) -> List[str]:
        """Get all keys in the cache.
        
        Returns:
            List of cache keys
        """
        with self._lock:
            return list(self._index.keys())


class CacheManager:
    """Manages multiple caches with different backends."""
    
    def __init__(self):
        """Initialize the cache manager."""
        self._caches: Dict[str, Union[MemoryCache, DiskCache]] = {}
        self._default_cache: Optional[str] = None
    
    def register_cache(
        self, 
        name: str, 
        cache: Union[MemoryCache, DiskCache],
        set_as_default: bool = False,
    ) -> None:
        """Register a cache with the manager.
        
        Args:
            name: Cache name
            cache: Cache instance
            set_as_default: Whether to set this as the default cache
        """
        self._caches[name] = cache
        
        if set_as_default or self._default_cache is None:
            self._default_cache = name
    
    def get_cache(self, name: Optional[str] = None) -> Union[MemoryCache, DiskCache]:
        """Get a cache by name.
        
        Args:
            name: Cache name (None for default)
            
        Returns:
            Cache instance
            
        Raises:
            KeyError: If the cache does not exist
        """
        cache_name = name or self._default_cache
        
        if cache_name is None:
            raise KeyError("No default cache set")
        
        if cache_name not in self._caches:
            raise KeyError(f"Cache '{cache_name}' not found")
        
        return self._caches[cache_name]
    
    def create_memory_cache(
        self, 
        name: str, 
        max_size: int = 1000,
        default_ttl: Optional[float] = None,
        set_as_default: bool = False,
    ) -> MemoryCache:
        """Create and register a memory cache.
        
        Args:
            name: Cache name
            max_size: Maximum number of entries
            default_ttl: Default time-to-live in seconds
            set_as_default: Whether to set as default
            
        Returns:
            The created cache
        """
        cache = MemoryCache(max_size=max_size, default_ttl=default_ttl)
        self.register_cache(name, cache, set_as_default)
        return cache
    
    def create_disk_cache(
        self, 
        name: str, 
        directory: str,
        max_size: int = 1000,
        default_ttl: Optional[float] = None,
        set_as_default: bool = False,
    ) -> DiskCache:
        """Create and register a disk cache.
        
        Args:
            name: Cache name
            directory: Directory to store cache files
            max_size: Maximum number of entries
            default_ttl: Default time-to-live in seconds
            set_as_default: Whether to set as default
            
        Returns:
            The created cache
        """
        cache = DiskCache(directory=directory, max_size=max_size, default_ttl=default_ttl)
        self.register_cache(name, cache, set_as_default)
        return cache
    
    def get(
        self, 
        key: str, 
        cache_name: Optional[str] = None,
    ) -> Optional[Any]:
        """Get a value from a cache.
        
        Args:
            key: Cache key
            cache_name: Cache name (None for default)
            
        Returns:
            Cached value or None if not found
        """
        cache = self.get_cache(cache_name)
        return cache.get(key)
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        cache_name: Optional[str] = None,
    ) -> None:
        """Set a value in a cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
            metadata: Additional metadata
            cache_name: Cache name (None for default)
        """
        cache = self.get_cache(cache_name)
        cache.set(key, value, ttl, metadata)
    
    def delete(self, key: str, cache_name: Optional[str] = None) -> bool:
        """Delete a value from a cache.
        
        Args:
            key: Cache key
            cache_name: Cache name (None for default)
            
        Returns:
            True if the key was found and deleted
        """
        cache = self.get_cache(cache_name)
        return cache.delete(key)
    
    def clear(self, cache_name: Optional[str] = None) -> int:
        """Clear a cache.
        
        Args:
            cache_name: Cache name (None for default)
            
        Returns:
            Number of entries cleared
        """
        cache = self.get_cache(cache_name)
        return cache.clear()
    
    def cleanup(self, cache_name: Optional[str] = None) -> int:
        """Remove expired entries from a cache.
        
        Args:
            cache_name: Cache name (None for default)
            
        Returns:
            Number of entries removed
        """
        cache = self.get_cache(cache_name)
        return cache.cleanup()
    
    def get_stats(self, cache_name: Optional[str] = None) -> Dict[str, Any]:
        """Get cache statistics.
        
        Args:
            cache_name: Cache name (None for default)
            
        Returns:
            Cache statistics
        """
        cache = self.get_cache(cache_name)
        return cache.get_stats()
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all caches.
        
        Returns:
            Dictionary mapping cache names to statistics
        """
        return {name: cache.get_stats() for name, cache in self._caches.items()}


# Global cache manager
_cache_manager = CacheManager()


def get_cache_manager() -> CacheManager:
    """Get the global cache manager.
    
    Returns:
        Global cache manager
    """
    return _cache_manager


def initialize_caches() -> None:
    """Initialize the default caches based on configuration."""
    manager = get_cache_manager()
    config = get_config()
    
    # Create memory cache
    memory_cache_size = config.cache.memory_cache_size
    memory_cache_ttl = config.cache.memory_cache_ttl
    manager.create_memory_cache(
        "memory",
        max_size=memory_cache_size,
        default_ttl=memory_cache_ttl,
        set_as_default=True,
    )
    
    # Create disk cache if enabled
    if config.cache.disk_cache_enabled:
        disk_cache_dir = config.cache.disk_cache_dir
        disk_cache_size = config.cache.disk_cache_size
        disk_cache_ttl = config.cache.disk_cache_ttl
        
        manager.create_disk_cache(
            "disk",
            directory=disk_cache_dir,
            max_size=disk_cache_size,
            default_ttl=disk_cache_ttl,
        )


# Cache key generation

def generate_cache_key(
    func: Optional[Callable] = None,
    args: Optional[tuple] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    namespace: Optional[str] = None,
    include_args: bool = True,
) -> str:
    """Generate a cache key for a function call.
    
    Args:
        func: Function being called
        args: Positional arguments
        kwargs: Keyword arguments
        namespace: Optional namespace for the key
        include_args: Whether to include arguments in the key
        
    Returns:
        Cache key
    """
    # Start with namespace if provided, otherwise use function name
    if namespace:
        key_parts = [namespace]
    elif func:
        key_parts = [func.__module__ + "." + func.__name__]
    else:
        key_parts = ["anonymous"]
    
    # Add arguments if requested
    if include_args and (args or kwargs):
        # Get a stable representation of the arguments
        arg_parts = []
        
        if args:
            arg_parts.extend(str(arg) for arg in args)
        
        if kwargs:
            # Sort kwargs for stable ordering
            arg_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        
        # Join argument parts and hash them
        arg_str = ",".join(arg_parts)
        arg_hash = hashlib.md5(arg_str.encode()).hexdigest()
        key_parts.append(arg_hash)
    
    # Join key parts
    return ":".join(key_parts)


# Decorators

def cached(
    ttl: Optional[float] = None,
    cache_name: Optional[str] = None,
    key_func: Optional[Callable[..., str]] = None,
) -> Callable[[F], F]:
    """Decorator to cache function results.
    
    Args:
        ttl: Time-to-live in seconds
        cache_name: Cache to use
        key_func: Function to generate cache keys
        
    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get the cache
            manager = get_cache_manager()
            
            # Generate cache key
            if key_func:
                key = key_func(func, *args, **kwargs)
            else:
                key = generate_cache_key(func, args, kwargs)
            
            # Try to get from cache
            cached_value = manager.get(key, cache_name)
            if cached_value is not None:
                return cached_value
            
            # Call the function
            result = func(*args, **kwargs)
            
            # Cache the result
            manager.set(key, result, ttl, cache_name=cache_name)
            
            return result
        
        return wrapper  # type: ignore
    
    return decorator


def async_cached(
    ttl: Optional[float] = None,
    cache_name: Optional[str] = None,
    key_func: Optional[Callable[..., str]] = None,
) -> Callable[[AsyncF], AsyncF]:
    """Decorator to cache async function results.
    
    Args:
        ttl: Time-to-live in seconds
        cache_name: Cache to use
        key_func: Function to generate cache keys
        
    Returns:
        Decorator function
    """
    def decorator(func: AsyncF) -> AsyncF:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get the cache
            manager = get_cache_manager()
            
            # Generate cache key
            if key_func:
                key = key_func(func, *args, **kwargs)
            else:
                key = generate_cache_key(func, args, kwargs)
            
            # Try to get from cache
            cached_value = manager.get(key, cache_name)
            if cached_value is not None:
                return cached_value
            
            # Call the function
            result = await func(*args, **kwargs)
            
            # Cache the result
            manager.set(key, result, ttl, cache_name=cache_name)
            
            return result
        
        return wrapper  # type: ignore
    
    return decorator


# Helper functions

def invalidate_cache(
    func: Optional[Callable] = None,
    args: Optional[tuple] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    namespace: Optional[str] = None,
    cache_name: Optional[str] = None,
) -> bool:
    """Invalidate a cached function result.
    
    Args:
        func: Function to invalidate
        args: Positional arguments
        kwargs: Keyword arguments
        namespace: Optional namespace
        cache_name: Cache name
        
    Returns:
        True if the cache entry was found and deleted
    """
    # Generate the cache key
    key = generate_cache_key(func, args, kwargs, namespace)
    
    # Delete from cache
    manager = get_cache_manager()
    return manager.delete(key, cache_name)


def cache_context(namespace: str, ttl: Optional[float] = None) -> 'CacheContext':
    """Create a context manager for manually caching values.
    
    Args:
        namespace: Namespace for cache keys
        ttl: Time-to-live in seconds
        
    Returns:
        Cache context manager
    """
    return CacheContext(namespace, ttl)


class CacheContext:
    """Context manager for manually caching values."""
    
    def __init__(self, namespace: str, ttl: Optional[float] = None):
        """Initialize the cache context.
        
        Args:
            namespace: Namespace for cache keys
            ttl: Time-to-live in seconds
        """
        self.namespace = namespace
        self.ttl = ttl
    
    def get(self, key: str, cache_name: Optional[str] = None) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: Cache key (will be prefixed with namespace)
            cache_name: Cache name
            
        Returns:
            Cached value or None if not found
        """
        full_key = f"{self.namespace}:{key}"
        manager = get_cache_manager()
        return manager.get(full_key, cache_name)
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        cache_name: Optional[str] = None,
    ) -> None:
        """Set a value in the cache.
        
        Args:
            key: Cache key (will be prefixed with namespace)
            value: Value to cache
            ttl: Time-to-live in seconds (overrides context TTL)
            metadata: Additional metadata
            cache_name: Cache name
        """
        full_key = f"{self.namespace}:{key}"
        manager = get_cache_manager()
        
        # Use provided TTL or context TTL
        actual_ttl = ttl if ttl is not None else self.ttl
        
        manager.set(full_key, value, actual_ttl, metadata, cache_name)
    
    def delete(self, key: str, cache_name: Optional[str] = None) -> bool:
        """Delete a value from the cache.
        
        Args:
            key: Cache key (will be prefixed with namespace)
            cache_name: Cache name
            
        Returns:
            True if the key was found and deleted
        """
        full_key = f"{self.namespace}:{key}"
        manager = get_cache_manager()
        return manager.delete(full_key, cache_name)
    
    def clear_namespace(self, cache_name: Optional[str] = None) -> int:
        """Clear all keys in this namespace.
        
        Args:
            cache_name: Cache name
            
        Returns:
            Number of entries cleared
        """
        # This is a best-effort implementation since not all caches
        # support prefix-based clearing. It fetches all keys and
        # deletes those with the matching namespace.
        manager = get_cache_manager()
        cache = manager.get_cache(cache_name)
        
        count = 0
        for key in cache.get_keys():
            if key.startswith(f"{self.namespace}:"):
                if cache.delete(key):
                    count += 1
        
        return count