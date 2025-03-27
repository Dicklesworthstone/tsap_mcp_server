"""
Cache invalidation strategies for the TSAP MCP Server.

This module provides various cache invalidation strategies and policies
to ensure cache freshness while maximizing cache hit rates.
"""

import time
import os
import hashlib
from typing import Dict, List, Any, Optional, Callable
from abc import ABC, abstractmethod

from tsap.utils.logging import logger


class InvalidationStrategy(ABC):
    """
    Base class for cache invalidation strategies.
    
    An invalidation strategy determines which cache entries should be
    removed when the cache is full or entries are stale.
    """
    
    @abstractmethod
    def should_invalidate(self, key: str, metadata: Dict[str, Any]) -> bool:
        """
        Determine if a cache entry should be invalidated.
        
        Args:
            key: Cache entry key
            metadata: Cache entry metadata
            
        Returns:
            True if the entry should be invalidated, False otherwise
        """
        pass
    
    @abstractmethod
    def get_entries_to_invalidate(
        self, 
        entries: Dict[str, Dict[str, Any]],
        needed_space: Optional[int] = None,
        max_entries: Optional[int] = None
    ) -> List[str]:
        """
        Get a list of cache entries to invalidate.
        
        Args:
            entries: Dictionary of cache entries (key -> metadata)
            needed_space: Amount of space needed (in bytes)
            max_entries: Maximum number of entries to invalidate
            
        Returns:
            List of cache keys to invalidate
        """
        pass


class TTLInvalidationStrategy(InvalidationStrategy):
    """
    Time-to-Live (TTL) invalidation strategy.
    
    Invalidates cache entries that have been in the cache longer than
    their time-to-live (TTL) value.
    """
    
    def __init__(self, default_ttl: int = 3600):
        """
        Initialize the TTL invalidation strategy.
        
        Args:
            default_ttl: Default time-to-live in seconds
        """
        self.default_ttl = default_ttl
    
    def should_invalidate(self, key: str, metadata: Dict[str, Any]) -> bool:
        """
        Determine if a cache entry should be invalidated based on TTL.
        
        Args:
            key: Cache entry key
            metadata: Cache entry metadata
            
        Returns:
            True if the entry has expired, False otherwise
        """
        # Get creation time or use current time if not available
        creation_time = metadata.get("creation_time", time.time())
        
        # Get TTL from metadata or use default
        ttl = metadata.get("ttl", self.default_ttl)
        
        # Check if expired
        return (time.time() - creation_time) > ttl
    
    def get_entries_to_invalidate(
        self, 
        entries: Dict[str, Dict[str, Any]],
        needed_space: Optional[int] = None,
        max_entries: Optional[int] = None
    ) -> List[str]:
        """
        Get a list of expired cache entries to invalidate.
        
        Args:
            entries: Dictionary of cache entries (key -> metadata)
            needed_space: Ignored (not used by TTL strategy)
            max_entries: Ignored (not used by TTL strategy)
            
        Returns:
            List of expired cache keys to invalidate
        """
        expired_keys = []
        
        for key, metadata in entries.items():
            if self.should_invalidate(key, metadata):
                expired_keys.append(key)
        
        return expired_keys


class LRUInvalidationStrategy(InvalidationStrategy):
    """
    Least Recently Used (LRU) invalidation strategy.
    
    Invalidates cache entries that have been used least recently
    when space is needed.
    """
    
    def should_invalidate(self, key: str, metadata: Dict[str, Any]) -> bool:
        """
        LRU doesn't invalidate individual entries based on metadata.
        
        Args:
            key: Cache entry key
            metadata: Cache entry metadata
            
        Returns:
            Always False (LRU invalidates based on access order)
        """
        return False
    
    def get_entries_to_invalidate(
        self, 
        entries: Dict[str, Dict[str, Any]],
        needed_space: Optional[int] = None,
        max_entries: Optional[int] = None
    ) -> List[str]:
        """
        Get a list of least recently used cache entries to invalidate.
        
        Args:
            entries: Dictionary of cache entries (key -> metadata)
            needed_space: Amount of space needed (in bytes)
            max_entries: Maximum number of entries to invalidate
            
        Returns:
            List of cache keys to invalidate
        """
        # Sort entries by last access time (oldest first)
        sorted_entries = sorted(
            entries.items(),
            key=lambda x: x[1].get("last_access", 0)
        )
        
        invalidate_keys = []
        freed_space = 0
        
        for key, metadata in sorted_entries:
            # Stop if we've freed enough space
            if needed_space is not None and freed_space >= needed_space:
                break
            
            # Stop if we've invalidated enough entries
            if max_entries is not None and len(invalidate_keys) >= max_entries:
                break
            
            # Add key to invalidation list
            invalidate_keys.append(key)
            
            # Track freed space
            freed_space += metadata.get("size", 0)
        
        return invalidate_keys


class LFUInvalidationStrategy(InvalidationStrategy):
    """
    Least Frequently Used (LFU) invalidation strategy.
    
    Invalidates cache entries that have been used least frequently
    when space is needed.
    """
    
    def should_invalidate(self, key: str, metadata: Dict[str, Any]) -> bool:
        """
        LFU doesn't invalidate individual entries based on metadata.
        
        Args:
            key: Cache entry key
            metadata: Cache entry metadata
            
        Returns:
            Always False (LFU invalidates based on access frequency)
        """
        return False
    
    def get_entries_to_invalidate(
        self, 
        entries: Dict[str, Dict[str, Any]],
        needed_space: Optional[int] = None,
        max_entries: Optional[int] = None
    ) -> List[str]:
        """
        Get a list of least frequently used cache entries to invalidate.
        
        Args:
            entries: Dictionary of cache entries (key -> metadata)
            needed_space: Amount of space needed (in bytes)
            max_entries: Maximum number of entries to invalidate
            
        Returns:
            List of cache keys to invalidate
        """
        # Sort entries by access count (lowest first)
        sorted_entries = sorted(
            entries.items(),
            key=lambda x: x[1].get("access_count", 0)
        )
        
        invalidate_keys = []
        freed_space = 0
        
        for key, metadata in sorted_entries:
            # Stop if we've freed enough space
            if needed_space is not None and freed_space >= needed_space:
                break
            
            # Stop if we've invalidated enough entries
            if max_entries is not None and len(invalidate_keys) >= max_entries:
                break
            
            # Add key to invalidation list
            invalidate_keys.append(key)
            
            # Track freed space
            freed_space += metadata.get("size", 0)
        
        return invalidate_keys


class FIFOInvalidationStrategy(InvalidationStrategy):
    """
    First In, First Out (FIFO) invalidation strategy.
    
    Invalidates oldest cache entries first, regardless of access patterns.
    """
    
    def should_invalidate(self, key: str, metadata: Dict[str, Any]) -> bool:
        """
        FIFO doesn't invalidate individual entries based on metadata.
        
        Args:
            key: Cache entry key
            metadata: Cache entry metadata
            
        Returns:
            Always False (FIFO invalidates based on creation order)
        """
        return False
    
    def get_entries_to_invalidate(
        self, 
        entries: Dict[str, Dict[str, Any]],
        needed_space: Optional[int] = None,
        max_entries: Optional[int] = None
    ) -> List[str]:
        """
        Get a list of oldest cache entries to invalidate.
        
        Args:
            entries: Dictionary of cache entries (key -> metadata)
            needed_space: Amount of space needed (in bytes)
            max_entries: Maximum number of entries to invalidate
            
        Returns:
            List of cache keys to invalidate
        """
        # Sort entries by creation time (oldest first)
        sorted_entries = sorted(
            entries.items(),
            key=lambda x: x[1].get("creation_time", 0)
        )
        
        invalidate_keys = []
        freed_space = 0
        
        for key, metadata in sorted_entries:
            # Stop if we've freed enough space
            if needed_space is not None and freed_space >= needed_space:
                break
            
            # Stop if we've invalidated enough entries
            if max_entries is not None and len(invalidate_keys) >= max_entries:
                break
            
            # Add key to invalidation list
            invalidate_keys.append(key)
            
            # Track freed space
            freed_space += metadata.get("size", 0)
        
        return invalidate_keys


class HybridInvalidationStrategy(InvalidationStrategy):
    """
    Hybrid invalidation strategy.
    
    Combines multiple strategies for better performance. First applies TTL
    to remove expired entries, then applies LRU or another strategy for
    remaining entries if more space is needed.
    """
    
    def __init__(
        self,
        ttl_strategy: Optional[TTLInvalidationStrategy] = None,
        backup_strategy: Optional[InvalidationStrategy] = None
    ):
        """
        Initialize the hybrid invalidation strategy.
        
        Args:
            ttl_strategy: TTL strategy to use first
            backup_strategy: Backup strategy to use if more space is needed
        """
        self.ttl_strategy = ttl_strategy or TTLInvalidationStrategy()
        self.backup_strategy = backup_strategy or LRUInvalidationStrategy()
    
    def should_invalidate(self, key: str, metadata: Dict[str, Any]) -> bool:
        """
        Determine if a cache entry should be invalidated.
        
        Args:
            key: Cache entry key
            metadata: Cache entry metadata
            
        Returns:
            True if the entry should be invalidated, False otherwise
        """
        # First check TTL
        if self.ttl_strategy.should_invalidate(key, metadata):
            return True
        
        # Then check backup strategy
        return self.backup_strategy.should_invalidate(key, metadata)
    
    def get_entries_to_invalidate(
        self, 
        entries: Dict[str, Dict[str, Any]],
        needed_space: Optional[int] = None,
        max_entries: Optional[int] = None
    ) -> List[str]:
        """
        Get a list of cache entries to invalidate.
        
        Args:
            entries: Dictionary of cache entries (key -> metadata)
            needed_space: Amount of space needed (in bytes)
            max_entries: Maximum number of entries to invalidate
            
        Returns:
            List of cache keys to invalidate
        """
        # First get expired entries
        expired_keys = self.ttl_strategy.get_entries_to_invalidate(entries)
        
        # Calculate freed space from expired entries
        freed_space = sum(
            entries[key].get("size", 0) for key in expired_keys
            if key in entries
        )
        
        # If we've freed enough space, or hit max entries, we're done
        if (needed_space is not None and freed_space >= needed_space) or \
           (max_entries is not None and len(expired_keys) >= max_entries):
            return expired_keys[:max_entries] if max_entries is not None else expired_keys
        
        # Otherwise, get more entries from backup strategy
        # Create a new dictionary without the expired entries
        remaining_entries = {
            key: metadata for key, metadata in entries.items()
            if key not in expired_keys
        }
        
        # Calculate remaining needed space and entries
        remaining_space = None
        if needed_space is not None:
            remaining_space = max(0, needed_space - freed_space)
        
        remaining_max_entries = None
        if max_entries is not None:
            remaining_max_entries = max(0, max_entries - len(expired_keys))
        
        # Get additional entries from backup strategy
        additional_keys = self.backup_strategy.get_entries_to_invalidate(
            remaining_entries,
            remaining_space,
            remaining_max_entries
        )
        
        # Combine expired and additional keys
        return expired_keys + additional_keys


class ContentAwareInvalidationStrategy(InvalidationStrategy):
    """
    Content-aware invalidation strategy.
    
    Invalidates cache entries based on content dependencies and changes.
    Particularly useful for code analysis results that depend on source files.
    """
    
    def __init__(self, file_checksum_func: Optional[Callable[[str], str]] = None):
        """
        Initialize the content-aware invalidation strategy.
        
        Args:
            file_checksum_func: Function to compute file checksums
        """
        self.file_checksum_func = file_checksum_func or self._default_file_checksum
    
    def _default_file_checksum(self, path: str) -> str:
        """
        Compute a checksum for a file.
        
        Args:
            path: Path to the file
            
        Returns:
            File checksum as a string
        """
        if not os.path.exists(path):
            return ""
        
        try:
            with open(path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.warning(
                f"Error computing checksum for {path}: {str(e)}",
                component="cache"
            )
            return ""
    
    def should_invalidate(self, key: str, metadata: Dict[str, Any]) -> bool:
        """
        Determine if a cache entry should be invalidated based on content changes.
        
        Args:
            key: Cache entry key
            metadata: Cache entry metadata
            
        Returns:
            True if the entry should be invalidated, False otherwise
        """
        # Get file dependencies
        dependencies = metadata.get("dependencies", [])
        
        # If no dependencies, don't invalidate
        if not dependencies:
            return False
        
        # Check if any dependencies have changed
        for dep in dependencies:
            # Skip if file doesn't exist
            if not os.path.exists(dep["path"]):
                return True
            
            # Get current checksum
            current_checksum = self.file_checksum_func(dep["path"])
            
            # Check if checksum has changed
            if current_checksum != dep["checksum"]:
                return True
        
        return False
    
    def get_entries_to_invalidate(
        self, 
        entries: Dict[str, Dict[str, Any]],
        needed_space: Optional[int] = None,
        max_entries: Optional[int] = None
    ) -> List[str]:
        """
        Get a list of cache entries to invalidate based on content changes.
        
        Args:
            entries: Dictionary of cache entries (key -> metadata)
            needed_space: Amount of space needed (in bytes)
            max_entries: Maximum number of entries to invalidate
            
        Returns:
            List of cache keys to invalidate
        """
        invalidate_keys = []
        
        for key, metadata in entries.items():
            # Check if entry should be invalidated
            if self.should_invalidate(key, metadata):
                invalidate_keys.append(key)
                
                # Check if we've hit limits
                if max_entries is not None and len(invalidate_keys) >= max_entries:
                    break
        
        # If we need to free specific amount of space and haven't freed enough,
        # fall back to LRU for additional entries
        if needed_space is not None:
            freed_space = sum(
                entries[key].get("size", 0) for key in invalidate_keys
                if key in entries
            )
            
            if freed_space < needed_space:
                # Create dict without already invalidated entries
                remaining_entries = {
                    key: metadata for key, metadata in entries.items()
                    if key not in invalidate_keys
                }
                
                # Use LRU to get more entries
                lru_strategy = LRUInvalidationStrategy()
                remaining_space = needed_space - freed_space
                
                remaining_max_entries = None
                if max_entries is not None:
                    remaining_max_entries = max_entries - len(invalidate_keys)
                    if remaining_max_entries <= 0:
                        return invalidate_keys
                
                additional_keys = lru_strategy.get_entries_to_invalidate(
                    remaining_entries,
                    remaining_space,
                    remaining_max_entries
                )
                
                invalidate_keys.extend(additional_keys)
        
        return invalidate_keys


class InvalidationPolicyManager:
    """
    Manages cache invalidation policies.
    
    This class provides a unified interface for cache invalidation based on
    different strategies and policies.
    """
    
    def __init__(self, default_strategy: str = "hybrid"):
        """
        Initialize the invalidation policy manager.
        
        Args:
            default_strategy: Default invalidation strategy name
        """
        self.strategies = {
            "ttl": TTLInvalidationStrategy(),
            "lru": LRUInvalidationStrategy(),
            "lfu": LFUInvalidationStrategy(),
            "fifo": FIFOInvalidationStrategy(),
            "hybrid": HybridInvalidationStrategy(),
            "content": ContentAwareInvalidationStrategy()
        }
        
        self.default_strategy = default_strategy
    
    def get_strategy(self, strategy_name: Optional[str] = None) -> InvalidationStrategy:
        """
        Get an invalidation strategy by name.
        
        Args:
            strategy_name: Strategy name or None for default
            
        Returns:
            Invalidation strategy
        """
        if strategy_name is None:
            strategy_name = self.default_strategy
        
        if strategy_name not in self.strategies:
            logger.warning(
                f"Unknown invalidation strategy: {strategy_name}, using default",
                component="cache"
            )
            strategy_name = self.default_strategy
        
        return self.strategies[strategy_name]
    
    def invalidate_entries(
        self,
        entries: Dict[str, Dict[str, Any]],
        strategy_name: Optional[str] = None,
        needed_space: Optional[int] = None,
        max_entries: Optional[int] = None
    ) -> List[str]:
        """
        Get a list of cache entries to invalidate.
        
        Args:
            entries: Dictionary of cache entries (key -> metadata)
            strategy_name: Strategy name or None for default
            needed_space: Amount of space needed (in bytes)
            max_entries: Maximum number of entries to invalidate
            
        Returns:
            List of cache keys to invalidate
        """
        strategy = self.get_strategy(strategy_name)
        return strategy.get_entries_to_invalidate(entries, needed_space, max_entries)
    
    def should_invalidate(
        self,
        key: str,
        metadata: Dict[str, Any],
        strategy_name: Optional[str] = None
    ) -> bool:
        """
        Determine if a cache entry should be invalidated.
        
        Args:
            key: Cache entry key
            metadata: Cache entry metadata
            strategy_name: Strategy name or None for default
            
        Returns:
            True if the entry should be invalidated, False otherwise
        """
        strategy = self.get_strategy(strategy_name)
        return strategy.should_invalidate(key, metadata)


# Global instance for convenience
_invalidation_manager: Optional[InvalidationPolicyManager] = None


def get_invalidation_manager() -> InvalidationPolicyManager:
    """
    Get the global invalidation policy manager instance.
    
    Returns:
        InvalidationPolicyManager instance
    """
    global _invalidation_manager
    
    if _invalidation_manager is None:
        _invalidation_manager = InvalidationPolicyManager()
    
    return _invalidation_manager


def get_entries_to_invalidate(
    entries: Dict[str, Dict[str, Any]],
    strategy_name: Optional[str] = None,
    needed_space: Optional[int] = None,
    max_entries: Optional[int] = None
) -> List[str]:
    """
    Convenience function to get a list of cache entries to invalidate.
    
    Args:
        entries: Dictionary of cache entries (key -> metadata)
        strategy_name: Strategy name or None for default
        needed_space: Amount of space needed (in bytes)
        max_entries: Maximum number of entries to invalidate
        
    Returns:
        List of cache keys to invalidate
    """
    manager = get_invalidation_manager()
    return manager.invalidate_entries(entries, strategy_name, needed_space, max_entries)


def should_invalidate(
    key: str,
    metadata: Dict[str, Any],
    strategy_name: Optional[str] = None
) -> bool:
    """
    Convenience function to determine if a cache entry should be invalidated.
    
    Args:
        key: Cache entry key
        metadata: Cache entry metadata
        strategy_name: Strategy name or None for default
        
    Returns:
        True if the entry should be invalidated, False otherwise
    """
    manager = get_invalidation_manager()
    return manager.should_invalidate(key, metadata, strategy_name)