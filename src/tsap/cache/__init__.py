"""
Cache package for the TSAP MCP Server.

This package provides caching functionality for operations
to improve performance by avoiding redundant computations.
"""

from tsap.cache.manager import (
    CacheManager,
    get_cache_manager,
    cache_result,
    get_cached_result,
    invalidate_cache_entry,
    clear_cache
)

__all__ = [
    'CacheManager',
    'get_cache_manager',
    'cache_result',
    'get_cached_result',
    'invalidate_cache_entry',
    'clear_cache'
]