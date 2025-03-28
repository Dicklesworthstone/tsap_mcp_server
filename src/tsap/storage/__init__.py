"""
Storage package marker for TSAP MCP Server.

This package provides functionality for persistent storage of various data types
like commands, patterns, and profiles, primarily using SQLite databases.
"""

from tsap.storage.database import (
    Database, DatabaseError, DatabaseRegistry,
    get_registry, get_database, create_database,
    get_system_database, close_all_databases,
    with_database, with_transaction, with_system_db, with_system_transaction,
    get_setting, set_setting, delete_setting, list_settings
)

from tsap.storage.history_store import (
    HistoryStore, HistoryStoreError,
    get_history_store, get_project_history_store,
    add_command_to_history, update_command_in_history,
    get_command_from_history, search_command_history,
    get_recent_commands_from_history
)

from tsap.storage.pattern_store import (
    PatternStore, PatternStoreError,
    get_pattern_store, add_pattern, search_patterns,
    get_pattern, get_pattern_by_name, get_popular_patterns,
    increment_pattern_usage, add_pattern_example, add_pattern_stats
)

from tsap.storage.profile_store import (
    ProfileStore, ProfileStoreError,
    get_profile_store, add_profile, get_profile,
    get_default_profile, set_profile_setting,
    get_profile_setting, add_document_profile,
    get_document_profile, find_similar_documents
)

from tsap.storage.strategy_store import (
    StrategyStore, StrategyStoreError,
    get_strategy_store, add_strategy, get_strategy,
    get_strategy_by_name, search_strategies,
    record_strategy_execution, get_popular_strategies,
    find_similar_strategies
)


__all__ = [
    # Database components
    'Database', 'DatabaseError', 'DatabaseRegistry',
    'get_registry', 'get_database', 'create_database',
    'get_system_database', 'close_all_databases',
    'with_database', 'with_transaction', 'with_system_db', 'with_system_transaction',
    'get_setting', 'set_setting', 'delete_setting', 'list_settings',
    
    # History store components
    'HistoryStore', 'HistoryStoreError',
    'get_history_store', 'get_project_history_store',
    'add_command_to_history', 'update_command_in_history',
    'get_command_from_history', 'search_command_history',
    'get_recent_commands_from_history',
    
    # Pattern store components
    'PatternStore', 'PatternStoreError',
    'get_pattern_store', 'add_pattern', 'search_patterns',
    'get_pattern', 'get_pattern_by_name', 'get_popular_patterns',
    'increment_pattern_usage', 'add_pattern_example', 'add_pattern_stats',
    
    # Profile store components
    'ProfileStore', 'ProfileStoreError',
    'get_profile_store', 'add_profile', 'get_profile',
    'get_default_profile', 'set_profile_setting',
    'get_profile_setting', 'add_document_profile',
    'get_document_profile', 'find_similar_documents',
    
    # Strategy store components
    'StrategyStore', 'StrategyStoreError',
    'get_strategy_store', 'add_strategy', 'get_strategy',
    'get_strategy_by_name', 'search_strategies',
    'record_strategy_execution', 'get_popular_strategies',
    'find_similar_strategies'
]