"""
Project package marker for TSAP MCP Server.

This package provides functionality for managing projects, which represent
self-contained sets of files, analyses, and operations with their own
configuration and historical data.
"""

from tsap.project.context import (
    ProjectContext, ProjectContextError, ProjectRegistry,
    get_project_registry, create_project, get_project,
    set_active_project, project_context, initialize_project_system
)

from tsap.project.history import (
    CommandEntry, CommandHistory, HistoryError,
    get_command_history, record_command, update_command_status,
    get_command, get_recent_commands, command_context
)

from tsap.project.profile import (
    ProjectProfile, ProfileManager, ProfileError,
    get_profile_manager, get_active_profile, create_profile,
    set_active_profile, export_profile, import_profile
)

from tsap.project.transfer import (
    ProjectExport, ProjectImport, TransferError,
    export_project, import_project, create_project_backup,
    restore_project_from_backup, transfer_results
)


__all__ = [
    # Context components
    'ProjectContext', 'ProjectContextError', 'ProjectRegistry',
    'get_project_registry', 'create_project', 'get_project',
    'set_active_project', 'project_context', 'initialize_project_system',
    
    # History components
    'CommandEntry', 'CommandHistory', 'HistoryError',
    'get_command_history', 'record_command', 'update_command_status',
    'get_command', 'get_recent_commands', 'command_context',
    
    # Profile components
    'ProjectProfile', 'ProfileManager', 'ProfileError',
    'get_profile_manager', 'get_active_profile', 'create_profile',
    'set_active_profile', 'export_profile', 'import_profile',
    
    # Transfer components
    'ProjectExport', 'ProjectImport', 'TransferError',
    'export_project', 'import_project', 'create_project_backup',
    'restore_project_from_backup', 'transfer_results'
]