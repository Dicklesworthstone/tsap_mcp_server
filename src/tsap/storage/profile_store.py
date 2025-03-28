"""
Profile storage for TSAP.

This module provides persistent storage for profiles, including both
project configuration profiles and document profiles.
"""

import os
import time
import uuid
import threading
from typing import Dict, List, Any, Optional, Tuple

from tsap.utils.errors import TSAPError
from tsap.storage.database import Database, get_database, create_database


class ProfileStoreError(TSAPError):
    """Exception raised for profile storage errors."""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.details = details or {}


class ProfileStore:
    """
    Persistent storage for profiles.
    
    This class provides methods for saving, retrieving, and querying
    profiles in a SQLite database.
    """
    
    def __init__(self, db: Database):
        """
        Initialize a profile store.
        
        Args:
            db: Database instance
        """
        self.db = db
        self._lock = threading.RLock()
        self._initialize_schema()
    
    def _initialize_schema(self) -> None:
        """Initialize the database schema for profile storage."""
        with self._lock:
            with self.db.transaction():
                # Create profiles table
                if not self.db.table_exists('profiles'):
                    self.db.create_table(
                        'profiles',
                        {
                            'profile_id': 'TEXT NOT NULL',
                            'project_id': 'TEXT',  # Nullable for global profiles
                            'name': 'TEXT NOT NULL',
                            'description': 'TEXT',
                            'type': 'TEXT NOT NULL',  # 'project', 'document', etc.
                            'is_default': 'INTEGER NOT NULL DEFAULT 0',
                            'is_system': 'INTEGER NOT NULL DEFAULT 0',
                            'created_at': 'REAL NOT NULL',
                            'updated_at': 'REAL NOT NULL',
                            'last_used_at': 'REAL'
                        },
                        primary_key='profile_id'
                    )
                    
                    # Create indices
                    self.db.create_index('idx_profiles_project_id', 'profiles', 'project_id')
                    self.db.create_index('idx_profiles_type', 'profiles', 'type')
                    self.db.create_index('idx_profiles_name', 'profiles', 'name')
                
                # Create profile settings table
                if not self.db.table_exists('profile_settings'):
                    self.db.create_table(
                        'profile_settings',
                        {
                            'profile_id': 'TEXT NOT NULL',
                            'key': 'TEXT NOT NULL',
                            'value': 'TEXT',
                            'updated_at': 'REAL NOT NULL'
                        },
                        primary_key=['profile_id', 'key']
                    )
                    
                    # Create index on profile_id for faster lookups
                    self.db.create_index('idx_ps_profile_id', 'profile_settings', 'profile_id')
                
                # Create tool configurations table
                if not self.db.table_exists('tool_configs'):
                    self.db.create_table(
                        'tool_configs',
                        {
                            'profile_id': 'TEXT NOT NULL',
                            'tool_name': 'TEXT NOT NULL',
                            'config': 'TEXT NOT NULL',
                            'updated_at': 'REAL NOT NULL'
                        },
                        primary_key=['profile_id', 'tool_name']
                    )
                    
                    # Create index on profile_id for faster lookups
                    self.db.create_index('idx_tc_profile_id', 'tool_configs', 'profile_id')
                
                # Create document profiles table
                if not self.db.table_exists('document_profiles'):
                    self.db.create_table(
                        'document_profiles',
                        {
                            'profile_id': 'TEXT NOT NULL',
                            'document_path': 'TEXT NOT NULL',
                            'content_hash': 'TEXT',
                            'basic_properties': 'TEXT NOT NULL',
                            'content_metrics': 'TEXT',
                            'language_info': 'TEXT',
                            'content_features': 'TEXT',
                            'structure_info': 'TEXT',
                            'type_specific_features': 'TEXT',
                            'created_at': 'REAL NOT NULL',
                            'updated_at': 'REAL NOT NULL'
                        },
                        primary_key='profile_id'
                    )
                    
                    # Create index on document_path
                    self.db.create_index('idx_dp_document_path', 'document_profiles', 'document_path')
                    self.db.create_index('idx_dp_content_hash', 'document_profiles', 'content_hash')
    
    def add_profile(self, name: str, profile_type: str, project_id: Optional[str] = None,
                  description: Optional[str] = None, is_default: bool = False,
                  is_system: bool = False, profile_id: Optional[str] = None) -> str:
        """
        Add a profile to the store.
        
        Args:
            name: Profile name
            profile_type: Profile type ('project', 'document', etc.)
            project_id: Optional project ID (None for global profiles)
            description: Optional description
            is_default: Whether this is the default profile
            is_system: Whether this is a system profile
            profile_id: Optional profile ID (generated if None)
            
        Returns:
            ID of the added profile
        """
        with self._lock:
            with self.db.transaction():
                # Generate profile ID if not provided
                if profile_id is None:
                    profile_id = str(uuid.uuid4())
                
                # Current time
                current_time = time.time()
                
                # If setting as default, clear existing default
                if is_default and project_id is not None:
                    self.db.update(
                        'profiles',
                        {'is_default': 0},
                        'project_id = ? AND type = ?',
                        (project_id, profile_type)
                    )
                
                # Insert profile
                self.db.insert(
                    'profiles',
                    {
                        'profile_id': profile_id,
                        'project_id': project_id,
                        'name': name,
                        'description': description,
                        'type': profile_type,
                        'is_default': 1 if is_default else 0,
                        'is_system': 1 if is_system else 0,
                        'created_at': current_time,
                        'updated_at': current_time,
                        'last_used_at': current_time if is_default else None
                    }
                )
                
                return profile_id
    
    def get_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a profile from the store.
        
        Args:
            profile_id: ID of the profile
            
        Returns:
            Profile data or None if not found
        """
        with self._lock:
            # Get profile data
            profile = self.db.query_one(
                'SELECT * FROM profiles WHERE profile_id = ?',
                (profile_id,)
            )
            
            if not profile:
                return None
            
            # Convert boolean fields
            profile['is_default'] = bool(profile['is_default'])
            profile['is_system'] = bool(profile['is_system'])
            
            # Get settings
            settings = self.db.query(
                'SELECT key, value FROM profile_settings WHERE profile_id = ?',
                (profile_id,)
            )
            
            profile_settings = {}
            for setting in settings:
                profile_settings[setting['key']] = self.db.json_deserialize(setting['value'])
            
            profile['settings'] = profile_settings
            
            # Get tool configurations
            tool_configs = self.db.query(
                'SELECT tool_name, config FROM tool_configs WHERE profile_id = ?',
                (profile_id,)
            )
            
            profile_tool_configs = {}
            for config in tool_configs:
                profile_tool_configs[config['tool_name']] = self.db.json_deserialize(config['config'])
            
            profile['tool_configs'] = profile_tool_configs
            
            # If this is a document profile, get document profile data
            if profile['type'] == 'document':
                doc_profile = self.db.query_one(
                    'SELECT * FROM document_profiles WHERE profile_id = ?',
                    (profile_id,)
                )
                
                if doc_profile:
                    doc_data = {
                        'document_path': doc_profile['document_path'],
                        'content_hash': doc_profile['content_hash'],
                        'basic_properties': self.db.json_deserialize(doc_profile['basic_properties']),
                        'content_metrics': self.db.json_deserialize(doc_profile['content_metrics']),
                        'language_info': self.db.json_deserialize(doc_profile['language_info']),
                        'content_features': self.db.json_deserialize(doc_profile['content_features']),
                        'structure_info': self.db.json_deserialize(doc_profile['structure_info']),
                        'type_specific_features': self.db.json_deserialize(doc_profile['type_specific_features']),
                        'created_at': doc_profile['created_at'],
                        'updated_at': doc_profile['updated_at']
                    }
                    
                    profile['document_profile'] = doc_data
            
            return profile
    
    def get_profile_by_name(self, name: str, profile_type: str, 
                          project_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get a profile by name.
        
        Args:
            name: Profile name
            profile_type: Profile type
            project_id: Optional project ID (None for global profiles)
            
        Returns:
            Profile data or None if not found
        """
        with self._lock:
            # Build query
            if project_id is None:
                query = '''
                SELECT profile_id FROM profiles 
                WHERE name = ? AND type = ? AND project_id IS NULL 
                LIMIT 1
                '''
                params = (name, profile_type)
            else:
                query = '''
                SELECT profile_id FROM profiles 
                WHERE name = ? AND type = ? AND project_id = ? 
                LIMIT 1
                '''
                params = (name, profile_type, project_id)
            
            # Get profile ID
            result = self.db.query_one(query, params)
            
            if not result:
                return None
            
            # Get full profile data
            return self.get_profile(result['profile_id'])
    
    def update_profile(self, profile_id: str, name: Optional[str] = None,
                     description: Optional[str] = None,
                     is_default: Optional[bool] = None) -> bool:
        """
        Update a profile in the store.
        
        Args:
            profile_id: ID of the profile
            name: New name
            description: New description
            is_default: Whether this is the default profile
            
        Returns:
            True if profile was updated, False if not found
        """
        with self._lock:
            # Check if profile exists
            profile = self.db.query_one(
                'SELECT project_id, type FROM profiles WHERE profile_id = ?',
                (profile_id,)
            )
            
            if not profile:
                return False
            
            # Build update data
            update_data = {
                'updated_at': time.time()
            }
            
            if name is not None:
                update_data['name'] = name
            
            if description is not None:
                update_data['description'] = description
            
            # Update profile
            with self.db.transaction():
                # If setting as default, clear existing default
                if is_default is not None:
                    update_data['is_default'] = 1 if is_default else 0
                    
                    if is_default and profile['project_id'] is not None:
                        self.db.update(
                            'profiles',
                            {'is_default': 0},
                            'project_id = ? AND type = ? AND profile_id != ?',
                            (profile['project_id'], profile['type'], profile_id)
                        )
                
                self.db.update(
                    'profiles',
                    update_data,
                    'profile_id = ?',
                    (profile_id,)
                )
                
                return True
    
    def delete_profile(self, profile_id: str) -> bool:
        """
        Delete a profile from the store.
        
        Args:
            profile_id: ID of the profile
            
        Returns:
            True if profile was deleted, False if not found
        """
        with self._lock:
            # Check if profile exists
            profile = self.db.query_one(
                'SELECT is_system FROM profiles WHERE profile_id = ?',
                (profile_id,)
            )
            
            if not profile:
                return False
            
            # Cannot delete system profiles
            if profile['is_system']:
                return False
            
            # Delete profile and related data
            with self.db.transaction():
                # Delete settings
                self.db.delete('profile_settings', 'profile_id = ?', (profile_id,))
                
                # Delete tool configurations
                self.db.delete('tool_configs', 'profile_id = ?', (profile_id,))
                
                # Delete document profile if present
                self.db.delete('document_profiles', 'profile_id = ?', (profile_id,))
                
                # Delete profile
                self.db.delete('profiles', 'profile_id = ?', (profile_id,))
                
                return True
    
    def set_profile_setting(self, profile_id: str, key: str, value: Any) -> bool:
        """
        Set a setting for a profile.
        
        Args:
            profile_id: ID of the profile
            key: Setting key
            value: Setting value
            
        Returns:
            True if setting was set, False if profile not found
        """
        with self._lock:
            # Check if profile exists
            profile_exists = self.db.query_one(
                'SELECT 1 FROM profiles WHERE profile_id = ?',
                (profile_id,)
            )
            
            if not profile_exists:
                return False
            
            # Serialize value
            value_json = self.db.json_serialize(value)
            current_time = time.time()
            
            # Check if setting exists
            setting_exists = self.db.query_one(
                'SELECT 1 FROM profile_settings WHERE profile_id = ? AND key = ?',
                (profile_id, key)
            )
            
            with self.db.transaction():
                if setting_exists:
                    # Update existing setting
                    self.db.update(
                        'profile_settings',
                        {
                            'value': value_json,
                            'updated_at': current_time
                        },
                        'profile_id = ? AND key = ?',
                        (profile_id, key)
                    )
                else:
                    # Insert new setting
                    self.db.insert(
                        'profile_settings',
                        {
                            'profile_id': profile_id,
                            'key': key,
                            'value': value_json,
                            'updated_at': current_time
                        }
                    )
                
                # Update profile updated_at
                self.db.update(
                    'profiles',
                    {'updated_at': current_time},
                    'profile_id = ?',
                    (profile_id,)
                )
                
                return True
    
    def get_profile_setting(self, profile_id: str, key: str, 
                          default: Any = None) -> Any:
        """
        Get a setting for a profile.
        
        Args:
            profile_id: ID of the profile
            key: Setting key
            default: Default value if setting not found
            
        Returns:
            Setting value or default
        """
        with self._lock:
            # Get setting
            setting = self.db.query_one(
                'SELECT value FROM profile_settings WHERE profile_id = ? AND key = ?',
                (profile_id, key)
            )
            
            if not setting:
                return default
            
            # Deserialize value
            return self.db.json_deserialize(setting['value'])
    
    def delete_profile_setting(self, profile_id: str, key: str) -> bool:
        """
        Delete a setting for a profile.
        
        Args:
            profile_id: ID of the profile
            key: Setting key
            
        Returns:
            True if setting was deleted, False if profile or setting not found
        """
        with self._lock:
            with self.db.transaction():
                # Delete setting
                rows_affected = self.db.delete(
                    'profile_settings',
                    'profile_id = ? AND key = ?',
                    (profile_id, key)
                )
                
                if rows_affected > 0:
                    # Update profile updated_at
                    self.db.update(
                        'profiles',
                        {'updated_at': time.time()},
                        'profile_id = ?',
                        (profile_id,)
                    )
                    
                    return True
                
                return False
    
    def set_tool_config(self, profile_id: str, tool_name: str, 
                      config: Dict[str, Any]) -> bool:
        """
        Set a tool configuration for a profile.
        
        Args:
            profile_id: ID of the profile
            tool_name: Name of the tool
            config: Tool configuration
            
        Returns:
            True if configuration was set, False if profile not found
        """
        with self._lock:
            # Check if profile exists
            profile_exists = self.db.query_one(
                'SELECT 1 FROM profiles WHERE profile_id = ?',
                (profile_id,)
            )
            
            if not profile_exists:
                return False
            
            # Serialize config
            config_json = self.db.json_serialize(config)
            current_time = time.time()
            
            # Check if config exists
            config_exists = self.db.query_one(
                'SELECT 1 FROM tool_configs WHERE profile_id = ? AND tool_name = ?',
                (profile_id, tool_name)
            )
            
            with self.db.transaction():
                if config_exists:
                    # Update existing config
                    self.db.update(
                        'tool_configs',
                        {
                            'config': config_json,
                            'updated_at': current_time
                        },
                        'profile_id = ? AND tool_name = ?',
                        (profile_id, tool_name)
                    )
                else:
                    # Insert new config
                    self.db.insert(
                        'tool_configs',
                        {
                            'profile_id': profile_id,
                            'tool_name': tool_name,
                            'config': config_json,
                            'updated_at': current_time
                        }
                    )
                
                # Update profile updated_at
                self.db.update(
                    'profiles',
                    {'updated_at': current_time},
                    'profile_id = ?',
                    (profile_id,)
                )
                
                return True
    
    def get_tool_config(self, profile_id: str, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a tool configuration for a profile.
        
        Args:
            profile_id: ID of the profile
            tool_name: Name of the tool
            
        Returns:
            Tool configuration or None if not found
        """
        with self._lock:
            # Get config
            config = self.db.query_one(
                'SELECT config FROM tool_configs WHERE profile_id = ? AND tool_name = ?',
                (profile_id, tool_name)
            )
            
            if not config:
                return None
            
            # Deserialize config
            return self.db.json_deserialize(config['config'])
    
    def delete_tool_config(self, profile_id: str, tool_name: str) -> bool:
        """
        Delete a tool configuration for a profile.
        
        Args:
            profile_id: ID of the profile
            tool_name: Name of the tool
            
        Returns:
            True if configuration was deleted, False if profile or config not found
        """
        with self._lock:
            with self.db.transaction():
                # Delete config
                rows_affected = self.db.delete(
                    'tool_configs',
                    'profile_id = ? AND tool_name = ?',
                    (profile_id, tool_name)
                )
                
                if rows_affected > 0:
                    # Update profile updated_at
                    self.db.update(
                        'profiles',
                        {'updated_at': time.time()},
                        'profile_id = ?',
                        (profile_id,)
                    )
                    
                    return True
                
                return False
    
    def mark_as_used(self, profile_id: str) -> bool:
        """
        Mark a profile as used (update last_used_at timestamp).
        
        Args:
            profile_id: ID of the profile
            
        Returns:
            True if profile was updated, False if not found
        """
        with self._lock:
            # Update profile
            rows_affected = self.db.update(
                'profiles',
                {'last_used_at': time.time()},
                'profile_id = ?',
                (profile_id,)
            )
            
            return rows_affected > 0
    
    def get_default_profile(self, profile_type: str, 
                          project_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get the default profile of a specific type.
        
        Args:
            profile_type: Profile type
            project_id: Optional project ID (None for global profiles)
            
        Returns:
            Default profile or None if not found
        """
        with self._lock:
            # Build query
            if project_id is None:
                query = '''
                SELECT profile_id FROM profiles 
                WHERE type = ? AND project_id IS NULL AND is_default = 1 
                LIMIT 1
                '''
                params = (profile_type,)
            else:
                query = '''
                SELECT profile_id FROM profiles 
                WHERE type = ? AND project_id = ? AND is_default = 1 
                LIMIT 1
                '''
                params = (profile_type, project_id)
            
            # Get profile ID
            result = self.db.query_one(query, params)
            
            if not result:
                return None
            
            # Get full profile data
            return self.get_profile(result['profile_id'])
    
    def search_profiles(self, profile_type: Optional[str] = None,
                      project_id: Optional[str] = None,
                      query: Optional[str] = None,
                      include_system: bool = True,
                      limit: int = 100,
                      offset: int = 0,
                      sort_by: str = 'updated_at',
                      sort_order: str = 'desc') -> Tuple[List[Dict[str, Any]], int]:
        """
        Search for profiles in the store.
        
        Args:
            profile_type: Filter by profile type
            project_id: Filter by project ID
            query: Text search query
            include_system: Whether to include system profiles
            limit: Maximum number of results to return
            offset: Offset for pagination
            sort_by: Field to sort by
            sort_order: Sort order ('asc' or 'desc')
            
        Returns:
            Tuple of (profile list, total count)
        """
        with self._lock:
            # Build the query
            query_parts = ['SELECT * FROM profiles']
            where_clauses = []
            params = []
            
            # Add filter conditions
            if profile_type:
                where_clauses.append('type = ?')
                params.append(profile_type)
            
            if project_id is not None:
                where_clauses.append('project_id = ?')
                params.append(project_id)
            elif project_id is None and not where_clauses:
                # Only filter by project_id IS NULL if no other filters are applied
                # This allows getting both project-specific and global profiles
                pass
            
            if not include_system:
                where_clauses.append('is_system = 0')
            
            if query:
                # Search in name and description
                where_clauses.append('(name LIKE ? OR description LIKE ?)')
                search_term = f'%{query}%'
                params.extend([search_term, search_term])
            
            # Combine WHERE clauses
            if where_clauses:
                query_parts.append('WHERE ' + ' AND '.join(where_clauses))
            
            # Build the full query for count
            count_query = ' '.join(['SELECT COUNT(*) as count FROM ('] + query_parts + [') as subquery'])
            
            # Get total count
            count_result = self.db.query_one(count_query, params)
            total_count = count_result['count'] if count_result else 0
            
            # Add sorting
            valid_sort_fields = {
                'name', 'created_at', 'updated_at', 'last_used_at'
            }
            if sort_by not in valid_sort_fields:
                sort_by = 'updated_at'
            
            sort_order = 'DESC' if sort_order.lower() == 'desc' else 'ASC'
            query_parts.append(f'ORDER BY {sort_by} {sort_order}')
            
            # Add pagination
            query_parts.append('LIMIT ? OFFSET ?')
            params.append(limit)
            params.append(offset)
            
            # Execute the query
            query = ' '.join(query_parts)
            profiles = self.db.query(query, params)
            
            # Process results
            result = []
            for profile in profiles:
                # Convert boolean fields
                profile['is_default'] = bool(profile['is_default'])
                profile['is_system'] = bool(profile['is_system'])
                
                # Get settings count
                settings_count = self.db.query_one(
                    'SELECT COUNT(*) as count FROM profile_settings WHERE profile_id = ?',
                    (profile['profile_id'],)
                )
                
                profile['settings_count'] = settings_count['count'] if settings_count else 0
                
                # Get tool configs count
                tool_configs_count = self.db.query_one(
                    'SELECT COUNT(*) as count FROM tool_configs WHERE profile_id = ?',
                    (profile['profile_id'],)
                )
                
                profile['tool_configs_count'] = tool_configs_count['count'] if tool_configs_count else 0
                
                result.append(profile)
            
            return result, total_count
    
    def add_document_profile(self, document_path: str, basic_properties: Dict[str, Any],
                          profile_id: Optional[str] = None, name: Optional[str] = None,
                          content_hash: Optional[str] = None,
                          content_metrics: Optional[Dict[str, Any]] = None,
                          language_info: Optional[Dict[str, Any]] = None,
                          content_features: Optional[Dict[str, Any]] = None,
                          structure_info: Optional[Dict[str, Any]] = None,
                          type_specific_features: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a document profile to the store.
        
        Args:
            document_path: Path to the document
            basic_properties: Basic document properties
            profile_id: Optional profile ID (generated if None)
            name: Optional profile name
            content_hash: Optional hash of document content
            content_metrics: Optional content metrics
            language_info: Optional language information
            content_features: Optional content features
            structure_info: Optional structure information
            type_specific_features: Optional type-specific features
            
        Returns:
            ID of the added profile
        """
        with self._lock:
            with self.db.transaction():
                # Use existing profile ID or generate a new one
                if profile_id is None:
                    profile_id = str(uuid.uuid4())
                
                # Current time
                current_time = time.time()
                
                # Default name if not provided
                if not name:
                    name = os.path.basename(document_path)
                
                # Check if a profile already exists for this document
                existing_profile = self.db.query_one(
                    'SELECT profile_id FROM document_profiles WHERE document_path = ?',
                    (document_path,)
                )
                
                if existing_profile:
                    # Update existing document profile
                    self.db.update(
                        'document_profiles',
                        {
                            'content_hash': content_hash,
                            'basic_properties': self.db.json_serialize(basic_properties),
                            'content_metrics': self.db.json_serialize(content_metrics),
                            'language_info': self.db.json_serialize(language_info),
                            'content_features': self.db.json_serialize(content_features),
                            'structure_info': self.db.json_serialize(structure_info),
                            'type_specific_features': self.db.json_serialize(type_specific_features),
                            'updated_at': current_time
                        },
                        'profile_id = ?',
                        (existing_profile['profile_id'],)
                    )
                    
                    # Update profile
                    self.db.update(
                        'profiles',
                        {
                            'name': name,
                            'updated_at': current_time,
                            'last_used_at': current_time
                        },
                        'profile_id = ?',
                        (existing_profile['profile_id'],)
                    )
                    
                    return existing_profile['profile_id']
                else:
                    # Create new profile
                    self.add_profile(
                        name=name,
                        profile_type='document',
                        description=f"Profile for {document_path}",
                        profile_id=profile_id
                    )
                    
                    # Add document profile
                    self.db.insert(
                        'document_profiles',
                        {
                            'profile_id': profile_id,
                            'document_path': document_path,
                            'content_hash': content_hash,
                            'basic_properties': self.db.json_serialize(basic_properties),
                            'content_metrics': self.db.json_serialize(content_metrics),
                            'language_info': self.db.json_serialize(language_info),
                            'content_features': self.db.json_serialize(content_features),
                            'structure_info': self.db.json_serialize(structure_info),
                            'type_specific_features': self.db.json_serialize(type_specific_features),
                            'created_at': current_time,
                            'updated_at': current_time
                        }
                    )
                    
                    return profile_id
    
    def get_document_profile(self, document_path: str) -> Optional[Dict[str, Any]]:
        """
        Get a document profile by document path.
        
        Args:
            document_path: Path to the document
            
        Returns:
            Document profile or None if not found
        """
        with self._lock:
            # Get profile ID
            result = self.db.query_one(
                'SELECT profile_id FROM document_profiles WHERE document_path = ?',
                (document_path,)
            )
            
            if not result:
                return None
            
            # Get full profile data
            return self.get_profile(result['profile_id'])
    
    def find_similar_documents(self, content_features: Dict[str, Any],
                             max_results: int = 5,
                             similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Find documents with similar content features.
        
        Args:
            content_features: Content features to compare
            max_results: Maximum number of results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of similar document profiles
        """
        with self._lock:
            # Get all document profiles
            profiles = self.db.query(
                'SELECT profile_id, document_path, content_features FROM document_profiles'
            )
            
            if not profiles:
                return []
            
            # Calculate similarity scores
            documents_with_scores = []
            
            for profile in profiles:
                profile_features = self.db.json_deserialize(profile['content_features'])
                
                if not profile_features:
                    continue
                
                similarity = self._calculate_content_similarity(content_features, profile_features)
                
                if similarity >= similarity_threshold:
                    documents_with_scores.append((profile['profile_id'], profile['document_path'], similarity))
            
            # Sort by similarity (highest first)
            documents_with_scores.sort(key=lambda x: x[2], reverse=True)
            
            # Take top results
            results = []
            for profile_id, document_path, similarity in documents_with_scores[:max_results]:
                profile = self.get_profile(profile_id)
                if profile:
                    profile['similarity'] = similarity
                    results.append(profile)
            
            return results
    
    def _calculate_content_similarity(self, features1: Dict[str, Any], 
                                   features2: Dict[str, Any]) -> float:
        """
        Calculate similarity between two sets of content features.
        
        Args:
            features1: First set of content features
            features2: Second set of content features
            
        Returns:
            Similarity score (0.0 - 1.0)
        """
        # Simple implementation using common terms
        if 'terms' in features1 and 'terms' in features2:
            terms1 = set(features1.get('terms', {}).keys())
            terms2 = set(features2.get('terms', {}).keys())
            
            if not terms1 or not terms2:
                return 0.0
            
            # Jaccard similarity
            intersection = len(terms1.intersection(terms2))
            union = len(terms1.union(terms2))
            
            if union == 0:
                return 0.0
                
            return intersection / union
        
        # Fallback to simple structural similarity
        return 0.5  # Default mid-range similarity


# Global registry of profile stores
_profile_stores = {}
_profile_lock = threading.RLock()


def get_profile_store(db_name: str = 'profiles') -> ProfileStore:
    """
    Get a profile store.
    
    Args:
        db_name: Database name
        
    Returns:
        ProfileStore instance
    """
    with _profile_lock:
        if db_name in _profile_stores:
            return _profile_stores[db_name]
        
        # Get or create the database
        db = get_database(db_name)
        
        if db is None:
            # Create profile database in data directory
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
        
        # Create profile store
        store = ProfileStore(db)
        _profile_stores[db_name] = store
        
        return store


def add_profile(name: str, profile_type: str, project_id: Optional[str] = None,
              description: Optional[str] = None, is_default: bool = False,
              is_system: bool = False) -> str:
    """
    Add a profile to the store.
    
    Args:
        name: Profile name
        profile_type: Profile type ('project', 'document', etc.)
        project_id: Optional project ID (None for global profiles)
        description: Optional description
        is_default: Whether this is the default profile
        is_system: Whether this is a system profile
        
    Returns:
        ID of the added profile
    """
    store = get_profile_store()
    return store.add_profile(
        name, profile_type, project_id, description, is_default, is_system
    )


def get_profile(profile_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a profile from the store.
    
    Args:
        profile_id: ID of the profile
        
    Returns:
        Profile data or None if not found
    """
    store = get_profile_store()
    return store.get_profile(profile_id)


def get_default_profile(profile_type: str, 
                      project_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Get the default profile of a specific type.
    
    Args:
        profile_type: Profile type
        project_id: Optional project ID (None for global profiles)
        
    Returns:
        Default profile or None if not found
    """
    store = get_profile_store()
    return store.get_default_profile(profile_type, project_id)


def set_profile_setting(profile_id: str, key: str, value: Any) -> bool:
    """
    Set a setting for a profile.
    
    Args:
        profile_id: ID of the profile
        key: Setting key
        value: Setting value
        
    Returns:
        True if setting was set, False if profile not found
    """
    store = get_profile_store()
    return store.set_profile_setting(profile_id, key, value)


def get_profile_setting(profile_id: str, key: str, default: Any = None) -> Any:
    """
    Get a setting for a profile.
    
    Args:
        profile_id: ID of the profile
        key: Setting key
        default: Default value if setting not found
        
    Returns:
        Setting value or default
    """
    store = get_profile_store()
    return store.get_profile_setting(profile_id, key, default)


def add_document_profile(document_path: str, basic_properties: Dict[str, Any],
                       content_hash: Optional[str] = None,
                       content_metrics: Optional[Dict[str, Any]] = None,
                       language_info: Optional[Dict[str, Any]] = None,
                       content_features: Optional[Dict[str, Any]] = None,
                       structure_info: Optional[Dict[str, Any]] = None,
                       type_specific_features: Optional[Dict[str, Any]] = None) -> str:
    """
    Add a document profile to the store.
    
    Args:
        document_path: Path to the document
        basic_properties: Basic document properties
        content_hash: Optional hash of document content
        content_metrics: Optional content metrics
        language_info: Optional language information
        content_features: Optional content features
        structure_info: Optional structure information
        type_specific_features: Optional type-specific features
        
    Returns:
        ID of the added profile
    """
    store = get_profile_store()
    return store.add_document_profile(
        document_path, basic_properties, None, None, content_hash,
        content_metrics, language_info, content_features,
        structure_info, type_specific_features
    )


def get_document_profile(document_path: str) -> Optional[Dict[str, Any]]:
    """
    Get a document profile by document path.
    
    Args:
        document_path: Path to the document
        
    Returns:
        Document profile or None if not found
    """
    store = get_profile_store()
    return store.get_document_profile(document_path)


def find_similar_documents(content_features: Dict[str, Any],
                         max_results: int = 5,
                         similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
    """
    Find documents with similar content features.
    
    Args:
        content_features: Content features to compare
        max_results: Maximum number of results to return
        similarity_threshold: Minimum similarity threshold
        
    Returns:
        List of similar document profiles
    """
    store = get_profile_store()
    return store.find_similar_documents(content_features, max_results, similarity_threshold)