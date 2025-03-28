"""
Project profile management for TSAP.

This module provides functionality for managing project-specific configurations,
user preferences, saved states, and profiles for optimizing project workflows.
"""

import os
import json
import copy
import time
import uuid
import threading
from typing import Dict, List, Any, Optional

import tsap.utils.logging as logging
from tsap.utils.errors import TSAPError
from tsap.project.context import ProjectContext


class ProfileError(TSAPError):
    """Exception raised for profile-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.details = details or {}


class ProjectProfile:
    """
    Represents a specific configuration profile for a project.
    
    A profile contains settings, preferences, and configurations that
    can be saved and loaded to customize project behavior.
    """
    
    def __init__(self, 
                profile_id: Optional[str] = None,
                name: Optional[str] = None,
                description: Optional[str] = None):
        """
        Initialize a project profile.
        
        Args:
            profile_id: Unique ID for the profile
            name: Name of the profile
            description: Description of the profile
        """
        self.profile_id = profile_id or str(uuid.uuid4())
        self.name = name or f"Profile-{self.profile_id[:8]}"
        self.description = description or ""
        
        self.created_at = time.time()
        self.updated_at = time.time()
        self.last_used_at = time.time()
        
        # Settings storage
        self._settings: Dict[str, Any] = {}
        
        # Tool configurations
        self._tool_configs: Dict[str, Dict[str, Any]] = {}
        
        # Search patterns and templates
        self._patterns: Dict[str, Dict[str, Any]] = {}
        self._templates: Dict[str, Dict[str, Any]] = {}
        
        # View states and UI preferences
        self._view_states: Dict[str, Dict[str, Any]] = {}
        self._ui_preferences: Dict[str, Any] = {}
        
        # Performance profiles
        self._performance_configs: Dict[str, Dict[str, Any]] = {}
        
        # Lock for thread safety
        self._lock = threading.RLock()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert profile to a dictionary.
        
        Returns:
            Dictionary representation of the profile
        """
        with self._lock:
            return {
                'profile_id': self.profile_id,
                'name': self.name,
                'description': self.description,
                'created_at': self.created_at,
                'updated_at': self.updated_at,
                'last_used_at': self.last_used_at,
                'settings': self._settings,
                'tool_configs': self._tool_configs,
                'patterns': self._patterns,
                'templates': self._templates,
                'view_states': self._view_states,
                'ui_preferences': self._ui_preferences,
                'performance_configs': self._performance_configs
            }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProjectProfile':
        """
        Create a profile from a dictionary.
        
        Args:
            data: Dictionary representation of a profile
            
        Returns:
            New ProjectProfile instance
        """
        profile = cls(
            profile_id=data.get('profile_id'),
            name=data.get('name'),
            description=data.get('description')
        )
        
        profile.created_at = data.get('created_at', profile.created_at)
        profile.updated_at = data.get('updated_at', profile.updated_at)
        profile.last_used_at = data.get('last_used_at', profile.last_used_at)
        
        # Load settings and configurations
        with profile._lock:
            profile._settings = data.get('settings', {})
            profile._tool_configs = data.get('tool_configs', {})
            profile._patterns = data.get('patterns', {})
            profile._templates = data.get('templates', {})
            profile._view_states = data.get('view_states', {})
            profile._ui_preferences = data.get('ui_preferences', {})
            profile._performance_configs = data.get('performance_configs', {})
        
        return profile
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Get a profile setting.
        
        Args:
            key: Setting key
            default: Default value if setting is not found
            
        Returns:
            Setting value or default
        """
        with self._lock:
            return self._settings.get(key, default)
    
    def set_setting(self, key: str, value: Any) -> None:
        """
        Set a profile setting.
        
        Args:
            key: Setting key
            value: Setting value
        """
        with self._lock:
            self._settings[key] = value
            self.updated_at = time.time()
    
    def delete_setting(self, key: str) -> bool:
        """
        Delete a profile setting.
        
        Args:
            key: Setting key
            
        Returns:
            True if setting was deleted, False if not found
        """
        with self._lock:
            if key in self._settings:
                del self._settings[key]
                self.updated_at = time.time()
                return True
            return False
    
    def get_tool_config(self, tool_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool configuration dictionary
        """
        with self._lock:
            return self._tool_configs.get(tool_name, {})
    
    def set_tool_config(self, tool_name: str, config: Dict[str, Any]) -> None:
        """
        Set configuration for a specific tool.
        
        Args:
            tool_name: Name of the tool
            config: Tool configuration
        """
        with self._lock:
            self._tool_configs[tool_name] = config
            self.updated_at = time.time()
    
    def update_tool_config(self, tool_name: str, updates: Dict[str, Any]) -> None:
        """
        Update configuration for a specific tool.
        
        Args:
            tool_name: Name of the tool
            updates: Configuration updates
        """
        with self._lock:
            if tool_name not in self._tool_configs:
                self._tool_configs[tool_name] = {}
                
            self._tool_configs[tool_name].update(updates)
            self.updated_at = time.time()
    
    def delete_tool_config(self, tool_name: str) -> bool:
        """
        Delete configuration for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            True if configuration was deleted, False if not found
        """
        with self._lock:
            if tool_name in self._tool_configs:
                del self._tool_configs[tool_name]
                self.updated_at = time.time()
                return True
            return False
    
    def get_all_tool_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all tool configurations.
        
        Returns:
            Dictionary of tool configurations
        """
        with self._lock:
            return copy.deepcopy(self._tool_configs)
    
    def get_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a search pattern.
        
        Args:
            pattern_id: ID of the pattern
            
        Returns:
            Pattern dictionary or None if not found
        """
        with self._lock:
            return copy.deepcopy(self._patterns.get(pattern_id))
    
    def add_pattern(self, pattern: Dict[str, Any], pattern_id: Optional[str] = None) -> str:
        """
        Add a search pattern.
        
        Args:
            pattern: Pattern dictionary
            pattern_id: Optional ID for the pattern
            
        Returns:
            ID of the added pattern
        """
        with self._lock:
            # Generate ID if not provided
            if pattern_id is None:
                pattern_id = str(uuid.uuid4())
            
            # Add metadata if not present
            if 'id' not in pattern:
                pattern['id'] = pattern_id
            if 'created_at' not in pattern:
                pattern['created_at'] = time.time()
            
            pattern['updated_at'] = time.time()
            
            self._patterns[pattern_id] = pattern
            self.updated_at = time.time()
            
            return pattern_id
    
    def update_pattern(self, pattern_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a search pattern.
        
        Args:
            pattern_id: ID of the pattern
            updates: Pattern updates
            
        Returns:
            True if pattern was updated, False if not found
        """
        with self._lock:
            if pattern_id in self._patterns:
                self._patterns[pattern_id].update(updates)
                self._patterns[pattern_id]['updated_at'] = time.time()
                self.updated_at = time.time()
                return True
            return False
    
    def delete_pattern(self, pattern_id: str) -> bool:
        """
        Delete a search pattern.
        
        Args:
            pattern_id: ID of the pattern
            
        Returns:
            True if pattern was deleted, False if not found
        """
        with self._lock:
            if pattern_id in self._patterns:
                del self._patterns[pattern_id]
                self.updated_at = time.time()
                return True
            return False
    
    def get_all_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all search patterns.
        
        Returns:
            Dictionary of pattern dictionaries
        """
        with self._lock:
            return copy.deepcopy(self._patterns)
    
    def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a template.
        
        Args:
            template_id: ID of the template
            
        Returns:
            Template dictionary or None if not found
        """
        with self._lock:
            return copy.deepcopy(self._templates.get(template_id))
    
    def add_template(self, template: Dict[str, Any], template_id: Optional[str] = None) -> str:
        """
        Add a template.
        
        Args:
            template: Template dictionary
            template_id: Optional ID for the template
            
        Returns:
            ID of the added template
        """
        with self._lock:
            # Generate ID if not provided
            if template_id is None:
                template_id = str(uuid.uuid4())
            
            # Add metadata if not present
            if 'id' not in template:
                template['id'] = template_id
            if 'created_at' not in template:
                template['created_at'] = time.time()
            
            template['updated_at'] = time.time()
            
            self._templates[template_id] = template
            self.updated_at = time.time()
            
            return template_id
    
    def update_template(self, template_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a template.
        
        Args:
            template_id: ID of the template
            updates: Template updates
            
        Returns:
            True if template was updated, False if not found
        """
        with self._lock:
            if template_id in self._templates:
                self._templates[template_id].update(updates)
                self._templates[template_id]['updated_at'] = time.time()
                self.updated_at = time.time()
                return True
            return False
    
    def delete_template(self, template_id: str) -> bool:
        """
        Delete a template.
        
        Args:
            template_id: ID of the template
            
        Returns:
            True if template was deleted, False if not found
        """
        with self._lock:
            if template_id in self._templates:
                del self._templates[template_id]
                self.updated_at = time.time()
                return True
            return False
    
    def get_all_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all templates.
        
        Returns:
            Dictionary of template dictionaries
        """
        with self._lock:
            return copy.deepcopy(self._templates)
    
    def get_view_state(self, view_id: str) -> Dict[str, Any]:
        """
        Get state for a specific view.
        
        Args:
            view_id: ID of the view
            
        Returns:
            View state dictionary
        """
        with self._lock:
            return copy.deepcopy(self._view_states.get(view_id, {}))
    
    def set_view_state(self, view_id: str, state: Dict[str, Any]) -> None:
        """
        Set state for a specific view.
        
        Args:
            view_id: ID of the view
            state: View state
        """
        with self._lock:
            self._view_states[view_id] = state
            self.updated_at = time.time()
    
    def update_view_state(self, view_id: str, updates: Dict[str, Any]) -> None:
        """
        Update state for a specific view.
        
        Args:
            view_id: ID of the view
            updates: State updates
        """
        with self._lock:
            if view_id not in self._view_states:
                self._view_states[view_id] = {}
                
            self._view_states[view_id].update(updates)
            self.updated_at = time.time()
    
    def get_ui_preference(self, key: str, default: Any = None) -> Any:
        """
        Get a UI preference.
        
        Args:
            key: Preference key
            default: Default value if preference is not found
            
        Returns:
            Preference value or default
        """
        with self._lock:
            return self._ui_preferences.get(key, default)
    
    def set_ui_preference(self, key: str, value: Any) -> None:
        """
        Set a UI preference.
        
        Args:
            key: Preference key
            value: Preference value
        """
        with self._lock:
            self._ui_preferences[key] = value
            self.updated_at = time.time()
    
    def get_all_ui_preferences(self) -> Dict[str, Any]:
        """
        Get all UI preferences.
        
        Returns:
            Dictionary of UI preferences
        """
        with self._lock:
            return copy.deepcopy(self._ui_preferences)
    
    def get_performance_config(self, config_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a performance configuration.
        
        Args:
            config_id: ID of the configuration
            
        Returns:
            Performance configuration or None if not found
        """
        with self._lock:
            return copy.deepcopy(self._performance_configs.get(config_id))
    
    def add_performance_config(self, config: Dict[str, Any], config_id: Optional[str] = None) -> str:
        """
        Add a performance configuration.
        
        Args:
            config: Performance configuration
            config_id: Optional ID for the configuration
            
        Returns:
            ID of the added configuration
        """
        with self._lock:
            # Generate ID if not provided
            if config_id is None:
                config_id = str(uuid.uuid4())
            
            # Add metadata if not present
            if 'id' not in config:
                config['id'] = config_id
            if 'created_at' not in config:
                config['created_at'] = time.time()
            
            config['updated_at'] = time.time()
            
            self._performance_configs[config_id] = config
            self.updated_at = time.time()
            
            return config_id
    
    def get_all_performance_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all performance configurations.
        
        Returns:
            Dictionary of performance configurations
        """
        with self._lock:
            return copy.deepcopy(self._performance_configs)
    
    def mark_as_used(self) -> None:
        """Mark profile as used (update last_used_at timestamp)."""
        with self._lock:
            self.last_used_at = time.time()


class ProfileManager:
    """
    Manages profiles for a project.
    
    This class handles loading, saving, and managing multiple profiles
    for a project, including selecting the active profile.
    """
    
    def __init__(self, project: ProjectContext):
        """
        Initialize profile manager for a project.
        
        Args:
            project: Project context
        """
        self.project = project
        self._profiles: Dict[str, ProjectProfile] = {}
        self._active_profile_id: Optional[str] = None
        self._default_profile_id: Optional[str] = None
        self._lock = threading.RLock()
        
        # Create profiles directory
        self._profiles_dir = None
        if project.output_directory:
            self._profiles_dir = os.path.join(project.output_directory, "profiles")
            os.makedirs(self._profiles_dir, exist_ok=True)
        
        # Create a default profile
        self._create_default_profile()
    
    def _create_default_profile(self) -> None:
        """Create a default profile."""
        with self._lock:
            default = ProjectProfile(
                name="Default Profile",
                description="Default configuration profile"
            )
            
            self._profiles[default.profile_id] = default
            self._default_profile_id = default.profile_id
            
            if self._active_profile_id is None:
                self._active_profile_id = default.profile_id
            
            # Save the default profile
            if self._profiles_dir:
                self._save_profile(default)
    
    def _save_profile(self, profile: ProjectProfile) -> str:
        """
        Save a profile to disk.
        
        Args:
            profile: Profile to save
            
        Returns:
            Path to saved file
        """
        if not self._profiles_dir:
            raise ProfileError("No profiles directory available")
        
        try:
            # Generate file path
            file_path = os.path.join(self._profiles_dir, f"{profile.profile_id}.json")
            
            # Convert to dictionary
            data = profile.to_dict()
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logging.debug(f"Saved profile {profile.profile_id} to {file_path}", 
                        component="profile")
            
            return file_path
        
        except Exception as e:
            raise ProfileError(f"Failed to save profile: {str(e)}")
    
    def _load_profile(self, file_path: str) -> ProjectProfile:
        """
        Load a profile from disk.
        
        Args:
            file_path: Path to profile file
            
        Returns:
            Loaded profile
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            profile = ProjectProfile.from_dict(data)
            logging.debug(f"Loaded profile {profile.profile_id} from {file_path}", 
                        component="profile")
            
            return profile
        
        except Exception as e:
            raise ProfileError(f"Failed to load profile from {file_path}: {str(e)}")
    
    def load_profiles(self) -> None:
        """Load all profiles from disk."""
        if not self._profiles_dir or not os.path.exists(self._profiles_dir):
            return
        
        with self._lock:
            # Clear existing profiles (except default)
            default_profile = self._profiles.get(self._default_profile_id) if self._default_profile_id else None
            self._profiles.clear()
            
            if default_profile:
                self._profiles[default_profile.profile_id] = default_profile
            
            # Load profiles from disk
            for filename in os.listdir(self._profiles_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self._profiles_dir, filename)
                    
                    try:
                        profile = self._load_profile(file_path)
                        self._profiles[profile.profile_id] = profile
                        
                        # Set as default if marked as such
                        if profile.get_setting('is_default', False):
                            self._default_profile_id = profile.profile_id
                    
                    except Exception as e:
                        logging.warning(f"Failed to load profile from {file_path}: {str(e)}", 
                                      component="profile")
            
            # Ensure we have an active profile
            if self._active_profile_id not in self._profiles:
                self._active_profile_id = self._default_profile_id
            
            logging.info(f"Loaded {len(self._profiles)} profiles", component="profile")
    
    def save_profiles(self) -> None:
        """Save all profiles to disk."""
        if not self._profiles_dir:
            return
        
        with self._lock:
            for profile_id, profile in self._profiles.items():
                try:
                    self._save_profile(profile)
                except Exception as e:
                    logging.warning(f"Failed to save profile {profile_id}: {str(e)}", 
                                  component="profile")
    
    def create_profile(self, name: str, description: Optional[str] = None) -> ProjectProfile:
        """
        Create a new profile.
        
        Args:
            name: Name for the profile
            description: Optional description
            
        Returns:
            New profile
        """
        with self._lock:
            profile = ProjectProfile(name=name, description=description)
            self._profiles[profile.profile_id] = profile
            
            # Save the profile
            if self._profiles_dir:
                try:
                    self._save_profile(profile)
                except Exception as e:
                    logging.warning(f"Failed to save new profile: {str(e)}", 
                                  component="profile")
            
            return profile
    
    def get_profile(self, profile_id: str) -> Optional[ProjectProfile]:
        """
        Get a profile by ID.
        
        Args:
            profile_id: ID of the profile
            
        Returns:
            Profile or None if not found
        """
        with self._lock:
            return self._profiles.get(profile_id)
    
    def list_profiles(self) -> List[Dict[str, Any]]:
        """
        List all profiles.
        
        Returns:
            List of profile metadata
        """
        with self._lock:
            return [
                {
                    'profile_id': p.profile_id,
                    'name': p.name,
                    'description': p.description,
                    'created_at': p.created_at,
                    'updated_at': p.updated_at,
                    'last_used_at': p.last_used_at,
                    'is_active': p.profile_id == self._active_profile_id,
                    'is_default': p.profile_id == self._default_profile_id
                }
                for p in self._profiles.values()
            ]
    
    def delete_profile(self, profile_id: str) -> bool:
        """
        Delete a profile.
        
        Args:
            profile_id: ID of the profile
            
        Returns:
            True if profile was deleted, False if not found or is default
        """
        with self._lock:
            # Cannot delete default profile
            if profile_id == self._default_profile_id:
                return False
            
            if profile_id not in self._profiles:
                return False
            
            # Delete the profile
            del self._profiles[profile_id]
            
            # If active profile was deleted, switch to default
            if profile_id == self._active_profile_id:
                self._active_profile_id = self._default_profile_id
            
            # Delete profile file
            if self._profiles_dir:
                file_path = os.path.join(self._profiles_dir, f"{profile_id}.json")
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        logging.warning(f"Failed to delete profile file: {str(e)}", 
                                      component="profile")
            
            return True
    
    def set_active_profile(self, profile_id: str) -> bool:
        """
        Set the active profile.
        
        Args:
            profile_id: ID of the profile
            
        Returns:
            True if active profile was set, False if profile not found
        """
        with self._lock:
            if profile_id not in self._profiles:
                return False
            
            old_active = self._active_profile_id
            self._active_profile_id = profile_id
            
            # Mark as used
            self._profiles[profile_id].mark_as_used()
            
            # Save the profile
            if self._profiles_dir:
                try:
                    self._save_profile(self._profiles[profile_id])
                except Exception as e:
                    logging.warning(f"Failed to save profile after activation: {str(e)}", 
                                  component="profile")
            
            if old_active != profile_id:
                logging.debug(f"Set active profile to {profile_id}", component="profile")
            
            return True
    
    def get_active_profile(self) -> Optional[ProjectProfile]:
        """
        Get the active profile.
        
        Returns:
            Active profile or None if no active profile
        """
        with self._lock:
            if self._active_profile_id:
                return self._profiles.get(self._active_profile_id)
            return None
    
    def set_default_profile(self, profile_id: str) -> bool:
        """
        Set the default profile.
        
        Args:
            profile_id: ID of the profile
            
        Returns:
            True if default profile was set, False if profile not found
        """
        with self._lock:
            if profile_id not in self._profiles:
                return False
            
            # Update old default
            if self._default_profile_id and self._default_profile_id in self._profiles:
                self._profiles[self._default_profile_id].set_setting('is_default', False)
                
                # Save old default
                if self._profiles_dir:
                    try:
                        self._save_profile(self._profiles[self._default_profile_id])
                    except Exception:
                        pass
            
            # Set new default
            self._default_profile_id = profile_id
            self._profiles[profile_id].set_setting('is_default', True)
            
            # Save new default
            if self._profiles_dir:
                try:
                    self._save_profile(self._profiles[profile_id])
                except Exception as e:
                    logging.warning(f"Failed to save profile after setting as default: {str(e)}", 
                                  component="profile")
            
            return True
    
    def get_default_profile(self) -> Optional[ProjectProfile]:
        """
        Get the default profile.
        
        Returns:
            Default profile or None if no default profile
        """
        with self._lock:
            if self._default_profile_id:
                return self._profiles.get(self._default_profile_id)
            return None
    
    def clone_profile(self, profile_id: str, new_name: str) -> Optional[ProjectProfile]:
        """
        Clone a profile.
        
        Args:
            profile_id: ID of the profile to clone
            new_name: Name for the cloned profile
            
        Returns:
            Cloned profile or None if source profile not found
        """
        with self._lock:
            source = self.get_profile(profile_id)
            if not source:
                return None
            
            # Create new profile
            clone = ProjectProfile(
                name=new_name,
                description=f"Clone of {source.name}"
            )
            
            # Clone settings and configurations
            clone._settings = copy.deepcopy(source._settings)
            clone._tool_configs = copy.deepcopy(source._tool_configs)
            clone._patterns = copy.deepcopy(source._patterns)
            clone._templates = copy.deepcopy(source._templates)
            clone._view_states = copy.deepcopy(source._view_states)
            clone._ui_preferences = copy.deepcopy(source._ui_preferences)
            clone._performance_configs = copy.deepcopy(source._performance_configs)
            
            # Ensure clone is not marked as default
            clone._settings.pop('is_default', None)
            
            # Add to profiles
            self._profiles[clone.profile_id] = clone
            
            # Save the cloned profile
            if self._profiles_dir:
                try:
                    self._save_profile(clone)
                except Exception as e:
                    logging.warning(f"Failed to save cloned profile: {str(e)}", 
                                  component="profile")
            
            return clone
    
    def export_profile(self, profile_id: str, file_path: str) -> bool:
        """
        Export a profile to a file.
        
        Args:
            profile_id: ID of the profile
            file_path: Path to export to
            
        Returns:
            True if profile was exported, False if profile not found
        """
        with self._lock:
            profile = self.get_profile(profile_id)
            if not profile:
                return False
            
            try:
                # Convert to dictionary
                data = profile.to_dict()
                
                # Save to file
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                logging.debug(f"Exported profile {profile_id} to {file_path}", 
                            component="profile")
                
                return True
                
            except Exception as e:
                logging.warning(f"Failed to export profile: {str(e)}", 
                              component="profile")
                return False
    
    def import_profile(self, file_path: str) -> Optional[ProjectProfile]:
        """
        Import a profile from a file.
        
        Args:
            file_path: Path to import from
            
        Returns:
            Imported profile or None on error
        """
        with self._lock:
            try:
                # Load the profile
                profile = self._load_profile(file_path)
                
                # Check for duplicate ID
                if profile.profile_id in self._profiles:
                    # Generate new ID
                    old_id = profile.profile_id
                    profile.profile_id = str(uuid.uuid4())
                    logging.debug(f"Changed duplicate profile ID from {old_id} to {profile.profile_id}", 
                                component="profile")
                
                # Update timestamps
                profile.created_at = time.time()
                profile.updated_at = time.time()
                profile.last_used_at = time.time()
                
                # Add to profiles
                self._profiles[profile.profile_id] = profile
                
                # Save the imported profile
                if self._profiles_dir:
                    try:
                        self._save_profile(profile)
                    except Exception as e:
                        logging.warning(f"Failed to save imported profile: {str(e)}", 
                                      component="profile")
                
                return profile
                
            except Exception as e:
                logging.warning(f"Failed to import profile from {file_path}: {str(e)}", 
                              component="profile")
                return None


# Registry of profile managers
_profile_managers = {}
_profile_lock = threading.RLock()


def get_profile_manager(project_id: Optional[str] = None) -> ProfileManager:
    """
    Get the profile manager for a project.
    
    Args:
        project_id: ID of the project (or None for active project)
        
    Returns:
        ProfileManager instance
        
    Raises:
        ProfileError: If no project is found or active
    """
    from tsap.project.context import get_project
    
    # Get the project
    project = get_project(project_id)
    
    if not project:
        if project_id:
            raise ProfileError(f"Project not found: {project_id}")
        else:
            raise ProfileError("No active project")
    
    with _profile_lock:
        # Check if we already have a profile manager for this project
        if project.project_id in _profile_managers:
            return _profile_managers[project.project_id]
        
        # Create a new profile manager
        manager = ProfileManager(project)
        _profile_managers[project.project_id] = manager
        
        # Load profiles
        manager.load_profiles()
        
        return manager


def get_active_profile(project_id: Optional[str] = None) -> Optional[ProjectProfile]:
    """
    Get the active profile for a project.
    
    Args:
        project_id: ID of the project (or None for active project)
        
    Returns:
        Active profile or None if no active profile
        
    Raises:
        ProfileError: If no project is found or active
    """
    # Get the profile manager
    manager = get_profile_manager(project_id)
    
    # Get the active profile
    return manager.get_active_profile()


def create_profile(name: str, description: Optional[str] = None, 
                 project_id: Optional[str] = None) -> ProjectProfile:
    """
    Create a new profile for a project.
    
    Args:
        name: Name for the profile
        description: Optional description
        project_id: ID of the project (or None for active project)
        
    Returns:
        New profile
        
    Raises:
        ProfileError: If no project is found or active
    """
    # Get the profile manager
    manager = get_profile_manager(project_id)
    
    # Create a new profile
    return manager.create_profile(name, description)


def set_active_profile(profile_id: str, project_id: Optional[str] = None) -> bool:
    """
    Set the active profile for a project.
    
    Args:
        profile_id: ID of the profile
        project_id: ID of the project (or None for active project)
        
    Returns:
        True if active profile was set, False if profile not found
        
    Raises:
        ProfileError: If no project is found or active
    """
    # Get the profile manager
    manager = get_profile_manager(project_id)
    
    # Set the active profile
    return manager.set_active_profile(profile_id)


def export_profile(profile_id: str, file_path: str, 
                 project_id: Optional[str] = None) -> bool:
    """
    Export a profile to a file.
    
    Args:
        profile_id: ID of the profile
        file_path: Path to export to
        project_id: ID of the project (or None for active project)
        
    Returns:
        True if profile was exported, False if profile not found
        
    Raises:
        ProfileError: If no project is found or active
    """
    # Get the profile manager
    manager = get_profile_manager(project_id)
    
    # Export the profile
    return manager.export_profile(profile_id, file_path)


def import_profile(file_path: str, 
                 project_id: Optional[str] = None) -> Optional[ProjectProfile]:
    """
    Import a profile from a file.
    
    Args:
        file_path: Path to import from
        project_id: ID of the project (or None for active project)
        
    Returns:
        Imported profile or None on error
        
    Raises:
        ProfileError: If no project is found or active
    """
    # Get the profile manager
    manager = get_profile_manager(project_id)
    
    # Import the profile
    return manager.import_profile(file_path)