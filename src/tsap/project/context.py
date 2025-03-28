"""
Project context management for TSAP.

This module provides classes and functions for managing project-specific
settings, loaded data, active analyses, and other contextual information
that should persist during a project session.
"""

import os
import json
import time
import uuid
import threading
import contextlib
from typing import Dict, List, Any, Optional, Generator

import tsap.utils.logging as logging
from tsap.utils.errors import TSAPError


class ProjectContextError(TSAPError):
    """Exception raised for project context errors."""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.details = details or {}


class ProjectContext:
    """
    Manages context for a TSAP project session.
    
    This class stores and provides access to project-specific settings,
    loaded data, active analyses, etc.
    """
    
    def __init__(self, project_id: Optional[str] = None, name: Optional[str] = None):
        """
        Initialize a new project context.
        
        Args:
            project_id: Unique ID for the project (generated if not provided)
            name: Name for the project
        """
        self.project_id = project_id or str(uuid.uuid4())
        self.name = name or f"Project-{self.project_id[:8]}"
        self.created_at = time.time()
        self.updated_at = time.time()
        
        # File paths and directories
        self.root_directory: Optional[str] = None
        self.config_file: Optional[str] = None
        self.output_directory: Optional[str] = None
        
        # Data storage
        self._data: Dict[str, Any] = {}
        self._files: Dict[str, str] = {}  # Map of file IDs to file paths
        self._analyses: Dict[str, Dict] = {}
        self._active_operations: Dict[str, Dict] = {}
        
        # Settings and configuration
        self._settings: Dict[str, Any] = {}
        
        # State flags
        self.is_active = True
        self.is_modified = False
        
        # Locking for thread safety
        self._lock = threading.RLock()
    
    def set_directories(self, root_dir: Optional[str] = None, 
                       output_dir: Optional[str] = None) -> None:
        """
        Set project directories.
        
        Args:
            root_dir: Root directory for project
            output_dir: Output directory for results
        """
        with self._lock:
            if root_dir:
                self.root_directory = os.path.abspath(root_dir)
            
            if output_dir:
                self.output_directory = os.path.abspath(output_dir)
            elif root_dir and not self.output_directory:
                # Default output directory inside root
                self.output_directory = os.path.join(self.root_directory, "output")
            
            # Create output directory if it doesn't exist
            if self.output_directory and not os.path.exists(self.output_directory):
                try:
                    os.makedirs(self.output_directory, exist_ok=True)
                    logging.debug(f"Created output directory: {self.output_directory}", 
                                component="project")
                except Exception as e:
                    logging.warning(f"Failed to create output directory: {str(e)}", 
                                  component="project")
            
            self.is_modified = True
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Get a project setting.
        
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
        Set a project setting.
        
        Args:
            key: Setting key
            value: Setting value
        """
        with self._lock:
            self._settings[key] = value
            self.updated_at = time.time()
            self.is_modified = True
    
    def get_data(self, key: str, default: Any = None) -> Any:
        """
        Get stored data by key.
        
        Args:
            key: Data key
            default: Default value if data is not found
            
        Returns:
            Stored data or default
        """
        with self._lock:
            return self._data.get(key, default)
    
    def set_data(self, key: str, value: Any) -> None:
        """
        Store data by key.
        
        Args:
            key: Data key
            value: Data value
        """
        with self._lock:
            self._data[key] = value
            self.updated_at = time.time()
            self.is_modified = True
    
    def list_data_keys(self) -> List[str]:
        """
        List all data keys.
        
        Returns:
            List of data keys
        """
        with self._lock:
            return list(self._data.keys())
    
    def delete_data(self, key: str) -> bool:
        """
        Delete data by key.
        
        Args:
            key: Data key
            
        Returns:
            True if data was deleted, False if key not found
        """
        with self._lock:
            if key in self._data:
                del self._data[key]
                self.updated_at = time.time()
                self.is_modified = True
                return True
            return False
    
    def register_file(self, file_path: str, file_id: Optional[str] = None) -> str:
        """
        Register a file with the project.
        
        Args:
            file_path: Path to the file
            file_id: Optional ID for the file (generated if not provided)
            
        Returns:
            ID of the registered file
        """
        with self._lock:
            # Generate ID if not provided
            if file_id is None:
                file_id = str(uuid.uuid4())
            
            # Store absolute path
            abs_path = os.path.abspath(file_path)
            
            self._files[file_id] = abs_path
            self.updated_at = time.time()
            self.is_modified = True
            
            logging.debug(f"Registered file {file_id}: {abs_path}", component="project")
            return file_id
    
    def get_file_path(self, file_id: str) -> Optional[str]:
        """
        Get file path by ID.
        
        Args:
            file_id: ID of the file
            
        Returns:
            Path to the file or None if not found
        """
        with self._lock:
            return self._files.get(file_id)
    
    def list_files(self) -> Dict[str, str]:
        """
        List all registered files.
        
        Returns:
            Dictionary mapping file IDs to file paths
        """
        with self._lock:
            return self._files.copy()
    
    def unregister_file(self, file_id: str) -> bool:
        """
        Unregister a file from the project.
        
        Args:
            file_id: ID of the file
            
        Returns:
            True if file was unregistered, False if not found
        """
        with self._lock:
            if file_id in self._files:
                del self._files[file_id]
                self.updated_at = time.time()
                self.is_modified = True
                return True
            return False
    
    def register_analysis(self, analysis_id: str, analysis_type: str, 
                        parameters: Dict[str, Any], 
                        result_path: Optional[str] = None) -> None:
        """
        Register an analysis with the project.
        
        Args:
            analysis_id: ID of the analysis
            analysis_type: Type of analysis
            parameters: Parameters used for the analysis
            result_path: Optional path to result file
        """
        with self._lock:
            self._analyses[analysis_id] = {
                'id': analysis_id,
                'type': analysis_type,
                'parameters': parameters,
                'result_path': result_path,
                'created_at': time.time()
            }
            self.updated_at = time.time()
            self.is_modified = True
    
    def get_analysis(self, analysis_id: str) -> Optional[Dict]:
        """
        Get analysis by ID.
        
        Args:
            analysis_id: ID of the analysis
            
        Returns:
            Analysis data or None if not found
        """
        with self._lock:
            return self._analyses.get(analysis_id)
    
    def list_analyses(self, analysis_type: Optional[str] = None) -> List[Dict]:
        """
        List analyses, optionally filtered by type.
        
        Args:
            analysis_type: Optional type to filter by
            
        Returns:
            List of analysis data
        """
        with self._lock:
            if analysis_type:
                return [a for a in self._analyses.values() if a['type'] == analysis_type]
            return list(self._analyses.values())
    
    def update_analysis(self, analysis_id: str, result_path: Optional[str] = None, 
                       **updates) -> bool:
        """
        Update an analysis.
        
        Args:
            analysis_id: ID of the analysis
            result_path: Optional new result path
            **updates: Additional updates to apply
            
        Returns:
            True if analysis was updated, False if not found
        """
        with self._lock:
            if analysis_id in self._analyses:
                if result_path is not None:
                    self._analyses[analysis_id]['result_path'] = result_path
                
                self._analyses[analysis_id].update(updates)
                self.updated_at = time.time()
                self.is_modified = True
                return True
            return False
    
    def delete_analysis(self, analysis_id: str) -> bool:
        """
        Delete an analysis.
        
        Args:
            analysis_id: ID of the analysis
            
        Returns:
            True if analysis was deleted, False if not found
        """
        with self._lock:
            if analysis_id in self._analyses:
                del self._analyses[analysis_id]
                self.updated_at = time.time()
                self.is_modified = True
                return True
            return False
    
    def register_operation(self, operation_id: str, operation_type: str, 
                         status: str = "pending",
                         parameters: Optional[Dict[str, Any]] = None) -> None:
        """
        Register an active operation.
        
        Args:
            operation_id: ID of the operation
            operation_type: Type of operation
            status: Initial status
            parameters: Operation parameters
        """
        with self._lock:
            self._active_operations[operation_id] = {
                'id': operation_id,
                'type': operation_type,
                'status': status,
                'parameters': parameters or {},
                'created_at': time.time(),
                'updated_at': time.time()
            }
            self.updated_at = time.time()
            self.is_modified = True
    
    def update_operation_status(self, operation_id: str, status: str, 
                              result: Optional[Any] = None) -> bool:
        """
        Update operation status.
        
        Args:
            operation_id: ID of the operation
            status: New status
            result: Optional operation result
            
        Returns:
            True if operation was updated, False if not found
        """
        with self._lock:
            if operation_id in self._active_operations:
                self._active_operations[operation_id]['status'] = status
                self._active_operations[operation_id]['updated_at'] = time.time()
                
                if result is not None:
                    self._active_operations[operation_id]['result'] = result
                
                self.updated_at = time.time()
                self.is_modified = True
                return True
            return False
    
    def get_operation(self, operation_id: str) -> Optional[Dict]:
        """
        Get operation by ID.
        
        Args:
            operation_id: ID of the operation
            
        Returns:
            Operation data or None if not found
        """
        with self._lock:
            return self._active_operations.get(operation_id)
    
    def list_operations(self, status: Optional[str] = None) -> List[Dict]:
        """
        List operations, optionally filtered by status.
        
        Args:
            status: Optional status to filter by
            
        Returns:
            List of operation data
        """
        with self._lock:
            if status:
                return [op for op in self._active_operations.values() 
                       if op['status'] == status]
            return list(self._active_operations.values())
    
    def cleanup_operations(self, max_age: Optional[float] = None, 
                         statuses: Optional[List[str]] = None) -> int:
        """
        Cleanup old or completed operations.
        
        Args:
            max_age: Maximum age in seconds
            statuses: List of statuses to clean up
            
        Returns:
            Number of operations cleaned up
        """
        with self._lock:
            if not max_age and not statuses:
                return 0
            
            current_time = time.time()
            to_remove = []
            
            for op_id, op in self._active_operations.items():
                age = current_time - op.get('updated_at', op.get('created_at', current_time))
                
                if (max_age and age >= max_age) or (statuses and op['status'] in statuses):
                    to_remove.append(op_id)
            
            for op_id in to_remove:
                del self._active_operations[op_id]
            
            if to_remove:
                self.updated_at = time.time()
                self.is_modified = True
            
            return len(to_remove)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert project context to a dictionary.
        
        Returns:
            Dictionary representation of the project context
        """
        with self._lock:
            return {
                'project_id': self.project_id,
                'name': self.name,
                'created_at': self.created_at,
                'updated_at': self.updated_at,
                'root_directory': self.root_directory,
                'output_directory': self.output_directory,
                'config_file': self.config_file,
                'settings': self._settings,
                'files': self._files,
                'analyses': self._analyses,
                'active_operations': self._active_operations
            }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProjectContext':
        """
        Create a project context from a dictionary.
        
        Args:
            data: Dictionary representing a project context
            
        Returns:
            New ProjectContext instance
        """
        context = cls(project_id=data.get('project_id'), name=data.get('name'))
        context.created_at = data.get('created_at', context.created_at)
        context.updated_at = data.get('updated_at', context.updated_at)
        context.root_directory = data.get('root_directory')
        context.output_directory = data.get('output_directory')
        context.config_file = data.get('config_file')
        context._settings = data.get('settings', {})
        context._files = data.get('files', {})
        context._analyses = data.get('analyses', {})
        context._active_operations = data.get('active_operations', {})
        return context
    
    def save(self, path: Optional[str] = None) -> str:
        """
        Save project context to file.
        
        Args:
            path: Path to save to (default: output_directory/project_id.json)
            
        Returns:
            Path to saved file
        """
        with self._lock:
            # Determine save path
            if not path:
                if not self.output_directory:
                    raise ProjectContextError("No output directory set and no path provided")
                
                # Ensure output directory exists
                os.makedirs(self.output_directory, exist_ok=True)
                path = os.path.join(self.output_directory, f"{self.project_id}.json")
            
            try:
                # Convert to dictionary
                data = self.to_dict()
                
                # Save to file
                with open(path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                self.is_modified = False
                logging.debug(f"Saved project context to {path}", component="project")
                return path
                
            except Exception as e:
                raise ProjectContextError(f"Failed to save project context: {str(e)}")
    
    @classmethod
    def load(cls, path: str) -> 'ProjectContext':
        """
        Load project context from file.
        
        Args:
            path: Path to load from
            
        Returns:
            Loaded ProjectContext instance
        """
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            context = cls.from_dict(data)
            logging.debug(f"Loaded project context from {path}", component="project")
            return context
            
        except Exception as e:
            raise ProjectContextError(f"Failed to load project context: {str(e)}")


# Global project context registry
class ProjectRegistry:
    """Registry for managing multiple project contexts."""
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ProjectRegistry, cls).__new__(cls)
                cls._instance._projects = {}
                cls._instance._active_project_id = None
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        with self._lock:
            if not self._initialized:
                self._projects = {}
                self._active_project_id = None
                self._initialized = True
    
    def register_project(self, project: ProjectContext) -> None:
        """
        Register a project with the registry.
        
        Args:
            project: Project context to register
        """
        with self._lock:
            self._projects[project.project_id] = project
            logging.debug(f"Registered project {project.project_id}: {project.name}", 
                        component="project")
    
    def get_project(self, project_id: str) -> Optional[ProjectContext]:
        """
        Get a project by ID.
        
        Args:
            project_id: ID of the project
            
        Returns:
            Project context or None if not found
        """
        with self._lock:
            return self._projects.get(project_id)
    
    def list_projects(self) -> List[Dict[str, Any]]:
        """
        List all registered projects.
        
        Returns:
            List of project metadata
        """
        with self._lock:
            return [
                {
                    'project_id': p.project_id,
                    'name': p.name,
                    'created_at': p.created_at,
                    'updated_at': p.updated_at,
                    'root_directory': p.root_directory,
                    'is_active': p.project_id == self._active_project_id
                }
                for p in self._projects.values()
            ]
    
    def unregister_project(self, project_id: str) -> bool:
        """
        Unregister a project.
        
        Args:
            project_id: ID of the project
            
        Returns:
            True if project was unregistered, False if not found
        """
        with self._lock:
            if project_id in self._projects:
                if project_id == self._active_project_id:
                    self._active_project_id = None
                
                del self._projects[project_id]
                logging.debug(f"Unregistered project {project_id}", component="project")
                return True
            return False
    
    def set_active_project(self, project_id: str) -> bool:
        """
        Set the active project.
        
        Args:
            project_id: ID of the project
            
        Returns:
            True if active project was set, False if project not found
        """
        with self._lock:
            if project_id in self._projects:
                old_active = self._active_project_id
                self._active_project_id = project_id
                
                if old_active != project_id:
                    logging.debug(f"Set active project to {project_id}", component="project")
                
                return True
            return False
    
    def get_active_project(self) -> Optional[ProjectContext]:
        """
        Get the active project.
        
        Returns:
            Active project context or None if no active project
        """
        with self._lock:
            if self._active_project_id:
                return self._projects.get(self._active_project_id)
            return None
    
    def save_all_projects(self, directory: Optional[str] = None) -> Dict[str, str]:
        """
        Save all projects.
        
        Args:
            directory: Directory to save to (defaults to each project's output directory)
            
        Returns:
            Dictionary mapping project IDs to save paths
        """
        with self._lock:
            results = {}
            
            for project_id, project in self._projects.items():
                if project.is_modified:
                    try:
                        path = None
                        if directory:
                            path = os.path.join(directory, f"{project_id}.json")
                        
                        save_path = project.save(path)
                        results[project_id] = save_path
                    except Exception as e:
                        logging.warning(f"Failed to save project {project_id}: {str(e)}", 
                                      component="project")
            
            return results
    
    def load_projects(self, directory: str) -> Dict[str, bool]:
        """
        Load projects from a directory.
        
        Args:
            directory: Directory to load from
            
        Returns:
            Dictionary mapping project IDs to success status
        """
        with self._lock:
            results = {}
            
            if not os.path.isdir(directory):
                logging.warning(f"Project directory does not exist: {directory}", 
                              component="project")
                return results
            
            for filename in os.listdir(directory):
                if filename.endswith('.json'):
                    path = os.path.join(directory, filename)
                    
                    try:
                        project = ProjectContext.load(path)
                        self.register_project(project)
                        results[project.project_id] = True
                    except Exception as e:
                        project_id = filename.rsplit('.', 1)[0]
                        logging.warning(f"Failed to load project {project_id}: {str(e)}", 
                                      component="project")
                        results[project_id] = False
            
            return results


# Global registry instance
_project_registry = ProjectRegistry()


def get_project_registry() -> ProjectRegistry:
    """
    Get the global project registry.
    
    Returns:
        Global ProjectRegistry instance
    """
    return _project_registry


def create_project(name: Optional[str] = None, 
                 root_directory: Optional[str] = None) -> ProjectContext:
    """
    Create a new project.
    
    Args:
        name: Name for the project
        root_directory: Root directory for the project
        
    Returns:
        New ProjectContext instance
    """
    project = ProjectContext(name=name)
    
    if root_directory:
        project.set_directories(root_directory)
    
    # Register with global registry
    registry = get_project_registry()
    registry.register_project(project)
    
    # Set as active project if no other project is active
    if registry.get_active_project() is None:
        registry.set_active_project(project.project_id)
    
    return project


def get_project(project_id: Optional[str] = None) -> Optional[ProjectContext]:
    """
    Get a project by ID or the active project.
    
    Args:
        project_id: ID of the project (or None for active project)
        
    Returns:
        Project context or None if not found
    """
    registry = get_project_registry()
    
    if project_id:
        return registry.get_project(project_id)
    else:
        return registry.get_active_project()


def set_active_project(project_id: str) -> bool:
    """
    Set the active project.
    
    Args:
        project_id: ID of the project
        
    Returns:
        True if active project was set, False if project not found
    """
    registry = get_project_registry()
    return registry.set_active_project(project_id)


@contextlib.contextmanager
def project_context(project_id: Optional[str] = None) -> Generator[ProjectContext, None, None]:
    """
    Context manager for working with a project.
    
    Args:
        project_id: ID of the project (or None for active project)
        
    Yields:
        Project context
        
    Raises:
        ProjectContextError: If no project is found or active
    """
    project = get_project(project_id)
    
    if not project:
        if project_id:
            raise ProjectContextError(f"Project not found: {project_id}")
        else:
            raise ProjectContextError("No active project")
    
    try:
        yield project
    finally:
        pass  # Auto-save could be implemented here


def initialize_project_system(projects_directory: Optional[str] = None) -> None:
    """
    Initialize the project system.
    
    Args:
        projects_directory: Directory to load projects from
    """
    if projects_directory:
        # Load projects from directory
        registry = get_project_registry()
        registry.load_projects(projects_directory)
        
        # Log results
        projects = registry.list_projects()
        loaded_count = len(projects)
        
        if loaded_count > 0:
            logging.info(f"Loaded {loaded_count} projects from {projects_directory}", 
                       component="project")
        else:
            logging.debug(f"No projects found in {projects_directory}", component="project")