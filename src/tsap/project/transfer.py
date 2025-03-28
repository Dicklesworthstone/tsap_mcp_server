"""
Project data transfer functionality for TSAP.

This module provides functions and classes for importing, exporting, and
transferring project data, results, and configurations between projects
or to external formats.
"""

import os
import re
import json
import zipfile
import shutil
import tempfile
import time
import threading
from typing import Dict, List, Any, Optional

import tsap.utils.logging as logging
from tsap.utils.errors import TSAPError
from tsap.project.context import ProjectContext
from tsap.project.profile import get_profile_manager


class TransferError(TSAPError):
    """Exception raised for data transfer errors."""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.details = details or {}


class ProjectExport:
    """
    Handles exporting project data to various formats.
    
    This class provides methods for exporting project data, configurations,
    results, and other project artifacts to various formats like ZIP, JSON, etc.
    """
    
    def __init__(self, project: ProjectContext):
        """
        Initialize project export for a project.
        
        Args:
            project: Project context
        """
        self.project = project
        self._lock = threading.RLock()
    
    def export_to_zip(self, output_path: str, 
                     include_profiles: bool = True,
                     include_history: bool = True,
                     include_results: bool = True,
                     include_files: bool = True,
                     password: Optional[str] = None) -> str:
        """
        Export project data to a ZIP archive.
        
        Args:
            output_path: Path for the output ZIP file
            include_profiles: Whether to include project profiles
            include_history: Whether to include command history
            include_results: Whether to include analysis results
            include_files: Whether to include registered files
            password: Optional password to encrypt the archive
            
        Returns:
            Path to the created ZIP file
            
        Raises:
            TransferError: If export fails
        """
        with self._lock:
            temp_dir = None
            try:
                # Create a temporary directory to organize files
                temp_dir = tempfile.mkdtemp(prefix="tsap_export_")
                
                # Create project manifest
                manifest = {
                    'project_id': self.project.project_id,
                    'name': self.project.name,
                    'created_at': self.project.created_at,
                    'updated_at': self.project.updated_at,
                    'version': '1.0',
                    'timestamp': time.time(),
                    'contents': {}
                }
                
                # Create project context export
                context_path = os.path.join(temp_dir, "project.json")
                with open(context_path, 'w') as f:
                    json.dump(self.project.to_dict(), f, indent=2)
                manifest['contents']['project'] = "project.json"
                
                # Create profiles export if requested
                if include_profiles and self.project.output_directory:
                    profiles_dir = os.path.join(self.project.output_directory, "profiles")
                    if os.path.exists(profiles_dir):
                        profile_export_dir = os.path.join(temp_dir, "profiles")
                        os.makedirs(profile_export_dir, exist_ok=True)
                        
                        # Copy all profile files
                        for filename in os.listdir(profiles_dir):
                            if filename.endswith('.json'):
                                src_path = os.path.join(profiles_dir, filename)
                                dst_path = os.path.join(profile_export_dir, filename)
                                shutil.copy2(src_path, dst_path)
                        
                        manifest['contents']['profiles'] = "profiles/"
                        
                        # Save profile manager state
                        profile_manager = get_profile_manager(self.project.project_id)
                        manager_state = {
                            'active_profile_id': profile_manager._active_profile_id,
                            'default_profile_id': profile_manager._default_profile_id
                        }
                        with open(os.path.join(profile_export_dir, "_manager.json"), 'w') as f:
                            json.dump(manager_state, f, indent=2)
                
                # Create history export if requested
                if include_history and self.project.output_directory:
                    history_db = os.path.join(
                        self.project.output_directory, 
                        f"{self.project.project_id}_history.db"
                    )
                    if os.path.exists(history_db):
                        history_export_path = os.path.join(temp_dir, "history.db")
                        shutil.copy2(history_db, history_export_path)
                        manifest['contents']['history'] = "history.db"
                
                # Export results if requested
                if include_results and self.project.output_directory:
                    results_dir = os.path.join(self.project.output_directory, "results")
                    if os.path.exists(results_dir):
                        results_export_dir = os.path.join(temp_dir, "results")
                        shutil.copytree(results_dir, results_export_dir)
                        manifest['contents']['results'] = "results/"
                
                # Export registered files if requested
                if include_files:
                    files = self.project.list_files()
                    if files:
                        files_export_dir = os.path.join(temp_dir, "files")
                        os.makedirs(files_export_dir, exist_ok=True)
                        
                        # Create files registry
                        files_registry = {}
                        
                        for file_id, file_path in files.items():
                            if os.path.exists(file_path):
                                # Create a safe filename
                                filename = os.path.basename(file_path)
                                safe_name = re.sub(r'[^\w\.-]', '_', filename)
                                
                                # Ensure uniqueness by adding ID as prefix if needed
                                safe_name = f"{file_id}_{safe_name}"
                                
                                # Copy the file
                                dst_path = os.path.join(files_export_dir, safe_name)
                                shutil.copy2(file_path, dst_path)
                                
                                # Register in files registry
                                files_registry[file_id] = {
                                    'original_path': file_path,
                                    'export_path': f"files/{safe_name}"
                                }
                        
                        # Save files registry
                        with open(os.path.join(files_export_dir, "_registry.json"), 'w') as f:
                            json.dump(files_registry, f, indent=2)
                            
                        manifest['contents']['files'] = "files/"
                
                # Write manifest
                manifest_path = os.path.join(temp_dir, "manifest.json")
                with open(manifest_path, 'w') as f:
                    json.dump(manifest, f, indent=2)
                
                # Create the ZIP archive
                if os.path.exists(output_path):
                    os.remove(output_path)
                
                if password:
                    # Use zipfile with password
                    import pyzipper  # Optional dependency
                    
                    with pyzipper.AESZipFile(
                        output_path, 
                        'w', 
                        compression=pyzipper.ZIP_LZMA,
                        encryption=pyzipper.WZ_AES
                    ) as zipf:
                        zipf.setpassword(password.encode())
                        
                        # Add all files in temp_dir to the ZIP
                        for root, _, files in os.walk(temp_dir):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, temp_dir)
                                zipf.write(file_path, arcname)
                else:
                    # Use standard zipfile
                    with zipfile.ZipFile(
                        output_path, 
                        'w', 
                        compression=zipfile.ZIP_DEFLATED
                    ) as zipf:
                        # Add all files in temp_dir to the ZIP
                        for root, _, files in os.walk(temp_dir):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, temp_dir)
                                zipf.write(file_path, arcname)
                
                logging.info(f"Exported project {self.project.project_id} to {output_path}", 
                           component="transfer")
                
                return output_path
            
            except Exception as e:
                raise TransferError(f"Failed to export project: {str(e)}")
            
            finally:
                # Clean up temporary directory
                if temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
    
    def export_to_json(self, output_path: str) -> str:
        """
        Export project metadata to a JSON file.
        
        This exports only the project context and basic metadata,
        not the full project contents.
        
        Args:
            output_path: Path for the output JSON file
            
        Returns:
            Path to the created JSON file
            
        Raises:
            TransferError: If export fails
        """
        with self._lock:
            try:
                # Create project export data
                export_data = {
                    'project_id': self.project.project_id,
                    'name': self.project.name,
                    'created_at': self.project.created_at,
                    'updated_at': self.project.updated_at,
                    'version': '1.0',
                    'timestamp': time.time(),
                    'context': self.project.to_dict(),
                    'file_count': len(self.project.list_files())
                }
                
                # Write JSON file
                with open(output_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                logging.info(f"Exported project metadata to {output_path}", 
                           component="transfer")
                
                return output_path
            
            except Exception as e:
                raise TransferError(f"Failed to export project metadata: {str(e)}")
    
    def export_results(self, output_dir: str, 
                      result_types: Optional[List[str]] = None) -> str:
        """
        Export analysis results to a directory.
        
        Args:
            output_dir: Directory to export results to
            result_types: Optional list of result types to export (None for all)
            
        Returns:
            Path to the export directory
            
        Raises:
            TransferError: If export fails
        """
        with self._lock:
            try:
                # Check if project has results
                if not self.project.output_directory:
                    raise TransferError("Project has no output directory")
                
                results_dir = os.path.join(self.project.output_directory, "results")
                if not os.path.exists(results_dir):
                    raise TransferError("Project has no results directory")
                
                # Create output directory
                os.makedirs(output_dir, exist_ok=True)
                
                # Export results
                results_count = 0
                
                for root, dirs, files in os.walk(results_dir):
                    # Determine result type from directory path
                    rel_path = os.path.relpath(root, results_dir)
                    if rel_path == '.':
                        result_type = None
                    else:
                        result_type = rel_path.split(os.sep)[0]
                    
                    # Skip if not in requested types
                    if result_types and result_type and result_type not in result_types:
                        continue
                    
                    # Create corresponding directory in output
                    rel_output_dir = os.path.join(output_dir, rel_path)
                    os.makedirs(rel_output_dir, exist_ok=True)
                    
                    # Copy files
                    for file in files:
                        src_path = os.path.join(root, file)
                        dst_path = os.path.join(rel_output_dir, file)
                        shutil.copy2(src_path, dst_path)
                        results_count += 1
                
                # Create metadata file
                metadata = {
                    'project_id': self.project.project_id,
                    'name': self.project.name,
                    'exported_at': time.time(),
                    'result_types': result_types,
                    'results_count': results_count
                }
                
                with open(os.path.join(output_dir, "export_metadata.json"), 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logging.info(f"Exported {results_count} results to {output_dir}", 
                           component="transfer")
                
                return output_dir
            
            except Exception as e:
                raise TransferError(f"Failed to export results: {str(e)}")
    
    def create_backup(self, output_path: Optional[str] = None) -> str:
        """
        Create a complete backup of the project.
        
        Args:
            output_path: Optional path for the backup file
                (default: <project_output_dir>/<project_id>_backup_<timestamp>.zip)
            
        Returns:
            Path to the created backup file
            
        Raises:
            TransferError: If backup fails
        """
        with self._lock:
            try:
                # Determine backup path
                if not output_path:
                    if not self.project.output_directory:
                        raise TransferError("Project has no output directory and no output path provided")
                    
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    output_path = os.path.join(
                        self.project.output_directory,
                        f"{self.project.project_id}_backup_{timestamp}.zip"
                    )
                
                # Create a complete export
                return self.export_to_zip(
                    output_path=output_path,
                    include_profiles=True,
                    include_history=True,
                    include_results=True,
                    include_files=True
                )
            
            except Exception as e:
                raise TransferError(f"Failed to create backup: {str(e)}")


class ProjectImport:
    """
    Handles importing project data from various formats.
    
    This class provides methods for importing project data, configurations,
    results, and other project artifacts from various formats like ZIP, JSON, etc.
    """
    
    def __init__(self, existing_project: Optional[ProjectContext] = None):
        """
        Initialize project import.
        
        Args:
            existing_project: Optional existing project to import into
        """
        self.existing_project = existing_project
        self._lock = threading.RLock()
    
    def import_from_zip(self, zip_path: str, 
                       import_profiles: bool = True,
                       import_history: bool = True,
                       import_results: bool = True,
                       import_files: bool = True,
                       password: Optional[str] = None) -> ProjectContext:
        """
        Import project data from a ZIP archive.
        
        Args:
            zip_path: Path to the ZIP file
            import_profiles: Whether to import project profiles
            import_history: Whether to import command history
            import_results: Whether to import analysis results
            import_files: Whether to import registered files
            password: Optional password to decrypt the archive
            
        Returns:
            Imported or updated project context
            
        Raises:
            TransferError: If import fails
        """
        with self._lock:
            temp_dir = None
            try:
                # Create a temporary directory to extract files
                temp_dir = tempfile.mkdtemp(prefix="tsap_import_")
                
                # Extract the ZIP archive
                if password:
                    # Use zipfile with password
                    import pyzipper  # Optional dependency
                    
                    with pyzipper.AESZipFile(zip_path, 'r') as zipf:
                        zipf.setpassword(password.encode())
                        zipf.extractall(temp_dir)
                else:
                    # Use standard zipfile
                    with zipfile.ZipFile(zip_path, 'r') as zipf:
                        zipf.extractall(temp_dir)
                
                # Check if manifest exists
                manifest_path = os.path.join(temp_dir, "manifest.json")
                if not os.path.exists(manifest_path):
                    raise TransferError("Invalid project archive: manifest.json not found")
                
                # Load manifest
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                
                # Check if project context exists
                project_path = os.path.join(temp_dir, manifest['contents'].get('project', 'project.json'))
                if not os.path.exists(project_path):
                    raise TransferError("Invalid project archive: project.json not found")
                
                # Load project context
                with open(project_path, 'r') as f:
                    project_data = json.load(f)
                
                # Create or update project
                if self.existing_project:
                    # Update existing project
                    project = self.existing_project
                    
                    # Merge project data
                    self._merge_project_data(project, project_data)
                else:
                    # Create new project
                    from tsap.project.context import create_project
                    
                    # Create with imported ID and name
                    project = create_project(
                        name=project_data.get('name'),
                    )
                    
                    # Replace project ID with the imported one
                    orig_project_id = project.project_id
                    project.project_id = project_data.get('project_id')
                    
                    # Update project registry
                    from tsap.project.context import get_project_registry
                    registry = get_project_registry()
                    if orig_project_id in registry._projects:
                        del registry._projects[orig_project_id]
                    registry._projects[project.project_id] = project
                    
                    # Set as active project
                    registry.set_active_project(project.project_id)
                    
                    # Merge project data
                    self._merge_project_data(project, project_data)
                
                # Import profiles if requested
                if import_profiles and 'profiles' in manifest['contents']:
                    profiles_dir = os.path.join(temp_dir, manifest['contents']['profiles'])
                    if os.path.exists(profiles_dir):
                        # Create profiles directory in project if needed
                        if project.output_directory:
                            project_profiles_dir = os.path.join(project.output_directory, "profiles")
                            os.makedirs(project_profiles_dir, exist_ok=True)
                            
                            # Copy all profile files
                            for filename in os.listdir(profiles_dir):
                                if filename.endswith('.json') and filename != "_manager.json":
                                    src_path = os.path.join(profiles_dir, filename)
                                    dst_path = os.path.join(project_profiles_dir, filename)
                                    shutil.copy2(src_path, dst_path)
                            
                            # Load profile manager state if available
                            manager_path = os.path.join(profiles_dir, "_manager.json")
                            if os.path.exists(manager_path):
                                with open(manager_path, 'r') as f:
                                    manager_state = json.load(f)
                                
                                # Apply profile manager state
                                profile_manager = get_profile_manager(project.project_id)
                                profile_manager.load_profiles()
                                
                                # Set active and default profiles
                                if 'active_profile_id' in manager_state:
                                    profile_manager.set_active_profile(manager_state['active_profile_id'])
                                if 'default_profile_id' in manager_state:
                                    profile_manager.set_default_profile(manager_state['default_profile_id'])
                
                # Import history if requested
                if import_history and 'history' in manifest['contents']:
                    history_path = os.path.join(temp_dir, manifest['contents']['history'])
                    if os.path.exists(history_path) and project.output_directory:
                        dst_path = os.path.join(
                            project.output_directory,
                            f"{project.project_id}_history.db"
                        )
                        shutil.copy2(history_path, dst_path)
                
                # Import results if requested
                if import_results and 'results' in manifest['contents']:
                    results_dir = os.path.join(temp_dir, manifest['contents']['results'])
                    if os.path.exists(results_dir) and project.output_directory:
                        dst_dir = os.path.join(project.output_directory, "results")
                        
                        # Clear existing results if any
                        if os.path.exists(dst_dir):
                            shutil.rmtree(dst_dir)
                        
                        # Copy results
                        shutil.copytree(results_dir, dst_dir)
                
                # Import files if requested
                if import_files and 'files' in manifest['contents']:
                    files_dir = os.path.join(temp_dir, manifest['contents']['files'])
                    if os.path.exists(files_dir):
                        # Load files registry
                        registry_path = os.path.join(files_dir, "_registry.json")
                        if os.path.exists(registry_path):
                            with open(registry_path, 'r') as f:
                                files_registry = json.load(f)
                            
                            # Import each file
                            for file_id, file_info in files_registry.items():
                                export_path = file_info.get('export_path')
                                if export_path:
                                    src_path = os.path.join(temp_dir, export_path)
                                    
                                    if os.path.exists(src_path):
                                        # Determine target path
                                        if project.output_directory:
                                            # Store in project files directory
                                            files_dir = os.path.join(project.output_directory, "files")
                                            os.makedirs(files_dir, exist_ok=True)
                                            
                                            # Use original filename if possible
                                            filename = os.path.basename(file_info.get('original_path', src_path))
                                            dst_path = os.path.join(files_dir, filename)
                                            
                                            # Ensure uniqueness
                                            if os.path.exists(dst_path):
                                                name, ext = os.path.splitext(filename)
                                                dst_path = os.path.join(files_dir, f"{name}_{file_id}{ext}")
                                            
                                            # Copy the file
                                            shutil.copy2(src_path, dst_path)
                                            
                                            # Register the file with its original ID
                                            project.register_file(dst_path, file_id)
                
                logging.info(f"Imported project {project.project_id} from {zip_path}", 
                           component="transfer")
                
                return project
            
            except Exception as e:
                raise TransferError(f"Failed to import project: {str(e)}")
            
            finally:
                # Clean up temporary directory
                if temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
    
    def import_from_json(self, json_path: str) -> ProjectContext:
        """
        Import project metadata from a JSON file.
        
        This imports only the project context and basic metadata,
        not the full project contents.
        
        Args:
            json_path: Path to the JSON file
            
        Returns:
            Imported or updated project context
            
        Raises:
            TransferError: If import fails
        """
        with self._lock:
            try:
                # Load JSON file
                with open(json_path, 'r') as f:
                    export_data = json.load(f)
                
                # Get project context data
                project_data = export_data.get('context')
                if not project_data:
                    raise TransferError("Invalid project JSON: context data not found")
                
                # Create or update project
                if self.existing_project:
                    # Update existing project
                    project = self.existing_project
                    
                    # Merge project data
                    self._merge_project_data(project, project_data)
                else:
                    # Create new project
                    from tsap.project.context import create_project
                    
                    # Create with imported ID and name
                    project = create_project(
                        name=export_data.get('name'),
                    )
                    
                    # Replace project ID with the imported one
                    orig_project_id = project.project_id
                    project.project_id = export_data.get('project_id')
                    
                    # Update project registry
                    from tsap.project.context import get_project_registry
                    registry = get_project_registry()
                    if orig_project_id in registry._projects:
                        del registry._projects[orig_project_id]
                    registry._projects[project.project_id] = project
                    
                    # Set as active project
                    registry.set_active_project(project.project_id)
                    
                    # Merge project data
                    self._merge_project_data(project, project_data)
                
                logging.info(f"Imported project metadata for {project.project_id} from {json_path}", 
                           component="transfer")
                
                return project
            
            except Exception as e:
                raise TransferError(f"Failed to import project metadata: {str(e)}")
    
    def import_results(self, results_dir: str, 
                      project: Optional[ProjectContext] = None) -> int:
        """
        Import analysis results from a directory.
        
        Args:
            results_dir: Directory with results to import
            project: Optional project to import into (uses existing_project if None)
            
        Returns:
            Number of imported results
            
        Raises:
            TransferError: If import fails
        """
        with self._lock:
            try:
                # Determine target project
                if project is None:
                    project = self.existing_project
                
                if project is None:
                    raise TransferError("No project specified for importing results")
                
                # Check if target project has an output directory
                if not project.output_directory:
                    raise TransferError("Project has no output directory")
                
                # Check metadata
                metadata_path = os.path.join(results_dir, "export_metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        
                    logging.debug(f"Importing results from export of project {metadata.get('project_id')}", 
                                component="transfer")
                
                # Create results directory in project
                project_results_dir = os.path.join(project.output_directory, "results")
                os.makedirs(project_results_dir, exist_ok=True)
                
                # Copy results
                results_count = 0
                
                for root, dirs, files in os.walk(results_dir):
                    # Skip metadata file
                    if os.path.basename(root) == results_dir and "export_metadata.json" in files:
                        files.remove("export_metadata.json")
                    
                    # Create corresponding directory in project
                    rel_path = os.path.relpath(root, results_dir)
                    if rel_path == '.':
                        dst_dir = project_results_dir
                    else:
                        dst_dir = os.path.join(project_results_dir, rel_path)
                    
                    os.makedirs(dst_dir, exist_ok=True)
                    
                    # Copy files
                    for file in files:
                        src_path = os.path.join(root, file)
                        dst_path = os.path.join(dst_dir, file)
                        shutil.copy2(src_path, dst_path)
                        results_count += 1
                
                logging.info(f"Imported {results_count} results to project {project.project_id}", 
                           component="transfer")
                
                return results_count
            
            except Exception as e:
                raise TransferError(f"Failed to import results: {str(e)}")
    
    def restore_from_backup(self, backup_path: str, 
                           password: Optional[str] = None) -> ProjectContext:
        """
        Restore a project from a backup file.
        
        Args:
            backup_path: Path to the backup file
            password: Optional password to decrypt the backup
            
        Returns:
            Restored project context
            
        Raises:
            TransferError: If restore fails
        """
        with self._lock:
            try:
                # Import from the backup ZIP
                return self.import_from_zip(
                    zip_path=backup_path,
                    import_profiles=True,
                    import_history=True,
                    import_results=True,
                    import_files=True,
                    password=password
                )
            
            except Exception as e:
                raise TransferError(f"Failed to restore from backup: {str(e)}")
    
    def _merge_project_data(self, project: ProjectContext, data: Dict[str, Any]) -> None:
        """
        Merge imported project data into a project context.
        
        Args:
            project: Target project context
            data: Project data to merge
            
        Raises:
            TransferError: If merge fails
        """
        try:
            # Import basic metadata
            if 'name' in data:
                project.name = data['name']
            
            if 'created_at' in data:
                project.created_at = data['created_at']
            
            if 'updated_at' in data:
                project.updated_at = data['updated_at']
            
            # Set directories if needed
            if 'root_directory' in data and not project.root_directory:
                project.root_directory = data['root_directory']
            
            if 'output_directory' in data and not project.output_directory:
                output_dir = data['output_directory']
                
                # Create output directory if it doesn't exist
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                
                project.output_directory = output_dir
            
            # Merge data
            if '_data' in data:
                with project._lock:
                    for key, value in data['_data'].items():
                        project._data[key] = value
            
            # Merge settings
            if '_settings' in data:
                with project._lock:
                    for key, value in data['_settings'].items():
                        project._settings[key] = value
        
        except Exception as e:
            raise TransferError(f"Failed to merge project data: {str(e)}")


# Functions for working with projects

def export_project(project_id: Optional[str] = None, 
                 output_path: Optional[str] = None,
                 include_profiles: bool = True,
                 include_history: bool = True,
                 include_results: bool = True,
                 include_files: bool = True) -> str:
    """
    Export a project to a ZIP archive.
    
    Args:
        project_id: ID of the project (or None for active project)
        output_path: Path for the output ZIP file
        include_profiles: Whether to include project profiles
        include_history: Whether to include command history
        include_results: Whether to include analysis results
        include_files: Whether to include registered files
        
    Returns:
        Path to the created ZIP file
        
    Raises:
        TransferError: If export fails
    """
    from tsap.project.context import get_project
    
    # Get the project
    project = get_project(project_id)
    
    if not project:
        if project_id:
            raise TransferError(f"Project not found: {project_id}")
        else:
            raise TransferError("No active project")
    
    # Create export handler
    exporter = ProjectExport(project)
    
    # Determine output path if not provided
    if not output_path:
        if not project.output_directory:
            raise TransferError("Project has no output directory and no output path provided")
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            project.output_directory,
            f"{project.project_id}_export_{timestamp}.zip"
        )
    
    # Export the project
    return exporter.export_to_zip(
        output_path=output_path,
        include_profiles=include_profiles,
        include_history=include_history,
        include_results=include_results,
        include_files=include_files
    )


def import_project(import_path: str, 
                 target_project_id: Optional[str] = None,
                 import_profiles: bool = True,
                 import_history: bool = True,
                 import_results: bool = True,
                 import_files: bool = True) -> ProjectContext:
    """
    Import a project from a ZIP archive or JSON file.
    
    Args:
        import_path: Path to the import file
        target_project_id: Optional ID of an existing project to import into
        import_profiles: Whether to import project profiles
        import_history: Whether to import command history
        import_results: Whether to import analysis results
        import_files: Whether to import registered files
        
    Returns:
        Imported or updated project context
        
    Raises:
        TransferError: If import fails
    """
    # Determine if importing into an existing project
    existing_project = None
    if target_project_id:
        from tsap.project.context import get_project
        existing_project = get_project(target_project_id)
        
        if not existing_project:
            raise TransferError(f"Target project not found: {target_project_id}")
    
    # Create import handler
    importer = ProjectImport(existing_project)
    
    # Determine file type
    if import_path.lower().endswith('.zip'):
        # Import from ZIP
        return importer.import_from_zip(
            zip_path=import_path,
            import_profiles=import_profiles,
            import_history=import_history,
            import_results=import_results,
            import_files=import_files
        )
    elif import_path.lower().endswith('.json'):
        # Import from JSON
        return importer.import_from_json(json_path=import_path)
    else:
        raise TransferError(f"Unsupported import file format: {import_path}")


def create_project_backup(project_id: Optional[str] = None, 
                        output_path: Optional[str] = None) -> str:
    """
    Create a complete backup of a project.
    
    Args:
        project_id: ID of the project (or None for active project)
        output_path: Optional path for the backup file
            
    Returns:
        Path to the created backup file
        
    Raises:
        TransferError: If backup fails
    """
    from tsap.project.context import get_project
    
    # Get the project
    project = get_project(project_id)
    
    if not project:
        if project_id:
            raise TransferError(f"Project not found: {project_id}")
        else:
            raise TransferError("No active project")
    
    # Create export handler
    exporter = ProjectExport(project)
    
    # Create backup
    return exporter.create_backup(output_path)


def restore_project_from_backup(backup_path: str) -> ProjectContext:
    """
    Restore a project from a backup file.
    
    Args:
        backup_path: Path to the backup file
        
    Returns:
        Restored project context
        
    Raises:
        TransferError: If restore fails
    """
    # Create import handler
    importer = ProjectImport()
    
    # Restore from backup
    return importer.restore_from_backup(backup_path)


def transfer_results(source_project_id: str, 
                    target_project_id: str,
                    result_types: Optional[List[str]] = None) -> int:
    """
    Transfer results between projects.
    
    Args:
        source_project_id: ID of the source project
        target_project_id: ID of the target project
        result_types: Optional list of result types to transfer (None for all)
        
    Returns:
        Number of transferred results
        
    Raises:
        TransferError: If transfer fails
    """
    from tsap.project.context import get_project
    
    # Get the source project
    source_project = get_project(source_project_id)
    if not source_project:
        raise TransferError(f"Source project not found: {source_project_id}")
    
    # Get the target project
    target_project = get_project(target_project_id)
    if not target_project:
        raise TransferError(f"Target project not found: {target_project_id}")
    
    # Create export handler
    exporter = ProjectExport(source_project)
    
    # Create import handler
    importer = ProjectImport(target_project)
    
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp(prefix="tsap_transfer_")
    
    try:
        # Export results to temporary directory
        export_dir = exporter.export_results(temp_dir, result_types)
        
        # Import results to target project
        return importer.import_results(export_dir, target_project)
        
    except Exception as e:
        raise TransferError(f"Failed to transfer results: {str(e)}")
        
    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)