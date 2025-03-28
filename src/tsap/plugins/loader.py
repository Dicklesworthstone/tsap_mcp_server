"""
TSAP Plugin Loader.

This module provides functionality for discovering, loading, and managing
TSAP plugins from various sources including built-in plugins, 
installed packages, and local directories.
"""

import os
import sys
import glob
import importlib
import importlib.util
from typing import Dict, List, Any, Optional, Union, Set, Type
from dataclasses import dataclass, field
from pathlib import Path

from tsap.utils.logging import logger
from tsap.config import get_config
from tsap.utils.errors import PluginError
from tsap.plugins.interface import (
    Plugin, 
    discover_plugins_in_module,
    discover_plugins_in_package,
    validate_plugin_class,
    PluginType,
    PluginCapability,
)


@dataclass
class PluginMetadata:
    """Metadata for a loaded plugin."""
    
    id: str
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    capabilities: List[PluginCapability]
    dependencies: List[str]
    path: Optional[str] = None
    module: Optional[str] = None
    class_name: Optional[str] = None
    enabled: bool = True
    errors: List[str] = field(default_factory=list)
    status: str = "unloaded"  # unloaded, loaded, initialized, active, error, disabled


@dataclass
class LoadedPlugin:
    """A loaded plugin with its instance and metadata."""
    
    metadata: PluginMetadata
    instance: Optional[Plugin] = None
    plugin_class: Optional[Type[Plugin]] = None
    components: Dict[str, Any] = field(default_factory=dict)


class PluginLoader:
    """Manages discovering, loading, and activating TSAP plugins."""
    
    def __init__(self):
        """Initialize the plugin loader."""
        self.plugins: Dict[str, LoadedPlugin] = {}
        self.discovered_plugins: Dict[str, PluginMetadata] = {}
        self.enabled_plugins: Set[str] = set()
        self.plugin_paths: List[str] = []
        self.initialized = False
    
    def initialize(self, plugin_paths: Optional[List[str]] = None) -> None:
        """Initialize the plugin loader.
        
        Args:
            plugin_paths: Additional paths to search for plugins
        """
        if self.initialized:
            return
        
        # Get plugin paths from configuration
        config = get_config()
        config_paths = config.plugins.paths or []
        
        # Combine paths and ensure they exist
        all_paths = []
        
        # Add built-in plugins path
        builtins_path = Path(__file__).parent / "builtin"
        if builtins_path.exists():
            all_paths.append(str(builtins_path))
        
        # Add configuration paths
        all_paths.extend(config_paths)
        
        # Add additional paths
        if plugin_paths:
            all_paths.extend(plugin_paths)
        
        # Store unique paths
        self.plugin_paths = list(set(all_paths))
        
        # Load enabled plugins from configuration
        self.enabled_plugins = set(config.plugins.enabled or [])
        
        logger.debug(f"Plugin loader initialized with paths: {self.plugin_paths}")
        logger.debug(f"Enabled plugins: {self.enabled_plugins}")
        
        self.initialized = True
    
    def discover_plugins(self) -> Dict[str, PluginMetadata]:
        """Discover available plugins.
        
        Returns:
            Dictionary of discovered plugin metadata
        """
        if not self.initialized:
            self.initialize()
        
        # Dictionary to hold discovered plugins
        discovered = {}
        
        # Discover plugins in built-in package
        builtin_plugins = self._discover_builtin_plugins()
        for metadata in builtin_plugins.values():
            discovered[metadata.id] = metadata
        
        # Discover plugins in plugin paths
        for path in self.plugin_paths:
            path_plugins = self._discover_plugins_in_path(path)
            for metadata in path_plugins.values():
                discovered[metadata.id] = metadata
        
        # Store discovered plugins
        self.discovered_plugins = discovered
        
        return discovered
    
    def _discover_builtin_plugins(self) -> Dict[str, PluginMetadata]:
        """Discover built-in plugins.
        
        Returns:
            Dictionary of built-in plugin metadata
        """
        discovered = {}
        
        try:
            plugin_classes = discover_plugins_in_package("tsap.plugins.builtin")
            
            for plugin_class in plugin_classes:
                # Validate plugin class
                errors = validate_plugin_class(plugin_class)
                
                # Create metadata
                metadata = PluginMetadata(
                    id=plugin_class.id,
                    name=plugin_class.name,
                    version=plugin_class.version,
                    description=plugin_class.description,
                    author=plugin_class.author,
                    plugin_type=plugin_class.plugin_type,
                    capabilities=plugin_class.capabilities,
                    dependencies=plugin_class.dependencies,
                    module=plugin_class.__module__,
                    class_name=plugin_class.__name__,
                    path="builtin",
                    errors=errors,
                    status="unloaded" if not errors else "error",
                )
                
                discovered[metadata.id] = metadata
        except ImportError as e:
            logger.warning(f"Failed to import builtin plugins: {e}")
        
        return discovered
    
    def _discover_plugins_in_path(self, path: str) -> Dict[str, PluginMetadata]:
        """Discover plugins in a directory path.
        
        Args:
            path: Directory path to search
            
        Returns:
            Dictionary of discovered plugin metadata
        """
        discovered = {}
        
        # Ensure path exists
        if not os.path.exists(path):
            logger.warning(f"Plugin path does not exist: {path}")
            return discovered
        
        # Search for Python files and packages
        python_files = glob.glob(os.path.join(path, "*.py"))
        package_dirs = [
            d for d in glob.glob(os.path.join(path, "*"))
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "__init__.py"))
        ]
        
        # Process Python files
        for file_path in python_files:
            try:
                file_plugins = self._discover_plugins_in_file(file_path)
                for metadata in file_plugins.values():
                    discovered[metadata.id] = metadata
            except Exception as e:
                logger.error(f"Error discovering plugins in file {file_path}: {e}")
        
        # Process packages
        for package_dir in package_dirs:
            try:
                package_plugins = self._discover_plugins_in_directory(package_dir)
                for metadata in package_plugins.values():
                    discovered[metadata.id] = metadata
            except Exception as e:
                logger.error(f"Error discovering plugins in directory {package_dir}: {e}")
        
        return discovered
    
    def _discover_plugins_in_file(self, file_path: str) -> Dict[str, PluginMetadata]:
        """Discover plugins in a Python file.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            Dictionary of discovered plugin metadata
        """
        discovered = {}
        
        try:
            # Import the module
            module_name = os.path.basename(file_path).replace(".py", "")
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                logger.warning(f"Could not load spec for {file_path}")
                return discovered
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin classes
            plugin_classes = discover_plugins_in_module(module)
            
            for plugin_class in plugin_classes:
                # Validate plugin class
                errors = validate_plugin_class(plugin_class)
                
                # Create metadata
                metadata = PluginMetadata(
                    id=plugin_class.id,
                    name=plugin_class.name,
                    version=plugin_class.version,
                    description=plugin_class.description,
                    author=plugin_class.author,
                    plugin_type=plugin_class.plugin_type,
                    capabilities=plugin_class.capabilities,
                    dependencies=plugin_class.dependencies,
                    module=module_name,
                    class_name=plugin_class.__name__,
                    path=file_path,
                    errors=errors,
                    status="unloaded" if not errors else "error",
                )
                
                discovered[metadata.id] = metadata
        except Exception as e:
            logger.error(f"Error loading plugins from file {file_path}: {e}")
        
        return discovered
    
    def _discover_plugins_in_directory(self, dir_path: str) -> Dict[str, PluginMetadata]:
        """Discover plugins in a directory (package).
        
        Args:
            dir_path: Path to directory
            
        Returns:
            Dictionary of discovered plugin metadata
        """
        discovered = {}
        
        try:
            # Add directory to sys.path temporarily
            parent_dir = os.path.dirname(dir_path)
            package_name = os.path.basename(dir_path)
            
            sys.path.insert(0, parent_dir)
            
            try:
                # Import the package
                package = importlib.import_module(package_name)  # noqa: F841
                
                # Find plugin classes
                plugin_classes = discover_plugins_in_package(package_name)
                
                for plugin_class in plugin_classes:
                    # Validate plugin class
                    errors = validate_plugin_class(plugin_class)
                    
                    # Create metadata
                    metadata = PluginMetadata(
                        id=plugin_class.id,
                        name=plugin_class.name,
                        version=plugin_class.version,
                        description=plugin_class.description,
                        author=plugin_class.author,
                        plugin_type=plugin_class.plugin_type,
                        capabilities=plugin_class.capabilities,
                        dependencies=plugin_class.dependencies,
                        module=plugin_class.__module__,
                        class_name=plugin_class.__name__,
                        path=dir_path,
                        errors=errors,
                        status="unloaded" if not errors else "error",
                    )
                    
                    discovered[metadata.id] = metadata
            finally:
                # Remove directory from sys.path
                sys.path.remove(parent_dir)
        except Exception as e:
            logger.error(f"Error loading plugins from directory {dir_path}: {e}")
        
        return discovered
    
    def load_plugin(self, plugin_id: str) -> Optional[LoadedPlugin]:
        """Load a plugin by ID.
        
        Args:
            plugin_id: Plugin ID
            
        Returns:
            Loaded plugin or None if loading fails
            
        Raises:
            PluginError: If plugin not found or loading fails
        """
        # Check if plugin is already loaded
        if plugin_id in self.plugins:
            return self.plugins[plugin_id]
        
        # Check if plugin is discovered
        if plugin_id not in self.discovered_plugins:
            # Try to discover plugins first
            if not self.discovered_plugins:
                self.discover_plugins()
            
            # Check again
            if plugin_id not in self.discovered_plugins:
                raise PluginError(f"Plugin not found: {plugin_id}")
        
        metadata = self.discovered_plugins[plugin_id]
        
        # Check for errors
        if metadata.errors:
            error_messages = ", ".join(metadata.errors)
            raise PluginError(f"Plugin validation failed: {error_messages}", plugin_id)
        
        # Load the plugin class
        try:
            plugin_class = self._load_plugin_class(metadata)
        except Exception as e:
            metadata.status = "error"
            metadata.errors.append(f"Failed to load plugin class: {e}")
            raise PluginError(f"Failed to load plugin class: {e}", plugin_id)
        
        # Check if plugin is enabled
        if plugin_id not in self.enabled_plugins and not metadata.path == "builtin":
            # Don't throw an error, just return None
            metadata.status = "disabled"
            logger.info(f"Plugin {plugin_id} is disabled")
            
            # Create a LoadedPlugin with no instance
            loaded_plugin = LoadedPlugin(
                metadata=metadata,
                plugin_class=plugin_class,
            )
            self.plugins[plugin_id] = loaded_plugin
            return loaded_plugin
        
        # Create plugin instance
        try:
            plugin_instance = plugin_class()
            metadata.status = "loaded"
        except Exception as e:
            metadata.status = "error"
            metadata.errors.append(f"Failed to instantiate plugin: {e}")
            raise PluginError(f"Failed to instantiate plugin: {e}", plugin_id)
        
        # Create LoadedPlugin
        loaded_plugin = LoadedPlugin(
            metadata=metadata,
            instance=plugin_instance,
            plugin_class=plugin_class,
        )
        
        # Store loaded plugin
        self.plugins[plugin_id] = loaded_plugin
        
        return loaded_plugin
    
    def _load_plugin_class(self, metadata: PluginMetadata) -> Type[Plugin]:
        """Load a plugin class from metadata.
        
        Args:
            metadata: Plugin metadata
            
        Returns:
            Plugin class
            
        Raises:
            PluginError: If loading fails
        """
        if metadata.path == "builtin":
            # Load from builtin package
            try:
                if not metadata.module or not metadata.class_name:
                    raise PluginError("Missing module or class name in metadata")
                
                module = importlib.import_module(metadata.module)
                plugin_class = getattr(module, metadata.class_name)
                
                return plugin_class
            except ImportError as e:
                raise PluginError(f"Failed to import builtin module {metadata.module}: {e}")
            except AttributeError as e:
                raise PluginError(f"Failed to find class {metadata.class_name} in module {metadata.module}: {e}")
        elif metadata.path and metadata.path.endswith(".py"):
            # Load from Python file
            try:
                if not metadata.class_name:
                    raise PluginError("Missing class name in metadata")
                
                module_name = os.path.basename(metadata.path).replace(".py", "")
                spec = importlib.util.spec_from_file_location(module_name, metadata.path)
                if spec is None or spec.loader is None:
                    raise PluginError(f"Could not load spec for {metadata.path}")
                
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                plugin_class = getattr(module, metadata.class_name)
                
                return plugin_class
            except Exception as e:
                raise PluginError(f"Failed to load plugin class from file {metadata.path}: {e}")
        elif metadata.path and os.path.isdir(metadata.path):
            # Load from package
            try:
                if not metadata.module or not metadata.class_name:
                    raise PluginError("Missing module or class name in metadata")
                
                # Add directory to sys.path temporarily
                parent_dir = os.path.dirname(metadata.path)
                sys.path.insert(0, parent_dir)
                
                try:
                    module = importlib.import_module(metadata.module)
                    plugin_class = getattr(module, metadata.class_name)
                    
                    return plugin_class
                finally:
                    # Remove directory from sys.path
                    if parent_dir in sys.path:
                        sys.path.remove(parent_dir)
            except ImportError as e:
                raise PluginError(f"Failed to import module {metadata.module}: {e}")
            except AttributeError as e:
                raise PluginError(f"Failed to find class {metadata.class_name} in module {metadata.module}: {e}")
        else:
            raise PluginError(f"Unsupported plugin path: {metadata.path}")
    
    def initialize_plugin(self, plugin_id: str) -> Optional[LoadedPlugin]:
        """Initialize a loaded plugin.
        
        Args:
            plugin_id: Plugin ID
            
        Returns:
            Initialized plugin or None if initialization fails
            
        Raises:
            PluginError: If plugin not found or initialization fails
        """
        # Check if plugin is loaded
        if plugin_id not in self.plugins:
            # Try to load the plugin
            loaded_plugin = self.load_plugin(plugin_id)
            if not loaded_plugin or not loaded_plugin.instance:
                raise PluginError(f"Plugin not loaded: {plugin_id}")
        else:
            loaded_plugin = self.plugins[plugin_id]
            if not loaded_plugin.instance:
                raise PluginError(f"Plugin disabled: {plugin_id}")
        
        # Get plugin instance
        plugin = loaded_plugin.instance
        metadata = loaded_plugin.metadata
        
        # Check for dependencies
        for dependency_id in metadata.dependencies:
            # Skip if already loaded
            if dependency_id in self.plugins and self.plugins[dependency_id].metadata.status == "initialized":
                continue
            
            # Load and initialize dependency
            try:
                self.initialize_plugin(dependency_id)
            except PluginError as e:
                metadata.status = "error"
                metadata.errors.append(f"Failed to load dependency {dependency_id}: {e}")
                raise PluginError(f"Failed to load dependency {dependency_id}: {e}", plugin_id)
        
        # Initialize the plugin
        try:
            plugin.initialize()
            metadata.status = "initialized"
        except Exception as e:
            metadata.status = "error"
            metadata.errors.append(f"Failed to initialize plugin: {e}")
            raise PluginError(f"Failed to initialize plugin: {e}", plugin_id)
        
        return loaded_plugin
    
    def register_plugin(self, plugin_id: str) -> Optional[LoadedPlugin]:
        """Register a loaded and initialized plugin.
        
        Args:
            plugin_id: Plugin ID
            
        Returns:
            Registered plugin or None if registration fails
            
        Raises:
            PluginError: If plugin not found or registration fails
        """
        # Check if plugin is initialized
        if plugin_id not in self.plugins:
            # Try to initialize the plugin
            loaded_plugin = self.initialize_plugin(plugin_id)
            if not loaded_plugin or not loaded_plugin.instance:
                raise PluginError(f"Plugin not initialized: {plugin_id}")
        else:
            loaded_plugin = self.plugins[plugin_id]
            if not loaded_plugin.instance:
                raise PluginError(f"Plugin disabled: {plugin_id}")
            if loaded_plugin.metadata.status != "initialized":
                raise PluginError(f"Plugin not initialized: {plugin_id}")
        
        # Get plugin instance
        plugin = loaded_plugin.instance
        metadata = loaded_plugin.metadata
        
        # Register the plugin
        try:
            plugin.register()
            metadata.status = "active"
            
            # Store components based on plugin type
            components = self._get_plugin_components(plugin)
            loaded_plugin.components = components
        except Exception as e:
            metadata.status = "error"
            metadata.errors.append(f"Failed to register plugin: {e}")
            raise PluginError(f"Failed to register plugin: {e}", plugin_id)
        
        return loaded_plugin
    
    def _get_plugin_components(self, plugin: Plugin) -> Dict[str, Any]:
        """Get components provided by a plugin based on its type.
        
        Args:
            plugin: Plugin instance
            
        Returns:
            Dictionary mapping component names to component objects
        """
        components = {}
        
        # Get components based on plugin type
        if plugin.plugin_type == PluginType.CORE_TOOL:
            # Core tool plugin
            try:
                components["tool_classes"] = plugin.get_tool_classes()  # type: ignore
                components["tool_instances"] = plugin.get_tool_instances()  # type: ignore
            except Exception as e:
                logger.error(f"Failed to get components from core tool plugin {plugin.id}: {e}")
                
        elif plugin.plugin_type == PluginType.COMPOSITE:
            # Composite plugin
            try:
                components["operations"] = plugin.get_operations()  # type: ignore
            except Exception as e:
                logger.error(f"Failed to get components from composite plugin {plugin.id}: {e}")
                
        elif plugin.plugin_type == PluginType.ANALYSIS:
            # Analysis plugin
            try:
                components["analysis_tools"] = plugin.get_analysis_tools()  # type: ignore
                components["analysis_functions"] = plugin.get_analysis_functions()  # type: ignore
            except Exception as e:
                logger.error(f"Failed to get components from analysis plugin {plugin.id}: {e}")
                
        elif plugin.plugin_type == PluginType.EVOLUTION:
            # Evolution plugin
            try:
                components["evolution_components"] = plugin.get_evolution_components()  # type: ignore
            except Exception as e:
                logger.error(f"Failed to get components from evolution plugin {plugin.id}: {e}")
                
        elif plugin.plugin_type == PluginType.TEMPLATE:
            # Template plugin
            try:
                components["templates"] = plugin.get_templates()  # type: ignore
            except Exception as e:
                logger.error(f"Failed to get components from template plugin {plugin.id}: {e}")
                
        elif plugin.plugin_type == PluginType.FORMAT:
            # Format plugin
            try:
                components["input_formats"] = plugin.get_input_formats()  # type: ignore
                components["output_formats"] = plugin.get_output_formats()  # type: ignore
            except Exception as e:
                logger.error(f"Failed to get components from format plugin {plugin.id}: {e}")
                
        elif plugin.plugin_type == PluginType.INTEGRATION:
            # Integration plugin
            try:
                components["integrations"] = plugin.get_integrations()  # type: ignore
            except Exception as e:
                logger.error(f"Failed to get components from integration plugin {plugin.id}: {e}")
                
        elif plugin.plugin_type == PluginType.UI:
            # UI plugin
            try:
                components["ui_components"] = plugin.get_ui_components()  # type: ignore
            except Exception as e:
                logger.error(f"Failed to get components from UI plugin {plugin.id}: {e}")
                
        elif plugin.plugin_type == PluginType.UTILITY:
            # Utility plugin
            try:
                components["utilities"] = plugin.get_utilities()  # type: ignore
            except Exception as e:
                logger.error(f"Failed to get components from utility plugin {plugin.id}: {e}")
                
        elif plugin.plugin_type == PluginType.EXTENSION:
            # Extension plugin
            try:
                components["extensions"] = plugin.get_extensions()  # type: ignore
            except Exception as e:
                logger.error(f"Failed to get components from extension plugin {plugin.id}: {e}")
        
        return components
    
    def unload_plugin(self, plugin_id: str) -> bool:
        """Unload a plugin.
        
        Args:
            plugin_id: Plugin ID
            
        Returns:
            True if plugin was unloaded, False otherwise
        """
        # Check if plugin is loaded
        if plugin_id not in self.plugins:
            logger.warning(f"Plugin not loaded: {plugin_id}")
            return False
        
        loaded_plugin = self.plugins[plugin_id]
        
        # Skip if no instance
        if not loaded_plugin.instance:
            # Remove from loaded plugins
            del self.plugins[plugin_id]
            return True
        
        # Shut down the plugin
        try:
            loaded_plugin.instance.shutdown()
        except Exception as e:
            logger.error(f"Error shutting down plugin {plugin_id}: {e}")
        
        # Remove from loaded plugins
        del self.plugins[plugin_id]
        
        return True
    
    def get_plugin(self, plugin_id: str) -> Optional[LoadedPlugin]:
        """Get a loaded plugin by ID.
        
        Args:
            plugin_id: Plugin ID
            
        Returns:
            Loaded plugin or None if not found
        """
        return self.plugins.get(plugin_id)
    
    def get_plugin_by_name(self, name: str) -> Optional[LoadedPlugin]:
        """Get a loaded plugin by name.
        
        Args:
            name: Plugin name
            
        Returns:
            Loaded plugin or None if not found
        """
        for plugin in self.plugins.values():
            if plugin.metadata.name == name:
                return plugin
        return None
    
    def get_plugins_by_type(self, plugin_type: Union[str, PluginType]) -> List[LoadedPlugin]:
        """Get loaded plugins by type.
        
        Args:
            plugin_type: Plugin type
            
        Returns:
            List of loaded plugins
        """
        # Convert string to enum
        if isinstance(plugin_type, str):
            try:
                plugin_type = PluginType(plugin_type)
            except ValueError:
                logger.warning(f"Invalid plugin type: {plugin_type}")
                return []
        
        return [p for p in self.plugins.values() if p.metadata.plugin_type == plugin_type]
    
    def get_plugins_by_capability(self, capability: Union[str, PluginCapability]) -> List[LoadedPlugin]:
        """Get loaded plugins by capability.
        
        Args:
            capability: Plugin capability
            
        Returns:
            List of loaded plugins
        """
        # Convert string to enum
        if isinstance(capability, str):
            try:
                capability = PluginCapability(capability)
            except ValueError:
                logger.warning(f"Invalid plugin capability: {capability}")
                return []
        
        return [p for p in self.plugins.values() if capability in p.metadata.capabilities]
    
    def enable_plugin(self, plugin_id: str) -> bool:
        """Enable a plugin.
        
        Args:
            plugin_id: Plugin ID
            
        Returns:
            True if plugin was enabled, False otherwise
        """
        # Check if plugin is available
        if plugin_id not in self.discovered_plugins:
            logger.warning(f"Plugin not found: {plugin_id}")
            return False
        
        # Add to enabled plugins
        self.enabled_plugins.add(plugin_id)
        
        # Save enabled plugins to configuration
        config = get_config()
        config.plugins.enabled = list(self.enabled_plugins)
        
        return True
    
    def disable_plugin(self, plugin_id: str) -> bool:
        """Disable a plugin.
        
        Args:
            plugin_id: Plugin ID
            
        Returns:
            True if plugin was disabled, False otherwise
        """
        # Check if plugin is enabled
        if plugin_id not in self.enabled_plugins:
            logger.warning(f"Plugin not enabled: {plugin_id}")
            return False
        
        # Check if plugin is loaded
        if plugin_id in self.plugins:
            # Unload plugin
            self.unload_plugin(plugin_id)
        
        # Remove from enabled plugins
        self.enabled_plugins.remove(plugin_id)
        
        # Save enabled plugins to configuration
        config = get_config()
        config.plugins.enabled = list(self.enabled_plugins)
        
        return True
    
    def load_all_plugins(self) -> Dict[str, LoadedPlugin]:
        """Load all enabled plugins.
        
        Returns:
            Dictionary mapping plugin IDs to loaded plugins
        """
        # Discover plugins
        self.discover_plugins()
        
        # Load enabled plugins
        loaded_plugins = {}
        for plugin_id in list(self.discovered_plugins.keys()):
            metadata = self.discovered_plugins[plugin_id]
            
            # Skip plugins with errors
            if metadata.errors:
                continue
            
            # Skip disabled plugins
            if plugin_id not in self.enabled_plugins and metadata.path != "builtin":
                continue
            
            try:
                # Load plugin
                loaded_plugin = self.load_plugin(plugin_id)
                if loaded_plugin and loaded_plugin.instance:
                    loaded_plugins[plugin_id] = loaded_plugin
            except PluginError as e:
                logger.error(f"Failed to load plugin {plugin_id}: {e}")
        
        return loaded_plugins
    
    def initialize_all_plugins(self) -> Dict[str, LoadedPlugin]:
        """Initialize all loaded plugins.
        
        Returns:
            Dictionary mapping plugin IDs to initialized plugins
        """
        # Load all plugins
        self.load_all_plugins()
        
        # Initialize loaded plugins
        initialized_plugins = {}
        for plugin_id, loaded_plugin in list(self.plugins.items()):
            # Skip plugins without instance
            if not loaded_plugin.instance:
                continue
            
            try:
                # Initialize plugin
                self.initialize_plugin(plugin_id)
                initialized_plugins[plugin_id] = loaded_plugin
            except PluginError as e:
                logger.error(f"Failed to initialize plugin {plugin_id}: {e}")
        
        return initialized_plugins
    
    def register_all_plugins(self) -> Dict[str, LoadedPlugin]:
        """Register all initialized plugins.
        
        Returns:
            Dictionary mapping plugin IDs to registered plugins
        """
        # Initialize all plugins
        self.initialize_all_plugins()
        
        # Register initialized plugins
        registered_plugins = {}
        for plugin_id, loaded_plugin in list(self.plugins.items()):
            # Skip plugins without instance
            if not loaded_plugin.instance:
                continue
            
            # Skip plugins that aren't initialized
            if loaded_plugin.metadata.status != "initialized":
                continue
            
            try:
                # Register plugin
                self.register_plugin(plugin_id)
                registered_plugins[plugin_id] = loaded_plugin
            except PluginError as e:
                logger.error(f"Failed to register plugin {plugin_id}: {e}")
        
        return registered_plugins
    
    def unload_all_plugins(self) -> None:
        """Unload all loaded plugins."""
        for plugin_id in list(self.plugins.keys()):
            self.unload_plugin(plugin_id)
    
    def get_component(self, component_type: str, component_name: str) -> Optional[Any]:
        """Get a component by type and name.
        
        Args:
            component_type: Component type (e.g., "tool_instances", "operations")
            component_name: Component name
            
        Returns:
            Component or None if not found
        """
        for plugin in self.plugins.values():
            if not plugin.instance:
                continue
            
            components = plugin.components.get(component_type, {})
            if component_name in components:
                return components[component_name]
        
        return None
    
    def get_components_by_type(self, component_type: str) -> Dict[str, Any]:
        """Get all components of a specific type.
        
        Args:
            component_type: Component type (e.g., "tool_instances", "operations")
            
        Returns:
            Dictionary mapping component names to components
        """
        components = {}
        
        for plugin in self.plugins.values():
            if not plugin.instance:
                continue
            
            plugin_components = plugin.components.get(component_type, {})
            components.update(plugin_components)
        
        return components
    
    def reload_plugin(self, plugin_id: str) -> Optional[LoadedPlugin]:
        """Reload a plugin.
        
        Args:
            plugin_id: Plugin ID
            
        Returns:
            Reloaded plugin or None if reloading fails
            
        Raises:
            PluginError: If plugin not found or reloading fails
        """
        # Unload plugin if loaded
        if plugin_id in self.plugins:
            self.unload_plugin(plugin_id)
        
        # Rediscover plugins
        self.discover_plugins()
        
        # Load plugin
        loaded_plugin = self.load_plugin(plugin_id)
        
        # Skip if no instance
        if not loaded_plugin or not loaded_plugin.instance:
            return loaded_plugin
        
        # Initialize and register plugin
        self.initialize_plugin(plugin_id)
        self.register_plugin(plugin_id)
        
        return loaded_plugin
    
    def get_plugin_status(self, plugin_id: str) -> Dict[str, Any]:
        """Get plugin status.
        
        Args:
            plugin_id: Plugin ID
            
        Returns:
            Dictionary with plugin status information
            
        Raises:
            PluginError: If plugin not found
        """
        # Check if plugin is loaded
        if plugin_id in self.plugins:
            plugin = self.plugins[plugin_id]
            
            # Get status from plugin instance if available
            if plugin.instance:
                try:
                    return plugin.instance.get_status()
                except Exception as e:
                    logger.error(f"Error getting plugin status from instance: {e}")
            
            # Return metadata status
            return {
                "id": plugin.metadata.id,
                "name": plugin.metadata.name,
                "version": plugin.metadata.version,
                "status": plugin.metadata.status,
                "enabled": plugin_id in self.enabled_plugins or plugin.metadata.path == "builtin",
                "errors": plugin.metadata.errors,
            }
        
        # Check if plugin is discovered
        if plugin_id in self.discovered_plugins:
            metadata = self.discovered_plugins[plugin_id]
            
            return {
                "id": metadata.id,
                "name": metadata.name,
                "version": metadata.version,
                "status": metadata.status,
                "enabled": plugin_id in self.enabled_plugins or metadata.path == "builtin",
                "errors": metadata.errors,
            }
        
        raise PluginError(f"Plugin not found: {plugin_id}")
    
    def get_all_plugin_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all discovered plugins.
        
        Returns:
            Dictionary mapping plugin IDs to status information
        """
        status = {}
        
        # Include discovered plugins
        for plugin_id, metadata in self.discovered_plugins.items():
            status[plugin_id] = {
                "id": metadata.id,
                "name": metadata.name,
                "version": metadata.version,
                "status": metadata.status,
                "enabled": plugin_id in self.enabled_plugins or metadata.path == "builtin",
                "errors": metadata.errors,
            }
        
        # Update with loaded plugin status
        for plugin_id, plugin in self.plugins.items():
            if plugin.instance:
                try:
                    status[plugin_id] = plugin.instance.get_status()
                except Exception as e:
                    logger.error(f"Error getting plugin status from instance: {e}")
            
        return status


# Global plugin loader instance
_plugin_loader = PluginLoader()


def get_plugin_loader() -> PluginLoader:
    """Get the global plugin loader instance.
    
    Returns:
        Global plugin loader
    """
    return _plugin_loader


def initialize_plugins() -> None:
    """Initialize the plugin system.
    
    This loads, initializes, and registers all enabled plugins.
    """
    loader = get_plugin_loader()
    loader.initialize()
    loader.register_all_plugins()


def get_plugin(plugin_id: str) -> Optional[Plugin]:
    """Get a plugin instance by ID.
    
    Args:
        plugin_id: Plugin ID
        
    Returns:
        Plugin instance or None if not found
        
    Note:
        This is a convenience function that gets the plugin from the global loader.
    """
    loader = get_plugin_loader()
    loaded_plugin = loader.get_plugin(plugin_id)
    return loaded_plugin.instance if loaded_plugin else None


def get_plugin_component(component_type: str, component_name: str) -> Optional[Any]:
    """Get a plugin component by type and name.
    
    Args:
        component_type: Component type (e.g., "tool_instances", "operations")
        component_name: Component name
        
    Returns:
        Component or None if not found
        
    Note:
        This is a convenience function that gets the component from the global loader.
    """
    loader = get_plugin_loader()
    return loader.get_component(component_type, component_name)


def install_plugin(plugin_path: str, enable: bool = True) -> Optional[str]:
    """Install a plugin from a path.
    
    Args:
        plugin_path: Path to plugin file or directory
        enable: Whether to enable the plugin after installation
        
    Returns:
        Plugin ID or None if installation fails
        
    Note:
        This is a placeholder for future implementation.
    """
    # TODO: Implement plugin installation
    logger.error("Plugin installation not implemented yet")
    return None


def uninstall_plugin(plugin_id: str) -> bool:
    """Uninstall a plugin.
    
    Args:
        plugin_id: Plugin ID
        
    Returns:
        True if plugin was uninstalled, False otherwise
        
    Note:
        This is a placeholder for future implementation.
    """
    # TODO: Implement plugin uninstallation
    logger.error("Plugin uninstallation not implemented yet")
    return False