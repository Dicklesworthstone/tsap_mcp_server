"""
TSAP Plugin Manager.

This module provides a high-level interface for interacting with the plugin system,
integrating the plugin loader, registry, and providing utility functions for plugin management.
"""

from typing import Dict, List, Any, Optional, Type, Callable, Union

from tsap.utils.logging import logger
from tsap.plugins.interface import (
    Plugin, 
    PluginType,
    PluginCapability,
)
from tsap.plugins.loader import get_plugin_loader, LoadedPlugin
from tsap.plugins.registry import get_registry


class PluginManager:
    """High-level manager for the TSAP plugin system."""
    
    def __init__(self):
        """Initialize the plugin manager."""
        self.loader = get_plugin_loader()
        self.registry = get_registry()
        self.initialized = False
    
    def initialize(self, plugin_paths: Optional[List[str]] = None) -> None:
        """Initialize the plugin system.
        
        Args:
            plugin_paths: Additional paths to search for plugins
        """
        if self.initialized:
            return
        
        # Initialize the plugin loader
        self.loader.initialize(plugin_paths)
        
        self.initialized = True
        logger.info("Plugin manager initialized")
    
    def discover_plugins(self) -> Dict[str, Any]:
        """Discover available plugins.
        
        Returns:
            Dictionary of discovered plugin metadata
        """
        if not self.initialized:
            self.initialize()
        
        return self.loader.discover_plugins()
    
    def load_plugins(self) -> Dict[str, LoadedPlugin]:
        """Load all enabled plugins.
        
        Returns:
            Dictionary mapping plugin IDs to loaded plugins
        """
        if not self.initialized:
            self.initialize()
        
        return self.loader.load_all_plugins()
    
    def initialize_plugins(self) -> Dict[str, LoadedPlugin]:
        """Initialize all loaded plugins.
        
        Returns:
            Dictionary mapping plugin IDs to initialized plugins
        """
        if not self.initialized:
            self.initialize()
        
        return self.loader.initialize_all_plugins()
    
    def register_plugins(self) -> Dict[str, LoadedPlugin]:
        """Register all initialized plugins.
        
        Returns:
            Dictionary mapping plugin IDs to registered plugins
        """
        if not self.initialized:
            self.initialize()
        
        plugins = self.loader.register_all_plugins()
        
        # Register components with the registry
        for plugin_id, loaded_plugin in plugins.items():
            if loaded_plugin.instance and loaded_plugin.metadata.status == "active":
                self.registry.register_plugin_components(plugin_id, loaded_plugin.components)
        
        return plugins
    
    def start_plugins(self) -> None:
        """Start the plugin system.
        
        This discovers, loads, initializes, and registers all enabled plugins.
        """
        if not self.initialized:
            self.initialize()
        
        # Discover plugins
        self.discover_plugins()
        
        # Register plugins
        registered_plugins = self.register_plugins()
        
        # Log summary
        plugin_count = len(registered_plugins)
        plugin_types = {}
        
        for plugin in registered_plugins.values():
            plugin_type = plugin.metadata.plugin_type
            plugin_types[plugin_type] = plugin_types.get(plugin_type, 0) + 1
        
        logger.info(f"Started plugin system with {plugin_count} plugins")
        for plugin_type, count in plugin_types.items():
            logger.info(f"  - {count} {plugin_type.value} plugins")
    
    def stop_plugins(self) -> None:
        """Stop the plugin system.
        
        This unloads all plugins and clears the registry.
        """
        if not self.initialized:
            return
        
        # Get registered plugins
        registered_plugins = self.registry.list_registered_plugins()
        
        # Unregister plugins from registry
        for plugin_id in registered_plugins:
            self.registry.unregister_plugin_components(plugin_id)
        
        # Unload all plugins
        self.loader.unload_all_plugins()
        
        logger.info(f"Stopped plugin system ({len(registered_plugins)} plugins unloaded)")
    
    def get_plugin(self, plugin_id: str) -> Optional[Plugin]:
        """Get a plugin instance by ID.
        
        Args:
            plugin_id: Plugin ID
            
        Returns:
            Plugin instance or None if not found
        """
        loaded_plugin = self.loader.get_plugin(plugin_id)
        return loaded_plugin.instance if loaded_plugin else None
    
    def get_plugin_by_name(self, name: str) -> Optional[Plugin]:
        """Get a plugin instance by name.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin instance or None if not found
        """
        loaded_plugin = self.loader.get_plugin_by_name(name)
        return loaded_plugin.instance if loaded_plugin else None
    
    def get_plugins_by_type(self, plugin_type: Union[str, PluginType]) -> List[Plugin]:
        """Get plugin instances by type.
        
        Args:
            plugin_type: Plugin type
            
        Returns:
            List of plugin instances
        """
        loaded_plugins = self.loader.get_plugins_by_type(plugin_type)
        return [p.instance for p in loaded_plugins if p.instance]
    
    def get_plugins_by_capability(self, capability: Union[str, PluginCapability]) -> List[Plugin]:
        """Get plugin instances by capability.
        
        Args:
            capability: Plugin capability
            
        Returns:
            List of plugin instances
        """
        loaded_plugins = self.loader.get_plugins_by_capability(capability)
        return [p.instance for p in loaded_plugins if p.instance]
    
    def get_plugin_status(self, plugin_id: str) -> Dict[str, Any]:
        """Get plugin status.
        
        Args:
            plugin_id: Plugin ID
            
        Returns:
            Dictionary with plugin status information
            
        Raises:
            PluginError: If plugin not found
        """
        return self.loader.get_plugin_status(plugin_id)
    
    def get_all_plugin_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all discovered plugins.
        
        Returns:
            Dictionary mapping plugin IDs to status information
        """
        return self.loader.get_all_plugin_status()
    
    def enable_plugin(self, plugin_id: str) -> bool:
        """Enable a plugin.
        
        Args:
            plugin_id: Plugin ID
            
        Returns:
            True if plugin was enabled, False otherwise
        """
        return self.loader.enable_plugin(plugin_id)
    
    def disable_plugin(self, plugin_id: str) -> bool:
        """Disable a plugin.
        
        Args:
            plugin_id: Plugin ID
            
        Returns:
            True if plugin was disabled, False otherwise
        """
        # Unregister plugin components if it's registered
        if plugin_id in self.registry.list_registered_plugins():
            self.registry.unregister_plugin_components(plugin_id)
        
        return self.loader.disable_plugin(plugin_id)
    
    def reload_plugin(self, plugin_id: str) -> Optional[Plugin]:
        """Reload a plugin.
        
        Args:
            plugin_id: Plugin ID
            
        Returns:
            Reloaded plugin or None if reloading fails
        """
        # Unregister plugin components if it's registered
        if plugin_id in self.registry.list_registered_plugins():
            self.registry.unregister_plugin_components(plugin_id)
        
        # Reload the plugin
        loaded_plugin = self.loader.reload_plugin(plugin_id)
        
        # Register components if the plugin was activated
        if loaded_plugin and loaded_plugin.instance and loaded_plugin.metadata.status == "active":
            self.registry.register_plugin_components(plugin_id, loaded_plugin.components)
        
        return loaded_plugin.instance if loaded_plugin else None
    
    def install_plugin(self, plugin_path: str, enable: bool = True) -> Optional[str]:
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
    
    def uninstall_plugin(self, plugin_id: str) -> bool:
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
    
    # Component access methods
    
    def get_tool_class(self, name: str) -> Optional[Type]:
        """Get a tool class by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool class or None if not found
        """
        return self.registry.get_tool_class(name)
    
    def get_tool_instance(self, name: str) -> Optional[Any]:
        """Get a tool instance by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool instance or None if not found
        """
        return self.registry.get_tool_instance(name)
    
    def get_operation(self, name: str) -> Optional[Callable]:
        """Get a composite operation by name.
        
        Args:
            name: Operation name
            
        Returns:
            Operation function or None if not found
        """
        return self.registry.get_operation(name)
    
    def get_analysis_tool(self, name: str) -> Optional[Type]:
        """Get an analysis tool class by name.
        
        Args:
            name: Tool name
            
        Returns:
            Analysis tool class or None if not found
        """
        return self.registry.get_analysis_tool(name)
    
    def get_analysis_function(self, name: str) -> Optional[Callable]:
        """Get an analysis function by name.
        
        Args:
            name: Function name
            
        Returns:
            Analysis function or None if not found
        """
        return self.registry.get_analysis_function(name)
    
    def get_evolution_component(self, name: str) -> Optional[Any]:
        """Get an evolution component by name.
        
        Args:
            name: Component name
            
        Returns:
            Evolution component or None if not found
        """
        return self.registry.get_evolution_component(name)
    
    def get_template(self, name: str) -> Optional[Any]:
        """Get a template by name.
        
        Args:
            name: Template name
            
        Returns:
            Template or None if not found
        """
        return self.registry.get_template(name)
    
    def get_input_format(self, name: str) -> Optional[Any]:
        """Get an input format handler by name.
        
        Args:
            name: Format name
            
        Returns:
            Input format handler or None if not found
        """
        return self.registry.get_input_format(name)
    
    def get_output_format(self, name: str) -> Optional[Any]:
        """Get an output format handler by name.
        
        Args:
            name: Format name
            
        Returns:
            Output format handler or None if not found
        """
        return self.registry.get_output_format(name)
    
    def get_integration(self, name: str) -> Optional[Any]:
        """Get an integration handler by name.
        
        Args:
            name: Integration name
            
        Returns:
            Integration handler or None if not found
        """
        return self.registry.get_integration(name)
    
    def get_ui_component(self, name: str) -> Optional[Any]:
        """Get a UI component by name.
        
        Args:
            name: Component name
            
        Returns:
            UI component or None if not found
        """
        return self.registry.get_ui_component(name)
    
    def get_utility(self, name: str) -> Optional[Any]:
        """Get a utility by name.
        
        Args:
            name: Utility name
            
        Returns:
            Utility or None if not found
        """
        return self.registry.get_utility(name)
    
    def get_extension(self, name: str) -> Optional[Any]:
        """Get an extension by name.
        
        Args:
            name: Extension name
            
        Returns:
            Extension or None if not found
        """
        return self.registry.get_extension(name)
    
    # Listing methods
    
    def list_plugins(self) -> Dict[str, Dict[str, Any]]:
        """Get a list of all discovered plugins with metadata.
        
        Returns:
            Dictionary mapping plugin IDs to metadata
        """
        return {
            plugin_id: metadata.get_metadata()
            for plugin_id, metadata in self.loader.discovered_plugins.items()
        }
    
    def list_registered_plugins(self) -> List[str]:
        """Get a list of all registered plugin IDs.
        
        Returns:
            List of plugin IDs
        """
        return self.registry.list_registered_plugins()
    
    def list_enabled_plugins(self) -> List[str]:
        """Get a list of all enabled plugin IDs.
        
        Returns:
            List of plugin IDs
        """
        return list(self.loader.enabled_plugins)
    
    def list_tool_classes(self) -> List[str]:
        """Get a list of all registered tool classes.
        
        Returns:
            List of tool class names
        """
        return self.registry.list_tool_classes()
    
    def list_tool_instances(self) -> List[str]:
        """Get a list of all registered tool instances.
        
        Returns:
            List of tool instance names
        """
        return self.registry.list_tool_instances()
    
    def list_operations(self) -> List[str]:
        """Get a list of all registered composite operations.
        
        Returns:
            List of operation names
        """
        return self.registry.list_operations()
    
    def list_analysis_tools(self) -> List[str]:
        """Get a list of all registered analysis tools.
        
        Returns:
            List of analysis tool names
        """
        return self.registry.list_analysis_tools()
    
    def list_analysis_functions(self) -> List[str]:
        """Get a list of all registered analysis functions.
        
        Returns:
            List of analysis function names
        """
        return self.registry.list_analysis_functions()
    
    def list_evolution_components(self) -> List[str]:
        """Get a list of all registered evolution components.
        
        Returns:
            List of evolution component names
        """
        return self.registry.list_evolution_components()
    
    def list_templates(self) -> List[str]:
        """Get a list of all registered templates.
        
        Returns:
            List of template names
        """
        return self.registry.list_templates()
    
    def list_input_formats(self) -> List[str]:
        """Get a list of all registered input formats.
        
        Returns:
            List of input format names
        """
        return self.registry.list_input_formats()
    
    def list_output_formats(self) -> List[str]:
        """Get a list of all registered output formats.
        
        Returns:
            List of output format names
        """
        return self.registry.list_output_formats()
    
    def list_integrations(self) -> List[str]:
        """Get a list of all registered integrations.
        
        Returns:
            List of integration names
        """
        return self.registry.list_integrations()
    
    def list_ui_components(self) -> List[str]:
        """Get a list of all registered UI components.
        
        Returns:
            List of UI component names
        """
        return self.registry.list_ui_components()
    
    def list_utilities(self) -> List[str]:
        """Get a list of all registered utilities.
        
        Returns:
            List of utility names
        """
        return self.registry.list_utilities()
    
    def list_extensions(self) -> List[str]:
        """Get a list of all registered extensions.
        
        Returns:
            List of extension names
        """
        return self.registry.list_extensions()


# Global plugin manager instance
_plugin_manager = PluginManager()


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance.
    
    Returns:
        Global plugin manager
    """
    return _plugin_manager


def initialize_plugin_system() -> None:
    """Initialize the plugin system.
    
    This is a convenience function that initializes and starts the plugin manager.
    """
    manager = get_plugin_manager()
    manager.start_plugins()


def shutdown_plugin_system() -> None:
    """Shut down the plugin system.
    
    This is a convenience function that stops the plugin manager.
    """
    manager = get_plugin_manager()
    manager.stop_plugins()


# Convenience functions for accessing plugin components

def get_tool_instance(name: str) -> Optional[Any]:
    """Get a tool instance by name.
    
    Args:
        name: Tool name
        
    Returns:
        Tool instance or None if not found
    """
    return get_plugin_manager().get_tool_instance(name)


def get_operation(name: str) -> Optional[Callable]:
    """Get a composite operation by name.
    
    Args:
        name: Operation name
        
    Returns:
        Operation function or None if not found
    """
    return get_plugin_manager().get_operation(name)


def get_analysis_function(name: str) -> Optional[Callable]:
    """Get an analysis function by name.
    
    Args:
        name: Function name
        
    Returns:
        Analysis function or None if not found
    """
    return get_plugin_manager().get_analysis_function(name)


def get_template(name: str) -> Optional[Any]:
    """Get a template by name.
    
    Args:
        name: Template name
        
    Returns:
        Template or None if not found
    """
    return get_plugin_manager().get_template(name)