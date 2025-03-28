"""
TSAP Plugin Registry.

This module provides a centralized registry for managing plugin components
and making them available to the rest of the TSAP system.
"""

from typing import Dict, List, Any, Optional, Type, Callable, Set, TypeVar

from tsap.utils.logging import logger


# Type variables for registry items
T = TypeVar('T')


class PluginRegistry:
    """Central registry for plugin components."""
    
    def __init__(self):
        """Initialize the plugin registry."""
        # Core tool components
        self.tool_classes: Dict[str, Type] = {}
        self.tool_instances: Dict[str, Any] = {}
        
        # Composite operation components
        self.operations: Dict[str, Callable] = {}
        
        # Analysis tool components
        self.analysis_tools: Dict[str, Type] = {}
        self.analysis_functions: Dict[str, Callable] = {}
        
        # Evolution components
        self.evolution_components: Dict[str, Any] = {}
        
        # Template components
        self.templates: Dict[str, Any] = {}
        
        # Format handlers
        self.input_formats: Dict[str, Any] = {}
        self.output_formats: Dict[str, Any] = {}
        
        # Integration handlers
        self.integrations: Dict[str, Any] = {}
        
        # UI components
        self.ui_components: Dict[str, Any] = {}
        
        # Utility components
        self.utilities: Dict[str, Any] = {}
        
        # Extension components
        self.extensions: Dict[str, Any] = {}
        
        # Mapping of components to plugins
        self.component_plugins: Dict[str, str] = {}
        
        # Keep track of registered plugins
        self.registered_plugins: Set[str] = set()
    
    def register_plugin_components(self, plugin_id: str, components: Dict[str, Dict[str, Any]]) -> None:
        """Register components from a plugin.
        
        Args:
            plugin_id: Plugin ID
            components: Dictionary mapping component types to components
        """
        # Register core tool components
        if "tool_classes" in components:
            for name, cls in components["tool_classes"].items():
                self.register_tool_class(name, cls, plugin_id)
        
        if "tool_instances" in components:
            for name, instance in components["tool_instances"].items():
                self.register_tool_instance(name, instance, plugin_id)
        
        # Register composite operations
        if "operations" in components:
            for name, operation in components["operations"].items():
                self.register_operation(name, operation, plugin_id)
        
        # Register analysis components
        if "analysis_tools" in components:
            for name, cls in components["analysis_tools"].items():
                self.register_analysis_tool(name, cls, plugin_id)
        
        if "analysis_functions" in components:
            for name, func in components["analysis_functions"].items():
                self.register_analysis_function(name, func, plugin_id)
        
        # Register evolution components
        if "evolution_components" in components:
            for name, component in components["evolution_components"].items():
                self.register_evolution_component(name, component, plugin_id)
        
        # Register templates
        if "templates" in components:
            for name, template in components["templates"].items():
                self.register_template(name, template, plugin_id)
        
        # Register format handlers
        if "input_formats" in components:
            for name, handler in components["input_formats"].items():
                self.register_input_format(name, handler, plugin_id)
        
        if "output_formats" in components:
            for name, handler in components["output_formats"].items():
                self.register_output_format(name, handler, plugin_id)
        
        # Register integrations
        if "integrations" in components:
            for name, handler in components["integrations"].items():
                self.register_integration(name, handler, plugin_id)
        
        # Register UI components
        if "ui_components" in components:
            for name, component in components["ui_components"].items():
                self.register_ui_component(name, component, plugin_id)
        
        # Register utilities
        if "utilities" in components:
            for name, utility in components["utilities"].items():
                self.register_utility(name, utility, plugin_id)
        
        # Register extensions
        if "extensions" in components:
            for name, extension in components["extensions"].items():
                self.register_extension(name, extension, plugin_id)
        
        # Mark plugin as registered
        self.registered_plugins.add(plugin_id)
    
    def unregister_plugin_components(self, plugin_id: str) -> None:
        """Unregister all components from a plugin.
        
        Args:
            plugin_id: Plugin ID
        """
        # Find all components registered by this plugin
        components_to_remove = [
            component_id for component_id, registered_plugin_id in self.component_plugins.items()
            if registered_plugin_id == plugin_id
        ]
        
        # Remove components
        for component_id in components_to_remove:
            parts = component_id.split(":", 1)
            if len(parts) != 2:
                continue
            
            component_type, component_name = parts
            
            # Remove from appropriate registry
            if component_type == "tool_class":
                self.unregister_tool_class(component_name)
            elif component_type == "tool_instance":
                self.unregister_tool_instance(component_name)
            elif component_type == "operation":
                self.unregister_operation(component_name)
            elif component_type == "analysis_tool":
                self.unregister_analysis_tool(component_name)
            elif component_type == "analysis_function":
                self.unregister_analysis_function(component_name)
            elif component_type == "evolution_component":
                self.unregister_evolution_component(component_name)
            elif component_type == "template":
                self.unregister_template(component_name)
            elif component_type == "input_format":
                self.unregister_input_format(component_name)
            elif component_type == "output_format":
                self.unregister_output_format(component_name)
            elif component_type == "integration":
                self.unregister_integration(component_name)
            elif component_type == "ui_component":
                self.unregister_ui_component(component_name)
            elif component_type == "utility":
                self.unregister_utility(component_name)
            elif component_type == "extension":
                self.unregister_extension(component_name)
        
        # Remove plugin from registered plugins
        if plugin_id in self.registered_plugins:
            self.registered_plugins.remove(plugin_id)
    
    def register_tool_class(self, name: str, cls: Type, plugin_id: str) -> None:
        """Register a tool class.
        
        Args:
            name: Tool name
            cls: Tool class
            plugin_id: Plugin ID
        """
        self.tool_classes[name] = cls
        self.component_plugins[f"tool_class:{name}"] = plugin_id
        logger.debug(f"Registered tool class '{name}' from plugin '{plugin_id}'")
    
    def unregister_tool_class(self, name: str) -> None:
        """Unregister a tool class.
        
        Args:
            name: Tool name
        """
        if name in self.tool_classes:
            del self.tool_classes[name]
            if f"tool_class:{name}" in self.component_plugins:
                del self.component_plugins[f"tool_class:{name}"]
            logger.debug(f"Unregistered tool class '{name}'")
    
    def register_tool_instance(self, name: str, instance: Any, plugin_id: str) -> None:
        """Register a tool instance.
        
        Args:
            name: Tool name
            instance: Tool instance
            plugin_id: Plugin ID
        """
        self.tool_instances[name] = instance
        self.component_plugins[f"tool_instance:{name}"] = plugin_id
        logger.debug(f"Registered tool instance '{name}' from plugin '{plugin_id}'")
    
    def unregister_tool_instance(self, name: str) -> None:
        """Unregister a tool instance.
        
        Args:
            name: Tool name
        """
        if name in self.tool_instances:
            del self.tool_instances[name]
            if f"tool_instance:{name}" in self.component_plugins:
                del self.component_plugins[f"tool_instance:{name}"]
            logger.debug(f"Unregistered tool instance '{name}'")
    
    def register_operation(self, name: str, operation: Callable, plugin_id: str) -> None:
        """Register a composite operation.
        
        Args:
            name: Operation name
            operation: Operation function
            plugin_id: Plugin ID
        """
        self.operations[name] = operation
        self.component_plugins[f"operation:{name}"] = plugin_id
        logger.debug(f"Registered operation '{name}' from plugin '{plugin_id}'")
    
    def unregister_operation(self, name: str) -> None:
        """Unregister a composite operation.
        
        Args:
            name: Operation name
        """
        if name in self.operations:
            del self.operations[name]
            if f"operation:{name}" in self.component_plugins:
                del self.component_plugins[f"operation:{name}"]
            logger.debug(f"Unregistered operation '{name}'")
    
    def register_analysis_tool(self, name: str, cls: Type, plugin_id: str) -> None:
        """Register an analysis tool class.
        
        Args:
            name: Tool name
            cls: Tool class
            plugin_id: Plugin ID
        """
        self.analysis_tools[name] = cls
        self.component_plugins[f"analysis_tool:{name}"] = plugin_id
        logger.debug(f"Registered analysis tool '{name}' from plugin '{plugin_id}'")
    
    def unregister_analysis_tool(self, name: str) -> None:
        """Unregister an analysis tool class.
        
        Args:
            name: Tool name
        """
        if name in self.analysis_tools:
            del self.analysis_tools[name]
            if f"analysis_tool:{name}" in self.component_plugins:
                del self.component_plugins[f"analysis_tool:{name}"]
            logger.debug(f"Unregistered analysis tool '{name}'")
    
    def register_analysis_function(self, name: str, function: Callable, plugin_id: str) -> None:
        """Register an analysis function.
        
        Args:
            name: Function name
            function: Analysis function
            plugin_id: Plugin ID
        """
        self.analysis_functions[name] = function
        self.component_plugins[f"analysis_function:{name}"] = plugin_id
        logger.debug(f"Registered analysis function '{name}' from plugin '{plugin_id}'")
    
    def unregister_analysis_function(self, name: str) -> None:
        """Unregister an analysis function.
        
        Args:
            name: Function name
        """
        if name in self.analysis_functions:
            del self.analysis_functions[name]
            if f"analysis_function:{name}" in self.component_plugins:
                del self.component_plugins[f"analysis_function:{name}"]
            logger.debug(f"Unregistered analysis function '{name}'")
    
    def register_evolution_component(self, name: str, component: Any, plugin_id: str) -> None:
        """Register an evolution component.
        
        Args:
            name: Component name
            component: Evolution component
            plugin_id: Plugin ID
        """
        self.evolution_components[name] = component
        self.component_plugins[f"evolution_component:{name}"] = plugin_id
        logger.debug(f"Registered evolution component '{name}' from plugin '{plugin_id}'")
    
    def unregister_evolution_component(self, name: str) -> None:
        """Unregister an evolution component.
        
        Args:
            name: Component name
        """
        if name in self.evolution_components:
            del self.evolution_components[name]
            if f"evolution_component:{name}" in self.component_plugins:
                del self.component_plugins[f"evolution_component:{name}"]
            logger.debug(f"Unregistered evolution component '{name}'")
    
    def register_template(self, name: str, template: Any, plugin_id: str) -> None:
        """Register a template.
        
        Args:
            name: Template name
            template: Template object
            plugin_id: Plugin ID
        """
        self.templates[name] = template
        self.component_plugins[f"template:{name}"] = plugin_id
        logger.debug(f"Registered template '{name}' from plugin '{plugin_id}'")
    
    def unregister_template(self, name: str) -> None:
        """Unregister a template.
        
        Args:
            name: Template name
        """
        if name in self.templates:
            del self.templates[name]
            if f"template:{name}" in self.component_plugins:
                del self.component_plugins[f"template:{name}"]
            logger.debug(f"Unregistered template '{name}'")
    
    def register_input_format(self, name: str, handler: Any, plugin_id: str) -> None:
        """Register an input format handler.
        
        Args:
            name: Format name
            handler: Format handler
            plugin_id: Plugin ID
        """
        self.input_formats[name] = handler
        self.component_plugins[f"input_format:{name}"] = plugin_id
        logger.debug(f"Registered input format '{name}' from plugin '{plugin_id}'")
    
    def unregister_input_format(self, name: str) -> None:
        """Unregister an input format handler.
        
        Args:
            name: Format name
        """
        if name in self.input_formats:
            del self.input_formats[name]
            if f"input_format:{name}" in self.component_plugins:
                del self.component_plugins[f"input_format:{name}"]
            logger.debug(f"Unregistered input format '{name}'")
    
    def register_output_format(self, name: str, handler: Any, plugin_id: str) -> None:
        """Register an output format handler.
        
        Args:
            name: Format name
            handler: Format handler
            plugin_id: Plugin ID
        """
        self.output_formats[name] = handler
        self.component_plugins[f"output_format:{name}"] = plugin_id
        logger.debug(f"Registered output format '{name}' from plugin '{plugin_id}'")
    
    def unregister_output_format(self, name: str) -> None:
        """Unregister an output format handler.
        
        Args:
            name: Format name
        """
        if name in self.output_formats:
            del self.output_formats[name]
            if f"output_format:{name}" in self.component_plugins:
                del self.component_plugins[f"output_format:{name}"]
            logger.debug(f"Unregistered output format '{name}'")
    
    def register_integration(self, name: str, handler: Any, plugin_id: str) -> None:
        """Register an integration handler.
        
        Args:
            name: Integration name
            handler: Integration handler
            plugin_id: Plugin ID
        """
        self.integrations[name] = handler
        self.component_plugins[f"integration:{name}"] = plugin_id
        logger.debug(f"Registered integration '{name}' from plugin '{plugin_id}'")
    
    def unregister_integration(self, name: str) -> None:
        """Unregister an integration handler.
        
        Args:
            name: Integration name
        """
        if name in self.integrations:
            del self.integrations[name]
            if f"integration:{name}" in self.component_plugins:
                del self.component_plugins[f"integration:{name}"]
            logger.debug(f"Unregistered integration '{name}'")
    
    def register_ui_component(self, name: str, component: Any, plugin_id: str) -> None:
        """Register a UI component.
        
        Args:
            name: Component name
            component: UI component
            plugin_id: Plugin ID
        """
        self.ui_components[name] = component
        self.component_plugins[f"ui_component:{name}"] = plugin_id
        logger.debug(f"Registered UI component '{name}' from plugin '{plugin_id}'")
    
    def unregister_ui_component(self, name: str) -> None:
        """Unregister a UI component.
        
        Args:
            name: Component name
        """
        if name in self.ui_components:
            del self.ui_components[name]
            if f"ui_component:{name}" in self.component_plugins:
                del self.component_plugins[f"ui_component:{name}"]
            logger.debug(f"Unregistered UI component '{name}'")
    
    def register_utility(self, name: str, utility: Any, plugin_id: str) -> None:
        """Register a utility.
        
        Args:
            name: Utility name
            utility: Utility object or function
            plugin_id: Plugin ID
        """
        self.utilities[name] = utility
        self.component_plugins[f"utility:{name}"] = plugin_id
        logger.debug(f"Registered utility '{name}' from plugin '{plugin_id}'")
    
    def unregister_utility(self, name: str) -> None:
        """Unregister a utility.
        
        Args:
            name: Utility name
        """
        if name in self.utilities:
            del self.utilities[name]
            if f"utility:{name}" in self.component_plugins:
                del self.component_plugins[f"utility:{name}"]
            logger.debug(f"Unregistered utility '{name}'")
    
    def register_extension(self, name: str, extension: Any, plugin_id: str) -> None:
        """Register an extension.
        
        Args:
            name: Extension name
            extension: Extension object
            plugin_id: Plugin ID
        """
        self.extensions[name] = extension
        self.component_plugins[f"extension:{name}"] = plugin_id
        logger.debug(f"Registered extension '{name}' from plugin '{plugin_id}'")
    
    def unregister_extension(self, name: str) -> None:
        """Unregister an extension.
        
        Args:
            name: Extension name
        """
        if name in self.extensions:
            del self.extensions[name]
            if f"extension:{name}" in self.component_plugins:
                del self.component_plugins[f"extension:{name}"]
            logger.debug(f"Unregistered extension '{name}'")
    
    def get_tool_class(self, name: str) -> Optional[Type]:
        """Get a tool class by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool class or None if not found
        """
        return self.tool_classes.get(name)
    
    def get_tool_instance(self, name: str) -> Optional[Any]:
        """Get a tool instance by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool instance or None if not found
        """
        return self.tool_instances.get(name)
    
    def get_operation(self, name: str) -> Optional[Callable]:
        """Get a composite operation by name.
        
        Args:
            name: Operation name
            
        Returns:
            Operation function or None if not found
        """
        return self.operations.get(name)
    
    def get_analysis_tool(self, name: str) -> Optional[Type]:
        """Get an analysis tool class by name.
        
        Args:
            name: Tool name
            
        Returns:
            Analysis tool class or None if not found
        """
        return self.analysis_tools.get(name)
    
    def get_analysis_function(self, name: str) -> Optional[Callable]:
        """Get an analysis function by name.
        
        Args:
            name: Function name
            
        Returns:
            Analysis function or None if not found
        """
        return self.analysis_functions.get(name)
    
    def get_evolution_component(self, name: str) -> Optional[Any]:
        """Get an evolution component by name.
        
        Args:
            name: Component name
            
        Returns:
            Evolution component or None if not found
        """
        return self.evolution_components.get(name)
    
    def get_template(self, name: str) -> Optional[Any]:
        """Get a template by name.
        
        Args:
            name: Template name
            
        Returns:
            Template or None if not found
        """
        return self.templates.get(name)
    
    def get_input_format(self, name: str) -> Optional[Any]:
        """Get an input format handler by name.
        
        Args:
            name: Format name
            
        Returns:
            Input format handler or None if not found
        """
        return self.input_formats.get(name)
    
    def get_output_format(self, name: str) -> Optional[Any]:
        """Get an output format handler by name.
        
        Args:
            name: Format name
            
        Returns:
            Output format handler or None if not found
        """
        return self.output_formats.get(name)
    
    def get_integration(self, name: str) -> Optional[Any]:
        """Get an integration handler by name.
        
        Args:
            name: Integration name
            
        Returns:
            Integration handler or None if not found
        """
        return self.integrations.get(name)
    
    def get_ui_component(self, name: str) -> Optional[Any]:
        """Get a UI component by name.
        
        Args:
            name: Component name
            
        Returns:
            UI component or None if not found
        """
        return self.ui_components.get(name)
    
    def get_utility(self, name: str) -> Optional[Any]:
        """Get a utility by name.
        
        Args:
            name: Utility name
            
        Returns:
            Utility or None if not found
        """
        return self.utilities.get(name)
    
    def get_extension(self, name: str) -> Optional[Any]:
        """Get an extension by name.
        
        Args:
            name: Extension name
            
        Returns:
            Extension or None if not found
        """
        return self.extensions.get(name)
    
    def get_component_plugin(self, component_type: str, name: str) -> Optional[str]:
        """Get the plugin ID that registered a component.
        
        Args:
            component_type: Component type (e.g., "tool_class", "operation")
            name: Component name
            
        Returns:
            Plugin ID or None if component not found
        """
        component_id = f"{component_type}:{name}"
        return self.component_plugins.get(component_id)
    
    def list_tool_classes(self) -> List[str]:
        """Get a list of all registered tool classes.
        
        Returns:
            List of tool class names
        """
        return list(self.tool_classes.keys())
    
    def list_tool_instances(self) -> List[str]:
        """Get a list of all registered tool instances.
        
        Returns:
            List of tool instance names
        """
        return list(self.tool_instances.keys())
    
    def list_operations(self) -> List[str]:
        """Get a list of all registered composite operations.
        
        Returns:
            List of operation names
        """
        return list(self.operations.keys())
    
    def list_analysis_tools(self) -> List[str]:
        """Get a list of all registered analysis tools.
        
        Returns:
            List of analysis tool names
        """
        return list(self.analysis_tools.keys())
    
    def list_analysis_functions(self) -> List[str]:
        """Get a list of all registered analysis functions.
        
        Returns:
            List of analysis function names
        """
        return list(self.analysis_functions.keys())
    
    def list_evolution_components(self) -> List[str]:
        """Get a list of all registered evolution components.
        
        Returns:
            List of evolution component names
        """
        return list(self.evolution_components.keys())
    
    def list_templates(self) -> List[str]:
        """Get a list of all registered templates.
        
        Returns:
            List of template names
        """
        return list(self.templates.keys())
    
    def list_input_formats(self) -> List[str]:
        """Get a list of all registered input formats.
        
        Returns:
            List of input format names
        """
        return list(self.input_formats.keys())
    
    def list_output_formats(self) -> List[str]:
        """Get a list of all registered output formats.
        
        Returns:
            List of output format names
        """
        return list(self.output_formats.keys())
    
    def list_integrations(self) -> List[str]:
        """Get a list of all registered integrations.
        
        Returns:
            List of integration names
        """
        return list(self.integrations.keys())
    
    def list_ui_components(self) -> List[str]:
        """Get a list of all registered UI components.
        
        Returns:
            List of UI component names
        """
        return list(self.ui_components.keys())
    
    def list_utilities(self) -> List[str]:
        """Get a list of all registered utilities.
        
        Returns:
            List of utility names
        """
        return list(self.utilities.keys())
    
    def list_extensions(self) -> List[str]:
        """Get a list of all registered extensions.
        
        Returns:
            List of extension names
        """
        return list(self.extensions.keys())
    
    def list_registered_plugins(self) -> List[str]:
        """Get a list of all registered plugin IDs.
        
        Returns:
            List of plugin IDs
        """
        return list(self.registered_plugins)


# Global registry instance
_registry = PluginRegistry()


def get_registry() -> PluginRegistry:
    """Get the global plugin registry instance.
    
    Returns:
        Global plugin registry
    """
    return _registry