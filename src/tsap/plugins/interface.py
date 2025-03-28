"""
TSAP Plugin Interface.

This module defines the base interfaces and abstractions for TSAP plugins,
enabling third-party extensions to the core TSAP functionality.
"""

import inspect
import importlib
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable, Type, ClassVar

from tsap.utils.logging import logger
from tsap.utils.helpers import generate_id


class PluginType(str, Enum):
    """Types of plugins supported by TSAP."""
    
    CORE_TOOL = "core_tool"            # Extends Layer 1 core tools
    COMPOSITE = "composite"            # Extends Layer 2 composite operations
    ANALYSIS = "analysis"              # Extends Layer 3 analysis tools
    EVOLUTION = "evolution"            # Extends Evolution & Learning systems
    TEMPLATE = "template"              # Provides task templates
    FORMAT = "format"                  # Provides support for additional formats
    INTEGRATION = "integration"        # Integrates with external systems
    UI = "ui"                          # Extends user interface
    UTILITY = "utility"                # Provides general utilities
    EXTENSION = "extension"            # General extension


class PluginCapability(str, Enum):
    """Capabilities that plugins can provide or extend."""
    
    SEARCH = "search"                  # Text searching capability
    TRANSFORMATION = "transformation"  # Data transformation
    EXTRACTION = "extraction"          # Data extraction
    ANALYSIS = "analysis"              # Data analysis
    VISUALIZATION = "visualization"    # Data visualization
    FORMAT = "format"                  # Data format support
    PROTOCOL = "protocol"              # Communication protocol
    INTEGRATION = "integration"        # External system integration
    STORAGE = "storage"                # Data storage
    WORKFLOW = "workflow"              # Workflow automation
    UI = "ui"                          # User interface enhancement


class Plugin(ABC):
    """Base interface for all TSAP plugins."""
    
    # Class attributes
    id: ClassVar[str]
    name: ClassVar[str]
    version: ClassVar[str]
    description: ClassVar[str]
    author: ClassVar[str]
    plugin_type: ClassVar[PluginType]
    capabilities: ClassVar[List[PluginCapability]]
    dependencies: ClassVar[List[str]]
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the plugin.
        
        This method is called once when the plugin is loaded.
        """
        pass
    
    @abstractmethod
    def register(self) -> None:
        """Register the plugin with the TSAP system.
        
        This method is called after initialization to register any
        components, tools, or other functionality with the system.
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Shut down the plugin.
        
        This method is called when the plugin is being unloaded
        or when the TSAP system is shutting down.
        """
        pass
    
    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        """Get plugin metadata.
        
        Returns:
            Dictionary with plugin metadata
        """
        return {
            "id": cls.id,
            "name": cls.name,
            "version": cls.version,
            "description": cls.description,
            "author": cls.author,
            "plugin_type": cls.plugin_type,
            "capabilities": cls.capabilities,
            "dependencies": cls.dependencies,
            "class": cls.__name__,
            "module": cls.__module__,
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get plugin status.
        
        Returns:
            Dictionary with plugin status information
            
        Note:
            Subclasses can override this method to provide additional status information.
        """
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "active": True,
        }
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate plugin configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation error messages (empty if valid)
            
        Note:
            Subclasses can override this method to provide custom validation.
        """
        return []


class CoreToolPlugin(Plugin):
    """Base interface for plugins that extend Layer 1 core tools."""
    
    plugin_type = PluginType.CORE_TOOL
    
    @abstractmethod
    def get_tool_classes(self) -> Dict[str, Type]:
        """Get core tool classes provided by this plugin.
        
        Returns:
            Dictionary mapping tool names to tool classes
        """
        pass
    
    @abstractmethod
    def get_tool_instances(self) -> Dict[str, Any]:
        """Get core tool instances provided by this plugin.
        
        Returns:
            Dictionary mapping tool names to tool instances
        """
        pass


class CompositePlugin(Plugin):
    """Base interface for plugins that extend Layer 2 composite operations."""
    
    plugin_type = PluginType.COMPOSITE
    
    @abstractmethod
    def get_operations(self) -> Dict[str, Callable]:
        """Get composite operations provided by this plugin.
        
        Returns:
            Dictionary mapping operation names to operation functions
        """
        pass


class AnalysisPlugin(Plugin):
    """Base interface for plugins that extend Layer 3 analysis tools."""
    
    plugin_type = PluginType.ANALYSIS
    
    @abstractmethod
    def get_analysis_tools(self) -> Dict[str, Type]:
        """Get analysis tool classes provided by this plugin.
        
        Returns:
            Dictionary mapping tool names to tool classes
        """
        pass
    
    @abstractmethod
    def get_analysis_functions(self) -> Dict[str, Callable]:
        """Get analysis functions provided by this plugin.
        
        Returns:
            Dictionary mapping function names to analysis functions
        """
        pass


class EvolutionPlugin(Plugin):
    """Base interface for plugins that extend Evolution & Learning systems."""
    
    plugin_type = PluginType.EVOLUTION
    
    @abstractmethod
    def get_evolution_components(self) -> Dict[str, Any]:
        """Get evolution components provided by this plugin.
        
        Returns:
            Dictionary mapping component names to component objects
        """
        pass


class TemplatePlugin(Plugin):
    """Base interface for plugins that provide task templates."""
    
    plugin_type = PluginType.TEMPLATE
    
    @abstractmethod
    def get_templates(self) -> Dict[str, Any]:
        """Get templates provided by this plugin.
        
        Returns:
            Dictionary mapping template names to template objects
        """
        pass


class FormatPlugin(Plugin):
    """Base interface for plugins that provide support for additional formats."""
    
    plugin_type = PluginType.FORMAT
    
    @abstractmethod
    def get_input_formats(self) -> Dict[str, Any]:
        """Get input format handlers provided by this plugin.
        
        Returns:
            Dictionary mapping format names to handler objects
        """
        pass
    
    @abstractmethod
    def get_output_formats(self) -> Dict[str, Any]:
        """Get output format handlers provided by this plugin.
        
        Returns:
            Dictionary mapping format names to handler objects
        """
        pass


class IntegrationPlugin(Plugin):
    """Base interface for plugins that integrate with external systems."""
    
    plugin_type = PluginType.INTEGRATION
    
    @abstractmethod
    def get_integrations(self) -> Dict[str, Any]:
        """Get integration handlers provided by this plugin.
        
        Returns:
            Dictionary mapping integration names to handler objects
        """
        pass


class UIPlugin(Plugin):
    """Base interface for plugins that extend the user interface."""
    
    plugin_type = PluginType.UI
    
    @abstractmethod
    def get_ui_components(self) -> Dict[str, Any]:
        """Get UI components provided by this plugin.
        
        Returns:
            Dictionary mapping component names to component objects
        """
        pass


class UtilityPlugin(Plugin):
    """Base interface for plugins that provide general utilities."""
    
    plugin_type = PluginType.UTILITY
    
    @abstractmethod
    def get_utilities(self) -> Dict[str, Any]:
        """Get utilities provided by this plugin.
        
        Returns:
            Dictionary mapping utility names to utility objects or functions
        """
        pass


class ExtensionPlugin(Plugin):
    """Base interface for general extension plugins."""
    
    plugin_type = PluginType.EXTENSION
    
    @abstractmethod
    def get_extensions(self) -> Dict[str, Any]:
        """Get extensions provided by this plugin.
        
        Returns:
            Dictionary mapping extension names to extension objects
        """
        pass


# Plugin metadata decorators

def plugin_info(
    name: str,
    version: str,
    description: str,
    author: str,
    plugin_id: Optional[str] = None,
) -> Callable[[Type[Plugin]], Type[Plugin]]:
    """Class decorator to set basic plugin information.
    
    Args:
        name: Plugin name
        version: Plugin version
        description: Plugin description
        author: Plugin author
        plugin_id: Optional plugin ID (generated if not provided)
        
    Returns:
        Class decorator function
    """
    def decorator(cls: Type[Plugin]) -> Type[Plugin]:
        cls.name = name
        cls.version = version
        cls.description = description
        cls.author = author
        cls.id = plugin_id or generate_id(f"{name.lower().replace(' ', '-')}-")
        return cls
    
    return decorator


def plugin_capabilities(*capabilities: Union[str, PluginCapability]) -> Callable[[Type[Plugin]], Type[Plugin]]:
    """Class decorator to set plugin capabilities.
    
    Args:
        *capabilities: Plugin capabilities
        
    Returns:
        Class decorator function
    """
    def decorator(cls: Type[Plugin]) -> Type[Plugin]:
        # Convert string capabilities to enum values
        enum_capabilities = []
        for capability in capabilities:
            if isinstance(capability, str):
                try:
                    capability = PluginCapability(capability)
                except ValueError:
                    raise ValueError(f"Invalid capability: {capability}")
            enum_capabilities.append(capability)
        
        cls.capabilities = enum_capabilities
        return cls
    
    return decorator


def plugin_dependencies(*dependencies: str) -> Callable[[Type[Plugin]], Type[Plugin]]:
    """Class decorator to set plugin dependencies.
    
    Args:
        *dependencies: Plugin dependencies (other plugin IDs)
        
    Returns:
        Class decorator function
    """
    def decorator(cls: Type[Plugin]) -> Type[Plugin]:
        cls.dependencies = list(dependencies)
        return cls
    
    return decorator


# Plugin discovery and validation

def is_plugin_class(cls: Any) -> bool:
    """Check if a class is a TSAP plugin.
    
    Args:
        cls: Class to check
        
    Returns:
        True if class is a plugin, False otherwise
    """
    return (
        inspect.isclass(cls) and
        issubclass(cls, Plugin) and
        cls is not Plugin and
        not inspect.isabstract(cls) and
        hasattr(cls, "id") and
        hasattr(cls, "name") and
        hasattr(cls, "version")
    )


def discover_plugins_in_module(module: Any) -> List[Type[Plugin]]:
    """Discover plugin classes in a module.
    
    Args:
        module: Module to search
        
    Returns:
        List of plugin classes
    """
    return [
        cls for _, cls in inspect.getmembers(module, is_plugin_class)
    ]


def discover_plugins_in_package(package_name: str) -> List[Type[Plugin]]:
    """Discover plugin classes in a package.
    
    Args:
        package_name: Package name
        
    Returns:
        List of plugin classes
    """
    try:
        package = importlib.import_module(package_name)
    except ImportError as e:
        logger.error(f"Failed to import package '{package_name}': {e}")
        return []
    
    plugins = discover_plugins_in_module(package)
    
    # Recursively discover plugins in subpackages
    if hasattr(package, "__path__"):
        for _, name, is_pkg in importlib.util.iter_modules(package.__path__, package.__name__ + "."):
            if is_pkg:
                # Discover plugins in subpackage
                plugins.extend(discover_plugins_in_package(name))
            else:
                # Discover plugins in module
                try:
                    module = importlib.import_module(name)
                    plugins.extend(discover_plugins_in_module(module))
                except ImportError as e:
                    logger.error(f"Failed to import module '{name}': {e}")
    
    return plugins


def validate_plugin_class(plugin_class: Type[Plugin]) -> List[str]:
    """Validate that a plugin class meets all requirements.
    
    Args:
        plugin_class: Plugin class to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Check required class attributes
    for attr in ["id", "name", "version", "description", "author", "plugin_type", "capabilities"]:
        if not hasattr(plugin_class, attr):
            errors.append(f"Missing required attribute: {attr}")
    
    # Check that plugin_type is valid
    if hasattr(plugin_class, "plugin_type"):
        plugin_type = plugin_class.plugin_type
        if not isinstance(plugin_type, PluginType):
            errors.append(f"plugin_type must be a PluginType enum value, got: {type(plugin_type)}")
    
    # Check that capabilities is a list of PluginCapability
    if hasattr(plugin_class, "capabilities"):
        capabilities = plugin_class.capabilities
        if not isinstance(capabilities, list):
            errors.append(f"capabilities must be a list, got: {type(capabilities)}")
        else:
            for capability in capabilities:
                if not isinstance(capability, PluginCapability):
                    errors.append(f"capabilities must contain PluginCapability enum values, got: {type(capability)}")
    
    # Check required methods
    for method_name in ["initialize", "register", "shutdown"]:
        if not hasattr(plugin_class, method_name) or not callable(getattr(plugin_class, method_name)):
            errors.append(f"Missing required method: {method_name}")
    
    # Check plugin type-specific requirements
    if hasattr(plugin_class, "plugin_type"):
        plugin_type = plugin_class.plugin_type
        
        if plugin_type == PluginType.CORE_TOOL:
            if not issubclass(plugin_class, CoreToolPlugin):
                errors.append(f"Plugin with type {plugin_type} must inherit from CoreToolPlugin")
            elif not hasattr(plugin_class, "get_tool_classes") or not callable(getattr(plugin_class, "get_tool_classes")):
                errors.append(f"Missing required method for {plugin_type} plugin: get_tool_classes")
            elif not hasattr(plugin_class, "get_tool_instances") or not callable(getattr(plugin_class, "get_tool_instances")):
                errors.append(f"Missing required method for {plugin_type} plugin: get_tool_instances")
                
        elif plugin_type == PluginType.COMPOSITE:
            if not issubclass(plugin_class, CompositePlugin):
                errors.append(f"Plugin with type {plugin_type} must inherit from CompositePlugin")
            elif not hasattr(plugin_class, "get_operations") or not callable(getattr(plugin_class, "get_operations")):
                errors.append(f"Missing required method for {plugin_type} plugin: get_operations")
                
        elif plugin_type == PluginType.ANALYSIS:
            if not issubclass(plugin_class, AnalysisPlugin):
                errors.append(f"Plugin with type {plugin_type} must inherit from AnalysisPlugin")
            elif not hasattr(plugin_class, "get_analysis_tools") or not callable(getattr(plugin_class, "get_analysis_tools")):
                errors.append(f"Missing required method for {plugin_type} plugin: get_analysis_tools")
            elif not hasattr(plugin_class, "get_analysis_functions") or not callable(getattr(plugin_class, "get_analysis_functions")):
                errors.append(f"Missing required method for {plugin_type} plugin: get_analysis_functions")
                
        elif plugin_type == PluginType.EVOLUTION:
            if not issubclass(plugin_class, EvolutionPlugin):
                errors.append(f"Plugin with type {plugin_type} must inherit from EvolutionPlugin")
            elif not hasattr(plugin_class, "get_evolution_components") or not callable(getattr(plugin_class, "get_evolution_components")):
                errors.append(f"Missing required method for {plugin_type} plugin: get_evolution_components")
                
        elif plugin_type == PluginType.TEMPLATE:
            if not issubclass(plugin_class, TemplatePlugin):
                errors.append(f"Plugin with type {plugin_type} must inherit from TemplatePlugin")
            elif not hasattr(plugin_class, "get_templates") or not callable(getattr(plugin_class, "get_templates")):
                errors.append(f"Missing required method for {plugin_type} plugin: get_templates")
                
        elif plugin_type == PluginType.FORMAT:
            if not issubclass(plugin_class, FormatPlugin):
                errors.append(f"Plugin with type {plugin_type} must inherit from FormatPlugin")
            elif not hasattr(plugin_class, "get_input_formats") or not callable(getattr(plugin_class, "get_input_formats")):
                errors.append(f"Missing required method for {plugin_type} plugin: get_input_formats")
            elif not hasattr(plugin_class, "get_output_formats") or not callable(getattr(plugin_class, "get_output_formats")):
                errors.append(f"Missing required method for {plugin_type} plugin: get_output_formats")
                
        elif plugin_type == PluginType.INTEGRATION:
            if not issubclass(plugin_class, IntegrationPlugin):
                errors.append(f"Plugin with type {plugin_type} must inherit from IntegrationPlugin")
            elif not hasattr(plugin_class, "get_integrations") or not callable(getattr(plugin_class, "get_integrations")):
                errors.append(f"Missing required method for {plugin_type} plugin: get_integrations")
                
        elif plugin_type == PluginType.UI:
            if not issubclass(plugin_class, UIPlugin):
                errors.append(f"Plugin with type {plugin_type} must inherit from UIPlugin")
            elif not hasattr(plugin_class, "get_ui_components") or not callable(getattr(plugin_class, "get_ui_components")):
                errors.append(f"Missing required method for {plugin_type} plugin: get_ui_components")
                
        elif plugin_type == PluginType.UTILITY:
            if not issubclass(plugin_class, UtilityPlugin):
                errors.append(f"Plugin with type {plugin_type} must inherit from UtilityPlugin")
            elif not hasattr(plugin_class, "get_utilities") or not callable(getattr(plugin_class, "get_utilities")):
                errors.append(f"Missing required method for {plugin_type} plugin: get_utilities")
                
        elif plugin_type == PluginType.EXTENSION:
            if not issubclass(plugin_class, ExtensionPlugin):
                errors.append(f"Plugin with type {plugin_type} must inherit from ExtensionPlugin")
            elif not hasattr(plugin_class, "get_extensions") or not callable(getattr(plugin_class, "get_extensions")):
                errors.append(f"Missing required method for {plugin_type} plugin: get_extensions")
    
    return errors


def create_plugin_instance(plugin_class: Type[Plugin]) -> Optional[Plugin]:
    """Create an instance of a plugin class.
    
    Args:
        plugin_class: Plugin class to instantiate
        
    Returns:
        Plugin instance or None if instantiation fails
    """
    try:
        return plugin_class()
    except Exception as e:
        logger.error(f"Failed to instantiate plugin {plugin_class.__name__}: {e}")
        return None