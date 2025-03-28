"""
TSAP Plugins Package.

This package provides a plugin system for extending TSAP functionality through
a modular architecture. Plugins can add new tools, operations, analysis methods,
and other components to the system.
"""

from tsap.plugins.interface import (
    Plugin,
    PluginType,
    PluginCapability,
    CoreToolPlugin,
    CompositePlugin,
    AnalysisPlugin,
    EvolutionPlugin,
    TemplatePlugin,
    FormatPlugin,
    IntegrationPlugin,
    UIPlugin, 
    UtilityPlugin,
    ExtensionPlugin,
    plugin_info,
    plugin_capabilities,
    plugin_dependencies,
)

from tsap.plugins.manager import (
    get_plugin_manager,
    initialize_plugin_system,
    shutdown_plugin_system,
    get_tool_instance,
    get_operation,
    get_analysis_function,
    get_template,
)

from tsap.plugins.loader import (
    get_plugin_loader,
    install_plugin,
    uninstall_plugin,
)

from tsap.plugins.registry import get_registry

# Initialize the plugin system on import if auto-loading is enabled
from tsap.config import get_config

# Don't auto-initialize if running in a test environment 
import os
import sys

if not os.environ.get("TSAP_DISABLE_PLUGIN_AUTOLOAD") and "pytest" not in sys.modules:
    config = get_config()
    if getattr(config.plugins, "auto_load", False):
        try:
            initialize_plugin_system()
        except Exception as e:
            from tsap.utils.logging import logger
            logger.warning(f"Error auto-initializing plugin system: {e}")
            logger.warning("Plugin system will need to be initialized manually")


__all__ = [
    # Plugin interfaces
    "Plugin",
    "PluginType",
    "PluginCapability",
    "CoreToolPlugin",
    "CompositePlugin",
    "AnalysisPlugin",
    "EvolutionPlugin",
    "TemplatePlugin",
    "FormatPlugin",
    "IntegrationPlugin",
    "UIPlugin",
    "UtilityPlugin",
    "ExtensionPlugin",
    
    # Plugin decorators
    "plugin_info",
    "plugin_capabilities",
    "plugin_dependencies",
    
    # Plugin management
    "get_plugin_manager",
    "initialize_plugin_system",
    "shutdown_plugin_system",
    
    # Plugin loader
    "get_plugin_loader",
    "install_plugin",
    "uninstall_plugin",
    
    # Plugin registry
    "get_registry",
    
    # Convenience functions
    "get_tool_instance",
    "get_operation",
    "get_analysis_function",
    "get_template",
]