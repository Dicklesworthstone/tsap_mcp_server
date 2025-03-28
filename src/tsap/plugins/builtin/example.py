"""
TSAP Example Plugin.

This is a simple example plugin that demonstrates the basic structure of a TSAP plugin.
It provides a utility function and a simple tool for demonstration purposes.
"""

from typing import Dict, Any

from tsap.utils.logging import logger
from tsap.plugins.interface import (
    UtilityPlugin,
    plugin_info,
    plugin_capabilities,
    plugin_dependencies,
    PluginCapability,
)
from tsap.core.base import BaseCoreTool


@plugin_info(
    name="Example Plugin",
    version="1.0.0",
    description="A simple example plugin for TSAP",
    author="TSAP Team",
    plugin_id="example-plugin",
)
@plugin_capabilities(
    PluginCapability.UTILITY,
)
@plugin_dependencies()
class ExamplePlugin(UtilityPlugin):
    """Example TSAP plugin that provides utility functions."""
    
    def __init__(self):
        """Initialize the example plugin."""
        logger.debug("Initializing ExamplePlugin")
        self.initialized = False
        self.active = False
    
    def initialize(self) -> None:
        """Initialize the plugin.
        
        This is called when the plugin is loaded.
        """
        logger.debug("Example plugin initializing")
        self.initialized = True
    
    def register(self) -> None:
        """Register the plugin with the TSAP system.
        
        This is called after initialization to register any components.
        """
        logger.debug("Example plugin registering")
        self.active = True
    
    def shutdown(self) -> None:
        """Shut down the plugin.
        
        This is called when the plugin is being unloaded or when TSAP is shutting down.
        """
        logger.debug("Example plugin shutting down")
        self.active = False
        self.initialized = False
    
    def get_utilities(self) -> Dict[str, Any]:
        """Get utilities provided by this plugin.
        
        Returns:
            Dictionary mapping utility names to utility objects or functions
        """
        return {
            "example_utility": self.example_utility,
            "example_tool": ExampleTool(),
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get plugin status.
        
        Returns:
            Dictionary with plugin status information
        """
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "active": self.active,
            "initialized": self.initialized,
        }
    
    def example_utility(self, text: str) -> str:
        """Example utility function.
        
        Args:
            text: Input text
            
        Returns:
            Processed text
        """
        return f"Example plugin processed: {text}"


class ExampleTool(BaseCoreTool):
    """Example tool provided by the example plugin."""
    
    def __init__(self):
        """Initialize the example tool."""
        super().__init__("example_tool")
    
    async def process(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process data with the example tool.
        
        Args:
            params: Processing parameters
            
        Returns:
            Processing results
        """
        async with self._measure_execution_time():
            # Extract parameters
            input_text = params.get("input_text", "")
            operation = params.get("operation", "uppercase")
            
            # Process based on operation
            if operation == "uppercase":
                result = input_text.upper()
            elif operation == "lowercase":
                result = input_text.lower()
            elif operation == "reverse":
                result = input_text[::-1]
            else:
                result = input_text
            
            # Return results
            return {
                "input": input_text,
                "operation": operation,
                "result": result,
                "status": "success",
            }


# Async wrapper for the example tool
async def example_tool_process(params: Dict[str, Any]) -> Dict[str, Any]:
    """Process data with the example tool.
    
    Args:
        params: Processing parameters
        
    Returns:
        Processing results
    """
    tool = ExampleTool()
    return await tool.process(params)