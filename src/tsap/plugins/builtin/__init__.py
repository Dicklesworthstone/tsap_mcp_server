"""
TSAP Built-in Plugins Package.

This package contains built-in plugins that are included with TSAP by default.
These plugins provide core functionality and serve as examples for plugin development.
"""

# Import all built-in plugins to make them discoverable
from tsap.plugins.builtin.example import ExamplePlugin

# Add new built-in plugins here as they are implemented
__all__ = [
    "ExamplePlugin",
]