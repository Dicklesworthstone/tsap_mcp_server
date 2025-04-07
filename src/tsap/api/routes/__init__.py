"""
API routes package for the TSAP ToolAPI Server API.

This package contains route definitions for the REST API endpoints
organized by functional area.
"""

# Explicitly import all routers to ensure they're available
from . import core
from . import composite
from . import analysis
from . import evolution
from . import plugins

# This file intentionally left mostly empty as it's just a package marker.