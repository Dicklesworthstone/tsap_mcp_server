"""
Version information for TSAP MCP Server.
"""

# Version of the TSAP package
__version__ = "0.1.0"

# Protocol version
PROTOCOL_VERSION = "1.0.0"

# Minimum supported Python version
MINIMUM_PYTHON_VERSION = "3.13"

# API version
API_VERSION = "v1"


def get_version_info():
    """Get comprehensive version information.
    
    Returns:
        dict: Dictionary with version details
    """
    return {
        "package_version": __version__,
        "protocol_version": PROTOCOL_VERSION,
        "api_version": API_VERSION,
        "minimum_python_version": MINIMUM_PYTHON_VERSION,
    }