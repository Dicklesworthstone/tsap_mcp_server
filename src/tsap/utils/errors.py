"""
TSAP Error Definitions.

This module defines custom error types used throughout the TSAP system to provide
more specific error information and handling.
"""

from typing import Optional, Dict, Any


class TSAPError(Exception):
    """Base exception class for all TSAP-related errors."""

    def __init__(
        self, message: str, code: Optional[str] = None, details: Optional[Dict[str, Any]] = None
    ):
        """Initialize a TSAPError with optional error code and details.

        Args:
            message: Human-readable error message
            code: Machine-readable error code
            details: Additional error context and details
        """
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(message)


class ConfigurationError(TSAPError):
    """Error raised when there's an issue with TSAP configuration."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "CONFIGURATION_ERROR", details)


class DependencyError(TSAPError):
    """Error raised when a required dependency is missing or incompatible."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "DEPENDENCY_ERROR", details)


class ValidationError(TSAPError):
    """Error raised when input validation fails."""

    def __init__(
        self, message: str, field: Optional[str] = None, details: Optional[Dict[str, Any]] = None
    ):
        error_details = details or {}
        if field:
            error_details["field"] = field
        super().__init__(message, "VALIDATION_ERROR", error_details)


class ToolExecutionError(TSAPError):
    """Error raised when a tool execution fails."""

    def __init__(
        self,
        message: str,
        tool: Optional[str] = None,
        command: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        error_details = details or {}
        if tool:
            error_details["tool"] = tool
        if command:
            error_details["command"] = command
        super().__init__(message, "TOOL_EXECUTION_ERROR", error_details)


class ResourceError(TSAPError):
    """Error raised when there's an issue with system resources."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "RESOURCE_ERROR", details)


class CacheError(TSAPError):
    """Error raised when there's an issue with the caching system."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "CACHE_ERROR", details)


class FilesystemError(TSAPError):
    """Error raised when there's an issue with filesystem operations."""

    def __init__(
        self, message: str, path: Optional[str] = None, details: Optional[Dict[str, Any]] = None
    ):
        error_details = details or {}
        if path:
            error_details["path"] = path
        super().__init__(message, "FILESYSTEM_ERROR", error_details)


class MCPError(TSAPError):
    """Error raised for MCP protocol-related issues."""

    def __init__(
        self,
        message: str,
        command: Optional[str] = None,
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        error_details = details or {}
        if command:
            error_details["command"] = command
        if request_id:
            error_details["request_id"] = request_id
        super().__init__(message, "MCP_ERROR", error_details)


class PluginError(TSAPError):
    """Error raised for plugin-related issues."""

    def __init__(
        self,
        message: str,
        plugin_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        error_details = details or {}
        if plugin_name:
            error_details["plugin_name"] = plugin_name
        super().__init__(message, "PLUGIN_ERROR", error_details)


class AnalysisError(TSAPError):
    """Error raised when an analysis operation fails."""

    def __init__(
        self,
        message: str,
        analysis_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        error_details = details or {}
        if analysis_type:
            error_details["analysis_type"] = analysis_type
        super().__init__(message, "ANALYSIS_ERROR", error_details)