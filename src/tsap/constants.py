"""
Global constants for the TSAP MCP Server.
"""

# Version constants (duplicated from version.py for convenience)
VERSION = "0.1.0"  # Will need to be updated to match version.py
PROTOCOL_VERSION = "0.1.0"  # Will need to be updated to match version.py

# Performance mode constants
PERFORMANCE_MODE_FAST = "fast"
PERFORMANCE_MODE_STANDARD = "standard"
PERFORMANCE_MODE_DEEP = "deep"

# File type categorization
TEXT_FILE_EXTENSIONS = {
    "txt", "md", "rst", "json", "yaml", "yml", "toml", "xml", "html", "htm",
    "css", "js", "py", "java", "c", "cpp", "h", "hpp", "cs", "go", "rs",
    "php", "rb", "pl", "sh", "bash", "sql", "ini", "cfg", "conf"
}

BINARY_FILE_EXTENSIONS = {
    "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "zip", "tar", "gz",
    "bz2", "7z", "rar", "exe", "dll", "so", "dylib", "bin", "jpg", "jpeg",
    "png", "gif", "bmp", "svg", "mp3", "mp4", "wav", "avi", "mov"
}

# Command constants
COMMAND_TIMEOUT_DEFAULT = 60.0  # seconds
COMMAND_MAX_OUTPUT_SIZE = 1024 * 1024 * 10  # 10 MB

# Path constants
DEFAULT_CACHE_DIR = ".tsap/cache"
DEFAULT_PLUGINS_DIR = ".tsap/plugins"
DEFAULT_TEMPLATES_DIR = ".tsap/templates"
DEFAULT_STORAGE_DIR = ".tsap/storage"

# HTTP status codes (commonly used ones)
HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
HTTP_FORBIDDEN = 403
HTTP_NOT_FOUND = 404
HTTP_METHOD_NOT_ALLOWED = 405
HTTP_INTERNAL_SERVER_ERROR = 500
HTTP_SERVICE_UNAVAILABLE = 503

# MCP status codes
MCP_STATUS_SUCCESS = "success"
MCP_STATUS_ERROR = "error"
MCP_STATUS_TIMEOUT = "timeout"
MCP_STATUS_CANCELLED = "cancelled"

# Common error codes
ERROR_VALIDATION = "validation_error"
ERROR_EXECUTION = "execution_error"
ERROR_TIMEOUT = "timeout_error"
ERROR_PERMISSION = "permission_error"
ERROR_NOT_FOUND = "not_found_error"
ERROR_UNSUPPORTED = "unsupported_operation"
ERROR_DEPENDENCY = "dependency_error"