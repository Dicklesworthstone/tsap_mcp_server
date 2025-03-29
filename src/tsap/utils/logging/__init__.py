"""
TSAP Logging Package.

This package provides enhanced logging capabilities with rich formatting,
progress tracking, and console output for the TSAP system.
"""

import logging
import logging.handlers
from typing import Dict, Any, Optional, List

# Import Rich-based console
from tsap.utils.logging.console import (
    console,
    create_progress,
    status,
    print_panel,
    print_syntax,
    print_table,
    print_tree,
    print_json,
    live_display,
)

# Import logger and related utilities
from tsap.utils.logging.logger import (
    Logger,
    debug,
    info,
    success,
    warning,
    error,
    critical,
    section,
)

# Import emojis
from tsap.utils.logging.emojis import (
    get_emoji,
    INFO,
    DEBUG,
    WARNING,
    ERROR,
    CRITICAL,
    SUCCESS,
    RUNNING,
    COMPLETED,
    FAILED,
)

# Import panels
from tsap.utils.logging.panels import (
    HeaderPanel,
    ResultPanel,
    InfoPanel,
    WarningPanel,
    ErrorPanel,
    ToolOutputPanel,
    CodePanel,
    display_header,
    display_results,
    display_info,
    display_warning,
    display_error,
    display_tool_output,
    display_code,
)

# Import progress tracking
from tsap.utils.logging.progress import (
    TSAPProgress,
    ProgressContext,
    track,
    task,
)

# Import formatters and handlers
from tsap.utils.logging.formatter import (
    TSAPLogRecord,
    SimpleLogFormatter,
    DetailedLogFormatter,
    RichLoggingHandler,
)

# Create a global logger instance for importing
logger = Logger("tsap")

# Removed configure_root_logger, initialize_logging, set_log_level functions
# Logging is now configured via dictConfig in server.py

def get_logger(name: str) -> Logger:
    """Get a logger for a specific component.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return Logger(name)

def capture_logs(level: Optional[str] = None) -> "LogCapture":
    """Create a context manager to capture logs.
    
    Args:
        level: Minimum log level to capture
        
    Returns:
        Log capture context manager
    """
    return LogCapture(level)

# Log capturing for testing
class LogCapture:
    """Context manager for capturing logs."""
    
    def __init__(self, level: Optional[str] = None):
        """Initialize the log capture.
        
        Args:
            level: Minimum log level to capture
        """
        self.level = level
        self.level_num = getattr(logging, self.level.upper(), 0) if self.level else 0
        self.logs: List[Dict[str, Any]] = []
        self.handler = self._create_handler()
    
    def _create_handler(self) -> logging.Handler:
        """Create a handler to capture logs.
        
        Returns:
            Log handler
        """
        class CaptureHandler(logging.Handler):
            def __init__(self, capture):
                super().__init__()
                self.capture = capture
            
            def emit(self, record):
                # Skip if record level is lower than minimum
                if record.levelno < self.capture.level_num:
                    return
                
                # Add log record to captured logs
                self.capture.logs.append({
                    "level": record.levelname,
                    "message": record.getMessage(),
                    "name": record.name,
                    "time": record.created,
                    "file": record.pathname,
                    "line": record.lineno,
                })
        
        return CaptureHandler(self)
    
    def __enter__(self) -> "LogCapture":
        """Enter the context manager.
        
        Returns:
            Self
        """
        # Add handler to root logger
        logging.getLogger().addHandler(self.handler)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager.
        
        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
        """
        # Remove handler from root logger
        logging.getLogger().removeHandler(self.handler)
    
    def get_logs(self, level: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get captured logs, optionally filtered by level.
        
        Args:
            level: Filter logs by level
            
        Returns:
            List of log records
        """
        if not level:
            return self.logs
        
        level_num = getattr(logging, level.upper(), 0)
        return [log for log in self.logs if getattr(logging, log["level"], 0) >= level_num]
    
    def get_messages(self, level: Optional[str] = None) -> List[str]:
        """Get captured log messages, optionally filtered by level.
        
        Args:
            level: Filter logs by level
            
        Returns:
            List of log messages
        """
        return [log["message"] for log in self.get_logs(level)]
    
    def contains(self, text: str, level: Optional[str] = None) -> bool:
        """Check if captured logs contain a specific text.
        
        Args:
            text: Text to search for
            level: Filter logs by level
            
        Returns:
            True if text is found in log messages
        """
        return any(text in message for message in self.get_messages(level))

__all__ = [
    # Console
    "console",
    "create_progress",
    "status",
    "print_panel",
    "print_syntax",
    "print_table",
    "print_tree",
    "print_json",
    "live_display",
    
    # Logger and utilities
    "logger",
    "Logger",
    "debug",
    "info",
    "success",
    "warning",
    "error",
    "critical",
    "section",
    "get_logger",
    "capture_logs",
    "LogCapture",
    
    # Emojis
    "get_emoji",
    "INFO",
    "DEBUG",
    "WARNING",
    "ERROR",
    "CRITICAL",
    "SUCCESS",
    "RUNNING",
    "COMPLETED",
    "FAILED",
    
    # Panels
    "HeaderPanel",
    "ResultPanel",
    "InfoPanel",
    "WarningPanel",
    "ErrorPanel",
    "ToolOutputPanel",
    "CodePanel",
    "display_header",
    "display_results",
    "display_info",
    "display_warning",
    "display_error",
    "display_tool_output",
    "display_code",
    
    # Progress tracking
    "TSAPProgress",
    "ProgressContext",
    "track",
    "task",
    
    # Formatters and handlers
    "TSAPLogRecord",
    "SimpleLogFormatter",
    "DetailedLogFormatter",
    "RichLoggingHandler",
]