"""
TSAP Logging Package.

This package provides enhanced logging capabilities with rich formatting,
progress tracking, and console output for the TSAP system.
"""

import logging
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
    COMPLETE,
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

# Configure root logger
def configure_root_logger(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure the root logger with rich formatting.
    
    Args:
        level: Log level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        log_file: Optional path to log file
    """
    # Convert level string to logging level
    logging_level = getattr(logging, level.upper(), logging.INFO)
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set level
    root_logger.setLevel(logging_level)
    
    # Add rich handler
    rich_handler = RichLoggingHandler(
        level=logging_level,
        console=console,
        show_time=True,
        show_level=True,
        show_path=False,
    )
    root_logger.addHandler(rich_handler)
    
    # Add file handler if specified
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging_level)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            logger.error(f"Failed to create log file handler: {e}")


def get_logger(name: str) -> Logger:
    """Get a logger for a specific component.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return Logger(name)


def set_log_level(level: str) -> None:
    """Set the log level for the global logger.
    
    Args:
        level: Log level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    """
    logger.set_level(level)


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


# Initialize the logging system
def initialize_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    show_time: bool = True,
    show_level: bool = True,
    show_path: bool = False,
) -> None:
    """Initialize the TSAP logging system.
    
    Args:
        level: Log level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        log_file: Optional path to log file
        show_time: Whether to show timestamp in console logs
        show_level: Whether to show log level in console logs
        show_path: Whether to show file path in console logs
    """
    # Configure root logger
    configure_root_logger(level, log_file)
    
    # Set global logger level
    logger.set_level(level)
    
    # Log initialization message
    logger.info(f"TSAP logging initialized (level: {level})")


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
    "set_log_level",
    "configure_root_logger",
    "initialize_logging",
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
    "COMPLETE",
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