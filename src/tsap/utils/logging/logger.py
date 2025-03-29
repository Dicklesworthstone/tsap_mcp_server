"""
Main Logger class for TSAP.

This module provides the central Logger class that integrates all TSAP logging
functionality with a beautiful, informative interface.
"""
import sys
import time
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from functools import wraps
from contextlib import contextmanager

from rich.console import Console

from .console import console
from .formatter import (
    TSAPLogRecord, SimpleLogFormatter, DetailedLogFormatter, RichLoggingHandler
)
from .emojis import get_emoji
from .progress import TSAPProgress
from .panels import (
    HeaderPanel, ResultPanel, InfoPanel, WarningPanel, ErrorPanel,
    ToolOutputPanel, CodePanel
)

# Set up standard Python logging with our custom handler
# logging.basicConfig( # Removed to centralize config in __init__.py
#     level=logging.INFO,
#     format="%(message)s",
#     datefmt="[%X]",
#     handlers=[RichLoggingHandler(console=console)]
# )

class Logger:
    """Advanced logger for TSAP with rich formatting and progress tracking."""
    
    def __init__(
        self,
        name: str = "tsap",
        console: Optional[Console] = None,
        level: str = "info",
        show_timestamps: bool = True,
        component: Optional[str] = None,
        capture_output: bool = False,
    ):
        """Initialize the TSAP logger.
        
        Args:
            name: Logger name
            console: Rich console to use
            level: Initial log level
            show_timestamps: Whether to show timestamps in logs
            component: Default component name
            capture_output: Whether to capture and store log output
        """
        self.name = name
        self.console = console or globals().get("console", Console())
        self.level = level.lower()
        self.show_timestamps = show_timestamps
        self.component = component
        self.capture_output = capture_output
        
        # Create a standard Python logger
        self.python_logger = logging.getLogger(name)
        
        # Set up formatters
        self.simple_formatter = SimpleLogFormatter(show_timestamps, True, True)
        self.detailed_formatter = DetailedLogFormatter(show_timestamps, True, True)
        
        # Progress tracker
        self.progress = TSAPProgress(console=self.console)
        
        # Output capture if enabled
        self.captured_logs = [] if capture_output else None
        
        # Restore propagation to allow messages to reach root handlers
        self.python_logger.propagate = True 
        
        # Set initial log level
        self.set_level(level)
    
    def set_level(self, level: str) -> None:
        """Set the log level.
        
        Args:
            level: Log level (debug, info, warning, error, critical)
        """
        level = level.lower()
        self.level = level
        
        # Map to Python logging levels
        level_map = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }
        
        python_level = level_map.get(level, logging.INFO)
        self.python_logger.setLevel(python_level)
    
    def get_level(self) -> str:
        """Get the current log level.
        
        Returns:
            Current log level
        """
        return self.level
    
    def should_log(self, level: str) -> bool:
        """Check if a message at the given level should be logged.
        
        Args:
            level: Log level to check
            
        Returns:
            Whether messages at this level should be logged
        """
        level_priority = {
            "debug": 0,
            "info": 1,
            "success": 1,  # Same as info
            "warning": 2,
            "error": 3,
            "critical": 4,
        }
        
        current_priority = level_priority.get(self.level, 1)
        message_priority = level_priority.get(level.lower(), 1)
        
        return message_priority >= current_priority
    
    def _log(
        self,
        level: str,
        message: str,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        emoji: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        use_detailed_formatter: bool = False,
        exception_info: Optional[Tuple] = None,
    ) -> None:
        """Internal method to handle logging.
        
        Args:
            level: Log level
            message: Log message
            component: TSAP component (core, composite, analysis, etc.)
            operation: Operation being performed
            emoji: Custom emoji override
            context: Additional contextual data
            use_detailed_formatter: Whether to use the detailed formatter
            exception_info: Exception info tuple (type, value, traceback)
        """
        # Check if we should log at this level
        if not self.should_log(level):
            return
            
        # Use default component if not provided
        component = component or self.component
        
        # Create our enhanced log record
        record = TSAPLogRecord(
            level=level,
            message=message,
            component=component,
            operation=operation,
            emoji=emoji,
            context=context,
            exception_info=exception_info,
        )
        
        # Format the record
        formatter = self.detailed_formatter if use_detailed_formatter else self.simple_formatter
        renderable = formatter.format_record(record)
        
        # Print to console - REMOVED to prevent duplicates
        # self.console.print(renderable)
        
        # Capture if enabled
        if self.captured_logs is not None:
            self.captured_logs.append({
                "level": level,
                "message": message,
                "component": component,
                "operation": operation,
                "timestamp": datetime.now().isoformat(),
                "context": context,
            })
            
        # Also log through Python's logging system (useful for file logging, etc.)
        log_func = getattr(self.python_logger, level if level != "success" else "info")
        
        # Add extra fields for our custom handler
        extras = {
            "component": component,
            "operation": operation,
            "emoji": emoji,
            "context": context,
        }
        
        # Log with Python logger
        if exception_info and level in ("error", "critical"):
            exc_type, exc_value, exc_tb = exception_info
            log_func(message, exc_info=(exc_type, exc_value, exc_tb), extra=extras)
        else:
            log_func(message, extra=extras)
    
    # Standard logging methods
    
    def debug(
        self,
        message: str,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a debug message.
        
        Args:
            message: Log message
            component: TSAP component
            operation: Operation being performed
            context: Additional contextual data
        """
        self._log("debug", message, component, operation, context=context)
    
    def info(
        self,
        message: str,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an info message.
        
        Args:
            message: Log message
            component: TSAP component
            operation: Operation being performed
            context: Additional contextual data
        """
        self._log("info", message, component, operation, context=context)
    
    def success(
        self,
        message: str,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a success message.
        
        Args:
            message: Log message
            component: TSAP component
            operation: Operation being performed
            context: Additional contextual data
        """
        self._log("success", message, component, operation, context=context)
    
    def warning(
        self,
        message: str,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        details: Optional[List[str]] = None,
    ) -> None:
        """Log a warning message.
        
        Args:
            message: Log message
            component: TSAP component
            operation: Operation being performed
            context: Additional contextual data
            details: Optional list of detail points
        """
        warning_context = context or {}
        if details:
            warning_context["details"] = details
            
        self._log(
            "warning",
            message,
            component,
            operation,
            context=warning_context,
            use_detailed_formatter=True,
        )
    
    def error(
        self,
        message: str,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        error_code: Optional[str] = None,
        resolution_steps: Optional[List[str]] = None,
    ) -> None:
        """Log an error message.
        
        Args:
            message: Log message
            component: TSAP component
            operation: Operation being performed
            context: Additional contextual data
            exception: Optional exception that caused the error
            error_code: Optional error code for reference
            resolution_steps: Optional list of steps to resolve the error
        """
        error_context = context or {}
        
        if error_code:
            error_context["error_code"] = error_code
            
        if resolution_steps:
            error_context["resolution_steps"] = resolution_steps
            
        # Get exception info if provided
        exc_info = None
        if exception:
            exc_info = (type(exception), exception, exception.__traceback__)
            
        self._log(
            "error",
            message,
            component,
            operation,
            context=error_context,
            use_detailed_formatter=True,
            exception_info=exc_info,
        )
    
    def critical(
        self,
        message: str,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        error_code: Optional[str] = None,
    ) -> None:
        """Log a critical error message.
        
        Args:
            message: Log message
            component: TSAP component
            operation: Operation being performed
            context: Additional contextual data
            exception: Optional exception that caused the error
            error_code: Optional error code for reference
        """
        critical_context = context or {}
        
        if error_code:
            critical_context["error_code"] = error_code
            
        # Get exception info if provided
        exc_info = None
        if exception:
            exc_info = (type(exception), exception, exception.__traceback__)
            
        self._log(
            "critical",
            message,
            component,
            operation,
            context=critical_context,
            use_detailed_formatter=True,
            exception_info=exc_info,
        )
    
    # Enhanced logging methods
    
    def operation(
        self,
        operation: str,
        message: str,
        component: Optional[str] = None,
        level: str = "info",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an operation.
        
        Args:
            operation: Operation name
            message: Log message
            component: TSAP component
            level: Log level
            context: Additional contextual data
        """
        # Get the operation emoji
        emoji = get_emoji("operation", operation)
        
        self._log(
            level,
            message,
            component,
            operation,
            emoji=emoji,
            context=context,
        )
    
    def tool(
        self,
        tool: str,
        command: str,
        output: str,
        status: str = "success",
        duration: Optional[float] = None,
        component: Optional[str] = None,
    ) -> None:
        """Log a tool command and its output.
        
        Args:
            tool: Tool name (ripgrep, awk, jq, etc.)
            command: Command that was executed
            output: Command output text
            status: Execution status (success, error)
            duration: Optional execution duration in seconds
            component: TSAP component
        """
        # Create a tool output panel
        panel = ToolOutputPanel(tool, command, output, status, duration)
        
        # Log a message about the tool execution
        level = "success" if status == "success" else "error"
        message = f"Executed {tool} command"
        
        if status != "success":
            message += f" (failed: {status})"
            
        if duration is not None:
            message += f" in {duration:.2f}s"
            
        self._log(
            level,
            message,
            component,
            operation=f"execute_{tool}",
            context={"command": command, "output": output},
        )
        
        # Print the panel
        self.console.print(panel)
    
    def code(
        self,
        code: str,
        language: str = "python",
        title: Optional[str] = None,
        line_numbers: bool = True,
        highlight_lines: Optional[List[int]] = None,
        message: Optional[str] = None,
        component: Optional[str] = None,
    ) -> None:
        """Log code with syntax highlighting.
        
        Args:
            code: The code to display
            language: Programming language for syntax highlighting
            title: Optional panel title
            line_numbers: Whether to show line numbers
            highlight_lines: List of line numbers to highlight
            message: Optional log message
            component: TSAP component
        """
        # Create a code panel
        panel = CodePanel(code, language, title, line_numbers, highlight_lines)
        
        # Log a message if provided
        if message:
            self._log(
                "info",
                message,
                component,
                operation="code_display",
                context={"language": language},
            )
            
        # Print the panel
        self.console.print(panel)
    
    def display_results(
        self,
        title: str,
        results: Union[List[Dict[str, Any]], Dict[str, Any]],
        status: str = "success",
        component: Optional[str] = None,
        show_count: bool = True,
        compact: bool = False,
        message: Optional[str] = None,
    ) -> None:
        """Display operation results.
        
        Args:
            title: Results title
            results: Results to display (list of dicts or single dict)
            status: Result status (success, warning, error)
            component: TSAP component
            show_count: Whether to show result count in title
            compact: Whether to use a compact display style
            message: Optional log message
        """
        # Create a result panel
        panel = ResultPanel(title, results, status, component, show_count, compact)
        
        # Log a message if provided
        if message:
            level = status if status in ("success", "warning", "error") else "info"
            context = {"results_count": len(results) if isinstance(results, list) else 1}
            
            self._log(
                level,
                message,
                component,
                operation="display_results",
                context=context,
            )
            
        # Print the panel
        self.console.print(panel)
    
    def section(
        self,
        title: str,
        subtitle: Optional[str] = None,
        component: Optional[str] = None,
    ) -> None:
        """Display a section header.
        
        Args:
            title: Section title
            subtitle: Optional subtitle
            component: TSAP component
        """
        # Create a header panel
        panel = HeaderPanel(title, subtitle, component=component)
        
        # Print the panel
        self.console.print(panel)
        
        # Log the section change
        self._log(
            "info",
            f"Section: {title}",
            component,
            operation="section",
        )
    
    def info_panel(
        self,
        title: str,
        content: Union[str, List[str], Dict[str, Any]],
        icon: Optional[str] = None,
        style: str = "info",
        component: Optional[str] = None,
    ) -> None:
        """Display an information panel.
        
        Args:
            title: Panel title
            content: Content to display (string, list, or dict)
            icon: Emoji or icon character
            style: Style name to apply (from theme)
            component: TSAP component
        """
        # Create an info panel
        panel = InfoPanel(title, content, icon, style)
        
        # Print the panel
        self.console.print(panel)
        
        # Log the info display
        self._log(
            "info",
            f"Info: {title}",
            component,
            operation="info_display",
        )
    
    def warning_panel(
        self,
        title: Optional[str] = None,
        message: str = "",
        details: Optional[List[str]] = None,
        component: Optional[str] = None,
    ) -> None:
        """Display a warning panel.
        
        Args:
            title: Optional panel title
            message: Main warning message
            details: Optional list of detail points
            component: TSAP component
        """
        # Create a warning panel
        panel = WarningPanel(title, message, details)
        
        # Print the panel
        self.console.print(panel)
        
        # Log the warning
        self._log(
            "warning",
            message,
            component,
            operation="warning_display",
            context={"details": details} if details else None,
            use_detailed_formatter=True,
        )
    
    def error_panel(
        self,
        title: Optional[str] = None,
        message: str = "",
        details: Optional[str] = None,
        resolution_steps: Optional[List[str]] = None,
        error_code: Optional[str] = None,
        component: Optional[str] = None,
        exception: Optional[Exception] = None,
    ) -> None:
        """Display an error panel.
        
        Args:
            title: Optional panel title
            message: Main error message
            details: Optional error details
            resolution_steps: Optional list of steps to resolve the error
            error_code: Optional error code for reference
            component: TSAP component
            exception: Optional exception that caused the error
        """
        # Create an error panel
        panel = ErrorPanel(title, message, details, resolution_steps, error_code)
        
        # Print the panel
        self.console.print(panel)
        
        # Get exception info if provided
        exc_info = None
        if exception:
            exc_info = (type(exception), exception, exception.__traceback__)
            
        # Log the error
        error_context = {}
        if details:
            error_context["details"] = details
        if resolution_steps:
            error_context["resolution_steps"] = resolution_steps
        if error_code:
            error_context["error_code"] = error_code
            
        self._log(
            "error",
            message,
            component,
            operation="error_display",
            context=error_context,
            use_detailed_formatter=True,
            exception_info=exc_info,
        )
    
    # Timing and performance methods
    
    @contextmanager
    def time_operation(
        self,
        operation: str,
        component: Optional[str] = None,
        level: str = "info",
    ):
        """Context manager to time an operation.
        
        Args:
            operation: Operation name
            component: TSAP component
            level: Log level for the completion message
            
        Yields:
            None
        """
        # Get operation emoji
        emoji = get_emoji("operation", operation)
        
        # Log start message
        start_msg = f"Starting {operation}"
        self._log(level, start_msg, component, operation, emoji=emoji)
        
        # Record start time
        start_time = time.time()
        
        try:
            # Yield control back to the caller
            yield
            
            # No exception, operation succeeded
            status = "success"
            status_msg = "Completed"
        except Exception:
            # Operation failed
            status = "error"
            status_msg = "Failed"
            # Re-raise the exception
            raise
        finally:
            # Calculate duration
            duration = time.time() - start_time
            
            # Log completion message
            end_msg = f"{status_msg} {operation} in {duration:.2f}s"
            
            # Adjust level based on status
            if status == "error":
                end_level = "error"
            else:
                end_level = level
                
            self._log(
                end_level,
                end_msg,
                component,
                operation,
                emoji=emoji,
                context={"duration": duration},
            )
    
    # Progress tracking methods
    
    def track(
        self,
        iterable: Any,
        description: str,
        name: Optional[str] = None,
        total: Optional[int] = None,
        parent: Optional[str] = None,
    ) -> Any:
        """Track progress through an iterable.
        
        Args:
            iterable: The iterable to track
            description: User-visible task description
            name: Unique task identifier
            total: Total items (obtained from iterable if not provided)
            parent: Optional parent task name
            
        Returns:
            Tracked iterable
        """
        return self.progress.track(iterable, description, name, total, parent)
    
    @contextmanager
    def task(
        self,
        description: str,
        name: Optional[str] = None,
        total: int = 100,
        parent: Optional[str] = None,
        component: Optional[str] = None,
    ):
        """Context manager for a task with automatic start/complete.
        
        Args:
            description: User-visible task description
            name: Unique task identifier
            total: Total work units for this task
            parent: Optional parent task name
            component: TSAP component
            
        Yields:
            Task name
        """
        # Log task start
        self._log(
            "info",
            f"Task: {description}",
            component,
            operation="task",
        )
        
        # Start progress tracking if not already started
        if self.progress._live is None:
            self.progress.start()
            
        try:
            with self.progress.task(description, name, total, parent) as task_name:
                yield task_name
                
            # Task completed successfully
            self._log(
                "success",
                f"Completed task: {description}",
                component,
                operation="task_complete",
            )
            
        except Exception as e:
            # Task failed
            self._log(
                "error",
                f"Failed task: {description}",
                component,
                operation="task_failed",
                exception=e,
            )
            # Re-raise the exception
            raise
            
        finally:
            # Stop progress tracking if it's still active
            if self.progress._live:
                self.progress.stop()
    
    # Exception handling methods
    
    @contextmanager
    def catch_and_log(
        self,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        reraise: bool = True,
        level: str = "error",
    ):
        """Context manager to catch and log exceptions.
        
        Args:
            component: TSAP component
            operation: Operation being performed
            reraise: Whether to re-raise the caught exception
            level: Log level for the exception
            
        Yields:
            None
        """
        try:
            # Yield control back to the caller
            yield
            
        except Exception as e:
            # Log the exception
            exc_type, exc_value, exc_tb = sys.exc_info()
            
            # Format exception message
            message = f"Exception: {str(e)}"
            
            # Log with appropriate level
            if level == "critical":
                self.critical(
                    message, component, operation, exception=e
                )
            else:
                self.error(
                    message, component, operation, exception=e
                )
                
            # Re-raise if requested
            if reraise:
                raise
    
    # Decorator for logging function calls
    def log_call(
        self,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        level: str = "debug",
        log_args: bool = True,
        log_result: bool = False,
        log_exceptions: bool = True,
    ):
        """Decorator to log function calls.
        
        Args:
            component: TSAP component
            operation: Operation being performed (defaults to function name)
            level: Log level
            log_args: Whether to log function arguments
            log_result: Whether to log function return value
            log_exceptions: Whether to log exceptions
            
        Returns:
            Decorated function
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Determine operation name
                op_name = operation or func.__name__
                
                # Prepare context with args if requested
                context = None
                if log_args:
                    # Convert args to a safe representation
                    safe_args = [repr(arg) for arg in args]
                    safe_kwargs = {k: repr(v) for k, v in kwargs.items()}
                    
                    context = {
                        "args": safe_args,
                        "kwargs": safe_kwargs,
                    }
                
                # Log function call
                self._log(
                    level,
                    f"Calling {func.__name__}",
                    component,
                    op_name,
                    context=context,
                )
                
                try:
                    # Call the function
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    # Log success with result if requested
                    result_context = None
                    if log_result:
                        # Use a safe representation of the result
                        result_context = {"result": repr(result), "duration": duration}
                    elif duration > 0.1:  # Only log duration if significant
                        result_context = {"duration": duration}
                        
                    self._log(
                        level,
                        f"Completed {func.__name__}",
                        component,
                        op_name,
                        context=result_context,
                    )
                    
                    return result
                    
                except Exception as e:
                    # Log exception if requested
                    if log_exceptions:
                        self.error(
                            f"Exception in {func.__name__}: {str(e)}",
                            component,
                            op_name,
                            exception=e,
                        )
                        
                    # Re-raise the exception
                    raise
                    
            return wrapper
            
        return decorator
    
    # System/state methods
    
    def startup(
        self,
        version: str,
        component: Optional[str] = None,
        mode: str = "standard",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log system startup.
        
        Args:
            version: TSAP version
            component: TSAP component
            mode: Performance mode (fast, standard, deep)
            context: Additional startup context
        """
        # Create combined context
        startup_context = {
            "version": version,
            "mode": mode,
            "python_version": sys.version,
            "timestamp": datetime.now().isoformat(),
        }
        
        if context:
            startup_context.update(context)
            
        # Create the header panel
        # panel = HeaderPanel(title="TSAP Startup", version=version, mode=mode)
        # self.console.print(panel)
        
        # Log the startup message
        self.section("TSAP Startup", component=component)
        
        # Log detailed message
        self._log(
            "info",
            f"Starting TSAP {version} in {mode} mode",
            component,
            operation="startup",
            emoji=get_emoji("system", "startup"),
            context=startup_context,
        )
    
    def shutdown(
        self,
        component: Optional[str] = None,
        duration: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log system shutdown.
        
        Args:
            component: TSAP component
            duration: Optional total runtime in seconds
            context: Additional shutdown context
        """
        # Create context
        shutdown_context = {
            "timestamp": datetime.now().isoformat(),
        }
        
        if duration:
            shutdown_context["duration"] = duration
            
        if context:
            shutdown_context.update(context)
            
        # Log shutdown message
        duration_str = f" after {duration:.2f}s" if duration else ""
        
        self._log(
            "info",
            f"Shutting down TSAP{duration_str}",
            component,
            operation="shutdown",
            emoji=get_emoji("system", "shutdown"),
            context=shutdown_context,
        )

# Create a default global logger
logger = Logger()

# Convenience functions using the global logger

def debug(
    message: str,
    component: Optional[str] = None,
    operation: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Log a debug message using the global logger.
    
    Args:
        message: Log message
        component: TSAP component
        operation: Operation being performed
        context: Additional contextual data
    """
    logger.debug(message, component, operation, context)

def info(
    message: str,
    component: Optional[str] = None,
    operation: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Log an info message using the global logger.
    
    Args:
        message: Log message
        component: TSAP component
        operation: Operation being performed
        context: Additional contextual data
    """
    logger.info(message, component, operation, context)

def success(
    message: str,
    component: Optional[str] = None,
    operation: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Log a success message using the global logger.
    
    Args:
        message: Log message
        component: TSAP component
        operation: Operation being performed
        context: Additional contextual data
    """
    logger.success(message, component, operation, context)

def warning(
    message: str,
    component: Optional[str] = None,
    operation: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    details: Optional[List[str]] = None,
) -> None:
    """Log a warning message using the global logger.
    
    Args:
        message: Log message
        component: TSAP component
        operation: Operation being performed
        context: Additional contextual data
        details: Optional list of detail points
    """
    logger.warning(message, component, operation, context, details)

def error(
    message: str,
    component: Optional[str] = None,
    operation: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    exception: Optional[Exception] = None,
    error_code: Optional[str] = None,
    resolution_steps: Optional[List[str]] = None,
) -> None:
    """Log an error message using the global logger.
    
    Args:
        message: Log message
        component: TSAP component
        operation: Operation being performed
        context: Additional contextual data
        exception: Optional exception that caused the error
        error_code: Optional error code for reference
        resolution_steps: Optional list of steps to resolve the error
    """
    logger.error(
        message, component, operation, context,
        exception, error_code, resolution_steps
    )

def critical(
    message: str,
    component: Optional[str] = None,
    operation: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    exception: Optional[Exception] = None,
    error_code: Optional[str] = None,
) -> None:
    """Log a critical error message using the global logger.
    
    Args:
        message: Log message
        component: TSAP component
        operation: Operation being performed
        context: Additional contextual data
        exception: Optional exception that caused the error
        error_code: Optional error code for reference
    """
    logger.critical(message, component, operation, context, exception, error_code)

def section(
    title: str,
    subtitle: Optional[str] = None,
    component: Optional[str] = None,
) -> None:
    """Display a section header using the global logger.
    
    Args:
        title: Section title
        subtitle: Optional subtitle
        component: TSAP component
    """
    logger.section(title, subtitle, component)