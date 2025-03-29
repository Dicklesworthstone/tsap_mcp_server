"""
Log formatters for TSAP logging system.

This module provides formatters that convert log records into Rich renderables
with consistent styling and visual elements.
"""
import time
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

from rich.console import Console, ConsoleRenderable
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.logging import RichHandler
from rich.traceback import Traceback
from rich.style import Style
from rich.columns import Columns

from .emojis import LEVEL_EMOJIS, get_emoji
from .themes import get_level_style, get_component_style

class TSAPLogRecord:
    """Enhanced log record with additional TSAP-specific fields."""
    
    def __init__(
        self,
        level: str,
        message: str,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        emoji: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
        exception_info: Optional[Tuple] = None,
    ):
        """Initialize a TSAP log record.
        
        Args:
            level: Log level (info, debug, warning, error, critical, success)
            message: Log message
            component: TSAP component (core, composite, analysis, etc.)
            operation: Operation being performed
            emoji: Custom emoji override
            context: Additional contextual data
            timestamp: Unix timestamp (defaults to current time)
            exception_info: Exception info tuple (type, value, traceback)
        """
        self.level = level.lower()
        self.message = message
        self.component = component.lower() if component else None
        self.operation = operation.lower() if operation else None
        self.custom_emoji = emoji
        self.context = context or {}
        self.timestamp = timestamp or time.time()
        self.exception_info = exception_info
        
    @property
    def emoji(self) -> str:
        """Get the appropriate emoji for this log record."""
        if self.custom_emoji:
            return self.custom_emoji
            
        # Use operation emoji if available
        if self.operation:
            operation_emoji = get_emoji("operation", self.operation)
            if operation_emoji != "❓":  # If not unknown
                return operation_emoji
        
        # Fall back to level emoji
        return LEVEL_EMOJIS.get(self.level, "❓")
    
    @property
    def style(self) -> Style:
        """Get the appropriate style for this log record."""
        return get_level_style(self.level)
    
    @property
    def component_style(self) -> Style:
        """Get the style for this record's component."""
        if not self.component:
            return self.style
        return get_component_style(self.component)
    
    @property
    def format_time(self) -> str:
        """Format the timestamp for display."""
        dt = datetime.fromtimestamp(self.timestamp)
        return dt.strftime("%H:%M:%S.%f")[:-3]  # Trim microseconds to milliseconds
    
    def has_exception(self) -> bool:
        """Check if this record contains exception information."""
        return self.exception_info is not None

class TSAPLogFormatter:
    """Base formatter for TSAP logs that converts to Rich renderables."""
    
    def __init__(self, show_time: bool = True, show_level: bool = True, show_component: bool = True):
        """Initialize the formatter.
        
        Args:
            show_time: Whether to show timestamp
            show_level: Whether to show log level
            show_component: Whether to show component
        """
        self.show_time = show_time
        self.show_level = show_level
        self.show_component = show_component
    
    def format_record(self, record: TSAPLogRecord) -> ConsoleRenderable:
        """Format a TSAPLogRecord into a Rich renderable.
        
        Args:
            record: The log record to format
            
        Returns:
            A Rich renderable object
        """
        # This should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement format_record")

class SimpleLogFormatter(TSAPLogFormatter):
    """Simple single-line log formatter."""
    
    def format_record(self, record: TSAPLogRecord) -> Text:
        """Format a record as a single line of text.
        
        Args:
            record: The log record to format
            
        Returns:
            Formatted Text object
        """
        result = Text()
        
        # Add timestamp if requested
        if self.show_time:
            result.append(f"[{record.format_time}] ", style="timestamp")
            
        # Add emoji
        result.append(f"{record.emoji} ", style=record.style)
        
        # Add level if requested
        if self.show_level:
            level_text = f"[{record.level.upper()}] "
            result.append(level_text, style=record.style)
            
        # Add component if available and requested
        if self.show_component and record.component:
            component_text = f"[{record.component}] "
            result.append(component_text, style=record.component_style)
            
        # Add operation if available
        if record.operation:
            operation_text = f"{record.operation}: "
            result.append(operation_text, style="operation")
            
        # Add message
        result.append(record.message)
        
        return result

class DetailedLogFormatter(TSAPLogFormatter):
    """Multi-line formatter that can include context data."""
    
    def __init__(
        self,
        show_time: bool = True,
        show_level: bool = True,
        show_component: bool = True,
        show_context: bool = True,
        context_max_depth: int = 2,
    ):
        """Initialize the formatter.
        
        Args:
            show_time: Whether to show timestamp
            show_level: Whether to show log level
            show_component: Whether to show component
            show_context: Whether to show context data
            context_max_depth: Maximum depth for context data display
        """
        super().__init__(show_time, show_level, show_component)
        self.show_context = show_context
        self.context_max_depth = context_max_depth
    
    def format_record(self, record: TSAPLogRecord) -> ConsoleRenderable:
        """Format a record with detailed information.
        
        Args:
            record: The log record to format
            
        Returns:
            Formatted Panel or Text object
        """
        # Create the header similar to SimpleLogFormatter
        header = Text()
        
        if self.show_time:
            header.append(f"[{record.format_time}] ", style="timestamp")
            
        header.append(f"{record.emoji} ", style=record.style)
        
        if self.show_level:
            level_text = f"[{record.level.upper()}] "
            header.append(level_text, style=record.style)
            
        if self.show_component and record.component:
            component_text = f"[{record.component}] "
            header.append(component_text, style=record.component_style)
            
        if record.operation:
            operation_text = f"{record.operation}: "
            header.append(operation_text, style="operation")
            
        header.append(record.message)
        
        # For simple cases with no context or exception, just return the header
        if not self.show_context or (not record.context and not record.has_exception()):
            return header
            
        # Otherwise, create a more detailed display
        elements = [header]
        
        # Add context data if available
        if record.context and self.show_context:
            context_table = Table(box=None, expand=False, padding=(0, 1))
            context_table.add_column("Key", style="bright_black")
            context_table.add_column("Value")
            
            # Flatten context data to key-value pairs up to max_depth
            def format_value(value, depth=0):
                if depth >= self.context_max_depth:
                    if isinstance(value, (dict, list)):
                        return f"<{type(value).__name__} with {len(value)} items>"
                    return str(value)
                    
                if isinstance(value, dict):
                    if not value:
                        return "{}"
                    return "{...}"  # Just indicate there's content
                elif isinstance(value, (list, tuple)):
                    if not value:
                        return "[]" if isinstance(value, list) else "()"
                    return "[...]" if isinstance(value, list) else "(...)"
                return str(value)
            
            # Add context entries
            for key, value in record.context.items():
                context_table.add_row(str(key), format_value(value))
                
            elements.append(context_table)
        
        # Add exception information if available
        if record.has_exception():
            exc_type, exc_value, exc_tb = record.exception_info
            elements.append(Traceback.from_exception(exc_type, exc_value, exc_tb))
        
        # Return a panel for error and critical, otherwise a simpler layout
        if record.level in ("error", "critical"):
            return Panel(
                Columns(elements, padding=(0, 1)),
                title=f"{record.level.upper()} in {record.component or 'tsap'}",
                border_style=record.style,
                padding=(1, 2),
            )
        
        return Columns(elements, padding=(0, 2))

class RichLoggingHandler(RichHandler):
    """Enhanced Rich logging handler for TSAP."""
    
    def __init__(
        self,
        level: int = logging.NOTSET,
        console: Optional[Console] = None,
        formatter: Optional[TSAPLogFormatter] = None,
        **kwargs
    ):
        """Initialize the Rich logging handler.
        
        Args:
            level: Logging level
            console: Rich console to use
            formatter: TSAP log formatter
            **kwargs: Additional arguments passed to RichHandler
        """
        super().__init__(level=level, console=console, **kwargs)
        self.tsap_formatter = formatter or SimpleLogFormatter()
    
    def render(
        self,
        record: logging.LogRecord,
        traceback: Optional[Traceback],
        message_renderable: ConsoleRenderable,
    ) -> ConsoleRenderable:
        """Render a log record.
        
        This overrides the Rich handler's render method to use our TSAP formatter.
        
        Args:
            record: Standard logging record
            traceback: Exception traceback if any
            message_renderable: The rendered message
            
        Returns:
            Formatted renderable
        """
        # Extract TSAP-specific fields from the record extras
        component = getattr(record, "component", None)
        operation = getattr(record, "operation", None)
        emoji = getattr(record, "emoji", None)
        context = getattr(record, "context", {})
        
        # Create our enhanced record
        tsap_record = TSAPLogRecord(
            level=record.levelname.lower(),
            message=record.getMessage(),
            component=component,
            operation=operation,
            emoji=emoji,
            context=context,
            timestamp=record.created,
            exception_info=(record.exc_info if record.exc_info else None),
        )
        
        # Use our formatter
        return self.tsap_formatter.format_record(tsap_record)

# Add factory function for dictConfig
def create_rich_console_handler(**kwargs):
    """Factory function to create a RichLoggingHandler for dictConfig."""
    # Import console instance here to avoid circular import issues at module level
    from .console import console

    # Get level if passed from config, default otherwise
    level = kwargs.get("level", logging.INFO)
    # Note: Formatter is implicitly handled by the handler itself or needs config?
    # If formatter needs config, it gets more complex. Assuming default/internal for now.

    return RichLoggingHandler(
        level=level,
        console=console,
        show_path=kwargs.get("show_path", False),
        markup=kwargs.get("markup", True),
        rich_tracebacks=kwargs.get("rich_tracebacks", True)
        # Add other relevant kwargs if needed by RichLoggingHandler
    )