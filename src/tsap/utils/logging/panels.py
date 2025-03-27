"""
Rich panel layouts for TSAP output.

This module provides specialized panel types for different output contexts,
with consistent styling and structure.
"""
from typing import Optional, List, Dict, Any, Union, Tuple
from datetime import datetime, timedelta

from rich.panel import Panel
from rich.console import Console, Group, ConsoleRenderable
from rich.text import Text
from rich.table import Table
from rich.box import Box, ROUNDED, SIMPLE, DOUBLE
from rich.style import Style
from rich.columns import Columns
from rich.layout import Layout
from rich.rule import Rule
from rich.markdown import Markdown
from rich.align import Align
from rich.syntax import Syntax

from .console import console
from .emojis import get_emoji
from .themes import get_level_style, get_component_style

class HeaderPanel:
    """Creates a header panel for major sections of output."""
    
    def __init__(
        self,
        title: str,
        subtitle: Optional[str] = None,
        icon: Optional[str] = None,
        component: Optional[str] = None,
        style: Optional[str] = "primary",
        width: Optional[int] = None,
    ):
        """Initialize a header panel.
        
        Args:
            title: Main title text
            subtitle: Optional subtitle text
            icon: Emoji or icon character
            component: TSAP component name (for styling)
            style: Style name to apply (from theme)
            width: Optional explicit width
        """
        self.title = title
        self.subtitle = subtitle
        self.icon = icon
        self.component = component
        self.style = style
        self.width = width
    
    def __rich__(self) -> ConsoleRenderable:
        """Render as a Rich console renderable.
        
        Returns:
            A Rich Panel
        """
        # Determine icon
        icon = self.icon
        if not icon and self.component:
            icon = get_emoji("component", self.component)
            
        # Create title with icon
        title_text = Text()
        if icon:
            title_text.append(f"{icon} ", style=self.style)
        title_text.append(self.title, style=f"bold {self.style}")
        
        # Create complete content
        content = [title_text]
        
        if self.subtitle:
            subtitle_text = Text(self.subtitle, style="dim")
            content.append(subtitle_text)
            
        # Determine panel style
        panel_style = self.style
        if self.component:
            component_style = get_component_style(self.component)
            panel_style = str(component_style)
            
        # Create and return the panel
        return Panel(
            Group(*content),
            box=DOUBLE,
            style=panel_style,
            width=self.width,
            padding=(1, 2),
        )

class ResultPanel:
    """Panel for displaying operation results with status indicators."""
    
    def __init__(
        self,
        title: str,
        results: Union[List[Dict[str, Any]], Dict[str, Any]],
        status: str = "success",
        component: Optional[str] = None,
        show_count: bool = True,
        compact: bool = False,
        width: Optional[int] = None,
    ):
        """Initialize a result panel.
        
        Args:
            title: Panel title
            results: Results to display (list of dicts or single dict)
            status: Result status (success, warning, error)
            component: TSAP component name (for styling)
            show_count: Whether to show result count in title
            compact: Whether to use a compact display style
            width: Optional explicit width
        """
        self.title = title
        self.results = results if isinstance(results, list) else [results]
        self.status = status.lower()
        self.component = component
        self.show_count = show_count
        self.compact = compact
        self.width = width
    
    def __rich__(self) -> ConsoleRenderable:
        """Render as a Rich console renderable.
        
        Returns:
            A Rich Panel or Group
        """
        # Determine status icon and style
        status_emoji = get_emoji("level", self.status)
        status_style = get_level_style(self.status)
        
        # Create title with count
        title_text = Text()
        title_text.append(f"{status_emoji} ", style=str(status_style))
        title_text.append(self.title, style="bold")
        
        if self.show_count and len(self.results) > 0:
            title_text.append(f" ({len(self.results)})", style="dim")
            
        # Create content based on results
        if len(self.results) == 0:
            # No results
            content = Text("No results found", style="dim")
        elif self.compact:
            # Compact display for many results
            content = self._create_compact_display()
        else:
            # Standard display
            content = self._create_standard_display()
            
        # Determine panel style
        panel_style = str(status_style)
        if self.component:
            component_style = get_component_style(self.component)
            # Blend the styles - component for the border, status for the content
            panel_style = str(component_style)
            
        # Create and return the panel
        return Panel(
            content,
            title=title_text,
            box=ROUNDED,
            style=panel_style,
            width=self.width,
            padding=(1, 2),
        )
    
    def _create_standard_display(self) -> ConsoleRenderable:
        """Create a standard display for the results.
        
        Returns:
            A Rich console renderable
        """
        # For a small number of results, show detailed info for each
        items = []
        
        for i, result in enumerate(self.results):
            # Create a sub-panel for each result
            if isinstance(result, dict) and len(result) > 0:
                # Create a table for the result properties
                table = Table(box=None, show_header=False, padding=(0, 1))
                table.add_column("Key", style="bright_black")
                table.add_column("Value")
                
                # Add rows for each property
                for key, value in result.items():
                    # Format the value based on type
                    if isinstance(value, (list, tuple)) and len(value) > 0:
                        # For lists, show as a comma-separated string or count
                        if len(value) > 3:
                            formatted = f"{len(value)} items"
                        else:
                            formatted = ", ".join(str(v) for v in value)
                    elif isinstance(value, dict) and len(value) > 0:
                        # For dicts, show count of key-value pairs
                        formatted = f"{len(value)} properties"
                    elif isinstance(value, bool):
                        # Format booleans with checkmarks/x
                        formatted = "✓" if value else "✗"
                    elif isinstance(value, (int, float)):
                        # Keep numbers as is
                        formatted = str(value)
                    elif value is None:
                        # Show None as empty
                        formatted = ""
                    else:
                        # Convert everything else to string
                        formatted = str(value)
                        
                    table.add_row(key, formatted)
                
                items.append(table)
            else:
                # If not a dict or empty, just show the result as a string
                items.append(Text(str(result)))
            
            # Add a separator between results
            if i < len(self.results) - 1:
                items.append(Rule(style="bright_black"))
                
        return Group(*items)
    
    def _create_compact_display(self) -> ConsoleRenderable:
        """Create a compact display for many results.
        
        Returns:
            A Rich console renderable
        """
        # For many results, create a more compact table
        if not self.results or not isinstance(self.results[0], dict):
            # If not dicts or empty, fall back to standard display
            return self._create_standard_display()
            
        # Extract all unique keys from the results
        all_keys = set()
        for result in self.results:
            all_keys.update(result.keys())
            
        # If too many keys, just show the most common ones
        if len(all_keys) > 5:
            # Count key occurrences
            key_counts = {}
            for result in self.results:
                for key in result:
                    key_counts[key] = key_counts.get(key, 0) + 1
                    
            # Select the most common keys (up to 5)
            top_keys = sorted(key_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            keys_to_show = [k for k, _ in top_keys]
        else:
            # Show all keys if not too many
            keys_to_show = sorted(all_keys)
            
        # Create a table with the selected keys as columns
        table = Table(box=SIMPLE, show_lines=False)
        
        # Add columns
        for key in keys_to_show:
            table.add_column(key)
            
        # Add rows for each result
        for result in self.results:
            row = []
            for key in keys_to_show:
                value = result.get(key, "")
                
                # Format the value
                if isinstance(value, (list, tuple)):
                    if len(value) > 3:
                        formatted = f"{len(value)} items"
                    else:
                        formatted = ", ".join(str(v) for v in value)
                elif isinstance(value, dict):
                    formatted = f"{len(value)} properties"
                elif isinstance(value, bool):
                    formatted = "✓" if value else "✗"
                elif value is None:
                    formatted = ""
                else:
                    formatted = str(value)
                    
                row.append(formatted)
                
            table.add_row(*row)
            
        return table

class InfoPanel:
    """Panel for displaying contextual information and help."""
    
    def __init__(
        self,
        title: str,
        content: Union[str, List[str], Dict[str, Any]],
        icon: Optional[str] = None,
        style: str = "info",
        width: Optional[int] = None,
    ):
        """Initialize an info panel.
        
        Args:
            title: Panel title
            content: Content to display (string, list, or dict)
            icon: Emoji or icon character
            style: Style name to apply (from theme)
            width: Optional explicit width
        """
        self.title = title
        self.content = content
        self.icon = icon or get_emoji("level", "info")
        self.style = style
        self.width = width
    
    def __rich__(self) -> ConsoleRenderable:
        """Render as a Rich console renderable.
        
        Returns:
            A Rich Panel
        """
        # Create title with icon
        title_text = Text()
        title_text.append(f"{self.icon} ", style=self.style)
        title_text.append(self.title, style=f"bold {self.style}")
        
        # Format content based on type
        if isinstance(self.content, str):
            # String content can be displayed directly
            content = self.content
        elif isinstance(self.content, list):
            # For a list, create a bullet list
            items = []
            for item in self.content:
                items.append(f"• {item}")
            content = "\n".join(items)
        elif isinstance(self.content, dict):
            # For a dict, create a key-value table
            table = Table(box=None, show_header=False, padding=(0, 1))
            table.add_column("Key", style="bright_black")
            table.add_column("Value")
            
            for key, value in self.content.items():
                table.add_row(key, str(value))
                
            content = table
        else:
            # Fallback for other types
            content = str(self.content)
            
        # Create and return the panel
        return Panel(
            content,
            title=title_text,
            box=ROUNDED,
            style=self.style,
            width=self.width,
            padding=(1, 2),
        )

class WarningPanel:
    """Panel for displaying warnings and cautions."""
    
    def __init__(
        self,
        title: Optional[str] = None,
        message: str = "",
        details: Optional[List[str]] = None,
        width: Optional[int] = None,
    ):
        """Initialize a warning panel.
        
        Args:
            title: Optional panel title
            message: Main warning message
            details: Optional list of detail points
            width: Optional explicit width
        """
        self.title = title or "Warning"
        self.message = message
        self.details = details or []
        self.width = width
    
    def __rich__(self) -> ConsoleRenderable:
        """Render as a Rich console renderable.
        
        Returns:
            A Rich Panel
        """
        # Create title with warning emoji
        title_text = Text()
        title_text.append(f"{get_emoji('level', 'warning')} ", style="warning")
        title_text.append(self.title, style="bold warning")
        
        # Create content with message and details
        content_items = [Text(self.message, style="warning")]
        
        if self.details:
            # Add a separator
            content_items.append(Text(""))
            
            # Add each detail as a bullet point
            for detail in self.details:
                detail_text = Text("• ", style="bright_black")
                detail_text.append(detail)
                content_items.append(detail_text)
                
        # Create and return the panel
        return Panel(
            Group(*content_items),
            title=title_text,
            box=ROUNDED,
            border_style="warning",
            width=self.width,
            padding=(1, 2),
        )

class ErrorPanel:
    """Panel for displaying errors with optional resolution steps."""
    
    def __init__(
        self,
        title: Optional[str] = None,
        message: str = "",
        details: Optional[str] = None,
        resolution_steps: Optional[List[str]] = None,
        error_code: Optional[str] = None,
        width: Optional[int] = None,
    ):
        """Initialize an error panel.
        
        Args:
            title: Optional panel title
            message: Main error message
            details: Optional error details
            resolution_steps: Optional list of steps to resolve the error
            error_code: Optional error code for reference
            width: Optional explicit width
        """
        self.title = title or "Error"
        self.message = message
        self.details = details
        self.resolution_steps = resolution_steps or []
        self.error_code = error_code
        self.width = width
    
    def __rich__(self) -> ConsoleRenderable:
        """Render as a Rich console renderable.
        
        Returns:
            A Rich Panel
        """
        # Create title with error emoji and optional error code
        title_text = Text()
        title_text.append(f"{get_emoji('level', 'error')} ", style="error")
        title_text.append(self.title, style="bold error")
        
        if self.error_code:
            title_text.append(f" [Code: {self.error_code}]", style="bright_black")
            
        # Create content with message, details, and resolution steps
        content_items = [Text(self.message, style="bold error")]
        
        if self.details:
            # Add details
            content_items.append(Text(""))
            content_items.append(Text(self.details))
            
        if self.resolution_steps:
            # Add resolution steps
            content_items.append(Text(""))
            content_items.append(Text("Resolution Steps:", style="bold"))
            
            for i, step in enumerate(self.resolution_steps, 1):
                step_text = Text(f"{i}. ", style="bright_black")
                step_text.append(step)
                content_items.append(step_text)
                
        # Create and return the panel
        return Panel(
            Group(*content_items),
            title=title_text,
            box=ROUNDED,
            border_style="error",
            width=self.width,
            padding=(1, 2),
        )

class ToolOutputPanel:
    """Panel for displaying tool command output."""
    
    def __init__(
        self,
        tool: str,
        command: str,
        output: str,
        status: str = "success",
        duration: Optional[float] = None,
        width: Optional[int] = None,
    ):
        """Initialize a tool output panel.
        
        Args:
            tool: Tool name (ripgrep, awk, jq, etc.)
            command: Command that was executed
            output: Command output text
            status: Execution status (success, error)
            duration: Optional execution duration in seconds
            width: Optional explicit width
        """
        self.tool = tool.lower()
        self.command = command
        self.output = output
        self.status = status.lower()
        self.duration = duration
        self.width = width
    
    def __rich__(self) -> ConsoleRenderable:
        """Render as a Rich console renderable.
        
        Returns:
            A Rich Panel
        """
        # Get tool emoji and style
        tool_emoji = get_emoji("tool", self.tool) or "🔧"
        
        # Create title
        title_text = Text()
        title_text.append(f"{tool_emoji} ", style=self.tool)
        title_text.append(self.tool.upper(), style=f"bold {self.tool}")
        
        if self.duration is not None:
            duration_text = f" ({self.duration:.2f}s)"
            title_text.append(duration_text, style="bright_black")
            
        # Status indicator
        status_emoji = get_emoji("level", self.status)
        status_text = Text(f"{status_emoji} {self.status.upper()}", style=self.status)
        
        # Command text
        command_text = Text("$ ", style="bright_black")
        command_text.append(self.command, style="bold")
        
        # Format output
        if not self.output:
            output_content = Text("(No output)", style="dim")
        else:
            output_content = Text(self.output)
        
        # Create full content
        content = Group(
            status_text,
            Text(""),  # Empty line
            command_text,
            Text(""),  # Empty line
            output_content,
        )
            
        # Create and return the panel
        return Panel(
            content,
            title=title_text,
            box=SIMPLE,
            border_style=self.tool,
            width=self.width,
            padding=(1, 2),
        )

class CodePanel:
    """Panel for displaying code snippets with syntax highlighting."""
    
    def __init__(
        self,
        code: str,
        language: str = "python",
        title: Optional[str] = None,
        line_numbers: bool = True,
        highlight_lines: Optional[List[int]] = None,
        width: Optional[int] = None,
    ):
        """Initialize a code panel.
        
        Args:
            code: The code to display
            language: Programming language for syntax highlighting
            title: Optional panel title
            line_numbers: Whether to show line numbers
            highlight_lines: List of line numbers to highlight
            width: Optional explicit width
        """
        self.code = code
        self.language = language
        self.title = title
        self.line_numbers = line_numbers
        self.highlight_lines = highlight_lines
        self.width = width
    
    def __rich__(self) -> ConsoleRenderable:
        """Render as a Rich console renderable.
        
        Returns:
            A Rich Panel
        """
        # Create syntax-highlighted code
        syntax = Syntax(
            self.code,
            self.language,
            theme="monokai",
            line_numbers=self.line_numbers,
            word_wrap=True,
            highlight_lines=set(self.highlight_lines or []),
        )
        
        # Create title if provided
        title_text = None
        if self.title:
            title_text = Text()
            title_text.append("📝 ", style="code")
            title_text.append(self.title, style="bold code")
            
        # Create and return the panel
        return Panel(
            syntax,
            title=title_text,
            box=ROUNDED,
            border_style="code",
            width=self.width,
            padding=(1, 2),
        )

# Convenience functions to create and display panels

def display_header(
    title: str,
    subtitle: Optional[str] = None,
    icon: Optional[str] = None,
    component: Optional[str] = None,
    style: Optional[str] = "primary",
    console: Optional[Console] = None,
) -> None:
    """Display a header panel.
    
    Args:
        title: Main title text
        subtitle: Optional subtitle text
        icon: Emoji or icon character
        component: TSAP component name (for styling)
        style: Style name to apply (from theme)
        console: Optional console to use
    """
    panel = HeaderPanel(title, subtitle, icon, component, style)
    (console or globals().get("console", Console())).print(panel)

def display_results(
    title: str,
    results: Union[List[Dict[str, Any]], Dict[str, Any]],
    status: str = "success",
    component: Optional[str] = None,
    show_count: bool = True,
    compact: bool = False,
    console: Optional[Console] = None,
) -> None:
    """Display a results panel.
    
    Args:
        title: Panel title
        results: Results to display (list of dicts or single dict)
        status: Result status (success, warning, error)
        component: TSAP component name (for styling)
        show_count: Whether to show result count in title
        compact: Whether to use a compact display style
        console: Optional console to use
    """
    panel = ResultPanel(title, results, status, component, show_count, compact)
    (console or globals().get("console", Console())).print(panel)

def display_info(
    title: str,
    content: Union[str, List[str], Dict[str, Any]],
    icon: Optional[str] = None,
    style: str = "info",
    console: Optional[Console] = None,
) -> None:
    """Display an info panel.
    
    Args:
        title: Panel title
        content: Content to display (string, list, or dict)
        icon: Emoji or icon character
        style: Style name to apply (from theme)
        console: Optional console to use
    """
    panel = InfoPanel(title, content, icon, style)
    (console or globals().get("console", Console())).print(panel)

def display_warning(
    title: Optional[str] = None,
    message: str = "",
    details: Optional[List[str]] = None,
    console: Optional[Console] = None,
) -> None:
    """Display a warning panel.
    
    Args:
        title: Optional panel title
        message: Main warning message
        details: Optional list of detail points
        console: Optional console to use
    """
    panel = WarningPanel(title, message, details)
    (console or globals().get("console", Console())).print(panel)

def display_error(
    title: Optional[str] = None,
    message: str = "",
    details: Optional[str] = None,
    resolution_steps: Optional[List[str]] = None,
    error_code: Optional[str] = None,
    console: Optional[Console] = None,
) -> None:
    """Display an error panel.
    
    Args:
        title: Optional panel title
        message: Main error message
        details: Optional error details
        resolution_steps: Optional list of steps to resolve the error
        error_code: Optional error code for reference
        console: Optional console to use
    """
    panel = ErrorPanel(title, message, details, resolution_steps, error_code)
    (console or globals().get("console", Console())).print(panel)

def display_tool_output(
    tool: str,
    command: str,
    output: str,
    status: str = "success",
    duration: Optional[float] = None,
    console: Optional[Console] = None,
) -> None:
    """Display a tool output panel.
    
    Args:
        tool: Tool name (ripgrep, awk, jq, etc.)
        command: Command that was executed
        output: Command output text
        status: Execution status (success, error)
        duration: Optional execution duration in seconds
        console: Optional console to use
    """
    panel = ToolOutputPanel(tool, command, output, status, duration)
    (console or globals().get("console", Console())).print(panel)

def display_code(
    code: str,
    language: str = "python",
    title: Optional[str] = None,
    line_numbers: bool = True,
    highlight_lines: Optional[List[int]] = None,
    console: Optional[Console] = None,
) -> None:
    """Display a code panel.
    
    Args:
        code: The code to display
        language: Programming language for syntax highlighting
        title: Optional panel title
        line_numbers: Whether to show line numbers
        highlight_lines: List of line numbers to highlight
        console: Optional console to use
    """
    panel = CodePanel(code, language, title, line_numbers, highlight_lines)
    (console or globals().get("console", Console())).print(panel)