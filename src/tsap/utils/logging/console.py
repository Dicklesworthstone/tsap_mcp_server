"""
Rich console configuration for TSAP logging system.

This module provides a configured Rich console instance for beautiful terminal output,
along with utility functions for common console operations.
"""
import sys
from typing import Optional, Dict, Any, List, Union, Tuple
from contextlib import contextmanager

from rich.console import Console, ConsoleOptions, RenderResult
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.box import Box, ROUNDED
from rich.progress import Progress, TaskID, TextColumn, BarColumn, TimeElapsedColumn
from rich.progress import TimeRemainingColumn, SpinnerColumn, ProgressColumn
from rich.status import Status
from rich.syntax import Syntax
from rich.traceback import install as install_rich_traceback
from rich.live import Live
from rich.tree import Tree
from rich.style import Style

from .themes import RICH_THEME, get_level_style, get_component_style

# Configure global console with our theme
console = Console(
    theme=RICH_THEME,
    highlight=True,
    markup=True,
    emoji=True,
    record=False,
    width=None,  # Auto-width
    color_system="auto",
)

# Install rich traceback handler for beautiful error tracebacks
install_rich_traceback(console=console, show_locals=True)

# Custom progress bar setup
def create_progress(
    transient: bool = True,
    auto_refresh: bool = True,
    disable: bool = False,
    **kwargs
) -> Progress:
    """Create a customized Rich Progress instance.
    
    Args:
        transient: Whether to remove the progress bar after completion
        auto_refresh: Whether to auto-refresh the progress bar
        disable: Whether to disable the progress bar
        **kwargs: Additional arguments passed to Progress constructor
        
    Returns:
        Configured Progress instance
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=transient,
        auto_refresh=auto_refresh,
        disable=disable,
        **kwargs
    )

@contextmanager
def status(message: str, spinner: str = "dots", **kwargs):
    """Context manager for displaying a status message during an operation.
    
    Args:
        message: The status message to display
        spinner: The spinner animation to use
        **kwargs: Additional arguments to pass to Status constructor
    
    Yields:
        Rich Status object that can be updated
    """
    with Status(message, console=console, spinner=spinner, **kwargs) as status:
        yield status

def print_panel(
    content: Union[str, Text],
    title: Optional[str] = None,
    style: Optional[str] = "info",
    box: Optional[Box] = ROUNDED,
    expand: bool = True,
    padding: Tuple[int, int] = (1, 2),
    **kwargs
) -> None:
    """Print content in a styled panel.
    
    Args:
        content: The content to display in the panel
        title: Optional panel title
        style: Style name to apply (from theme)
        box: Box style to use
        expand: Whether the panel should expand to fill width
        padding: Panel padding (vertical, horizontal)
        **kwargs: Additional arguments to pass to Panel constructor
    """
    if isinstance(content, str):
        content = Text(content)
    
    panel = Panel(
        content,
        title=title,
        style=style,
        box=box,
        expand=expand,
        padding=padding,
        **kwargs
    )
    console.print(panel)

def print_syntax(
    code: str,
    language: str = "python",
    line_numbers: bool = True,
    theme: str = "monokai",
    title: Optional[str] = None,
    **kwargs
) -> None:
    """Print syntax-highlighted code.
    
    Args:
        code: The code to highlight
        language: The programming language
        line_numbers: Whether to show line numbers
        theme: Syntax highlighting theme
        title: Optional title for the code block
        **kwargs: Additional arguments to pass to Syntax constructor
    """
    syntax = Syntax(
        code,
        language,
        theme=theme,
        line_numbers=line_numbers,
        **kwargs
    )
    
    if title:
        print_panel(syntax, title=title)
    else:
        console.print(syntax)

def print_table(
    title: Optional[str] = None,
    columns: Optional[List[str]] = None,
    rows: Optional[List[List[Any]]] = None,
    box: Box = ROUNDED,
    **kwargs
) -> Table:
    """Create and print a Rich table.
    
    Args:
        title: Optional table title
        columns: List of column names
        rows: List of rows, each a list of values
        box: Box style to use
        **kwargs: Additional arguments to pass to Table constructor
        
    Returns:
        The created Table instance (in case further modification is needed)
    """
    table = Table(title=title, box=box, **kwargs)
    
    if columns:
        for column in columns:
            table.add_column(column)
            
    if rows:
        for row in rows:
            table.add_row(*[str(item) for item in row])
    
    console.print(table)
    return table

def print_tree(
    name: str,
    data: Dict[str, Any],
    guide_style: str = "bright_black",
    **kwargs
) -> None:
    """Print a hierarchical tree structure from nested data.
    
    Args:
        name: The root name of the tree
        data: Nested dictionary to render as a tree
        guide_style: Style for the tree guides
        **kwargs: Additional arguments to pass to Tree constructor
    """
    tree = Tree(name, guide_style=guide_style, **kwargs)
    
    def build_tree(tree, data):
        """Recursively build the tree from nested data."""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    branch = tree.add(str(key))
                    build_tree(branch, value)
                else:
                    tree.add(f"{key}: {value}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    branch = tree.add(f"[{i}]")
                    build_tree(branch, item)
                else:
                    tree.add(str(item))
    
    build_tree(tree, data)
    console.print(tree)

def print_json(data: Dict[str, Any], title: Optional[str] = None) -> None:
    """Print formatted JSON data.
    
    Args:
        data: Dictionary to print as JSON
        title: Optional title for the JSON block
    """
    import json
    
    json_str = json.dumps(data, indent=2)
    print_syntax(json_str, language="json", title=title)

# Utility for creating a live updating display
@contextmanager
def live_display(**kwargs):
    """Context manager for a live updating display.
    
    Args:
        **kwargs: Arguments to pass to Live constructor
        
    Yields:
        Rich Live object
    """
    with Live(console=console, **kwargs) as live:
        yield live