"""
TSAP Terminal Output Formatter.

This module provides terminal-specific output formatting with color support,
tables, and other rich terminal features using the Rich library.
"""

import os
from enum import Enum
from typing import Dict, List, Any, Optional, Union, TextIO

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich.syntax import Syntax
from rich.box import ROUNDED
from rich.tree import Tree

from tsap.output.formatter import OutputFormatter
from tsap.utils.logging import logger, console


class OutputStyle(str, Enum):
    """Style options for terminal output."""
    
    NORMAL = "normal"
    COMPACT = "compact"
    DETAILED = "detailed"
    MINIMAL = "minimal"


class TerminalFormatter(OutputFormatter):
    """Formatter for rich terminal output."""

    def __init__(
        self,
        pretty: bool = True,
        style: OutputStyle = OutputStyle.NORMAL,
        use_color: Optional[bool] = None,
        use_emoji: bool = True,
    ):
        """Initialize the terminal formatter.
        
        Args:
            pretty: Whether to use more decorative formatting
            style: Overall formatting style
            use_color: Force color on/off (None for auto-detection)
            use_emoji: Whether to use emoji in output
        """
        super().__init__(pretty)
        self.style = style if isinstance(style, OutputStyle) else OutputStyle(style)
        self.console = Console(
            color_system="auto" if use_color is None else ("standard" if use_color else None),
            highlight=pretty,
        )
        self.use_emoji = use_emoji

    def format(self, data: Any) -> str:
        """Format the data for terminal display.
        
        Returns a string representation that would be rendered in the terminal.
        For proper rich rendering, use format_stream instead.
        
        Args:
            data: The data to format
            
        Returns:
            String representation of formatted data
        """
        # Create a string buffer to capture the output
        string_io = StringIO()
        
        # Write the formatted output to the buffer
        self.format_stream(data, string_io)
        
        # Return the contents of the buffer
        return string_io.getvalue()

    def format_stream(self, data: Any, stream: TextIO) -> None:
        """Write formatted output directly to a stream.
        
        For proper terminal rendering, use print_to_console instead.
        
        Args:
            data: The data to format
            stream: Output stream to write to
        """
        # For now, just convert to string and write to stream
        # More advanced handling would involve using rich's file export
        if isinstance(data, dict):
            self._format_dict(data, stream)
        elif isinstance(data, list):
            self._format_list(data, stream)
        else:
            stream.write(str(data))
            stream.write("\n")

    def print_to_console(self, data: Any) -> None:
        """Print the formatted data directly to the terminal.
        
        This uses rich's console for proper rendering.
        
        Args:
            data: The data to format and print
        """
        if isinstance(data, dict):
            self._print_dict(data)
        elif isinstance(data, list):
            self._print_list(data)
        else:
            self.console.print(str(data))

    def _print_dict(self, data: Dict[str, Any]) -> None:
        """Print a dictionary to the console with rich formatting.
        
        Args:
            data: Dictionary to print
        """
        if self.style == OutputStyle.MINIMAL:
            # Just print key-value pairs
            for key, value in data.items():
                self.console.print(f"{key}: {value}")
        elif self.style == OutputStyle.COMPACT:
            # Print as a simple table
            table = Table(box=None, show_header=False)
            table.add_column("Key")
            table.add_column("Value")
            
            for key, value in sorted(data.items()):
                table.add_row(
                    Text(str(key), style="bold"),
                    self._format_value_for_table(value)
                )
            
            self.console.print(table)
        else:  # NORMAL or DETAILED
            # Use a panel with a title if available
            title = data.get("title", "Result")
            
            if self.style == OutputStyle.DETAILED and "metadata" in data:
                # Show metadata in a separate panel
                metadata_panel = self._create_metadata_panel(data["metadata"])
                self.console.print(metadata_panel)
            
            # Create a table for the main content
            table = Table(box=ROUNDED if self.pretty else None)
            table.add_column("Key")
            table.add_column("Value")
            
            for key, value in sorted(data.items()):
                if key == "metadata" and self.style == OutputStyle.DETAILED:
                    continue  # Skip metadata as it's already displayed
                
                table.add_row(
                    Text(str(key), style="bold"),
                    self._format_value_for_table(value)
                )
            
            panel = Panel(
                table,
                title=title,
                border_style="blue" if self.pretty else None,
                expand=False,
            )
            
            self.console.print(panel)

    def _print_list(self, data: List[Any]) -> None:
        """Print a list to the console with rich formatting.
        
        Args:
            data: List to print
        """
        if not data:
            self.console.print("[]")
            return
            
        if self.style == OutputStyle.MINIMAL:
            # Just print items
            for item in data:
                self.console.print(str(item))
        elif all(isinstance(item, dict) for item in data):
            # List of dictionaries - try to create a table
            self._print_dict_list(data)
        elif self.style == OutputStyle.COMPACT:
            # Print as a simple list with numbers
            for i, item in enumerate(data):
                self.console.print(f"{i+1}. {str(item)}")
        else:  # NORMAL or DETAILED
            # Use a panel with numbered items
            items = []
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    # For complex items, create a string representation
                    item_str = self._format_complex_value(item)
                else:
                    item_str = str(item)
                
                items.append(f"[bold]{i+1}.[/bold] {item_str}")
            
            panel = Panel(
                "\n".join(items),
                title=f"List ({len(data)} items)",
                border_style="green" if self.pretty else None,
                expand=False,
            )
            
            self.console.print(panel)

    def _print_dict_list(self, data: List[Dict[str, Any]]) -> None:
        """Print a list of dictionaries as a table.
        
        Args:
            data: List of dictionaries to print
        """
        # Collect all field names
        all_fields = set()
        for item in data:
            all_fields.update(item.keys())
        
        # Create a table with the fields as columns
        table = Table(box=ROUNDED if self.pretty else None)
        
        # If there are too many fields, select the most common ones
        if len(all_fields) > 10:
            # Count field occurrences
            field_counts = {}
            for item in data:
                for field in item.keys():
                    field_counts[field] = field_counts.get(field, 0) + 1
            
            # Select the most common fields
            selected_fields = sorted(
                all_fields,
                key=lambda f: (-field_counts.get(f, 0), f)
            )[:10]
            
            # Add a warning about truncated fields
            logger.warning(
                f"Too many fields ({len(all_fields)}), showing only the {len(selected_fields)} most common."
            )
        else:
            selected_fields = sorted(all_fields)
        
        # Add columns
        for field in selected_fields:
            table.add_column(str(field))
        
        # Add rows
        for item in data:
            row = []
            for field in selected_fields:
                value = item.get(field, "")
                row.append(self._format_value_for_table(value))
            
            table.add_row(*row)
        
        self.console.print(table)

    def _format_value_for_table(self, value: Any) -> str:
        """Format a value for display in a table cell.
        
        Args:
            value: Value to format
            
        Returns:
            Formatted string value
        """
        if isinstance(value, (dict, list)):
            # For complex values, use a summary representation
            return self._format_complex_value(value)
        elif value is None:
            return ""
        else:
            return str(value)

    def _format_complex_value(self, value: Union[Dict[str, Any], List[Any]]) -> str:
        """Create a compact string representation of a complex value.
        
        Args:
            value: Complex value to format
            
        Returns:
            Compact string representation
        """
        if isinstance(value, dict):
            if len(value) == 0:
                return "{}"
            
            # For dictionaries, show a count and some keys
            if len(value) <= 3:
                items = [f"{k}: {self._format_simple_value(v)}" for k, v in value.items()]
                return "{" + ", ".join(items) + "}"
            else:
                sample_keys = list(value.keys())[:3]
                sample = ", ".join(f"{k}: {self._format_simple_value(value[k])}" for k in sample_keys)
                return f"{{...{len(value)} keys: {sample}...}}"
        elif isinstance(value, list):
            if len(value) == 0:
                return "[]"
            
            # For lists, show a count and some items
            if len(value) <= 3:
                items = [self._format_simple_value(item) for item in value]
                return "[" + ", ".join(items) + "]"
            else:
                sample = ", ".join(self._format_simple_value(item) for item in value[:3])
                return f"[...{len(value)} items: {sample}...]"
        else:
            return str(value)

    def _format_simple_value(self, value: Any) -> str:
        """Format a value for compact display.
        
        Args:
            value: Value to format
            
        Returns:
            Simple string representation
        """
        if isinstance(value, (dict, list)):
            if isinstance(value, dict):
                return f"{{...{len(value)} keys}}"
            else:
                return f"[...{len(value)} items]"
        elif isinstance(value, str) and len(value) > 30:
            return f'"{value[:27]}..."'
        else:
            return str(value)

    def _create_metadata_panel(self, metadata: Dict[str, Any]) -> Panel:
        """Create a panel to display metadata.
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            Rich Panel with formatted metadata
        """
        # Create a table for the metadata
        table = Table(box=None, show_header=False)
        table.add_column("Key")
        table.add_column("Value")
        
        for key, value in sorted(metadata.items()):
            table.add_row(
                Text(str(key), style="dim bold"),
                Text(str(value), style="dim")
            )
        
        return Panel(
            table,
            title="Metadata",
            border_style="dim blue" if self.pretty else None,
            expand=False,
        )

    def _format_dict(self, data: Dict[str, Any], stream: TextIO) -> None:
        """Format a dictionary to a text stream.
        
        Args:
            data: Dictionary to format
            stream: Output stream
        """
        for key, value in sorted(data.items()):
            if isinstance(value, (dict, list)):
                stream.write(f"{key}:\n")
                if isinstance(value, dict):
                    # Indent the nested dictionary
                    lines = self._get_dict_lines(value)
                    for line in lines:
                        stream.write(f"  {line}\n")
                else:
                    # Indent the nested list
                    lines = self._get_list_lines(value)
                    for line in lines:
                        stream.write(f"  {line}\n")
            else:
                stream.write(f"{key}: {value}\n")

    def _format_list(self, data: List[Any], stream: TextIO) -> None:
        """Format a list to a text stream.
        
        Args:
            data: List to format
            stream: Output stream
        """
        if not data:
            stream.write("[]\n")
            return
            
        for i, item in enumerate(data):
            if isinstance(item, (dict, list)):
                stream.write(f"{i+1}.\n")
                if isinstance(item, dict):
                    # Indent the nested dictionary
                    lines = self._get_dict_lines(item)
                    for line in lines:
                        stream.write(f"  {line}\n")
                else:
                    # Indent the nested list
                    lines = self._get_list_lines(item)
                    for line in lines:
                        stream.write(f"  {line}\n")
            else:
                stream.write(f"{i+1}. {item}\n")

    def _get_dict_lines(self, data: Dict[str, Any]) -> List[str]:
        """Get a list of formatted lines for a dictionary.
        
        Args:
            data: Dictionary to format
            
        Returns:
            List of formatted lines
        """
        lines = []
        for key, value in sorted(data.items()):
            if isinstance(value, (dict, list)):
                lines.append(f"{key}:")
                if isinstance(value, dict):
                    # Indent the nested dictionary
                    nested_lines = self._get_dict_lines(value)
                    for line in nested_lines:
                        lines.append(f"  {line}")
                else:
                    # Indent the nested list
                    nested_lines = self._get_list_lines(value)
                    for line in nested_lines:
                        lines.append(f"  {line}")
            else:
                lines.append(f"{key}: {value}")
        return lines

    def _get_list_lines(self, data: List[Any]) -> List[str]:
        """Get a list of formatted lines for a list.
        
        Args:
            data: List to format
            
        Returns:
            List of formatted lines
        """
        lines = []
        for i, item in enumerate(data):
            if isinstance(item, (dict, list)):
                lines.append(f"{i+1}.")
                if isinstance(item, dict):
                    # Indent the nested dictionary
                    nested_lines = self._get_dict_lines(item)
                    for line in nested_lines:
                        lines.append(f"  {line}")
                else:
                    # Indent the nested list
                    nested_lines = self._get_list_lines(item)
                    for line in nested_lines:
                        lines.append(f"  {line}")
            else:
                lines.append(f"{i+1}. {item}")
        return lines


# Helper functions

def print_search_results(results: Dict[str, Any]) -> None:
    """Print search results with nice formatting.
    
    Args:
        results: Search results dictionary
    """
    console = Console()
    
    # Print summary
    if "total_matches" in results:
        console.print(
            f"[bold green]{results.get('total_matches', 0)}[/bold green] matches found "
            f"in [bold]{results.get('total_files', 0)}[/bold] files"
        )
    
    # Print matches
    if "matches" in results and results["matches"]:
        for match in results["matches"]:
            file_path = match.get("file_path", "")
            line_number = match.get("line_number", 0)
            
            # Create a header with file and line info
            header = Text()
            header.append(os.path.basename(file_path), style="bold cyan")
            header.append(":")
            header.append(str(line_number), style="bold yellow")
            header.append(f" ({file_path})", style="dim")
            
            # Get the match content
            content = match.get("content", "").strip()
            
            # Create a panel with the content
            panel = Panel(
                Syntax(content, "text", theme="ansi_dark", line_numbers=False),
                title=header,
                border_style="blue",
                expand=False,
            )
            
            console.print(panel)
    else:
        console.print("[yellow]No matches found[/yellow]")


def print_table_from_data(
    data: List[Dict[str, Any]],
    columns: Optional[List[str]] = None,
    title: Optional[str] = None,
    show_index: bool = False,
) -> None:
    """Print a table from a list of dictionaries.
    
    Args:
        data: List of dictionaries with data
        columns: List of columns to include (None for all)
        title: Optional table title
        show_index: Whether to show row indices
    """
    if not data:
        console.print("[yellow]No data to display[/yellow]")
        return
    
    # If columns not specified, use all keys from the first dict
    if columns is None:
        columns = sorted(data[0].keys())
    
    # Create the table
    table = Table(title=title, box=ROUNDED)
    
    # Add index column if requested
    if show_index:
        table.add_column("#", style="cyan")
    
    # Add the columns
    for column in columns:
        table.add_column(str(column), style="green")
    
    # Add rows
    for i, row in enumerate(data):
        row_data = []
        
        if show_index:
            row_data.append(str(i + 1))
        
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, (dict, list)):
                # Format complex values
                value = str(type(value).__name__) + f"({len(value)})"
            row_data.append(str(value))
        
        table.add_row(*row_data)
    
    # Print the table
    console.print(table)


def print_tree_from_dict(
    data: Dict[str, Any],
    title: Optional[str] = None,
    exclude_keys: Optional[List[str]] = None,
) -> None:
    """Print a tree visualization of a nested dictionary.
    
    Args:
        data: Dictionary to visualize
        title: Optional tree title
        exclude_keys: Keys to exclude from visualization
    """
    exclude_keys = exclude_keys or []
    
    # Create the tree
    tree = Tree(
        title or "Dictionary Structure",
        guide_style="dim",
    )
    
    # Add items recursively
    def add_items(node, items, prefix=""):
        for key, value in sorted(items.items()):
            if key in exclude_keys:
                continue
                
            if isinstance(value, dict):
                # Add dictionary as a branch
                branch = node.add(f"[bold]{key}[/bold]")
                add_items(branch, value, f"{prefix}.{key}" if prefix else key)
            elif isinstance(value, list):
                # Add list as a branch
                if not value:
                    node.add(f"[bold]{key}[/bold]: []")
                elif all(isinstance(item, dict) for item in value):
                    # List of dictionaries
                    branch = node.add(f"[bold]{key}[/bold] [{len(value)} items]")
                    # Add first few dictionaries
                    for i, item in enumerate(value[:3]):
                        sub_branch = branch.add(f"[{i}]")
                        add_items(sub_branch, item, f"{prefix}.{key}[{i}]" if prefix else f"{key}[{i}]")
                    if len(value) > 3:
                        branch.add(f"... {len(value) - 3} more items ...")
                else:
                    # List of simple values
                    items_sample = str(value[:5])[1:-1]
                    if len(value) > 5:
                        items_sample += ", ..."
                    node.add(f"[bold]{key}[/bold]: [{items_sample}]")
            else:
                # Add simple value as a leaf
                node.add(f"[bold]{key}[/bold]: {value}")
    
    # Populate the tree
    add_items(tree, data)
    
    # Print the tree
    console.print(tree)


# Get a terminal formatter with specific style
def get_terminal_formatter(
    style: str = "normal",
    pretty: bool = True,
    use_color: Optional[bool] = None,
    use_emoji: bool = True,
) -> TerminalFormatter:
    """Get a configured TerminalFormatter.
    
    Args:
        style: Style name ("normal", "compact", "detailed", "minimal")
        pretty: Whether to use decorative formatting
        use_color: Force color on/off (None for auto-detection)
        use_emoji: Whether to use emoji in output
        
    Returns:
        Configured TerminalFormatter
        
    Raises:
        ValueError: If style is not recognized
    """
    try:
        return TerminalFormatter(
            style=OutputStyle(style),
            pretty=pretty,
            use_color=use_color,
            use_emoji=use_emoji,
        )
    except ValueError:
        raise ValueError(
            f"Invalid style '{style}'. Valid styles: {', '.join(s.value for s in OutputStyle)}"
        )


# Add StringIO for testing that doesn't depend on the external io module
class StringIO:
    """Simple string buffer for testing."""
    
    def __init__(self):
        """Initialize an empty buffer."""
        self.buffer = []
    
    def write(self, s: str) -> int:
        """Write a string to the buffer.
        
        Args:
            s: String to write
            
        Returns:
            Number of characters written
        """
        self.buffer.append(s)
        return len(s)
    
    def getvalue(self) -> str:
        """Get the contents of the buffer.
        
        Returns:
            Contents as a string
        """
        return "".join(self.buffer)