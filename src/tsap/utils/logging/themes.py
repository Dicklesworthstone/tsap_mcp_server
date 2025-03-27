"""
Color themes for TSAP logging system.

This module defines color schemes for different log levels, operations, and components
to provide visual consistency and improve readability of log output.
"""
from typing import Dict, Any, Tuple, Optional
from rich.theme import Theme
from rich.style import Style

# Base color definitions
COLORS = {
    # Main colors
    "primary": "bright_blue",
    "secondary": "cyan",
    "accent": "magenta",
    "success": "green",
    "warning": "yellow",
    "error": "red",
    "critical": "bright_red",
    "info": "bright_blue",
    "debug": "bright_black",
    "trace": "bright_black",
    
    # Component-specific colors
    "core": "blue",
    "composite": "cyan",
    "analysis": "green",
    "evolution": "magenta",
    "plugin": "yellow",
    "template": "bright_cyan",
    "cache": "bright_magenta",
    "output": "bright_green",
    "api": "bright_yellow",
    "mcp": "bright_blue",
    "project": "bright_cyan",
    "storage": "bright_magenta",
    
    # Tool-specific colors
    "ripgrep": "blue",
    "awk": "green",
    "jq": "yellow",
    "sqlite": "magenta",
    "html": "cyan",
    "pdf": "red",
    "table": "bright_blue",
    
    # Performance mode colors
    "fast": "green",
    "standard": "blue",
    "deep": "magenta",
    
    # Result confidence colors
    "high_confidence": "green",
    "medium_confidence": "yellow",
    "low_confidence": "red",
    
    # Evolution-related colors
    "evolving": "cyan",
    "evolved": "green",
    "mutation": "magenta",
    "selection": "yellow",
    "improvement": "green",
    "regression": "red",
    "plateau": "yellow",
    
    # Misc
    "muted": "bright_black",
    "highlight": "bright_white",
    "timestamp": "bright_black",
    "path": "bright_blue",
    "code": "bright_cyan",
    "data": "bright_yellow",
}

# Style definitions (combining color with additional attributes)
STYLES = {
    # Base styles for log levels
    "info": Style(color=COLORS["info"]),
    "debug": Style(color=COLORS["debug"]),
    "warning": Style(color=COLORS["warning"], bold=True),
    "error": Style(color=COLORS["error"], bold=True),
    "critical": Style(color=COLORS["critical"], bold=True, reverse=True),
    "success": Style(color=COLORS["success"], bold=True),
    "trace": Style(color=COLORS["trace"], dim=True),
    
    # Component styles
    "core": Style(color=COLORS["core"], bold=True),
    "composite": Style(color=COLORS["composite"], bold=True),
    "analysis": Style(color=COLORS["analysis"], bold=True),
    "evolution": Style(color=COLORS["evolution"], bold=True),
    "plugin": Style(color=COLORS["plugin"], bold=True),
    "template": Style(color=COLORS["template"], bold=True),
    "cache": Style(color=COLORS["cache"], bold=True),
    "output": Style(color=COLORS["output"], bold=True),
    "api": Style(color=COLORS["api"], bold=True),
    "mcp": Style(color=COLORS["mcp"], bold=True),
    "project": Style(color=COLORS["project"], bold=True),
    "storage": Style(color=COLORS["storage"], bold=True),
    
    # Operation styles
    "operation": Style(color=COLORS["accent"], bold=True),
    "search": Style(color=COLORS["info"], bold=True),
    "analyze": Style(color=COLORS["analysis"], bold=True),
    "transform": Style(color=COLORS["secondary"], bold=True),
    
    # Performance mode styles
    "fast_mode": Style(color=COLORS["fast"], bold=True),
    "standard_mode": Style(color=COLORS["standard"], bold=True),
    "deep_mode": Style(color=COLORS["deep"], bold=True),
    
    # Confidence level styles
    "high_confidence": Style(color=COLORS["high_confidence"], bold=True),
    "medium_confidence": Style(color=COLORS["medium_confidence"], bold=True),
    "low_confidence": Style(color=COLORS["low_confidence"], bold=True),
    
    # Evolution styles
    "evolving": Style(color=COLORS["evolving"]),
    "evolved": Style(color=COLORS["evolved"], bold=True),
    "mutation": Style(color=COLORS["mutation"]),
    "improvement": Style(color=COLORS["improvement"], bold=True),
    "regression": Style(color=COLORS["regression"], bold=True),
    "plateau": Style(color=COLORS["plateau"]),
    
    # Misc styles
    "timestamp": Style(color=COLORS["timestamp"], dim=True),
    "path": Style(color=COLORS["path"], underline=True),
    "code": Style(color=COLORS["code"], italic=True),
    "data": Style(color=COLORS["data"]),
    "muted": Style(color=COLORS["muted"], dim=True),
    "highlight": Style(color=COLORS["highlight"], bold=True),
}

# Rich theme that can be used directly with Rich Console
RICH_THEME = Theme({name: style for name, style in STYLES.items()})

# Get the appropriate style for a log level
def get_level_style(level: str) -> Style:
    """Get the Rich style for a specific log level.
    
    Args:
        level: The log level (info, debug, warning, error, critical, success, trace)
        
    Returns:
        The corresponding Rich Style
    """
    level = level.lower()
    return STYLES.get(level, STYLES["info"])

# Get style for a component
def get_component_style(component: str) -> Style:
    """Get the Rich style for a specific component.
    
    Args:
        component: The component name (core, composite, analysis, etc.)
        
    Returns:
        The corresponding Rich Style
    """
    component = component.lower()
    return STYLES.get(component, STYLES["info"])

# Get color by name
def get_color(name: str) -> str:
    """Get a color by name.
    
    Args:
        name: The color name
        
    Returns:
        The color string that can be used with Rich
    """
    return COLORS.get(name.lower(), COLORS["primary"])

# Apply style to text directly
def style_text(text: str, style_name: str) -> str:
    """Apply a named style to text (for use without Rich console).
    
    This is a utility function that doesn't depend on Rich, useful for
    simple terminal output or when Rich console isn't available.
    
    Args:
        text: The text to style
        style_name: The name of the style to apply
        
    Returns:
        Text with ANSI color codes applied
    """
    # This is a simplified version - in a real implementation, you would
    # convert Rich styles to ANSI escape codes here
    return f"[{style_name}]{text}[/{style_name}]"

# Get foreground and background colors for a specific context
def get_context_colors(
    context: str, component: Optional[str] = None
) -> Tuple[str, Optional[str]]:
    """Get appropriate foreground and background colors for a log context.
    
    Args:
        context: The log context (e.g., 'search', 'analyze')
        component: Optional component name for more specific styling
        
    Returns:
        Tuple of (foreground_color, background_color)
    """
    fg_color = COLORS.get(context.lower(), COLORS["primary"])
    bg_color = None
    
    # Special cases for specific combinations
    if context.lower() == "error":
        bg_color = "bright_red"
        fg_color = "white"
    elif context.lower() == "critical":
        bg_color = "red"
        fg_color = "white"
    elif context.lower() == "warning" and component == "security":
        bg_color = "yellow"
        fg_color = "black"
    
    return (fg_color, bg_color)