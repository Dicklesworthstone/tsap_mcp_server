"""
Command-line interface for TSAP MCP Server.

This module provides the main entry point for the TSAP command-line interface.
"""
import sys
from typing import Optional, List

import typer
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

from tsap.utils.logging import logger, console
from tsap.version import get_version_info
from tsap.config import load_config, get_config, save_config
from tsap.performance_mode import (
    get_performance_mode, set_performance_mode,
    describe_current_mode, PerformanceMode
)
from tsap.server import start_server
from tsap.dependencies import check_dependencies

# Create the Typer app
app = typer.Typer(
    name="tsap",
    help="TSAP MCP Server - Text Search and Processing Model Context Protocol",
    add_completion=False,
)

# Create sub-commands
server_app = typer.Typer(help="Server management commands")
config_app = typer.Typer(help="Configuration management commands")
tool_app = typer.Typer(help="Tool execution commands")
plugin_app = typer.Typer(help="Plugin management commands")
template_app = typer.Typer(help="Template management commands")

# Add sub-commands to main app
app.add_typer(server_app, name="server")
app.add_typer(config_app, name="config")
app.add_typer(tool_app, name="tool")
app.add_typer(plugin_app, name="plugin")
app.add_typer(template_app, name="template")


@app.callback()
def callback(
    ctx: typer.Context,
    config_file: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
    mode: str = typer.Option(
        None, "--mode", "-m", help="Performance mode (fast, standard, deep)"
    ),
):
    """TSAP MCP Server command-line interface."""
    # Logging is now configured by Uvicorn via dictConfig in server.py
    # We just need to determine the level to pass to start_server

    # Load configuration first
    config = None
    try:
        config = load_config(config_file)
    except FileNotFoundError:
        config = load_config() 
    except Exception as e:
        # Log early errors using basic logger before full config might be active
        print(f"ERROR: Failed to load configuration: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine the final log level based on config and verbose flag
    log_level_from_config = config.server.log_level if config and config.server else "INFO"
    final_log_level = "DEBUG" if verbose else log_level_from_config.upper()
    # Store the final level determined by global options/config in context
    ctx.meta["final_log_level"] = final_log_level

    # Set performance mode if provided
    if mode:
        try:
            set_performance_mode(mode)
        except ValueError as e:
            logger.error(f"Invalid performance mode: {e}")
            sys.exit(1)


@app.command()
def version():
    """Show version information."""
    info = get_version_info()
    
    # Create a nice panel with version info
    title = Text()
    title.append("🚀 ", style="bright_blue")
    title.append("TSAP MCP Server", style="bold bright_blue")
    
    version_table = Table(box=None, show_header=False, padding=(0, 2))
    version_table.add_column("Key", style="bright_black")
    version_table.add_column("Value", style="bright_blue")
    
    version_table.add_row("Version", info["package_version"])
    version_table.add_row("Protocol Version", info["protocol_version"])
    version_table.add_row("API Version", info["api_version"])
    version_table.add_row("Min Python Version", info["minimum_python_version"])
    version_table.add_row("Current Mode", get_performance_mode())
    
    panel = Panel(
        version_table,
        title=title,
        border_style="bright_blue",
        padding=(1, 2),
    )
    
    console.print(panel)


@app.command()
def check(
    fix: bool = typer.Option(
        False, "--fix", "-f", help="Attempt to fix missing dependencies"
    ),
):
    """Check if all dependencies are installed and available."""
    logger.info("Checking dependencies...", operation="check_dependencies")
    
    # Check dependencies
    result = check_dependencies(fix=fix)
    
    if all(dep["installed"] for dep in result):
        logger.success("All dependencies are installed and available")
    else:
        logger.warning(
            "Some dependencies are missing or not available",
            details=[
                f"{dep['name']}: {dep['message']}"
                for dep in result if not dep["installed"]
            ]
        )
        
        if not fix:
            logger.info("Run 'tsap check --fix' to attempt to fix missing dependencies")
            
    # Display detailed results
    deps_table = Table(title="Dependencies")
    deps_table.add_column("Name", style="bright_blue")
    deps_table.add_column("Status", style="")
    deps_table.add_column("Version", style="")
    deps_table.add_column("Path", style="")
    
    for dep in result:
        status_style = "green" if dep["installed"] else "red"
        status_icon = "✅" if dep["installed"] else "❌"
        
        deps_table.add_row(
            dep["name"],
            f"[{status_style}]{status_icon} {dep['status']}[/{status_style}]",
            dep.get("version", ""),
            dep.get("path", ""),
        )
        
    console.print(deps_table)


@app.command()
def shell():
    """Open an interactive TSAP shell."""
    # This is a placeholder for now
    logger.error_panel(
        "Not Implemented",
        "The interactive shell is not implemented yet.",
        resolution_steps=["Check for updates in future versions."]
    )


# Server commands

@server_app.command("start")
def server_start(
    ctx: typer.Context,
    host: str = typer.Option(
        None, "--host", "-h", help="Host to bind to"
    ),
    port: int = typer.Option(
        None, "--port", "-p", help="Port to bind to"
    ),
    workers: int = typer.Option(
        None, "--workers", "-w", help="Number of worker processes"
    ),
    log_level: Optional[str] = typer.Option(
        None, "--log-level", "-l", help="Logging level (overrides default/verbose)"
    ),
    reload: bool = typer.Option(
        False, "--reload", "-r", help="Enable auto-reload (development)"
    ),
):
    """Start the TSAP MCP Server."""
    # Prioritize the command-specific --log-level option if provided
    # Otherwise, use the level determined by the global --verbose flag / config
    level_from_callback = ctx.meta.get("final_log_level", "INFO")
    effective_log_level = (log_level or level_from_callback).upper()
    
    try:
        # Pass the effective level to start_server
        start_server(
            host=host,
            port=port,
            workers=workers,
            log_level=effective_log_level, # Pass prioritized level
            reload=reload,
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exception=e)
        sys.exit(1)


@server_app.command("status")
def server_status():
    """Check the status of the TSAP MCP Server."""
    # This is a placeholder for now
    logger.error_panel(
        "Not Implemented",
        "The server status check is not implemented yet.",
        resolution_steps=["Check for updates in future versions."]
    )


# Configuration commands

@config_app.command("show")
def config_show(
    format: str = typer.Option(
        "table", "--format", "-f", help="Output format (table, json, yaml)"
    ),
):
    """Show the current configuration."""
    config = get_config()
    
    if format.lower() == "json":
        import json
        console.print_json(json.dumps(config.dict(), indent=2))
    elif format.lower() == "yaml":
        import yaml
        console.print(yaml.dump(config.dict(), default_flow_style=False))
    else:
        # Show as a table (default)
        config_dict = config.dict()
        
        # Create a table view of the configuration
        logger.section("Current Configuration")
        
        # Function to create a table for a config section
        def create_section_table(section_name, section_data):
            table = Table(title=f"{section_name.title()} Configuration")
            table.add_column("Setting", style="bright_blue")
            table.add_column("Value", style="")
            
            for key, value in section_data.items():
                if isinstance(value, dict):
                    # Skip nested dictionaries, they'll be shown in their own table
                    continue
                table.add_row(key, str(value))
                
            return table
        
        # Create tables for each top-level section
        for section, data in config_dict.items():
            if isinstance(data, dict):
                console.print(create_section_table(section, data))
                
        # Show extra section separately if it has items
        if config_dict.get("extra"):
            console.print(create_section_table("Extra", config_dict["extra"]))


@config_app.command("save")
def config_save(
    path: str = typer.Argument(..., help="Path to save configuration to"),
):
    """Save the current configuration to a file."""
    try:
        save_config(path)
        logger.success(f"Configuration saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}", exception=e)
        sys.exit(1)


@config_app.command("mode")
def config_mode(
    mode: str = typer.Argument(
        None, help="Performance mode to set (fast, standard, deep)"
    ),
):
    """View or set the current performance mode."""
    if mode:
        try:
            set_performance_mode(mode)
            logger.success(f"Performance mode set to {mode}")
        except ValueError as e:
            logger.error(f"Invalid performance mode: {e}")
            sys.exit(1)
    
    # Show current mode info
    mode_info = describe_current_mode()
    
    # Create a nice panel with mode info
    title = Text()
    title.append("⚡ ", style="bright_blue")
    title.append("Performance Mode", style="bold bright_blue")
    
    mode_text = Text()
    mode_text.append("Current mode: ", style="bright_black")
    mode_style = {
        PerformanceMode.FAST: "green",
        PerformanceMode.STANDARD: "blue",
        PerformanceMode.DEEP: "magenta",
    }.get(mode_info["mode"], "blue")
    mode_text.append(mode_info["mode"], style=f"bold {mode_style}")
    mode_text.append("\n\n")
    mode_text.append(mode_info["description"])
    mode_text.append("\n\n")
    
    # Add key parameters
    mode_text.append("Key parameters:", style="bold")
    mode_text.append("\n")
    
    for param, value in mode_info["config"].items():
        mode_text.append(f"• {param}: ", style="bright_black")
        mode_text.append(str(value))
        mode_text.append("\n")
    
    panel = Panel(
        mode_text,
        title=title,
        border_style="bright_blue",
        padding=(1, 2),
    )
    
    console.print(panel)


# Tool commands

@tool_app.command("ripgrep")
def tool_ripgrep(
    pattern: str = typer.Argument(..., help="Search pattern"),
    path: List[str] = typer.Argument(None, help="Files or directories to search"),
    recursive: bool = typer.Option(
        True, "--recursive/--no-recursive", "-r/-R", help="Search recursively"
    ),
    case_sensitive: bool = typer.Option(
        False, "--case-sensitive", "-s", help="Case sensitive search"
    ),
    file_pattern: Optional[str] = typer.Option(
        None, "--file-pattern", "-g", help="File pattern glob"
    ),
    context_lines: int = typer.Option(
        2, "--context", "-C", help="Context lines around match"
    ),
):
    """Run ripgrep search tool."""
    # This is a placeholder for now
    logger.error_panel(
        "Not Implemented",
        "The ripgrep tool command is not implemented yet.",
        resolution_steps=["Check for updates in future versions."]
    )


# Plugin commands

@plugin_app.command("list")
def plugin_list():
    """List installed plugins."""
    # This is a placeholder for now
    logger.error_panel(
        "Not Implemented",
        "The plugin list command is not implemented yet.",
        resolution_steps=["Check for updates in future versions."]
    )


@plugin_app.command("install")
def plugin_install(
    plugin_path: str = typer.Argument(..., help="Path to plugin package or directory"),
):
    """Install a plugin."""
    # This is a placeholder for now
    logger.error_panel(
        "Not Implemented",
        "The plugin install command is not implemented yet.",
        resolution_steps=["Check for updates in future versions."]
    )


# Template commands

@template_app.command("list")
def template_list():
    """List available templates."""
    # This is a placeholder for now
    logger.error_panel(
        "Not Implemented",
        "The template list command is not implemented yet.",
        resolution_steps=["Check for updates in future versions."]
    )


@template_app.command("run")
def template_run(
    template: str = typer.Argument(..., help="Template to run"),
    args: List[str] = typer.Argument(None, help="Template arguments"),
):
    """Run a template."""
    # This is a placeholder for now
    logger.error_panel(
        "Not Implemented",
        "The template run command is not implemented yet.",
        resolution_steps=["Check for updates in future versions."]
    )


# Main entry point
def main():
    """Main entry point for the CLI."""
    try:
        app()
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exception=e)
        sys.exit(1)


if __name__ == "__main__":
    main()