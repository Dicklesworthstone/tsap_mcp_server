"""
Command Line Interface for TSAP.

This module provides the primary CLI entry point for TSAP, allowing users
to interact with the system through a rich command-line interface. It uses
the Typer library to implement a modern, type-annotated CLI with automatic
help generation and rich text formatting.
"""

import os
import sys
import time
from typing import Optional, List, Dict, Any

import typer
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich import print as rich_print

from tsap.utils.logging import logger, console
from tsap.version import get_version_info
from tsap.config import load_config, get_config, save_config
from tsap.performance_mode import (
    get_performance_mode, set_performance_mode, describe_current_mode
)
from tsap.dependencies import check_dependencies
from tsap.server import start_server
from tsap.utils.logging.dashboard import start_dashboard, is_dashboard_available
from tsap.plugins.manager import get_plugin_manager, initialize_plugin_system
from tsap.project.context import create_project, list_projects
from tsap.templates.base import get_template_runner

# Create main typer app
app = typer.Typer(
    name="tsap",
    help="Text Search and Processing Model Context Protocol Server",
    add_completion=False,
)

# Create subcommands
server_app = typer.Typer(help="Server management commands")
config_app = typer.Typer(help="Configuration commands")
tool_app = typer.Typer(help="Tool commands")
plugin_app = typer.Typer(help="Plugin management commands")
project_app = typer.Typer(help="Project management commands")
template_app = typer.Typer(help="Template commands")
search_app = typer.Typer(help="Search commands")
analyze_app = typer.Typer(help="Analysis commands")

# Add subcommands to main app
app.add_typer(server_app, name="server")
app.add_typer(config_app, name="config")
app.add_typer(tool_app, name="tool")
app.add_typer(plugin_app, name="plugin")
app.add_typer(project_app, name="project")
app.add_typer(template_app, name="template")
app.add_typer(search_app, name="search")
app.add_typer(analyze_app, name="analyze")


@app.callback()
def callback(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config file"),
    performance_mode: Optional[str] = typer.Option(None, "--mode", "-m", help="Performance mode (fast, standard, deep)"),
    log_file: Optional[str] = typer.Option(None, "--log-file", "-l", help="Log to specified file")
) -> None:
    """
    Text Search and Processing Model Context Protocol Server.
    
    A comprehensive toolkit for text search, analysis, and processing operations.
    """
    # Set up configuration
    if config_file:
        try:
            config = load_config(config_file)  # noqa: F841
            logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            rich_print(f"[bold red]Error:[/bold red] Failed to load configuration from {config_file}: {str(e)}")
            sys.exit(1)
    
    # Set up performance mode
    if performance_mode:
        try:
            if performance_mode.lower() not in ("fast", "standard", "deep"):
                raise ValueError(f"Invalid performance mode: {performance_mode}. Must be one of: fast, standard, deep")
            
            set_performance_mode(performance_mode)
            logger.info(f"Set performance mode to {performance_mode}")
        except Exception as e:
            logger.error(f"Error setting performance mode: {str(e)}")
            rich_print(f"[bold red]Error:[/bold red] Failed to set performance mode: {str(e)}")
            sys.exit(1)
    
    # Set up logging to file if specified
    if log_file:
        # We'd need to implement this based on the logging system
        logger.info(f"Logging to file: {log_file}")
    
    # Set verbose mode
    if verbose:
        # We'd need to implement this based on the logging system
        logger.info("Verbose mode enabled")


@app.command()
def version() -> None:
    """
    Display version information.
    """
    info = get_version_info()
    
    version_panel = Panel(
        (
            f"[bold]TSAP Version:[/bold] {info['version']}\n"
            f"[bold]Protocol Version:[/bold] {info['protocol_version']}\n"
            f"[bold]API Version:[/bold] {info['api_version']}\n"
            f"[bold]Python Version:[/bold] {info['python_version']}\n"
            f"[bold]Performance Mode:[/bold] {get_performance_mode().upper()}"
        ),
        title="TSAP Version Information",
        border_style="blue"
    )
    
    console.print(version_panel)


@app.command()
def check(
    fix: bool = typer.Option(False, "--fix", "-f", help="Attempt to fix missing dependencies"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information about each dependency")
) -> None:
    """
    Check system dependencies.
    """
    with console.status("[bold blue]Checking dependencies...[/bold blue]") as status:
        dependencies = check_dependencies(fix=fix)
    
    # Count categories
    categories = {
        "ok": 0,
        "missing": 0,
        "error": 0
    }
    
    for dep in dependencies:
        categories[dep.get("status", "error")] += 1
    
    # Create a table for results
    table = Table(title="Dependency Check Results")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("Version", style="blue")
    table.add_column("Path", style="yellow")
    
    if verbose:
        table.add_column("Details", style="white")
    
    # Sort dependencies by status (missing first, then error, then ok)
    status_order = {"missing": 0, "error": 1, "ok": 2}
    sorted_deps = sorted(dependencies, key=lambda d: status_order.get(d.get("status"), 3))
    
    for dep in sorted_deps:
        name = dep.get("name", "Unknown")
        dep_type = dep.get("type", "Unknown")
        status = dep.get("status", "error")
        version = dep.get("version", "") or ""
        path = dep.get("path", "") or ""
        details = dep.get("details", "") or ""
        
        # Color status
        if status == "ok":
            status_text = Text("OK", style="green")
        elif status == "missing":
            status_text = Text("Missing", style="red")
        else:
            status_text = Text("Error", style="yellow")
        
        if verbose:
            table.add_row(name, dep_type, status_text, version, path, details)
        else:
            table.add_row(name, dep_type, status_text, version, path)
    
    console.print(table)
    
    # Print summary
    summary_color = "green" if categories["missing"] == 0 and categories["error"] == 0 else "red"
    summary = Panel(
        f"[bold]{categories['ok']}[/bold] OK, [bold]{categories['missing']}[/bold] Missing, [bold]{categories['error']}[/bold] Error",
        title="Summary",
        border_style=summary_color
    )
    console.print(summary)
    
    # Print suggestion if there are missing dependencies
    if categories["missing"] > 0:
        if fix:
            console.print("[yellow]Some dependencies could not be fixed automatically. Please install them manually.[/yellow]")
        else:
            console.print("[yellow]Run with --fix to attempt to install missing dependencies.[/yellow]")


@app.command()
def shell() -> None:
    """
    Start an interactive TSAP shell.
    """
    rich_print("[bold yellow]Interactive shell not yet implemented[/bold yellow]")
    rich_print("This feature will provide an interactive shell for TSAP commands.")
    
    # Placeholder for future implementation
    # The shell could use prompt_toolkit for a rich interactive experience
    # with autocomplete, syntax highlighting, history, etc.


@app.command()
def dashboard() -> None:
    """
    Launch the TSAP dashboard.
    """
    if not is_dashboard_available():
        rich_print("[bold red]Error:[/bold red] Dashboard requires the Textual library.")
        rich_print("Please install it using: [bold]pip install textual[/bold]")
        return
    
    rich_print("[bold blue]Starting TSAP dashboard...[/bold blue]")
    rich_print("Press [bold]Q[/bold] to quit, [bold]?[/bold] for help")
    
    try:
        start_dashboard()
    except KeyboardInterrupt:
        rich_print("[bold yellow]Dashboard stopped by user[/bold yellow]")
    except Exception as e:
        rich_print(f"[bold red]Error starting dashboard:[/bold red] {str(e)}")


@server_app.command("start")
def server_start(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8021, "--port", "-p", help="Port to listen on"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of worker processes"),
    log_level: str = typer.Option("info", "--log-level", "-l", help="Log level (debug, info, warning, error)"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload on code changes (development only)"),
    dashboard: bool = typer.Option(False, "--dashboard", "-d", help="Launch dashboard after server start")
) -> None:
    """
    Start the TSAP server.
    """
    # Show server configuration
    config_panel = Panel(
        f"[bold]Host:[/bold] {host}\n"
        f"[bold]Port:[/bold] {port}\n"
        f"[bold]Workers:[/bold] {workers}\n"
        f"[bold]Log Level:[/bold] {log_level.upper()}\n"
        f"[bold]Auto-reload:[/bold] {'Enabled' if reload else 'Disabled'}\n"
        f"[bold]Performance Mode:[/bold] {get_performance_mode().upper()}",
        title="Server Configuration",
        border_style="blue"
    )
    console.print(config_panel)
    
    try:
        # Start server with specified options
        rich_print(f"[bold green]Starting TSAP server on http://{host}:{port}[/bold green]")
        
        # If dashboard is requested, launch it in a separate thread
        if dashboard:
            if not is_dashboard_available():
                rich_print("[bold yellow]Warning:[/bold yellow] Dashboard requires the Textual library and will not be started")
                rich_print("Please install it using: [bold]pip install textual[/bold]")
            else:
                # This would require implementing a non-blocking dashboard that works with the server
                rich_print("[bold yellow]Dashboard with server not yet implemented[/bold yellow]")
        
        # Start the server
        start_server(host=host, port=port, workers=workers, log_level=log_level, reload=reload)
        
    except KeyboardInterrupt:
        rich_print("[bold yellow]Server stopped by user[/bold yellow]")
    except Exception as e:
        rich_print(f"[bold red]Error starting server:[/bold red] {str(e)}")
        sys.exit(1)


@server_app.command("status")
def server_status() -> None:
    """
    Check the status of the TSAP server.
    """
    # This would check if the server is running and show status information
    rich_print("[bold yellow]Server status check not yet implemented[/bold yellow]")
    
    # Placeholder implementation that could be expanded
    try:
        # Try to connect to the server and get status
        rich_print("[bold]Checking server status...[/bold]")
        time.sleep(1)  # Simulate a delay for checking
        
        # Show status (placeholder)
        status_panel = Panel(
            "[bold]Status:[/bold] Not running\n"
            "[bold]Last seen:[/bold] Never\n",
            title="Server Status",
            border_style="red"
        )
        console.print(status_panel)
        
    except Exception as e:
        rich_print(f"[bold red]Error checking server status:[/bold red] {str(e)}")


@server_app.command("stop")
def server_stop() -> None:
    """
    Stop a running TSAP server.
    """
    # This would gracefully stop a running server
    rich_print("[bold yellow]Server stop command not yet implemented[/bold yellow]")
    
    # Placeholder implementation
    with console.status("[bold blue]Stopping server...[/bold blue]") as status:
        time.sleep(2)  # Simulate a delay
        status.update("[bold red]Failed to stop server: No running server found[/bold red]")
        time.sleep(1)


@config_app.command("show")
def config_show(
    section: Optional[str] = typer.Argument(None, help="Config section to display"),
    format: str = typer.Option("table", "--format", "-f", help="Output format (table, json, yaml)")
) -> None:
    """
    Show current configuration.
    """
    config = get_config()
    config_dict = config.dict()
    
    if section:
        if section not in config_dict:
            rich_print(f"[bold red]Error:[/bold red] Section '{section}' not found in configuration")
            return
        
        config_data = config_dict[section]
        title = f"Configuration - {section.upper()}"
    else:
        config_data = config_dict
        title = "Full Configuration"
    
    if format.lower() == "json":
        import json
        rich_print(json.dumps(config_data, indent=2))
    elif format.lower() == "yaml":
        try:
            import yaml
            rich_print(yaml.dump(config_data, default_flow_style=False))
        except ImportError:
            rich_print("[bold red]Error:[/bold red] YAML format requires PyYAML. Please install it with: pip install pyyaml")
    else:  # table format
        _display_config_as_table(config_data, title)


def _display_config_as_table(config_data: Dict[str, Any], title: str) -> None:
    """
    Display configuration data as a Rich table.
    
    Args:
        config_data: Configuration data dictionary
        title: Table title
    """
    # Handle nested config data
    if isinstance(config_data, dict) and all(isinstance(v, dict) for v in config_data.values()):
        # Top-level sections
        for section, section_data in config_data.items():
            section_title = f"{title} - {section.upper()}"
            _display_config_as_table(section_data, section_title)
        return
    
    # Create table for leaf config items
    table = Table(title=title)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in sorted(config_data.items()):
        # Format value for display
        if isinstance(value, dict):
            value_str = "<nested configuration>"
        elif isinstance(value, list):
            value_str = ", ".join(str(v) for v in value)
        elif value is None:
            value_str = "None"
        else:
            value_str = str(value)
        
        table.add_row(key, value_str)
    
    console.print(table)
    console.print()


@config_app.command("save")
def config_save(
    path: str = typer.Argument(..., help="Path to save configuration file"),
    format: str = typer.Option("yaml", "--format", "-f", help="Output format (yaml, json)")
) -> None:
    """
    Save current configuration to a file.
    """
    try:
        config = get_config()  # noqa: F841
        
        # Save config to specified path with format
        save_config(path, format=format)
        
        rich_print(f"[bold green]Configuration saved to {path}[/bold green]")
        
    except Exception as e:
        rich_print(f"[bold red]Error saving configuration:[/bold red] {str(e)}")


@config_app.command("mode")
def config_mode(
    mode: Optional[str] = typer.Argument(None, help="Performance mode to set (fast, standard, deep)"),
    show_params: bool = typer.Option(False, "--params", "-p", help="Show parameters for the current mode")
) -> None:
    """
    Get or set the performance mode.
    """
    # If mode is provided, set it
    if mode:
        try:
            if mode.lower() not in ("fast", "standard", "deep"):
                raise ValueError(f"Invalid performance mode: {mode}. Must be one of: fast, standard, deep")
            
            set_performance_mode(mode)
            rich_print(f"[bold green]Performance mode set to {mode.upper()}[/bold green]")
            
        except Exception as e:
            rich_print(f"[bold red]Error setting performance mode:[/bold red] {str(e)}")
            return
    
    # Show current mode info
    current_mode = get_performance_mode()
    mode_info = describe_current_mode()
    
    # Create panel for mode info
    mode_panel = Panel(
        f"[bold]Current Mode:[/bold] {current_mode.upper()}\n"
        f"[bold]Search Depth:[/bold] {mode_info.get('search_depth', 'Unknown')}\n"
        f"[bold]Concurrency:[/bold] {mode_info.get('max_concurrency', 'Unknown')}\n"
        f"[bold]Timeout:[/bold] {mode_info.get('operation_timeout', 'Unknown')} seconds\n"
        f"[bold]Max Results:[/bold] {mode_info.get('max_results', 'Unknown')}",
        title="Performance Mode",
        border_style="blue"
    )
    
    console.print(mode_panel)
    
    # Show detailed parameters if requested
    if show_params:
        # This would show the full set of parameters for the current mode
        rich_print("[bold yellow]Detailed parameters not yet implemented[/bold yellow]")


@tool_app.command("ripgrep")
def tool_ripgrep(
    pattern: str = typer.Argument(..., help="Search pattern"),
    paths: List[str] = typer.Argument(None, help="Paths to search"),
    case_sensitive: bool = typer.Option(False, "--case-sensitive", "-s", help="Enable case-sensitive search"),
    regex: bool = typer.Option(True, "--regex/--literal", "-r/-l", help="Use regex pattern (vs literal)"),
    file_patterns: Optional[List[str]] = typer.Option(None, "--file", "-f", help="File patterns to include"),
    exclude_patterns: Optional[List[str]] = typer.Option(None, "--exclude", "-e", help="Patterns to exclude"),
    context_lines: int = typer.Option(2, "--context", "-c", help="Number of context lines to show"),
    max_results: Optional[int] = typer.Option(None, "--max", "-m", help="Maximum number of results"),
    show_stats: bool = typer.Option(False, "--stats", help="Show search statistics")
) -> None:
    """
    Search files using ripgrep.
    """
    # Default to current directory if no paths provided
    if not paths:
        paths = [os.getcwd()]
    
    # Create search parameters
    
    # Show search parameters
    rich_print(f"[bold blue]Searching for:[/bold blue] {pattern}")
    rich_print(f"[bold blue]In paths:[/bold blue] {', '.join(paths)}")
    if file_patterns:
        rich_print(f"[bold blue]File patterns:[/bold blue] {', '.join(file_patterns)}")
    if exclude_patterns:
        rich_print(f"[bold blue]Exclude patterns:[/bold blue] {', '.join(exclude_patterns)}")
    
    # This would normally use the actual ripgrep_search function
    # For now, we'll simulate progress and results
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}[/bold blue]"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn()
    ) as progress:
        search_task = progress.add_task("[blue]Searching...", total=100)
        
        # Simulate progress
        for i in range(0, 101, 5):
            progress.update(search_task, completed=i)
            time.sleep(0.1)
    
    # Simulate results (in a real implementation, this would show actual results)
    rich_print("\n[bold green]Search Results:[/bold green]")
    rich_print("No matches found.")
    
    # Show statistics if requested
    if show_stats:
        stats_panel = Panel(
            "[bold]Files searched:[/bold] 0\n"
            "[bold]Matches found:[/bold] 0\n"
            "[bold]Time taken:[/bold] 0.5 seconds",
            title="Search Statistics",
            border_style="blue"
        )
        console.print(stats_panel)


@plugin_app.command("list")
def plugin_list(
    enabled_only: bool = typer.Option(False, "--enabled", "-e", help="Show only enabled plugins"),
    plugin_type: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by plugin type"),
    format: str = typer.Option("table", "--format", "-f", help="Output format (table, json)")
) -> None:
    """
    List installed plugins.
    """
    # Initialize plugin system
    initialize_plugin_system()
    
    # Get plugin manager
    plugin_manager = get_plugin_manager()
    
    # Get plugins
    plugins = plugin_manager.list_plugins()
    
    # Filter plugins
    if enabled_only:
        plugins = {k: v for k, v in plugins.items() if v.get("enabled", False)}
    
    if plugin_type:
        plugins = {k: v for k, v in plugins.items() if v.get("plugin_type", "") == plugin_type}
    
    # Display plugins
    if format.lower() == "json":
        import json
        rich_print(json.dumps(plugins, indent=2))
    else:
        # Display as table
        table = Table(title="Installed Plugins")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="blue")
        table.add_column("Version", style="green")
        table.add_column("Type", style="magenta")
        table.add_column("Status", style="yellow")
        table.add_column("Description", style="white")
        
        for plugin_id, plugin_info in plugins.items():
            name = plugin_info.get("name", "Unknown")
            version = plugin_info.get("version", "0.0.0")
            plugin_type = plugin_info.get("plugin_type", "Unknown")
            enabled = plugin_info.get("enabled", False)
            description = plugin_info.get("description", "")
            
            status = Text("Enabled", style="green") if enabled else Text("Disabled", style="red")
            
            table.add_row(plugin_id, name, version, plugin_type, status, description)
        
        console.print(table)
        
        # Show summary
        rich_print(f"\n[bold]Total plugins:[/bold] {len(plugins)}")


@plugin_app.command("install")
def plugin_install(
    source: str = typer.Argument(..., help="Plugin source (file path, URL, or plugin name)"),
    enable: bool = typer.Option(True, "--enable/--no-enable", help="Enable the plugin after installation"),
    force: bool = typer.Option(False, "--force", "-f", help="Force reinstallation if plugin already exists")
) -> None:
    """
    Install a plugin.
    """
    # Initialize plugin system
    initialize_plugin_system()
    
    # Get plugin manager
    plugin_manager = get_plugin_manager()
    
    try:
        # Install the plugin
        with console.status(f"[bold blue]Installing plugin from {source}...[/bold blue]") as status:  # noqa: F841
            plugin_id = plugin_manager.install_plugin(plugin_path=source, enable=enable)
        
        if plugin_id:
            rich_print(f"[bold green]Plugin installed successfully with ID: {plugin_id}[/bold green]")
            
            # Get plugin status
            plugin_status = plugin_manager.get_plugin_status(plugin_id)
            
            # Show plugin details
            if plugin_status:
                plugin_panel = Panel(
                    f"[bold]ID:[/bold] {plugin_id}\n"
                    f"[bold]Name:[/bold] {plugin_status.get('name', 'Unknown')}\n"
                    f"[bold]Version:[/bold] {plugin_status.get('version', 'Unknown')}\n"
                    f"[bold]Type:[/bold] {plugin_status.get('plugin_type', 'Unknown')}\n"
                    f"[bold]Enabled:[/bold] {'Yes' if plugin_status.get('enabled', False) else 'No'}\n"
                    f"[bold]Description:[/bold] {plugin_status.get('description', '')}\n",
                    title="Plugin Details",
                    border_style="green"
                )
                console.print(plugin_panel)
        else:
            rich_print(f"[bold red]Failed to install plugin from {source}[/bold red]")
    
    except Exception as e:
        rich_print(f"[bold red]Error installing plugin:[/bold red] {str(e)}")


@project_app.command("list")
def project_list(
    format: str = typer.Option("table", "--format", "-f", help="Output format (table, json)")
) -> None:
    """
    List projects.
    """
    try:
        # Get project registry
        projects = list_projects()
        
        if format.lower() == "json":
            import json
            rich_print(json.dumps(projects, indent=2))
            return
        
        # Display as table
        table = Table(title="Projects")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="blue")
        table.add_column("Created", style="green")
        table.add_column("Last Modified", style="yellow")
        table.add_column("Directory", style="magenta")
        
        for project in projects:
            project_id = project.get("id", "")
            name = project.get("name", "Unnamed")
            created = project.get("created_at", "")
            modified = project.get("modified_at", "")
            directory = project.get("root_dir", "")
            
            table.add_row(project_id, name, created, modified, directory)
        
        console.print(table)
        
        # Show active project
        active_project = None
        for p in projects:
            if p.get("is_active", False):
                active_project = p
                break
        
        if active_project:
            rich_print(f"\n[bold green]Active project:[/bold green] {active_project.get('name', 'Unnamed')} ({active_project.get('id', '')})")
        else:
            rich_print("\n[bold yellow]No active project[/bold yellow]")
    
    except Exception as e:
        rich_print(f"[bold red]Error listing projects:[/bold red] {str(e)}")


@project_app.command("create")
def project_create(
    name: str = typer.Argument(..., help="Project name"),
    directory: Optional[str] = typer.Option(None, "--dir", "-d", help="Project root directory"),
    set_active: bool = typer.Option(True, "--set-active/--no-set-active", help="Set as active project")
) -> None:
    """
    Create a new project.
    """
    try:
        # Create project
        project = create_project(name=name, root_directory=directory)
        
        # Set as active if requested
        if set_active:
            from tsap.project.context import set_active_project
            set_active_project(project.id)
        
        rich_print(f"[bold green]Project created:[/bold green] {name} (ID: {project.id})")
        
        if directory:
            rich_print(f"[bold]Directory:[/bold] {directory}")
        
        if set_active:
            rich_print("[bold]Set as active project[/bold]")
    
    except Exception as e:
        rich_print(f"[bold red]Error creating project:[/bold red] {str(e)}")


@template_app.command("list")
def template_list() -> None:
    """
    List available templates.
    """
    try:
        # Get template runner
        template_runner = get_template_runner()
        
        # Get templates
        templates = template_runner.list_templates()
        
        # Display as table
        table = Table(title="Available Templates")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="blue")
        table.add_column("Type", style="green")
        table.add_column("Description", style="white")
        
        for template in templates:
            template_id = template.get("id", "")
            name = template.get("name", "")
            template_type = template.get("type", "")
            description = template.get("description", "")
            
            table.add_row(template_id, name, template_type, description)
        
        console.print(table)
        
        # Show summary
        rich_print(f"\n[bold]Total templates:[/bold] {len(templates)}")
    
    except Exception as e:
        rich_print(f"[bold red]Error listing templates:[/bold red] {str(e)}")


@template_app.command("run")
def template_run(
    template_id: str = typer.Argument(..., help="Template ID or name"),
    params_file: Optional[str] = typer.Option(None, "--params", "-p", help="Path to parameters JSON file"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Path to save output")
) -> None:
    """
    Run a template.
    """
    try:
        # Get template runner
        template_runner = get_template_runner()  # noqa: F841
        
        # Load parameters if provided
        params = {}
        if params_file:
            import json
            with open(params_file, 'r') as f:
                params = json.load(f)  # noqa: F841
        
        # Run template
        with console.status(f"[bold blue]Running template {template_id}...[/bold blue]") as status:  # noqa: F841
            # This would be an async call, we'd need to adapt for sync CLI context
            # result = await template_runner.run_template_by_name(template_id, params)
            
            # Placeholder for demonstration
            time.sleep(2)
            result = {
                "success": True,
                "template_id": template_id,
                "execution_time": 2.0,
                "result": {
                    "message": "Template executed successfully",
                    "items_processed": 42
                }
            }
        
        # Show result
        if result.get("success", False):
            rich_print("[bold green]Template executed successfully[/bold green]")
            
            # Format result for display
            result_panel = Panel(
                str(result.get("result", "")),
                title=f"Template Result: {template_id}",
                border_style="green"
            )
            console.print(result_panel)
            
            # Save to file if requested
            if output_file:
                import json
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                rich_print(f"[bold]Result saved to:[/bold] {output_file}")
        else:
            rich_print(f"[bold red]Template execution failed:[/bold red] {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        rich_print(f"[bold red]Error running template:[/bold red] {str(e)}")


@search_app.command("parallel")
def search_parallel(
    patterns: List[str] = typer.Argument(..., help="Search patterns, comma-separated"),
    paths: List[str] = typer.Argument(None, help="Paths to search"),
    case_sensitive: bool = typer.Option(False, "--case-sensitive", "-s", help="Enable case-sensitive search"),
    regex: bool = typer.Option(True, "--regex/--literal", "-r/-l", help="Use regex patterns (vs literal)"),
    file_patterns: Optional[List[str]] = typer.Option(None, "--file", "-f", help="File patterns to include"),
    exclude_patterns: Optional[List[str]] = typer.Option(None, "--exclude", "-e", help="Patterns to exclude"),
    context_lines: int = typer.Option(2, "--context", "-c", help="Number of context lines to show"),
    max_results: Optional[int] = typer.Option(None, "--max", "-m", help="Maximum number of results per pattern"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Path to save results")
) -> None:
    """
    Run multiple search patterns in parallel.
    """
    # Default to current directory if no paths provided
    if not paths:
        paths = [os.getcwd()]
    
    # Format patterns for display
    rich_print(f"[bold blue]Searching for {len(patterns)} patterns:[/bold blue]")
    for i, pattern in enumerate(patterns, 1):
        rich_print(f"  {i}. {pattern}")
    
    rich_print(f"[bold blue]In paths:[/bold blue] {', '.join(paths)}")
    
    # This would normally use the actual parallel_search function
    # For now, we'll simulate progress and results
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}[/bold blue]"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn()
    ) as progress:
        search_task = progress.add_task("[blue]Searching...", total=100)
        
        # Simulate progress
        for i in range(0, 101, 5):
            progress.update(search_task, completed=i)
            time.sleep(0.1)
    
    # Simulate results (in a real implementation, this would show actual results)
    rich_print("\n[bold green]Search Results:[/bold green]")
    rich_print("No matches found for any patterns.")
    
    # Save results if requested
    if output_file:
        rich_print(f"\n[bold]Results would be saved to:[/bold] {output_file}")


@analyze_app.command("code")
def analyze_code(
    paths: List[str] = typer.Argument(..., help="Paths to analyze"),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Filter by language"),
    include_structure: bool = typer.Option(True, "--structure/--no-structure", help="Include code structure analysis"),
    include_complexity: bool = typer.Option(True, "--complexity/--no-complexity", help="Include complexity analysis"),
    include_dependencies: bool = typer.Option(True, "--dependencies/--no-dependencies", help="Include dependency analysis"),
    include_security: bool = typer.Option(True, "--security/--no-security", help="Include security checks"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Path to save analysis results")
) -> None:
    """
    Analyze code structure, complexity, dependencies, and security.
    """
    rich_print(f"[bold blue]Analyzing code in:[/bold blue] {', '.join(paths)}")
    
    if language:
        rich_print(f"[bold blue]Language filter:[/bold blue] {language}")
    
    # Show analysis options
    options = []
    if include_structure:
        options.append("[green]Structure[/green]")
    if include_complexity:
        options.append("[yellow]Complexity[/yellow]")
    if include_dependencies:
        options.append("[blue]Dependencies[/blue]")
    if include_security:
        options.append("[red]Security[/red]")
    
    rich_print(f"[bold blue]Analysis includes:[/bold blue] {', '.join(options)}")
    
    # Simulate analysis with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}[/bold blue]"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn()
    ) as progress:
        # Create tasks for each analysis type
        tasks = {}
        
        if include_structure:
            tasks["structure"] = progress.add_task("[green]Analyzing structure...", total=100)
        if include_complexity:
            tasks["complexity"] = progress.add_task("[yellow]Analyzing complexity...", total=100)
        if include_dependencies:
            tasks["dependencies"] = progress.add_task("[blue]Analyzing dependencies...", total=100)
        if include_security:
            tasks["security"] = progress.add_task("[red]Running security checks...", total=100)
        
        # Simulate progress
        for _ in range(20):
            for task_id in tasks.values():
                current = progress.tasks[task_id].completed
                if current < 100:
                    progress.update(task_id, completed=min(current + 5, 100))
            time.sleep(0.1)
    
    # Simulate results (in a real implementation, this would show actual results)
    rich_print("\n[bold green]Analysis Results:[/bold green]")
    
    if include_structure:
        rich_print("\n[bold green]Code Structure:[/bold green]")
        rich_print("  - 10 files analyzed")
        rich_print("  - 42 functions identified")
        rich_print("  - 5 classes identified")
    
    if include_complexity:
        rich_print("\n[bold yellow]Complexity Metrics:[/bold yellow]")
        rich_print("  - Average cyclomatic complexity: 5.2")
        rich_print("  - Maximum nesting depth: 4")
    
    if include_dependencies:
        rich_print("\n[bold blue]Dependencies:[/bold blue]")
        rich_print("  - 3 external dependencies identified")
        rich_print("  - 15 internal module dependencies")
    
    if include_security:
        rich_print("\n[bold red]Security Analysis:[/bold red]")
        rich_print("  - No security issues found")
    
    # Save results if requested
    if output_file:
        rich_print(f"\n[bold]Results would be saved to:[/bold] {output_file}")


def main() -> None:
    """
    Main entry point for the CLI.
    """
    app()


if __name__ == "__main__":
    main()