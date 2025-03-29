#!/usr/bin/env python3
"""
Plugin Development Example

This script demonstrates how to create custom plugins for TSAP.
It shows a simplified version of plugin development without using
the full plugin manager infrastructure.
"""
import os
import sys
import asyncio
from typing import Dict, Any, List, Optional, Callable

from rich import print as rich_print
from rich.panel import Panel
from rich.syntax import Syntax
from rich.console import Console
from rich.rule import Rule
from rich.table import Table

# Ensure the src directory is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    # Import necessary components
    from mcp_client_example import MCPClient  # Use MCPClient
except ImportError as e:
    rich_print(f"[bold red]Error: Failed to import required modules: {e}[/bold red]")
    rich_print("[yellow]Make sure you're running from the TSAP project root, the virtual environment is activated, and mcp_client_example.py exists.[/yellow]")
    sys.exit(1)

console = Console()

# --- Plugin Registration System (Simplified) ---
class PluginRegistry:
    """Simple plugin registry for demonstration purposes."""
    
    def __init__(self):
        """Initialize the registry."""
        self.plugins = {}
        self.analysis_functions = {}
    
    def register_plugin(self, plugin_id: str, plugin: Any) -> None:
        """Register a plugin."""
        self.plugins[plugin_id] = plugin
        rich_print(f"[green]Registered plugin: {plugin_id}[/green]")
    
    def register_analysis_function(self, name: str, function: Callable) -> None:
        """Register an analysis function."""
        self.analysis_functions[name] = function
        rich_print(f"[green]Registered analysis function: {name}[/green]")
    
    def get_plugin(self, plugin_id: str) -> Optional[Any]:
        """Get a plugin by ID."""
        return self.plugins.get(plugin_id)
    
    def get_analysis_function(self, name: str) -> Optional[Callable]:
        """Get an analysis function by name."""
        return self.analysis_functions.get(name)
    
    def list_plugins(self) -> Dict[str, Any]:
        """List all registered plugins."""
        return self.plugins
    
    def list_analysis_functions(self) -> Dict[str, Callable]:
        """List all registered analysis functions."""
        return self.analysis_functions

# Create a global registry
registry = PluginRegistry()

# --- Custom Plugin Definition ---
class CodeLinesCounterPlugin:
    """Custom plugin for counting lines of code by language."""
    
    def __init__(self, name: str, version: str, description: str, author: str):
        """Initialize the plugin with metadata."""
        self.id = "code_lines_counter"
        self.name = name
        self.version = version
        self.description = description
        self.author = author
        self.active = False
    
    def initialize(self) -> None:
        """Initialize the plugin."""
        self.active = True
        rich_print(f"[green]Initializing {self.name} v{self.version}...[/green]")
        
        # Register analysis function (Note: analyze_code no longer needs client)
        registry.register_analysis_function("count_lines", self.analyze_code)
    
    def shutdown(self) -> None:
        """Shutdown the plugin."""
        self.active = False
        rich_print(f"[yellow]Shutting down {self.name}...[/yellow]")
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get plugin metadata."""
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "active": self.active
        }
    
    # Removed MCPClient from signature as it's no longer used for core counting
    async def analyze_code(self, file_paths: List[str], **kwargs) -> Dict[str, Any]:
        """Analyze code files and count lines by language locally.
        
        Args:
            file_paths: List of file paths to analyze
            **kwargs: Additional parameters (currently unused)
            
        Returns:
            Analysis results
        """
        if not self.active:
            raise RuntimeError("Plugin is not active. Call initialize() first.")
            
        rich_print(f"[bold cyan]Analyzing {len(file_paths)} files locally with {self.name}...[/bold cyan]")
        
        # Define language patterns (mapping extension to language name)
        # Simplified to use endswith for clarity and performance
        language_extensions = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".html": "html",
            ".css": "css",
            ".md": "markdown",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml", # Added .yml variant
        }
        # Create reverse mapping for stats dictionary keys
        language_names = set(language_extensions.values()) | {"other"}
        
        # Initialize counters
        total_files = 0 
        total_lines = 0
        language_stats = {lang: {"files": 0, "lines": 0} for lang in language_names}
        
        # Process files locally
        rich_print("[dim]Counting lines locally for all files...[/dim]")
        for file_path in file_paths:
            total_files += 1 # Count every file encountered
            file_lang = "other" # Default language
            
            # Determine language from extension
            # Check specific extensions first
            _, ext = os.path.splitext(file_path)
            ext = ext.lower() # Case-insensitive extension matching
            if ext in language_extensions:
                 file_lang = language_extensions[ext]
            # Could add more sophisticated checks here if needed (e.g., shebang)
            
            # Count lines locally for the file
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    line_count = sum(1 for _ in f) # Simple line count
                
                language_stats[file_lang]["files"] += 1
                language_stats[file_lang]["lines"] += line_count
                total_lines += line_count 
            except Exception as e:
                 rich_print(f"[dim red]Could not count lines locally for {os.path.basename(file_path)}: {e}[/dim]")
                 # Still count the file, but with 0 lines for stats
                 language_stats[file_lang]["files"] += 1

        rich_print(f"[dim]Finished counting lines locally for {total_files} files.[/dim]")
        
        # Calculate percentages
        if total_lines > 0:
            for lang_data in language_stats.values():
                 if lang_data["lines"] > 0:
                     lang_data["percentage"] = round(
                         (lang_data["lines"] / total_lines) * 100, 1
                     )
                 else:
                     lang_data["percentage"] = 0.0
        
        # Return results
        return {
            "total_files": total_files, # Now reflects all files attempted
            "total_lines": total_lines,
            "language_stats": language_stats,
            "summary": f"Analyzed {total_files} files with {total_lines} lines of code (local analysis)"
        }

# --- Demo Functions ---
def create_and_register_plugin():
    """Create and register a custom plugin."""
    rich_print(Rule("[bold cyan]Creating and Registering Plugin[/bold cyan]"))
    
    # Create the plugin
    plugin = CodeLinesCounterPlugin(
        name="Code Lines Counter",
        version="1.0.0",
        description="Counts lines of code by language and provides statistics",
        author="TSAP Example"
    )
    
    # Display the plugin's source code
    rich_print("\n[bold yellow]Plugin Definition:[/bold yellow]")
    import inspect
    source = inspect.getsource(CodeLinesCounterPlugin)
    console.print(Syntax(source, "python", line_numbers=True))
    
    # Register and initialize the plugin
    registry.register_plugin(plugin.id, plugin)
    plugin.initialize()
    
    # Display plugin metadata
    metadata = plugin.get_metadata()
    metadata_panel = Panel(
        "\n".join(f"[bold]{k}:[/bold] {v}" for k, v in metadata.items()),
        title="Plugin Metadata",
        expand=False
    )
    console.print(metadata_panel)
    
    return plugin

# Updated signature to REMOVE MCPClient as analyze_code no longer needs it
async def run_plugin_analysis(client: MCPClient, directory: str):
    """Run analysis using the plugin (fetches file list via client)."""
    rich_print(Rule("[bold yellow]Running Plugin Analysis (via Server File List)[/bold yellow]"))
    
    # Verify directory exists - Removed, client handles this implicitly
    # if not os.path.exists(directory):
    #     rich_print(f"[bold red]Error: Directory '{directory}' not found[/bold red]")
    #     return
    
    # Get list of files using MCPClient and ripgrep
    file_paths = []
    try:
        rich_print(f"[dim]Querying server for files in '{directory}'...[/dim]")
        # Use ripgrep to list files by matching the start of any line
        response = await client.ripgrep_search(
            pattern='^', 
            paths=[directory], 
            max_total_matches=10000, # Allow many files
            stats=False # Don't need stats, just paths
        )

        if response.get("status") != "success" or "data" not in response or "matches" not in response["data"]:
            error_msg = response.get("error", {}).get("message", "Unknown error")
            rich_print(f"[bold red]Error fetching file list from server: {error_msg}[/bold red]")
            return

        # Extract unique file paths from matches
        file_paths_set = set()
        for match in response["data"]["matches"]:
            if "path" in match:
                # Make path absolute relative to the current working directory 
                # Assumes server returns paths relative to workspace root, 
                # and script runs from workspace root.
                abs_path = os.path.abspath(match["path"]) 
                file_paths_set.add(abs_path)
        file_paths = list(file_paths_set)
       
        if not file_paths:
            rich_print(f"[yellow]No files found in '{directory}' via server query.[/yellow]")
            # Don't return here, let the analysis run with an empty list to show 0 counts
            # return 
           
    except Exception as e:
        rich_print(f"[bold red]Error communicating with server: {e}[/bold red]")
        import traceback
        rich_print(traceback.format_exc())
        return

    # Get list of files in directory (recursively) - Old code removed
    # file_paths = []
    # for root, _, files in os.walk(directory):
    #     for file in files:
    #         # Make path absolute for consistency
    #         file_paths.append(os.path.abspath(os.path.join(root, file)))
    
    rich_print(f"[bold]Server reported {len(file_paths)} files in {directory}[/bold]")
    
    # Get the analysis function from the registry
    analysis_fn = registry.get_analysis_function("count_lines")
    if not analysis_fn:
        rich_print("[bold red]Error: Analysis function 'count_lines' not found[/bold red]")
        return
    
    # Run the analysis (no client needed anymore)
    try:
        results = await analysis_fn(file_paths)
        
        # Display results
        rich_print(Panel(f"[bold blue]Code Analysis Results: {results['summary']}[/bold blue]", expand=False))
        
        # Create a table of language statistics
        table = Table(title="Code Distribution by Language")
        table.add_column("Language", style="cyan")
        table.add_column("Files", style="yellow", justify="right")
        table.add_column("Lines", style="green", justify="right")
        table.add_column("Percentage", style="magenta", justify="right")
        
        # Sort languages by number of lines (descending)
        languages = sorted(
            results["language_stats"].keys(),
            key=lambda x: results["language_stats"][x]["lines"],
            reverse=True
        )
        
        for lang in languages:
            stats = results["language_stats"][lang]
            # Skip languages with 0 files/lines
            if stats["files"] == 0 and stats["lines"] == 0:
                continue
                
            table.add_row(
                lang.capitalize(),
                str(stats["files"]),
                str(stats["lines"]),
                f"{stats.get('percentage', 0.0)}%"
            )
        
        console.print(table)
        
        # Visual representation of the distribution
        rich_print("\n[bold]Language Distribution:[/bold]")
        for lang in languages:
            stats = results["language_stats"][lang]
            if stats["files"] == 0 and stats["lines"] == 0:
                continue
                
            percentage = stats.get("percentage", 0)
            bar_length = int(percentage / 2)  # Scale to fit console width
            
            # Use different colors for different languages
            color = "green" # Default for other/unknown
            if lang == "python":
                color = "blue"
            elif lang == "javascript":
                color = "yellow"
            elif lang == "typescript":
                color = "cyan"
            elif lang == "html":
                color = "red"
            elif lang == "css":
                color = "magenta"
            elif lang == "markdown":
                color = "bright_green"
            elif lang == "json":
                color = "bright_blue"
            elif lang == "yaml":
                color = "bright_cyan"
            
            rich_print(
                f"{lang.capitalize():12} " # Wider padding for language names
                f"[white]{percentage:5.1f}%[/white] "
                f"[bold {color}]{'█' * bar_length}[/bold {color}]"
            )
            
    except Exception as e:
        rich_print(f"[bold red]Error running plugin analysis: {str(e)}[/bold red]")
        import traceback
        rich_print(traceback.format_exc())

def show_how_plugins_work():
    """Explain how plugins work in TSAP."""
    rich_print(Rule("[bold magenta]How TSAP Plugins Work[/bold magenta]"))
    
    explanation = """
[bold]1. Plugin Architecture[/bold]
TSAP's plugin system follows a modular architecture that allows extending core functionality:
- Plugins implement standard interfaces based on their type
- The plugin registry maintains references to all plugins and their components
- The plugin manager handles loading, enabling, and configuring plugins

[bold]2. Plugin Types[/bold]
TSAP supports various plugin types for different extension points:
- [cyan]Core Tool Plugins[/cyan]: Extend Layer 1 tools (search, extraction, etc.)
- [cyan]Composite Plugins[/cyan]: Extend Layer 2 composite operations
- [cyan]Analysis Plugins[/cyan]: Extend Layer 3 analysis capabilities
- [cyan]Format Plugins[/cyan]: Add support for additional file formats
- [cyan]Integration Plugins[/cyan]: Connect with external systems
- [cyan]UI Plugins[/cyan]: Extend the user interface

[bold]3. Plugin Lifecycle[/bold]
Plugins go through several phases during their lifecycle:
- Registration: Plugin is registered with the system
- Initialization: Plugin resources are initialized
- Operation: Plugin functionality is available to the system
- Shutdown: Plugin resources are cleaned up

[bold]4. Creating Plugins[/bold]
To create a TSAP plugin:
1. Implement the appropriate plugin interface
2. Define metadata using decorators
3. Implement required methods for initialization and registration
4. Register plugin components with the system
5. Package the plugin for distribution

[bold]5. Plugin Discovery[/bold]
TSAP discovers plugins in several ways:
- Built-in plugins included with TSAP
- Installed plugins in the plugins directory
- Dynamically loaded plugins at runtime
- Manually registered plugins

This example demonstrates a simplified version of the TSAP plugin system,
focusing on the core concepts and plugin development workflow.
"""
    
    rich_print(explanation)

async def main():
    """Main function."""
    console.print(Panel("[bold blue]TSAP Plugin Development Example[/bold blue]", expand=False))
    
    # Show how plugins work in TSAP
    show_how_plugins_work()
    
    # Parse arguments
    target_dir = "tsap_example_data"
    if len(sys.argv) > 1 and sys.argv[1] != "--debug":
        target_dir = sys.argv[1]
    elif len(sys.argv) == 1:
         rich_print(f"[yellow]No directory specified, defaulting to '{target_dir}'[/yellow]")
    
    # Create and register the plugin
    plugin = create_and_register_plugin()
    
    try:
        # Create MCPClient and run analysis
        async with MCPClient() as client:
             # Optional: Check client connection
             try:
                 info = await client.info()
                 if info.get("status") != "success":
                    rich_print(f"[bold red]Failed to connect to MCP Server: {info.get('error', 'Unknown error')}[/bold red]")
                    return # Exit if connection failed
                 rich_print("[green]Connected to MCP Server.[/green]")
             except Exception as conn_err:
                 rich_print(f"[bold red]Error connecting to MCP Server: {conn_err}[/bold red]")
                 return # Exit if connection failed

             # Use the plugin to analyze code, passing the client
             await run_plugin_analysis(client, target_dir) 
             
    except Exception as e:
        rich_print(f"[bold red]Error during client operation or analysis: {e}[/bold red]")
        import traceback
        rich_print(traceback.format_exc())
    finally:
        # Shutdown the plugin
        if plugin:
            plugin.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
