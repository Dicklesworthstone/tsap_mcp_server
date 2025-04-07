#!/usr/bin/env python3
"""
Plugin Development Example (MCP Tools Version)

This script demonstrates how to create custom plugins for TSAP MCP.
It shows a simplified version of plugin development without using
the full plugin manager infrastructure, using the MCP tools interface.
"""
import os
import sys
import asyncio
import json
from typing import Dict, Any, List, Optional, Callable

from rich import print as rich_print
from rich.panel import Panel
from rich.syntax import Syntax
from rich.console import Console
from rich.rule import Rule
from rich.table import Table

# Import proper MCP client modules
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Ensure the src directory is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
        
        # Register analysis function
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

async def run_plugin_analysis(session, directory: str):
    """Run analysis using the plugin and MCP tools to get file list."""
    rich_print(Rule("[bold yellow]Running Plugin Analysis (via MCP Tools File List)[/bold yellow]"))
    
    # Get list of files using MCP tools search
    file_paths = []
    try:
        rich_print(f"[dim]Querying server for files in '{directory}'...[/dim]")
        
        # Get the list of available tools
        tools_result = await session.list_tools()
        tools = tools_result.tools
        
        # Find the search tool
        search_tool = next((t for t in tools if t.name == "search"), None)
        if not search_tool:
            rich_print("[bold red]Error: search tool not found![/bold red]")
            return
        
        # Use search tool to list files by matching anything
        result = await session.call_tool(
            search_tool.name,
            arguments={
                "query": ".",  # Match any character
                "paths": [directory],
                "max_matches": 10000  # Allow many files
            }
        )
        
        # Extract the text content
        result_text = None
        for content in result.content:
            if content.type == "text":
                result_text = content.text
                break
        
        if result_text:
            try:
                # Parse the JSON response
                response = json.loads(result_text)
                
                # Extract unique file paths from matches
                file_paths_set = set()
                for match in response.get("matches", []):
                    if "file" in match:
                        # Make path absolute relative to the current working directory
                        abs_path = os.path.abspath(match["file"])
                        file_paths_set.add(abs_path)
                file_paths = list(file_paths_set)
                
                if not file_paths:
                    rich_print(f"[yellow]No files found in '{directory}' via server query.[/yellow]")
            except json.JSONDecodeError:
                rich_print("[bold red]Failed to parse search response as JSON[/bold red]")
                return
        else:
            rich_print("[bold red]No text content in search response[/bold red]")
            return
        
    except Exception as e:
        rich_print(f"[bold red]Error communicating with server: {e}[/bold red]")
        import traceback
        rich_print(traceback.format_exc())
        return

    rich_print(f"[bold]Server reported {len(file_paths)} files in {directory}[/bold]")
    
    # Get the analysis function from the registry
    analysis_fn = registry.get_analysis_function("count_lines")
    if not analysis_fn:
        rich_print("[bold red]Error: Analysis function 'count_lines' not found[/bold red]")
        return
    
    # Run the analysis
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
    """Explain how plugins work in TSAP MCP."""
    rich_print(Rule("[bold magenta]How TSAP MCP Plugins Work[/bold magenta]"))
    
    explanation = """
[bold]1. Plugin Architecture[/bold]
TSAP MCP's plugin system follows a modular architecture that allows extending core functionality:
- Plugins implement standard interfaces based on their type
- The plugin registry maintains references to all plugins and their components
- The plugin manager handles loading, enabling, and configuring plugins

[bold]2. Plugin Types[/bold]
TSAP MCP supports various plugin types for different extension points:
- [cyan]Tool Plugins[/cyan]: Extend the MCP tools interface with new capabilities
- [cyan]Composite Plugins[/cyan]: Combine multiple tools for complex operations
- [cyan]Analysis Plugins[/cyan]: Add specialized analysis functionality
- [cyan]Format Plugins[/cyan]: Add support for additional file formats
- [cyan]Integration Plugins[/cyan]: Connect with external systems
- [cyan]UI Plugins[/cyan]: Extend the user interface

[bold]3. Plugin Lifecycle[/bold]
Plugins go through several phases during their lifecycle:
- Registration: Plugin is registered with the system
- Initialization: Plugin resources are initialized and context is provided
- Operation: Plugin functionality is available through the MCP tools interface
- Shutdown: Plugin resources are cleaned up

[bold]4. Creating Plugins[/bold]
To create a TSAP MCP plugin:
1. Implement the appropriate plugin interface
2. Define metadata using decorators
3. Implement required methods for initialization and tool registration
4. Register plugin components with the system
5. Package the plugin for distribution

[bold]5. Using Plugins with MCP Tools[/bold]
Once registered, plugins are accessible through the MCP tools interface:
- Tools provided by plugins appear in the client.tools namespace
- Plugin functions can be called with appropriate context
- Results from plugin operations follow the standard MCP response format
"""
    
    rich_print(Panel(explanation, title="Plugin System Overview", expand=False))

async def main():
    """Main function to run the plugin development example."""
    rich_print(Panel(
        "[bold blue]TSAP MCP Plugin Development Example[/bold blue]",
        subtitle="Demonstrating how to create and use custom plugins"
    ))
    
    # Show explanation of how plugins work
    show_how_plugins_work()
    
    # Create and register plugin
    plugin = create_and_register_plugin()
    
    # Path to the proxy script
    proxy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_proxy.py")
    rich_print(f"Using proxy script: {proxy_path}")
    
    # Configure server parameters for stdio connection to our proxy
    server_params = StdioServerParameters(
        command="python",
        args=[proxy_path],
        env={"MCP_DEBUG": "0"}  # Disable debug logging by default
    )
    
    try:
        # Connect to the MCP server via proxy
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()
                
                # Find the info tool for initial check
                tools_result = await session.list_tools()
                tools = tools_result.tools
                info_tool = next((t for t in tools if t.name == "info"), None)
                
                # Check server health
                if info_tool:
                    rich_print("Checking server info...")
                    try:
                        info_result = await session.call_tool(info_tool.name, arguments={})
                        info_text = None
                        for content in info_result.content:
                            if content.type == "text":
                                info_text = content.text
                                break
                        
                        if info_text:
                            try:
                                json.loads(info_text)
                                rich_print("[green]Initial info check successful.[/green]")
                            except json.JSONDecodeError:
                                rich_print("[yellow]Info response is not JSON[/yellow]")
                        else:
                            rich_print("[yellow]Info call succeeded but returned no readable content.[/yellow]")
                    except Exception as e:
                        rich_print(f"[bold red]Error during info check:[/bold red] {str(e)}")
                        return
                
                # Run plugin analysis on examples directory
                await run_plugin_analysis(session, "examples")
    
    except Exception as e:
        rich_print(f"[bold red]Error connecting to MCP server: {str(e)}[/bold red]")
        import traceback
        rich_print(traceback.format_exc())
    
    # Shutdown plugin
    plugin.shutdown()
    rich_print("[bold green]Plugin example completed successfully![/bold green]")

if __name__ == "__main__":
    asyncio.run(main()) 