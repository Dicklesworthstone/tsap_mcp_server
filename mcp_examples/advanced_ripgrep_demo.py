#!/usr/bin/env python3
"""
Advanced Ripgrep Demo (MCP Version)

This script demonstrates the comprehensive features of the ripgrep integration
in TSAP MCP by searching through the example documents in various ways.
"""
import asyncio
import os
import sys
import json
from datetime import datetime
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.console import Console
from rich.rule import Rule

# Import the proper MCP client modules
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

console = Console()

# Add some basic debugging
DEBUG = False

def debug_print(msg):
    """Print debug message if DEBUG is True."""
    if DEBUG:
        console.print(f"[dim][DEBUG] {msg}[/dim]")

async def ripgrep_demo():
    """Demonstrate ripgrep's advanced features with example documents using MCP tools."""
    console.print(Panel.fit(
        "[bold blue]TSAP MCP Ripgrep Advanced Features Demo[/bold blue]",
        subtitle="Searching through tsap_example_data with MCP tools"
    ))
    
    # Path to the proxy script
    proxy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_proxy.py")
    debug_print(f"Proxy path: {proxy_path}")
    
    # Configure server parameters for stdio connection to our proxy
    server_params = StdioServerParameters(
        command="python",
        args=[proxy_path],
        env={"MCP_DEBUG": "1"}  # Enable debug logging
    )
    
    try:
        # Connect to the MCP server via proxy
        async with stdio_client(server_params) as (read, write):
            debug_print("Connected to MCP proxy")
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()
                debug_print("Session initialized successfully")
                
                # Get the list of available tools
                tools_result = await session.list_tools()
                tools = tools_result.tools
                debug_print(f"Found {len(tools)} tools")
                
                # Find the ripgrep tool
                ripgrep_tool = next((t for t in tools if t.name == "ripgrep_search"), None)
                if not ripgrep_tool:
                    console.print("[bold red]Error: ripgrep_search tool not found![/bold red]")
                    return
                
                # Find the info tool for initial check
                info_tool = next((t for t in tools if t.name == "info"), None)
                
                # --- Initial info check ---
                if info_tool:
                    console.print("Attempting to get server info...")
                    try:
                        info_result = await session.call_tool(info_tool.name, arguments={})
                        info_text = None
                        for content in info_result.content:
                            if content.type == "text":
                                info_text = content.text
                                break
                        
                        if info_text:
                            console.print("[green]Initial server info check successful.[/green]")
                            info_data = json.loads(info_text)
                            console.print(f"Server version: {info_data.get('version', 'unknown')}")
                        else:
                            console.print("[yellow]Info call succeeded but returned no readable content.[/yellow]")
                    except Exception as e:
                        console.print(f"[bold red]Error during info check:[/bold red] {str(e)}")
                        debug_print(f"Info check error details: {e}")
                # --------------------------
                
                # DEMO 1: Basic search with context
                console.print(Rule("[bold yellow]Basic Search with Context & Encoding[/bold yellow]"))
                console.print("[italic]Demonstrates search with context lines and specific file encoding[/italic]\n")
                
                await run_demo(
                    session,
                    ripgrep_tool,
                    None,  # Title handled above
                    None,  # Description handled above
                    pattern="MCP",
                    paths=["tsap_example_data/documents/"],
                    file_patterns=["*.md", "*.txt"],
                    context_lines=2,
                    encoding="utf-8",  # Specify file encoding
                    show_title=False
                )
                
                # Let's limit our demos for debugging
                if DEBUG:
                    debug_print("DEBUG mode: Only running first demo")
                    return
                
                # DEMO 2: Case sensitivity
                console.print(Rule("[bold yellow]Case Sensitive vs Insensitive Search[/bold yellow]"))
                console.print("[italic]Demonstrates the difference between case sensitive and insensitive searches[/italic]\n")
                
                # First do case-sensitive search
                console.print("[bold cyan]Case Sensitive Search (pattern='API')[/bold cyan]")
                await run_demo(
                    session,
                    ripgrep_tool,
                    None,
                    None,
                    pattern="API",
                    paths=["tsap_example_data/documents/"],
                    file_patterns=["*.md"],
                    case_sensitive=True,
                    max_count=5,
                    show_title=False
                )
                
                # Then do case-insensitive search
                console.print("\n[bold cyan]Case Insensitive Search (pattern='API')[/bold cyan]")
                await run_demo(
                    session,
                    ripgrep_tool,
                    None,
                    None,
                    pattern="API",
                    paths=["tsap_example_data/documents/"],
                    file_patterns=["*.md"],
                    case_sensitive=False,
                    max_count=5,
                    show_title=False
                )
                
                # DEMO 3: Regular expressions
                console.print(Rule("[bold yellow]Regular Expression Search with Specific Context Controls[/bold yellow]"))
                console.print("[italic]Demonstrates regex patterns with separate before and after context controls[/italic]\n")
                
                await run_demo(
                    session,
                    ripgrep_tool,
                    None,
                    None,
                    pattern=r"def\s+\w+\(",
                    paths=["tsap_example_data/code/"],
                    file_patterns=["*.py"],
                    regex=True,
                    before_context=1,  # Only show 1 line before matches
                    after_context=2,   # Show 2 lines after matches
                    max_count=5,
                    binary=False,      # Explicitly don't search binary files
                    show_title=False
                )
                
                # DEMO 4: File type filtering
                console.print(Rule("[bold yellow]Advanced File Filtering[/bold yellow]"))
                console.print("[italic]Demonstrates filtering with hidden files and .gitignore override options[/italic]\n")
                
                await run_demo(
                    session,
                    ripgrep_tool,
                    None,
                    None,
                    pattern="import",
                    paths=["tsap_example_data/"],
                    file_patterns=["*.py"],
                    hidden=True,       # Search hidden files (dot files)
                    no_ignore=True,    # Ignore .gitignore rules
                    max_total_matches=15,
                    show_title=False
                )
                
                # DEMO 5: Exclusion patterns
                console.print(Rule("[bold yellow]Exclusion Patterns & Symlink Following[/bold yellow]"))
                console.print("[italic]Demonstrates excluding files and symlink handling[/italic]\n")
                
                # First search all files
                console.print("[bold cyan]All Files (searching for 'class')[/bold cyan]")
                await run_demo(
                    session,
                    ripgrep_tool,
                    None,
                    None,
                    pattern="class",
                    paths=["tsap_example_data/code/"],
                    max_count=5,
                    follow_symlinks=True,  # Follow symbolic links during search
                    show_title=False
                )
                
                # Then exclude the tournament file
                console.print("\n[bold cyan]Excluding Tournament File & Following Symlinks[/bold cyan]")
                await run_demo(
                    session,
                    ripgrep_tool,
                    None,
                    None,
                    pattern="class",
                    paths=["tsap_example_data/code/"],
                    exclude_patterns=["*tournament*"],
                    follow_symlinks=True,  # Follow symbolic links during search
                    max_count=5,
                    show_title=False
                )
                
                # DEMO 6: Word boundary matching
                console.print(Rule("[bold yellow]Word Boundary Matching[/bold yellow]"))
                console.print("[italic]Demonstrates searching for whole words only vs. partial matches[/italic]\n")
                
                # First search without word boundaries
                console.print("[bold cyan]Without Word Boundaries (pattern='file')[/bold cyan]")
                await run_demo(
                    session,
                    ripgrep_tool,
                    None,
                    None,
                    pattern="file",
                    paths=["tsap_example_data/documents/"],
                    max_count=10,
                    show_title=False
                )
                
                # Then search with word boundaries
                console.print("\n[bold cyan]With Word Boundaries (pattern='file' with whole_word=True)[/bold cyan]")
                await run_demo(
                    session,
                    ripgrep_tool,
                    None,
                    None,
                    pattern="file",
                    paths=["tsap_example_data/documents/"],
                    whole_word=True,
                    max_count=10,
                    show_title=False
                )
                
                # DEMO 7: Invert match
                console.print(Rule("[bold yellow]Inverted Matching[/bold yellow]"))
                console.print("[italic]Demonstrates finding lines that DON'T contain the pattern[/italic]\n")
                
                await run_demo(
                    session, 
                    ripgrep_tool,
                    None,
                    None,
                    pattern="import",
                    paths=["tsap_example_data/code/main.py"],
                    invert_match=True,
                    show_title=False
                )
                
                # DEMO 8: Advanced regex with capture groups
                console.print(Rule("[bold yellow]Advanced Regex with Capture Groups[/bold yellow]"))
                console.print("[italic]Demonstrates complex pattern matching with regex capture groups[/italic]\n")
                
                await run_demo(
                    session,
                    ripgrep_tool,
                    None,
                    None,
                    pattern=r"class\s+(\w+)[\s\(:]",
                    paths=["tsap_example_data/code/"],
                    regex=True,
                    max_count=10,
                    show_title=False
                )
                
                # DEMO 9: Directory depth limiting with timeout
                console.print(Rule("[bold yellow]Directory Depth Limiting with Timeout Control[/bold yellow]"))
                console.print("[italic]Demonstrates limiting search depth with performance timeout control[/italic]\n")
                
                await run_demo(
                    session,
                    ripgrep_tool,
                    None,
                    None,
                    pattern="README",
                    paths=["tsap_example_data/"],
                    max_depth=2,
                    timeout=5.0,  # Set a 5-second timeout for the search
                    show_title=False
                )
                
    except Exception as e:
        console.print(f"[bold red]Error running demo: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())

async def run_demo(session, ripgrep_tool, title, description, **search_params):
    """Run a ripgrep demo with the given parameters."""
    # Whether to show the title (defaults to True)
    show_title = search_params.pop("show_title", True)
    
    if show_title and title:
        console.print(Rule(f"[bold yellow]{title}[/bold yellow]"))
        
    if description:
        console.print(f"[italic]{description}[/italic]\n")
    
    # Show the search parameters
    console.print("[bold cyan]Search Parameters:[/bold cyan]")
    params_table = Table(show_header=False, box=None)
    params_table.add_column("Parameter", style="green")
    params_table.add_column("Value", style="white")
    
    # Create a mutable copy of search params
    params = dict(search_params)
    
    # Check if we have exclude_patterns and handle differently for display
    exclude_patterns = params.pop("exclude_patterns", None)

    for key, value in params.items():
        # Skip our internal parameter
        if key == "show_title":
            continue
            
        # Format certain parameters for better display
        if isinstance(value, list) and len(value) > 0:
            formatted_value = ", ".join(map(str, value)) # Ensure all items are strings
        else:
            formatted_value = str(value)
            
        params_table.add_row(key, formatted_value)

    # Add exclude_patterns back for display if it existed
    if exclude_patterns:
        params_table.add_row("exclude_patterns", ", ".join(exclude_patterns))
        # Add it back to params for the call
        params["exclude_patterns"] = exclude_patterns
    
    console.print(params_table)
    console.print()
    
    # Execute the search
    start_time = datetime.now()
    console.print("[bold]Executing search...[/bold]")
    
    try:
        # Use proper MCP tool call
        result = await session.call_tool(ripgrep_tool.name, arguments=params)
        
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
            except json.JSONDecodeError:
                console.print("[bold red]Failed to parse response as JSON[/bold red]")
                console.print(f"Raw response: {result_text[:200]}...")
                return
        else:
            console.print("[bold red]No text content in response[/bold red]")
            return
            
    except Exception as e:
        console.print(f"[bold red]Error during ripgrep_search call: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        response = {"matches": [], "error": str(e)}
    
    # Calculate execution time
    execution_time = (datetime.now() - start_time).total_seconds()
    
    # Process and display results
    if isinstance(response, dict) and response:
        if "matches" in response:
            matches = response["matches"]
            match_count = len(matches)
            
            # Show summary statistics
            console.print(f"[green]Found {match_count} matches in {execution_time:.2f} seconds[/green]")
            
            # Show command that was executed if available
            if "command" in response:
                console.print("\n[dim]Command:[/dim]")
                console.print(Syntax(response["command"], "bash", theme="monokai"))
            
            # Display matches in a table
            if match_count > 0:
                results_table = Table(title="Search Results")
                results_table.add_column("File", style="cyan", no_wrap=True)
                results_table.add_column("Line", style="yellow")
                results_table.add_column("Content", style="white")
                
                # Add matches to the table (max 15 for readability)
                for match in matches[:15]:
                    file_path = match.get("path", "")
                    # Make path relative and shorter for display
                    if file_path.startswith("tsap_example_data/"):
                        file_path = os.path.relpath(file_path, "tsap_example_data")
                        
                    line_num = str(match.get("line_number", ""))
                    line_text = match.get("line_text", "").strip()
                    
                    # Highlight the match in the content if possible
                    match_text = match.get("match_text", "")
                    if match_text and match_text in line_text:
                        highlighted = line_text.replace(
                            match_text, 
                            f"[bold red]{match_text}[/bold red]"
                        )
                        results_table.add_row(file_path, line_num, highlighted)
                    else:
                        results_table.add_row(file_path, line_num, line_text)
                
                console.print(results_table)
                
                if match_count > 15:
                    console.print(f"[dim]... and {match_count - 15} more matches not shown[/dim]")
                
                # Show some context for the first match if available
                if matches and (matches[0].get("before_context") or matches[0].get("after_context")):
                    console.print("\n[bold]Context for first match:[/bold]")
                    
                    before = matches[0].get("before_context", [])
                    after = matches[0].get("after_context", [])
                    
                    if before:
                        console.print("[dim italic]Before:[/dim italic]")
                        for line in before:
                            console.print(f"  {line}")
                    
                    console.print(f"[bold red]> {matches[0].get('line_text', '').strip()}[/bold red]")
                    
                    if after:
                        console.print("[dim italic]After:[/dim italic]")
                        for line in after:
                            console.print(f"  {line}")
            else:
                console.print("[yellow]No matches found[/yellow]")
                
            # Show stats if available
            if "stats" in response:
                console.print("\n[bold cyan]Search Statistics:[/bold cyan]")
                stats = response["stats"]
                
                stats_table = Table(show_header=False, box=None)
                stats_table.add_column("Statistic", style="green")
                stats_table.add_column("Value", style="white")
                
                # Add relevant stats
                if "elapsed_total" in stats:
                    stats_table.add_row("Engine Time", f"{stats['elapsed_total']:.3f}s")
                if "files_searched" in stats:
                    stats_table.add_row("Files Searched", str(stats.get("files_searched", "N/A")))
                if "files_with_matches" in stats:
                    stats_table.add_row("Files With Matches", str(stats.get("files_with_matches", "N/A")))
                if "total_matches" in stats:
                    stats_table.add_row("Total Matches", str(stats.get("total_matches", "N/A")))
                if "truncated" in response:
                    stats_table.add_row("Results Truncated", str(response["truncated"]))
                
                console.print(stats_table)
            
        else:
            console.print("[yellow]Search successful but no matches returned[/yellow]")
    else:
        console.print("[bold red]Search failed or returned no data[/bold red]")
        if isinstance(response, dict) and "error" in response:
            console.print(f"Error: {response.get('error')}")
    
    console.print("\n")  # Add space between demos

if __name__ == "__main__":
    if "--debug" in sys.argv:
        DEBUG = True
        debug_print("Debug mode enabled")
    
    try:
        asyncio.run(ripgrep_demo())
    except KeyboardInterrupt:
        console.print("[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Unhandled error: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc()) 