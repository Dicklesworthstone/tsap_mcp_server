#!/usr/bin/env python3
"""
Search Documents Example (MCP Tools Version)

This script searches for patterns in example documents using the MCP tools interface.
"""
import asyncio
import json
import os
import sys
import argparse
from rich import print as rich_print
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.console import Console

# Import proper MCP client modules
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

console = Console()

async def search_documents(pattern: str, paths: list[str], file_patterns: list[str], context_lines: int):
    """Search for content in example documents using MCP tools."""
    rich_print(Panel(f"[bold blue]Search Documents (MCP Tools Version): '{pattern}'[/bold blue]", expand=False))
    rich_print(f"Paths: {paths}\nFile Patterns: {file_patterns}\nContext: {context_lines}")
    
    # Path to the proxy script
    proxy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_proxy.py")
    console.print(f"Using proxy script: {proxy_path}")
    
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
                
                # Get the list of available tools
                tools_result = await session.list_tools()
                tools = tools_result.tools
                
                # Find the search and info tools
                search_tool = next((t for t in tools if t.name == "search"), None)
                info_tool = next((t for t in tools if t.name == "info"), None)
                
                if not search_tool:
                    console.print("[bold red]Error: search tool not found![/bold red]")
                    return
                
                # --- Add initial info check ---
                if info_tool:
                    console.print("Checking server info...")
                    try:
                        info_result = await session.call_tool(info_tool.name, arguments={})
                        info_text = None
                        for content in info_result.content:
                            if content.type == "text":
                                info_text = content.text
                                break
                        
                        if info_text:
                            try:
                                info_data = json.loads(info_text)  # noqa: F841
                                console.print("[green]Initial info check successful.[/green]")
                            except json.JSONDecodeError:
                                console.print("[yellow]Info response is not JSON[/yellow]")
                        else:
                            console.print("[yellow]Info call succeeded but returned no readable content.[/yellow]")
                    except Exception as e:
                        console.print(f"[bold red]Error during info check:[/bold red] {str(e)}")
                        return
                # ------------------------------
                
                # Use ripgrep tool to search using provided parameters
                rich_print(f"\n[bold]Searching for '{pattern}'...[/bold]")
                
                try:
                    # Call the search tool
                    result = await session.call_tool(
                        search_tool.name,
                        arguments={
                            "query": pattern,
                            "paths": paths,
                            "file_patterns": file_patterns,
                            "context_lines": context_lines
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
                            search_results = json.loads(result_text)
                            
                            # Print the raw response for debugging (only in verbose mode)
                            if "--verbose" in sys.argv:
                                rich_print(Syntax(json.dumps(search_results, indent=2, default=str), "json"))
                            
                            # Process and display results
                            matches = search_results.get("matches", [])
                            match_count = len(matches)
                            
                            if match_count > 0:
                                rich_print(f"\n[green]Found {match_count} matches for '{pattern}'[/green]")
                                
                                # Display matches in a table
                                table = Table(title=f"Search Results for '{pattern}' ({match_count} matches)")
                                table.add_column("File", style="cyan", no_wrap=True)
                                table.add_column("Line", style="yellow", justify="right")
                                table.add_column("Content", style="white", max_width=80)
                                
                                # Add matches to the table
                                for match in matches:
                                    # Get file path
                                    file_path = match.get("file", "")
                                    
                                    # Make path relative and shorter for display
                                    if file_path.startswith("tsap_example_data/"):
                                        file_path = os.path.relpath(file_path, "tsap_example_data")
                                        
                                    line_num = str(match.get("line", ""))
                                    line_text = match.get("text", "").strip()
                                    
                                    # Highlight the match in the content if possible
                                    matched_text = match.get("matched_text", "")
                                    if matched_text and matched_text in line_text:
                                        highlighted = line_text.replace(
                                            matched_text, 
                                            f"[bold red]{matched_text}[/bold red]"
                                        )
                                        table.add_row(file_path, line_num, highlighted)
                                    else:
                                        table.add_row(file_path, line_num, line_text)
                                
                                rich_print(table)
                            else:
                                rich_print(f"[yellow]No matches found for '{pattern}'[/yellow]")
                        except json.JSONDecodeError:
                            console.print("[bold red]Failed to parse search response as JSON[/bold red]")
                            console.print(f"Raw response: {result_text[:200]}...")
                    else:
                        console.print("[bold red]No text content in search response[/bold red]")
                
                except Exception as e:
                    rich_print(f"[bold red]Error during search operation: {str(e)}[/bold red]")
                    import traceback
                    rich_print(traceback.format_exc())
            
    except Exception as e:
        rich_print(f"[bold red]Error connecting to MCP server: {str(e)}[/bold red]")
        import traceback
        rich_print(traceback.format_exc())

if __name__ == "__main__":
    # --- Add Argument Parsing ---
    parser = argparse.ArgumentParser(description="Search documents using TSAP MCP tools.")
    parser.add_argument(
        "-p", "--pattern",
        type=str,
        default="MCP",
        help="The text pattern to search for (default: MCP)"
    )
    parser.add_argument(
        "--paths",
        nargs='+',
        default=["tsap_example_data/documents/"],
        help="List of paths (files or directories) to search in (default: tsap_example_data/documents/)"
    )
    parser.add_argument(
        "--file-patterns",
        nargs='+',
        default=["*.md", "*.txt", "*.json"],
        help="Glob patterns for file types to include (default: *.md *.txt *.json)"
    )
    parser.add_argument(
        "-C", "--context-lines",
        type=int,
        default=2,
        help="Number of context lines to show around matches (default: 2)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the raw JSON response from the server"
    )
    args = parser.parse_args()

    # Add --verbose to sys.argv if set, for the existing check within the function
    if args.verbose:
        if "--verbose" not in sys.argv:
             sys.argv.append("--verbose")

    # Run the async function with parsed arguments
    asyncio.run(search_documents(
        pattern=args.pattern,
        paths=args.paths,
        file_patterns=args.file_patterns,
        context_lines=args.context_lines
    )) 