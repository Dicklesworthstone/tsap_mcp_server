#!/usr/bin/env python3
"""
Incremental Processing Example (MCP Tools Version)

This script demonstrates how to process large data efficiently using 
the MCP tools interface, which handles incremental processing internally.
"""
import asyncio
import os
import json

from rich import print as rich_print
from rich.panel import Panel
from rich.syntax import Syntax

# Import proper MCP client modules
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# --- Configuration ---
DATA_DIR = "tsap_example_data/large_data" # Directory with potentially large log files

# --- Main Incremental Example Function ---
async def run_incremental_example(session, search_tool):
    """Demonstrates an operation that likely uses incremental processing internally using MCP tools."""
    rich_print(Panel("[bold blue]Demonstrating Large Data Search with MCP Tools...[/bold blue]", expand=False))
    rich_print(f"Searching for 'ERROR' in directory: {DATA_DIR}")
    rich_print("[dim]Note: TSAP MCP handles large directories/files internally, potentially using incremental processing "
               "even if not explicitly requested.[/dim]")
    
    # Define search configuration
    config = {
        "query": "ERROR",
        "paths": [DATA_DIR],
        "case_sensitive": False,
        "file_patterns": ["*.log"],
        "context_lines": 1,
        "max_matches": 50
    }
    
    rich_print("[cyan]Sending Search Request (on directory):[/cyan]")
    rich_print(Syntax(json.dumps(config, indent=2), "json", theme="default"))

    try:
        # Call the search tool using MCP protocol
        result = await session.call_tool(
            search_tool.name,
            arguments={
                "query": "ERROR",
                "paths": [DATA_DIR],  # Target a directory
                "case_sensitive": False,
                "file_patterns": ["*.log"],  # Specify file types if known
                "context_lines": 1,
                "max_matches": 50  # Limit results for the example output
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
                
                # Process results
                matches = response.get("matches", [])
                truncated = response.get("truncated", False)
                stats = response.get("stats", {})
                
                rich_print(Panel(f"[bold green]Search Complete:[/bold green] Found {len(matches)} matches.", expand=False))
                
                if truncated:
                    rich_print("[yellow]Results were truncated based on max_matches.[/yellow]")
                    
                # Display first few matches
                if matches:
                    rich_print("[bold]Sample Matches:[/bold]")
                    for match in matches[:5]:  # Show first 5 matches
                        file_name = os.path.basename(match.get('file', ''))
                        line_num = match.get('line', '?')
                        line_text = match.get('text', '').strip()
                        rich_print(f"  [cyan]{file_name}[/cyan]:[yellow]{line_num}[/yellow] - {line_text}")
                        
                # Display stats
                if stats:
                    stats_text = (
                        f"[bold]Files Searched:[/bold] {stats.get('files_searched', 'N/A')}\n"
                        f"[bold]Files with Matches:[/bold] {stats.get('files_with_matches', 'N/A')}\n"
                        f"[bold]Total Matches (reported):[/bold] {len(matches)}\n"
                        # Add more stats if available in the actual response
                    )
                    rich_print(Panel(stats_text, title="Search Statistics", border_style="blue"))
            except json.JSONDecodeError:
                rich_print("[bold red]Failed to parse response as JSON[/bold red]")
                rich_print(f"Raw response: {result_text[:200]}...")
        else:
            rich_print("[bold red]No text content in response[/bold red]")
            
    except Exception as e:
        rich_print(f"[bold red]Error during search operation: {str(e)}[/bold red]")
        import traceback
        rich_print(traceback.format_exc())

# --- Main Execution ---
async def main():
    """Main function to run the incremental processing example."""
    # Ensure the example data directory exists
    if not os.path.exists(DATA_DIR):
        rich_print(f"[bold red]Error:[/bold red] Data directory '{DATA_DIR}' not found.")
        rich_print("Please create it and add some sample .log files (e.g., data_part1.log, data_part2.log).")
        return
    if not os.listdir(DATA_DIR):
         rich_print(f"[bold yellow]Warning:[/bold yellow] Data directory '{DATA_DIR}' is empty.")

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
                
                # Get the list of available tools
                tools_result = await session.list_tools()
                tools = tools_result.tools
                
                # Find the search and info tools
                search_tool = next((t for t in tools if t.name == "search"), None)
                info_tool = next((t for t in tools if t.name == "info"), None)
                
                if not search_tool:
                    rich_print("[bold red]Error: search tool not found![/bold red]")
                    return
                
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
                                info_data = json.loads(info_text)  # noqa: F841
                                rich_print("[green]Initial info check successful.[/green]")
                            except json.JSONDecodeError:
                                rich_print("[yellow]Info response is not JSON[/yellow]")
                        else:
                            rich_print("[yellow]Info call succeeded but returned no readable content.[/yellow]")
                    except Exception as e:
                        rich_print(f"[bold red]Error during info check: {str(e)}[/bold red]")
                        return
                
                await run_incremental_example(session, search_tool)
    
    except Exception as e:
        rich_print(f"[bold red]Error connecting to MCP server: {str(e)}[/bold red]")
        import traceback
        rich_print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main()) 