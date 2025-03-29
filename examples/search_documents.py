#!/usr/bin/env python3
"""
Search Documents Example

This script searches for MCP-related content in the example documents.
"""
import asyncio
import json
import os
import sys
from rich import print as rich_print
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.console import Console

from mcp_client_example import MCPClient  # Import our existing client

console = Console()

async def search_documents():
    """Search for MCP-related content in example documents."""
    rich_print(Panel("[bold blue]Search MCP-related Content in Documents[/bold blue]", expand=False))
    
    try:
        async with MCPClient() as client:
            # --- Add initial info check ---
            console.print(f"Attempting to get server info from {client.base_url}...")
            info = await client.info()
            if info.get("status") != "success" or info.get("error") is not None:
                console.print(f"[bold red]Error during initial client.info() check:[/bold red] {info.get('error', 'Status was not success')}")
                return # Exit if server check fails
            else:
               console.print("[green]Initial client.info() check successful.[/green]")
            # ------------------------------

            # Use ripgrep to search for "MCP" in the example documents
            rich_print("\n[bold]Searching for MCP in example documents...[/bold]")
            search_response = await client.ripgrep_search(
                pattern="MCP",
                paths=["tsap_example_data/documents/"],
                file_patterns=["*.md", "*.txt", "*.json"],
                context_lines=2
            )
            
            # Print the raw response for debugging (only in verbose mode)
            if "--verbose" in sys.argv:
                rich_print(Syntax(json.dumps(search_response, indent=2, default=str), "json"))
            
            # --- Refined Success Check Logic ---
            if search_response.get("status") == "success":
                if "data" in search_response and search_response["data"] is not None:
                    if "matches" in search_response["data"]:
                        matches = search_response["data"]["matches"]
                        match_count = len(matches)
                        
                        rich_print(f"\n[green]Found {match_count} matches for 'MCP' in documents[/green]")
                        
                        # Display matches in a table
                        table = Table(title=f"MCP Content in Documents ({match_count} matches)")
                        table.add_column("File", style="cyan", no_wrap=True)
                        table.add_column("Line", style="yellow", justify="right")
                        table.add_column("Content", style="white", max_width=80)
                        
                        # Add matches to the table
                        for match in matches:
                            # Use 'path' instead of 'file' to access the file path
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
                                table.add_row(file_path, line_num, highlighted)
                            else:
                                table.add_row(file_path, line_num, line_text)
                        
                        rich_print(table)
                    else:
                        rich_print("[yellow]Search successful but no matches field found in data[/yellow]")
                else:
                    rich_print("[yellow]Search successful but no data returned[/yellow]")
            else:
                # Handle unsuccessful status
                error_info = search_response.get("error", "Unknown error")
                rich_print(f"[bold red]Search failed.[/bold red] Status: {search_response.get('status', 'N/A')}")
                rich_print(f"Error details: {error_info}")
            # --- End Refined Success Check Logic ---

    except Exception as e:
        rich_print(f"[bold red]Error during search: {str(e)}[/bold red]")
        import traceback
        rich_print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(search_documents()) 