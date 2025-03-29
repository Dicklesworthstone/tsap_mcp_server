#!/usr/bin/env python3
"""
Simple MCP Search Example

This script demonstrates a simple MCP search using the ripgrep command.
"""
import asyncio
# import httpx # No longer needed directly

from mcp_client_example import MCPClient # Import MCPClient
from rich.panel import Panel # Added
from rich.table import Table # Added
from rich.console import Console # Added
import os # Added for path manipulation
from datetime import datetime # Added for timing

console = Console() # Create console instance

# Server URL is handled by MCPClient
# SERVER_URL = "http://localhost:8021"

async def simple_search():
    """Run a simple search test using MCPClient with rich formatting."""
    console.print(Panel.fit("[bold blue]Simple MCP Ripgrep Search Demo[/bold blue]"))
    # console.print("=====================") # Replaced by Panel
    
    # Define search parameters
    search_args = {
        "pattern": "MCP",
        "paths": ["tsap_example_data/documents/"],
        "case_sensitive": False,
        "context_lines": 0, # Reduced context for cleaner table
        "file_patterns": ["*.md", "*.txt", "*.json"],
        "stats": True # Request stats for summary
    }
    
    # Display search parameters
    console.print("[bold cyan]Search Parameters:[/bold cyan]")
    params_table = Table(show_header=False, box=None, padding=(0, 1))
    params_table.add_column("Parameter", style="green")
    params_table.add_column("Value", style="white")
    for key, value in search_args.items():
        if isinstance(value, list):
            formatted_value = ", ".join(map(str, value))
        else:
            formatted_value = str(value)
        params_table.add_row(key, formatted_value)
    console.print(params_table)
    console.print()

    async with MCPClient() as client:
        try:
            # --- Add initial info check ---
            console.print(f"Attempting to get server info from {client.base_url}...")
            info = await client.info()
            if info.get("status") != "success" or info.get("error") is not None:
                console.print(f"[bold red]MCP server check failed:[/bold red] {info.get('error', 'Status was not success')}")
                return
            else:
                console.print("[green]MCP server check successful.[/green]")
            # ------------------------------

            console.print("\nSending ripgrep_search request via MCPClient...")
            start_time = datetime.now()
            
            # Send request using MCPClient method
            result = await client.ripgrep_search(**search_args)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Remove raw JSON print
            # console.print("\n[cyan]Response:[/cyan]")
            # console.print(Syntax(json.dumps(result, indent=2), "json", theme="default"))
            
            # Check for matches (using refined status check)
            if result.get("status") == "success":
                if result.get("data") and "matches" in result["data"]:
                    matches = result["data"]["matches"]
                    match_count = len(matches)
                    console.print(f"\n[green]Found {match_count} matches in {execution_time:.2f} seconds[/green]")

                    if match_count > 0:
                        results_table = Table(title="Search Results")
                        results_table.add_column("File", style="cyan", no_wrap=True)
                        results_table.add_column("Line", style="yellow", justify="right")
                        results_table.add_column("Content", style="white")
                        
                        # Add matches to the table (max 15 for readability)
                        for match in matches[:15]:
                            file_path = match.get("path", "")
                            # Make path relative if possible
                            try:
                                if os.path.commonpath([os.getcwd(), os.path.abspath(file_path)]) == os.getcwd():
                                     rel_path = os.path.relpath(os.path.abspath(file_path), os.getcwd())
                                     # Shorten if it starts with the target dir
                                     if rel_path.startswith(search_args["paths"][0]):
                                         file_path = os.path.join("...", os.path.basename(rel_path))
                                     else:
                                         file_path = rel_path
                            except ValueError:
                                pass # Keep original path if common path fails
                                
                            line_num = str(match.get("line_number", ""))
                            line_text = match.get("line_text", "").strip().replace('\r', '') # Clean up line text
                            
                            # Highlight the match in the content if possible
                            match_text = match.get("match_text", "")
                            if match_text and match_text in line_text: # Check if substring exists (case-sensitive)
                                # Use simple string replace - ripgrep provides match_text with original casing
                                highlighted = line_text.replace(match_text, 
                                                     f"[bold red]{match_text}[/bold red]"
                                                     ) # Removed count=1
                                
                                results_table.add_row(file_path, line_num, highlighted)
                            else:
                                # Fallback if match_text isn't directly in line_text (shouldn't happen often)
                                results_table.add_row(file_path, line_num, line_text)
                        
                        console.print(results_table)
                        
                        if match_count > 15:
                            console.print(f"[dim]... and {match_count - 15} more matches not shown[/dim]")

                    # Show stats if available
                    if "stats" in result["data"]:
                        console.print("\n[bold cyan]Search Statistics:[/bold cyan]")
                        stats = result["data"]["stats"]
                        
                        stats_table = Table(show_header=False, box=None, padding=(0, 1))
                        stats_table.add_column("Statistic", style="green")
                        stats_table.add_column("Value", style="white")
                        
                        # Add relevant stats
                        if "elapsed_total" in stats and stats["elapsed_total"] is not None:
                             try:
                                 stats_table.add_row("Engine Time", f"{float(stats['elapsed_total']):.4f}s")
                             except (ValueError, TypeError):
                                  stats_table.add_row("Engine Time", str(stats['elapsed_total'])) # Fallback
                        elif "stats" in stats and "elapsed" in stats["stats"]:
                            try:
                                nanos = stats["stats"]["elapsed"]["nanos"]
                                secs = stats["stats"]["elapsed"]["secs"]
                                stats_table.add_row("Engine Time (rg)", f"{secs + nanos / 1e9:.4f}s")
                            except (KeyError, TypeError):
                                pass # Ignore if format is unexpected
                        if "files_searched" in stats and stats["files_searched"] is not None:
                            stats_table.add_row("Files Searched", str(stats.get("files_searched", "N/A")))
                        if "files_with_matches" in stats and stats["files_with_matches"] is not None:
                            stats_table.add_row("Files With Matches", str(stats.get("files_with_matches", "N/A")))
                        if "total_matches" in stats and stats["total_matches"] is not None:
                            stats_table.add_row("Total Matches (Engine)", str(stats.get("total_matches", "N/A")))
                        elif "stats" in stats and "matches" in stats["stats"]:
                             stats_table.add_row("Total Matches (Engine)", str(stats["stats"]["matches"])) 
                        if "truncated" in result["data"]:
                            stats_table.add_row("Results Truncated", str(result["data"]["truncated"]))
                        
                        console.print(stats_table)
                else:
                    console.print("[yellow]\nSearch successful, but no matches found in data.[/yellow]")
            else:
                error_info = result.get("error", "Unknown error")
                console.print(f"[bold red]\nSearch failed.[/bold red] Status: {result.get('status', 'N/A')}")
                console.print(f"Error details: {error_info}")
                
        except Exception as e:
            console.print(f"[bold red]Error during client operation or search: {str(e)}[/bold red]")
            import traceback
            console.print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(simple_search()) 