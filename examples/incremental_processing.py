# examples/incremental_processing.py
import asyncio
import os
import json

from rich import print as rich_print
from rich.panel import Panel
from rich.syntax import Syntax

from tsap.toolapi import ToolAPIClient # Import from the library

# --- Configuration ---
DATA_DIR = "tsap_example_data/documents" # Directory with sample documents

# --- Main Incremental Example Function ---
async def run_incremental_example(client: ToolAPIClient):
    """Demonstrates an operation that likely uses incremental processing internally."""
    rich_print(Panel("[bold blue]Demonstrating Large Data Search (using Ripgrep)...[/bold blue]", expand=False))
    rich_print(f"Searching for 'the' in directory: {DATA_DIR}")
    rich_print("[dim]Note: TSAP handles large directories/files internally, potentially using incremental processing "
               "even if not explicitly requested via the API.[/dim]")

    # --- Define Request Parameters ---
    # We use ripgrep search on a directory. TSAP's ripgrep tool is expected
    # to handle large directories and files efficiently, possibly streaming results
    # or processing files one by one, which aligns with the *spirit* of incremental processing.
    search_params = {
        "pattern": "the",
        "paths": [DATA_DIR], # Target a directory
        "case_sensitive": False,
        "file_patterns": ["*.md", "*.txt"], # Specify file types if known
        "context_lines": 1,
        "max_total_matches": 50 # Limit results for the example output
    }

    rich_print("[cyan]Sending Ripgrep Request (on directory):[/cyan]")
    rich_print(Syntax(json.dumps(search_params, indent=2), "json", theme="default"))

    # --- Make API Call using ToolAPIClient ---
    response = await client.ripgrep_search(**search_params)

    # --- Process Response ---
    if not response or response.get("status") != "success":
        rich_print("[bold red]Search request failed.[/bold red]", response)
        return

    data = response.get("data", {})
    matches = data.get("matches", [])
    stats = data.get("stats", {})
    truncated = data.get("truncated", False)

    rich_print(Panel(f"[bold green]Search Complete:[/bold green] Found {len(matches)} matches.", expand=False))
    if truncated:
        rich_print("[yellow]Results were truncated based on max_total_matches.[/yellow]")

    # Display first few matches
    if matches:
        rich_print("[bold]Sample Matches:[/bold]")
        for match in matches[:5]: # Show first 5 matches
            file_name = os.path.basename(match.get('path', ''))
            line_num = match.get('line_number', '?')
            line_text = match.get('line_text', '').strip()
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

# --- Main Execution ---
async def main():
    """Main function to run the incremental processing example."""
    # Ensure the example data directory exists
    if not os.path.exists(DATA_DIR):
        rich_print(f"[bold red]Error:[/bold red] Data directory '{DATA_DIR}' not found.")
        rich_print("Please create it and add some sample text files (e.g., sample.txt, sample.md).")
        return
    if not os.listdir(DATA_DIR):
         rich_print(f"[bold yellow]Warning:[/bold yellow] Data directory '{DATA_DIR}' is empty.")

    async with ToolAPIClient() as client:
        # Check server health
        rich_print(f"Attempting to get server info from {client.base_url}...")
        info = await client.info()
        if info.get("status") != "success" or info.get("error") is not None:
            rich_print("[bold red]ToolAPI server check failed. Make sure the ToolAPI server is running.[/bold red]")
            rich_print(f"Info response: {info}")
            return
        else:
            rich_print("[green]ToolAPI server check successful.[/green]")
            
        await run_incremental_example(client)

if __name__ == "__main__":
    asyncio.run(main())