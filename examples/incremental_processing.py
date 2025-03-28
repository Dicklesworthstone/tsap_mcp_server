# examples/incremental_processing.py
import asyncio
import os
import json

from rich import print as rich_print
from rich.panel import Panel
from rich.syntax import Syntax

from api_client import TSAPClient, API_KEY # Import from our client module

# --- Configuration ---
DATA_DIR = "tsap_example_data/large_data" # Directory with potentially large log files

# --- Main Incremental Example Function ---
async def run_incremental_example(client: TSAPClient):
    """Demonstrates an operation that likely uses incremental processing internally."""
    rich_print(Panel("[bold blue]Demonstrating Large Data Search (using Ripgrep)...[/bold blue]", expand=False))
    rich_print(f"Searching for 'ERROR' in directory: {DATA_DIR}")
    rich_print("[dim]Note: TSAP handles large directories/files internally, potentially using incremental processing "
               "even if not explicitly requested via the API.[/dim]")

    # --- Define Request Payload ---
    # We use ripgrep search on a directory. TSAP's ripgrep tool is expected
    # to handle large directories and files efficiently, possibly streaming results
    # or processing files one by one, which aligns with the *spirit* of incremental processing.
    # There isn't a specific "incremental_process" API endpoint based on the provided code.
    request_payload = {
        "params": {
            "pattern": "ERROR",
            "paths": [DATA_DIR], # Target a directory
            "case_sensitive": False,
            "file_patterns": ["*.log"], # Specify file types if known
            "context_lines": 1,
            "max_total_matches": 50 # Limit results for the example output
        },
        "performance_mode": "standard",
        "async_execution": False # Run sync for this example
    }

    rich_print("[cyan]Sending Ripgrep Request (on directory):[/cyan]")
    rich_print(Syntax(json.dumps(request_payload, indent=2), "json", theme="default"))

    # --- Make API Call ---
    # Using the core ripgrep endpoint
    response = await client.post("/api/core/ripgrep", payload=request_payload)

    # --- Process Response ---
    if not response or "error" in response:
        rich_print("[bold red]Search request failed.[/bold red]", response)
        return

    if "result" not in response:
         rich_print("[bold red]Unexpected response format:[/bold red]", response)
         return

    search_result = response["result"]
    matches = search_result.get("matches", [])
    stats = search_result.get("stats", {})
    truncated = search_result.get("truncated", False)

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
        rich_print("Please create it and add some sample .log files (e.g., data_part1.log, data_part2.log).")
        return
    if not os.listdir(DATA_DIR):
         rich_print(f"[bold yellow]Warning:[/bold yellow] Data directory '{DATA_DIR}' is empty.")

    if API_KEY == "your-default-api-key":
        rich_print("[bold yellow]Warning:[/bold yellow] Using default API key. Set the TSAP_API_KEY environment variable.")

    async with TSAPClient() as client:
        await run_incremental_example(client)

if __name__ == "__main__":
    asyncio.run(main())