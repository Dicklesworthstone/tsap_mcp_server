# examples/log_pattern_evolution.py
import asyncio
import json

from rich import print as rich_print
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.console import Console

# Assuming mcp_client_example.py is in the same directory orPYTHONPATH
from tsap.mcp import MCPClient

console = Console()

# --- Configuration: Define our target and non-target log lines --- 

# Goal: Find ERROR messages specifically from APIService
POSITIVE_EXAMPLES = [
    "2024-07-30 10:02:05 ERROR [APIService] Failed processing request for /v1/orders: Missing 'Authorization' header. Request ID: abc-123",
    "2024-07-30 10:04:10 ERROR [APIService] Internal server error processing /v1/products: NullPointerException at ProductMapper.java:52. Request ID: def-456",
    "2024-07-30 10:07:50 ERROR [APIService] Invalid input format for /v1/users/update: Unexpected field 'email_address'. Expected 'email'. Request ID: ghi-789"
]

NEGATIVE_EXAMPLES = [
    "2024-07-30 10:00:01 INFO [AuthService] User 'admin' logged in successfully.",
    "2024-07-30 10:01:15 WARN [APIService] Request rate limit approaching for endpoint /v1/users. Limit: 100/min.", # Correct service, wrong level
    "2024-07-30 10:02:30 INFO [DatabaseService] Connection established to primary DB.",
    "2024-07-30 10:03:45 ERROR [DatabaseService] Query failed: Timeout waiting for connection pool. Query: SELECT * FROM products.", # Correct level, wrong service
    "2024-07-30 10:05:00 DEBUG [APIService] Request payload for /v1/widgets: {'name': 'Thingamajig', 'quantity': 5}", # Correct service, wrong level
    "2024-07-30 10:06:22 INFO [AuthService] User 'guest' session expired."
]

# --- Main Evolution Function --- 
async def run_log_pattern_analysis(client: MCPClient):
    """Runs pattern analysis using the MCP pattern_analyze command for log data."""
    rich_print(Panel("[bold blue]Starting Pattern Analysis for APIService Errors...[/bold blue]", expand=False))

    # --- Define MCP Command Arguments --- 
    # Initial attempt: Find lines with ERROR followed by [APIService]
    initial_pattern = r"ERROR.*\[APIService\].*"
    
    mcp_args = {
        "pattern": initial_pattern,
        "description": "Evolve pattern to find APIService errors in logs",
        "is_regex": True,
        "case_sensitive": False, # Log levels might be upper/lower case
        "reference_set": { 
            "positive": POSITIVE_EXAMPLES,
            "negative": NEGATIVE_EXAMPLES
        },
        "generate_variants": True, 
        "num_variants": 5, # Request a few variants
        # "paths": ["examples/sample.log"], # Could use paths instead of reference_set strings, but examples are clearer for demo
    }

    rich_print("[cyan]Sending Pattern Analysis Request:[/cyan]")
    rich_print(Syntax(json.dumps(mcp_args, indent=2), "json", theme="default"))

    # --- Make MCP Call --- 
    start_time = asyncio.get_event_loop().time()
    response = await client.send_request("pattern_analyze", mcp_args)
    end_time = asyncio.get_event_loop().time()
    client_exec_time = end_time - start_time

    # --- Process Response --- 
    if not response:
        rich_print("[bold red]MCP request failed: No response received.[/bold red]")
        return

    if response.get("status") != "success":
        error_info = response.get("error", "Unknown error")
        rich_print(f"[bold red]Pattern analysis command failed.[/bold red] Status: {response.get('status', 'N/A')}")
        rich_print(f"Error details: {error_info}")
        return

    if "data" not in response or not response["data"]:
        rich_print("[bold yellow]Analysis successful, but no data returned.[/bold yellow]")
        return

    # --- Display Results --- 
    analysis_result = response["data"]
    main_stats = analysis_result.get("stats", {})
    variants = analysis_result.get("variants", [])
    # Use total execution time reported by the server if available, else use client time
    server_exec_time = analysis_result.get("execution_time") 
    display_exec_time = server_exec_time if server_exec_time is not None else client_exec_time
    
    # Display stats for the initial pattern
    summary_panel = Panel(
        f"[bold]Initial Pattern:[/bold] [green]{initial_pattern}[/green]\n"
        f"[bold]Precision:[/bold] {main_stats.get('precision', 0.0):.4f}\n"
        f"[bold]Recall:[/bold] {main_stats.get('recall', 0.0):.4f}\n"
        f"[bold]F1 Score:[/bold] {main_stats.get('f1_score', 0.0):.4f}\n"
        f"[bold]Server Exec Time:[/bold] {display_exec_time:.4f}s",
        title="Initial Pattern Analysis Result",
        border_style="blue"
    )
    console.print(summary_panel)

    # Display generated variants if available
    if variants:
        # Sort variants by F1-score descending (primary), then Precision (secondary)
        variants.sort(key=lambda v: (
            v.get("stats", {}).get("f1_score", 0.0),
            v.get("stats", {}).get("precision", 0.0)
            ), reverse=True)
        
        alt_table = Table(title=f"Generated Variants ({len(variants)} shown, sorted by F1 then Precision)")
        alt_table.add_column("Variant Pattern", style="cyan", overflow="fold")
        alt_table.add_column("Precision", style="yellow", justify="right")
        alt_table.add_column("Recall", style="yellow", justify="right")
        alt_table.add_column("F1 Score", style="magenta", justify="right")
        alt_table.add_column("Exec Time (s)", style="dim", justify="right")

        for variant in variants:
            variant_stats = variant.get("stats", {})
            alt_table.add_row(
                variant.get("pattern", "N/A"),
                f"{variant_stats.get('precision', 0.0):.4f}",
                f"{variant_stats.get('recall', 0.0):.4f}",
                f"{variant_stats.get('f1_score', 0.0):.4f}",
                f"{variant_stats.get('execution_time', 0.0):.5f}"
            )
        console.print(alt_table)
    else:
        rich_print("[dim]No pattern variants were generated or returned.[/dim]")

# --- Main Execution --- 
async def main():
    """Main function to run the log pattern analysis example."""
    async with MCPClient() as client:
        # Check server health
        rich_print(f"Attempting to get server info from {client.base_url}...")
        info = await client.info()
        if info.get("status") != "success" or info.get("error") is not None:
            rich_print("[bold red]MCP server check failed. Make sure the MCP server is running.[/bold red]")
            rich_print(f"Info response: {info}")
            return
        else:
            rich_print("[green]MCP server check successful.[/green]")
            
        await run_log_pattern_analysis(client)

if __name__ == "__main__":
    # Make sure the server has the latest pattern_analyzer code loaded!
    # If you changed server code, restart the server before running this.
    asyncio.run(main()) 