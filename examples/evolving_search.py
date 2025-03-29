# examples/evolving_search.py
import asyncio
import json

from rich import print as rich_print
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.console import Console

from mcp_client_example import MCPClient

console = Console() # Instantiate Console


# --- Configuration ---
POSITIVE_EXAMPLES = [
    "user@example.com",
    "john.doe123@company.co.uk",
    "support+alias@my-site.info",
    "test.email@domain.name",
]
NEGATIVE_EXAMPLES = [
    "not an email",
    "user@",
    "@domain.com",
    "user@domain",
    "user@domain.",
    "user @ domain.com",
]

# --- Main Evolution Function ---
async def run_pattern_analysis(client: MCPClient):
    """Runs pattern analysis using the MCP pattern_analyze command."""
    rich_print(Panel("[bold blue]Starting Pattern Analysis for Email Addresses...[/bold blue]", expand=False))

    # --- Define MCP Command Arguments ---
    # Based on handle_pattern_analyze in mcp/handler.py
    initial_pattern = r"\b[\w._%+-]+@[\w.-]+\.\w{2,}\b" # Use the first initial pattern
    mcp_args = {
        "pattern": initial_pattern,
        "description": "Evolved email address pattern analysis", # Updated description
        "is_regex": True,
        "case_sensitive": False,
        "reference_set": { # Assumed structure for reference set
            "positive": POSITIVE_EXAMPLES,
            "negative": NEGATIVE_EXAMPLES
        },
        "generate_variants": True, # Request variants
        "num_variants": 5, # Generate a few variants
        # "paths": [], # Omitted, not needed for example-based analysis
    }

    rich_print("[cyan]Sending Pattern Analysis Request:[/cyan]")
    rich_print(Syntax(json.dumps(mcp_args, indent=2), "json", theme="default")) # Display MCP args

    # --- Make MCP Call (Synchronous) ---
    response = await client.send_request("pattern_analyze", mcp_args)

    # --- Process Direct Response ---
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
    # Assuming structure based on PatternStats model from pattern_analyzer.py
    main_stats = analysis_result.get("stats", {})
    variants = analysis_result.get("variants", [])
    # Safely get and convert execution time
    exec_time_raw = analysis_result.get("execution_time", 0.0) # Default to 0.0
    try:
        exec_time = float(exec_time_raw)
    except (ValueError, TypeError):
        exec_time = 0.0 # Fallback if conversion fails

    # Display stats for the initial pattern
    summary_panel = Panel(
        f"[bold]Initial Pattern:[/bold] [green]{initial_pattern}[/green]\n"
        f"[bold]Precision:[/bold] {main_stats.get('precision', 0.0):.4f}\n"
        f"[bold]Recall:[/bold] {main_stats.get('recall', 0.0):.4f}\n"
        f"[bold]F1 Score:[/bold] {main_stats.get('f1_score', 0.0):.4f}\n"
        f"[bold]Execution Time:[/bold] {exec_time:.2f}s",
        title="Initial Pattern Analysis Result",
        border_style="blue"
    )
    console.print(summary_panel)

    # Display generated variants if available
    if variants:
        alt_table = Table(title=f"Generated Variants ({len(variants)} shown)")
        alt_table.add_column("Variant Pattern", style="cyan")
        alt_table.add_column("Precision", style="yellow", justify="right")
        alt_table.add_column("Recall", style="yellow", justify="right")
        alt_table.add_column("F1 Score", style="yellow", justify="right")

        # Sort variants by F1-score descending
        variants.sort(key=lambda v: v.get("stats", {}).get("f1_score", 0.0), reverse=True)

        for variant in variants:
            variant_stats = variant.get("stats", {})
            alt_table.add_row(
                variant.get("pattern", "N/A"),
                f"{variant_stats.get('precision', 0.0):.4f}",
                f"{variant_stats.get('recall', 0.0):.4f}",
                f"{variant_stats.get('f1_score', 0.0):.4f}"
            )
        console.print(alt_table)
    else:
        rich_print("[dim]No pattern variants were generated or returned.[/dim]")

# --- Main Execution ---
async def main():
    """Main function to run the pattern analysis example."""
    # Removed API_KEY check
    # if API_KEY == "your-default-api-key":
    #     rich_print("[bold yellow]Warning:[/bold yellow] Using default API key. Set the TSAP_API_KEY environment variable.")

    async with MCPClient() as client: # Changed to MCPClient
        # Check server health
        rich_print(f"Attempting to get server info from {client.base_url}...")
        info = await client.info()
        if info.get("status") != "success" or info.get("error") is not None:
            rich_print("[bold red]MCP server check failed. Make sure the MCP server is running.[/bold red]")
            rich_print(f"Info response: {info}")
            return
        else:
            rich_print("[green]MCP server check successful.[/green]")
            
        await run_pattern_analysis(client) # Call renamed function

if __name__ == "__main__":
    asyncio.run(main())