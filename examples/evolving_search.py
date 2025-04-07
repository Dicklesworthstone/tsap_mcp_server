# examples/evolving_search.py
import asyncio
import json

from rich import print as rich_print
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.console import Console

from tsap.toolapi import ToolAPIClient

console = Console() # Instantiate Console

# Add verbose flag for debugging
verbose = False

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

# --- Pattern Analysis Function ---
async def analyze_pattern(client, pattern, description, paths, reference_set=None):
    """Runs pattern analysis using the ToolAPI pattern_analyze command."""
    rich_print(f"[bold cyan]Analyzing Pattern:[/bold cyan] '{pattern}'")
    
    # --- Define ToolAPI Command Arguments ---
    # Based on evolution/pattern_analyzer.py
    args = {
        "pattern": pattern,
        "description": description,
        "is_regex": True,  # Treat as regex
        "paths": paths,    # Paths to search
        "generate_variants": True,
        "num_variants": 3,
    }
    
    # Add reference set if provided
    if reference_set:
        args["reference_set"] = reference_set
    
    # Display parameters
    if verbose:
        rich_print(Syntax(json.dumps(args, indent=2), "json", theme="default")) # Display ToolAPI args
    
    # --- Make ToolAPI Call ---
    rich_print("Sending pattern analyze request...")
    response = await client.send_request("pattern_analyze", args)
    
    # --- Process Response ---
    if not response:
        rich_print("[bold red]ToolAPI request failed: No response received.[/bold red]")
        return None
    
    if response.get("status") != "success":
        error_info = response.get("error", "Unknown error")
        rich_print(f"[bold red]Pattern analysis command failed.[/bold red] Status: {response.get('status', 'N/A')}")
        rich_print(f"Error details: {error_info}")
        return None
    
    if "data" not in response or not response["data"]:
        rich_print("[bold yellow]Analysis successful, but no data returned.[/bold yellow]")
        return None
    
    return response["data"]

# --- Main Evolution Function ---
async def run_pattern_analysis(client):
    """Runs pattern analysis using example data for email patterns."""
    rich_print(Panel("[bold blue]Starting Pattern Analysis for Email Addresses...[/bold blue]", expand=False))
    
    # Initial pattern for email addresses
    initial_pattern = r"\b[\w._%+-]+@[\w.-]+\.\w{2,}\b"
    
    # Set up reference examples
    reference_set = {
        "positive": POSITIVE_EXAMPLES,
        "negative": NEGATIVE_EXAMPLES
    }
    
    # Run the analysis
    analysis_result = await analyze_pattern(
        client=client,
        pattern=initial_pattern,
        description="Evolved email address pattern analysis",
        paths=[],  # Not needed for example-based analysis
        reference_set=reference_set
    )
    
    if not analysis_result:
        return
    
    # --- Display Results ---
    # Get main stats for the initial pattern
    main_stats = analysis_result.get("stats", {})
    variants = analysis_result.get("variants", [])
    exec_time = analysis_result.get("execution_time", 0.0)
    
    # Display stats for the initial pattern
    summary_panel = Panel(
        f"[bold]Initial Pattern:[/bold] [green]{initial_pattern}[/green]\n"
        f"[bold]Precision:[/bold] {main_stats.get('precision', 0.0):.4f}\n"
        f"[bold]Recall:[/bold] {main_stats.get('recall', 0.0):.4f}\n"
        f"[bold]F1 Score:[/bold] {main_stats.get('f1_score', 0.0):.4f}\n"
        f"[bold]Execution Time:[/bold] {exec_time:.4f}s",
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
        sorted_variants = sorted(variants, key=lambda v: v.get("stats", {}).get("f1_score", 0.0), reverse=True)
        
        for variant in sorted_variants:
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
            
        await run_pattern_analysis(client)

if __name__ == "__main__":
    asyncio.run(main())