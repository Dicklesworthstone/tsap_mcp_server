#!/usr/bin/env python3
"""
Performance Modes Example

This script demonstrates the different performance modes available in TSAP,
showing how they affect search results, analysis depth, and execution time.
"""
import asyncio
import time
import statistics
from typing import List, Dict, Any
import argparse

from rich import print as rich_print
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from tsap.mcp import MCPClient

# Create console instance
console = Console()

# Performance mode descriptions
MODE_DESCRIPTIONS = {
    "fast": "Optimized for speed with potential trade-offs in result quality or depth",
    "standard": "Balanced mode providing good results with reasonable performance",
    "deep": "Thorough analysis prioritizing comprehensive results over performance"
}

async def benchmark_ripgrep_search(client: MCPClient, mode: str) -> Dict[str, Any]:
    """Benchmark ripgrep search with specified performance mode."""
    rich_print(f"\n[bold cyan]Running ripgrep search in [bold]{mode}[/bold] mode...[/bold cyan]")
    
    # Search parameters - complex enough to show performance differences
    pattern = "function|class|method|async|await"
    paths = ["tsap_example_data/"]
    
    start_time = time.time()
    response = await client.send_request(
        "ripgrep_search", 
        {
            "pattern": pattern,
            "paths": paths,
            "regex": True,
            "case_sensitive": False,
            "file_patterns": ["*.py", "*.md", "*.txt", "*.json"],
            "max_total_matches": 1000  # Reasonable limit
        },
        mode=mode  # Set performance mode
    )
    elapsed = time.time() - start_time
    
    # Extract results
    match_count = 0
    if "data" in response and response["data"] is not None:
        if "matches" in response["data"]:
            match_count = len(response["data"]["matches"])
    
    # Return results metrics
    return {
        "mode": mode,
        "execution_time": elapsed,
        "match_count": match_count,
        "success": response.get("status") == "success"
    }

async def compare_performance_modes(client: MCPClient, operations: List[str] = None, iterations: int = 1):
    """Compare different performance modes for various operations."""
    if operations is None:
        operations = ["ripgrep"]
    
    modes = ["fast", "standard", "deep"]
    
    # Results storage
    all_results = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[yellow]Running performance benchmarks...", total=len(operations) * len(modes) * iterations)
        
        for operation in operations:
            all_results[operation] = {}
            
            for mode in modes:
                all_results[operation][mode] = []
                
                for i in range(iterations):
                    progress.update(task, description=f"[bold]{operation}[/bold] in [cyan]{mode}[/cyan] mode (iteration {i+1}/{iterations})")
                    
                    if operation == "ripgrep":
                        result = await benchmark_ripgrep_search(client, mode)
                        all_results[operation][mode].append(result)
                    # Add more operations here as needed
                    
                    progress.advance(task)
    
    # Display results
    for operation in operations:
        rich_print(Panel(f"[bold blue]Performance Comparison: {operation.upper()}[/bold blue]", expand=False))
        
        # Create performance comparison table
        table = Table(title=f"{operation.capitalize()} Performance by Mode")
        table.add_column("Mode", style="cyan")
        table.add_column("Description", style="dim", max_width=60)
        table.add_column("Avg. Time (s)", style="yellow", justify="right")
        table.add_column("Matches", style="green", justify="right")
        table.add_column("Success Rate", style="magenta", justify="right")
        
        for mode in modes:
            results = all_results[operation][mode]
            
            # Skip if no results for this mode
            if not results:
                continue
                
            # Calculate metrics
            avg_time = statistics.mean([r["execution_time"] for r in results])
            avg_matches = statistics.mean([r["match_count"] for r in results])
            success_rate = sum(1 for r in results if r["success"]) / len(results) * 100
            
            # Add row to table
            table.add_row(
                f"[bold]{mode.upper()}[/bold]",
                MODE_DESCRIPTIONS.get(mode, ""),
                f"{avg_time:.2f}",
                f"{int(avg_matches)}",
                f"{success_rate:.0f}%"
            )
            
        console.print(table)
        
        # Show visual comparison of execution times
        if all(all_results[operation][mode] for mode in modes):
            rich_print("\n[bold]Execution Time Comparison:[/bold]")
            max_time = max(statistics.mean([r["execution_time"] for r in all_results[operation][mode]]) for mode in modes)
            
            for mode in modes:
                avg_time = statistics.mean([r["execution_time"] for r in all_results[operation][mode]])
                bar_width = int((avg_time / max_time) * 40) if max_time > 0 else 0
                bar_color = "green" if mode == "fast" else "yellow" if mode == "standard" else "red"
                rich_print(f"{mode.upper():8} [bold]{avg_time:.2f}s[/bold] [bold {bar_color}]{'█' * bar_width}[/bold {bar_color}]")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark TSAP performance modes")
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations for each benchmark")
    parser.add_argument("--operations", nargs="+", default=["ripgrep"], help="Operations to benchmark")
    return parser.parse_args()

async def main():
    """Main function."""
    args = parse_args()
    
    rich_print(Panel(
        "[bold blue]TSAP Performance Modes Benchmark[/bold blue]",
        subtitle=f"Comparing fast, standard, and deep modes across {', '.join(args.operations)}"
    ))
    
    try:
        async with MCPClient() as client:
            # --- Restored initial info check ---
            # Check if the server is running
            rich_print(f"Attempting to get server info from {client.base_url}...") # Debug print
            info = await client.info()
            if info.get("status") != "success" or info.get("error") is not None:
                rich_print(f"[bold red]Error during initial client.info() check:[/bold red] {info.get('error', 'Status was not success')}")
                return 
            else:
               rich_print("[green]Initial client.info() check successful.[/green]")
            # -------------------------------------------------
                
            rich_print(f"[green]Connected to client instance for server: {client.base_url}[/green]") # Modified message
            rich_print(f"Running {args.iterations} iterations for each mode")
            
            await compare_performance_modes(
                client,
                operations=args.operations,
                iterations=args.iterations
            )
            
    except Exception as e:
        rich_print(f"[bold red]Error: {str(e)}[/bold red]")
        import traceback
        rich_print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main())
