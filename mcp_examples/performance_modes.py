#!/usr/bin/env python3
"""
Performance Modes Example (MCP Tools Version)

This script demonstrates the different performance modes available in TSAP MCP,
showing how they affect search results, analysis depth, and execution time.
"""
import asyncio
import time
import statistics
import os
import json
from typing import List, Dict, Any
import argparse

from rich import print as rich_print
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

# Import proper MCP client modules
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Create console instance
console = Console()

# Performance mode descriptions
MODE_DESCRIPTIONS = {
    "fast": "Optimized for speed with potential trade-offs in result quality or depth",
    "standard": "Balanced mode providing good results with reasonable performance",
    "deep": "Thorough analysis prioritizing comprehensive results over performance"
}

async def benchmark_search(session, search_tool, mode: str) -> Dict[str, Any]:
    """Benchmark search with specified performance mode using MCP tools."""
    rich_print(f"\n[bold cyan]Running search in [bold]{mode}[/bold] mode...[/bold cyan]")
    
    # Search parameters - complex enough to show performance differences
    pattern = "function|class|method|async|await"
    paths = ["tsap_example_data/"]
    
    start_time = time.time()
    try:
        # Call the search tool from MCP tools with performance mode
        result = await session.call_tool(
            search_tool.name,
            arguments={
                "query": pattern,
                "paths": paths,
                "regex": True,
                "case_sensitive": False,
                "file_patterns": ["*.py", "*.md", "*.txt", "*.json"],
                "max_matches": 1000,  # Reasonable limit
                "mode": mode  # Pass performance mode as a parameter
            }
        )
        
        elapsed = time.time() - start_time
        
        # Extract the text content from the response
        result_text = None
        for content in result.content:
            if content.type == "text":
                result_text = content.text
                break
        
        if result_text:
            try:
                # Parse the JSON response
                response = json.loads(result_text)
                
                # Extract match count
                match_count = len(response.get("matches", []))
                
                # Return results metrics
                return {
                    "mode": mode,
                    "execution_time": elapsed,
                    "match_count": match_count,
                    "success": True
                }
            except json.JSONDecodeError:
                rich_print("[bold red]Failed to parse search response as JSON[/bold red]")
                return {
                    "mode": mode,
                    "execution_time": elapsed,
                    "match_count": 0,
                    "success": False
                }
        else:
            rich_print("[bold red]No text content in search response[/bold red]")
            return {
                "mode": mode,
                "execution_time": elapsed,
                "match_count": 0,
                "success": False
            }
    except Exception as e:
        elapsed = time.time() - start_time
        rich_print(f"[yellow]Error during search in {mode} mode: {str(e)}[/yellow]")
        
        # Return error results
        return {
            "mode": mode,
            "execution_time": elapsed,
            "match_count": 0,
            "success": False
        }

async def compare_performance_modes(session, operations: List[str] = None, iterations: int = 1):
    """Compare different performance modes for various operations using MCP tools."""
    if operations is None:
        operations = ["search"]
    
    modes = ["fast", "standard", "deep"]
    
    # Get the list of available tools
    tools_result = await session.list_tools()
    tools = tools_result.tools
    
    # Find the search tool
    search_tool = next((t for t in tools if t.name == "search"), None)
    if not search_tool:
        rich_print("[bold red]Error: search tool not found![/bold red]")
        return
    
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
                    
                    if operation == "search":
                        result = await benchmark_search(session, search_tool, mode)
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
    parser = argparse.ArgumentParser(description="Benchmark TSAP MCP performance modes")
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations for each benchmark")
    parser.add_argument("--operations", nargs="+", default=["search"], help="Operations to benchmark")
    return parser.parse_args()

async def main():
    """Main function."""
    args = parse_args()
    
    rich_print(Panel(
        "[bold blue]TSAP MCP Performance Modes Benchmark[/bold blue]",
        subtitle=f"Comparing fast, standard, and deep modes across {', '.join(args.operations)}"
    ))
    
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
                
                # Find the info tool for initial check
                tools_result = await session.list_tools()
                tools = tools_result.tools
                info_tool = next((t for t in tools if t.name == "info"), None)
                
                # Check if the server is running
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
                        rich_print(f"[bold red]Error during info check:[/bold red] {str(e)}")
                        return
                
                rich_print("[green]Connected to MCP server successfully[/green]")
                rich_print(f"Running {args.iterations} iterations for each mode")
                
                await compare_performance_modes(
                    session,
                    operations=args.operations,
                    iterations=args.iterations
                )
            
    except Exception as e:
        rich_print(f"[bold red]Error: {str(e)}[/bold red]")
        import traceback
        rich_print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main()) 