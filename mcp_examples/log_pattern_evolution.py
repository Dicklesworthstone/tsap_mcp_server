#!/usr/bin/env python3
"""
Log Pattern Evolution Demo (MCP Tools Version)

This script demonstrates how to evolve and optimize patterns for log analysis
using the MCP tools interface.
"""
import asyncio
import json
import time
import os

from rich import print as rich_print
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.console import Console

# Import proper MCP client modules
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

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
async def run_log_pattern_analysis(session, analyze_pattern_tool):
    """Runs pattern analysis using the MCP tools for log data."""
    rich_print(Panel("[bold blue]Starting Pattern Analysis for APIService Errors (MCP Tools Version)...[/bold blue]", expand=False))

    # Initial attempt: Find lines with ERROR followed by [APIService]
    initial_pattern = r"ERROR.*\[APIService\].*"
    
    rich_print("[cyan]Analyzing pattern with the following configuration:[/cyan]")
    config = {
        "pattern": initial_pattern,
        "description": "Evolve pattern to find APIService errors in logs",
        "positive_examples": POSITIVE_EXAMPLES,
        "negative_examples": NEGATIVE_EXAMPLES,
        "generate_variants": True,
        "num_variants": 5
    }
    rich_print(Syntax(json.dumps(config, indent=2), "json", theme="default"))

    try:
        # Record start time for client-side execution time calculation
        start_time = time.time()
        
        # Call the analyze_pattern tool
        result = await session.call_tool(
            analyze_pattern_tool.name,
            arguments={
                "pattern": initial_pattern,
                "description": "Evolve pattern to find APIService errors in logs",
                "positive_examples": POSITIVE_EXAMPLES,
                "negative_examples": NEGATIVE_EXAMPLES,
                "generate_variants": True,
                "num_variants": 5
            }
        )
        
        # Calculate client-side execution time
        end_time = time.time()
        client_exec_time = end_time - start_time
        
        # Extract the text content
        result_text = None
        for content in result.content:
            if content.type == "text":
                result_text = content.text
                break
        
        if result_text:
            try:
                # Parse the JSON response
                analysis_result = json.loads(result_text)
                
                # Extract results
                main_stats = analysis_result.get("stats", {})
                variants = analysis_result.get("variants", [])
                
                # Use server execution time if available, otherwise use client time
                server_exec_time = analysis_result.get("execution_time")
                display_exec_time = server_exec_time if server_exec_time is not None else client_exec_time
                
                # Display stats for the initial pattern
                summary_panel = Panel(
                    f"[bold]Initial Pattern:[/bold] [green]{initial_pattern}[/green]\n"
                    f"[bold]Precision:[/bold] {main_stats.get('precision', 0.0):.4f}\n"
                    f"[bold]Recall:[/bold] {main_stats.get('recall', 0.0):.4f}\n"
                    f"[bold]F1 Score:[/bold] {main_stats.get('f1_score', 0.0):.4f}\n"
                    f"[bold]Execution Time:[/bold] {display_exec_time:.4f}s",
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
                
                # Optional: Show best pattern explicitly
                if variants:
                    best_variant = variants[0]  # Already sorted above
                    best_pattern = best_variant.get("pattern", initial_pattern)
                    best_f1 = best_variant.get("stats", {}).get("f1_score", 0.0)
                    if best_f1 > main_stats.get("f1_score", 0.0):
                        rich_print(f"\n[bold green]Best Pattern Found:[/bold green] {best_pattern}")
                        rich_print(f"F1 Score: {best_f1:.4f} (improved from {main_stats.get('f1_score', 0.0):.4f})")
            except json.JSONDecodeError:
                rich_print("[bold red]Failed to parse response as JSON[/bold red]")
                rich_print(f"Raw response: {result_text[:200]}...")
        else:
            rich_print("[bold red]No text content in response[/bold red]")
    
    except Exception as e:
        rich_print(f"[bold red]Error during pattern analysis: {str(e)}[/bold red]")
        import traceback
        rich_print(traceback.format_exc())

# --- Main Execution --- 
async def main():
    """Main function to run the log pattern analysis example."""
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
                
                # Find the analyze_pattern and info tools
                analyze_pattern_tool = next((t for t in tools if t.name == "analyze_pattern"), None)
                info_tool = next((t for t in tools if t.name == "info"), None)
                
                if not analyze_pattern_tool:
                    rich_print("[bold red]Error: analyze_pattern tool not found![/bold red]")
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
                                rich_print("[green]MCP server check successful.[/green]")
                            except json.JSONDecodeError:
                                rich_print("[yellow]Info response is not JSON[/yellow]")
                        else:
                            rich_print("[yellow]Info call succeeded but returned no readable content.[/yellow]")
                    except Exception as e:
                        rich_print(f"[bold red]Error during info check: {str(e)}[/bold red]")
                        return
                    
                await run_log_pattern_analysis(session, analyze_pattern_tool)
    
    except Exception as e:
        rich_print(f"[bold red]Error connecting to MCP server: {str(e)}[/bold red]")
        import traceback
        rich_print(traceback.format_exc())

if __name__ == "__main__":
    # Make sure the server has the latest pattern_analyzer code loaded!
    # If you changed server code, restart the server before running this.
    asyncio.run(main()) 