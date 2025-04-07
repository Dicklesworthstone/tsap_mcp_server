#!/usr/bin/env python3
"""
Evolving Search Demo (MCP Tools Version)

This script demonstrates how to perform pattern evolution and analysis
using the MCP tools interface to generate improved search patterns.
"""
import asyncio
import json
import os

from rich import print as rich_print
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.console import Console

# Import proper MCP client modules
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

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
async def run_pattern_analysis(session, analyze_pattern_tool):
    """Runs pattern analysis using the MCP tools interface."""
    rich_print(Panel("[bold blue]Starting Pattern Analysis for Email Addresses (MCP Tools Version)...[/bold blue]", expand=False))

    # Initial pattern for email addresses
    initial_pattern = r"\b[\w._%+-]+@[\w.-]+\.\w{2,}\b"
    
    rich_print("[cyan]Analyzing pattern with the following configuration:[/cyan]")
    config = {
        "pattern": initial_pattern,
        "description": "Evolved email address pattern analysis",
        "positive_examples": POSITIVE_EXAMPLES,
        "negative_examples": NEGATIVE_EXAMPLES,
        "generate_variants": True,
        "num_variants": 5
    }
    rich_print(Syntax(json.dumps(config, indent=2), "json", theme="default"))

    try:
        # Call the analyze_pattern tool
        result = await session.call_tool(
            analyze_pattern_tool.name,
            arguments={
                "pattern": initial_pattern,
                "description": "Evolved email address pattern analysis",
                "positive_examples": POSITIVE_EXAMPLES,
                "negative_examples": NEGATIVE_EXAMPLES,
                "generate_variants": True,
                "num_variants": 5
            }
        )
        
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
    """Main function to run the pattern analysis example."""
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
                
                await run_pattern_analysis(session, analyze_pattern_tool)
    
    except Exception as e:
        rich_print(f"[bold red]Error connecting to MCP server: {str(e)}[/bold red]")
        import traceback
        rich_print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main()) 