#!/usr/bin/env python3
"""
LLM-based Pattern Evolution Example (MCP Tools Version)

This example demonstrates the use of LLM-powered regex pattern evolution
to find API error messages in logs. It shows how to use the MCP tools interface
for LLM-based pattern generation.
"""
import os
import asyncio
import json
from typing import Dict, List, Any

from rich import print as rich_print
from rich.panel import Panel
from rich.table import Table

# Import proper MCP client modules
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Enable LLM-based pattern generation through the context
# These environment variables are set at client initialization
os.environ["USE_LLM_PATTERN_GENERATION"] = "true"
os.environ["LLM_MCP_SERVER_URL"] = "http://localhost:8013"


async def evaluate_pattern(session, analyze_pattern_tool, pattern: str, description: str, positive_examples: List[str], negative_examples: List[str]) -> Dict[str, Any]:
    """Evaluate a regex pattern against example strings using MCP tools.
    
    Args:
        session: The ClientSession instance
        analyze_pattern_tool: The analyze_pattern tool
        pattern: The regex pattern to evaluate
        description: Description of what the pattern is trying to match
        positive_examples: List of strings that should match the pattern
        negative_examples: List of strings that should not match the pattern
        
    Returns:
        Analysis results including stats and generated variants
    """
    try:
        # Call the analyze_pattern tool 
        result = await session.call_tool(
            analyze_pattern_tool.name,
            arguments={
                "pattern": pattern,
                "description": description,
                "positive_examples": positive_examples,
                "negative_examples": negative_examples,
                "generate_variants": True,
                "num_variants": 5,
                "use_llm": True  # Enable LLM-based pattern generation
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
                response_data = json.loads(result_text)
                
                # Create a response format similar to the original for compatibility
                response = {
                    "original_pattern": pattern,
                    "best_pattern": response_data.get("best_pattern", pattern),
                    "stats": response_data.get("stats", {}),
                    "variants": response_data.get("variants", []),
                    "execution_time": response_data.get("execution_time", 0.0)
                }
                
                return response
            except json.JSONDecodeError:
                rich_print("[bold red]Failed to parse response as JSON[/bold red]")
                rich_print(f"Raw response: {result_text[:200]}...")
                return {
                    "original_pattern": pattern,
                    "best_pattern": pattern,
                    "stats": {"precision": 0.0, "recall": 0.0, "f1_score": 0.0},
                    "variants": [],
                    "execution_time": 0.0
                }
        else:
            rich_print("[bold red]No text content in response[/bold red]")
            return {
                "original_pattern": pattern,
                "best_pattern": pattern,
                "stats": {"precision": 0.0, "recall": 0.0, "f1_score": 0.0},
                "variants": [],
                "execution_time": 0.0
            }
        
    except Exception as e:
        rich_print(f"[bold red]Error during pattern analysis: {str(e)}[/bold red]")
        import traceback
        rich_print(traceback.format_exc())
        return {
            "original_pattern": pattern,
            "best_pattern": pattern,
            "stats": {"precision": 0.0, "recall": 0.0, "f1_score": 0.0},
            "variants": [],
            "execution_time": 0.0
        }


async def main():
    """Run example pattern evolution using MCP tools."""
    rich_print(Panel("[bold blue]TSAP MCP LLM Pattern Evolution Example[/bold blue]", expand=False))
    
    # Log lines for testing
    positive_examples = [
        "2024-07-30 10:02:05 ERROR [APIService] Failed processing request: Invalid token",
        "2024-07-30 10:04:10 ERROR [APIService] Internal server error in /api/v1/users: Database connection timeout",
        "2024-07-30 10:07:50 ERROR [APIService] Invalid input parameters for request ID 83749: Missing required field",
    ]
    
    negative_examples = [
        "2024-07-30 10:00:01 INFO [AuthService] User 'admin' logged in",
        "2024-07-30 10:01:15 WARN [APIService] Request rate limit approaching for client 12345",
        "2024-07-30 10:02:30 INFO [DatabaseService] Connected to database",
        "2024-07-30 10:03:45 DEBUG [APIService] Request parameters: {'id': 123, 'action': 'get'}",
        "2024-07-30 10:05:20 INFO [CacheService] Cache invalidated for user 456",
        "2024-07-30 10:06:35 ERROR [AuthService] Authentication failed for user 'test': Invalid credentials",
    ]
    
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
                                rich_print("[green]Initial info check successful.[/green]")
                            except json.JSONDecodeError:
                                rich_print("[yellow]Info response is not JSON[/yellow]")
                        else:
                            rich_print("[yellow]Info call succeeded but returned no readable content.[/yellow]")
                    except Exception as e:
                        rich_print(f"[bold red]Error during info check: {str(e)}[/bold red]")
                        return
                
                # Initial pattern to evolve
                initial_pattern = r"ERROR.*\[APIService\].*"
                
                rich_print("\n[bold]Original Pattern:[/bold] ", initial_pattern)
                rich_print("[bold]Description:[/bold] Evolve pattern to find APIService errors in logs")
                
                # Run the pattern analysis
                result = await evaluate_pattern(
                    session,
                    analyze_pattern_tool,
                    pattern=initial_pattern,
                    description="Evolve pattern to find APIService errors in logs",
                    positive_examples=positive_examples,
                    negative_examples=negative_examples,
                )
                
                # Display results in a table
                table = Table(title="Pattern Evaluation Results")
                table.add_column("Pattern", style="cyan")
                table.add_column("Precision", style="magenta")
                table.add_column("Recall", style="green")
                table.add_column("F1 Score", style="yellow")
                table.add_column("Source", style="blue")
                
                # Add original pattern
                table.add_row(
                    result["original_pattern"],
                    f"{result['stats'].get('precision', 0.0):.2f}",
                    f"{result['stats'].get('recall', 0.0):.2f}",
                    f"{result['stats'].get('f1_score', 0.0):.2f}",
                    "Original"
                )
                
                # Add variants
                for variant in result["variants"]:
                    # Extract source (LLM or Rule-based) from description
                    variant_desc = variant.get("description", "")
                    if variant_desc and "(LLM)" in variant_desc:
                        source = "LLM"
                    else:
                        source = "Rule-based"
                        
                    variant_stats = variant.get("stats", {})
                    table.add_row(
                        variant.get("pattern", "N/A"),
                        f"{variant_stats.get('precision', 0.0):.2f}",
                        f"{variant_stats.get('recall', 0.0):.2f}",
                        f"{variant_stats.get('f1_score', 0.0):.2f}",
                        source
                    )
                
                rich_print(table)
                
                # Show the best pattern
                rich_print("\n[bold green]Best Pattern:[/bold green] ", result["best_pattern"])
                
                # Show some details about the execution
                rich_print(f"\n[dim]Total execution time: {result['execution_time']:.2f} seconds[/dim]")
                rich_print(f"[dim]Variants generated: {len(result['variants'])}[/dim]")
                
                # Try a more challenging pattern
                rich_print("\n\n[bold]Testing a more challenging pattern...[/bold]")
                
                complex_pattern = r"ERROR.*error.*timeout"
                
                rich_print("\n[bold]Original Pattern:[/bold] ", complex_pattern)
                rich_print("[bold]Description:[/bold] Find timeout error messages regardless of service")
                
                # Run the pattern analysis
                complex_result = await evaluate_pattern(
                    session,
                    analyze_pattern_tool,
                    pattern=complex_pattern,
                    description="Find timeout error messages regardless of service",
                    positive_examples=positive_examples,
                    negative_examples=negative_examples,
                )
                
                # Display results in a table
                complex_table = Table(title="Complex Pattern Evaluation Results")
                complex_table.add_column("Pattern", style="cyan")
                complex_table.add_column("Precision", style="magenta")
                complex_table.add_column("Recall", style="green")
                complex_table.add_column("F1 Score", style="yellow")
                complex_table.add_column("Source", style="blue")
                
                # Add original pattern
                complex_table.add_row(
                    complex_result["original_pattern"],
                    f"{complex_result['stats'].get('precision', 0.0):.2f}",
                    f"{complex_result['stats'].get('recall', 0.0):.2f}",
                    f"{complex_result['stats'].get('f1_score', 0.0):.2f}",
                    "Original"
                )
                
                # Add variants
                for variant in complex_result["variants"]:
                    # Extract source (LLM or Rule-based) from description
                    variant_desc = variant.get("description", "")
                    if variant_desc and "(LLM)" in variant_desc:
                        source = "LLM"
                    else:
                        source = "Rule-based"
                        
                    variant_stats = variant.get("stats", {})
                    complex_table.add_row(
                        variant.get("pattern", "N/A"),
                        f"{variant_stats.get('precision', 0.0):.2f}",
                        f"{variant_stats.get('recall', 0.0):.2f}",
                        f"{variant_stats.get('f1_score', 0.0):.2f}",
                        source
                    )
                
                rich_print(complex_table)
                
                # Show the best pattern
                rich_print("\n[bold green]Best Pattern:[/bold green] ", complex_result["best_pattern"])
    
    except Exception as e:
        rich_print(f"[bold red]Error connecting to MCP server: {str(e)}[/bold red]")
        import traceback
        rich_print(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main()) 