#!/usr/bin/env python3
"""
Simple test for LLM pattern generation (MCP Tools Version)

This is a minimal script to verify the LLM pattern generation feature
using the MCP tools interface.
"""
import asyncio
import os
import json
from rich.console import Console
from rich.panel import Panel

# Import proper MCP client modules
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

console = Console()

async def test_llm_pattern_generation():
    """Test the LLM pattern generation functionality using MCP tools."""
    console.print(Panel.fit(
        "[bold blue]TSAP MCP LLM Pattern Generation Test[/bold blue]",
        subtitle="Testing LLM-based pattern evolution"
    ))
    
    # Test data
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
    
    # Initial pattern
    pattern = r"ERROR.*\[APIService\].*"
    description = "Evolve pattern to find APIService errors in logs"
    
    console.print(f"[bold]Initial pattern:[/bold] {pattern}")
    console.print(f"[bold]Description:[/bold] {description}")
    console.print(f"[bold]Positive examples:[/bold] {len(positive_examples)}")
    console.print(f"[bold]Negative examples:[/bold] {len(negative_examples)}")
    
    # Path to the proxy script
    proxy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_proxy.py")
    console.print(f"Using proxy script: {proxy_path}")
    
    # Configure server parameters for stdio connection to our proxy
    server_params = StdioServerParameters(
        command="python",
        args=[proxy_path],
        env={"MCP_DEBUG": "0"}  # Disable debug logging by default
    )
    
    try:
        # Connect to the MCP server via proxy
        console.print("Connecting to MCP server...")
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()
                
                # Get the list of available tools
                tools_result = await session.list_tools()
                tools = tools_result.tools
                
                # Find the pattern evolution tool
                pattern_evolution_tool = next((t for t in tools if t.name == "pattern_evolution"), None)
                if not pattern_evolution_tool:
                    console.print("[bold red]Error: pattern_evolution tool not found![/bold red]")
                    return
                
                # Call the LLM pattern generation tool
                console.print("\n[bold]Calling LLM pattern generation...[/bold]")
                
                # Use the MCP tools pattern_evolution
                try:
                    result = await session.call_tool(
                        pattern_evolution_tool.name, 
                        arguments={
                            "initial_pattern": pattern,
                            "description": description,
                            "positive_examples": positive_examples,
                            "negative_examples": negative_examples,
                            "num_variants": 3
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
                            result_data = json.loads(result_text)
                            
                            # Check if the call was successful
                            if result_data.get("status") == "success":
                                patterns = result_data.get("patterns", [])
                                console.print("[green]Successfully generated patterns:[/green]")
                                
                                # Display the generated patterns
                                for i, p in enumerate(patterns):
                                    pattern_str = p.get("pattern", "N/A")
                                    score = p.get("score", 0.0)
                                    console.print(f"Pattern {i+1} (score: {score:.2f}): [bold cyan]{pattern_str}[/bold cyan]")
                                    
                                    # Show matches if available
                                    matches = p.get("matches", [])
                                    if matches:
                                        console.print("  [dim]Sample matches:[/dim]")
                                        for match in matches[:3]:
                                            console.print(f"  - {match}")
                                        if len(matches) > 3:
                                            console.print(f"  [dim]... and {len(matches) - 3} more matches[/dim]")
                            else:
                                error = result_data.get("error", {})
                                error_msg = error.get("message", "Unknown error")
                                console.print(f"[bold red]Error during LLM pattern generation: {error_msg}[/bold red]")
                        except json.JSONDecodeError:
                            console.print("[bold red]Failed to parse response as JSON[/bold red]")
                            console.print(f"Raw response: {result_text[:200]}...")
                    else:
                        console.print("[bold red]No text content in response[/bold red]")
                
                except Exception as e:
                    console.print(f"[bold red]Error calling pattern_evolution: {str(e)}[/bold red]")
    
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(test_llm_pattern_generation()) 