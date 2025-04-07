#!/usr/bin/env python3
"""
Debug Tools

A simple script to directly test the TSAP server API and log the response.
"""
import asyncio
import json
import httpx
from rich.console import Console
from rich.syntax import Syntax

console = Console()

async def main():
    """Test the TSAP server API directly."""
    server_url = "http://localhost:8021/mcp/"
    console.print(f"Connecting to TSAP server at: {server_url}")
    
    # Create the client
    async with httpx.AsyncClient(base_url=server_url, timeout=30.0) as client:
        # Test list_tools
        console.print("\n[bold]Testing list_tools command...[/bold]")
        request = {
            "request_id": "test-1",
            "command": "list_tools",
            "args": {}
        }
        
        response = await client.post("", json=request)
        result = response.json()
        
        # Check for success
        if result.get("status") == "success":
            console.print("[green]Successfully received tools list[/green]")
            
            # Log the exact structure
            console.print("\n[bold cyan]Response structure:[/bold cyan]")
            console.print(Syntax(json.dumps(result, indent=2), "json"))
            
            # Parse the structure
            data = result.get("data", {})
            
            # Log each category
            console.print("\n[bold yellow]Tool categories and counts:[/bold yellow]")
            for category, tools in data.items():
                if isinstance(tools, list):
                    console.print(f"  {category}: {len(tools)} tools")
                    # Show first 3 tools in each category
                    for i, tool in enumerate(tools[:3]):
                        console.print(f"    {i+1}. [cyan]{tool.get('name')}[/cyan]: {tool.get('description')}")
                    if len(tools) > 3:
                        console.print(f"    ... and {len(tools) - 3} more")
            
            # Count total tools
            total_tools = sum([len(tools) for category, tools in data.items() 
                              if isinstance(tools, list)])
            console.print(f"\nTotal tools: {total_tools}")
            
            # Check if tools have arguments
            console.print("\n[bold yellow]Checking for argument structures...[/bold yellow]")
            for category, tools in data.items():
                if not isinstance(tools, list):
                    continue
                
                for tool in tools:
                    tool_name = tool.get("name", "")
                    args = tool.get("arguments", [])
                    if args:
                        console.print(f"  [cyan]{tool_name}[/cyan] has {len(args)} arguments")
                        # Show the first argument
                        if args:
                            arg = args[0]
                            console.print(f"    Example arg: {arg.get('name')} - {arg.get('description')}")
                        break
        else:
            console.print(f"[red]Failed to get tools: {result.get('error')}[/red]")

if __name__ == "__main__":
    asyncio.run(main()) 