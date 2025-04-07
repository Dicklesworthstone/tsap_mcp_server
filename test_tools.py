#!/usr/bin/env python3
"""Test script to check available tools and their parameters"""
import asyncio
import os
import json
from rich.console import Console
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

console = Console()

async def main():
    """Run the tools test."""
    console.print("[bold]Testing MCP Tools[/bold]")

    # Path to the proxy script
    proxy_path = os.path.join("mcp_examples", "mcp_proxy.py")
    
    # Configure server parameters for stdio connection
    server_params = StdioServerParameters(
        command="python",
        args=[proxy_path],
        env={"MCP_DEBUG": "1"}
    )
    
    try:
        # Connect to the MCP server via proxy
        async with stdio_client(server_params) as (read, write):
            console.print("Connected to MCP proxy")
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()
                console.print("Session initialized successfully")
                
                # Get available tools
                tools_result = await session.list_tools()
                tools = tools_result.tools
                console.print(f"Found {len(tools)} tools")
                
                # Find the document tools
                document_profile_tool = next((t for t in tools if t.name == "document_profile"), None)
                document_explore_tool = next((t for t in tools if t.name == "document_explore"), None)
                
                if document_profile_tool:
                    console.print("[bold]Document Profile Tool:[/bold]")
                    console.print(f"Description: {document_profile_tool.description}")
                    console.print("Input Schema:")
                    console.print(document_profile_tool.inputSchema)
                else:
                    console.print("[red]Document Profile Tool not found[/red]")
                
                if document_explore_tool:
                    console.print("\n[bold]Document Explore Tool:[/bold]")
                    console.print(f"Description: {document_explore_tool.description}")
                    console.print("Input Schema:")
                    console.print(document_explore_tool.inputSchema)
                else:
                    console.print("[red]Document Explore Tool not found[/red]")
                
    except Exception as e:
        console.print(f"[bold red]Error running test: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main()) 