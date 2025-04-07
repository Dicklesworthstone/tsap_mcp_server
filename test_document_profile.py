#!/usr/bin/env python3
"""Test script for document_profile tool."""
import asyncio
import os
import json
from rich.console import Console
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

console = Console()

async def main():
    """Run a simple test of the document_profile tool."""
    console.print("[bold]Testing document_profile tool[/bold]")

    # Path to the proxy script
    proxy_path = os.path.join("mcp_examples", "mcp_proxy.py")
    
    # Configure server parameters for stdio connection
    server_params = StdioServerParameters(
        command="python",
        args=[proxy_path],
        env={"MCP_DEBUG": "1"}
    )
    
    # Test file path
    test_file = "tsap_example_data/documents/strategic_thinking.txt"
    abs_path = os.path.abspath(test_file)
    console.print(f"Testing with file: {abs_path}")
    
    try:
        # Connect to the MCP server via proxy
        async with stdio_client(server_params) as (read, write):
            console.print("Connected to MCP proxy")
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()
                console.print("Session initialized")
                
                # Get available tools
                tools_result = await session.list_tools()
                tools = tools_result.tools
                
                # Find the document_profile tool
                profile_tool = next((t for t in tools if t.name == "document_profile"), None)
                
                if not profile_tool:
                    console.print("[red]Error: document_profile tool not found![/red]")
                    return
                
                # Try different query formats
                query_formats = [
                    f"file={abs_path}",
                    f"path={abs_path}",
                    f"file_path={abs_path}",
                    f"document={abs_path}",
                    abs_path,  # Just the path itself
                ]
                
                for i, query in enumerate(query_formats):
                    console.print(f"\n[bold]Test #{i+1}[/bold]: query=\"{query}\"")
                    try:
                        result = await session.call_tool(
                            profile_tool.name,
                            arguments={"query": query}
                        )
                        
                        # Extract the text content
                        result_text = None
                        for content in result.content:
                            if content.type == "text":
                                result_text = content.text
                                break
                        
                        if result_text:
                            try:
                                response = json.loads(result_text)
                                console.print("[green]Successfully received JSON response![/green]")
                                console.print(f"Response data: {response}")
                            except json.JSONDecodeError:
                                console.print("[red]Failed to parse response as JSON[/red]")
                                console.print(f"Raw response: {result_text[:200]}...")
                        else:
                            console.print("[red]No text content in response[/red]")
                    
                    except Exception as e:
                        console.print(f"[red]Error during test #{i+1}: {str(e)}[/red]")
                        
    except Exception as e:
        console.print(f"[bold red]Error running test: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main()) 