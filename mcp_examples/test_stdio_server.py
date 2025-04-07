#!/usr/bin/env python3
"""
Test MCP StdIO Server

This script tests the MCP StdIO server by connecting to it using the standard MCP client.
"""
import asyncio
import subprocess
import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule

# Import standard MCP client libraries
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

console = Console()

async def test_server():
    """Test the MCP StdIO server."""
    console.print(Panel("[bold blue]MCP StdIO Server Test[/bold blue]"))
    
    # Start the server process
    console.print("[bold]Starting MCP StdIO server...[/bold]")
    server_process = subprocess.Popen(
        ["python", "mcp_examples/mcp_stdio_server.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    # Wait a moment for the server to initialize
    await asyncio.sleep(1)
    
    # Create server parameters
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_examples/mcp_stdio_server.py"]
    )
    
    console.print("[bold]Connecting to MCP StdIO server...[/bold]")
    
    try:
        # Connect to the server
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                console.print("\n[bold]Initializing session...[/bold]")
                init_result = await session.initialize()
                console.print(f"[green]✓ Successfully initialized connection to {init_result.serverInfo.name} {init_result.serverInfo.version}[/green]")
                
                # List available tools
                console.print("\n[bold]Listing available tools...[/bold]")
                tools_result = await session.list_tools()
                console.print(f"[green]Found {len(tools_result.tools)} tools[/green]")
                
                tools_table = Table(title="Available Tools")
                tools_table.add_column("Name", style="cyan")
                tools_table.add_column("Description", style="green")
                
                for tool in tools_result.tools:
                    tools_table.add_row(tool.name, tool.description)
                
                console.print(tools_table)
                
                # Test echo tool
                console.print(Rule("[bold]Testing 'echo' tool[/bold]"))
                echo_result = await session.call_tool("echo", arguments={"message": "Hello, MCP StdIO Server!"})
                
                # Extract text content
                echo_text = None
                for content in echo_result.content:
                    if content.type == "text":
                        echo_text = content.text
                        break
                
                console.print(f"[bold]Echo Result:[/bold] {echo_text}")
                
                # Test calculator tool
                console.print(Rule("[bold]Testing 'calculator' tool[/bold]"))
                calc_result = await session.call_tool(
                    "calculator", 
                    arguments={
                        "operation": "add",
                        "a": 5,
                        "b": 3
                    }
                )
                
                # Extract text content
                calc_text = None
                for content in calc_result.content:
                    if content.type == "text":
                        calc_text = content.text
                        break
                
                console.print(f"[bold]Calculator Result (5 + 3):[/bold] {calc_text}")
                
                # Test list_files tool
                console.print(Rule("[bold]Testing 'list_files' tool[/bold]"))
                files_result = await session.call_tool("list_files", arguments={})
                
                # Extract text content
                files_text = None
                for content in files_result.content:
                    if content.type == "text":
                        files_text = content.text
                        break
                
                if files_text:
                    files_list = json.loads(files_text)
                    console.print(f"[bold]Available Files:[/bold] {', '.join(files_list)}")
                    
                    # Test read_file tool with one of the files
                    if files_list:
                        test_file = files_list[0]
                        console.print(Rule(f"[bold]Testing 'read_file' tool with '{test_file}'[/bold]"))
                        file_result = await session.call_tool(
                            "read_file", 
                            arguments={"filename": test_file}
                        )
                        
                        # Extract text content
                        file_content = None
                        for content in file_result.content:
                            if content.type == "text":
                                file_content = content.text
                                break
                        
                        console.print(f"[bold]File Content:[/bold] {file_content}")
                
                # Test get_user tool
                console.print(Rule("[bold]Testing 'get_user' tool[/bold]"))
                user_result = await session.call_tool(
                    "get_user", 
                    arguments={"user_id": "user1"}
                )
                
                # Extract text content
                user_text = None
                for content in user_result.content:
                    if content.type == "text":
                        user_text = content.text
                        break
                
                if user_text:
                    user_data = json.loads(user_text)
                    console.print(f"[bold]User Data:[/bold] {user_data}")
                
                # Test info tool
                console.print(Rule("[bold]Testing 'info' tool[/bold]"))
                info_result = await session.call_tool("info", arguments={})
                
                # Extract text content
                info_text = None
                for content in info_result.content:
                    if content.type == "text":
                        info_text = content.text
                        break
                
                if info_text:
                    info_data = json.loads(info_text)
                    
                    info_table = Table(title="Server Information")
                    info_table.add_column("Property", style="cyan")
                    info_table.add_column("Value", style="green")
                    
                    for key, value in info_data.items():
                        if isinstance(value, list):
                            info_table.add_row(key, ", ".join(value))
                        else:
                            info_table.add_row(key, str(value))
                    
                    console.print(info_table)
                
                # Test resource access
                console.print(Rule("[bold]Testing resource access[/bold]"))
                
                try:
                    # Get user resource
                    user_resource = await session.get_resource("user/user1")
                    resource_text = None
                    for content in user_resource.content:
                        if content.type == "text":
                            resource_text = content.text
                            break
                    
                    if resource_text:
                        resource_data = json.loads(resource_text)
                        console.print(f"[bold]User Resource Data:[/bold] {resource_data}")
                    
                    # Get file resource
                    if files_list:
                        file_resource = await session.get_resource(f"file/{files_list[0]}")
                        resource_text = None
                        for content in file_resource.content:
                            if content.type == "text":
                                resource_text = content.text
                                break
                        
                        console.print(f"[bold]File Resource Content:[/bold] {resource_text}")
                
                except Exception as e:
                    console.print(f"[yellow]Resource access error: {e}[/yellow]")
                
                console.print("\n[green]✓ All tests completed successfully![/green]")
    
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
    
    finally:
        # Stop the server process
        console.print("\n[bold]Stopping MCP StdIO server...[/bold]")
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    try:
        asyncio.run(test_server())
    except KeyboardInterrupt:
        console.print("[yellow]Test interrupted[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Unhandled error: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc()) 