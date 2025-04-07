#!/usr/bin/env python3
"""
Simple MCP Test

Tests connection to the TSAP MCP server using the standard MCP protocol client.
"""
import asyncio
import os
from rich.console import Console
from rich.panel import Panel

# Import standard MCP components as documented
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

console = Console()

async def main():
    """Test connecting to the MCP server using standard client patterns."""
    console.print(Panel("[bold blue]Standard MCP Client Test (stdio)[/bold blue]"))
    
    # Create a proxy environment with debug mode
    env = os.environ.copy()
    env["MCP_DEBUG"] = "1"
    
    # Create proxy parameters
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_examples/mcp_proxy.py"],
        env=env
    )
    
    console.print("Connecting to MCP server via proxy...")
    
    # Use stdio_client and ClientSession as documented
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            console.print("[bold]Initializing session...[/bold]")
            result = await session.initialize()
            console.print(f"[green]✓ Successfully initialized connection to {result.serverInfo.name} {result.serverInfo.version}[/green]")
            
            # List tools
            console.print("\n[bold]Listing available tools...[/bold]")
            try:
                result = await session.list_tools()
                
                # Debug output the actual structure 
                console.print("[bold cyan]Tools result type:[/bold cyan]", type(result))
                console.print("[bold cyan]Tools result members:[/bold cyan]", dir(result))
                
                # Access the tools through the .tools attribute on ListToolsResult
                if hasattr(result, 'tools'):
                    console.print(f"[green]Found {len(result.tools)} tools[/green]")
                    for tool in result.tools:
                        console.print(f"  - [cyan]{tool.name}[/cyan]: {tool.description}")
                else:
                    console.print("[yellow]Result has no 'tools' attribute![/yellow]")
                    
            except Exception as e:
                console.print(f"[red]Error listing tools: {str(e)}[/red]")
                import traceback
                console.print(traceback.format_exc())
                
            # Stop here for debugging
            console.print("\n[yellow]Completed test[/yellow]")
            return

if __name__ == "__main__":
    # Add a timeout to prevent hanging
    async def run_with_timeout():
        # Set a 15 second timeout
        try:
            await asyncio.wait_for(main(), timeout=15)
        except asyncio.TimeoutError:
            console.print("[red]Operation timed out - check mcp_proxy_debug.log for details[/red]")
    
    try:
        asyncio.run(run_with_timeout())
    except KeyboardInterrupt:
        console.print("[yellow]Test interrupted[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Unhandled error: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc()) 