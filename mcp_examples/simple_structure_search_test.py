#!/usr/bin/env python3
"""
Simple Structure Search Test (MCP Protocol Version)

This script demonstrates the basic usage of the structure search tool 
using the standard MCP protocol client.
"""
import asyncio
import os
import sys
import json
from rich.console import Console
from rich.panel import Panel

# Import proper MCP client modules
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Add paths for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, 'src')

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_path not in sys.path:
    sys.path.insert(0, src_path)


console = Console()

# Add debug support
DEBUG = False

def debug_print(msg):
    """Print debug message if DEBUG is True."""
    if DEBUG:
        console.print(f"[dim][DEBUG] {msg}[/dim]")

async def main():
    """Main test function using standard MCP protocol"""
    console.print(Panel("[bold]Structure Search Test Script - MCP Protocol Version[/bold]"))
    
    # Test file
    test_file = os.path.join("tsap_example_data", "structure", "sample_code.py")
    
    # Verify file exists
    if not os.path.exists(test_file):
        console.print(f"[red]Test file not found: {test_file}[/red]")
        return
    
    console.print(f"[green]Found test file: {test_file}[/green]")
    
    # Print some content from the file for debugging
    with open(test_file, 'r') as f:
        content = f.read()
        console.print(f"File contains {len(content)} characters and {'Sample' in content} that 'Sample' is in it")
    
    # Path to the proxy script
    proxy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_proxy.py")
    debug_print(f"Proxy path: {proxy_path}")
    
    # Configure server parameters for stdio connection
    server_params = StdioServerParameters(
        command="python",
        args=[proxy_path],
        env={"MCP_DEBUG": "1" if DEBUG else "0"}  # Enable debug logging if needed
    )
    
    try:
        # Connect to the MCP server via proxy
        async with stdio_client(server_params) as (read, write):
            debug_print("Connected to MCP proxy")
            async with ClientSession(read, write) as session:
                # Initialize the connection
                init_result = await session.initialize()
                console.print(f"[green]Connected to {init_result.serverInfo.name} {init_result.serverInfo.version}[/green]")
                
                # Get the list of available tools
                tools_result = await session.list_tools()
                tools = tools_result.tools
                debug_print(f"Found {len(tools)} tools")
                
                # Debug output of available tools
                if DEBUG:
                    debug_print("Available tools:")
                    for tool in tools:
                        debug_print(f"  - {tool.name}: {tool.description}")
                
                # Find the structure search and analysis tools
                search_structure_tool = next((t for t in tools if t.name == "structure_search"), None)
                analyze_structure_tool = next((t for t in tools if t.name == "structure_analyze"), None)
                
                if not search_structure_tool:
                    console.print("[red]Error: structure_search tool not found![/red]")
                    return
                    
                if not analyze_structure_tool:
                    console.print("[yellow]Warning: structure_analyze tool not found. Test 3 will be skipped.[/yellow]")
                
                # Test 1: Basic structure search
                console.print("\n[bold]Test 1: Basic structure search with MCP protocol[/bold]")
                
                try:
                    # Call the structure search tool
                    result1 = await session.call_tool(search_structure_tool.name, arguments={
                        "action": "search",
                        "structure_pattern": "Sample",
                        "paths": [test_file]
                    })
                    
                    # Extract the text content
                    response_text1 = None
                    for content in result1.content:
                        if content.type == "text":
                            response_text1 = content.text
                            break
                    
                    if response_text1:
                        try:
                            response1 = json.loads(response_text1)
                            debug_print(f"Result 1: {response1}")
                            
                            # Navigate through response structure to find matches
                            # Sometimes it's in response['result']['matches']
                            # Sometimes it's in response['data']['result']['matches']
                            matches1 = None
                            
                            if "result" in response1 and "matches" in response1["result"]:
                                matches1 = response1["result"]["matches"]
                            elif "data" in response1 and "result" in response1["data"] and "matches" in response1["data"]["result"]:
                                matches1 = response1["data"]["result"]["matches"]
                            
                            # Display matches if found
                            if matches1:
                                console.print(f"[green]Found {len(matches1)} matches[/green]")
                                
                                # Display first 3 matches if available
                                for i, match in enumerate(matches1[:3]):
                                    console.print(f"Match {i+1}: {match.get('match_text', 'No match text')} - Type: {match.get('element_type', 'Unknown type')}")
                            else:
                                console.print("[yellow]No matches found in result[/yellow]")
                                
                        except json.JSONDecodeError:
                            console.print("[red]Failed to parse response as JSON[/red]")
                            console.print(f"Raw response: {response_text1[:200]}...")
                    else:
                        console.print("[red]No text content in response[/red]")
                
                except Exception as e:
                    console.print(f"[red]Error in Test 1: {e}[/red]")
                
                # Test 2: Search for specific element types
                console.print("\n[bold]Test 2: Search for specific element types with MCP protocol[/bold]")
                
                try:
                    # Call the structure search tool with element_types parameter
                    result2 = await session.call_tool(search_structure_tool.name, arguments={
                        "action": "search",
                        "structure_pattern": "Sample",
                        "paths": [test_file],
                        "structure_type": "function_def"  # Using function_def which is a valid ElementType value
                    })
                    
                    # Extract the text content
                    response_text2 = None
                    for content in result2.content:
                        if content.type == "text":
                            response_text2 = content.text
                            break
                    
                    if response_text2:
                        try:
                            response2 = json.loads(response_text2)
                            debug_print(f"Result 2: {response2}")
                            
                            # Navigate through response structure to find matches
                            matches2 = None
                            
                            if "result" in response2 and "matches" in response2["result"]:
                                matches2 = response2["result"]["matches"]
                            elif "data" in response2 and "result" in response2["data"] and "matches" in response2["data"]["result"]:
                                matches2 = response2["data"]["result"]["matches"]
                            
                            # Display matches if found
                            if matches2:
                                console.print(f"[green]Found {len(matches2)} function matches[/green]")
                                
                                # Display first 3 matches if available
                                for i, match in enumerate(matches2[:3]):
                                    console.print(f"Function match {i+1}: {match.get('match_text', 'No match text')} - Type: {match.get('element_type', 'Unknown type')}")
                            else:
                                console.print("[yellow]No function matches found in result[/yellow]")
                                
                        except json.JSONDecodeError:
                            console.print("[red]Failed to parse response as JSON[/red]")
                            console.print(f"Raw response: {response_text2[:200]}...")
                    else:
                        console.print("[red]No text content in response[/red]")
                
                except Exception as e:
                    console.print(f"[red]Error in Test 2: {e}[/red]")
                
                # Test 3: Proper structure search instead of invalid search_by_structure
                console.print("\n[bold]Test 3: Structure analysis through search[/bold]")
                
                try:
                    # Call the structure search tool with correct parameters
                    result3 = await session.call_tool(search_structure_tool.name, arguments={
                        "action": "search",
                        "structure_pattern": "class",  # Look for classes instead
                        "paths": [test_file],
                        "include_content": True,
                        "max_matches": 10
                    })
                    
                    # Extract the text content
                    response_text3 = None
                    for content in result3.content:
                        if content.type == "text":
                            response_text3 = content.text
                            break
                    
                    if response_text3:
                        try:
                            response3 = json.loads(response_text3)
                            debug_print(f"Result 3: {response3}")
                            
                            # Navigate through response structure to find matches
                            matches3 = None
                            
                            if "result" in response3 and "matches" in response3["result"]:
                                matches3 = response3["result"]["matches"]
                            elif "data" in response3 and "result" in response3["data"] and "matches" in response3["data"]["result"]:
                                matches3 = response3["data"]["result"]["matches"]
                            
                            # Display matches if found
                            if matches3:
                                console.print(f"[green]Found {len(matches3)} structural elements[/green]")
                                
                                # Display structural element types
                                element_types = {}
                                for match in matches3:
                                    element_type = match.get('element_type', 'unknown')
                                    element_types[element_type] = element_types.get(element_type, 0) + 1
                                
                                console.print("Element types found:")
                                for element_type, count in element_types.items():
                                    console.print(f"  - {element_type}: {count}")
                                
                                # Display first 3 matches if available
                                console.print("\nSample matches:")
                                for i, match in enumerate(matches3[:3]):
                                    element_type = match.get('element_type', 'Unknown type')
                                    match_text = match.get('match_text', 'No match text')
                                    content_snippet = match.get('element_content', 'No content')[:100] + "..." if match.get('element_content') else 'No content'
                                    console.print(f"Match {i+1}: {match_text} - Type: {element_type}")
                                    console.print(f"  Content: {content_snippet}")
                            else:
                                console.print("[yellow]No structural elements found[/yellow]")
                                
                        except json.JSONDecodeError:
                            console.print("[red]Failed to parse response as JSON[/red]")
                            console.print(f"Raw response: {response_text3[:200]}...")
                    else:
                        console.print("[red]No text content in response[/red]")
                
                except Exception as e:
                    console.print(f"[red]Error in Test 3: {e}[/red]")
    
    except Exception as e:
        console.print(f"[red]Error connecting to MCP server: {str(e)}[/red]")
        import traceback
        console.print(traceback.format_exc())

if __name__ == "__main__":
    # Check for debug flag
    if "--debug" in sys.argv:
        DEBUG = True
        debug_print("Debug mode enabled")
        
    asyncio.run(main()) 