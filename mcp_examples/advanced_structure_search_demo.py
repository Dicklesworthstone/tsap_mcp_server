#!/usr/bin/env python3
"""
Advanced Structure Search Demo (MCP Tools Version)

This script demonstrates the comprehensive features of the Structure Search
composite tool in TSAP MCP, which enables context-aware searching by understanding
the structural elements in different document types.
"""
import asyncio
import sys
import os
import json
from pathlib import Path
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich.rule import Rule
from rich.box import ROUNDED
from rich.tree import Tree
from typing import List

# Import proper MCP client modules
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# --- Path Setup --- #
# Add the project root and src directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go up one level from mcp_examples/
src_path = os.path.join(project_root, 'src')

# Add project root if not already present
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add src path if not already present
if src_path not in sys.path:
    sys.path.insert(0, src_path)  # Prepend src directory

# Add examples path itself
if script_dir not in sys.path:
    sys.path.insert(1, script_dir)
# --- End Path Setup ---


console = Console()

DEBUG = False

def debug_print(msg):
    """Print debug message if DEBUG is True."""
    if DEBUG:
        console.print(f"[dim][DEBUG] {msg}[/dim]")

async def structure_search_demo():
    """Demonstrate Structure Search's advanced features using MCP tools."""
    console.print(Panel.fit(
        "[bold blue]TSAP MCP Structure Search Advanced Features Demo[/bold blue]",
        subtitle="Context-aware searching based on document structure using MCP tools"
    ))

    # Define file paths (relative to workspace root)
    base_data_dir = Path("tsap_example_data/structure/")
    sample_files = [
        base_data_dir / "sample_code.py",
        base_data_dir / "sample_markdown.md",
        base_data_dir / "sample_html.html",
    ]

    # Check existence of essential files
    missing_files = [f for f in sample_files if not f.exists()]
    if missing_files:
        console.print(f"[bold red]Error:[/bold red] Required example files not found: {', '.join(map(str, missing_files))}")
        console.print(f"Please ensure the '{base_data_dir}' directory contains the required files.")
        return

    # Path to the proxy script
    proxy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_proxy.py")
    debug_print(f"Proxy path: {proxy_path}")
    
    # Configure server parameters for stdio connection to our proxy
    server_params = StdioServerParameters(
        command="python",
        args=[proxy_path],
        env={"MCP_DEBUG": "1" if DEBUG else "0"}  # Enable debug logging if needed
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
                
                # Find required tools
                search_structure_tool = next((t for t in tools if t.name == "search_structure"), None)
                analyze_structure_tool = next((t for t in tools if t.name == "analyze_structure"), None)
                info_tool = next((t for t in tools if t.name == "info"), None)
                
                if not search_structure_tool or not analyze_structure_tool:
                    console.print("[bold red]Error: Required structure search tools not found![/bold red]")
                    return

                # --- Add initial info check ---
                if info_tool:
                    console.print("Checking server info...")
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
                                console.print("[green]Initial info check successful.[/green]")
                            except json.JSONDecodeError:
                                console.print("[yellow]Info response is not JSON[/yellow]")
                        else:
                            console.print("[yellow]Info call succeeded but returned no readable content.[/yellow]")
                    except Exception as e:
                        console.print(f"[bold red]Error during info check:[/bold red] {str(e)}")
                        return
                # ------------------------------

                # First, visualize the structure of one of the example files
                console.print(Rule("[bold yellow]Document Structure Visualization[/bold yellow]"))
                console.print("[italic]Visualizing document structure helps understand search targets.[/italic]\n")
                
                # Take first file that exists
                for sample_file in sample_files:
                    if sample_file.exists():
                        await visualize_document_structure(session, analyze_structure_tool, str(sample_file))
                        break

                # Let's limit our demos for debugging
                if DEBUG:
                    debug_print("DEBUG mode: Only running visualization demo")
                    return

                # Run the basic search demo
                console.print(Rule("[bold yellow]Basic Structure Search Demo[/bold yellow]"))
                console.print("[italic]Searching for content with basic structure awareness.[/italic]\n")
                
                await run_basic_search_demo(session, search_structure_tool, sample_files)

                # Run the element-specific search demo
                console.print(Rule("[bold yellow]Element-Specific Structure Search Demo[/bold yellow]"))
                console.print("[italic]Searching for content within specific structural elements.[/italic]\n")
                
                await run_element_specific_search_demo(session, search_structure_tool, sample_files)

                # Run the context-aware search demo
                console.print(Rule("[bold yellow]Context-Aware Structure Search Demo[/bold yellow]"))
                console.print("[italic]Searching for content with awareness of surrounding context.[/italic]\n")
                
                await run_context_aware_search_demo(session, search_structure_tool, sample_files)

    except Exception as e:
        console.print(f"[bold red]Error running demo: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())

async def run_basic_search_demo(session, search_structure_tool, sample_files: List[Path]):
    """Run the basic structure search demo with example files."""
    console.print("[bold cyan]Basic Structure Search[/bold cyan]")
    console.print("This demonstrates simple structure-aware searches across different file types.\n")
    
    # Example 1: Search for function definitions
    console.print("[bold]Example 1: Search for function definitions[/bold]")
    
    try:
        # Use the search_structure tool
        result = await session.call_tool(
            search_structure_tool.name,
            arguments={
                "query": "function",
                "files": [str(f) for f in sample_files if f.exists()],
                "element_types": ["function_declaration", "function"]
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
                results = json.loads(result_text)
                display_search_results(results, "Function Definitions")
            except json.JSONDecodeError:
                console.print("[bold red]Failed to parse response as JSON[/bold red]")
                console.print(f"Raw response: {result_text[:200]}...")
        else:
            console.print("[bold red]No text content in response[/bold red]")
        
    except Exception as e:
        console.print(f"[bold red]Error during structure search: {e}[/bold red]")
    
    # Example 2: Search for list items with "example" keyword
    console.print("\n[bold]Example 2: Search for list items containing 'example'[/bold]")
    
    try:
        # Use the search_structure tool
        result = await session.call_tool(
            search_structure_tool.name,
            arguments={
                "query": "example",
                "files": [str(f) for f in sample_files if f.exists()],
                "element_types": ["list_item"]
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
                results = json.loads(result_text)
                display_search_results(results, "List Items with 'example'")
            except json.JSONDecodeError:
                console.print("[bold red]Failed to parse response as JSON[/bold red]")
                console.print(f"Raw response: {result_text[:200]}...")
        else:
            console.print("[bold red]No text content in response[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Error during structure search: {e}[/bold red]")
    
    # Example 3: Search for headings
    console.print("\n[bold]Example 3: Search for headings with 'Example' in the title[/bold]")
    
    try:
        # Use the search_structure tool
        result = await session.call_tool(
            search_structure_tool.name,
            arguments={
                "query": "Example",
                "files": [str(f) for f in sample_files if f.exists()],
                "element_types": ["heading"]
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
                results = json.loads(result_text)
                display_search_results(results, "Headings with 'Example'")
            except json.JSONDecodeError:
                console.print("[bold red]Failed to parse response as JSON[/bold red]")
                console.print(f"Raw response: {result_text[:200]}...")
        else:
            console.print("[bold red]No text content in response[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Error during structure search: {e}[/bold red]")

async def run_element_specific_search_demo(session, search_structure_tool, sample_files: List[Path]):
    """Run the element-specific structure search demo."""
    console.print("[bold cyan]Element-Specific Structure Search[/bold cyan]")
    console.print("This demonstrates searching for specific structural elements with precise targeting.\n")
    
    # Example 1: Search for class definitions with "data" in the name
    console.print("[bold]Example 1: Search for class definitions with 'data' in the name[/bold]")
    
    try:
        # Use the search_structure tool
        result = await session.call_tool(
            search_structure_tool.name,
            arguments={
                "query": "data",
                "files": [str(f) for f in sample_files if f.suffix == '.py' and f.exists()],
                "element_types": ["class_declaration"]
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
                results = json.loads(result_text)
                display_search_results(results, "Class Definitions with 'data'")
            except json.JSONDecodeError:
                console.print("[bold red]Failed to parse response as JSON[/bold red]")
                console.print(f"Raw response: {result_text[:200]}...")
        else:
            console.print("[bold red]No text content in response[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Error during structure search: {e}[/bold red]")
    
    # Example 2: Search for tables in HTML and Markdown
    console.print("\n[bold]Example 2: Search for tables[/bold]")
    
    try:
        # Use the search_structure tool
        result = await session.call_tool(
            search_structure_tool.name,
            arguments={
                "query": "",  # Empty query to match all tables
                "files": [str(f) for f in sample_files if f.suffix in ('.html', '.md') and f.exists()],
                "element_types": ["table"]
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
                results = json.loads(result_text)
                display_search_results(results, "Tables in HTML and Markdown")
            except json.JSONDecodeError:
                console.print("[bold red]Failed to parse response as JSON[/bold red]")
                console.print(f"Raw response: {result_text[:200]}...")
        else:
            console.print("[bold red]No text content in response[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Error during structure search: {e}[/bold red]")
    
    # Example 3: Search for code blocks with "print" statements
    console.print("\n[bold]Example 3: Search for code blocks with 'print' statements[/bold]")
    
    try:
        # Use the search_structure tool
        result = await session.call_tool(
            search_structure_tool.name,
            arguments={
                "query": "print",
                "files": [str(f) for f in sample_files if f.suffix in ('.md', '.html') and f.exists()],
                "element_types": ["code_block"]
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
                results = json.loads(result_text)
                display_search_results(results, "Code Blocks with 'print' statements")
            except json.JSONDecodeError:
                console.print("[bold red]Failed to parse response as JSON[/bold red]")
                console.print(f"Raw response: {result_text[:200]}...")
        else:
            console.print("[bold red]No text content in response[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Error during structure search: {e}[/bold red]")

async def run_context_aware_search_demo(session, search_structure_tool, sample_files: List[Path]):
    """Run the context-aware structure search demo."""
    console.print("[bold cyan]Context-Aware Structure Search[/bold cyan]")
    console.print("This demonstrates searches that are aware of their containing elements and surrounding context.\n")
    
    # Example 1: Search for comments within function definitions
    console.print("[bold]Example 1: Search for comments within function definitions[/bold]")
    
    try:
        # Use the search_structure tool with parent element type specified
        result = await session.call_tool(
            search_structure_tool.name,
            arguments={
                "query": "",  # Empty query to match all comments
                "files": [str(f) for f in sample_files if f.suffix == '.py' and f.exists()],
                "element_types": ["comment"],
                "parent_element_types": ["function_declaration"]
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
                results = json.loads(result_text)
                display_search_results(results, "Comments within Functions")
            except json.JSONDecodeError:
                console.print("[bold red]Failed to parse response as JSON[/bold red]")
                console.print(f"Raw response: {result_text[:200]}...")
        else:
            console.print("[bold red]No text content in response[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Error during structure search: {e}[/bold red]")
    
    # Example 2: Search for list items within a section about structure
    console.print("\n[bold]Example 2: Search for list items within a section about 'structure'[/bold]")
    
    try:
        # Use the search_structure tool with section context
        result = await session.call_tool(
            search_structure_tool.name,
            arguments={
                "query": "",  # Match all list items
                "files": [str(f) for f in sample_files if f.suffix in ('.md', '.html') and f.exists()],
                "element_types": ["list_item"],
                "context_query": "structure"  # Look for sections containing "structure"
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
                results = json.loads(result_text)
                display_search_results(results, "List Items in Sections about 'structure'")
            except json.JSONDecodeError:
                console.print("[bold red]Failed to parse response as JSON[/bold red]")
                console.print(f"Raw response: {result_text[:200]}...")
        else:
            console.print("[bold red]No text content in response[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Error during structure search: {e}[/bold red]")
    
    # Example 3: Search for specific type code examples
    console.print("\n[bold]Example 3: Search for Python code blocks within sections containing 'example'[/bold]")
    
    try:
        # Use the search_structure tool with complex context
        result = await session.call_tool(
            search_structure_tool.name,
            arguments={
                "query": "python",
                "files": [str(f) for f in sample_files if f.suffix in ('.md', '.html') and f.exists()],
                "element_types": ["code_block"],
                "context_query": "example"
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
                results = json.loads(result_text)
                display_search_results(results, "Python Code Blocks in 'example' Sections")
            except json.JSONDecodeError:
                console.print("[bold red]Failed to parse response as JSON[/bold red]")
                console.print(f"Raw response: {result_text[:200]}...")
        else:
            console.print("[bold red]No text content in response[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Error during structure search: {e}[/bold red]")

async def visualize_document_structure(session, analyze_structure_tool, file_path: str):
    """Visualize the document structure using structure analyzer."""
    console.print(f"[bold]Visualizing structure for:[/bold] {os.path.basename(file_path)}")
    
    try:
        # Use the analyze_structure tool
        result = await session.call_tool(
            analyze_structure_tool.name,
            arguments={
                "file": file_path
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
                structure = json.loads(result_text)
                
                if structure and "elements" in structure:
                    elements = structure["elements"]
                    
                    # Create a rich Tree to display the structure
                    root = Tree(f"[bold blue]{os.path.basename(file_path)}[/bold blue]")
                    
                    # Keep track of current path in tree
                    current_path = [root]
                    current_depth = 0
                    
                    # Output first 30 elements for demo purposes
                    for i, element in enumerate(elements[:30]):
                        element_type = element.get("type", "unknown")
                        content = element.get("content", "")
                        
                        # Truncate content for display
                        if content and len(content) > 50:
                            content = content[:47] + "..."
                        
                        # Show depth/nesting
                        depth = element.get("depth", 0)
                        
                        # Adjust tree based on depth
                        if depth > current_depth:
                            # Going deeper, use last node as new parent
                            current_depth = depth
                        elif depth < current_depth:
                            # Going up, pop from path stack
                            for _ in range(current_depth - depth):
                                if len(current_path) > 1:  # Always keep root
                                    current_path.pop()
                            current_depth = depth
                        
                        # Create node text based on element type and content
                        element_display = f"[yellow]{element_type}[/yellow]"
                        if content:
                            element_display += f": [dim]\"{content}\"[/dim]"
                        
                        # Add element to the tree
                        current_path[-1].add(element_display)
                    
                    # Display the tree
                    console.print(root)
                    
                    if len(elements) > 30:
                        console.print(f"[dim]... and {len(elements) - 30} more elements not shown.[/dim]")
                else:
                    console.print("[yellow]No structure analysis results returned.[/yellow]")
            except json.JSONDecodeError:
                console.print("[bold red]Failed to parse response as JSON[/bold red]")
                console.print(f"Raw response: {result_text[:200]}...")
        else:
            console.print("[bold red]No text content in response[/bold red]")
            
    except Exception as e:
        console.print(f"[bold red]Error during structure analysis: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())

def display_search_results(results, title: str):
    """Display structure search results in a formatted table."""
    if not results or "matches" not in results:
        console.print(f"[yellow]No results found for {title}[/yellow]")
        return
        
    matches = results["matches"]
    if not matches:
        console.print(f"[yellow]No matches found for {title}[/yellow]")
        return
    
    # Create result table
    result_table = Table(title=f"{title} ({len(matches)} matches)", box=ROUNDED)
    result_table.add_column("File", style="cyan")
    result_table.add_column("Element Type", style="green")
    result_table.add_column("Content", style="white")
    result_table.add_column("Location", style="yellow")
    
    # Add matches to table (limit to 10 for display)
    for match in matches[:10]:
        # Extract fields
        file_path = match.get("file", "")
        element_type = match.get("element_type", "")
        content = match.get("content", "")
        location = match.get("location", {})
        
        # Format filename for display
        file_name = os.path.basename(file_path)
        
        # Format location for display
        location_str = ""
        if location:
            if "line" in location:
                location_str = f"Line {location['line']}"
                if "column" in location:
                    location_str += f", Col {location['column']}"
        
        # Truncate content for display
        if content and len(content) > 50:
            content = content[:47] + "..."
        
        # Add row to table
        result_table.add_row(file_name, element_type, content, location_str)
    
    # Show indication if results were truncated
    if len(matches) > 10:
        result_table.caption = f"... and {len(matches) - 10} more matches not shown"
    
    # Display the table
    console.print(result_table)

if __name__ == "__main__":
    if "--debug" in sys.argv:
        DEBUG = True
        debug_print("Debug mode enabled")
    
    try:
        asyncio.run(structure_search_demo())
    except KeyboardInterrupt:
        console.print("[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Unhandled error: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc()) 