#!/usr/bin/env python3
"""
Simple test script for structure search
"""
import asyncio
import os
import sys
from rich.console import Console

# Import the MCP client
from tsap.mcp import MCPClient
# Add paths for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, 'src')

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_path not in sys.path:
    sys.path.insert(0, src_path)


console = Console()

async def main():
    """Main test function"""
    console.print("[bold]Structure Search Test Script[/bold]")
    
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
    
    # Create client
    async with MCPClient() as client:
        # Basic test using the original parameters as in the client
        console.print("\n[bold]Test 1: Using client convenience method[/bold]")
        result1 = await client.structure_search(
            search_term="Sample",
            file_paths=[test_file]
        )
        console.print(f"Result 1: {result1}")
        
        # Test using the direct send_request with proper parameters
        console.print("\n[bold]Test 2: Using direct request with proper params[/bold]")
        result2 = await client.send_request(
            "structure_search",
            {
                "action": "search",
                "structure_pattern": "Sample",
                "paths": [test_file],
                "structure_type": "function_def"  # Using function_def which is a valid ElementType value
            }
        )
        console.print(f"Result 2: {result2}")
        
        # Test 3: Direct use of search_by_structure
        console.print("\n[bold]Test 3: Testing search_by_structure wrapper[/bold]")
        result3 = await client.send_request(
            "structure_search",
            {
                "action": "search_by_structure",
                "pattern": "Sample",
                "files": [test_file]
            }
        )
        console.print(f"Result 3: {result3}")

if __name__ == "__main__":
    asyncio.run(main()) 