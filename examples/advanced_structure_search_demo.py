#!/usr/bin/env python3
"""
Advanced Structure Search Demo

This script demonstrates the comprehensive features of the Structure Search
composite tool in TSAP, which enables context-aware searching by understanding
the structural elements in different document types.
"""
import asyncio
import sys
import os
from pathlib import Path
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich.rule import Rule
from rich.box import ROUNDED
from rich.progress import Progress
from rich.tree import Tree
from typing import List

# Import the MCP client from the library
from tsap.mcp import MCPClient

# --- Path Setup --- #
# Add the project root and src directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go up one level from examples/
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

def safe_extract(response):
    """Safely extract data from an MCP response without risking None returns.
    
    Args:
        response: The API response dictionary
        
    Returns:
        Dictionary with success flag, result data, and error info
    """
    if response is None:
        return {"success": False, "error": "No response received"}
    
    if not isinstance(response, dict):
        return {"success": False, "error": f"Invalid response type: {type(response)}"}
    
    # Check top-level status
    if response.get("status") != "success":
        return {"success": False, "error": response.get("error", "Unknown error")}
    
    # Check data section
    data = response.get("data")
    if not data:
        return {"success": False, "error": "No data in response"}
    
    # Check data status
    if isinstance(data, dict) and data.get("status") != "success":
        return {"success": False, "error": data.get("error", "Unknown error in data")}
    
    return {"success": True, "result": data.get("result"), "error": None}

async def structure_search_demo():
    """Demonstrate Structure Search's advanced features."""
    console.print(Panel.fit(
        "[bold blue]TSAP Structure Search Advanced Features Demo[/bold blue]",
        subtitle="Context-aware searching based on document structure"
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
        console.print(f"Please ensure the '{base_data_dir}' directory contains the required files. Creating sample files for demo purposes...")
        
        # Create sample directory and files for demo if they don't exist
        os.makedirs(base_data_dir, exist_ok=True)
        
        # Sample Python code file
        sample_code_content = """#!/usr/bin/env python3
'''
Sample Python Module
This module demonstrates various structural elements that can be searched.
'''
import os
import sys
import re
from typing import List, Dict, Any, Optional

# Global variable
VERSION = "1.0.0"

# Global constants
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3

class DataProcessor:
    '''
    A class for processing data with various methods.
    This class demonstrates structural elements within a class.
    '''
    
    def __init__(self, config: Dict[str, Any] = None):
        '''Initialize the data processor with optional config.'''
        self.config = config or {}
        self.initialized = True
        self.data = []
    
    def process_data(self, input_data: List[Any]) -> List[Any]:
        '''
        Process the input data and return results.
        
        Args:
            input_data: The data to process
            
        Returns:
            Processed data
        '''
        results = []
        
        # Process each item
        for item in input_data:
            # Skip invalid items
            if not item:
                continue
                
            # Transform the item
            processed = self._transform_item(item)
            results.append(processed)
            
        return results
    
    def _transform_item(self, item: Any) -> Any:
        '''Internal method to transform an item.'''
        # Apply transformations based on config
        if 'uppercase' in self.config and self.config['uppercase']:
            if isinstance(item, str):
                return item.upper()
        
        return item

def main():
    '''Main function to demonstrate the module.'''
    processor = DataProcessor({'uppercase': True})
    
    # Sample data
    data = ["hello", "world", None, "example"]
    
    # Process the data
    results = processor.process_data(data)
    
    # Print results
    print(f"Processed {len(results)} items:")
    for item in results:
        print(f"- {item}")

if __name__ == "__main__":
    main()
"""
        
        # Sample Markdown file
        sample_markdown_content = """# Sample Markdown Document

## Introduction

This is a sample Markdown document that contains various structural elements
that can be searched and analyzed by the structure search tool.

### Purpose

The purpose of this document is to:

1. Demonstrate headings of different levels
2. Show list structures (like this one)
3. Include code blocks
4. Demonstrate tables

## Code Examples

Here's a simple Python code example:

```python
def hello_world():
    '''Print a greeting message'''
    print("Hello, world!")
    
hello_world()
```

And here's some JavaScript:

```javascript
function calculateSum(a, b) {
    // Return the sum of two numbers
    return a + b;
}

console.log(calculateSum(5, 3));  // Output: 8
```

## Data Section

### Tables

Here's a sample table:

| Name | Age | Occupation |
|------|-----|------------|
| John | 32  | Developer  |
| Mary | 28  | Designer   |
| Sam  | 45  | Manager    |

### Quotes

> This is a blockquote that could be matched by structure search.
> It continues across multiple lines.

## Conclusion

This document has demonstrated various Markdown structural elements including:

* Headings (h1, h2, h3)
* Ordered and unordered lists
* Code blocks with syntax highlighting
* Tables
* Blockquotes

These elements can all be identified and searched with the structure search tool.
"""
        
        # Sample HTML file
        sample_html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Structure Search Example</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3 {
            color: #333;
        }
        .section {
            margin-bottom: 30px;
            border-bottom: 1px solid #eee;
            padding-bottom: 20px;
        }
        code {
            background-color: #f4f4f4;
            padding: 2px 5px;
            border-radius: 3px;
        }
        pre {
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }
        table {
            border-collapse: collapse;
            width: 100%;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f4f4f4;
        }
        .note {
            background-color: #ffffd9;
            padding: 10px;
            border-left: 4px solid #e6db55;
        }
    </style>
</head>
<body>
    <header>
        <h1>Structure Search HTML Example</h1>
        <p>This HTML document demonstrates various structural elements that can be searched.</p>
    </header>
    
    <div class="section">
        <h2>Introduction</h2>
        <p>The structure search tool can identify and search through different HTML elements:</p>
        <ul>
            <li>Headings (h1, h2, h3, etc.)</li>
            <li>Paragraphs</li>
            <li>Lists (ordered and unordered)</li>
            <li>Tables</li>
            <li>Code blocks</li>
            <li>Sections and divs</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>Code Examples</h2>
        <h3>Python Example</h3>
        <pre><code>
def fibonacci(n):
    // Generate the Fibonacci sequence up to n
    a, b = 0, 1
    result = []
    while a < n:
        result.append(a)
        a, b = b, a + b
    return result

print(fibonacci(100))
        </code></pre>
        
        <h3>JavaScript Example</h3>
        <pre><code>
function sortNumbers(array) {
    // Sort an array of numbers in ascending order
    return array.sort((a, b) => a - b);
}

const numbers = [5, 2, 9, 1, 5, 6];
console.log(sortNumbers(numbers));  // Output: [1, 2, 5, 5, 6, 9]
        </code></pre>
    </div>
    
    <div class="section">
        <h2>Tables</h2>
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Name</th>
                    <th>Category</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>1</td>
                    <td>Item One</td>
                    <td>Category A</td>
                    <td>100</td>
                </tr>
                <tr>
                    <td>2</td>
                    <td>Item Two</td>
                    <td>Category B</td>
                    <td>75</td>
                </tr>
                <tr>
                    <td>3</td>
                    <td>Item Three</td>
                    <td>Category A</td>
                    <td>50</td>
                </tr>
            </tbody>
        </table>
    </div>
    
    <div class="section">
        <h2>Notes and Comments</h2>
        <div class="note">
            <p>This is a special note section that can be targeted by structure search.</p>
            <p>You can search for elements with specific classes or nested within certain parent elements.</p>
        </div>
        <!-- This is an HTML comment that could be matched by structure search -->
    </div>
    
    <footer>
        <p>This example document was created for the TSAP Structure Search demonstration.</p>
    </footer>
</body>
</html>
"""

        # Write the sample files
        with open(sample_files[0], 'w') as f:
            f.write(sample_code_content)
        with open(sample_files[1], 'w') as f:
            f.write(sample_markdown_content)
        with open(sample_files[2], 'w') as f:
            f.write(sample_html_content)
        
        console.print("[green]Sample files created successfully.[/green]")

    # Create our client
    try:
        debug_print("Creating MCPClient...")
        async with MCPClient() as client:
            debug_print("Running demos...")

            # Initial server info check
            console.print(f"Attempting to get server info from {client.base_url}...")
            try:
                info = await client.info()
                if info.get("status") != "success" or info.get("error") is not None:
                    error_message = info.get('error', "Status was not success")
                    console.print(f"[bold red]Error during initial client.info() check:[/bold red] {error_message}")
                    return
                else:
                    console.print("[green]Initial client.info() check successful.[/green]")
            except Exception as e:
                console.print(f"[bold red]Error during initial client.info() check:[/bold red] {e}")
                import traceback
                console.print(traceback.format_exc())
                return
                
            # Check if structure_search is supported
            test_response = await client.send_request(
                "structure_search",
                {
                    "action": "search",
                    "structure_pattern": "test",
                    "paths": [str(sample_files[0])],
                    "structure_type": "document"
                }
            )
            
            # Use our safe extract function
            test_extracted = safe_extract(test_response)
            
            # Check if command is not supported
            error_code = ""
            if not test_extracted.get("success", False) and test_extracted.get("error"):
                error = test_extracted.get("error")
                if isinstance(error, dict):
                    error_code = error.get("code", "")
                    
            if error_code == "unknown_command":
                console.print(Rule("[bold red]Feature Not Available[/bold red]"))
                console.print("[italic yellow]The structure_search command is not currently implemented on the server.[/italic yellow]")
                console.print()
                console.print("[bold]This demo requires the structure_search feature, which appears to be unavailable.[/bold]")
                console.print("The structure_search feature provides context-aware searching capabilities by understanding document structure.")
                console.print()
                console.print("When implemented, this feature will allow:")
                console.print("1. Searching within specific structural elements (functions, classes, headings, etc.)")
                console.print("2. Filtering searches based on element types")
                console.print("3. Understanding document hierarchies for more precise results")
                console.print()
                console.print("[dim]To implement this feature, the server would need to add a handler for the 'structure_search' command.[/dim]")
                return

            # DEMO 1: Basic Structure Search
            console.print(Rule("[bold yellow]Demo 1: Basic Structure Search[/bold yellow]"))
            console.print("[italic]Search for text across multiple files with structural context.[/italic]")
            console.print()
            await run_basic_search_demo(
                client,
                sample_files
            )

            # DEMO 2: Element-Specific Searching
            console.print(Rule("[bold yellow]Demo 2: Element-Specific Searching[/bold yellow]"))
            console.print("[italic]Search within specific structural elements like classes, functions, or headings.[/italic]")
            console.print()
            await run_element_specific_search_demo(
                client,
                sample_files
            )

            # DEMO 3: Context-Aware Searching
            console.print(Rule("[bold yellow]Demo 3: Context-Aware Searching[/bold yellow]"))
            console.print("[italic]Search with awareness of element hierarchy and relationships.[/italic]")
            console.print()
            await run_context_aware_search_demo(
                client,
                sample_files
            )

            console.print(Rule("[bold green]Structure Search Demo Complete[/bold green]"))

    except Exception as e:
        console.print(f"[bold red]Error during demo:[/bold red] {str(e)}")
        import traceback
        console.print(traceback.format_exc())

async def run_basic_search_demo(
    client: MCPClient,
    sample_files: List[Path],
    console: Console = console
):
    """Run a basic structure search demo."""
    console.print("Performing basic structure search across sample files")
    
    # Convert file paths to strings
    file_paths = [str(f) for f in sample_files]
    console.print(f"[dim]Searching in files: {', '.join(file_paths)}[/dim]")

    # Basic search term - use a term that definitely appears in the files
    search_term = "Sample"
    
    # Check if the term exists in the files (for debugging)
    for file_path in file_paths:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                if search_term in content:
                    console.print(f"[dim]Found '{search_term}' in {file_path}[/dim]")
                else:
                    console.print(f"[dim]Term '{search_term}' NOT found in {file_path}[/dim]")
        except Exception as e:
            console.print(f"[dim]Error reading {file_path}: {str(e)}[/dim]")
    
    with Progress() as progress:
        task = progress.add_task("[green]Searching...", total=1)
        
        # Show the exact request we're making
        console.print(f"[dim]Making request with: structure_pattern='{search_term}', structure_type='document'[/dim]")
        
        # Use send_request_and_extract like in patterns demo
        response = await client.send_request(
            "structure_search",
            {
                "action": "search",
                "structure_pattern": search_term,
                "paths": file_paths,
                "structure_type": "document"  # Use document type instead of None to search the entire document
            }
        )
        
        # Print raw response for debugging
        console.print(f"[dim]Raw response: {response}[/dim]")
        
        # Extract the result properly
        extracted = safe_extract(response)
        
        progress.update(task, advance=1)
    
    if extracted is None:
        # Handle extraction failure
        if isinstance(response, dict) and response.get("status") == "success":
            # Response contains data but extraction failed
            console.print("[yellow]Warning: Response extraction issue, but continuing with raw data.[/yellow]")
            # Try to get the result from the raw response
            if "data" in response and "result" in response["data"]:
                search_results = response["data"]["result"]
            else:
                console.print("[bold red]Error: Could not extract search results from response[/bold red]")
                return
        else:
            console.print("[bold red]Error during structure search: Extract result failed[/bold red]")
            return
    elif not extracted.get("success", False):
        console.print(f"[bold red]Error during structure search:[/bold red] {extracted.get('error', 'Unknown error')}")
        return
    else:
        search_results = extracted.get("result", {})
    
    # Handle the case where search_results might still be None
    if not search_results:
        search_results = {}
    
    console.print("Found matches in search results")
    display_search_results(search_results, f"Basic Search for '{search_term}'")

async def run_element_specific_search_demo(client, sample_files):
    """Run a search demo targeting specific structural elements."""
    console.print("Searching within specific structural elements")
    
    # Convert file paths to strings
    file_paths = [str(f) for f in sample_files]
    
    # Define searches to demonstrate
    element_searches = [
        {
            "title": "Functions and Methods",
            "pattern": "main",
            "element_type": "FUNCTION_DEF",
            "description": "Find 'main' in functions/methods"
        },
        {
            "title": "Class Definitions",
            "pattern": "DataProcessor",
            "element_type": "CLASS_DEF",
            "description": "Find 'DataProcessor' in class definitions"
        },
        {
            "title": "Markdown Headings",
            "pattern": "Sample",
            "element_type": "HEADING",
            "description": "Find 'Sample' in headings"
        }
    ]
    
    # Perform each search
    for search in element_searches:
        console.print(f"\n[bold]Searching for: {search['description']}[/bold]")
        
        with Progress() as progress:
            task = progress.add_task(f"[green]Searching for {search['pattern']} in {search['element_type']}...", total=1)
            
            # Use send_request instead of directly calling structure_search
            response = await client.send_request(
                "structure_search",
                {
                    "action": "search",
                    "structure_pattern": search["pattern"],
                    "paths": file_paths,
                    "structure_type": search["element_type"]
                }
            )
            
            # Extract the result properly
            extracted = safe_extract(response)
            
            progress.update(task, advance=1)
        
        if extracted is None:
            # Handle extraction failure
            if isinstance(response, dict) and response.get("status") == "success":
                # Response contains data but extraction failed
                console.print("[yellow]Warning: Response extraction issue, but continuing with raw data.[/yellow]")
                # Try to get the result from the raw response
                if "data" in response and "result" in response["data"]:
                    search_results = response["data"]["result"]
                else:
                    console.print(f"[bold red]Error: Could not extract search results for {search['title']}[/bold red]")
                    continue
            else:
                console.print("[bold red]Error during element-specific search: Extract result failed[/bold red]")
                continue
        elif not extracted.get("success", False):
            console.print(f"[bold red]Error during element-specific search:[/bold red] {extracted.get('error', 'Unknown error')}")
            continue
        else:
            search_results = extracted.get("result", {})
        
        # Handle the case where search_results might still be None
        if not search_results:
            search_results = {}
            
        display_search_results(search_results, f"{search['title']} Search")

async def run_context_aware_search_demo(client, sample_files):
    """Run a search demo showing context-aware capabilities."""
    console.print("Demonstrating context-aware structure searching")
    
    # Convert file paths to strings
    file_paths = [str(f) for f in sample_files]
    
    # Define a context-aware search
    console.print("\n[bold]Searching for patterns with specific parent elements:[/bold]")
    
    with Progress() as progress:
        task = progress.add_task("[green]Performing context-aware search...", total=1)
        
        # Use send_request instead of directly calling structure_search
        response = await client.send_request(
            "structure_search",
            {
                "action": "search",
                "structure_pattern": "import",
                "paths": file_paths,
                "structure_type": "IMPORT_STMT",  # Changed from IMPORT_STATEMENT to match valid types
                "parent_elements": ["DOCUMENT"]
            }
        )
        
        # Extract the result properly
        extracted = safe_extract(response)
        
        progress.update(task, advance=1)
    
    if extracted is None:
        # Handle extraction failure
        if isinstance(response, dict) and response.get("status") == "success":
            # Response contains data but extraction failed
            console.print("[yellow]Warning: Response extraction issue, but continuing with raw data.[/yellow]")
            # Try to get the result from the raw response
            if "data" in response and "result" in response["data"]:
                search_results = response["data"]["result"]
            else:
                console.print("[bold red]Error: Could not extract search results from response[/bold red]")
                return
        else:
            console.print("[bold red]Error during context-aware search: Extract result failed[/bold red]")
            return
    elif not extracted.get("success", False):
        console.print(f"[bold red]Error during context-aware search:[/bold red] {extracted.get('error', 'Unknown error')}")
        return
    else:
        search_results = extracted.get("result", {})
    
    # Handle the case where search_results might still be None
    if not search_results:
        search_results = {}
        
    display_search_results(
        search_results, 
        "Context-Aware Search: 'import' statements at the document level"
    )

async def visualize_document_structure(client, file_path):
    """Visualize the structural hierarchy of a document."""
    console.print(f"Analyzing structural elements in {os.path.basename(file_path)}")
    
    # We would typically call a specialized endpoint for this
    # For demo purposes, we'll simulate it with a simplified structure
    
    # Create a tree visualization
    file_name = os.path.basename(file_path)
    tree = Tree(f"[bold blue]📄 {file_name}[/bold blue]")
    
    # Check file type to determine how to display structure
    _, ext = os.path.splitext(str(file_path))
    
    if ext == '.py':
        # Python structure
        module_node = tree.add("[bold cyan]Module[/bold cyan]")
        
        # Add classes
        class_node = module_node.add("[bold green]Class: DataProcessor[/bold green]")
        class_node.add("[yellow]Method: __init__[/yellow]")
        class_node.add("[yellow]Method: process_data[/yellow]")
        class_node.add("[yellow]Method: _transform_item[/yellow]")
        
        # Add functions
        module_node.add("[bold magenta]Function: main[/bold magenta]")
        
    elif ext == '.md':
        # Markdown structure
        h1_node = tree.add("[bold cyan]Heading 1: Sample Markdown Document[/bold cyan]")
        
        # Introduction section
        h2_intro = h1_node.add("[bold green]Heading 2: Introduction[/bold green]")
        h3_purpose = h2_intro.add("[green]Heading 3: Purpose[/green]")
        h3_purpose.add("[yellow]List (ordered)[/yellow]")
        
        # Code examples section
        h2_code = h1_node.add("[bold green]Heading 2: Code Examples[/bold green]")
        h2_code.add("[yellow]Code Block: Python[/yellow]")
        h2_code.add("[yellow]Code Block: JavaScript[/yellow]")
        
        # Data section
        h2_data = h1_node.add("[bold green]Heading 2: Data Section[/bold green]")
        h3_tables = h2_data.add("[green]Heading 3: Tables[/green]")
        h3_tables.add("[yellow]Table[/yellow]")
        h3_quotes = h2_data.add("[green]Heading 3: Quotes[/green]")
        h3_quotes.add("[yellow]Blockquote[/yellow]")
        
        # Conclusion
        h1_node.add("[bold green]Heading 2: Conclusion[/bold green]")
        
    elif ext in ['.html', '.htm']:
        # HTML structure
        html_node = tree.add("[bold cyan]html[/bold cyan]")
        
        head = html_node.add("[bold green]head[/bold green]")
        head.add("[yellow]title: Structure Search Example[/yellow]")
        head.add("[yellow]style[/yellow]")
        
        body = html_node.add("[bold green]body[/bold green]")
        
        header = body.add("[green]header[/green]")
        header.add("[yellow]h1: Structure Search HTML Example[/yellow]")
        
        sections = body.add("[green]div.section (4 sections)[/green]")
        sections.add("[yellow]h2 headings (4)[/yellow]")
        sections.add("[yellow]h3 headings (2)[/yellow]")
        sections.add("[yellow]tables (1)[/yellow]")
        
        footer = body.add("[green]footer[/green]")  # noqa: F841
    
    console.print(tree)

def display_search_results(results, title):
    """Display structure search results in a formatted way."""
    # Check if results is a dictionary with a 'matches' key (as returned by structure_search)
    if isinstance(results, dict) and 'matches' in results:
        matches = results['matches']
    else:
        matches = results  # Assume it's already a list of matches
    
    if not matches:
        console.print(f"[yellow]No matches found for {title}.[/yellow]")
        return
    
    console.print(Panel(f"Found [bold]{len(matches)}[/bold] matches", title=title))
    
    # Create a table for the results
    table = Table(box=ROUNDED)
    table.add_column("File", style="cyan", no_wrap=True)
    table.add_column("Element Type", style="green")
    table.add_column("Line", style="yellow", justify="right")
    table.add_column("Match", style="white")
    table.add_column("Context", style="dim")
    
    # Add up to 10 results to the table
    for result in matches[:10]:
        file_name = os.path.basename(result.get("file_path", "Unknown"))
        element_type = result.get("element_type", "Unknown")
        line_num = result.get("match_line", 0)
        match_text = result.get("match_text", "")
        
        # Get context (parent elements)
        parent_elements = result.get("parent_elements", [])
        parent_info = ""
        if parent_elements:
            parent_types = [p.get("element_type", "Unknown") for p in parent_elements]
            parent_info = " → ".join(parent_types)
        
        # Use context text if available, otherwise the element content
        context_text = result.get("context_text", result.get("element_content", ""))
        if len(context_text) > 50:
            context_text = context_text[:47] + "..."
        
        table.add_row(
            file_name,
            element_type,
            str(line_num),
            match_text,
            parent_info if parent_info else context_text
        )
    
    console.print(table)
    
    if len(matches) > 10:
        console.print(f"[dim]... and {len(matches) - 10} more matches not shown.[/dim]")

async def main():
    """Main entry point for the structure search demo."""
    try:
        await structure_search_demo()
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        import traceback
        console.print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main()) 