#!/usr/bin/env python3
import sys
import os
import time
import argparse
import asyncio
import json
import glob
from typing import List, Dict, Any
# Use absolute import relative to project root in sys.path
from mcp_client_example import MCPClient
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.rule import Rule

# Initialize console after rich imports
console = Console()

"""
Semantic Search Demo Script (MCP Client Version)

This script demonstrates the semantic search capabilities of TSAP
by interacting with the MCP server via the MCPClient.
"""

# --- Configuration ---
EXAMPLE_DATA_DIR = "tsap_example_data/documents/"
DEFAULT_QUERY = "What is the nature of strategy?"
MAX_DOCS_TO_INDEX = 50  # Limit number of docs for demo speed
MAX_RESULTS_TO_SHOW = 5

# --- Robust Path Setup ---
def find_project_root(marker_dirs=["src", "examples"]):
    """Find the project root directory containing marker directories."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while True:
        if all(os.path.exists(os.path.join(current_dir, marker)) for marker in marker_dirs):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir: # Reached filesystem root
            raise RuntimeError("Could not find project root containing: " + ", ".join(marker_dirs))
        current_dir = parent_dir

PROJECT_ROOT = find_project_root()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --- End Path Setup ---

# --- Helper Functions ---

def print_section(title: str):
    """Print a section title using rich Rule."""
    console.print(Rule(f"[bold cyan]{title}[/bold cyan]"))
    console.print()

def load_documents(data_dir: str, max_docs: int) -> List[Dict[str, Any]]:
    """Load text documents from the specified directory."""
    documents = []
    filepaths = glob.glob(os.path.join(data_dir, "*.md")) + \
                glob.glob(os.path.join(data_dir, "*.txt"))
                
    if not filepaths:
        console.print(f"[bold red]Error:[/bold red] No .md or .txt files found in '{data_dir}'.")
        console.print("[yellow]Please ensure the 'tsap_example_data' directory exists and contains documents.[/yellow]")
        sys.exit(1)
        
    console.print(f"Loading documents from '{data_dir}'...")
    for i, filepath in enumerate(filepaths):
        if i >= max_docs:
            console.print(f"[dim]Reached max documents ({max_docs}), stopping loading.[/dim]")
            break
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append({
                    "id": f"doc_{os.path.basename(filepath)}_{i}",
                    "text": content,
                    "metadata": {
                        "source_file": os.path.basename(filepath),
                        "path": filepath
                    }
                })
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Could not read file '{filepath}': {e}")
            
    console.print(f"Loaded {len(documents)} documents.")
    return documents

def display_results(results_data: Dict[str, Any]):
    """Display semantic search results using rich Table."""
    if not results_data or "results" not in results_data:
        console.print("[yellow]No results found or invalid response format.[/yellow]")
        return

    matches = results_data.get("results", [])
    query = results_data.get("query", "N/A")
    backend = results_data.get("faiss_backend", "N/A")
    top_k = results_data.get("top_k", "N/A")
    match_count = len(matches)

    console.print(Panel(f"Query: '[bold]{query}[/bold]' | Backend: {backend} | Top K: {top_k}", title="Search Summary"))

    if match_count == 0:
        console.print("[yellow]No matches found for this query.[/yellow]")
        return

    table = Table(title=f"Top {min(match_count, MAX_RESULTS_TO_SHOW)} Semantic Search Results")
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Score", style="yellow", width=8)
    table.add_column("Source File", style="cyan", no_wrap=True)
    table.add_column("Text Snippet", style="white")

    for i, result in enumerate(matches[:MAX_RESULTS_TO_SHOW]):
        score = f"{result.get('score', 0.0):.4f}"
        text_snippet = result.get('text', '')[:150].strip().replace('\n', ' ') + "..."
        metadata = result.get('metadata', {})
        source_file = metadata.get('source_file', 'N/A')
        
        table.add_row(
            str(i + 1),
            score,
            source_file,
            text_snippet
        )

    console.print(table)
    if match_count > MAX_RESULTS_TO_SHOW:
        console.print(f"[dim]... and {match_count - MAX_RESULTS_TO_SHOW} more matches not shown.[/dim]")
    console.print()

# --- Main Demo Function ---

async def run_semantic_search_demo(query: str, data_dir: str):
    """Run the semantic search demo using MCPClient."""
    console.print(Panel.fit(
        "[bold blue]TSAP Semantic Search Demo (MCP Client)[/bold blue]",
        subtitle=f"Querying data from '{data_dir}'"
    ))

    # Load documents
    print_section("Loading Data")
    documents = load_documents(data_dir, MAX_DOCS_TO_INDEX)
    if not documents:
        console.print("[bold red]Failed to load documents. Exiting.[/bold red]")
        return

    texts_to_index = [doc["text"] for doc in documents]
    ids_to_index = [doc["id"] for doc in documents]
    metadata_to_index = [doc["metadata"] for doc in documents]

    # Run search via MCP Client
    print_section("Running Semantic Search via MCP")
    async with MCPClient() as client:
        # Check server connection
        console.print("Checking server connection...")
        info = await client.info()
        # Improved connection check
        if info.get("status") != "success" or info.get("error") is not None:
            console.print("[bold red]Error connecting to server or server status not success:[/bold red]")
            console.print(f"Response: {info}") # Print the full response for debugging
            return
        else:
            console.print("[green]Server connection successful.[/green]")
            # console.print(Syntax(json.dumps(info.get('data', {}), indent=2), "json", theme="default"))
            console.print()

        # Perform the search
        console.print(f"Sending search request for query: '[bold]{query}[/bold]'...")
        start_time = time.time()
        
        try:
            search_response = await client.semantic_search(
                texts=texts_to_index,
                query=query,
                ids=ids_to_index,
                metadata=metadata_to_index,
                top_k=10 # Request more than we display initially
            )
            execution_time = time.time() - start_time
            console.print(f"Search request completed in {execution_time:.2f} seconds.")

        except Exception as e:
            console.print(f"[bold red]Error during semantic_search call:[/bold red] {e}")
            import traceback
            console.print(traceback.format_exc())
            return

        # Process and display results
        print_section("Search Results")
        # Improved check: Verify status is success AND error is None/null
        if search_response.get("status") != "success" or search_response.get("error") is not None:
            console.print("[bold red]Server returned an error or unexpected status:[/bold red]")
            # Print the whole response if it's not success or has an error
            error_content = search_response.get("error", "No error field found")
            try:
                # Try pretty printing if it's json-serializable
                error_display = json.dumps(error_content, indent=2)
            except TypeError:
                # Otherwise, just convert to string
                error_display = str(error_content)
                
            console.print(f"Status: {search_response.get('status', 'N/A')}")
            console.print("Error Content:")
            console.print(Syntax(error_display, "json" if isinstance(error_content, (dict, list)) else "text", theme="default", line_numbers=True))
            # Optionally print the full response for more context
            # console.print("[dim]Full response:[/dim]")
            # console.print(Syntax(json.dumps(search_response, indent=2), "json", theme="default"))
        elif "data" in search_response:
            # This path is now only taken if status is 'success' and error is None
            display_results(search_response["data"])
        else:
            # This case might indicate success status but missing data field
            console.print("[bold red]Unexpected response format: Status success but no data field.[/bold red]")
            console.print(Syntax(json.dumps(search_response, indent=2), "json", theme="default", line_numbers=True))

# --- Entry Point ---

async def main():
    # Declare global first
    global MAX_DOCS_TO_INDEX
    
    parser = argparse.ArgumentParser(description="Run TSAP Semantic Search Demo using MCP Client.")
    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY,
        help=f"The semantic query to run. Default: '{DEFAULT_QUERY}'"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=EXAMPLE_DATA_DIR,
        help=f"Directory containing documents to search. Default: '{EXAMPLE_DATA_DIR}'"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=MAX_DOCS_TO_INDEX,
        help=f"Maximum number of documents to load and index. Default: {MAX_DOCS_TO_INDEX}"
    )

    args = parser.parse_args()
    
    # Update global config if needed (e.g., MAX_DOCS_TO_INDEX)
    MAX_DOCS_TO_INDEX = args.max_docs

    await run_semantic_search_demo(args.query, args.data_dir)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("[yellow]Demo interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Unhandled error in demo:[/bold red] {str(e)}")
        import traceback
        console.print(traceback.format_exc()) 