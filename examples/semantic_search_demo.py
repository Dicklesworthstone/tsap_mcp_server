#!/usr/bin/env python3
import sys
import os
import time
import argparse
import asyncio
import json
import glob
from typing import List, Dict, Any, Optional
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.rule import Rule
from tsap.mcp import MCPClient

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

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Basic text chunking by paragraphs, ensuring chunks aren't too small."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current_chunk = ""
    min_chunk_length = 100 # Avoid tiny chunks

    for paragraph in paragraphs:
        if not current_chunk:
            current_chunk = paragraph
        # If adding the next paragraph fits within chunk_size or the current chunk is too small
        elif len(current_chunk) < min_chunk_length or len(current_chunk) + len(paragraph) + 1 <= chunk_size:
            current_chunk += "\n\n" + paragraph
        else:
            # Chunk is full enough, add it
            chunks.append(current_chunk)
            # Start new chunk, potentially with overlap (last part of previous paragraph)
            # Simple overlap: just start with the new paragraph
            current_chunk = paragraph

    # Add the last remaining chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)

    # Filter out any potentially remaining very small chunks after processing
    return [chunk for chunk in chunks if len(chunk) >= min_chunk_length // 2]

def load_and_chunk_documents(data_dir: str, max_docs: int) -> List[Dict[str, Any]]:
    """Load text documents from the specified directory and chunk them."""
    chunked_documents = []
    filepaths = glob.glob(os.path.join(data_dir, "*.md")) + \
                glob.glob(os.path.join(data_dir, "*.txt"))
                
    if not filepaths:
        console.print(f"[bold red]Error:[/bold red] No .md or .txt files found in '{data_dir}'.")
        console.print("[yellow]Please ensure the 'tsap_example_data' directory exists and contains documents.[/yellow]")
        sys.exit(1)
        
    console.print(f"Loading and chunking documents from '{data_dir}'...")
    doc_count = 0
    chunk_count = 0
    for i, filepath in enumerate(filepaths):
        if doc_count >= max_docs:
            console.print(f"[dim]Reached max documents ({max_docs}), stopping loading.[/dim]")
            break
        doc_count += 1
        base_filename = os.path.basename(filepath)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                chunks = chunk_text(content) # Use the new chunking function
                for chunk_idx, chunk_text_content in enumerate(chunks):
                    chunk_id = f"doc_{base_filename}_{i}_chunk_{chunk_idx}"
                    metadata = {
                        "source_file": base_filename,
                        "path": filepath,
                        "chunk_index": chunk_idx,
                        "total_chunks": len(chunks)
                    }
                    chunked_documents.append({
                        "id": chunk_id,
                        "text": chunk_text_content,
                        "metadata": metadata
                    })
                    chunk_count += 1
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Could not read or chunk file '{filepath}': {e}")
            
    console.print(f"Loaded {doc_count} documents, created {chunk_count} chunks.")
    return chunked_documents

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
        chunk_info = f" (Chunk {metadata.get('chunk_index', '?')}/{metadata.get('total_chunks', '?')})" if 'chunk_index' in metadata else ""
        
        table.add_row(
            str(i + 1),
            score,
            f"{source_file}{chunk_info}", # Display chunk info alongside filename
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

    # Load and chunk documents
    print_section("Loading & Chunking Data")
    documents = load_and_chunk_documents(data_dir, MAX_DOCS_TO_INDEX)
    if not documents:
        console.print("[bold red]Failed to load or chunk documents. Exiting.[/bold red]")
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
        
        # Pass the extracted payload
        display_results(search_response.get("data"))

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
        default=os.path.join(PROJECT_ROOT, EXAMPLE_DATA_DIR),
        help=f"Directory containing documents to search. Default: '{EXAMPLE_DATA_DIR}' relative to project root"
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