#!/usr/bin/env python3
"""
Semantic Search Demo Script (MCP Protocol Version)

This script demonstrates the semantic search capabilities of TSAP
by using the standard MCP protocol client.
"""
import sys
import os
import time
import argparse
import asyncio
import json
import glob
from typing import List, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule

# Import proper MCP client modules
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Initialize console after rich imports
console = Console()

# --- Configuration ---
EXAMPLE_DATA_DIR = "tsap_example_data/documents/"
DEFAULT_QUERY = "What is the nature of strategy?"
MAX_DOCS_TO_INDEX = 50  # Limit number of docs for demo speed
MAX_RESULTS_TO_SHOW = 5
DEBUG = False

def debug_print(msg):
    """Print debug message if DEBUG is True."""
    if DEBUG:
        console.print(f"[dim][DEBUG] {msg}[/dim]")

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

def display_results(results: Dict[str, Any], query: str):
    """Display semantic search results using rich Table."""
    if not results:
        console.print("[yellow]No results found or invalid response format.[/yellow]")
        return

    console.print(Panel(f"Query: '[bold]{query}[/bold]'", title="Search Summary"))

    if DEBUG:
        # Print the exact structure of the results for debugging
        console.print("[dim][DEBUG] Results structure:[/dim]")
        console.print(f"[dim]{results}[/dim]")
        console.print("[dim][DEBUG] Results type: {0}[/dim]".format(type(results)))
        console.print("[dim][DEBUG] Results keys: {0}[/dim]".format(results.keys() if isinstance(results, dict) else "N/A"))

    # Check for error response
    if isinstance(results, dict) and "error" in results:
        console.print("[bold red]Error in search response:[/bold red]")
        error = results["error"]
        if isinstance(error, dict):
            console.print(f"[red]Code: {error.get('code', 'unknown')}[/red]")
            console.print(f"[red]Message: {error.get('message', 'No message')}[/red]")
            if "details" in error and error["details"]:
                console.print("[dim]Details: (truncated)[/dim]")
                details = error["details"]
                if len(details) > 500:
                    details = details[:500] + "..."
                console.print(f"[dim]{details}[/dim]")
        else:
            console.print(f"[red]{error}[/red]")
        return

    # Handle different response formats
    if isinstance(results, dict):
        if "matches" in results:
            matches = results["matches"]
        elif "results" in results:
            matches = results["results"]
        else:
            # If we can't identify a clear list of matches, just use the whole dict as one result
            console.print("[yellow]Unexpected response format. Using entire response as a single result.[/yellow]")
            matches = [results]
    else:
        matches = results  # Assume it's already a list of matches

    match_count = len(matches)

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
        text = result.get('text', '')
        text_snippet = text[:150].strip().replace('\n', ' ') + "..." if len(text) > 150 else text
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
    """Run the semantic search demo using proper MCP protocol."""
    console.print(Panel.fit(
        "[bold blue]TSAP Semantic Search Demo (MCP Protocol)[/bold blue]",
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

    # Path to the proxy script
    proxy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_proxy.py")
    debug_print(f"Proxy path: {proxy_path}")
    
    # Configure server parameters for stdio connection
    server_params = StdioServerParameters(
        command="python",
        args=[proxy_path],
        env={"MCP_DEBUG": "1" if DEBUG else "0"}  # Enable debug logging if needed
    )
    
    # Run search via MCP Protocol
    print_section("Running Semantic Search via MCP Protocol")
    
    try:
        # Connect to the MCP server via proxy
        async with stdio_client(server_params) as (read, write):
            debug_print("Connected to MCP proxy")
            async with ClientSession(read, write) as session:
                # Initialize the connection
                console.print("Initializing session...")
                init_result = await session.initialize()
                console.print(f"[green]Connected to {init_result.serverInfo.name} {init_result.serverInfo.version}[/green]")
                
                # Get the list of available tools
                tools_result = await session.list_tools()
                tools = tools_result.tools
                debug_print(f"Found {len(tools)} tools")
                
                # Find the semantic search tool
                semantic_search_tool = next((t for t in tools if t.name == "semantic_search"), None)
                if not semantic_search_tool:
                    console.print("[bold red]Error: semantic_search tool not found![/bold red]")
                    available_tools = [t.name for t in tools]
                    console.print(f"Available tools: {', '.join(available_tools)}")
                    return
                
                console.print(f"Using semantic search tool: {semantic_search_tool.name}")
                console.print()
                
                # Perform the search using the semantic_search tool
                console.print(f"Sending search request for query: '[bold]{query}[/bold]'...")
                start_time = time.time()
                
                try:
                    # Call the semantic search tool through the MCP session
                    result = await session.call_tool(semantic_search_tool.name, arguments={
                        "query": query,
                        "texts": texts_to_index,
                        "model": "all-MiniLM-L6-v2",
                        "top_k": 10,
                        "threshold": 0.5
                    })
                    
                    # Extract the text content from the response
                    response_text = None
                    for content in result.content:
                        if content.type == "text":
                            response_text = content.text
                            break
                    
                    if response_text:
                        try:
                            search_results = json.loads(response_text)
                            execution_time = time.time() - start_time
                            
                            # Debug print the response structure for debugging
                            if DEBUG:
                                console.print("[dim][DEBUG] Raw search_results structure:[/dim]")
                                console.print(f"[dim]{search_results}[/dim]")
                                console.print("[dim][DEBUG] search_results type: {0}[/dim]".format(type(search_results)))
                                console.print("[dim][DEBUG] search_results keys: {0}[/dim]".format(search_results.keys() if isinstance(search_results, dict) else "N/A"))
                            
                            # Check for different response formats
                            if isinstance(search_results, dict):
                                if "data" in search_results:
                                    search_results = search_results["data"]
                                    if DEBUG:
                                        console.print("[dim][DEBUG] Extracted data from search_results[/dim]")
                                        console.print(f"[dim]{search_results}[/dim]")
                                
                                # Some responses have results inside "matches" or "results" keys
                                if "matches" in search_results:
                                    search_results = search_results["matches"]
                                    if DEBUG:
                                        console.print("[dim][DEBUG] Extracted matches from search_results[/dim]")
                                elif "results" in search_results:
                                    search_results = search_results["results"]
                                    if DEBUG:
                                        console.print("[dim][DEBUG] Extracted results from search_results[/dim]")
                            
                            # Enhance results with metadata
                            if isinstance(search_results, list):
                                for i, result in enumerate(search_results):
                                    if DEBUG:
                                        console.print(f"[dim][DEBUG] Result {i}: {result}[/dim]")
                                        
                                    # Check for different formats of corpus index
                                    corpus_idx = None
                                    if isinstance(result, dict):
                                        if 'corpus_index' in result:
                                            corpus_idx = result['corpus_index']
                                        elif 'index' in result:
                                            corpus_idx = result['index']
                                            
                                        if corpus_idx is not None and 0 <= corpus_idx < len(metadata_to_index):
                                            # Add metadata and ID to the result
                                            result['metadata'] = metadata_to_index[corpus_idx]
                                            result['id'] = ids_to_index[corpus_idx]
                                            if DEBUG:
                                                console.print(f"[dim][DEBUG] Added metadata for result {i} with corpus_idx {corpus_idx}[/dim]")
                            
                            console.print(f"Search request completed in {execution_time:.2f} seconds.")
                            
                        except json.JSONDecodeError:
                            console.print("[bold red]Failed to parse response as JSON[/bold red]")
                            console.print(f"Raw response: {response_text[:200]}...")
                            return
                    else:
                        console.print("[bold red]No text content in response[/bold red]")
                        return
                
                except Exception as e:
                    console.print(f"[bold red]Error during semantic search call:[/bold red] {e}")
                    import traceback
                    console.print(traceback.format_exc())
                    return

                # Process and display results
                print_section("Search Results")
                display_results(search_results, query)

    except Exception as e:
        console.print(f"[bold red]Error connecting to MCP server: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())

# --- Entry Point ---

async def main():
    # Declare global first
    global MAX_DOCS_TO_INDEX, DEBUG
    
    parser = argparse.ArgumentParser(description="Run TSAP Semantic Search Demo using MCP Protocol.")
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )

    args = parser.parse_args()
    
    # Update global config if needed
    MAX_DOCS_TO_INDEX = args.max_docs
    DEBUG = args.debug
    
    if DEBUG:
        debug_print("Debug mode enabled")

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