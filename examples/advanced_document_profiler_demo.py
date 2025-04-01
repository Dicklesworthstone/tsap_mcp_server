#!/usr/bin/env python3
"""
Advanced Document Profiler Demo

This script demonstrates the comprehensive features of the Document Profiler
composite tool in TSAP, including structural profiling, content analysis,
language detection, and document comparison capabilities.
"""
import asyncio
import sys
import os
from pathlib import Path
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich.box import ROUNDED
from rich.progress import Progress
from rich.tree import Tree
from tabulate import tabulate

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

# Import the MCP client
try:
    from tsap.mcp import MCPClient
except ImportError as e:
    print("Error: Could not import MCPClient or its dependencies.")
    print(f"Import Error: {e}")
    print("Make sure mcp_client_example.py is in the 'examples' directory and all dependencies are importable.")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

console = Console()

DEBUG = False

def debug_print(msg):
    """Print debug message if DEBUG is True."""
    if DEBUG:
        console.print(f"[dim][DEBUG] {msg}[/dim]")

async def document_profiler_demo():
    """Main demonstration of document profiler capabilities."""
    banner = Panel(
        "[bold]TSAP Document Profiler Advanced Features Demo[/bold]\n[dim]Analyzing and comparing document structures[/dim]",
        box=ROUNDED,
    )
    console.print(banner)

    # Setup MCP client
    # This client will connect to the standalone MCP server which should be running
    client = MCPClient("http://localhost:8021")
    
    # Check client connectivity
    console.print("Attempting to get server info from http://localhost:8021...")
    try:
        info = await client.info()  # noqa: F841
        console.print("Initial client.info() check successful.")
    except Exception as e:
        console.print(f"[bold red]Error connecting to MCP server:[/bold red] {e}")
        console.print("\nMake sure the standalone MCP server is running with:")
        console.print("[blue]python examples/standalone_mcp_server.py[/blue]")
        return

    # DEMO 1: Basic Document Profiling
    console.rule("Demo 1: Basic Document Profiling")
    console.print("Creates a basic profile of a text document including metrics and structure.\n")
    await run_basic_profile_demo(client)

    # DEMO 2: Profiling Different Document Types 
    console.rule("Demo 2: Profiling Different Document Types")
    console.print("Demonstrates profiling capabilities across different file types.\n")
    await run_multiple_file_types_demo(client)

    # DEMO 3: Document Comparison
    console.rule("Demo 3: Document Comparison")
    console.print("Compares multiple documents and shows similarity scores.\n")
    
    # Get sample documents of different types 
    documents = [
        "tsap_example_data/documents/strategic_thinking.txt",
        "tsap_example_data/documents/technical_specification.md",
        "tsap_example_data/documents/code_sample.py"
    ]
    
    # Pass file_paths and set include_content to True
    await run_document_comparison_demo(
        client, 
        documents,
        include_content=True
    )
    
    console.rule("Document Profiler Demo Complete")

async def run_basic_profile_demo(client, file_path="tsap_example_data/documents/strategic_thinking.txt"):
    """Run a demonstration of basic document profiling."""
    print(f"Profiling document: {file_path}")
    abs_path = os.path.abspath(file_path)
    
    with Progress() as progress:
        task = progress.add_task("[green]Profiling document...", total=1)
        
        # Call MCP server to profile the document
        result = await client.send_request(
            "tsap.composite.document_profiler.profile_document",
            {
                "document_path": abs_path,  # Use absolute path
                "include_content_features": True
            }
        )
        
        progress.update(task, advance=1)

    if result.get("status") != "success":
        console.print(f"[bold red]Error profiling document:[/bold red] {result.get('error', 'Unknown error')}")
        return
    
    # Display the profile results - NOTE: The profile data is in the 'data' field, not 'result'
    profile_data = result.get("data", {})
    display_document_profile(profile_data)

async def run_multiple_file_types_demo(client, document_paths=None):
    """Run a demonstration of profiling multiple document types."""
    print("\nProfiling multiple document types")
    
    # Default document paths if none provided
    if document_paths is None:
        document_paths = [
            "tsap_example_data/documents/strategic_thinking.txt",
            "tsap_example_data/documents/technical_specification.md",
            "tsap_example_data/documents/code_sample.py"
        ]
    
    with Progress() as progress:
        task = progress.add_task("[green]Profiling documents...", total=len(document_paths))
        
        results = []
        for file_path in document_paths:
            abs_path = str(Path(file_path).absolute())
            console.print(f"\nProcessing: [cyan]{file_path}[/cyan]")
            
            result = await client.send_request(
                "tsap.composite.document_profiler.profile_document",
                {
                    "document_path": abs_path,  # Use absolute path
                    "include_content_features": True
                }
            )
            
            if result.get("status") == "success":
                # Get profile data from the 'data' field
                results.append((file_path, result.get("data", {})))
            else:
                console.print(f"[yellow]Warning:[/yellow] Failed to profile {file_path}: {result.get('error', 'Unknown error')}")
            
            progress.update(task, advance=1)
    
    # Display summary table comparing the different file types
    display_file_type_comparison(results)

async def run_document_comparison_demo(client, document_paths, include_content=True):
    """Run a demonstration of comparing multiple documents."""
    print("\nComparing documents for similarity")
    
    # Convert to absolute paths
    absolute_paths = []
    for path in document_paths:
        abs_path = os.path.abspath(path)
        absolute_paths.append(abs_path)

    try:
        result = await client.send_request(
            "tsap.composite.document_profiler.profile_documents",
            {
                "document_paths": absolute_paths,
                "include_content_features": include_content
            }
        )
        
        if result.get("status") == "success":
            comparison_data = result.get("data", {})
            # Update: Pass only the comparison_data to display_document_comparison
            display_document_comparison(comparison_data)
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()

def display_document_profile(profile_data):
    """Display a single document profile in a structured format."""
    if not profile_data:
        console.print("[yellow]No profile data to display.[/yellow]")
        return
    
    # Extract basic properties
    file_name = profile_data.get("file_name", "Unknown")
    file_extension = profile_data.get("file_extension", "")
    basic_props = profile_data.get("basic_properties", {})
    content_metrics = profile_data.get("content_metrics", {})
    structure_metrics = profile_data.get("structure_metrics", {})
    language_features = profile_data.get("language_features", {})
    content_features = profile_data.get("content_features", {})
    
    # Create the main panel
    console.print(Panel(
        f"Document: [bold]{file_name}[/bold] ({file_extension})\n"
        f"Size: {basic_props.get('file_size', 0)} bytes\n"
        f"Modified: {basic_props.get('modification_time', 'Unknown')}",
        title="Document Information",
        border_style="blue"
    ))
    
    # Create metrics table
    metrics_table = Table(title="Document Metrics", box=ROUNDED)
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green")
    
    # Add content metrics
    metrics_table.add_row("Lines", str(content_metrics.get("line_count", 0)))
    metrics_table.add_row("Words", str(content_metrics.get("word_count", 0)))
    metrics_table.add_row("Characters", str(content_metrics.get("char_count", 0)))
    metrics_table.add_row("Paragraphs", str(structure_metrics.get("paragraph_count", 0)))
    metrics_table.add_row("Headings", str(structure_metrics.get("heading_count", 0)))
    
    # Add language info
    metrics_table.add_row("Language", language_features.get("language", "Unknown"))
    metrics_table.add_row("Language Confidence", f"{language_features.get('language_confidence', 0):.2f}")
    metrics_table.add_row("Readability Score", f"{content_features.get('readability_score', 0):.2f}")
    
    console.print(metrics_table)
    
    # Show top terms
    top_terms = content_features.get("top_terms", [])
    if top_terms:
        terms_table = Table(title="Top Terms", box=ROUNDED)
        terms_table.add_column("Term", style="cyan")
        terms_table.add_column("Frequency", style="green")
        
        for term in top_terms[:10]:  # Show top 10
            if isinstance(term, list) and len(term) == 2:
                terms_table.add_row(term[0], str(term[1]))
        
        console.print(terms_table)
    
    # Show structure as a tree
    heading_hierarchy = profile_data.get("structure_features", {}).get("heading_hierarchy", [])
    if heading_hierarchy:
        console.print("Document Structure:")
        structure_tree = Tree("📄 Document")
        
        # Build tree from heading hierarchy (simplified)
        for heading in heading_hierarchy[:5]:  # Show first few for demo
            if isinstance(heading, dict):
                structure_tree.add(f"[bold]{heading.get('text', 'Untitled')}[/bold] (Level {heading.get('level', '?')})")
        
        console.print(structure_tree)

def display_file_type_comparison(results):
    """Display a comparison of different file types."""
    if not results:
        console.print("[yellow]No comparison data to display.[/yellow]")
        return
    
    table = Table(title="File Type Comparison", box=ROUNDED)
    table.add_column("File", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Words", style="yellow")
    table.add_column("Lines", style="yellow")
    table.add_column("Language", style="magenta")
    table.add_column("Special Features", style="blue")
    
    for file_path, profile in results:
        file_name = os.path.basename(str(file_path))
        file_type = profile.get("file_extension", "Unknown")
        words = profile.get("content_metrics", {}).get("word_count", 0)
        lines = profile.get("content_metrics", {}).get("line_count", 0)
        language = profile.get("language_features", {}).get("language", "Unknown")
        
        # Get special features based on file type
        special_features = []
        content_type_features = profile.get("content_type_features", {})
        if file_type == ".py":
            special_features.append(f"Classes: {content_type_features.get('class_count', 0)}")
            special_features.append(f"Functions: {content_type_features.get('function_count', 0)}")
        elif file_type == ".md":
            special_features.append(f"Links: {content_type_features.get('link_count', 0)}")
            special_features.append(f"Images: {content_type_features.get('image_count', 0)}")
        elif file_type == ".txt":
            special_features.append(f"Readability: {profile.get('content_features', {}).get('readability_score', 0):.2f}")
        
        table.add_row(
            file_name,
            file_type,
            str(words),
            str(lines),
            language,
            ", ".join(special_features) if special_features else "N/A"
        )
    
    console.print(table)

def display_document_comparison(doc_comparison_result):
    """
    Display the document comparison results including a similarity matrix.
    """
    if not doc_comparison_result:
        print("No document comparison data available.")
        return

    # Extract profiles and comparisons from the result
    profiles = doc_comparison_result.get("profiles", {})
    comparisons = doc_comparison_result.get("comparisons", {})
    errors = doc_comparison_result.get("errors", {})  # noqa: F841
    
    if not profiles or not comparisons:
        print("Missing profile or comparison data.")
        return
    
    # Get the document names
    doc_names = list(profiles.keys())
    for name in doc_names:
        # Get just the filename from the path
        profiles[name]["short_name"] = os.path.basename(name)
    
    # Create the similarity matrix
    matrix = []
    header = ["Document"] + [profiles[name]["short_name"] for name in doc_names]
    
    # For each document, create a row in the matrix
    for doc1 in doc_names:
        row = [profiles[doc1]["short_name"]]
        
        # For each column, find the similarity value
        for doc2 in doc_names:
            if doc1 == doc2:
                # Self-similarity is always 1.0
                row.append("1.00")
            else:
                # Look up the similarity in comparisons
                # Try different possible key formats
                key1 = f"{doc1}|{doc2}"
                key2 = f"{doc2}|{doc1}"
                key3 = f"{doc1} <-> {doc2}"
                key4 = f"{doc2} <-> {doc1}"
                
                similarity = None
                
                if key1 in comparisons:
                    similarity = comparisons[key1].get('similarity')
                elif key2 in comparisons:
                    similarity = comparisons[key2].get('similarity')
                elif key3 in comparisons:
                    similarity = comparisons[key3].get('similarity')
                elif key4 in comparisons:
                    similarity = comparisons[key4].get('similarity')
                
                if similarity is not None:
                    row.append(f"{similarity:.2f}")
                else:
                    row.append("N/A")
        
        matrix.append(row)
    
    # Print the similarity matrix
    print(f"\n{' ' * 37}Document Similarity Matrix{' ' * 38}")
    table = tabulate(matrix, headers=header, tablefmt="fancy_grid")
    print(table)
    
    # Also print the most similar documents in descending order
    print("\nMost Similar Documents:")
    similarities = []
    
    # Try to extract similarities using the detected format
    for key, value in comparisons.items():
        if isinstance(key, str) and (" <-> " in key or "|" in key):
            if "|" in key:
                parts = key.split("|")
            else:
                parts = key.split(" <-> ")
                
            if len(parts) == 2:
                doc1, doc2 = parts
                doc1_name = profiles.get(doc1, {}).get("short_name", os.path.basename(doc1))
                doc2_name = profiles.get(doc2, {}).get("short_name", os.path.basename(doc2))
                similarity = value.get("similarity", 0)
                similarities.append((doc1_name, doc2_name, similarity))
    
    # Sort by similarity descending
    if similarities:
        similarities.sort(key=lambda x: x[2], reverse=True)
        
        # Print each similarity
        for doc1, doc2, sim in similarities:
            print(f"{doc1} and {doc2}: {sim:.2f} similarity")
    else:
        print("No similarity data available between documents.")

async def main():
    """Main entry point for the document profiler demo."""
    try:
        await document_profiler_demo()
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        import traceback
        console.print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main()) 