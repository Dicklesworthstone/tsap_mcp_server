#!/usr/bin/env python3
"""
Advanced Document Profiler Demo (MCP Tools Version)

This script demonstrates the comprehensive features of the Document Profiler
composite tool in TSAP MCP, including structural profiling, content analysis,
language detection, and document comparison capabilities.
"""
import asyncio
import sys
import os
import json
from pathlib import Path
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich.box import ROUNDED
from rich.progress import Progress
from rich.tree import Tree
from tabulate import tabulate
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

async def document_profiler_demo():
    """Main demonstration of document profiler capabilities using MCP tools."""
    banner = Panel(
        "[bold]TSAP MCP Document Profiler Advanced Features Demo[/bold]\n[dim]Analyzing and comparing document structures using MCP tools[/dim]",
        box=ROUNDED,
    )
    console.print(banner)

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
            debug_print("Connected to MCP proxy")
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()
                debug_print("Session initialized successfully")
                
                # Get the list of available tools
                tools_result = await session.list_tools()
                tools = tools_result.tools
                debug_print(f"Found {len(tools)} tools")
                
                # Find the document profiler tools
                profile_document_tool = next((t for t in tools if t.name == "document_profile"), None)
                profile_documents_tool = next((t for t in tools if t.name == "document_explore"), None)
                
                if not profile_document_tool:
                    console.print("[bold red]Error: document_profile tool not found![/bold red]")
                    # Print available tools to help diagnose the issue
                    available_tools = [t.name for t in tools]
                    console.print(f"Available tools: {', '.join(available_tools)}")
                    return
                
                if not profile_documents_tool:
                    console.print("[bold red]Error: document_explore tool not found![/bold red]")
                    # Print available tools to help diagnose the issue
                    available_tools = [t.name for t in tools]
                    console.print(f"Available tools: {', '.join(available_tools)}")
                    return
                
                # Find the info tool for initial check
                info_tool = next((t for t in tools if t.name == "info"), None)
                
                # Check client connectivity
                if info_tool:
                    console.print("Attempting to get server info...")
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
                        console.print(f"[bold red]Error during initial info check:[/bold red] {str(e)}")
                        return # Exit if server check fails
                
                # DEMO 1: Basic Document Profiling
                console.rule("Demo 1: Basic Document Profiling")
                console.print("Creates a basic profile of a text document including metrics and structure.\n")
                await run_basic_profile_demo(session, profile_document_tool)

                # DEMO 2: Profiling Different Document Types 
                console.rule("Demo 2: Profiling Different Document Types")
                console.print("Demonstrates profiling capabilities across different file types.\n")
                await run_multiple_file_types_demo(session, profile_document_tool)

                # DEMO 3: Document Comparison
                console.rule("Demo 3: Document Comparison")
                console.print("Compares multiple documents and shows similarity scores.\n")
                
                # Get sample documents of different types 
                documents = [
                    "tsap_example_data/documents/strategic_thinking.txt",
                    "tsap_example_data/documents/technical_specification.md",
                    "tsap_example_data/documents/code_sample.py"
                ]
                
                # Pass documents and set include_content to True
                await run_document_comparison_demo(
                    session, 
                    profile_documents_tool,
                    documents,
                    include_content=True
                )
                
                console.rule("Document Profiler Demo Complete")

    except Exception as e:
        console.print(f"[bold red]Error running demo: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())

async def run_basic_profile_demo(session, profile_document_tool, file_path="tsap_example_data/documents/strategic_thinking.txt"):
    """Run a demonstration of basic document profiling using MCP tools."""
    print(f"Profiling document: {file_path}")
    
    with Progress() as progress:
        task = progress.add_task("[green]Profiling document...", total=1)
        
        # Call MCP tools to profile the document
        try:
            # Use proper MCP tool call with the correct format matching the curl command
            result = await session.call_tool(
                profile_document_tool.name, 
                arguments={
                    "document_paths": [file_path],
                    "include_content_features": True
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
                    response = json.loads(result_text)
                    
                    # Extract profile data directly - we don't need to check status
                    profiles = response.get("profiles", {})
                    
                    # Get the profile for this specific file
                    profile_data = profiles.get(file_path)
                    
                    progress.update(task, advance=1)
                    
                    # Display the profile results
                    if profile_data:
                        display_document_profile(profile_data)
                    else:
                        console.print("[yellow]No profile data found for this document.[/yellow]")
                        console.print(f"Available profiles: {list(profiles.keys())}")
                        
                except json.JSONDecodeError:
                    console.print("[bold red]Failed to parse response as JSON[/bold red]")
                    console.print(f"Raw response: {result_text[:200]}...")
                    progress.update(task, advance=1)
            else:
                console.print("[bold red]No text content in response[/bold red]")
                progress.update(task, advance=1)
            
        except Exception as e:
            console.print(f"[bold red]Error profiling document:[/bold red] {str(e)}")
            progress.update(task, advance=1)

async def run_multiple_file_types_demo(session, profile_document_tool, document_paths=None):
    """Run a demonstration of profiling multiple document types using MCP tools."""
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
            console.print(f"\nProcessing: [cyan]{file_path}[/cyan]")
            
            try:
                # Use proper MCP tool call with the correct format matching the curl command
                result = await session.call_tool(
                    profile_document_tool.name, 
                    arguments={
                        "document_paths": [file_path],
                        "include_content_features": True
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
                        response = json.loads(result_text)
                        
                        # Extract profile data directly without checking status
                        profiles = response.get("profiles", {})
                        
                        # Get the profile for this specific file
                        profile_data = profiles.get(file_path)
                        
                        if profile_data:
                            results.append((file_path, profile_data))
                        else:
                            console.print(f"[yellow]No profile data found for {file_path}[/yellow]")
                            if profiles:
                                console.print(f"Available profiles: {list(profiles.keys())}")
                    except json.JSONDecodeError:
                        console.print(f"[yellow]Failed to parse response for {file_path} as JSON[/yellow]")
                else:
                    console.print(f"[yellow]No text content in response for {file_path}[/yellow]")
                
            except Exception as e:
                console.print(f"[yellow]Failed to profile {file_path}: {str(e)}[/yellow]")
            
            progress.update(task, advance=1)
    
    # Display summary table comparing the different file types
    display_file_type_comparison(results)

async def run_document_comparison_demo(session, profile_documents_tool, document_paths, include_content=True):
    """Run a demonstration of comparing multiple documents using MCP tools."""
    print("\nComparing documents for similarity")
    
    # Call document_profile with all document paths in a single call
    try:
        # Use document_profile directly since document_explore isn't finding files
        result = await session.call_tool(
            "document_profile", 
            arguments={
                "document_paths": document_paths,
                "include_content_features": include_content,
                "compare_documents": True  # Enable comparison
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
                response = json.loads(result_text)
                
                # Display the comparison results directly - we don't need to check status
                if response.get("profiles") and response.get("comparisons"):
                    display_document_comparison(response)
                else:
                    console.print("[yellow]No document profiles or comparisons found.[/yellow]")
                    if response.get("profiles"):
                        console.print(f"Found {len(response.get('profiles'))} profiles but no comparisons.")
                    else:
                        console.print("No profiles found.")
            except json.JSONDecodeError:
                console.print("[bold red]Failed to parse document comparison response as JSON[/bold red]")
                console.print(f"Raw response: {result_text[:200]}...")
        else:
            console.print("[bold red]No text content in document comparison response[/bold red]")
        
    except Exception as e:
        print(f"Error during document comparison: {e}")
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

    # First determine what type of result we have (document_profile or document_explore)
    if "profiles" in doc_comparison_result:
        # This is from document_profile
        profiles_data = doc_comparison_result.get("profiles", {})
        comparisons_data = doc_comparison_result.get("comparisons", {})
    elif "documents" in doc_comparison_result:
        # This is from document_explore
        documents = doc_comparison_result.get("documents", [])
        profiles_data = {}
        comparisons_data = {}
        
        # Extract profiles from documents
        for doc in documents:
            file_path = doc.get("path", "unknown")
            profiles_data[file_path] = {
                "file_name": os.path.basename(file_path),
                "content_metrics": doc.get("metrics", {}),
                "language_features": doc.get("language", {}),
                "short_name": os.path.basename(file_path)
            }
        
        # Extract comparisons if available
        similarities = doc_comparison_result.get("similarities", {})
        for pair, score in similarities.items():
            if isinstance(pair, str) and " <-> " in pair:
                comparisons_data[pair] = {"similarity": score}
    else:
        print("Unrecognized comparison data format.")
        return
    
    # Check if we have any profiles
    if not profiles_data:
        console.print("[yellow]No document profiles found for comparison.[/yellow]")
        return
        
    # Check if we have any comparisons
    if not comparisons_data:
        console.print("[yellow]No document comparisons found.[/yellow]")
        # If we have profiles but no comparisons, just list the documents
        docs_table = Table(title="Documents Found")
        docs_table.add_column("Document", style="cyan")
        docs_table.add_column("Size", style="yellow", justify="right")
        
        for path, profile in profiles_data.items():
            docs_table.add_row(
                profile.get("file_name", os.path.basename(path)),
                str(profile.get("content_metrics", {}).get("char_count", "N/A"))
            )
        
        console.print(docs_table)
        return
    
    # Add short_name to each profile if missing
    for name in profiles_data:
        if "short_name" not in profiles_data[name]:
            profiles_data[name]["short_name"] = os.path.basename(name)
    
    # Create the similarity matrix
    matrix = []
    doc_names = list(profiles_data.keys())
    header = ["Document"] + [profiles_data[name]["short_name"] for name in doc_names]
    
    # For each document, create a row in the matrix
    for doc1 in doc_names:
        row = [profiles_data[doc1]["short_name"]]
        
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
                
                if key1 in comparisons_data:
                    similarity = comparisons_data[key1].get('similarity')
                elif key2 in comparisons_data:
                    similarity = comparisons_data[key2].get('similarity')
                elif key3 in comparisons_data:
                    similarity = comparisons_data[key3].get('similarity')
                elif key4 in comparisons_data:
                    similarity = comparisons_data[key4].get('similarity')
                
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
    for key, value in comparisons_data.items():
        if isinstance(key, str) and (" <-> " in key or "|" in key):
            separator = "|" if "|" in key else " <-> "
            parts = key.split(separator)
                
            if len(parts) == 2:
                doc1, doc2 = parts
                similarity = value.get("similarity")
                
                if similarity is not None:
                    doc1_name = profiles_data.get(doc1, {}).get("short_name", doc1)
                    doc2_name = profiles_data.get(doc2, {}).get("short_name", doc2)
                    similarities.append((doc1_name, doc2_name, similarity))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[2], reverse=True)
    
    # Print top similarities
    for i, (doc1, doc2, similarity) in enumerate(similarities[:5]):  # Show top 5
        print(f"{i+1}. {doc1} <-> {doc2}: {similarity:.2f}")

async def main():
    """Run the document profiler demo."""
    try:
        await document_profiler_demo()
    except KeyboardInterrupt:
        console.print("[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Unhandled error: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())

if __name__ == "__main__":
    if "--debug" in sys.argv:
        DEBUG = True
        debug_print("Debug mode enabled")
    
    asyncio.run(main()) 