#!/usr/bin/env python3
"""
Document DNA Profiling (MCP Tools Version)

This script demonstrates how to perform document DNA profiling and comparison
using the MCP tools interface.
"""
import asyncio
import os
import json
from typing import List
from rich import print as rich_print
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.console import Console

# Import proper MCP client modules
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

console = Console()

# --- Configuration ---
DOC_DIR = "tsap_example_data/documents"
DOCUMENTS_TO_PROFILE = [
    os.path.join(DOC_DIR, "report_v1.txt"),
    os.path.join(DOC_DIR, "report_v2.txt"),
    # Add more documents here if needed, e.g., os.path.join(DOC_DIR, "contract.pdf")
]

# --- Main Profiling Function ---
async def run_batch_profiling(session, profile_documents_tool, paths: List[str]):
    """Runs batch document profiling and comparison using MCP tools."""
    rich_print(Panel("[bold blue]Starting Batch Document Profiling & Comparison (MCP Tools Version)...[/bold blue]", expand=False))
    rich_print(f"Profiling {len(paths)} documents.")
    
    # Convert paths to absolute paths
    abs_paths = [os.path.abspath(path) for path in paths]

    rich_print("[cyan]Sending Profile Documents Request:[/cyan]")
    config = {
        "document_paths": abs_paths,
        "include_content_features": True,
        "compare_documents": True,
        "cluster_documents": True
    }
    rich_print(Syntax(json.dumps(config, indent=2), "json", theme="default"))

    try:
        # Call the profile_documents tool
        result = await session.call_tool(
            profile_documents_tool.name,
            arguments={
                "document_paths": abs_paths,
                "include_content_features": True,  # Required for meaningful comparison
                "compare_documents": True,         # Compare documents with each other
                "cluster_documents": True          # Optional: Ask server to cluster similar docs
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
                profile_result = json.loads(result_text)
                
                profiles = profile_result.get("profiles", {})
                comparisons = profile_result.get("comparisons", {})
                clustering = profile_result.get("clustering")  # May be None if not requested or failed

                num_profiles = len(profiles)
                num_comparisons = len(comparisons)
                num_clusters = len(clustering.get("clusters", [])) if clustering else 0

                rich_print(Panel(f"[bold green]Profiling Complete:[/bold green] {num_profiles} profiles created, {num_comparisons} comparisons made, {num_clusters} clusters found.", expand=False))

                # --- Display Profiles (Summary) ---
                if profiles:
                    profile_table = Table(title="Document Profile Summary")
                    profile_table.add_column("Document", style="cyan")
                    profile_table.add_column("Size (Bytes)", style="yellow", justify="right")
                    profile_table.add_column("Lines", style="yellow", justify="right")
                    profile_table.add_column("Words", style="yellow", justify="right")
                    profile_table.add_column("Language", style="magenta")
                    profile_table.add_column("Top Terms", style="white")

                    for path, profile_data in profiles.items():
                        filename = profile_data.get("file_name", os.path.basename(path))
                        basic_props = profile_data.get("basic_properties", {})
                        content_metrics = profile_data.get("content_metrics", {})
                        lang_features = profile_data.get("language_features", {})
                        content_features = profile_data.get("content_features", {})
                        top_terms_list = content_features.get("top_terms", [])
                        top_terms = ", ".join([f"{term}({count})" for term, count in top_terms_list[:3]]) # Show top 3

                        profile_table.add_row(
                            filename,
                            str(basic_props.get("file_size", "N/A")),
                            str(content_metrics.get("line_count", "N/A")),
                            str(content_metrics.get("word_count", "N/A")),
                            f"{lang_features.get('language', 'N/A')} ({lang_features.get('language_confidence', 0.0):.1%})",
                            top_terms + ("..." if len(top_terms_list) > 3 else "")
                        )
                    console.print(profile_table)

                # --- Display Comparisons ---
                if comparisons:
                    comp_table = Table(title="Document Comparisons")
                    comp_table.add_column("Document Pair", style="cyan")
                    comp_table.add_column("Similarity", style="yellow", justify="right")

                    # Sort by similarity
                    # Handle different formats of comparison keys (pipe or arrow)
                    sorted_comps = []
                    for pair_key, comp_data in comparisons.items():
                        if ' <-> ' in pair_key:
                            doc1, doc2 = pair_key.split(' <-> ')
                        elif '|' in pair_key:
                            doc1, doc2 = pair_key.split('|')
                        else:
                            continue  # Skip if unexpected format
                        
                        # Format for display
                        filename1 = os.path.basename(doc1)
                        filename2 = os.path.basename(doc2)
                        display_pair = f"{filename1} <-> {filename2}"
                        similarity = comp_data.get("similarity", 0.0)
                        sorted_comps.append((display_pair, similarity, comp_data))
                    
                    # Sort by similarity
                    sorted_comps.sort(key=lambda x: x[1], reverse=True)
                    
                    for display_pair, similarity, _ in sorted_comps:
                        comp_table.add_row(display_pair, f"{similarity:.2%}")
                    console.print(comp_table)

                # --- Display Clustering ---
                if clustering:
                    cluster_table = Table(title="Document Clusters")
                    cluster_table.add_column("Cluster ID", style="cyan")
                    cluster_table.add_column("Suggested Name", style="blue")
                    cluster_table.add_column("Size", style="yellow", justify="right")
                    cluster_table.add_column("Avg Similarity", style="yellow", justify="right")
                    cluster_table.add_column("Documents", style="white")

                    for cluster in clustering.get("clusters", []):
                        cluster_id = cluster.get("id", "N/A")
                        name = cluster.get("suggested_name", "N/A")
                        size = cluster.get("size", 0)
                        avg_sim = cluster.get("average_similarity", 0.0)
                        docs = ", ".join([os.path.basename(d) for d in cluster.get("documents", [])])
                        cluster_table.add_row(cluster_id, name, str(size), f"{avg_sim:.2%}", docs)
                    console.print(cluster_table)
                    rich_print(f"[dim]Clustering Method: {clustering.get('method', 'N/A')}, Silhouette Score: {clustering.get('silhouette_score', 'N/A'):.3f}[/dim]")
            except json.JSONDecodeError:
                rich_print("[bold red]Failed to parse response as JSON[/bold red]")
                rich_print(f"Raw response: {result_text[:200]}...")
        else:
            rich_print("[bold red]No text content in response[/bold red]")
    
    except Exception as e:
        rich_print(f"[bold red]Error during profile_documents operation: {str(e)}[/bold red]")
        import traceback
        rich_print(traceback.format_exc())

# --- Main Execution ---
async def main():
    """Main function to run the document profiling example."""
    # Check if documents exist
    missing_docs = [path for path in DOCUMENTS_TO_PROFILE if not os.path.exists(path)]
    if missing_docs:
        rich_print("[bold red]Error:[/bold red] The following documents were not found:")
        for path in missing_docs:
            rich_print(f"  - {path}")
        rich_print("Please create them or update the DOCUMENTS_TO_PROFILE list.")
        return

    # Path to the proxy script
    proxy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_proxy.py")
    rich_print(f"Using proxy script: {proxy_path}")
    
    # Configure server parameters for stdio connection to our proxy
    server_params = StdioServerParameters(
        command="python",
        args=[proxy_path],
        env={"MCP_DEBUG": "0"}  # Disable debug logging by default
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
                
                # Find the profile_documents and info tools
                profile_documents_tool = next((t for t in tools if t.name == "profile_documents"), None)
                info_tool = next((t for t in tools if t.name == "info"), None)
                
                if not profile_documents_tool:
                    rich_print("[bold red]Error: profile_documents tool not found![/bold red]")
                    return
                
                # Check server health
                if info_tool:
                    rich_print("Checking server info...")
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
                                rich_print("[green]MCP server check successful.[/green]")
                            except json.JSONDecodeError:
                                rich_print("[yellow]Info response is not JSON[/yellow]")
                        else:
                            rich_print("[yellow]Info call succeeded but returned no readable content.[/yellow]")
                    except Exception as e:
                        rich_print(f"[bold red]Error during info check: {str(e)}[/bold red]")
                        return
                
                await run_batch_profiling(session, profile_documents_tool, paths=DOCUMENTS_TO_PROFILE)
    
    except Exception as e:
        rich_print(f"[bold red]Error connecting to MCP server: {str(e)}[/bold red]")
        import traceback
        rich_print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main()) 