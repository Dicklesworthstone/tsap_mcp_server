# examples/document_dna_profiling.py
import asyncio
import os
import json
from typing import List

from rich import print as rich_print
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.console import console

from api_client import TSAPClient, API_KEY # Import from our client module

# --- Configuration ---
DOC_DIR = "tsap_example_data/documents"
DOCUMENTS_TO_PROFILE = [
    os.path.join(DOC_DIR, "report_v1.txt"),
    os.path.join(DOC_DIR, "report_v2.txt"),
    # Add more documents here if needed, e.g., os.path.join(DOC_DIR, "contract.pdf")
]

# --- Main Profiling Function ---
async def run_batch_profiling(client: TSAPClient, paths: List[str]):
    """Runs batch document profiling and comparison."""
    rich_print(Panel("[bold blue]Starting Batch Document Profiling & Comparison...[/bold blue]", expand=False))
    rich_print(f"Profiling {len(paths)} documents.")

    # --- Define Request Payload ---
    # Uses the /api/composite/batch_profile endpoint
    # Based on api/models/composite.py (BatchProfileRequest) and api/routes/composite.py
    request_payload = {
        "document_paths": paths,
        "include_content_features": True, # Required for meaningful comparison
        "compare_documents": True,        # Compare documents with each other
        "cluster_documents": True,        # Optional: Ask server to cluster similar docs
        "performance_mode": "standard",   # Optional override
        # "max_documents": 10 # Optional limit
    }

    rich_print("[cyan]Sending Batch Profile Request:[/cyan]")
    rich_print(Syntax(json.dumps(request_payload, indent=2), "json", theme="default"))

    # --- Make API Call ---
    response = await client.post("/api/composite/batch_profile", payload=request_payload)

    # --- Process Response ---
    if not response or "error" in response:
        rich_print("[bold red]Batch profiling request failed.[/bold red]", response)
        return

    if "profiles" not in response:
         rich_print("[bold red]Unexpected response format:[/bold red]", response)
         return

    profiles = response.get("profiles", {})
    comparisons = response.get("comparisons", {})
    clustering = response.get("clustering") # May be None if not requested or failed

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
        sorted_comps = sorted(comparisons.items(), key=lambda item: item[1].get("similarity", 0), reverse=True)

        for pair_key, comp_data in sorted_comps:
             # Extract filenames for display
             doc1, doc2 = pair_key.split(' <-> ')
             filename1 = os.path.basename(doc1)
             filename2 = os.path.basename(doc2)
             display_pair = f"{filename1} <-> {filename2}"
             similarity = comp_data.get("similarity", 0.0)
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

    if API_KEY == "your-default-api-key":
        rich_print("[bold yellow]Warning:[/bold yellow] Using default API key. Set the TSAP_API_KEY environment variable.")

    async with TSAPClient() as client:
        await run_batch_profiling(client, paths=DOCUMENTS_TO_PROFILE)

if __name__ == "__main__":
    asyncio.run(main())