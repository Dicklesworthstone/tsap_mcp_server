#!/usr/bin/env python3
"""
Advanced HTML Processor Demo

This script demonstrates the comprehensive features of the HTML Processor
integration in TSAP, including parsing local files and URLs, extracting
various content types (text, links, tables, metadata), using selectors,
and demonstrating synergy with Ripgrep and JQ.
"""
import asyncio
import sys
import os
import json
import re
from datetime import datetime
from pathlib import Path
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.console import Console
from rich.rule import Rule
from rich.box import ROUNDED
from rich.markup import escape
from typing import Optional, Dict

# --- Path Setup --- #
# Add the project root and src directory to the Python path
# This allows importing modules from 'src' (like tsap) and finding 'mcp_client_example'
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) # Go up one level from examples/
src_path = os.path.join(project_root, 'src')

# Add project root if not already present (handles running from root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add src path if not already present
if src_path not in sys.path:
    sys.path.insert(0, src_path) # Prepend src directory

# Add examples path itself (helps ensure mcp_client_example is found directly)
if script_dir not in sys.path:
     sys.path.insert(1, script_dir) # Insert after src

# --- End Path Setup ---

# Assuming mcp_client_example.py is in the same directory or accessible
try:
    from tsap.mcp import MCPClient
except ImportError as e:
    print("Error: Could not import MCPClient or its dependencies.")
    print(f"Import Error: {e}")
    print("Make sure mcp_client_example.py is in the 'examples' directory and all dependencies (like httpx, rich, tsap) are importable.")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

console = Console()

# Add some basic debugging
DEBUG = False

def debug_print(msg):
    """Print debug message if DEBUG is True."""
    if DEBUG:
        console.print(f"[dim][DEBUG] {msg}[/dim]")

async def html_processor_demo():
    """Demonstrate HTML Processor's advanced features."""
    console.print(Panel.fit(
        "[bold blue]TSAP HTML Processor Advanced Features Demo[/bold blue]",
        subtitle="Parsing and extracting data from HTML content"
    ))

    # Define file paths (relative to workspace root)
    base_data_dir = Path("tsap_example_data/html/")
    algebra_wiki_file = base_data_dir / "Algebraic_topology_Wikipedia.html"
    sec_filing_file = base_data_dir / "10q_filing.html"
    example_url = "http://example.com" # Simple external URL for demo

    # Check existence of essential files
    required_files = [algebra_wiki_file, sec_filing_file]
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        console.print(f"[bold red]Error:[/bold red] Required example files not found: {', '.join(map(str, missing_files))}")
        console.print(f"Please ensure the '{base_data_dir}' directory contains the required HTML files.")
        return

    # Create our client
    try:
        debug_print("Creating MCPClient...")
        async with MCPClient() as client:
            debug_print("Running demos...")

            # --- Add initial info check ---
            console.print(f"Attempting to get server info from {client.base_url}...")
            try:
                info = await client.info()
                if info.get("status") != "success" or info.get("error") is not None:
                    error_message = info.get('error', "Status was not success")
                    console.print(f"[bold red]Error during initial client.info() check:[/bold red] {error_message}")
                    return # Exit if server check fails
                else:
                   console.print("[green]Initial client.info() check successful.[/green]")
            except Exception as e:
                 console.print(f"[bold red]Error during initial client.info() check:[/bold red] {e}")
                 import traceback
                 console.print(traceback.format_exc())
                 return
            # ------------------------------

            # Check if we should only run a specific demo
            if "--js-only" in sys.argv:
                console.print("[yellow]Running only JavaScript demo (Demo 9)...[/yellow]")
                # Skip to Demo 9
                await run_js_demo(client, algebra_wiki_file, base_data_dir)
                return

            # DEMO 1: Basic File Metadata Extraction
            console.print(Rule("[bold yellow]Demo 1: Basic File Metadata (Wikipedia Page)[/bold yellow]"))
            console.print(f"[italic]Extracts metadata (title, H1, stats) from '{algebra_wiki_file.name}'.[/italic]")
            console.print() # Add newline
            await run_html_demo(
                client,
                demo_title="Demo 1: Basic File Metadata",
                file_path=str(algebra_wiki_file),
                extract_metadata=True,
                show_title=False
            )

            # Let's limit our demos for debugging
            if DEBUG:
                debug_print("DEBUG mode: Only running first demo")
                return

            # DEMO 2: Extracting Links from File
            console.print(Rule("[bold yellow]Demo 2: Extracting Links (Wikipedia Page)[/bold yellow]"))
            console.print(f"[italic]Extracts all links from '{algebra_wiki_file.name}', showing resolved URLs (limited output).[/italic]")
            console.print() # Add newline
            await run_html_demo(
                client,
                demo_title="Demo 2: Link Extraction",
                file_path=str(algebra_wiki_file),
                extract_links=True,
                show_title=False,
                max_links_display=15 # Custom arg for display limiting
            )

            # DEMO 3: Extracting Tables from File
            console.print(Rule("[bold yellow]Demo 3: Extracting Tables (SEC Filing)[/bold yellow]"))
            console.print(f"[italic]Extracts tables from '{sec_filing_file.name}' (showing first few rows of first table).[/italic]")
            console.print() # Add newline
            await run_html_demo(
                client,
                demo_title="Demo 3: Table Extraction",
                file_path=str(sec_filing_file),
                extract_tables=True,
                show_title=False,
                max_table_rows_display=5 # Custom arg for display limiting
            )

            # DEMO 4: Extracting Clean Text
            console.print(Rule("[bold yellow]Demo 4: Extracting Clean Text (Wikipedia Page)[/bold yellow]"))
            console.print(f"[italic]Gets clean, readable text content from '{algebra_wiki_file.name}'.[/italic]")
            console.print() # Add newline
            await run_html_demo(
                client,
                demo_title="Demo 4: Clean Text Extraction",
                file_path=str(algebra_wiki_file),
                extract_text=True,
                show_title=False,
                max_text_display_chars=500 # Custom arg for display limiting
            )

            # DEMO 5: Using CSS Selectors
            console.print(Rule("[bold yellow]Demo 5: Using CSS Selectors (Wikipedia Page)[/bold yellow]"))
            console.print(f"[italic]Extracts specific heading elements using the CSS selector 'h2 > span.mw-headline' from '{algebra_wiki_file.name}'.[/italic]")
            console.print() # Add newline
            await run_html_demo(
                client,
                demo_title="Demo 5: CSS Selector",
                file_path=str(algebra_wiki_file),
                selector='h2 > span.mw-headline', # Select heading spans within H2s
                show_title=False,
                max_elements_display=10 # Custom arg for display limiting
            )

            # DEMO 6: Processing a URL
            console.print(Rule("[bold yellow]Demo 6: Processing a URL (example.com)[/bold yellow]"))
            console.print(f"[italic]Fetches '{example_url}' and extracts its title and links.[/italic]")
            console.print() # Add newline
            await run_html_demo(
                client,
                demo_title="Demo 6: URL Processing",
                url=example_url,
                extract_metadata=True,
                extract_links=True,
                show_title=False
            )

            # DEMO 7: Synergy with Ripgrep (Find -> Process)
            console.print(Rule("[bold yellow]Demo 7: Synergy (Ripgrep -> HTML Processor)[/bold yellow]"))
            console.print(f"[italic]Uses Ripgrep to find HTML files in '{base_data_dir}' containing 'Table' (case-insensitive), then extracts tables from those files.[/italic]")
            console.print() # Add newline

            # Step 7.1: Run Ripgrep
            console.print(f"[bold cyan]Step 7.1: Running Ripgrep to find relevant HTML files in '{base_data_dir}'...[/bold cyan]")
            files_found_rg = []
            try:
                rg_params = {
                    "pattern": "Table",
                    "paths": [str(base_data_dir)],
                    "file_patterns": ["*.html"],
                    "case_sensitive": False,
                    "files_with_matches": True,
                    "max_total_matches": 10
                }
                debug_print(f"Ripgrep params: {rg_params}")
                rg_response = await client.ripgrep_search(**rg_params)
                debug_print(f"Ripgrep response: {rg_response}")

                if isinstance(rg_response, dict) and rg_response.get("data") and rg_response["data"].get("matches"):
                    files_found_rg = [match.get("path") for match in rg_response["data"]["matches"] if match.get("path")]
                    files_found_rg = sorted(list(set(filter(None, files_found_rg))))
                    console.print(f"[green]Ripgrep found {len(files_found_rg)} file(s): {', '.join(files_found_rg)}[/green]")
                elif isinstance(rg_response, dict) and rg_response.get("data") and not rg_response["data"].get("matches"):
                    console.print("[yellow]Ripgrep did not find any matching files.[/yellow]")
                else:
                    console.print("[yellow]Ripgrep search failed or returned unexpected data.[/yellow]")
                    if isinstance(rg_response, dict) and rg_response.get("error"):
                        console.print(f"[red]Ripgrep Error: {rg_response['error']}[/red]")
                    else:
                        console.print(f"[dim]Ripgrep Response: {rg_response}[/dim]")

            except Exception as e:
                console.print(f"[bold red]Error during Ripgrep search: {e}[/bold red]")
                import traceback
                console.print(traceback.format_exc())

            # Step 7.2: Run HTML Processor on found files
            if files_found_rg:
                console.print("\n[bold cyan]Step 7.2: Running HTML Processor to extract tables from found files...[/bold cyan]")
                for html_file in files_found_rg:
                    console.print(f"\n[bold]Processing file: {html_file}[/bold]")
                    await run_html_demo(
                        client,
                        demo_title=f"Demo 7.2: Tables from {Path(html_file).name}",
                        file_path=html_file,
                        extract_tables=True,
                        show_title=True, # Show title for each file
                        max_table_rows_display=3
                    )
            else:
                console.print("[yellow]Skipping HTML processing as Ripgrep found no files.[/yellow]")
                console.print() # Added newline


            # DEMO 8: Synergy with JQ (Process -> Filter/Transform)
            console.print(Rule("[bold yellow]Demo 8: Synergy (HTML Processor -> JQ)[/bold yellow]"))
            console.print(f"[italic]Extracts specific DIV elements ('div.reftoggle.show') from '{algebra_wiki_file.name}', then uses JQ to count them.[/italic]")
            console.print() # Add newline

            # Step 8.1: Run HTML Processor
            console.print("[bold cyan]Step 8.1: Running HTML Processor to extract elements ('div.reftoggle.show')...[/bold cyan]")
            html_elements_result = None
            try:
                html_params = {
                    "file_path": str(algebra_wiki_file),
                    "selector": 'div.reftoggle.show', # Select specific divs
                }
                # Call directly, not via run_html_demo, as we need the raw result data
                html_response = await client.html_process(**html_params)
                debug_print(f"HTML Processor response: {html_response}")

                if isinstance(html_response, dict) and html_response.get("data") and "elements" in html_response["data"]:
                     html_elements_result = html_response["data"]["elements"] # Get the list of element dicts
                     console.print(f"[green]HTML Processor found {len(html_elements_result)} matching elements.[/green]")
                else:
                    console.print("[yellow]HTML Processor did not return expected element data.[/yellow]")
                    if isinstance(html_response, dict) and html_response.get("error"):
                        console.print(f"[red]HTML Processor Error: {html_response['error']}[/red]")
                    else:
                        console.print(f"[dim]HTML Processor Response: {html_response}[/dim]")

            except Exception as e:
                console.print(f"[bold red]Error during HTML Processor call: {e}[/bold red]")
                import traceback
                console.print(traceback.format_exc())
            else:
                 console.print("[yellow]Skipping JQ processing as HTML Processor did not return element data.[/yellow]")
                 console.print() # Added newline

            # DEMO 9: JavaScript Rendering (Optional Feature)
            console.print(Rule("[bold yellow]Demo 9: JavaScript Rendering (Optional Feature)[/bold yellow]"))
            console.print("[italic]Fetches a JavaScript-heavy site and renders it with Playwright (if available).[/italic]")
            console.print() # Add newline
            
            # Define a JS-heavy site for the demo (GitHub's trending page has a lot of JS content)
            js_site_url = "https://github.com/trending"
            
            # First, fetch without JS rendering
            console.print("[bold cyan]Step 9.1: Fetching without JavaScript rendering...[/bold cyan]")
            await run_html_demo(
                client,
                demo_title="Demo 9.1: Without JavaScript",
                url=js_site_url,
                extract_metadata=True,
                extract_text=True,
                show_title=False,
                max_text_display_chars=300
            )
            
            # Then, fetch with JS rendering
            console.print("\n[bold cyan]Step 9.2: Fetching with JavaScript rendering...[/bold cyan]")
            console.print("[italic yellow]Note: This may take longer and requires Playwright to be installed.[/italic yellow]")
            console.print("[italic yellow]If this is the first time, the system will attempt to install required dependencies.[/italic yellow]")
            
            try:
                # Use plain print for parts that might cause Rich markup issues
                print("\nAttempting JavaScript rendering...")
                response = await client.html_process(
                    url=js_site_url,
                    extract_metadata=True,
                    extract_text=True,
                    render_js=True,
                    js_timeout=30
                )
                
                # Check if we succeeded or got an error
                if response and isinstance(response, dict):
                    if response.get("status") == "success" and "data" in response:
                        # Display results using the standard demo function
                        print("✓ JavaScript rendering successful!")
                        await display_html_results(
                            response["data"], 
                            {"url": js_site_url, "render_js": True, "extract_metadata": True, "extract_text": True},
                            max_text_display_chars=300
                        )
                    elif response.get("error"):
                        error_message = response.get("error", {}).get("message", "Unknown error")
                        print(f"JavaScript rendering failed: {error_message}")
                        print("This is often due to missing system dependencies for browser automation.")
                        print("You can install them manually with: sudo playwright install-deps")
                        print("Processing will continue without JavaScript rendering.")
                else:
                    print("Unexpected response format from HTML processor.")
            
            except Exception as e:
                print(f"Error during JavaScript rendering: {str(e)}")
                print("The demo will continue with standard HTML processing.")
                
            # DEMO 10: Static vs JavaScript Rendering Comparison
            console.print(Rule("[bold yellow]Demo 10: Static vs JavaScript Content Comparison[/bold yellow]"))
            console.print("[italic]Compares the content extracted with and without JavaScript rendering from a dynamic site.[/italic]")
            console.print() # Add newline

            dynamic_site_url = "https://news.ycombinator.com/"  # Hacker News has dynamic voting counts and interaction elements

            # First, fetch without JS rendering
            console.print("[bold cyan]Step 10.1: Extracting element counts WITHOUT JavaScript...[/bold cyan]")
            try:
                static_response = await client.html_process(
                    url=dynamic_site_url,
                    extract_metadata=True
                )
                console.print("[green]Successfully fetched static content.[/green]")
            except Exception as e:
                console.print(f"[bold red]Error fetching static content: {escape(str(e))}[/bold red]")
                static_response = None

            # Then with JS rendering
            console.print("\n[bold cyan]Step 10.2: Extracting element counts WITH JavaScript...[/bold cyan]")
            console.print("[italic yellow]Note: This may take longer as the browser needs to load and execute JavaScript.[/italic yellow]")
            try:
                js_response = await client.html_process(
                    url=dynamic_site_url,
                    extract_metadata=True,
                    render_js=True,
                    js_timeout=30
                )
                console.print("[green]Successfully fetched JavaScript-rendered content.[/green]")
            except Exception as e:
                console.print(f"[bold red]Error fetching JavaScript-rendered content: {escape(str(e))}[/bold red]")
                js_response = None

            # Compare the results
            if all(r and isinstance(r, dict) and r.get("status") == "success" for r in [static_response, js_response]):
                static_stats = static_response.get("data", {}).get("metadata", {}).get("stats", {})
                js_stats = js_response.get("data", {}).get("metadata", {}).get("stats", {})
                
                # Create comparison table
                comparison_table = Table(title="Static vs JavaScript Rendering Comparison", box=ROUNDED)
                comparison_table.add_column("Metric", style="cyan")
                comparison_table.add_column("Static HTML", style="yellow")
                comparison_table.add_column("JS Rendered", style="green")
                comparison_table.add_column("Difference", style="red")
                
                for key in sorted(set(static_stats.keys()) | set(js_stats.keys())):
                    static_val = static_stats.get(key, 0)
                    js_val = js_stats.get(key, 0)
                    diff = js_val - static_val
                    diff_str = f"+{diff}" if diff > 0 else str(diff)
                    comparison_table.add_row(key, str(static_val), str(js_val), diff_str)
                
                console.print(comparison_table)
            else:
                console.print("[yellow]Couldn't compare static and JavaScript-rendered content due to errors.[/yellow]")
                
            # DEMO 11: JavaScript Extraction → JQ Processing Demo
            console.print(Rule("[bold yellow]Demo 11: JS Rendering → JQ Processing[/bold yellow]"))
            console.print("[italic]Extracts dynamic content with JavaScript rendering, then processes it with JQ[/italic]")
            console.print() # Add newline

            # GitHub trending repositories contains dynamic star counts and other metrics
            github_url = "https://github.com/trending"

            # Step 1: Extract with JS rendering
            console.print("[bold cyan]Step 11.1: Extracting GitHub trending repositories with JavaScript...[/bold cyan]")
            try:
                repos_response = await client.html_process(
                    url=github_url,
                    selector=".Box article.Box-row",  # Selector for trending repository items
                    render_js=True
                )
                console.print("[green]Successfully extracted repository data.[/green]")
            except Exception as e:
                console.print(f"[bold red]Error extracting repository data: {escape(str(e))}[/bold red]")
                repos_response = None

            if repos_response and repos_response.get("status") == "success":
                elements = repos_response.get("data", {}).get("elements", [])
                if elements:
                    # Step 2: Transform elements to structured JSON
                    repos_data = []
                    for elem in elements:
                        # Extract repository data from the element
                        elem_text = elem.get("text", "").split("\n")
                        repo_name = next((line.strip() for line in elem_text if line.strip()), "")
                        repo_url = next((a.get("href") for a in elem.get("links", []) 
                                    if a.get("href", "").startswith("/") and a.get("href", "").count("/") == 2), "")
                        # Find description (usually second or third non-empty line)
                        repo_desc = ""
                        non_empty_lines = [line.strip() for line in elem_text if line.strip()]
                        if len(non_empty_lines) > 1:
                            repo_desc = non_empty_lines[1]
                        
                        repos_data.append({"name": repo_name, "url": repo_url, "description": repo_desc})
                    
                    console.print(f"[green]Parsed {len(repos_data)} repositories.[/green]")
                    
                    # Step 3: Process with JQ to extract names and format as markdown list
                    console.print("\n[bold cyan]Step 11.2: Processing repository data with JQ...[/bold cyan]")
                    try:
                        jq_response = await client.jq_process(
                            input_json=json.dumps(repos_data),
                            query="map(\"- [\" + .name + \"](https://github.com\" + .url + \") - \" + .description) | join(\"\\n\")"
                        )
                        console.print("[green]Successfully processed data with JQ.[/green]")
                    except Exception as e:
                        console.print(f"[bold red]Error processing data with JQ: {escape(str(e))}[/bold red]")
                        jq_response = None
                    
                    # Show the formatted output
                    if jq_response and jq_response.get("status") == "success":
                        jq_output = jq_response.get("data", {}).get("output", "")
                        console.print(Panel(jq_output, title="Trending Repositories (Formatted with JQ)", border_style="green"))
                    else:
                        console.print("[yellow]Error processing data with JQ[/yellow]")
                else:
                    console.print("[yellow]No repository elements found[/yellow]")
            else:
                console.print("[yellow]Failed to extract data from GitHub[/yellow]")
                
            # DEMO 12: Ripgrep → JavaScript Rendering → Analysis Demo
            console.print(Rule("[bold yellow]Demo 12: Ripgrep → JS Rendering → Analysis[/bold yellow]"))
            console.print("[italic]Uses Ripgrep to find URLs in markdown files, then renders them with JavaScript and analyzes the content[/italic]")
            console.print() # Add newline

            # Step 1: Use Ripgrep to find URLs in markdown files
            console.print("[bold cyan]Step 12.1: Finding URLs in markdown files with Ripgrep...[/bold cyan]")
            url_pattern = r"https?://[^\s)'\"]+"  # Simple pattern to match URLs
            try:
                # Adjust paths based on your repository structure
                search_paths = ["README.md"]
                # Check if docs directory exists
                if os.path.exists("docs"):
                    search_paths.append("docs/")
                
                rg_response = await client.ripgrep_search(
                    pattern=url_pattern,
                    paths=search_paths,
                    file_patterns=["*.md"],
                    max_total_matches=5  # Limit to 5 matches for demo
                )
                console.print("[green]Successfully searched files with Ripgrep.[/green]")
            except Exception as e:
                console.print(f"[bold red]Error searching with Ripgrep: {escape(str(e))}[/bold red]")
                rg_response = None

            if rg_response and rg_response.get("status") == "success":
                matches = rg_response.get("data", {}).get("matches", [])
                if matches:
                    urls = []
                    for match in matches:
                        # Extract the URL from the match
                        url_match = re.search(url_pattern, match.get("line_text", ""))
                        if url_match:
                            url = url_match.group(0)
                            if url not in urls:
                                urls.append(url)
                    
                    console.print(f"[green]Found {len(urls)} unique URLs in markdown files.[/green]")
                    
                    # Step 2: Use JS rendering to process each URL
                    console.print(f"\n[bold cyan]Step 12.2: Rendering {len(urls)} URLs with JavaScript...[/bold cyan]")
                    console.print("[italic yellow]Note: This may take some time as each URL needs to be rendered.[/italic yellow]")
                    
                    results = []
                    for url in urls:
                        console.print(f"Processing: {url}")
                        try:
                            js_response = await client.html_process(
                                url=url,
                                extract_metadata=True,
                                extract_text=True,
                                render_js=True,
                                js_timeout=15
                            )
                            
                            if js_response and js_response.get("status") == "success":
                                metadata = js_response.get("data", {}).get("metadata", {})
                                title = metadata.get("title", "No title")
                                text = js_response.get("data", {}).get("text", "")
                                text_sample = text[:150] + "..." if len(text) > 150 else text
                                word_count = len(text.split())
                                results.append({
                                    "url": url,
                                    "title": title,
                                    "word_count": word_count,
                                    "text_sample": text_sample
                                })
                                console.print(f"[green]✓ Successfully processed {url}[/green]")
                            else:
                                console.print(f"[yellow]Failed to process {url}[/yellow]")
                        except Exception as e:
                            console.print(f"[yellow]Error processing {url}: {escape(str(e))}[/yellow]")
                    
                    # Step 3: Display results table
                    if results:
                        console.print("\n[bold cyan]Step 12.3: Analyzing rendered content...[/bold cyan]")
                        results_table = Table(title="URL Content Analysis", box=ROUNDED)
                        results_table.add_column("URL", style="cyan", no_wrap=True)
                        results_table.add_column("Title", style="green")
                        results_table.add_column("Word Count", style="yellow")
                        results_table.add_column("Text Sample", style="white")
                        
                        for result in results:
                            url_display = result["url"]
                            if len(url_display) > 40:
                                url_display = url_display[:37] + "..."
                            
                            results_table.add_row(
                                url_display,
                                result["title"],
                                str(result["word_count"]),
                                result["text_sample"]
                            )
                        
                        console.print(results_table)
                    else:
                        console.print("[yellow]No URLs were successfully processed.[/yellow]")
                else:
                    console.print("[yellow]No URL matches found in the repository[/yellow]")
            else:
                console.print("[yellow]Error executing Ripgrep search[/yellow]")
                
            # DEMO 13: Interactive Page Navigation Demo
            console.print(Rule("[bold yellow]Demo 13: Interactive Page Navigation[/bold yellow]"))
            console.print("[italic]Demonstrates how to use JavaScript rendering to navigate through interactive page elements[/italic]")
            console.print() # Add newline
            
            # Check if we have the necessary parameters
            console.print("[bold cyan]Checking if interactive_actions parameter is supported...[/bold cyan]")
            
            try:
                # First get server info to see if it supports the interactive_actions parameter
                info = await client.info()
                supported_params = info.get("data", {}).get("html_processor_params", [])
                interactive_supported = "interactive_actions" in supported_params
                
                if not interactive_supported:
                    console.print("[yellow]The interactive_actions parameter is not supported by the current server version.[/yellow]")
                    console.print("[yellow]This demo will be skipped. Please update the HTML processor implementation to support interactive actions.[/yellow]")
                else:
                    interactive_url = "https://www.wikipedia.org/"
                    
                    console.print("\n[bold cyan]Step 13.1: Opening Wikipedia homepage and performing interactive actions...[/bold cyan]")
                    console.print("[italic yellow]Note: This requires advanced JavaScript rendering capabilities.[/italic yellow]")
                    
                    # Perform the interactive navigation
                    js_response = await client.html_process(
                        url=interactive_url,
                        extract_metadata=True,
                        extract_text=True,
                        render_js=True,
                        interactive_actions=[
                            # Click on the English Wikipedia link
                            {"action": "click", "selector": "#js-link-box-en"},
                            # Wait for navigation to complete
                            {"action": "wait_for_navigation"},
                            # Wait for specific element to appear
                            {"action": "wait_for_selector", "selector": "#searchInput"},
                            # Type in search term
                            {"action": "fill", "selector": "#searchInput", "text": "JavaScript"},
                            # Click search button 
                            {"action": "click", "selector": "button.pure-button.pure-button-primary-progressive"},
                            # Wait for page to load
                            {"action": "wait_for_load_state"}
                        ]
                    )
                    
                    if js_response and js_response.get("status") == "success":
                        # Extract title and first paragraph
                        metadata = js_response.get("data", {}).get("metadata", {})
                        text = js_response.get("data", {}).get("text", "")
                        
                        # Extract the first paragraph (approximate)
                        first_para = text.split("\n\n")[0] if text else "No text found."
                        
                        # Display the results
                        console.print(Panel(
                            f"[bold]{metadata.get('title', 'No title')}[/bold]\n\n" + 
                            first_para,
                            title="Interactive Navigation Result",
                            border_style="green"
                        ))
                    else:
                        error = js_response.get("error", {}).get("message", "Unknown error") if js_response else "No response"
                        console.print(f"[yellow]Error during interactive navigation: {error}[/yellow]")
            except Exception as e:
                console.print(f"[bold red]Error checking for interactive navigation support: {escape(str(e))}[/bold red]")
                
            # DEMO 14: CSS-in-JS Style Analysis Demo
            console.print(Rule("[bold yellow]Demo 14: CSS-in-JS Style Analysis[/bold yellow]"))
            console.print("[italic]Analyzes websites that use CSS-in-JS by extracting computed styles[/italic]")
            console.print() # Add newline
            
            # Check if the extract_computed_styles parameter is supported
            console.print("[bold cyan]Checking if extract_computed_styles parameter is supported...[/bold cyan]")
            
            try:
                # First get server info to see if it supports the extract_computed_styles parameter
                info = await client.info()
                supported_params = info.get("data", {}).get("html_processor_params", [])
                computed_styles_supported = "extract_computed_styles" in supported_params
                
                if not computed_styles_supported:
                    console.print("[yellow]The extract_computed_styles parameter is not supported by the current server version.[/yellow]")
                    console.print("[yellow]This demo will be skipped. Please update the HTML processor implementation to support computed style extraction.[/yellow]")
                else:
                    js_framework_url = "https://reactjs.org/"
                    
                    console.print("\n[bold cyan]Step 14.1: Extracting computed styles from React website...[/bold cyan]")
                    console.print("[italic yellow]Note: This requires advanced JavaScript rendering capabilities.[/italic yellow]")
                    
                    # Extract computed styles
                    js_response = await client.html_process(
                        url=js_framework_url,
                        selector="header,main,nav",  # Extract main site sections
                        extract_computed_styles=True,  # New parameter for HTML processor
                        render_js=True
                    )
                    
                    if js_response and js_response.get("status") == "success":
                        elements = js_response.get("data", {}).get("elements", [])
                        if elements:
                            # Extract color palette information
                            colors = {}
                            for elem in elements:
                                styles = elem.get("computed_styles", {})
                                color = styles.get("color")
                                bg_color = styles.get("background-color")
                                
                                if color and color != "rgba(0, 0, 0, 0)":
                                    colors[color] = colors.get(color, 0) + 1
                                if bg_color and bg_color != "rgba(0, 0, 0, 0)":
                                    colors[bg_color] = colors.get(bg_color, 0) + 1
                            
                            # Display color analysis
                            color_table = Table(title="Color Palette Analysis", box=ROUNDED)
                            color_table.add_column("Color", style="white")
                            color_table.add_column("Occurrences", style="cyan")
                            
                            for color, count in sorted(colors.items(), key=lambda x: x[1], reverse=True)[:10]:
                                color_table.add_row(color, str(count))
                            
                            console.print(color_table)
                        else:
                            console.print("[yellow]No elements with computed styles found.[/yellow]")
                    else:
                        error = js_response.get("error", {}).get("message", "Unknown error") if js_response else "No response"
                        console.print(f"[yellow]Error extracting computed styles: {error}[/yellow]")
            except Exception as e:
                console.print(f"[bold red]Error checking for computed styles support: {escape(str(e))}[/bold red]")

    except Exception as e:
        # Use standard Python print to avoid any Rich markup issues  
        print("\n========== ERROR ==========")
        print(f"Error running demo: {str(e)}")
        import traceback
        print("\nTraceback:")
        traceback.print_exc()
        print("============================")


async def run_html_demo(client, demo_title: str, show_title: bool = True, **params):
    """Run an HTML Processor demo with the given parameters and display results."""

    # --- Pop custom display parameters (not part of HtmlProcessParams) ---
    max_elements_display = params.pop("max_elements_display", 5)
    max_links_display = params.pop("max_links_display", 10)
    max_tables_display = params.pop("max_tables_display", 1)
    max_table_rows_display = params.pop("max_table_rows_display", 5)
    max_text_display_chars = params.pop("max_text_display_chars", 300)
    # ---------------------------------------------------------------------

    if show_title and demo_title:
        console.print(Rule(f"[bold yellow]{demo_title}[/bold yellow]"))

    # Show the parameters used
    console.print("[bold cyan]HTML Processor Parameters:[/bold cyan]")
    params_table = Table(show_header=False, box=None)
    params_table.add_column("Parameter", style="green")
    params_table.add_column("Value", style="white")

    for key, value in params.items():
        display_value = str(value)
        if key == "html" and value is not None:
             display_value = (value[:100] + '...' if len(value) > 100 else value)
        params_table.add_row(key, display_value)

    console.print(params_table)
    console.print()

    # Execute the HTML process call
    start_time_dt = datetime.now()
    console.print("[bold]Executing HTML process...[/bold]")

    response: Optional[Dict] = None
    error_info: Optional[Dict] = None
    result_data: Optional[Dict] = None

    try:
        response = await client.html_process(**params)
        debug_print(f"Raw response: {response}") # Debugging

        if isinstance(response, dict):
            result_data = response.get("data")
            error_info = response.get("error")
        else:
             # Handle unexpected response format
             error_info = {"code": "UNEXPECTED_RESPONSE", "message": f"Received type {type(response)} instead of dict."}


    except Exception as e:
        console.print(f"[bold red]Error during HTML process: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        error_info = {"code": "CLIENT_ERROR", "message": str(e)}


    # Calculate execution time
    execution_time_s = (datetime.now() - start_time_dt).total_seconds()
    server_exec_time = result_data.get('execution_time') if result_data else None
    display_exec_time = server_exec_time if server_exec_time is not None else execution_time_s

    # Process and display results
    console.print("[bold cyan]HTML Processor Results:[/bold cyan]")
    results_summary_table = Table(show_header=False, box=None, title="Summary")
    results_summary_table.add_column("Field", style="green")
    results_summary_table.add_column("Value", style="white")

    is_success = error_info is None and result_data is not None

    if is_success:
        results_summary_table.add_row("Status", "[bold green]Success[/bold green]")
        results_summary_table.add_row("Execution Time", f"{display_exec_time:.4f}s")

        elements = result_data.get("elements")
        tables = result_data.get("tables")
        links = result_data.get("links")
        text = result_data.get("text")
        metadata = result_data.get("metadata")

        if elements is not None:
            results_summary_table.add_row("Elements Found", str(len(elements)))
        if tables is not None:
            results_summary_table.add_row("Tables Found", str(len(tables)))
        if links is not None:
            results_summary_table.add_row("Links Found", str(len(links)))
        if text is not None:
            results_summary_table.add_row("Text Length", f"{len(text)} chars")
        if metadata is not None:
             # Simple check if metadata dict has keys
             results_summary_table.add_row("Metadata Found", "Yes" if metadata else "No")

        console.print(results_summary_table)

        # Display details for each extracted section
        if elements is not None and len(elements) > 0:
             console.print(Rule("[cyan]Extracted Elements (Sample)[/cyan]"))
             display_count = min(len(elements), max_elements_display)
             for i, elem in enumerate(elements[:display_count]):
                 elem_table = Table(title=f"Element {i+1}", box=None, show_header=False)
                 elem_table.add_column("Attr", style="dim")
                 elem_table.add_column("Value")
                 elem_table.add_row("Tag", elem.get("tag_name"))
                 # Truncate long text/html for display
                 # Escape element text before display
                 elem_text = escape(elem.get('text', ''))
                 elem_html = elem.get('html', '')
                 elem_table.add_row("Text", (elem_text[:100] + '...' if len(elem_text) > 100 else elem_text))
                 elem_table.add_row("HTML", Syntax((elem_html[:200] + '...' if len(elem_html) > 200 else elem_html), "html", theme="monokai", word_wrap=True))
                 elem_table.add_row("Attributes", Syntax(json.dumps(elem.get("attributes"), indent=2), "json", theme="monokai"))
                 # Escape paths before display
                 elem_table.add_row("XPath", escape(elem.get("xpath", "")))
                 elem_table.add_row("CSS Path", escape(elem.get("css_path", "")))
                 console.print(elem_table)
             if len(elements) > display_count:
                 console.print(f"[dim]... and {len(elements) - display_count} more elements not shown.[/dim]")

        if tables is not None and len(tables) > 0:
             console.print(Rule("[cyan]Extracted Tables (Sample)[/cyan]"))
             display_table_count = min(len(tables), max_tables_display)
             for i, table_data in enumerate(tables[:display_table_count]):
                 if not table_data: 
                     continue # Skip empty tables
                 # Handle cases where tables might not have a consistent header/row structure
                 if not table_data[0]: 
                     continue # Skip if header row is empty
                 header = table_data[0]
                 # Create a table with a proper box object
                 table_view = Table(title=f"Table {i+1}", show_header=True, box=ROUNDED)
                 for col_header in header:
                     table_view.add_column(str(col_header))
                 display_row_count = min(len(table_data), max_table_rows_display)
                 for row in table_data[1:display_row_count]:
                     # Pad row with empty strings if shorter than header
                     padded_row = [str(cell) for cell in row] + [''] * (len(header) - len(row))
                     table_view.add_row(*padded_row[:len(header)]) # Ensure only header number of cells
                 console.print(table_view)
                 if len(table_data) > display_row_count:
                      console.print(f"[dim]... and {len(table_data) - display_row_count} more rows not shown.[/dim]")
             if len(tables) > display_table_count:
                 console.print(f"[dim]... and {len(tables) - display_table_count} more tables not shown.[/dim]")

        if links is not None and len(links) > 0:
             console.print(Rule("[cyan]Extracted Links (Sample)[/cyan]"))
             link_table = Table(title="Links", box=ROUNDED)
             link_table.add_column("Text", style="white")
             link_table.add_column("HREF", style="cyan")
             link_table.add_column("Resolved HREF", style="magenta")
             display_count = min(len(links), max_links_display)
             for link in links[:display_count]:
                 # Escape link text before adding to table row
                 link_table.add_row(
                     escape(link.get('text', '')),
                     link.get('href', ''),
                     link.get('href_resolved', '[dim]N/A[/dim]')
                 )
             console.print(link_table)
             if len(links) > display_count:
                 console.print(f"[dim]... and {len(links) - display_count} more links not shown.[/dim]")

        if text is not None:
             console.print(Rule("[cyan]Extracted Text (Sample)[/cyan]"))
             display_text = (text[:max_text_display_chars] + '...' if len(text) > max_text_display_chars else text)
             console.print(Syntax(display_text, "text", theme="default", word_wrap=True))
             if len(text) > max_text_display_chars:
                  console.print(f"[dim]... text truncated ({len(text)} total chars).[/dim]")

        if metadata is not None:
             console.print(Rule("[cyan]Extracted Metadata[/cyan]"))
             # Create a copy of metadata to modify for display (truncating large values)
             display_metadata = metadata.copy() if isinstance(metadata, dict) else metadata
             
             # Truncate large values (like base64 encoded images)
             if isinstance(display_metadata, dict):
                 # Handle icon specifically which is often base64 encoded
                 if 'links' in display_metadata and isinstance(display_metadata['links'], dict):
                     if 'icon' in display_metadata['links']:
                         icon_val = display_metadata['links']['icon']
                         if isinstance(icon_val, str) and len(icon_val) > 100:
                             display_metadata['links']['icon'] = icon_val[:100] + '... [truncated base64 data]'
                 
                 # Process any other large string values recursively
                 def truncate_large_values(obj, max_length=500):
                     if isinstance(obj, dict):
                         for key, value in obj.items():
                             if isinstance(value, (dict, list)):
                                 truncate_large_values(value, max_length)
                             elif isinstance(value, str) and len(value) > max_length:
                                 obj[key] = value[:max_length] + f'... [truncated, total {len(value)} chars]'
                     elif isinstance(obj, list):
                         for i, item in enumerate(obj):
                             if isinstance(item, (dict, list)):
                                 truncate_large_values(item, max_length)
                             elif isinstance(item, str) and len(item) > max_length:
                                 obj[i] = item[:max_length] + f'... [truncated, total {len(item)} chars]'
                 
                 # Apply the truncation function
                 truncate_large_values(display_metadata)
                 
                 # Pretty print the truncated metadata dictionary
                 metadata_str = json.dumps(display_metadata, indent=2, ensure_ascii=False)
                 console.print(Syntax(metadata_str, "json", theme="monokai", line_numbers=False, word_wrap=True))

    else:
        results_summary_table.add_row("Status", "[bold red]Failed[/bold red]")
        if error_info:
            results_summary_table.add_row("Error Code", error_info.get('code', 'N/A'))
            results_summary_table.add_row("Message", error_info.get('message', 'Unknown error'))
            if error_info.get('details'):
                 # Try to pretty print if it's JSON, otherwise show raw
                 details = error_info['details']
                 try:
                      details_str = json.dumps(json.loads(details), indent=2) if isinstance(details, str) else json.dumps(details, indent=2)
                      results_summary_table.add_row("Details", Syntax(details_str, "json", theme="monokai", word_wrap=True))
                 except (json.JSONDecodeError, TypeError):
                      results_summary_table.add_row("Details", str(details))

        else:
             results_summary_table.add_row("Error", "[dim]Unknown error details.[/dim]")
             if response is not None:
                  results_summary_table.add_row("Raw Response", str(response))

        console.print(results_summary_table)

    console.print() # Add space between demos


async def display_html_results(result_data, params, **display_params):
    """Display HTML processor results without redoing the request.
    Similar to run_html_demo but takes pre-fetched results."""
    
    # --- Pop custom display parameters (not part of HtmlProcessParams) ---
    max_elements_display = display_params.get("max_elements_display", 5)
    max_links_display = display_params.get("max_links_display", 10)
    max_tables_display = display_params.get("max_tables_display", 1)
    max_table_rows_display = display_params.get("max_table_rows_display", 5)
    max_text_display_chars = display_params.get("max_text_display_chars", 300)
    # ---------------------------------------------------------------------
    
    # Show the parameters used
    console.print("[bold cyan]HTML Processor Parameters:[/bold cyan]")
    params_table = Table(show_header=False, box=None)
    params_table.add_column("Parameter", style="green")
    params_table.add_column("Value", style="white")

    for key, value in params.items():
        display_value = str(value)
        if key == "html" and value is not None:
             display_value = (value[:100] + '...' if len(value) > 100 else value)
        params_table.add_row(key, display_value)

    console.print(params_table)
    console.print()
    
    # Display results
    console.print("[bold cyan]HTML Processor Results:[/bold cyan]")
    results_summary_table = Table(show_header=False, box=None, title="Summary")
    results_summary_table.add_column("Field", style="green")
    results_summary_table.add_column("Value", style="white")

    server_exec_time = result_data.get('execution_time')
    
    results_summary_table.add_row("Status", "[bold green]Success[/bold green]")
    if server_exec_time is not None:
        results_summary_table.add_row("Execution Time", f"{server_exec_time:.4f}s")

    elements = result_data.get("elements")
    tables = result_data.get("tables")
    links = result_data.get("links")
    text = result_data.get("text")
    metadata = result_data.get("metadata")

    if elements is not None:
        results_summary_table.add_row("Elements Found", str(len(elements)))
    if tables is not None:
        results_summary_table.add_row("Tables Found", str(len(tables)))
    if links is not None:
        results_summary_table.add_row("Links Found", str(len(links)))
    if text is not None:
        results_summary_table.add_row("Text Length", f"{len(text)} chars")
    if metadata is not None:
         # Simple check if metadata dict has keys
         results_summary_table.add_row("Metadata Found", "Yes" if metadata else "No")

    console.print(results_summary_table)

    # Display details for each extracted section
    if elements is not None and len(elements) > 0:
         console.print(Rule("[cyan]Extracted Elements (Sample)[/cyan]"))
         display_count = min(len(elements), max_elements_display)
         for i, elem in enumerate(elements[:display_count]):
             elem_table = Table(title=f"Element {i+1}", box=None, show_header=False)
             elem_table.add_column("Attr", style="dim")
             elem_table.add_column("Value")
             elem_table.add_row("Tag", elem.get("tag_name"))
             # Truncate long text/html for display
             # Escape element text before display
             elem_text = escape(elem.get('text', ''))
             elem_html = elem.get('html', '')
             elem_table.add_row("Text", (elem_text[:100] + '...' if len(elem_text) > 100 else elem_text))
             elem_table.add_row("HTML", Syntax((elem_html[:200] + '...' if len(elem_html) > 200 else elem_html), "html", theme="monokai", word_wrap=True))
             elem_table.add_row("Attributes", Syntax(json.dumps(elem.get("attributes"), indent=2), "json", theme="monokai"))
             # Escape paths before display
             elem_table.add_row("XPath", escape(elem.get("xpath", "")))
             elem_table.add_row("CSS Path", escape(elem.get("css_path", "")))
             console.print(elem_table)
         if len(elements) > display_count:
             console.print(f"[dim]... and {len(elements) - display_count} more elements not shown.[/dim]")

    if tables is not None and len(tables) > 0:
         console.print(Rule("[cyan]Extracted Tables (Sample)[/cyan]"))
         display_table_count = min(len(tables), max_tables_display)
         for i, table_data in enumerate(tables[:display_table_count]):
             if not table_data: 
                 continue # Skip empty tables
             # Handle cases where tables might not have a consistent header/row structure
             if not table_data[0]: 
                 continue # Skip if header row is empty
             header = table_data[0]
             # Create a table with a proper box object
             table_view = Table(title=f"Table {i+1}", show_header=True, box=ROUNDED)
             for col_header in header:
                 table_view.add_column(str(col_header))
             display_row_count = min(len(table_data), max_table_rows_display)
             for row in table_data[1:display_row_count]:
                 # Pad row with empty strings if shorter than header
                 padded_row = [str(cell) for cell in row] + [''] * (len(header) - len(row))
                 table_view.add_row(*padded_row[:len(header)]) # Ensure only header number of cells
             console.print(table_view)
             if len(table_data) > display_row_count:
                  console.print(f"[dim]... and {len(table_data) - display_row_count} more rows not shown.[/dim]")
         if len(tables) > display_table_count:
             console.print(f"[dim]... and {len(tables) - display_table_count} more tables not shown.[/dim]")

    if links is not None and len(links) > 0:
         console.print(Rule("[cyan]Extracted Links (Sample)[/cyan]"))
         link_table = Table(title="Links", box=ROUNDED)
         link_table.add_column("Text", style="white")
         link_table.add_column("HREF", style="cyan")
         link_table.add_column("Resolved HREF", style="magenta")
         display_count = min(len(links), max_links_display)
         for link in links[:display_count]:
             # Escape link text before adding to table row
             link_table.add_row(
                 escape(link.get('text', '')),
                 link.get('href', ''),
                 link.get('href_resolved', '[dim]N/A[/dim]')
             )
         console.print(link_table)
         if len(links) > display_count:
             console.print(f"[dim]... and {len(links) - display_count} more links not shown.[/dim]")

    if text is not None:
         console.print(Rule("[cyan]Extracted Text (Sample)[/cyan]"))
         display_text = (text[:max_text_display_chars] + '...' if len(text) > max_text_display_chars else text)
         console.print(Syntax(display_text, "text", theme="default", word_wrap=True))
         if len(text) > max_text_display_chars:
              console.print(f"[dim]... text truncated ({len(text)} total chars).[/dim]")

    if metadata is not None:
         console.print(Rule("[cyan]Extracted Metadata[/cyan]"))
         # Create a copy of metadata to modify for display (truncating large values)
         display_metadata = metadata.copy() if isinstance(metadata, dict) else metadata
         
         # Truncate large values (like base64 encoded images)
         if isinstance(display_metadata, dict):
             # Handle icon specifically which is often base64 encoded
             if 'links' in display_metadata and isinstance(display_metadata['links'], dict):
                 if 'icon' in display_metadata['links']:
                     icon_val = display_metadata['links']['icon']
                     if isinstance(icon_val, str) and len(icon_val) > 100:
                         display_metadata['links']['icon'] = icon_val[:100] + '... [truncated base64 data]'
             
             # Process any other large string values recursively
             def truncate_large_values(obj, max_length=500):
                 if isinstance(obj, dict):
                     for key, value in obj.items():
                         if isinstance(value, (dict, list)):
                             truncate_large_values(value, max_length)
                         elif isinstance(value, str) and len(value) > max_length:
                             obj[key] = value[:max_length] + f'... [truncated, total {len(value)} chars]'
                 elif isinstance(obj, list):
                     for i, item in enumerate(obj):
                         if isinstance(item, (dict, list)):
                             truncate_large_values(item, max_length)
                         elif isinstance(item, str) and len(item) > max_length:
                             obj[i] = item[:max_length] + f'... [truncated, total {len(item)} chars]'
             
             # Apply the truncation function
             truncate_large_values(display_metadata)
             
             # Pretty print the truncated metadata dictionary
             metadata_str = json.dumps(display_metadata, indent=2, ensure_ascii=False)
             console.print(Syntax(metadata_str, "json", theme="monokai", line_numbers=False, word_wrap=True))

    console.print() # Add space between demos


# Add a new function to run just the JavaScript demo
async def run_js_demo(client, algebra_wiki_file, base_data_dir):
    """Run only the JavaScript rendering demo (Demo 9)."""
    # DEMO 9: JavaScript Rendering (Optional Feature)
    console.print(Rule("[bold yellow]Demo 9: JavaScript Rendering (Optional Feature)[/bold yellow]"))
    console.print("[italic]Fetches a JavaScript-heavy site and renders it with Playwright (if available).[/italic]")
    console.print() # Add newline
    
    # Define a JS-heavy site for the demo (GitHub's trending page has a lot of JS content)
    js_site_url = "https://github.com/trending"
    
    # First, fetch without JS rendering
    console.print("[bold cyan]Step 9.1: Fetching without JavaScript rendering...[/bold cyan]")
    await run_html_demo(
        client,
        demo_title="Demo 9.1: Without JavaScript",
        url=js_site_url,
        extract_metadata=True,
        extract_text=True,
        show_title=False,
        max_text_display_chars=300
    )
    
    # Then, fetch with JS rendering
    console.print("\n[bold cyan]Step 9.2: Fetching with JavaScript rendering...[/bold cyan]")
    console.print("[italic yellow]Note: This may take longer and requires Playwright to be installed.[/italic yellow]")
    console.print("[italic yellow]If this is the first time, the system will attempt to install required dependencies.[/italic yellow]")
    
    try:
        # Use plain print for parts that might cause Rich markup issues
        print("\nAttempting JavaScript rendering...")
        response = await client.html_process(
            url=js_site_url,
            extract_metadata=True,
            extract_text=True,
            render_js=True,
            js_timeout=30
        )
        
        # Check if we succeeded or got an error
        if response and isinstance(response, dict):
            if response.get("status") == "success" and "data" in response:
                # Display results using the standard demo function
                print("✓ JavaScript rendering successful!")
                await display_html_results(
                    response["data"], 
                    {"url": js_site_url, "render_js": True, "extract_metadata": True, "extract_text": True},
                    max_text_display_chars=300
                )
            elif response.get("error"):
                error_message = response.get("error", {}).get("message", "Unknown error")
                print(f"JavaScript rendering failed: {error_message}")
                print("This is often due to missing system dependencies for browser automation.")
                print("You can install them manually with: sudo playwright install-deps")
                print("Processing will continue without JavaScript rendering.")
        else:
            print("Unexpected response format from HTML processor.")
    
    except Exception as e:
        print(f"Error during JavaScript rendering: {str(e)}")
        print("The demo will continue with standard HTML processing.")


if __name__ == "__main__":
    if "--debug" in sys.argv:
        DEBUG = True
        debug_print("Debug mode enabled")

    # Ensure example data directory and files exist before running
    example_html_dir = Path("tsap_example_data/html")
    if not example_html_dir.is_dir():
         console.print(f"[bold red]Error:[/bold red] Example HTML directory not found: '{example_html_dir}'")
         console.print("Please ensure the 'tsap_example_data/html' directory exists and contains the required files.")
         sys.exit(1)

    # Check specific files
    required_files_main = [
         example_html_dir / "Algebraic_topology_Wikipedia.html",
         example_html_dir / "10q_filing.html"
    ]
    missing_files_main = [f for f in required_files_main if not f.exists()]
    if missing_files_main:
          console.print(f"[bold red]Error:[/bold red] Required example HTML files missing: {', '.join(map(str, missing_files_main))}")
          sys.exit(1)


    try:
        asyncio.run(html_processor_demo())
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
    except NameError as e:
        if 'MCPClient' in str(e):
             # Import error already handled
             pass
        else:
            console.print(f"[bold red]Unhandled NameError: {escape(str(e))}[/bold red]")
            import traceback
            console.print(escape(traceback.format_exc()))
    except FileNotFoundError as e:
        console.print(f"[bold red]File Not Found Error: {escape(str(e))}[/bold red]")
        console.print("Ensure the required example files and directories exist.")
    except Exception as e:
        # Use standard Python print to avoid any Rich markup issues
        print("\n========== ERROR ==========")
        print(f"Unhandled error: {str(e)}")
        import traceback
        print("\nTraceback:")
        traceback.print_exc()
        print("============================")
