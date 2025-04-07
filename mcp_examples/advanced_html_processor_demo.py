#!/usr/bin/env python3
"""
Advanced HTML Processor Demo (MCP Tools Version)

This script demonstrates the comprehensive features of the HTML Processor
integration in TSAP MCP, including parsing local files and URLs, extracting
various content types (text, links, tables, metadata), using selectors,
and demonstrating synergy with Ripgrep and JQ.
"""
import asyncio
import sys
import os
import json
from datetime import datetime
from pathlib import Path
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.console import Console
from rich.rule import Rule
from rich.box import ROUNDED
from rich.markup import escape

# Import proper MCP client modules
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# --- Path Setup --- #
# Add the project root and src directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) # Go up one level from mcp_examples/
src_path = os.path.join(project_root, 'src')

# Add project root if not already present (handles running from root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add src path if not already present
if src_path not in sys.path:
    sys.path.insert(0, src_path) # Prepend src directory

# Add examples path itself
if script_dir not in sys.path:
     sys.path.insert(1, script_dir) # Insert after src

# --- End Path Setup ---

console = Console()

# Add some basic debugging
DEBUG = False

def debug_print(msg):
    """Print debug message if DEBUG is True."""
    if DEBUG:
        console.print(f"[dim][DEBUG] {msg}[/dim]")

async def html_processor_demo():
    """Demonstrate HTML Processor's advanced features using MCP tools."""
    console.print(Panel.fit(
        "[bold blue]TSAP MCP HTML Processor Advanced Features Demo[/bold blue]",
        subtitle="Parsing and extracting data from HTML content using MCP tools"
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

    # Path to the proxy script
    proxy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_proxy.py")
    debug_print(f"Proxy path: {proxy_path}")
    
    # Configure server parameters for stdio connection to our proxy
    server_params = StdioServerParameters(
        command="python",
        args=[proxy_path],
        env={"MCP_DEBUG": "1"}  # Enable debug logging
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
                
                # Find the html_process tool, ripgrep tool, and jq tool
                html_process_tool = next((t for t in tools if t.name == "html_process"), None)
                ripgrep_tool = next((t for t in tools if t.name == "ripgrep_search"), None)
                jq_tool = next((t for t in tools if t.name == "jq_query"), None)
                
                if not html_process_tool:
                    console.print("[bold red]Error: html_process tool not found![/bold red]")
                    return

                # --- Initial info check ---
                info_tool = next((t for t in tools if t.name == "info"), None)
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
                            info_data = json.loads(info_text)  # noqa: F841
                            console.print("[green]Initial server info check successful.[/green]")
                        else:
                            console.print("[yellow]Info call succeeded but returned no readable content.[/yellow]")
                    except Exception as e:
                        console.print(f"[bold red]Error during info check:[/bold red] {str(e)}")
                        debug_print(f"Info check error details: {e}")
                # ------------------------------

                # Check if we should only run a specific demo
                if "--js-only" in sys.argv:
                    console.print("[yellow]Running only JavaScript demo (Demo 9)...[/yellow]")
                    # Skip to Demo 9
                    await run_js_demo(session, html_process_tool, algebra_wiki_file, base_data_dir)
                    return

                # DEMO 1: Basic File Metadata Extraction
                console.print(Rule("[bold yellow]Demo 1: Basic File Metadata (Wikipedia Page)[/bold yellow]"))
                console.print(f"[italic]Extracts metadata (title, H1, stats) from '{algebra_wiki_file.name}'.[/italic]")
                console.print() # Add newline
                await run_html_demo(
                    session,
                    html_process_tool,
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
                    session,
                    html_process_tool,
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
                    session,
                    html_process_tool,
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
                    session,
                    html_process_tool,
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
                    session,
                    html_process_tool,
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
                    session,
                    html_process_tool,
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

                # Step 7.1: Run Ripgrep using proper MCP pattern
                if ripgrep_tool:
                    console.print(f"[bold cyan]Step 7.1: Running Ripgrep to find relevant HTML files in '{base_data_dir}'...[/bold cyan]")
                    files_found_rg = []
                    try:
                        # Call ripgrep using MCP protocol
                        ripgrep_params = {
                            "pattern": "Table",
                            "paths": [str(base_data_dir)],
                            "file_patterns": ["*.html"],
                            "case_sensitive": False,
                            "max_total_matches": 10
                        }
                        
                        rg_result = await session.call_tool(ripgrep_tool.name, arguments=ripgrep_params)
                        
                        # Extract the text content
                        rg_text = None
                        for content in rg_result.content:
                            if content.type == "text":
                                rg_text = content.text
                                break
                        
                        if rg_text:
                            # Parse the JSON response
                            rg_response = json.loads(rg_text)
                            debug_print(f"Ripgrep response: {rg_response}")
                            
                            if rg_response and "matches" in rg_response:
                                files_found_rg = [match.get("path") for match in rg_response["matches"] if match.get("path")]
                                files_found_rg = sorted(list(set(filter(None, files_found_rg))))
                                console.print(f"[green]Ripgrep found {len(files_found_rg)} file(s): {', '.join(files_found_rg)}[/green]")
                            else:
                                console.print("[yellow]Ripgrep did not find any matching files.[/yellow]")
                        else:
                            console.print("[yellow]Ripgrep call succeeded but returned no text content.[/yellow]")
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
                                session,
                                html_process_tool,
                                demo_title=f"Demo 7.2: Tables from {Path(html_file).name}",
                                file_path=html_file,
                                extract_tables=True,
                                show_title=True, # Show title for each file
                                max_table_rows_display=3
                            )
                    else:
                        console.print("[yellow]Skipping HTML processing as Ripgrep found no files.[/yellow]")
                        console.print() # Added newline
                else:
                    console.print("[yellow]Skipping demo 7: ripgrep_search tool not available.[/yellow]")

                # DEMO 8: Synergy with JQ (Process -> Filter/Transform)
                console.print(Rule("[bold yellow]Demo 8: Synergy (HTML Processor -> JQ)[/bold yellow]"))
                console.print(f"[italic]Extracts specific DIV elements ('div.reftoggle.show') from '{algebra_wiki_file.name}', then uses JQ to count them.[/italic]")
                console.print() # Add newline

                if jq_tool:
                    # Step 8.1: Run HTML Processor to extract elements
                    console.print("[bold cyan]Step 8.1: Running HTML Processor to extract elements ('div.reftoggle.show')...[/bold cyan]")
                    html_elements_result = None
                    try:
                        # Call HTML processor with MCP protocol
                        html_params = {
                            "file_path": str(algebra_wiki_file),
                            "selector": 'div.reftoggle.show' # Select specific divs
                        }
                        
                        html_result = await session.call_tool(html_process_tool.name, arguments=html_params)
                        
                        # Extract the text content
                        html_text = None
                        for content in html_result.content:
                            if content.type == "text":
                                html_text = content.text
                                break
                        
                        if html_text:
                            # Parse the JSON response
                            html_response = json.loads(html_text)
                            debug_print(f"HTML Processor response: {html_response}")
                            
                            if html_response and "elements" in html_response:
                                html_elements_result = html_response["elements"] # Get the list of element dicts
                                console.print(f"[green]HTML Processor found {len(html_elements_result)} matching elements.[/green]")
                            else:
                                console.print("[yellow]HTML Processor did not return expected element data.[/yellow]")
                        else:
                            console.print("[yellow]HTML Processor call succeeded but returned no text content.[/yellow]")
                    except Exception as e:
                        console.print(f"[bold red]Error during HTML Processor call: {e}[/bold red]")
                        import traceback
                        console.print(traceback.format_exc())
                    
                    # Step 8.2: Process with JQ tool if we have elements
                    if html_elements_result:
                        console.print("\n[bold cyan]Step 8.2: Using JQ to process and analyze the HTML elements...[/bold cyan]")
                        try:
                            # Convert the elements to JSON for JQ processing
                            elements_json = json.dumps(html_elements_result)
                            
                            # Call JQ using MCP protocol
                            jq_params = {
                                "input_json": elements_json,
                                "query": "length" # Count the elements
                            }
                            
                            jq_result = await session.call_tool(jq_tool.name, arguments=jq_params)
                            
                            # Extract the text content
                            jq_text = None
                            for content in jq_result.content:
                                if content.type == "text":
                                    jq_text = content.text
                                    break
                            
                            if jq_text:
                                try:
                                    # Parse the JSON response
                                    jq_response = json.loads(jq_text)
                                    console.print(f"[green]JQ count of elements: {jq_response.get('output', 'N/A')}[/green]")
                                except json.JSONDecodeError:
                                    console.print(f"[green]JQ count of elements: {jq_text}[/green]")
                            
                            # Another JQ query to get unique tag names
                            jq_tags_params = {
                                "input_json": elements_json,
                                "query": "map(.tag_name) | unique" # Extract unique tag names
                            }
                            
                            jq_tags_result = await session.call_tool(jq_tool.name, arguments=jq_tags_params)
                            
                            # Extract the text content
                            jq_tags_text = None
                            for content in jq_tags_result.content:
                                if content.type == "text":
                                    jq_tags_text = content.text
                                    break
                            
                            if jq_tags_text:
                                try:
                                    # Parse the JSON response
                                    jq_tags_response = json.loads(jq_tags_text)
                                    console.print(f"[green]Unique tag names in elements: {jq_tags_response.get('output', 'N/A')}[/green]")
                                except json.JSONDecodeError:
                                    console.print(f"[green]Unique tag names in elements: {jq_tags_text}[/green]")
                            
                        except Exception as e:
                            console.print(f"[bold red]Error during JQ processing: {e}[/bold red]")
                            import traceback
                            console.print(traceback.format_exc())
                    else:
                        console.print("[yellow]Skipping JQ processing as HTML Processor did not return element data.[/yellow]")
                        console.print() # Added newline
                else:
                    console.print("[yellow]Skipping demo 8: jq_query tool not available.[/yellow]")

                # DEMO 9: JavaScript Rendering (Optional Feature)
                console.print(Rule("[bold yellow]Demo 9: JavaScript Rendering (Optional Feature)[/bold yellow]"))
                console.print("[italic]Fetches a JavaScript-heavy site and renders it with Playwright (if available).[/italic]")
                console.print() # Add newline
                
                # Define a JS-heavy site for the demo (GitHub's trending page has a lot of JS content)
                js_site_url = "https://github.com/trending"
                
                # First, fetch without JS rendering
                console.print("[bold cyan]Step 9.1: Fetching without JavaScript rendering...[/bold cyan]")
                await run_html_demo(
                    session,
                    html_process_tool,
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
                    
                    # Call HTML processor with JS rendering enabled
                    js_params = {
                        "url": js_site_url,
                        "extract_metadata": True,
                        "extract_text": True,
                        "render_js": True,
                        "js_timeout": 30
                    }
                    
                    js_result = await session.call_tool(html_process_tool.name, arguments=js_params)
                    
                    # Extract the text content
                    js_text = None
                    for content in js_result.content:
                        if content.type == "text":
                            js_text = content.text
                            break
                    
                    # Check if we succeeded
                    if js_text:
                        # Parse the JSON response
                        try:
                            response = json.loads(js_text)
                            # Display results using the standard demo function
                            print("✓ JavaScript rendering successful!")
                            await display_html_results(
                                response, 
                                {"url": js_site_url, "render_js": True, "extract_metadata": True, "extract_text": True},
                                max_text_display_chars=300
                            )
                        except json.JSONDecodeError:
                            print(f"✓ JavaScript rendering returned non-JSON response: {js_text[:100]}...")
                    else:
                        print("JavaScript rendering failed or returned no data.")
                        print("This is often due to missing system dependencies for browser automation.")
                        print("You can install them manually with: sudo playwright install-deps")
                        print("Processing will continue without JavaScript rendering.")
                
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
                static_response = None
                try:
                    # Call HTML processor for static content
                    static_params = {
                        "url": dynamic_site_url,
                        "extract_metadata": True
                    }
                    
                    static_result = await session.call_tool(html_process_tool.name, arguments=static_params)
                    
                    # Extract the text content
                    static_text = None
                    for content in static_result.content:
                        if content.type == "text":
                            static_text = content.text
                            break
                    
                    if static_text:
                        # Parse the JSON response
                        static_response = json.loads(static_text)
                        console.print("[green]Successfully fetched static content.[/green]")
                    else:
                        console.print("[yellow]Static content request returned no text content.[/yellow]")
                except Exception as e:
                    console.print(f"[bold red]Error fetching static content: {escape(str(e))}[/bold red]")
                    static_response = None

                # Then with JS rendering
                console.print("\n[bold cyan]Step 10.2: Extracting element counts WITH JavaScript...[/bold cyan]")
                console.print("[italic yellow]Note: This may take longer as the browser needs to load and execute JavaScript.[/italic yellow]")
                js_response = None
                try:
                    # Call HTML processor with JS rendering enabled
                    js_params = {
                        "url": dynamic_site_url,
                        "extract_metadata": True,
                        "render_js": True,
                        "js_timeout": 30
                    }
                    
                    js_result = await session.call_tool(html_process_tool.name, arguments=js_params)
                    
                    # Extract the text content
                    js_text = None
                    for content in js_result.content:
                        if content.type == "text":
                            js_text = content.text
                            break
                    
                    if js_text:
                        # Parse the JSON response
                        js_response = json.loads(js_text)
                        console.print("[green]Successfully fetched JavaScript-rendered content.[/green]")
                    else:
                        console.print("[yellow]JavaScript-rendered content request returned no text content.[/yellow]")
                except Exception as e:
                    console.print(f"[bold red]Error fetching JavaScript-rendered content: {escape(str(e))}[/bold red]")
                    js_response = None

                # Compare the results
                if static_response and js_response:
                    static_stats = static_response.get("metadata", {}).get("stats", {})
                    js_stats = js_response.get("metadata", {}).get("stats", {})
                    
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
                        
                        # Format numeric values for better display
                        if isinstance(static_val, (int, float)) and isinstance(js_val, (int, float)):
                            diff_display = f"{diff:+d}" if isinstance(diff, int) else f"{diff:+.2f}"
                            # Add arrows to indicate increase/decrease
                            if diff > 0:
                                diff_display = f"↑ {diff_display}"
                            elif diff < 0:
                                diff_display = f"↓ {diff_display}"
                            else:
                                diff_display = f"= {diff_display}"
                        else:
                            # Handle non-numeric differences
                            diff_display = "Different" if static_val != js_val else "Same"
                            
                        comparison_table.add_row(
                            key, 
                            str(static_val), 
                            str(js_val),
                            diff_display
                        )
                        
                    console.print(comparison_table)
                    
                    # Add summary observations
                    console.print("\n[bold]Summary Observations:[/bold]")
                    element_diff = js_stats.get("elements", 0) - static_stats.get("elements", 0)
                    script_diff = js_stats.get("scripts", 0) - static_stats.get("scripts", 0)
                    
                    observations = []
                    if element_diff > 0:
                        observations.append(f"JavaScript rendering added {element_diff} more DOM elements.")
                    if script_diff != 0:
                        observations.append(f"JavaScript rendering {'added' if script_diff > 0 else 'removed'} {abs(script_diff)} script elements.")
                    
                    if observations:
                        for obs in observations:
                            console.print(f"• {obs}")
                    else:
                        console.print("• No significant differences detected between static and JavaScript rendering.")
                else:
                    console.print("[yellow]Cannot compare results as one or both requests failed.[/yellow]")
    
    except Exception as e:
        console.print(f"[bold red]Error running demo: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())

async def run_html_demo(session, html_process_tool, demo_title: str, show_title: bool = True, **params):
    """Run an HTML processor demo with the given parameters using MCP tools."""
    # Extract display parameters that are not for the API
    max_links_display = params.pop("max_links_display", 10)
    max_table_rows_display = params.pop("max_table_rows_display", 5)
    max_text_display_chars = params.pop("max_text_display_chars", 500)
    max_elements_display = params.pop("max_elements_display", 10)
    
    if show_title:
        console.print(f"[bold cyan]{demo_title}[/bold cyan]")
    
    # Show the parameters
    console.print("[bold]HTML Processing Parameters:[/bold]")
    params_table = Table(show_header=False, box=None)
    params_table.add_column("Parameter", style="green")
    params_table.add_column("Value", style="white")
    
    for key, value in params.items():
        params_table.add_row(key, str(value))
    
    console.print(params_table)
    console.print()
    
    # Execute the HTML processing
    start_time = datetime.now()
    console.print("[bold]Executing HTML processing...[/bold]")
    
    try:
        # Call the HTML process tool using MCP protocol
        result = await session.call_tool(html_process_tool.name, arguments=params)
        
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
                
                # Calculate execution time
                execution_time = (datetime.now() - start_time).total_seconds()
                
                # Process and display results
                console.print(f"[green]Processing completed in {execution_time:.2f} seconds[/green]")
                
                # Display results with custom display parameters
                display_params = {
                    "max_links_display": max_links_display,
                    "max_table_rows_display": max_table_rows_display,
                    "max_text_display_chars": max_text_display_chars,
                    "max_elements_display": max_elements_display
                }
                await display_html_results(response, params, **display_params)
            except json.JSONDecodeError:
                console.print("[bold red]Failed to parse response as JSON[/bold red]")
                console.print(f"Raw response: {result_text[:200]}...")
        else:
            console.print("[bold red]No text content in response[/bold red]")
            
    except Exception as e:
        console.print(f"[bold red]Error during HTML processing: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
    
    console.print("\n") # Add spacing between demos

async def display_html_results(result_data, params, **display_params):
    """Display HTML processing results using rich formatting."""
    if not result_data:
        console.print("[yellow]No results returned from HTML processor.[/yellow]")
        return
        
    # Extract display parameters
    max_links_display = display_params.get("max_links_display", 10)
    max_table_rows_display = display_params.get("max_table_rows_display", 5)
    max_text_display_chars = display_params.get("max_text_display_chars", 500)
    max_elements_display = display_params.get("max_elements_display", 10)
    
    # Define a helper function to truncate large values for display
    def truncate_large_values(obj, max_length=500):
        """Truncate string values that are too large for display."""
        if isinstance(obj, str) and len(obj) > max_length:
            return obj[:max_length] + "..."
        elif isinstance(obj, dict):
            return {k: truncate_large_values(v, max_length) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [truncate_large_values(i, max_length) for i in obj]
        else:
            return obj

    # Display metadata if available and requested
    if params.get("extract_metadata") and "metadata" in result_data:
        metadata = result_data["metadata"]
        console.print("\n[bold cyan]Page Metadata:[/bold cyan]")
        
        # Display title and description
        if "title" in metadata:
            console.print(f"[bold]Title:[/bold] {metadata['title']}")
        if "description" in metadata:
            console.print(f"[bold]Description:[/bold] {metadata['description']}")
            
        # Display other metadata
        if "stats" in metadata:
            stats_table = Table(title="Page Statistics")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="yellow")
            
            for key, value in metadata["stats"].items():
                stats_table.add_row(key, str(value))
                
            console.print(stats_table)
            
    # Display links if available and requested
    if params.get("extract_links") and "links" in result_data:
        links = result_data["links"]
        link_count = len(links)
        
        console.print(f"\n[bold cyan]Links ({link_count} found):[/bold cyan]")
        
        if link_count > 0:
            links_table = Table(title=f"Page Links (showing {min(link_count, max_links_display)} of {link_count})")
            links_table.add_column("Text", style="green")
            links_table.add_column("URL", style="yellow")
            links_table.add_column("Type", style="cyan")
            
            for i, link in enumerate(links[:max_links_display]):
                # Truncate link text if too long
                link_text = link.get("text", "")
                if len(link_text) > 50:
                    link_text = link_text[:47] + "..."
                    
                links_table.add_row(
                    link_text,
                    link.get("url", ""),
                    link.get("type", "")
                )
                
            console.print(links_table)
            
            if link_count > max_links_display:
                console.print(f"[dim]... and {link_count - max_links_display} more links not shown[/dim]")
    
    # Display tables if available and requested
    if params.get("extract_tables") and "tables" in result_data:
        tables = result_data["tables"]
        table_count = len(tables)
        
        console.print(f"\n[bold cyan]Tables ({table_count} found):[/bold cyan]")
        
        for i, table_data in enumerate(tables):
            # Skip if we have too many tables to display
            if i >= 3:  # Limit to 3 tables max
                console.print(f"[dim]... and {table_count - 3} more tables not shown[/dim]")
                break
                
            # Handle different possible structures of table_data
            if isinstance(table_data, dict):
                # Dictionary format with keys
                rows = table_data.get("rows", [])
                headers = table_data.get("headers", [])
                caption = table_data.get("caption", f"Table {i+1}")
            elif isinstance(table_data, list):
                # List format - assume it's directly the rows
                rows = table_data
                headers = []
                caption = f"Table {i+1}"
            else:
                console.print(f"[yellow]Unexpected table data format: {type(table_data)}[/yellow]")
                continue
            
            console.print(f"\n[bold]Table {i+1}: {caption}[/bold]")
            
            if rows:
                # Create the display table
                display_table = Table(title=f"{caption} (showing {min(len(rows), max_table_rows_display)} of {len(rows)} rows)")
                
                # Add columns based on headers or first row
                if headers:
                    for header in headers:
                        display_table.add_column(str(header), overflow="fold")
                elif rows:
                    # Use first row to determine column count
                    for j in range(len(rows[0])):
                        display_table.add_column(f"Column {j+1}", overflow="fold")
                        
                # Add rows to the table (limited)
                for row_idx, row in enumerate(rows[:max_table_rows_display]):
                    # Convert all values to strings and truncate if needed
                    str_row = [truncate_large_values(cell, 50) if cell is not None else "" for cell in row]
                    display_table.add_row(*str_row)
                    
                console.print(display_table)
                
                if len(rows) > max_table_rows_display:
                    console.print(f"[dim]... and {len(rows) - max_table_rows_display} more rows not shown[/dim]")
            else:
                console.print("[yellow]Table contains no rows[/yellow]")
    
    # Display text if available and requested
    if params.get("extract_text") and "text" in result_data:
        text = result_data["text"]
        
        console.print("\n[bold cyan]Extracted Text:[/bold cyan]")
        
        # Truncate text if it's too long
        if len(text) > max_text_display_chars:
            display_text = text[:max_text_display_chars] + "..."
            console.print(display_text)
            console.print(f"[dim]... {len(text) - max_text_display_chars} more characters not shown[/dim]")
        else:
            console.print(text)
            
    # Display elements if selector was provided
    if params.get("selector") and "elements" in result_data:
        elements = result_data["elements"]
        element_count = len(elements)
        
        console.print(f"\n[bold cyan]Selected Elements ({element_count} found with selector '{params['selector']}'):[/bold cyan]")
        
        if element_count > 0:
            for i, element in enumerate(elements[:max_elements_display]):
                console.print(f"\n[bold]Element {i+1}:[/bold]")
                console.print(f"Tag: {element.get('tag_name', 'unknown')}")
                if "attributes" in element:
                    console.print("Attributes:")
                    for attr, value in element.get("attributes", {}).items():
                        console.print(f"  {attr}: {value}")
                
                element_text = element.get("text", "")
                if element_text:
                    if len(element_text) > 100:
                        element_text = element_text[:100] + "..."
                    console.print(f"Text: {element_text}")
                
                if "html" in element:
                    html = element.get("html", "")
                    if len(html) > 100:
                        html = html[:100] + "..."
                    console.print(f"HTML: {html}")
            
            if element_count > max_elements_display:
                console.print(f"[dim]... and {element_count - max_elements_display} more elements not shown[/dim]")
        else:
            console.print("[yellow]No elements matched the selector[/yellow]")
            
    # Display raw HTML if available and requested (in collapsed syntax)
    if params.get("extract_html") and "html" in result_data:
        html = result_data["html"]
        
        console.print("\n[bold cyan]Raw HTML (collapsed):[/bold cyan]")
        
        # Display only first part of the HTML to avoid overwhelming output
        if len(html) > 300:
            console.print(Syntax(html[:300] + "...", "html", theme="monokai"))
            console.print(f"[dim]... {len(html) - 300} more characters not shown[/dim]")
        else:
            console.print(Syntax(html, "html", theme="monokai"))

async def run_js_demo(session, html_process_tool, algebra_wiki_file, base_data_dir):
    """Run the JavaScript rendering demo separately."""
    console.print(Rule("[bold yellow]JavaScript Rendering Demo[/bold yellow]"))
    console.print("[italic]Demonstrates JavaScript rendering capabilities on dynamic websites.[/italic]")
    console.print() # Add newline
    
    # Define a JS-heavy site
    dynamic_site = "https://github.com/trending"
    
    console.print(f"[bold]Fetching and rendering '{dynamic_site}' with JavaScript...[/bold]")
    try:
        # Call HTML processor with JS rendering enabled
        js_params = {
            "url": dynamic_site,
            "extract_metadata": True,
            "extract_text": True,
            "render_js": True,
            "js_timeout": 30
        }
        
        js_result = await session.call_tool(html_process_tool.name, arguments=js_params)
        
        # Extract the text content
        js_text = None
        for content in js_result.content:
            if content.type == "text":
                js_text = content.text
                break
        
        if js_text:
            # Parse the JSON response
            try:
                response = json.loads(js_text)
                console.print("[green]JavaScript rendering successful![/green]")
                
                # Display the results
                if response:
                    # Show metadata
                    if "metadata" in response:
                        metadata = response["metadata"]
                        console.print("\n[bold cyan]Page Metadata:[/bold cyan]")
                        if "title" in metadata:
                            console.print(f"[bold]Title:[/bold] {metadata['title']}")
                        if "description" in metadata:
                            console.print(f"[bold]Description:[/bold] {metadata['description']}")
                            
                        # Show statistics
                        if "stats" in metadata:
                            stats_table = Table(title="Page Statistics After JS Rendering")
                            stats_table.add_column("Metric", style="cyan")
                            stats_table.add_column("Value", style="yellow")
                            
                            for key, value in metadata["stats"].items():
                                stats_table.add_row(key, str(value))
                                
                            console.print(stats_table)
                    
                    # Show text preview
                    if "text" in response:
                        text = response["text"]
                        console.print("\n[bold cyan]Extracted Text (First 300 chars):[/bold cyan]")
                        if len(text) > 300:
                            console.print(text[:300] + "...")
                        else:
                            console.print(text)
                else:
                    console.print("[yellow]No results returned from HTML processor.[/yellow]")
            except json.JSONDecodeError:
                console.print("[bold red]Failed to parse response as JSON[/bold red]")
                console.print(f"Raw response: {js_text[:200]}...")
        else:
            console.print("[bold red]No text content in response[/bold red]")
    
    except Exception as e:
        console.print(f"[bold red]Error during JavaScript rendering: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())

if __name__ == "__main__":
    if "--debug" in sys.argv:
        DEBUG = True
        debug_print("Debug mode enabled")
    
    try:
        asyncio.run(html_processor_demo())
    except KeyboardInterrupt:
        console.print("[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Unhandled error: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc()) 