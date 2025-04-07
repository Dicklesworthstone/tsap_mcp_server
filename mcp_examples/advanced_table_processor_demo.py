#!/usr/bin/env python3
"""
Advanced Table Processor Demo (MCP Tools Version)

This script demonstrates the comprehensive features of the Table Processor
integration in TSAP MCP, including reading various formats, data transformation,
analysis, writing output, and synergy with Ripgrep and JQ.
"""
import asyncio
import os
import sys
import json
import tempfile
from datetime import datetime
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.console import Console
from rich.rule import Rule
from rich.json import JSON
from typing import Dict, Any

# Import proper MCP client modules
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

console = Console()

# Add some basic debugging
DEBUG = False

def debug_print(msg):
    """Print debug message if DEBUG is True."""
    if DEBUG:
        console.print(f"[dim][DEBUG] {msg}[/dim]")

async def table_processor_demo():
    """Demonstrate Table Processor's advanced features with example data using MCP tools."""
    console.print(Panel.fit(
        "[bold blue]TSAP MCP Table Processor Advanced Features Demo[/bold blue]",
        subtitle="Processing and analyzing tabular data with Table Processor using MCP tools"
    ))

    # Define file paths (relative to workspace root)
    products_csv = "tsap_example_data/tables/products.csv"
    orders_tsv = "tsap_example_data/tables/orders.tsv"
    sensor_json = "tsap_example_data/tables/sensor_data.json"
    inventory_csv = "tsap_example_data/tables/inventory.csv" # Using CSV instead of Excel for broader compatibility
    sales_csv = "tsap_example_data/tables/sales_data.csv"
    report_dir = "tsap_example_data/documents/" # For synergy demo

    # Check existence of essential files
    required_files = [products_csv, orders_tsv, sensor_json, inventory_csv, sales_csv]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        console.print(f"[bold red]Error:[/bold red] Required example files not found: {', '.join(missing_files)}")
        console.print("Please ensure the 'tsap_example_data/tables' directory is correctly populated.")
        return

    if not os.path.isdir(report_dir):
         console.print(f"[bold red]Error:[/bold red] Required report directory not found: {report_dir}")
         return

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
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()
                
                debug_print("Running demos...")

                # Get the list of available tools
                tools_result = await session.list_tools()
                tools = tools_result.tools
                
                # Find required tools
                process_table_tool = next((t for t in tools if t.name == "process_table"), None)
                process_json_tool = next((t for t in tools if t.name == "process_json"), None)
                ripgrep_search_tool = next((t for t in tools if t.name == "ripgrep_search"), None)
                info_tool = next((t for t in tools if t.name == "info"), None)
                
                if not process_table_tool:
                    console.print("[bold red]Error: Required process_table tool not found![/bold red]")
                    return

                # --- Add initial info check ---
                if info_tool:
                    console.print("Checking server info...")
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
                        console.print(f"[bold red]Error during info check:[/bold red] {str(e)}")
                        return
                # ------------------------------

                # DEMO 1: Basic CSV Reading and Selection
                console.print(Rule("[bold yellow]Demo 1: Basic CSV Reading & Column Selection[/bold yellow]"))
                console.print(f"[italic]Reads '{os.path.basename(products_csv)}' and selects specific columns.[/italic]\n")
                await run_table_demo(
                    session,
                    process_table_tool,
                    demo_title="Demo 1: Read CSV & Select Columns",
                    params={
                        "file_path": products_csv,
                        "transform": {
                            "columns": ["ProductName", "Category", "Price"]
                        },
                        "max_rows_return": 5 # Limit rows for display
                    },
                    show_title=False
                )

                # Let's limit our demos for debugging
                if DEBUG:
                    debug_print("DEBUG mode: Only running first demo")
                    return

                # DEMO 2: Reading TSV and Filtering Rows
                console.print(Rule("[bold yellow]Demo 2: Reading TSV & Filtering Rows[/bold yellow]"))
                console.print(f"[italic]Reads '{os.path.basename(orders_tsv)}' (tab-separated) and filters by total amount.[/italic]\n")
                await run_table_demo(
                    session,
                    process_table_tool,
                    demo_title="Demo 2: Read TSV & Filter",
                    params={
                        "file_path": orders_tsv,
                        "input_format": "tsv", # Explicitly specify TSV
                        "transform": {
                            "filter_expr": "float(row.get('Total Amount', 0)) > 100" # Filter for orders > 100
                        }
                    },
                    show_title=False
                )

                # DEMO 3: Reading JSON and Basic Analysis
                console.print(Rule("[bold yellow]Demo 3: Reading JSON & Basic Analysis[/bold yellow]"))
                console.print(f"[italic]Reads '{os.path.basename(sensor_json)}', performs basic analysis, and shows results.[/italic]\n")
                await run_table_demo(
                    session,
                    process_table_tool,
                    demo_title="Demo 3: Read JSON & Analyze",
                    params={
                        "file_path": sensor_json,
                        "input_format": "json", # Specify JSON format
                        "analyze": True # Request analysis
                    },
                    show_analysis=True, # Tell the demo runner to display analysis
                    show_title=False
                )

                # DEMO 4: Data Transformation - Computed Columns & Sorting
                console.print(Rule("[bold yellow]Demo 4: Computed Columns & Sorting[/bold yellow]"))
                console.print(f"[italic]Reads '{os.path.basename(products_csv)}', adds a discounted price, and sorts by stock.[/italic]\n")
                await run_table_demo(
                    session,
                    process_table_tool,
                    demo_title="Demo 4: Computed Columns & Sort",
                    params={
                        "file_path": products_csv,
                        "transform": {
                            "computed_columns": {
                                 # Add a 10% discount, ensure Price is float
                                "DiscountedPrice": "round(float(row.get('Price', 0)) * 0.9, 2)"
                            },
                            "columns": ["ProductName", "Price", "DiscountedPrice", "Stock"], # Select relevant columns
                            "sort_by": "Stock", # Sort by the original Stock column
                            "sort_desc": True # Sort descending (highest stock first)
                        },
                        "max_rows_return": 10
                    },
                    show_title=False
                )

                # DEMO 5: Reading with Different Options (Header, Encoding - simulated)
                console.print(Rule("[bold yellow]Demo 5: Reading with Options (e.g., Header Row)[/bold yellow]"))
                console.print(f"[italic]Reads '{os.path.basename(inventory_csv)}' demonstrating header detection (default).[/italic]\n")
                # We'll use inventory.csv which has a header
                await run_table_demo(
                    session,
                    process_table_tool,
                    demo_title="Demo 5: Reading Options",
                    params={
                        "file_path": inventory_csv,
                        # "has_header": True, # This is usually default, but explicit for demo
                        "max_rows_return": 5
                    },
                    show_title=False
                )

                # DEMO 6: Writing Output to CSV
                console.print(Rule("[bold yellow]Demo 6: Writing Output to CSV[/bold yellow]"))
                console.print(f"[italic]Reads '{os.path.basename(sales_csv)}', filters for 'North' region, and writes to a new CSV.[/italic]\n")
                # Create a temporary file for the output
                temp_output_file = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_f:
                        temp_output_file = temp_f.name

                    await run_table_demo(
                        session,
                        process_table_tool,
                        demo_title="Demo 6: Write Output CSV",
                        params={
                            "file_path": sales_csv,
                            "transform": {
                                "filter_expr": "row.get('REGION') == 'North'"
                            },
                            "output_format": "csv",
                            "output_path": temp_output_file # Specify the temp file path
                        },
                        show_output_path=True, # Ask runner to show path and content
                        show_title=False
                    )
                finally:
                    # Clean up the temporary file
                    if temp_output_file and os.path.exists(temp_output_file):
                        try:
                            os.remove(temp_output_file)
                            console.print(f"[dim]Cleaned up temporary output file: {temp_output_file}[/dim]")
                        except OSError as e:
                            console.print(f"[yellow]Warning: Could not delete temp file {temp_output_file}: {e}[/yellow]")


                # DEMO 7: Writing Output to JSON
                console.print(Rule("[bold yellow]Demo 7: Writing Output to JSON[/bold yellow]"))
                console.print(f"[italic]Reads '{os.path.basename(inventory_csv)}', transforms, and writes to a new JSON file.[/italic]\n")
                temp_json_output_file = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_f:
                        temp_json_output_file = temp_f.name

                    await run_table_demo(
                        session,
                        process_table_tool,
                        demo_title="Demo 7: Write Output JSON",
                        params={
                            "file_path": inventory_csv,
                            "transform": {
                                "filter_expr": "int(row.get('Quantity', 0)) < 50", # Low stock items
                                "columns": ["ItemCode", "Quantity", "Location"],
                                "sort_by": "Quantity"
                            },
                            "output_format": "json",
                            "output_path": temp_json_output_file,
                            "json_orient": "records" # Specify JSON orientation
                        },
                        show_output_path=True,
                        show_title=False
                    )
                finally:
                    if temp_json_output_file and os.path.exists(temp_json_output_file):
                        try:
                            os.remove(temp_json_output_file)
                            console.print(f"[dim]Cleaned up temporary output file: {temp_json_output_file}[/dim]")
                        except OSError as e:
                            console.print(f"[yellow]Warning: Could not delete temp file {temp_json_output_file}: {e}[/yellow]")


                # --- Synergy Demos ---

                # DEMO 8: Synergy with Ripgrep - Find CSVs and Process
                console.print(Rule("[bold magenta]Demo 8: Synergy - Ripgrep + Table Processor[/bold magenta]"))
                console.print(f"[italic]Uses Ripgrep to find report CSVs in '{report_dir}', then uses Table Processor to read, combine (implicitly by reading multiple), and analyze them.[/italic]\n")

                # Skip if ripgrep tool not available
                if not ripgrep_search_tool:
                    console.print("[yellow]Skipping Demo 8: ripgrep_search tool not found[/yellow]")
                else:
                    # Step 8.1: Use Ripgrep to find relevant CSV files
                    console.print("[bold cyan]Step 8.1: Running Ripgrep to find report*.csv files...[/bold cyan]")
                    report_files_found = []
                    try:
                        # Use the ripgrep_search tool
                        rg_result = await session.call_tool(
                            ripgrep_search_tool.name, 
                            arguments={
                                "pattern": r"report_.*\.csv", 
                                "paths": [report_dir],
                                "regex": True,
                                "max_total_matches": 10
                            }
                        )
                        
                        # Extract the text content
                        rg_text = None
                        for content in rg_result.content:
                            if content.type == "text":
                                rg_text = content.text
                                break
                        
                        if rg_text:
                            try:
                                # Parse the JSON response
                                rg_response = json.loads(rg_text)
                                
                                if rg_response and "matches" in rg_response:
                                    for match in rg_response["matches"]:
                                        if "path" in match:
                                            # Add unique file paths
                                            if match["path"] not in report_files_found:
                                                report_files_found.append(match["path"])
                            except json.JSONDecodeError:
                                console.print("[bold red]Failed to parse ripgrep response as JSON[/bold red]")
                        else:
                            console.print("[bold red]No text content in ripgrep response[/bold red]")

                        # Alternative approach: list directory and filter for report*.csv
                        if not report_files_found:
                            console.print(f"[dim]Falling back to directory listing in {report_dir} for 'report_*.csv'[/dim]")
                            for entry in os.listdir(report_dir):
                                if entry.startswith("report_") and entry.endswith(".csv"):
                                    full_path = os.path.join(report_dir, entry)
                                    if os.path.isfile(full_path):
                                        report_files_found.append(full_path)

                        if report_files_found:
                            console.print(f"[green]Found {len(report_files_found)} report file(s): {', '.join(report_files_found)}[/green]")

                            # Step 8.2: Use Table Processor on the found files
                            console.print("\n[bold cyan]Step 8.2: Running Table Processor to analyze the first report...[/bold cyan]")
                            
                            # Process the first found file
                            await run_table_demo(
                                session,
                                process_table_tool,
                                demo_title="Demo 8.2: Process First Found Report",
                                params={
                                    "file_path": report_files_found[0],
                                    "analyze": True,
                                    "transform": {
                                        "columns": ["ID", "Metric", "Value"]
                                    }
                                },
                                show_analysis=True,
                                show_title=False
                            )
                            
                            if len(report_files_found) > 1:
                                console.print(f"[dim]Note: Demo processed only the first file ({report_files_found[0]}). A real workflow might process all.[/dim]")
                        else:
                            console.print("[yellow]Could not find any 'report_*.csv' files in the specified directory.[/yellow]\n")

                    except Exception as e:
                        console.print(f"[bold red]Error during Ripgrep/Table Processor synergy demo: {e}[/bold red]")
                        import traceback
                        console.print(traceback.format_exc())


                # DEMO 9: Synergy with JQ - Process JSON Table, Filter with JQ
                console.print(Rule("[bold magenta]Demo 9: Synergy - Table Processor + JQ[/bold magenta]"))
                console.print("[italic]Uses Table Processor to read JSON, then pipes to JQ for complex filtering/transformation.[/italic]\n")

                # Skip if JQ tool not available
                if not process_json_tool:
                    console.print("[yellow]Skipping Demo 9: process_json tool not found[/yellow]")
                else:
                    # Step 9.1: Read JSON with Table Processor
                    console.print("[bold cyan]Step 9.1: Reading sensor data with Table Processor...[/bold cyan]")
                    
                    tp_result_jq = None
                    try:
                        result = await session.call_tool(
                            process_table_tool.name,
                            arguments={
                                "file_path": sensor_json,
                                "input_format": "json"
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
                                tp_result_jq = json.loads(result_text)
                            except json.JSONDecodeError:
                                console.print("[bold red]Failed to parse process_table response as JSON[/bold red]")
                                tp_result_jq = None
                        else:
                            console.print("[bold red]No text content in process_table response[/bold red]")
                    except Exception as e:
                        console.print(f"[bold red]Error calling process_table in Step 9.1: {e}[/bold red]")
                        tp_result_jq = None

                    if tp_result_jq and "result" in tp_result_jq:
                        # Get the table data
                        sensor_data_list = tp_result_jq["result"]
                        console.print(f"[green]Table Processor successfully read {len(sensor_data_list)} records.[/green]")
                        
                        # Convert list of dicts to JSON string for JQ input
                        jq_input_json_str = json.dumps(sensor_data_list)

                        # Step 9.2: Process with JQ using MCP tools
                        console.print("\n[bold cyan]Step 9.2: Running JQ to filter for temperature readings > 22.6...[/bold cyan]")
                        
                        try:
                            # Use the process_json tool
                            jq_result = await session.call_tool(
                                process_json_tool.name,
                                arguments={
                                    "json_string": jq_input_json_str,
                                    "query": '.[] | select(.unit == "Celsius" and .reading > 22.6)',
                                    "compact_output": True
                                }
                            )
                            
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
                                    
                                    # Display JQ results
                                    if jq_response and "result" in jq_response:
                                        console.print("[bold green]JQ Processing Results:[/bold green]")
                                        output = jq_response["result"]
                                        
                                        # Check if output is already a list/dict
                                        if isinstance(output, (list, dict)):
                                            console.print(JSON(json.dumps(output, indent=2)))
                                        elif isinstance(output, str):
                                            # Try parsing if it's a string (might be compact JSON lines)
                                            try:
                                                # Try parsing as separate JSON objects per line
                                                jq_results_list = [json.loads(line) for line in output.strip().split('\n') if line.strip()]
                                                console.print(JSON(json.dumps(jq_results_list, indent=2)))
                                            except json.JSONDecodeError:
                                                try:
                                                    # If line-by-line fails, try parsing as one JSON object
                                                    jq_result_obj = json.loads(output)
                                                    console.print(JSON(json.dumps(jq_result_obj, indent=2)))
                                                except json.JSONDecodeError:
                                                    console.print("[yellow]Could not parse JQ output string as JSON, showing raw:[/yellow]")
                                                    console.print(Syntax(output, "json", theme="monokai", word_wrap=True))
                                        else:
                                            console.print(f"[yellow]Unexpected JQ output type ({type(output)}), showing raw:[/yellow]")
                                            console.print(str(output))
                                    else:
                                        console.print("[yellow]JQ processing did not return expected output.[/yellow]")
                                except json.JSONDecodeError:
                                    console.print("[bold red]Failed to parse process_json response as JSON[/bold red]")
                                    console.print(f"Raw response: {jq_text[:200]}...")
                            else:
                                console.print("[bold red]No text content in process_json response[/bold red]")

                        except Exception as e:
                            console.print(f"[bold red]Error during JQ processing step: {e}[/bold red]")
                            import traceback
                            console.print(traceback.format_exc())
                    else:
                        console.print("[bold red]Failed to read data with Table Processor in Step 9.1.[/bold red]")

    except Exception as e:
        console.print(f"[bold red]Error running demo: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())

async def run_table_demo(session, process_table_tool, params: Dict, demo_title: str = None, show_title: bool = True,
                         show_analysis: bool = False, show_output_path: bool = False):
    """Run a Table Processor demo with the given parameters using MCP tools."""
    if show_title and demo_title:
        console.print(Rule(f"[bold yellow]{demo_title}[/bold yellow]"))
    
    # Helper function to format parameter values for display
    def format_param_value(key: str, value: Any) -> str:
        """Format a parameter value for display in the table."""
        if key == "transform" and isinstance(value, dict):
            return json.dumps(value, indent=2)
        elif isinstance(value, dict):
            return json.dumps(value)
        elif isinstance(value, list):
            return str(value)
        else:
            return str(value)
    
    # Show the parameters
    console.print("[bold cyan]Table Processor Parameters:[/bold cyan]")
    params_table = Table(show_header=False, box=None)
    params_table.add_column("Parameter", style="green")
    params_table.add_column("Value", style="white")
    
    for key, value in params.items():
        # Format parameter value for display
        formatted_value = format_param_value(key, value)
        params_table.add_row(key, formatted_value)
    
    console.print(params_table)
    console.print()
    
    # Execute the Table Processor
    start_time = datetime.now()
    console.print("[bold]Executing Table Processor...[/bold]")
    
    try:
        # Use the process_table tool
        result = await session.call_tool(
            process_table_tool.name,
            arguments=params
        )
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        
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
                
                # Process and display results
                console.print(f"[green]Processing completed in {execution_time:.2f} seconds[/green]")
                
                if response and "result" in response:
                    # Show analysis if requested and available
                    if show_analysis and "analysis" in response:
                        analysis = response["analysis"]
                        console.print("\n[bold cyan]Data Analysis:[/bold cyan]")
                        
                        analysis_table = Table(show_header=False, box=None)
                        analysis_table.add_column("Metric", style="green")
                        analysis_table.add_column("Value", style="white")
                        
                        for key, value in analysis.items():
                            if isinstance(value, dict):
                                # For column statistics
                                console.print(f"\n[bold]Column Statistics: {key}[/bold]")
                                col_stats_table = Table(show_header=False, box=None)
                                col_stats_table.add_column("Statistic", style="cyan")
                                col_stats_table.add_column("Value", style="yellow")
                                
                                for stat_key, stat_value in value.items():
                                    col_stats_table.add_row(stat_key, str(stat_value))
                                
                                console.print(col_stats_table)
                            else:
                                analysis_table.add_row(key, str(value))
                        
                        console.print(analysis_table)
                    
                    # Show result rows
                    result_data = response["result"]
                    row_count = len(result_data) if isinstance(result_data, list) else 0
                    
                    if row_count > 0:
                        console.print(f"\n[bold cyan]Result Data ({row_count} rows):[/bold cyan]")
                        
                        # Get column names from first row
                        first_row = result_data[0]
                        columns = list(first_row.keys())
                        
                        # Create a rich table
                        data_table = Table(title="Table Data")
                        for col in columns:
                            data_table.add_column(col, style="cyan")
                        
                        # Add rows (limit to first 10 for display)
                        max_display = min(10, row_count)
                        for i in range(max_display):
                            row = result_data[i]
                            data_table.add_row(*[str(row.get(col, "")) for col in columns])
                        
                        console.print(data_table)
                        
                        if row_count > max_display:
                            console.print(f"[dim]... and {row_count - max_display} more rows not shown[/dim]")
                    else:
                        console.print("[yellow]No result rows returned[/yellow]")
                    
                    # Show output file path and content if requested
                    if show_output_path and "output_path" in params:
                        output_path = params["output_path"]
                        console.print(f"\n[bold cyan]Output File:[/bold cyan] {output_path}")
                        
                        # Show first few lines of the output file if it exists
                        if os.path.exists(output_path):
                            with open(output_path, "r") as f:
                                content = f.read(1000) # Read first 1000 chars
                            
                            console.print("[bold]Output File Content (first 1000 chars):[/bold]")
                            file_ext = os.path.splitext(output_path)[1].lower()
                            lang = "csv" if file_ext == ".csv" else "json" if file_ext == ".json" else "text"
                            console.print(Syntax(content, lang, theme="monokai"))
                            
                            # Show file size
                            file_size = os.path.getsize(output_path)
                            console.print(f"File size: {file_size} bytes")
                        else:
                            console.print("[yellow]Output file not found or not accessible[/yellow]")
                    
                else:
                    console.print("[bold red]Processing failed or returned no data[/bold red]")
            except json.JSONDecodeError:
                console.print("[bold red]Failed to parse response as JSON[/bold red]")
                console.print(f"Raw response: {result_text[:200]}...")
        else:
            console.print("[bold red]No text content in response[/bold red]")
    
    except Exception as e:
        console.print(f"[bold red]Error during Table Processor execution: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
    
    console.print("\n")  # Add spacing between demos

if __name__ == "__main__":
    if "--debug" in sys.argv:
        DEBUG = True
        debug_print("Debug mode enabled")
    
    try:
        asyncio.run(table_processor_demo())
    except KeyboardInterrupt:
        console.print("[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Unhandled error: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc()) 