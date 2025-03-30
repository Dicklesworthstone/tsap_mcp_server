#!/usr/bin/env python3
"""
Advanced Table Processor Demo

This script demonstrates the comprehensive features of the Table Processor
integration in TSAP, including reading various formats, data transformation,
analysis, writing output, and synergy with Ripgrep and JQ.
"""
import asyncio
import os
import sys
import json
import tempfile
from datetime import datetime
from pathlib import Path
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.console import Console
from rich.rule import Rule
from rich.json import JSON
from typing import Dict, Optional

# Assuming mcp_client_example.py is in the same directory or accessible
# Adjust the import path if necessary
try:
    from mcp_client_example import MCPClient
except ImportError:
    print("Error: Could not import MCPClient. Make sure mcp_client_example.py is accessible.")
    sys.exit(1)

console = Console()

# Add some basic debugging
DEBUG = False

def debug_print(msg):
    """Print debug message if DEBUG is True."""
    if DEBUG:
        console.print(f"[dim][DEBUG] {msg}[/dim]")

async def table_processor_demo():
    """Demonstrate Table Processor's advanced features with example data."""
    console.print(Panel.fit(
        "[bold blue]TSAP Table Processor Advanced Features Demo[/bold blue]",
        subtitle="Processing and analyzing tabular data with Table Processor"
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

    # Create our client
    try:
        debug_print("Creating MCPClient...")
        async with MCPClient() as client:
            debug_print("Running demos...")

            # --- Add initial info check ---
            console.print(f"Attempting to get server info from {client.base_url}...")
            info = await client.info()
            if info.get("status") != "success" or info.get("error") is not None:
                error_message = info.get('error', "Status was not success")
                console.print(f"[bold red]Error during initial client.info() check:[/bold red] {error_message}")
                return # Exit if server check fails
            else:
               console.print("[green]Initial client.info() check successful.[/green]")
            # ------------------------------

            # DEMO 1: Basic CSV Reading and Selection
            console.print(Rule("[bold yellow]Demo 1: Basic CSV Reading & Column Selection[/bold yellow]"))
            console.print(f"[italic]Reads '{os.path.basename(products_csv)}' and selects specific columns.[/italic]\n")
            await run_table_demo(
                client,
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
                client,
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
                client,
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
                client,
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
                client,
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
                    client,
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
                    client,
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

            # Step 8.1: Use Ripgrep to find relevant CSV files
            console.print("[bold cyan]Step 8.1: Running Ripgrep to find report*.csv files...[/bold cyan]")
            report_files_found = []
            try:
                rg_params = {  # noqa: F841
                    "pattern": r"report_.*\.csv", # Pattern to find filenames
                    "paths": [report_dir],
                    "files_matching": True, # Find files matching the pattern itself
                    "max_total_matches": 10
                }
                # Note: files_matching isn't a standard rg feature, simulate with filename search
                # Correct approach: search *for* filenames or use file listing + filtering.
                # Let's simulate by listing files and filtering client-side for this demo.
                # In a real scenario, you might list dir or use rg -l pattern if content matters.

                # --- Alternative: List directory and filter ---
                console.print(f"[dim]Simulating file search in {report_dir} for 'report_*.csv'[/dim]")
                for entry in os.listdir(report_dir):
                    if entry.startswith("report_") and entry.endswith(".csv"):
                         full_path = os.path.join(report_dir, entry)
                         if os.path.isfile(full_path):
                              report_files_found.append(full_path)

                # --- End Alternative ---

                # Debug print the full Ripgrep response if you were using it
                # debug_print(f"Ripgrep response: {rg_response}")

                if report_files_found:
                    console.print(f"[green]Found {len(report_files_found)} report file(s): {', '.join(report_files_found)}[/green]")

                    # Step 8.2: Use Table Processor on the found files
                    console.print("\n[bold cyan]Step 8.2: Running Table Processor to combine and analyze reports...[/bold cyan]")
                    await run_table_demo(
                        client,
                        demo_title="Demo 8.2: Process First Found Report",
                        params={
                             # Table processor might not directly support combining multiple files in one call this way.
                             # We'll process the first found file for this demo.
                             # A real workflow might involve multiple calls or a different tool.
                            "file_path": report_files_found[0], # Process the first found file
                             # "input_paths": report_files_found, # Idealized parameter
                            "analyze": True,
                            "transform": {
                                # Example: Add region based on filename (requires more logic)
                                # "computed_columns": {"Region": "os.path.basename(input_filename).split('_')[1].split('.')[0]"} # Idealized
                                "columns": ["ID", "Metric", "Value"] # Select columns
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
            console.print("[italic]Uses Table Processor to read JSON, then pipes to JQ (simulated) for complex filtering/transformation.[/italic]\n")

            # Step 9.1: Read JSON with Table Processor
            console.print("[bold cyan]Step 9.1: Reading sensor data with Table Processor...[/bold cyan]")
            tp_params_jq = {
                "file_path": sensor_json,
                "input_format": "json",
                 # No transformation needed here, get all data for JQ
            }
            tp_result_jq = None
            try:
                tp_result_jq = await client.process_table(**tp_params_jq)
            except Exception as e:
                console.print(f"[bold red]Error calling process_table in Step 9.1: {e}[/bold red]")
                tp_result_jq = {"success": False, "error": {"message": str(e)}}

            if tp_result_jq and tp_result_jq.get("status") == "success" and tp_result_jq.get("data", {}).get("result") is not None:
                # Get the actual table data from the 'result' field inside 'data'
                sensor_data_list = tp_result_jq["data"]["result"]
                console.print(f"[green]Table Processor successfully read {len(sensor_data_list)} records.[/green]")
                # Convert list of dicts back to JSON string for JQ input
                jq_input_json_str = json.dumps(sensor_data_list)

                # Step 9.2: Process with JQ
                console.print("\n[bold cyan]Step 9.2: Running JQ to filter for temperature readings > 22.6...[/bold cyan]")
                try:
                    jq_params = {
                        "input_json": jq_input_json_str,
                         # Select objects where unit is Celsius and reading > 22.6
                        "query": '.[] | select(.unit == "Celsius" and .reading > 22.6)',
                        "compact_output": True # Get compact JSON objects as output
                    }
                    jq_response = await client.jq_process(**jq_params)

                    # Display JQ results
                    if jq_response and jq_response.get("status") == "success" and jq_response.get("data", {}).get("output") is not None:
                         console.print("[bold green]JQ Processing Results:[/bold green]")
                         output = jq_response["data"]["output"]
                         # Check if output is already a list/dict (parsed by client or server)
                         if isinstance(output, (list, dict)):
                             console.print(JSON(json.dumps(output, indent=2))) # Pretty print the parsed object
                         elif isinstance(output, str):
                             # Attempt to parse if it's a string (might be compact JSON lines)
                             try:
                                # Try parsing as separate JSON objects per line first
                                jq_results_list = [json.loads(line) for line in output.strip().split('\n') if line.strip()]
                                console.print(JSON(json.dumps(jq_results_list, indent=2)))
                             except json.JSONDecodeError:
                                 try:
                                     # If line-by-line fails, try parsing the whole string as one JSON object
                                     jq_result_obj = json.loads(output)
                                     console.print(JSON(json.dumps(jq_result_obj, indent=2)))
                                 except json.JSONDecodeError:
                                     console.print("[yellow]Could not parse JQ output string as JSON, showing raw:[/yellow]")
                                     console.print(Syntax(output, "json", theme="monokai", word_wrap=True))
                             except Exception as parse_err:
                                 console.print(f"[yellow]Error processing JQ output string ({parse_err}), showing raw:[/yellow]")
                                 console.print(Syntax(output, "text", theme="monokai", word_wrap=True))
                         else:
                            console.print(f"[yellow]Unexpected JQ output type ({type(output)}), showing raw:[/yellow]")
                            console.print(str(output))

                    elif jq_response and jq_response.get("error"):
                         console.print(f"[bold red]JQ Processing Failed:[/bold red] {jq_response['error']}")
                    else:
                         console.print("[yellow]JQ processing did not return expected output.[/yellow]")

                except Exception as e:
                    console.print(f"[bold red]Error during JQ processing step: {e}[/bold red]")
                    import traceback
                    console.print(traceback.format_exc())
            else:
                console.print("[bold red]Failed to read data with Table Processor in Step 9.1.[/bold red]")
                if tp_result_jq and tp_result_jq.get("error"):
                    console.print(f"Error: {tp_result_jq['error']}")


    except Exception as e:
        console.print(f"[bold red]Error running demo: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())

async def run_table_demo(client, params: Dict, demo_title: str = None, show_title: bool = True,
                         show_analysis: bool = False, show_output_path: bool = False):
    """Run a Table Processor demo with the given parameters."""
    if show_title and demo_title:
        console.print(Rule(f"[bold yellow]{demo_title}[/bold yellow]"))

    # Show the parameters
    console.print("[bold cyan]Table Processor Parameters:[/bold cyan]")
    params_table = Table(show_header=False, box=None)
    params_table.add_column("Parameter", style="green")
    params_table.add_column("Value", style="white")

    # Function to safely format parameter values for display
    def format_param_value(key, value):
        if isinstance(value, str) and len(value) > 100:
            return f"'{value[:100]}...'" # Truncate long strings
        if isinstance(value, dict):
             # Pretty print dicts using rich JSON
             return JSON.from_data(value)
        if isinstance(value, list):
            # Show lists concisely
            return f"[{', '.join(map(str, value[:5]))}{', ...' if len(value) > 5 else ''}]"
        return str(value)

    for key, value in params.items():
        params_table.add_row(key, format_param_value(key, value))

    console.print(params_table)
    console.print()

    # Execute the Table Processor process
    start_time_dt = datetime.now()
    console.print("[bold]Executing Table Processor process...[/bold]")

    response: Optional[Dict] = None
    try:
        # Assuming client.process_table returns a dict like {'success': True/False, 'data': ..., 'analysis': ..., 'output_path': ..., 'error': ...}
        response = await client.process_table(**params)
    except Exception as e:
        console.print(f"[bold red]Error during client.process_table call: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        response = {
             "success": False,
             "error": {"code": "CLIENT_SIDE_ERROR", "message": str(e)},
             "execution_time": (datetime.now() - start_time_dt).total_seconds()
        }

    # Calculate client-side execution time (fallback)
    client_execution_time_s = (datetime.now() - start_time_dt).total_seconds()

    # Process and display results
    console.print("[bold cyan]Table Processor Results:[/bold cyan]")
    results_summary_table = Table(show_header=False, box=None, expand=True)
    results_summary_table.add_column("Field", style="green", width=20)
    results_summary_table.add_column("Value", style="white", ratio=1) # Let value take remaining space

    # Check 'status' field for success
    is_success = response and response.get("status") == "success"
    # Access tool-specific results under the top-level 'data' key
    tool_data = response.get("data") if response else {}
    tool_data = tool_data if isinstance(tool_data, dict) else {} # Ensure it's a dict

    error = response.get("error") if response else None
    # Get specific fields from tool_data
    table_result_data = tool_data.get("result") # Actual table rows
    analysis = tool_data.get("analysis")
    output_path = tool_data.get("output_path")
    row_count_processed = tool_data.get("row_count", "N/A") # Rows processed by tool
    # Use server execution time if available (from tool_data), else client time
    execution_time = tool_data.get("execution_time", client_execution_time_s)

    if is_success:
        results_summary_table.add_row("Status", "[bold green]Success[/bold green]")
        results_summary_table.add_row("Execution Time", f"{execution_time:.4f}s")
        results_summary_table.add_row("Rows Processed", str(row_count_processed))

        # Display returned data (limited preview) using 'table_result_data'
        if table_result_data is not None and isinstance(table_result_data, list):
            returned_rows = len(table_result_data)
            results_summary_table.add_row("Rows Returned", str(returned_rows))
            if returned_rows > 0:
                # Display data in a rich Table
                console.print("\n[bold green]Returned Data (Preview):[/bold green]")
                data_table = Table(show_header=True, header_style="bold magenta", box=None)
                # Dynamically add columns based on the keys of the first row
                headers = list(table_result_data[0].keys()) if table_result_data else []
                for header in headers:
                    data_table.add_column(header, overflow="fold") # Fold long text
                # Add rows (limited to 10 for preview)
                for row_dict in table_result_data[:10]:
                     # Convert all values to string for display in table
                     row_values = [str(row_dict.get(h, '')) for h in headers]
                     data_table.add_row(*row_values)
                console.print(data_table)
                if returned_rows > 10:
                    console.print(f"[dim]... displaying first 10 of {returned_rows} returned rows.[/dim]")
            else:
                 pass # Handled by Rows Returned = 0
        else:
             results_summary_table.add_row("Returned Data", "[dim]None or invalid format[/dim]")
             if table_result_data is not None:
                  debug_print(f"Unexpected data format: {type(table_result_data)}")

        # Display analysis if requested and available (already using correct field)
        if show_analysis and analysis is not None:
            results_summary_table.add_row("Analysis", JSON.from_data(analysis))
        elif show_analysis:
             results_summary_table.add_row("Analysis", "[dim]Not available or not generated[/dim]")

        # Display output path if requested and available
        if show_output_path and output_path:
            results_summary_table.add_row("Output File", f"[cyan]{output_path}[/cyan]")
            debug_print(f"Output path received: {output_path}") # DEBUG PRINT
            # Try to show a snippet of the output file
            try:
                with open(output_path, 'r', encoding='utf-8', errors='ignore') as f:
                     content_snippet = "".join(f.readline() for _ in range(5)) # Read first 5 lines
                     if content_snippet:
                          console.print("\n[bold green]Output File Content (Snippet):[/bold green]")
                          # Determine language for syntax highlighting
                          lang = "json" if output_path.endswith(".json") else "csv"
                          console.print(Syntax(content_snippet.strip(), lang, theme="monokai"))
                     else:
                          console.print("[dim]Output file is empty.[/dim]")
            except Exception as e:
                console.print(f"[yellow]Could not read output file snippet: {e}[/yellow]")
        elif show_output_path:
             results_summary_table.add_row("Output File", "[dim]Not generated or path not returned[/dim]")

    else:
        results_summary_table.add_row("Status", "[bold red]Failed[/bold red]")
        results_summary_table.add_row("Execution Time", f"{execution_time:.4f}s")
        if error and isinstance(error, dict):
            results_summary_table.add_row("Error Code", error.get('code', 'N/A'))
            # Use rich JSON for potentially nested error messages/details
            results_summary_table.add_row("Error Details", JSON.from_data(error, indent=1))
        elif isinstance(error, str):
             results_summary_table.add_row("Error", error)
        else:
             results_summary_table.add_row("Error", "[dim]Unknown error details.[/dim]")
             debug_print(f"Raw response on failure: {response}")

        # Show partial data/analysis if available even on failure
        if table_result_data is not None and isinstance(table_result_data, list) and table_result_data:
             results_summary_table.add_row("Partial Data", f"[dim]({len(table_result_data)} rows returned before/during failure)[/dim]")
        if analysis is not None:
             results_summary_table.add_row("Partial Analysis", JSON.from_data(analysis))


    console.print(results_summary_table)
    console.print("\n") # Add space between demos

if __name__ == "__main__":
    if "--debug" in sys.argv:
        DEBUG = True
        debug_print("Debug mode enabled")

    # Ensure example data directory exists
    example_data_dir = Path("tsap_example_data/tables")
    if not example_data_dir.is_dir():
         console.print(f"[bold red]Error:[/bold red] Example data directory '{example_data_dir}' not found.")
         console.print("Please ensure the 'tsap_example_data/tables' directory exists and contains the required files.")
         sys.exit(1)

    # Check essential files again just before running
    required_files_main = [
        example_data_dir / "products.csv",
        example_data_dir / "orders.tsv",
        example_data_dir / "sensor_data.json",
        example_data_dir / "inventory.csv",
        example_data_dir / "sales_data.csv"
    ]
    missing_files_main = [f for f in required_files_main if not f.exists()]
    if missing_files_main:
         console.print(f"[bold red]Error:[/bold red] Required example files missing: {', '.join(map(str, missing_files_main))}")
         sys.exit(1)

    # Also check synergy report dir
    report_dir_main = Path("tsap_example_data/documents")
    if not report_dir_main.is_dir():
        console.print(f"[bold red]Error:[/bold red] Required directory for synergy demo '{report_dir_main}' not found.")
        sys.exit(1)
    # Check for at least one report file for synergy demo
    if not any(f.name.startswith("report_") and f.name.endswith(".csv") for f in report_dir_main.iterdir()):
         console.print(f"[bold yellow]Warning:[/bold yellow] No 'report_*.csv' files found in '{report_dir_main}'. Synergy demo 8 might not run correctly.")


    try:
        asyncio.run(table_processor_demo())
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
    except NameError as e:
        if 'MCPClient' in str(e):
             # Import error already handled at the top
             pass
        else:
            console.print(f"[bold red]Unhandled NameError: {str(e)}[/bold red]")
            import traceback
            console.print(traceback.format_exc())
    except FileNotFoundError as e:
         console.print(f"[bold red]File Not Found Error: {str(e)}[/bold red]")
         console.print("Ensure the required example files and directories exist.")
    except Exception as e:
        console.print(f"[bold red]Unhandled error in main execution: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
