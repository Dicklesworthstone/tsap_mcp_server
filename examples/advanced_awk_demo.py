#!/usr/bin/env python3
"""
Advanced AWK Demo

This script demonstrates the comprehensive features of the AWK integration
in TSAP, including processing strings, files, using variables, and synergy
with other tools like Ripgrep.
"""
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.console import Console
from rich.rule import Rule

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

async def awk_demo():
    """Demonstrate AWK's advanced features with example data."""
    console.print(Panel.fit(
        "[bold blue]TSAP AWK Advanced Features Demo[/bold blue]",
        subtitle="Processing data with AWK"
    ))

    # Create our client
    try:
        debug_print("Creating MCPClient...")
        async with MCPClient() as client:
            debug_print("Running demos...")

            # --- Add initial info check ---
            console.print(f"Attempting to get server info from {client.base_url}...")
            info = await client.info()
            if info.get("status") != "success" or info.get("error") is not None:
                console.print(f"[bold red]Error during initial client.info() check:[/bold red] {info.get('error', 'Status was not success')}")
                return # Exit if server check fails
            else:
               console.print("[green]Initial client.info() check successful.[/green]")
            # ------------------------------

            # Example data for demos
            sample_text_string = """line 1 field1 field2 10
line 2 fieldA fieldB 25
line 3 fieldX fieldY 30
line 1 again field1 field2 15"""

            sample_csv_string = "header1,header2,value\nitem1,desc1,100\nitem2,desc2,200\nitem3,desc3,150"

            # DEMO 1: Basic Field Extraction (Input String)
            console.print(Rule("[bold yellow]Demo 1: Basic Field Extraction (String Input)[/bold yellow]"))
            console.print("[italic]Extracts the 1st and 3rd fields from a multi-line string.[/italic]\n")
            await run_awk_demo(
                client,
                demo_title="Demo 1: Basic Field Extraction",
                script='{print $1, $3}',
                input_text=sample_text_string,
                show_title=False
            )

            # Let's limit our demos for debugging
            if DEBUG:
                debug_print("DEBUG mode: Only running first demo")
                return

            # DEMO 2: Using Field Separators (CSV String)
            console.print(Rule("[bold yellow]Demo 2: Custom Field Separators (CSV String)[/bold yellow]"))
            console.print("[italic]Processes a CSV string using ',' as input and '|' as output separator.[/italic]\n")
            await run_awk_demo(
                client,
                demo_title="Demo 2: Custom Field Separators",
                script='NR > 1 {print $1, $3}', # Skip header row (NR > 1)
                input_text=sample_csv_string,
                field_separator=',',
                output_field_separator=' | ',
                show_title=False
            )

            # DEMO 3: Using Variables
            console.print(Rule("[bold yellow]Demo 3: Passing Variables to AWK[/bold yellow]"))
            console.print("[italic]Filters lines based on a variable passed to the script.[/italic]\n")
            await run_awk_demo(
                client,
                demo_title="Demo 3: Passing Variables",
                script='($NF+0) > (threshold+0) {print "Value over threshold:", $0}',
                input_text=sample_text_string,
                variables={"threshold": "20"},
                show_title=False
            )

            # DEMO 4: Processing Files
            console.print(Rule("[bold yellow]Demo 4: Processing an Input File[/bold yellow]"))
            console.print("[italic]Processes 'tsap_example_data/documents/sample.txt', printing line number and line.[/italic]\n")
            # Check if file exists
            sample_file = "tsap_example_data/documents/sample.txt"
            if os.path.exists(sample_file):
                await run_awk_demo(
                    client,
                    demo_title="Demo 4: Processing File",
                    script='{print NR ": " $0}', # NR is the record (line) number
                    input_files=[sample_file],
                    show_title=False
                )
            else:
                console.print(f"[yellow]Skipping Demo 4: File not found '{sample_file}'[/yellow]\n")


            # DEMO 5: Pattern Matching and Actions
            console.print(Rule("[bold yellow]Demo 5: Pattern Matching and Actions[/bold yellow]"))
            console.print("[italic]Prints lines containing the word 'line' from the sample string.[/italic]\n")
            await run_awk_demo(
                client,
                demo_title="Demo 5: Pattern Matching",
                script='/line/ {print "Found match on line " NR ":", $0}', # Only operates on lines matching /line/
                input_text=sample_text_string,
                show_title=False
            )

            # DEMO 6: Summing Column (Using $NF)
            console.print(Rule("[bold yellow]Demo 6: Performing Calculations[/bold yellow]"))
            console.print("[italic]Sums the values in the last column ($NF) of the sample string.[/italic]\n")
            await run_awk_demo(
                client,
                demo_title="Demo 6: Summing Column",
                script='BEGIN { sum=0 } { sum += ($NF+0) } END { print "Total sum of column 5:", sum }',
                input_text=sample_text_string,
                show_title=False
            )

            # DEMO 7: Synergy with Ripgrep - Find Python Function Definitions
            console.print(Rule("[bold yellow]Demo 7: Synergy with Ripgrep[/bold yellow]"))
            console.print("[italic]Uses Ripgrep to find Python function definitions, then AWK to extract names.[/italic]\n")
            # Check if code directory exists
            code_dir = "tsap_example_data/code/"
            if os.path.isdir(code_dir):
                # Step 1: Run Ripgrep to find lines starting with 'def '
                console.print("[bold cyan]Step 7.1: Running Ripgrep to find function definitions...[/bold cyan]")
                try:
                    rg_params = {
                        "pattern": r"^[ \t]*def\s+", # Regex for lines starting with optional space/tabs then 'def '
                        "paths": [code_dir],
                        "file_patterns": ["*.py"],
                        "regex": True,
                        "max_total_matches": 20 # Limit matches
                    }
                    rg_response = await client.ripgrep_search(**rg_params)

                    if isinstance(rg_response, dict) and rg_response.get("data") and rg_response["data"].get("matches"):
                        matches = rg_response["data"]["matches"]
                        console.print(f"[green]Ripgrep found {len(matches)} potential function definitions.[/green]")

                        # Concatenate the line text from matches for AWK input
                        rg_output_lines = [match.get("line_text", "") for match in matches]
                        awk_input_from_rg = "\n".join(rg_output_lines).strip()

                        if awk_input_from_rg:
                            console.print("\n[bold cyan]Step 7.2: Running AWK to extract function names...[/bold cyan]")
                            # Step 2: Run AWK to process Ripgrep's output
                            await run_awk_demo(
                                client,
                                demo_title="Demo 7.2: AWK Processing Ripgrep Output",
                                # Refined regex to capture function name up to '('
                                script=r'/^[ \t]*def / { match($0, /def +([^(]+)/, arr); if (arr[1]) print arr[1] }',
                                input_text=awk_input_from_rg,
                                show_title=False
                            )
                        else:
                            console.print("[yellow]No lines extracted from Ripgrep output for AWK processing.[/yellow]\n")

                    else:
                        console.print("[yellow]Ripgrep found no matches or encountered an error.[/yellow]")
                        if isinstance(rg_response, dict) and rg_response.get("error"):
                            console.print(f"[red]Ripgrep Error: {rg_response['error']}[/red]\n")
                        else:
                            console.print(f"[dim]Ripgrep Response: {rg_response}[/dim]\n")

                except Exception as e:
                    console.print(f"[bold red]Error during Ripgrep/AWK synergy demo: {e}[/bold red]")
                    import traceback
                    console.print(traceback.format_exc())
            else:
                console.print(f"[yellow]Skipping Demo 7: Directory not found '{code_dir}'[/yellow]\n")

            # DEMO 8: BEGIN and END blocks
            console.print(Rule("[bold yellow]Demo 8: BEGIN and END Blocks[/bold yellow]"))
            console.print("[italic]Demonstrates setup (BEGIN) and summary (END) actions.[/italic]\n")
            await run_awk_demo(
                client,
                demo_title="Demo 8: BEGIN/END Blocks",
                script='BEGIN { print "Starting processing..."; count = 0 } { count++ } END { print "Processed " count " lines." }',
                input_text=sample_text_string,
                show_title=False
            )

            # DEMO 9: Frequency Counting (Associative Arrays)
            console.print(Rule("[bold yellow]Demo 9: Frequency Counting (Associative Arrays)[/bold yellow]"))
            console.print("[italic]Counts the frequency of each unique value in the first field ($1) of the sample string.[/italic]\n")
            await run_awk_demo(
                client,
                demo_title="Demo 9: Frequency Counting",
                script='{ count[$1]++ } END { for (item in count) print item, count[item] }',
                input_text=sample_text_string,
                show_title=False
            )

            # DEMO 10: Grouping and Aggregation (Associative Arrays)
            console.print(Rule("[bold yellow]Demo 10: Grouping and Aggregation (Associative Arrays)[/bold yellow]"))
            console.print("[italic]Groups lines by the first field ($1) and sums the last field ($NF) for each group.[/italic]\n")
            await run_awk_demo(
                client,
                demo_title="Demo 10: Grouping & Summation",
                script='{ sum[$1] += $NF } END { for (item in sum) printf "%s: Total = %d\\n", item, sum[item] }',
                input_text=sample_text_string,
                show_title=False
            )


    except Exception as e:
        console.print(f"[bold red]Error running demo: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())

async def run_awk_demo(client, script: str, input_text: str = None, input_files: list = None,
                       field_separator: str = None, output_field_separator: str = None,
                       variables: dict = None, show_title: bool = True, demo_title: str = None):
    """Run an AWK demo with the given parameters."""
    if show_title and demo_title:
        console.print(Rule(f"[bold yellow]{demo_title}[/bold yellow]"))

    # Build the parameters dict cleanly
    awk_params = {
        "script": script
    }
    if input_text is not None:
        awk_params["input_text"] = input_text
    if input_files is not None:
        awk_params["input_files"] = input_files
    if field_separator is not None:
        awk_params["field_separator"] = field_separator
    if output_field_separator is not None:
        awk_params["output_field_separator"] = output_field_separator
    if variables is not None:
        awk_params["variables"] = variables

    # Show the search parameters
    console.print("[bold cyan]AWK Parameters:[/bold cyan]")
    params_table = Table(show_header=False, box=None)
    params_table.add_column("Parameter", style="green")
    params_table.add_column("Value", style="white")

    for key, value in awk_params.items():
        # Format values for display
        if key == "input_text" and value is not None:
             # Truncate long input text for display
            display_value = repr(value[:100] + '...' if len(value) > 100 else value)
        elif isinstance(value, list):
            display_value = ", ".join(map(str, value))
        elif isinstance(value, dict):
             display_value = ", ".join(f'{k}={v}' for k, v in value.items())
        elif key == "script":
             display_value = Syntax(value, "awk", theme="monokai", word_wrap=True)
        else:
            display_value = str(value)
        params_table.add_row(key, display_value)

    console.print(params_table)
    console.print()

    # Execute the AWK process
    start_time_dt = datetime.now()
    console.print("[bold]Executing AWK process...[/bold]")

    response = None
    try:
        # Assuming client.awk_process returns a dict like {'data': {...}, 'error': ...} or throws
        response = await client.awk_process(**awk_params)
    except Exception as e:
        console.print(f"[bold red]Error during client.awk_process call: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        # Structure the error similarly to a potential server error response
        response = {
             "error": {
                 "code": "CLIENT_SIDE_ERROR",
                 "message": str(e)
                 },
             "command": "N/A (Client Error Before Call)" # Indicate command wasn't formed/sent
        }

    # Calculate execution time
    execution_time = (datetime.now() - start_time_dt).total_seconds()

    # Process and display results
    console.print("[bold cyan]AWK Results:[/bold cyan]")
    results_table = Table(show_header=False, box=None)
    results_table.add_column("Field", style="green")
    results_table.add_column("Value", style="white")

    # Determine success based on presence of 'data' and no 'error'
    is_success = isinstance(response, dict) and "data" in response and not response.get("error")
    data = response.get("data", {}) if isinstance(response, dict) else {}
    error = response.get("error") if isinstance(response, dict) else None

    if is_success:
        results_table.add_row("Status", "[bold green]Success[/bold green]")
        results_table.add_row("Exit Code", str(data.get("exit_code", "N/A")))
        results_table.add_row("Execution Time", f"{data.get('execution_time', execution_time):.4f}s")
        results_table.add_row("Command", Syntax(data.get("command", "N/A"), "bash", theme="monokai", word_wrap=True))

        output = data.get("output", "")
        if output:
            # Basic syntax highlighting guess
            lang = "python" if "def " in output else "text"
            results_table.add_row("Output", Syntax(output.strip(), lang, theme="monokai", line_numbers=True, word_wrap=True))
        else:
            results_table.add_row("Output", "[dim]No output[/dim]")
    else:
        results_table.add_row("Status", "[bold red]Failed[/bold red]")
        if error and isinstance(error, dict):
            results_table.add_row("Error Code", error.get('code', 'N/A'))
            results_table.add_row("Message", error.get('message', 'Unknown error'))
        elif isinstance(error, str):
             results_table.add_row("Error", error)
        else:
             results_table.add_row("Error", "[dim]Unknown error details[/dim]")

        # Show command even on failure if available
        cmd_attempt = data.get("command") or (response.get("command") if isinstance(response, dict) else None)
        if cmd_attempt and cmd_attempt != "N/A (Client Error Before Call)":
            results_table.add_row("Attempted Command", Syntax(cmd_attempt, "bash", theme="monokai", word_wrap=True))
        elif cmd_attempt:
             results_table.add_row("Attempted Command", cmd_attempt)

        # Show partial output if available (e.g., from stderr captured in output)
        output = data.get("output", "")
        if output:
             results_table.add_row("Partial Output/Stderr", Syntax(output.strip(), "text", theme="monokai", word_wrap=True))

        # Fallback for completely unexpected response
        if not error and not data:
            results_table.add_row("Response", str(response))

    console.print(results_table)
    console.print("\n")  # Add space between demos

if __name__ == "__main__":
    if "--debug" in sys.argv:
        DEBUG = True
        debug_print("Debug mode enabled")

    # Ensure example data directory exists or skip file-based demos
    example_data_dir = Path("tsap_example_data")
    if not example_data_dir.is_dir():
         console.print(f"[bold yellow]Warning:[/bold yellow] '{example_data_dir}' directory not found.")
         console.print("File-based and synergy demos might be skipped or fail.")
         # Create dummy dirs/files if needed for basic script execution without full data
         # (Path(example_data_dir) / "documents").mkdir(parents=True, exist_ok=True)
         # (Path(example_data_dir) / "documents" / "sample.txt").touch()
         # (Path(example_data_dir) / "code").mkdir(parents=True, exist_ok=True)

    try:
        asyncio.run(awk_demo())
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
    except NameError as e:
        if 'MCPClient' in str(e):
             # Already handled the import error message
             pass
        else:
            console.print(f"[bold red]Unhandled error: {str(e)}[/bold red]")
            import traceback
            console.print(traceback.format_exc())
    except Exception as e:
        console.print(f"[bold red]Unhandled error: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
