#!/usr/bin/env python3
"""
Advanced JQ Demo

This script demonstrates the comprehensive features of the JQ query
tool in TSAP, including powerful JSON data manipulation and extraction.
"""
import asyncio
from datetime import datetime
import os
import sys
import json
import tempfile
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.rule import Rule
from typing import Dict, Any, List, Optional, Union

# Import the MCP client from the library
from tsap.mcp import MCPClient

console = Console()

# Add some basic debugging
DEBUG = False

def debug_print(msg):
    """Print debug message if DEBUG is True."""
    if DEBUG:
        console.print(f"[dim][DEBUG] {msg}[/dim]")

async def jq_demo():
    """Demonstrate JQ's advanced features with example data."""
    console.print(Panel.fit(
        "[bold blue]TSAP JQ Advanced Features Demo[/bold blue]",
        subtitle="Querying and transforming JSON data with JQ"
    ))

    # Define file paths (relative to workspace root)
    users_file = "tsap_example_data/documents/users.json"
    nested_file = "tsap_example_data/documents/nested_data.json"
    logs_file = "tsap_example_data/documents/logs.jsonl"
    # code_dir = "tsap_example_data/code/" # For synergy demo - Removed as it's unused

    # Check existence of essential files
    required_files = [users_file, nested_file, logs_file]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        console.print(f"[bold red]Error:[/bold red] Required example files not found: {', '.join(missing_files)}")
        console.print("Please ensure the 'tsap_example_data' directory is correctly populated.")
        return

    # Example JSON string for demos
    sample_json_string = '{"name": "example", "version": 1.0, "enabled": true, "items": [10, 20, 30]}'

    # Create our client
    try:
        debug_print("Creating MCPClient...")
        async with MCPClient() as client:
            debug_print("Running demos...")

            # --- Add initial info check ---
            console.print(f"Attempting to get server info from {client.base_url}...")
            info = await client.info()
            if info.get("status") != "success" or info.get("error") is not None:
                # Use double quotes for the default message inside get()
                error_message = info.get('error', "Status was not success")
                console.print(f"[bold red]Error during initial client.info() check:[/bold red] {error_message}")
                return # Exit if server check fails
            else:
               console.print("[green]Initial client.info() check successful.[/green]")
            # ------------------------------

            # DEMO 1: Basic Field Selection (Input String)
            console.print(Rule("[bold yellow]Demo 1: Basic Field Selection (String Input)[/bold yellow]"))
            console.print("[italic]Extracts the 'name' and 'version' fields from a JSON string.[/italic]\n")
            await run_jq_demo(
                client,
                demo_title="Demo 1: Basic Selection (String)",
                query='{name: .name, ver: .version}',
                input_json=sample_json_string,
                show_title=False
            )

            # Let's limit our demos for debugging
            if DEBUG:
                debug_print("DEBUG mode: Only running first demo")
                return

            # DEMO 2: Selecting Fields from File (users.json)
            console.print(Rule("[bold yellow]Demo 2: Field Selection from File (users.json)[/bold yellow]"))
            console.print("[italic]Extracts 'id', 'username', and 'email' from each user object in the file.[/italic]\n")
            await run_jq_demo(
                client,
                demo_title="Demo 2: Field Selection (File)",
                query='.[] | {id, username, email}', # Selects from each element in the array
                input_files=[users_file],
                show_title=False
            )

            # DEMO 3: Filtering Array Elements (users.json)
            console.print(Rule("[bold yellow]Demo 3: Filtering Array Elements (users.json)[/bold yellow]"))
            console.print("[italic]Selects only users where 'active' is true.[/italic]\n")
            await run_jq_demo(
                client,
                demo_title="Demo 3: Filtering Users",
                query='.[] | select(.active == true)',
                input_files=[users_file],
                show_title=False
            )

            # DEMO 4: Transforming Data Structure (users.json)
            console.print(Rule("[bold yellow]Demo 4: Transforming Data Structure (users.json)[/bold yellow]"))
            console.print("[italic]Creates a new structure mapping username to role for active users.[/italic]\n")
            await run_jq_demo(
                client,
                demo_title="Demo 4: Transforming Data",
                query='[.[] | select(.active == true) | {(.username): .role}] | add', # Create key-value pairs and merge
                input_files=[users_file],
                show_title=False
            )

            # DEMO 5: Accessing Nested Fields (nested_data.json)
            console.print(Rule("[bold yellow]Demo 5: Accessing Nested Fields (nested_data.json)[/bold yellow]"))
            console.print("[italic]Extracts the 'engine' and 'version' for database components.[/italic]\n")
            await run_jq_demo(
                client,
                demo_title="Demo 5: Nested Field Access",
                query='.components[] | select(.type == "database") | .details | {engine, version}',
                input_files=[nested_file],
                show_title=False
            )

            # DEMO 6: Using JQ Operators and Functions (users.json)
            console.print(Rule("[bold yellow]Demo 6: JQ Operators and Functions (users.json)[/bold yellow]"))
            console.print("[italic]Calculates the number of users and lists usernames with their first tag.[/italic]\n")
            await run_jq_demo(
                client,
                demo_title="Demo 6: Operators and Functions",
                query='{user_count: length, users: map({user: .username, first_tag: (if (.tags | length > 0) then .tags[0] else "None" end)})}',
                input_files=[users_file],
                show_title=False
            )

            # DEMO 7: Processing JSON Lines File (logs.jsonl)
            console.print(Rule("[bold yellow]Demo 7: Processing JSON Lines (logs.jsonl)[/bold yellow]"))
            console.print("[italic]Extracts messages from log entries with level 'ERROR'.[/italic]\n")
            await run_jq_demo(
                client,
                demo_title="Demo 7: JSON Lines Processing",
                query='select(.level == "ERROR") | .message',
                input_files=[logs_file], # jq handles JSON Lines automatically when reading files
                show_title=False,
                raw_output=True # Get raw strings for messages
            )

            # DEMO 8: Raw Output (-r) for Plain Strings (users.json)
            console.print(Rule("[bold yellow]Demo 8: Raw Output (-r) (users.json)[/bold yellow]"))
            console.print("[italic]Extracts usernames as plain strings, not JSON strings.[/italic]\n")
            await run_jq_demo(
                client,
                demo_title="Demo 8: Raw Output",
                query='.[] | .username',
                input_files=[users_file],
                raw_output=True,
                show_title=False
            )

            # DEMO 9: Compact Output (-c) (users.json)
            console.print(Rule("[bold yellow]Demo 9: Compact Output (-c) (users.json)[/bold yellow]"))
            console.print("[italic]Outputs filtered active user objects, each on a single compact line.[/italic]\n")
            await run_jq_demo(
                client,
                demo_title="Demo 9: Compact Output",
                query='.[] | select(.active == true)',
                input_files=[users_file],
                compact_output=True,
                show_title=False
            )

            # DEMO 10: Synergy with Ripgrep - Find and Process
            console.print(Rule("[bold yellow]Demo 10: Synergy with Ripgrep[/bold yellow]"))
            console.print('[italic]Uses Ripgrep to find JSON files containing "role": "admin", then JQ extracts user info.[/italic]\n')

            # Step 1: Run Ripgrep to find files
            console.print("[bold cyan]Step 10.1: Running Ripgrep to find relevant JSON files...[/bold cyan]")
            try:
                rg_params = {
                    # Use raw string for the regex pattern
                    "pattern": r'"role":\s*"admin"', 
                    "paths": ["tsap_example_data/documents/"],
                    "file_patterns": ["*.json"], # Only search .json files
                    "regex": True,
                    "files_with_matches": True, # Only output filenames
                    "max_total_matches": 10 # Limit number of files
                }
                rg_response = await client.ripgrep_search(**rg_params)

                # Debug print the full Ripgrep response
                # debug_print(f"Ripgrep response: {rg_response}")

                files_found = []
                # Check response structure carefully based on files_with_matches=True
                if isinstance(rg_response, dict) and rg_response.get("data") and rg_response["data"].get("matches"):
                     # When files_with_matches=True, matches contain file paths directly in 'path'
                    files_found = [match.get("path") for match in rg_response["data"]["matches"] if match.get("path")]
                    # Remove potential duplicates and None values
                    files_found = sorted(list(set(filter(None, files_found))))


                if files_found:
                    console.print(f"[green]Ripgrep found {len(files_found)} file(s) containing the pattern: {', '.join(files_found)}[/green]")

                    console.print("\n[bold cyan]Step 10.2: Running JQ to extract info from found files...[/bold cyan]")
                    # Step 2: Run JQ on the found files
                    await run_jq_demo(
                        client,
                        demo_title="Demo 10.2: JQ Processing Ripgrep Results",
                        query='.[] | select(.role == "admin") | {file: input_filename, user: .username, email: .email}',
                        input_files=files_found, # Pass the list of files found by ripgrep
                        show_title=False
                    )
                elif isinstance(rg_response, dict) and rg_response.get("data") and not rg_response["data"].get("matches"):
                     console.print("[yellow]Ripgrep did not find any matching files.[/yellow]\n")
                else:
                    console.print("[yellow]Ripgrep search failed or returned unexpected data.[/yellow]")
                    if isinstance(rg_response, dict) and rg_response.get("error"):
                        error_info = rg_response['error']
                        console.print(f"[red]Ripgrep Error: {error_info}[/red]\n")
                    else:
                        response_info = rg_response
                        console.print(f"[dim]Ripgrep Response: {response_info}[/dim]\n")

            except Exception as e:
                console.print(f"[bold red]Error during Ripgrep/JQ synergy demo: {e}[/bold red]")
                import traceback
                console.print(traceback.format_exc())


    except Exception as e:
        console.print(f"[bold red]Error running demo: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())

async def run_jq_demo(client, query: str, input_json: str = None, input_files: list = None,
                      raw_output: bool = False, compact_output: bool = False,
                      monochrome_output: bool = False,
                      show_title: bool = True, demo_title: str = None):
    """Run a JQ demo with the given parameters."""
    if show_title and demo_title:
        console.print(Rule(f"[bold yellow]{demo_title}[/bold yellow]"))

    # Build the parameters dict cleanly
    jq_params = {
        "query": query
    }
    if input_json is not None:
        jq_params["input_json"] = input_json
    if input_files is not None:
        jq_params["input_files"] = input_files
    if raw_output:
        jq_params["raw_output"] = True
    if compact_output:
        jq_params["compact_output"] = True
    if monochrome_output:
        jq_params["monochrome_output"] = True


    # Show the search parameters
    console.print("[bold cyan]JQ Parameters:[/bold cyan]")
    params_table = Table(show_header=False, box=None)
    params_table.add_column("Parameter", style="green")
    params_table.add_column("Value", style="white")

    for key, value in jq_params.items():
        # Format values for display
        display_value: Any
        if key == "input_json" and value is not None:
             # Truncate long input JSON for display (removed repr)
            display_value = (value[:100] + '...' if len(value) > 100 else value)
        elif isinstance(value, list):
            display_value = ", ".join(map(str, value))
        elif key == "query":
             display_value = Syntax(value, "jq", theme="monokai", word_wrap=True)
        else:
            display_value = str(value)
        params_table.add_row(key, display_value)

    console.print(params_table)
    console.print()

    # Execute the JQ process
    start_time_dt = datetime.now()
    console.print("[bold]Executing JQ process...[/bold]")

    response: Optional[dict] = None
    try:
        # Assuming client.jq_query returns a dict like {'data': {...}, 'error': ...} or throws
        # Corrected method name from jq_query to jq_process
        response = await client.jq_process(**jq_params)
    except Exception as e:
        console.print(f"[bold red]Error during client.jq_process call: {e}[/bold red]")
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
    execution_time_s = (datetime.now() - start_time_dt).total_seconds()

    # Process and display results
    console.print("[bold cyan]JQ Results:[/bold cyan]")
    results_table = Table(show_header=False, box=None)
    results_table.add_column("Field", style="green")
    results_table.add_column("Value", style="white")

    # Determine success based on presence of 'data' and no 'error'
    is_success = isinstance(response, dict) and "data" in response and not response.get("error")
    # Ensure 'data' is always a dict, defaulting to {} if response["data"] is None or response is not a dict
    data_from_response = response.get("data") if isinstance(response, dict) else None
    data = data_from_response if isinstance(data_from_response, dict) else {}
    error = response.get("error") if isinstance(response, dict) else None

    # Adjust execution time if available from server response
    server_exec_time = data.get('execution_time')
    display_exec_time = server_exec_time if server_exec_time is not None else execution_time_s

    if is_success:
        results_table.add_row("Status", "[bold green]Success[/bold green]")
        results_table.add_row("Exit Code", str(data.get("exit_code", "N/A")))
        results_table.add_row("Parsed as JSON", str(data.get("parsed", "N/A")))
        results_table.add_row("Execution Time", f"{display_exec_time:.4f}s")
        results_table.add_row("Command", Syntax(data.get("command", "N/A"), "bash", theme="monokai", word_wrap=True))

        output_data = data.get("output")
        if output_data is not None:
            # Determine how to display the output
            if data.get("parsed") and not raw_output:
                # Pretty print JSON if it was parsed and not raw
                try:
                    output_str = json.dumps(output_data, indent=2)
                    output_syntax = Syntax(output_str, "json", theme="monokai", line_numbers=True, word_wrap=True)
                    results_table.add_row("Output (JSON)", output_syntax)
                except TypeError:
                     # Fallback if json.dumps fails (shouldn't happen often)
                     results_table.add_row("Output", str(output_data))
            else:
                # Display as plain text if raw or not parsed
                output_str = str(output_data)
                # Heuristic: If it looks like JSON, highlight it anyway (using double quotes for startswith)
                lang = "json" if (output_str.strip().startswith("{") or output_str.strip().startswith("[")) and not raw_output else "text"
                output_syntax = Syntax(output_str.strip(), lang, theme="monokai", line_numbers=True, word_wrap=True)
                results_table.add_row("Output (Text/Raw)", output_syntax)

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
             # Handle case where response might be None or not a dict
             results_table.add_row("Error", f"[dim]Unknown error details. Response: {response}[/dim]")


        # Show command even on failure if available
        cmd_attempt = data.get("command") or (response.get("command") if isinstance(response, dict) else None) or jq_params.get("command") # Last fallback
        if cmd_attempt and cmd_attempt != "N/A (Client Error Before Call)":
            results_table.add_row("Attempted Command", Syntax(cmd_attempt, "bash", theme="monokai", word_wrap=True))
        elif cmd_attempt:
             results_table.add_row("Attempted Command", cmd_attempt)

        # Show partial output if available (e.g., from stderr captured in output field on error)
        output_data = data.get("output", "") # Usually contains stderr on failure
        if output_data:
             results_table.add_row("Partial Output/Stderr", Syntax(str(output_data).strip(), "text", theme="monokai", word_wrap=True))

        # Fallback for completely unexpected response
        if not error and not data and response is not None:
            results_table.add_row("Response", str(response))


    console.print(results_table)
    console.print("\n")

if __name__ == "__main__":
    if "--debug" in sys.argv:
        DEBUG = True
        debug_print("Debug mode enabled")

    # Ensure example data directory exists
    example_data_dir = Path("tsap_example_data")
    if not example_data_dir.is_dir():
         console.print(f"[bold yellow]Warning:[/bold yellow] '{example_data_dir}' directory not found.")
         console.print("File-based and synergy demos might be skipped or fail if files are missing.")
         # Check for specific files needed by demos if dir exists
         if example_data_dir.is_dir():
             required_files_main = [
                 example_data_dir / "documents" / "users.json",
                 example_data_dir / "documents" / "nested_data.json",
                 example_data_dir / "documents" / "logs.jsonl"
             ]
             missing_files_main = [f for f in required_files_main if not f.exists()]
             if missing_files_main:
                  console.print(f"[bold red]Error:[/bold red] Required example files missing: {', '.join(map(str, missing_files_main))}")
                  sys.exit(1)

    try:
        asyncio.run(jq_demo())
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
    except NameError as e:
        if 'MCPClient' in str(e):
             # Already handled the import error message
             pass
        else:
            console.print(f"[bold red]Unhandled NameError: {str(e)}[/bold red]")
            import traceback
            console.print(traceback.format_exc())
    except FileNotFoundError as e:
         console.print(f"[bold red]File Not Found Error: {str(e)}[/bold red]")
         console.print("Ensure the required example files and directories exist.")
    except Exception as e:
        console.print(f"[bold red]Unhandled error: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())


