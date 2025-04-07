#!/usr/bin/env python3
"""
Advanced JQ Demo (MCP Tools Version)

This script demonstrates the comprehensive features of the JQ query
tool in TSAP, including powerful JSON data manipulation and extraction
using the MCP tools interface.
"""
import asyncio
from datetime import datetime
import os
import sys
import json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.rule import Rule
from typing import Any

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

async def jq_demo():
    """Demonstrate JQ's advanced features with example data."""
    console.print(Panel.fit(
        "[bold blue]TSAP MCP JQ Advanced Features Demo[/bold blue]",
        subtitle="Querying and transforming JSON data with JQ tools"
    ))

    # Define file paths (relative to workspace root)
    users_file = "tsap_example_data/documents/users.json"
    nested_file = "tsap_example_data/documents/nested_data.json"
    logs_file = "tsap_example_data/documents/logs.jsonl"

    # Check existence of essential files
    required_files = [users_file, nested_file, logs_file]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        console.print(f"[bold red]Error:[/bold red] Required example files not found: {', '.join(missing_files)}")
        console.print("Please ensure the 'tsap_example_data' directory is correctly populated.")
        return

    # Example JSON string for demos
    sample_json_string = '{"name": "example", "version": 1.0, "enabled": true, "items": [10, 20, 30]}'

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
                
                # Find the jq tool and search tool
                jq_tool = next((t for t in tools if t.name == "jq_query"), None)
                search_tool = next((t for t in tools if t.name == "ripgrep_search"), None)
                
                if not jq_tool:
                    console.print("[bold red]Error: jq_query tool not found![/bold red]")
                    return
                
                # --- Initial info check ---
                info_tool = next((t for t in tools if t.name == "info"), None)
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
                            info_data = json.loads(info_text)  # noqa: F841
                            console.print("[green]Initial server info check successful.[/green]")
                        else:
                            console.print("[yellow]Info call succeeded but returned no readable content.[/yellow]")
                    except Exception as e:
                        console.print(f"[bold red]Error during info check:[/bold red] {str(e)}")
                        debug_print(f"Info check error details: {e}")
                # ------------------------------

                # DEMO 1: Basic Field Selection (Input String)
                console.print(Rule("[bold yellow]Demo 1: Basic Field Selection (String Input)[/bold yellow]"))
                console.print("[italic]Extracts the 'name' and 'version' fields from a JSON string.[/italic]\n")
                await run_jq_demo(
                    session,
                    jq_tool,
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
                    session,
                    jq_tool,
                    demo_title="Demo 2: Field Selection (File)",
                    query='.[] | {id, username, email}', # Selects from each element in the array
                    input_files=[users_file],
                    show_title=False
                )

                # DEMO 3: Filtering Array Elements (users.json)
                console.print(Rule("[bold yellow]Demo 3: Filtering Array Elements (users.json)[/bold yellow]"))
                console.print("[italic]Selects only users where 'active' is true.[/italic]\n")
                await run_jq_demo(
                    session,
                    jq_tool,
                    demo_title="Demo 3: Filtering Users",
                    query='.[] | select(.active == true)',
                    input_files=[users_file],
                    show_title=False
                )

                # DEMO 4: Transforming Data Structure (users.json)
                console.print(Rule("[bold yellow]Demo 4: Transforming Data Structure (users.json)[/bold yellow]"))
                console.print("[italic]Creates a new structure mapping username to role for active users.[/italic]\n")
                await run_jq_demo(
                    session,
                    jq_tool,
                    demo_title="Demo 4: Transforming Data",
                    query='[.[] | select(.active == true) | {(.username): .role}] | add', # Create key-value pairs and merge
                    input_files=[users_file],
                    show_title=False
                )

                # DEMO 5: Accessing Nested Fields (nested_data.json)
                console.print(Rule("[bold yellow]Demo 5: Accessing Nested Fields (nested_data.json)[/bold yellow]"))
                console.print("[italic]Extracts the 'engine' and 'version' for database components.[/italic]\n")
                await run_jq_demo(
                    session,
                    jq_tool,
                    demo_title="Demo 5: Nested Field Access",
                    query='.components[] | select(.type == "database") | .details | {engine, version}',
                    input_files=[nested_file],
                    show_title=False
                )

                # DEMO 6: Using JQ Operators and Functions (users.json)
                console.print(Rule("[bold yellow]Demo 6: JQ Operators and Functions (users.json)[/bold yellow]"))
                console.print("[italic]Calculates the number of users and lists usernames with their first tag.[/italic]\n")
                await run_jq_demo(
                    session,
                    jq_tool,
                    demo_title="Demo 6: Operators and Functions",
                    query='{user_count: length, users: map({user: .username, first_tag: (if (.tags | length > 0) then .tags[0] else "None" end)})}',
                    input_files=[users_file],
                    show_title=False
                )

                # DEMO 7: Processing JSON Lines File (logs.jsonl)
                console.print(Rule("[bold yellow]Demo 7: Processing JSON Lines (logs.jsonl)[/bold yellow]"))
                console.print("[italic]Extracts messages from log entries with level 'ERROR'.[/italic]\n")
                await run_jq_demo(
                    session,
                    jq_tool,
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
                    session,
                    jq_tool,
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
                    session,
                    jq_tool,
                    demo_title="Demo 9: Compact Output",
                    query='.[] | select(.active == true)',
                    input_files=[users_file],
                    compact_output=True,
                    show_title=False
                )

                # DEMO 10: Synergy with Search - Find and Process
                console.print(Rule("[bold yellow]Demo 10: Synergy with Search Tool[/bold yellow]"))
                console.print('[italic]Uses search to find JSON files containing "role": "admin", then JQ extracts user info.[/italic]\n')

                if search_tool:
                    # Step 1: Run search to find files
                    console.print("[bold cyan]Step 10.1: Running search to find relevant JSON files...[/bold cyan]")
                    try:
                        # Use ripgrep_search tool with proper MCP calling pattern
                        search_params = {
                            "pattern": r'"role":\s*"admin"', # Regex for "role": "admin"
                            "paths": ["tsap_example_data/documents/"],
                            "file_patterns": ["*.json"], # Only search .json files
                            "regex": True,
                            "files_with_matches": True # Only output filenames
                        }
                        
                        search_result = await session.call_tool(search_tool.name, arguments=search_params)
                        
                        # Extract the text content
                        search_text = None
                        for content in search_result.content:
                            if content.type == "text":
                                search_text = content.text
                                break
                        
                        if search_text:
                            # Parse JSON response
                            search_response = json.loads(search_text)
                            
                            files_found = []
                            # Check response structure for files_with_matches=True
                            if "matches" in search_response:
                                # When files_with_matches=True, each match represents a file
                                files_found = [match.get("path") for match in search_response.get("matches", []) if match.get("path")]
                                # Remove potential duplicates and None values
                                files_found = sorted(list(set(filter(None, files_found))))
                        else:
                            console.print("[yellow]Search returned no text content[/yellow]")
                            files_found = []

                        if files_found:
                            console.print(f"[green]Search found {len(files_found)} file(s) containing the pattern: {', '.join(files_found)}[/green]")

                            console.print("\n[bold cyan]Step 10.2: Running JQ to extract info from found files...[/bold cyan]")
                            # Step 2: Run JQ on the found files
                            await run_jq_demo(
                                session,
                                jq_tool,
                                demo_title="Demo 10.2: JQ Processing Search Results",
                                query='.[] | select(.role == "admin") | {file: input_filename, user: .username, email: .email}',
                                input_files=files_found, # Pass the list of files found by search
                                show_title=False
                            )
                        else:
                            console.print("[yellow]Search did not find any matching files.[/yellow]\n")
                            if search_response.get("error"):
                                console.print(f"[red]Search Error: {search_response.get('error')}[/red]\n")

                    except Exception as e:
                        console.print(f"[bold red]Error during Search/JQ synergy demo: {e}[/bold red]")
                        import traceback
                        console.print(traceback.format_exc())
                else:
                    console.print("[yellow]Search tool not found, skipping Demo 10[/yellow]")


    except Exception as e:
        console.print(f"[bold red]Error running demo: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())

async def run_jq_demo(session, jq_tool, query: str, input_json: str = None, input_files: list = None,
                      raw_output: bool = False, compact_output: bool = False,
                      monochrome_output: bool = False,
                      show_title: bool = True, demo_title: str = None):
    """Run a JQ demo with the given parameters."""
    if show_title and demo_title:
        console.print(Rule(f"[bold yellow]{demo_title}[/bold yellow]"))

    # Build the parameters dict for MCP tool call
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
             # Truncate long input JSON for display
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

    try:
        # Call the JQ tool using MCP protocol
        result = await session.call_tool(jq_tool.name, arguments=jq_params)
        
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
            except json.JSONDecodeError:
                console.print("[bold red]Failed to parse response as JSON[/bold red]")
                console.print(f"Raw response: {result_text[:200]}...")
                return
        else:
            console.print("[bold red]No text content in response[/bold red]")
            return
            
    except Exception as e:
        console.print(f"[bold red]Error during jq_query call: {e}[/bold red]")
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

    # Determine success based on direct keys in response
    # The service might return a flat structure without status/data division
    has_error = "error" in response and response["error"]
    exit_code = response.get("exit_code", 0)
    command = response.get("command", "N/A")
    output = response.get("output")
    parsed = response.get("parsed", False)
    execution_time = response.get("execution_time", execution_time_s)
    
    # Check if it's a success (either no error or exit_code 0)
    is_success = not has_error and (exit_code == 0 or "status" in response and response["status"] == "success")

    if is_success:
        results_table.add_row("Status", "[bold green]Success[/bold green]")
        results_table.add_row("Exit Code", str(exit_code))
        results_table.add_row("Parsed as JSON", str(parsed))
        results_table.add_row("Execution Time", f"{execution_time:.4f}s")
        
        if command:
            results_table.add_row("Command", Syntax(command, "bash", theme="monokai", word_wrap=True))

        if output is not None:
            # Determine how to display the output
            if parsed and not raw_output:
                # Pretty print JSON if it was parsed and not raw
                try:
                    if isinstance(output, str):
                        # If output is already a string, try to parse it as JSON
                        output_data = json.loads(output)
                        output_str = json.dumps(output_data, indent=2)
                    else:
                        # Otherwise assume it's already a Python object
                        output_str = json.dumps(output, indent=2)
                    output_syntax = Syntax(output_str, "json", theme="monokai", line_numbers=True, word_wrap=True)
                    results_table.add_row("Output (JSON)", output_syntax)
                except (TypeError, ValueError, json.JSONDecodeError):
                    # Fallback if json.dumps fails
                    results_table.add_row("Output", str(output))
            else:
                # Display as plain text if raw or not parsed
                output_str = str(output)
                # Heuristic: If it looks like JSON, highlight it anyway
                lang = "json" if (output_str.strip().startswith("{") or output_str.strip().startswith("[")) and not raw_output else "text"
                output_syntax = Syntax(output_str.strip(), lang, theme="monokai", line_numbers=True, word_wrap=True)
                results_table.add_row("Output (Text/Raw)", output_syntax)
        else:
            results_table.add_row("Output", "[dim]No output[/dim]")
    else:
        results_table.add_row("Status", "[bold red]Failed[/bold red]")
        
        # Handle various error formats
        if has_error:
            error = response["error"]
            if isinstance(error, dict):
                results_table.add_row("Error Code", error.get('code', 'N/A'))
                results_table.add_row("Message", error.get('message', 'Unknown error'))
            elif isinstance(error, str):
                results_table.add_row("Error", error)
        # If there's no explicit error but the command returned non-zero exit code
        elif exit_code != 0:
            results_table.add_row("Error", f"Command failed with exit code {exit_code}")
        # Fallback error message for other cases
        else:
            results_table.add_row("Error", "Unknown error")

        # Show command even on failure if available
        if command and command != "N/A (Client Error Before Call)":
            results_table.add_row("Attempted Command", Syntax(command, "bash", theme="monokai", word_wrap=True))

        # Show partial output if available (e.g., from stderr captured in output field on error)
        if output:
            results_table.add_row("Partial Output", Syntax(str(output).strip(), "text", theme="monokai", word_wrap=True))

    console.print(results_table)
    console.print("\n")

async def main():
    """Run the JQ demo."""
    if "--debug" in sys.argv:
        global DEBUG
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
        await jq_demo()
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
    except FileNotFoundError as e:
         console.print(f"[bold red]File Not Found Error: {str(e)}[/bold red]")
         console.print("Ensure the required example files and directories exist.")
    except Exception as e:
        console.print(f"[bold red]Unhandled error: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main()) 