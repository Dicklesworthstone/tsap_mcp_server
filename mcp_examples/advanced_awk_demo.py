#!/usr/bin/env python3
"""
Advanced AWK Demo (MCP Protocol Version)

This script demonstrates the comprehensive features of the AWK Process
tool in TSAP, including text processing, data extraction, and transformation
using the standard MCP protocol client.
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

async def awk_demo():
    """Demonstrate AWK's advanced features with example data."""
    console.print(Panel.fit(
        "[bold blue]TSAP MCP AWK Advanced Features Demo[/bold blue]",
        subtitle="Processing data with AWK tools"
    ))

    # Path to the proxy script
    proxy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_proxy.py")
    debug_print(f"Proxy path: {proxy_path}")
    
    # Configure server parameters for stdio connection
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
                init_result = await session.initialize()
                console.print(f"[green]Connected to {init_result.serverInfo.name} {init_result.serverInfo.version}[/green]")
                debug_print("Session initialized successfully")
                
                # Get available tools
                tools_result = await session.list_tools()
                tools = tools_result.tools
                debug_print(f"Found {len(tools)} tools")
                
                # Find the awk tool
                awk_tool = next((t for t in tools if t.name == "awk_process"), None)
                if not awk_tool:
                    console.print("[bold red]Error: awk_process tool not found![/bold red]")
                    available_tools = [t.name for t in tools]
                    console.print(f"Available tools: {', '.join(available_tools)}")
                    return
                
                # Find the info tool for initial check
                info_tool = next((t for t in tools if t.name == "info"), None)
                
                # --- Add initial info check ---
                if info_tool:
                    console.print("Checking server info...")
                    try:
                        info_result = await session.call_tool(info_tool.name, arguments={})
                        
                        # Extract the text content
                        info_text = None
                        for content in info_result.content:
                            if content.type == "text":
                                info_text = content.text
                                break
                        
                        if info_text:
                            try:
                                info_data = json.loads(info_text)  # noqa: F841
                                console.print("[green]Initial server info check successful.[/green]")
                            except json.JSONDecodeError:
                                console.print(f"[yellow]Info response is not JSON: {info_text[:100]}...[/yellow]")
                        else:
                            console.print("[yellow]No text content in info response[/yellow]")
                    except Exception as e:
                        console.print(f"[bold red]Error during info check: {str(e)}[/bold red]")
                        debug_print(f"Info check error details: {e}")
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
                    session,
                    awk_tool,
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
                    session,
                    awk_tool,
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
                    session,
                    awk_tool,
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
                        session,
                        awk_tool,
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
                    session,
                    awk_tool,
                    demo_title="Demo 5: Pattern Matching",
                    script='/line/ {print "Found match on line " NR ":", $0}', # Only operates on lines matching /line/
                    input_text=sample_text_string,
                    show_title=False
                )

                # DEMO 6: Summing Column (Using $NF)
                console.print(Rule("[bold yellow]Demo 6: Performing Calculations[/bold yellow]"))
                console.print("[italic]Sums the values in the last column ($NF) of the sample string.[/italic]\n")
                await run_awk_demo(
                    session,
                    awk_tool,
                    demo_title="Demo 6: Summing Column",
                    script='BEGIN { sum=0 } { sum += ($NF+0) } END { print "Total sum of column 5:", sum }',
                    input_text=sample_text_string,
                    show_title=False
                )

                # DEMO 7: Synergy with Search Tool - Find Python Function Definitions
                console.print(Rule("[bold yellow]Demo 7: Synergy with Search Tool[/bold yellow]"))
                console.print("[italic]Uses search to find Python function definitions, then AWK to extract names.[/italic]\n")
                # Check if code directory exists
                code_dir = "tsap_example_data/code/"
                if os.path.isdir(code_dir):
                    # Step 1: Run search to find lines starting with 'def '
                    console.print("[bold cyan]Step 7.1: Running search to find function definitions...[/bold cyan]")
                    try:
                        # Find search tool
                        search_tool = next((t for t in tools if t.name == "search" or t.name == "ripgrep_search"), None)
                        if not search_tool:
                            console.print("[yellow]Search tool not found, skipping demo 7[/yellow]")
                        else:
                            # Use the MCP search tool
                            search_params = {
                                "pattern": r"^[ \t]*def\s+", # Regex for lines starting with optional space/tabs then 'def '
                                "paths": [code_dir],
                                "file_patterns": ["*.py"],
                                "regex": True,
                                "max_count": 20 # Limit matches
                            }
                            
                            search_result = await session.call_tool(search_tool.name, arguments=search_params)
                            
                            # Extract the text content
                            search_response_text = None
                            for content in search_result.content:
                                if content.type == "text":
                                    search_response_text = content.text
                                    break
                            
                            if search_response_text:
                                try:
                                    search_response = json.loads(search_response_text)
                                    
                                    # Different tools might have different response structures
                                    if isinstance(search_response, dict) and "matches" in search_response:
                                        matches = search_response["matches"]
                                    elif isinstance(search_response, dict) and "data" in search_response and "matches" in search_response["data"]:
                                        matches = search_response["data"]["matches"]
                                    else:
                                        matches = []
                                        
                                    if matches:
                                        console.print(f"[green]Search found {len(matches)} potential function definitions.[/green]")

                                        # Concatenate the line text from matches for AWK input
                                        search_output_lines = [match.get("line_text", "") for match in matches]
                                        awk_input_from_search = "\n".join(search_output_lines).strip()

                                        if awk_input_from_search:
                                            console.print("\n[bold cyan]Step 7.2: Running AWK to extract function names...[/bold cyan]")
                                            # Step 2: Run AWK to process search output
                                            await run_awk_demo(
                                                session,
                                                awk_tool,
                                                demo_title="Demo 7.2: AWK Processing Search Output",
                                                # Refined regex to capture function name up to '('
                                                script=r'/^[ \t]*def / { match($0, /def +([^(]+)/, arr); if (arr[1]) print arr[1] }',
                                                input_text=awk_input_from_search,
                                                show_title=False
                                            )
                                        else:
                                            console.print("[yellow]No lines extracted from search output for AWK processing.[/yellow]\n")

                                    else:
                                        console.print("[yellow]Search found no matches.[/yellow]")
                                
                                except json.JSONDecodeError:
                                    console.print("[bold red]Failed to parse search response as JSON[/bold red]")
                                    console.print(f"Raw response: {search_response_text[:200]}...")
                            else:
                                console.print("[bold red]No text content in search response[/bold red]")
                    except Exception as e:
                        console.print(f"[bold red]Error during Search/AWK synergy demo: {e}[/bold red]")
                        import traceback
                        console.print(traceback.format_exc())
                else:
                    console.print(f"[yellow]Skipping Demo 7: Directory not found '{code_dir}'[/yellow]\n")

                # DEMO 8: BEGIN and END blocks
                console.print(Rule("[bold yellow]Demo 8: BEGIN and END Blocks[/bold yellow]"))
                console.print("[italic]Demonstrates setup (BEGIN) and summary (END) actions.[/italic]\n")
                await run_awk_demo(
                    session,
                    awk_tool,
                    demo_title="Demo 8: BEGIN/END Blocks",
                    script='BEGIN { print "Starting processing..."; count = 0 } { count++ } END { print "Processed " count " lines." }',
                    input_text=sample_text_string,
                    show_title=False
                )

                # DEMO 9: Frequency Counting (Associative Arrays)
                console.print(Rule("[bold yellow]Demo 9: Frequency Counting (Associative Arrays)[/bold yellow]"))
                console.print("[italic]Counts the frequency of each unique value in the first field ($1) of the sample string.[/italic]\n")
                await run_awk_demo(
                    session,
                    awk_tool,
                    demo_title="Demo 9: Frequency Counting",
                    script='{ count[$1]++ } END { for (item in count) print item, count[item] }',
                    input_text=sample_text_string,
                    show_title=False
                )

                # DEMO 10: Grouping and Aggregation (Associative Arrays)
                console.print(Rule("[bold yellow]Demo 10: Grouping and Aggregation (Associative Arrays)[/bold yellow]"))
                console.print("[italic]Groups lines by the first field ($1) and sums the last field ($NF) for each group.[/italic]\n")
                await run_awk_demo(
                    session,
                    awk_tool,
                    demo_title="Demo 10: Grouping & Summation",
                    script='{ sum[$1] += $NF } END { for (item in sum) printf "%s: Total = %d\\n", item, sum[item] }',
                    input_text=sample_text_string,
                    show_title=False
                )

    except Exception as e:
        console.print(f"[bold red]Error running demo: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())

async def run_awk_demo(session, awk_tool, script: str, input_text: str = None, input_files: list = None,
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

    try:
        # Call the awk tool through the MCP session
        result = await session.call_tool(awk_tool.name, arguments=awk_params)
        
        # Extract the text content from the response
        response_text = None
        for content in result.content:
            if content.type == "text":
                response_text = content.text
                break
        
        if response_text:
            try:
                response = json.loads(response_text)
            except json.JSONDecodeError:
                console.print("[bold red]Failed to parse response as JSON[/bold red]")
                console.print(f"Raw response: {response_text[:200]}...")
                return
        else:
            console.print("[bold red]No text content in response[/bold red]")
            return
    except Exception as e:
        console.print(f"[bold red]Error during awk tool call: {e}[/bold red]")
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

    # Handle different response formats
    if isinstance(response, dict) and "data" in response:
        data = response.get("data", {})
    else:
        data = response
        
    error = response.get("error") if isinstance(response, dict) else None

    # The command appears to be executing successfully even though the status doesn't indicate success
    # Look for output as the primary indicator of success instead of status
    has_output = False
    if isinstance(response, dict):
        if isinstance(data, dict) and "output" in data and data["output"].strip():
            has_output = True
        elif "output" in response and response["output"].strip():
            has_output = True
    
    # Check command execution success based on output presence
    is_success = has_output

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

    # Fallback for completely unexpected response
    if not response:
        results_table.add_row("Response", str(response))

    console.print(results_table)
    console.print("\n")  # Add space between demos

async def main():
    """Run the AWK demo."""
    if "--debug" in sys.argv:
        global DEBUG
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
        await awk_demo()
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Unhandled error: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main()) 