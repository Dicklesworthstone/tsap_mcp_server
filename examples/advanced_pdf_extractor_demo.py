#!/usr/bin/env python3
"""
Advanced PDF Extractor Demo

This script demonstrates the comprehensive features of the PDF Extractor
tool in TSAP, including text extraction, table extraction, image analysis,
and metadata retrieval from PDF documents.
"""
import asyncio
import os
import sys
import re
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich.text import Text
from rich import box
from typing import Dict, Any, Optional, Union

# Import the MCP client from the library
from tsap.mcp import MCPClient

console = Console()

# --- Configuration ---
DEBUG = False
MAX_TEXT_DISPLAY_CHARS = 500 # Limit text output in demos
# --- End Configuration ---

# --- Example File Paths ---
# Ensure these paths are relative to the workspace root where the script is run
EXAMPLE_DATA_DIR = "tsap_example_data"
PDF_LAMPORT = os.path.join(EXAMPLE_DATA_DIR, "pdfs/lamport-paxos.pdf")
PDF_SLIDES = os.path.join(EXAMPLE_DATA_DIR, "pdfs/durnovo_presentation_slides.pdf")
# --- End Example File Paths ---


def debug_print(msg):
    """Print debug message if DEBUG is True."""
    if DEBUG:
        console.print(f"[dim][DEBUG] {msg}[/dim]")


async def pdf_extractor_demo():
    """Demonstrate PDF Extractor's advanced features."""
    console.print(Panel.fit(
        "[bold blue]TSAP PDF Extractor Advanced Features Demo[/bold blue]",
        subtitle="Extracting data from PDF files"
    ))

    # Check existence of essential files
    required_files = [PDF_LAMPORT, PDF_SLIDES]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        console.print(f"[bold red]Error:[/bold red] Required example PDF files not found: {', '.join(missing_files)}")
        console.print(f"Please ensure the '{EXAMPLE_DATA_DIR}' directory contains the required PDFs.")
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

            # DEMO 1: Metadata Extraction
            console.print(Rule("[bold yellow]Demo 1: Metadata Extraction[/bold yellow]"))
            console.print(f"[italic]Extracts metadata from '{os.path.basename(PDF_LAMPORT)}'.[/italic]\n")
            await run_pdf_demo(
                client,
                demo_title="Demo 1: Metadata Extraction",
                pdf_path=PDF_LAMPORT,
                extract_metadata=True,
                extract_text=False, # Only get metadata
                show_title=False
            )

            # Let's limit our demos for debugging
            if DEBUG:
                debug_print("DEBUG mode: Only running first demo")
                return

            # DEMO 2: Full Text Extraction
            console.print(Rule("[bold yellow]Demo 2: Full Text Extraction[/bold yellow]"))
            console.print(f"[italic]Extracts all text from '{os.path.basename(PDF_SLIDES)}' (output truncated).[/italic]\n")
            await run_pdf_demo(
                client,
                demo_title="Demo 2: Full Text Extraction",
                pdf_path=PDF_SLIDES,
                extract_text=True,
                pages="all", # Explicitly request all pages
                show_title=False
            )

            # DEMO 3: Specific Page Text Extraction
            console.print(Rule("[bold yellow]Demo 3: Specific Page Text Extraction[/bold yellow]"))
            console.print(f"[italic]Extracts text from pages 1-3 of '{os.path.basename(PDF_LAMPORT)}'.[/italic]\n")
            await run_pdf_demo(
                client,
                demo_title="Demo 3: Specific Page Text",
                pdf_path=PDF_LAMPORT,
                extract_text=True,
                pages="1-3", # Page range
                show_title=False
            )

            # DEMO 4: Specific Page List Text Extraction
            console.print(Rule("[bold yellow]Demo 4: Specific Page List Text Extraction[/bold yellow]"))
            console.print(f"[italic]Extracts text from pages 1, 5, and 10 of '{os.path.basename(PDF_LAMPORT)}'.[/italic]\n")
            await run_pdf_demo(
                client,
                demo_title="Demo 4: Page List Text",
                pdf_path=PDF_LAMPORT,
                extract_text=True,
                pages=[1, 5, 10], # List of pages
                show_title=False
            )

            # DEMO 5: Image Information Extraction
            console.print(Rule("[bold yellow]Demo 5: Image Information Extraction[/bold yellow]"))
            console.print(f"[italic]Extracts information about images in '{os.path.basename(PDF_SLIDES)}'. (Requires PyMuPDF on server)[/italic]\n")
            await run_pdf_demo(
                client,
                demo_title="Demo 5: Image Info Extraction",
                pdf_path=PDF_SLIDES,
                extract_images=True,
                pages="all",
                show_title=False
            )

            # DEMO 6: Table Information Extraction (Attempt)
            console.print(Rule("[bold yellow]Demo 6: Table Information Extraction (Attempt)[/bold yellow]"))
            console.print(f"[italic]Attempts to extract table information from '{os.path.basename(PDF_LAMPORT)}'. (Feature may be limited)[/italic]\n")
            await run_pdf_demo(
                client,
                demo_title="Demo 6: Table Info Extraction",
                pdf_path=PDF_LAMPORT,
                extract_tables=True,
                pages="1-5", # Limit pages for demo
                show_title=False
            )

            # DEMO 7: Combined Extraction
            console.print(Rule("[bold yellow]Demo 7: Combined Extraction[/bold yellow]"))
            console.print(f"[italic]Extracts metadata, text (page 1), and image info from '{os.path.basename(PDF_SLIDES)}'.[/italic]\n")
            await run_pdf_demo(
                client,
                demo_title="Demo 7: Combined Extraction",
                pdf_path=PDF_SLIDES,
                extract_metadata=True,
                extract_text=True,
                extract_images=True,
                pages=[1], # Only text from page 1
                show_title=False
            )

            # DEMO 8: Synergy with Ripgrep - Find PDFs then get Metadata
            console.print(Rule("[bold yellow]Demo 8: Synergy - Find PDFs + PDF Metadata[/bold yellow]"))
            console.print(f"[italic]Uses Python's glob to find PDF files in '{EXAMPLE_DATA_DIR}', then extracts metadata for each.[/italic]\n")

            console.print("[bold cyan]Step 8.1: Using glob to find PDF files...[/bold cyan]")
            files_found = []
            try:
                # Use glob to find files matching the pattern
                import glob
                search_pattern = os.path.join(EXAMPLE_DATA_DIR, "pdfs/*.pdf")
                files_found = sorted(glob.glob(search_pattern))

                if files_found:
                    console.print(f"[green]Glob found {len(files_found)} PDF file(s): {', '.join(files_found)}[/green]")
                else:
                     console.print(f"[yellow]Glob did not find any PDF files matching '{search_pattern}'.[/yellow]\n")

            except Exception as e:
                console.print(f"[bold red]Error during glob file search: {e}[/bold red]")
                import traceback
                console.print(traceback.format_exc())

            # Proceed only if files were found
            if files_found:
                 console.print("\n[bold cyan]Step 8.2: Extracting metadata for each found PDF...[/bold cyan]")
                 for pdf_file in files_found:
                     await run_pdf_demo(
                         client,
                         demo_title=f"Metadata for {os.path.basename(pdf_file)}",
                         pdf_path=pdf_file,
                         extract_metadata=True,
                         extract_text=False,
                         show_title=True # Show title for each file
                     )

            # DEMO 9: Synergy - Extract Text then Process with Regex
            console.print(Rule("[bold yellow]Demo 9: Synergy - Extract Text + Regex Search[/bold yellow]"))
            console.print(f"[italic]Extracts text from '{os.path.basename(PDF_LAMPORT)}' and searches for email-like patterns within the script.[/italic]\n")

            console.print("[bold cyan]Step 9.1: Extracting text from PDF...[/bold cyan]")
            text_content: Optional[Union[str, Dict[int, str]]] = None
            try:
                extract_response = await client.pdf_extract(
                    pdf_path=PDF_LAMPORT,
                    extract_text=True,
                    pages="all" # Get all text
                )
                if extract_response and extract_response.get("data") and extract_response["data"].get("text"):
                    text_content = extract_response["data"]["text"]
                    console.print(f"[green]Successfully extracted text (Total chars: {len(text_content) if isinstance(text_content, str) else 'N/A'}).[/green]")
                else:
                    console.print("[yellow]Failed to extract text or text was empty.[/yellow]")
                    err = extract_response.get('error') if isinstance(extract_response, dict) else "Unknown error"
                    if err: 
                        console.print(f"[red]Error: {err}[/red]")


            except Exception as e:
                console.print(f"[bold red]Error during PDF text extraction for Regex demo: {e}[/bold red]")

            if isinstance(text_content, str) and text_content:
                console.print("\n[bold cyan]Step 9.2: Searching extracted text for email patterns using Regex...[/bold cyan]")
                # Simple regex for things that look like emails
                email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
                matches = re.findall(email_pattern, text_content)

                if matches:
                    console.print(f"[green]Found {len(matches)} potential email patterns:[/green]")
                    # Print unique matches
                    unique_matches = sorted(list(set(matches)))
                    for match in unique_matches:
                        console.print(f"- {match}")
                else:
                    console.print("[yellow]No email-like patterns found in the extracted text.[/yellow]")
            elif text_content:
                 console.print("[yellow]Cannot perform regex search because extracted text is per-page, not a single string.[/yellow]")


    except Exception as e:
        console.print(f"[bold red]Error running demo: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())


async def run_pdf_demo(client, demo_title: str, show_title: bool = True, **kwargs):
    """Run a PDF extraction demo with the given parameters."""
    if show_title:
        console.print(Rule(f"[bold yellow]{demo_title}[/bold yellow]"))

    # Build the parameters dict cleanly
    extract_params = {k: v for k, v in kwargs.items() if v is not None} # Filter out None values for clarity

    # Show the parameters
    console.print("[bold cyan]PDF Extractor Parameters:[/bold cyan]")
    params_table = Table(show_header=False, box=None)
    params_table.add_column("Parameter", style="green")
    params_table.add_column("Value", style="white")

    # Sort params for consistent display
    sorted_params = sorted(extract_params.items())

    for key, value in sorted_params:
         display_value: Any
         if key == "pdf_path":
             display_value = os.path.basename(value) # Show only filename
         elif isinstance(value, list):
             display_value = ", ".join(map(str, value))
         else:
             display_value = str(value)
         params_table.add_row(key, display_value)

    console.print(params_table)
    console.print()

    # Execute the PDF extraction process
    start_time_dt = datetime.now()
    console.print("[bold]Executing PDF extraction...[/bold]")

    response: Optional[dict] = None
    try:
        # Use the client's generic request or a dedicated method if available
        # Assuming a generic request structure for now
        response = await client.pdf_extract(**extract_params)

    except Exception as e:
        console.print(f"[bold red]Error during client.pdf_extract call: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        response = {"error": {"code": "CLIENT_SIDE_ERROR", "message": str(e)}}

    # Calculate execution time
    execution_time_s = (datetime.now() - start_time_dt).total_seconds()

    # Process and display results
    console.print("[bold cyan]PDF Extractor Results:[/bold cyan]")
    results_table = Table(show_header=False, box=None, padding=(0, 1))
    results_table.add_column("Field", style="green", justify="right")
    results_table.add_column("Value", style="white", justify="left")

    # Determine success based on presence of 'data' and no 'error'
    is_success = isinstance(response, dict) and "data" in response and not response.get("error")
    data = response.get("data") if isinstance(response, dict) else None
    error = response.get("error") if isinstance(response, dict) else None

    # Adjust execution time if available from server response
    server_exec_time = data.get('execution_time') if isinstance(data, dict) else None
    display_exec_time = server_exec_time if server_exec_time is not None else execution_time_s

    if is_success and isinstance(data, dict):
        results_table.add_row("Status", "[bold green]Success[/bold green]")
        results_table.add_row("Execution Time", f"{display_exec_time:.4f}s")
        results_table.add_row("Page Count", str(data.get("page_count", "N/A")))

        # Display Metadata if present
        metadata_content = data.get("metadata")
        if isinstance(metadata_content, dict):
            meta_table = Table(title="Metadata", show_header=False, box=box.ROUNDED, padding=(0,1))
            meta_table.add_column("Key", style="blue")
            meta_table.add_column("Value", style="magenta")
            if metadata_content:
                for k, v in sorted(metadata_content.items()):
                     # Truncate long values like page_sizes for display
                     v_str = str(v)
                     if len(v_str) > 100:
                          v_str = v_str[:100] + "..."
                     meta_table.add_row(k, v_str)
            results_table.add_row("Metadata", meta_table)
        elif extract_params.get("extract_metadata"):
             results_table.add_row("Metadata", "[dim]Not extracted or empty[/dim]")

        # Display Text if present
        text_content = data.get("text")
        if text_content:
             text_display = Text()
             if isinstance(text_content, str):
                 num_chars = len(text_content)
                 text_display.append(f"Extracted Text ({num_chars} chars):\n")
                 truncated_text = text_content[:MAX_TEXT_DISPLAY_CHARS]
                 text_display.append(truncated_text, style="dim")
                 if num_chars > MAX_TEXT_DISPLAY_CHARS:
                     text_display.append(f"\n... [Truncated - {num_chars - MAX_TEXT_DISPLAY_CHARS} more chars]")
             elif isinstance(text_content, dict):
                 text_display.append(f"Extracted Text (by Page - {len(text_content)} pages):\n")
                 for page_num, page_text in sorted(text_content.items()):
                      num_chars = len(page_text)
                      truncated_text = page_text[:MAX_TEXT_DISPLAY_CHARS//len(text_content)] # Rough split
                      text_display.append(f"\n--- Page {page_num} ({num_chars} chars) ---\n", style="bold yellow")
                      text_display.append(truncated_text, style="dim")
                      if num_chars > len(truncated_text):
                           text_display.append(f"\n... [Truncated - {num_chars - len(truncated_text)} more chars]")
             else:
                  text_display.append(f"Text (Unknown Format): {type(text_content)}")

             results_table.add_row("Text", text_display)
        elif extract_params.get("extract_text"):
             results_table.add_row("Text", "[dim]Not extracted or empty[/dim]")

        # Display Images info if present
        images_info = data.get("images")
        if isinstance(images_info, list):
            img_count = len(images_info)
            img_table = Table(title=f"Images Found ({img_count})", box=box.ROUNDED)
            if images_info:
                img_table.add_column("Page", style="cyan")
                img_table.add_column("Index", style="yellow")
                img_table.add_column("Size (WxH)", style="magenta")
                img_table.add_column("Type", style="blue")
                img_table.add_column("XRef", style="green")
                for img in images_info[:10]: # Show first 10 images
                     img_table.add_row(
                          str(img.get('page', '?')),
                          str(img.get('index', '?')),
                          f"{img.get('width','?')}x{img.get('height','?')}",
                          str(img.get('type', '?')),
                          str(img.get('xref', '?')),
                     )
                if len(images_info) > 10:
                     img_table.caption = f"... and {len(images_info) - 10} more images not shown"
            else:
                 img_table.add_row("[dim]No images found or extracted[/dim]")
            results_table.add_row("Images", img_table)
        elif extract_params.get("extract_images"):
             results_table.add_row("Images", "[dim]Not extracted or empty[/dim]")


        # Display Tables info if present
        tables_info = data.get("tables")
        if isinstance(tables_info, list):
            tbl_count = len(tables_info)
            tbl_table = Table(title=f"Tables Found ({tbl_count})", box=box.ROUNDED)
            if tables_info:
                tbl_table.add_column("Page", style="cyan")
                tbl_table.add_column("Index", style="yellow")
                tbl_table.add_column("Details", style="magenta") # Placeholder
                for tbl_idx, tbl in enumerate(tables_info[:10]): # Show first 10 tables
                     tbl_table.add_row(
                          str(tbl.get('page', '?')),
                          str(tbl_idx + 1),
                          str(tbl) # Display raw table info for now
                     )
                if len(tables_info) > 10:
                     tbl_table.caption = f"... and {len(tables_info) - 10} more tables not shown"
                tbl_table.add_row("[dim i]Note: Table content extraction might be limited.[/dim i]", span=3)
            else:
                 tbl_table.add_row("[dim]No tables found or extracted[/dim]")
            results_table.add_row("Tables", tbl_table)
        elif extract_params.get("extract_tables"):
             results_table.add_row("Tables", "[dim]Not extracted or empty (or feature limited)[/dim]")


    else: # Handle failure case
        results_table.add_row("Status", "[bold red]Failed[/bold red]")
        if error and isinstance(error, dict):
            results_table.add_row("Error Code", error.get('code', 'N/A'))
            results_table.add_row("Message", error.get('message', 'Unknown error'))
        elif isinstance(error, str):
             results_table.add_row("Error", error)
        else:
             results_table.add_row("Error", f"[dim]Unknown error details. Response: {response}[/dim]")

        # Show partial data if available
        if isinstance(data, dict):
            if data.get("page_count"):
                results_table.add_row("Page Count (Partial)", str(data["page_count"]))
            if data.get("execution_time"):
                 results_table.add_row("Execution Time (Partial)", f"{data['execution_time']:.4f}s")


    console.print(results_table)
    console.print("\n") # Add space between demos


if __name__ == "__main__":
    if "--debug" in sys.argv:
        DEBUG = True
        debug_print("Debug mode enabled")

    # Check example data directory and files again before running
    example_data_path = Path(EXAMPLE_DATA_DIR)
    if not example_data_path.is_dir():
         console.print(f"[bold red]Error:[/bold red] Example data directory '{EXAMPLE_DATA_DIR}' not found.")
         console.print("Please ensure the directory exists and contains the required PDF files.")
         sys.exit(1)

    required_files_main = [Path(PDF_LAMPORT), Path(PDF_SLIDES)]
    missing_files_main = [f for f in required_files_main if not f.exists()]
    if missing_files_main:
        console.print(f"[bold red]Error:[/bold red] Required example PDF files missing: {', '.join(map(str, missing_files_main))}")
        sys.exit(1)


    try:
        asyncio.run(pdf_extractor_demo())
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
    except NameError as e:
        if 'MCPClient' in str(e):
             # Import error already handled
             pass
        else:
            console.print(f"[bold red]Unhandled NameError: {str(e)}[/bold red]")
            import traceback
            console.print(traceback.format_exc())
    except FileNotFoundError as e:
         console.print(f"[bold red]File Not Found Error: {str(e)}[/bold red]")
         console.print("Ensure the required example files and directories exist relative to your workspace.")
    except Exception as e:
        console.print(f"[bold red]Unhandled error: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
