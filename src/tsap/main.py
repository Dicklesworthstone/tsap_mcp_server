#!/usr/bin/env python3
"""
TSAP Command Line Interface

This is the main entry point for the TSAP command-line interface.
It provides direct access to TSAP's core functionalities.
"""
import json
import time
from typing import Optional, Any, Dict

import typer

from tsap.utils.logging import console
from tsap.core.html_processor import get_html_processor

# Create the app
app = typer.Typer(
    name="tsap",
    help="TSAP MCP Server - Text Search and Processing Model Context Protocol",
    add_completion=False,
)

# Track start time
@app.callback()
def callback(ctx: typer.Context):
    """Set up the context for the TSAP CLI."""
    ctx.ensure_object(dict)
    ctx.obj["start_time"] = time.time()

def format_json(data: Dict[str, Any]) -> str:
    """Format JSON data for display."""
    return json.dumps(data, indent=2, ensure_ascii=False)

@app.command()
async def html_process(
    ctx: typer.Context,
    html: Optional[str] = typer.Option(None, "--html", help="HTML content to process"),
    url: Optional[str] = typer.Option(None, "--url", help="URL to fetch HTML from"),
    file_path: Optional[str] = typer.Option(None, "--file", help="Path to HTML file"),
    selector: Optional[str] = typer.Option(None, "--selector", help="CSS selector to extract elements"),
    xpath: Optional[str] = typer.Option(None, "--xpath", help="XPath to extract elements"),
    extract_tables: bool = typer.Option(False, "--tables", help="Extract tables from HTML"),
    extract_links: bool = typer.Option(False, "--links", help="Extract links from HTML"),
    extract_text: bool = typer.Option(False, "--text", help="Extract plain text from HTML"),
    extract_metadata: bool = typer.Option(False, "--metadata", help="Extract metadata from HTML"),
    render_js: bool = typer.Option(False, "--render-js", help="Render JavaScript before processing HTML"),
    js_timeout: int = typer.Option(30, "--js-timeout", help="Timeout in seconds for JavaScript rendering"),
    interactive_actions_file: Optional[str] = typer.Option(
        None, 
        "--interactive-actions", 
        help="Path to JSON file containing interactive actions for JavaScript rendering"
    ),
    extract_computed_styles: bool = typer.Option(
        False, 
        "--computed-styles", 
        help="Extract computed CSS styles for elements (requires --render-js)"
    ),
    output_file: Optional[str] = typer.Option(None, "--output", help="Path to write output JSON"),
) -> Any:
    """Process HTML content."""
    try:
        # Check that at least one source is provided
        if all(s is None for s in [html, url, file_path]):
            error_json = {"status": "error", "error": {"code": "MISSING_SOURCE", "message": "At least one of --html, --url, or --file must be provided"}}
            console.print(format_json(error_json))
            return error_json
            
        # Parse interactive actions from file if provided
        interactive_actions = None
        if interactive_actions_file:
            try:
                with open(interactive_actions_file, 'r') as f:
                    interactive_actions = json.load(f)
            except Exception as e:
                error_json = {"status": "error", "error": {"code": "INVALID_ACTIONS_FILE", "message": f"Failed to parse interactive actions file: {str(e)}"}}
                console.print(format_json(error_json))
                return error_json
        
        # Make sure computed styles extraction is only requested with render_js
        if extract_computed_styles and not render_js:
            console.print("[yellow]Warning: --computed-styles requires --render-js, enabling JavaScript rendering.[/yellow]")
            render_js = True
        
        # Process HTML
        html_processor = get_html_processor()
        result = await html_processor.process_html(
            html=html,
            url=url,
            file_path=file_path,
            selector=selector,
            xpath=xpath,
            extract_tables=extract_tables,
            extract_links=extract_links,
            extract_text=extract_text,
            extract_metadata=extract_metadata,
            render_js=render_js,
            js_timeout=js_timeout,
            interactive_actions=interactive_actions,
            extract_computed_styles=extract_computed_styles
        )
        
        # Add execution time to the result
        result["execution_time"] = time.time() - ctx.obj["start_time"]
        
        # Format output
        output = {
            "status": "success",
            "data": result
        }
        
        # Write to file if requested
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
                
        # Print to console
        console.print(format_json(output))
        return output
        
    except Exception as e:
        error_json = {"status": "error", "error": {"code": "PROCESSING_ERROR", "message": str(e)}}
        console.print(format_json(error_json))
        return error_json

if __name__ == "__main__":
    # Use standard asyncio event loop by default
    typer.run(app()) 