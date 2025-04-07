#!/usr/bin/env python3
"""
Code Security Audit (MCP Tools Version)

This script demonstrates how to perform a security audit on code files
using the MCP tools interface.
"""
import asyncio
import os
import json
import argparse
from typing import List

from rich import print as rich_print
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.console import Console

# Import proper MCP client modules
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Create console instance
console = Console()

# --- Configuration ---
DEFAULT_CODE_DIR = "tsap_example_data/code"  # Directory containing code to audit

# --- Parse Command-Line Arguments ---
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run a security audit on code using MCP tools.")
    parser.add_argument(
        "--paths", 
        nargs="+", 
        default=[DEFAULT_CODE_DIR],
        help=f"Paths to audit (default: {DEFAULT_CODE_DIR})"
    )
    parser.add_argument(
        "--severity", 
        choices=["low", "medium", "high", "critical"],
        default="low",
        help="Minimum severity level to report (default: low)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    return parser.parse_args()

# --- Main Audit Function ---
async def run_code_audit(session, analyze_code_tool, paths: List[str], severity: str = "low", verbose: bool = False):
    """Runs a security audit on the specified code paths using MCP tools."""
    rich_print(Panel(f"[bold blue]Starting Security Audit on:[/bold blue] {', '.join(paths)}", expand=False))
    rich_print(f"Minimum severity level: [bold]{severity.upper()}[/bold]")

    # --- Define Tool Parameters ---
    tool_params = {
        "paths": paths,
        "analysis_types": ["security"], # Focus on security analysis
    }
    
    if verbose:
        rich_print("[cyan]Sending Analyze Code Tool Request:[/cyan]")
        rich_print(Syntax(json.dumps(tool_params, indent=2), "json", theme="default"))

    try:
        # --- Make MCP Tools Request ---
        result = await session.call_tool(
            analyze_code_tool.name,
            arguments=tool_params
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
                response = json.loads(result_text)
                
                # --- Process MCP Response ---
                if not response:
                    rich_print("[bold red]Tool request failed: No response received.[/bold red]")
                    return
                
                # Check for security results
                security_results = response.get("security")

                if not security_results:
                    rich_print("[bold green]No security issues found (or security analysis not performed).[/bold green]")
                    return

                # Filter findings based on severity threshold
                findings = []
                severity_map = {"low": 1, "medium": 2, "high": 3, "critical": 4}
                min_severity_level = severity_map.get(severity.lower(), 2)

                for category, category_issues in security_results.get("issues_by_category", {}).items():
                    for issue in category_issues:
                        # Add category and severity from the pattern definition if available
                        issue_severity_str = issue.get("severity", "low") # Default to low if not present
                        issue_severity_level = severity_map.get(issue_severity_str.lower(), 1)

                        if issue_severity_level >= min_severity_level:
                            findings.append({
                                "file": issue.get("file", "N/A"),
                                "line": issue.get("line", "N/A"),
                                "category": category,
                                "description": issue.get("description", "N/A"),
                                "text": issue.get("text", "N/A"),
                                "severity": issue_severity_str.upper()
                            })

                # Sort findings by severity (desc) then file/line
                findings.sort(key=lambda x: (-severity_map.get(x["severity"].lower(), 0), x["file"], x["line"]))

                # --- Display Results ---
                num_findings = len(findings)
                summary_color = "green" if num_findings == 0 else "yellow" if num_findings > 0 and not any(f['severity'] in ['HIGH', 'CRITICAL'] for f in findings) else "red"
                rich_print(Panel(f"[bold {summary_color}]Found {num_findings} security issues (severity >= {severity.upper()})[/bold {summary_color}]", expand=False))

                if num_findings > 0:
                    table = Table(title="Security Audit Findings")
                    table.add_column("Severity", style="bold", justify="center")
                    table.add_column("File", style="cyan")
                    table.add_column("Line", style="yellow", justify="right")
                    table.add_column("Category", style="magenta")
                    table.add_column("Description", style="white")
                    table.add_column("Finding", style="default")

                    severity_styles = {
                        "CRITICAL": "bold red",
                        "HIGH": "red",
                        "MEDIUM": "yellow",
                        "LOW": "dim"
                    }

                    for finding in findings:
                        severity_style = severity_styles.get(finding["severity"], "default")
                        table.add_row(
                            f"[{severity_style}]{finding['severity']}[/]",
                            finding["file"],
                            str(finding["line"]),
                            finding["category"],
                            finding["description"],
                            finding["text"].strip()
                        )
                    console.print(table)
            except json.JSONDecodeError:
                rich_print("[bold red]Failed to parse response as JSON[/bold red]")
                rich_print(f"Raw response: {result_text[:200]}...")
        else:
            rich_print("[bold red]No text content in response[/bold red]")

    except Exception as e:
        rich_print(f"[bold red]Error during security audit: {str(e)}[/bold red]")
        if verbose:
            import traceback
            rich_print(traceback.format_exc())

# --- Main Execution ---
async def main():
    """Main function to run the security audit example."""
    args = parse_args()
    
    # Validate paths
    for path in args.paths:
        if not os.path.exists(path):
            rich_print(f"[bold red]Error:[/bold red] Path '{path}' not found.")
            return

    # Path to the proxy script
    proxy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_proxy.py")
    rich_print(f"Using proxy script: {proxy_path}")
    
    # Configure server parameters for stdio connection to our proxy
    server_params = StdioServerParameters(
        command="python",
        args=[proxy_path],
        env={"MCP_DEBUG": "1" if args.verbose else "0"}  # Enable debug logging if verbose
    )
    
    try:
        # Connect to the MCP server via proxy
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()
                
                # Get the list of available tools
                tools_result = await session.list_tools()
                tools = tools_result.tools
                
                # Find the analyze_code and info tools
                analyze_code_tool = next((t for t in tools if t.name == "analyze_code"), None)
                info_tool = next((t for t in tools if t.name == "info"), None)
                
                if not analyze_code_tool:
                    rich_print("[bold red]Error: analyze_code tool not found![/bold red]")
                    return
                
                # Check server health
                if info_tool:
                    rich_print("Checking server info...")
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
                                rich_print("[green]MCP server check successful.[/green]")
                            except json.JSONDecodeError:
                                rich_print("[yellow]Info response is not JSON[/yellow]")
                        else:
                            rich_print("[yellow]Info call succeeded but returned no readable content.[/yellow]")
                    except Exception as e:
                        rich_print(f"[bold red]Error during info check: {str(e)}[/bold red]")
                        return
                
                await run_code_audit(
                    session, 
                    analyze_code_tool,
                    paths=args.paths, 
                    severity=args.severity,
                    verbose=args.verbose
                )
    except Exception as e:
        rich_print(f"[bold red]Error connecting to MCP server: {str(e)}[/bold red]")
        if args.verbose:
            import traceback
            rich_print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main()) 