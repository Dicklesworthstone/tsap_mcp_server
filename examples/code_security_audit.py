# examples/code_security_audit.py
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

# from api_client import TSAPClient, API_KEY # Import from our client module
from mcp_client_example import MCPClient # Use MCPClient instead

# Create console instance
console = Console()

# --- Configuration ---
DEFAULT_CODE_DIR = "tsap_example_data/code"  # Directory containing code to audit

# --- Parse Command-Line Arguments ---
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run a security audit on code.")
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
async def run_code_audit(client: MCPClient, paths: List[str], severity: str = "low", verbose: bool = False):
    """Runs a security audit on the specified code paths using MCPClient."""
    rich_print(Panel(f"[bold blue]Starting Security Audit on:[/bold blue] {', '.join(paths)}", expand=False))
    rich_print(f"Minimum severity level: [bold]{severity.upper()}[/bold]")

    # --- Define MCP Command Arguments ---
    # Based on CodeAnalyzerParams model
    mcp_args = {
        "file_paths": paths,
        "analysis_types": ["security"], # Focus on security analysis
        # Add other relevant params from CodeAnalyzerParams if needed, e.g.:
        # "language": None, # Auto-detect language
        # "include_metrics": False,
    }
    
    # Removed params specific to the old API client:
    # "include_file_content", "highlight_matches", "performance_mode", "async_execution"
    # MCPClient handles requests differently; check MCP command definition for equivalent options if necessary.

    if verbose:
        rich_print("[cyan]Sending Code Analyze MCP Command:[/cyan]")
        rich_print(Syntax(json.dumps({"command": "code_analyze", "args": mcp_args}, indent=2), "json", theme="default"))

    try:
        # --- Make MCP Request ---
        response = await client.send_request("code_analyze", mcp_args)

        # --- Process MCP Response ---
        if not response:
            rich_print("[bold red]MCP request failed: No response received.[/bold red]")
            return

        # Use the consistent status check
        if response.get("status") != "success":
            error_info = response.get("error", "Unknown error")
            rich_print(f"[bold red]Code analysis command failed.[/bold red] Status: {response.get('status', 'N/A')}")
            rich_print(f"Error details: {error_info}")
            return
        
        # Check for data field
        if "data" not in response or not response["data"]:
            rich_print("[bold yellow]Analysis successful, but no data returned.[/bold yellow]")
            return
        
        # Expecting results based on CodeAnalyzerResult model
        analysis_result = response["data"] 
        security_results = analysis_result.get("security")

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

    # Remove API_KEY check as MCPClient doesn't use it
    # if API_KEY == "your-default-api-key":
    #     rich_print("[bold yellow]Warning:[/bold yellow] Using default API key. Set the TSAP_API_KEY environment variable.")

    async with MCPClient() as client:
        # Check server health before proceeding
        # Use client.info() instead of health_check()
        rich_print(f"Attempting to get server info from {client.base_url}...")
        info = await client.info()
        if info.get("status") != "success" or info.get("error") is not None:
            rich_print("[bold red]MCP server check failed. Make sure the MCP server is running.[/bold red]")
            rich_print(f"Info response: {info}")
            return
        else:
            rich_print("[green]MCP server check successful.[/green]")
            
        await run_code_audit(
            client, 
            paths=args.paths, 
            severity=args.severity,
            verbose=args.verbose
        )

if __name__ == "__main__":
    asyncio.run(main())