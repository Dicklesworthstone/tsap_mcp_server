# examples/code_security_audit.py
import asyncio
import os
import json
from typing import List

from rich import print as rich_print
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.console import console

from api_client import TSAPClient, API_KEY # Import from our client module

# --- Configuration ---
CODE_DIR = "tsap_example_data/code"  # Directory containing code to audit

# --- Main Audit Function ---
async def run_code_audit(client: TSAPClient, paths: List[str], severity: str = "medium"):
    """Runs a security audit on the specified code paths."""
    rich_print(Panel(f"[bold blue]Starting Security Audit on:[/bold blue] {', '.join(paths)}", expand=False))
    rich_print(f"Minimum severity level: [bold]{severity.upper()}[/bold]")

    # --- Define Request Payload ---
    # We use the analysis/code endpoint with the 'security' analysis type
    # Based on api/models/analysis.py and api/routes/analysis.py
    request_payload = {
        "params": {
            "file_paths": paths,
            "analysis_types": ["security"], # Focus on security analysis
            "language": None, # Auto-detect language
            # 'repository_path': CODE_DIR # Alternative if it's a repo
        },
        # Optional API-specific parameters from CodeAnalysisRequest
        "include_file_content": False, # Don't need full file content for this example
        "highlight_matches": True, # Ask server to highlight (though we don't display it here)
        "performance_mode": "standard", # Optional override
        "async_execution": False # Run synchronously for simplicity in this example
    }

    rich_print("[cyan]Sending Code Analysis Request:[/cyan]")
    rich_print(Syntax(json.dumps(request_payload, indent=2), "json", theme="default"))

    # --- Make API Call ---
    response = await client.post("/api/analysis/code", payload=request_payload)

    # --- Process Response ---
    if not response or "error" in response:
        rich_print("[bold red]Code analysis request failed.[/bold red]", response)
        return

    if "result" not in response:
         rich_print("[bold red]Unexpected response format:[/bold red]", response)
         return

    analysis_result = response["result"]
    security_results = analysis_result.get("security")

    if not security_results:
        rich_print("[bold green]No security issues found (or analysis type not run).[/bold green]")
        return

    # Filter findings based on severity threshold
    findings = []
    severity_map = {"low": 1, "medium": 2, "high": 3, "critical": 4}
    min_severity_level = severity_map.get(severity.lower(), 2)

    for category, category_issues in security_results.get("issues_by_category", {}).items():
        for issue in category_issues:
             # Add category and severity from the pattern definition if available
             # (Assuming CodeAnalyzerResult structure from analysis/code.py includes this detail)
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
    findings.sort(key=lambda x: (severity_map.get(x["severity"].lower(), 0), x["file"], x["line"]), reverse=True)

    # --- Display Results ---
    num_findings = len(findings)
    summary_color = "green" if num_findings == 0 else "yellow" if any(f['severity'] == 'MEDIUM' for f in findings) else "red"
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

# --- Main Execution ---
async def main():
    """Main function to run the security audit example."""
    if not os.path.exists(CODE_DIR):
        rich_print(f"[bold red]Error:[/bold red] Code directory '{CODE_DIR}' not found. Please create it and add sample files.")
        return

    if API_KEY == "your-default-api-key":
        rich_print("[bold yellow]Warning:[/bold yellow] Using default API key. Set the TSAP_API_KEY environment variable.")

    async with TSAPClient() as client:
        await run_code_audit(client, paths=[CODE_DIR], severity="medium")

if __name__ == "__main__":
    asyncio.run(main())