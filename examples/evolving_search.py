# examples/evolving_search.py
import asyncio
import json

from rich import print as rich_print
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.console import console

from api_client import TSAPClient, API_KEY # Import from our client module

# --- Configuration ---
POSITIVE_EXAMPLES = [
    "user@example.com",
    "john.doe123@company.co.uk",
    "support+alias@my-site.info",
    "test.email@domain.name",
]
NEGATIVE_EXAMPLES = [
    "not an email",
    "user@",
    "@domain.com",
    "user@domain",
    "user@domain.",
    "user @ domain.com",
]

# --- Main Evolution Function ---
async def run_pattern_evolution(client: TSAPClient):
    """Runs regex pattern evolution using the API."""
    rich_print(Panel("[bold blue]Starting Regex Pattern Evolution for Email Addresses...[/bold blue]", expand=False))

    # --- Define Request Payload ---
    # Uses the /api/evolution/pattern endpoint
    # Based on api/models/evolution.py (PatternEvolutionRequest) and api/routes/evolution.py
    request_payload = {
        "params": { # Corresponds to PatternLibraryParams (used for context, description etc.)
             "description": "Evolved email address pattern",
             "pattern_type": "regex", # Assuming this is a valid field
             "tags": ["email", "evolved", "example"]
        },
        "positive_examples": POSITIVE_EXAMPLES,
        "negative_examples": NEGATIVE_EXAMPLES,
        "initial_patterns": [ # Optional: provide starting points
            r"\b[\w._%+-]+@[\w.-]+\.\w{2,}\b",
            r".+@.+\..+"
        ],
        "config": { # Corresponds to EvolutionConfigRequest
            "population_size": 20, # Smaller for quicker example
            "generations": 10,     # Fewer generations for example
            "mutation_rate": 0.3,
            "crossover_rate": 0.7,
            "elitism": 1,
            "selection_method": "tournament",
            "tournament_size": 3, # Need to add this if using tournament
            "fitness_target": 0.98, # Stop if a good pattern is found
            "max_runtime": 120 # Max 2 minutes
        },
        "async_execution": True # Evolution can take time, run async
    }

    rich_print("[cyan]Sending Pattern Evolution Request:[/cyan]")
    rich_print(Syntax(json.dumps(request_payload, indent=2), "json", theme="default"))

    # --- Make API Call ---
    response = await client.post("/api/evolution/pattern", payload=request_payload)

    # --- Process Response (Async Job Handling) ---
    if not response or "error" in response:
        rich_print("[bold red]Pattern evolution request failed.[/bold red]", response)
        return

    if "job_id" in response:
        # --- Wait for Async Job ---
        try:
            final_result_response = await client.wait_for_job(response, poll_interval=5.0) # Poll every 5s

            # --- Display Final Result ---
            if not final_result_response or "result" not in final_result_response:
                 rich_print("[bold red]Failed to retrieve valid job result.[/bold red]", final_result_response)
                 return

            evolution_result = final_result_response["result"]
            best_pattern = evolution_result.get("pattern", "N/A")
            fitness = evolution_result.get("fitness", 0.0)
            generations = evolution_result.get("generations", "N/A")
            exec_time = evolution_result.get("execution_time", "N/A")

            summary_panel = Panel(
                f"[bold]Best Pattern:[/bold] [green]{best_pattern}[/green]\n"
                f"[bold]Fitness:[/bold] {fitness:.4f}\n"
                f"[bold]Generations:[/bold] {generations}\n"
                f"[bold]Execution Time:[/bold] {exec_time:.2f}s",
                title="Evolution Result Summary",
                border_style="green"
            )
            console.print(summary_panel)

            # Display alternative patterns if available
            alternatives = evolution_result.get("alternative_patterns", [])
            if alternatives:
                alt_table = Table(title="Alternative Patterns")
                alt_table.add_column("Pattern", style="cyan")
                alt_table.add_column("Fitness", style="yellow", justify="right")
                alt_table.add_column("Precision", style="yellow", justify="right")
                alt_table.add_column("Recall", style="yellow", justify="right")

                for alt in alternatives[:5]: # Show top 5 alternatives
                    alt_table.add_row(
                        alt.get("pattern", "N/A"),
                        f"{alt.get('fitness', 0.0):.4f}",
                        f"{alt.get('precision', 0.0):.3f}",
                        f"{alt.get('recall', 0.0):.3f}"
                    )
                console.print(alt_table)

        except Exception as e:
             rich_print(f"[bold red]Error during pattern evolution job:[/bold red] {e}")
    else:
         rich_print("[bold red]Expected an async job response but got something else.[/bold red]", response)


# --- Main Execution ---
async def main():
    """Main function to run the pattern evolution example."""
    if API_KEY == "your-default-api-key":
        rich_print("[bold yellow]Warning:[/bold yellow] Using default API key. Set the TSAP_API_KEY environment variable.")

    async with TSAPClient() as client:
        await run_pattern_evolution(client)

if __name__ == "__main__":
    asyncio.run(main())