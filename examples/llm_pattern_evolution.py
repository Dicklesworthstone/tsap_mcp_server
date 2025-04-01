#!/usr/bin/env python3
"""
LLM-based Pattern Evolution Example

This example demonstrates the use of LLM-powered regex pattern evolution
to find API error messages in logs. It shows how to use the environment
variables to enable LLM-based pattern generation instead of rule-based.
"""
import os
import asyncio
from typing import Dict, List, Any

from tsap.evolution.pattern_analyzer import analyze_pattern
from rich import print as rich_print
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

# Enable LLM-based pattern generation
os.environ["USE_LLM_PATTERN_GENERATION"] = "true"
# Use the LLM gateway on port 8013
os.environ["LLM_MCP_SERVER_URL"] = "http://localhost:8013"


async def evaluate_pattern(pattern: str, description: str, positive_examples: List[str], negative_examples: List[str]) -> Dict[str, Any]:
    """Evaluate a regex pattern against example strings.
    
    Args:
        pattern: The regex pattern to evaluate
        description: Description of what the pattern is trying to match
        positive_examples: List of strings that should match the pattern
        negative_examples: List of strings that should not match the pattern
        
    Returns:
        Analysis results including stats and generated variants
    """
    reference_set = {
        "positive": positive_examples,
        "negative": negative_examples
    }
    
    result = await analyze_pattern(
        pattern=pattern,
        description=description,
        is_regex=True,
        case_sensitive=False,
        paths=[],  # No file paths, using reference_set instead
        reference_set=reference_set,
        generate_variants=True,
        num_variants=5,
    )
    
    return result


async def main():
    """Run example pattern evolution."""
    rich_print(Panel("[bold blue]TSAP LLM Pattern Evolution Example[/bold blue]", expand=False))
    
    # Log lines for testing
    positive_examples = [
        "2024-07-30 10:02:05 ERROR [APIService] Failed processing request: Invalid token",
        "2024-07-30 10:04:10 ERROR [APIService] Internal server error in /api/v1/users: Database connection timeout",
        "2024-07-30 10:07:50 ERROR [APIService] Invalid input parameters for request ID 83749: Missing required field",
    ]
    
    negative_examples = [
        "2024-07-30 10:00:01 INFO [AuthService] User 'admin' logged in",
        "2024-07-30 10:01:15 WARN [APIService] Request rate limit approaching for client 12345",
        "2024-07-30 10:02:30 INFO [DatabaseService] Connected to database",
        "2024-07-30 10:03:45 DEBUG [APIService] Request parameters: {'id': 123, 'action': 'get'}",
        "2024-07-30 10:05:20 INFO [CacheService] Cache invalidated for user 456",
        "2024-07-30 10:06:35 ERROR [AuthService] Authentication failed for user 'test': Invalid credentials",
    ]
    
    # Initial pattern to evolve
    initial_pattern = r"ERROR.*\[APIService\].*"
    
    rich_print("\n[bold]Original Pattern:[/bold] ", initial_pattern)
    rich_print("[bold]Description:[/bold] Evolve pattern to find APIService errors in logs")
    
    # Run the pattern analysis
    result = await evaluate_pattern(
        pattern=initial_pattern,
        description="Evolve pattern to find APIService errors in logs",
        positive_examples=positive_examples,
        negative_examples=negative_examples,
    )
    
    # Display results in a table
    table = Table(title="Pattern Evaluation Results")
    table.add_column("Pattern", style="cyan")
    table.add_column("Precision", style="magenta")
    table.add_column("Recall", style="green")
    table.add_column("F1 Score", style="yellow")
    table.add_column("Source", style="blue")
    
    # Add original pattern
    table.add_row(
        result["original_pattern"],
        f"{result['stats']['precision']:.2f}",
        f"{result['stats']['recall']:.2f}",
        f"{result['stats']['f1_score']:.2f}",
        "Original"
    )
    
    # Add variants
    for variant in result["variants"]:
        # Extract source (LLM or Rule-based) from description
        if "(LLM)" in variant["description"]:
            source = "LLM"
        else:
            source = "Rule-based"
            
        table.add_row(
            variant["pattern"],
            f"{variant['stats']['precision']:.2f}",
            f"{variant['stats']['recall']:.2f}",
            f"{variant['stats']['f1_score']:.2f}",
            source
        )
    
    rich_print(table)
    
    # Show the best pattern
    rich_print("\n[bold green]Best Pattern:[/bold green] ", result["best_pattern"])
    
    # Show some details about the execution
    rich_print(f"\n[dim]Total execution time: {result['execution_time']:.2f} seconds[/dim]")
    rich_print(f"[dim]Variants generated: {len(result['variants'])}[/dim]")
    
    # Try a more challenging pattern
    rich_print("\n\n[bold]Testing a more challenging pattern...[/bold]")
    
    complex_pattern = r"ERROR.*error.*timeout"
    
    rich_print("\n[bold]Original Pattern:[/bold] ", complex_pattern)
    rich_print("[bold]Description:[/bold] Find timeout error messages regardless of service")
    
    # Run the pattern analysis
    complex_result = await evaluate_pattern(
        pattern=complex_pattern,
        description="Find timeout error messages regardless of service",
        positive_examples=positive_examples,
        negative_examples=negative_examples,
    )
    
    # Display results in a table
    complex_table = Table(title="Complex Pattern Evaluation Results")
    complex_table.add_column("Pattern", style="cyan")
    complex_table.add_column("Precision", style="magenta")
    complex_table.add_column("Recall", style="green")
    complex_table.add_column("F1 Score", style="yellow")
    complex_table.add_column("Source", style="blue")
    
    # Add original pattern
    complex_table.add_row(
        complex_result["original_pattern"],
        f"{complex_result['stats']['precision']:.2f}",
        f"{complex_result['stats']['recall']:.2f}",
        f"{complex_result['stats']['f1_score']:.2f}",
        "Original"
    )
    
    # Add variants
    for variant in complex_result["variants"]:
        # Extract source (LLM or Rule-based) from description
        if "(LLM)" in variant["description"]:
            source = "LLM"
        else:
            source = "Rule-based"
            
        complex_table.add_row(
            variant["pattern"],
            f"{variant['stats']['precision']:.2f}",
            f"{variant['stats']['recall']:.2f}",
            f"{variant['stats']['f1_score']:.2f}",
            source
        )
    
    rich_print(complex_table)
    
    # Show the best pattern
    rich_print("\n[bold green]Best Pattern:[/bold green] ", complex_result["best_pattern"])


if __name__ == "__main__":
    asyncio.run(main()) 