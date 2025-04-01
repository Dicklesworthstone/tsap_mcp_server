#!/usr/bin/env python3
"""
Advanced Patterns Demo

This script demonstrates the comprehensive features of the Patterns 
composite tool in TSAP, including pattern creation, testing, management,
and applying patterns to search through documents.
"""
import asyncio
import sys
import os
from pathlib import Path
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich.rule import Rule
from rich.box import ROUNDED

# Import the MCP client from the library
from tsap.mcp import MCPClient

# --- Path Setup --- #
# Add the project root and src directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go up one level from examples/
src_path = os.path.join(project_root, 'src')

# Add project root if not already present
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add src path if not already present
if src_path not in sys.path:
    sys.path.insert(0, src_path)  # Prepend src directory

# Add examples path itself
if script_dir not in sys.path:
    sys.path.insert(1, script_dir)
# --- End Path Setup ---


console = Console()

DEBUG = False

def debug_print(msg):
    """Print debug message if DEBUG is True."""
    if DEBUG:
        console.print(f"[dim][DEBUG] {msg}[/dim]")

async def patterns_demo():
    """Demonstrate Pattern's advanced features."""
    console.print(Panel.fit(
        "[bold blue]TSAP Patterns Advanced Features Demo[/bold blue]",
        subtitle="Creating, managing and applying search patterns"
    ))

    # Define file paths (relative to workspace root)
    base_data_dir = Path("tsap_example_data/code/")
    sample_files = [
        base_data_dir / "sample_python.py",
        base_data_dir / "sample_config.yaml",
        base_data_dir / "sample_log.txt",
    ]

    # Check existence of essential files
    missing_files = [f for f in sample_files if not f.exists()]
    if missing_files:
        console.print(f"[bold red]Error:[/bold red] Required example files not found: {', '.join(map(str, missing_files))}")
        console.print(f"Please ensure the '{base_data_dir}' directory contains the required files. Creating sample files for demo purposes...")
        
        # Create sample directory and files for demo if they don't exist
        os.makedirs(base_data_dir, exist_ok=True)
        
        # Sample Python file with patterns to detect
        sample_python_content = """#!/usr/bin/env python3
import os
import sys
import re
import time
import json
from typing import Dict, List, Any, Optional

# TODO: Implement error handling for edge cases
# FIXME: This is a temporary solution, needs proper implementation

def insecure_function(user_input):
    # This is an insecure function that directly executes user input
    result = eval(user_input)  # Security risk: direct eval of user input
    return result

def load_config(filename):
    # Loads configuration from a file
    with open(filename, 'r') as f:
        return json.load(f)

def process_data(data):
    # Process the data with a raw SQL query
    sql = f"SELECT * FROM users WHERE username = '{data['username']}'"  # SQL Injection risk
    
    # Hardcoded credentials - bad practice
    password = "supersecretpassword123"
    api_key = "AIzaSyA1X-GsHPQ-5W6C8aXg9l1C1JnJPXfLdH"
    
    return sql

class DataProcessor:
    def __init__(self):
        self.debug = True
        
    def log_error(self, message):
        print(f"ERROR: {message}")
        
    def process(self, data):
        try:
            return process_data(data)
        except Exception as e:
            self.log_error(f"Failed to process data: {e}")
            return None
"""
        
        # Sample YAML config file
        sample_config_content = """# Application Configuration

# Server settings
server:
  host: 0.0.0.0
  port: 8080
  debug: true
  
# Database connection
database:
  host: localhost
  port: 5432
  name: app_db
  user: admin
  password: secret_password  # TODO: Move to environment variable
  
# API settings
api:
  rate_limit: 100
  timeout: 30
  token: "abcdef123456789"  # FIXME: Replace with proper token generation
  
# Logging configuration
logging:
  level: INFO
  file: /var/log/app.log
  rotation: daily
  
# Security
security:
  # TODO: Implement proper key rotation
  ssl: true
  key: "/etc/ssl/private.key"
  cert: "/etc/ssl/cert.pem"
"""

        # Sample log file
        sample_log_content = """[2023-04-01 12:34:56] [INFO] Application started
[2023-04-01 12:35:02] [DEBUG] Connection established to database server
[2023-04-01 12:35:10] [ERROR] Failed to authenticate user 'john': Invalid password
[2023-04-01 12:36:23] [WARNING] High memory usage detected: 85%
[2023-04-01 12:40:15] [ERROR] Exception in thread "main" java.lang.NullPointerException at com.example.Main.process(Main.java:42)
[2023-04-01 12:45:30] [INFO] Request processed successfully in 235ms
[2023-04-01 12:46:11] [CRITICAL] Database connection lost, attempting reconnection
[2023-04-01 12:46:15] [ERROR] Unable to connect to database: Connection refused
[2023-04-01 12:50:22] [INFO] Server shutting down gracefully
"""

        # Write the sample files
        with open(sample_files[0], 'w') as f:
            f.write(sample_python_content)
        with open(sample_files[1], 'w') as f:
            f.write(sample_config_content)
        with open(sample_files[2], 'w') as f:
            f.write(sample_log_content)
        
        console.print("[green]Sample files created successfully.[/green]")

    # Create our client
    try:
        debug_print("Creating MCPClient...")
        async with MCPClient() as client:
            debug_print("Running demos...")

            # Initial server info check
            console.print(f"Attempting to get server info from {client.base_url}...")
            try:
                info = await client.info()
                if info.get("status") != "success" or info.get("error") is not None:
                    error_message = info.get('error', "Status was not success")
                    console.print(f"[bold red]Error during initial client.info() check:[/bold red] {error_message}")
                    return
                else:
                    console.print("[green]Initial client.info() check successful.[/green]")
            except Exception as e:
                console.print(f"[bold red]Error during initial client.info() check:[/bold red] {e}")
                import traceback
                console.print(traceback.format_exc())
                return

            # DEMO 1: Exploring Built-in Patterns
            console.print(Rule("[bold yellow]Demo 1: Exploring Built-in Patterns[/bold yellow]"))
            console.print("[italic]Browse through the built-in patterns in the pattern library.[/italic]")
            console.print()
            await explore_patterns_demo(client)

            # DEMO 2: Creating and Testing Custom Patterns
            console.print(Rule("[bold yellow]Demo 2: Creating and Testing Custom Patterns[/bold yellow]"))
            console.print("[italic]Create custom search patterns and test them against examples.[/italic]")
            console.print()
            await create_pattern_demo(client)

            # DEMO 3: Applying Patterns to Files
            console.print(Rule("[bold yellow]Demo 3: Applying Patterns to Files[/bold yellow]"))
            console.print("[italic]Apply patterns to search for matches in different file types.[/italic]")
            console.print()
            await apply_patterns_demo(client, sample_files)

            console.print(Rule("[bold green]Patterns Demo Complete[/bold green]"))

    except Exception as e:
        console.print(f"[bold red]Error during demo:[/bold red] {str(e)}")
        import traceback
        console.print(traceback.format_exc())

async def explore_patterns_demo(client):
    """Demo exploring built-in patterns in the pattern library."""
    console.print("Exploring built-in pattern categories and examples")
    
    try:
        # Get all available pattern categories using the new helper method
        result = await client.send_request_and_extract(
            "pattern_analyze", 
            {
                "action": "list_categories"
            }
        )
        
        if not result["success"]:
            console.print(f"[bold red]Error getting pattern categories:[/bold red] {result['error']}")
            return
            
        categories_data = result["result"] or {}
        categories = categories_data.get("categories", [])
        descriptions = categories_data.get("descriptions", {})
        
        # Display the available pattern categories
        table = Table(title="Pattern Categories", box=ROUNDED)
        table.add_column("Category", style="cyan")
        table.add_column("Description", style="green")
        
        for category in categories:
            description = descriptions.get(category, "No description available")
            table.add_row(category, description)
        
        console.print(table)
        
        # Get patterns for a specific category (security) using the new helper method
        console.print("\n[bold]Getting sample patterns from 'security' category:[/bold]")
        
        result = await client.send_request_and_extract(
            "pattern_analyze",
            {
                "action": "list_patterns",
                "category": "security"
            }
        )
        
        if not result["success"]:
            console.print(f"[bold red]Error getting security patterns:[/bold red] {result['error']}")
            return
            
        security_patterns = result["result"] or []
        if security_patterns:
            display_patterns(security_patterns, "Security Patterns")
        else:
            console.print("[yellow]No patterns found in the security category.[/yellow]")

    except Exception as e:
        console.print(f"[bold red]Error during explore_patterns_demo:[/bold red] {str(e)}")
        import traceback
        console.print(traceback.format_exc())

async def create_pattern_demo(client):
    """Demo creating and testing custom patterns."""
    console.print("Creating and testing custom search patterns")
    
    # Define a custom pattern for testing
    custom_pattern = {
        "name": "API Key Pattern",
        "pattern": r"api[_-]?key[_-]?=\s*['\"]([a-zA-Z0-9_-]{20,})['\"]",
        "description": "Detects potential API keys in code",
        "category": "security",
        "is_regex": True,
        "case_sensitive": False,
        "tags": ["security", "credentials", "api"],
        "examples": [
            "api_key='AbCdEfGhIjKlMnOpQrSt1234567890'",
            "API-KEY = \"XyZ_AbCdEfGhIjKlMnOpQrStUv_12345\""
        ],
        "negative_examples": [
            "apikey='short'",
            "not_an_api_key='AbCdEfGhIjKlMn'"
        ]
    }
    
    # Test the pattern
    console.print("\n[bold]Testing pattern before adding to library:[/bold]")
    
    response = await client.send_request(
        "pattern_analyze",
        {
            "action": "test_pattern",
            "pattern": custom_pattern["pattern"],
            "examples": custom_pattern["examples"],
            "negative_examples": custom_pattern["negative_examples"],
            "is_regex": custom_pattern["is_regex"],
            "case_sensitive": custom_pattern["case_sensitive"]
        }
    )
    
    # Print the raw response for debugging
    if DEBUG:
        console.print("[bold cyan]DEBUG: Raw test pattern response:[/bold cyan]")
        console.print(response)
    
    # Use the new helper method to extract the result
    extracted = client.extract_result(response)
    if not extracted["success"]:
        console.print(f"[bold red]Error testing pattern:[/bold red] {extracted['error']}")
        return
        
    test_results = extracted["result"] or {}
    display_pattern_test_results(test_results)
    
    # Add the pattern if tests passed
    success = test_results.get("success", False)
    
    # Display a warning if test wasn't fully successful but continue anyway
    if not success:
        console.print("[yellow]Pattern test wasn't fully successful. Proceeding anyway for demo purposes.[/yellow]")
    
    console.print("\n[bold]Adding pattern to library:[/bold]")
    
    response = await client.send_request(
        "pattern_analyze",
        {
            "action": "add_custom_pattern",
            "name": custom_pattern["name"],
            "pattern": custom_pattern["pattern"],
            "description": custom_pattern["description"],
            "category": custom_pattern["category"],
            "is_regex": custom_pattern["is_regex"],
            "tags": custom_pattern["tags"]
        }
    )
    
    # Print the raw response for debugging
    if DEBUG:
        console.print("[bold cyan]DEBUG: Raw add custom pattern response:[/bold cyan]")
        console.print(response)
    
    # Use the new helper method to extract the result
    extracted = client.extract_result(response)
    if not extracted["success"]:
        console.print(f"[bold red]Error adding pattern:[/bold red] {extracted['error']}")
        return
    
    pattern_id = extracted["result"] or ""
    console.print(f"[green]Pattern added successfully with ID: {pattern_id}[/green]")

async def apply_patterns_demo(client, file_paths=None):
    """Demo applying patterns to different file types."""
    console.print("Applying patterns to different file types")
    
    # Default file paths if not provided
    if file_paths is None:
        base_dir = "tsap_example_data"
        python_file = os.path.join(base_dir, "code/sample_code.py")
        config_file = os.path.join(base_dir, "config/sample_config.yaml")
        log_file = os.path.join(base_dir, "logs/sample_logs.log")
        file_paths = [python_file, config_file, log_file]
    else:
        # Convert Path objects to strings
        python_file = str(file_paths[0])
        config_file = str(file_paths[1])
        log_file = str(file_paths[2])

    try:
        # 1. Apply code patterns to Python file
        console.print(f"\n[bold]Applying code patterns to {os.path.basename(python_file)}:[/bold]")
        
        response = await client.send_request(
            "pattern_analyze",
            {
                "action": "apply_patterns",
                "file_path": python_file,
                "category": "code"
            }
        )
        
        # Print the raw response for debugging
        if DEBUG:
            console.print(f"[bold cyan]DEBUG: Raw apply patterns to {python_file} response:[/bold cyan]")
            console.print(response)
        
        # Use the new helper method to extract the result
        extracted = client.extract_result(response)
        if not extracted["success"]:
            console.print(f"[bold red]Error applying code patterns:[/bold red] {extracted['error']}")
            return
            
        code_pattern_results = extracted["result"] or []
        if code_pattern_results:
            display_file_matches(code_pattern_results, os.path.basename(python_file))
        else:
            console.print(f"No pattern matches found in {os.path.basename(python_file)}")
        
        # 2. Apply configuration patterns to config file
        console.print(f"\n[bold]Applying configuration patterns to {os.path.basename(config_file)}:[/bold]")
        
        response = await client.send_request(
            "pattern_analyze",
            {
                "action": "apply_patterns",
                "file_path": config_file,
                "category": "configuration"
            }
        )
        
        # Print the raw response for debugging
        if DEBUG:
            console.print(f"[bold cyan]DEBUG: Raw apply patterns to {config_file} response:[/bold cyan]")
            console.print(response)
        
        # Use the new helper method to extract the result
        extracted = client.extract_result(response)
        if not extracted["success"]:
            console.print(f"[bold red]Error applying configuration patterns:[/bold red] {extracted['error']}")
            return
            
        config_pattern_results = extracted["result"] or []
        if config_pattern_results:
            display_file_matches(config_pattern_results, os.path.basename(config_file))
        else:
            console.print(f"No pattern matches found in {os.path.basename(config_file)}")
        
        # 3. Apply documentation patterns to log file (since there's no log category)
        console.print(f"\n[bold]Applying documentation patterns to {os.path.basename(log_file)}:[/bold]")
        
        response = await client.send_request(
            "pattern_analyze",
            {
                "action": "apply_patterns",
                "file_path": log_file,
                "category": "documentation"
            }
        )
        
        # Print the raw response for debugging
        if DEBUG:
            console.print(f"[bold cyan]DEBUG: Raw apply patterns to {log_file} response:[/bold cyan]")
            console.print(response)
        
        # Use the new helper method to extract the result
        extracted = client.extract_result(response)
        if not extracted["success"]:
            console.print(f"[bold red]Error applying documentation patterns:[/bold red] {extracted['error']}")
            return
            
        doc_pattern_results = extracted["result"] or []
        if doc_pattern_results:
            display_file_matches(doc_pattern_results, os.path.basename(log_file))
        else:
            console.print(f"No pattern matches found in {os.path.basename(log_file)}")
        
    except Exception as e:
        console.print(f"[bold red]Error during apply_patterns_demo:[/bold red] {str(e)}")
        import traceback
        console.print(traceback.format_exc())

def display_patterns(patterns, title):
    """Display a list of patterns."""
    if not patterns:
        console.print("[yellow]No patterns to display.[/yellow]")
        return
    
    table = Table(title=title, box=ROUNDED)
    table.add_column("Name", style="cyan")
    table.add_column("Category", style="green")
    table.add_column("Description", style="white")
    table.add_column("Priority", style="yellow")
    table.add_column("Confidence", style="magenta")
    table.add_column("Tags", style="blue")
    
    for pattern in patterns[:5]:  # Limit to 5 patterns for the demo
        name = pattern.get("name", "Unnamed")
        category = pattern.get("category", "")
        description = pattern.get("description", "")
        priority = pattern.get("priority", "medium")
        confidence = pattern.get("confidence", 0.0)
        tags = ", ".join(pattern.get("tags", []))
        
        # Color-code priority
        priority_style = {
            "critical": "red",
            "high": "yellow",
            "medium": "green",
            "low": "blue"
        }.get(priority.lower(), "white")
        
        # Color-code confidence
        confidence_style = "red" if confidence < 0.5 else "yellow" if confidence < 0.8 else "green"
        
        table.add_row(
            name,
            category,
            description,
            f"[{priority_style}]{priority}[/{priority_style}]",
            f"[{confidence_style}]{confidence:.2f}[/{confidence_style}]",
            tags
        )
    
    console.print(table)
    
    if len(patterns) > 5:
        console.print(f"[dim]... and {len(patterns) - 5} more patterns not shown.[/dim]")

def display_pattern_test_results(results):
    """Display pattern test results."""
    if not results:
        console.print("No test results to display")
        return
        
    # Check if results has the expected structure
    if 'success' not in results:
        console.print("Unexpected test results format")
        return
        
    console.print(f"Test Results (Success: {'Yes' if results['success'] else 'No'})")
    
    # Display statistics if available
    if 'stats' in results:
        stats = results['stats']
        console.print(f"F1 Score: {stats.get('f1_score', 0.0):.2f}")
        console.print(f"Execution Time: {stats.get('execution_time', 0.0):.4f}s")
    
    # Display matches
    console.print("\nSuccessful Matches:")
    for match in results.get('matches', []):
        console.print(f"  ✓ {match}")
    
    # Display failed examples
    console.print("\nFailed Matches:")
    for failed in results.get('failed_examples', []):
        console.print(f"  ✗ {failed}")
    
    # Display failed negatives if they exist
    if results.get('failed_negatives'):
        console.print("\nFailed Negative Examples:")
        for neg in results.get('failed_negatives', []):
            console.print(f"  ✗ {neg}")

def display_file_matches(matches, filename):
    """Display pattern matches found in a file."""
    if not matches:
        console.print(f"No matches found in {filename}")
        return
    
    console.print(f"Matches in {filename}:")
    total_match_count = sum(len(pattern.get('matches', [])) for pattern in matches)
    console.print(f"Found {total_match_count} matches across {len(matches)} patterns\n")
    
    for pattern_result in matches:
        pattern_matches = pattern_result.get('matches', [])
        if not pattern_matches:
            continue
            
        console.print(f"\n[bold]Pattern:[/bold] {pattern_result.get('name', 'Unnamed pattern')}")
        console.print(f"[bold]Category:[/bold] {pattern_result.get('category', 'Unknown')}")
        console.print(f"[bold]Matches found:[/bold] {len(pattern_matches)}")
        
        for match in pattern_matches[:5]:  # Limit to 5 matches per pattern
            line = match.get('line', '?')
            match_text = match.get('match', '')
            context = match.get('context', '')
            
            # Trim context if too long
            context_display = context[:60] + "..." if len(context) > 60 else context
            
            console.print(f"  Line {line}: {match_text}")
            console.print(f"  Context: {context_display}")
        
        # Show if we have more matches that aren't displayed
        if len(pattern_matches) > 5:
            console.print(f"  ... and {len(pattern_matches) - 5} more matches not shown.")

async def main():
    """Main entry point for the patterns demo."""
    try:
        await patterns_demo()
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        import traceback
        console.print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main()) 