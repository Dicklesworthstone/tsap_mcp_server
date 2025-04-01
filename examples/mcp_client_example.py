#!/usr/bin/env python3
"""
MCP Client Example

This example demonstrates how to interact with the TSAP MCP Server 
using direct MCP protocol requests.
"""
import asyncio
import json
import uuid
from typing import Dict, Any, Optional, List, Union

from rich import print as rich_print
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax

from tsap.utils.errors import TSAPError
from tsap.mcp import MCPClient, DEFAULT_SERVER_URL

# Use the server URL from the library
SERVER_URL = DEFAULT_SERVER_URL

# Example function that extends functionality of the base MCPClient
async def ripgrep_search_example(
    client: MCPClient,
    pattern: str,
    paths: List[str],
    case_sensitive: bool = False,
    file_patterns: Optional[List[str]] = None,
    context_lines: int = 0,
    **kwargs
) -> Dict[str, Any]:
    """Perform a ripgrep search with enhanced parameters.
    
    Args:
        client: The MCPClient instance to use
        pattern: Search pattern
        paths: Paths to search
        case_sensitive: Whether to use case-sensitive matching
        file_patterns: Optional list of file patterns to search
        context_lines: Number of context lines to include
        **kwargs: Additional parameters to pass to ripgrep
            
    Returns:
        Ripgrep search results
    """
    # Basic search parameters
    args = {
        "pattern": pattern,
        "paths": paths,
        "case_sensitive": case_sensitive,
    }
    
    # Add context lines if provided
    if "before_context" in kwargs and "after_context" in kwargs:
        args["before_context"] = kwargs.pop("before_context")
        args["after_context"] = kwargs.pop("after_context")
    elif context_lines > 0:
        args["context_lines"] = context_lines
    
    # Add file patterns if provided
    if file_patterns:
        args["file_patterns"] = file_patterns
    
    # Handle exclude patterns as negative file patterns
    if "exclude_patterns" in kwargs:
        exclude_patterns = kwargs.pop("exclude_patterns")
        if not args.get("file_patterns"):
            args["file_patterns"] = []
        elif isinstance(args["file_patterns"], str):
            args["file_patterns"] = [args["file_patterns"]]
            
        # Add negative patterns for exclusion
        for pattern in exclude_patterns:
            args["file_patterns"].append(f"!{pattern}")
    
    # Add any remaining keyword arguments
    for key, value in kwargs.items():
        args[key] = value
        
    # Send the request
    return await client.send_request("ripgrep_search", args)

async def main():
    """Run example MCP client requests."""
    rich_print(Panel("[bold blue]TSAP MCP Client Example[/bold blue]", expand=False))
    
    async with MCPClient() as client:
        # Test command for debugging
        rich_print("\n[bold]Running Test Command...[/bold]")
        test_response = await client.send_request("test", {"param": "value"})
        
        if "data" in test_response:
            rich_print(Syntax(json.dumps(test_response["data"], indent=2), "json"))
        else:
            rich_print("[bold red]Test command failed[/bold red]")
            rich_print(test_response)
        
        # Get server info
        rich_print("\n[bold]Getting Server Info...[/bold]")
        info_response = await client.info()
        
        if "data" in info_response:
            rich_print(Syntax(json.dumps(info_response["data"], indent=2), "json"))
        else:
            rich_print("[bold red]Failed to get server info[/bold red]")
            rich_print(info_response)
        
        # Perform a basic ripgrep search
        rich_print("\n[bold]Performing Basic Ripgrep Search...[/bold]")
        search_response = await ripgrep_search_example(
            client,
            pattern="TSAP",
            paths=["src/"],
            file_patterns=["*.py"],
            context_lines=1
        )
        
        # Perform an advanced ripgrep search with additional parameters
        rich_print("\n[bold]Performing Advanced Ripgrep Search...[/bold]")
        advanced_search = await ripgrep_search_example(  # noqa: F841
            client,
            pattern="function",
            paths=["src/"],
            regex=True,
            whole_word=True,
            max_count=10,
            exclude_patterns=["*__pycache__*"],
            follow_symlinks=True
        )
        
        # Print the response regardless of contents
        rich_print(Syntax(json.dumps(search_response, indent=2, default=str), "json"))
        
        # Check for success and data
        if "data" in search_response and search_response["data"] is not None:
            # Check for matches
            if "matches" in search_response["data"]:
                matches = search_response["data"]["matches"]
                match_count = len(matches)
                
                rich_print(f"[green]Found {match_count} matches[/green]")
                
                # Display matches in a table
                table = Table(title=f"Ripgrep Search Results ({match_count} matches)")
                table.add_column("File", style="cyan")
                table.add_column("Line", style="yellow")
                table.add_column("Text", style="white")
                
                # Add the first 10 matches to the table
                for match in matches[:10]:
                    file_path = match.get("path", "")
                    line_num = str(match.get("line_number", ""))
                    line_text = match.get("line_text", "").strip()
                    table.add_row(file_path, line_num, line_text)
                    
                rich_print(table)
                
                if match_count > 10:
                    rich_print(f"[dim](Showing 10 of {match_count} matches)[/dim]")
            else:
                rich_print("[yellow]Ripgrep search successful but no matches field in response[/yellow]")
        else:
            rich_print("[bold red]Ripgrep search failed or returned no matches[/bold red]")


if __name__ == "__main__":
    asyncio.run(main()) 