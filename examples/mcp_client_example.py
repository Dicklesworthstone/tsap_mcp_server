#!/usr/bin/env python3
"""
MCP Client Example

This example demonstrates how to interact with the TSAP MCP Server 
using direct MCP protocol requests.
"""
import asyncio
import json
import httpx
import uuid
from typing import Dict, Any, Optional, List

from rich import print as rich_print
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax

# Base MCP server URL - default is the local server
SERVER_URL = "http://localhost:8021"

class MCPClient:
    """Simple client for interacting with the TSAP MCP Server."""

    def __init__(self, base_url: str = SERVER_URL):
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
        }
        self._client = httpx.AsyncClient(
            base_url=self.base_url, 
            headers=self.headers, 
            timeout=60.0
        )

    async def close(self):
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def send_request(self, command: str, args: Dict[str, Any], mode: Optional[str] = None) -> Dict[str, Any]:
        """Send an MCP request to the server.
        
        Args:
            command: The MCP command name
            args: Command arguments
            mode: Optional performance mode (fast, standard, deep)
            
        Returns:
            MCP response as a dictionary
        """
        # Create an MCP request payload
        request = {
            "request_id": str(uuid.uuid4()),
            "command": command,
            "args": args,
        }
        
        if mode:
            request["mode"] = mode
            
        try:
            # Commented out verbose request printing
            # rich_print(f"Sending to: {self.base_url}/mcp/")
            # rich_print(f"Request: {json.dumps(request, indent=2)}")
            response = await self._client.post("/mcp/", json=request)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            rich_print(f"[bold red]HTTP Error:[/bold red] {e.response.status_code} - {e.response.text}")
            return {"error": {"code": f"HTTP_{e.response.status_code}", "message": e.response.text}}
        except httpx.ConnectError as e:
            rich_print(f"[bold red]Connection Error:[/bold red] {e}")
            rich_print(f"[yellow]Try running: curl {self.base_url}/health[/yellow]")
            return {"error": {"code": "CONNECTION_ERROR", "message": str(e)}}
        except Exception as e:
            rich_print(f"[bold red]Unexpected Error During Request:[/bold red] {e}")
            rich_print(f"[yellow]Exception type: {type(e).__name__}[/yellow]")
            import traceback
            rich_print("[yellow]Traceback:[/yellow]")
            rich_print(traceback.format_exc())
            return {"error": {"code": "CLIENT_ERROR", "message": str(e)}}

    async def info(self) -> Dict[str, Any]:
        """Get server information."""
        return await self.send_request("info", {})
        
    async def ripgrep_search(
        self,
        pattern: str,
        paths: List[str],
        case_sensitive: bool = False,
        file_patterns: Optional[List[str]] = None,
        context_lines: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        """Perform a ripgrep search with enhanced parameters.
        
        Args:
            pattern: Search pattern
            paths: Paths to search
            case_sensitive: Whether to use case-sensitive matching
            file_patterns: Optional list of file patterns to search
            context_lines: Number of context lines to include
            **kwargs: Additional parameters to pass to ripgrep
                - before_context: Number of lines before match to show
                - after_context: Number of lines after match to show
                - regex: Whether to use regex matching (bool)
                - whole_word: Whether to match whole words only (bool)
                - invert_match: Whether to invert the match (bool)
                - max_count: Maximum matches per file
                - max_depth: Maximum directory depth to search
                - max_total_matches: Maximum total matches to return
                - binary: Whether to search binary files (bool)
                - hidden: Whether to search hidden files (bool)
                - no_ignore: Whether to ignore .gitignore rules (bool)
                - follow_symlinks: Whether to follow symlinks (bool)
                - encoding: File encoding to use
                - timeout: Timeout in seconds for the search
            
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
        return await self.send_request("ripgrep_search", args)

    async def semantic_search(
        self,
        texts: List[str],
        query: str,
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 10,
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform a semantic search.

        Args:
            texts: List of texts to index and search.
            query: The search query.
            ids: Optional list of unique IDs for each text. If None, generated.
            metadata: Optional list of metadata dictionaries corresponding to texts.
            top_k: Number of top results to return.
            mode: Optional performance mode.

        Returns:
            Semantic search results.
        """
        args = {
            "texts": texts,
            "query": query,
            "top_k": top_k,
        }
        if ids:
            args["ids"] = ids
        if metadata:
            args["metadata"] = metadata
            
        # Send the request using the specified mode
        return await self.send_request("semantic_search", args, mode=mode)


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
        search_response = await client.ripgrep_search(
            pattern="TSAP",
            paths=["src/"],
            file_patterns=["*.py"],
            context_lines=1
        )
        
        # Perform an advanced ripgrep search with additional parameters
        rich_print("\n[bold]Performing Advanced Ripgrep Search...[/bold]")
        advanced_search = await client.ripgrep_search(  # noqa: F841
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