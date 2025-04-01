"""
MCP Client - Base Implementation

This module provides the base client class for interacting with TSAP's MCP protocol.
"""
import uuid
import httpx
from typing import Dict, Any, Optional, List, Union

# Default MCP server URL
DEFAULT_SERVER_URL = "http://localhost:8021"

class MCPClient:
    """Client for interacting with the TSAP MCP Server.
    
    This class provides methods for sending commands to an MCP server
    and processing responses, with convenient utilities for handling the
    nested response structure.
    """

    def __init__(self, base_url: str = DEFAULT_SERVER_URL):
        """Initialize the MCP client.
        
        Args:
            base_url: Base URL of the MCP server, defaults to localhost:8021
        """
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
            response = await self._client.post("/mcp/", json=request)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {
                "status": "error",
                "error": {
                    "code": f"HTTP_{e.response.status_code}",
                    "message": e.response.text
                }
            }
        except httpx.ConnectError as e:
            return {
                "status": "error",
                "error": {
                    "code": "CONNECTION_ERROR",
                    "message": f"Failed to connect to MCP server at {self.base_url}: {str(e)}"
                }
            }
        except Exception as e:
            import traceback
            return {
                "status": "error",
                "error": {
                    "code": "CLIENT_ERROR",
                    "message": str(e),
                    "traceback": traceback.format_exc()
                }
            }

    def extract_result(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Safely extract result data from a nested MCP response.
        
        Handles the common nested data structure in MCP responses:
        {
            "status": "success",
            "data": {
                "status": "success",
                "result": {...actual data...}
            }
        }
        
        Args:
            response: The MCP response dictionary
            
        Returns:
            A dictionary containing:
            {
                "success": True/False,
                "result": The extracted result data or None,
                "error": Error message if any error occurred, or None
            }
        """
        result = {
            "success": False,
            "result": None,
            "error": None
        }
        
        # Check the top-level response status
        if response.get("status") != "success":
            result["error"] = response.get("error", "Unknown error in response")
            return result
            
        # Check the data section exists
        data = response.get("data", {})
        if not data:
            result["error"] = "No data section in response"
            return result
            
        # Check the data status
        if data.get("status") != "success":
            result["error"] = data.get("error", "Unknown error in data section")
            return result
            
        # Extract the result
        result["success"] = True
        result["result"] = data.get("result")
        return result

    async def send_request_and_extract(self, command: str, args: Dict[str, Any], mode: Optional[str] = None) -> Dict[str, Any]:
        """Send an MCP request and extract the result data in one step.
        
        This is a convenience method that combines send_request and extract_result.
        
        Args:
            command: The MCP command name
            args: Command arguments
            mode: Optional performance mode (fast, standard, deep)
            
        Returns:
            A dictionary containing:
            {
                "success": True/False,
                "result": The extracted result data or None,
                "error": Error message if any error occurred, or None
            }
        """
        response = await self.send_request(command, args, mode)
        return self.extract_result(response)

    async def info(self) -> Dict[str, Any]:
        """Get server information."""
        return await self.send_request("info", {})

    # Tool-specific utility methods used by example scripts
    
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
        texts: Optional[List[str]] = None,
        query: str = "",
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 10,
        mode: Optional[str] = None,
        files: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Perform a semantic search.

        Args:
            texts: List of texts to index and search. Takes precedence over files.
            query: The search query.
            ids: Optional list of unique IDs for each text. If None, generated.
            metadata: Optional list of metadata dictionaries corresponding to texts.
            top_k: Number of top results to return.
            mode: Optional performance mode.
            files: Optional list of files to search (backward compatibility).
            **kwargs: Additional parameters for backward compatibility.

        Returns:
            Semantic search results.
        """
        # If texts is None but files is provided, we'll pass files directly to the server
        # This is for backward compatibility with the old API
        if texts is None and files is not None:
            args = {
                "files": files,
                "query": query,
                **kwargs  # Include any extra parameters
            }
            return await self.send_request("semantic_search", args, mode=mode)
        
        # Otherwise use the new API
        args = {
            "texts": texts,
            "query": query,
            "top_k": top_k,
        }
        if ids:
            args["ids"] = ids
        if metadata:
            args["metadata"] = metadata
            
        # Include any additional kwargs
        for key, value in kwargs.items():
            args[key] = value
            
        # Send the request using the specified mode
        return await self.send_request("semantic_search", args, mode=mode)

    async def awk_process(
        self,
        script: str,
        input_text: Optional[str] = None,
        input_files: Optional[List[str]] = None,
        field_separator: Optional[str] = None,
        output_field_separator: Optional[str] = None,
        variables: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Process text using AWK.

        Args:
            script: The AWK script to execute.
            input_text: Optional string input to process via stdin.
            input_files: Optional list of file paths to process.
            field_separator: Optional input field separator (-F).
            output_field_separator: Optional output field separator (OFS).
            variables: Optional dictionary of variables to pass (-v).
            **kwargs: Additional parameters (currently unused but for future).

        Returns:
            AWK processing results.
        """
        args = {
            "script": script,
        }
        if input_text is not None:
            args["input_text"] = input_text
        if input_files is not None:
            args["input_files"] = input_files
        if field_separator is not None:
            args["field_separator"] = field_separator
        if output_field_separator is not None:
            args["output_field_separator"] = output_field_separator
        if variables is not None:
            args["variables"] = variables

        # Add any remaining keyword arguments (if needed in future)
        for key, value in kwargs.items():
            args[key] = value

        # Send the request
        return await self.send_request("awk_process", args)

    async def html_process(
        self,
        html: Optional[str] = None,
        url: Optional[str] = None,
        file_path: Optional[str] = None,
        selector: Optional[str] = None,
        xpath: Optional[str] = None,
        extract_tables: bool = False,
        extract_links: bool = False,
        extract_text: bool = False,
        extract_metadata: bool = False,
        mode: Optional[str] = None,
        **kwargs # Catch any other potential future args
    ) -> Dict[str, Any]:
        """Process HTML content.

        Args:
            html: Optional HTML content string.
            url: Optional URL to fetch HTML from.
            file_path: Optional file path to read HTML from.
            selector: Optional CSS selector to extract elements.
            xpath: Optional XPath expression (basic support).
            extract_tables: Whether to extract tables.
            extract_links: Whether to extract links.
            extract_text: Whether to extract clean text.
            extract_metadata: Whether to extract metadata.
            mode: Optional performance mode.
            **kwargs: Any additional arguments for future compatibility.

        Returns:
            HTML processing results.
        """
        args = {
            "html": html,
            "url": url,
            "file_path": file_path,
            "selector": selector,
            "xpath": xpath,
            "extract_tables": extract_tables,
            "extract_links": extract_links,
            "extract_text": extract_text,
            "extract_metadata": extract_metadata,
            **kwargs # Include any extra args passed
        }
        # Remove keys with None values as the server might expect them to be absent
        args = {k: v for k, v in args.items() if v is not None and v is not False}

        return await self.send_request("html_process", args, mode=mode)

    async def jq_process(
        self,
        query: str,
        input_json: Optional[str] = None,
        input_files: Optional[List[str]] = None,
        raw_output: bool = False,
        compact_output: bool = False,
        monochrome_output: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Query JSON data using JQ.

        Args:
            query: The JQ query string.
            input_json: Optional JSON string input to process via stdin.
            input_files: Optional list of file paths containing JSON data.
            raw_output: If True, output raw strings (-r).
            compact_output: If True, output compact JSON (-c).
            monochrome_output: If True, disable colorized output (-M).
            **kwargs: Additional parameters (currently unused).

        Returns:
            JQ processing results.
        """
        args = {
            "query": query,
        }
        if input_json is not None:
            args["input_json"] = input_json
        if input_files is not None:
            args["input_files"] = input_files
        if raw_output:
            args["raw_output"] = raw_output
        if compact_output:
            args["compact_output"] = compact_output
        if monochrome_output:
            args["monochrome_output"] = monochrome_output

        # Add any remaining keyword arguments (if needed in future)
        for key, value in kwargs.items():
            args[key] = value

        # Send the request
        # The handler expects the command name defined in MCPCommandType
        return await self.send_request("jq_query", args)

    async def pdf_extract(
        self,
        pdf_path: Optional[str] = None,
        file_path: Optional[str] = None,
        pages: Optional[Union[List[int], str]] = None,
        extract_text: bool = True,
        extract_tables: bool = False,
        extract_images: bool = False,
        extract_metadata: bool = False,
        password: Optional[str] = None,
        **kwargs # Allow for future expansion
    ) -> Dict[str, Any]:
        """Extract data from a PDF file using the PdfExtractor tool.

        Args:
            pdf_path: Path to the PDF file to process (will be used if file_path is None).
            file_path: Path to the PDF file to process (alternative to pdf_path).
            pages: Optional specification of pages to extract (e.g., "1-5", [1, 3, 5], "all"). Defaults to all if not specified for text/images/tables.
            extract_text: Whether to extract text content.
            extract_tables: Whether to attempt table extraction.
            extract_images: Whether to extract information about images.
            extract_metadata: Whether to extract PDF metadata.
            password: Optional password for encrypted PDFs.
            **kwargs: Additional parameters (currently unused).

        Returns:
            PDF extraction results.
        """
        # For backward compatibility, we support both pdf_path and file_path
        # with pdf_path taking precedence
        actual_path = pdf_path if pdf_path is not None else file_path
        
        args = {
            "pdf_path": actual_path,  # Server expects pdf_path, not file_path
            "extract_text": extract_text,
            "extract_tables": extract_tables,
            "extract_images": extract_images,
            "extract_metadata": extract_metadata,
        }
        
        # Only include optional args if they are provided (not None/False)
        if pages is not None:
            args["pages"] = pages
        if password is not None:
            args["password"] = password

        # Add any unexpected kwargs (for forward compatibility)
        args.update(kwargs)

        # Send the request
        return await self.send_request("pdf_extract", args)

    async def process_table(self, **kwargs) -> Dict[str, Any]:
        """Process tabular data using the TableProcessor tool.

        Args:
            **kwargs: Parameters for the table processor, matching TableProcessParams.
                Common examples include:
                - input_path (str): Path to the input file.
                - input_data (List[Dict]): Direct data input.
                - input_format (str): e.g., 'csv', 'tsv', 'json'.
                - analyze (bool): Whether to perform analysis.
                - transform (Dict): Transformation parameters (filter_expr, columns, etc.).
                - output_format (str): e.g., 'csv', 'json'.
                - output_path (str): Path for the output file.
                - max_rows_return (int): Max rows to include in the response data.
                - (and other specific format/reading options)

        Returns:
            Table processing results.
        """
        # The kwargs directly match the expected 'args' for the server command
        args = kwargs
        # Use the correct command name expected by the server handler
        return await self.send_request("table_process", args)

    # Also include aliases to maintain backward compatibility with code that may have used old method names
    async def table_process(self, **kwargs) -> Dict[str, Any]:
        """Alias for process_table for backward compatibility."""
        return await self.process_table(**kwargs)

    async def sqlite_query(
        self,
        query: str,
        database: str,
        parameters: Optional[List[Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute a SQLite query via the MCP server.
        
        Args:
            query: SQL query to execute
            database: Path to the SQLite database file
            parameters: Optional list of parameters for the query
            **kwargs: Additional parameters:
                - timeout: Timeout in seconds
                - result_mode: Result mode (objects, tuples, raw)
                
        Returns:
            Dictionary containing query results or error information
        """
        args = {
            "query": query,
            "database": database  # Using database parameter name to match server expectation
        }
        
        if parameters is not None:
            args["parameters"] = parameters
            
        # Add all valid keyword arguments
        for key, value in kwargs.items():
            args[key] = value
            
        return await self.send_request("sqlite_query", args)
    
    async def structure_search(
        self,
        search_term: str,
        file_paths: List[str],
        element_type: Optional[str] = None,
        parent_elements: Optional[List[str]] = None,
        action: str = "search",
        **kwargs
    ) -> Dict[str, Any]:
        """Perform a structure-aware search that understands document structure.
        
        Args:
            search_term: The text pattern to search for
            file_paths: List of file paths to search within
            element_type: Optional structural element type to restrict search to 
                         (e.g., "function_def", "class_def", "heading")
            parent_elements: Optional list of parent element types to further refine search context
            action: The action to perform (default: "search")
            **kwargs: Additional parameters for future expansion
            
        Returns:
            Dictionary containing search results or error information
        """
        args = {
            "action": action,
            "search_term": search_term,
            "file_paths": file_paths
        }
        
        if element_type is not None:
            args["element_type"] = element_type
            
        if parent_elements is not None:
            args["parent_elements"] = parent_elements
            
        # Add any additional keyword arguments
        args.update(kwargs)
        
        return await self.send_request("structure_search", args) 