"""
Adapter layer for the TSAP MCP server.

This module provides a compatibility layer between the original TSAP ToolAPI implementation
and the MCP-compliant server, allowing them to work together.
"""
import logging
import importlib
from typing import Any, Dict, List, Optional
from mcp.server.fastmcp import FastMCP, Context

logger = logging.getLogger("tsap_mcp.adapter")


def initialize_adapter(mcp: FastMCP) -> None:
    """Initialize the adapter layer for the original TSAP ToolAPI implementation.
    
    This function sets up bridges between the original ToolAPI implementation and the
    MCP-compliant server, allowing them to interoperate.
    
    Args:
        mcp: FastMCP server instance
    """
    logger.info("Initializing adapter layer for original TSAP ToolAPI implementation")
    
    # Verify that the original implementation is available
    try:
        importlib.import_module("tsap.toolapi")
    except ImportError:
        logger.warning("Original TSAP ToolAPI implementation not found, adapter disabled")
        return
    
    # Create adapter tools and resources
    _create_search_adapter(mcp)
    _create_processing_adapter(mcp)
    _create_analysis_adapter(mcp)
    _create_event_bridge(mcp)


def _create_search_adapter(mcp: FastMCP) -> None:
    """Create adapters for search functionality.
    
    Args:
        mcp: FastMCP server instance
    """
    from tsap.operations.search import search as original_search
    from tsap.models.search import SearchParams
    
    @mcp.tool(internal=True, name="adapter.search")
    async def search_adapter(
        query: str,
        path: str = ".",
        max_results: int = 10,
        include_pattern: Optional[str] = None,
        exclude_pattern: Optional[str] = None,
        ctx: Context = None,
    ) -> List[Dict[str, Any]]:
        """Adapter for the original search operation.
        
        Args:
            query: Search query
            path: Path to search in
            max_results: Maximum number of results to return
            include_pattern: Pattern to include in search
            exclude_pattern: Pattern to exclude from search
            ctx: MCP context
            
        Returns:
            Search results
        """
        if ctx:
            ctx.info(f"Performing search via adapter: {query}")
        
        # Create parameters for original implementation
        params = SearchParams(
            query=query,
            path=path,
            max_results=max_results,
            include_pattern=include_pattern,
            exclude_pattern=exclude_pattern,
        )
        
        # Call original implementation
        result = await original_search(params)
        
        # Convert result to MCP format
        if hasattr(result, "results"):
            return [
                {
                    "path": item.path if hasattr(item, "path") else None,
                    "line": item.line if hasattr(item, "line") else None,
                    "content": item.content if hasattr(item, "content") else None,
                    "metadata": item.metadata if hasattr(item, "metadata") else {},
                }
                for item in result.results
            ]
        else:
            return []


def _create_processing_adapter(mcp: FastMCP) -> None:
    """Create adapters for processing functionality.
    
    Args:
        mcp: FastMCP server instance
    """
    try:
        from tsap.operations.process import (
            process_text as original_process_text,
            extract_data as original_extract_data,
        )
        from tsap.models.process import ProcessParams, ExtractParams
        
        @mcp.tool(internal=True, name="adapter.process_text")
        async def process_text_adapter(
            text: str,
            operation: str = "clean",
            options: Optional[Dict[str, Any]] = None,
            ctx: Context = None,
        ) -> str:
            """Adapter for the original text processing operation.
            
            Args:
                text: Text to process
                operation: Processing operation to perform
                options: Additional options for the operation
                ctx: MCP context
                
            Returns:
                Processed text
            """
            if ctx:
                ctx.info(f"Processing text via adapter: {operation}")
            
            # Create parameters for original implementation
            params = ProcessParams(
                text=text,
                operation=operation,
                options=options or {},
            )
            
            # Call original implementation
            result = await original_process_text(params)
            
            # Return result
            if hasattr(result, "text"):
                return result.text
            else:
                return text
        
        @mcp.tool(internal=True, name="adapter.extract_data")
        async def extract_data_adapter(
            text: str,
            patterns: List[str],
            format: str = "json",
            ctx: Context = None,
        ) -> Dict[str, Any]:
            """Adapter for the original data extraction operation.
            
            Args:
                text: Text to extract data from
                patterns: Patterns to extract
                format: Output format
                ctx: MCP context
                
            Returns:
                Extracted data
            """
            if ctx:
                ctx.info(f"Extracting data via adapter: {len(patterns)} patterns")
            
            # Create parameters for original implementation
            params = ExtractParams(
                text=text,
                patterns=patterns,
                format=format,
            )
            
            # Call original implementation
            result = await original_extract_data(params)
            
            # Return result
            if hasattr(result, "data"):
                return result.data
            else:
                return {}
    
    except ImportError:
        logger.warning("Original processing operations not found, processing adapter disabled")


def _create_analysis_adapter(mcp: FastMCP) -> None:
    """Create adapters for analysis functionality.
    
    Args:
        mcp: FastMCP server instance
    """
    try:
        from tsap.operations.analyze import (
            analyze_code as original_analyze_code,
            analyze_text as original_analyze_text,
        )
        from tsap.models.analyze import CodeAnalysisParams, TextAnalysisParams
        
        @mcp.tool(internal=True, name="adapter.analyze_code")
        async def analyze_code_adapter(
            code: str,
            language: Optional[str] = None,
            analysis_type: str = "quality",
            ctx: Context = None,
        ) -> Dict[str, Any]:
            """Adapter for the original code analysis operation.
            
            Args:
                code: Code to analyze
                language: Programming language
                analysis_type: Type of analysis to perform
                ctx: MCP context
                
            Returns:
                Analysis results
            """
            if ctx:
                ctx.info(f"Analyzing code via adapter: {analysis_type}")
            
            # Create parameters for original implementation
            params = CodeAnalysisParams(
                code=code,
                language=language,
                analysis_type=analysis_type,
            )
            
            # Call original implementation
            result = await original_analyze_code(params)
            
            # Return result
            if hasattr(result, "analysis"):
                return result.analysis
            else:
                return {}
        
        @mcp.tool(internal=True, name="adapter.analyze_text")
        async def analyze_text_adapter(
            text: str,
            analysis_type: str = "sentiment",
            options: Optional[Dict[str, Any]] = None,
            ctx: Context = None,
        ) -> Dict[str, Any]:
            """Adapter for the original text analysis operation.
            
            Args:
                text: Text to analyze
                analysis_type: Type of analysis to perform
                options: Additional options for the analysis
                ctx: MCP context
                
            Returns:
                Analysis results
            """
            if ctx:
                ctx.info(f"Analyzing text via adapter: {analysis_type}")
            
            # Create parameters for original implementation
            params = TextAnalysisParams(
                text=text,
                analysis_type=analysis_type,
                options=options or {},
            )
            
            # Call original implementation
            result = await original_analyze_text(params)
            
            # Return result
            if hasattr(result, "analysis"):
                return result.analysis
            else:
                return {}
    
    except ImportError:
        logger.warning("Original analysis operations not found, analysis adapter disabled")


def _create_event_bridge(mcp: FastMCP) -> None:
    """Create an event bridge between the original implementation and MCP.
    
    Args:
        mcp: FastMCP server instance
    """
    try:
        from tsap.events import EventBus
        
        # Set up event listeners
        original_bus = EventBus()
        
        # Listen for events from original implementation
        @original_bus.on("operation.start")
        async def on_operation_start(data: Dict[str, Any]) -> None:
            """Handle operation start event from original implementation."""
            logger.debug(f"Original implementation operation started: {data.get('operation')}")
        
        @original_bus.on("operation.complete")
        async def on_operation_complete(data: Dict[str, Any]) -> None:
            """Handle operation complete event from original implementation."""
            logger.debug(f"Original implementation operation completed: {data.get('operation')}")
        
        @original_bus.on("error")
        async def on_error(data: Dict[str, Any]) -> None:
            """Handle error event from original implementation."""
            logger.error(f"Original implementation error: {data.get('message')}")
        
        logger.info("Event bridge established with original implementation")
    
    except ImportError:
        logger.warning("Original event bus not found, event bridge disabled")


# Adapter initialization function for entry point
def init_adapter() -> None:
    """Initialize the adapter when called as an entry point."""
    try:
        from tsap_mcp.server import mcp
        initialize_adapter(mcp)
    except ImportError:
        logger.error("Failed to import MCP server module")
    except Exception as e:
        logger.error(f"Failed to initialize adapter: {e}")


if __name__ == "__main__":
    init_adapter() 