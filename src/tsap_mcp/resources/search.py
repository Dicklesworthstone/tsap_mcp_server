"""
Search resources for TSAP MCP Server.

This module provides MCP resource implementations for accessing search
capabilities and pattern matching information.
"""
import json
from mcp.server.fastmcp import FastMCP


def register_search_resources(mcp: FastMCP) -> None:
    """Register all search-related resources with the MCP server.
    
    Args:
        mcp: FastMCP server instance
    """
    
    @mcp.resource("search://ripgrep/options")
    async def get_ripgrep_options() -> str:
        """Get information about ripgrep search options.
        
        This resource provides information about ripgrep search options,
        including their syntax, purpose, and examples.
        
        Returns:
            Ripgrep options information as JSON string
        """
        # Get ripgrep options from the original implementation
        from tsap.core.ripgrep import get_ripgrep_options as original_get_options
        
        try:
            options = await original_get_options()
            
            # Format as JSON
            if isinstance(options, dict):
                return json.dumps(options, indent=2)
            elif hasattr(options, "dict"):
                return json.dumps(options.dict(), indent=2)
            else:
                return json.dumps({"error": "Options format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to retrieve ripgrep options: {str(e)}"
            }, indent=2)
    
    @mcp.resource("search://text/patterns")
    async def get_text_search_patterns() -> str:
        """Get common text search patterns.
        
        This resource provides information about commonly used
        text search patterns with examples and explanations.
        
        Returns:
            Text search patterns as JSON string
        """
        # Get text search patterns from the original implementation
        from tsap.core.text_search import get_common_patterns as original_get_patterns
        
        try:
            patterns = await original_get_patterns()
            
            # Format as JSON
            if isinstance(patterns, dict):
                return json.dumps(patterns, indent=2)
            elif hasattr(patterns, "dict"):
                return json.dumps(patterns.dict(), indent=2)
            else:
                return json.dumps({"error": "Patterns format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to retrieve text search patterns: {str(e)}"
            }, indent=2)
    
    @mcp.resource("search://regex/examples")
    async def get_regex_examples() -> str:
        """Get examples of regex patterns for common use cases.
        
        This resource provides examples of regex patterns for
        common search scenarios, with explanations.
        
        Returns:
            Regex examples as JSON string
        """
        # Get regex examples from the original implementation
        from tsap.core.text_search import get_regex_examples as original_get_examples
        
        try:
            examples = await original_get_examples()
            
            # Format as JSON
            if isinstance(examples, dict):
                return json.dumps(examples, indent=2)
            elif hasattr(examples, "dict"):
                return json.dumps(examples.dict(), indent=2)
            else:
                return json.dumps({"error": "Examples format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to retrieve regex examples: {str(e)}"
            }, indent=2)
    
    @mcp.resource("search://semantic/models")
    async def get_semantic_models() -> str:
        """Get information about available semantic search models.
        
        This resource provides information about semantic search models,
        including their capabilities, dimensions, and specific uses.
        
        Returns:
            Semantic model information as JSON string
        """
        # Get model information from the original implementation
        from tsap.core.semantic_search import get_models as original_get_models
        
        try:
            models = await original_get_models()
            
            # Format as JSON
            if isinstance(models, dict):
                return json.dumps(models, indent=2)
            elif hasattr(models, "dict"):
                return json.dumps(models.dict(), indent=2)
            else:
                return json.dumps({"error": "Models format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to retrieve semantic models: {str(e)}"
            }, indent=2)
    
    @mcp.resource("search://history")
    async def get_search_history() -> str:
        """Get search history information.
        
        This resource provides information about recent searches,
        including queries, filters, and results summary.
        
        Returns:
            Search history as JSON string
        """
        # Get search history from the original implementation
        from tsap.core.search_history import get_history as original_get_history
        
        try:
            history = await original_get_history()
            
            # Format as JSON
            if isinstance(history, dict):
                return json.dumps(history, indent=2)
            elif hasattr(history, "dict"):
                return json.dumps(history.dict(), indent=2)
            else:
                return json.dumps({"error": "History format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to retrieve search history: {str(e)}"
            }, indent=2)
    
    @mcp.resource("search://filters/{filter_type}")
    async def get_search_filters(filter_type: str) -> str:
        """Get information about search filters.
        
        This resource provides information about available search filters
        for narrowing search results.
        
        Args:
            filter_type: Type of filter (file, content, etc.)
            
        Returns:
            Filter information as JSON string
        """
        # Get filter information from the original implementation
        from tsap.core.search_filters import get_filters as original_get_filters
        
        try:
            filters = await original_get_filters(filter_type)
            
            # Format as JSON
            if isinstance(filters, dict):
                return json.dumps(filters, indent=2)
            elif hasattr(filters, "dict"):
                return json.dumps(filters.dict(), indent=2)
            else:
                return json.dumps({"error": "Filters format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to retrieve search filters: {str(e)}",
                "filter_type": filter_type
            }, indent=2)
    
    @mcp.resource("search://file-types")
    async def get_file_types() -> str:
        """Get information about searchable file types.
        
        This resource provides information about file types that can be
        searched, including extensions, characteristics, and search tips.
        
        Returns:
            File type information as JSON string
        """
        # Get file type information from the original implementation
        from tsap.core.file_types import get_file_types as original_get_file_types
        
        try:
            file_types = await original_get_file_types()
            
            # Format as JSON
            if isinstance(file_types, dict):
                return json.dumps(file_types, indent=2)
            elif hasattr(file_types, "dict"):
                return json.dumps(file_types.dict(), indent=2)
            else:
                return json.dumps({"error": "File types format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to retrieve file types: {str(e)}"
            }, indent=2) 