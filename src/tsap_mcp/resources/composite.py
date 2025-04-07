"""
Composite resources for TSAP MCP Server.

This module provides MCP resource implementations for accessing composite operations
that combine multiple core functionalities.
"""
import json
from typing import Optional
from mcp.server.fastmcp import FastMCP


def register_composite_resources(mcp: FastMCP) -> None:
    """Register all composite-related resources with the MCP server.
    
    Args:
        mcp: FastMCP server instance
    """
    
    @mcp.resource("composite://profile/{document_path:path}")
    async def get_document_profile(document_path: str) -> str:
        """Get profile information for a document.
        
        This resource provides comprehensive profile information about a document,
        including its structure, content types, and patterns.
        
        Args:
            document_path: Path to the document to profile
            
        Returns:
            Document profile as JSON string
        """
        # Get document profile from the original implementation
        from tsap.composite.document_profiler import get_document_profile as original_get_profile
        
        try:
            profile = await original_get_profile(document_path)
            
            # Format as JSON
            if isinstance(profile, dict):
                return json.dumps(profile, indent=2)
            elif hasattr(profile, "dict"):
                return json.dumps(profile.dict(), indent=2)
            else:
                return json.dumps({"error": "Profile format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to profile document: {str(e)}",
                "document_path": document_path
            }, indent=2)
    
    @mcp.resource("composite://patterns/{file_path:path}")
    async def get_patterns(file_path: str) -> str:
        """Get pattern information for a file.
        
        This resource identifies and provides information about patterns found
        in the specified file.
        
        Args:
            file_path: Path to the file to analyze for patterns
            
        Returns:
            Pattern information as JSON string
        """
        # Get patterns from the original implementation
        from tsap.composite.patterns import get_patterns as original_get_patterns
        
        try:
            patterns = await original_get_patterns(file_path)
            
            # Format as JSON
            if isinstance(patterns, dict):
                return json.dumps(patterns, indent=2)
            elif hasattr(patterns, "dict"):
                return json.dumps(patterns.dict(), indent=2)
            else:
                return json.dumps({"error": "Patterns format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to identify patterns: {str(e)}",
                "file_path": file_path
            }, indent=2)
    
    @mcp.resource("composite://context/{file_path:path}/{query}")
    async def get_context(file_path: str, query: str) -> str:
        """Get context information around a query in a file.
        
        This resource extracts relevant context around the specified query in the file.
        
        Args:
            file_path: Path to the file to extract context from
            query: Query to find context for
            
        Returns:
            Context information as JSON string
        """
        # Get context from the original implementation
        from tsap.composite.context import get_context as original_get_context
        
        try:
            context = await original_get_context(file_path, query)
            
            # Format as JSON
            if isinstance(context, dict):
                return json.dumps(context, indent=2)
            elif hasattr(context, "dict"):
                return json.dumps(context.dict(), indent=2)
            else:
                return json.dumps({"error": "Context format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to extract context: {str(e)}",
                "file_path": file_path,
                "query": query
            }, indent=2)
    
    @mcp.resource("composite://diff/{original_path:path}/{modified_path:path}")
    async def get_diff(original_path: str, modified_path: str) -> str:
        """Get diff information between two files.
        
        This resource creates a diff between two files and provides
        detailed information about the changes.
        
        Args:
            original_path: Path to the original file
            modified_path: Path to the modified file
            
        Returns:
            Diff information as JSON string
        """
        # Get diff from the original implementation
        from tsap.composite.diff_generator import get_diff as original_get_diff
        
        try:
            diff = await original_get_diff(original_path, modified_path)
            
            # Format as JSON
            if isinstance(diff, dict):
                return json.dumps(diff, indent=2)
            elif hasattr(diff, "dict"):
                return json.dumps(diff.dict(), indent=2)
            else:
                return json.dumps({"error": "Diff format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to generate diff: {str(e)}",
                "original_path": original_path,
                "modified_path": modified_path
            }, indent=2)
    
    @mcp.resource("composite://filenames/{directory_path:path}")
    async def analyze_filenames(directory_path: str) -> str:
        """Analyze filenames in a directory.
        
        This resource analyzes filenames in a directory to identify patterns,
        naming conventions, and potential improvements.
        
        Args:
            directory_path: Path to the directory with filenames to analyze
            
        Returns:
            Filename analysis as JSON string
        """
        # Get filename analysis from the original implementation
        from tsap.composite.filenames import analyze_filenames as original_analyze_filenames
        
        try:
            analysis = await original_analyze_filenames(directory_path)
            
            # Format as JSON
            if isinstance(analysis, dict):
                return json.dumps(analysis, indent=2)
            elif hasattr(analysis, "dict"):
                return json.dumps(analysis.dict(), indent=2)
            else:
                return json.dumps({"error": "Analysis format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to analyze filenames: {str(e)}",
                "directory_path": directory_path
            }, indent=2)
    
    @mcp.resource("composite://regex/generate")
    async def generate_regex(matches: str, non_matches: Optional[str] = None) -> str:
        """Generate regex pattern from examples.
        
        This resource creates a regex pattern that matches the provided examples
        and doesn't match the non-examples.
        
        Args:
            matches: Examples of strings to match (newline separated)
            non_matches: Examples of strings not to match (newline separated)
            
        Returns:
            Generated regex information as JSON string
        """
        # Get regex pattern from the original implementation
        from tsap.composite.regex_generator import generate_regex as original_generate_regex
        
        try:
            # Convert newline-separated strings to lists
            matches_list = [m.strip() for m in matches.split('\n') if m.strip()]
            non_matches_list = [m.strip() for m in non_matches.split('\n') if m.strip()] if non_matches else []
            
            regex_info = await original_generate_regex(matches_list, non_matches_list)
            
            # Format as JSON
            if isinstance(regex_info, dict):
                return json.dumps(regex_info, indent=2)
            elif hasattr(regex_info, "dict"):
                return json.dumps(regex_info.dict(), indent=2)
            else:
                return json.dumps({"error": "Regex info format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to generate regex: {str(e)}",
                "matches_count": len(matches_list) if 'matches_list' in locals() else 0,
                "non_matches_count": len(non_matches_list) if 'non_matches_list' in locals() else 0
            }, indent=2)
    
    @mcp.resource("composite://structure/{format}")
    async def get_structure_search_templates(format: str) -> str:
        """Get structure search templates for a specific format.
        
        This resource provides templates for searching within specific
        structured data formats.
        
        Args:
            format: Structure format (json, xml, code, etc.)
            
        Returns:
            Structure search templates as JSON string
        """
        # Get structure search templates from the original implementation
        from tsap.composite.structure_search import get_templates as original_get_templates
        
        try:
            templates = await original_get_templates(format)
            
            # Format as JSON
            if isinstance(templates, dict):
                return json.dumps(templates, indent=2)
            elif hasattr(templates, "dict"):
                return json.dumps(templates.dict(), indent=2)
            else:
                return json.dumps({"error": "Templates format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to retrieve templates: {str(e)}",
                "format": format
            }, indent=2) 