"""
Search utilities for TSAP MCP Server.

This module provides utilities for working with search results,
pattern matching, and query formatting.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import re


def format_search_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format search results for MCP compatibility.
    
    Args:
        results: Raw search results
        
    Returns:
        Formatted search results
    """
    formatted_results = []
    
    for result in results:
        # Create a new result entry with standardized keys
        formatted_result = {
            "file": result.get("file", result.get("path", "")),
            "line": result.get("line", result.get("line_number", 0)),
            "text": result.get("text", result.get("content", "")),
        }
        
        # Add optional fields if present
        if "column" in result:
            formatted_result["column"] = result["column"]
        if "match" in result:
            formatted_result["match"] = result["match"]
        if "context" in result:
            formatted_result["context"] = result["context"]
        if "score" in result:
            formatted_result["score"] = result["score"]
        
        formatted_results.append(formatted_result)
    
    return formatted_results


def sanitize_regex_pattern(pattern: str) -> str:
    """Sanitize a regex pattern to ensure it's valid.
    
    Args:
        pattern: Regex pattern to sanitize
        
    Returns:
        Sanitized regex pattern
    """
    # Check if the pattern is valid
    try:
        re.compile(pattern)
        return pattern
    except re.error:
        # Basic sanitization: escape special characters
        return re.escape(pattern)


def build_ripgrep_command(
    pattern: str,
    paths: List[str],
    case_sensitive: bool = False,
    file_pattern: Optional[str] = None,
    exclude_pattern: Optional[str] = None,
    context_lines: Optional[int] = None,
) -> str:
    """Build a ripgrep command for given search parameters.
    
    Args:
        pattern: Pattern to search for
        paths: Paths to search in
        case_sensitive: Whether the search is case-sensitive
        file_pattern: Pattern to filter files to search
        exclude_pattern: Pattern to exclude files from search
        context_lines: Number of context lines to include
        
    Returns:
        Ripgrep command string
    """
    # Start with base command
    cmd = ["rg"]
    
    # Add options
    if not case_sensitive:
        cmd.append("-i")
    
    if file_pattern:
        cmd.append(f"-g '{file_pattern}'")
    
    if exclude_pattern:
        cmd.append(f"-g '!{exclude_pattern}'")
    
    if context_lines is not None:
        cmd.append(f"-C{context_lines}")
    
    # Add pattern with proper quoting
    cmd.append(f"'{pattern}'")
    
    # Add paths
    cmd.extend(paths)
    
    # Join into string
    return " ".join(cmd)


def normalize_search_query(query: str) -> str:
    """Normalize a search query for consistent handling.
    
    Args:
        query: Raw search query
        
    Returns:
        Normalized search query
    """
    # Trim whitespace
    query = query.strip()
    
    # Remove redundant whitespace
    query = re.sub(r'\s+', ' ', query)
    
    return query


def categorize_search_results(
    results: List[Dict[str, Any]],
    categories: List[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """Categorize search results into groups.
    
    Args:
        results: Search results to categorize
        categories: Categories to use (e.g., file types)
        
    Returns:
        Results categorized into groups
    """
    categorized = {}
    
    # Initialize categories
    for category in categories:
        categorized[category] = []
    
    # Add uncategorized for items that don't match
    categorized["other"] = []
    
    # Categorize results
    for result in results:
        file_path = result.get("file", "")
        
        # Find matching category
        matched = False
        for category in categories:
            if file_path.endswith(f".{category}"):
                categorized[category].append(result)
                matched = True
                break
        
        # Add to other if no category matched
        if not matched:
            categorized["other"].append(result)
    
    return categorized 