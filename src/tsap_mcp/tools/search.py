"""
Search tools for TSAP MCP Server.

This module provides MCP tool implementations for searching capabilities,
including ripgrep, text search, and semantic search functionality.
"""
import logging
from typing import Dict, List, Any, Optional, Union

from mcp.server.fastmcp import FastMCP, Context
from tsap_mcp.server import AppContext

# Import original implementations
# Ripgrep search
from tsap.core.ripgrep import RipgrepTool, RipgrepParams
# Text search
from tsap.core.text_search import TextSearchTool, TextSearchParams
# Semantic search
from tsap.core.semantic_search import SemanticSearchTool, SemanticSearchParams

logger = logging.getLogger("tsap_mcp.tools.search")


def register_search_tools(mcp: FastMCP) -> None:
    """Register all search-related tools with the MCP server.
    
    Args:
        mcp: FastMCP server instance
    """
    
    @mcp.tool()
    async def search_text(
        text: str,
        query: str,
        regex: bool = False,
        case_sensitive: bool = False,
        whole_word: bool = False,
        context_lines: int = 0,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """Search within text for matches.
        
        This tool searches within text for specific patterns, with options
        for regex, case sensitivity, whole word matching, and context.
        
        Args:
            text: Text to search in
            query: Pattern to search for
            regex: Whether to use regex for pattern matching
            case_sensitive: Whether the search is case sensitive
            whole_word: Whether to match whole words only
            context_lines: Number of context lines to include
            ctx: MCP context
            
        Returns:
            Search results with matches and context
        """
        if ctx:
            # Access the strongly typed context
            app_ctx: AppContext = ctx.request_context.lifespan_context
            debug_mode = app_ctx.config["debug"]
            if debug_mode:
                ctx.info(f"[DEBUG] Searching text for: {query} (length: {len(text)})")
            else:
                ctx.info(f"Searching text for: {query}")
        
        # Use original implementation
        params = TextSearchParams(
            text=text,
            query=query,
            regex=regex,
            case_sensitive=case_sensitive,
            whole_word=whole_word,
            context_lines=context_lines,
        )
        
        tool = TextSearchTool()
        result = await tool.search(params)
        
        if hasattr(result, 'results'):
            return result.results
        return {}
    
    @mcp.tool()
    async def search_regex(
        text: str,
        pattern: str,
        case_sensitive: bool = True,
        multiline: bool = False,
        group_results: bool = False,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """Search text using regular expressions.
        
        This tool searches text using regex patterns, with options for
        case sensitivity, multiline mode, and result grouping.
        
        Args:
            text: Text to search in
            pattern: Regex pattern to search for
            case_sensitive: Whether the search is case sensitive
            multiline: Whether to use multiline mode (^/$ match line start/end)
            group_results: Whether to include capture groups in results
            ctx: MCP context
            
        Returns:
            Regex search results with matches and groups
        """
        if ctx:
            ctx.info(f"Searching with regex: {pattern}")
        
        # Use original implementation
        params = TextSearchParams(
            text=text,
            pattern=pattern,
            case_sensitive=case_sensitive,
            multiline=multiline,
            group_results=group_results,
        )
        
        tool = TextSearchTool()
        result = await tool.search_regex(params)
        
        if hasattr(result, 'results'):
            return result.results
        return {}
    
    @mcp.tool()
    async def ripgrep_search(
        pattern: str,
        paths: List[str],
        regex: bool = False,
        case_sensitive: bool = True,
        whole_word: bool = False,
        file_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        context_lines: Optional[int] = None,
        before_context: Optional[int] = None,
        after_context: Optional[int] = None,
        max_count: Optional[int] = None,
        max_depth: Optional[int] = None,
        invert_match: bool = False,
        binary: bool = False,
        follow_symlinks: bool = False,
        hidden: bool = False,
        no_ignore: bool = False,
        encoding: Optional[str] = None,
        max_total_matches: Optional[int] = None,
        timeout: Optional[float] = None,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """Search files with ripgrep.
        
        This tool searches files using ripgrep, a high-performance 
        text searching tool, with extensive options for controlling
        the search behavior.
        
        Args:
            pattern: Pattern to search for
            paths: Paths to search in
            regex: Whether to use regex for pattern matching
            case_sensitive: Whether the search is case sensitive
            whole_word: Whether to match whole words only
            file_patterns: Patterns of files to include
            exclude_patterns: Patterns of files to exclude
            context_lines: Number of context lines to include
            before_context: Number of context lines before match
            after_context: Number of context lines after match
            max_count: Maximum number of matches per file
            max_depth: Maximum directory search depth
            invert_match: Whether to return non-matching lines
            binary: Whether to search binary files
            follow_symlinks: Whether to follow symbolic links
            hidden: Whether to search hidden files
            no_ignore: Whether to ignore .gitignore files
            encoding: File encoding to use
            max_total_matches: Maximum total matches to return
            timeout: Search timeout in seconds
            ctx: MCP context
            
        Returns:
            Ripgrep search results with matches and stats
        """
        if ctx:
            ctx.info(f"Searching with ripgrep: {pattern} in {', '.join(paths)}")
        
        # Use original implementation
        params = RipgrepParams(
            pattern=pattern,
            paths=paths,
            regex=regex,
            case_sensitive=case_sensitive,
            whole_word=whole_word,
            file_patterns=file_patterns or [],
            exclude_patterns=exclude_patterns or [],
            context_lines=context_lines,
            before_context=before_context,
            after_context=after_context,
            max_count=max_count,
            max_depth=max_depth,
            invert_match=invert_match,
            binary=binary,
            follow_symlinks=follow_symlinks,
            hidden=hidden,
            no_ignore=no_ignore,
            encoding=encoding,
            max_total_matches=max_total_matches,
            timeout=timeout,
        )
        
        tool = RipgrepTool()
        result = await tool.search(params)
        
        if hasattr(result, 'results'):
            return result.results
        return {}
    
    @mcp.tool()
    async def search_semantic(
        query: str,
        corpus: Union[str, List[str]],
        model: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        threshold: float = 0.5,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """Search text semantically for similar content.
        
        This tool performs semantic search on text, finding content that is
        semantically similar to the query, not just exact matches.
        
        Args:
            query: Query text to search for
            corpus: Text or list of texts to search in
            model: Embedding model to use
            top_k: Maximum number of results to return
            threshold: Minimum similarity threshold
            ctx: MCP context
            
        Returns:
            Semantic search results with similarity scores
        """
        if ctx:
            ctx.info(f"Performing semantic search for: {query}")
        
        # Use original implementation
        params = SemanticSearchParams(
            query=query,
            corpus=corpus,
            model=model,
            top_k=top_k,
            threshold=threshold,
        )
        
        tool = SemanticSearchTool()
        result = await tool.search(params)
        
        if hasattr(result, 'results'):
            return result.results
        return {}
        
    # Register more search tools as needed 