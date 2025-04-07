"""
Composite tools for TSAP MCP Server.

This module provides MCP tool implementations for composite operations that
combine multiple core functionalities, such as document profiling, structure
searching, pattern analysis, and more.
"""
import logging
from typing import Dict, List, Any, Optional, Union
import json

from mcp.server.fastmcp import FastMCP, Context

# Import original implementations
# Document profiler
from tsap.composite.document_profiler import DocumentProfiler, ProfileParams
# Structure search
from tsap.composite.structure_search import StructureSearch, StructureSearchParams
# Pattern matching
from tsap.composite.patterns import PatternMatcher, PatternParams
# Context extraction
from tsap.composite.context import ContextExtractor, ContextParams
# Semantic search
from tsap.core.semantic_search_tool import SemanticSearchTool, SemanticSearchParams
# Diff generation
from tsap.composite.diff_generator import DiffGenerator, DiffParams
# Regex generation
from tsap.composite.regex_generator import RegexGenerator, RegexParams
# Filename operations
from tsap.composite.filenames import FilenameProcessor, FilenameParams

logger = logging.getLogger("tsap_mcp.tools.composite")


def register_composite_tools(mcp: FastMCP) -> None:
    """Register all composite tools with the MCP server.
    
    Args:
        mcp: FastMCP server instance
    """
    
    @mcp.tool()
    async def profile_document(
        text: str,
        profile_type: str = "general",
        options: Optional[Dict[str, Any]] = None,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """Profile a document to extract key characteristics.
        
        This tool analyzes a document to extract information about its structure,
        content types, patterns, and other characteristics.
        
        Args:
            text: Document text to analyze
            profile_type: Type of profile to generate (general, code, data, etc.)
            options: Additional options for profiling
            ctx: MCP context
            
        Returns:
            Document profile with various characteristics
        """
        if ctx:
            ctx.info(f"Profiling document with profile type: {profile_type}")
        
        # Use original implementation
        params = ProfileParams(
            text=text,
            profile_type=profile_type,
            options=options or {},
        )
        
        profiler = DocumentProfiler()
        result = await profiler.profile(params)
        
        if hasattr(result, 'profile'):
            return result.profile
        return {}
    
    @mcp.tool()
    async def search_structure(
        text: str,
        pattern: str,
        options: Optional[Dict[str, Any]] = None,
        ctx: Optional[Context] = None,
    ) -> List[Dict[str, Any]]:
        """Search for structural patterns in text.
        
        This tool identifies structural patterns like code blocks, lists,
        tables, and other formatted content within text.
        
        Args:
            text: Text to search for structures
            pattern: Structure pattern to search for (code, list, table, etc.)
            options: Additional options for structure search
            ctx: MCP context
            
        Returns:
            List of matched structures with locations and details
        """
        if ctx:
            ctx.info(f"Searching for structure pattern: {pattern}")
        
        # Use original implementation
        params = StructureSearchParams(
            text=text,
            pattern=pattern,
            options=options or {},
        )
        
        searcher = StructureSearch()
        result = await searcher.search(params)
        
        if hasattr(result, 'matches'):
            return result.matches
        return []
    
    @mcp.tool()
    async def match_patterns(
        text: str,
        patterns: List[str],
        options: Optional[Dict[str, Any]] = None,
        ctx: Optional[Context] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Match patterns in text using various techniques.
        
        This tool finds instances of patterns in text, using techniques like
        regex, fuzzy matching, semantic matching, and others.
        
        Args:
            text: Text to search for patterns
            patterns: List of patterns to match
            options: Additional options for pattern matching
            ctx: MCP context
            
        Returns:
            Dictionary of pattern matches by pattern name
        """
        if ctx:
            ctx.info(f"Matching {len(patterns)} patterns in text")
        
        # Use original implementation
        params = PatternParams(
            text=text,
            patterns=patterns,
            options=options or {},
        )
        
        matcher = PatternMatcher()
        result = await matcher.match(params)
        
        if hasattr(result, 'matches'):
            return result.matches
        return {}
    
    @mcp.tool()
    async def extract_context(
        text: str,
        query: str,
        context_lines: int = 3,
        options: Optional[Dict[str, Any]] = None,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """Extract context around matches in text.
        
        This tool finds matches for a query and extracts surrounding context
        to provide more information about the matches.
        
        Args:
            text: Text to search
            query: Query to search for
            context_lines: Number of context lines to include before/after
            options: Additional options for context extraction
            ctx: MCP context
            
        Returns:
            Matched text with surrounding context
        """
        if ctx:
            ctx.info(f"Extracting context for query: {query}")
        
        # Use original implementation
        params = ContextParams(
            text=text,
            query=query,
            context_before=context_lines,
            context_after=context_lines,
            options=options or {},
        )
        
        extractor = ContextExtractor()
        result = await extractor.extract(params)
        
        if hasattr(result, 'contexts'):
            return result.contexts
        return {}
    
    @mcp.tool()
    async def generate_diff(
        text1: str,
        text2: str,
        format: str = "unified",
        context_lines: int = 3,
        ctx: Optional[Context] = None,
    ) -> str:
        """Generate a diff between two texts.
        
        This tool generates a difference report between two texts, showing
        what was added, removed, or changed.
        
        Args:
            text1: Original text
            text2: Modified text
            format: Diff format (unified, context, or side-by-side)
            context_lines: Number of context lines to include
            ctx: MCP context
            
        Returns:
            Diff output in the specified format
        """
        if ctx:
            ctx.info(f"Generating {format} diff with {context_lines} context lines")
        
        # Use original implementation
        params = DiffParams(
            text1=text1,
            text2=text2,
            format=format,
            context_lines=context_lines,
        )
        
        generator = DiffGenerator()
        result = await generator.generate(params)
        
        if hasattr(result, 'diff'):
            return result.diff
        return ""
    
    @mcp.tool()
    async def generate_regex(
        examples: List[str],
        antiexamples: Optional[List[str]] = None,
        options: Optional[Dict[str, Any]] = None,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """Generate a regex pattern from examples.
        
        This tool creates regex patterns that match the provided examples
        and don't match the antiexamples, if provided.
        
        Args:
            examples: Strings that should match the pattern
            antiexamples: Strings that should not match the pattern
            options: Additional options for regex generation
            ctx: MCP context
            
        Returns:
            Generated regex with confidence score and explanation
        """
        if ctx:
            ctx.info(f"Generating regex from {len(examples)} examples")
        
        # Use original implementation
        params = RegexParams(
            examples=examples,
            antiexamples=antiexamples or [],
            options=options or {},
        )
        
        generator = RegexGenerator()
        result = await generator.generate(params)
        
        if hasattr(result, 'result'):
            return result.result
        return {}
    
    @mcp.tool()
    async def semantic_search(
        query: str,
        corpus: Union[str, List[str]],
        top_k: int = 5,
        threshold: float = 0.5,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """Search for semantically similar content.
        
        This tool finds content that is semantically similar to the query,
        not just exact matches, using embeddings-based search.
        
        Args:
            query: Search query
            corpus: Text or list of documents to search in
            top_k: Maximum number of results to return
            threshold: Minimum similarity threshold
            ctx: MCP context
            
        Returns:
            Semantic search results with similarity scores
        """
        if ctx:
            ctx.info(f"Searching semantically for: {query}")
        
        # Use original implementation
        # Convert corpus to list of documents if it's a string
        if isinstance(corpus, str):
            # Split by double newlines (paragraphs) for better results
            corpus = [doc.strip() for doc in corpus.split("\n\n") if doc.strip()]
            
        params = SemanticSearchParams(
            query=query,
            corpus=corpus,
            top_k=top_k,
            threshold=threshold,
        )
        
        tool = SemanticSearchTool()
        result = await tool.search(params)
        
        if hasattr(result, 'results'):
            return result.results
        return {}
    
    @mcp.tool()
    async def analyze_filenames(
        filenames: List[str],
        operation: str = "extract_patterns",
        options: Optional[Dict[str, Any]] = None,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """Analyze patterns and information in filenames.
        
        This tool analyzes filenames to extract patterns, categorize files,
        suggest renames, and perform other filename-related operations.
        
        Args:
            filenames: List of filenames to analyze
            operation: Analysis operation to perform
            options: Additional options for the operation
            ctx: MCP context
            
        Returns:
            Analysis results based on the operation
        """
        if ctx:
            ctx.info(f"Analyzing filenames with operation: {operation}")
        
        # Use original implementation
        params = FilenameParams(
            filenames=filenames,
            operation=operation,
            options=options or {},
        )
        
        processor = FilenameProcessor()
        result = await processor.process(params)
        
        if hasattr(result, 'result'):
            return result.result
        return {}
    
    @mcp.tool()
    async def structure_search(
        query: str,
        content: Union[str, Dict[str, Any]],
        structure_type: str = "auto",
        max_results: int = 10,
        include_context: bool = True,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """Search for patterns within structured data.
        
        This tool searches for patterns within structured data like JSON, XML, or
        code, with specialized search capabilities for each structure type.
        
        Args:
            query: Search query
            content: Structured content to search
            structure_type: Type of structure (json, xml, code, auto)
            max_results: Maximum number of results to return
            include_context: Whether to include surrounding context
            ctx: MCP context
            
        Returns:
            Search results with matches and context
        """
        if ctx:
            ctx.info(f"Searching {structure_type} structure for: {query}")
        
        # Use original implementation
        # We'll reuse StructureSearch but with different parameters
        if isinstance(content, dict):
            # Convert dict to JSON string for consistent handling
            content_str = json.dumps(content, indent=2)
            detected_type = "json"
        else:
            content_str = content
            # Auto-detect structure type if needed
            if structure_type == "auto":
                if content_str.strip().startswith("{") or content_str.strip().startswith("["):
                    detected_type = "json"
                elif content_str.strip().startswith("<"):
                    detected_type = "xml"
                else:
                    detected_type = "code"
            else:
                detected_type = structure_type
                
        params = StructureSearchParams(
            text=content_str,
            pattern=query,
            options={
                "structure_type": detected_type,
                "max_results": max_results,
                "include_context": include_context,
            },
        )
        
        searcher = StructureSearch()
        result = await searcher.search_structured(params)
        
        if hasattr(result, 'matches'):
            return {
                "query": query,
                "structure_type": detected_type,
                "matches": result.matches,
                "match_count": len(result.matches),
            }
        return {
            "query": query,
            "structure_type": detected_type,
            "matches": [],
            "match_count": 0,
        } 