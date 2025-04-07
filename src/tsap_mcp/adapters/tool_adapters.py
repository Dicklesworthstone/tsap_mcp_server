"""
Tool adapters for TSAP MCP Server.

This module provides adapter functions that bridge between the original TSAP
tool implementations and the MCP native format.
"""
from typing import Dict, Any, Optional, List, Union
from mcp.server.fastmcp import Context


async def adapted_ripgrep_search(
    pattern: str, 
    path: str = ".", 
    case_sensitive: bool = False,
    include_pattern: Optional[str] = None,
    exclude_pattern: Optional[str] = None,
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """Adapt the original ripgrep implementation to MCP format.
    
    Args:
        pattern: The regex pattern to search for
        path: Directory path to search in
        case_sensitive: Whether the search is case-sensitive
        include_pattern: Pattern to include files (e.g., "*.py")
        exclude_pattern: Pattern to exclude files (e.g., "*.pyc")
        ctx: MCP context object
        
    Returns:
        Search results in MCP-compatible format
    """
    # Import from original implementation
    from tsap.toolapi.models import RipgrepSearchParams
    from tsap.core.ripgrep import ripgrep_search
    
    # Map MCP parameters to original format
    params = RipgrepSearchParams(
        pattern=pattern,
        path=path,
        case_sensitive=case_sensitive,
        include_pattern=include_pattern,
        exclude_pattern=exclude_pattern,
    )
    
    # Get performance mode from context if available
    performance_mode = None
    if ctx and hasattr(ctx, "request_context") and hasattr(ctx.request_context, "lifespan_context"):
        tsap_context = ctx.request_context.lifespan_context
        performance_mode = tsap_context.get("performance_mode", "standard")
    
    # Progress reporting
    if ctx:
        ctx.info(f"Searching for pattern: {pattern}")
        await ctx.report_progress(0, 100, "Starting search")
    
    # Call original implementation
    result = await ripgrep_search(params, mode=performance_mode)
    
    # Progress reporting
    if ctx:
        await ctx.report_progress(100, 100, "Search complete")
        match_count = len(result.matches) if hasattr(result, "matches") else 0
        ctx.info(f"Found {match_count} matches")
    
    # Convert to dictionary for MCP output
    if hasattr(result, "dict"):
        return result.dict()
    return result


async def adapted_jq_query(
    json_data: str,
    query: str,
    raw_output: bool = False,
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """Adapt the original jq implementation to MCP format.
    
    Args:
        json_data: JSON data to query
        query: jq query string
        raw_output: Whether to return raw output
        ctx: MCP context object
        
    Returns:
        Query results in MCP-compatible format
    """
    # Import from original implementation
    from tsap.toolapi.models import JqQueryParams
    from tsap.core.jq import jq_query
    
    # Map MCP parameters to original format
    params = JqQueryParams(
        json_data=json_data,
        query=query,
        raw_output=raw_output,
    )
    
    # Get performance mode from context if available
    performance_mode = None
    if ctx and hasattr(ctx, "request_context") and hasattr(ctx.request_context, "lifespan_context"):
        tsap_context = ctx.request_context.lifespan_context
        performance_mode = tsap_context.get("performance_mode", "standard")
    
    # Progress reporting
    if ctx:
        ctx.info(f"Executing jq query: {query}")
        await ctx.report_progress(0, 100, "Starting query")
    
    # Call original implementation
    result = await jq_query(params, mode=performance_mode)
    
    # Progress reporting
    if ctx:
        await ctx.report_progress(100, 100, "Query complete")
    
    # Convert to dictionary for MCP output
    if hasattr(result, "dict"):
        return result.dict()
    return result


async def adapted_html_process(
    html: Optional[str] = None,
    url: Optional[str] = None,
    file_path: Optional[str] = None,
    selector: Optional[str] = None,
    xpath: Optional[str] = None,
    extract_tables: bool = False,
    extract_links: bool = False,
    extract_text: bool = True,
    extract_metadata: bool = False,
    render_js: bool = False,
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """Adapt the original HTML processor to MCP format.
    
    Args:
        html: HTML content as string
        url: URL to fetch HTML from
        file_path: Path to HTML file
        selector: CSS selector
        xpath: XPath expression
        extract_tables: Whether to extract tables
        extract_links: Whether to extract links
        extract_text: Whether to extract text
        extract_metadata: Whether to extract metadata
        render_js: Whether to render JavaScript
        ctx: MCP context object
        
    Returns:
        Processing results in MCP-compatible format
    """
    # Import from original implementation
    from tsap.toolapi.models import HtmlProcessParams
    from tsap.core.html_processor import process_html
    
    # Map MCP parameters to original format
    params = HtmlProcessParams(
        html=html,
        url=url,
        file_path=file_path,
        selector=selector,
        xpath=xpath,
        extract_tables=extract_tables,
        extract_links=extract_links,
        extract_text=extract_text,
        extract_metadata=extract_metadata,
        render_js=render_js,
    )
    
    # Get performance mode from context if available
    performance_mode = None
    if ctx and hasattr(ctx, "request_context") and hasattr(ctx.request_context, "lifespan_context"):
        tsap_context = ctx.request_context.lifespan_context
        performance_mode = tsap_context.get("performance_mode", "standard")
    
    # Progress reporting
    if ctx:
        source_type = "HTML string" if html else "URL" if url else "file"
        ctx.info(f"Processing HTML from {source_type}")
        await ctx.report_progress(0, 100, "Starting HTML processing")
    
    # Call original implementation
    result = await process_html(params, mode=performance_mode)
    
    # Progress reporting
    if ctx:
        await ctx.report_progress(100, 100, "HTML processing complete")
    
    # Convert to dictionary for MCP output
    if hasattr(result, "dict"):
        return result.dict()
    return result


async def adapted_document_profiler(
    document_text: str,
    profile_type: str = "general",
    include_patterns: bool = True,
    include_metrics: bool = True,
    include_structure: bool = True,
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """Adapt the original document profiler to MCP format.
    
    Args:
        document_text: Text of the document to profile
        profile_type: Type of profile to generate (general, code, text)
        include_patterns: Whether to include pattern analysis
        include_metrics: Whether to include text metrics
        include_structure: Whether to include structure analysis
        ctx: MCP context object
        
    Returns:
        Profiling results in MCP-compatible format
    """
    # Import from original implementation
    from tsap.toolapi.models import ProfileParams
    from tsap.composite.document_profiler import profile_document
    
    # Map MCP parameters to original format
    params = ProfileParams(
        text=document_text,
        profile_type=profile_type,
        options={
            "include_patterns": include_patterns,
            "include_metrics": include_metrics,
            "include_structure": include_structure,
        },
    )
    
    # Get performance mode from context if available
    performance_mode = None
    if ctx and hasattr(ctx, "request_context") and hasattr(ctx.request_context, "lifespan_context"):
        tsap_context = ctx.request_context.lifespan_context
        performance_mode = tsap_context.get("performance_mode", "standard")
    
    # Progress reporting
    if ctx:
        ctx.info(f"Profiling document with profile type: {profile_type}")
        await ctx.report_progress(0, 100, "Starting document profiling")
    
    # Call original implementation
    result = await profile_document(params, mode=performance_mode)
    
    # Progress reporting
    if ctx:
        await ctx.report_progress(100, 100, "Document profiling complete")
    
    # Convert to dictionary for MCP output
    if hasattr(result, "dict"):
        return result.dict()
    return result


async def adapted_regex_generator(
    matches: List[str],
    non_matches: Optional[List[str]] = None,
    simplify: bool = True,
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """Adapt the regex generator to MCP format.
    
    Args:
        matches: Examples of strings to match
        non_matches: Examples of strings not to match
        simplify: Whether to simplify the resulting pattern
        ctx: MCP context object
        
    Returns:
        Regex generation results in MCP-compatible format
    """
    # Import from original implementation
    from tsap.toolapi.models import RegexGenParams
    from tsap.composite.regex_generator import generate_regex as original_generate_regex
    
    # Map MCP parameters to original format
    params = RegexGenParams(
        matches=matches,
        non_matches=non_matches or [],
        options={
            "simplify": simplify,
        },
    )
    
    # Get performance mode from context if available
    performance_mode = None
    if ctx and hasattr(ctx, "request_context") and hasattr(ctx.request_context, "lifespan_context"):
        tsap_context = ctx.request_context.lifespan_context
        performance_mode = tsap_context.get("performance_mode", "standard")
    
    # Progress reporting
    if ctx:
        ctx.info(f"Generating regex from {len(matches)} examples")
        await ctx.report_progress(0, 100, "Starting regex generation")
    
    # Call original implementation
    result = await original_generate_regex(params, mode=performance_mode)
    
    # Progress reporting
    if ctx:
        await ctx.report_progress(100, 100, "Regex generation complete")
    
    # Convert to dictionary for MCP output
    if hasattr(result, "dict"):
        return result.dict()
    return result


async def adapted_pattern_finder(
    text: str,
    pattern_types: Optional[List[str]] = None,
    min_occurrences: int = 2,
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """Adapt the pattern finder to MCP format.
    
    Args:
        text: Text to analyze for patterns
        pattern_types: Types of patterns to look for
        min_occurrences: Minimum number of occurrences
        ctx: MCP context object
        
    Returns:
        Pattern finding results in MCP-compatible format
    """
    # Import from original implementation
    from tsap.toolapi.models import PatternFinderParams
    from tsap.composite.patterns import find_patterns as original_find_patterns
    
    # Default pattern types if not specified
    if pattern_types is None:
        pattern_types = ["dates", "emails", "urls", "numbers", "identifiers"]
    
    # Map MCP parameters to original format
    params = PatternFinderParams(
        text=text,
        pattern_types=pattern_types,
        options={
            "min_occurrences": min_occurrences,
        },
    )
    
    # Get performance mode from context if available
    performance_mode = None
    if ctx and hasattr(ctx, "request_context") and hasattr(ctx.request_context, "lifespan_context"):
        tsap_context = ctx.request_context.lifespan_context
        performance_mode = tsap_context.get("performance_mode", "standard")
    
    # Progress reporting
    if ctx:
        ctx.info(f"Finding patterns in text ({', '.join(pattern_types)})")
        await ctx.report_progress(0, 100, "Starting pattern analysis")
    
    # Call original implementation
    result = await original_find_patterns(params, mode=performance_mode)
    
    # Progress reporting
    if ctx:
        await ctx.report_progress(100, 100, "Pattern analysis complete")
        
        # Report found patterns
        if hasattr(result, "patterns"):
            pattern_count = sum(len(patterns) for patterns in result.patterns.values())
            ctx.info(f"Found {pattern_count} patterns")
    
    # Convert to dictionary for MCP output
    if hasattr(result, "dict"):
        return result.dict()
    return result


async def adapted_diff_generator(
    original_text: str,
    modified_text: str,
    context_lines: int = 3,
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """Adapt the diff generator to MCP format.
    
    Args:
        original_text: Original text
        modified_text: Modified text
        context_lines: Number of context lines
        ctx: MCP context object
        
    Returns:
        Diff generation results in MCP-compatible format
    """
    # Import from original implementation
    from tsap.toolapi.models import DiffParams
    from tsap.composite.diff_generator import generate_diff as original_generate_diff
    
    # Map MCP parameters to original format
    params = DiffParams(
        original_text=original_text,
        modified_text=modified_text,
        options={
            "context_lines": context_lines,
        },
    )
    
    # Get performance mode from context if available
    performance_mode = None
    if ctx and hasattr(ctx, "request_context") and hasattr(ctx.request_context, "lifespan_context"):
        tsap_context = ctx.request_context.lifespan_context
        performance_mode = tsap_context.get("performance_mode", "standard")
    
    # Progress reporting
    if ctx:
        ctx.info(f"Generating diff with {context_lines} lines of context")
        await ctx.report_progress(0, 100, "Starting diff generation")
    
    # Call original implementation
    result = await original_generate_diff(params, mode=performance_mode)
    
    # Progress reporting
    if ctx:
        await ctx.report_progress(100, 100, "Diff generation complete")
    
    # Convert to dictionary for MCP output
    if hasattr(result, "dict"):
        return result.dict()
    return result


async def adapted_code_structure_analyzer(
    code: str,
    language: str,
    include_docstrings: bool = True,
    include_metrics: bool = False,
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """Adapt the code structure analyzer to MCP format.
    
    Args:
        code: Source code to analyze
        language: Programming language
        include_docstrings: Whether to include docstrings
        include_metrics: Whether to include code metrics
        ctx: MCP context object
        
    Returns:
        Structure analysis results in MCP-compatible format
    """
    # Import from original implementation
    from tsap.toolapi.models import StructureParams
    from tsap.core.structure import analyze_structure
    
    # Map MCP parameters to original format
    params = StructureParams(
        code=code,
        language=language,
        options={
            "include_docstrings": include_docstrings,
            "include_metrics": include_metrics,
        },
    )
    
    # Get performance mode from context if available
    performance_mode = None
    if ctx and hasattr(ctx, "request_context") and hasattr(ctx.request_context, "lifespan_context"):
        tsap_context = ctx.request_context.lifespan_context
        performance_mode = tsap_context.get("performance_mode", "standard")
    
    # Progress reporting
    if ctx:
        ctx.info(f"Analyzing {language} code structure")
        await ctx.report_progress(0, 100, "Starting structure analysis")
    
    # Call original implementation
    result = await analyze_structure(params, mode=performance_mode)
    
    # Progress reporting
    if ctx:
        await ctx.report_progress(100, 100, "Structure analysis complete")
    
    # Convert to dictionary for MCP output
    if hasattr(result, "dict"):
        return result.dict()
    return result


async def adapted_security_analyzer(
    code: str,
    language: str,
    severity_level: str = "medium",
    include_suggestions: bool = True,
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """Adapt the security analyzer to MCP format.
    
    Args:
        code: Source code to analyze
        language: Programming language
        severity_level: Minimum severity level to report
        include_suggestions: Whether to include remediation suggestions
        ctx: MCP context object
        
    Returns:
        Security analysis results in MCP-compatible format
    """
    # Import from original implementation
    from tsap.toolapi.models import SecurityParams
    from tsap.core.security import analyze_security
    
    # Map MCP parameters to original format
    params = SecurityParams(
        code=code,
        language=language,
        options={
            "severity_level": severity_level,
            "include_suggestions": include_suggestions,
        },
    )
    
    # Get performance mode from context if available
    performance_mode = None
    if ctx and hasattr(ctx, "request_context") and hasattr(ctx.request_context, "lifespan_context"):
        tsap_context = ctx.request_context.lifespan_context
        performance_mode = tsap_context.get("performance_mode", "standard")
    
    # Progress reporting
    if ctx:
        ctx.info(f"Performing security analysis on {language} code")
        await ctx.report_progress(0, 100, "Starting security analysis")
    
    # Call original implementation
    result = await analyze_security(params, mode=performance_mode)
    
    # Progress reporting
    if ctx:
        await ctx.report_progress(100, 100, "Security analysis complete")
        finding_count = len(result.findings) if hasattr(result, "findings") else 0
        ctx.info(f"Found {finding_count} security issues")
    
    # Convert to dictionary for MCP output
    if hasattr(result, "dict"):
        return result.dict()
    return result


async def adapted_chart_generator(
    data: Union[str, List[Dict[str, Any]]],
    chart_type: str = "bar",
    title: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    format: str = "png",
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """Adapt the chart generator to MCP format.
    
    Args:
        data: Data to visualize (JSON string or list of dicts)
        chart_type: Type of chart to generate
        title: Chart title
        options: Additional chart options
        format: Output format (png, svg, json)
        ctx: MCP context object
        
    Returns:
        Chart generation results in MCP-compatible format
    """
    # Import from original implementation
    from tsap.toolapi.models import ChartParams
    from tsap.core.visualization import generate_chart
    
    # Convert string data to JSON if needed
    if isinstance(data, str):
        import json
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            # Return error in MCP format
            return {
                "error": "Invalid JSON data",
                "data": data[:100] + "..." if len(data) > 100 else data
            }
    
    # Map MCP parameters to original format
    params = ChartParams(
        data=data,
        chart_type=chart_type,
        title=title,
        options=options or {},
        format=format,
    )
    
    # Get performance mode from context if available
    performance_mode = None
    if ctx and hasattr(ctx, "request_context") and hasattr(ctx.request_context, "lifespan_context"):
        tsap_context = ctx.request_context.lifespan_context
        performance_mode = tsap_context.get("performance_mode", "standard")
    
    # Progress reporting
    if ctx:
        ctx.info(f"Generating {chart_type} chart")
        await ctx.report_progress(0, 100, "Starting chart generation")
    
    # Call original implementation
    result = await generate_chart(params, mode=performance_mode)
    
    # Progress reporting
    if ctx:
        await ctx.report_progress(100, 100, "Chart generation complete")
    
    # Process image data if present and format is binary
    if format in ["png", "jpg", "svg"] and hasattr(result, "image_data"):
        from tsap_mcp.utils.visualization import encode_image_base64
        if isinstance(result.image_data, bytes):
            result.image_data = encode_image_base64(result.image_data, format)
    
    # Convert to dictionary for MCP output
    if hasattr(result, "dict"):
        return result.dict()
    return result


async def adapted_graph_visualizer(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    layout: str = "force",
    options: Optional[Dict[str, Any]] = None,
    format: str = "png",
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """Adapt the graph visualizer to MCP format.
    
    Args:
        nodes: List of node definitions
        edges: List of edge definitions
        layout: Graph layout algorithm
        options: Additional visualization options
        format: Output format (png, svg, json)
        ctx: MCP context object
        
    Returns:
        Graph visualization results in MCP-compatible format
    """
    # Import from original implementation
    from tsap.toolapi.models import GraphParams
    from tsap.core.visualization import visualize_graph
    
    # Map MCP parameters to original format
    params = GraphParams(
        nodes=nodes,
        edges=edges,
        layout=layout,
        options=options or {},
        format=format,
    )
    
    # Get performance mode from context if available
    performance_mode = None
    if ctx and hasattr(ctx, "request_context") and hasattr(ctx.request_context, "lifespan_context"):
        tsap_context = ctx.request_context.lifespan_context
        performance_mode = tsap_context.get("performance_mode", "standard")
    
    # Progress reporting
    if ctx:
        ctx.info(f"Generating graph visualization with {layout} layout")
        await ctx.report_progress(0, 100, "Starting graph visualization")
    
    # Call original implementation
    result = await visualize_graph(params, mode=performance_mode)
    
    # Progress reporting
    if ctx:
        await ctx.report_progress(100, 100, "Graph visualization complete")
    
    # Process image data if present and format is binary
    if format in ["png", "jpg", "svg"] and hasattr(result, "image_data"):
        from tsap_mcp.utils.visualization import encode_image_base64
        if isinstance(result.image_data, bytes):
            result.image_data = encode_image_base64(result.image_data, format)
    
    # Convert to dictionary for MCP output
    if hasattr(result, "dict"):
        return result.dict()
    return result 