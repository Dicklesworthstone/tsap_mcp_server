"""
MCP protocol request handler.

This module provides handlers for processing MCP protocol requests and
dispatching them to the appropriate tool implementations.
"""
import time
import asyncio
import traceback
from typing import Dict, Optional, Callable, Any

from tsap.utils.logging import logger
from tsap.performance_mode import get_performance_mode, set_performance_mode
from tsap.version import __version__, get_version_info
from tsap.core.ripgrep import ripgrep_search
from tsap.core.jq import jq_query
from tsap.core.awk import awk_process
from tsap.core.sqlite import sqlite_query
from tsap.composite.parallel import parallel_search
from tsap.composite.context import extract_context
from tsap.analysis.code import analyze_code
from tsap.analysis.documents import explore_documents
from tsap.analysis.strategy_compiler import compile_strategy
from tsap.evolution.pattern_analyzer import analyze_pattern
from tsap.composite.document_profiler import profile_documents
import faiss
from tsap.composite.semantic_search import SemanticSearchParams, get_operation

from .protocol import (
    MCPRequest, MCPResponse, MCPCommandType,
    create_success_response, create_error_response,
)

from .models import (
    RipgrepSearchParams,
)


# Registry of command handlers
_command_handlers: Dict[str, Callable] = {}


def register_command_handler(command: str, handler: Callable):
    """Register a handler function for an MCP command.
    
    Args:
        command: Command name to handle
        handler: Handler function
    """
    _command_handlers[command] = handler
    logger.debug(
        f"Registered handler for MCP command: {command}",
        component="mcp",
        operation="register_handler"
    )


def get_command_handler(command: str) -> Optional[Callable]:
    """Get the handler function for an MCP command.
    
    Args:
        command: Command name
        
    Returns:
        Handler function or None if not found
    """
    return _command_handlers.get(command)


async def handle_request(request: MCPRequest) -> MCPResponse:
    """Handle an MCP request.
    
    This function dispatches the request to the appropriate handler
    based on the command.
    
    Args:
        request: MCP request to handle
        
    Returns:
        MCP response
    """
    start_time = time.time()
    
    # Set performance mode if specified in request
    if request.mode:
        try:
            set_performance_mode(request.mode)
        except ValueError as e:
            return create_error_response(
                request_id=request.request_id,
                command=request.command,
                error_code="invalid_mode",
                error_message=str(e),
                execution_time=time.time() - start_time,
            )
    
    # Log request
    logger.info(
        f"Processing MCP request: {request.command}",
        component="mcp",
        operation="handle_request",
        context={
            "request_id": request.request_id,
            "mode": get_performance_mode(),
        }
    )
    
    # Find handler for this command
    handler = get_command_handler(request.command)
    
    if handler is None:
        # Handle meta-commands directly
        if request.command == MCPCommandType.INFO:
            return handle_info_command(request, start_time)
        elif request.command == MCPCommandType.STATUS:
            return handle_status_command(request, start_time)
        elif request.command == MCPCommandType.LIST_TOOLS:
            return handle_list_tools_command(request, start_time)
        elif request.command == MCPCommandType.CANCEL:
            # Not implemented yet
            return create_error_response(
                request_id=request.request_id,
                command=request.command,
                error_code="not_implemented",
                error_message="Cancel command is not implemented yet",
                execution_time=time.time() - start_time,
            )
        else:
            # Unknown command
            return create_error_response(
                request_id=request.request_id,
                command=request.command,
                error_code="unknown_command",
                error_message=f"Unknown command: {request.command}",
                execution_time=time.time() - start_time,
            )
    
    # Execute handler with timeout if specified
    try:
        if request.timeout:
            # Use asyncio.wait_for for timeout
            result = await asyncio.wait_for(
                handler(request.args),
                timeout=request.timeout,
            )
        else:
            # No timeout
            result = await handler(request.args)
            
        # Create success response
        return create_success_response(
            request_id=request.request_id,
            command=request.command,
            data=result,
            execution_time=time.time() - start_time,
        )
        
    except asyncio.TimeoutError:
        logger.warning(
            f"MCP request timed out after {request.timeout}s: {request.command}",
            component="mcp",
            operation="handle_request",
            context={"request_id": request.request_id}
        )
        
        return create_error_response(
            request_id=request.request_id,
            command=request.command,
            error_code="timeout",
            error_message=f"Request timed out after {request.timeout}s",
            execution_time=time.time() - start_time,
        )
        
    except Exception as e:
        logger.error(
            f"Error handling MCP request: {str(e)}",
            component="mcp",
            operation="handle_request",
            exception=e,
            context={"request_id": request.request_id}
        )
        
        return create_error_response(
            request_id=request.request_id,
            command=request.command,
            error_code="handler_error",
            error_message=str(e),
            error_details=traceback.format_exc(),
            execution_time=time.time() - start_time,
        )


def handle_info_command(request: MCPRequest, start_time: float) -> MCPResponse:
    """Handle the info command.
    
    Args:
        request: MCP request
        start_time: Request start time
        
    Returns:
        MCP response
    """
    info = {
        "version": __version__,
        "version_info": get_version_info(),
        "performance_mode": get_performance_mode(),
        "available_commands": list(MCPCommandType),
    }
    
    return create_success_response(
        request_id=request.request_id,
        command=request.command,
        data=info,
        execution_time=time.time() - start_time,
    )


def handle_status_command(request: MCPRequest, start_time: float) -> MCPResponse:
    """Handle the status command.
    
    Args:
        request: MCP request
        start_time: Request start time
        
    Returns:
        MCP response
    """
    status = {
        "status": "operational",
        "uptime": 0,  # TODO: Track uptime
        "active_requests": 0,  # TODO: Track active requests
        "performance_mode": get_performance_mode(),
    }
    
    return create_success_response(
        request_id=request.request_id,
        command=request.command,
        data=status,
        execution_time=time.time() - start_time,
    )


def handle_list_tools_command(request: MCPRequest, start_time: float) -> MCPResponse:
    """Handle the list_tools command.
    
    Args:
        request: MCP request
        start_time: Request start time
        
    Returns:
        MCP response
    """
    # Group commands by category
    tools = {
        "core": [
            {"name": MCPCommandType.RIPGREP_SEARCH, "description": "Search files using ripgrep"},
            {"name": MCPCommandType.AWK_PROCESS, "description": "Process text with AWK"},
            {"name": MCPCommandType.JQ_QUERY, "description": "Query JSON with jq"},
            {"name": MCPCommandType.SQLITE_QUERY, "description": "Query SQLite database"},
            {"name": MCPCommandType.HTML_PROCESS, "description": "Process HTML content"},
            {"name": MCPCommandType.PDF_EXTRACT, "description": "Extract text from PDF"},
            {"name": MCPCommandType.TABLE_PROCESS, "description": "Process tabular data"},
            {"name": MCPCommandType.SEMANTIC_SEARCH, "description": "Semantic search"},
        ],
        "composite": [
            {"name": MCPCommandType.PARALLEL_SEARCH, "description": "Search with multiple patterns in parallel"},
            {"name": MCPCommandType.RECURSIVE_REFINE, "description": "Recursively refine search results"},
            {"name": MCPCommandType.CONTEXT_EXTRACT, "description": "Extract context around matches"},
            {"name": MCPCommandType.PATTERN_ANALYZE, "description": "Analyze patterns in text"},
            {"name": MCPCommandType.FILENAME_DISCOVER, "description": "Discover filename patterns"},
            {"name": MCPCommandType.STRUCTURE_ANALYZE, "description": "Analyze document structure"},
            {"name": MCPCommandType.STRUCTURE_SEARCH, "description": "Search based on structure"},
            {"name": MCPCommandType.DIFF_GENERATE, "description": "Generate diff between texts"},
            {"name": MCPCommandType.REGEX_GENERATE, "description": "Generate regex for patterns"},
            {"name": MCPCommandType.DOCUMENT_PROFILE, "description": "Create document profile"},
            {"name": MCPCommandType.SEMANTIC_SEARCH, "description": "Semantic search"},
        ],
        "analysis": [
            {"name": MCPCommandType.CODE_ANALYZE, "description": "Analyze code"},
            {"name": MCPCommandType.DOCUMENT_EXPLORE, "description": "Explore document collection"},
            {"name": MCPCommandType.METADATA_EXTRACT, "description": "Extract metadata"},
            {"name": MCPCommandType.CORPUS_MAP, "description": "Map document corpus"},
            {"name": MCPCommandType.COUNTERFACTUAL_ANALYZE, "description": "Analyze counterfactuals"},
            {"name": MCPCommandType.STRATEGY_COMPILE, "description": "Compile search strategy"},
        ],
        "meta": [
            {"name": MCPCommandType.INFO, "description": "Get server information"},
            {"name": MCPCommandType.STATUS, "description": "Get server status"},
            {"name": MCPCommandType.CANCEL, "description": "Cancel a request"},
            {"name": MCPCommandType.LIST_TOOLS, "description": "List available tools"},
            {"name": MCPCommandType.LIST_STRATEGIES, "description": "List available strategies"},
            {"name": MCPCommandType.LIST_TEMPLATES, "description": "List available templates"},
        ],
    }
    
    return create_success_response(
        request_id=request.request_id,
        command=request.command,
        data=tools,
        execution_time=time.time() - start_time,
    )


def command_handler(command: str):
    """Decorator to register a function as an MCP command handler.
    
    Args:
        command: Command name to handle
        
    Returns:
        Decorator function
    """
    def decorator(func):
        register_command_handler(command, func)
        return func
    return decorator


# Initialize default handlers
def initialize_handlers():
    """Initialize default command handlers."""
    # Register core tools
    register_command_handler(MCPCommandType.RIPGREP_SEARCH, handle_ripgrep_search)
    register_command_handler(MCPCommandType.JQ_QUERY, handle_jq_query)
    register_command_handler(MCPCommandType.AWK_PROCESS, handle_awk_process)
    register_command_handler(MCPCommandType.SQLITE_QUERY, handle_sqlite_query)
    
    # Register composite operations
    register_command_handler(MCPCommandType.PARALLEL_SEARCH, handle_parallel_search)
    register_command_handler(MCPCommandType.CONTEXT_EXTRACT, handle_context_extract)
    
    # Register analysis tools
    register_command_handler(MCPCommandType.CODE_ANALYZE, handle_code_analyze)
    register_command_handler(MCPCommandType.DOCUMENT_EXPLORE, handle_document_explore)
    register_command_handler(MCPCommandType.STRATEGY_COMPILE, handle_strategy_compile)
    
    # Register evolution tools
    register_command_handler(MCPCommandType.PATTERN_ANALYZE, handle_pattern_analyze)
    register_command_handler(MCPCommandType.DOCUMENT_PROFILE, handle_document_profile)
    
    # Register test handler
    register_command_handler("test", handle_test)
    
    logger.info(
        "Registered MCP command handlers",
        component="mcp",
        operation="register_handlers",
        context={
            "handler_count": len(_command_handlers),
        }
    )


@command_handler(MCPCommandType.RIPGREP_SEARCH)
async def handle_ripgrep_search(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle ripgrep_search command.
    
    Args:
        args: Command arguments
        
    Returns:
        Command result
    """
    logger.info(
        "Handling ripgrep_search command",
        component="mcp",
        operation="ripgrep_search"
    )
    
    # Convert args dictionary to RipgrepSearchParams object with field adaptations
    adapted_args = args.copy()
    
    # Handle file_types/file_patterns and exclude_file_types/exclude_patterns differences
    if "file_types" in adapted_args:
        adapted_args["file_patterns"] = adapted_args.pop("file_types")
        
    if "exclude_file_types" in adapted_args:
        adapted_args["exclude_patterns"] = adapted_args.pop("exclude_file_types")
    
    # Add defaults for required fields in models.py that are not in protocol.py
    if "whole_word" not in adapted_args:
        adapted_args["whole_word"] = False
        
    if "regex" not in adapted_args:
        adapted_args["regex"] = True
        
    if "invert_match" not in adapted_args:
        adapted_args["invert_match"] = False
        
    # Create params object
    params = RipgrepSearchParams(**adapted_args)
    
    # Execute ripgrep search
    result = await ripgrep_search(params)
    
    # Convert to serializable form
    return result.model_dump()


@command_handler(MCPCommandType.JQ_QUERY)
async def handle_jq_query(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle jq_query command.
    
    Args:
        args: Command arguments
        
    Returns:
        Command result
    """
    logger.info(
        "Handling jq_query command",
        component="mcp",
        operation="jq_query"
    )
    
    # Execute jq query
    result = await jq_query(args)
    
    # Convert to serializable form
    return result.model_dump()


@command_handler(MCPCommandType.AWK_PROCESS)
async def handle_awk_process(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle awk_process command.
    
    Args:
        args: Command arguments
        
    Returns:
        Command result
    """
    logger.info(
        "Handling awk_process command",
        component="mcp",
        operation="awk_process"
    )
    
    # Execute awk process
    result = await awk_process(args)
    
    # Convert to serializable form
    return result.model_dump()


@command_handler(MCPCommandType.PARALLEL_SEARCH)
async def handle_parallel_search(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle parallel_search command.
    
    Args:
        args: Command arguments
        
    Returns:
        Command result
    """
    logger.info(
        "Handling parallel_search command",
        component="mcp",
        operation="parallel_search"
    )
    
    # Execute parallel search
    result = await parallel_search(args)
    
    # Convert to serializable form
    return result.model_dump()


@command_handler(MCPCommandType.CONTEXT_EXTRACT)
async def handle_context_extract(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle context_extract command.
    
    Args:
        args: Command arguments
        
    Returns:
        Command result
    """
    logger.info(
        "Handling context_extract command",
        component="mcp",
        operation="context_extract"
    )
    
    # Execute context extraction
    result = await extract_context(args)
    
    # Convert to serializable form
    return result.model_dump()


@command_handler(MCPCommandType.CODE_ANALYZE)
async def handle_code_analyze(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle code_analyze command.
    
    Args:
        args: Command arguments
        
    Returns:
        Command result
    """
    logger.info(
        "Handling code_analyze command",
        component="mcp",
        operation="code_analyze"
    )
    
    # Execute code analysis
    result = await analyze_code(args)
    
    # Convert to serializable form
    return result.model_dump()


@command_handler(MCPCommandType.DOCUMENT_EXPLORE)
async def handle_document_explore(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle document_explore command.
    
    Args:
        args: Command arguments
        
    Returns:
        Command result
    """
    logger.info(
        "Handling document_explore command",
        component="mcp",
        operation="document_explore"
    )
    
    # Execute document exploration
    result = await explore_documents(args)
    
    # Convert to serializable form
    return result.model_dump()


@command_handler(MCPCommandType.PATTERN_ANALYZE)
async def handle_pattern_analyze(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle pattern_analyze command.
    
    Args:
        args: Command arguments
        
    Returns:
        Command result
    """
    logger.info(
        "Handling pattern_analyze command",
        component="mcp",
        operation="pattern_analyze"
    )
    
    # Execute pattern analysis
    result = await analyze_pattern(
        pattern=args.get("pattern"),
        description=args.get("description", ""),
        is_regex=args.get("is_regex", True),
        case_sensitive=args.get("case_sensitive", False),
        paths=args.get("paths", []),
        reference_set=args.get("reference_set"),
        generate_variants=args.get("generate_variants", True),
        num_variants=args.get("num_variants", 3),
    )
    
    return result


@command_handler(MCPCommandType.STRATEGY_COMPILE)
async def handle_strategy_compile(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle strategy_compile command.
    
    Args:
        args: Command arguments
        
    Returns:
        Command result
    """
    logger.info(
        "Handling strategy_compile command",
        component="mcp",
        operation="strategy_compile"
    )
    
    # Execute strategy compilation
    result = await compile_strategy(args)
    
    # Convert to serializable form
    return result.model_dump()


@command_handler(MCPCommandType.SQLITE_QUERY)
async def handle_sqlite_query(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle sqlite_query command.
    
    Args:
        args: Command arguments
        
    Returns:
        Command result
    """
    logger.info(
        "Handling sqlite_query command",
        component="mcp",
        operation="sqlite_query"
    )
    
    # Execute sqlite query
    result = await sqlite_query(args)
    
    # Convert to serializable form
    return result.model_dump()


# Document Profiling Handler
@command_handler(MCPCommandType.DOCUMENT_PROFILE)
async def handle_document_profile(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle document_profile command.
    
    Args:
        args: Command arguments (e.g., document_paths, include_content_features, compare_documents)
        
    Returns:
        Command result containing profiles and comparisons
    """
    logger.info(
        "Handling document_profile command",
        component="mcp",
        operation="document_profile"
    )
    
    # Execute document profiling
    # Note: Mapping args directly, assuming keys match function parameters
    result = await profile_documents(
        document_paths=args.get("document_paths", []),
        include_content_features=args.get("include_content_features", True),
        compare_documents=args.get("compare_documents", True)
    )
    
    # Result is already a dict, no model_dump needed
    return result


# Semantic Search Handler
@command_handler(MCPCommandType.SEMANTIC_SEARCH)
async def handle_semantic_search(args: Dict[str, Any]) -> Dict[str, Any]:
    texts = args["texts"]
    query = args["query"]
    ids = args.get("ids", [f"doc_{i}" for i in range(len(texts))])
    top_k = args.get("top_k", 10)
    # Extract metadata from args, providing a default if missing
    metadata = args.get("metadata", [{} for _ in range(len(texts))])

    op = get_operation("semantic_search")
    # Pass metadata when creating params
    params = SemanticSearchParams(texts=texts, query=query, ids=ids, top_k=top_k, metadata=metadata)
    result = await op.execute_with_stats(params)

    backend = "gpu" if hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0 else "cpu"
    logger.info(
        f"Semantic search executed with {backend} FAISS backend",
        component="mcp",
        operation="semantic_search",
        context={"backend": backend, "query": query, "result_count": len(result)}
    )

    return {
        "results": result,
        "faiss_backend": backend,
        "query": query,
        "top_k": top_k
    }


# Simple test handler for debugging
async def handle_test(args: Dict[str, Any]) -> Dict[str, Any]:
    """Test handler for debugging.
    
    Args:
        args: Command arguments
        
    Returns:
        Test result
    """
    return {
        "message": "Test handler called successfully",
        "args": args,
        "handler_count": len(_command_handlers),
        "available_handlers": list(_command_handlers.keys()),
    }


# Register default handlers
initialize_handlers()