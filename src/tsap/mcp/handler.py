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

# Log module import
logger.info("Importing src.tsap.mcp.handler module...", component="mcp", operation="module_import")

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
from tsap.core.table_processor import process_table, get_table_processor
from tsap.core.html_processor import process_html, HtmlProcessParams

from .protocol import (
    MCPRequest, MCPResponse, MCPCommandType,
    create_success_response, create_error_response,
)

from .models import (
    RipgrepSearchParams,
    JqQueryParams,
    TableProcessParams,
    TableProcessResult,
    PdfExtractParams,
    PdfExtractResult,
)


# Registry of command handlers
_command_handlers: Dict[str, Callable] = {}


def register_command_handler(command: str, handler: Callable):
    """Register a handler function for an MCP command.
    
    Args:
        command: Command name to handle
        handler: Handler function
    """
    # Ensure the key is always the string value
    key = command.value if isinstance(command, MCPCommandType) else command
    _command_handlers[key] = handler
    logger.debug(
        f"Registered handler for MCP command: {key}", # Log the key being used
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
    logger.info(
        f"Getting handler for MCP command: {command}",
        component="mcp",
        operation="get_handler"
    )
    # Log dictionary ID at lookup time
    logger.info(f"Checking _command_handlers ID at lookup: {id(_command_handlers)}", component="mcp", operation="get_handler")
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
    # Get tool-specific parameter information
    html_processor_params = [
        "html", "url", "file_path", "selector", "xpath", 
        "extract_tables", "extract_links", "extract_text", "extract_metadata",
        "render_js", "js_timeout", "interactive_actions", "extract_computed_styles"
    ]
    
    ripgrep_params = [
        "pattern", "paths", "file_patterns", "ignore_case", "word_regexp", 
        "max_count", "max_depth", "context_lines", "include_hidden", 
        "follow_symlinks", "max_matches_per_file", "max_total_matches"
    ]
    
    jq_params = [
        "query", "input_json", "input_files", "compact_output", 
        "raw_output", "slurp", "sort_keys"
    ]
    
    info = {
        "version": __version__,
        "version_info": get_version_info(),
        "performance_mode": get_performance_mode(),
        "available_commands": list(MCPCommandType),
        "html_processor_params": html_processor_params,
        "ripgrep_params": ripgrep_params,
        "jq_params": jq_params,
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
    register_command_handler(MCPCommandType.PDF_EXTRACT, handle_pdf_extract)
    register_command_handler(MCPCommandType.TABLE_PROCESS, handle_table_process)
    register_command_handler(MCPCommandType.HTML_PROCESS, handle_html_processor)
    
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
    register_command_handler(MCPCommandType.SEMANTIC_SEARCH, handle_semantic_search)
    
    # Register test handler
    register_command_handler("test", handle_test)
    
    # Add logging to check handlers dict *after* registration
    logger.info(
        f"_command_handlers after initialization: {list(_command_handlers.keys())}",
        component="mcp",
        operation="register_handlers",
        context={
            "handler_count": len(_command_handlers),
        }
    )
    # Log dictionary ID after initialization
    logger.info(f"_command_handlers ID after init: {id(_command_handlers)}", component="mcp", operation="register_handlers")
    
    # Original logging (slightly redundant now but keep for consistency)
    logger.info(
        "Registered MCP command handlers",
        component="mcp",
        operation="register_handlers",
        context={
            "handler_count": len(_command_handlers),
        }
    )


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
    
    try:
        # Validate args into the Pydantic model
        params = JqQueryParams(**args)
        
        # Execute jq query using the validated params object
        result = await jq_query(params)
        
        # Convert to serializable form
        return result.model_dump()
    except Exception as e:
        logger.error(
            f"Error handling JQ query: {str(e)}",
            component="mcp",
            operation="handle_jq_query",
            exception=e,
            context=args
        )
        # Re-raise to be caught by the main request handler
        raise


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
    
    try:
        # Convert dict to SqliteQueryParams
        from tsap.mcp.models import SqliteQueryParams
        params = SqliteQueryParams(**args)
        
        # Execute sqlite query with proper parameters
        result = await sqlite_query(params)
        
        # Convert to serializable form
        return result.model_dump()
    except Exception as e:
        logger.error(
            f"Error handling SQLite query: {str(e)}",
            component="mcp",
            operation="handle_sqlite_query",
            exception=e,
            context=args
        )
        # Re-raise to be caught by the main request handler
        raise


async def handle_pdf_extract(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle pdf_extract command.

    Args:
        args: Command arguments

    Returns:
        Command result
    """
    logger.info(
        "Handling pdf_extract command",
        component="mcp",
        operation="pdf_extract"
    )
    
    try:
        # Import the core extraction function and models
        from tsap.core.pdf_extractor import extract_pdf_text, extract_pdf_metadata # We might need the full PdfExtractor class or its main method
        from tsap.core.pdf_extractor import get_pdf_extractor # Get the registered tool instance
        from .models import PdfExtractParams, PdfExtractResult # Import Pydantic models

        # Validate arguments using the Pydantic model
        params = PdfExtractParams(**args)
        
        # Get the PdfExtractor tool instance
        extractor = get_pdf_extractor()
        
        # Call the extractor's main processing method
        # Assuming the PdfExtractor class has an `extract_text` or similar method
        # that takes PdfExtractParams and returns PdfExtractResult
        # Adjust the method call based on the actual PdfExtractor implementation
        result: PdfExtractResult = await extractor.extract_text(params)
        
        # Convert result to serializable dictionary
        return result.model_dump()
    except Exception as e:
        logger.error(
            f"Error handling PDF Extract query: {str(e)}",
            component="mcp",
            operation="handle_pdf_extract",
            exception=e,
            context=args
        )
        # Re-raise to be caught by the main request handler
        raise


async def handle_table_process(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle table_process command.
    
    Args:
        args: Command arguments matching TableProcessParams.
        
    Returns:
        Command result as a dictionary.
    """
    logger.info(
        "Handling table_process command",
        component="mcp",
        operation="table_process",
        context={"args_keys": list(args.keys())} # Log keys for debugging
    )
    try:
        # Validate/parse args using the Pydantic model
        # This ensures args match the expected structure
        params = TableProcessParams(**args)

        # Get the singleton TableProcessor instance
        processor = get_table_processor()

        # Call the synchronous process method directly
        # No need for run_in_executor if the method is not CPU-bound for long,
        # or if the underlying I/O operations are already async (which they aren't here).
        # If significant CPU work is done (complex transforms), executor might be needed.
        # For now, assume direct call is acceptable for typical cases.
        result: TableProcessResult = processor.process(params)

        # Convert the result Pydantic model to a dictionary for the MCP response
        # Ensure None values are included so client sees expected fields
        return result.model_dump(exclude_none=False)
    except Exception as e:
        logger.error(
            f"Error during table processing: {e}",
            component="mcp",
            operation="table_process",
            exception=e
        )
        # Re-raise the exception so the main handle_request error handler
        # can catch it and create a proper error response.
        raise


# Document Profiling Handler
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
async def handle_semantic_search(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle the semantic_search command."""
    # Example of using a Pydantic model for parsing/validation
    try:
        params = SemanticSearchParams(**args)
        operation = get_operation("semantic_search")
        result = await operation.execute(params)
        return result.dict() # Convert result model to dict
    except Exception as e:
        logger.error(f"Error during semantic search: {e}", exception=e)
        # Re-raise or handle appropriately
        raise # Reraise to be caught by the main handler


# --- Add Handler for HTML Processor --- #
async def handle_html_processor(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle the html_processor command."""
    try:
        # Log the HTML processing request
        logger.info(
            "Handling HTML process command",
            component="mcp",
            operation="html_process"
        )
        
        # Validate/parse args using the Pydantic model from core
        params = HtmlProcessParams(**args)
        # Call the core function
        result = await process_html(params)
        # Convert the result Pydantic model back to a dictionary using model_dump()
        return result.model_dump() 
    except Exception as e:
        logger.error(f"Error during HTML processing: {e}", exception=e)
        # Re-raise the exception to be caught by the main request handler
        raise
# --- End Handler for HTML Processor --- #


async def handle_test(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle the test command (for debugging)."""
    logger.info(
        "Handling test command",
        component="mcp",
        operation="test_handler"
    )
    return {"message": "Test command successful", "received_args": args}


# Register default handlers
initialize_handlers()