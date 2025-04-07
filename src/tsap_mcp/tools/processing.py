"""
Processing tools for TSAP MCP Server.

This module provides MCP tool implementations for various data processing
operations, including text processing, HTML manipulation, PDF extraction,
table handling, and more.
"""
import json
import logging
from typing import Dict, List, Any, Optional, Union

# Import MCP components
from mcp.server.fastmcp import FastMCP, Context

# Import original implementations
# HTML processing
from tsap.core.html_processor import HTMLProcessor, HTMLProcessParams
# SQLite operations
from tsap.core.sqlite import SQLiteProcessor, SQLiteParams
# PDF extraction
from tsap.core.pdf_extractor import PDFExtractor, PDFExtractParams
# Table processing
from tsap.core.table_processor import TableProcessor, TableProcessParams
# JQ JSON querying
from tsap.core.jq import JQProcessor, JQParams
# AWK text processing
from tsap.core.awk import AWKProcessor, AWKParams
# Text processing
from tsap.core.process import TextProcessor, TextProcessParams
# Validation
from tsap.core.validation import Validator, ValidationParams

logger = logging.getLogger("tsap_mcp.tools.processing")


def register_processing_tools(mcp: FastMCP) -> None:
    """Register all processing-related tools with the MCP server.
    
    Args:
        mcp: FastMCP server instance
    """
    
    @mcp.tool()
    async def process_text(
        text: str,
        operation: str = "clean",
        options: Optional[Dict[str, Any]] = None,
        ctx: Optional[Context] = None,
    ) -> str:
        """Process text using various operations.
        
        This tool applies various processing operations to text, such as
        cleaning, normalization, tokenization, etc.
        
        Args:
            text: Text to process
            operation: Processing operation (clean, normalize, tokenize, etc.)
            options: Additional options for the operation
            ctx: MCP context
            
        Returns:
            Processed text
        """
        if ctx:
            ctx.info(f"Processing text with operation: {operation}")
        
        # Use original implementation
        params = TextProcessParams(
            text=text,
            operation=operation,
            options=options or {},
        )
        
        tool = TextProcessor()
        result = await tool.process_text(params)
        
        if hasattr(result, 'text'):
            return result.text
        return str(result)
    
    @mcp.tool()
    async def extract_data(
        text: str,
        pattern: str,
        options: Optional[Dict[str, Any]] = None,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """Extract structured data from text using patterns.
        
        This tool extracts data from text using regular expressions or other
        pattern matching techniques.
        
        Args:
            text: Text to extract data from
            pattern: Pattern to extract (regex or named pattern)
            options: Additional options for extraction
            ctx: MCP context
            
        Returns:
            Extracted data as a dictionary
        """
        if ctx:
            ctx.info(f"Extracting data using pattern: {pattern}")
        
        # Use original implementation
        params = TextProcessParams(
            text=text,
            pattern=pattern,
            options=options or {},
        )
        
        tool = TextProcessor()
        result = await tool.extract_data(params)
        
        if hasattr(result, 'data'):
            return result.data
        return {}
    
    @mcp.tool()
    async def process_html(
        html: str,
        operation: str = "extract_text",
        selector: Optional[str] = None,
        attributes: Optional[List[str]] = None,
        include_links: bool = False,
        clean_text: bool = True,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """Process HTML content with various operations.
        
        This tool processes HTML content, performing operations like
        extracting text, selecting elements, parsing tables, and more.
        
        Args:
            html: HTML content to process
            operation: Operation to perform (extract_text, select, extract_tables, etc.)
            selector: CSS selector for targeting elements
            attributes: HTML attributes to extract (if applicable)
            include_links: Whether to include links in extraction
            clean_text: Whether to clean and normalize extracted text
            ctx: MCP context
            
        Returns:
            Processing results based on the operation
        """
        if ctx:
            ctx.info(f"Processing HTML with operation: {operation}")
        
        # Use original implementation
        params = HTMLProcessParams(
            html=html,
            operation=operation,
            selector=selector,
            attributes=attributes,
            include_links=include_links,
            clean_text=clean_text,
        )
        
        processor = HTMLProcessor()
        result = await processor.process(params)
        
        if hasattr(result, 'result'):
            return result.result
        return {}
    
    @mcp.tool()
    async def query_sqlite(
        db_path: str,
        query: str,
        params: Optional[List[Any]] = None,
        fetch_mode: str = "all",
        include_schema: bool = False,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """Execute a SQLite query and return results.
        
        This tool runs SQL queries against a SQLite database,
        with options for parameterized queries and schema retrieval.
        
        Args:
            db_path: Path to SQLite database file
            query: SQL query to execute
            params: Parameters for parameterized query
            fetch_mode: Result fetch mode (all, one, many, schema)
            include_schema: Whether to include schema information
            ctx: MCP context
            
        Returns:
            Query results with optional schema
        """
        if ctx:
            ctx.info(f"Executing SQLite query: {query}")
        
        # Use original implementation
        params = SQLiteParams(
            db_path=db_path,
            query=query,
            params=params or [],
            fetch_mode=fetch_mode,
            include_schema=include_schema,
        )
        
        processor = SQLiteProcessor()
        result = await processor.query(params)
        
        if hasattr(result, 'results'):
            return result.results
        return {}
    
    @mcp.tool()
    async def extract_pdf(
        pdf_path: str,
        operation: str = "text",
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        password: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """Extract content from a PDF file.
        
        This tool extracts various content from PDF files, including
        text, images, metadata, and structural information.
        
        Args:
            pdf_path: Path to PDF file
            operation: Extraction operation (text, images, metadata, structure)
            start_page: First page to extract (1-indexed)
            end_page: Last page to extract (inclusive)
            password: PDF password if encrypted
            options: Additional operation-specific options
            ctx: MCP context
            
        Returns:
            Extracted PDF content
        """
        if ctx:
            ctx.info(f"Extracting PDF content from {pdf_path} with operation: {operation}")
        
        # Use original implementation
        params = PDFExtractParams(
            pdf_path=pdf_path,
            operation=operation,
            start_page=start_page,
            end_page=end_page,
            password=password,
            options=options or {},
        )
        
        extractor = PDFExtractor()
        result = await extractor.extract(params)
        
        if hasattr(result, 'results'):
            return result.results
        return {}
    
    @mcp.tool()
    async def process_table(
        table_data: Union[str, List[List[Any]], Dict[str, Any]],
        format: str = "auto",
        operation: str = "convert",
        target_format: str = "json",
        options: Optional[Dict[str, Any]] = None,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """Process tabular data with various operations.
        
        This tool processes tables in various formats (CSV, markdown, HTML),
        with operations like conversion, filtering, sorting, and analysis.
        
        Args:
            table_data: Table data in various formats
            format: Input format (csv, markdown, html, json, auto)
            operation: Processing operation (convert, filter, sort, analyze)
            target_format: Target format for conversion
            options: Additional operation-specific options
            ctx: MCP context
            
        Returns:
            Processed table data
        """
        if ctx:
            ctx.info(f"Processing table with operation: {operation}")
        
        # Use original implementation
        params = TableProcessParams(
            table_data=table_data,
            format=format,
            operation=operation,
            target_format=target_format,
            options=options or {},
        )
        
        processor = TableProcessor()
        result = await processor.process(params)
        
        if hasattr(result, 'result'):
            return result.result
        return {}
    
    @mcp.tool()
    async def process_json(
        json_data: Union[str, Dict[str, Any], List[Any]],
        query: str,
        options: Optional[Dict[str, Any]] = None,
        ctx: Optional[Context] = None,
    ) -> Union[str, Dict[str, Any], List[Any]]:
        """Process JSON data with query expressions.
        
        This tool processes JSON data using query expressions
        to extract, filter, and transform the data.
        
        Args:
            json_data: JSON data as string or parsed object
            query: Query expression
            options: Additional processing options
            ctx: MCP context
            
        Returns:
            Processed JSON result
        """
        if ctx:
            ctx.info(f"Processing JSON with query: {query}")
        
        # Use original implementation
        params = JQParams(
            json_data=json_data if isinstance(json_data, str) else json.dumps(json_data),
            query=query,
            options=options or {},
        )
        
        processor = JQProcessor()
        result = await processor.process(params)
        
        if hasattr(result, 'result'):
            return result.result
        return {}
    
    @mcp.tool()
    async def process_awk(
        text: str,
        pattern: str,
        action: Optional[str] = None,
        field_separator: str = " ",
        options: Optional[Dict[str, Any]] = None,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """Process text using AWK-like pattern-action statements.
        
        This tool applies AWK-style pattern matching and actions to text data,
        useful for complex text processing tasks.
        
        Args:
            text: Text to process
            pattern: AWK pattern (condition for applying action)
            action: AWK action to apply when pattern matches
            field_separator: Field separator for line parsing
            options: Additional processing options
            ctx: MCP context
            
        Returns:
            Processing results
        """
        if ctx:
            ctx.info(f"Processing text with AWK pattern: {pattern}")
        
        # Use original implementation
        params = AWKParams(
            text=text,
            pattern=pattern,
            action=action,
            field_separator=field_separator,
            options=options or {},
        )
        
        processor = AWKProcessor()
        result = await processor.process(params)
        
        if hasattr(result, 'result'):
            return result.result
        return {}
    
    @mcp.tool()
    async def validate_data(
        data: Any,
        schema: Dict[str, Any],
        format: str = "json",
        options: Optional[Dict[str, Any]] = None,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """Validate data against a schema.
        
        This tool validates data against various schema formats,
        such as JSON Schema, dataclass definitions, or custom rules.
        
        Args:
            data: Data to validate
            schema: Validation schema
            format: Schema format (json_schema, dataclass, custom)
            options: Additional validation options
            ctx: MCP context
            
        Returns:
            Validation results
        """
        if ctx:
            ctx.info(f"Validating data against schema format: {format}")
        
        # Use original implementation
        params = ValidationParams(
            data=data,
            schema=schema,
            format=format,
            options=options or {},
        )
        
        validator = Validator()
        result = await validator.validate(params)
        
        if hasattr(result, 'result'):
            return result.result
        return {}
    
    @mcp.tool()
    async def validate_format(
        text: str,
        format_type: str,
        options: Optional[Dict[str, Any]] = None,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """Validate text against a specific format.
        
        This tool validates whether text conforms to a specific format,
        such as JSON, YAML, CSV, regex pattern, etc.
        
        Args:
            text: Text to validate
            format_type: Format type to validate against
            options: Additional validation options
            ctx: MCP context
            
        Returns:
            Validation results
        """
        if ctx:
            ctx.info(f"Validating text against format: {format_type}")
        
        # Use original implementation
        params = ValidationParams(
            text=text,
            format_type=format_type,
            options=options or {},
        )
        
        validator = Validator()
        result = await validator.validate_format(params)
        
        if hasattr(result, 'result'):
            return result.result
        return {}
    
    @mcp.tool()
    async def process_jq(
        json_data: Union[str, Dict[str, Any], List[Any]],
        query: str,
        options: Optional[Dict[str, Any]] = None,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """Process JSON data using jq query language.
        
        This tool processes JSON data using the jq query language,
        which provides powerful capabilities for filtering, mapping,
        and transforming JSON data.
        
        Args:
            json_data: JSON data as string or parsed object
            query: jq query expression
            options: Additional processing options
            ctx: MCP context
            
        Returns:
            Processing results
        """
        if ctx:
            ctx.info(f"Processing JSON with jq query: {query}")
        
        # Use original implementation
        if isinstance(json_data, (dict, list)):
            json_data = json.dumps(json_data)
            
        params = JQParams(
            json_data=json_data,
            query=query,
            options=options or {},
        )
        
        processor = JQProcessor()
        result = await processor.process(params)
        
        if hasattr(result, 'result'):
            return result.result
        return {}