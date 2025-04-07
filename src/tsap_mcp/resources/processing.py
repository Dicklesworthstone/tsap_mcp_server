"""
Processing resources for TSAP MCP Server.

This module provides MCP resource implementations for accessing data processing
capabilities and transformation information.
"""
import os
import json
from typing import Dict, Any, List, Optional
from mcp.server.fastmcp import FastMCP


def register_processing_resources(mcp: FastMCP) -> None:
    """Register all processing-related resources with the MCP server.
    
    Args:
        mcp: FastMCP server instance
    """
    
    @mcp.resource("processing://text/{operation}")
    async def get_text_operations(operation: str) -> str:
        """Get information about a text processing operation.
        
        This resource provides information about available text processing
        operations, their parameters, and usage examples.
        
        Args:
            operation: Text processing operation to get information about
            
        Returns:
            Operation information as JSON string
        """
        # Get operation info from the original implementation
        from tsap.core.process import get_text_operation_info as original_get_info
        
        try:
            operation_info = await original_get_info(operation)
            
            # Format as JSON
            if isinstance(operation_info, dict):
                return json.dumps(operation_info, indent=2)
            elif hasattr(operation_info, "dict"):
                return json.dumps(operation_info.dict(), indent=2)
            else:
                return json.dumps({"error": "Operation info format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to retrieve operation info: {str(e)}",
                "operation": operation
            }, indent=2)
    
    @mcp.resource("processing://html/selectors")
    async def get_html_selectors() -> str:
        """Get information about available HTML selectors.
        
        This resource provides a list of commonly used CSS selectors
        for HTML processing, with examples and descriptions.
        
        Returns:
            HTML selector information as JSON string
        """
        # Get selector info from the original implementation
        from tsap.core.html_processor import get_selector_info as original_get_selector_info
        
        try:
            selector_info = await original_get_selector_info()
            
            # Format as JSON
            if isinstance(selector_info, dict):
                return json.dumps(selector_info, indent=2)
            elif hasattr(selector_info, "dict"):
                return json.dumps(selector_info.dict(), indent=2)
            else:
                return json.dumps({"error": "Selector info format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to retrieve selector info: {str(e)}"
            }, indent=2)
    
    @mcp.resource("processing://sqlite/{db_path:path}/schema")
    async def get_sqlite_schema(db_path: str) -> str:
        """Get schema information for a SQLite database.
        
        This resource provides database schema information,
        including tables, columns, and relationships.
        
        Args:
            db_path: Path to the SQLite database
            
        Returns:
            Schema information as JSON string
        """
        # Get schema from the original implementation
        from tsap.core.sqlite import get_schema as original_get_schema
        
        try:
            schema = await original_get_schema(db_path)
            
            # Format as JSON
            if isinstance(schema, dict):
                return json.dumps(schema, indent=2)
            elif hasattr(schema, "dict"):
                return json.dumps(schema.dict(), indent=2)
            else:
                return json.dumps({"error": "Schema format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to retrieve database schema: {str(e)}",
                "db_path": db_path
            }, indent=2)
    
    @mcp.resource("processing://pdf/{pdf_path:path}/metadata")
    async def get_pdf_metadata(pdf_path: str) -> str:
        """Get metadata for a PDF file.
        
        This resource provides metadata information for a PDF file,
        such as author, creation date, page count, etc.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            PDF metadata as JSON string
        """
        # Get metadata from the original implementation
        from tsap.core.pdf_extractor import get_metadata as original_get_metadata
        
        try:
            metadata = await original_get_metadata(pdf_path)
            
            # Format as JSON
            if isinstance(metadata, dict):
                return json.dumps(metadata, indent=2)
            elif hasattr(metadata, "dict"):
                return json.dumps(metadata.dict(), indent=2)
            else:
                return json.dumps({"error": "Metadata format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to retrieve PDF metadata: {str(e)}",
                "pdf_path": pdf_path
            }, indent=2)
    
    @mcp.resource("processing://table/formats")
    async def get_table_formats() -> str:
        """Get information about supported table formats.
        
        This resource provides information about the table formats
        that can be processed, including their characteristics.
        
        Returns:
            Table format information as JSON string
        """
        # Get format info from the original implementation
        from tsap.core.table_processor import get_format_info as original_get_format_info
        
        try:
            format_info = await original_get_format_info()
            
            # Format as JSON
            if isinstance(format_info, dict):
                return json.dumps(format_info, indent=2)
            elif hasattr(format_info, "dict"):
                return json.dumps(format_info.dict(), indent=2)
            else:
                return json.dumps({"error": "Format info not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to retrieve table format info: {str(e)}"
            }, indent=2)
    
    @mcp.resource("processing://jq/operators")
    async def get_jq_operators() -> str:
        """Get information about JQ query operators.
        
        This resource provides information about JQ operators
        for JSON processing, with examples and explanations.
        
        Returns:
            JQ operator information as JSON string
        """
        # Get operator info from the original implementation
        from tsap.core.jq import get_operator_info as original_get_operator_info
        
        try:
            operator_info = await original_get_operator_info()
            
            # Format as JSON
            if isinstance(operator_info, dict):
                return json.dumps(operator_info, indent=2)
            elif hasattr(operator_info, "dict"):
                return json.dumps(operator_info.dict(), indent=2)
            else:
                return json.dumps({"error": "Operator info not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to retrieve JQ operator info: {str(e)}"
            }, indent=2)
    
    @mcp.resource("processing://awk/patterns")
    async def get_awk_patterns() -> str:
        """Get information about AWK patterns.
        
        This resource provides information about common AWK patterns
        for text processing, with examples and explanations.
        
        Returns:
            AWK pattern information as JSON string
        """
        # Get pattern info from the original implementation
        from tsap.core.awk import get_pattern_info as original_get_pattern_info
        
        try:
            pattern_info = await original_get_pattern_info()
            
            # Format as JSON
            if isinstance(pattern_info, dict):
                return json.dumps(pattern_info, indent=2)
            elif hasattr(pattern_info, "dict"):
                return json.dumps(pattern_info.dict(), indent=2)
            else:
                return json.dumps({"error": "Pattern info not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to retrieve AWK pattern info: {str(e)}"
            }, indent=2)
    
    @mcp.resource("processing://validation/schemas/{format}")
    async def get_validation_schemas(format: str) -> str:
        """Get validation schemas for a specific format.
        
        This resource provides common validation schemas for
        various data formats.
        
        Args:
            format: Data format (json, csv, etc.)
            
        Returns:
            Validation schema information as JSON string
        """
        # Get schema info from the original implementation
        from tsap.core.validation import get_schemas as original_get_schemas
        
        try:
            schemas = await original_get_schemas(format)
            
            # Format as JSON
            if isinstance(schemas, dict):
                return json.dumps(schemas, indent=2)
            elif hasattr(schemas, "dict"):
                return json.dumps(schemas.dict(), indent=2)
            else:
                return json.dumps({"error": "Schema info not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to retrieve validation schemas: {str(e)}",
                "format": format
            }, indent=2) 