"""
Analysis resources for TSAP MCP Server.

This module provides MCP resource implementations for accessing code analysis
capabilities and structure information.
"""
import os
import json
from typing import Dict, Any, List, Optional
from mcp.server.fastmcp import FastMCP


def register_analysis_resources(mcp: FastMCP) -> None:
    """Register all analysis-related resources with the MCP server.
    
    Args:
        mcp: FastMCP server instance
    """
    
    @mcp.resource("analysis://structure/{file_path:path}")
    async def get_file_structure(file_path: str) -> str:
        """Get code structure information for a file.
        
        This resource provides structured information about code organization,
        including classes, functions, and relationships.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Code structure information as JSON string
        """
        # Get file structure from the original implementation
        from tsap.core.structure import get_file_structure as original_get_file_structure
        
        try:
            structure = await original_get_file_structure(file_path)
            
            # Format as JSON
            if isinstance(structure, dict):
                return json.dumps(structure, indent=2)
            elif hasattr(structure, "dict"):
                return json.dumps(structure.dict(), indent=2)
            else:
                return json.dumps({"error": "Structure format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to analyze file structure: {str(e)}",
                "file_path": file_path
            }, indent=2)
    
    @mcp.resource("analysis://dependencies/{file_path:path}")
    async def get_dependencies(file_path: str) -> str:
        """Get dependency information for a file.
        
        This resource provides information about dependencies,
        including imported modules, libraries, and references.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Dependency information as JSON string
        """
        # Get dependencies from the original implementation
        from tsap.core.dependencies import get_dependencies as original_get_dependencies
        
        try:
            dependencies = await original_get_dependencies(file_path)
            
            # Format as JSON
            if isinstance(dependencies, dict):
                return json.dumps(dependencies, indent=2)
            elif hasattr(dependencies, "dict"):
                return json.dumps(dependencies.dict(), indent=2)
            else:
                return json.dumps({"error": "Dependencies format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to retrieve dependencies: {str(e)}",
                "file_path": file_path
            }, indent=2)
    
    @mcp.resource("analysis://metrics/{file_path:path}")
    async def get_code_metrics(file_path: str) -> str:
        """Get code quality metrics for a file.
        
        This resource provides various code quality metrics,
        such as complexity, maintainability, and documentation levels.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Code metrics as JSON string
        """
        # Get code metrics from the original implementation
        from tsap.core.metrics import get_code_metrics as original_get_code_metrics
        
        try:
            metrics = await original_get_code_metrics(file_path)
            
            # Format as JSON
            if isinstance(metrics, dict):
                return json.dumps(metrics, indent=2)
            elif hasattr(metrics, "dict"):
                return json.dumps(metrics.dict(), indent=2)
            else:
                return json.dumps({"error": "Metrics format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to calculate code metrics: {str(e)}",
                "file_path": file_path
            }, indent=2)
    
    @mcp.resource("analysis://security/{file_path:path}")
    async def get_security_report(file_path: str) -> str:
        """Get security analysis for a file.
        
        This resource provides security analysis results,
        including potential vulnerabilities and best practice violations.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Security report as JSON string
        """
        # Get security report from the original implementation
        from tsap.core.security import get_security_report as original_get_security_report
        
        try:
            security_report = await original_get_security_report(file_path)
            
            # Format as JSON
            if isinstance(security_report, dict):
                return json.dumps(security_report, indent=2)
            elif hasattr(security_report, "dict"):
                return json.dumps(security_report.dict(), indent=2)
            else:
                return json.dumps({"error": "Security report format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to generate security report: {str(e)}",
                "file_path": file_path
            }, indent=2)
    
    @mcp.resource("analysis://docstrings/{file_path:path}")
    async def get_docstring_coverage(file_path: str) -> str:
        """Get docstring coverage information for a file.
        
        This resource provides information about docstring coverage,
        quality, and completeness.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Docstring coverage information as JSON string
        """
        # Get docstring coverage from the original implementation
        from tsap.core.documentation import get_docstring_coverage as original_get_docstring_coverage
        
        try:
            coverage = await original_get_docstring_coverage(file_path)
            
            # Format as JSON
            if isinstance(coverage, dict):
                return json.dumps(coverage, indent=2)
            elif hasattr(coverage, "dict"):
                return json.dumps(coverage.dict(), indent=2)
            else:
                return json.dumps({"error": "Coverage format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to analyze docstring coverage: {str(e)}",
                "file_path": file_path
            }, indent=2)
    
    @mcp.resource("analysis://class/{file_path:path}/{class_name}")
    async def get_class_info(file_path: str, class_name: str) -> str:
        """Get detailed information about a specific class.
        
        This resource provides detailed information about a class,
        including methods, attributes, relationships, and documentation.
        
        Args:
            file_path: Path to the file containing the class
            class_name: Name of the class to analyze
            
        Returns:
            Class information as JSON string
        """
        # Get class info from the original implementation
        from tsap.core.structure import get_class_info as original_get_class_info
        
        try:
            class_info = await original_get_class_info(file_path, class_name)
            
            # Format as JSON
            if isinstance(class_info, dict):
                return json.dumps(class_info, indent=2)
            elif hasattr(class_info, "dict"):
                return json.dumps(class_info.dict(), indent=2)
            else:
                return json.dumps({"error": "Class info format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to retrieve class info: {str(e)}",
                "file_path": file_path,
                "class_name": class_name
            }, indent=2)
    
    @mcp.resource("analysis://function/{file_path:path}/{function_name}")
    async def get_function_info(file_path: str, function_name: str) -> str:
        """Get detailed information about a specific function.
        
        This resource provides detailed information about a function,
        including parameters, return values, complexity, and documentation.
        
        Args:
            file_path: Path to the file containing the function
            function_name: Name of the function to analyze
            
        Returns:
            Function information as JSON string
        """
        # Get function info from the original implementation
        from tsap.core.structure import get_function_info as original_get_function_info
        
        try:
            function_info = await original_get_function_info(file_path, function_name)
            
            # Format as JSON
            if isinstance(function_info, dict):
                return json.dumps(function_info, indent=2)
            elif hasattr(function_info, "dict"):
                return json.dumps(function_info.dict(), indent=2)
            else:
                return json.dumps({"error": "Function info format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to retrieve function info: {str(e)}",
                "file_path": file_path,
                "function_name": function_name
            }, indent=2)
    
    @mcp.resource("analysis://call-graph/{file_path:path}")
    async def get_call_graph(file_path: str) -> str:
        """Get call graph for a file.
        
        This resource provides a call graph showing function calls
        and dependencies within the file.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Call graph information as JSON string
        """
        # Get call graph from the original implementation
        from tsap.core.structure import get_call_graph as original_get_call_graph
        
        try:
            call_graph = await original_get_call_graph(file_path)
            
            # Format as JSON
            if isinstance(call_graph, dict):
                return json.dumps(call_graph, indent=2)
            elif hasattr(call_graph, "dict"):
                return json.dumps(call_graph.dict(), indent=2)
            else:
                return json.dumps({"error": "Call graph format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to generate call graph: {str(e)}",
                "file_path": file_path
            }, indent=2)
    
    @mcp.resource("analysis://complexity/{directory_path:path}")
    async def get_directory_complexity(directory_path: str) -> str:
        """Get complexity analysis for a directory.
        
        This resource provides aggregated complexity metrics
        for all files in a directory.
        
        Args:
            directory_path: Path to the directory to analyze
            
        Returns:
            Directory complexity information as JSON string
        """
        # Get directory complexity from the original implementation
        from tsap.core.metrics import get_directory_complexity as original_get_directory_complexity
        
        try:
            complexity = await original_get_directory_complexity(directory_path)
            
            # Format as JSON
            if isinstance(complexity, dict):
                return json.dumps(complexity, indent=2)
            elif hasattr(complexity, "dict"):
                return json.dumps(complexity.dict(), indent=2)
            else:
                return json.dumps({"error": "Complexity format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to analyze directory complexity: {str(e)}",
                "directory_path": directory_path
            }, indent=2) 