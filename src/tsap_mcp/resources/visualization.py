"""
Visualization resources for TSAP MCP Server.

This module provides MCP resource implementations for accessing visualization
capabilities and chart/graph information.
"""
import json
from mcp.server.fastmcp import FastMCP


def register_visualization_resources(mcp: FastMCP) -> None:
    """Register all visualization-related resources with the MCP server.
    
    Args:
        mcp: FastMCP server instance
    """
    
    @mcp.resource("visualization://charts/types")
    async def get_chart_types() -> str:
        """Get information about available chart types.
        
        This resource provides information about various chart types,
        their characteristics, and appropriate use cases.
        
        Returns:
            Chart type information as JSON string
        """
        # Get chart type information from the original implementation
        from tsap.core.visualization import get_chart_types as original_get_chart_types
        
        try:
            chart_types = await original_get_chart_types()
            
            # Format as JSON
            if isinstance(chart_types, dict):
                return json.dumps(chart_types, indent=2)
            elif hasattr(chart_types, "dict"):
                return json.dumps(chart_types.dict(), indent=2)
            else:
                return json.dumps({"error": "Chart types format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to retrieve chart types: {str(e)}"
            }, indent=2)
    
    @mcp.resource("visualization://graphs/layouts")
    async def get_graph_layouts() -> str:
        """Get information about available graph layouts.
        
        This resource provides information about various graph layouts,
        their characteristics, and appropriate use cases.
        
        Returns:
            Graph layout information as JSON string
        """
        # Get graph layout information from the original implementation
        from tsap.core.visualization import get_graph_layouts as original_get_layouts
        
        try:
            layouts = await original_get_layouts()
            
            # Format as JSON
            if isinstance(layouts, dict):
                return json.dumps(layouts, indent=2)
            elif hasattr(layouts, "dict"):
                return json.dumps(layouts.dict(), indent=2)
            else:
                return json.dumps({"error": "Layout format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to retrieve graph layouts: {str(e)}"
            }, indent=2)
    
    @mcp.resource("visualization://code/{language}/options")
    async def get_code_visualization_options(language: str) -> str:
        """Get code visualization options for a specific language.
        
        This resource provides information about visualization options
        for code in a specific programming language.
        
        Args:
            language: Programming language
            
        Returns:
            Code visualization options as JSON string
        """
        # Get code visualization options from the original implementation
        from tsap.core.visualization import get_code_vis_options as original_get_options
        
        try:
            options = await original_get_options(language)
            
            # Format as JSON
            if isinstance(options, dict):
                return json.dumps(options, indent=2)
            elif hasattr(options, "dict"):
                return json.dumps(options.dict(), indent=2)
            else:
                return json.dumps({"error": "Options format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to retrieve code visualization options: {str(e)}",
                "language": language
            }, indent=2)
    
    @mcp.resource("visualization://color-schemes")
    async def get_color_schemes() -> str:
        """Get information about available color schemes.
        
        This resource provides information about various color schemes
        for data visualization, with examples and use cases.
        
        Returns:
            Color scheme information as JSON string
        """
        # Get color scheme information from the original implementation
        from tsap.core.visualization import get_color_schemes as original_get_schemes
        
        try:
            schemes = await original_get_schemes()
            
            # Format as JSON
            if isinstance(schemes, dict):
                return json.dumps(schemes, indent=2)
            elif hasattr(schemes, "dict"):
                return json.dumps(schemes.dict(), indent=2)
            else:
                return json.dumps({"error": "Color schemes format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to retrieve color schemes: {str(e)}"
            }, indent=2)
    
    @mcp.resource("visualization://formats")
    async def get_visualization_formats() -> str:
        """Get information about available visualization formats.
        
        This resource provides information about output formats for
        visualizations, such as image types, interactive formats, etc.
        
        Returns:
            Visualization format information as JSON string
        """
        # Get format information from the original implementation
        from tsap.core.visualization import get_formats as original_get_formats
        
        try:
            formats = await original_get_formats()
            
            # Format as JSON
            if isinstance(formats, dict):
                return json.dumps(formats, indent=2)
            elif hasattr(formats, "dict"):
                return json.dumps(formats.dict(), indent=2)
            else:
                return json.dumps({"error": "Formats not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to retrieve visualization formats: {str(e)}"
            }, indent=2)
    
    @mcp.resource("visualization://templates/{visualization_type}")
    async def get_visualization_templates(visualization_type: str) -> str:
        """Get templates for a specific visualization type.
        
        This resource provides templates and examples for creating
        visualizations of a specific type.
        
        Args:
            visualization_type: Type of visualization (chart, graph, etc.)
            
        Returns:
            Template information as JSON string
        """
        # Get template information from the original implementation
        from tsap.core.visualization import get_templates as original_get_templates
        
        try:
            templates = await original_get_templates(visualization_type)
            
            # Format as JSON
            if isinstance(templates, dict):
                return json.dumps(templates, indent=2)
            elif hasattr(templates, "dict"):
                return json.dumps(templates.dict(), indent=2)
            else:
                return json.dumps({"error": "Templates format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to retrieve visualization templates: {str(e)}",
                "visualization_type": visualization_type
            }, indent=2)
    
    @mcp.resource("visualization://best-practices/{data_type}")
    async def get_visualization_best_practices(data_type: str) -> str:
        """Get visualization best practices for a specific data type.
        
        This resource provides best practices and recommendations for
        visualizing specific types of data.
        
        Args:
            data_type: Type of data (numeric, categorical, temporal, etc.)
            
        Returns:
            Best practice information as JSON string
        """
        # Get best practice information from the original implementation
        from tsap.core.visualization import get_best_practices as original_get_practices
        
        try:
            practices = await original_get_practices(data_type)
            
            # Format as JSON
            if isinstance(practices, dict):
                return json.dumps(practices, indent=2)
            elif hasattr(practices, "dict"):
                return json.dumps(practices.dict(), indent=2)
            else:
                return json.dumps({"error": "Best practices format not recognized"}, indent=2)
                
        except Exception as e:
            return json.dumps({
                "error": f"Failed to retrieve visualization best practices: {str(e)}",
                "data_type": data_type
            }, indent=2) 