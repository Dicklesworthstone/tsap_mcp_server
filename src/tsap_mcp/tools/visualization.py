"""
Visualization tools for TSAP MCP Server.

This module provides MCP tool implementations for various visualization functions,
including chart generation, graph visualization, and data plotting.
"""
import os
import base64
import logging
from io import BytesIO
from typing import Dict, List, Any, Optional, Union, Tuple

from mcp.server.fastmcp import FastMCP, Context

# Import original implementations
# Import visualization tools from original implementation
from tsap.core.visualization import ChartGenerator, ChartParams
from tsap.core.visualization import GraphVisualizer, GraphParams
from tsap.core.visualization import CodeVisualizer, CodeVisParams

logger = logging.getLogger("tsap_mcp.tools.visualization")


def register_visualization_tools(mcp: FastMCP) -> None:
    """Register all visualization-related tools with the MCP server.
    
    Args:
        mcp: FastMCP server instance
    """
    
    @mcp.tool()
    async def generate_chart(
        data: Union[str, List[Dict[str, Any]]],
        chart_type: str = "bar",
        title: str = "",
        x_label: str = "",
        y_label: str = "",
        format: str = "png",
        options: Optional[Dict[str, Any]] = None,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """Generate a chart visualization from data.
        
        This tool creates chart visualizations from data in various formats,
        including bar charts, line charts, scatter plots, and more.
        
        Args:
            data: Data to visualize (JSON string or list of dictionaries)
            chart_type: Type of chart (bar, line, scatter, pie, etc.)
            title: Chart title
            x_label: X-axis label
            y_label: Y-axis label
            format: Output format (png, svg, base64)
            options: Additional options for chart generation
            ctx: MCP context
            
        Returns:
            Generated chart data, with base64-encoded image and metadata
        """
        if ctx:
            ctx.info(f"Generating {chart_type} chart visualization")
        
        # Use original implementation
        params = ChartParams(
            data=data,
            chart_type=chart_type,
            title=title,
            x_label=x_label,
            y_label=y_label,
            format=format,
            options=options or {},
        )
        
        generator = ChartGenerator()
        result = await generator.generate_chart(params)
        
        if hasattr(result, 'chart_data'):
            return result.chart_data
        return {}
    
    @mcp.tool()
    async def visualize_graph(
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        layout: str = "force",
        format: str = "png",
        options: Optional[Dict[str, Any]] = None,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """Visualize graph structure with nodes and edges.
        
        This tool creates visualizations of graph structures, such as
        dependency graphs, network diagrams, hierarchies, etc.
        
        Args:
            nodes: List of node definitions with IDs and attributes
            edges: List of edge definitions connecting nodes
            layout: Graph layout algorithm (force, circular, hierarchical)
            format: Output format (png, svg, base64)
            options: Additional options for graph visualization
            ctx: MCP context
            
        Returns:
            Generated graph visualization data
        """
        if ctx:
            ctx.info(f"Generating graph visualization with {layout} layout")
        
        # Use original implementation
        params = GraphParams(
            nodes=nodes,
            edges=edges,
            layout=layout,
            format=format,
            options=options or {},
        )
        
        visualizer = GraphVisualizer()
        result = await visualizer.visualize(params)
        
        if hasattr(result, 'graph_data'):
            return result.graph_data
        return {}
    
    @mcp.tool()
    async def visualize_code_structure(
        code: str,
        language: str,
        view_type: str = "class",
        format: str = "png",
        options: Optional[Dict[str, Any]] = None,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """Visualize code structure as diagrams.
        
        This tool creates visual representations of code structure, such as
        class diagrams, dependency graphs, call graphs, etc.
        
        Args:
            code: Source code to visualize
            language: Programming language of the code
            view_type: Type of visualization (class, module, dependency, call)
            format: Output format (png, svg, base64)
            options: Additional options for visualization
            ctx: MCP context
            
        Returns:
            Generated code visualization
        """
        if ctx:
            ctx.info(f"Visualizing {language} code structure as {view_type} diagram")
        
        # Use original implementation
        params = CodeVisParams(
            code=code,
            language=language,
            view_type=view_type,
            format=format,
            options=options or {},
        )
        
        visualizer = CodeVisualizer()
        result = await visualizer.visualize(params)
        
        if hasattr(result, 'visualization'):
            return result.visualization
        return {}


# Helper functions for visualization

def _create_bar_chart(
    data: List[Dict[str, Any]], 
    title: str = "", 
    x_label: str = "", 
    y_label: str = "", 
    options: Dict[str, Any] = None
) -> Tuple[Any, Any]:
    """Create a bar chart using matplotlib.
    
    Args:
        data: Chart data
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        options: Chart options
        
    Returns:
        Matplotlib figure and axes objects
    """
    import matplotlib.pyplot as plt
    
    options = options or {}
    
    # Extract data
    x_values = [item.get("x", item.get("label", i)) for i, item in enumerate(data)]
    y_values = [item.get("y", item.get("value")) for item in data]
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=options.get("figsize", (10, 6)))
    
    # Create bars
    bars = ax.bar(
        x_values, 
        y_values, 
        color=options.get("color", "skyblue"),
        edgecolor=options.get("edgecolor", "black"),
        alpha=options.get("alpha", 0.7),
        width=options.get("width", 0.8)
    )
    
    # Add data labels if requested
    if options.get("data_labels", False):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., 
                height * 1.01,
                f'{height}', 
                ha='center', 
                va='bottom'
            )
    
    # Set title and labels
    if title:
        ax.set_title(title)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    
    # Set grid
    if options.get("grid", True):
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Rotate x-axis labels if needed
    if options.get("rotate_xlabels", False):
        plt.xticks(rotation=options.get("rotation", 45))
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax 