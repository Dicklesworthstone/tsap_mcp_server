"""
TSAP Visualization Module.

This module provides functionality for generating visualizations from TSAP results,
including charts, graphs, and other data visualizations.
"""

import os
import re
import io
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Set

# Import optional visualization libraries with fallbacks
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from tsap.utils.logging import logger

# Visualization types
class VisualizationType(str, Enum):
    """Types of visualizations supported."""
    
    BAR_CHART = "bar_chart"
    LINE_CHART = "line_chart"
    PIE_CHART = "pie_chart"
    SCATTER_PLOT = "scatter_plot"
    HEATMAP = "heatmap"
    GRAPH = "graph"
    TREE = "tree"
    BOX_PLOT = "box_plot"
    HISTOGRAM = "histogram"
    VENN_DIAGRAM = "venn_diagram"
    SANKEY_DIAGRAM = "sankey_diagram"


class VisualizationFormat(str, Enum):
    """Output formats for visualizations."""
    
    PNG = "png"
    SVG = "svg"
    PDF = "pdf"
    HTML = "html"
    JSON = "json"


class Visualization:
    """Base class for all visualizations."""
    
    def __init__(
        self,
        title: str,
        data: Any,
        width: Optional[int] = None,
        height: Optional[int] = None,
        theme: str = "default",
    ):
        """Initialize a visualization.
        
        Args:
            title: Visualization title
            data: Data to visualize
            width: Width in pixels
            height: Height in pixels
            theme: Visualization theme
        """
        self.title = title
        self.data = data
        self.width = width
        self.height = height
        self.theme = theme
        self.figure: Optional[Any] = None
    
    def render(self) -> Any:
        """Render the visualization.
        
        Returns:
            Rendered visualization
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement render()")
    
    def save(self, path: str, format: Optional[Union[str, VisualizationFormat]] = None) -> str:
        """Save the visualization to a file.
        
        Args:
            path: Output file path
            format: Output format (inferred from path if not specified)
            
        Returns:
            Path to the saved file
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement save()")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the visualization to a dictionary.
        
        Returns:
            Dictionary representation of the visualization
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement to_dict()")


class MatplotlibVisualization(Visualization):
    """Base class for Matplotlib-based visualizations."""
    
    def __init__(
        self,
        title: str,
        data: Any,
        width: Optional[int] = None,
        height: Optional[int] = None,
        theme: str = "default",
        dpi: int = 100,
    ):
        """Initialize a Matplotlib visualization.
        
        Args:
            title: Visualization title
            data: Data to visualize
            width: Width in inches
            height: Height in inches
            theme: Matplotlib style
            dpi: Resolution in dots per inch
        """
        super().__init__(title, data, width, height, theme)
        
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib is required for this visualization")
        
        self.dpi = dpi
        
        # Set theme
        if theme != "default":
            try:
                plt.style.use(theme)
            except Exception as e:
                logger.warning(f"Failed to set Matplotlib style '{theme}': {e}")
        
        # Create figure
        figsize = None
        if width is not None and height is not None:
            # Convert pixels to inches
            figsize = (width / self.dpi, height / self.dpi)
        
        self.figure = plt.figure(figsize=figsize, dpi=self.dpi)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title(title)
    
    def render(self) -> Figure:
        """Render the visualization.
        
        Returns:
            Matplotlib figure
        """
        if self.figure is None:
            raise RuntimeError("Visualization has not been created")
        
        return self.figure
    
    def save(self, path: str, format: Optional[Union[str, VisualizationFormat]] = None) -> str:
        """Save the visualization to a file.
        
        Args:
            path: Output file path
            format: Output format (inferred from path if not specified)
            
        Returns:
            Path to the saved file
        """
        if self.figure is None:
            raise RuntimeError("Visualization has not been created")
        
        # Infer format from path if not specified
        if format is None:
            ext = os.path.splitext(path)[1][1:].lower()
            if ext in [f.value for f in VisualizationFormat]:
                format = ext
            else:
                format = VisualizationFormat.PNG.value
        
        # Ensure format is a string
        if isinstance(format, VisualizationFormat):
            format = format.value
        
        # Save the figure
        self.figure.savefig(path, format=format, dpi=self.dpi, bbox_inches="tight")
        
        return path
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the visualization to a dictionary.
        
        Returns:
            Dictionary representation of the visualization
        """
        return {
            "type": "matplotlib",
            "title": self.title,
            "theme": self.theme,
            "width": self.width,
            "height": self.height,
            "dpi": self.dpi,
        }
    
    def to_base64(self, format: Union[str, VisualizationFormat] = VisualizationFormat.PNG) -> str:
        """Convert the visualization to a base64 string.
        
        Args:
            format: Output format
            
        Returns:
            Base64-encoded string
        """
        import base64
        
        if self.figure is None:
            raise RuntimeError("Visualization has not been created")
        
        # Ensure format is a string
        if isinstance(format, VisualizationFormat):
            format = format.value
        
        # Save to a buffer
        buffer = io.BytesIO()
        self.figure.savefig(buffer, format=format, dpi=self.dpi, bbox_inches="tight")
        buffer.seek(0)
        
        # Convert to base64
        return base64.b64encode(buffer.read()).decode("utf-8")


class BarChart(MatplotlibVisualization):
    """Bar chart visualization."""
    
    def __init__(
        self,
        title: str,
        data: Dict[str, Union[int, float]],
        width: Optional[int] = None,
        height: Optional[int] = None,
        theme: str = "default",
        dpi: int = 100,
        orientation: str = "vertical",
        sort_values: bool = False,
        limit: Optional[int] = None,
        color: Optional[Union[str, List[str]]] = None,
    ):
        """Initialize a bar chart.
        
        Args:
            title: Chart title
            data: Dictionary of {category: value}
            width: Width in pixels
            height: Height in pixels
            theme: Matplotlib style
            dpi: Resolution in dots per inch
            orientation: Bar orientation ('vertical' or 'horizontal')
            sort_values: Whether to sort bars by value
            limit: Maximum number of bars to show
            color: Bar color(s)
        """
        super().__init__(title, data, width, height, theme, dpi)
        
        self.orientation = orientation
        self.sort_values = sort_values
        self.limit = limit
        self.color = color
        
        # Create the bar chart
        self._create_bar_chart()
    
    def _create_bar_chart(self) -> None:
        """Create the bar chart."""
        # Extract categories and values
        categories = list(self.data.keys())
        values = list(self.data.values())
        
        # Sort if requested
        if self.sort_values:
            sorted_data = sorted(
                zip(categories, values),
                key=lambda x: x[1],
                reverse=True,
            )
            categories, values = zip(*sorted_data)
        
        # Limit if requested
        if self.limit is not None and len(categories) > self.limit:
            categories = categories[:self.limit]
            values = values[:self.limit]
        
        # Create the bar chart
        if self.orientation == "horizontal":
            self.ax.barh(categories, values, color=self.color)
            self.ax.set_xlabel("Value")
            self.ax.set_ylabel("Category")
        else:
            self.ax.bar(categories, values, color=self.color)
            self.ax.set_xlabel("Category")
            self.ax.set_ylabel("Value")
            
            # Rotate x-axis labels if there are many categories
            if len(categories) > 5:
                plt.xticks(rotation=45, ha="right")
        
        # Adjust layout
        self.figure.tight_layout()


class LineChart(MatplotlibVisualization):
    """Line chart visualization."""
    
    def __init__(
        self,
        title: str,
        data: Union[Dict[str, List[Union[int, float]]], Dict[str, Dict[Any, Union[int, float]]]],
        x_labels: Optional[List[Any]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        theme: str = "default",
        dpi: int = 100,
        marker: Optional[str] = None,
        line_style: Optional[str] = None,
        include_points: bool = True,
    ):
        """Initialize a line chart.
        
        Args:
            title: Chart title
            data: Dictionary of {series_name: values} or {series_name: {x: y}}
            x_labels: Labels for x-axis points
            width: Width in pixels
            height: Height in pixels
            theme: Matplotlib style
            dpi: Resolution in dots per inch
            marker: Point marker style
            line_style: Line style
            include_points: Whether to include data points
        """
        super().__init__(title, data, width, height, theme, dpi)
        
        self.x_labels = x_labels
        self.marker = marker
        self.line_style = line_style
        self.include_points = include_points
        
        # Create the line chart
        self._create_line_chart()
    
    def _create_line_chart(self) -> None:
        """Create the line chart."""
        # Check data format
        first_value = next(iter(self.data.values()))
        
        if isinstance(first_value, dict):
            # Handle {series_name: {x: y}} format
            for series_name, series_data in self.data.items():
                x_vals = list(series_data.keys())
                y_vals = list(series_data.values())
                
                # Sort by x values
                sorted_data = sorted(zip(x_vals, y_vals))
                x_vals, y_vals = zip(*sorted_data) if sorted_data else ([], [])
                
                self.ax.plot(
                    x_vals,
                    y_vals,
                    label=series_name,
                    marker=self.marker if self.include_points else None,
                    linestyle=self.line_style,
                )
        else:
            # Handle {series_name: [y1, y2, ...]} format
            for series_name, y_vals in self.data.items():
                if self.x_labels:
                    x_vals = self.x_labels
                else:
                    x_vals = list(range(len(y_vals)))
                
                self.ax.plot(
                    x_vals,
                    y_vals,
                    label=series_name,
                    marker=self.marker if self.include_points else None,
                    linestyle=self.line_style,
                )
        
        # Add legend if multiple series
        if len(self.data) > 1:
            self.ax.legend()
        
        # Set x-axis labels
        if self.x_labels:
            if len(self.x_labels) > 10:
                # If many labels, show every nth label
                n = len(self.x_labels) // 10 + 1
                self.ax.set_xticks(self.x_labels[::n])
                self.ax.set_xticklabels(self.x_labels[::n])
            else:
                self.ax.set_xticks(self.x_labels)
                self.ax.set_xticklabels(self.x_labels)
        
        # Add grid
        self.ax.grid(True, linestyle="--", alpha=0.7)
        
        # Set labels
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        
        # Adjust layout
        self.figure.tight_layout()


class PieChart(MatplotlibVisualization):
    """Pie chart visualization."""
    
    def __init__(
        self,
        title: str,
        data: Dict[str, Union[int, float]],
        width: Optional[int] = None,
        height: Optional[int] = None,
        theme: str = "default",
        dpi: int = 100,
        colors: Optional[List[str]] = None,
        explode: Optional[Dict[str, float]] = None,
        show_labels: bool = True,
        show_percentages: bool = True,
        start_angle: float = 0,
    ):
        """Initialize a pie chart.
        
        Args:
            title: Chart title
            data: Dictionary of {category: value}
            width: Width in pixels
            height: Height in pixels
            theme: Matplotlib style
            dpi: Resolution in dots per inch
            colors: List of colors for pie slices
            explode: Dictionary of {category: explode_amount}
            show_labels: Whether to show category labels
            show_percentages: Whether to show percentages
            start_angle: Starting angle in degrees
        """
        super().__init__(title, data, width, height, theme, dpi)
        
        self.colors = colors
        self.explode = explode
        self.show_labels = show_labels
        self.show_percentages = show_percentages
        self.start_angle = start_angle
        
        # Create the pie chart
        self._create_pie_chart()
    
    def _create_pie_chart(self) -> None:
        """Create the pie chart."""
        # Extract categories and values
        categories = list(self.data.keys())
        values = list(self.data.values())
        
        # Prepare explode values
        explode_values = None
        if self.explode:
            explode_values = [self.explode.get(cat, 0) for cat in categories]
        
        # Prepare labels
        if self.show_labels:
            if self.show_percentages:
                # Both labels and percentages
                labels = categories
                autopct = "%1.1f%%"
            else:
                # Only labels
                labels = categories
                autopct = None
        else:
            if self.show_percentages:
                # Only percentages
                labels = None
                autopct = "%1.1f%%"
            else:
                # No labels
                labels = None
                autopct = None
        
        # Create the pie chart
        self.ax.pie(
            values,
            labels=labels,
            autopct=autopct,
            explode=explode_values,
            colors=self.colors,
            shadow=True,
            startangle=self.start_angle,
        )
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        self.ax.axis("equal")
        
        # Adjust layout
        self.figure.tight_layout()


class HeatMap(MatplotlibVisualization):
    """Heat map visualization."""
    
    def __init__(
        self,
        title: str,
        data: List[List[Union[int, float]]],
        x_labels: Optional[List[str]] = None,
        y_labels: Optional[List[str]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        theme: str = "default",
        dpi: int = 100,
        colormap: str = "viridis",
        show_values: bool = True,
    ):
        """Initialize a heat map.
        
        Args:
            title: Chart title
            data: 2D array of values
            x_labels: Labels for x-axis
            y_labels: Labels for y-axis
            width: Width in pixels
            height: Height in pixels
            theme: Matplotlib style
            dpi: Resolution in dots per inch
            colormap: Matplotlib colormap name
            show_values: Whether to show values in cells
        """
        super().__init__(title, data, width, height, theme, dpi)
        
        self.x_labels = x_labels
        self.y_labels = y_labels
        self.colormap = colormap
        self.show_values = show_values
        
        # Create the heat map
        self._create_heatmap()
    
    def _create_heatmap(self) -> None:
        """Create the heat map."""
        import numpy as np
        
        # Convert data to numpy array
        data_array = np.array(self.data)
        
        # Create the heat map
        im = self.ax.imshow(data_array, cmap=self.colormap)
        
        # Add colorbar
        cbar = self.figure.colorbar(im, ax=self.ax)
        cbar.ax.set_ylabel("Value")
        
        # Set axis labels
        if self.x_labels:
            self.ax.set_xticks(np.arange(len(self.x_labels)))
            self.ax.set_xticklabels(self.x_labels)
            plt.setp(self.ax.get_xticklabels(), rotation=45, ha="right")
        
        if self.y_labels:
            self.ax.set_yticks(np.arange(len(self.y_labels)))
            self.ax.set_yticklabels(self.y_labels)
        
        # Show values in cells
        if self.show_values:
            # For each cell
            for i in range(data_array.shape[0]):
                for j in range(data_array.shape[1]):
                    value = data_array[i, j]
                    
                    # Determine text color based on background
                    threshold = (data_array.max() + data_array.min()) / 2
                    text_color = "white" if value > threshold else "black"
                    
                    # Add text
                    self.ax.text(
                        j, i, f"{value:.1f}",
                        ha="center", va="center",
                        color=text_color
                    )
        
        # Set gridlines
        self.ax.set_xticks(np.arange(data_array.shape[1] + 1) - 0.5, minor=True)
        self.ax.set_yticks(np.arange(data_array.shape[0] + 1) - 0.5, minor=True)
        self.ax.grid(which="minor", color="w", linestyle="-", linewidth=2)
        self.ax.tick_params(which="minor", bottom=False, left=False)
        
        # Adjust layout
        self.figure.tight_layout()


class NetworkGraph(Visualization):
    """Network graph visualization."""
    
    def __init__(
        self,
        title: str,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        width: Optional[int] = None,
        height: Optional[int] = None,
        theme: str = "default",
        layout: str = "spring",
        node_size_field: Optional[str] = None,
        node_color_field: Optional[str] = None,
        edge_width_field: Optional[str] = None,
        edge_color_field: Optional[str] = None,
        node_label_field: Optional[str] = "id",
    ):
        """Initialize a network graph.
        
        Args:
            title: Graph title
            nodes: List of node dictionaries
            edges: List of edge dictionaries
            width: Width in pixels
            height: Height in pixels
            theme: Visualization theme
            layout: Graph layout algorithm
            node_size_field: Node field to use for size
            node_color_field: Node field to use for color
            edge_width_field: Edge field to use for width
            edge_color_field: Edge field to use for color
            node_label_field: Node field to use for labels
        """
        super().__init__(title, {"nodes": nodes, "edges": edges}, width, height, theme)
        
        if not HAS_NETWORKX:
            raise ImportError("NetworkX is required for this visualization")
        
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib is required for this visualization")
        
        self.layout = layout
        self.node_size_field = node_size_field
        self.node_color_field = node_color_field
        self.edge_width_field = edge_width_field
        self.edge_color_field = edge_color_field
        self.node_label_field = node_label_field
        
        # Create the graph
        self._create_graph()
    
    def _create_graph(self) -> None:
        """Create the network graph."""
        # Create a new graph
        self.graph = nx.Graph()
        
        # Add nodes
        for node in self.data["nodes"]:
            self.graph.add_node(node.get("id"), **node)
        
        # Add edges
        for edge in self.data["edges"]:
            self.graph.add_edge(
                edge.get("source"),
                edge.get("target"),
                **edge
            )
        
        # Create figure
        figsize = None
        if self.width is not None and self.height is not None:
            # Convert pixels to inches
            dpi = 100
            figsize = (self.width / dpi, self.height / dpi)
        
        self.figure = plt.figure(figsize=figsize)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title(self.title)
        
        # Compute layout
        if self.layout == "spring":
            self.pos = nx.spring_layout(self.graph)
        elif self.layout == "circular":
            self.pos = nx.circular_layout(self.graph)
        elif self.layout == "shell":
            self.pos = nx.shell_layout(self.graph)
        elif self.layout == "spectral":
            self.pos = nx.spectral_layout(self.graph)
        else:
            # Default to spring layout
            self.pos = nx.spring_layout(self.graph)
        
        # Get node sizes
        if self.node_size_field and self.node_size_field in next(iter(self.graph.nodes(data=True)))[1]:
            node_sizes = [
                data.get(self.node_size_field, 300)
                for _, data in self.graph.nodes(data=True)
            ]
        else:
            node_sizes = 300
        
        # Get node colors
        if self.node_color_field and self.node_color_field in next(iter(self.graph.nodes(data=True)))[1]:
            node_colors = [
                data.get(self.node_color_field, "blue")
                for _, data in self.graph.nodes(data=True)
            ]
        else:
            node_colors = "skyblue"
        
        # Get edge widths
        if self.edge_width_field and self.graph.edges and self.edge_width_field in next(iter(self.graph.edges(data=True)))[2]:
            edge_widths = [
                data.get(self.edge_width_field, 1.0)
                for _, _, data in self.graph.edges(data=True)
            ]
        else:
            edge_widths = 1.0
        
        # Get edge colors
        if self.edge_color_field and self.graph.edges and self.edge_color_field in next(iter(self.graph.edges(data=True)))[2]:
            edge_colors = [
                data.get(self.edge_color_field, "gray")
                for _, _, data in self.graph.edges(data=True)
            ]
        else:
            edge_colors = "gray"
        
        # Draw the graph
        nx.draw_networkx(
            self.graph,
            pos=self.pos,
            ax=self.ax,
            with_labels=True,
            node_size=node_sizes,
            node_color=node_colors,
            width=edge_widths,
            edge_color=edge_colors,
            labels={
                node: data.get(self.node_label_field, str(node))
                for node, data in self.graph.nodes(data=True)
            } if self.node_label_field else None,
        )
        
        # Remove axis
        self.ax.axis("off")
        
        # Adjust layout
        self.figure.tight_layout()
    
    def render(self) -> Any:
        """Render the visualization.
        
        Returns:
            Matplotlib figure
        """
        if self.figure is None:
            raise RuntimeError("Visualization has not been created")
        
        return self.figure
    
    def save(self, path: str, format: Optional[Union[str, VisualizationFormat]] = None) -> str:
        """Save the visualization to a file.
        
        Args:
            path: Output file path
            format: Output format (inferred from path if not specified)
            
        Returns:
            Path to the saved file
        """
        if self.figure is None:
            raise RuntimeError("Visualization has not been created")
        
        # Infer format from path if not specified
        if format is None:
            ext = os.path.splitext(path)[1][1:].lower()
            if ext in [f.value for f in VisualizationFormat]:
                format = ext
            else:
                format = VisualizationFormat.PNG.value
        
        # Ensure format is a string
        if isinstance(format, VisualizationFormat):
            format = format.value
        
        # Save the figure
        self.figure.savefig(path, format=format, bbox_inches="tight")
        
        return path
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the visualization to a dictionary.
        
        Returns:
            Dictionary representation of the visualization
        """
        return {
            "type": "network_graph",
            "title": self.title,
            "nodes": len(self.data["nodes"]),
            "edges": len(self.data["edges"]),
            "layout": self.layout,
        }


# Factory function
def create_visualization(
    type: Union[str, VisualizationType],
    title: str,
    data: Any,
    **kwargs: Any,
) -> Visualization:
    """Create a visualization of the specified type.
    
    Args:
        type: Visualization type
        title: Visualization title
        data: Data to visualize
        **kwargs: Additional type-specific parameters
        
    Returns:
        Visualization instance
        
    Raises:
        ValueError: If visualization type is not supported
    """
    # Ensure type is a string
    if isinstance(type, VisualizationType):
        type = type.value
    
    # Create the appropriate visualization
    if type == VisualizationType.BAR_CHART.value:
        return BarChart(title, data, **kwargs)
    elif type == VisualizationType.LINE_CHART.value:
        return LineChart(title, data, **kwargs)
    elif type == VisualizationType.PIE_CHART.value:
        return PieChart(title, data, **kwargs)
    elif type == VisualizationType.HEATMAP.value:
        return HeatMap(title, data, **kwargs)
    elif type == VisualizationType.GRAPH.value:
        return NetworkGraph(title, data.get("nodes", []), data.get("edges", []), **kwargs)
    else:
        raise ValueError(f"Unsupported visualization type: {type}")


# Utility functions
def visualize_search_results(
    results: Dict[str, Any],
    title: str = "Search Results",
    visualization_type: str = "bar_chart",
    **kwargs: Any,
) -> Visualization:
    """Create a visualization from search results.
    
    Args:
        results: Search results dictionary
        title: Visualization title
        visualization_type: Type of visualization to create
        **kwargs: Additional visualization parameters
        
    Returns:
        Visualization instance
    """
    if visualization_type == "bar_chart":
        # Count matches by file
        file_counts = {}
        for match in results.get("matches", []):
            file_path = match.get("file_path")
            if file_path:
                file_name = os.path.basename(file_path)
                file_counts[file_name] = file_counts.get(file_name, 0) + 1
        
        # Create bar chart
        return BarChart(
            title,
            file_counts,
            sort_values=True,
            **kwargs,
        )
    
    elif visualization_type == "pie_chart":
        # Count matches by file
        file_counts = {}
        for match in results.get("matches", []):
            file_path = match.get("file_path")
            if file_path:
                file_name = os.path.basename(file_path)
                file_counts[file_name] = file_counts.get(file_name, 0) + 1
        
        # Create pie chart
        return PieChart(
            title,
            file_counts,
            **kwargs,
        )
    
    else:
        raise ValueError(f"Unsupported visualization type for search results: {visualization_type}")


def visualize_word_frequencies(
    text: str,
    title: str = "Word Frequencies",
    limit: int = 20,
    stop_words: Optional[Set[str]] = None,
    **kwargs: Any,
) -> Visualization:
    """Create a visualization of word frequencies in text.
    
    Args:
        text: Text to analyze
        title: Visualization title
        limit: Maximum number of words to include
        stop_words: Set of words to exclude
        **kwargs: Additional visualization parameters
        
    Returns:
        Visualization instance
    """
    import re
    from collections import Counter
    
    # Tokenize the text
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Remove stop words
    if stop_words:
        words = [word for word in words if word not in stop_words]
    
    # Count word frequencies
    word_counts = Counter(words)
    
    # Get the most common words
    most_common = dict(word_counts.most_common(limit))
    
    # Create the visualization
    return BarChart(
        title,
        most_common,
        sort_values=True,
        **kwargs,
    )


def visualize_data_distribution(
    data: List[Union[int, float]],
    title: str = "Data Distribution",
    bins: Optional[int] = None,
    **kwargs: Any,
) -> Visualization:
    """Create a histogram visualization of data distribution.
    
    Args:
        data: List of numeric values
        title: Visualization title
        bins: Number of histogram bins
        **kwargs: Additional visualization parameters
        
    Returns:
        Visualization instance
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib is required for this visualization")
    
    # Create the histogram
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # Plot the histogram
    n, bins, patches = ax.hist(data, bins=bins)
    
    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    
    # Create a custom visualization
    class Histogram(MatplotlibVisualization):
        def __init__(self, fig: plt.Figure, data: List[Union[int, float]], **kwargs: Any):
            self.title = title
            self.data = data
            self.figure = fig
            self.ax = ax
            # Additional properties
            self.bins = bins
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        def to_dict(self) -> Dict[str, Any]:
            return {
                "type": "histogram",
                "title": self.title,
                "data_size": len(self.data),
                "bins": self.bins,
            }
    
    # Return the histogram
    return Histogram(fig, data, **kwargs)


def visualize_time_series(
    timestamps: List[Union[int, float, str]],
    values: List[Union[int, float]],
    title: str = "Time Series",
    **kwargs: Any,
) -> Visualization:
    """Create a line chart visualization of time series data.
    
    Args:
        timestamps: List of timestamps
        values: List of values
        title: Visualization title
        **kwargs: Additional visualization parameters
        
    Returns:
        Visualization instance
    """
    import pandas as pd
    
    # Convert timestamps to datetime if they're strings
    if timestamps and isinstance(timestamps[0], str):
        try:
            timestamps = [pd.to_datetime(ts) for ts in timestamps]
        except Exception as e:
            logger.warning(f"Failed to convert timestamps to datetime: {e}")
    
    # Create a dictionary for the line chart
    data = {"Time Series": dict(zip(timestamps, values))}
    
    # Create the line chart
    return LineChart(
        title,
        data,
        include_points=True,
        **kwargs,
    )


def save_visualizations(
    visualizations: List[Visualization],
    directory: str,
    format: Union[str, VisualizationFormat] = VisualizationFormat.PNG,
    prefix: str = "",
) -> List[str]:
    """Save multiple visualizations to a directory.
    
    Args:
        visualizations: List of visualizations to save
        directory: Output directory
        format: Output format
        prefix: Filename prefix
        
    Returns:
        List of paths to saved files
    """
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Save each visualization
    paths = []
    for i, viz in enumerate(visualizations):
        # Generate a filename from the title or index
        if viz.title:
            # Clean the title for use as a filename
            filename = re.sub(r'[^\w\-_.]', '_', viz.title)
        else:
            filename = f"visualization_{i+1}"
        
        # Add prefix and ensure format
        filename = f"{prefix}{filename}.{format.value}" if isinstance(format, VisualizationFormat) else f"{prefix}{filename}.{format}"
        
        # Save the visualization
        path = viz.save(os.path.join(directory, filename), format)
        paths.append(path)
    
    return paths