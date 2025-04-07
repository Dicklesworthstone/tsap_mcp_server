"""
Visualization utilities for TSAP MCP Server.

This module provides utilities for working with visualization data,
format conversions, and chart generation.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import base64
import io
import json


def encode_image_base64(image_bytes: bytes, format: str = "png") -> str:
    """Encode image bytes as base64 for embedding in JSON.
    
    Args:
        image_bytes: Raw image bytes
        format: Image format (png, jpg, svg, etc.)
        
    Returns:
        Base64-encoded string
    """
    base64_str = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:image/{format};base64,{base64_str}"


def decode_image_base64(base64_str: str) -> Tuple[bytes, str]:
    """Decode base64 image string to bytes.
    
    Args:
        base64_str: Base64-encoded image string with format prefix
        
    Returns:
        Tuple of (image_bytes, format)
    """
    # Extract format from string
    _, format_data = base64_str.split("data:image/", 1)
    format, data = format_data.split(";base64,", 1)
    
    # Decode base64 data
    image_bytes = base64.b64decode(data)
    return image_bytes, format


def convert_chart_data_for_mcp(chart_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert chart data to a format suitable for MCP transmission.
    
    This function ensures that chart data is properly formatted for
    transmission through MCP, handling binary data and special formats.
    
    Args:
        chart_data: Original chart data dictionary
        
    Returns:
        MCP-compatible chart data
    """
    mcp_data = {}
    
    # Copy basic properties
    for key, value in chart_data.items():
        if key != "image_data" and key != "raw_data":
            mcp_data[key] = value
    
    # Handle image data if present
    if "image_data" in chart_data and chart_data["image_data"]:
        image_data = chart_data["image_data"]
        if isinstance(image_data, bytes):
            format = chart_data.get("format", "png")
            mcp_data["image_data"] = encode_image_base64(image_data, format)
        else:
            mcp_data["image_data"] = image_data
    
    # Handle raw data if present
    if "raw_data" in chart_data and chart_data["raw_data"]:
        # Ensure raw data is JSON serializable
        try:
            json.dumps(chart_data["raw_data"])
            mcp_data["raw_data"] = chart_data["raw_data"]
        except (TypeError, OverflowError):
            # Convert to string if not serializable
            mcp_data["raw_data"] = str(chart_data["raw_data"])
    
    return mcp_data


def sanitize_chart_options(options: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize chart options to ensure they're valid.
    
    This function validates and corrects chart options to ensure
    they're compatible with the charting library.
    
    Args:
        options: Chart options dictionary
        
    Returns:
        Sanitized options dictionary
    """
    sanitized = {}
    
    # Copy valid options
    for key, value in options.items():
        if key in ["title", "width", "height", "legend", "colors", "theme"]:
            sanitized[key] = value
    
    # Ensure width and height are integers
    if "width" in sanitized and not isinstance(sanitized["width"], int):
        try:
            sanitized["width"] = int(sanitized["width"])
        except (ValueError, TypeError):
            sanitized["width"] = 800  # Default width
    
    if "height" in sanitized and not isinstance(sanitized["height"], int):
        try:
            sanitized["height"] = int(sanitized["height"])
        except (ValueError, TypeError):
            sanitized["height"] = 600  # Default height
    
    return sanitized 