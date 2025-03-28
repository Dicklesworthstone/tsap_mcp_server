"""
TSAP JSON Output Formatter.

This module provides enhanced JSON formatting capabilities
for TSAP result output.
"""

import json
import datetime
import uuid
from typing import Dict, Any, Optional, TextIO

from tsap.output.formatter import OutputFormatter, JsonFormatter


class EnhancedJsonEncoder(json.JSONEncoder):
    """Enhanced JSON encoder that handles additional Python types.
    
    Provides custom serialization for:
    - datetime, date, time objects
    - UUID objects
    - Set objects
    - Objects with a to_dict() method
    - Objects with a __dict__ attribute
    """

    def default(self, obj: Any) -> Any:
        """Implement custom serialization for special types.
        
        Args:
            obj: Object to serialize
            
        Returns:
            JSON-serializable representation
        """
        # Handle datetime objects
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        if isinstance(obj, datetime.time):
            return obj.isoformat()
            
        # Handle UUID objects
        if isinstance(obj, uuid.UUID):
            return str(obj)
            
        # Handle sets
        if isinstance(obj, set):
            return list(obj)
            
        # Handle objects with a to_dict method
        if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
            return obj.to_dict()
            
        # Handle objects with a __dict__ attribute (excluding modules, etc.)
        if hasattr(obj, "__dict__") and not obj.__class__.__module__ == "builtins":
            return {key: value for key, value in obj.__dict__.items() 
                    if not key.startswith("_")}
            
        # Let the parent class handle it or raise TypeError
        return super().default(obj)


class EnhancedJsonFormatter(JsonFormatter):
    """Enhanced JSON formatter with additional capabilities.
    
    Extends the base JsonFormatter with:
    - Support for additional data types
    - Schema version inclusion
    - Metadata inclusion
    """

    def __init__(
        self,
        pretty: bool = True,
        indent: int = 2,
        include_schema_version: bool = False,
        schema_version: str = "1.0",
        include_metadata: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the enhanced JSON formatter.
        
        Args:
            pretty: Whether to format the output in a human-readable way
            indent: Number of spaces for indentation when pretty is True
            include_schema_version: Whether to include schema version in output
            schema_version: Schema version string to include
            include_metadata: Whether to include metadata in output
            metadata: Additional metadata to include
        """
        super().__init__(pretty, indent)
        self.include_schema_version = include_schema_version
        self.schema_version = schema_version
        self.include_metadata = include_metadata
        self.metadata = metadata or {}

    def format(self, data: Any) -> str:
        """Format the data as a JSON string with enhanced capabilities.
        
        Args:
            data: The data to format
            
        Returns:
            JSON string
        """
        # If we need to add metadata or schema version, wrap the data
        if self.include_schema_version or self.include_metadata:
            wrapped_data = {"data": data}
            
            if self.include_schema_version:
                wrapped_data["schema_version"] = self.schema_version
                
            if self.include_metadata:
                wrapped_data["metadata"] = self.metadata
            
            data = wrapped_data
        
        return json.dumps(
            data,
            indent=self.indent,
            sort_keys=self.pretty,
            ensure_ascii=False,
            cls=EnhancedJsonEncoder,
        )

    def format_stream(self, data: Any, stream: TextIO) -> None:
        """Write enhanced JSON directly to a stream.
        
        Args:
            data: The data to format
            stream: Output stream to write to
        """
        # If we need to add metadata or schema version, wrap the data
        if self.include_schema_version or self.include_metadata:
            wrapped_data = {"data": data}
            
            if self.include_schema_version:
                wrapped_data["schema_version"] = self.schema_version
                
            if self.include_metadata:
                wrapped_data["metadata"] = self.metadata
            
            data = wrapped_data
        
        json.dump(
            data,
            stream,
            indent=self.indent,
            sort_keys=self.pretty,
            ensure_ascii=False,
            cls=EnhancedJsonEncoder,
        )


class JsonLinesFormatter(OutputFormatter):
    """Formatter for JSON Lines format (one JSON object per line)."""

    def __init__(self, include_metadata: bool = False, metadata: Optional[Dict[str, Any]] = None):
        """Initialize the JSON Lines formatter.
        
        Args:
            include_metadata: Whether to include metadata in each line
            metadata: Additional metadata to include in each line
        """
        super().__init__(pretty=False)  # Pretty printing doesn't apply to JSON Lines
        self.include_metadata = include_metadata
        self.metadata = metadata or {}

    def format(self, data: Any) -> str:
        """Format the data as JSON Lines.
        
        Args:
            data: The data to format (must be a list for JSON Lines)
            
        Returns:
            JSON Lines string (one JSON object per line)
        """
        if not isinstance(data, list):
            # Wrap non-list data in a list
            data = [data]
            
        output = []
        for item in data:
            if self.include_metadata:
                # Add metadata to each item
                if isinstance(item, dict):
                    item = {**item, "metadata": self.metadata}
                else:
                    item = {"data": item, "metadata": self.metadata}
            
            # Convert to JSON without newlines
            line = json.dumps(item, ensure_ascii=False, cls=EnhancedJsonEncoder)
            output.append(line)
            
        return "\n".join(output)

    def format_stream(self, data: Any, stream: TextIO) -> None:
        """Write JSON Lines directly to a stream.
        
        Args:
            data: The data to format (must be a list for JSON Lines)
            stream: Output stream to write to
        """
        if not isinstance(data, list):
            # Wrap non-list data in a list
            data = [data]
            
        for item in data:
            if self.include_metadata:
                # Add metadata to each item
                if isinstance(item, dict):
                    item = {**item, "metadata": self.metadata}
                else:
                    item = {"data": item, "metadata": self.metadata}
            
            # Write each item as a separate line
            line = json.dumps(item, ensure_ascii=False, cls=EnhancedJsonEncoder)
            stream.write(line)
            stream.write("\n")


# Factory functions for JSON formatters
def get_enhanced_json_formatter(
    pretty: bool = True,
    indent: int = 2,
    include_schema_version: bool = False,
    schema_version: str = "1.0",
    include_metadata: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> EnhancedJsonFormatter:
    """Get a configured enhanced JSON formatter.
    
    Args:
        pretty: Whether to format the output in a human-readable way
        indent: Number of spaces for indentation when pretty is True
        include_schema_version: Whether to include schema version
        schema_version: Schema version string to include
        include_metadata: Whether to include metadata
        metadata: Additional metadata to include
        
    Returns:
        Configured EnhancedJsonFormatter
    """
    return EnhancedJsonFormatter(
        pretty=pretty,
        indent=indent,
        include_schema_version=include_schema_version,
        schema_version=schema_version,
        include_metadata=include_metadata,
        metadata=metadata,
    )


def get_jsonl_formatter(
    include_metadata: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> JsonLinesFormatter:
    """Get a configured JSON Lines formatter.
    
    Args:
        include_metadata: Whether to include metadata in each line
        metadata: Additional metadata to include
        
    Returns:
        Configured JsonLinesFormatter
    """
    return JsonLinesFormatter(
        include_metadata=include_metadata,
        metadata=metadata,
    )