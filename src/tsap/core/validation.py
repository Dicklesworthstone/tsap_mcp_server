"""
Input and output validation for TSAP.

This module provides functions and classes for validating inputs, outputs,
and configurations for TSAP core tools and operations.
"""
import os
import re
from typing import Dict, List, Any, Optional, Union, Callable, Type
import json
from enum import Enum


class ValidationError(Exception):
    """Exception raised for validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None):
        """Initialize validation error.
        
        Args:
            message: Error message
            field: Optional field name that caused the error
        """
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}" if field else message)


class ValidationLevel(str, Enum):
    """Validation level for inputs and outputs."""
    
    NONE = "none"  # No validation
    MINIMAL = "minimal"  # Basic type and range checks
    STRICT = "strict"  # Comprehensive validation
    PARANOID = "paranoid"  # Extra security checks and sanitization


def validate_path(
    path: str,
    must_exist: bool = True,
    should_be_file: bool = False,
    should_be_dir: bool = False,
    access_mode: Optional[str] = None,
    field_name: Optional[str] = None,
) -> str:
    """Validate a file or directory path.
    
    Args:
        path: Path to validate
        must_exist: Whether the path must exist
        should_be_file: Whether the path should be a file
        should_be_dir: Whether the path should be a directory
        access_mode: Optional access mode to check ('r', 'w', 'x')
        field_name: Optional field name for error reporting
        
    Returns:
        Validated path
        
    Raises:
        ValidationError: If validation fails
    """
    # Check existence
    if must_exist and not os.path.exists(path):
        raise ValidationError(f"Path does not exist: {path}", field_name)
    
    # Check file/dir type
    if should_be_file and os.path.exists(path) and not os.path.isfile(path):
        raise ValidationError(f"Path is not a file: {path}", field_name)
    
    if should_be_dir and os.path.exists(path) and not os.path.isdir(path):
        raise ValidationError(f"Path is not a directory: {path}", field_name)
    
    # Check access mode
    if access_mode and os.path.exists(path):
        if 'r' in access_mode and not os.access(path, os.R_OK):
            raise ValidationError(f"Path is not readable: {path}", field_name)
        if 'w' in access_mode and not os.access(path, os.W_OK):
            raise ValidationError(f"Path is not writable: {path}", field_name)
        if 'x' in access_mode and not os.access(path, os.X_OK):
            raise ValidationError(f"Path is not executable: {path}", field_name)
    
    return path


def validate_regex(
    pattern: str,
    field_name: Optional[str] = None,
) -> str:
    """Validate a regular expression pattern.
    
    Args:
        pattern: Regex pattern to validate
        field_name: Optional field name for error reporting
        
    Returns:
        Validated pattern
        
    Raises:
        ValidationError: If the pattern is not a valid regex
    """
    try:
        re.compile(pattern)
        return pattern
    except re.error as e:
        raise ValidationError(f"Invalid regex pattern: {str(e)}", field_name)


def validate_json(
    json_str: str,
    schema: Optional[Dict[str, Any]] = None,
    field_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate a JSON string.
    
    Args:
        json_str: JSON string to validate
        schema: Optional JSON schema to validate against
        field_name: Optional field name for error reporting
        
    Returns:
        Parsed JSON object
        
    Raises:
        ValidationError: If the JSON is not valid or does not match the schema
    """
    try:
        data = json.loads(json_str)
        
        # TODO: Implement schema validation if schema is provided
        
        return data
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON: {str(e)}", field_name)


def validate_type(
    value: Any,
    expected_type: Union[Type, List[Type]],
    field_name: Optional[str] = None,
) -> Any:
    """Validate a value against expected type(s).
    
    Args:
        value: Value to validate
        expected_type: Expected type or list of types
        field_name: Optional field name for error reporting
        
    Returns:
        Validated value
        
    Raises:
        ValidationError: If the value is not of the expected type
    """
    if not isinstance(expected_type, list):
        expected_type = [expected_type]
    
    if not any(isinstance(value, t) for t in expected_type):
        type_names = [t.__name__ for t in expected_type]
        type_str = " or ".join(type_names)
        raise ValidationError(
            f"Expected {type_str}, got {type(value).__name__}",
            field_name
        )
    
    return value


def validate_range(
    value: Union[int, float],
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
    field_name: Optional[str] = None,
) -> Union[int, float]:
    """Validate a numeric value within a range.
    
    Args:
        value: Value to validate
        min_value: Optional minimum value (inclusive)
        max_value: Optional maximum value (inclusive)
        field_name: Optional field name for error reporting
        
    Returns:
        Validated value
        
    Raises:
        ValidationError: If the value is outside the range
    """
    if min_value is not None and value < min_value:
        raise ValidationError(
            f"Value {value} is less than minimum {min_value}",
            field_name
        )
    
    if max_value is not None and value > max_value:
        raise ValidationError(
            f"Value {value} is greater than maximum {max_value}",
            field_name
        )
    
    return value


def validate_list(
    values: List[Any],
    item_validator: Optional[Callable[[Any], Any]] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    field_name: Optional[str] = None,
) -> List[Any]:
    """Validate a list and optionally its items.
    
    Args:
        values: List to validate
        item_validator: Optional function to validate each item
        min_length: Optional minimum list length
        max_length: Optional maximum list length
        field_name: Optional field name for error reporting
        
    Returns:
        Validated list
        
    Raises:
        ValidationError: If the list or any item is invalid
    """
    # Validate type
    validate_type(values, list, field_name)
    
    # Validate length
    if min_length is not None and len(values) < min_length:
        raise ValidationError(
            f"List length {len(values)} is less than minimum {min_length}",
            field_name
        )
    
    if max_length is not None and len(values) > max_length:
        raise ValidationError(
            f"List length {len(values)} is greater than maximum {max_length}",
            field_name
        )
    
    # Validate items
    if item_validator:
        validated_items = []
        for i, item in enumerate(values):
            try:
                validated_item = item_validator(item)
                validated_items.append(validated_item)
            except ValidationError as e:
                # Add index to field name
                item_field = f"{field_name}[{i}]" if field_name else f"item[{i}]"
                raise ValidationError(e.message, item_field)
        return validated_items
    
    return values


def validate_dict(
    values: Dict[str, Any],
    schema: Optional[Dict[str, Dict[str, Any]]] = None,
    required_keys: Optional[List[str]] = None,
    field_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate a dictionary against a schema.
    
    Args:
        values: Dictionary to validate
        schema: Optional schema mapping keys to validator functions/types
        required_keys: Optional list of required keys
        field_name: Optional field name for error reporting
        
    Returns:
        Validated dictionary
        
    Raises:
        ValidationError: If the dictionary or any value is invalid
    """
    # Validate type
    validate_type(values, dict, field_name)
    
    # Check required keys
    if required_keys:
        for key in required_keys:
            if key not in values:
                raise ValidationError(f"Missing required key: {key}", field_name)
    
    # Validate against schema
    if schema:
        validated_values = {}
        for key, value in values.items():
            if key in schema:
                validator_info = schema[key]
                validator = validator_info.get("validator")
                if validator:
                    try:
                        validated_value = validator(value)
                        validated_values[key] = validated_value
                    except ValidationError as e:
                        # Add key to field name
                        key_field = f"{field_name}.{key}" if field_name else key
                        raise ValidationError(e.message, key_field)
                else:
                    # No validator, keep as is
                    validated_values[key] = value
            else:
                # Key not in schema, keep as is
                validated_values[key] = value
        return validated_values
    
    return values


def sanitize_command_args(
    args: List[str],
    level: ValidationLevel = ValidationLevel.STRICT,
    allow_flags: bool = True,
    allow_pipes: bool = False,
    allow_redirects: bool = False,
    field_name: Optional[str] = None,
) -> List[str]:
    """Sanitize command arguments for security.
    
    Args:
        args: Command arguments to sanitize
        level: Validation level
        allow_flags: Whether to allow flag arguments (--flag, -f)
        allow_pipes: Whether to allow pipe characters (|)
        allow_redirects: Whether to allow redirects (>, >>, <)
        field_name: Optional field name for error reporting
        
    Returns:
        Sanitized command arguments
        
    Raises:
        ValidationError: If any argument is not allowed
    """
    # No validation
    if level == ValidationLevel.NONE:
        return args
    
    # Minimal validation just checks for empty args
    if level == ValidationLevel.MINIMAL:
        return [arg for arg in args if arg]
    
    # Strict and paranoid validation
    sanitized_args = []
    
    for i, arg in enumerate(args):
        if not arg:
            continue
        
        # Check for shell control characters
        if level == ValidationLevel.PARANOID:
            if any(c in arg for c in ['$', '`', '\\', '&&', ';', '||']):
                raise ValidationError(
                    f"Shell control characters not allowed: {arg}",
                    f"{field_name}[{i}]" if field_name else None
                )
        
        # Check for pipes
        if not allow_pipes and '|' in arg:
            raise ValidationError(
                f"Pipe characters not allowed: {arg}",
                f"{field_name}[{i}]" if field_name else None
            )
        
        # Check for redirects
        if not allow_redirects and any(r in arg for r in ['>', '>>', '<']):
            raise ValidationError(
                f"Redirect characters not allowed: {arg}",
                f"{field_name}[{i}]" if field_name else None
            )
        
        # Check for flags if not allowed
        if not allow_flags and arg.startswith(('-', '--')):
            raise ValidationError(
                f"Flag arguments not allowed: {arg}",
                f"{field_name}[{i}]" if field_name else None
            )
        
        sanitized_args.append(arg)
    
    return sanitized_args


def validate_file_content(
    content: str,
    max_size: Optional[int] = None,
    content_type: Optional[str] = None,
    field_name: Optional[str] = None,
) -> str:
    """Validate file content.
    
    Args:
        content: File content to validate
        max_size: Optional maximum content size
        content_type: Optional content type to validate against
        field_name: Optional field name for error reporting
        
    Returns:
        Validated content
        
    Raises:
        ValidationError: If the content is invalid
    """
    # Check size
    if max_size is not None and len(content) > max_size:
        raise ValidationError(
            f"Content size {len(content)} exceeds maximum {max_size}",
            field_name
        )
    
    # Check content type
    if content_type:
        if content_type == 'json':
            try:
                json.loads(content)
            except json.JSONDecodeError as e:
                raise ValidationError(f"Invalid JSON content: {str(e)}", field_name)
        elif content_type == 'xml':
            # Very basic XML validation
            if not (content.startswith('<?xml') or content.startswith('<')):
                raise ValidationError("Invalid XML content", field_name)
    
    return content