"""
Processing utilities for TSAP MCP Server.

This module provides utilities for data processing, transformation,
and format conversions.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import csv
import io
import json
import re


def convert_csv_to_json(csv_data: str) -> List[Dict[str, Any]]:
    """Convert CSV data to JSON format.
    
    Args:
        csv_data: CSV data as string
        
    Returns:
        List of dictionaries representing rows
    """
    # Process CSV data
    csv_reader = csv.DictReader(io.StringIO(csv_data))
    return [row for row in csv_reader]


def convert_json_to_csv(json_data: List[Dict[str, Any]]) -> str:
    """Convert JSON data to CSV format.
    
    Args:
        json_data: List of dictionaries to convert
        
    Returns:
        CSV data as string
    """
    if not json_data:
        return ""
    
    # Get field names from first item
    fieldnames = list(json_data[0].keys())
    
    # Write to CSV
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(json_data)
    
    return output.getvalue()


def clean_text(text: str, options: Optional[Dict[str, Any]] = None) -> str:
    """Clean text by removing unwanted characters and patterns.
    
    Args:
        text: Text to clean
        options: Cleaning options
        
    Returns:
        Cleaned text
    """
    if not options:
        options = {}
    
    # Make a copy of the original text
    cleaned = text
    
    # Apply specified cleaning operations
    if options.get("lowercase", False):
        cleaned = cleaned.lower()
    
    if options.get("strip_html", False):
        cleaned = re.sub(r'<[^>]+>', '', cleaned)
    
    if options.get("strip_punctuation", False):
        cleaned = re.sub(r'[^\w\s]', '', cleaned)
    
    if options.get("normalize_whitespace", False):
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    if options.get("strip_numbers", False):
        cleaned = re.sub(r'\d+', '', cleaned)
    
    return cleaned


def extract_table_from_markdown(markdown: str) -> List[Dict[str, str]]:
    """Extract a table from markdown text.
    
    Args:
        markdown: Markdown text containing a table
        
    Returns:
        List of dictionaries representing table rows
    """
    # Find table lines
    lines = [line.strip() for line in markdown.strip().split('\n')]
    table_lines = []
    in_table = False
    
    for line in lines:
        if line.startswith('|') and line.endswith('|'):
            table_lines.append(line)
            in_table = True
        elif in_table and not line:
            # Empty line after table means end of table
            in_table = False
    
    if len(table_lines) < 3:
        # Not enough lines for a proper table
        return []
    
    # Extract headers and rows
    headers = [h.strip() for h in table_lines[0].strip('|').split('|')]
    rows = []
    
    for line in table_lines[2:]:  # Skip header and separator
        if not line.strip('| '):
            continue
        values = [v.strip() for v in line.strip('|').split('|')]
        row = {}
        for i, value in enumerate(values):
            if i < len(headers):
                row[headers[i]] = value
        rows.append(row)
    
    return rows


def parse_jq_query(query: str) -> Dict[str, Any]:
    """Parse a JQ query into its components.
    
    Args:
        query: JQ query string
        
    Returns:
        Dictionary with query components
    """
    parts = {}
    
    # Extract filters
    filters = []
    current = ""
    depth = 0
    
    for char in query:
        if char == '|' and depth == 0:
            if current.strip():
                filters.append(current.strip())
            current = ""
        else:
            if char == '[':
                depth += 1
            elif char == ']':
                depth -= 1
            current += char
    
    if current.strip():
        filters.append(current.strip())
    
    parts["filters"] = filters
    
    # Extract array indices
    indices = re.findall(r'\[\d+\]', query)
    parts["indices"] = [idx.strip('[]') for idx in indices]
    
    # Extract field selections
    fields = re.findall(r'\.([a-zA-Z_][a-zA-Z0-9_]*)', query)
    parts["fields"] = fields
    
    return parts


def generate_awk_command(pattern: str, action: str, separator: str = None) -> str:
    """Generate an AWK command with pattern and action.
    
    Args:
        pattern: AWK pattern
        action: AWK action
        separator: Field separator
        
    Returns:
        AWK command string
    """
    cmd = ["awk"]
    
    # Add field separator if specified
    if separator:
        cmd.append(f"-F'{separator}'")
    
    # Add pattern and action
    if pattern and action:
        cmd.append(f"'{pattern} {{ {action} }}'")
    elif pattern:
        cmd.append(f"'{pattern}'")
    elif action:
        cmd.append(f"'{{ {action} }}'")
    
    return " ".join(cmd)


def extract_data_by_pattern(text: str, pattern: str) -> List[Dict[str, str]]:
    """Extract structured data from text using regex pattern.
    
    Args:
        text: Text to extract data from
        pattern: Regex pattern with named groups
        
    Returns:
        List of dictionaries with extracted data
    """
    # Compile regex
    try:
        regex = re.compile(pattern)
    except re.error:
        return []
    
    # Find all matches
    matches = []
    for match in regex.finditer(text):
        if match.groupdict():
            matches.append(match.groupdict())
        else:
            # If no named groups, create dict with group indices
            group_dict = {}
            for i, group in enumerate(match.groups(), 1):
                group_dict[f"group{i}"] = group
            matches.append(group_dict)
    
    return matches 