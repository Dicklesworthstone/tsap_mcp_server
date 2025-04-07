"""
Composite utilities for TSAP MCP Server.

This module provides utilities for composite operations that
combine multiple core functionalities.
"""
from typing import Any, Dict, List, Optional, Tuple, Union, Pattern
import re
import difflib
import json


def generate_diff(text1: str, text2: str, context_lines: int = 3) -> str:
    """Generate a unified diff between two texts.
    
    Args:
        text1: Original text
        text2: Modified text
        context_lines: Number of context lines to include
        
    Returns:
        Unified diff string
    """
    lines1 = text1.splitlines(keepends=True)
    lines2 = text2.splitlines(keepends=True)
    
    diff = difflib.unified_diff(
        lines1, 
        lines2,
        n=context_lines,
        lineterm=''
    )
    
    return ''.join(diff)


def extract_text_sections(text: str, pattern: Optional[str] = None) -> Dict[str, str]:
    """Extract sections from text based on headers or patterns.
    
    Args:
        text: Text to extract sections from
        pattern: Optional regex pattern for section headers
        
    Returns:
        Dictionary of section name to section text
    """
    sections = {}
    
    if not pattern:
        # Default pattern looks for markdown-style headers
        pattern = r"^(#+)\s+(.+)$"
    
    # Find all headers
    headers = []
    for match in re.finditer(pattern, text, re.MULTILINE):
        if len(match.groups()) >= 2:
            level = len(match.group(1)) if match.group(1).startswith('#') else 1
            title = match.group(2).strip()
            headers.append((match.start(), level, title))
        else:
            title = match.group(0).strip()
            headers.append((match.start(), 1, title))
    
    # Extract sections based on headers
    if headers:
        for i, (start, level, title) in enumerate(headers):
            # Section ends at the next header or the end of text
            end = headers[i+1][0] if i < len(headers) - 1 else len(text)
            section_text = text[start:end].strip()
            sections[title] = section_text
    else:
        # If no headers found, return the whole text as one section
        sections["text"] = text
    
    return sections


def compile_regex_from_examples(
    examples: List[str], 
    non_examples: Optional[List[str]] = None,
    simplify: bool = True
) -> str:
    """Generate a regex pattern that matches examples but not non-examples.
    
    Args:
        examples: List of strings the pattern should match
        non_examples: List of strings the pattern should not match
        simplify: Whether to simplify the resulting pattern
        
    Returns:
        Regex pattern string
    """
    if not examples:
        return ""
    
    # Start with a very specific pattern from the first example
    pattern = re.escape(examples[0])
    
    # Iteratively refine the pattern with more examples
    for example in examples[1:]:
        # Find common prefix
        common_prefix = ""
        for i, (c1, c2) in enumerate(zip(pattern, re.escape(example))):
            if c1 == c2:
                common_prefix += c1
            else:
                break
        
        # Find common suffix
        common_suffix = ""
        for i, (c1, c2) in enumerate(zip(reversed(pattern), reversed(re.escape(example)))):
            if c1 == c2:
                common_suffix = c1 + common_suffix
            else:
                break
        
        # Update pattern to capture both examples
        middle1 = pattern[len(common_prefix):-len(common_suffix) if common_suffix else None]
        middle2 = re.escape(example)[len(common_prefix):-len(common_suffix) if common_suffix else None]
        
        if middle1 and middle2:
            middle = f"({middle1}|{middle2})"
        elif middle1:
            middle = middle1
        elif middle2:
            middle = middle2
        else:
            middle = ""
        
        pattern = common_prefix + middle + common_suffix
    
    # Check against non-examples and refine if needed
    if non_examples:
        # Ensure our pattern doesn't match non-examples
        refinements_needed = []
        for non_example in non_examples:
            if re.match(f"^{pattern}$", non_example):
                refinements_needed.append(non_example)
        
        if refinements_needed:
            # This is a complex problem that may require machine learning
            # Here we just add some basic negative lookahead assertions
            for non_example in refinements_needed:
                # Find a unique substring in the non-example
                for i in range(1, len(non_example)):
                    substr = non_example[i-1:i+1]
                    if all(substr not in ex for ex in examples):
                        pattern = f"(?!.*{re.escape(substr)}){pattern}"
                        break
    
    # Simplify pattern if requested
    if simplify:
        # Replace escaped sequences with character classes where appropriate
        simplifications = [
            (r"\d+", r"\d+"),
            (r"[0-9]+", r"\d+"),
            (r"[a-zA-Z]+", r"\w+"),
            (r"\w+", r"\w+"),
            (r"\s+", r"\s+"),
        ]
        
        for complex_pattern, simple_pattern in simplifications:
            if re.search(complex_pattern, pattern):
                pattern = re.sub(complex_pattern, simple_pattern, pattern)
    
    return pattern


def analyze_file_naming_patterns(filenames: List[str]) -> Dict[str, Any]:
    """Analyze patterns in a list of filenames.
    
    Args:
        filenames: List of filenames to analyze
        
    Returns:
        Dictionary with pattern analysis
    """
    if not filenames:
        return {"error": "No filenames provided"}
    
    patterns = {
        "extensions": {},
        "prefixes": {},
        "naming_formats": [],
        "suggested_pattern": "",
    }
    
    # Analyze extensions
    for filename in filenames:
        parts = filename.split(".")
        if len(parts) > 1:
            ext = parts[-1].lower()
            patterns["extensions"][ext] = patterns["extensions"].get(ext, 0) + 1
    
    # Analyze prefixes (first part before any separator)
    for filename in filenames:
        # Remove extension
        name = filename.split(".")[0]
        
        # Check for common separators
        for sep in ["_", "-", " "]:
            if sep in name:
                prefix = name.split(sep)[0]
                patterns["prefixes"][prefix] = patterns["prefixes"].get(prefix, 0) + 1
                break
    
    # Determine naming formats
    formats = []
    for filename in filenames:
        format_parts = []
        # Split by extension
        name_parts = filename.split(".")
        if len(name_parts) > 1:
            basename = ".".join(name_parts[:-1])
            extension = name_parts[-1]
            format_parts.append(f"name.{extension}")
        else:
            basename = filename
            format_parts.append("name")
        
        # Analyze basename structure
        has_numbers = bool(re.search(r"\d", basename))
        has_underscores = "_" in basename
        has_hyphens = "-" in basename
        
        if has_numbers and (has_underscores or has_hyphens):
            if has_underscores:
                format_parts.append("contains_numbers_and_underscores")
            if has_hyphens:
                format_parts.append("contains_numbers_and_hyphens")
        elif has_numbers:
            format_parts.append("contains_numbers")
        elif has_underscores:
            format_parts.append("contains_underscores")
        elif has_hyphens:
            format_parts.append("contains_hyphens")
        
        formats.append(" + ".join(format_parts))
    
    # Count format frequencies
    format_counts = {}
    for fmt in formats:
        format_counts[fmt] = format_counts.get(fmt, 0) + 1
    
    # Sort by frequency
    patterns["naming_formats"] = [
        {"format": fmt, "count": count}
        for fmt, count in sorted(format_counts.items(), key=lambda x: x[1], reverse=True)
    ]
    
    # Generate a suggested pattern
    if filenames:
        patterns["suggested_pattern"] = compile_regex_from_examples(filenames)
    
    return patterns


def extract_context_around_match(
    text: str, 
    pattern: str, 
    context_lines: int = 3,
    max_matches: int = 10
) -> List[Dict[str, Any]]:
    """Extract context around pattern matches in text.
    
    Args:
        text: Text to search in
        pattern: Regex pattern to search for
        context_lines: Number of context lines before and after match
        max_matches: Maximum number of matches to return
        
    Returns:
        List of match information with context
    """
    results = []
    
    # Compile regex
    try:
        regex = re.compile(pattern, re.MULTILINE)
    except re.error:
        return [{"error": f"Invalid regex pattern: {pattern}"}]
    
    # Find all matches
    lines = text.splitlines()
    match_count = 0
    
    for match in regex.finditer(text):
        if match_count >= max_matches:
            break
            
        # Find the line number of the match
        match_pos = match.start()
        line_no = text[:match_pos].count("\n")
        
        # Extract matched text
        matched_text = match.group(0)
        
        # Calculate context line ranges
        start_line = max(0, line_no - context_lines)
        end_line = min(len(lines), line_no + context_lines + 1)
        
        # Extract context lines
        context_before = lines[start_line:line_no]
        context_after = lines[line_no+1:end_line]
        
        # Get the matched line
        matched_line = lines[line_no] if line_no < len(lines) else ""
        
        results.append({
            "match": matched_text,
            "line": line_no + 1,  # 1-indexed line number
            "context_before": context_before,
            "matched_line": matched_line,
            "context_after": context_after,
        })
        
        match_count += 1
    
    return results


def merge_results(
    results1: List[Dict[str, Any]],
    results2: List[Dict[str, Any]],
    merge_key: str = "path"
) -> List[Dict[str, Any]]:
    """Merge two sets of results based on a common key.
    
    Args:
        results1: First set of results
        results2: Second set of results
        merge_key: Key to use for merging
        
    Returns:
        Merged results
    """
    merged = {}
    
    # Add results from first set
    for item in results1:
        if merge_key in item:
            key = item[merge_key]
            merged[key] = item.copy()
    
    # Merge with second set
    for item in results2:
        if merge_key in item:
            key = item[merge_key]
            if key in merged:
                # Merge items with the same key
                for k, v in item.items():
                    if k != merge_key:
                        if k not in merged[key]:
                            merged[key][k] = v
                        elif isinstance(merged[key][k], list) and isinstance(v, list):
                            merged[key][k].extend(v)
                        elif isinstance(merged[key][k], dict) and isinstance(v, dict):
                            merged[key][k].update(v)
            else:
                merged[key] = item.copy()
    
    return list(merged.values()) 