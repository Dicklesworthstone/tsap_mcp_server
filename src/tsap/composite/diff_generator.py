"""
Diff Generator for the TSAP MCP Server.

This module implements tools for generating meaningful differences between
text files, documents, or document versions.
"""

import asyncio
import os
import re
import difflib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from tsap.utils.logging import logger
from tsap.mcp.models import DiffGeneratorParams, DiffGeneratorResult, DiffChunk


@dataclass
class DiffOptions:
    """Options for diff generation."""
    context_lines: int = 3
    ignore_whitespace: bool = False
    ignore_case: bool = False
    ignore_blank_lines: bool = False
    structural_diff: bool = False
    semantic_diff: bool = False
    highlight_changes: bool = True
    max_diff_size: Optional[int] = None


@dataclass
class DiffResult:
    """Result of a diff operation."""
    source_path: str
    target_path: str
    diff_chunks: List[DiffChunk] = field(default_factory=list)
    total_lines_changed: int = 0
    lines_added: int = 0
    lines_removed: int = 0
    lines_modified: int = 0
    source_size: int = 0
    target_size: int = 0
    diff_ratio: float = 0.0
    error: Optional[str] = None


async def _read_file(file_path: str) -> List[str]:
    """
    Read a file and return its contents as a list of lines.
    
    Args:
        file_path: Path to the file
        
    Returns:
        List of lines in the file
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.readlines()
    except Exception as e:
        logger.error(
            f"Error reading file {file_path}: {str(e)}",
            component="composite",
            operation="diff_generator"
        )
        raise


async def _preprocess_lines(
    lines: List[str],
    options: DiffOptions
) -> List[str]:
    """
    Preprocess lines according to diff options.
    
    Args:
        lines: List of lines to preprocess
        options: Diff options
        
    Returns:
        Preprocessed lines
    """
    result = lines.copy()
    
    # Apply preprocessing options
    if options.ignore_whitespace:
        result = [re.sub(r'\s+', ' ', line).strip() for line in result]
    
    if options.ignore_case:
        result = [line.lower() for line in result]
    
    if options.ignore_blank_lines:
        result = [line for line in result if line.strip()]
    
    return result


async def _generate_basic_diff(
    source_lines: List[str],
    target_lines: List[str],
    options: DiffOptions
) -> List[DiffChunk]:
    """
    Generate a basic diff between source and target lines.
    
    Args:
        source_lines: Source file lines
        target_lines: Target file lines
        options: Diff options
        
    Returns:
        List of diff chunks
    """
    # Preprocess lines according to options
    preprocessed_source = await _preprocess_lines(source_lines, options)
    preprocessed_target = await _preprocess_lines(target_lines, options)
    
    # Generate a unified diff
    diff_generator = difflib.unified_diff(
        preprocessed_source,
        preprocessed_target,
        n=options.context_lines,
        lineterm=''
    )
    
    # Skip the first two lines (headers)
    try:
        next(diff_generator)  # Skip "--- file1"
        next(diff_generator)  # Skip "+++ file2"
    except StopIteration:
        # Empty diff
        return []
    
    chunks = []
    current_chunk = None
    
    for line in diff_generator:
        # Check for chunk header (e.g., "@@ -1,7 +1,6 @@")
        if line.startswith("@@"):
            # Process the previous chunk
            if current_chunk is not None:
                chunks.append(current_chunk)
            
            # Parse the chunk header
            match = re.match(r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@", line)
            if match:
                source_start = int(match.group(1))
                source_length = int(match.group(2))
                target_start = int(match.group(3))
                target_length = int(match.group(4))
                
                # Create a new chunk
                current_chunk = DiffChunk(
                    source_start=source_start,
                    source_length=source_length,
                    target_start=target_start,
                    target_length=target_length,
                    lines=[],
                    changes={
                        "additions": 0,
                        "deletions": 0,
                        "modifications": 0
                    }
                )
        elif current_chunk is not None:
            # Add line to current chunk
            current_chunk.lines.append(line)
            
            # Update change counts
            if line.startswith("+"):
                current_chunk.changes["additions"] += 1
            elif line.startswith("-"):
                current_chunk.changes["deletions"] += 1
    
    # Add the last chunk
    if current_chunk is not None:
        chunks.append(current_chunk)
    
    # Calculate modifications (lines that were changed rather than added/deleted)
    for chunk in chunks:
        # Count consecutive "+" and "-" lines as modifications
        i = 0
        while i < len(chunk.lines):
            if i < len(chunk.lines) - 1 and chunk.lines[i].startswith("-") and chunk.lines[i+1].startswith("+"):
                # Count as a modification
                chunk.changes["modifications"] += 1
                chunk.changes["additions"] -= 1
                chunk.changes["deletions"] -= 1
                i += 2
            else:
                i += 1
    
    return chunks


async def _generate_structural_diff(
    source_lines: List[str],
    target_lines: List[str],
    options: DiffOptions
) -> List[DiffChunk]:
    """
    Generate a structural diff that understands code structure.
    
    Args:
        source_lines: Source file lines
        target_lines: Target file lines
        options: Diff options
        
    Returns:
        List of diff chunks
    """
    # This would ideally use a proper parser for the specific language
    # For now, use a simple heuristic to identify structural blocks
    
    # First, get basic diff chunks
    basic_chunks = await _generate_basic_diff(source_lines, target_lines, options)
    
    # Expand chunks to include complete structural blocks
    expanded_chunks = []
    
    for chunk in basic_chunks:
        # Determine the file type based on common patterns
        file_extension = None  # noqa: F841
        
        # Look for function/class/block boundaries
        # Start and end lines for expanded chunk
        source_start = max(1, chunk.source_start - 5)
        source_end = min(len(source_lines), chunk.source_start + chunk.source_length + 5)
        
        # Look for block start/end (like function/class definitions, brackets, etc.)
        # This is a simplistic approach - a real implementation would use language-specific parsers
        
        # Create expanded chunk
        expanded_chunk = DiffChunk(
            source_start=source_start,
            source_length=source_end - source_start,
            target_start=chunk.target_start,
            target_length=chunk.target_length,
            lines=chunk.lines,
            changes=chunk.changes
        )
        
        expanded_chunks.append(expanded_chunk)
    
    return expanded_chunks


async def _generate_semantic_diff(
    source_lines: List[str],
    target_lines: List[str],
    options: DiffOptions
) -> List[DiffChunk]:
    """
    Generate a semantic diff that understands the meaning of changes.
    
    Args:
        source_lines: Source file lines
        target_lines: Target file lines
        options: Diff options
        
    Returns:
        List of diff chunks with semantic annotations
    """
    # First, get basic or structural diff chunks
    base_chunks = []
    if options.structural_diff:
        base_chunks = await _generate_structural_diff(source_lines, target_lines, options)
    else:
        base_chunks = await _generate_basic_diff(source_lines, target_lines, options)
    
    # Add semantic information to chunks
    for chunk in base_chunks:
        # Analyze the changes to detect semantic meaning
        added_lines = [line[1:] for line in chunk.lines if line.startswith("+")]
        removed_lines = [line[1:] for line in chunk.lines if line.startswith("-")]
        
        # Add semantic labels (this would be more sophisticated in a real implementation)
        semantic_labels = []
        
        # Simple heuristics for semantic changes
        if any("function" in line or "def " in line for line in added_lines + removed_lines):
            semantic_labels.append("function_change")
        
        if any("class" in line for line in added_lines + removed_lines):
            semantic_labels.append("class_change")
        
        if any("import" in line for line in added_lines):
            semantic_labels.append("import_added")
        
        if any("import" in line for line in removed_lines):
            semantic_labels.append("import_removed")
        
        # Add the semantic information to the chunk
        chunk.semantic_labels = semantic_labels
    
    return base_chunks


async def generate_diff(params: DiffGeneratorParams) -> DiffGeneratorResult:
    """
    Generate a diff between two files or file versions.
    
    Args:
        params: Parameters for diff generation
        
    Returns:
        Diff generation result
    """
    try:
        # Create options object
        options = DiffOptions(
            context_lines=params.context_lines,
            ignore_whitespace=params.ignore_whitespace,
            ignore_case=params.ignore_case,
            ignore_blank_lines=params.ignore_blank_lines,
            structural_diff=params.structural_diff,
            semantic_diff=params.semantic_diff,
            highlight_changes=params.highlight_changes,
            max_diff_size=params.max_diff_size
        )
        
        # Read the files
        source_lines = await _read_file(params.source_path)
        target_lines = await _read_file(params.target_path)
        
        # Generate diff chunks based on options
        diff_chunks = []
        if options.semantic_diff:
            diff_chunks = await _generate_semantic_diff(source_lines, target_lines, options)
        elif options.structural_diff:
            diff_chunks = await _generate_structural_diff(source_lines, target_lines, options)
        else:
            diff_chunks = await _generate_basic_diff(source_lines, target_lines, options)
        
        # Calculate statistics
        total_lines_changed = 0
        lines_added = 0
        lines_removed = 0
        lines_modified = 0
        
        for chunk in diff_chunks:
            lines_added += chunk.changes["additions"]
            lines_removed += chunk.changes["deletions"]
            lines_modified += chunk.changes["modifications"]
        
        total_lines_changed = lines_added + lines_removed + lines_modified
        
        # Calculate diff ratio (percentage of lines changed)
        source_size = len(source_lines)
        target_size = len(target_lines)
        diff_ratio = total_lines_changed / max(source_size, target_size) if max(source_size, target_size) > 0 else 0
        
        # Create result object
        result = DiffGeneratorResult(
            source_path=params.source_path,
            target_path=params.target_path,
            diff_chunks=diff_chunks,
            total_lines_changed=total_lines_changed,
            lines_added=lines_added,
            lines_removed=lines_removed,
            lines_modified=lines_modified,
            source_size=source_size,
            target_size=target_size,
            diff_ratio=diff_ratio,
            structural_diff=options.structural_diff,
            semantic_diff=options.semantic_diff
        )
        
        return result
    except Exception as e:
        logger.error(
            f"Error generating diff: {str(e)}",
            component="composite",
            operation="diff_generator"
        )
        
        # Create error result
        return DiffGeneratorResult(
            source_path=params.source_path,
            target_path=params.target_path,
            diff_chunks=[],
            total_lines_changed=0,
            lines_added=0,
            lines_removed=0,
            lines_modified=0,
            source_size=0,
            target_size=0,
            diff_ratio=0,
            structural_diff=False,
            semantic_diff=False,
            error=str(e)
        )


async def generate_diffs(
    source_paths: List[str],
    target_paths: List[str],
    params: Dict[str, Any]
) -> List[DiffGeneratorResult]:
    """
    Generate diffs for multiple file pairs.
    
    Args:
        source_paths: List of source file paths
        target_paths: List of target file paths
        params: Parameters for diff generation
        
    Returns:
        List of diff generation results
    """
    if len(source_paths) != len(target_paths):
        raise ValueError("Source and target path lists must have the same length")
    
    # Create tasks for each file pair
    tasks = []
    for source, target in zip(source_paths, target_paths):
        # Create params for this file pair
        file_params = DiffGeneratorParams(
            source_path=source,
            target_path=target,
            **params
        )
        
        # Create task
        task = generate_diff(file_params)
        tasks.append(task)
    
    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)
    
    return results


async def generate_directory_diff(
    source_dir: str,
    target_dir: str,
    params: Dict[str, Any],
    file_pattern: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate diffs for all matching files in two directories.
    
    Args:
        source_dir: Source directory
        target_dir: Target directory
        params: Parameters for diff generation
        file_pattern: Optional pattern to filter files
        
    Returns:
        Dictionary with diff results and statistics
    """
    # Find files in both directories
    source_files = set()
    target_files = set()
    
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file_pattern is None or re.match(file_pattern, file):
                rel_path = os.path.relpath(os.path.join(root, file), source_dir)
                source_files.add(rel_path)
    
    for root, _, files in os.walk(target_dir):
        for file in files:
            if file_pattern is None or re.match(file_pattern, file):
                rel_path = os.path.relpath(os.path.join(root, file), target_dir)
                target_files.add(rel_path)
    
    # Categorize files
    common_files = source_files.intersection(target_files)
    source_only = source_files - common_files
    target_only = target_files - common_files
    
    # Generate diffs for common files
    source_paths = [os.path.join(source_dir, f) for f in common_files]
    target_paths = [os.path.join(target_dir, f) for f in common_files]
    
    diff_results = await generate_diffs(source_paths, target_paths, params)
    
    # Create file-to-result mapping
    file_results = {
        file: result for file, result in zip(common_files, diff_results)
    }
    
    # Calculate statistics
    total_files = len(common_files) + len(source_only) + len(target_only)
    changed_files = sum(1 for result in diff_results if result.total_lines_changed > 0)
    unchanged_files = len(common_files) - changed_files
    
    total_lines_added = sum(result.lines_added for result in diff_results)
    total_lines_removed = sum(result.lines_removed for result in diff_results)
    total_lines_modified = sum(result.lines_modified for result in diff_results)
    
    # Return the results
    return {
        "file_results": file_results,
        "common_files": list(common_files),
        "source_only": list(source_only),
        "target_only": list(target_only),
        "statistics": {
            "total_files": total_files,
            "common_files": len(common_files),
            "source_only": len(source_only),
            "target_only": len(target_only),
            "changed_files": changed_files,
            "unchanged_files": unchanged_files,
            "total_lines_added": total_lines_added,
            "total_lines_removed": total_lines_removed,
            "total_lines_modified": total_lines_modified
        }
    }