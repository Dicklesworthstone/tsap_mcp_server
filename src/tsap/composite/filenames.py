"""
Filename Pattern Discoverer for the TSAP MCP Server.

This module implements tools for discovering patterns and conventions in filenames
within a directory structure, including prefixes, suffixes, naming conventions,
and organization schemes.
"""

import os
import re
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set, Counter as CounterType
from dataclasses import dataclass, field
from collections import Counter, defaultdict

from tsap.utils.logging import logger
from tsap.mcp.models import FilenamePatternParams, FilenamePatternResult


@dataclass
class PatternCandidate:
    """
    Represents a candidate pattern discovered in filenames.
    """
    pattern: str
    regex: str
    description: str
    examples: List[str] = field(default_factory=list)
    match_count: int = 0
    confidence: float = 0.0


@dataclass
class DirectoryStructure:
    """
    Represents a discovered directory structure pattern.
    """
    path: str
    depth: int
    pattern: str
    description: str
    examples: List[str] = field(default_factory=list)
    file_count: int = 0
    subdir_count: int = 0


async def _list_files(
    directory: str,
    recursive: bool = True,
    exclude_patterns: Optional[List[str]] = None,
    max_depth: Optional[int] = None,
    include_hidden: bool = False
) -> List[str]:
    """
    List files in a directory.
    
    Args:
        directory: Directory to list files from
        recursive: Whether to recursively traverse subdirectories
        exclude_patterns: Patterns to exclude
        max_depth: Maximum directory depth to traverse
        include_hidden: Whether to include hidden files/directories
        
    Returns:
        List of file paths
    """
    files = []
    exclude_regex = None
    
    if exclude_patterns:
        exclude_regex = re.compile('|'.join(exclude_patterns))
    
    for root, dirs, filenames in os.walk(directory):
        # Calculate current depth
        current_depth = root.count(os.path.sep) - directory.count(os.path.sep)
        
        # Skip if we've reached max depth
        if max_depth is not None and current_depth >= max_depth:
            dirs.clear()  # Clear dirs to prevent further recursion
            continue
        
        # Filter out hidden directories if needed
        if not include_hidden:
            dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for filename in filenames:
            # Skip hidden files if needed
            if not include_hidden and filename.startswith('.'):
                continue
            
            # Skip excluded patterns
            if exclude_regex and exclude_regex.search(filename):
                continue
            
            files.append(os.path.join(root, filename))
        
        # Stop recursion if not requested
        if not recursive:
            break
    
    return files


def _extract_filename_parts(path: str) -> Dict[str, str]:
    """
    Extract parts of a filename.
    
    Args:
        path: File path
        
    Returns:
        Dictionary with filename parts
    """
    # Extract various parts of the path
    dirname = os.path.dirname(path)
    basename = os.path.basename(path)
    filename, ext = os.path.splitext(basename)
    
    # Get parent directory name
    parent_dir = os.path.basename(dirname) if dirname else ""
    
    # Clean up extension (remove leading dot)
    ext = ext[1:] if ext.startswith('.') else ext
    
    # Split filename into components
    # Check for common separators: underscore, dash, dot, camelCase
    components = []
    
    # First try explicit separators
    if '_' in filename:
        components = filename.split('_')
    elif '-' in filename:
        components = filename.split('-')
    elif '.' in filename:
        components = filename.split('.')
    else:
        # Try to split camelCase
        components = re.findall(r'[A-Z]?[a-z0-9]+', filename)
    
    return {
        "path": path,
        "dirname": dirname,
        "basename": basename,
        "filename": filename,
        "extension": ext,
        "parent_dir": parent_dir,
        "components": components
    }


async def _analyze_extensions(file_parts: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Analyze file extensions.
    
    Args:
        file_parts: List of dictionaries with filename parts
        
    Returns:
        Dictionary with extension analysis
    """
    # Count extensions
    extensions = Counter(part["extension"].lower() for part in file_parts if part["extension"])
    
    # Group extensions by type
    extension_groups = defaultdict(list)
    
    # Define extension groups
    groups = {
        "source_code": ["py", "js", "ts", "java", "c", "cpp", "h", "go", "rb", "php"],
        "web": ["html", "htm", "css", "scss", "less", "svg", "jsx", "tsx"],
        "data": ["json", "csv", "tsv", "xml", "yaml", "yml", "toml"],
        "document": ["md", "txt", "rst", "pdf", "doc", "docx", "odt", "rtf"],
        "image": ["jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp", "svg"],
        "video": ["mp4", "avi", "mov", "wmv", "flv", "mkv", "webm"],
        "audio": ["mp3", "wav", "ogg", "flac", "aac", "wma"],
        "archive": ["zip", "tar", "gz", "bz2", "7z", "rar"]
    }
    
    # Categorize extensions
    for ext, count in extensions.items():
        categorized = False
        for group, exts in groups.items():
            if ext in exts:
                extension_groups[group].append((ext, count))
                categorized = True
                break
        
        if not categorized:
            extension_groups["other"].append((ext, count))
    
    # Sort groups by total count
    sorted_groups = {}
    for group, exts in extension_groups.items():
        total = sum(count for _, count in exts)
        sorted_groups[group] = {
            "extensions": exts,
            "total": total
        }
    
    # Sort groups by total
    sorted_groups = dict(sorted(sorted_groups.items(), key=lambda x: x[1]["total"], reverse=True))
    
    return {
        "extension_counts": dict(extensions),
        "grouped_extensions": sorted_groups,
        "unique_extensions": len(extensions)
    }


async def _analyze_naming_patterns(file_parts: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Analyze naming patterns in filenames.
    
    Args:
        file_parts: List of dictionaries with filename parts
        
    Returns:
        Dictionary with naming pattern analysis
    """
    # Analyze common prefixes
    prefixes = Counter()
    for part in file_parts:
        # Use the first component as a potential prefix
        if part["components"] and len(part["components"]) > 1:
            prefixes[part["components"][0]] += 1
    
    # Analyze common suffixes
    suffixes = Counter()
    for part in file_parts:
        # Use the last component as a potential suffix
        if part["components"] and len(part["components"]) > 1:
            suffixes[part["components"][-1]] += 1
    
    # Analyze naming conventions
    conventions = Counter()
    for part in file_parts:
        filename = part["filename"]
        
        # Check for snake_case
        if re.match(r'^[a-z0-9]+(_[a-z0-9]+)*$', filename):
            conventions["snake_case"] += 1
        
        # Check for kebab-case
        elif re.match(r'^[a-z0-9]+(-[a-z0-9]+)*$', filename):
            conventions["kebab-case"] += 1
        
        # Check for camelCase
        elif re.match(r'^[a-z][a-zA-Z0-9]*$', filename) and any(c.isupper() for c in filename):
            conventions["camelCase"] += 1
        
        # Check for PascalCase
        elif re.match(r'^[A-Z][a-zA-Z0-9]*$', filename):
            conventions["PascalCase"] += 1
        
        # Check for UPPER_CASE
        elif re.match(r'^[A-Z0-9]+(_[A-Z0-9]+)*$', filename):
            conventions["UPPER_CASE"] += 1
        
        # Check for dot.case
        elif '.' in filename and not filename.endswith('.'):
            conventions["dot.case"] += 1
        
        # Other
        else:
            conventions["other"] += 1
    
    # Analyze component counts
    component_counts = Counter(len(part["components"]) for part in file_parts)
    
    # Identify common patterns
    pattern_candidates = []
    
    # Look for numeric patterns
    numeric_pattern = re.compile(r'^\d+$')
    numeric_files = [part for part in file_parts if any(numeric_pattern.match(comp) for comp in part["components"])]
    
    if numeric_files:
        pattern_candidates.append(PatternCandidate(
            pattern="numeric_component",
            regex=r".*?\d+.*?",
            description="Files with numeric components (e.g., version numbers, dates)",
            examples=[part["basename"] for part in numeric_files[:5]],
            match_count=len(numeric_files),
            confidence=len(numeric_files) / len(file_parts) if file_parts else 0
        ))
    
    # Look for date patterns
    date_pattern = re.compile(r'^\d{4}[_\-\.]\d{1,2}[_\-\.]\d{1,2}$|^\d{1,2}[_\-\.]\d{1,2}[_\-\.]\d{4}$')
    date_files = [part for part in file_parts if any(date_pattern.match(comp) for comp in part["components"])]
    
    if date_files:
        pattern_candidates.append(PatternCandidate(
            pattern="date_component",
            regex=r".*?\d{4}[_\-\.]\d{1,2}[_\-\.]\d{1,2}.*?|.*?\d{1,2}[_\-\.]\d{1,2}[_\-\.]\d{4}.*?",
            description="Files with date components",
            examples=[part["basename"] for part in date_files[:5]],
            match_count=len(date_files),
            confidence=len(date_files) / len(file_parts) if file_parts else 0
        ))
    
    # Common separators
    for separator, name in [('_', 'underscore'), ('-', 'dash'), ('.', 'dot')]:
        sep_files = [part for part in file_parts if separator in part["filename"]]
        if len(sep_files) > len(file_parts) * 0.2:  # At least 20%
            pattern_candidates.append(PatternCandidate(
                pattern=f"{name}_separator",
                regex=f".*?{re.escape(separator)}.*?",
                description=f"Files using {name} as separator",
                examples=[part["basename"] for part in sep_files[:5]],
                match_count=len(sep_files),
                confidence=len(sep_files) / len(file_parts) if file_parts else 0
            ))
    
    # Common prefixes (if used in more than 20% of files)
    for prefix, count in prefixes.items():
        if count > len(file_parts) * 0.2:
            prefix_files = [part for part in file_parts if part["components"] and part["components"][0] == prefix]
            pattern_candidates.append(PatternCandidate(
                pattern=f"prefix_{prefix}",
                regex=f"^{re.escape(prefix)}[_\-\.].*",
                description=f"Files with '{prefix}' prefix",
                examples=[part["basename"] for part in prefix_files[:5]],
                match_count=count,
                confidence=count / len(file_parts) if file_parts else 0
            ))
    
    # Common suffixes (if used in more than 20% of files)
    for suffix, count in suffixes.items():
        if count > len(file_parts) * 0.2:
            suffix_files = [part for part in file_parts if part["components"] and part["components"][-1] == suffix]
            pattern_candidates.append(PatternCandidate(
                pattern=f"suffix_{suffix}",
                regex=f".*[_\-\.]{re.escape(suffix)}$",
                description=f"Files with '{suffix}' suffix",
                examples=[part["basename"] for part in suffix_files[:5]],
                match_count=count,
                confidence=count / len(file_parts) if file_parts else 0
            ))
    
    return {
        "common_prefixes": dict(prefixes.most_common(10)),
        "common_suffixes": dict(suffixes.most_common(10)),
        "naming_conventions": dict(conventions),
        "component_counts": dict(component_counts),
        "pattern_candidates": [vars(candidate) for candidate in pattern_candidates]
    }


async def _analyze_directory_structure(
    files: List[str],
    base_dir: str
) -> Dict[str, Any]:
    """
    Analyze directory structure.
    
    Args:
        files: List of file paths
        base_dir: Base directory
        
    Returns:
        Dictionary with directory structure analysis
    """
    # Get all directories
    directories = set()
    for file_path in files:
        dir_path = os.path.dirname(file_path)
        if dir_path != base_dir:  # Skip the base directory
            directories.add(dir_path)
    
    # Count files per directory
    files_per_dir = Counter(os.path.dirname(file_path) for file_path in files)
    
    # Analyze directory depths
    dir_depths = {}
    for dir_path in directories:
        # Calculate depth relative to base_dir
        rel_path = os.path.relpath(dir_path, base_dir)
        depth = rel_path.count(os.path.sep) + 1  # Add 1 for the directory itself
        dir_depths[dir_path] = depth
    
    # Find common depth patterns
    depth_counts = Counter(dir_depths.values())
    avg_depth = sum(dir_depths.values()) / len(dir_depths) if dir_depths else 0
    
    # Identify directory structure patterns
    structure_patterns = []
    
    # Simple check for flat vs nested
    flat_ratio = depth_counts.get(1, 0) / len(dir_depths) if dir_depths else 0
    if flat_ratio > 0.7:  # More than 70% of directories are at depth 1
        structure_patterns.append(DirectoryStructure(
            path=base_dir,
            depth=1,
            pattern="flat",
            description="Flat directory structure with most files in top-level directories",
            examples=[os.path.relpath(d, base_dir) for d in directories if dir_depths[d] == 1][:5],
            file_count=sum(files_per_dir[d] for d in directories if dir_depths[d] == 1),
            subdir_count=sum(1 for d in directories if dir_depths[d] == 1)
        ))
    elif max(depth_counts.keys()) > 3:  # Directories deeper than 3 levels
        structure_patterns.append(DirectoryStructure(
            path=base_dir,
            depth=max(depth_counts.keys()),
            pattern="deeply_nested",
            description="Deeply nested directory structure",
            examples=[os.path.relpath(d, base_dir) for d in directories if dir_depths[d] > 3][:5],
            file_count=sum(files_per_dir[d] for d in directories if dir_depths[d] > 3),
            subdir_count=sum(1 for d in directories if dir_depths[d] > 3)
        ))
    else:
        structure_patterns.append(DirectoryStructure(
            path=base_dir,
            depth=2,
            pattern="moderately_nested",
            description="Moderately nested directory structure",
            examples=[os.path.relpath(d, base_dir) for d in directories if dir_depths[d] == 2][:5],
            file_count=sum(files_per_dir[d] for d in directories if dir_depths[d] == 2),
            subdir_count=sum(1 for d in directories if dir_depths[d] == 2)
        ))
    
    # Check for type-based organization
    # Count files by extension in each directory
    ext_by_dir = defaultdict(Counter)
    for file_path in files:
        dir_path = os.path.dirname(file_path)
        ext = os.path.splitext(file_path)[1][1:].lower()  # Remove leading dot
        ext_by_dir[dir_path][ext] += 1
    
    # Check for directories with mostly one type of file
    type_focused_dirs = []
    for dir_path, ext_counts in ext_by_dir.items():
        total = sum(ext_counts.values())
        if total > 0:
            main_ext, main_count = ext_counts.most_common(1)[0]
            if main_count / total > 0.8 and total > 5:  # 80% of files are same type and at least 5 files
                type_focused_dirs.append((dir_path, main_ext, main_count, total))
    
    if len(type_focused_dirs) > len(directories) * 0.3:  # At least 30% of directories are type-focused
        structure_patterns.append(DirectoryStructure(
            path=base_dir,
            depth=0,  # Not applicable
            pattern="type_based",
            description="Type-based organization with directories focusing on specific file types",
            examples=[f"{os.path.relpath(d, base_dir)} ({ext})" for d, ext, _, _ in type_focused_dirs[:5]],
            file_count=sum(total for _, _, _, total in type_focused_dirs),
            subdir_count=len(type_focused_dirs)
        ))
    
    # Check for feature-based organization (e.g., src, tests, docs)
    common_dir_names = Counter(os.path.basename(d) for d in directories)
    feature_dirs = ["src", "tests", "test", "docs", "doc", "examples", "lib", "include"]
    found_feature_dirs = [d for d in feature_dirs if d in common_dir_names]
    
    if found_feature_dirs:
        structure_patterns.append(DirectoryStructure(
            path=base_dir,
            depth=0,  # Not applicable
            pattern="feature_based",
            description="Feature-based organization with directories for specific purposes",
            examples=found_feature_dirs,
            file_count=sum(files_per_dir[os.path.join(base_dir, d)] for d in found_feature_dirs if os.path.join(base_dir, d) in files_per_dir),
            subdir_count=len(found_feature_dirs)
        ))
    
    return {
        "directory_count": len(directories),
        "files_per_directory": dict(files_per_dir),
        "directory_depths": dir_depths,
        "depth_counts": dict(depth_counts),
        "avg_depth": avg_depth,
        "structure_patterns": [vars(pattern) for pattern in structure_patterns]
    }


async def discover_filename_patterns(params: FilenamePatternParams) -> FilenamePatternResult:
    """
    Discover patterns in filenames.
    
    Args:
        params: Parameters for filename pattern discovery
        
    Returns:
        Results of filename pattern discovery
    """
    try:
        # List files
        files = await _list_files(
            params.directory_path,
            params.recursive,
            params.exclude_patterns,
            params.max_depth,
            params.include_hidden
        )
        
        # Extract filename parts
        file_parts = [_extract_filename_parts(file) for file in files]
        
        # Analyze extensions
        extension_analysis = await _analyze_extensions(file_parts)
        
        # Analyze naming patterns
        naming_analysis = await _analyze_naming_patterns(file_parts)
        
        # Analyze directory structure
        structure_analysis = await _analyze_directory_structure(files, params.directory_path)
        
        # Compile results
        return FilenamePatternResult(
            directory_path=params.directory_path,
            file_count=len(files),
            extension_analysis=extension_analysis,
            naming_patterns=naming_analysis,
            directory_structure=structure_analysis,
            error=None
        )
    
    except Exception as e:
        logger.error(
            f"Error discovering filename patterns: {str(e)}",
            component="composite",
            operation="filename_patterns"
        )
        
        # Return error result
        return FilenamePatternResult(
            directory_path=params.directory_path,
            file_count=0,
            extension_analysis={},
            naming_patterns={},
            directory_structure={},
            error=str(e)
        )


async def discover_patterns_in_files(
    directory_path: str,
    recursive: bool = True,
    file_types: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Discover patterns in a set of files.
    
    Args:
        directory_path: Path to directory
        recursive: Whether to recursively search
        file_types: Types of files to include
        exclude_patterns: Patterns to exclude
        
    Returns:
        Dictionary with discovered patterns
    """
    # Create parameters
    params = FilenamePatternParams(
        directory_path=directory_path,
        recursive=recursive,
        file_patterns=[f"*.{ft}" for ft in file_types] if file_types else None,
        exclude_patterns=exclude_patterns,
        max_depth=None,
        include_hidden=False
    )
    
    # Discover patterns
    result = await discover_filename_patterns(params)
    
    return result.dict()