"""
Context extraction operations for TSAP.

This module provides functionality to extract meaningful contexts around search
matches, such as code blocks, functions, paragraphs, and other logical units.
"""
import os
import re
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from tsap.utils.logging import logger
from tsap.mcp.models import (
    ContextExtractParams, ContextExtractResult
)


@dataclass
class Context:
    """A context extracted around a match."""
    
    file_path: str
    start_line: int
    end_line: int
    content: List[str]
    context_type: str
    matches: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def line_count(self) -> int:
        """Get the number of lines in this context."""
        return self.end_line - self.start_line + 1
    
    def contains_line(self, line_number: int) -> bool:
        """Check if this context contains a specific line number.
        
        Args:
            line_number: Line number to check
            
        Returns:
            Whether this context contains the line
        """
        return self.start_line <= line_number <= self.end_line
    
    def overlaps(self, other: 'Context') -> bool:
        """Check if this context overlaps with another context.
        
        Contexts overlap if they share any lines and are in the same file.
        
        Args:
            other: Another context
            
        Returns:
            Whether the contexts overlap
        """
        if self.file_path != other.file_path:
            return False
            
        return (
            (self.start_line <= other.start_line <= self.end_line) or
            (self.start_line <= other.end_line <= self.end_line) or
            (other.start_line <= self.start_line <= other.end_line) or
            (other.start_line <= self.end_line <= other.end_line)
        )
    
    def merge(self, other: 'Context') -> 'Context':
        """Merge this context with another overlapping context.
        
        Args:
            other: Another context
            
        Returns:
            Merged context
        """
        if not self.overlaps(other):
            raise ValueError("Cannot merge non-overlapping contexts")
            
        # Determine the expanded range
        start_line = min(self.start_line, other.start_line)
        end_line = max(self.end_line, other.end_line)
        
        # Determine merged context type
        context_type = self.context_type
        if self.context_type != other.context_type:
            # If types differ, use the larger context type
            if self.line_count >= other.line_count:
                context_type = self.context_type
            else:
                context_type = other.context_type
        
        # Combine matches
        all_matches = self.matches.copy()
        all_matches.extend(other.matches)
        
        # We need to fetch the content for the expanded range
        # For now, just combine the existing content (might have gaps or overlap)
        content = []
        
        # Create a mapping of line number to content line
        line_map = {}
        
        # Add lines from first context
        for i, line in enumerate(self.content):
            line_number = self.start_line + i
            line_map[line_number] = line
            
        # Add lines from second context
        for i, line in enumerate(other.content):
            line_number = other.start_line + i
            line_map[line_number] = line
            
        # Create merged content in order
        for line_number in range(start_line, end_line + 1):
            if line_number in line_map:
                content.append(line_map[line_number])
            else:
                # Missing line - use a placeholder
                content.append(f"[Missing line {line_number}]")
        
        return Context(
            file_path=self.file_path,
            start_line=start_line,
            end_line=end_line,
            content=content,
            context_type=context_type,
            matches=all_matches
        )


async def _read_file_lines(
    file_path: str, 
    start_line: Optional[int] = None,
    end_line: Optional[int] = None
) -> List[str]:
    """Read lines from a file.
    
    Args:
        file_path: Path to the file
        start_line: Optional start line (1-based, inclusive)
        end_line: Optional end line (1-based, inclusive)
        
    Returns:
        List of lines
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if start_line is None and end_line is None:
                # Read the entire file
                return f.read().splitlines()
                
            # Read only the specified lines
            if start_line is None:
                start_line = 1
            if end_line is None:
                end_line = float('inf')
                
            # Convert to 0-based index
            start_idx = start_line - 1
            
            lines = []
            for i, line in enumerate(f):
                if i >= start_idx:
                    lines.append(line.rstrip('\n'))
                if i >= end_line - 1:
                    break
                    
            return lines
    except UnicodeDecodeError:
        # Try with a different encoding
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                lines = f.read().splitlines()
                
                if start_line is None and end_line is None:
                    return lines
                    
                if start_line is None:
                    start_line = 1
                if end_line is None:
                    end_line = len(lines)
                    
                return lines[start_line-1:end_line]
        except Exception as e:
            logger.error(
                f"Failed to read file with latin-1 encoding: {str(e)}",
                component="composite",
                operation="read_file",
                exception=e
            )
            return []
    except Exception as e:
        logger.error(
            f"Failed to read file: {str(e)}",
            component="composite",
            operation="read_file",
            exception=e
        )
        return []


async def _extract_line_context(
    file_path: str,
    line_number: int,
    context_lines: int = 5
) -> Context:
    """Extract a simple line-based context around a match.
    
    Args:
        file_path: Path to the file
        line_number: Line number of the match
        context_lines: Number of context lines to include
        
    Returns:
        Extracted context
    """
    # Calculate the line range
    start_line = max(1, line_number - context_lines)
    end_line = line_number + context_lines
    
    # Read the lines
    lines = await _read_file_lines(file_path, start_line, end_line)
    
    return Context(
        file_path=file_path,
        start_line=start_line,
        end_line=start_line + len(lines) - 1,
        content=lines,
        context_type="lines"
    )


async def _extract_function_context(
    file_path: str,
    line_number: int,
    language: Optional[str] = None
) -> Optional[Context]:
    """Extract a function context around a match.
    
    Args:
        file_path: Path to the file
        line_number: Line number of the match
        language: Optional programming language
        
    Returns:
        Extracted context or None if not found
    """
    # Determine the language if not specified
    if not language:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.py', '.pyx', '.pyw']:
            language = 'python'
        elif ext in ['.js', '.jsx', '.ts', '.tsx']:
            language = 'javascript'
        elif ext in ['.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.hh', '.hxx']:
            language = 'c'
        elif ext in ['.java']:
            language = 'java'
        elif ext in ['.go']:
            language = 'go'
        elif ext in ['.rb']:
            language = 'ruby'
        elif ext in ['.php']:
            language = 'php'
        else:
            # Default to a generic pattern
            language = 'generic'
    
    # Read the entire file
    all_lines = await _read_file_lines(file_path)
    
    if not all_lines:
        return None
    
    # Function detection patterns by language
    patterns = {
        'python': {
            'start': r'^\s*(async\s+)?def\s+\w+\s*\(',
            'end': r'^\S'  # Dedent indicates end of function
        },
        'javascript': {
            'start': r'^\s*(function\s+\w+\s*\(|const\s+\w+\s*=\s*(?:async\s*)?\([^)]*\)\s*=>|class\s+\w+|[a-zA-Z_$][a-zA-Z0-9_$]*\s*\([^)]*\)\s*{)',
            'end': r'^\s*}'
        },
        'c': {
            'start': r'^\s*\w+\s+\w+\s*\([^;]*\)\s*{',
            'end': r'^\s*}'
        },
        'java': {
            'start': r'^\s*(public|private|protected|static|\s)+[\w\<\>\[\]]+\s+\w+\s*\([^\)]*\)\s*(\{|[^;])',
            'end': r'^\s*}'
        },
        'go': {
            'start': r'^\s*func\s+\w+\s*\(',
            'end': r'^\s*}'
        },
        'ruby': {
            'start': r'^\s*(def\s+\w+|class\s+\w+)',
            'end': r'^\s*end'
        },
        'php': {
            'start': r'^\s*(function\s+\w+\s*\(|class\s+\w+)',
            'end': r'^\s*}'
        },
        'generic': {
            'start': r'^\s*(\w+\s+)?\w+\s*\([^;]*\)\s*\{',
            'end': r'^\s*}'
        }
    }
    
    pattern = patterns.get(language, patterns['generic'])
    
    # Find function boundaries
    function_start = None
    function_end = None
    
    # First, look backward from the match line to find function start
    for i in range(line_number - 1, -1, -1):
        if re.match(pattern['start'], all_lines[i]):
            function_start = i + 1  # Convert to 1-based
            break
    
    # If we found a start, look forward for function end
    if function_start is not None:
        # For Python, need to track indentation level
        if language == 'python':
            # Get the indentation of the function definition line
            def_line = all_lines[function_start - 1]
            def_indent = len(def_line) - len(def_line.lstrip())
            
            # Find the first line with less indentation
            for i in range(function_start, len(all_lines)):
                line = all_lines[i]
                # Skip empty lines
                if not line.strip():
                    continue
                    
                # Check indentation
                indent = len(line) - len(line.lstrip())
                if indent <= def_indent:
                    function_end = i  # Convert to 1-based
                    break
            
            # If we reach the end of the file, that's the end of the function
            if function_end is None:
                function_end = len(all_lines)
        else:
            # For other languages, look for the end pattern
            # Also track brace/bracket matching for C-like languages
            if language in ['javascript', 'c', 'java', 'go', 'php', 'generic']:
                brace_level = 0
                found_first_brace = False
                
                for i in range(function_start - 1, len(all_lines)):
                    line = all_lines[i]
                    
                    # Count opening braces
                    brace_level += line.count('{')
                    
                    # Mark that we've found at least one brace
                    if not found_first_brace and '{' in line:
                        found_first_brace = True
                    
                    # Count closing braces
                    brace_level -= line.count('}')
                    
                    # Check if we're back to level 0 after finding at least one brace
                    if found_first_brace and brace_level == 0 and '}' in line:
                        function_end = i + 1  # Convert to 1-based
                        break
            else:
                # For other languages, just look for the end pattern
                for i in range(function_start, len(all_lines)):
                    if re.match(pattern['end'], all_lines[i]):
                        function_end = i + 1  # Convert to 1-based
                        break
            
            # If we reach the end of the file, that's the end of the function
            if function_end is None:
                function_end = len(all_lines)
    
    # If we couldn't find a function, return None
    if function_start is None or function_end is None:
        return None
    
    # Get the function content
    function_lines = all_lines[function_start - 1:function_end]
    
    return Context(
        file_path=file_path,
        start_line=function_start,
        end_line=function_end,
        content=function_lines,
        context_type="function"
    )


async def _extract_block_context(
    file_path: str,
    line_number: int,
    language: Optional[str] = None
) -> Optional[Context]:
    """Extract a code block context around a match.
    
    Args:
        file_path: Path to the file
        line_number: Line number of the match
        language: Optional programming language
        
    Returns:
        Extracted context or None if not found
    """
    # Determine the language if not specified
    if not language:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.py', '.pyx', '.pyw']:
            language = 'python'
        elif ext in ['.js', '.jsx', '.ts', '.tsx']:
            language = 'javascript'
        elif ext in ['.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.hh', '.hxx']:
            language = 'c'
        elif ext in ['.java']:
            language = 'java'
        elif ext in ['.go']:
            language = 'go'
        elif ext in ['.rb']:
            language = 'ruby'
        elif ext in ['.php']:
            language = 'php'
        else:
            # Default to a generic pattern
            language = 'generic'
    
    # Read the entire file
    all_lines = await _read_file_lines(file_path)
    
    if not all_lines:
        return None
    
    # Convert line_number to 0-based index
    line_idx = line_number - 1
    
    # Block detection based on language
    if language == 'python':
        # For Python, blocks are determined by indentation
        # Get the indentation of the match line
        if line_idx >= len(all_lines):
            return None
            
        match_line = all_lines[line_idx]
        match_indent = len(match_line) - len(match_line.lstrip())
        
        # Find the start of the block (first line with same or less indentation)
        block_start = line_idx
        for i in range(line_idx - 1, -1, -1):
            line = all_lines[i]
            # Skip empty lines
            if not line.strip():
                continue
                
            # Check indentation
            indent = len(line) - len(line.lstrip())
            if indent <= match_indent:
                block_start = i
                break
        
        # Find the end of the block (next line with same or less indentation)
        block_end = line_idx
        for i in range(line_idx + 1, len(all_lines)):
            line = all_lines[i]
            # Skip empty lines
            if not line.strip():
                continue
                
            # Check indentation
            indent = len(line) - len(line.lstrip())
            if indent <= match_indent:
                block_end = i - 1
                break
            else:
                block_end = i
    else:
        # For most other languages, blocks are defined by braces
        # Find the nearest opening brace before the match
        block_start = None
        brace_level = 0
        
        for i in range(line_idx, -1, -1):
            line = all_lines[i]
            
            # Count closing braces (going backward)
            brace_level += line.count('}')
            
            # Count opening braces (going backward)
            brace_level -= line.count('{')
            
            # If we find an opening brace when level is 1, that's our start
            if '{' in line and brace_level == 0:
                block_start = i
                break
        
        # If we couldn't find a starting brace, use the match line
        if block_start is None:
            block_start = line_idx
        
        # Find the matching closing brace after the match
        block_end = None
        brace_level = 0
        
        for i in range(block_start, len(all_lines)):
            line = all_lines[i]
            
            # Count opening braces
            brace_level += line.count('{')
            
            # Count closing braces
            brace_level -= line.count('}')
            
            # If we find a closing brace that brings us back to level 0, that's our end
            if '}' in line and brace_level == 0:
                block_end = i
                break
        
        # If we couldn't find an ending brace, use the match line
        if block_end is None:
            block_end = line_idx
    
    # Get the block content
    block_lines = all_lines[block_start:block_end + 1]
    
    # Convert back to 1-based line numbers
    return Context(
        file_path=file_path,
        start_line=block_start + 1,
        end_line=block_end + 1,
        content=block_lines,
        context_type="block"
    )


async def _extract_paragraph_context(
    file_path: str,
    line_number: int
) -> Optional[Context]:
    """Extract a paragraph context around a match.
    
    Args:
        file_path: Path to the file
        line_number: Line number of the match
        
    Returns:
        Extracted context or None if not found
    """
    # Read the entire file
    all_lines = await _read_file_lines(file_path)
    
    if not all_lines:
        return None
    
    # Convert line_number to 0-based index
    line_idx = line_number - 1
    
    if line_idx >= len(all_lines):
        return None
    
    # Find the start of the paragraph (first empty line before match)
    para_start = line_idx
    for i in range(line_idx - 1, -1, -1):
        if not all_lines[i].strip():
            para_start = i + 1
            break
    
    # Find the end of the paragraph (next empty line after match)
    para_end = line_idx
    for i in range(line_idx + 1, len(all_lines)):
        if not all_lines[i].strip():
            para_end = i - 1
            break
        else:
            para_end = i
    
    # Get the paragraph content
    para_lines = all_lines[para_start:para_end + 1]
    
    # Convert back to 1-based line numbers
    return Context(
        file_path=file_path,
        start_line=para_start + 1,
        end_line=para_end + 1,
        content=para_lines,
        context_type="paragraph"
    )


async def _extract_section_context(
    file_path: str,
    line_number: int
) -> Optional[Context]:
    """Extract a section context around a match.
    
    Args:
        file_path: Path to the file
        line_number: Line number of the match
        
    Returns:
        Extracted context or None if not found
    """
    # Read the entire file
    all_lines = await _read_file_lines(file_path)
    
    if not all_lines:
        return None
    
    # Convert line_number to 0-based index
    line_idx = line_number - 1
    
    if line_idx >= len(all_lines):
        return None
    
    # Look for section headers (lines starting with #, ==, --, etc.)
    section_header_patterns = [
        r'^\s*#{1,6}\s+\w+',  # Markdown headers
        r'^\s*[A-Z][A-Z0-9 ]+$',  # ALL CAPS headers
        r'^\s*[A-Z][a-zA-Z0-9 ]+:$',  # Title: headers
        r'^\s*[=-]{3,}$',  # === or --- underlines
        r'^\s*\*{3,}$',  # *** separators
    ]
    
    # Find the start of the section (previous section header)
    section_start = 0
    for i in range(line_idx - 1, -1, -1):
        line = all_lines[i]
        if any(re.match(pattern, line) for pattern in section_header_patterns):
            section_start = i
            break
    
    # Find the end of the section (next section header)
    section_end = len(all_lines) - 1
    for i in range(line_idx + 1, len(all_lines)):
        line = all_lines[i]
        if any(re.match(pattern, line) for pattern in section_header_patterns):
            section_end = i - 1
            break
    
    # Get the section content
    section_lines = all_lines[section_start:section_end + 1]
    
    # Convert back to 1-based line numbers
    return Context(
        file_path=file_path,
        start_line=section_start + 1,
        end_line=section_end + 1,
        content=section_lines,
        context_type="section"
    )


async def _extract_file_context(
    file_path: str,
    line_number: int,
    max_lines: Optional[int] = None
) -> Context:
    """Extract the entire file as context.
    
    Args:
        file_path: Path to the file
        line_number: Line number of the match
        max_lines: Maximum number of lines to include
        
    Returns:
        Extracted context
    """
    # Read the entire file
    all_lines = await _read_file_lines(file_path)
    
    # Limit the number of lines if specified
    if max_lines and len(all_lines) > max_lines:
        # Extract lines centered around the match
        half_max = max_lines // 2
        start_idx = max(0, line_number - half_max - 1)
        end_idx = min(len(all_lines), start_idx + max_lines)
        
        # Adjust start if we're near the end of the file
        if end_idx == len(all_lines):
            start_idx = max(0, end_idx - max_lines)
            
        all_lines = all_lines[start_idx:end_idx]
        return Context(
            file_path=file_path,
            start_line=start_idx + 1,
            end_line=end_idx,
            content=all_lines,
            context_type="file_excerpt"
        )
    
    return Context(
        file_path=file_path,
        start_line=1,
        end_line=len(all_lines),
        content=all_lines,
        context_type="file"
    )


async def _extract_context_for_match(
    match: Dict[str, Any],
    context_type: str,
    context_lines: Optional[int] = None,
    include_file_context: bool = False
) -> List[Context]:
    """Extract context for a single match.
    
    Args:
        match: Match dictionary with file and line information
        context_type: Type of context to extract
        context_lines: Optional number of context lines
        include_file_context: Whether to include file-level context
        
    Returns:
        List of extracted contexts
    """
    file_path = match.get('path')
    line_number = match.get('line_number')
    
    if not file_path or not line_number:
        logger.warning(
            "Match missing file path or line number",
            component="composite",
            operation="extract_context"
        )
        return []
    
    contexts = []
    
    # Extract the specified context type
    context = None
    
    if context_type == "lines":
        # Simple line-based context
        lines = context_lines or 5
        context = await _extract_line_context(file_path, line_number, lines)
    elif context_type == "function":
        # Function context
        context = await _extract_function_context(file_path, line_number)
    elif context_type == "block":
        # Code block context
        context = await _extract_block_context(file_path, line_number)
    elif context_type == "paragraph":
        # Paragraph context
        context = await _extract_paragraph_context(file_path, line_number)
    elif context_type == "section":
        # Section context
        context = await _extract_section_context(file_path, line_number)
    elif context_type == "file":
        # Entire file context
        max_lines = 1000  # Limit to 1000 lines to avoid huge contexts
        context = await _extract_file_context(file_path, line_number, max_lines)
    
    # Add the match to the context
    if context:
        context.matches.append(match)
        contexts.append(context)
    
    # Add file-level context if requested
    if include_file_context and context_type != "file":
        file_context = await _extract_file_context(file_path, line_number)
        
        # Don't add file context if it's identical to the primary context
        if not context or context.line_count < file_context.line_count:
            # Add the match to the file context
            file_context.matches.append(match)
            contexts.append(file_context)
    
    return contexts


def _consolidate_contexts(contexts: List[Context]) -> List[Context]:
    """Consolidate overlapping contexts.
    
    Args:
        contexts: List of contexts to consolidate
        
    Returns:
        Consolidated contexts
    """
    if not contexts:
        return []
    
    # Group contexts by file
    contexts_by_file = {}
    for ctx in contexts:
        if ctx.file_path not in contexts_by_file:
            contexts_by_file[ctx.file_path] = []
        contexts_by_file[ctx.file_path].append(ctx)
    
    # Consolidate contexts within each file
    consolidated = []
    
    for file_path, file_contexts in contexts_by_file.items():
        # Sort contexts by start line
        sorted_contexts = sorted(file_contexts, key=lambda c: c.start_line)
        
        # Merge overlapping contexts
        merged = [sorted_contexts[0]]
        
        for current in sorted_contexts[1:]:
            last = merged[-1]
            
            if last.overlaps(current):
                # Merge with the last context
                merged[-1] = last.merge(current)
            else:
                # Add as a new context
                merged.append(current)
        
        consolidated.extend(merged)
    
    return consolidated


async def extract_context(params: ContextExtractParams) -> ContextExtractResult:
    """Extract context around matches.
    
    Args:
        params: Context extraction parameters
        
    Returns:
        Extracted contexts
    """
    start_time = asyncio.get_event_loop().time()
    
    # Log the operation
    logger.info(
        f"Extracting {params.context_type} context for {len(params.matches)} matches",
        component="composite",
        operation="extract_context",
        context={
            "match_count": len(params.matches),
            "context_type": params.context_type,
        }
    )
    
    # Extract context for each match
    all_contexts = []
    
    for match in params.matches:
        contexts = await _extract_context_for_match(
            match=match,
            context_type=params.context_type,
            context_lines=params.context_lines,
            include_file_context=params.include_file_context
        )
        all_contexts.extend(contexts)
    
    # Consolidate overlapping contexts if requested
    contexts = all_contexts
    consolidated_count = 0
    
    if params.consolidate_contexts and all_contexts:
        original_count = len(all_contexts)
        contexts = _consolidate_contexts(all_contexts)
        consolidated_count = original_count - len(contexts)
        
        logger.info(
            f"Consolidated {original_count} contexts into {len(contexts)} contexts",
            component="composite",
            operation="extract_context"
        )
    
    # Convert contexts to dictionaries
    context_dicts = []
    
    for ctx in contexts:
        context_dict = {
            "file_path": ctx.file_path,
            "start_line": ctx.start_line,
            "end_line": ctx.end_line,
            "line_count": ctx.line_count,
            "content": ctx.content,
            "context_type": ctx.context_type,
            "matches": ctx.matches,
        }
        context_dicts.append(context_dict)
    
    # Calculate execution time
    execution_time = asyncio.get_event_loop().time() - start_time
    
    # Log completion
    logger.success(
        f"Context extraction completed: {len(contexts)} contexts",
        component="composite",
        operation="extract_context",
        context={
            "context_count": len(contexts),
            "consolidated_count": consolidated_count,
            "execution_time": execution_time,
        }
    )
    
    # Create and return result
    return ContextExtractResult(
        contexts=context_dicts,
        consolidated_count=consolidated_count,
        execution_time=execution_time,
    )