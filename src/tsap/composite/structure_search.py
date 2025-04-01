"""
Structure Search module for the TSAP MCP Server.

This module implements search capabilities that leverage document structure information
rather than just textual content, enabling more precise and context-aware searching.
"""

import os
import re
import time
from typing import Dict, List, Any, Optional, Generator
from dataclasses import dataclass, field

from tsap.utils.logging import logger
from tsap.composite.structure import (
    ElementType,
    StructuralElement,
    DocumentStructure,
    StructuralModel
)
from tsap.mcp.models import StructureSearchParams, StructureSearchResult


@dataclass
class StructuralMatch:
    """
    Represents a match within a structural element.
    """
    file_path: str
    element_id: str
    element_type: ElementType
    element_content: str
    match_line: int
    match_text: str
    context_text: Optional[str] = None
    confidence: float = 1.0
    parent_elements: List[Dict[str, Any]] = field(default_factory=list)
    siblings: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "file_path": self.file_path,
            "element_id": self.element_id,
            "element_type": self.element_type.value,
            "element_content": self.element_content[:100] + "..." if len(self.element_content) > 100 else self.element_content,
            "match_line": self.match_line,
            "match_text": self.match_text,
            "context_text": self.context_text,
            "confidence": self.confidence,
            "parent_elements": self.parent_elements,
            "siblings": self.siblings
        }


async def _build_structural_model(files: List[str]) -> StructuralModel:
    """
    Build a structural model from a set of files.
    
    Args:
        files: List of file paths
        
    Returns:
        Structural model
    """
    model = StructuralModel()
    
    for file_path in files:
        # Detect file type
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        # For demonstration, we'll implement simplified structure parsing
        # A real implementation would use proper parsers for different file types
        
        try:
            # Read the file
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Create document root
            root = StructuralElement(
                element_type=ElementType.DOCUMENT,
                position=None,  # Will be set by the parser
                content="",  # Will be filled by the parser
                element_id=f"doc:{os.path.basename(file_path)}",
                source_file=file_path
            )
            
            # Parse structure based on file type
            if ext in ['.py', '.js', '.java', '.c', '.cpp']:
                document = await _parse_code_structure(file_path, content, root)
            elif ext in ['.md', '.txt', '.rst']:
                document = await _parse_text_structure(file_path, content, root)
            elif ext in ['.html', '.htm', '.xml']:
                document = await _parse_markup_structure(file_path, content, root)
            elif ext in ['.json', '.yaml', '.yml']:
                document = await _parse_data_structure(file_path, content, root)
            else:
                # Generic structure parsing
                document = await _parse_generic_structure(file_path, content, root)
            
            # Add document to model
            model.add_document(document)
            
        except Exception as e:
            logger.error(
                f"Error building structural model for {file_path}: {str(e)}",
                component="composite",
                operation="structure_search"
            )
    
    return model


async def _parse_code_structure(
    file_path: str,
    content: str,
    root: StructuralElement
) -> DocumentStructure:
    """
    Parse structure from a code file.
    
    Args:
        file_path: Path to the file
        content: File content
        root: Root element
        
    Returns:
        Document structure
    """
    # Split into lines
    lines = content.split('\n')
    root.content = content
    
    # Set root position
    root.position = None  # Will be set later
    
    # Create a simplified structure
    # This would be much more sophisticated in a real implementation
    
    # Process lines to identify structural elements
    current_element = root
    element_stack = [root]
    current_line = 0  # noqa: F841
    
    for line_idx, line in enumerate(lines):
        line_text = line.strip()
        
        # Skip empty lines
        if not line_text:
            continue
        
        # Check for structural elements based on indentation
        indentation = len(line) - len(line.lstrip())
        
        # Simple heuristic: indentation change indicates structure change
        if element_stack[-1] != root:
            prev_indentation = len(lines[element_stack[-1].position.start_line]) - len(lines[element_stack[-1].position.start_line].lstrip())
            
            # End of block if indentation decreases
            if indentation <= prev_indentation:
                # Pop from stack until we find a matching indentation level
                while (len(element_stack) > 1 and 
                       len(lines[element_stack[-1].position.start_line]) - len(lines[element_stack[-1].position.start_line].lstrip()) >= indentation):
                    ended_element = element_stack.pop()
                    ended_element.position.end_line = line_idx - 1
                    ended_element.content = '\n'.join(lines[ended_element.position.start_line:ended_element.position.end_line+1])
                
                current_element = element_stack[-1]
        
        # Check for class definitions
        if re.match(r'^\s*(class|interface)\s+\w+', line):
            class_name = re.search(r'(class|interface)\s+(\w+)', line).group(2)
            
            class_element = StructuralElement(
                element_type=ElementType.CLASS_DEF,
                position=None,  # Will be set below
                content="",  # Will be set when the block ends
                element_id=f"class:{class_name}",
                source_file=file_path
            )
            
            class_element.position = None  # Will be set after initializing it
            class_element.position = None  # Will be set after initializing it
            
            # Add to parent
            current_element.add_child(class_element)
            
            # Update current element and stack
            current_element = class_element
            element_stack.append(class_element)
            
            # Set position
            class_element.position = type('Position', (), {
                'start_line': line_idx,
                'end_line': line_idx,  # Will be updated when the block ends
                'start_offset': indentation,
                'end_offset': None
            })
        
        # Check for function/method definitions
        elif re.match(r'^\s*(def|function|public|private|protected)\s+\w+', line):
            func_name = re.search(r'(def|function|public|private|protected)\s+(\w+)', line).group(2)
            
            if current_element.element_type == ElementType.CLASS_DEF:
                # Method within a class
                func_element = StructuralElement(
                    element_type=ElementType.METHOD_DEF,
                    position=None,  # Will be set below
                    content="",  # Will be set when the block ends
                    element_id=f"method:{func_name}",
                    source_file=file_path
                )
            else:
                # Standalone function
                func_element = StructuralElement(
                    element_type=ElementType.FUNCTION_DEF,
                    position=None,  # Will be set below
                    content="",  # Will be set when the block ends
                    element_id=f"function:{func_name}",
                    source_file=file_path
                )
            
            # Add to parent
            current_element.add_child(func_element)
            
            # Update current element and stack
            current_element = func_element
            element_stack.append(func_element)
            
            # Set position
            func_element.position = type('Position', (), {
                'start_line': line_idx,
                'end_line': line_idx,  # Will be updated when the block ends
                'start_offset': indentation,
                'end_offset': None
            })
        
        # Check for import statements
        elif re.match(r'^\s*(import|from|require|include)', line):
            import_element = StructuralElement(
                element_type=ElementType.IMPORT_STMT,
                position=type('Position', (), {
                    'start_line': line_idx,
                    'end_line': line_idx,
                    'start_offset': indentation,
                    'end_offset': None
                }),
                content=line,
                element_id=f"import:{line_idx}",
                source_file=file_path
            )
            
            # Add to parent
            current_element.add_child(import_element)
        
        # Check for comments
        elif re.match(r'^\s*(#|//|/\*|\*)', line):
            comment_element = StructuralElement(
                element_type=ElementType.COMMENT,
                position=type('Position', (), {
                    'start_line': line_idx,
                    'end_line': line_idx,
                    'start_offset': indentation,
                    'end_offset': None
                }),
                content=line,
                element_id=f"comment:{line_idx}",
                source_file=file_path
            )
            
            # Add to parent
            current_element.add_child(comment_element)
    
    # Close any remaining open elements
    while len(element_stack) > 1:
        ended_element = element_stack.pop()
        ended_element.position.end_line = len(lines) - 1
        ended_element.content = '\n'.join(lines[ended_element.position.start_line:ended_element.position.end_line+1])
    
    # Set root position
    root.position = type('Position', (), {
        'start_line': 0,
        'end_line': len(lines) - 1,
        'start_offset': 0,
        'end_offset': None
    })
    
    # Create document structure
    document = DocumentStructure(
        root=root,
        file_path=file_path,
        language=os.path.splitext(file_path)[1][1:]  # Use extension as language
    )
    
    return document


async def _parse_text_structure(
    file_path: str,
    content: str,
    root: StructuralElement
) -> DocumentStructure:
    """
    Parse structure from a text file (Markdown, RST, etc.).
    
    Args:
        file_path: Path to the file
        content: File content
        root: Root element
        
    Returns:
        Document structure
    """
    # Split into lines
    lines = content.split('\n')
    root.content = content
    
    # Set root position
    root.position = type('Position', (), {
        'start_line': 0,
        'end_line': len(lines) - 1,
        'start_offset': 0,
        'end_offset': None
    })
    
    # Process lines to identify structural elements
    current_section = None
    current_paragraph = None
    paragraph_start = None
    
    for line_idx, line in enumerate(lines):
        line_text = line.strip()
        
        # Check for headings
        if line_text.startswith('#'):
            # Markdown heading
            level = 0
            for char in line_text:
                if char == '#':
                    level += 1
                else:
                    break
            
            heading_text = line_text[level:].strip()
            
            # Close current paragraph if any
            if current_paragraph is not None:
                current_paragraph.position.end_line = line_idx - 1
                current_paragraph.content = '\n'.join(lines[paragraph_start:line_idx])
                current_paragraph = None
            
            # Create heading element
            heading_element = StructuralElement(
                element_type=ElementType.HEADING,
                position=type('Position', (), {
                    'start_line': line_idx,
                    'end_line': line_idx,
                    'start_offset': 0,
                    'end_offset': None
                }),
                content=heading_text,
                element_id=f"heading:{line_idx}",
                source_file=file_path
            )
            
            # Add level attribute
            heading_element.add_attribute("level", level)
            
            # Add to root
            root.add_child(heading_element)
            
            # Create a new section
            section_element = StructuralElement(
                element_type=ElementType.SECTION,
                position=type('Position', (), {
                    'start_line': line_idx,
                    'end_line': line_idx,  # Will be updated when a new section starts
                    'start_offset': 0,
                    'end_offset': None
                }),
                content="",  # Will be set when the section ends
                element_id=f"section:{line_idx}",
                source_file=file_path
            )
            
            # Add to root
            root.add_child(section_element)
            
            # Update current section
            if current_section is not None:
                current_section.position.end_line = line_idx - 1
                current_section.content = '\n'.join(lines[current_section.position.start_line:current_section.position.end_line+1])
            
            current_section = section_element
        
        # Check for blank lines (paragraph boundaries)
        elif not line_text:
            if current_paragraph is not None:
                current_paragraph.position.end_line = line_idx - 1
                current_paragraph.content = '\n'.join(lines[paragraph_start:line_idx])
                current_paragraph = None
        
        # Regular text - part of a paragraph
        else:
            if current_paragraph is None:
                # Start a new paragraph
                paragraph_start = line_idx
                current_paragraph = StructuralElement(
                    element_type=ElementType.PARAGRAPH,
                    position=type('Position', (), {
                        'start_line': line_idx,
                        'end_line': line_idx,  # Will be updated when the paragraph ends
                        'start_offset': 0,
                        'end_offset': None
                    }),
                    content="",  # Will be set when the paragraph ends
                    element_id=f"paragraph:{line_idx}",
                    source_file=file_path
                )
                
                # Add to current section or root
                if current_section is not None:
                    current_section.add_child(current_paragraph)
                else:
                    root.add_child(current_paragraph)
    
    # Close any remaining open elements
    if current_paragraph is not None:
        current_paragraph.position.end_line = len(lines) - 1
        current_paragraph.content = '\n'.join(lines[paragraph_start:len(lines)])
    
    if current_section is not None:
        current_section.position.end_line = len(lines) - 1
        current_section.content = '\n'.join(lines[current_section.position.start_line:current_section.position.end_line+1])
    
    # Create document structure
    document = DocumentStructure(
        root=root,
        file_path=file_path,
        language=os.path.splitext(file_path)[1][1:]  # Use extension as language
    )
    
    return document


async def _parse_markup_structure(
    file_path: str,
    content: str,
    root: StructuralElement
) -> DocumentStructure:
    """Placeholder for HTML/XML structure parsing."""
    # This would use a proper HTML/XML parser in a real implementation
    
    # For now, just create a simple document structure
    lines = content.split('\n')
    root.content = content
    root.position = type('Position', (), {
        'start_line': 0,
        'end_line': len(lines) - 1,
        'start_offset': 0,
        'end_offset': None
    })
    
    document = DocumentStructure(
        root=root,
        file_path=file_path,
        language=os.path.splitext(file_path)[1][1:]
    )
    
    return document


async def _parse_data_structure(
    file_path: str,
    content: str,
    root: StructuralElement
) -> DocumentStructure:
    """Placeholder for JSON/YAML structure parsing."""
    # This would use a proper JSON/YAML parser in a real implementation
    
    # For now, just create a simple document structure
    lines = content.split('\n')
    root.content = content
    root.position = type('Position', (), {
        'start_line': 0,
        'end_line': len(lines) - 1,
        'start_offset': 0,
        'end_offset': None
    })
    
    document = DocumentStructure(
        root=root,
        file_path=file_path,
        language=os.path.splitext(file_path)[1][1:]
    )
    
    return document


async def _parse_generic_structure(
    file_path: str,
    content: str,
    root: StructuralElement
) -> DocumentStructure:
    """Parse generic structure for unknown file types."""
    # Split into lines
    lines = content.split('\n')
    root.content = content
    
    # Set root position
    root.position = type('Position', (), {
        'start_line': 0,
        'end_line': len(lines) - 1,
        'start_offset': 0,
        'end_offset': None
    })
    
    # Simple paragraph detection
    current_paragraph = None
    paragraph_start = None
    
    for line_idx, line in enumerate(lines):
        line_text = line.strip()
        
        # Check for blank lines (paragraph boundaries)
        if not line_text:
            if current_paragraph is not None:
                current_paragraph.position.end_line = line_idx - 1
                current_paragraph.content = '\n'.join(lines[paragraph_start:line_idx])
                current_paragraph = None
        
        # Regular text - part of a paragraph
        else:
            if current_paragraph is None:
                # Start a new paragraph
                paragraph_start = line_idx
                current_paragraph = StructuralElement(
                    element_type=ElementType.PARAGRAPH,
                    position=type('Position', (), {
                        'start_line': line_idx,
                        'end_line': line_idx,  # Will be updated when the paragraph ends
                        'start_offset': 0,
                        'end_offset': None
                    }),
                    content="",  # Will be set when the paragraph ends
                    element_id=f"paragraph:{line_idx}",
                    source_file=file_path
                )
                
                # Add to root
                root.add_child(current_paragraph)
    
    # Close any remaining open paragraph
    if current_paragraph is not None:
        current_paragraph.position.end_line = len(lines) - 1
        current_paragraph.content = '\n'.join(lines[paragraph_start:len(lines)])
    
    # Create document structure
    document = DocumentStructure(
        root=root,
        file_path=file_path,
        language="text"  # Generic language
    )
    
    return document


async def _search_with_ripgrep(
    pattern: str,
    files: List[str],
    case_sensitive: bool,
    is_regex: bool
) -> List[Dict[str, Any]]:
    """
    Perform a basic search with ripgrep.
    
    Args:
        pattern: Search pattern
        files: List of files to search
        case_sensitive: Whether to use case-sensitive matching
        is_regex: Whether the pattern is a regular expression
        
    Returns:
        List of matches
    """
    from tsap.mcp.models import RipgrepSearchParams
    
    # Create search parameters
    params = RipgrepSearchParams(
        pattern=pattern,
        paths=files,
        case_sensitive=case_sensitive,
        is_regex=is_regex,
        recursive=False,  # We're passing the files directly
        context_lines=1  # Get a bit of context
    )
    
    # Perform search
    from tsap.core.ripgrep import ripgrep_search
    results = await ripgrep_search(params)
    
    return [match.dict() for match in results.matches]


async def _search_structural_model(
    model: StructuralModel,
    params: StructureSearchParams
) -> List[StructuralMatch]:
    """
    Search within a structural model.
    
    Args:
        model: Structural model to search
        params: Search parameters
        
    Returns:
        List of structural matches
    """
    matches = []
    
    # First, find elements matching the structural constraints
    candidate_elements = []
    
    # Convert structure_type to a valid ElementType or use a default
    try:
        # Process structure type - handle various input formats
        if params.structure_type:
            # Map common alternative names to valid ElementType values
            type_mapping = {
                "function": "function_def",
                "class": "class_def",
                "method": "method_def",
            }
            
            # Apply mapping if needed
            structure_type = params.structure_type.lower()
            if structure_type in type_mapping:
                structure_type = type_mapping[structure_type]
                
            # Try to convert to ElementType
            try:
                element_type = ElementType(structure_type)
                
                # Filter by element type
                for file_path, document in model.documents.items():
                    elements = document.find_elements_by_type(element_type)
                    candidate_elements.extend([(file_path, element) for element in elements])
            except ValueError:
                logger.warning(
                    f"Invalid structure type '{params.structure_type}', falling back to all elements",
                    component="composite",
                    operation="structure_search"
                )
                # Fall back to all elements
                for file_path, document in model.documents.items():
                    def collect_elements(element):
                        elements = [element]
                        for child in element.children:
                            elements.extend(collect_elements(child))
                        return elements
                    
                    all_elements = collect_elements(document.root)
                    candidate_elements.extend([(file_path, element) for element in all_elements])
        else:
            # No structure type specified, use all elements
            for file_path, document in model.documents.items():
                def collect_elements(element):
                    elements = [element]
                    for child in element.children:
                        elements.extend(collect_elements(child))
                    return elements
                
                all_elements = collect_elements(document.root)
                candidate_elements.extend([(file_path, element) for element in all_elements])
    except Exception as e:
        logger.error(
            f"Error processing structure type: {str(e)}",
            component="composite",
            operation="structure_search"
        )
        # Fall back to all elements
        for file_path, document in model.documents.items():
            def collect_elements(element):
                elements = [element]
                for child in element.children:
                    elements.extend(collect_elements(child))
                return elements
            
            all_elements = collect_elements(document.root)
            candidate_elements.extend([(file_path, element) for element in all_elements])
    
    # Filter by parent type if specified
    if hasattr(params, 'parent_type') and params.parent_type:
        try:
            # Map common alternative names to valid ElementType values
            type_mapping = {
                "function": "function_def",
                "class": "class_def",
                "method": "method_def",
            }
            
            # Apply mapping if needed
            parent_type_str = params.parent_type.lower()
            if parent_type_str in type_mapping:
                parent_type_str = type_mapping[parent_type_str]
                
            # Try to convert to ElementType
            try:
                parent_type = ElementType(parent_type_str)
                candidate_elements = [
                    (file_path, element) for file_path, element in candidate_elements
                    if element.parent and element.parent.element_type == parent_type
                ]
            except ValueError:
                logger.warning(
                    f"Invalid parent type '{params.parent_type}', ignoring parent type filter",
                    component="composite",
                    operation="structure_search"
                )
        except Exception as e:
            logger.error(
                f"Error processing parent type: {str(e)}",
                component="composite",
                operation="structure_search"
            )
    
    # Now search within the candidate elements
    for file_path, element in candidate_elements:
        # Skip elements without content
        if not element.content:
            continue
        
        # Search for the pattern in the element's content
        if params.structure_pattern:
            case_sensitive = getattr(params, 'case_sensitive', False)
            is_regex = getattr(params, 'is_regex', False)
            
            if is_regex:
                try:
                    # Compile regex with appropriate flags
                    flags = 0 if case_sensitive else re.IGNORECASE
                    regex = re.compile(params.structure_pattern, flags)
                    
                    # Find all matches
                    for match in regex.finditer(element.content):
                        # Get the line number for the match
                        match_text = match.group(0)
                        content_before_match = element.content[:match.start()]
                        line_number = content_before_match.count('\n') + element.position.start_line
                        
                        # Create a structural match
                        structural_match = StructuralMatch(
                            file_path=file_path,
                            element_id=element.element_id,
                            element_type=element.element_type,
                            element_content=element.content,
                            match_line=line_number,
                            match_text=match_text,
                            context_text=_extract_context(element.content, match.start(), 50),
                            confidence=1.0
                        )
                        
                        # Add parent elements info
                        parent = element.parent
                        while parent:
                            structural_match.parent_elements.append({
                                "element_id": parent.element_id,
                                "element_type": parent.element_type.value
                            })
                            parent = parent.parent
                        
                        # Add to matches
                        matches.append(structural_match)
                        
                except re.error as e:
                    logger.error(
                        f"Invalid regex pattern: {str(e)}",
                        component="composite",
                        operation="structure_search"
                    )
            else:
                # Simple text search
                search_text = params.structure_pattern if case_sensitive else params.structure_pattern.lower()
                content_text = element.content if case_sensitive else element.content.lower()
                
                start = 0
                while start < len(content_text):
                    # Find the next match
                    start_idx = content_text.find(search_text, start)
                    if start_idx == -1:
                        break
                    
                    # Get match text
                    match_text = element.content[start_idx:start_idx + len(search_text)]
                    
                    # Calculate line number
                    content_before_match = element.content[:start_idx]
                    line_number = content_before_match.count('\n') + element.position.start_line
                    
                    # Create a structural match
                    structural_match = StructuralMatch(
                        file_path=file_path,
                        element_id=element.element_id,
                        element_type=element.element_type,
                        element_content=element.content,
                        match_line=line_number,
                        match_text=match_text,
                        context_text=_extract_context(element.content, start_idx, 50),
                        confidence=1.0
                    )
                    
                    # Add parent elements info
                    parent = element.parent
                    while parent:
                        structural_match.parent_elements.append({
                            "element_id": parent.element_id,
                            "element_type": parent.element_type.value
                        })
                        parent = parent.parent
                    
                    # Add to matches
                    matches.append(structural_match)
                    
                    # Move to the position after this match
                    start = start_idx + len(search_text)
    
    return matches


def _extract_context(text: str, pos: int, context_size: int) -> str:
    """
    Extract context around a position in text.
    
    Args:
        text: Text to extract context from
        pos: Position in text
        context_size: Number of characters of context to extract
        
    Returns:
        Context string
    """
    start = max(0, pos - context_size)
    end = min(len(text), pos + context_size)
    
    context = text[start:end]
    
    # Add ellipsis if we're not at the beginning/end
    if start > 0:
        context = "..." + context
    if end < len(text):
        context = context + "..."
    
    return context


async def structure_search(params: StructureSearchParams) -> StructureSearchResult:
    """
    Perform a structure-aware search.
    
    Args:
        params: Structure search parameters
        
    Returns:
        Structure search results
    """
    start_time = time.time()
    try:
        # Build structural model
        model_start_time = time.time()
        model = await _build_structural_model(params.paths)
        model_build_time = time.time() - model_start_time
        
        # Search the model
        search_start_time = time.time()
        matches = await _search_structural_model(model, params)
        search_time = time.time() - search_start_time
        
        # Convert matches to dictionaries
        match_dicts = [match.to_dict() for match in matches]
        
        # Create result
        result = StructureSearchResult(
            matches=match_dicts,
            structures=[],  # We'll leave this empty for now
            stats={
                "model_build_time": model_build_time,
                "search_time": search_time,
                "total_elements": sum(len(list(_collect_elements(doc.root))) for doc in model.documents.values()),
                "total_files": len(model.documents)
            },
            truncated=False,
            execution_time=time.time() - start_time
        )
        
        return result
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(
            f"Error during structure search: {str(e)}",
            component="composite",
            operation="structure_search"
        )
        
        # Return empty result with error
        return StructureSearchResult(
            matches=[],
            structures=[],
            stats={},
            truncated=False,
            execution_time=total_time,
            error=str(e)
        )


def _collect_elements(element: StructuralElement) -> Generator[StructuralElement, None, None]:
    """
    Recursively collect all elements in a tree.
    
    Args:
        element: Root element
        
    Yields:
        All elements in the tree
    """
    yield element
    for child in element.children:
        yield from _collect_elements(child)


async def search_by_structure(
    pattern: str,
    files: List[str],
    element_type: Optional[str] = None,
    parent_type: Optional[str] = None,
    case_sensitive: bool = False,
    is_regex: bool = False
) -> Dict[str, Any]:
    """
    Convenience function for structure search.
    
    Args:
        pattern: Search pattern
        files: Files to search
        element_type: Type of element to search within
        parent_type: Type of parent structure to filter by
        case_sensitive: Whether to use case-sensitive matching
        is_regex: Whether the pattern is a regular expression
        
    Returns:
        Structure search results as a dictionary
    """
    try:
        # Log files being searched
        logger.info(
            f"Structure search for pattern '{pattern}' in {len(files)} files",
            component="composite",
            operation="structure_search",
            context={
                "element_type": element_type,
                "parent_type": parent_type,
                "case_sensitive": case_sensitive,
                "is_regex": is_regex,
                "files": files[:5]  # Log first 5 files max
            }
        )
        
        # Verify files exist
        valid_files = []
        for file_path in files:
            if os.path.exists(file_path):
                valid_files.append(file_path)
            else:
                logger.warning(
                    f"File not found: {file_path}",
                    component="composite",
                    operation="structure_search"
                )
        
        if len(valid_files) < len(files):
            logger.warning(
                f"{len(files) - len(valid_files)} files not found",
                component="composite",
                operation="structure_search"
            )
        
        if not valid_files:
            logger.warning(
                "No valid files to search",
                component="composite",
                operation="structure_search"
            )
            return {
                "matches": [],
                "structures": [],
                "stats": {},
                "truncated": False,
                "execution_time": 0.0
            }
            
        # Create parameters
        params = StructureSearchParams(
            paths=valid_files,
            structure_pattern=pattern,
            structure_type=element_type,
            parent_type=parent_type,
            case_sensitive=case_sensitive,
            is_regex=is_regex
        )
        
        # Perform search
        try:
            result = await structure_search(params)
            
            # Log results
            logger.info(
                f"Structure search completed with {len(result.matches)} matches",
                component="composite",
                operation="structure_search",
                context={
                    "total_files": len(valid_files),
                    "execution_time": result.execution_time
                }
            )
            
            return result.dict()
        except Exception as e:
            logger.error(
                f"Error during structure search: {str(e)}",
                component="composite",
                operation="structure_search"
            )
            # Return empty result
            return {
                "matches": [],
                "structures": [],
                "stats": {},
                "truncated": False,
                "execution_time": 0.0,
                "error": str(e)
            }
    except Exception as e:
        # Catch all errors and return a valid response
        logger.error(
            f"Unexpected error in structure search: {str(e)}",
            component="composite",
            operation="structure_search"
        )
        import traceback
        logger.error(
            traceback.format_exc(),
            component="composite",
            operation="structure_search"
        )
        return {
            "matches": [],
            "structures": [],
            "stats": {},
            "truncated": False,
            "execution_time": 0.0,
            "error": f"Unexpected error: {str(e)}"
        }