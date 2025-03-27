"""
Content Structure Analysis for the TSAP MCP Server.

This module provides data structures and utilities for content structure analysis,
which serve as the foundation for structure-aware searching and processing.
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field


class ElementType(str, Enum):
    """Types of structural elements in documents."""
    # Common element types
    DOCUMENT = "document"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    LIST = "list"
    LIST_ITEM = "list_item"
    TABLE = "table"
    TABLE_ROW = "table_row"
    TABLE_CELL = "table_cell"
    CODE_BLOCK = "code_block"
    BLOCKQUOTE = "blockquote"
    FIGURE = "figure"
    CAPTION = "caption"
    
    # Text-level element types
    SPAN = "span"
    LINK = "link"
    EMPHASIS = "emphasis"
    STRONG = "strong"
    CODE = "code"
    
    # Programming-specific element types
    CLASS_DEF = "class_def"
    FUNCTION_DEF = "function_def"
    METHOD_DEF = "method_def"
    VARIABLE_DEF = "variable_def"
    IMPORT_STMT = "import_stmt"
    CONTROL_FLOW = "control_flow"
    COMMENT = "comment"
    DOCSTRING = "docstring"
    
    # Specialized element types
    METADATA = "metadata"
    HEADER = "header"
    FOOTER = "footer"
    SIDEBAR = "sidebar"
    NAVIGATION = "navigation"
    FORM = "form"
    FORM_FIELD = "form_field"
    
    # Generic
    OTHER = "other"


@dataclass
class Attribute:
    """
    Represents an attribute (or property) of a structural element.
    """
    name: str
    value: Any
    source: Optional[str] = None  # Where the attribute was defined or inferred


@dataclass
class Position:
    """
    Represents the position of a structural element in a document.
    """
    start_line: int
    end_line: int
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    
    def contains(self, line: int) -> bool:
        """Check if this position contains the given line."""
        return self.start_line <= line <= self.end_line
    
    def overlaps(self, other: 'Position') -> bool:
        """Check if this position overlaps with another."""
        return (
            (self.start_line <= other.end_line and self.end_line >= other.start_line) or
            (other.start_line <= self.end_line and other.end_line >= self.start_line)
        )
    
    def contains_position(self, other: 'Position') -> bool:
        """Check if this position fully contains another."""
        return (
            self.start_line <= other.start_line and
            self.end_line >= other.end_line
        )


@dataclass
class StructuralElement:
    """
    Represents a structural element in a document.
    """
    element_type: ElementType
    position: Position
    content: str
    element_id: Optional[str] = None
    attributes: Dict[str, Attribute] = field(default_factory=dict)
    parent: Optional['StructuralElement'] = None
    children: List['StructuralElement'] = field(default_factory=list)
    source_file: Optional[str] = None
    language: Optional[str] = None
    importance: float = 1.0  # Scale of 0.0 to 1.0
    
    def __post_init__(self):
        """Initialize after creation."""
        # Ensure element_type is an ElementType enum
        if isinstance(self.element_type, str):
            self.element_type = ElementType(self.element_type)
    
    def add_child(self, child: 'StructuralElement') -> None:
        """
        Add a child element.
        
        Args:
            child: Child element to add
        """
        self.children.append(child)
        child.parent = self
    
    def add_attribute(self, name: str, value: Any, source: Optional[str] = None) -> None:
        """
        Add an attribute.
        
        Args:
            name: Attribute name
            value: Attribute value
            source: Source of the attribute
        """
        self.attributes[name] = Attribute(name=name, value=value, source=source)
    
    def get_attribute(self, name: str) -> Optional[Any]:
        """
        Get an attribute value.
        
        Args:
            name: Attribute name
            
        Returns:
            Attribute value or None if not found
        """
        if name in self.attributes:
            return self.attributes[name].value
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "element_type": self.element_type.value,
            "position": {
                "start_line": self.position.start_line,
                "end_line": self.position.end_line,
                "start_offset": self.position.start_offset,
                "end_offset": self.position.end_offset,
                "start_char": self.position.start_char,
                "end_char": self.position.end_char
            },
            "content": self.content[:100] + "..." if len(self.content) > 100 else self.content,
            "element_id": self.element_id,
            "attributes": {name: attr.value for name, attr in self.attributes.items()},
            "children": [child.to_dict() for child in self.children],
            "source_file": self.source_file,
            "language": self.language,
            "importance": self.importance
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StructuralElement':
        """
        Create from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            StructuralElement instance
        """
        position = Position(
            start_line=data["position"]["start_line"],
            end_line=data["position"]["end_line"],
            start_offset=data["position"].get("start_offset"),
            end_offset=data["position"].get("end_offset"),
            start_char=data["position"].get("start_char"),
            end_char=data["position"].get("end_char")
        )
        
        element = cls(
            element_type=data["element_type"],
            position=position,
            content=data["content"],
            element_id=data.get("element_id"),
            source_file=data.get("source_file"),
            language=data.get("language"),
            importance=data.get("importance", 1.0)
        )
        
        # Add attributes
        for name, value in data.get("attributes", {}).items():
            element.add_attribute(name, value)
        
        # Add children
        for child_data in data.get("children", []):
            child = cls.from_dict(child_data)
            element.add_child(child)
        
        return element
    
    def get_all_text(self) -> str:
        """
        Get all text from this element and its children.
        
        Returns:
            Combined text
        """
        if not self.children:
            return self.content
        
        text = self.content
        for child in self.children:
            text += "\n" + child.get_all_text()
        
        return text
    
    def find_element_by_id(self, element_id: str) -> Optional['StructuralElement']:
        """
        Find an element by ID.
        
        Args:
            element_id: Element ID to find
            
        Returns:
            Found element or None
        """
        if self.element_id == element_id:
            return self
        
        for child in self.children:
            found = child.find_element_by_id(element_id)
            if found:
                return found
        
        return None
    
    def find_elements_by_type(self, element_type: ElementType) -> List['StructuralElement']:
        """
        Find elements by type.
        
        Args:
            element_type: Element type to find
            
        Returns:
            List of found elements
        """
        results = []
        
        if self.element_type == element_type:
            results.append(self)
        
        for child in self.children:
            results.extend(child.find_elements_by_type(element_type))
        
        return results
    
    def find_elements_by_attribute(self, name: str, value: Any) -> List['StructuralElement']:
        """
        Find elements by attribute value.
        
        Args:
            name: Attribute name
            value: Attribute value
            
        Returns:
            List of found elements
        """
        results = []
        
        if name in self.attributes and self.attributes[name].value == value:
            results.append(self)
        
        for child in self.children:
            results.extend(child.find_elements_by_attribute(name, value))
        
        return results
    
    def find_element_at_position(self, line: int) -> Optional['StructuralElement']:
        """
        Find the deepest element at a specific line.
        
        Args:
            line: Line number
            
        Returns:
            Found element or None
        """
        if not self.position.contains(line):
            return None
        
        # Check children first to find the deepest element
        for child in self.children:
            found = child.find_element_at_position(line)
            if found:
                return found
        
        # If no children contain the line, return self
        return self
    
    def get_path(self) -> List['StructuralElement']:
        """
        Get the path from the root to this element.
        
        Returns:
            List of elements from root to this element
        """
        if not self.parent:
            return [self]
        
        return self.parent.get_path() + [self]
    
    def get_nearest_sibling(self, direction: str = "next") -> Optional['StructuralElement']:
        """
        Get the nearest sibling element.
        
        Args:
            direction: Direction to look ("next" or "previous")
            
        Returns:
            Nearest sibling or None
        """
        if not self.parent:
            return None
        
        siblings = self.parent.children
        index = siblings.index(self)
        
        if direction == "next" and index < len(siblings) - 1:
            return siblings[index + 1]
        elif direction == "previous" and index > 0:
            return siblings[index - 1]
        
        return None
    
    def get_heading_hierarchy(self) -> List[str]:
        """
        Get the hierarchy of headings above this element.
        
        Returns:
            List of heading texts from highest to lowest level
        """
        headings = []
        current = self
        
        while current:
            if current.element_type == ElementType.HEADING:
                headings.insert(0, current.content)
            current = current.parent
        
        return headings


@dataclass
class DocumentStructure:
    """
    Represents the structure of a document.
    """
    root: StructuralElement
    file_path: str
    language: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "root": self.root.to_dict(),
            "file_path": self.file_path,
            "language": self.language,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentStructure':
        """
        Create from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            DocumentStructure instance
        """
        root = StructuralElement.from_dict(data["root"])
        
        return cls(
            root=root,
            file_path=data["file_path"],
            language=data.get("language"),
            metadata=data.get("metadata", {})
        )
    
    def find_element_by_id(self, element_id: str) -> Optional[StructuralElement]:
        """
        Find an element by ID.
        
        Args:
            element_id: Element ID to find
            
        Returns:
            Found element or None
        """
        return self.root.find_element_by_id(element_id)
    
    def find_elements_by_type(self, element_type: ElementType) -> List[StructuralElement]:
        """
        Find elements by type.
        
        Args:
            element_type: Element type to find
            
        Returns:
            List of found elements
        """
        return self.root.find_elements_by_type(element_type)
    
    def find_element_at_position(self, line: int) -> Optional[StructuralElement]:
        """
        Find the deepest element at a specific line.
        
        Args:
            line: Line number
            
        Returns:
            Found element or None
        """
        return self.root.find_element_at_position(line)
    
    def get_headings(self) -> List[StructuralElement]:
        """
        Get all headings in the document.
        
        Returns:
            List of heading elements
        """
        return self.find_elements_by_type(ElementType.HEADING)
    
    def get_table_of_contents(self) -> List[Dict[str, Any]]:
        """
        Generate a table of contents from the document's headings.
        
        Returns:
            List of heading entries with level and text
        """
        headings = self.get_headings()
        toc = []
        
        for heading in headings:
            # Try to detect heading level
            level = 1  # Default level
            
            # Check attributes first
            level_attr = heading.get_attribute("level")
            if level_attr is not None:
                level = int(level_attr)
            else:
                # Infer from position in hierarchy
                path = heading.get_path()
                level = len([el for el in path if el.element_type == ElementType.HEADING])
            
            toc.append({
                "level": level,
                "text": heading.content,
                "position": {
                    "start_line": heading.position.start_line,
                    "end_line": heading.position.end_line
                },
                "element_id": heading.element_id
            })
        
        return toc


class StructuralModel:
    """
    Manages structural models for multiple documents.
    """
    
    def __init__(self):
        """Initialize the structural model."""
        self.documents: Dict[str, DocumentStructure] = {}
    
    def add_document(self, document: DocumentStructure) -> None:
        """
        Add a document to the model.
        
        Args:
            document: Document structure to add
        """
        self.documents[document.file_path] = document
    
    def get_document(self, file_path: str) -> Optional[DocumentStructure]:
        """
        Get a document structure.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Document structure or None if not found
        """
        return self.documents.get(file_path)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "documents": {
                path: doc.to_dict()
                for path, doc in self.documents.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StructuralModel':
        """
        Create from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            StructuralModel instance
        """
        model = cls()
        
        for path, doc_data in data.get("documents", {}).items():
            document = DocumentStructure.from_dict(doc_data)
            model.add_document(document)
        
        return model
    
    def find_element_by_id(self, element_id: str) -> Optional[Tuple[str, StructuralElement]]:
        """
        Find an element by ID across all documents.
        
        Args:
            element_id: Element ID to find
            
        Returns:
            Tuple of (file path, element) or None if not found
        """
        for path, document in self.documents.items():
            element = document.find_element_by_id(element_id)
            if element:
                return (path, element)
        
        return None
    
    def find_elements_by_type(self, element_type: ElementType) -> Dict[str, List[StructuralElement]]:
        """
        Find elements by type across all documents.
        
        Args:
            element_type: Element type to find
            
        Returns:
            Dictionary mapping file paths to lists of found elements
        """
        results = {}
        
        for path, document in self.documents.items():
            elements = document.find_elements_by_type(element_type)
            if elements:
                results[path] = elements
        
        return results
    
    def get_element_context(self, file_path: str, line: int) -> Dict[str, Any]:
        """
        Get context information for a specific line.
        
        Args:
            file_path: Path to the document
            line: Line number
            
        Returns:
            Dictionary with context information
        """
        document = self.get_document(file_path)
        if not document:
            return {}
        
        element = document.find_element_at_position(line)
        if not element:
            return {}
        
        # Get parent hierarchy
        path = element.get_path()
        
        # Get heading hierarchy
        headings = element.get_heading_hierarchy()
        
        # Get siblings
        next_sibling = element.get_nearest_sibling("next")
        prev_sibling = element.get_nearest_sibling("previous")
        
        # Get context
        context = {
            "element": element.to_dict(),
            "path": [el.element_type.value for el in path],
            "headings": headings,
            "siblings": {
                "next": next_sibling.to_dict() if next_sibling else None,
                "previous": prev_sibling.to_dict() if prev_sibling else None
            }
        }
        
        return context