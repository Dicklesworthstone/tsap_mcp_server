"""
Common search patterns and pattern management for composite operations.

This module provides a library of pre-defined patterns for common search tasks,
along with utilities for pattern categorization, sharing, and optimization.
"""

import re
import time
import uuid
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple, Pattern
from pydantic import BaseModel, Field, validator

from tsap.utils.logging import logger
from tsap.utils.errors import TSAPError
from tsap.mcp.models import SearchPattern
from tsap.storage.pattern_store import get_popular_patterns, add_pattern


class PatternError(TSAPError):
    """
    Exception raised for errors related to search patterns.
    
    Attributes:
        message: Error message
        pattern_id: ID of the pattern that caused the error
        details: Additional error details
    """
    def __init__(self, message: str, pattern_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, code="PATTERN_ERROR", details=details)
        self.pattern_id = pattern_id


class PatternCategory(str, Enum):
    """Categories for search patterns."""
    CODE = "code"
    SECURITY = "security"
    DOCUMENTATION = "documentation"
    DATA = "data"
    CONFIGURATION = "configuration"
    LOGGING = "logging"
    ERROR = "error"
    TESTING = "testing"
    PERFORMANCE = "performance"
    GENERAL = "general"
    CUSTOM = "custom"


class PatternPriority(str, Enum):
    """Priority levels for search patterns."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PatternDefinition(BaseModel):
    """
    Definition of a search pattern.
    
    Attributes:
        id: Unique identifier for the pattern
        name: Human-readable name of the pattern
        pattern: Regular expression pattern
        description: Description of what the pattern searches for
        category: Category of the pattern
        subcategory: Optional subcategory for more specific categorization
        is_regex: Whether the pattern is a regular expression (vs. literal)
        case_sensitive: Whether the pattern is case-sensitive
        multiline: Whether the pattern should match across multiple lines
        priority: Priority level of the pattern
        examples: Examples of text that should match the pattern
        negative_examples: Examples of text that should not match the pattern
        tags: Tags for organizing and finding patterns
        confidence: Confidence score for the pattern (0-1)
        created_at: Timestamp when the pattern was created
        updated_at: Timestamp when the pattern was last updated
        author: Optional author of the pattern
        source: Optional source of the pattern
        metadata: Additional metadata about the pattern
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    pattern: str
    description: str
    category: PatternCategory = PatternCategory.GENERAL
    subcategory: Optional[str] = None
    is_regex: bool = True
    case_sensitive: bool = False
    multiline: bool = False
    priority: PatternPriority = PatternPriority.MEDIUM
    examples: List[str] = Field(default_factory=list)
    negative_examples: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    confidence: float = 0.8
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    author: Optional[str] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('confidence')
    def validate_confidence(cls, v):
        """Validate that confidence is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        return v
    
    def to_search_pattern(self) -> SearchPattern:
        """
        Convert to a SearchPattern object for use with parallel_search.
        
        Returns:
            SearchPattern object
        """
        return SearchPattern(
            pattern=self.pattern,
            description=self.description,
            is_regex=self.is_regex,
            case_sensitive=self.case_sensitive,
            multiline=self.multiline
        )
    
    @classmethod
    def from_search_pattern(cls, search_pattern: SearchPattern, name: Optional[str] = None, category: PatternCategory = PatternCategory.GENERAL) -> 'PatternDefinition':
        """
        Create a PatternDefinition from a SearchPattern.
        
        Args:
            search_pattern: SearchPattern object
            name: Name for the pattern (defaults to description if None)
            category: Category for the pattern
            
        Returns:
            PatternDefinition object
        """
        return cls(
            name=name or search_pattern.description,
            pattern=search_pattern.pattern,
            description=search_pattern.description,
            category=category,
            is_regex=search_pattern.is_regex,
            case_sensitive=search_pattern.case_sensitive,
            multiline=search_pattern.multiline
        )
    
    def compile(self) -> Pattern:
        """
        Compile the pattern into a regex Pattern object.
        
        Returns:
            Compiled regex Pattern
            
        Raises:
            PatternError: If the pattern is invalid
        """
        if not self.is_regex:
            # Escape the pattern if it's a literal string
            pattern_str = re.escape(self.pattern)
        else:
            pattern_str = self.pattern
        
        try:
            flags = 0
            if not self.case_sensitive:
                flags |= re.IGNORECASE
            if self.multiline:
                flags |= re.MULTILINE | re.DOTALL
                
            return re.compile(pattern_str, flags)
        except re.error as e:
            raise PatternError(
                f"Invalid regular expression: {str(e)}",
                pattern_id=self.id,
                details={"pattern": self.pattern, "error": str(e)}
            )
    
    def matches(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Find all matches of the pattern in the text.
        
        Args:
            text: Text to search in
            
        Returns:
            List of tuples (start_pos, end_pos, matched_text)
            
        Raises:
            PatternError: If the pattern is invalid
        """
        compiled = self.compile()
        
        try:
            matches = []
            for match in compiled.finditer(text):
                start, end = match.span()
                matched_text = match.group(0)
                matches.append((start, end, matched_text))
            return matches
        except Exception as e:
            raise PatternError(
                f"Error matching pattern: {str(e)}",
                pattern_id=self.id,
                details={"pattern": self.pattern, "error": str(e)}
            )
    
    def test(self) -> Dict[str, Any]:
        """
        Test the pattern against its examples and negative examples.
        
        Returns:
            Dictionary with test results
            
        Raises:
            PatternError: If the pattern is invalid
        """
        compiled = self.compile()
        
        results = {
            "positive_examples": [],
            "negative_examples": [],
            "true_positives": 0,
            "false_negatives": 0,
            "true_negatives": 0,
            "false_positives": 0
        }
        
        # Test positive examples
        for example in self.examples:
            match = bool(compiled.search(example))
            results["positive_examples"].append({
                "example": example,
                "matched": match
            })
            
            if match:
                results["true_positives"] += 1
            else:
                results["false_negatives"] += 1
        
        # Test negative examples
        for example in self.negative_examples:
            match = bool(compiled.search(example))
            results["negative_examples"].append({
                "example": example,
                "matched": match
            })
            
            if match:
                results["false_positives"] += 1
            else:
                results["true_negatives"] += 1
        
        # Calculate metrics
        total_positives = results["true_positives"] + results["false_negatives"]
        total_negatives = results["true_negatives"] + results["false_positives"]  # noqa: F841
        
        if total_positives > 0:
            results["recall"] = results["true_positives"] / total_positives
        else:
            results["recall"] = 1.0
            
        if results["true_positives"] + results["false_positives"] > 0:
            results["precision"] = results["true_positives"] / (results["true_positives"] + results["false_positives"])
        else:
            results["precision"] = 1.0
            
        if results["precision"] + results["recall"] > 0:
            results["f1_score"] = 2 * (results["precision"] * results["recall"]) / (results["precision"] + results["recall"])
        else:
            results["f1_score"] = 0.0
        
        return results


class PatternLibrary:
    """
    Library of predefined search patterns.
    
    Provides access to common search patterns organized by category.
    """
    
    def __init__(self) -> None:
        """Initialize the pattern library."""
        self._patterns: Dict[str, PatternDefinition] = {}
        self._categories: Dict[PatternCategory, Dict[str, PatternDefinition]] = {
            category: {} for category in PatternCategory
        }
        self._tags: Dict[str, List[str]] = {}
        
        # Initialize with built-in patterns
        self._initialize_built_in_patterns()
    
    def _initialize_built_in_patterns(self) -> None:
        """Initialize the library with built-in patterns."""
        # Add code patterns
        self._add_pattern(CODE_PATTERNS["function_definition"])
        self._add_pattern(CODE_PATTERNS["class_definition"])
        self._add_pattern(CODE_PATTERNS["import_statement"])
        self._add_pattern(CODE_PATTERNS["todo_comment"])
        self._add_pattern(CODE_PATTERNS["error_handling"])
        
        # Add security patterns
        self._add_pattern(SECURITY_PATTERNS["api_key"])
        self._add_pattern(SECURITY_PATTERNS["password"])
        self._add_pattern(SECURITY_PATTERNS["sql_injection"])
        self._add_pattern(SECURITY_PATTERNS["insecure_hash"])
        
        # Add documentation patterns
        self._add_pattern(DOCUMENTATION_PATTERNS["docstring"])
        self._add_pattern(DOCUMENTATION_PATTERNS["markdown_heading"])
        self._add_pattern(DOCUMENTATION_PATTERNS["code_comment"])
        
        # Add configuration patterns
        self._add_pattern(CONFIGURATION_PATTERNS["env_var"])
        self._add_pattern(CONFIGURATION_PATTERNS["json_config"])
        self._add_pattern(CONFIGURATION_PATTERNS["yaml_config"])
        self._add_pattern(CONFIGURATION_PATTERNS["ini_config"])
    
    def _add_pattern(self, pattern: PatternDefinition) -> None:
        """
        Add a pattern to the library.
        
        Args:
            pattern: Pattern to add
        """
        self._patterns[pattern.id] = pattern
        self._categories[pattern.category][pattern.id] = pattern
        
        # Add to tag index
        for tag in pattern.tags:
            if tag not in self._tags:
                self._tags[tag] = []
            self._tags[tag].append(pattern.id)
    
    def get_pattern(self, pattern_id: str) -> Optional[PatternDefinition]:
        """
        Get a pattern by ID.
        
        Args:
            pattern_id: ID of the pattern to get
            
        Returns:
            The pattern, or None if not found
        """
        return self._patterns.get(pattern_id)
    
    def get_pattern_by_name(self, name: str) -> Optional[PatternDefinition]:
        """
        Get a pattern by name.
        
        Args:
            name: Name of the pattern to get
            
        Returns:
            The pattern, or None if not found
        """
        for pattern in self._patterns.values():
            if pattern.name.lower() == name.lower():
                return pattern
        return None
    
    def list_patterns(
        self,
        category: Optional[Union[PatternCategory, str]] = None,
        subcategory: Optional[str] = None,
        tags: Optional[List[str]] = None,
        priority: Optional[Union[PatternPriority, str]] = None
    ) -> List[PatternDefinition]:
        """
        List patterns with optional filtering.
        
        Args:
            category: Optional category to filter by
            subcategory: Optional subcategory to filter by
            tags: Optional list of tags to filter by
            priority: Optional priority to filter by
            
        Returns:
            List of matching patterns
        """
        # Convert string inputs to enums if needed
        if category and isinstance(category, str):
            try:
                category = PatternCategory(category.lower())
            except ValueError:
                # Invalid category, return empty list
                return []
                
        if priority and isinstance(priority, str):
            try:
                priority = PatternPriority(priority.lower())
            except ValueError:
                # Invalid priority, return empty list
                return []
        
        # Start with all patterns or a specific category
        if category:
            patterns = list(self._categories[category].values())
        else:
            patterns = list(self._patterns.values())
        
        # Filter by subcategory
        if subcategory:
            patterns = [p for p in patterns if p.subcategory == subcategory]
        
        # Filter by tags (patterns must have all specified tags)
        if tags:
            patterns = [p for p in patterns if all(tag in p.tags for tag in tags)]
        
        # Filter by priority
        if priority:
            patterns = [p for p in patterns if p.priority == priority]
        
        return patterns
    
    def list_categories(self) -> List[str]:
        """
        List all categories with at least one pattern.
        
        Returns:
            List of category names
        """
        return [c.value for c in self._categories if self._categories[c]]
    
    def list_tags(self) -> List[str]:
        """
        List all tags used by patterns.
        
        Returns:
            List of tags
        """
        return list(self._tags.keys())
    
    def list_subcategories(self, category: Optional[Union[PatternCategory, str]] = None) -> List[str]:
        """
        List all subcategories, optionally within a specific category.
        
        Args:
            category: Optional category to filter by
            
        Returns:
            List of subcategories
        """
        # Convert string input to enum if needed
        if category and isinstance(category, str):
            try:
                category = PatternCategory(category.lower())
            except ValueError:
                # Invalid category, return empty list
                return []
        
        # Get patterns from all categories or a specific one
        if category:
            patterns = list(self._categories[category].values())
        else:
            patterns = list(self._patterns.values())
        
        # Extract unique subcategories
        subcategories = set()
        for pattern in patterns:
            if pattern.subcategory:
                subcategories.add(pattern.subcategory)
        
        return sorted(list(subcategories))
    
    def add_custom_pattern(self, pattern: PatternDefinition) -> str:
        """
        Add a custom pattern to the library.
        
        Args:
            pattern: Pattern to add
            
        Returns:
            ID of the added pattern
        """
        # Ensure the pattern has a unique ID
        if pattern.id in self._patterns:
            pattern.id = str(uuid.uuid4())
        
        # Add the pattern
        self._add_pattern(pattern)
        
        # Save the pattern to the persistent store if available
        try:
            add_pattern(
                name=pattern.name,
                pattern=pattern.pattern,
                description=pattern.description,
                is_regex=pattern.is_regex,
                category=pattern.category.value,
                tags=pattern.tags
            )
        except Exception as e:
            logger.warning(f"Failed to save pattern to persistent store: {str(e)}")
        
        return pattern.id
    
    def remove_pattern(self, pattern_id: str) -> bool:
        """
        Remove a pattern from the library.
        
        Args:
            pattern_id: ID of the pattern to remove
            
        Returns:
            True if the pattern was removed, False if not found
        """
        pattern = self._patterns.get(pattern_id)
        if not pattern:
            return False
        
        # Remove from main dictionary
        del self._patterns[pattern_id]
        
        # Remove from category dictionary
        if pattern_id in self._categories[pattern.category]:
            del self._categories[pattern.category][pattern_id]
        
        # Remove from tag index
        for tag in pattern.tags:
            if tag in self._tags and pattern_id in self._tags[tag]:
                self._tags[tag].remove(pattern_id)
                if not self._tags[tag]:
                    del self._tags[tag]
        
        return True
    
    def get_search_patterns(
        self,
        category: Optional[Union[PatternCategory, str]] = None,
        tags: Optional[List[str]] = None,
        priority: Optional[Union[PatternPriority, str]] = None,
        limit: Optional[int] = None
    ) -> List[SearchPattern]:
        """
        Get a list of SearchPattern objects for use with parallel_search.
        
        Args:
            category: Optional category to filter by
            tags: Optional list of tags to filter by
            priority: Optional priority to filter by
            limit: Optional maximum number of patterns to return
            
        Returns:
            List of SearchPattern objects
        """
        patterns = self.list_patterns(category=category, tags=tags, priority=priority)
        
        # Sort by priority and confidence
        patterns.sort(key=lambda p: (
            PRIORITY_VALUES[p.priority],
            p.confidence
        ), reverse=True)
        
        # Apply limit
        if limit is not None:
            patterns = patterns[:limit]
        
        # Convert to SearchPattern objects
        return [p.to_search_pattern() for p in patterns]
    
    def load_from_persistent_store(self) -> int:
        """
        Load patterns from the persistent pattern store.
        
        Returns:
            Number of patterns loaded
        """
        loaded_count = 0
        
        try:
            # Get popular patterns from the store
            patterns = get_popular_patterns(limit=100)
            
            for p in patterns:
                # Skip patterns that are already in the library
                if self.get_pattern_by_name(p["name"]):
                    continue
                
                # Create a PatternDefinition
                try:
                    category = PatternCategory(p.get("category", "general").lower())
                except ValueError:
                    category = PatternCategory.GENERAL
                
                try:
                    priority = PatternPriority(p.get("priority", "medium").lower())
                except ValueError:
                    priority = PatternPriority.MEDIUM
                
                pattern = PatternDefinition(
                    id=p.get("id", str(uuid.uuid4())),
                    name=p["name"],
                    pattern=p["pattern"],
                    description=p.get("description", ""),
                    category=category,
                    subcategory=p.get("subcategory"),
                    is_regex=p.get("is_regex", True),
                    case_sensitive=p.get("case_sensitive", False),
                    multiline=p.get("multiline", False),
                    priority=priority,
                    tags=p.get("tags", []),
                    confidence=p.get("confidence", 0.8),
                    author=p.get("author"),
                    source=p.get("source")
                )
                
                # Add the pattern
                self._add_pattern(pattern)
                loaded_count += 1
                
        except Exception as e:
            logger.warning(f"Failed to load patterns from persistent store: {str(e)}")
        
        return loaded_count


# Initialize priority values for sorting
PRIORITY_VALUES = {
    PatternPriority.LOW: 0,
    PatternPriority.MEDIUM: 1,
    PatternPriority.HIGH: 2,
    PatternPriority.CRITICAL: 3
}


# Built-in code patterns
CODE_PATTERNS = {
    "function_definition": PatternDefinition(
        name="Function Definition",
        pattern=r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*(?:->.*?)?:",
        description="Find function definitions in Python code",
        category=PatternCategory.CODE,
        subcategory="python",
        tags=["python", "function", "definition"],
        examples=[
            "def hello_world():",
            "def process_data(input_file: str, options: Dict) -> List[str]:",
            "def calculate_total(items: List[Item], tax_rate: float = 0.1) -> float:",
            "async def fetch_data(url: str, timeout: int = 30) -> Dict[str, Any]:"
        ]
    ),
    "class_definition": PatternDefinition(
        name="Class Definition",
        pattern=r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\([^)]*\))?:",
        description="Find class definitions in Python code",
        category=PatternCategory.CODE,
        subcategory="python",
        tags=["python", "class", "definition"],
        examples=[
            "class MyClass:",
            "class DataProcessor(BaseProcessor):",
            "class UserManager(metaclass=Singleton):",
            "class AsyncDatabaseConnection(ABC):"
        ]
    ),
    "import_statement": PatternDefinition(
        name="Import Statement",
        pattern=r"(?:from\s+[a-zA-Z_][a-zA-Z0-9_.]+\s+)?import\s+(?:[a-zA-Z_][a-zA-Z0-9_]+(?:\s+as\s+[a-zA-Z_][a-zA-Z0-9_]+)?(?:,\s*)?)+",
        description="Find import statements in Python code",
        category=PatternCategory.CODE,
        subcategory="python",
        tags=["python", "import"],
        examples=[
            "import os",
            "from typing import Dict, List, Any",
            "import numpy as np",
            "from datetime import datetime, timezone",
            "from .models import User, Profile",
            "import pandas as pd, matplotlib.pyplot as plt"
        ]
    ),
    "todo_comment": PatternDefinition(
        name="TODO Comment",
        pattern=r"(?:^|\s)#\s*TODO\s*:?\s*(.*?)(?:$|\n)",
        description="Find TODO comments in code",
        category=PatternCategory.CODE,
        subcategory="comments",
        tags=["todo", "comment"],
        examples=[
            "# TODO: Fix this bug",
            "    # TODO: Implement error handling",
            "# TODO: Add input validation",
            "# TODO: Optimize performance for large datasets"
        ]
    ),
    "error_handling": PatternDefinition(
        name="Error Handling",
        pattern=r"try\s*:[\s\S]*?except\s+(?:\([^)]+\)|[^:]+):",
        description="Find error handling blocks in Python code",
        category=PatternCategory.CODE,
        subcategory="python",
        multiline=True,
        tags=["python", "error", "exception"],
        examples=[
            "try:\n    result = process_data()\nexcept Exception as e:\n    logger.error(e)",
            "try:\n    with open(file) as f:\n        data = f.read()\nexcept (IOError, OSError) as e:\n    print(f'Error: {e}')"
        ]
    )
}


# Built-in security patterns
SECURITY_PATTERNS = {
    "api_key": PatternDefinition(
        name="API Key",
        pattern=r"(?:api|token|secret|key)(?:_|\s)?(?:key|token|secret)?(?:_|\s)?(?:=\"|\s=\s\"|:\s\"|=\')([a-zA-Z0-9]{16,})",
        description="Find potential API keys or tokens in code",
        category=PatternCategory.SECURITY,
        subcategory="credentials",
        priority=PatternPriority.HIGH,
        tags=["security", "api", "key", "credentials"],
        examples=[
            'api_key = "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"',
            'API_TOKEN="a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"',
            'secret_key = "sk_live_51H7J8K9L0M1N2O3P4Q5R6S7T8U9V0W1X2Y3Z4"',
            'jwt_secret = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"'
        ]
    ),
    "password": PatternDefinition(
        name="Password",
        pattern=r"(?:password|passwd|pwd)(?:_|\s)?(?:=\s|\s=\s|\:)(?:\"|\'|)([^\"\'\n]+)",
        description="Find hardcoded passwords in code",
        category=PatternCategory.SECURITY,
        subcategory="credentials",
        priority=PatternPriority.CRITICAL,
        tags=["security", "password", "credentials"],
        examples=[
            'password = "supersecret"',
            'pwd="p@ssw0rd123"',
            'db_password = "Admin@123!"',
            'root_password: "Root@2023#"'
        ]
    ),
    "sql_injection": PatternDefinition(
        name="SQL Injection Vulnerability",
        pattern=r"(?:execute|query|run_query)\s*\(\s*(?:[\"\']\s*SELECT|[\"\']\s*INSERT|[\"\']\s*UPDATE|[\"\']\s*DELETE).*?\+|f[\"\'](?:SELECT|INSERT|UPDATE|DELETE)",
        description="Find potential SQL injection vulnerabilities",
        category=PatternCategory.SECURITY,
        subcategory="injection",
        priority=PatternPriority.CRITICAL,
        tags=["security", "sql", "injection"],
        examples=[
            'cursor.execute("SELECT * FROM users WHERE id = " + user_id)',
            'query = f"SELECT * FROM users WHERE username = \'{username}\'"',
            'db.execute("UPDATE users SET status = " + status + " WHERE id = " + id)',
            'sql = f"INSERT INTO logs (user, action) VALUES (\'{user}\', \'{action}\')"'
        ]
    ),
    "insecure_hash": PatternDefinition(
        name="Insecure Hash Function",
        pattern=r"(?:md5|sha1)\s*\(\s*[\"\'][^\"\']+[\"\']\s*\)",
        description="Find usage of cryptographically weak hash functions",
        category=PatternCategory.SECURITY,
        subcategory="crypto",
        priority=PatternPriority.HIGH,
        tags=["security", "crypto", "hash"],
        examples=[
            'hash_value = md5("password123")',
            'file_hash = sha1(file_content)',
            'checksum = md5("sensitive_data")',
            'digest = sha1(user_input)'
        ]
    )
}


# Built-in documentation patterns
DOCUMENTATION_PATTERNS = {
    "docstring": PatternDefinition(
        name="Docstring",
        pattern=r"(?:^|\s)(?:\"\"\"|\'\'\')(.*?)(?:\"\"\"|\'\'\')(?:\s|$)",
        description="Find docstrings in Python code",
        category=PatternCategory.DOCUMENTATION,
        subcategory="python",
        multiline=True,
        tags=["python", "docstring", "documentation"],
        examples=[
            '"""This is a docstring."""',
            '    """Multi-line\n    docstring\n    example."""',
            '"""Calculate the total price including tax.\n\nArgs:\n    price: Base price\n    tax_rate: Tax rate (0-1)\n\nReturns:\n    Total price with tax\n"""',
            "'''Process user data and generate report.\n\nThis function handles data validation,\nformatting, and report generation.\n'''"
        ]
    ),
    "markdown_heading": PatternDefinition(
        name="Markdown Heading",
        pattern=r"^#+\s+(.*)$",
        description="Find headings in Markdown files",
        category=PatternCategory.DOCUMENTATION,
        subcategory="markdown",
        tags=["markdown", "heading", "documentation"],
        examples=[
            "# Heading Level 1",
            "### Heading Level 3",
            "## Project Setup",
            "#### Configuration Options"
        ]
    ),
    "code_comment": PatternDefinition(
        name="Code Comment",
        pattern=r"(?:^|\s)#\s+[^#\n]+",
        description="Find descriptive code comments",
        category=PatternCategory.DOCUMENTATION,
        subcategory="comments",
        tags=["comment", "documentation"],
        examples=[
            "# Initialize database connection",
            "    # Process each item in the batch",
            "# Validate input parameters",
            "    # Handle edge cases"
        ]
    )
}


# Built-in configuration patterns
CONFIGURATION_PATTERNS = {
    "env_var": PatternDefinition(
        name="Environment Variable",
        pattern=r"(?:os\.(?:getenv|environ\.get)|dotenv\.get_key)\s*\(\s*(?:\"|\')([A-Z0-9_]+)",
        description="Find environment variable usage",
        category=PatternCategory.CONFIGURATION,
        subcategory="environment",
        tags=["configuration", "environment", "variable"],
        examples=[
            'os.getenv("API_KEY")',
            'os.environ.get("DATABASE_URL", default_url)',
            'dotenv.get_key(".env", "SECRET_KEY")',
            'os.environ.get("DEBUG_MODE", "False")'
        ]
    ),
    "json_config": PatternDefinition(
        name="JSON Configuration",
        pattern=r"with\s+open\s*\(\s*[\"']([^\"']+)[\"'](?:\s*,\s*[\"']r[\"'])?\s*as\s+[^:]+:\s*\n\s*(?:config|settings|conf)\s*=\s*json\.load",
        description="Find JSON configuration file loading",
        category=PatternCategory.CONFIGURATION,
        subcategory="json",
        multiline=True,
        tags=["configuration", "json", "file"],
        examples=[
            'with open("config.json") as f:\n    config = json.load(f)',
            'with open("./settings/app_config.json", "r") as config_file:\n    settings = json.load(config_file)',
            'with open("config/production.json", "r") as f:\n    conf = json.load(f)',
            'with open("settings.json") as settings_file:\n    app_config = json.load(settings_file)'
        ]
    ),
    "yaml_config": PatternDefinition(
        name="YAML Configuration",
        pattern=r"([A-Za-z_]+):\s*(?:[\"']?)(?:[^\"'\n]+)(?:[\"']?)",
        description="Find YAML configuration entries",
        category=PatternCategory.CONFIGURATION,
        subcategory="yaml",
        tags=["configuration", "yaml", "file"],
        examples=[
            'host: 0.0.0.0',
            'password: "Admin@123!"',
            'api_key: "sk_live_51H7..."'
        ]
    ),
    "ini_config": PatternDefinition(
        name="INI Configuration",
        pattern=r"(?:configparser\.ConfigParser|config\.ConfigParser)\(\s*\)\.read\s*\(\s*[\"']([^\"']+\.ini)[\"']\s*\)",
        description="Find INI configuration file loading",
        category=PatternCategory.CONFIGURATION,
        subcategory="ini",
        tags=["configuration", "ini", "file"],
        examples=[
            'config = configparser.ConfigParser().read("config.ini")',
            'settings = config.ConfigParser().read("app_settings.ini")',
            'conf = configparser.ConfigParser().read("production.ini")'
        ]
    )
}


# Singleton instance of the pattern library
_pattern_library: Optional[PatternLibrary] = None


def get_pattern_library() -> PatternLibrary:
    """
    Get the global pattern library instance.
    
    Returns:
        The global pattern library
    """
    global _pattern_library
    if _pattern_library is None:
        _pattern_library = PatternLibrary()
    return _pattern_library


def get_patterns_by_category(category: Union[PatternCategory, str]) -> List[PatternDefinition]:
    """
    Get patterns by category.
    
    Args:
        category: Category to get patterns for
        
    Returns:
        List of patterns in the category
    """
    library = get_pattern_library()
    return library.list_patterns(category=category)


def get_patterns_by_tags(tags: List[str]) -> List[PatternDefinition]:
    """
    Get patterns by tags.
    
    Args:
        tags: Tags to get patterns for
        
    Returns:
        List of patterns with the specified tags
    """
    library = get_pattern_library()
    return library.list_patterns(tags=tags)


def get_search_patterns_for_operation(
    operation_type: str,
    limit: Optional[int] = 10
) -> List[SearchPattern]:
    """
    Get search patterns appropriate for a specific operation type.
    
    Args:
        operation_type: Type of operation (e.g., "security_audit", "code_analysis")
        limit: Maximum number of patterns to return
        
    Returns:
        List of SearchPattern objects
    """
    library = get_pattern_library()
    
    # Map operation types to categories and tags
    operation_map = {
        "security_audit": {
            "category": PatternCategory.SECURITY,
            "tags": ["security"]
        },
        "code_analysis": {
            "category": PatternCategory.CODE,
            "tags": ["code"]
        },
        "documentation_check": {
            "category": PatternCategory.DOCUMENTATION,
            "tags": ["documentation"]
        },
        "configuration_analysis": {
            "category": PatternCategory.CONFIGURATION,
            "tags": ["configuration"]
        }
    }
    
    mapping = operation_map.get(operation_type, {"category": None, "tags": None})
    
    return library.get_search_patterns(
        category=mapping["category"],
        tags=mapping["tags"],
        limit=limit
    )


def add_custom_pattern(
    name: str,
    pattern: str,
    description: str,
    category: Union[PatternCategory, str] = PatternCategory.CUSTOM,
    is_regex: bool = True,
    tags: Optional[List[str]] = None
) -> str:
    """
    Add a custom pattern to the library.
    
    Args:
        name: Name of the pattern
        pattern: Regular expression or literal pattern
        description: Description of what the pattern searches for
        category: Category for the pattern
        is_regex: Whether the pattern is a regular expression
        tags: Optional tags for the pattern
        
    Returns:
        ID of the added pattern
        
    Raises:
        PatternError: If the pattern is invalid
    """
    # Convert string category to enum if needed
    if isinstance(category, str):
        try:
            category = PatternCategory(category.lower())
        except ValueError:
            category = PatternCategory.CUSTOM
    
    # Create pattern definition
    pattern_def = PatternDefinition(
        name=name,
        pattern=pattern,
        description=description,
        category=category,
        is_regex=is_regex,
        tags=tags or []
    )
    
    # Validate the pattern by compiling it
    try:
        pattern_def.compile()
    except PatternError as e:
        # Re-raise with additional context
        raise PatternError(
            f"Invalid pattern '{name}': {str(e)}",
            details={"pattern": pattern, "error": str(e)}
        )
    
    # Add to library
    library = get_pattern_library()
    return library.add_custom_pattern(pattern_def)


def get_pattern_by_id(pattern_id: str) -> Optional[PatternDefinition]:
    """
    Get a pattern by ID.
    
    Args:
        pattern_id: ID of the pattern to get
        
    Returns:
        The pattern, or None if not found
    """
    library = get_pattern_library()
    return library.get_pattern(pattern_id)


def test_pattern(
    pattern: str,
    examples: List[str],
    negative_examples: Optional[List[str]] = None,
    is_regex: bool = True,
    case_sensitive: bool = False,
    multiline: bool = False
) -> Dict[str, Any]:
    """
    Test a pattern against examples.
    
    Args:
        pattern: Regular expression or literal pattern
        examples: Examples that should match
        negative_examples: Examples that should not match
        is_regex: Whether the pattern is a regular expression
        case_sensitive: Whether the pattern is case-sensitive
        multiline: Whether the pattern should match across multiple lines
        
    Returns:
        Dictionary with test results
        
    Raises:
        PatternError: If the pattern is invalid
    """
    # Create a temporary pattern definition
    pattern_def = PatternDefinition(
        name="Test Pattern",
        pattern=pattern,
        description="Temporary pattern for testing",
        is_regex=is_regex,
        case_sensitive=case_sensitive,
        multiline=multiline,
        examples=examples,
        negative_examples=negative_examples or []
    )
    
    # Test the pattern
    return pattern_def.test()