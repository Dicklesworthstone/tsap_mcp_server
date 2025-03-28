"""
TSAP Composite Operations Package.

This package provides pre-configured combinations of Layer 1 tools that perform
common multi-step operations efficiently. These composite operations build on
the core tools to provide higher-level functionality.
"""

from tsap.composite.parallel import parallel_search
from tsap.composite.context import extract_context
from tsap.composite.structure_search import structure_search
from tsap.composite.diff_generator import generate_diff
from tsap.composite.regex_generator import generate_regex
from tsap.composite.filenames import discover_filename_patterns
from tsap.composite.document_profiler import profile_document, profile_documents

__all__ = [
    "parallel_search",
    "extract_context",
    "structure_search",
    "generate_diff",
    "generate_regex",
    "discover_filename_patterns",
    "profile_document",
    "profile_documents",
]