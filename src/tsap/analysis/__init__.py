"""
Analysis tools for the TSAP MCP Server.

This package contains high-level semantic tools that build on core tools and composite operations
to perform complex analysis tasks.
"""

from tsap.analysis.base import (
    BaseAnalysisTool,
    AnalysisRegistry,
    AnalysisContext,
    register_analysis_tool
)
from tsap.analysis.cartographer import CorpusCartographer
from tsap.analysis.code import CodeAnalyzer, analyze_code
from tsap.analysis.documents import DocumentExplorer, explore_documents

__all__ = [
    'BaseAnalysisTool',
    'AnalysisRegistry',
    'AnalysisContext',
    'register_analysis_tool',
    'CorpusCartographer',
    'CodeAnalyzer',
    'analyze_code',
    'DocumentExplorer',
    'explore_documents',
]