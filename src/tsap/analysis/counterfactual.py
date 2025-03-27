"""
Counterfactual Analyzer for the TSAP MCP Server.

This module implements analysis tools for identifying missing or unusual elements 
in documents compared to expectations or references.
"""

from typing import Dict, List, Any

from tsap.utils.logging import logger
from tsap.analysis.base import BaseAnalysisTool, register_analysis_tool, AnalysisContext
from tsap.mcp.models import CounterfactualAnalyzerParams, CounterfactualAnalyzerResult, Counterfactual
from tsap.core.ripgrep import ripgrep_search


@register_analysis_tool("counterfactual_analyzer")
class CounterfactualAnalyzer(BaseAnalysisTool):
    """
    Analyzer for identifying missing or unusual elements in documents.
    
    The CounterfactualAnalyzer compares document content against reference patterns,
    expectations, or other documents to identify:
    1. Expected patterns that are not present (missing elements)
    2. Unusual patterns that differ from expectations (anomalies)
    3. Structural differences between similar documents
    """
    
    def __init__(self, name: str = "counterfactual_analyzer"):
        """Initialize the counterfactual analyzer."""
        super().__init__(name)
    
    async def _identify_missing_patterns(
        self, 
        documents: List[str], 
        expected_patterns: List[Dict[str, Any]],
        context: AnalysisContext
    ) -> List[Counterfactual]:
        """
        Identify expected patterns that are missing from the documents.
        
        Args:
            documents: List of document paths to analyze
            expected_patterns: List of patterns expected to be found
            context: Analysis context
            
        Returns:
            List of counterfactuals representing missing patterns
        """
        missing_patterns = []
        
        # Search for each expected pattern
        for pattern in expected_patterns:
            pattern_found = False  # noqa: F841
            
            # Use ripgrep to search for the pattern
            search_result = await ripgrep_search({
                "pattern": pattern["pattern"],
                "paths": documents,
                "recursive": True,
                "case_sensitive": pattern.get("case_sensitive", False),
                "is_regex": pattern.get("is_regex", True),
                "max_matches": 1  # We only need to know if it exists
            })
            
            # If no matches found, this is a missing pattern
            if not search_result.matches:
                missing_patterns.append(Counterfactual(
                    type="missing_pattern",
                    description=f"Expected pattern '{pattern['description']}' not found",
                    pattern=pattern["pattern"],
                    confidence=pattern.get("confidence", 0.8),
                    locations=[],
                    related_elements=pattern.get("related_elements", [])
                ))
            
        context.add_statistic("missing_patterns_count", len(missing_patterns))
        return missing_patterns
    
    async def _identify_anomalies(
        self, 
        documents: List[str], 
        normal_patterns: List[Dict[str, Any]],
        anomaly_patterns: List[Dict[str, Any]],
        context: AnalysisContext
    ) -> List[Counterfactual]:
        """
        Identify anomalous patterns in the documents.
        
        Args:
            documents: List of document paths to analyze
            normal_patterns: Patterns that describe normal/expected elements
            anomaly_patterns: Patterns that might indicate anomalies
            context: Analysis context
            
        Returns:
            List of counterfactuals representing anomalies
        """
        anomalies = []
        
        # Search for potential anomaly patterns
        for pattern in anomaly_patterns:
            # Use ripgrep to search for the pattern
            search_result = await ripgrep_search({
                "pattern": pattern["pattern"],
                "paths": documents,
                "recursive": True,
                "case_sensitive": pattern.get("case_sensitive", False),
                "is_regex": pattern.get("is_regex", True),
                "max_matches": pattern.get("max_matches", 100)
            })
            
            # If matches found, these are potential anomalies
            if search_result.matches:
                locations = [
                    {
                        "file": match.file_path,
                        "line": match.line_number,
                        "context": match.line_text
                    }
                    for match in search_result.matches
                ]
                
                anomalies.append(Counterfactual(
                    type="anomaly",
                    description=f"Potential anomaly: {pattern['description']}",
                    pattern=pattern["pattern"],
                    confidence=pattern.get("confidence", 0.7),
                    locations=locations,
                    related_elements=pattern.get("related_elements", [])
                ))
        
        context.add_statistic("anomalies_count", len(anomalies))
        return anomalies
    
    async def _compare_structures(
        self, 
        target_documents: List[str], 
        reference_documents: List[str],
        context: AnalysisContext
    ) -> List[Counterfactual]:
        """
        Compare document structures to identify differences.
        
        Args:
            target_documents: Documents to analyze
            reference_documents: Reference documents for comparison
            context: Analysis context
            
        Returns:
            List of counterfactuals representing structural differences
        """
        structure_differences = []
        
        # This is a simplified placeholder implementation
        # A real implementation would parse document structures and compare them
        
        context.add_statistic("structure_differences_count", len(structure_differences))
        return structure_differences
    
    async def analyze(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform counterfactual analysis on the specified documents.
        
        Args:
            params: CounterfactualAnalyzerParams as a dictionary
            
        Returns:
            CounterfactualAnalyzerResult as a dictionary
        """
        params = CounterfactualAnalyzerParams.parse_obj(params)
        context = AnalysisContext()
        
        try:
            async with self._measure_execution_time():
                # Initialize counterfactuals list
                counterfactuals = []
                
                # 1. Identify missing patterns
                if params.expected_patterns:
                    missing_patterns = await self._identify_missing_patterns(
                        params.documents,
                        params.expected_patterns,
                        context
                    )
                    counterfactuals.extend(missing_patterns)
                
                # 2. Identify anomalies
                if params.anomaly_patterns:
                    anomalies = await self._identify_anomalies(
                        params.documents,
                        params.normal_patterns or [],
                        params.anomaly_patterns,
                        context
                    )
                    counterfactuals.extend(anomalies)
                
                # 3. Compare structures
                if params.reference_documents:
                    structure_differences = await self._compare_structures(
                        params.documents,
                        params.reference_documents,
                        context
                    )
                    counterfactuals.extend(structure_differences)
                
                # Sort counterfactuals by confidence (descending)
                counterfactuals.sort(key=lambda x: x.confidence, reverse=True)
                
                # Create the result
                result = CounterfactualAnalyzerResult(
                    counterfactuals=counterfactuals,
                    document_count=len(params.documents),
                    total_counterfactuals=len(counterfactuals),
                    execution_stats=self.get_statistics(),
                    summary=(
                        f"Found {len(counterfactuals)} counterfactuals "
                        f"({context.get_parameter('missing_patterns_count', 0)} missing patterns, "
                        f"{context.get_parameter('anomalies_count', 0)} anomalies, "
                        f"{context.get_parameter('structure_differences_count', 0)} structural differences)"
                    )
                )
                
                return result.dict()
        except Exception as e:
            logger.error(
                f"Error during counterfactual analysis: {str(e)}",
                component="analysis",
                operation="counterfactual_analyze"
            )
            raise
        

async def analyze_counterfactuals(params: CounterfactualAnalyzerParams) -> CounterfactualAnalyzerResult:
    """
    Convenience function to perform counterfactual analysis.
    
    Args:
        params: Parameters for the counterfactual analysis
        
    Returns:
        Results of the counterfactual analysis
    """
    analyzer = CounterfactualAnalyzer()
    result = await analyzer.analyze(params.dict())
    return CounterfactualAnalyzerResult.parse_obj(result)