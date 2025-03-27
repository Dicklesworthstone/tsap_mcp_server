"""
Strategy Compiler for the TSAP MCP Server.

This module implements a tool for compiling high-level search and analysis
objectives into optimized sequences of TSAP operations (strategies).
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from collections import defaultdict

from tsap.utils.logging import logger
from tsap.analysis.base import BaseAnalysisTool, register_analysis_tool, AnalysisContext
from tsap.mcp.models import (
    StrategyCompilerParams,
    StrategyCompilerResult,
    StrategyOperation,
    CompiledStrategy
)
from tsap.performance_mode import get_performance_mode


@register_analysis_tool("strategy_compiler")
class StrategyCompiler(BaseAnalysisTool):
    """
    Compile high-level objectives into optimized TSAP operation sequences.
    
    The StrategyCompiler takes a high-level objective and compiles it into
    an optimized sequence of TSAP operations (a strategy) based on the
    available tools, data characteristics, and performance requirements.
    """
    
    def __init__(self, name: str = "strategy_compiler"):
        """Initialize the strategy compiler."""
        super().__init__(name)
        
        # Available operations by category
        self._operations = {
            "search": [
                {"name": "ripgrep_search", "description": "Basic text search with regex", "layer": "core"},
                {"name": "parallel_search", "description": "Run multiple search patterns simultaneously", "layer": "composite"},
                {"name": "recursive_refinement", "description": "Iteratively narrow search scope", "layer": "composite"},
                {"name": "structure_search", "description": "Search based on document structure", "layer": "composite"}
            ],
            "extract": [
                {"name": "context_extractor", "description": "Extract context around matches", "layer": "composite"},
                {"name": "pdf_extractor", "description": "Extract text from PDF files", "layer": "core"},
                {"name": "html_processor", "description": "Process HTML content", "layer": "core"},
                {"name": "adaptive_metadata_extractor", "description": "Extract metadata from documents", "layer": "analysis"}
            ],
            "analyze": [
                {"name": "code_analyzer", "description": "Analyze code repositories", "layer": "analysis"},
                {"name": "document_explorer", "description": "Analyze document collections", "layer": "analysis"},
                {"name": "corpus_cartographer", "description": "Map relationships between documents", "layer": "analysis"},
                {"name": "counterfactual_analyzer", "description": "Find what's missing in documents", "layer": "analysis"},
                {"name": "pattern_analyzer", "description": "Identify and count patterns", "layer": "composite"}
            ],
            "transform": [
                {"name": "awk_process", "description": "Transform text with AWK", "layer": "core"},
                {"name": "jq_query", "description": "Transform JSON data", "layer": "core"},
                {"name": "sqlite_query", "description": "Run SQL queries on data", "layer": "core"},
                {"name": "table_processor", "description": "Process tabular data", "layer": "core"}
            ],
            "discover": [
                {"name": "pattern_analyzer", "description": "Discover patterns in text", "layer": "composite"},
                {"name": "filename_pattern_discoverer", "description": "Discover filename patterns", "layer": "composite"},
                {"name": "content_structure_analyzer", "description": "Discover document structure", "layer": "composite"},
                {"name": "document_profiler", "description": "Create document fingerprints", "layer": "composite"}
            ]
        }
        
        # Common strategy templates
        self._templates = {
            "code_security_audit": [
                {"operation": "code_analyzer", "params": {"analysis_type": "security"}, "description": "Analyze code for security issues"},
                {"operation": "pattern_analyzer", "params": {"categories": ["secrets", "vulnerabilities"]}, "description": "Find security-related patterns"},
                {"operation": "counterfactual_analyzer", "params": {"expected_patterns": ["security_checks"]}, "description": "Find missing security checks"}
            ],
            "document_extraction": [
                {"operation": "document_explorer", "params": {"extract_metadata": True}, "description": "Find and explore documents"},
                {"operation": "adaptive_metadata_extractor", "params": {"discover_patterns": True}, "description": "Extract metadata from documents"},
                {"operation": "corpus_cartographer", "params": {"include_topics": True}, "description": "Map document relationships"}
            ],
            "pattern_discovery": [
                {"operation": "content_structure_analyzer", "params": {}, "description": "Analyze content structure"},
                {"operation": "pattern_analyzer", "params": {"generate_regex": True}, "description": "Discover and analyze patterns"},
                {"operation": "document_profiler", "params": {}, "description": "Create document profiles"}
            ]
        }
    
    def _parse_objective(self, objective: str) -> Dict[str, Any]:
        """
        Parse a high-level objective into component parts.
        
        Args:
            objective: High-level objective string
            
        Returns:
            Dictionary with parsed components
        """
        # This would ideally use LLM-based parsing
        # For now, use a simple keyword-based approach
        keywords = {
            "search": ["search", "find", "locate", "grep", "look for"],
            "extract": ["extract", "pull out", "get", "retrieve"],
            "analyze": ["analyze", "examine", "study", "investigate"],
            "transform": ["transform", "convert", "process", "change"],
            "discover": ["discover", "identify", "detect", "recognize"]
        }
        
        # Count category keywords
        category_counts = defaultdict(int)
        for category, terms in keywords.items():
            for term in terms:
                if term in objective.lower():
                    category_counts[category] += 1
        
        # Determine primary and secondary categories
        primary_category = max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else "search"
        
        # Remove primary category from counts
        if primary_category in category_counts:
            del category_counts[primary_category]
        
        # Get secondary categories
        secondary_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        secondary_categories = [cat for cat, count in secondary_categories if count > 0]
        
        # Extract potential targets
        code_keywords = ["code", "source", "repository", "function", "class", "method"]
        document_keywords = ["document", "file", "text", "content", "paper", "report"]
        data_keywords = ["data", "json", "csv", "table", "database", "record"]
        
        targets = []
        if any(kw in objective.lower() for kw in code_keywords):
            targets.append("code")
        if any(kw in objective.lower() for kw in document_keywords):
            targets.append("document")
        if any(kw in objective.lower() for kw in data_keywords):
            targets.append("data")
        
        # Default target if none detected
        if not targets:
            targets = ["document"]
        
        return {
            "primary_category": primary_category,
            "secondary_categories": secondary_categories,
            "targets": targets,
            "complexity": "medium",  # Default complexity
            "original_objective": objective
        }
    
    def _select_operations(
        self,
        parsed_objective: Dict[str, Any],
        data_types: List[str],
        max_operations: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Select operations based on the parsed objective and data types.
        
        Args:
            parsed_objective: Parsed objective
            data_types: Types of data to process
            max_operations: Maximum number of operations to include
            
        Returns:
            List of operations
        """
        operations = []
        
        # Get operations for primary category
        primary_category = parsed_objective["primary_category"]
        if primary_category in self._operations:
            operations.extend(self._operations[primary_category])
        
        # Add operations from secondary categories
        for category in parsed_objective["secondary_categories"]:
            if category in self._operations:
                operations.extend(self._operations[category])
        
        # Filter operations based on data types
        filtered_operations = []
        for op in operations:
            if "code" in data_types and op["name"] in ["code_analyzer", "ripgrep_search", "context_extractor"]:
                filtered_operations.append(op)
            elif "document" in data_types and op["name"] in ["document_explorer", "pdf_extractor", "html_processor", "adaptive_metadata_extractor"]:
                filtered_operations.append(op)
            elif "data" in data_types and op["name"] in ["table_processor", "jq_query", "sqlite_query"]:
                filtered_operations.append(op)
            else:
                # Include general-purpose operations
                if op["name"] in ["parallel_search", "recursive_refinement", "pattern_analyzer"]:
                    filtered_operations.append(op)
        
        # Remove duplicates
        seen = set()
        unique_operations = []
        for op in filtered_operations:
            if op["name"] not in seen:
                seen.add(op["name"])
                unique_operations.append(op)
        
        # Limit to max operations
        selected_operations = unique_operations[:max_operations]
        
        # Sort by layer (core -> composite -> analysis)
        layer_order = {"core": 0, "composite": 1, "analysis": 2}
        selected_operations.sort(key=lambda x: layer_order.get(x["layer"], 3))
        
        return selected_operations
    
    def _is_template_applicable(
        self, 
        template_name: str, 
        parsed_objective: Dict[str, Any],
        data_types: List[str]
    ) -> bool:
        """
        Check if a template is applicable to the objective and data types.
        
        Args:
            template_name: Name of template
            parsed_objective: Parsed objective
            data_types: Types of data to process
            
        Returns:
            True if template is applicable, False otherwise
        """
        if template_name == "code_security_audit":
            return (
                "code" in data_types and
                parsed_objective["primary_category"] in ["search", "analyze"] and
                ("security" in parsed_objective["original_objective"].lower() or
                 "vulnerabilit" in parsed_objective["original_objective"].lower())
            )
        elif template_name == "document_extraction":
            return (
                "document" in data_types and
                parsed_objective["primary_category"] in ["extract", "analyze"]
            )
        elif template_name == "pattern_discovery":
            return (
                parsed_objective["primary_category"] in ["discover", "analyze"] or
                "pattern" in parsed_objective["original_objective"].lower()
            )
        
        return False
    
    def _compile_strategy(
        self,
        parsed_objective: Dict[str, Any],
        operations: List[Dict[str, Any]],
        file_paths: List[str],
        file_types: List[str],
        parameters: Dict[str, Any]
    ) -> CompiledStrategy:
        """
        Compile a strategy from selected operations.
        
        Args:
            parsed_objective: Parsed objective
            operations: Selected operations
            file_paths: Paths to process
            file_types: Types of files to process
            parameters: Additional parameters
            
        Returns:
            Compiled strategy
        """
        # Check if any templates are applicable
        strategy_ops = []
        template_used = None
        
        for template_name, template_ops in self._templates.items():
            if self._is_template_applicable(template_name, parsed_objective, parameters.get("data_types", [])):
                strategy_ops = template_ops
                template_used = template_name
                break
        
        # If no template used, build strategy from operations
        if not template_used:
            # Build operations list
            for op in operations:
                strategy_op = StrategyOperation(
                    operation=op["name"],
                    description=op["description"],
                    params={},
                    dependencies=[]
                )
                
                # Add basic parameters based on operation
                if op["name"] in ["ripgrep_search", "parallel_search"]:
                    strategy_op.params = {
                        "paths": file_paths,
                        "file_patterns": file_types if file_types else None,
                        "recursive": True
                    }
                elif op["name"] == "code_analyzer":
                    strategy_op.params = {
                        "paths": file_paths,
                        "language": None,  # Auto-detect
                        "analysis_types": ["structure", "complexity", "security"]
                    }
                elif op["name"] == "document_explorer":
                    strategy_op.params = {
                        "directory_path": file_paths[0] if file_paths else ".",
                        "recursive": True,
                        "extract_metadata": True,
                        "extract_summaries": True
                    }
                
                # Add to strategy
                strategy_ops.append(strategy_op)
            
            # Add dependencies between operations
            for i in range(1, len(strategy_ops)):
                strategy_ops[i].dependencies.append(strategy_ops[i-1].operation)
        
        # Create the strategy
        strategy = CompiledStrategy(
            name=f"Strategy for: {parsed_objective['original_objective'][:50]}...",
            description=parsed_objective["original_objective"],
            operations=strategy_ops,
            template_used=template_used,
            performance_mode=get_performance_mode(),
            estimated_execution_time=len(strategy_ops) * 10,  # Rough estimate: 10s per operation
            parameters={
                "file_paths": file_paths,
                "file_types": file_types,
                **parameters
            }
        )
        
        return strategy
    
    def _optimize_strategy(
        self,
        strategy: CompiledStrategy,
        optimization_level: int = 1
    ) -> CompiledStrategy:
        """
        Optimize a compiled strategy.
        
        Args:
            strategy: Compiled strategy
            optimization_level: Level of optimization (1-3)
            
        Returns:
            Optimized strategy
        """
        if optimization_level <= 0:
            return strategy
        
        # Clone the strategy
        optimized = CompiledStrategy(
            name=strategy.name,
            description=strategy.description,
            operations=[StrategyOperation(**op.dict()) for op in strategy.operations],
            template_used=strategy.template_used,
            performance_mode=strategy.performance_mode,
            estimated_execution_time=strategy.estimated_execution_time,
            parameters=dict(strategy.parameters)
        )
        
        # Basic optimizations (level 1)
        if optimization_level >= 1:
            # Identify and remove redundant operations
            op_names = [op.operation for op in optimized.operations]
            redundant_indices = []
            
            for i, op_name in enumerate(op_names):
                if op_names.count(op_name) > 1:
                    # Keep only the last instance of each operation type
                    last_index = len(op_names) - 1 - op_names[::-1].index(op_name)
                    if i != last_index:
                        redundant_indices.append(i)
            
            # Remove redundant operations
            optimized.operations = [
                op for i, op in enumerate(optimized.operations)
                if i not in redundant_indices
            ]
            
            # Update dependencies
            for op in optimized.operations:
                op.dependencies = [
                    dep for dep in op.dependencies
                    if any(o.operation == dep for o in optimized.operations)
                ]
        
        # Advanced optimizations (level 2)
        if optimization_level >= 2:
            # Merge similar operations
            i = 0
            while i < len(optimized.operations) - 1:
                current_op = optimized.operations[i]
                next_op = optimized.operations[i + 1]
                
                # Check if operations can be merged
                if (current_op.operation == "ripgrep_search" and
                    next_op.operation == "ripgrep_search"):
                    # Merge into parallel_search
                    merged_op = StrategyOperation(
                        operation="parallel_search",
                        description="Combined search operations",
                        params={
                            "patterns": [
                                {"pattern": current_op.params.get("pattern", ""), "description": "Pattern 1"},
                                {"pattern": next_op.params.get("pattern", ""), "description": "Pattern 2"}
                            ],
                            "paths": current_op.params.get("paths", []),
                            "file_patterns": current_op.params.get("file_patterns", None),
                            "recursive": current_op.params.get("recursive", True)
                        },
                        dependencies=current_op.dependencies
                    )
                    
                    # Replace the two operations with the merged one
                    optimized.operations[i] = merged_op
                    optimized.operations.pop(i + 1)
                    
                    # Update dependencies for later operations
                    for j in range(i + 1, len(optimized.operations)):
                        if next_op.operation in optimized.operations[j].dependencies:
                            optimized.operations[j].dependencies.remove(next_op.operation)
                            if merged_op.operation not in optimized.operations[j].dependencies:
                                optimized.operations[j].dependencies.append(merged_op.operation)
                else:
                    i += 1
        
        # Expert optimizations (level 3)
        if optimization_level >= 3:
            # Add parallelization where possible
            for i, op in enumerate(optimized.operations):
                # Identify operations that could benefit from resource allocation
                if op.operation in ["parallel_search", "code_analyzer", "document_explorer"]:
                    # Add resource allocation parameters
                    op.params["resource_allocation"] = {
                        "parallel": True,
                        "max_threads": 4,
                        "batch_size": 100
                    }
        
        # Update execution time estimate
        optimized.estimated_execution_time = sum(
            10 if op.operation.startswith("parallel") else 15
            for op in optimized.operations
        )
        
        return optimized
    
    async def analyze(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compile an optimized strategy from a high-level objective.
        
        Args:
            params: StrategyCompilerParams as a dictionary
            
        Returns:
            StrategyCompilerResult as a dictionary
        """
        params = StrategyCompilerParams.parse_obj(params)
        context = AnalysisContext()
        
        try:
            async with self._measure_execution_time():
                # Parse the objective
                parsed_objective = self._parse_objective(params.objective)
                context.add_result("parsed_objective", parsed_objective)
                
                # Select operations
                operations = self._select_operations(
                    parsed_objective,
                    params.data_types,
                    params.max_operations
                )
                context.add_result("selected_operations", operations)
                
                # Compile the strategy
                strategy = self._compile_strategy(
                    parsed_objective,
                    operations,
                    params.file_paths,
                    params.file_types,
                    params.parameters
                )
                context.add_result("initial_strategy", strategy)
                
                # Optimize the strategy
                optimized_strategy = self._optimize_strategy(
                    strategy,
                    params.optimization_level
                )
                context.add_result("optimized_strategy", optimized_strategy)
                
                # Create the result
                result = StrategyCompilerResult(
                    strategy=optimized_strategy,
                    parsed_objective=parsed_objective,
                    optimization_applied=params.optimization_level > 0,
                    execution_stats=self.get_statistics()
                )
                
                return result.dict()
        except Exception as e:
            logger.error(
                f"Error during strategy compilation: {str(e)}",
                component="analysis",
                operation="strategy_compiler"
            )
            raise


async def compile_strategy(params: StrategyCompilerParams) -> StrategyCompilerResult:
    """
    Convenience function to compile a strategy.
    
    Args:
        params: Parameters for strategy compilation
        
    Returns:
        Results of strategy compilation
    """
    compiler = StrategyCompiler()
    result = await compiler.analyze(params.dict())
    return StrategyCompilerResult.parse_obj(result)