"""
Runtime learning for pattern and strategy optimization.

This module provides functionality for real-time learning and adaptation
of search patterns and analysis strategies based on feedback from users
or from the results of previous executions.
"""

import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from tsap.mcp.models import RuntimeLearningParams, RuntimeLearningResult
from tsap.evolution.pattern_analyzer import PatternAnalyzer, analyze_pattern
from tsap.evolution.genetic import evolve_regex_pattern
from tsap.utils.errors import TSAPError


class RuntimeLearningError(TSAPError):
    """
    Exception for errors in runtime learning operations.
    
    Attributes:
        message: Error message
        learning_type: Type of learning operation that failed
        details: Additional error details
    """
    def __init__(
        self, 
        message: str, 
        learning_type: Optional[str] = None, 
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            message, 
            code=f"RUNTIME_LEARNING_{learning_type.upper()}_ERROR" if learning_type else "RUNTIME_LEARNING_ERROR",
            details=details
        )
        self.learning_type = learning_type


@dataclass
class PatternLearningContext:
    """
    Context for pattern learning operations.
    
    Attributes:
        pattern_id: ID of the pattern to optimize
        original_pattern: Original pattern string
        feedback_items: User feedback on matches
        positive_examples: Matches that are relevant
        negative_examples: Matches that are not relevant
        learning_rate: Rate of learning (0.0-1.0)
        metadata: Additional metadata for learning
    """
    pattern_id: str
    original_pattern: str
    feedback_items: List[Dict[str, Any]] = field(default_factory=list)
    positive_examples: List[str] = field(default_factory=list)
    negative_examples: List[str] = field(default_factory=list)
    learning_rate: float = 0.1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyLearningContext:
    """
    Context for strategy learning operations.
    
    Attributes:
        strategy_id: ID of the strategy to optimize
        original_strategy: Original strategy configuration
        feedback_items: User feedback on strategy results
        effectiveness_score: Effectiveness score (0.0-1.0)
        learning_rate: Rate of learning (0.0-1.0)
        metadata: Additional metadata for learning
    """
    strategy_id: str
    original_strategy: Dict[str, Any]
    feedback_items: List[Dict[str, Any]] = field(default_factory=list)
    effectiveness_score: float = 0.0
    learning_rate: float = 0.1
    metadata: Dict[str, Any] = field(default_factory=dict)


class RuntimeLearning:
    """
    Class for real-time learning and adaptation of patterns and strategies.
    """
    def __init__(self) -> None:
        """Initialize the runtime learning system."""
        # Initialize learning contexts by ID
        self._pattern_contexts: Dict[str, PatternLearningContext] = {}
        self._strategy_contexts: Dict[str, StrategyLearningContext] = {}
        
        # Initialize pattern analyzer
        self._pattern_analyzer = PatternAnalyzer()
    
    async def optimize_pattern(
        self, 
        pattern_id: str, 
        feedback: List[Dict[str, Any]], 
        learning_rate: float = 0.1,
        evolve: bool = True
    ) -> Dict[str, Any]:
        """
        Optimize a search pattern based on feedback.
        
        Args:
            pattern_id: ID of the pattern to optimize
            feedback: List of feedback items, each containing a match and relevance flag
            learning_rate: Rate of learning (0.0-1.0)
            evolve: Whether to evolve a new pattern or just analyze
            
        Returns:
            Dictionary with optimization results
            
        Raises:
            RuntimeLearningError: If pattern optimization fails
        """
        try:
            # Get pattern information (from pattern_library or other storage)
            pattern_info = await self._get_pattern_info(pattern_id)
            
            if not pattern_info:
                raise RuntimeLearningError(
                    message=f"Pattern not found: {pattern_id}",
                    learning_type="pattern"
                )
            
            # Get or create learning context
            if pattern_id not in self._pattern_contexts:
                self._pattern_contexts[pattern_id] = PatternLearningContext(
                    pattern_id=pattern_id,
                    original_pattern=pattern_info["pattern"],
                    learning_rate=learning_rate
                )
            
            context = self._pattern_contexts[pattern_id]
            
            # Update learning context with new feedback
            context.feedback_items.extend(feedback)
            
            # Process feedback into positive and negative examples
            for item in feedback:
                match_text = item.get("match", "")
                relevant = item.get("relevant", False)
                
                if match_text:
                    if relevant:
                        context.positive_examples.append(match_text)
                    else:
                        context.negative_examples.append(match_text)
            
            # Analyze the pattern
            analysis_result = await analyze_pattern(
                pattern=context.original_pattern,
                description=pattern_info.get("description", ""),
                is_regex=pattern_info.get("is_regex", True),
                case_sensitive=pattern_info.get("case_sensitive", False),
                paths=[],  # Paths are not needed for analysis
                reference_set=None
            )
            
            # Generate optimized pattern if requested
            optimized_pattern = context.original_pattern
            evolution_result = None
            
            if evolve and (context.positive_examples or context.negative_examples):
                # Use genetic algorithm to evolve a new pattern
                evolution_result = await evolve_regex_pattern(
                    positive_examples=context.positive_examples,
                    negative_examples=context.negative_examples,
                    initial_patterns=[context.original_pattern]
                )
                
                # Get the optimized pattern
                if evolution_result and "pattern" in evolution_result:
                    optimized_pattern = evolution_result["pattern"]
            
            # Calculate similarity between original and optimized pattern
            pattern_similarity = self._calculate_pattern_similarity(
                context.original_pattern, 
                optimized_pattern
            )
            
            # Calculate effective learning rate based on feedback quality and quantity
            effective_rate = min(
                1.0, 
                context.learning_rate * (len(context.feedback_items) / 10)
            )
            
            # Update context with results
            context.metadata.update({
                "last_optimized_at": time.time(),
                "optimized_pattern": optimized_pattern,
                "pattern_similarity": pattern_similarity,
                "effective_rate": effective_rate,
                "positive_examples_count": len(context.positive_examples),
                "negative_examples_count": len(context.negative_examples)
            })
            
            # Return optimization results
            return {
                "pattern_id": pattern_id,
                "original_pattern": context.original_pattern,
                "optimized_pattern": optimized_pattern,
                "pattern_similarity": pattern_similarity,
                "positive_examples": context.positive_examples,
                "negative_examples": context.negative_examples,
                "effective_learning_rate": effective_rate,
                "evolution_result": evolution_result,
                "analysis_result": analysis_result,
                "timestamp": time.time()
            }
            
        except Exception as e:
            if not isinstance(e, RuntimeLearningError):
                raise RuntimeLearningError(
                    message=str(e),
                    learning_type="pattern",
                    details={"original_error": str(e)}
                )
            raise
    
    async def optimize_strategy(
        self, 
        strategy_id: str, 
        feedback: List[Dict[str, Any]], 
        effectiveness_score: float,
        learning_rate: float = 0.1
    ) -> Dict[str, Any]:
        """
        Optimize a search or analysis strategy based on feedback.
        
        Args:
            strategy_id: ID of the strategy to optimize
            feedback: List of feedback items on strategy results
            effectiveness_score: Overall effectiveness score (0.0-1.0)
            learning_rate: Rate of learning (0.0-1.0)
            
        Returns:
            Dictionary with optimization results
            
        Raises:
            RuntimeLearningError: If strategy optimization fails
        """
        try:
            # Get strategy information (from strategy library or other storage)
            strategy_info = await self._get_strategy_info(strategy_id)
            
            if not strategy_info:
                raise RuntimeLearningError(
                    message=f"Strategy not found: {strategy_id}",
                    learning_type="strategy"
                )
            
            # Get or create learning context
            if strategy_id not in self._strategy_contexts:
                self._strategy_contexts[strategy_id] = StrategyLearningContext(
                    strategy_id=strategy_id,
                    original_strategy=strategy_info,
                    learning_rate=learning_rate
                )
            
            context = self._strategy_contexts[strategy_id]
            
            # Update learning context with new feedback
            context.feedback_items.extend(feedback)
            context.effectiveness_score = effectiveness_score
            
            # Strategy optimization is more complex and would typically require
            # specialized algorithms based on the strategy type
            # For this placeholder implementation, we'll just generate some mock results
            
            # Calculate recommended adjustments (placeholder implementation)
            adjustments = self._calculate_strategy_adjustments(
                context.original_strategy,
                context.feedback_items,
                context.effectiveness_score,
                context.learning_rate
            )
            
            # Apply adjustments to create optimized strategy
            optimized_strategy = self._apply_strategy_adjustments(
                context.original_strategy,
                adjustments
            )
            
            # Update context with results
            context.metadata.update({
                "last_optimized_at": time.time(),
                "optimized_strategy": optimized_strategy,
                "adjustments": adjustments,
                "effectiveness_score": effectiveness_score
            })
            
            # Return optimization results
            return {
                "strategy_id": strategy_id,
                "original_strategy": context.original_strategy,
                "optimized_strategy": optimized_strategy,
                "adjustments": adjustments,
                "effectiveness_score": effectiveness_score,
                "feedback_count": len(context.feedback_items),
                "timestamp": time.time()
            }
            
        except Exception as e:
            if not isinstance(e, RuntimeLearningError):
                raise RuntimeLearningError(
                    message=str(e),
                    learning_type="strategy",
                    details={"original_error": str(e)}
                )
            raise
    
    def _calculate_pattern_similarity(self, pattern1: str, pattern2: str) -> float:
        """
        Calculate similarity between two patterns.
        
        Args:
            pattern1: First pattern
            pattern2: Second pattern
            
        Returns:
            Similarity score (0.0-1.0)
        """
        # Simple character-based similarity metric
        # More sophisticated metrics would be used in a real implementation
        
        if pattern1 == pattern2:
            return 1.0
        
        if not pattern1 or not pattern2:
            return 0.0
        
        # Calculate longest common subsequence length
        m, n = len(pattern1), len(pattern2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pattern1[i-1] == pattern2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        
        # Normalize by the average length of the two patterns
        return lcs_length / ((m + n) / 2)
    
    def _calculate_strategy_adjustments(
        self, 
        strategy: Dict[str, Any], 
        feedback: List[Dict[str, Any]],
        effectiveness_score: float,
        learning_rate: float
    ) -> Dict[str, Any]:
        """
        Calculate recommended adjustments for a strategy.
        
        Args:
            strategy: Original strategy configuration
            feedback: User feedback on strategy results
            effectiveness_score: Overall effectiveness score (0.0-1.0)
            learning_rate: Learning rate (0.0-1.0)
            
        Returns:
            Dictionary with recommended adjustments
        """
        # Placeholder implementation - in a real system, this would analyze feedback
        # and recommend specific adjustments to the strategy
        
        # Mock adjustments based on effectiveness score
        if effectiveness_score < 0.3:
            # Strategy needs major improvements
            return {
                "confidence_threshold": "decrease",
                "search_depth": "increase",
                "max_results": "increase",
                "context_window": "increase",
                "severity": "significant"
            }
        elif effectiveness_score < 0.7:
            # Strategy needs minor improvements
            return {
                "confidence_threshold": "slight_decrease",
                "search_depth": "slight_increase",
                "pattern_specificity": "adjust",
                "severity": "moderate"
            }
        else:
            # Strategy is performing well
            return {
                "pattern_specificity": "slight_adjust",
                "confidence_threshold": "fine_tune",
                "severity": "minor"
            }
    
    def _apply_strategy_adjustments(
        self, 
        strategy: Dict[str, Any], 
        adjustments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply recommended adjustments to a strategy.
        
        Args:
            strategy: Original strategy configuration
            adjustments: Recommended adjustments
            
        Returns:
            Optimized strategy configuration
        """
        # Deep copy the strategy to avoid modifying the original
        optimized = json.loads(json.dumps(strategy))
        
        # Placeholder implementation - in a real system, this would apply
        # specific adjustments to the strategy configuration
        
        # Apply mock adjustments based on severity
        severity = adjustments.get("severity", "minor")
        
        if "confidence_threshold" in strategy:
            if adjustments.get("confidence_threshold") == "decrease":
                optimized["confidence_threshold"] = max(0.1, strategy["confidence_threshold"] - 0.2)
            elif adjustments.get("confidence_threshold") == "slight_decrease":
                optimized["confidence_threshold"] = max(0.1, strategy["confidence_threshold"] - 0.1)
            elif adjustments.get("confidence_threshold") == "fine_tune":
                optimized["confidence_threshold"] = max(0.1, strategy["confidence_threshold"] - 0.05)
        
        if "search_depth" in strategy:
            if adjustments.get("search_depth") == "increase":
                optimized["search_depth"] = strategy["search_depth"] + 2
            elif adjustments.get("search_depth") == "slight_increase":
                optimized["search_depth"] = strategy["search_depth"] + 1
        
        if "max_results" in strategy:
            if adjustments.get("max_results") == "increase":
                optimized["max_results"] = strategy["max_results"] * 2
        
        if "context_window" in strategy:
            if adjustments.get("context_window") == "increase":
                optimized["context_window"] = strategy["context_window"] + 2
        
        # Add a note about the adjustments
        optimized["_adjusted"] = {
            "timestamp": time.time(),
            "severity": severity,
            "adjustments": adjustments
        }
        
        return optimized
    
    async def _get_pattern_info(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a pattern.
        
        Args:
            pattern_id: ID of the pattern
            
        Returns:
            Pattern information or None if not found
        """
        # Placeholder implementation - in a real system, this would retrieve
        # pattern information from a pattern library or other storage
        
        # Mock pattern information for demonstration
        if pattern_id == "test_pattern":
            return {
                "id": pattern_id,
                "pattern": r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
                "description": "Email address pattern",
                "is_regex": True,
                "case_sensitive": False
            }
        
        # In a real implementation, this would query a pattern library
        # from tsap.evolution.pattern_library import get_pattern_library
        # pattern_library = get_pattern_library()
        # return await pattern_library.get_pattern(pattern_id)
        
        # For now, just return a mock pattern
        return {
            "id": pattern_id,
            "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            "description": f"Pattern {pattern_id}",
            "is_regex": True,
            "case_sensitive": False
        }
    
    async def _get_strategy_info(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a strategy.
        
        Args:
            strategy_id: ID of the strategy
            
        Returns:
            Strategy information or None if not found
        """
        # Placeholder implementation - in a real system, this would retrieve
        # strategy information from a strategy library or other storage
        
        # Mock strategy information for demonstration
        if strategy_id == "test_strategy":
            return {
                "id": strategy_id,
                "name": "Test Strategy",
                "objective": "Find security vulnerabilities",
                "confidence_threshold": 0.7,
                "search_depth": 3,
                "max_results": 100,
                "context_window": 2,
                "patterns": [
                    {"pattern": r"password\s*=", "description": "Hardcoded password"},
                    {"pattern": r"api_key\s*=", "description": "Hardcoded API key"}
                ]
            }
        
        # In a real implementation, this would query a strategy library
        # For now, just return a mock strategy
        return {
            "id": strategy_id,
            "name": f"Strategy {strategy_id}",
            "objective": "Generic search strategy",
            "confidence_threshold": 0.5,
            "search_depth": 2,
            "max_results": 50,
            "context_window": 2,
            "patterns": [
                {"pattern": r"example", "description": "Example pattern"}
            ]
        }


# Global instance
_runtime_learning = None


def get_runtime_learning() -> RuntimeLearning:
    """
    Get or create the global RuntimeLearning instance.
    
    Returns:
        RuntimeLearning instance
    """
    global _runtime_learning
    if _runtime_learning is None:
        _runtime_learning = RuntimeLearning()
    return _runtime_learning


async def apply_runtime_learning(params: RuntimeLearningParams) -> RuntimeLearningResult:
    """
    Apply runtime learning to optimize patterns or strategies.
    
    Args:
        params: Runtime learning parameters
            - learning_type: Type of learning ("pattern_optimization" or "strategy_optimization")
            - target_pattern_id: ID of the pattern to optimize (for pattern learning)
            - target_strategy_id: ID of the strategy to optimize (for strategy learning)
            - feedback: List of feedback items
            - effectiveness_score: Overall effectiveness score (for strategy learning)
            - learning_rate: Rate of learning (0.0-1.0)
            
    Returns:
        Runtime learning results
    """
    # Get runtime learning instance
    learning = get_runtime_learning()
    
    # Process the request based on learning type
    learning_type = params.learning_type
    
    if learning_type == "pattern_optimization":
        # Optimize pattern
        result = await learning.optimize_pattern(
            pattern_id=params.target_pattern_id,
            feedback=params.feedback,
            learning_rate=params.learning_rate,
            evolve=True
        )
        
        return RuntimeLearningResult(
            learning_type=learning_type,
            success=True,
            optimized_pattern=result["optimized_pattern"],
            original_pattern=result["original_pattern"],
            pattern_similarity=result["pattern_similarity"],
            positive_examples=result["positive_examples"],
            negative_examples=result["negative_examples"],
            effectiveness_score=None,
            timestamp=result["timestamp"],
            details=result
        )
        
    elif learning_type == "strategy_optimization":
        # Optimize strategy
        result = await learning.optimize_strategy(
            strategy_id=params.target_strategy_id,
            feedback=params.feedback,
            effectiveness_score=params.effectiveness_score,
            learning_rate=params.learning_rate
        )
        
        return RuntimeLearningResult(
            learning_type=learning_type,
            success=True,
            optimized_strategy=result["optimized_strategy"],
            original_strategy=result["original_strategy"],
            effectiveness_score=params.effectiveness_score,
            adjustments=result["adjustments"],
            timestamp=result["timestamp"],
            details=result
        )
        
    else:
        # Unknown learning type
        raise RuntimeLearningError(
            message=f"Unknown learning type: {learning_type}",
            learning_type=learning_type
        )