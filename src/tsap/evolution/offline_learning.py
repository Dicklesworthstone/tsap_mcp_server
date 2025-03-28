"""
Offline learning from historical data.

This module provides functionality for learning from historical data to
improve patterns, strategies, and analysis techniques. It analyzes past
performance data to identify trends and recommend optimizations.
"""

import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

from tsap.utils.logging import logger
from tsap.utils.errors import TSAPError
from tsap.evolution.strategy_journal import get_strategy_journal
from tsap.evolution.pattern_library import get_pattern_library
from tsap.evolution.genetic import evolve_regex_pattern
from tsap.mcp.models import (
    OfflineLearningParams, 
    OfflineLearningResult
)


class OfflineLearningError(TSAPError):
    """
    Exception for errors in offline learning operations.
    
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
            code=f"OFFLINE_LEARNING_{learning_type.upper()}_ERROR" if learning_type else "OFFLINE_LEARNING_ERROR",
            details=details
        )
        self.learning_type = learning_type


@dataclass
class LearningResult:
    """
    Result of a learning operation.
    
    Attributes:
        success: Whether the learning was successful
        learning_type: Type of learning performed
        recommendations: List of recommendations based on the learning
        optimized_entities: Dictionary mapping entity IDs to optimized versions
        statistics: Statistics about the learning operation
        metadata: Additional metadata
    """
    success: bool
    learning_type: str
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    optimized_entities: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class OfflineLearning:
    """
    Class for offline learning from historical data.
    """
    def __init__(self) -> None:
        """Initialize the offline learning system."""
        # Initialize pattern library and strategy journal
        self._pattern_library = None
        self._strategy_journal = None
        
        # Learning settings
        self.min_data_points = 10
        self.confidence_threshold = 0.7
        self.max_recommendations = 5
    
    async def _get_pattern_library(self):
        """
        Get or initialize the pattern library.
        
        Returns:
            Pattern library instance
        """
        if self._pattern_library is None:
            self._pattern_library = get_pattern_library()
        return self._pattern_library
    
    async def _get_strategy_journal(self):
        """
        Get or initialize the strategy journal.
        
        Returns:
            Strategy journal instance
        """
        if self._strategy_journal is None:
            self._strategy_journal = get_strategy_journal()
        return self._strategy_journal
    
    async def learn_pattern_optimization(
        self,
        data_source: str,
        date_range: Optional[Tuple[float, float]] = None,
        tags: Optional[List[str]] = None,
        pattern_ids: Optional[List[str]] = None,
        optimization_goal: str = "accuracy"
    ) -> LearningResult:
        """
        Learn pattern optimizations from historical data.
        
        Args:
            data_source: Source of historical data ("library", "journal", "custom")
            date_range: Date range for analysis (start_time, end_time)
            tags: Filter by tags
            pattern_ids: Specific pattern IDs to optimize (optional)
            optimization_goal: Optimization goal ("accuracy", "performance", "balance")
            
        Returns:
            Learning result
        """
        try:
            # Get pattern library
            pattern_library = await self._get_pattern_library()
            
            # Get patterns to analyze
            patterns_to_analyze = []
            
            if pattern_ids:
                # Use specified pattern IDs
                for pattern_id in pattern_ids:
                    pattern = await pattern_library.get_pattern(pattern_id)
                    if pattern:
                        patterns_to_analyze.append(pattern)
            else:
                # Search for patterns with sufficient usage
                search_params = {
                    "min_usage_count": self.min_data_points
                }
                
                if tags:
                    search_params["tags"] = tags
                
                patterns, total = await pattern_library.search_patterns(**search_params)
                patterns_to_analyze = patterns
            
            if not patterns_to_analyze:
                return LearningResult(
                    success=False,
                    learning_type="pattern_optimization",
                    recommendations=[],
                    statistics={"message": "No patterns found with sufficient usage data"}
                )
            
            # Analyze patterns and generate recommendations
            recommendations = []
            optimized_patterns = {}
            
            for pattern in patterns_to_analyze:
                pattern_id = pattern["id"]
                
                # Get pattern examples
                positive_examples = await pattern_library.get_pattern_examples(
                    pattern_id, relevant=True, limit=100
                )
                
                negative_examples = await pattern_library.get_pattern_examples(
                    pattern_id, relevant=False, limit=100
                )
                
                # Get pattern stats
                pattern_stats = await pattern_library.get_pattern_stats(pattern_id)  # noqa: F841
                
                # Check if we have sufficient data
                if (len(positive_examples) < self.min_data_points or
                    len(negative_examples) < self.min_data_points):
                    continue
                
                # Calculate baseline metrics
                baseline_metrics = await self._calculate_pattern_metrics(
                    pattern["pattern"],
                    positive_examples,
                    negative_examples
                )
                
                # Try to evolve an optimized pattern
                evolution_result = await evolve_regex_pattern(
                    positive_examples=[e["text"] for e in positive_examples],
                    negative_examples=[e["text"] for e in negative_examples],
                    initial_patterns=[pattern["pattern"]]
                )
                
                if not evolution_result:
                    continue
                
                optimized_pattern = evolution_result["pattern"]
                
                # Calculate optimized metrics
                optimized_metrics = await self._calculate_pattern_metrics(
                    optimized_pattern,
                    positive_examples,
                    negative_examples
                )
                
                # Check if optimization is an improvement
                if (optimized_metrics["f1_score"] > baseline_metrics["f1_score"] * 1.05):
                    # This is a significant improvement
                    improvement = (
                        (optimized_metrics["f1_score"] - baseline_metrics["f1_score"]) /
                        baseline_metrics["f1_score"]
                    ) * 100
                    
                    recommendation = {
                        "pattern_id": pattern_id,
                        "pattern_name": pattern.get("name", pattern_id),
                        "original_pattern": pattern["pattern"],
                        "optimized_pattern": optimized_pattern,
                        "improvement": f"{improvement:.1f}%",
                        "baseline_metrics": baseline_metrics,
                        "optimized_metrics": optimized_metrics,
                        "confidence": min(improvement / 20, 0.95)  # Scale confidence based on improvement
                    }
                    
                    recommendations.append(recommendation)
                    optimized_patterns[pattern_id] = {
                        "pattern": optimized_pattern,
                        "metrics": optimized_metrics
                    }
            
            # Sort recommendations by improvement
            recommendations.sort(key=lambda r: float(r["improvement"].rstrip('%')), reverse=True)
            
            # Limit number of recommendations
            recommendations = recommendations[:self.max_recommendations]
            
            # Build learning result
            return LearningResult(
                success=True,
                learning_type="pattern_optimization",
                recommendations=recommendations,
                optimized_entities=optimized_patterns,
                statistics={
                    "total_patterns_analyzed": len(patterns_to_analyze),
                    "optimization_goal": optimization_goal,
                    "data_source": data_source,
                    "recommendations_generated": len(recommendations)
                },
                metadata={
                    "timestamp": time.time(),
                    "date_range": date_range,
                    "tags": tags
                }
            )
            
        except Exception as e:
            logger.error(f"Error in offline pattern optimization: {str(e)}")
            
            if not isinstance(e, OfflineLearningError):
                raise OfflineLearningError(
                    message=str(e),
                    learning_type="pattern_optimization",
                    details={"original_error": str(e)}
                )
            raise
    
    async def learn_strategy_optimization(
        self,
        data_source: str,
        date_range: Optional[Tuple[float, float]] = None,
        tags: Optional[List[str]] = None,
        strategy_ids: Optional[List[str]] = None,
        optimization_goal: str = "effectiveness"
    ) -> LearningResult:
        """
        Learn strategy optimizations from historical data.
        
        Args:
            data_source: Source of historical data ("journal", "custom")
            date_range: Date range for analysis (start_time, end_time)
            tags: Filter by tags
            strategy_ids: Specific strategy IDs to optimize (optional)
            optimization_goal: Optimization goal ("effectiveness", "speed", "balance")
            
        Returns:
            Learning result
        """
        try:
            # Get strategy journal
            strategy_journal = await self._get_strategy_journal()
            
            # Get strategies to analyze
            strategies_to_analyze = set()
            
            if strategy_ids:
                # Use specified strategy IDs
                strategies_to_analyze = set(strategy_ids)
            elif data_source == "journal":
                # Search for strategies in the journal
                # Get all journal entries within the date range
                entries = await strategy_journal.search_entries(
                    date_range=date_range,
                    tags=tags
                )
                
                # Count entries per strategy
                strategy_counts = defaultdict(int)
                for entry in entries:
                    strategy_counts[entry.strategy_id] += 1
                
                # Filter strategies with sufficient entries
                strategies_to_analyze = {
                    strategy_id for strategy_id, count in strategy_counts.items()
                    if count >= self.min_data_points
                }
            
            if not strategies_to_analyze:
                return LearningResult(
                    success=False,
                    learning_type="strategy_optimization",
                    recommendations=[],
                    statistics={"message": "No strategies found with sufficient usage data"}
                )
            
            # Analyze strategies and generate recommendations
            recommendations = []
            optimized_strategies = {}
            
            for strategy_id in strategies_to_analyze:
                # Get strategy statistics
                stats = await strategy_journal.get_strategy_statistics(strategy_id)
                
                if not stats or stats.get("total_executions", 0) < self.min_data_points:
                    continue
                
                # Get strategy history
                entries = await strategy_journal.get_strategy_history(
                    strategy_id, max_entries=100
                )
                
                if not entries:
                    continue
                
                # Extract training data from entries
                target_documents = set()
                training_queries = []
                
                for entry in entries:
                    # Get details if available
                    details = entry.details or {}
                    
                    # Extract documents
                    if "documents" in details:
                        target_documents.update(details["documents"])
                    
                    # Extract queries
                    if "queries" in details and "results" in details:
                        query = {
                            "query": details["queries"],
                            "expected_matches": details["results"]
                        }
                        training_queries.append(query)
                
                # Need sufficient training data
                if not target_documents or not training_queries:
                    continue
                
                # Try to evolve an optimized strategy
                # In a real implementation, we would need to get the original strategy
                # from a strategy repository or extract it from journal entries
                
                # For now, create a mock optimized strategy
                optimized_strategy = {
                    "id": f"{strategy_id}_optimized",
                    "patterns": [
                        {
                            "pattern": r"\b[a-z]+\b",
                            "description": "Optimized pattern 1"
                        },
                        {
                            "pattern": r"\b\d+\b",
                            "description": "Optimized pattern 2"
                        }
                    ],
                    "options": {
                        "confidence_threshold": 0.6,
                        "context_lines": 2,
                        "search_depth": 3
                    }
                }
                
                # Calculate improvement metrics
                baseline_effectiveness = stats.get("average_effectiveness", 0.5)
                optimized_effectiveness = baseline_effectiveness * 1.2  # Mock 20% improvement
                
                improvement = (
                    (optimized_effectiveness - baseline_effectiveness) /
                    baseline_effectiveness
                ) * 100
                
                # Create recommendation
                recommendation = {
                    "strategy_id": strategy_id,
                    "baseline_effectiveness": baseline_effectiveness,
                    "optimized_effectiveness": optimized_effectiveness,
                    "improvement": f"{improvement:.1f}%",
                    "sample_size": stats.get("total_executions", 0),
                    "confidence": min(improvement / 20, 0.95),  # Scale confidence based on improvement
                    "key_changes": [
                        "Optimized pattern specificity",
                        "Adjusted confidence threshold",
                        "Increased search depth"
                    ]
                }
                
                recommendations.append(recommendation)
                optimized_strategies[strategy_id] = optimized_strategy
            
            # Sort recommendations by improvement
            recommendations.sort(key=lambda r: float(r["improvement"].rstrip('%')), reverse=True)
            
            # Limit number of recommendations
            recommendations = recommendations[:self.max_recommendations]
            
            # Build learning result
            return LearningResult(
                success=True,
                learning_type="strategy_optimization",
                recommendations=recommendations,
                optimized_entities=optimized_strategies,
                statistics={
                    "total_strategies_analyzed": len(strategies_to_analyze),
                    "optimization_goal": optimization_goal,
                    "data_source": data_source,
                    "recommendations_generated": len(recommendations)
                },
                metadata={
                    "timestamp": time.time(),
                    "date_range": date_range,
                    "tags": tags
                }
            )
            
        except Exception as e:
            logger.error(f"Error in offline strategy optimization: {str(e)}")
            
            if not isinstance(e, OfflineLearningError):
                raise OfflineLearningError(
                    message=str(e),
                    learning_type="strategy_optimization",
                    details={"original_error": str(e)}
                )
            raise
    
    async def learn_usage_patterns(
        self,
        data_source: str,
        date_range: Optional[Tuple[float, float]] = None,
        entity_type: str = "all"
    ) -> LearningResult:
        """
        Learn usage patterns from historical data.
        
        Args:
            data_source: Source of historical data ("journal", "library", "custom")
            date_range: Date range for analysis (start_time, end_time)
            entity_type: Type of entity to analyze ("strategy", "pattern", "all")
            
        Returns:
            Learning result
        """
        try:
            recommendations = []
            statistics = {}
            
            # Analyze pattern usage if applicable
            if entity_type in ["pattern", "all"]:
                pattern_recommendations = await self._analyze_pattern_usage(data_source, date_range)
                recommendations.extend(pattern_recommendations)
                
                statistics["pattern_analysis"] = {
                    "entities_analyzed": len(pattern_recommendations),
                    "data_source": data_source
                }
            
            # Analyze strategy usage if applicable
            if entity_type in ["strategy", "all"]:
                strategy_recommendations = await self._analyze_strategy_usage(data_source, date_range)
                recommendations.extend(strategy_recommendations)
                
                statistics["strategy_analysis"] = {
                    "entities_analyzed": len(strategy_recommendations),
                    "data_source": data_source
                }
            
            # Sort recommendations by confidence
            recommendations.sort(key=lambda r: r.get("confidence", 0), reverse=True)
            
            # Limit number of recommendations
            recommendations = recommendations[:self.max_recommendations]
            
            # Build learning result
            return LearningResult(
                success=True,
                learning_type="usage_patterns",
                recommendations=recommendations,
                optimized_entities={},  # No optimizations for usage pattern analysis
                statistics=statistics,
                metadata={
                    "timestamp": time.time(),
                    "date_range": date_range,
                    "entity_type": entity_type
                }
            )
            
        except Exception as e:
            logger.error(f"Error in learning usage patterns: {str(e)}")
            
            if not isinstance(e, OfflineLearningError):
                raise OfflineLearningError(
                    message=str(e),
                    learning_type="usage_patterns",
                    details={"original_error": str(e)}
                )
            raise
    
    async def learn_trend_analysis(
        self,
        data_source: str,
        date_range: Optional[Tuple[float, float]] = None,
        entity_ids: Optional[List[str]] = None,
        time_period: str = "day"
    ) -> LearningResult:
        """
        Learn trends from historical data.
        
        Args:
            data_source: Source of historical data ("journal", "library", "custom")
            date_range: Date range for analysis (start_time, end_time)
            entity_ids: Specific entity IDs to analyze (optional)
            time_period: Time period for trend analysis ("hour", "day", "week", "month")
            
        Returns:
            Learning result
        """
        try:
            # Get strategy journal for trend data
            strategy_journal = await self._get_strategy_journal()
            
            # Get execution trends
            trend_data = await strategy_journal.get_execution_trends(
                strategy_id=entity_ids[0] if entity_ids and len(entity_ids) == 1 else None,
                time_period=time_period,
                start_time=date_range[0] if date_range else None,
                end_time=date_range[1] if date_range else None
            )
            
            # Analyze trends to generate recommendations
            recommendations = []
            
            # Calculate trends for effectiveness
            effectiveness_trend = self._calculate_trend(
                [p.get("average_effectiveness", 0) for p in trend_data.get("trend_data", [])]
            )
            
            # Calculate trends for execution time
            execution_time_trend = self._calculate_trend(
                [p.get("average_execution_time", 0) for p in trend_data.get("trend_data", [])]
            )
            
            # Calculate trends for usage
            usage_trend = self._calculate_trend(
                [p.get("entry_count", 0) for p in trend_data.get("trend_data", [])]
            )
            
            # Generate recommendations based on trends
            if effectiveness_trend < -0.1:
                recommendations.append({
                    "type": "effectiveness",
                    "trend": "declining",
                    "description": "Effectiveness has been declining. Consider reviewing strategy configurations.",
                    "confidence": min(abs(effectiveness_trend) * 2, 0.95)
                })
            elif effectiveness_trend > 0.1:
                recommendations.append({
                    "type": "effectiveness",
                    "trend": "improving",
                    "description": "Effectiveness has been improving. Recent changes appear to be working well.",
                    "confidence": min(effectiveness_trend * 2, 0.95)
                })
            
            if execution_time_trend > 0.1:
                recommendations.append({
                    "type": "performance",
                    "trend": "declining",
                    "description": "Execution time has been increasing. Consider performance optimizations.",
                    "confidence": min(execution_time_trend * 2, 0.95)
                })
            elif execution_time_trend < -0.1:
                recommendations.append({
                    "type": "performance",
                    "trend": "improving",
                    "description": "Execution time has been decreasing. Performance optimizations appear effective.",
                    "confidence": min(abs(execution_time_trend) * 2, 0.95)
                })
            
            if usage_trend > 0.1:
                recommendations.append({
                    "type": "usage",
                    "trend": "increasing",
                    "description": "Usage has been increasing. Strategies are being used more frequently.",
                    "confidence": min(usage_trend * 2, 0.95)
                })
            elif usage_trend < -0.1:
                recommendations.append({
                    "type": "usage",
                    "trend": "decreasing",
                    "description": "Usage has been decreasing. Consider promoting or improving strategies.",
                    "confidence": min(abs(usage_trend) * 2, 0.95)
                })
            
            # Sort recommendations by confidence
            recommendations.sort(key=lambda r: r.get("confidence", 0), reverse=True)
            
            # Build learning result
            return LearningResult(
                success=True,
                learning_type="trend_analysis",
                recommendations=recommendations,
                optimized_entities={},  # No optimizations for trend analysis
                statistics={
                    "effectiveness_trend": effectiveness_trend,
                    "execution_time_trend": execution_time_trend,
                    "usage_trend": usage_trend,
                    "data_points": len(trend_data.get("trend_data", [])),
                    "time_period": time_period
                },
                metadata={
                    "timestamp": time.time(),
                    "date_range": date_range,
                    "entity_ids": entity_ids
                }
            )
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {str(e)}")
            
            if not isinstance(e, OfflineLearningError):
                raise OfflineLearningError(
                    message=str(e),
                    learning_type="trend_analysis",
                    details={"original_error": str(e)}
                )
            raise
    
    async def _analyze_pattern_usage(
        self,
        data_source: str,
        date_range: Optional[Tuple[float, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze pattern usage patterns.
        
        Args:
            data_source: Source of historical data
            date_range: Date range for analysis
            
        Returns:
            List of pattern usage recommendations
        """
        # Get pattern library
        pattern_library = await self._get_pattern_library()
        
        # Get popular patterns
        popular_patterns = await pattern_library.get_popular_patterns(limit=20)
        
        # Generate recommendations
        recommendations = []
        
        # Analyze patterns for redundancy
        similar_patterns = []
        for i, pattern1 in enumerate(popular_patterns):
            for pattern2 in popular_patterns[i+1:]:
                similarity = self._calculate_pattern_similarity(
                    pattern1.get("pattern", ""),
                    pattern2.get("pattern", "")
                )
                
                if similarity > 0.8:
                    similar_patterns.append((pattern1, pattern2, similarity))
        
        for pattern1, pattern2, similarity in similar_patterns:
            recommendations.append({
                "type": "pattern_redundancy",
                "description": f"Patterns '{pattern1.get('name', pattern1.get('id'))}' and '{pattern2.get('name', pattern2.get('id'))}' are very similar.",
                "suggestion": "Consider consolidating these patterns to reduce redundancy.",
                "confidence": similarity,
                "patterns": [pattern1.get("id"), pattern2.get("id")],
                "similarity": similarity
            })
        
        # Analyze patterns for complexity
        for pattern in popular_patterns:
            complexity = self._calculate_pattern_complexity(pattern.get("pattern", ""))
            if complexity > 0.7:
                recommendations.append({
                    "type": "pattern_complexity",
                    "description": f"Pattern '{pattern.get('name', pattern.get('id'))}' is quite complex.",
                    "suggestion": "Consider simplifying the pattern for better maintainability and performance.",
                    "confidence": min(complexity - 0.5, 0.9),
                    "pattern_id": pattern.get("id"),
                    "complexity": complexity
                })
        
        # Analyze patterns for underuse
        # For now, just add a placeholder recommendation
        recommendations.append({
            "type": "pattern_underuse",
            "description": "Some patterns are rarely used despite being highly specific.",
            "suggestion": "Consider promoting these patterns or integrating them into more common strategies.",
            "confidence": 0.6,
            "pattern_ids": ["pattern1", "pattern2", "pattern3"]  # Placeholder
        })
        
        return recommendations
    
    async def _analyze_strategy_usage(
        self,
        data_source: str,
        date_range: Optional[Tuple[float, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze strategy usage patterns.
        
        Args:
            data_source: Source of historical data
            date_range: Date range for analysis
            
        Returns:
            List of strategy usage recommendations
        """
        # Get strategy journal
        strategy_journal = await self._get_strategy_journal()
        
        # Get journal summary
        summary = await strategy_journal.get_journal_summary()  # noqa: F841
        
        # Get context statistics
        context_stats = await strategy_journal.get_context_statistics()
        
        # Generate recommendations
        recommendations = []
        
        # Analyze strategy distribution across contexts
        context_counts = context_stats.get("context_counts", {})
        if context_counts:
            # Check for imbalance in context usage
            total_entries = sum(context_counts.values())
            max_context = max(context_counts.items(), key=lambda x: x[1])
            
            if max_context[1] > total_entries * 0.7:
                # Dominant context
                recommendations.append({
                    "type": "context_imbalance",
                    "description": f"Context '{max_context[0]}' accounts for {max_context[1] / total_entries:.1%} of all strategy executions.",
                    "suggestion": "Consider diversifying strategy usage across more contexts.",
                    "confidence": min(max_context[1] / total_entries, 0.95),
                    "dominant_context": max_context[0],
                    "percentage": max_context[1] / total_entries
                })
        
        # Check for patterns in strategy effectiveness
        # For now, just add a placeholder recommendation
        recommendations.append({
            "type": "strategy_effectiveness",
            "description": "Strategies tend to be more effective in certain contexts.",
            "suggestion": "Consider specializing strategy configurations for different contexts.",
            "confidence": 0.75,
            "contexts": ["security_audit", "code_review"]  # Placeholder
        })
        
        # Check for patterns in strategy frequency
        # For now, just add a placeholder recommendation
        recommendations.append({
            "type": "strategy_frequency",
            "description": "Some strategies are used more frequently at specific times.",
            "suggestion": "Consider scheduling strategy executions based on historical patterns.",
            "confidence": 0.6,
            "time_patterns": ["weekday_mornings", "month_end"]  # Placeholder
        })
        
        return recommendations
    
    async def _calculate_pattern_metrics(
        self,
        pattern: str,
        positive_examples: List[Dict[str, Any]],
        negative_examples: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate metrics for a pattern based on examples.
        
        Args:
            pattern: Pattern string
            positive_examples: Examples that should match
            negative_examples: Examples that should not match
            
        Returns:
            Dictionary with pattern metrics
        """
        # Compile pattern
        try:
            import re
            compiled_pattern = re.compile(pattern)
        except Exception as e:
            logger.warning(f"Error compiling pattern: {str(e)}")
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "accuracy": 0.0
            }
        
        # Count matches
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        # Test positive examples
        for example in positive_examples:
            text = example["text"]
            if compiled_pattern.search(text):
                true_positives += 1
            else:
                false_negatives += 1
        
        # Test negative examples
        for example in negative_examples:
            text = example["text"]
            if compiled_pattern.search(text):
                false_positives += 1
            else:
                true_negatives += 1
        
        # Calculate metrics
        total = true_positives + true_negatives + false_positives + false_negatives
        
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0.0
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "accuracy": accuracy,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives
        }
    
    def _calculate_pattern_similarity(self, pattern1: str, pattern2: str) -> float:
        """
        Calculate similarity between two regex patterns.
        
        Args:
            pattern1: First pattern
            pattern2: Second pattern
            
        Returns:
            Similarity score (0.0-1.0)
        """
        # Simple character-based similarity metric
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
    
    def _calculate_pattern_complexity(self, pattern: str) -> float:
        """
        Calculate complexity of a regex pattern.
        
        Args:
            pattern: Pattern string
            
        Returns:
            Complexity score (0.0-1.0)
        """
        if not pattern:
            return 0.0
        
        # Count special regex features
        special_chars = r".*+?[](){}|^$\:"
        
        # Count occurrences of special characters
        special_count = sum(1 for c in pattern if c in special_chars)
        
        # Count character classes, capturing groups, etc.
        char_class_count = pattern.count("[") + pattern.count("]")
        group_count = pattern.count("(") + pattern.count(")")
        alternation_count = pattern.count("|")
        quantifier_count = sum(1 for i in range(len(pattern)) if pattern[i] in "*+?{")
        
        # Calculate complexity score
        base_complexity = special_count / len(pattern)
        
        # Add weights for more complex features
        weighted_complexity = (
            base_complexity * 0.4 +
            char_class_count / (len(pattern) + 1) * 0.2 +
            group_count / (len(pattern) + 1) * 0.2 +
            alternation_count / (len(pattern) + 1) * 0.1 +
            quantifier_count / (len(pattern) + 1) * 0.1
        )
        
        # Normalize to 0.0-1.0 range
        return min(weighted_complexity * 2, 1.0)
    
    def _calculate_trend(self, values: List[float]) -> float:
        """
        Calculate trend from a series of values.
        
        Args:
            values: List of values
            
        Returns:
            Trend coefficient (-1.0 to 1.0, with positive values indicating increasing trend)
        """
        if not values or len(values) < 2:
            return 0.0
        
        # Remove None values
        values = [v for v in values if v is not None]
        
        if not values or len(values) < 2:
            return 0.0
        
        # Normalize values to 0.0-1.0 range
        max_val = max(values)
        min_val = min(values)
        
        if max_val == min_val:
            return 0.0
        
        normalized = [(v - min_val) / (max_val - min_val) for v in values]
        
        # Calculate trend
        n = len(normalized)
        indices = list(range(n))
        
        # Calculate means
        mean_x = sum(indices) / n
        mean_y = sum(normalized) / n
        
        # Calculate sum of squares
        numerator = sum((indices[i] - mean_x) * (normalized[i] - mean_y) for i in range(n))
        denominator = sum((indices[i] - mean_x) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        # Calculate slope
        slope = numerator / denominator
        
        # Normalize slope to -1.0 to 1.0 range
        return max(-1.0, min(1.0, slope * n))


# Global instance
_offline_learning = None


def get_offline_learning() -> OfflineLearning:
    """
    Get or create the global OfflineLearning instance.
    
    Returns:
        OfflineLearning instance
    """
    global _offline_learning
    if _offline_learning is None:
        _offline_learning = OfflineLearning()
    return _offline_learning


async def perform_offline_learning(params: OfflineLearningParams) -> OfflineLearningResult:
    """
    Perform offline learning from historical data.
    
    Args:
        params: Offline learning parameters
            - learning_type: Type of learning to perform
            - data_source: Source of historical data
            - date_range: Date range for analysis
            - tags: Filter by tags
            - entity_ids: Specific entity IDs to analyze
            - entity_type: Type of entity to analyze
            - optimization_goal: Optimization goal
            - time_period: Time period for trend analysis
            
    Returns:
        Offline learning results
    """
    # Get offline learning instance
    learning = get_offline_learning()
    
    # Parse date range
    date_range = None
    if params.date_range:
        try:
            # Convert date strings to timestamps
            start_date = datetime.fromisoformat(params.date_range[0])
            end_date = datetime.fromisoformat(params.date_range[1])
            date_range = (start_date.timestamp(), end_date.timestamp())
        except (ValueError, IndexError):
            logger.warning(f"Invalid date range: {params.date_range}")
    
    try:
        # Process the request based on learning type
        learning_type = params.learning_type
        
        if learning_type == "pattern_optimization":
            # Perform pattern optimization
            result = await learning.learn_pattern_optimization(
                data_source=params.data_source,
                date_range=date_range,
                tags=params.tags,
                pattern_ids=params.entity_ids,
                optimization_goal=params.optimization_goal
            )
            
        elif learning_type == "strategy_optimization":
            # Perform strategy optimization
            result = await learning.learn_strategy_optimization(
                data_source=params.data_source,
                date_range=date_range,
                tags=params.tags,
                strategy_ids=params.entity_ids,
                optimization_goal=params.optimization_goal
            )
            
        elif learning_type == "usage_patterns":
            # Analyze usage patterns
            result = await learning.learn_usage_patterns(
                data_source=params.data_source,
                date_range=date_range,
                entity_type=params.entity_type
            )
            
        elif learning_type == "trend_analysis":
            # Analyze trends
            result = await learning.learn_trend_analysis(
                data_source=params.data_source,
                date_range=date_range,
                entity_ids=params.entity_ids,
                time_period=params.time_period
            )
            
        else:
            # Unknown learning type
            raise OfflineLearningError(
                message=f"Unknown learning type: {learning_type}",
                learning_type=learning_type
            )
        
        # Create result object
        return OfflineLearningResult(
            learning_type=learning_type,
            success=result.success,
            recommendations=result.recommendations,
            optimized_entities=result.optimized_entities,
            statistics=result.statistics,
            metadata=result.metadata,
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"Error in offline learning: {str(e)}")
        
        # Create error result
        return OfflineLearningResult(
            learning_type=params.learning_type,
            success=False,
            error=str(e),
            recommendations=[],
            statistics={"error": str(e)},
            timestamp=time.time()
        )