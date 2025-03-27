"""
Metrics for evaluating search patterns and strategies in the TSAP MCP Server.

This module defines metrics used to evaluate the effectiveness and efficiency
of search patterns, extraction strategies, and analysis approaches.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field



@dataclass
class SearchMetrics:
    """Metrics for evaluating search operations."""
    # Results metrics
    true_positives: int = 0  # Matches that are actually relevant
    false_positives: int = 0  # Matches that are not relevant
    true_negatives: int = 0  # Non-matches that are correctly excluded
    false_negatives: int = 0  # Relevant items that were not matched
    
    # Performance metrics
    execution_time: float = 0.0  # Seconds
    files_processed: int = 0
    lines_processed: int = 0
    memory_usage: float = 0.0  # MB
    
    # Derived metrics (calculated on demand)
    _precision: Optional[float] = None
    _recall: Optional[float] = None
    _f1_score: Optional[float] = None
    _accuracy: Optional[float] = None
    _specificity: Optional[float] = None
    
    @property
    def precision(self) -> float:
        """
        Calculate precision (positive predictive value).
        
        Precision = TP / (TP + FP)
        
        Returns:
            Precision score (0-1)
        """
        if self._precision is None:
            if self.true_positives + self.false_positives == 0:
                self._precision = 0.0
            else:
                self._precision = self.true_positives / (self.true_positives + self.false_positives)
        return self._precision
    
    @property
    def recall(self) -> float:
        """
        Calculate recall (sensitivity, true positive rate).
        
        Recall = TP / (TP + FN)
        
        Returns:
            Recall score (0-1)
        """
        if self._recall is None:
            if self.true_positives + self.false_negatives == 0:
                self._recall = 0.0
            else:
                self._recall = self.true_positives / (self.true_positives + self.false_negatives)
        return self._recall
    
    @property
    def f1_score(self) -> float:
        """
        Calculate F1 score (harmonic mean of precision and recall).
        
        F1 = 2 * (precision * recall) / (precision + recall)
        
        Returns:
            F1 score (0-1)
        """
        if self._f1_score is None:
            if self.precision + self.recall == 0:
                self._f1_score = 0.0
            else:
                self._f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        return self._f1_score
    
    @property
    def accuracy(self) -> float:
        """
        Calculate accuracy.
        
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        
        Returns:
            Accuracy score (0-1)
        """
        if self._accuracy is None:
            total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
            if total == 0:
                self._accuracy = 0.0
            else:
                self._accuracy = (self.true_positives + self.true_negatives) / total
        return self._accuracy
    
    @property
    def specificity(self) -> float:
        """
        Calculate specificity (true negative rate).
        
        Specificity = TN / (TN + FP)
        
        Returns:
            Specificity score (0-1)
        """
        if self._specificity is None:
            if self.true_negatives + self.false_positives == 0:
                self._specificity = 0.0
            else:
                self._specificity = self.true_negatives / (self.true_negatives + self.false_positives)
        return self._specificity
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metrics to a dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "result_metrics": {
                "true_positives": self.true_positives,
                "false_positives": self.false_positives,
                "true_negatives": self.true_negatives,
                "false_negatives": self.false_negatives,
                
                "precision": self.precision,
                "recall": self.recall,
                "f1_score": self.f1_score,
                "accuracy": self.accuracy,
                "specificity": self.specificity
            },
            "performance_metrics": {
                "execution_time": self.execution_time,
                "files_processed": self.files_processed,
                "lines_processed": self.lines_processed,
                "memory_usage": self.memory_usage,
                "processing_rate": self.lines_processed / self.execution_time if self.execution_time > 0 else 0
            }
        }


@dataclass
class ExtractionMetrics:
    """Metrics for evaluating information extraction operations."""
    # Results metrics
    correct_extractions: int = 0  # Correctly extracted items
    incorrect_extractions: int = 0  # Incorrectly extracted items
    missed_extractions: int = 0  # Items that should have been extracted but weren't
    
    # Quality metrics
    extraction_completeness: Dict[str, float] = field(default_factory=dict)  # By field type
    extraction_accuracy: Dict[str, float] = field(default_factory=dict)  # By field type
    
    # Performance metrics
    execution_time: float = 0.0  # Seconds
    documents_processed: int = 0
    fields_processed: int = 0
    
    # Derived metrics (calculated on demand)
    _precision: Optional[float] = None
    _recall: Optional[float] = None
    _f1_score: Optional[float] = None
    
    @property
    def precision(self) -> float:
        """
        Calculate precision for extractions.
        
        Precision = correct / (correct + incorrect)
        
        Returns:
            Precision score (0-1)
        """
        if self._precision is None:
            if self.correct_extractions + self.incorrect_extractions == 0:
                self._precision = 0.0
            else:
                self._precision = self.correct_extractions / (self.correct_extractions + self.incorrect_extractions)
        return self._precision
    
    @property
    def recall(self) -> float:
        """
        Calculate recall for extractions.
        
        Recall = correct / (correct + missed)
        
        Returns:
            Recall score (0-1)
        """
        if self._recall is None:
            if self.correct_extractions + self.missed_extractions == 0:
                self._recall = 0.0
            else:
                self._recall = self.correct_extractions / (self.correct_extractions + self.missed_extractions)
        return self._recall
    
    @property
    def f1_score(self) -> float:
        """
        Calculate F1 score for extractions.
        
        F1 = 2 * (precision * recall) / (precision + recall)
        
        Returns:
            F1 score (0-1)
        """
        if self._f1_score is None:
            if self.precision + self.recall == 0:
                self._f1_score = 0.0
            else:
                self._f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        return self._f1_score
    
    @property
    def overall_accuracy(self) -> float:
        """
        Calculate overall extraction accuracy.
        
        Returns:
            Accuracy score (0-1)
        """
        total = self.correct_extractions + self.incorrect_extractions + self.missed_extractions
        if total == 0:
            return 0.0
        return self.correct_extractions / total
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metrics to a dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "result_metrics": {
                "correct_extractions": self.correct_extractions,
                "incorrect_extractions": self.incorrect_extractions,
                "missed_extractions": self.missed_extractions,
                
                "precision": self.precision,
                "recall": self.recall,
                "f1_score": self.f1_score,
                "overall_accuracy": self.overall_accuracy
            },
            "quality_metrics": {
                "extraction_completeness": self.extraction_completeness,
                "extraction_accuracy": self.extraction_accuracy
            },
            "performance_metrics": {
                "execution_time": self.execution_time,
                "documents_processed": self.documents_processed,
                "fields_processed": self.fields_processed,
                "extraction_rate": self.fields_processed / self.execution_time if self.execution_time > 0 else 0
            }
        }


@dataclass
class StrategyMetrics:
    """Metrics for evaluating search/extraction strategies."""
    # Component metrics
    search_metrics: Optional[SearchMetrics] = None
    extraction_metrics: Optional[ExtractionMetrics] = None
    
    # Strategy-specific metrics
    strategy_accuracy: float = 0.0  # How accurately the strategy matches user intent
    generalizability: float = 0.0  # How well the strategy generalizes to new data
    efficiency: float = 0.0  # Computational efficiency (0-1 scale)
    
    # Additional metrics
    complexity: int = 0  # Number of operations in the strategy
    diversity: float = 0.0  # Diversity of approaches (0-1 scale)
    adaptability: float = 0.0  # How well the strategy adapts (0-1 scale)
    
    # Performance metrics
    execution_time: float = 0.0  # Seconds
    execution_success: bool = True
    errors_encountered: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metrics to a dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "search_metrics": self.search_metrics.to_dict() if self.search_metrics else None,
            "extraction_metrics": self.extraction_metrics.to_dict() if self.extraction_metrics else None,
            "strategy_metrics": {
                "strategy_accuracy": self.strategy_accuracy,
                "generalizability": self.generalizability,
                "efficiency": self.efficiency,
                "complexity": self.complexity,
                "diversity": self.diversity,
                "adaptability": self.adaptability
            },
            "performance_metrics": {
                "execution_time": self.execution_time,
                "execution_success": self.execution_success,
                "errors_encountered": self.errors_encountered
            }
        }


class MetricsCalculator:
    """
    Utility class for calculating and evaluating metrics.
    """
    
    @staticmethod
    def calculate_search_metrics(
        matches: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        execution_time: float,
        files_processed: int,
        lines_processed: int,
        memory_usage: Optional[float] = None
    ) -> SearchMetrics:
        """
        Calculate search metrics from search results and ground truth.
        
        Args:
            matches: List of search matches
            ground_truth: List of ground truth items
            execution_time: Execution time in seconds
            files_processed: Number of files processed
            lines_processed: Number of lines processed
            memory_usage: Memory usage in MB (optional)
            
        Returns:
            SearchMetrics object
        """
        # Create sets of match locations for comparison
        match_locations = set()
        for match in matches:
            file_path = match.get("file_path", "")
            line_number = match.get("line_number", 0)
            match_locations.add((file_path, line_number))
        
        ground_truth_locations = set()
        for item in ground_truth:
            file_path = item.get("file_path", "")
            line_number = item.get("line_number", 0)
            ground_truth_locations.add((file_path, line_number))
        
        # Calculate true positives, false positives, and false negatives
        true_positives = len(match_locations.intersection(ground_truth_locations))
        false_positives = len(match_locations - ground_truth_locations)
        false_negatives = len(ground_truth_locations - match_locations)
        
        # True negatives are harder to calculate without knowing all possible locations
        # For simplicity, we'll estimate using total lines
        true_negatives = max(0, lines_processed - (true_positives + false_positives + false_negatives))
        
        # Create metrics object
        metrics = SearchMetrics(
            true_positives=true_positives,
            false_positives=false_positives,
            true_negatives=true_negatives,
            false_negatives=false_negatives,
            execution_time=execution_time,
            files_processed=files_processed,
            lines_processed=lines_processed,
            memory_usage=memory_usage or 0.0
        )
        
        return metrics
    
    @staticmethod
    def calculate_extraction_metrics(
        extractions: Dict[str, Any],
        ground_truth: Dict[str, Any],
        execution_time: float,
        documents_processed: int
    ) -> ExtractionMetrics:
        """
        Calculate extraction metrics from extraction results and ground truth.
        
        Args:
            extractions: Dictionary of extracted items
            ground_truth: Dictionary of ground truth items
            execution_time: Execution time in seconds
            documents_processed: Number of documents processed
            
        Returns:
            ExtractionMetrics object
        """
        # Initialize counters
        correct_extractions = 0
        incorrect_extractions = 0
        missed_extractions = 0
        fields_processed = 0
        
        # Calculate field-specific metrics
        extraction_completeness = {}
        extraction_accuracy = {}
        
        # Get all field types
        all_fields = set(extractions.keys()) | set(ground_truth.keys())
        
        for field_name in all_fields:
            # Count fields processed
            fields_processed += 1
            
            # Get values for this field
            extracted_values = extractions.get(field_name, [])
            ground_truth_values = ground_truth.get(field_name, [])
            
            if not isinstance(extracted_values, list):
                extracted_values = [extracted_values]
            
            if not isinstance(ground_truth_values, list):
                ground_truth_values = [ground_truth_values]
            
            # Create sets for comparison
            extracted_set = set(str(v) for v in extracted_values)
            ground_truth_set = set(str(v) for v in ground_truth_values)
            
            # Calculate field-specific metrics
            field_correct = len(extracted_set.intersection(ground_truth_set))
            field_incorrect = len(extracted_set - ground_truth_set)
            field_missed = len(ground_truth_set - extracted_set)
            
            # Update counters
            correct_extractions += field_correct
            incorrect_extractions += field_incorrect
            missed_extractions += field_missed
            
            # Calculate field-specific rates
            if len(ground_truth_values) > 0:
                extraction_completeness[field_name] = field_correct / len(ground_truth_values)
            else:
                extraction_completeness[field_name] = 0.0
            
            if len(extracted_values) > 0:
                extraction_accuracy[field_name] = field_correct / len(extracted_values)
            else:
                extraction_accuracy[field_name] = 0.0
        
        # Create metrics object
        metrics = ExtractionMetrics(
            correct_extractions=correct_extractions,
            incorrect_extractions=incorrect_extractions,
            missed_extractions=missed_extractions,
            extraction_completeness=extraction_completeness,
            extraction_accuracy=extraction_accuracy,
            execution_time=execution_time,
            documents_processed=documents_processed,
            fields_processed=fields_processed
        )
        
        return metrics
    
    @staticmethod
    def calculate_strategy_metrics(
        strategy_results: Dict[str, Any],
        ground_truth: Dict[str, Any],
        execution_time: float,
        strategy_complexity: int
    ) -> StrategyMetrics:
        """
        Calculate strategy metrics from strategy execution results.
        
        Args:
            strategy_results: Results of strategy execution
            ground_truth: Ground truth data
            execution_time: Execution time in seconds
            strategy_complexity: Number of operations in the strategy
            
        Returns:
            StrategyMetrics object
        """
        # Extract component metrics if available
        search_metrics = None
        if "search_results" in strategy_results and "search_ground_truth" in ground_truth:
            search_metrics = MetricsCalculator.calculate_search_metrics(
                matches=strategy_results["search_results"],
                ground_truth=ground_truth["search_ground_truth"],
                execution_time=strategy_results.get("search_time", 0.0),
                files_processed=strategy_results.get("files_processed", 0),
                lines_processed=strategy_results.get("lines_processed", 0)
            )
        
        extraction_metrics = None
        if "extractions" in strategy_results and "extraction_ground_truth" in ground_truth:
            extraction_metrics = MetricsCalculator.calculate_extraction_metrics(
                extractions=strategy_results["extractions"],
                ground_truth=ground_truth["extraction_ground_truth"],
                execution_time=strategy_results.get("extraction_time", 0.0),
                documents_processed=strategy_results.get("documents_processed", 0)
            )
        
        # Calculate strategy-specific metrics
        # For demonstration, we'll use simplified calculations
        strategy_accuracy = 0.0
        if search_metrics:
            strategy_accuracy += 0.5 * search_metrics.f1_score
        if extraction_metrics:
            strategy_accuracy += 0.5 * extraction_metrics.f1_score
        
        # Generalizability would typically be calculated using cross-validation
        # This is a placeholder value
        generalizability = 0.8 * strategy_accuracy
        
        # Efficiency based on execution time and complexity
        base_efficiency = 1.0 / (1.0 + (execution_time / (strategy_complexity * 0.1)))
        efficiency = min(1.0, max(0.0, base_efficiency))
        
        # Create metrics object
        metrics = StrategyMetrics(
            search_metrics=search_metrics,
            extraction_metrics=extraction_metrics,
            strategy_accuracy=strategy_accuracy,
            generalizability=generalizability,
            efficiency=efficiency,
            complexity=strategy_complexity,
            diversity=0.5,  # Placeholder
            adaptability=0.7,  # Placeholder
            execution_time=execution_time,
            execution_success=True,
            errors_encountered=strategy_results.get("errors", 0)
        )
        
        return metrics
    
    @staticmethod
    def compare_strategies(
        metrics_a: StrategyMetrics,
        metrics_b: StrategyMetrics
    ) -> Dict[str, Any]:
        """
        Compare two strategies based on their metrics.
        
        Args:
            metrics_a: Metrics for strategy A
            metrics_b: Metrics for strategy B
            
        Returns:
            Dictionary with comparison results
        """
        # Calculate the weighted score for each strategy
        weights = {
            "strategy_accuracy": 0.4,
            "generalizability": 0.2,
            "efficiency": 0.2,
            "diversity": 0.1,
            "adaptability": 0.1
        }
        
        score_a = (
            weights["strategy_accuracy"] * metrics_a.strategy_accuracy +
            weights["generalizability"] * metrics_a.generalizability +
            weights["efficiency"] * metrics_a.efficiency +
            weights["diversity"] * metrics_a.diversity +
            weights["adaptability"] * metrics_a.adaptability
        )
        
        score_b = (
            weights["strategy_accuracy"] * metrics_b.strategy_accuracy +
            weights["generalizability"] * metrics_b.generalizability +
            weights["efficiency"] * metrics_b.efficiency +
            weights["diversity"] * metrics_b.diversity +
            weights["adaptability"] * metrics_b.adaptability
        )
        
        # Determine the winner
        if score_a > score_b:
            winner = "A"
            margin = score_a - score_b
        elif score_b > score_a:
            winner = "B"
            margin = score_b - score_a
        else:
            winner = "Tie"
            margin = 0.0
        
        # Create comparison report
        comparison = {
            "scores": {
                "strategy_a": score_a,
                "strategy_b": score_b
            },
            "winner": winner,
            "margin": margin,
            "breakdown": {
                "strategy_accuracy": {
                    "strategy_a": metrics_a.strategy_accuracy,
                    "strategy_b": metrics_b.strategy_accuracy,
                    "difference": metrics_a.strategy_accuracy - metrics_b.strategy_accuracy
                },
                "generalizability": {
                    "strategy_a": metrics_a.generalizability,
                    "strategy_b": metrics_b.generalizability,
                    "difference": metrics_a.generalizability - metrics_b.generalizability
                },
                "efficiency": {
                    "strategy_a": metrics_a.efficiency,
                    "strategy_b": metrics_b.efficiency,
                    "difference": metrics_a.efficiency - metrics_b.efficiency
                },
                "execution_time": {
                    "strategy_a": metrics_a.execution_time,
                    "strategy_b": metrics_b.execution_time,
                    "difference": metrics_a.execution_time - metrics_b.execution_time
                }
            }
        }
        
        return comparison


def evaluate_search_results(
    matches: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    execution_time: float,
    files_processed: int,
    lines_processed: int
) -> Dict[str, Any]:
    """
    Evaluate search results against ground truth.
    
    Args:
        matches: List of search matches
        ground_truth: List of ground truth items
        execution_time: Execution time in seconds
        files_processed: Number of files processed
        lines_processed: Number of lines processed
        
    Returns:
        Dictionary with evaluation metrics
    """
    metrics = MetricsCalculator.calculate_search_metrics(
        matches=matches,
        ground_truth=ground_truth,
        execution_time=execution_time,
        files_processed=files_processed,
        lines_processed=lines_processed
    )
    
    return metrics.to_dict()


def evaluate_extraction_results(
    extractions: Dict[str, Any],
    ground_truth: Dict[str, Any],
    execution_time: float,
    documents_processed: int
) -> Dict[str, Any]:
    """
    Evaluate extraction results against ground truth.
    
    Args:
        extractions: Dictionary of extracted items
        ground_truth: Dictionary of ground truth items
        execution_time: Execution time in seconds
        documents_processed: Number of documents processed
        
    Returns:
        Dictionary with evaluation metrics
    """
    metrics = MetricsCalculator.calculate_extraction_metrics(
        extractions=extractions,
        ground_truth=ground_truth,
        execution_time=execution_time,
        documents_processed=documents_processed
    )
    
    return metrics.to_dict()


def evaluate_strategy(
    strategy_results: Dict[str, Any],
    ground_truth: Dict[str, Any],
    execution_time: float,
    strategy_complexity: int
) -> Dict[str, Any]:
    """
    Evaluate a strategy's results against ground truth.
    
    Args:
        strategy_results: Results of strategy execution
        ground_truth: Ground truth data
        execution_time: Execution time in seconds
        strategy_complexity: Number of operations in the strategy
        
    Returns:
        Dictionary with evaluation metrics
    """
    metrics = MetricsCalculator.calculate_strategy_metrics(
        strategy_results=strategy_results,
        ground_truth=ground_truth,
        execution_time=execution_time,
        strategy_complexity=strategy_complexity
    )
    
    return metrics.to_dict()