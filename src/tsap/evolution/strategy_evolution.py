"""
Advanced evolutionary algorithms for search and analysis strategies.

This module provides sophisticated functionality for evolving, optimizing, and adapting
search and analysis strategies using advanced genetic algorithms, coevolution,
multi-objective optimization, and adaptive parameter control techniques.
"""

import asyncio
import re
import time
import random
import uuid
import math
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum, auto

from tsap.utils.logging import logger
from tsap.utils.errors import TSAPError
from tsap.evolution.base import (
    EvolutionAlgorithm, 
    EvolutionConfig, 
    Individual,
    Population,
    register_algorithm
)

# Conditionally import these libraries if available
# These would be used in a real implementation for more advanced features
try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False
    
try:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    sklearn_available = True
except ImportError:
    sklearn_available = False


class StrategyEvolutionError(TSAPError):
    """
    Exception for errors in strategy evolution.
    
    Attributes:
        message: Error message
        strategy_type: Type of strategy that caused the error
        details: Additional error details
    """
    def __init__(
        self, 
        message: str, 
        strategy_type: Optional[str] = None, 
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            message, 
            code=f"STRATEGY_EVOLUTION_{strategy_type.upper()}_ERROR" if strategy_type else "STRATEGY_EVOLUTION_ERROR",
            details=details
        )
        self.strategy_type = strategy_type


class PatternType(Enum):
    """Enumeration of pattern types for more expressive strategy genomes."""
    LITERAL = auto()  # Exact text match
    REGEX = auto()    # Regular expression
    SEMANTIC = auto() # Semantic/meaning-based match
    FUZZY = auto()    # Fuzzy/approximate match
    NEGATION = auto() # Pattern that should NOT match
    COMBINED = auto() # Combination of other patterns (AND/OR)


class PatternOperator(Enum):
    """Logical operators for combining patterns."""
    AND = auto()
    OR = auto()
    NOT = auto()
    NEAR = auto()     # Patterns near each other
    FOLLOWED_BY = auto() # Sequential patterns
    WITHIN = auto()   # One pattern within N tokens of another


@dataclass
class PatternNode:
    """
    Node in a pattern tree representing a complex search pattern.
    
    For leaf nodes, pattern_text contains the actual pattern.
    For non-leaf nodes, children contains sub-patterns combined with operator.
    """
    pattern_type: PatternType
    pattern_text: Optional[str] = None
    operator: Optional[PatternOperator] = None
    children: List['PatternNode'] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "pattern_type": self.pattern_type.name,
        }
        
        if self.pattern_text:
            result["pattern_text"] = self.pattern_text
            
        if self.operator:
            result["operator"] = self.operator.name
            
        if self.children:
            result["children"] = [child.to_dict() for child in self.children]
            
        if self.parameters:
            result["parameters"] = self.parameters
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatternNode':
        """Create from dictionary representation."""
        children = []
        if "children" in data:
            children = [PatternNode.from_dict(child) for child in data["children"]]
            
        return cls(
            pattern_type=PatternType[data["pattern_type"]],
            pattern_text=data.get("pattern_text"),
            operator=PatternOperator[data["operator"]] if "operator" in data else None,
            children=children,
            parameters=data.get("parameters", {})
        )
    
    def complexity(self) -> int:
        """Calculate pattern complexity for fitness penalties and mutation probability."""
        if not self.children:
            return 1
        return 1 + sum(child.complexity() for child in self.children)
    
    def depth(self) -> int:
        """Calculate tree depth."""
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)
    
    def get_all_text_patterns(self) -> List[str]:
        """Extract all text patterns from the tree."""
        if self.pattern_text:
            return [self.pattern_text]
        
        result = []
        for child in self.children:
            result.extend(child.get_all_text_patterns())
        return result
    
    def to_ripgrep_pattern(self) -> Dict[str, Any]:
        """Convert to ripgrep pattern format."""
        if self.pattern_type == PatternType.LITERAL:
            return {
                "pattern": re.escape(self.pattern_text or ""),
                "description": f"Literal: {self.pattern_text}",
                "is_regex": True,
                "case_sensitive": self.parameters.get("case_sensitive", False)
            }
        elif self.pattern_type == PatternType.REGEX:
            return {
                "pattern": self.pattern_text or "",
                "description": f"Regex: {self.pattern_text}",
                "is_regex": True,
                "case_sensitive": self.parameters.get("case_sensitive", False)
            }
        elif self.pattern_type == PatternType.FUZZY:
            # Convert fuzzy pattern to regex with character variations
            if not self.pattern_text:
                return {"pattern": "", "description": "Empty fuzzy pattern", "is_regex": True}
            
            # Create a regex that allows for character insertions, deletions, and substitutions
            fuzzy_level = self.parameters.get("fuzzy_level", 1)
            parts = []
            for char in self.pattern_text:
                if char.isalnum():
                    # Allow for substitutions (character class)
                    if char.isalpha():
                        parts.append(f"[{char.lower()}{char.upper()}]{{0,1}}")
                    else:
                        parts.append(f"{char}{{0,1}}")
                else:
                    parts.append(f"{re.escape(char)}{{0,1}}")
            
            # Allow for insertions with .{0,N} where N is fuzzy_level
            if fuzzy_level > 0:
                pattern = f".*?({'.*?'.join(parts)}).*?"
            else:
                pattern = ''.join(parts)
                
            return {
                "pattern": pattern,
                "description": f"Fuzzy({fuzzy_level}): {self.pattern_text}",
                "is_regex": True,
                "case_sensitive": self.parameters.get("case_sensitive", False)
            }
        elif self.pattern_type == PatternType.NEGATION:
            if self.pattern_text:
                return {
                    "pattern": f"^((?!{self.pattern_text}).)*$",
                    "description": f"NOT: {self.pattern_text}",
                    "is_regex": True,
                    "case_sensitive": self.parameters.get("case_sensitive", False)
                }
            elif self.children and self.children[0].pattern_text:
                return {
                    "pattern": f"^((?!{self.children[0].pattern_text}).)*$",
                    "description": f"NOT: {self.children[0].pattern_text}",
                    "is_regex": True,
                    "case_sensitive": self.parameters.get("case_sensitive", False)
                }
            else:
                return {"pattern": "", "description": "Empty negation pattern", "is_regex": True}
        elif self.pattern_type == PatternType.COMBINED:
            if not self.children:
                return {"pattern": "", "description": "Empty combined pattern", "is_regex": True}
            
            child_patterns = [child.to_ripgrep_pattern()["pattern"] for child in self.children if child.to_ripgrep_pattern()["pattern"]]
            
            if not child_patterns:
                return {"pattern": "", "description": "Empty combined pattern", "is_regex": True}
            
            if self.operator == PatternOperator.AND:
                # AND: Positive lookahead for each pattern
                pattern = "".join(f"(?=.*{p})" for p in child_patterns) + ".*"
                description = f"AND: {' AND '.join(p for p in child_patterns)}"
            elif self.operator == PatternOperator.OR:
                # OR: Alternative patterns
                pattern = f"({('|'.join(child_patterns))})"
                description = f"OR: {' OR '.join(p for p in child_patterns)}"
            elif self.operator == PatternOperator.NEAR:
                # NEAR: Patterns within N tokens
                distance = self.parameters.get("distance", 10)
                # This is a simplification - true "near" requires more complex regex or post-processing
                pattern = f"({('|'.join(child_patterns))})"
                description = f"NEAR({distance}): {', '.join(p for p in child_patterns)}"
            elif self.operator == PatternOperator.FOLLOWED_BY:
                # FOLLOWED_BY: Sequential patterns with optional content between
                max_gap = self.parameters.get("max_gap", 50)  # noqa: F841
                pattern = "".join(f"{p}.*?" for p in child_patterns[:-1]) + child_patterns[-1]
                description = f"SEQUENCE: {' -> '.join(p for p in child_patterns)}"
            else:
                # Default to OR
                pattern = f"({('|'.join(child_patterns))})"
                description = f"Pattern group: {', '.join(p for p in child_patterns)}"
            
            return {
                "pattern": pattern,
                "description": description,
                "is_regex": True,
                "case_sensitive": self.parameters.get("case_sensitive", False)
            }
        else:
            # Fallback
            return {
                "pattern": self.pattern_text or "",
                "description": f"Pattern: {self.pattern_text}",
                "is_regex": True,
                "case_sensitive": self.parameters.get("case_sensitive", False)
            }


@dataclass
class StrategyGenome:
    """
    Enhanced genome representation for a search or analysis strategy.
    
    Attributes:
        patterns: List of complex pattern trees
        options: Additional options for the strategy
        meta: Metadata about the strategy
        weights: Importance weights for different patterns
        constraints: Constraints on matches (e.g., must occur in same file)
        filters: Post-processing filters
        adaptive_params: Parameters that adapt during evolution
    """
    patterns: List[PatternNode]
    options: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    weights: Dict[int, float] = field(default_factory=dict)  # Pattern index -> weight
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    filters: List[Dict[str, Any]] = field(default_factory=list)
    adaptive_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default adaptive parameters if not present."""
        if not self.adaptive_params:
            self.adaptive_params = {
                "mutation_rate": random.uniform(0.1, 0.3),
                "crossover_points": random.randint(1, 3),
                "pattern_complexity_penalty": random.uniform(0.01, 0.1),
                "learning_rate": random.uniform(0.01, 0.1)
            }
    
    def complexity(self) -> float:
        """Calculate overall genome complexity for fitness penalties."""
        pattern_complexity = sum(p.complexity() for p in self.patterns)
        option_complexity = len(self.options) * 0.5
        constraint_complexity = len(self.constraints) * 0.7
        filter_complexity = len(self.filters) * 0.6
        
        return pattern_complexity + option_complexity + constraint_complexity + filter_complexity
    
    def adapt(self, fitness_delta: float, gen_fitness_stats: Dict[str, float]) -> None:
        """Adapt parameters based on fitness trends."""
        learning_rate = self.adaptive_params.get("learning_rate", 0.05)
        
        # If fitness improved, slightly decrease mutation rate
        if fitness_delta > 0:
            self.adaptive_params["mutation_rate"] = max(
                0.01, 
                self.adaptive_params.get("mutation_rate", 0.2) * (1 - learning_rate * 0.5)
            )
        # If fitness worsened or plateaued, increase mutation rate
        else:
            self.adaptive_params["mutation_rate"] = min(
                0.5, 
                self.adaptive_params.get("mutation_rate", 0.2) * (1 + learning_rate)
            )
        
        # Adjust pattern complexity penalty based on population stats
        mean_fitness = gen_fitness_stats.get("mean_fitness", 0.5)
        if mean_fitness > 0.7:  # If population is doing well
            # Increase penalty to favor simpler patterns
            self.adaptive_params["pattern_complexity_penalty"] = min(
                0.2,
                self.adaptive_params.get("pattern_complexity_penalty", 0.05) * (1 + learning_rate * 0.2)
            )
        else:  # If population is struggling
            # Decrease penalty to allow more complex patterns
            self.adaptive_params["pattern_complexity_penalty"] = max(
                0.01,
                self.adaptive_params.get("pattern_complexity_penalty", 0.05) * (1 - learning_rate * 0.2)
            )


class FitnessComponent(NamedTuple):
    """Named tuple for multi-objective fitness components."""
    name: str
    value: float
    weight: float = 1.0
    
    @property
    def weighted_value(self) -> float:
        """Get weighted fitness value."""
        return self.value * self.weight


@dataclass
class MultiFitness:
    """
    Multi-objective fitness representation.
    
    Supports both scalar fitness (for backward compatibility) and
    multi-objective optimization with weighted components.
    """
    components: List[FitnessComponent] = field(default_factory=list)
    scalar_value: Optional[float] = None
    
    def __post_init__(self):
        # Calculate scalar value if not provided
        if self.scalar_value is None and self.components:
            self.scalar_value = sum(c.weighted_value for c in self.components) / sum(c.weight for c in self.components)
    
    @classmethod
    def from_scalar(cls, value: float) -> 'MultiFitness':
        """Create from a scalar fitness value."""
        return cls(scalar_value=value, components=[FitnessComponent("scalar", value)])
    
    def dominates(self, other: 'MultiFitness') -> bool:
        """
        Check if this fitness dominates another in Pareto sense.
        
        In multi-objective optimization, solution A dominates B if
        A is at least as good as B in all objectives and better in at least one.
        """
        if not self.components or not other.components:
            # Fall back to scalar comparison
            return (self.scalar_value or 0) > (other.scalar_value or 0)
        
        # Map component names for comparison
        self_components = {c.name: c.value for c in self.components}
        other_components = {c.name: c.value for c in other.components}
        
        # Combined set of component names
        all_names = set(self_components.keys()) | set(other_components.keys())
        
        # Check domination criteria
        at_least_one_better = False
        for name in all_names:
            self_value = self_components.get(name, 0.0)
            other_value = other_components.get(name, 0.0)
            
            if self_value < other_value:
                return False  # Not dominating if worse in any component
            if self_value > other_value:
                at_least_one_better = True
                
        return at_least_one_better
    
    def distance(self, other: 'MultiFitness') -> float:
        """Calculate Euclidean distance between fitness vectors."""
        # Use scalar values if components not available
        if not self.components or not other.components:
            return abs((self.scalar_value or 0) - (other.scalar_value or 0))
        
        # Map component names for comparison
        self_components = {c.name: c.value for c in self.components}
        other_components = {c.name: c.value for c in other.components}
        
        # Combined set of component names
        all_names = set(self_components.keys()) | set(other_components.keys())
        
        # Calculate Euclidean distance in normalized space
        sum_squared_diff = 0.0
        for name in all_names:
            self_value = self_components.get(name, 0.0)
            other_value = other_components.get(name, 0.0)
            sum_squared_diff += (self_value - other_value) ** 2
            
        return math.sqrt(sum_squared_diff)


@dataclass
class StrategyIsland:
    """
    Island model for maintaining diverse sub-populations.
    
    Each island evolves independently with migration between islands.
    """
    id: str
    population: Population = field(default_factory=Population)
    focus: Dict[str, Any] = field(default_factory=dict)  # Specialization parameters
    
    def get_emigrants(self, count: int) -> List[Individual]:
        """Select individuals for migration to other islands."""
        if not self.population.individuals:
            return []
        
        # Sort by fitness
        sorted_individuals = sorted(
            self.population.individuals,
            key=lambda i: i.fitness.scalar_value if i.fitness else 0,
            reverse=True
        )
        
        # Select emigrants - mixture of best and diverse
        best_count = max(1, count // 2)
        diverse_count = count - best_count
        
        # Best individuals
        emigrants = sorted_individuals[:best_count]
        
        # Add diverse individuals
        if diverse_count > 0 and len(sorted_individuals) > best_count:
            candidates = sorted_individuals[best_count:]
            # Select based on genotypic diversity
            diverse_emigrants = []
            
            # Simple greedy selection for diversity
            for _ in range(diverse_count):
                if not candidates:
                    break
                    
                # Start with a random individual
                if not diverse_emigrants:
                    idx = random.randrange(len(candidates))
                    diverse_emigrants.append(candidates.pop(idx))
                    continue
                
                # Find most different from current emigrants
                most_diverse_idx = 0
                max_diversity = -1
                
                for i, candidate in enumerate(candidates):
                    # Calculate diversity as minimum genome difference from current emigrants
                    min_diff = min(genome_difference(candidate.genome, emigrant.genome) 
                                for emigrant in diverse_emigrants + emigrants)
                    
                    if min_diff > max_diversity:
                        max_diversity = min_diff
                        most_diverse_idx = i
                
                diverse_emigrants.append(candidates.pop(most_diverse_idx))
            
            emigrants.extend(diverse_emigrants)
        
        return emigrants
    
    def receive_immigrants(self, immigrants: List[Individual]) -> None:
        """Integrate immigrants into the island population."""
        for immigrant in immigrants:
            # Clone to avoid affecting original
            clone = Individual(
                genome=immigrant.genome,
                generation=immigrant.generation,
                parent_ids=immigrant.parent_ids,
                id=str(uuid.uuid4())
            )
            
            # Mark as immigrant in metadata
            clone.metadata["immigrant"] = True
            clone.metadata["source_island"] = immigrant.metadata.get("island_id")
            
            # Add to population
            self.population.add_individual(clone)


def genome_difference(genome1: StrategyGenome, genome2: StrategyGenome) -> float:
    """
    Calculate difference between two genomes.
    
    Uses pattern tree similarity, option differences, and other genome attributes.
    Returns a value between 0 (identical) and 1 (completely different).
    """
    # Pattern difference
    pattern_diff = 0.0
    if genome1.patterns and genome2.patterns:
        # Compare pattern sets using Jaccard distance
        patterns1 = set()
        patterns2 = set()
        
        for p in genome1.patterns:
            patterns1.update(p.get_all_text_patterns())
            
        for p in genome2.patterns:
            patterns2.update(p.get_all_text_patterns())
            
        if patterns1 or patterns2:
            # Jaccard distance: 1 - |A∩B|/|A∪B|
            intersection = len(patterns1.intersection(patterns2))
            union = len(patterns1.union(patterns2))
            pattern_diff = 1.0 - (intersection / union if union > 0 else 0.0)
        else:
            pattern_diff = 0.0
    else:
        pattern_diff = 1.0 if (genome1.patterns and not genome2.patterns) or (not genome1.patterns and genome2.patterns) else 0.0
    
    # Option difference
    option_diff = 0.0
    all_options = set(genome1.options.keys()) | set(genome2.options.keys())
    if all_options:
        option_diffs = []
        for option in all_options:
            if option in genome1.options and option in genome2.options:
                # Compare option values
                value1 = genome1.options[option]
                value2 = genome2.options[option]
                
                if isinstance(value1, bool) and isinstance(value2, bool):
                    option_diffs.append(0.0 if value1 == value2 else 1.0)
                elif isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                    # Normalize numeric difference
                    max_val = max(abs(value1), abs(value2))
                    if max_val > 0:
                        option_diffs.append(min(1.0, abs(value1 - value2) / max_val))
                    else:
                        option_diffs.append(0.0)
                else:
                    option_diffs.append(0.0 if value1 == value2 else 1.0)
            else:
                option_diffs.append(1.0)  # Missing option in one genome
                
        option_diff = sum(option_diffs) / len(all_options) if option_diffs else 0.0
    
    # Constraint difference
    constraint_diff = 0.0
    if genome1.constraints or genome2.constraints:
        len1 = len(genome1.constraints)
        len2 = len(genome2.constraints)
        constraint_diff = abs(len1 - len2) / max(len1, len2, 1)
    
    # Combine differences with weights
    return 0.6 * pattern_diff + 0.3 * option_diff + 0.1 * constraint_diff


@register_algorithm("strategy_evolution")
class StrategyEvolutionAlgorithm(EvolutionAlgorithm[StrategyGenome, MultiFitness]):
    """
    Advanced evolutionary algorithm for optimizing search and analysis strategies.
    
    Features:
    - Multi-objective fitness evaluation
    - Island model for diversity preservation
    - Adaptive parameter control
    - Sophisticated pattern tree representation
    - Advanced genetic operators
    - Fitness shaping and scaling
    - Elitism with diversity preservation
    """
    def __init__(
        self, 
        name: str = "strategy_evolution", 
        config: Optional[EvolutionConfig] = None
    ) -> None:
        """
        Initialize the strategy evolution algorithm.
        
        Args:
            name: Name of the algorithm
            config: Configuration for the algorithm
        """
        super().__init__(name, config)
        
        # Target documents and training queries for evaluation
        self.target_documents: List[str] = []
        self.training_queries: List[Dict[str, Any]] = []
        
        # Additional parameters
        self.strategy_type = "search"  # "search" or "analysis"
        self.objective = "general"  # General objective of the strategy
        
        # Pattern generation parameters
        self.max_patterns = 8
        self.min_patterns = 1
        self.max_pattern_depth = 4
        
        # Island model parameters
        self.num_islands = 4
        self.migration_interval = 5  # Generations between migrations
        self.migration_rate = 0.1    # Fraction of population that migrates
        self.islands: List[StrategyIsland] = []
        
        # Multi-objective parameters
        self.fitness_weights = {
            "precision": 0.4,
            "recall": 0.4,
            "complexity": 0.1,
            "runtime": 0.1
        }
        
        # Advanced parameters
        self.elitism_rate = 0.1
        self.diversity_preservation = True
        self.fitness_sharing = True
        self.fitness_sharing_radius = 0.2
        self.novelty_search = False
        self.archive_size = 10
        
        # Adaptive parameters
        self.adaptive_selection_pressure = True
        self.adaptive_operator_rates = True
        self.adaptive_population_size = False
        self.early_stopping = True
        self.restart_stagnation = True
        self.stagnation_tolerance = 5  # Generations without improvement
        
        # Pattern complexity control
        self.complexity_penalty_factor = 0.05
        self.depth_penalty_factor = 0.03
        
        # Training data clustering
        self.query_clusters = []
        
        # Evaluation cache
        self._evaluation_cache: Dict[str, MultiFitness] = {}
        
        # Novelty archive
        self._novelty_archive: List[Individual[StrategyGenome, MultiFitness]] = []
        
        # Pattern bank - store effective patterns for reuse
        self._pattern_bank: List[Tuple[PatternNode, float]] = []
        
        # Performance tracking
        self._evaluation_times: List[float] = []
        self._crossover_times: List[float] = []
        self._mutation_times: List[float] = []
        
        # Runtime statistics for adaptive control
        self.generation_stats: List[Dict[str, Any]] = []
        
        # Initialize additional components if advanced libraries are available
        if nltk_available:
            try:
                # Initialize NLTK for NLP-enhanced pattern generation
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                self.use_nlp = True
            except Exception:
                self.use_nlp = False
        else:
            self.use_nlp = False
    
    def set_target_documents(self, documents: List[str]) -> None:
        """
        Set the target documents for strategy evaluation.
        
        Args:
            documents: List of document paths
        """
        self.target_documents = documents
    
    def set_training_queries(self, queries: List[Dict[str, Any]]) -> None:
        """
        Set the training queries for strategy evaluation.
        
        Args:
            queries: List of training queries with expected results
        """
        self.training_queries = queries
        
        # Cluster training queries for specialized islands
        self._cluster_training_queries()
    
    def _cluster_training_queries(self) -> None:
        """
        Cluster training queries for specialized islands.
        
        Each island will focus on a different cluster of queries.
        Uses advanced clustering techniques if sklearn is available,
        otherwise falls back to simpler methods.
        """
        if not self.training_queries or len(self.training_queries) < self.num_islands:
            # Not enough queries to cluster
            self.query_clusters = [self.training_queries]
            return
            
        # Advanced clustering if sklearn is available
        if sklearn_available:
            # Extract query texts and expected matches
            query_texts = []
            for query in self.training_queries:
                query_text = query.get("query", "")
                expected_matches = " ".join(query.get("expected_matches", []))
                combined_text = f"{query_text} {expected_matches}"
                query_texts.append(combined_text)
            
            # Vectorize the queries using TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=1000, 
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
            
            try:
                # Transform queries to TF-IDF vectors
                query_vectors = vectorizer.fit_transform(query_texts)
                
                # Apply K-means clustering
                kmeans = KMeans(
                    n_clusters=self.num_islands,
                    random_state=42,
                    n_init=10
                )
                
                cluster_labels = kmeans.fit_predict(query_vectors)
                
                # Create clusters
                clusters = [[] for _ in range(self.num_islands)]
                for i, label in enumerate(cluster_labels):
                    clusters[label].append(self.training_queries[i])
                
                # Handle empty clusters
                for i, cluster in enumerate(clusters):
                    if not cluster:
                        # Assign random queries to empty clusters
                        random_queries = random.sample(self.training_queries, 
                                                     k=min(3, len(self.training_queries)))
                        clusters[i].extend(random_queries)
                
                self.query_clusters = clusters
                
                # Analyze cluster characteristics for specialization
                for i, cluster in enumerate(clusters):
                    cluster_texts = []
                    num_expected_matches = []
                    
                    for query in cluster:
                        query_text = query.get("query", "")
                        expected_matches = query.get("expected_matches", [])
                        
                        cluster_texts.append(query_text)
                        num_expected_matches.append(len(expected_matches))
                    
                    # Determine cluster characteristics
                    avg_matches = sum(num_expected_matches) / max(len(num_expected_matches), 1)
                    
                    # Set specialization based on characteristics
                    if avg_matches > 10:
                        # Many expected matches - optimize for recall
                        self.islands[i].focus["specialization"] = "recall"
                    elif avg_matches < 3:
                        # Few expected matches - optimize for precision
                        self.islands[i].focus["specialization"] = "precision"
                    else:
                        # Balanced - choose a random specialization
                        self.islands[i].focus["specialization"] = random.choice([
                            "precision", "recall", "complexity", "runtime"
                        ])
                
                return
                
            except Exception as e:
                logger.warning(f"Error in K-means clustering: {str(e)}")
                # Fall back to simpler method
        
        # Simple clustering based on query terms if sklearn not available
        # or if the advanced clustering failed
        
        # Extract terms from queries
        query_terms = []
        for query in self.training_queries:
            query_text = query.get("query", "")
            # Simple tokenization
            terms = [t.lower() for t in re.findall(r'\b\w+\b', query_text)]
            query_terms.append(terms)
        
        # Use simple term-based similarity for clustering
        clusters = [[] for _ in range(self.num_islands)]
        assigned = [False] * len(self.training_queries)
        
        # Start with random seeds for each cluster
        seed_indices = random.sample(range(len(self.training_queries)), 
                                    k=min(self.num_islands, len(self.training_queries)))
        
        for i, seed_idx in enumerate(seed_indices):
            clusters[i].append(self.training_queries[seed_idx])
            assigned[seed_idx] = True
        
        # Assign remaining queries to closest cluster
        for i, terms in enumerate(query_terms):
            if assigned[i]:
                continue
                
            # Find closest cluster
            max_similarity = -1
            best_cluster = 0
            
            for c, cluster in enumerate(clusters):
                if not cluster:
                    # Empty cluster
                    best_cluster = c
                    break
                    
                # Calculate similarity to this cluster
                similarity = 0
                for query in cluster:
                    query_text = query.get("query", "")
                    cluster_terms = [t.lower() for t in re.findall(r'\b\w+\b', query_text)]
                    
                    # Jaccard similarity
                    intersection = set(terms) & set(cluster_terms)
                    union = set(terms) | set(cluster_terms)
                    
                    similarity += len(intersection) / max(len(union), 1)
                
                avg_similarity = similarity / len(cluster)
                
                if avg_similarity > max_similarity:
                    max_similarity = avg_similarity
                    best_cluster = c
            
            # Assign to best cluster
            clusters[best_cluster].append(self.training_queries[i])
            assigned[i] = True
                
        # Handle empty clusters
        for i, cluster in enumerate(clusters):
            if not cluster:
                # Assign random queries to empty clusters
                unassigned = [j for j, a in enumerate(assigned) if not a]
                
                if unassigned:
                    # Use unassigned queries if available
                    for j in unassigned:
                        clusters[i].append(self.training_queries[j])
                        assigned[j] = True
                        
                        if clusters[i]:
                            break
                
                if not clusters[i]:
                    # No unassigned queries, use random ones
                    random_queries = random.sample(self.training_queries, 
                                                 k=min(3, len(self.training_queries)))
                    clusters[i].extend(random_queries)
        
        self.query_clusters = clusters
        
        # Assign specializations
        specializations = ["precision", "recall", "complexity", "runtime"]
        for i in range(min(len(self.islands), len(specializations))):
            self.islands[i].focus["specialization"] = specializations[i]
    
    def set_strategy_type(self, strategy_type: str) -> None:
        """
        Set the type of strategy to evolve.
        
        Args:
            strategy_type: Type of strategy ("search" or "analysis")
        """
        self.strategy_type = strategy_type
    
    def set_objective(self, objective: str) -> None:
        """
        Set the objective of the strategy.
        
        Args:
            objective: Objective of the strategy
        """
        self.objective = objective
    
    def configure_islands(self, num_islands: int = 4, migration_interval: int = 5, migration_rate: float = 0.1) -> None:
        """
        Configure island model parameters.
        
        Args:
            num_islands: Number of islands (sub-populations)
            migration_interval: Generations between migrations
            migration_rate: Fraction of population that migrates
        """
        self.num_islands = num_islands
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
    
    async def initialize_population(self) -> Population[StrategyGenome, MultiFitness]:
        """
        Initialize the initial population of strategies using the island model.
        
        Returns:
            Initial population
        """
        # Create islands
        self.islands = []
        for i in range(self.num_islands):
            island_id = f"island_{i}"
            island = StrategyIsland(id=island_id)
            
            # Set island focus based on query clusters
            if self.query_clusters and i < len(self.query_clusters):
                island.focus = {
                    "queries": self.query_clusters[i],
                    "specialization": random.choice(["precision", "recall", "complexity", "runtime"])
                }
            else:
                island.focus = {
                    "queries": self.training_queries,
                    "specialization": random.choice(["precision", "recall", "complexity", "runtime"])
                }
                
            self.islands.append(island)
        
        # Initialize island populations
        island_size = max(4, self.config.population_size // self.num_islands)
        
        initialization_tasks = []
        for island in self.islands:
            # Generate initial strategies for this island
            task = asyncio.create_task(self._initialize_island_population(island, island_size))
            initialization_tasks.append(task)
            
        await asyncio.gather(*initialization_tasks)
        
        # Create main population from all islands
        main_population = Population[StrategyGenome, MultiFitness]()
        for island in self.islands:
            for individual in island.population.individuals:
                main_population.add_individual(individual)
                
        return main_population
    
    async def _initialize_island_population(self, island: StrategyIsland, size: int) -> None:
        """
        Initialize the population for a specific island.
        
        Args:
            island: Island to initialize
            size: Population size for this island
        """
        # Create island population
        population = Population[StrategyGenome, MultiFitness]()
        
        # Determine island specialization
        specialization = island.focus.get("specialization", "general")
        queries = island.focus.get("queries", self.training_queries)  # noqa: F841
        
        # Generate initial strategies
        for i in range(size):
            # Generate a random strategy genome with island specialization
            genome = await self._generate_random_genome(specialization=specialization)
            
            # Create an individual
            individual = Individual[StrategyGenome, MultiFitness](
                genome=genome,
                generation=0
            )
            
            # Tag with island metadata
            individual.metadata["island_id"] = island.id
            individual.metadata["specialization"] = specialization
            
            # Add the individual to the population
            population.add_individual(individual)
            
        # Set island population
        island.population = population
    
    async def evaluate_fitness(self, individual: Individual[StrategyGenome, MultiFitness]) -> MultiFitness:
        """
        Evaluate the fitness of a strategy with multiple objectives.
        
        Args:
            individual: Individual to evaluate
            
        Returns:
            Multi-objective fitness
        """
        # Check cache first
        genome_hash = self._hash_genome(individual.genome)
        if genome_hash in self._evaluation_cache:
            return self._evaluation_cache[genome_hash]
        
        # Convert genome to strategy configuration
        strategy_config = self._genome_to_strategy(individual.genome)
        
        # Track evaluation start time
        start_time = time.time()
        
        # Get island specialization
        specialization = individual.metadata.get("specialization", "general")
        
        # Get target queries based on island
        island_id = individual.metadata.get("island_id")
        target_queries = None
        
        if island_id:
            for island in self.islands:
                if island.id == island_id:
                    target_queries = island.focus.get("queries")
                    break
        
        if not target_queries:
            target_queries = self.training_queries
        
        # Evaluate strategy against training queries
        evaluation_result = await self._evaluate_strategy(strategy_config, target_queries)
        
        # Calculate evaluation time
        evaluation_time = time.time() - start_time
        
        # Extract components
        precision = evaluation_result.get("precision", 0.0)
        recall = evaluation_result.get("recall", 0.0)
        f1_score = evaluation_result.get("f1_score", 0.0)  # noqa: F841
        
        # Apply complexity penalty
        complexity = individual.genome.complexity()
        complexity_penalty = self.complexity_penalty_factor * complexity / max(1, self.max_patterns * 2)
        
        # Normalize evaluation time
        max_time = 5.0  # seconds
        runtime_score = max(0, 1.0 - (evaluation_time / max_time))
        
        # Create fitness components based on specialization
        components = []
        
        if specialization == "precision":
            # Focus on precision
            components = [
                FitnessComponent("precision", precision, weight=0.7),
                FitnessComponent("recall", recall, weight=0.1),
                FitnessComponent("complexity", 1.0 - complexity_penalty, weight=0.1),
                FitnessComponent("runtime", runtime_score, weight=0.1)
            ]
        elif specialization == "recall":
            # Focus on recall
            components = [
                FitnessComponent("precision", precision, weight=0.1),
                FitnessComponent("recall", recall, weight=0.7),
                FitnessComponent("complexity", 1.0 - complexity_penalty, weight=0.1),
                FitnessComponent("runtime", runtime_score, weight=0.1)
            ]
        elif specialization == "complexity":
            # Focus on simplicity
            components = [
                FitnessComponent("precision", precision, weight=0.3),
                FitnessComponent("recall", recall, weight=0.3),
                FitnessComponent("complexity", 1.0 - complexity_penalty, weight=0.3),
                FitnessComponent("runtime", runtime_score, weight=0.1)
            ]
        elif specialization == "runtime":
            # Focus on fast execution
            components = [
                FitnessComponent("precision", precision, weight=0.3),
                FitnessComponent("recall", recall, weight=0.3),
                FitnessComponent("complexity", 1.0 - complexity_penalty, weight=0.1),
                FitnessComponent("runtime", runtime_score, weight=0.3)
            ]
        else:
            # Balanced fitness
            components = [
                FitnessComponent("precision", precision, weight=self.fitness_weights["precision"]),
                FitnessComponent("recall", recall, weight=self.fitness_weights["recall"]),
                FitnessComponent("complexity", 1.0 - complexity_penalty, weight=self.fitness_weights["complexity"]),
                FitnessComponent("runtime", runtime_score, weight=self.fitness_weights["runtime"])
            ]
        
        # Create multi-fitness object
        fitness = MultiFitness(components=components)
        
        # Update pattern bank with effective patterns
        if fitness.scalar_value > 0.7:
            self._update_pattern_bank(individual.genome, fitness.scalar_value)
        
        # Cache the result
        self._evaluation_cache[genome_hash] = fitness
        
        return fitness
    
    def _update_pattern_bank(self, genome: StrategyGenome, fitness: float) -> None:
        """
        Update pattern bank with effective patterns.
        
        Args:
            genome: Genome with patterns to potentially add
            fitness: Fitness value of the genome
        """
        # Extract patterns with good fitness
        for pattern in genome.patterns:
            # Check if pattern should be added
            pattern_complexity = pattern.complexity()
            
            # Skip very simple patterns
            if pattern_complexity < 2:
                continue
                
            # Calculate pattern fitness contribution
            pattern_fitness = fitness * (1.0 - (pattern_complexity / 10.0))
            
            # Add to bank if good enough
            if pattern_fitness > 0.6:
                # Check if similar pattern exists
                similar_exists = False
                for bank_pattern, _ in self._pattern_bank:
                    # Check pattern text similarity
                    bank_texts = set(bank_pattern.get_all_text_patterns())
                    pattern_texts = set(pattern.get_all_text_patterns())
                    
                    if bank_texts and pattern_texts:
                        intersection = len(bank_texts.intersection(pattern_texts))
                        union = len(bank_texts.union(pattern_texts))
                        similarity = intersection / union if union > 0 else 0
                        
                        if similarity > 0.7:
                            similar_exists = True
                            break
                
                if not similar_exists:
                    # Limit bank size
                    if len(self._pattern_bank) >= 20:
                        # Remove worst pattern
                        self._pattern_bank.sort(key=lambda x: x[1])
                        self._pattern_bank.pop(0)
                    
                    # Add pattern
                    self._pattern_bank.append((pattern, pattern_fitness))
    
    async def select_parent(self, population: Population[StrategyGenome, MultiFitness]) -> Individual[StrategyGenome, MultiFitness]:
        """
        Select a parent individual for reproduction using advanced selection methods.
        
        Args:
            population: Population to select from
            
        Returns:
            Selected parent individual
        """
        # Select parent based on the configured selection method
        if self.config.selection_method == "tournament":
            # Enhanced tournament selection with diversity
            tournament_size = min(5, len(population.individuals))
            tournament = random.sample(population.individuals, tournament_size)
            
            if self.diversity_preservation and random.random() < 0.3:
                # Sometimes select the most diverse individual in tournament
                if len(tournament) > 1:
                    most_diverse_idx = 0
                    max_diversity = -1
                    
                    for i, candidate in enumerate(tournament):
                        # Calculate diversity as average difference from other candidates
                        total_diff = 0
                        for j, other in enumerate(tournament):
                            if i != j:
                                total_diff += genome_difference(candidate.genome, other.genome)
                        
                        avg_diff = total_diff / (len(tournament) - 1)
                        
                        if avg_diff > max_diversity:
                            max_diversity = avg_diff
                            most_diverse_idx = i
                    
                    return tournament[most_diverse_idx]
            
            # Default tournament selection based on fitness
            return max(tournament, key=lambda i: i.fitness.scalar_value if i.fitness else 0)
            
        elif self.config.selection_method == "roulette":
            # Enhanced roulette wheel selection with fitness sharing
            if self.fitness_sharing:
                # Apply fitness sharing
                shared_fitness = self._apply_fitness_sharing(population)
                total_fitness = sum(shared_fitness.values())
                
                if total_fitness <= 0:
                    return random.choice(population.individuals)
                
                # Normalize fitness values
                normalized_fitness = {i.id: (shared_fitness[i.id] / total_fitness) for i in population.individuals}
                
                # Select based on fitness probabilities
                r = random.random()
                cum_prob = 0
                for individual in population.individuals:
                    prob = normalized_fitness[individual.id]
                    cum_prob += prob
                    if cum_prob >= r:
                        return individual
                
                # Fallback
                return population.individuals[-1]
            else:
                # Standard roulette selection
                total_fitness = sum(i.fitness.scalar_value for i in population.individuals if i.fitness)
                if total_fitness <= 0:
                    return random.choice(population.individuals)
                
                # Normalize fitness values
                normalized_fitness = [(i, (i.fitness.scalar_value / total_fitness) if i.fitness else 0) 
                                    for i in population.individuals]
                
                # Select based on fitness probabilities
                r = random.random()
                cum_prob = 0
                for individual, prob in normalized_fitness:
                    cum_prob += prob
                    if cum_prob >= r:
                        return individual
                
                # Fallback
                return population.individuals[-1]
            
        elif self.config.selection_method == "rank":
            # Enhanced rank-based selection with non-linear ranking
            sorted_individuals = sorted(
                population.individuals,
                key=lambda i: i.fitness.scalar_value if i.fitness else 0,
                reverse=True
            )
            
            # Calculate non-linear rank-based probabilities
            n = len(sorted_individuals)
            
            # Non-linear ranking uses exponential decay of probabilities
            selection_pressure = 1.5  # Higher values increase selection pressure
            rank_weights = [selection_pressure ** (n - i - 1) for i in range(n)]
            total_weight = sum(rank_weights)
            probabilities = [w / total_weight for w in rank_weights]
            
            # Select based on rank probabilities
            r = random.random()
            cum_prob = 0
            for i, prob in enumerate(probabilities):
                cum_prob += prob
                if cum_prob >= r:
                    return sorted_individuals[i]
            
            # Fallback
            return sorted_individuals[0]
        
        elif self.config.selection_method == "pareto":
            # Pareto-based selection for multi-objective optimization
            # Find non-dominated individuals (Pareto front)
            pareto_front = []
            for ind1 in population.individuals:
                is_dominated = False
                for ind2 in population.individuals:
                    if ind2.fitness and ind1.fitness and ind2.fitness.dominates(ind1.fitness):
                        is_dominated = True
                        break
                if not is_dominated:
                    pareto_front.append(ind1)
            
            if pareto_front:
                return random.choice(pareto_front)
            else:
                return random.choice(population.individuals)
            
        else:
            # Default to random selection
            return random.choice(population.individuals)
    
    def _apply_fitness_sharing(self, population: Population[StrategyGenome, MultiFitness]) -> Dict[str, float]:
        """
        Apply fitness sharing to promote diversity.
        
        Args:
            population: Population to apply fitness sharing to
            
        Returns:
            Dictionary mapping individual IDs to shared fitness values
        """
        shared_fitness = {}
        
        for i, ind1 in enumerate(population.individuals):
            if not ind1.fitness:
                shared_fitness[ind1.id] = 0
                continue
                
            raw_fitness = ind1.fitness.scalar_value
            niche_count = 0
            
            for j, ind2 in enumerate(population.individuals):
                # Calculate distance between individuals
                distance = genome_difference(ind1.genome, ind2.genome)
                
                # Apply sharing function
                if distance < self.fitness_sharing_radius:
                    # Linear sharing function
                    sh = 1.0 - (distance / self.fitness_sharing_radius)
                    niche_count += sh
            
            # Apply shared fitness
            if niche_count > 0:
                shared_fitness[ind1.id] = raw_fitness / niche_count
            else:
                shared_fitness[ind1.id] = raw_fitness
                
        return shared_fitness
    
    async def crossover(
        self, 
        parent1: Individual[StrategyGenome, MultiFitness], 
        parent2: Individual[StrategyGenome, MultiFitness]
    ) -> Individual[StrategyGenome, MultiFitness]:
        """
        Perform advanced crossover between two parent strategies.
        
        Args:
            parent1: First parent individual
            parent2: Second parent individual
            
        Returns:
            Child individual
        """
        # Get adaptive parameters
        crossover_points = parent1.genome.adaptive_params.get("crossover_points", 2)
        
        # Select crossover operator based on parent properties
        crossover_op = random.choices(
            [self._crossover_pattern_tree, self._crossover_full, self._crossover_pattern_level, self._crossover_options],
            weights=[0.4, 0.3, 0.2, 0.1],
            k=1
        )[0]
        
        # Apply crossover
        child_genome = await crossover_op(parent1.genome, parent2.genome, crossover_points)
        
        # Update adaptive parameters
        child_genome.adaptive_params = parent1.genome.adaptive_params.copy()
        
        # Merge metadata from parents
        child_genome.meta = {
            "objective": self.objective,
            "strategy_type": self.strategy_type,
            "parent1_id": parent1.id,
            "parent2_id": parent2.id,
            "created_at": time.time(),
            "crossover_op": crossover_op.__name__
        }
        
        # Inherit island from better parent
        parent1_fitness = parent1.fitness.scalar_value if parent1.fitness else 0
        parent2_fitness = parent2.fitness.scalar_value if parent2.fitness else 0
        better_parent = parent1 if parent1_fitness >= parent2_fitness else parent2
        island_id = better_parent.metadata.get("island_id")
        specialization = better_parent.metadata.get("specialization", "general")
        
        # Create child individual
        child = Individual[StrategyGenome, MultiFitness](
            genome=child_genome,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.id, parent2.id]
        )
        
        # Add metadata
        child.metadata["island_id"] = island_id
        child.metadata["specialization"] = specialization
        
        return child
    
    async def _crossover_pattern_tree(
        self, 
        genome1: StrategyGenome, 
        genome2: StrategyGenome,
        crossover_points: int = 2
    ) -> StrategyGenome:
        """
        Perform crossover at the pattern tree level.
        
        This exchanges subtrees between parents' pattern trees.
        
        Args:
            genome1: First parent genome
            genome2: Second parent genome
            crossover_points: Number of crossover points
            
        Returns:
            Child genome
        """
        # Create a copy of genome1 as basis for child
        child_genome = StrategyGenome(
            patterns=self._deep_copy_patterns(genome1.patterns),
            options=genome1.options.copy(),
            meta=genome1.meta.copy(),
            weights=genome1.weights.copy(),
            constraints=genome1.constraints.copy(),
            filters=genome1.filters.copy(),
            adaptive_params=genome1.adaptive_params.copy()
        )
        
        # If either parent has no patterns, use the other parent's patterns
        if not genome1.patterns:
            child_genome.patterns = self._deep_copy_patterns(genome2.patterns)
            return child_genome
        elif not genome2.patterns:
            return child_genome
        
        # Collect all subtrees from both parents
        subtrees1 = self._collect_subtrees(genome1.patterns)
        subtrees2 = self._collect_subtrees(genome2.patterns)
        
        if not subtrees1 or not subtrees2:
            return child_genome
            
        # Number of crossover operations to perform
        operations = min(crossover_points, min(len(subtrees1), len(subtrees2)))
        
        for _ in range(operations):
            # Select random subtree from parent1
            parent1_idx = random.randrange(len(subtrees1))
            parent1_node, parent1_path = subtrees1[parent1_idx]
            
            # Select random subtree from parent2
            parent2_idx = random.randrange(len(subtrees2))
            parent2_node, parent2_path = subtrees2[parent2_idx]
            
            # Copy parent2's subtree
            parent2_subtree_copy = self._deep_copy_node(parent2_node)
            
            # Replace subtree in child's patterns
            if not parent1_path:  # Root node
                if parent1_node in child_genome.patterns:
                    idx = child_genome.patterns.index(parent1_node)
                    child_genome.patterns[idx] = parent2_subtree_copy
            else:
                # Navigate to parent of the node to replace
                current = child_genome.patterns[parent1_path[0]]
                for i in range(1, len(parent1_path) - 1):
                    idx = parent1_path[i]
                    current = current.children[idx]
                
                # Replace child
                last_idx = parent1_path[-1]
                current.children[last_idx] = parent2_subtree_copy
        
        # Ensure patterns are within limits
        while len(child_genome.patterns) > self.max_patterns:
            child_genome.patterns.pop()
            
        while len(child_genome.patterns) < self.min_patterns and genome2.patterns:
            idx = random.randrange(len(genome2.patterns))
            pattern_copy = self._deep_copy_node(genome2.patterns[idx])
            child_genome.patterns.append(pattern_copy)
            
        return child_genome
    
    def _collect_subtrees(self, patterns: List[PatternNode]) -> List[Tuple[PatternNode, List[int]]]:
        """
        Collect all subtrees from a list of pattern trees.
        
        Args:
            patterns: List of pattern trees
            
        Returns:
            List of (node, path) tuples, where path is a list of indices to reach the node
        """
        result = []
        
        # Add root nodes
        for i, pattern in enumerate(patterns):
            result.append((pattern, [i]))
            
            # Recursively add subtrees
            self._collect_subtrees_recursive(pattern, [i], result)
            
        return result
    
    def _collect_subtrees_recursive(
        self, 
        node: PatternNode, 
        path: List[int], 
        result: List[Tuple[PatternNode, List[int]]]
    ) -> None:
        """
        Recursively collect subtrees from a pattern tree.
        
        Args:
            node: Current node
            path: Path of indices to reach this node
            result: List to store (node, path) tuples
        """
        for i, child in enumerate(node.children):
            child_path = path + [i]
            result.append((child, child_path))
            
            # Recurse on children
            self._collect_subtrees_recursive(child, child_path, result)
    
    async def _crossover_full(
        self, 
        genome1: StrategyGenome, 
        genome2: StrategyGenome,
        crossover_points: int = 2
    ) -> StrategyGenome:
        """
        Perform full genome crossover.
        
        This combines patterns from both parents and mixes options and constraints.
        
        Args:
            genome1: First parent genome
            genome2: Second parent genome
            crossover_points: Not used for this crossover type
            
        Returns:
            Child genome
        """
        # Create a new child genome
        child_genome = StrategyGenome(
            patterns=[],
            options={},
            meta={},
            weights={},
            constraints=[],
            filters=[],
            adaptive_params={}
        )
        
        # Mix patterns from both parents
        all_patterns = genome1.patterns + genome2.patterns
        if all_patterns:
            # Deduplicate similar patterns
            unique_patterns = []
            for pattern in all_patterns:
                # Check if similar pattern already included
                similar_exists = False
                for existing in unique_patterns:
                    # Check pattern text similarity
                    existing_texts = set(existing.get_all_text_patterns())
                    pattern_texts = set(pattern.get_all_text_patterns())
                    
                    if existing_texts and pattern_texts:
                        intersection = len(existing_texts.intersection(pattern_texts))
                        union = len(existing_texts.union(pattern_texts))
                        similarity = intersection / union if union > 0 else 0
                        
                        if similarity > 0.7:
                            similar_exists = True
                            break
                
                if not similar_exists:
                    unique_patterns.append(pattern)
            
            # Randomly select patterns from the unique set
            num_patterns = random.randint(
                min(self.min_patterns, len(unique_patterns)),
                min(self.max_patterns, len(unique_patterns))
            )
            
            selected_pattern_indices = random.sample(range(len(unique_patterns)), num_patterns)
            child_genome.patterns = [self._deep_copy_node(unique_patterns[i]) for i in selected_pattern_indices]
            
            # Set pattern weights
            for i, pattern in enumerate(child_genome.patterns):
                # Inherit weight from parent if available
                if i < len(genome1.patterns) and i in genome1.weights:
                    child_genome.weights[i] = genome1.weights[i]
                elif i < len(genome2.patterns) and i in genome2.weights:
                    child_genome.weights[i] = genome2.weights[i]
                else:
                    # Default weight
                    child_genome.weights[i] = 1.0
        
        # Mix options
        # For each option, randomly choose from parent1 or parent2
        all_options = set(genome1.options.keys()) | set(genome2.options.keys())
        for option in all_options:
            if option in genome1.options and option in genome2.options:
                # Both parents have this option
                if isinstance(genome1.options[option], (int, float)) and isinstance(genome2.options[option], (int, float)):
                    # Numeric option - blend values with random weight
                    alpha = random.random()
                    child_genome.options[option] = (
                        alpha * genome1.options[option] + 
                        (1 - alpha) * genome2.options[option]
                    )
                else:
                    # Non-numeric option - choose one
                    child_genome.options[option] = (
                        genome1.options[option] if random.random() < 0.5 
                        else genome2.options[option]
                    )
            elif option in genome1.options:
                # Only parent1 has this option - 70% chance to inherit
                if random.random() < 0.7:
                    child_genome.options[option] = genome1.options[option]
            else:
                # Only parent2 has this option - 70% chance to inherit
                if random.random() < 0.7:
                    child_genome.options[option] = genome2.options[option]
        
        # Mix constraints using similar logic
        all_constraints = genome1.constraints + genome2.constraints
        if all_constraints:
            # Deduplicate similar constraints
            unique_constraints = []
            for constraint in all_constraints:
                # Check if similar constraint already included
                similar_exists = False
                for existing in unique_constraints:
                    if constraint.get("type") == existing.get("type"):
                        similarity = random.random()  # Simplified similarity check
                        if similarity > 0.7:
                            similar_exists = True
                            break
                
                if not similar_exists:
                    unique_constraints.append(constraint)
            
            # Randomly select constraints
            num_constraints = min(len(unique_constraints), random.randint(0, 3))
            if num_constraints > 0:
                selected_constraint_indices = random.sample(range(len(unique_constraints)), num_constraints)
                child_genome.constraints = [unique_constraints[i].copy() for i in selected_constraint_indices]
        
        # Mix filters
        all_filters = genome1.filters + genome2.filters
        if all_filters:
            # Randomly select filters
            num_filters = min(len(all_filters), random.randint(0, 2))
            if num_filters > 0:
                selected_filter_indices = random.sample(range(len(all_filters)), num_filters)
                child_genome.filters = [all_filters[i].copy() for i in selected_filter_indices]
        
        # Mix adaptive parameters
        for param, value in genome1.adaptive_params.items():
            if param in genome2.adaptive_params:
                # Blend numeric parameters
                if isinstance(value, (int, float)) and isinstance(genome2.adaptive_params[param], (int, float)):
                    alpha = random.random()
                    child_genome.adaptive_params[param] = (
                        alpha * value + (1 - alpha) * genome2.adaptive_params[param]
                    )
                else:
                    # Choose one for non-numeric
                    child_genome.adaptive_params[param] = value if random.random() < 0.5 else genome2.adaptive_params[param]
            else:
                child_genome.adaptive_params[param] = value
                
        # Add any params from genome2 not in genome1
        for param, value in genome2.adaptive_params.items():
            if param not in child_genome.adaptive_params:
                child_genome.adaptive_params[param] = value
                
        # Ensure required adaptive params
        if "mutation_rate" not in child_genome.adaptive_params:
            child_genome.adaptive_params["mutation_rate"] = random.uniform(0.1, 0.3)
            
        if "crossover_points" not in child_genome.adaptive_params:
            child_genome.adaptive_params["crossover_points"] = random.randint(1, 3)
            
        return child_genome
    
    async def _crossover_pattern_level(
        self, 
        genome1: StrategyGenome, 
        genome2: StrategyGenome,
        crossover_points: int = 2
    ) -> StrategyGenome:
        """
        Perform pattern-level crossover.
        
        This exchanges entire patterns between parents.
        
        Args:
            genome1: First parent genome
            genome2: Second parent genome
            crossover_points: Number of crossover points
            
        Returns:
            Child genome
        """
        # Create a copy of genome1 as basis for child
        child_genome = StrategyGenome(
            patterns=self._deep_copy_patterns(genome1.patterns),
            options=genome1.options.copy(),
            meta=genome1.meta.copy(),
            weights=genome1.weights.copy(),
            constraints=genome1.constraints.copy(),
            filters=genome1.filters.copy(),
            adaptive_params=genome1.adaptive_params.copy()
        )
        
        # Handle edge cases
        if not genome2.patterns:
            return child_genome
            
        if not child_genome.patterns:
            child_genome.patterns = self._deep_copy_patterns(genome2.patterns[:self.max_patterns])
            return child_genome
        
        # Number of patterns to exchange
        exchange_count = min(crossover_points, min(len(child_genome.patterns), len(genome2.patterns)))
        
        # Select random positions to replace
        positions = random.sample(range(len(child_genome.patterns)), exchange_count)
        
        # Select random patterns from genome2
        g2_patterns = random.sample(genome2.patterns, exchange_count)
        
        # Replace patterns
        for i, pos in enumerate(positions):
            child_genome.patterns[pos] = self._deep_copy_node(g2_patterns[i])
            
            # Update weight if available
            pattern_idx = genome2.patterns.index(g2_patterns[i])
            if pattern_idx in genome2.weights:
                child_genome.weights[pos] = genome2.weights[pattern_idx]
        
        return child_genome
    
    async def _crossover_options(
        self, 
        genome1: StrategyGenome, 
        genome2: StrategyGenome,
        crossover_points: int = 2
    ) -> StrategyGenome:
        """
        Perform options-focused crossover.
        
        This keeps patterns from one parent and options from the other.
        
        Args:
            genome1: First parent genome
            genome2: Second parent genome
            crossover_points: Not used for this crossover type
            
        Returns:
            Child genome
        """
        # Randomly decide which parent provides patterns
        if random.random() < 0.5:
            patterns_genome = genome1
            options_genome = genome2
        else:
            patterns_genome = genome2
            options_genome = genome1
        
        # Create child genome with patterns from one parent
        child_genome = StrategyGenome(
            patterns=self._deep_copy_patterns(patterns_genome.patterns),
            options=options_genome.options.copy(),
            meta={},
            weights=patterns_genome.weights.copy(),
            constraints=options_genome.constraints.copy(),
            filters=options_genome.filters.copy(),
            adaptive_params={}
        )
        
        # Mix adaptive parameters
        for param, value in genome1.adaptive_params.items():
            if param in genome2.adaptive_params:
                # Blend numeric parameters
                if isinstance(value, (int, float)) and isinstance(genome2.adaptive_params[param], (int, float)):
                    alpha = random.random()
                    child_genome.adaptive_params[param] = (
                        alpha * value + (1 - alpha) * genome2.adaptive_params[param]
                    )
                else:
                    # Choose one for non-numeric
                    child_genome.adaptive_params[param] = value if random.random() < 0.5 else genome2.adaptive_params[param]
            else:
                child_genome.adaptive_params[param] = value
                
        # Add any params from genome2 not in genome1
        for param, value in genome2.adaptive_params.items():
            if param not in child_genome.adaptive_params:
                child_genome.adaptive_params[param] = value
                
        # Ensure required adaptive params
        if "mutation_rate" not in child_genome.adaptive_params:
            child_genome.adaptive_params["mutation_rate"] = random.uniform(0.1, 0.3)
            
        if "crossover_points" not in child_genome.adaptive_params:
            child_genome.adaptive_params["crossover_points"] = random.randint(1, 3)
        
        return child_genome
    
    def _deep_copy_patterns(self, patterns: List[PatternNode]) -> List[PatternNode]:
        """
        Create a deep copy of a list of pattern trees.
        
        Args:
            patterns: List of pattern trees to copy
            
        Returns:
            Deep copy of the pattern list
        """
        return [self._deep_copy_node(pattern) for pattern in patterns]
    
    def _deep_copy_node(self, node: PatternNode) -> PatternNode:
        """
        Create a deep copy of a pattern tree node.
        
        Args:
            node: Pattern node to copy
            
        Returns:
            Deep copy of the node
        """
        return PatternNode(
            pattern_type=node.pattern_type,
            pattern_text=node.pattern_text,
            operator=node.operator,
            children=[self._deep_copy_node(child) for child in node.children],
            parameters=node.parameters.copy()
        )
    
    async def mutate(self, individual: Individual[StrategyGenome, MultiFitness]) -> Individual[StrategyGenome, MultiFitness]:
        """
        Mutate a strategy individual using adaptive mutation.
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Mutated individual
        """
        # Get mutation rate from adaptive parameters
        mutation_rate = individual.genome.adaptive_params.get("mutation_rate", 0.2)
        
        # Deep copy the genome
        mutated_genome = StrategyGenome(
            patterns=self._deep_copy_patterns(individual.genome.patterns),
            options=individual.genome.options.copy(),
            meta=individual.genome.meta.copy(),
            weights=individual.genome.weights.copy(),
            constraints=[c.copy() for c in individual.genome.constraints],
            filters=[f.copy() for f in individual.genome.filters],
            adaptive_params=individual.genome.adaptive_params.copy()
        )
        
        # Multiple mutation operations can be applied
        num_mutations = 1
        if random.random() < mutation_rate:
            num_mutations += 1
            if random.random() < mutation_rate / 2:
                num_mutations += 1
                
        # Apply mutations
        mutation_ops = []
        
        # Always include these mutations
        base_mutations = [
            self._mutate_add_pattern,
            self._mutate_remove_pattern,
            self._mutate_modify_pattern,
            self._mutate_pattern_tree,
            self._mutate_options
        ]
        
        # Advanced mutations only if genome is somewhat evolved
        advanced_mutations = []
        if len(mutated_genome.patterns) >= 2 or individual.generation > 5:
            advanced_mutations = [
                self._mutate_combine_patterns,
                self._mutate_pattern_weights,
                self._mutate_constraints,
                self._mutate_filters,
                self._mutate_pattern_bank
            ]
            
        mutation_ops = base_mutations + advanced_mutations
        
        if mutation_ops:
            for _ in range(num_mutations):
                # Apply a random mutation operation
                op = random.choice(mutation_ops)
                mutated_genome = await op(mutated_genome)
        
        # Adapt mutation parameters based on context
        parent_fitness = individual.fitness.scalar_value if individual.fitness else 0  # noqa: F841
        
        # Update metadata
        mutated_genome.meta["mutated_from"] = individual.id
        mutated_genome.meta["mutated_at"] = time.time()
        
        # Create mutated individual
        mutated = Individual[StrategyGenome, MultiFitness](
            genome=mutated_genome,
            generation=individual.generation,  # Keep same generation
            parent_ids=[individual.id]
        )
        
        # Keep metadata from parent
        mutated.metadata = individual.metadata.copy()
        mutated.metadata["mutation_ops"] = [op.__name__ for op in mutation_ops[:num_mutations]]
        
        return mutated
    
    async def _mutate_add_pattern(self, genome: StrategyGenome) -> StrategyGenome:
        """
        Mutation: Add a new pattern to the genome.
        
        Args:
            genome: Genome to mutate
            
        Returns:
            Mutated genome
        """
        # Check if we're already at max patterns
        if len(genome.patterns) >= self.max_patterns:
            return genome
    
    async def _mutate_filters(self, genome: StrategyGenome) -> StrategyGenome:
        """
        Mutation: Modify post-processing filters.
        
        Args:
            genome: Genome to mutate
            
        Returns:
            Mutated genome
        """
        # Different filter mutations
        mutation_type = random.choice(["add", "remove", "modify"])
        
        if mutation_type == "add":
            # Add a new filter
            filter_type = random.choice(["min_confidence", "exclude_pattern", "content_type"])
            
            if filter_type == "min_confidence":
                filter_config = {
                    "type": "min_confidence",
                    "description": "Minimum confidence threshold",
                    "threshold": random.uniform(0.5, 0.9)
                }
                
            elif filter_type == "exclude_pattern":
                filter_config = {
                    "type": "exclude_pattern",
                    "description": "Exclude matches containing pattern",
                    "pattern": ''.join([random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(random.randint(3, 8))]),
                    "is_regex": random.choice([True, False])
                }
                
            elif filter_type == "content_type":
                filter_config = {
                    "type": "content_type",
                    "description": "Filter by content type",
                    "content_types": random.sample(["code", "comment", "string", "docstring"], 
                                                 k=random.randint(1, 3))
                }
                
            # Add the filter if not already present
            if not any(f.get("type") == filter_config["type"] for f in genome.filters):
                genome.filters.append(filter_config)
                
        elif mutation_type == "remove" and genome.filters:
            # Remove a random filter
            idx = random.randrange(len(genome.filters))
            genome.filters.pop(idx)
            
        elif mutation_type == "modify" and genome.filters:
            # Modify a random filter
            idx = random.randrange(len(genome.filters))
            filter_config = genome.filters[idx]
            
            if filter_config["type"] == "min_confidence" and "threshold" in filter_config:
                # Modify threshold
                delta = random.uniform(-0.1, 0.1)
                filter_config["threshold"] = max(0.1, min(0.95, filter_config["threshold"] + delta))
                
            elif filter_config["type"] == "exclude_pattern" and "pattern" in filter_config:
                # Modify pattern
                if random.random() < 0.5:
                    # Change pattern text
                    chars = list(filter_config["pattern"])
                    if chars:
                        idx = random.randrange(len(chars))
                        chars[idx] = random.choice('abcdefghijklmnopqrstuvwxyz')
                        filter_config["pattern"] = ''.join(chars)
                else:
                    # Toggle regex flag
                    filter_config["is_regex"] = not filter_config.get("is_regex", False)
                    
            elif filter_config["type"] == "content_type" and "content_types" in filter_config:
                # Modify content types
                all_types = ["code", "comment", "string", "docstring"]
                current_types = filter_config["content_types"]
                
                if len(current_types) < len(all_types) and random.random() < 0.7:
                    # Add a type
                    available_types = [t for t in all_types if t not in current_types]
                    if available_types:
                        type_to_add = random.choice(available_types)
                        filter_config["content_types"].append(type_to_add)
                elif len(current_types) > 1:
                    # Remove a type
                    type_to_remove = random.choice(current_types)
                    filter_config["content_types"].remove(type_to_remove)
            
            # Update the filter
            genome.filters[idx] = filter_config
            
        return genome
        
    async def _mutate_pattern_bank(self, genome: StrategyGenome) -> StrategyGenome:
        """
        Mutation: Incorporate patterns from the pattern bank.
        
        Args:
            genome: Genome to mutate
            
        Returns:
            Mutated genome
        """
        # Check if we have patterns in the bank
        if not self._pattern_bank:
            return genome
            
        # Different pattern bank mutations
        mutation_type = random.choice(["replace", "incorporate", "combine"])
        
        if mutation_type == "replace" and genome.patterns:
            # Replace a random pattern with one from the bank
            idx = random.randrange(len(genome.patterns))
            
            # Select a pattern from the bank (weighted by fitness)
            bank_patterns = [(pattern, fitness) for pattern, fitness in self._pattern_bank]
            weights = [fitness for _, fitness in bank_patterns]
            
            selected_pattern, _ = random.choices(bank_patterns, weights=weights, k=1)[0]
            
            # Replace the pattern
            genome.patterns[idx] = self._deep_copy_node(selected_pattern)
            
        elif mutation_type == "incorporate":
            # Add a pattern from the bank
            if len(genome.patterns) >= self.max_patterns:
                # Remove a random pattern to make room
                idx = random.randrange(len(genome.patterns))
                genome.patterns.pop(idx)
                
            # Select a pattern from the bank (weighted by fitness)
            bank_patterns = [(pattern, fitness) for pattern, fitness in self._pattern_bank]
            weights = [fitness for _, fitness in bank_patterns]
            
            selected_pattern, _ = random.choices(bank_patterns, weights=weights, k=1)[0]
            
            # Add the pattern
            genome.patterns.append(self._deep_copy_node(selected_pattern))
            
            # Add weight
            genome.weights[len(genome.patterns) - 1] = 1.0
            
        elif mutation_type == "combine" and genome.patterns:
            # Combine a pattern from the bank with an existing pattern
            idx = random.randrange(len(genome.patterns))
            existing_pattern = genome.patterns[idx]
            
            # Select a pattern from the bank (weighted by fitness)
            bank_patterns = [(pattern, fitness) for pattern, fitness in self._pattern_bank]
            weights = [fitness for _, fitness in bank_patterns]
            
            selected_pattern, _ = random.choices(bank_patterns, weights=weights, k=1)[0]
            
            # Create a combined pattern
            operator = random.choice(list(PatternOperator))
            
            combined = PatternNode(
                pattern_type=PatternType.COMBINED,
                operator=operator,
                children=[self._deep_copy_node(existing_pattern), self._deep_copy_node(selected_pattern)],
                parameters={"case_sensitive": random.choice([True, False])}
            )
            
            # Set operator-specific parameters
            if operator == PatternOperator.NEAR:
                combined.parameters["distance"] = random.randint(1, 20)
            elif operator == PatternOperator.FOLLOWED_BY:
                combined.parameters["max_gap"] = random.randint(1, 50)
                
            # Replace the existing pattern
            genome.patterns[idx] = combined
            
        return genome
        
        # Generate a new pattern
        mutation_type = random.choices(
            ["random", "bank", "combine"],
            weights=[0.5, 0.3, 0.2],
            k=1
        )[0]
        
        if mutation_type == "bank" and self._pattern_bank:
            # Use a pattern from the bank
            bank_pattern, _ = random.choice(self._pattern_bank)
            new_pattern = self._deep_copy_node(bank_pattern)
            
        elif mutation_type == "combine" and len(genome.patterns) >= 2:
            # Combine two existing patterns
            patterns = random.sample(genome.patterns, 2)
            
            # Create a combined pattern
            operator = random.choice(list(PatternOperator))
            new_pattern = PatternNode(
                pattern_type=PatternType.COMBINED,
                operator=operator,
                children=[self._deep_copy_node(p) for p in patterns],
                parameters={"case_sensitive": random.choice([True, False])}
            )
            
            if operator == PatternOperator.NEAR:
                new_pattern.parameters["distance"] = random.randint(1, 20)
            elif operator == PatternOperator.FOLLOWED_BY:
                new_pattern.parameters["max_gap"] = random.randint(1, 50)
                
        else:
            # Generate a completely new pattern
            new_pattern = await self._generate_random_pattern()
        
        # Add to the genome
        genome.patterns.append(new_pattern)
        
        # Add weight
        genome.weights[len(genome.patterns) - 1] = 1.0
        
        return genome
    
    async def _mutate_remove_pattern(self, genome: StrategyGenome) -> StrategyGenome:
        """
        Mutation: Remove a pattern from the genome.
        
        Args:
            genome: Genome to mutate
            
        Returns:
            Mutated genome
        """
        # Check if we can remove a pattern
        if len(genome.patterns) <= self.min_patterns:
            return genome
        
        # Remove a random pattern
        idx = random.randrange(len(genome.patterns))
        genome.patterns.pop(idx)
        
        # Remove weight
        if idx in genome.weights:
            del genome.weights[idx]
            
        # Update remaining weights
        new_weights = {}
        for i, (pattern_idx, weight) in enumerate(genome.weights.items()):
            if pattern_idx > idx:
                new_weights[pattern_idx - 1] = weight
            elif pattern_idx < idx:
                new_weights[pattern_idx] = weight
                
        genome.weights = new_weights
        
        return genome
    
    async def _mutate_modify_pattern(self, genome: StrategyGenome) -> StrategyGenome:
        """
        Mutation: Modify an existing pattern.
        
        Args:
            genome: Genome to mutate
            
        Returns:
            Mutated genome
        """
        # Check if there are patterns to modify
        if not genome.patterns:
            return genome
        
        # Select a random pattern
        idx = random.randrange(len(genome.patterns))
        pattern = genome.patterns[idx]
        
        # Determine modification type
        mod_type = random.choice(["modify_regex", "toggle_case", "modify_type", "modify_parameters"])
        
        if mod_type == "modify_regex" and pattern.pattern_text:
            # Modify the regex pattern
            original = pattern.pattern_text
            
            # Different modification strategies
            strategy = random.choice(["add_alternative", "add_quantifier", "character_class", "word_boundary", "small_change"])
            
            if strategy == "add_alternative" and '|' not in original:
                # Add an alternative
                if random.random() < 0.3:
                    # Add a term from the pattern bank if available
                    terms = []
                    for bank_pattern, _ in self._pattern_bank:
                        terms.extend(bank_pattern.get_all_text_patterns())
                    
                    if terms:
                        alt_term = random.choice(terms)
                        pattern.pattern_text = f"({original}|{alt_term})"
                    else:
                        # Generate a random alternative
                        words = original.split()
                        if words:
                            word_idx = random.randrange(len(words))
                            alt_word = ''.join([random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(len(words[word_idx]))])
                            words[word_idx] = f"(?:{words[word_idx]}|{alt_word})"
                            pattern.pattern_text = ' '.join(words)
                        else:
                            # Make a small modification
                            chars = list(original)
                            idx = random.randrange(len(chars))
                            chars[idx] = random.choice('abcdefghijklmnopqrstuvwxyz.*+?[](){}\\')
                            pattern.pattern_text = ''.join(chars)
                else:
                    # Add alternative with random word
                    alt_word = ''.join([random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(random.randint(3, 8))])
                    pattern.pattern_text = f"({original}|{alt_word})"
                    
            elif strategy == "add_quantifier":
                # Add a quantifier
                if re.search(r'[a-zA-Z0-9]', original):
                    # Find a character to add quantifier to
                    chars = list(original)
                    valid_indices = [i for i, c in enumerate(chars) if re.match(r'[a-zA-Z0-9]', c)]
                    
                    if valid_indices:
                        idx = random.choice(valid_indices)
                        quantifier = random.choice(['*', '+', '?', '{1,3}'])
                        
                        chars.insert(idx + 1, quantifier)
                        pattern.pattern_text = ''.join(chars)
                else:
                    # Default modification
                    pattern.pattern_text = original + random.choice(['*', '+', '?'])
                    
            elif strategy == "character_class":
                # Add or modify a character class
                if '[' in original and ']' in original:
                    # Modify existing character class
                    start = original.find('[')
                    end = original.find(']', start)
                    
                    if start >= 0 and end > start:
                        class_content = original[start+1:end]
                        
                        # Add or remove a character
                        if random.random() < 0.7:
                            # Add character
                            new_char = random.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                            class_content += new_char
                        else:
                            # Remove character if more than one
                            if len(class_content) > 1:
                                chars = list(class_content)
                                chars.pop(random.randrange(len(chars)))
                                class_content = ''.join(chars)
                                
                        pattern.pattern_text = original[:start+1] + class_content + original[end:]
                else:
                    # Add a new character class
                    if re.search(r'[a-zA-Z0-9]', original):
                        chars = list(original)
                        valid_indices = [i for i, c in enumerate(chars) if re.match(r'[a-zA-Z0-9]', c)]
                        
                        if valid_indices:
                            idx = random.choice(valid_indices)
                            char = chars[idx]
                            
                            # Create character class
                            if char.isalpha():
                                if char.islower():
                                    class_content = f"[{char}{char.upper()}]"
                                else:
                                    class_content = f"[{char}{char.lower()}]"
                            else:
                                # For digits, create a range
                                class_content = f"[{char}-9]"
                                
                            chars[idx] = class_content
                            pattern.pattern_text = ''.join(chars)
                            
            elif strategy == "word_boundary":
                # Add or remove word boundary
                if r'\b' in original:
                    # Remove a word boundary
                    pattern.pattern_text = original.replace(r'\b', '', 1)
                else:
                    # Add word boundary
                    pattern.pattern_text = r'\b' + original + r'\b'
                    
            else:  # small_change
                # Make a small random change
                if original:
                    chars = list(original)
                    idx = random.randrange(len(chars))
                    
                    # Random change type
                    change_type = random.choice(["replace", "insert", "delete"])
                    
                    if change_type == "replace":
                        chars[idx] = random.choice('abcdefghijklmnopqrstuvwxyz.*+?[](){}\\')
                    elif change_type == "insert":
                        chars.insert(idx, random.choice('abcdefghijklmnopqrstuvwxyz.*+?[](){}\\'))
                    elif change_type == "delete" and len(chars) > 1:
                        chars.pop(idx)
                        
                    pattern.pattern_text = ''.join(chars)
                else:
                    # Empty pattern, create a new one
                    letters = [random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(random.randint(3, 8))]
                    pattern.pattern_text = ''.join(letters)
            
        elif mod_type == "toggle_case":
            # Toggle case sensitivity
            pattern.parameters["case_sensitive"] = not pattern.parameters.get("case_sensitive", False)
            
        elif mod_type == "modify_type" and not pattern.children:
            # Change pattern type
            current_type = pattern.pattern_type
            new_type = random.choice([t for t in PatternType if t != current_type and t != PatternType.COMBINED])
            
            pattern.pattern_type = new_type
            
            # Update parameters for new type
            if new_type == PatternType.FUZZY:
                pattern.parameters["fuzzy_level"] = random.randint(1, 3)
                
        elif mod_type == "modify_parameters":
            # Modify parameters
            if pattern.pattern_type == PatternType.FUZZY and "fuzzy_level" in pattern.parameters:
                # Change fuzzy level
                delta = random.choice([-1, 1])
                pattern.parameters["fuzzy_level"] = max(1, min(5, pattern.parameters["fuzzy_level"] + delta))
                
            elif pattern.operator == PatternOperator.NEAR and "distance" in pattern.parameters:
                # Change distance
                delta = random.randint(-5, 5)
                pattern.parameters["distance"] = max(1, pattern.parameters["distance"] + delta)
                
            elif pattern.operator == PatternOperator.FOLLOWED_BY and "max_gap" in pattern.parameters:
                # Change max gap
                delta = random.randint(-10, 10)
                pattern.parameters["max_gap"] = max(1, pattern.parameters["max_gap"] + delta)
        
        # Update the pattern
        genome.patterns[idx] = pattern
        
        return genome
    
    async def _mutate_pattern_tree(self, genome: StrategyGenome) -> StrategyGenome:
        """
        Mutation: Modify the structure of a pattern tree.
        
        Args:
            genome: Genome to mutate
            
        Returns:
            Mutated genome
        """
        # Check if there are patterns to modify
        if not genome.patterns:
            return genome
        
        # Select a random pattern
        idx = random.randrange(len(genome.patterns))
        pattern = genome.patterns[idx]
        
        # Different tree mutations
        mutation_type = random.choice(["add_node", "remove_node", "swap_nodes", "change_operator"])
        
        if mutation_type == "add_node":
            # Add a child node to an existing node
            if pattern.pattern_type == PatternType.COMBINED:
                # Add to combined pattern
                new_node = await self._generate_random_pattern()
                pattern.children.append(new_node)
                
            elif not pattern.children:
                # Convert leaf to combined
                original_pattern = PatternNode(
                    pattern_type=pattern.pattern_type,
                    pattern_text=pattern.pattern_text,
                    parameters=pattern.parameters.copy()
                )
                
                new_node = await self._generate_random_pattern()
                
                pattern.pattern_type = PatternType.COMBINED
                pattern.operator = random.choice(list(PatternOperator))
                pattern.children = [original_pattern, new_node]
                pattern.pattern_text = None
                
                # Set parameters for operator
                if pattern.operator == PatternOperator.NEAR:
                    pattern.parameters["distance"] = random.randint(1, 20)
                elif pattern.operator == PatternOperator.FOLLOWED_BY:
                    pattern.parameters["max_gap"] = random.randint(1, 50)
                
        elif mutation_type == "remove_node" and pattern.children:
            # Remove a random child node
            if len(pattern.children) > 1:
                # Remove a random child
                child_idx = random.randrange(len(pattern.children))
                pattern.children.pop(child_idx)
                
            else:
                # Only one child - replace with it
                child = pattern.children[0]
                pattern.pattern_type = child.pattern_type
                pattern.pattern_text = child.pattern_text
                pattern.operator = child.operator
                pattern.children = child.children
                pattern.parameters = child.parameters.copy()
                
        elif mutation_type == "swap_nodes" and len(pattern.children) >= 2:
            # Swap two random child nodes
            idx1, idx2 = random.sample(range(len(pattern.children)), 2)
            pattern.children[idx1], pattern.children[idx2] = pattern.children[idx2], pattern.children[idx1]
            
        elif mutation_type == "change_operator" and pattern.pattern_type == PatternType.COMBINED:
            # Change the operator
            current_op = pattern.operator
            new_op = random.choice([op for op in PatternOperator if op != current_op])
            pattern.operator = new_op
            
            # Update parameters for new operator
            if new_op == PatternOperator.NEAR:
                pattern.parameters["distance"] = random.randint(1, 20)
            elif new_op == PatternOperator.FOLLOWED_BY:
                pattern.parameters["max_gap"] = random.randint(1, 50)
        
        # Apply tree depth limit
        if pattern.depth() > self.max_pattern_depth:
            # Simplify tree by removing deepest nodes
            self._limit_tree_depth(pattern, self.max_pattern_depth)
                
        # Update the pattern
        genome.patterns[idx] = pattern
        
        return genome
    
    def _limit_tree_depth(self, node: PatternNode, max_depth: int, current_depth: int = 1) -> None:
        """
        Limit the depth of a pattern tree.
        
        Args:
            node: Pattern node to limit
            max_depth: Maximum allowed depth
            current_depth: Current depth in the tree
        """
        if current_depth >= max_depth:
            # At max depth, remove all children
            node.children = []
            return
            
        # Process children
        for child in node.children:
            self._limit_tree_depth(child, max_depth, current_depth + 1)
    
    async def _mutate_options(self, genome: StrategyGenome) -> StrategyGenome:
        """
        Mutation: Modify strategy options.
        
        Args:
            genome: Genome to mutate
            
        Returns:
            Mutated genome
        """
        # Different option mutations
        mutation_type = random.choice(["modify_existing", "add_option", "remove_option"])
        
        if mutation_type == "modify_existing" and genome.options:
            # Select a random option to modify
            option = random.choice(list(genome.options.keys()))
            value = genome.options[option]
            
            # Modify the option based on its type
            if isinstance(value, bool):
                # Toggle boolean option
                genome.options[option] = not value
                
            elif isinstance(value, int):
                # Modify integer option
                delta = random.randint(-value // 5 or -1, value // 5 or 1)
                genome.options[option] = max(1, value + delta)
                
            elif isinstance(value, float):
                # Modify float option
                delta = random.uniform(-value / 5, value / 5)
                genome.options[option] = max(0.01, min(0.99, value + delta))
                
        elif mutation_type == "add_option":
            # Add a new option
            option_type = random.choice(["bool", "int", "float"])
            
            if option_type == "bool":
                options = ["use_word_boundaries", "match_whole_words", "ignore_comments", 
                         "search_in_imports", "search_in_strings", "ignore_case"]
                
                for option in options:
                    if option not in genome.options:
                        genome.options[option] = random.choice([True, False])
                        break
                        
            elif option_type == "int":
                options = [("max_results", lambda: random.randint(10, 1000)),
                        ("context_lines", lambda: random.randint(0, 10)),
                        ("search_depth", lambda: random.randint(1, 5))]
                
                for option, value_func in options:
                    if option not in genome.options:
                        genome.options[option] = value_func()
                        break
                        
            elif option_type == "float":
                options = [("confidence_threshold", lambda: random.uniform(0.1, 0.9)),
                        ("similarity_threshold", lambda: random.uniform(0.5, 0.95)),
                        ("timeout", lambda: random.uniform(1.0, 10.0))]
                
                for option, value_func in options:
                    if option not in genome.options:
                        genome.options[option] = value_func()
                        break
                
        elif mutation_type == "remove_option" and genome.options:
            # Remove a random option if we have more than 1
            if len(genome.options) > 1:
                option = random.choice(list(genome.options.keys()))
                del genome.options[option]
                
        return genome
    
    async def _mutate_combine_patterns(self, genome: StrategyGenome) -> StrategyGenome:
        """
        Mutation: Combine existing patterns into a more complex pattern.
        
        Args:
            genome: Genome to mutate
            
        Returns:
            Mutated genome
        """
        # Need at least two patterns to combine
        if len(genome.patterns) < 2:
            return genome
            
        # Select two random patterns
        indices = random.sample(range(len(genome.patterns)), 2)
        pattern1 = genome.patterns[indices[0]]
        pattern2 = genome.patterns[indices[1]]
        
        # Create a combined pattern
        operator = random.choice(list(PatternOperator))
        
        combined = PatternNode(
            pattern_type=PatternType.COMBINED,
            operator=operator,
            children=[self._deep_copy_node(pattern1), self._deep_copy_node(pattern2)],
            parameters={"case_sensitive": random.choice([True, False])}
        )
        
        # Set operator-specific parameters
        if operator == PatternOperator.NEAR:
            combined.parameters["distance"] = random.randint(1, 20)
        elif operator == PatternOperator.FOLLOWED_BY:
            combined.parameters["max_gap"] = random.randint(1, 50)
            
        # Remove original patterns and add combined
        # Remove in reverse order to avoid index issues
        indices.sort(reverse=True)
        for idx in indices:
            genome.patterns.pop(idx)
            
        # Add the combined pattern
        genome.patterns.append(combined)
        
        # Update weights
        weight1 = genome.weights.get(indices[0], 1.0)
        weight2 = genome.weights.get(indices[1], 1.0)
        
        # Remove old weights
        for idx in indices:
            if idx in genome.weights:
                del genome.weights[idx]
                
        # Update remaining weights
        new_weights = {}
        for pattern_idx, weight in genome.weights.items():
            new_idx = pattern_idx
            for idx in indices:
                if pattern_idx > idx:
                    new_idx -= 1
            new_weights[new_idx] = weight
            
        genome.weights = new_weights
        
        # Add weight for combined pattern
        genome.weights[len(genome.patterns) - 1] = (weight1 + weight2) / 2
        
        return genome
    
    async def _mutate_pattern_weights(self, genome: StrategyGenome) -> StrategyGenome:
        """
        Mutation: Modify pattern weights.
        
        Args:
            genome: Genome to mutate
            
        Returns:
            Mutated genome
        """
        if not genome.patterns:
            return genome
            
        # Select a random pattern
        idx = random.randrange(len(genome.patterns))
        
        # Get current weight or default
        current_weight = genome.weights.get(idx, 1.0)
        
        # Modify weight
        delta = random.uniform(-0.3, 0.3)
        new_weight = max(0.1, min(3.0, current_weight + delta))
        
        # Update weight
        genome.weights[idx] = new_weight
        
        return genome
    
    async def _mutate_constraints(self, genome: StrategyGenome) -> StrategyGenome:
        """
        Mutation: Modify search constraints.
        
        Args:
            genome: Genome to mutate
            
        Returns:
            Mutated genome
        """
        # Different constraint mutations
        mutation_type = random.choice(["add", "remove", "modify"])
        
        if mutation_type == "add":
            # Add a new constraint
            constraint_type = random.choice(["same_file", "within_distance", "ordered"])
            
            if constraint_type == "same_file":
                constraint = {
                    "type": "same_file",
                    "description": "Patterns must match in the same file"
                }
                
            elif constraint_type == "within_distance":
                constraint = {
                    "type": "within_distance",
                    "description": "Patterns must be within N lines",
                    "distance": random.randint(1, 50)
                }
                
            elif constraint_type == "ordered":
                constraint = {
                    "type": "ordered",
                    "description": "Patterns must match in order",
                    "strict": random.choice([True, False])
                }
                
                # Add the constraint if not already present
                if not any(c.get("type") == constraint["type"] for c in genome.constraints):
                    genome.constraints.append(constraint)
                
            return genome
            
        elif mutation_type == "remove" and genome.constraints:
            # Remove a random constraint
            idx = random.randrange(len(genome.constraints))
            genome.constraints.pop(idx)
            return genome
            
        elif mutation_type == "modify" and genome.constraints:
            # Modify a random constraint
            idx = random.randrange(len(genome.constraints))
            constraint = genome.constraints[idx]
            
            if constraint["type"] == "within_distance" and "distance" in constraint:
                # Modify distance
                delta = random.randint(-10, 10)
                constraint["distance"] = max(1, constraint["distance"] + delta)
                
            elif constraint["type"] == "ordered" and "strict" in constraint:
                # Toggle strict flag
                constraint["strict"] = not constraint["strict"]
                
            # Update the constraint
            genome.constraints[idx] = constraint
            
        return genome
    
async def _evolve_island(
        self,
        island: StrategyIsland,
        generation: int,
        selection_pressure: float,
        crossover_probability: float,
        mutation_probability: float
    ) -> None:
        """
        Evolve a single island population for one generation.
        
        Args:
            island: Island to evolve
            generation: Current generation number
            selection_pressure: Current selection pressure
            crossover_probability: Probability of crossover
            mutation_probability: Probability of mutation
        """
        # Get current population
        population = island.population
        
        # Create next generation
        next_generation = Population[StrategyGenome, MultiFitness]()
        
        # Elitism - preserve the best individuals
        elitism_count = max(1, int(len(population.individuals) * self.elitism_rate))
        elites = population.get_fittest(elitism_count)
        
        for elite in elites:
            # Clone elite to avoid modifying original
            elite_clone = Individual(
                genome=elite.genome,
                generation=elite.generation,
                parent_ids=elite.parent_ids,
                id=str(uuid.uuid4())
            )
            elite_clone.fitness = elite.fitness
            elite_clone.metadata = elite.metadata.copy()
            
            next_generation.add_individual(elite_clone)
            
        # Generate children until the next generation is filled
        island_size = len(population.individuals)
        
        # Use fixed tournament size for deterministic selection pressure
        base_tournament_size = 3
        
        # Apply selection pressure to adjust tournament size
        tournament_size = max(2, min(len(population.individuals) - 1, 
                                     int(base_tournament_size * selection_pressure)))
        
        # Generate children
        while len(next_generation.individuals) < island_size:
            # Use tournament selection with adaptive pressure
            parent1 = self._tournament_selection(population, tournament_size)
            parent2 = self._tournament_selection(population, tournament_size)
            
            # Skip duplicate parents if possible
            retry_count = 0
            while parent1.id == parent2.id and retry_count < 3 and len(population.individuals) > 1:
                parent2 = self._tournament_selection(population, tournament_size)
                retry_count += 1
            
            # Crossover
            crossover_start = time.time()
            
            if random.random() < crossover_probability:
                child = await self.crossover(parent1, parent2)
            else:
                # No crossover, clone parent1
                child = Individual(
                    genome=parent1.genome,
                    generation=generation,
                    parent_ids=[parent1.id],
                    id=str(uuid.uuid4())
                )
                child.metadata = parent1.metadata.copy()
                
            self._crossover_times.append(time.time() - crossover_start)
            
            # Mutation
            mutation_start = time.time()
            
            if random.random() < mutation_probability:
                child = await self.mutate(child)
                
            self._mutation_times.append(time.time() - mutation_start)
            
            # Evaluate child if not already evaluated
            if child.fitness is None:
                evaluation_start = time.time()
                child.fitness = await self.evaluate_fitness(child)
                self._evaluation_times.append(time.time() - evaluation_start)
            
            # Add to next generation
            next_generation.add_individual(child)
            
        # Check if some diversity preservation is needed
        if self.diversity_preservation and random.random() < 0.3:
            # Add some diversity
            self._add_diversity(next_generation, island.focus.get("specialization", "general"))
        
        # Replace current population with next generation
        island.population = next_generation
    
        def _tournament_selection(
            self, 
            population: Population[StrategyGenome, MultiFitness], 
            tournament_size: int
        ) -> Individual[StrategyGenome, MultiFitness]:
            """
            Tournament selection with specified tournament size.
            
            Args:
                population: Population to select from
                tournament_size: Tournament size
                
            Returns:
                Selected individual
            """
            # Adjust tournament size if needed
            tournament_size = min(tournament_size, len(population.individuals))
            
            # Select random individuals for tournament
            tournament = random.sample(population.individuals, tournament_size)
            
            # Return the best
            return max(tournament, key=lambda i: i.fitness.scalar_value if hasattr(i.fitness, "scalar_value") else (float(i.fitness) if i.fitness else float('-inf')))
            
        def _add_diversity(
            self, 
            population: Population[StrategyGenome, MultiFitness],
            specialization: str
        ) -> None:
            """
            Add diversity to the population by replacing least fit individuals.
            
            Args:
                population: Population to diversify
                specialization: Specialization focus
            """
            # Sort population by fitness
            population.sort_by_fitness()
            
            # Replace bottom 10% with diverse individuals
            replace_count = max(1, int(len(population.individuals) * 0.1))
            
            # Remove least fit individuals
            for _ in range(replace_count):
                if population.individuals:
                    population.individuals.pop(0)  # Remove least fit
            
            # Generate new diverse individuals
            for _ in range(replace_count):
                # Generate a completely fresh individual
                async def generate_new():
                    genome = await self._generate_random_genome(specialization=specialization)
                    
                    individual = Individual[StrategyGenome, MultiFitness](
                        genome=genome,
                        generation=0
                    )
                    
                    individual.metadata["specialization"] = specialization
                    
                    # Use pattern bank with some probability
                    if self._pattern_bank and random.random() < 0.5:
                        bank_pattern, _ = random.choice(self._pattern_bank)
                        
                        if genome.patterns and random.random() < 0.5:
                            # Replace a random pattern
                            idx = random.randrange(len(genome.patterns))
                            genome.patterns[idx] = self._deep_copy_node(bank_pattern)
                        elif len(genome.patterns) < self.max_patterns:
                            # Add to patterns
                            genome.patterns.append(self._deep_copy_node(bank_pattern))
                    
                    # Evaluate fitness
                    individual.fitness = await self.evaluate_fitness(individual)
                    
                    return individual
                
                # Use asyncio.run in a thread to avoid blocking
                new_individual = asyncio.run(generate_new())
                
                # Add to population
                population.add_individual(new_individual)
                
        async def _perform_migration(self) -> None:
            """
            Perform migration between islands.
            
            Implements island model migration where individuals move between islands
            to maintain diversity and share beneficial traits.
            """
            if len(self.islands) <= 1:
                return
                
            # Calculate number of migrants per island
            migrants_per_island = max(1, int(self.migration_rate * min(
                len(island.population.individuals) for island in self.islands
            )))
            
            # Select emigrants from each island
            all_emigrants = []
            for island in self.islands:
                emigrants = island.get_emigrants(migrants_per_island)
                
                # Tag with source island
                for emigrant in emigrants:
                    emigrant.metadata["source_island"] = island.id
                    
                all_emigrants.append(emigrants)
                
            # Assign emigrants to destination islands
            for i, source_island in enumerate(self.islands):
                # Select destination islands (all except source)
                destination_indices = [j for j in range(len(self.islands)) if j != i]
                
                # Get emigrants from this island
                emigrants = all_emigrants[i]
                
                # Distribute emigrants among destinations
                for j, emigrant in enumerate(emigrants):
                    dest_idx = destination_indices[j % len(destination_indices)]
                    dest_island = self.islands[dest_idx]
                    
                    # Create clone of emigrant to avoid modifying original
                    clone = Individual(
                        genome=emigrant.genome,
                        generation=emigrant.generation,
                        parent_ids=emigrant.parent_ids,
                        id=str(uuid.uuid4())
                    )
                    clone.fitness = emigrant.fitness
                    clone.metadata = emigrant.metadata.copy()
                    
                    # Send to destination island
                    dest_island.receive_immigrants([clone])
                    
        async def _reinitialize_island(self, island: StrategyIsland) -> None:
            """
            Reinitialize an island with a fresh population while preserving specialization.
            
            Args:
                island: Island to reinitialize
            """
            # Get island specialization
            specialization = island.focus.get("specialization", "general")
            
            # Get island size
            island_size = len(island.population.individuals) if island.population.individuals else 10
            
            # Create new population
            new_population = Population[StrategyGenome, MultiFitness]()
            
            # Generate new individuals
            tasks = []
            for _ in range(island_size):
                task = asyncio.create_task(self._generate_random_genome(specialization=specialization))
                tasks.append(task)
                
            genomes = await asyncio.gather(*tasks)
            
            # Create individuals
            for genome in genomes:
                individual = Individual[StrategyGenome, MultiFitness](
                    genome=genome,
                    generation=0
                )
                
                # Tag with island metadata
                individual.metadata["island_id"] = island.id
                individual.metadata["specialization"] = specialization
                
                # Evaluate fitness
                individual.fitness = await self.evaluate_fitness(individual)
                
                # Add to population
                new_population.add_individual(individual)
                
            # Replace island population
            island.population = new_population
                
        def _calculate_population_diversity(self, population: Population[StrategyGenome, MultiFitness]) -> float:
            """
            Calculate population diversity based on genome differences.
            
            Args:
                population: Population to analyze
                
            Returns:
                Diversity measure between 0 (identical) and 1 (completely diverse)
            """
            if len(population.individuals) < 2:
                return 0.0
                
            # For efficiency, sample pairs when population is large
            if len(population.individuals) > 20:
                # Sample random pairs
                num_pairs = 100
                total_diff = 0.0
                
                for _ in range(num_pairs):
                    ind1, ind2 = random.sample(population.individuals, 2)
                    total_diff += genome_difference(ind1.genome, ind2.genome)
                    
                return total_diff / num_pairs
            else:
                # Calculate all pairwise differences
                total_diff = 0.0
                num_pairs = 0
                
                for i in range(len(population.individuals)):
                    for j in range(i + 1, len(population.individuals)):
                        ind1 = population.individuals[i]
                        ind2 = population.individuals[j]
                        
                        total_diff += genome_difference(ind1.genome, ind2.genome)
                        num_pairs += 1
                        
                return total_diff / num_pairs if num_pairs > 0 else 0.0