"""
Pattern analyzer for TSAP's evolution system.

This module provides functionality to analyze and optimize search patterns,
learning from previous results to improve future searches.
"""
import os
import re
import asyncio
import json
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from dataclasses import dataclass, field
import time
import datetime
import statistics
from collections import Counter, defaultdict

from tsap.utils.logging import logger
from tsap.core.ripgrep import ripgrep_search
from tsap.composite.parallel import parallel_search
from tsap.mcp.models import SearchPattern, ParallelSearchParams, RipgrepSearchParams


@dataclass
class PatternStats:
    """Statistics for a search pattern."""
    
    pattern: str
    total_matches: int = 0
    files_with_matches: int = 0
    match_locations: List[Tuple[str, int]] = field(default_factory=list)
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 1.0
    recall: float = 1.0
    f1_score: float = 1.0
    avg_context_relevance: float = 1.0
    execution_time: float = 0.0
    
    def calculate_metrics(self, total_relevant: int = 0):
        """Calculate precision, recall, and F1 score.
        
        Args:
            total_relevant: Total number of relevant items
        """
        true_positives = self.total_matches - self.false_positives
        
        # Calculate precision
        if self.total_matches > 0:
            self.precision = true_positives / self.total_matches
        else:
            self.precision = 1.0  # By convention, precision is 1.0 when no matches
        
        # Calculate recall
        if total_relevant > 0:
            self.recall = true_positives / total_relevant
        else:
            self.recall = 1.0  # By convention, recall is 1.0 when no relevant items
        
        # Calculate F1 score
        if self.precision > 0 or self.recall > 0:
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        else:
            self.f1_score = 0.0


@dataclass
class PatternVariant:
    """A variant of a search pattern with its performance metrics."""
    
    pattern: str
    description: str
    is_regex: bool = True
    case_sensitive: bool = False
    stats: PatternStats = None
    generation: int = 0
    parent_pattern: Optional[str] = None
    
    def __post_init__(self):
        if not self.stats:
            self.stats = PatternStats(pattern=self.pattern)
    
    @property
    def score(self) -> float:
        """Calculate an overall score for this pattern variant.
        
        Returns:
            Score value (higher is better)
        """
        # Weighted combination of metrics
        # You can adjust weights based on your priorities
        weights = {
            "precision": 0.4,
            "recall": 0.4,
            "f1": 0.2,
        }
        
        return (
            weights["precision"] * self.stats.precision +
            weights["recall"] * self.stats.recall +
            weights["f1"] * self.stats.f1_score
        )


class PatternAnalyzer:
    """Analyzer for search patterns with evolutionary capabilities."""
    
    def __init__(self):
        """Initialize the pattern analyzer."""
        self.pattern_variants = {}  # Maps pattern ID to list of variants
        self.pattern_stats = {}  # Maps pattern to PatternStats
        self.pattern_history = {}  # Maps pattern ID to history of variants
    
    @staticmethod
    def _extract_pattern_components(pattern: str) -> List[str]:
        """Extract components from a regex pattern.
        
        Args:
            pattern: Regex pattern
            
        Returns:
            List of pattern components
        """
        # Simple component extraction for common regex constructs
        components = []
        
        # Extract literals (text not in special regex constructs)
        literal_parts = re.split(r'[\\[\](){}?*+^$.|]', pattern)
        components.extend([p for p in literal_parts if p])
        
        # Extract character classes
        char_classes = re.findall(r'\[[^\]]+\]', pattern)
        components.extend(char_classes)
        
        # Extract groups
        groups = re.findall(r'\([^)]+\)', pattern)
        components.extend(groups)
        
        return components
    
    @staticmethod
    def _generate_simpler_pattern(pattern: str) -> str:
        """Generate a simpler version of a pattern.
        
        Args:
            pattern: Original pattern
            
        Returns:
            Simplified pattern
        """
        # Replace complex regex constructs with simpler alternatives
        simplified = pattern
        
        # Replace character classes with wildcards
        simplified = re.sub(r'\[[^\]]+\]', '.', simplified)
        
        # Replace non-capturing groups with their content
        simplified = re.sub(r'\(\?:[^)]+\)', '.+', simplified)
        
        # Replace capturing groups with their content
        simplified = re.sub(r'\([^)]+\)', '.+', simplified)
        
        # Replace complex quantifiers
        simplified = re.sub(r'\{\d+,\d+\}', '+', simplified)
        simplified = re.sub(r'\{\d+\}', '+', simplified)
        
        # Replace word boundaries with spaces
        simplified = simplified.replace('\\b', ' ')
        
        # Remove anchors
        simplified = simplified.replace('^', '')
        simplified = simplified.replace('$', '')
        
        return simplified
    
    @staticmethod
    def _generate_more_specific_pattern(pattern: str) -> str:
        """Generate a more specific version of a pattern.
        
        Args:
            pattern: Original pattern
            
        Returns:
            More specific pattern
        """
        # Add more constraints to make the pattern more specific
        specific = pattern
        
        # Add word boundaries around words
        if '\\b' not in specific:
            words = re.findall(r'[a-zA-Z0-9_]+', specific)
            for word in words:
                if len(word) > 3:  # Only add boundaries to longer words
                    specific = specific.replace(word, f'\\b{word}\\b')
        
        # Replace wildcards with more specific character classes
        specific = specific.replace('.', '[a-zA-Z0-9_]')
        
        # Add start/end anchors if not present
        if not specific.startswith('^'):
            specific = '^' + specific
        if not specific.endswith('$'):
            specific = specific + '$'
        
        return specific
    
    @staticmethod
    def _generate_pattern_variations(pattern: str, is_regex: bool) -> List[str]:
        """Generate variations of a pattern.
        
        Args:
            pattern: Original pattern
            is_regex: Whether the pattern is a regex
            
        Returns:
            List of pattern variations
        """
        variations = []
        
        if is_regex:
            # For regex patterns, generate variations
            # Simpler version
            variations.append(PatternAnalyzer._generate_simpler_pattern(pattern))
            
            # More specific version
            variations.append(PatternAnalyzer._generate_more_specific_pattern(pattern))
            
            # Replace quantifiers with alternatives
            variations.append(pattern.replace('+', '*'))
            variations.append(pattern.replace('*', '+'))
            
            # Add/remove word boundaries
            if '\\b' in pattern:
                variations.append(pattern.replace('\\b', ''))
            else:
                words = re.findall(r'[a-zA-Z0-9_]+', pattern)
                for word in words:
                    if len(word) > 3:  # Only add boundaries to longer words
                        variations.append(pattern.replace(word, f'\\b{word}\\b'))
        else:
            # For literal patterns, generate variations
            # Add common misspellings or variations
            words = pattern.split()
            for i, word in enumerate(words):
                if len(word) > 3:
                    # Add a variation with one character changed
                    for j in range(len(word)):
                        varied_word = word[:j] + '.' + word[j+1:]
                        varied_words = words.copy()
                        varied_words[i] = varied_word
                        variations.append(' '.join(varied_words))
        
        # Remove duplicates and the original pattern
        return [v for v in variations if v != pattern]
    
    @staticmethod
    def _mutate_pattern(
        pattern: str, 
        is_regex: bool, 
        mutation_strength: float = 0.2
    ) -> str:
        """Mutate a pattern to create a new variant.
        
        Args:
            pattern: Original pattern
            is_regex: Whether the pattern is a regex
            mutation_strength: Strength of mutation (0.0-1.0)
            
        Returns:
            Mutated pattern
        """
        if mutation_strength <= 0:
            return pattern
            
        # Randomly apply mutations based on mutation_strength
        mutated = pattern
        
        # For regex patterns
        if is_regex:
            # Possible mutations
            mutations = [
                # Replace a dot with a character class
                (r'\.', lambda m: '[a-zA-Z0-9_]'),
                # Replace a character class with a dot
                (r'\[[^\]]+\]', lambda m: '.'),
                # Add/remove word boundary
                (r'\\b', lambda m: ''),
                (r'[a-zA-Z0-9_]+', lambda m: f'\\b{m.group(0)}\\b'),
                # Modify quantifiers
                (r'\+', lambda m: '*'),
                (r'\*', lambda m: '+'),
                (r'\{\d+\}', lambda m: '+'),
                # Add/remove start/end anchors
                (r'^\^', lambda m: ''),
                (r'\$$', lambda m: ''),
            ]
            
            # Apply mutations with probability based on mutation_strength
            import random
            for pattern, replacement in mutations:
                if random.random() < mutation_strength:
                    # Find all matches
                    matches = list(re.finditer(pattern, mutated))
                    if matches:
                        # Select a random match
                        match = random.choice(matches)
                        # Apply the replacement
                        mutated = (
                            mutated[:match.start()] + 
                            replacement(match) + 
                            mutated[match.end():]
                        )
        else:
            # For literal patterns, make small changes
            import random
            
            # Possible mutations
            if random.random() < mutation_strength:
                # Change a random character
                if len(mutated) > 0:
                    pos = random.randint(0, len(mutated) - 1)
                    mutated = mutated[:pos] + '.' + mutated[pos+1:]
        
        return mutated
    
    @staticmethod
    def _crossover_patterns(
        pattern1: str, 
        pattern2: str, 
        is_regex: bool
    ) -> str:
        """Create a new pattern by combining two patterns.
        
        Args:
            pattern1: First pattern
            pattern2: Second pattern
            is_regex: Whether the patterns are regexes
            
        Returns:
            Combined pattern
        """
        if is_regex:
            # For regex patterns, try to combine meaningful parts
            components1 = PatternAnalyzer._extract_pattern_components(pattern1)
            components2 = PatternAnalyzer._extract_pattern_components(pattern2)
            
            # Combine components from both patterns
            import random
            
            # Select components from each pattern
            selected = []
            for comp in components1:
                if random.random() < 0.5:
                    selected.append(comp)
            for comp in components2:
                if random.random() < 0.5:
                    selected.append(comp)
                    
            # Ensure we have at least one component
            if not selected and components1 and components2:
                selected = [random.choice(components1), random.choice(components2)]
            elif not selected and components1:
                selected = [random.choice(components1)]
            elif not selected and components2:
                selected = [random.choice(components2)]
                
            # Join components with a reasonable separator
            if all(re.match(r'^[a-zA-Z0-9_]+$', comp) for comp in selected):
                # For word components, join with spaces or pipes
                if random.random() < 0.5:
                    return ' '.join(selected)
                else:
                    return '|'.join(selected)
            else:
                # For regex components, join with a regex alternation
                return '|'.join(selected)
        else:
            # For literal patterns, combine words
            words1 = pattern1.split()
            words2 = pattern2.split()
            
            # Take words from both patterns
            import random
            combined = []
            
            # Select words from each pattern
            for word in words1:
                if random.random() < 0.5:
                    combined.append(word)
            for word in words2:
                if random.random() < 0.5:
                    combined.append(word)
                    
            # Ensure we have at least one word
            if not combined:
                if words1 and words2:
                    combined = [random.choice(words1), random.choice(words2)]
                elif words1:
                    combined = [random.choice(words1)]
                elif words2:
                    combined = [random.choice(words2)]
                    
            # Join the words
            return ' '.join(combined)
    
    async def evaluate_pattern(
        self,
        pattern: str,
        is_regex: bool,
        case_sensitive: bool,
        paths: List[str],
        reference_set: Optional[List[Tuple[str, int]]] = None
    ) -> PatternStats:
        """Evaluate a pattern against a set of files.
        
        Args:
            pattern: Pattern to evaluate
            is_regex: Whether the pattern is a regex
            case_sensitive: Whether the pattern is case-sensitive
            paths: Paths to search
            reference_set: Optional reference set of (file, line) locations
            
        Returns:
            Statistics for the pattern
        """
        # Create search parameters
        params = RipgrepSearchParams(
            pattern=pattern,
            paths=paths,
            regex=is_regex,
            case_sensitive=case_sensitive,
            context_lines=0,
        )
        
        start_time = time.time()
        
        # Execute the search
        try:
            result = await ripgrep_search(params)
        except Exception as e:
            logger.error(
                f"Pattern evaluation failed: {str(e)}",
                component="evolution",
                operation="evaluate_pattern",
                exception=e,
                context={"pattern": pattern}
            )
            
            # Return empty stats
            return PatternStats(
                pattern=pattern,
                execution_time=time.time() - start_time
            )
        
        execution_time = time.time() - start_time
        
        # Extract match locations
        matches = result.matches
        match_locations = [(m.path, m.line_number) for m in matches]
        
        # Count unique files with matches
        files_with_matches = len(set(m.path for m in matches))
        
        # Calculate false positives and false negatives if reference set is provided
        false_positives = 0
        false_negatives = 0
        
        if reference_set:
            reference_locations = set(reference_set)
            result_locations = set(match_locations)
            
            # False positives: in results but not in reference
            false_positives = len(result_locations - reference_locations)
            
            # False negatives: in reference but not in results
            false_negatives = len(reference_locations - result_locations)
        
        # Create pattern stats
        stats = PatternStats(
            pattern=pattern,
            total_matches=len(matches),
            files_with_matches=files_with_matches,
            match_locations=match_locations,
            false_positives=false_positives,
            false_negatives=false_negatives,
            execution_time=execution_time,
        )
        
        # Calculate metrics
        if reference_set:
            stats.calculate_metrics(len(reference_set))
        
        return stats
    
    async def analyze_pattern(
        self,
        pattern: str,
        description: str,
        is_regex: bool,
        case_sensitive: bool,
        paths: List[str],
        reference_set: Optional[List[Tuple[str, int]]] = None,
        generate_variants: bool = True,
        num_variants: int = 3,
    ) -> Dict[str, Any]:
        """Analyze a pattern and optionally generate variations.
        
        Args:
            pattern: Pattern to analyze
            description: Description of the pattern
            is_regex: Whether the pattern is a regex
            case_sensitive: Whether the pattern is case-sensitive
            paths: Paths to search
            reference_set: Optional reference set of (file, line) locations
            generate_variants: Whether to generate variants
            num_variants: Number of variants to generate
            
        Returns:
            Analysis results
        """
        # Create a pattern ID
        import hashlib
        pattern_id = hashlib.md5(pattern.encode()).hexdigest()
        
        # Log the operation
        logger.info(
            f"Analyzing pattern: {pattern}",
            component="evolution",
            operation="analyze_pattern",
            context={
                "pattern_id": pattern_id,
                "pattern": pattern,
                "is_regex": is_regex,
                "generate_variants": generate_variants,
            }
        )
        
        # Evaluate the pattern
        stats = await self.evaluate_pattern(
            pattern=pattern,
            is_regex=is_regex,
            case_sensitive=case_sensitive,
            paths=paths,
            reference_set=reference_set,
        )
        
        # Store the stats
        self.pattern_stats[pattern] = stats
        
        # Initialize variant list if needed
        if pattern_id not in self.pattern_variants:
            self.pattern_variants[pattern_id] = []
            self.pattern_history[pattern_id] = []
        
        # Create the base variant
        base_variant = PatternVariant(
            pattern=pattern,
            description=description,
            is_regex=is_regex,
            case_sensitive=case_sensitive,
            stats=stats,
        )
        
        # Add to variants and history
        self.pattern_variants[pattern_id].append(base_variant)
        self.pattern_history[pattern_id].append(base_variant)
        
        # Generate variants if requested
        variant_results = []
        
        if generate_variants and num_variants > 0:
            # Generate variations of the pattern
            variations = self._generate_pattern_variations(pattern, is_regex)
            
            # Evaluate each variation
            for i, var_pattern in enumerate(variations[:num_variants]):
                # Skip exact duplicates
                if var_pattern == pattern:
                    continue
                    
                # Generate a description for the variant
                var_description = f"Variant {i+1} of '{description}'"
                
                # Evaluate the variant
                var_stats = await self.evaluate_pattern(
                    pattern=var_pattern,
                    is_regex=is_regex,
                    case_sensitive=case_sensitive,
                    paths=paths,
                    reference_set=reference_set,
                )
                
                # Create variant object
                variant = PatternVariant(
                    pattern=var_pattern,
                    description=var_description,
                    is_regex=is_regex,
                    case_sensitive=case_sensitive,
                    stats=var_stats,
                    parent_pattern=pattern,
                )
                
                # Add to variants and history
                self.pattern_variants[pattern_id].append(variant)
                self.pattern_history[pattern_id].append(variant)
                
                # Add to results
                variant_results.append({
                    "pattern": var_pattern,
                    "description": var_description,
                    "is_regex": is_regex,
                    "case_sensitive": case_sensitive,
                    "stats": {
                        "total_matches": var_stats.total_matches,
                        "files_with_matches": var_stats.files_with_matches,
                        "precision": var_stats.precision,
                        "recall": var_stats.recall,
                        "f1_score": var_stats.f1_score,
                        "execution_time": var_stats.execution_time,
                    },
                    "score": variant.score,
                })
        
        # Sort variants by score
        self.pattern_variants[pattern_id].sort(key=lambda v: v.score, reverse=True)
        
        # Create result
        result = {
            "pattern_id": pattern_id,
            "original_pattern": pattern,
            "description": description,
            "is_regex": is_regex,
            "case_sensitive": case_sensitive,
            "stats": {
                "total_matches": stats.total_matches,
                "files_with_matches": stats.files_with_matches,
                "precision": stats.precision,
                "recall": stats.recall,
                "f1_score": stats.f1_score,
                "execution_time": stats.execution_time,
            },
            "variants": variant_results,
            "best_pattern": self.pattern_variants[pattern_id][0].pattern
            if self.pattern_variants[pattern_id] else pattern,
        }
        
        # Log completion
        logger.success(
            f"Pattern analysis completed: {pattern}",
            component="evolution",
            operation="analyze_pattern",
            context={
                "pattern_id": pattern_id,
                "matches": stats.total_matches,
                "variant_count": len(variant_results),
            }
        )
        
        return result
    
    async def evolve_pattern(
        self,
        pattern_id: str,
        paths: List[str],
        reference_set: Optional[List[Tuple[str, int]]] = None,
        generations: int = 3,
        population_size: int = 5,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.3,
    ) -> Dict[str, Any]:
        """Evolve a pattern through multiple generations.
        
        Args:
            pattern_id: ID of the pattern to evolve
            paths: Paths to search
            reference_set: Optional reference set of (file, line) locations
            generations: Number of generations to evolve
            population_size: Size of each generation
            mutation_rate: Rate of mutation (0.0-1.0)
            crossover_rate: Rate of crossover (0.0-1.0)
            
        Returns:
            Evolution results
        """
        # Check if pattern exists
        if pattern_id not in self.pattern_variants:
            raise ValueError(f"Pattern ID not found: {pattern_id}")
            
        # Get the current variants
        variants = self.pattern_variants[pattern_id]
        
        if not variants:
            raise ValueError(f"No variants found for pattern ID: {pattern_id}")
            
        # Log the operation
        logger.info(
            f"Evolving pattern: {variants[0].pattern}",
            component="evolution",
            operation="evolve_pattern",
            context={
                "pattern_id": pattern_id,
                "generations": generations,
                "population_size": population_size,
            }
        )
        
        # Track evolution history
        evolution_history = []
        
        # Initial population is the current variants
        population = variants.copy()
        
        # Ensure we have enough variants for the initial population
        while len(population) < population_size:
            # Clone random variants with mutations
            import random
            parent = random.choice(variants)
            
            # Create a mutated variant
            mutated_pattern = self._mutate_pattern(
                pattern=parent.pattern,
                is_regex=parent.is_regex,
                mutation_strength=mutation_rate,
            )
            
            # Skip exact duplicates
            if any(v.pattern == mutated_pattern for v in population):
                continue
                
            # Evaluate the new variant
            stats = await self.evaluate_pattern(
                pattern=mutated_pattern,
                is_regex=parent.is_regex,
                case_sensitive=parent.case_sensitive,
                paths=paths,
                reference_set=reference_set,
            )
            
            # Create variant object
            variant = PatternVariant(
                pattern=mutated_pattern,
                description=f"Mutation of '{parent.description}'",
                is_regex=parent.is_regex,
                case_sensitive=parent.case_sensitive,
                stats=stats,
                generation=1,
                parent_pattern=parent.pattern,
            )
            
            # Add to population
            population.append(variant)
            
            # Add to history
            self.pattern_history[pattern_id].append(variant)
        
        # Evolve through generations
        for gen in range(1, generations + 1):
            # Record the current generation
            gen_stats = {
                "generation": gen,
                "population": [
                    {
                        "pattern": v.pattern,
                        "description": v.description,
                        "score": v.score,
                        "matches": v.stats.total_matches,
                        "precision": v.stats.precision,
                        "recall": v.stats.recall,
                        "f1_score": v.stats.f1_score,
                    }
                    for v in population
                ],
                "best_pattern": max(population, key=lambda v: v.score).pattern,
                "best_score": max(population, key=lambda v: v.score).score,
                "avg_score": statistics.mean(v.score for v in population),
            }
            evolution_history.append(gen_stats)
            
            # Log generation progress
            logger.info(
                f"Evolution generation {gen}: {len(population)} variants",
                component="evolution",
                operation="evolve_pattern",
                context={
                    "generation": gen,
                    "best_score": gen_stats["best_score"],
                    "avg_score": gen_stats["avg_score"],
                }
            )
            
            # Generate new population through selection, crossover, and mutation
            new_population = []
            
            # Keep the best variants (elitism)
            elite_count = max(1, population_size // 5)
            elites = sorted(population, key=lambda v: v.score, reverse=True)[:elite_count]
            new_population.extend(elites)
            
            # Fill the rest of the population
            while len(new_population) < population_size:
                # Select parents through tournament selection
                import random
                
                tournament_size = min(3, len(population))
                tournament = random.sample(population, tournament_size)
                parent1 = max(tournament, key=lambda v: v.score)
                
                tournament = random.sample(population, tournament_size)
                parent2 = max(tournament, key=lambda v: v.score)
                
                # Determine if we do crossover
                if random.random() < crossover_rate and parent1.pattern != parent2.pattern:
                    # Create a crossover variant
                    child_pattern = self._crossover_patterns(
                        pattern1=parent1.pattern,
                        pattern2=parent2.pattern,
                        is_regex=parent1.is_regex,
                    )
                    
                    # Determine if we also do mutation
                    if random.random() < mutation_rate:
                        child_pattern = self._mutate_pattern(
                            pattern=child_pattern,
                            is_regex=parent1.is_regex,
                            mutation_strength=mutation_rate,
                        )
                    
                    # Skip exact duplicates
                    if any(v.pattern == child_pattern for v in new_population + population):
                        continue
                        
                    # Evaluate the new variant
                    stats = await self.evaluate_pattern(
                        pattern=child_pattern,
                        is_regex=parent1.is_regex,
                        case_sensitive=parent1.case_sensitive,
                        paths=paths,
                        reference_set=reference_set,
                    )
                    
                    # Create variant object
                    variant = PatternVariant(
                        pattern=child_pattern,
                        description=f"Crossover of '{parent1.description}' and '{parent2.description}'",
                        is_regex=parent1.is_regex,
                        case_sensitive=parent1.case_sensitive,
                        stats=stats,
                        generation=gen + 1,
                        parent_pattern=f"{parent1.pattern} + {parent2.pattern}",
                    )
                    
                    # Add to new population
                    new_population.append(variant)
                    
                    # Add to history
                    self.pattern_history[pattern_id].append(variant)
                else:
                    # Just do mutation
                    parent = parent1
                    
                    # Create a mutated variant
                    mutated_pattern = self._mutate_pattern(
                        pattern=parent.pattern,
                        is_regex=parent.is_regex,
                        mutation_strength=mutation_rate,
                    )
                    
                    # Skip exact duplicates
                    if any(v.pattern == mutated_pattern for v in new_population + population):
                        continue
                        
                    # Evaluate the new variant
                    stats = await self.evaluate_pattern(
                        pattern=mutated_pattern,
                        is_regex=parent.is_regex,
                        case_sensitive=parent.case_sensitive,
                        paths=paths,
                        reference_set=reference_set,
                    )
                    
                    # Create variant object
                    variant = PatternVariant(
                        pattern=mutated_pattern,
                        description=f"Mutation of '{parent.description}'",
                        is_regex=parent.is_regex,
                        case_sensitive=parent.case_sensitive,
                        stats=stats,
                        generation=gen + 1,
                        parent_pattern=parent.pattern,
                    )
                    
                    # Add to new population
                    new_population.append(variant)
                    
                    # Add to history
                    self.pattern_history[pattern_id].append(variant)
            
            # Set the new population
            population = new_population
        
        # Final generation stats
        final_gen_stats = {
            "generation": generations,
            "population": [
                {
                    "pattern": v.pattern,
                    "description": v.description,
                    "score": v.score,
                    "matches": v.stats.total_matches,
                    "precision": v.stats.precision,
                    "recall": v.stats.recall,
                    "f1_score": v.stats.f1_score,
                }
                for v in population
            ],
            "best_pattern": max(population, key=lambda v: v.score).pattern,
            "best_score": max(population, key=lambda v: v.score).score,
            "avg_score": statistics.mean(v.score for v in population),
        }
        evolution_history.append(final_gen_stats)
        
        # Find the best overall pattern
        best_variant = max(population, key=lambda v: v.score)
        
        # Update the pattern variants
        self.pattern_variants[pattern_id] = sorted(
            population, key=lambda v: v.score, reverse=True
        )
        
        # Calculate improvement
        original_variant = variants[0]
        improvement = (best_variant.score - original_variant.score) / original_variant.score
        improvement_percentage = improvement * 100
        
        # Log completion
        logger.success(
            f"Pattern evolution completed: {best_variant.pattern}",
            component="evolution",
            operation="evolve_pattern",
            context={
                "pattern_id": pattern_id,
                "generations": generations,
                "best_score": best_variant.score,
                "improvement": f"{improvement_percentage:.2f}%",
            }
        )
        
        # Create result
        result = {
            "pattern_id": pattern_id,
            "original_pattern": original_variant.pattern,
            "best_pattern": best_variant.pattern,
            "improvement_percentage": improvement_percentage,
            "generations": generations,
            "population_size": population_size,
            "history": evolution_history,
            "final_population": [
                {
                    "pattern": v.pattern,
                    "description": v.description,
                    "score": v.score,
                    "stats": {
                        "total_matches": v.stats.total_matches,
                        "files_with_matches": v.stats.files_with_matches,
                        "precision": v.stats.precision,
                        "recall": v.stats.recall,
                        "f1_score": v.stats.f1_score,
                    },
                }
                for v in self.pattern_variants[pattern_id]
            ],
        }
        
        return result
    
    def get_pattern_history(self, pattern_id: str) -> List[Dict[str, Any]]:
        """Get the history of a pattern's evolution.
        
        Args:
            pattern_id: Pattern ID
            
        Returns:
            Pattern history
        """
        if pattern_id not in self.pattern_history:
            return []
            
        return [
            {
                "pattern": v.pattern,
                "description": v.description,
                "score": v.score,
                "generation": v.generation,
                "parent_pattern": v.parent_pattern,
                "stats": {
                    "total_matches": v.stats.total_matches,
                    "files_with_matches": v.stats.files_with_matches,
                    "precision": v.stats.precision,
                    "recall": v.stats.recall,
                    "f1_score": v.stats.f1_score,
                },
            }
            for v in self.pattern_history[pattern_id]
        ]


# Create a singleton instance
_pattern_analyzer = None


def get_pattern_analyzer() -> PatternAnalyzer:
    """Get the singleton PatternAnalyzer instance.
    
    Returns:
        PatternAnalyzer instance
    """
    global _pattern_analyzer
    
    if _pattern_analyzer is None:
        _pattern_analyzer = PatternAnalyzer()
        
    return _pattern_analyzer


async def analyze_pattern(
    pattern: str,
    description: str,
    is_regex: bool,
    case_sensitive: bool,
    paths: List[str],
    reference_set: Optional[List[Tuple[str, int]]] = None,
    generate_variants: bool = True,
    num_variants: int = 3,
) -> Dict[str, Any]:
    """Analyze a pattern and optionally generate variations.
    
    This is a convenience function that uses the singleton PatternAnalyzer.
    
    Args:
        pattern: Pattern to analyze
        description: Description of the pattern
        is_regex: Whether the pattern is a regex
        case_sensitive: Whether the pattern is case-sensitive
        paths: Paths to search
        reference_set: Optional reference set of (file, line) locations
        generate_variants: Whether to generate variants
        num_variants: Number of variants to generate
        
    Returns:
        Analysis results
    """
    analyzer = get_pattern_analyzer()
    
    return await analyzer.analyze_pattern(
        pattern=pattern,
        description=description,
        is_regex=is_regex,
        case_sensitive=case_sensitive,
        paths=paths,
        reference_set=reference_set,
        generate_variants=generate_variants,
        num_variants=num_variants,
    )


async def evolve_pattern(
    pattern_id: str,
    paths: List[str],
    reference_set: Optional[List[Tuple[str, int]]] = None,
    generations: int = 3,
    population_size: int = 5,
    mutation_rate: float = 0.2,
    crossover_rate: float = 0.3,
) -> Dict[str, Any]:
    """Evolve a pattern through multiple generations.
    
    This is a convenience function that uses the singleton PatternAnalyzer.
    
    Args:
        pattern_id: ID of the pattern to evolve
        paths: Paths to search
        reference_set: Optional reference set of (file, line) locations
        generations: Number of generations to evolve
        population_size: Size of each generation
        mutation_rate: Rate of mutation (0.0-1.0)
        crossover_rate: Rate of crossover (0.0-1.0)
        
    Returns:
        Evolution results
    """
    analyzer = get_pattern_analyzer()
    
    return await analyzer.evolve_pattern(
        pattern_id=pattern_id,
        paths=paths,
        reference_set=reference_set,
        generations=generations,
        population_size=population_size,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
    )