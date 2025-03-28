"""
Composite operations for iterative refinement of search results.

This module provides operations that enable progressive filtering and iterative
refinement of search results, allowing for more precise search and analysis through
multiple steps of narrowing the focus based on initial results.
"""

from typing import List, Optional, Union
from dataclasses import dataclass, field

from tsap.utils.logging import logger
from tsap.composite.base import CompositeOperation, register_operation, CompositeError
from tsap.core.ripgrep import ripgrep_search
from tsap.composite.parallel import parallel_search
from tsap.mcp.models import (
    RipgrepSearchParams, RipgrepSearchResult,
    ParallelSearchParams, ParallelSearchResult,
    SearchPattern, RecursiveRefinementParams, RecursiveRefinementResult
)


@dataclass
class RefinementStep:
    """Represents a single step in the refinement process."""
    patterns: List[SearchPattern]
    file_paths: Optional[List[str]] = None
    file_patterns: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None
    context_lines: int = 2
    max_matches_per_file: Optional[int] = None
    max_total_matches: Optional[int] = None
    result: Optional[Union[RipgrepSearchResult, ParallelSearchResult]] = None
    matched_files: List[str] = field(default_factory=list)
    confidence: float = 0.0
    description: Optional[str] = None


class RecursiveRefinementOperation(CompositeOperation[RecursiveRefinementParams, RecursiveRefinementResult]):
    """
    Operation that performs recursive refinement of search results.
    
    This operation starts with a broad search pattern and iteratively narrows
    the scope based on the results of each step, until a desired level of
    precision is achieved or a maximum number of steps is reached.
    """
    
    async def execute(self, params: RecursiveRefinementParams) -> RecursiveRefinementResult:
        """
        Execute the recursive refinement operation.
        
        Args:
            params: Parameters for the operation
            
        Returns:
            Result of the recursive refinement operation
        """
        logger.info(f"Starting recursive refinement with initial patterns: {len(params.initial_patterns)}")
        
        # Validate parameters
        if not params.initial_patterns:
            raise CompositeError(
                message="At least one initial pattern is required for recursive refinement",
                operation="recursive_refinement"
            )
        
        if not params.paths and not params.file_paths:
            raise CompositeError(
                message="Either paths or file_paths must be provided",
                operation="recursive_refinement"
            )
        
        # Initialize the result
        result = RecursiveRefinementResult(
            steps=[],
            final_result=None,
            confidence=0.0,
            matched_files=[],
            execution_time=0.0,
            statistics={}
        )
        
        # Initialize the first step
        current_step = RefinementStep(
            patterns=params.initial_patterns.copy(),
            file_paths=params.file_paths.copy() if params.file_paths else None,
            file_patterns=params.file_patterns.copy() if params.file_patterns else None,
            exclude_patterns=params.exclude_patterns.copy() if params.exclude_patterns else None,
            context_lines=params.context_lines,
            max_matches_per_file=params.max_matches_per_file,
            max_total_matches=params.max_total_matches,
            description="Initial search with broad patterns"
        )
        
        # Execute the initial step
        try:
            await self._execute_step(current_step, params.paths if params.paths else [])
            result.steps.append(current_step)
            result.matched_files.extend(current_step.matched_files)
        except Exception as e:
            logger.error(f"Error executing initial refinement step: {str(e)}")
            raise CompositeError(
                message=f"Error executing initial refinement step: {str(e)}",
                operation="recursive_refinement",
                details={"step": 0, "patterns": [p.dict() for p in current_step.patterns]}
            ) from e
        
        # Early return if no matches or max_steps is 1
        if not current_step.matched_files or params.max_steps <= 1:
            result.final_result = current_step.result
            result.confidence = current_step.confidence
            return result
        
        # Iteratively refine the search
        step_num = 1
        while step_num < params.max_steps:
            # Generate refined patterns based on previous results
            refined_step = await self._generate_refined_step(
                current_step, 
                params.refinement_strategy,
                step_num
            )
            
            if not refined_step:
                logger.info(f"Refinement complete after {step_num} steps - no further refinement possible")
                break
            
            # Execute the refined step
            try:
                await self._execute_step(
                    refined_step, 
                    params.paths if params.paths else [],
                    current_step.matched_files if params.search_only_matched_files else None
                )
                result.steps.append(refined_step)
                
                # Update matched files
                result.matched_files = list(set(result.matched_files) | set(refined_step.matched_files))
                
                # Check if refinement should continue
                if not refined_step.matched_files:
                    logger.info(f"Refinement complete after {step_num + 1} steps - no matches in refined search")
                    break
                
                if refined_step.confidence >= params.min_confidence:
                    logger.info(f"Refinement complete after {step_num + 1} steps - confidence threshold reached")
                    break
                
                # Continue with next step
                current_step = refined_step
                step_num += 1
                
            except Exception as e:
                logger.error(f"Error executing refinement step {step_num}: {str(e)}")
                # Continue with the best results so far instead of failing
                break
        
        # Set the final result and confidence
        result.final_result = current_step.result
        result.confidence = current_step.confidence
        
        return result
    
    async def _execute_step(
        self, 
        step: RefinementStep, 
        paths: List[str],
        file_subset: Optional[List[str]] = None
    ) -> None:
        """
        Execute a single refinement step.
        
        Args:
            step: The refinement step to execute
            paths: Base paths to search in
            file_subset: Optional subset of files to search in
        """
        # Use either the file subset or the full paths
        search_paths = file_subset if file_subset else (step.file_paths if step.file_paths else paths)
        
        # Execute search using parallel search for multiple patterns or ripgrep for a single pattern
        if len(step.patterns) > 1:
            search_params = ParallelSearchParams(
                patterns=step.patterns,
                paths=search_paths,
                file_patterns=step.file_patterns,
                exclude_patterns=step.exclude_patterns,
                context_lines=step.context_lines,
                max_matches_per_file=step.max_matches_per_file,
                max_total_matches=step.max_total_matches
            )
            
            result = await parallel_search(search_params)
            step.result = result
            
            # Extract matched files
            for match in result.matches:
                if match.file_path not in step.matched_files:
                    step.matched_files.append(match.file_path)
            
        else:
            # Single pattern search using ripgrep directly
            pattern = step.patterns[0]
            search_params = RipgrepSearchParams(
                pattern=pattern.pattern,
                paths=search_paths,
                is_regex=pattern.is_regex,
                case_sensitive=pattern.case_sensitive,
                file_patterns=step.file_patterns,
                exclude_patterns=step.exclude_patterns,
                context_lines=step.context_lines,
                max_matches_per_file=step.max_matches_per_file,
                max_total_matches=step.max_total_matches
            )
            
            result = await ripgrep_search(search_params)
            step.result = result
            
            # Extract matched files
            for match in result.matches:
                if match.file_path not in step.matched_files:
                    step.matched_files.append(match.file_path)
        
        # Calculate confidence based on results
        step.confidence = self._calculate_step_confidence(step)
    
    async def _generate_refined_step(
        self, 
        previous_step: RefinementStep,
        strategy: str,
        step_num: int
    ) -> Optional[RefinementStep]:
        """
        Generate a refined step based on the results of the previous step.
        
        Args:
            previous_step: The previous refinement step
            strategy: Strategy to use for refinement
            step_num: Current step number
            
        Returns:
            A new refinement step, or None if no further refinement is possible
        """
        if not previous_step.matched_files:
            return None
        
        if strategy == "narrow_by_context":
            return await self._refine_by_context(previous_step, step_num)
        elif strategy == "narrow_by_frequency":
            return await self._refine_by_frequency(previous_step, step_num)
        elif strategy == "narrow_by_proximity":
            return await self._refine_by_proximity(previous_step, step_num)
        else:
            # Default strategy
            return await self._refine_by_context(previous_step, step_num)
    
    async def _refine_by_context(
        self, 
        previous_step: RefinementStep,
        step_num: int
    ) -> Optional[RefinementStep]:
        """
        Refine search based on context of previous matches.
        
        Args:
            previous_step: The previous refinement step
            step_num: Current step number
            
        Returns:
            A new refinement step with refined patterns
        """
        # This is a placeholder implementation that would need to be expanded
        # based on actual requirements
        
        # For now, just add context terms from the matched lines
        if not previous_step.result:
            return None
        
        # Extract common terms from matched lines
        common_terms = self._extract_common_terms(previous_step)
        if not common_terms:
            return None
        
        # Create refined patterns by combining original patterns with common terms
        refined_patterns = []
        for original_pattern in previous_step.patterns:
            for term in common_terms[:3]:  # Limit to top 3 terms
                refined_patterns.append(SearchPattern(
                    pattern=f"{original_pattern.pattern}.*{term}",
                    description=f"Refined pattern combining '{original_pattern.pattern}' with context term '{term}'",
                    is_regex=True,
                    case_sensitive=original_pattern.case_sensitive
                ))
        
        return RefinementStep(
            patterns=refined_patterns,
            file_patterns=previous_step.file_patterns,
            exclude_patterns=previous_step.exclude_patterns,
            context_lines=previous_step.context_lines,
            max_matches_per_file=previous_step.max_matches_per_file,
            max_total_matches=previous_step.max_total_matches,
            description=f"Refinement step {step_num} using context terms"
        )
    
    async def _refine_by_frequency(
        self, 
        previous_step: RefinementStep,
        step_num: int
    ) -> Optional[RefinementStep]:
        """
        Refine search based on frequency of terms in previous matches.
        
        Args:
            previous_step: The previous refinement step
            step_num: Current step number
            
        Returns:
            A new refinement step with refined patterns
        """
        # Placeholder implementation
        return None
    
    async def _refine_by_proximity(
        self, 
        previous_step: RefinementStep,
        step_num: int
    ) -> Optional[RefinementStep]:
        """
        Refine search based on proximity of terms in previous matches.
        
        Args:
            previous_step: The previous refinement step
            step_num: Current step number
            
        Returns:
            A new refinement step with refined patterns
        """
        # Placeholder implementation
        return None
    
    def _extract_common_terms(self, step: RefinementStep) -> List[str]:
        """
        Extract common terms from matched lines.
        
        Args:
            step: The refinement step
            
        Returns:
            List of common terms
        """
        # Placeholder implementation
        # In a real implementation, this would use NLP techniques to extract meaningful terms
        return []
    
    def _calculate_step_confidence(self, step: RefinementStep) -> float:
        """
        Calculate confidence score for a refinement step.
        
        Args:
            step: The refinement step
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Placeholder implementation
        # In a real implementation, this would use various factors to calculate confidence
        
        # For now, just base it on the number of matches
        if not step.result:
            return 0.0
        
        match_count = (
            len(step.result.matches) 
            if isinstance(step.result, (RipgrepSearchResult, ParallelSearchResult)) 
            else 0
        )
        
        if match_count == 0:
            return 0.0
        elif match_count < 5:
            return 0.3
        elif match_count < 20:
            return 0.6
        else:
            return 0.8


# Register the operation
recursive_refinement = register_operation("recursive_refinement")(RecursiveRefinementOperation)


async def refine_search(params: RecursiveRefinementParams) -> RecursiveRefinementResult:
    """
    Recursively refine a search to narrow down results.
    
    This function creates and executes a RecursiveRefinementOperation with the given parameters.
    
    Args:
        params: Parameters for the recursive refinement
        
    Returns:
        Result of the recursive refinement operation
    """
    operation = recursive_refinement("recursive_refinement")
    return await operation.execute_with_stats(params)