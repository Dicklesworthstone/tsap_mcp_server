"""
Parallel search operations for TSAP.

This module provides functionality to run multiple search patterns in parallel
and consolidate the results, offering more efficient and powerful searching.
"""
import asyncio
from typing import Dict, List, Any, Optional, Tuple
import time

from tsap.utils.logging import logger
from tsap.core.ripgrep import ripgrep_search
from tsap.mcp.models import (
    SearchPattern, ParallelSearchParams, ParallelSearchMatch,
    ParallelSearchResult, RipgrepSearchParams
)


async def _search_with_pattern(
    pattern: SearchPattern, 
    paths: List[str],
    file_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    context_lines: int = 2,
    max_matches: Optional[int] = None,
) -> Tuple[SearchPattern, List[ParallelSearchMatch], Dict[str, Any]]:
    """Execute a search with a single pattern.
    
    Args:
        pattern: Search pattern
        paths: Paths to search
        file_patterns: Optional file patterns to include
        exclude_patterns: Optional file patterns to exclude
        context_lines: Number of context lines around matches
        max_matches: Maximum matches to return
        
    Returns:
        Tuple of (pattern, matches, stats)
    """
    # Create ripgrep search parameters
    params = RipgrepSearchParams(
        pattern=pattern.pattern,
        paths=paths,
        case_sensitive=pattern.case_sensitive,
        whole_word=pattern.whole_word,
        regex=pattern.regex,
        file_patterns=file_patterns,
        exclude_patterns=exclude_patterns,
        context_lines=context_lines,
        max_total_matches=max_matches,
    )
    
    # Execute the search
    result = await ripgrep_search(params)
    
    # Convert RipgrepMatch to ParallelSearchMatch
    parallel_matches = []
    for rg_match in result.matches:
        parallel_match = ParallelSearchMatch(
            path=rg_match.path,
            line_number=rg_match.line_number,
            match_text=rg_match.match_text,
            line_text=rg_match.line_text,
            pattern=pattern.pattern,
            pattern_description=pattern.description,
            pattern_category=pattern.category,
            before_context=rg_match.before_context,
            after_context=rg_match.after_context,
            tags=pattern.tags,
            confidence=pattern.confidence,
        )
        parallel_matches.append(parallel_match)
    
    # Create stats
    stats = {
        "pattern": pattern.pattern,
        "match_count": len(parallel_matches),
        "truncated": result.truncated,
        "execution_time": result.execution_time,
    }
    
    return pattern, parallel_matches, stats


def _are_matches_overlapping(
    match1: ParallelSearchMatch, 
    match2: ParallelSearchMatch,
    context_window: int = 5
) -> bool:
    """Check if two matches are overlapping or close enough to consolidate.
    
    Args:
        match1: First match
        match2: Second match
        context_window: Number of lines to consider as overlapping context
        
    Returns:
        Whether the matches are overlapping
    """
    # Must be in the same file
    if match1.path != match2.path:
        return False
    
    # Check if line numbers are close enough
    line_distance = abs(match1.line_number - match2.line_number)
    return line_distance <= context_window


def _consolidate_matches(
    matches: List[ParallelSearchMatch],
    context_window: int = 5
) -> List[ParallelSearchMatch]:
    """Consolidate overlapping matches.
    
    Args:
        matches: List of matches to consolidate
        context_window: Number of lines to consider as overlapping context
        
    Returns:
        Consolidated matches
    """
    if not matches:
        return []
        
    # Sort matches by path and line number
    sorted_matches = sorted(
        matches, 
        key=lambda m: (m.path, m.line_number)
    )
    
    consolidated = []
    current_group = [sorted_matches[0]]
    
    for i in range(1, len(sorted_matches)):
        current_match = sorted_matches[i]
        last_match = current_group[-1]
        
        if _are_matches_overlapping(last_match, current_match, context_window):
            # Matches overlap, add to current group
            current_group.append(current_match)
        else:
            # No overlap, consolidate current group and start a new one
            if len(current_group) == 1:
                # Only one match in group, no need to consolidate
                consolidated.append(current_group[0])
            else:
                # Multiple matches, consolidate
                consolidated_match = _consolidate_match_group(current_group)
                consolidated.append(consolidated_match)
                
            # Start a new group
            current_group = [current_match]
    
    # Handle the last group
    if len(current_group) == 1:
        consolidated.append(current_group[0])
    else:
        consolidated_match = _consolidate_match_group(current_group)
        consolidated.append(consolidated_match)
        
    return consolidated


def _consolidate_match_group(
    matches: List[ParallelSearchMatch]
) -> ParallelSearchMatch:
    """Consolidate a group of overlapping matches into a single match.
    
    Args:
        matches: Group of overlapping matches
        
    Returns:
        Consolidated match
    """
    # Use the first match as the base
    base_match = matches[0]
    
    # Collect all unique patterns, descriptions, and tags
    patterns = []
    descriptions = []
    categories = set()
    all_tags = set()
    
    for match in matches:
        if match.pattern not in patterns:
            patterns.append(match.pattern)
            
        if match.pattern_description and match.pattern_description not in descriptions:
            descriptions.append(match.pattern_description)
            
        if match.pattern_category:
            categories.add(match.pattern_category)
            
        all_tags.update(match.tags)
    
    # Combine patterns and descriptions
    combined_pattern = " | ".join(patterns)
    combined_description = "; ".join(desc for desc in descriptions if desc)
    
    # Calculate average confidence
    avg_confidence = sum(match.confidence for match in matches) / len(matches)
    
    # Create consolidated match
    return ParallelSearchMatch(
        path=base_match.path,
        line_number=base_match.line_number,
        match_text=base_match.match_text,
        line_text=base_match.line_text,
        pattern=combined_pattern,
        pattern_description=combined_description,
        pattern_category=", ".join(categories) if categories else None,
        before_context=base_match.before_context,
        after_context=base_match.after_context,
        tags=list(all_tags),
        confidence=avg_confidence,
    )


async def parallel_search(params: ParallelSearchParams) -> ParallelSearchResult:
    """Execute multiple search patterns in parallel.
    
    Args:
        params: Parallel search parameters
        
    Returns:
        Parallel search results
    """
    start_time = time.time()
    
    # Log the operation
    logger.info(
        f"Starting parallel search with {len(params.patterns)} patterns",
        component="composite",
        operation="parallel_search",
        context={
            "pattern_count": len(params.patterns),
            "paths": params.paths,
        }
    )
    
    # Apply confidence filter if specified
    patterns_to_search = params.patterns
    if params.min_confidence is not None:
        patterns_to_search = [
            p for p in params.patterns 
            if p.confidence >= params.min_confidence
        ]
        
        if len(patterns_to_search) < len(params.patterns):
            logger.info(
                f"Filtered {len(params.patterns) - len(patterns_to_search)} patterns based on confidence threshold",
                component="composite",
                operation="parallel_search"
            )
    
    # Determine max matches per pattern
    max_total = params.max_total_matches
    max_per_pattern = params.max_matches_per_pattern
    
    if max_total is not None and max_per_pattern is None:
        # Distribute max total matches across patterns
        max_per_pattern = max(1, max_total // len(patterns_to_search))
    
    # Create tasks for each pattern
    tasks = []
    for pattern in patterns_to_search:
        task = _search_with_pattern(
            pattern=pattern,
            paths=params.paths,
            file_patterns=params.file_patterns,
            exclude_patterns=params.exclude_patterns,
            context_lines=params.context_lines,
            max_matches=max_per_pattern,
        )
        tasks.append(task)
    
    # Execute all search tasks in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    all_matches = []
    pattern_stats = {}
    total_matches = 0
    files_searched = set()
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            # Handle search failure
            pattern = patterns_to_search[i]
            logger.error(
                f"Search failed for pattern '{pattern.pattern}': {str(result)}",
                component="composite",
                operation="parallel_search",
                exception=result,
                context={"pattern": pattern.pattern}
            )
            
            pattern_stats[pattern.pattern] = {
                "error": str(result),
                "match_count": 0,
            }
            continue
            
        # Unpack successful result
        pattern, matches, stats = result
        
        # Add matches to overall results
        all_matches.extend(matches)
        total_matches += len(matches)
        
        # Track files with matches
        for match in matches:
            files_searched.add(match.path)
            
        # Store pattern statistics
        pattern_stats[pattern.pattern] = stats
        
        # Check if we've reached the maximum total matches
        if max_total is not None and total_matches >= max_total:
            logger.info(
                f"Reached maximum total matches ({max_total}), truncating results",
                component="composite",
                operation="parallel_search"
            )
            # Mark as truncated but don't break, process all patterns
            for p in patterns_to_search[i+1:]:
                pattern_stats[p.pattern] = {
                    "skipped": True,
                    "reason": "max_total_matches_reached",
                    "match_count": 0,
                }
    
    # Consolidate overlapping matches if requested
    truncated = total_matches >= (max_total or float('inf'))
    final_matches = all_matches
    
    if params.consolidate_overlapping and all_matches:
        logger.debug(
            "Consolidating overlapping matches",
            component="composite",
            operation="parallel_search"
        )
        
        final_matches = _consolidate_matches(
            all_matches, 
            context_window=params.context_lines * 2
        )
        
        logger.info(
            f"Consolidated {len(all_matches)} matches into {len(final_matches)} results",
            component="composite",
            operation="parallel_search"
        )
    
    # Sort results by confidence and path/line number
    final_matches.sort(
        key=lambda m: (-m.confidence, m.path, m.line_number)
    )
    
    # Trim to max_total_matches if needed
    if max_total is not None and len(final_matches) > max_total:
        final_matches = final_matches[:max_total]
        truncated = True
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # Log completion
    logger.success(
        f"Parallel search completed: {len(final_matches)} matches across {len(files_searched)} files",
        component="composite",
        operation="parallel_search",
        context={
            "match_count": len(final_matches),
            "files_with_matches": len(files_searched),
            "execution_time": execution_time,
            "truncated": truncated,
        }
    )
    
    # Create and return result
    return ParallelSearchResult(
        matches=final_matches,
        pattern_stats=pattern_stats,
        total_patterns=len(patterns_to_search),
        total_files_searched=len(files_searched),
        total_matches=len(final_matches),
        truncated=truncated,
        execution_time=execution_time,
    )