"""
TSAP Incremental Aggregator.

This module provides utilities for aggregating results from incremental processing,
combining partial results into a coherent final output.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, TypeVar, Generic, Callable
from collections import Counter, defaultdict

from tsap.utils.logging import logger
from tsap.utils.errors import TSAPError


# Type variables for result type
R = TypeVar('R')
T = TypeVar('T')


class AggregationError(TSAPError):
    """Error raised when result aggregation fails."""
    
    def __init__(
        self,
        message: str,
        aggregator: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize an aggregation error.
        
        Args:
            message: Error message
            aggregator: Aggregator name
            details: Additional error details
        """
        error_details = details or {}
        if aggregator:
            error_details["aggregator"] = aggregator
        
        super().__init__(message, "AGGREGATION_ERROR", error_details)


class Aggregator(Generic[R, T], ABC):
    """Base class for result aggregators.
    
    Aggregators combine partial results from incremental processing into a
    coherent final output.
    """
    
    def __init__(self, name: str):
        """Initialize an aggregator.
        
        Args:
            name: Aggregator name
        """
        self.name = name
    
    @abstractmethod
    async def aggregate(self, results: List[R]) -> T:
        """Aggregate results.
        
        Args:
            results: List of partial results
            
        Returns:
            Aggregated result
        """
        pass
    
    async def preprocess_result(self, result: R) -> R:
        """Preprocess a result before aggregation.
        
        Override this method to implement custom preprocessing.
        
        Args:
            result: Result to preprocess
            
        Returns:
            Preprocessed result
        """
        return result
    
    async def postprocess_result(self, result: T) -> T:
        """Postprocess the aggregated result.
        
        Override this method to implement custom postprocessing.
        
        Args:
            result: Aggregated result
            
        Returns:
            Postprocessed result
        """
        return result
    
    async def process(self, results: List[R]) -> T:
        """Process and aggregate results.
        
        Args:
            results: List of partial results
            
        Returns:
            Aggregated result
            
        Raises:
            AggregationError: If aggregation fails
        """
        try:
            # Preprocess results
            processed_results = []
            for result in results:
                processed_result = await self.preprocess_result(result)
                processed_results.append(processed_result)
            
            # Aggregate results
            aggregated_result = await self.aggregate(processed_results)
            
            # Postprocess result
            final_result = await self.postprocess_result(aggregated_result)
            
            return final_result
            
        except Exception as e:
            # Wrap in AggregationError if needed
            if not isinstance(e, AggregationError):
                raise AggregationError(
                    str(e),
                    aggregator=self.name,
                    details={"original_error": str(e)},
                ) from e
            
            raise


class ListAggregator(Aggregator[List[Any], List[Any]]):
    """Aggregator for lists.
    
    Combines lists into a single list, with various options for combination.
    """
    
    def __init__(
        self,
        deduplicate: bool = False,
        flatten: bool = True,
        sort: bool = False,
        sort_key: Optional[Callable[[Any], Any]] = None,
        reverse: bool = False,
        limit: Optional[int] = None,
        name: str = "list_aggregator",
    ):
        """Initialize a list aggregator.
        
        Args:
            deduplicate: Whether to remove duplicates
            flatten: Whether to flatten nested lists
            sort: Whether to sort the result
            sort_key: Key function for sorting
            reverse: Whether to reverse the sort order
            limit: Maximum number of items in the result
            name: Aggregator name
        """
        super().__init__(name)
        self.deduplicate = deduplicate
        self.flatten = flatten
        self.sort = sort
        self.sort_key = sort_key
        self.reverse = reverse
        self.limit = limit
    
    async def aggregate(self, results: List[List[Any]]) -> List[Any]:
        """Aggregate lists.
        
        Args:
            results: List of lists
            
        Returns:
            Aggregated list
        """
        # Combine lists
        if self.flatten:
            # Flatten nested lists
            combined = []
            for result in results:
                combined.extend(result)
        else:
            # Keep lists separate
            combined = []
            for result in results:
                combined.append(result)
        
        # Deduplicate
        if self.deduplicate:
            # Use a set to remove duplicates while preserving order
            seen = set()
            deduplicated = []
            for item in combined:
                # Only hashable items can be deduplicated
                try:
                    item_hash = hash(item)
                    if item_hash not in seen:
                        seen.add(item_hash)
                        deduplicated.append(item)
                except TypeError:
                    # Unhashable item (e.g., list or dict), keep it
                    deduplicated.append(item)
            combined = deduplicated
        
        # Sort
        if self.sort:
            combined.sort(key=self.sort_key, reverse=self.reverse)
        
        # Limit
        if self.limit is not None:
            combined = combined[:self.limit]
        
        return combined


class DictionaryAggregator(Aggregator[Dict[Any, Any], Dict[Any, Any]]):
    """Aggregator for dictionaries.
    
    Combines dictionaries with various options for handling conflicts.
    """
    
    def __init__(
        self,
        conflict_resolver: str = "overwrite",
        name: str = "dictionary_aggregator",
    ):
        """Initialize a dictionary aggregator.
        
        Args:
            conflict_resolver: Conflict resolution strategy
                ('overwrite', 'keep_first', 'merge', 'list', 'sum', 'min', 'max', 'concatenate')
            name: Aggregator name
        """
        super().__init__(name)
        self.conflict_resolver = conflict_resolver
    
    async def aggregate(self, results: List[Dict[Any, Any]]) -> Dict[Any, Any]:
        """Aggregate dictionaries.
        
        Args:
            results: List of dictionaries
            
        Returns:
            Aggregated dictionary
        """
        if not results:
            return {}
        
        # Initialize with the first dictionary
        combined = {}
        
        # Process each dictionary
        for result in results:
            if self.conflict_resolver == "keep_first":
                # Only add keys that don't already exist
                for key, value in result.items():
                    if key not in combined:
                        combined[key] = value
            elif self.conflict_resolver == "overwrite":
                # Overwrite existing keys
                combined.update(result)
            elif self.conflict_resolver == "merge":
                # Recursively merge dictionaries
                for key, value in result.items():
                    if key in combined and isinstance(combined[key], dict) and isinstance(value, dict):
                        # Recursive merge for nested dictionaries
                        nested_merger = DictionaryAggregator(self.conflict_resolver)
                        combined[key] = await nested_merger.aggregate([combined[key], value])
                    else:
                        # Overwrite for non-dictionary values
                        combined[key] = value
            elif self.conflict_resolver == "list":
                # Collect values in a list
                for key, value in result.items():
                    if key not in combined:
                        combined[key] = [value]
                    else:
                        if not isinstance(combined[key], list):
                            combined[key] = [combined[key]]
                        combined[key].append(value)
            elif self.conflict_resolver == "sum":
                # Sum numeric values, overwrite others
                for key, value in result.items():
                    if key not in combined:
                        combined[key] = value
                    elif isinstance(combined[key], (int, float)) and isinstance(value, (int, float)):
                        combined[key] += value
                    else:
                        # Non-numeric values, just overwrite
                        combined[key] = value
            elif self.conflict_resolver == "min":
                # Take the minimum of numeric values, overwrite others
                for key, value in result.items():
                    if key not in combined:
                        combined[key] = value
                    elif isinstance(combined[key], (int, float)) and isinstance(value, (int, float)):
                        combined[key] = min(combined[key], value)
                    else:
                        # Non-numeric values, just overwrite
                        combined[key] = value
            elif self.conflict_resolver == "max":
                # Take the maximum of numeric values, overwrite others
                for key, value in result.items():
                    if key not in combined:
                        combined[key] = value
                    elif isinstance(combined[key], (int, float)) and isinstance(value, (int, float)):
                        combined[key] = max(combined[key], value)
                    else:
                        # Non-numeric values, just overwrite
                        combined[key] = value
            elif self.conflict_resolver == "concatenate":
                # Concatenate string values, convert others to strings
                for key, value in result.items():
                    if key not in combined:
                        combined[key] = value
                    elif isinstance(combined[key], str) and isinstance(value, str):
                        combined[key] += value
                    else:
                        # Non-string values, convert to strings and concatenate
                        combined[key] = str(combined[key]) + str(value)
            else:
                # Unknown conflict resolver, just overwrite
                combined.update(result)
        
        return combined


class CounterAggregator(Aggregator[Dict[Any, int], Dict[Any, int]]):
    """Aggregator for counters.
    
    Combines counter dictionaries by summing the counts.
    """
    
    def __init__(self, name: str = "counter_aggregator"):
        """Initialize a counter aggregator.
        
        Args:
            name: Aggregator name
        """
        super().__init__(name)
    
    async def aggregate(self, results: List[Dict[Any, int]]) -> Dict[Any, int]:
        """Aggregate counters.
        
        Args:
            results: List of counter dictionaries
            
        Returns:
            Aggregated counter
        """
        # Combine counters
        combined = Counter()
        for result in results:
            counter = Counter(result)
            combined.update(counter)
        
        return dict(combined)


class StatisticsAggregator(Aggregator[Dict[str, Any], Dict[str, Any]]):
    """Aggregator for statistics.
    
    Combines statistics with various options for handling different types of values.
    """
    
    def __init__(
        self,
        count_fields: Optional[List[str]] = None,
        sum_fields: Optional[List[str]] = None,
        min_fields: Optional[List[str]] = None,
        max_fields: Optional[List[str]] = None,
        avg_fields: Optional[List[str]] = None,
        list_fields: Optional[List[str]] = None,
        merge_fields: Optional[List[str]] = None,
        name: str = "statistics_aggregator",
    ):
        """Initialize a statistics aggregator.
        
        Args:
            count_fields: Fields to count (number of occurrences)
            sum_fields: Fields to sum
            min_fields: Fields to take the minimum value
            max_fields: Fields to take the maximum value
            avg_fields: Fields to average
            list_fields: Fields to collect in a list
            merge_fields: Dictionary fields to merge recursively
            name: Aggregator name
        """
        super().__init__(name)
        self.count_fields = count_fields or []
        self.sum_fields = sum_fields or []
        self.min_fields = min_fields or []
        self.max_fields = max_fields or []
        self.avg_fields = avg_fields or []
        self.list_fields = list_fields or []
        self.merge_fields = merge_fields or []
    
    async def aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate statistics.
        
        Args:
            results: List of statistic dictionaries
            
        Returns:
            Aggregated statistics
        """
        if not results:
            return {}
        
        # Initialize counters and aggregates
        count_stats = {field: 0 for field in self.count_fields}
        sum_stats = {field: 0 for field in self.sum_fields}
        min_stats = {field: None for field in self.min_fields}
        max_stats = {field: None for field in self.max_fields}
        avg_stats = {field: {"sum": 0, "count": 0} for field in self.avg_fields}
        list_stats = {field: [] for field in self.list_fields}
        merge_stats = {field: {} for field in self.merge_fields}
        
        # Process each result
        for result in results:
            # Count fields
            for field in self.count_fields:
                if field in result:
                    count_stats[field] += 1
            
            # Sum fields
            for field in self.sum_fields:
                if field in result and isinstance(result[field], (int, float)):
                    sum_stats[field] += result[field]
            
            # Min fields
            for field in self.min_fields:
                if field in result and isinstance(result[field], (int, float)):
                    if min_stats[field] is None or result[field] < min_stats[field]:
                        min_stats[field] = result[field]
            
            # Max fields
            for field in self.max_fields:
                if field in result and isinstance(result[field], (int, float)):
                    if max_stats[field] is None or result[field] > max_stats[field]:
                        max_stats[field] = result[field]
            
            # Average fields
            for field in self.avg_fields:
                if field in result and isinstance(result[field], (int, float)):
                    avg_stats[field]["sum"] += result[field]
                    avg_stats[field]["count"] += 1
            
            # List fields
            for field in self.list_fields:
                if field in result:
                    list_stats[field].append(result[field])
            
            # Merge fields
            for field in self.merge_fields:
                if field in result and isinstance(result[field], dict):
                    # Use the dictionary aggregator for recursive merging
                    merger = DictionaryAggregator("merge")
                    merge_stats[field] = await merger.aggregate([merge_stats[field], result[field]])
        
        # Combine all statistics
        combined = {}
        
        # Add count stats
        for field, value in count_stats.items():
            combined[field] = value
        
        # Add sum stats
        for field, value in sum_stats.items():
            combined[f"{field}_sum"] = value
        
        # Add min stats
        for field, value in min_stats.items():
            if value is not None:
                combined[f"{field}_min"] = value
        
        # Add max stats
        for field, value in max_stats.items():
            if value is not None:
                combined[f"{field}_max"] = value
        
        # Add average stats
        for field, data in avg_stats.items():
            if data["count"] > 0:
                combined[f"{field}_avg"] = data["sum"] / data["count"]
        
        # Add list stats
        for field, value in list_stats.items():
            combined[f"{field}_list"] = value
        
        # Add merge stats
        for field, value in merge_stats.items():
            combined[field] = value
        
        return combined


class GroupByAggregator(Aggregator[Dict[str, Any], Dict[Any, List[Dict[str, Any]]]]):
    """Aggregator for grouping results by a key.
    
    Groups dictionaries by a specified key field.
    """
    
    def __init__(
        self,
        key_field: str,
        sort_groups: bool = False,
        sort_key: Optional[Callable[[Any], Any]] = None,
        reverse: bool = False,
        name: str = "group_by_aggregator",
    ):
        """Initialize a group-by aggregator.
        
        Args:
            key_field: Field to group by
            sort_groups: Whether to sort the groups by key
            sort_key: Key function for sorting
            reverse: Whether to reverse the sort order
            name: Aggregator name
        """
        super().__init__(name)
        self.key_field = key_field
        self.sort_groups = sort_groups
        self.sort_key = sort_key
        self.reverse = reverse
    
    async def aggregate(self, results: List[Dict[str, Any]]) -> Dict[Any, List[Dict[str, Any]]]:
        """Aggregate by grouping.
        
        Args:
            results: List of dictionaries
            
        Returns:
            Dictionary mapping keys to lists of dictionaries
        """
        # Group by key field
        groups = defaultdict(list)
        
        for result in results:
            if self.key_field in result:
                key = result[self.key_field]
                groups[key].append(result)
            else:
                # Handle missing key field
                groups[None].append(result)
        
        # Convert to regular dictionary
        grouped = dict(groups)
        
        # Sort groups if requested
        if self.sort_groups:
            # Sort keys
            sorted_keys = sorted(grouped.keys(), key=self.sort_key, reverse=self.reverse)
            
            # Rebuild dictionary with sorted keys
            sorted_groups = {}
            for key in sorted_keys:
                sorted_groups[key] = grouped[key]
            
            grouped = sorted_groups
        
        return grouped


class AggregatorRegistry:
    """Registry for aggregators."""
    
    _aggregators: Dict[str, Aggregator] = {}
    
    @classmethod
    def register(cls, name: str, aggregator: Aggregator) -> None:
        """Register an aggregator.
        
        Args:
            name: Aggregator name
            aggregator: Aggregator instance
        """
        cls._aggregators[name] = aggregator
        logger.debug(f"Registered aggregator: {name}")
    
    @classmethod
    def get_aggregator(cls, name: str) -> Optional[Aggregator]:
        """Get an aggregator by name.
        
        Args:
            name: Aggregator name
            
        Returns:
            Aggregator instance or None if not found
        """
        return cls._aggregators.get(name)
    
    @classmethod
    def list_aggregators(cls) -> List[str]:
        """Get a list of all registered aggregators.
        
        Returns:
            List of aggregator names
        """
        return list(cls._aggregators.keys())


def register_aggregator(aggregator: Aggregator) -> None:
    """Register an aggregator.
    
    Args:
        aggregator: Aggregator to register
    """
    AggregatorRegistry.register(aggregator.name, aggregator)


# Register default aggregators
register_aggregator(ListAggregator())
register_aggregator(DictionaryAggregator())
register_aggregator(CounterAggregator())
register_aggregator(StatisticsAggregator())
register_aggregator(GroupByAggregator("group"))


# Convenience functions

def get_aggregator(name: str) -> Optional[Aggregator]:
    """Get an aggregator by name.
    
    Args:
        name: Aggregator name
        
    Returns:
        Aggregator instance or None if not found
    """
    return AggregatorRegistry.get_aggregator(name)


def list_aggregators() -> List[str]:
    """Get a list of all registered aggregators.
    
    Returns:
        List of aggregator names
    """
    return AggregatorRegistry.list_aggregators()


async def aggregate_lists(
    lists: List[List[Any]],
    deduplicate: bool = False,
    flatten: bool = True,
    sort: bool = False,
    sort_key: Optional[Callable[[Any], Any]] = None,
    reverse: bool = False,
    limit: Optional[int] = None,
) -> List[Any]:
    """Aggregate lists.
    
    Args:
        lists: List of lists to aggregate
        deduplicate: Whether to remove duplicates
        flatten: Whether to flatten nested lists
        sort: Whether to sort the result
        sort_key: Key function for sorting
        reverse: Whether to reverse the sort order
        limit: Maximum number of items in the result
        
    Returns:
        Aggregated list
    """
    aggregator = ListAggregator(
        deduplicate=deduplicate,
        flatten=flatten,
        sort=sort,
        sort_key=sort_key,
        reverse=reverse,
        limit=limit,
    )
    
    return await aggregator.process(lists)


async def aggregate_dictionaries(
    dicts: List[Dict[Any, Any]],
    conflict_resolver: str = "overwrite",
) -> Dict[Any, Any]:
    """Aggregate dictionaries.
    
    Args:
        dicts: List of dictionaries to aggregate
        conflict_resolver: Conflict resolution strategy
            ('overwrite', 'keep_first', 'merge', 'list', 'sum', 'min', 'max', 'concatenate')
        
    Returns:
        Aggregated dictionary
    """
    aggregator = DictionaryAggregator(conflict_resolver=conflict_resolver)
    return await aggregator.process(dicts)


async def aggregate_counters(counters: List[Dict[Any, int]]) -> Dict[Any, int]:
    """Aggregate counters.
    
    Args:
        counters: List of counter dictionaries to aggregate
        
    Returns:
        Aggregated counter
    """
    aggregator = CounterAggregator()
    return await aggregator.process(counters)


async def group_by(
    dicts: List[Dict[str, Any]],
    key_field: str,
    sort_groups: bool = False,
) -> Dict[Any, List[Dict[str, Any]]]:
    """Group dictionaries by a key field.
    
    Args:
        dicts: List of dictionaries to group
        key_field: Field to group by
        sort_groups: Whether to sort the groups by key
        
    Returns:
        Dictionary mapping keys to lists of dictionaries
    """
    aggregator = GroupByAggregator(key_field=key_field, sort_groups=sort_groups)
    return await aggregator.process(dicts)