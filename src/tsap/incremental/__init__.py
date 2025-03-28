"""
Incremental processing package marker for TSAP MCP Server.

This package provides functionality for processing data in chunks, 
with support for splitting, aggregating, streaming, and parallel execution.
It enables handling large datasets that may not fit in memory all at once.
"""

from tsap.incremental.processor import (
    IncrementalProcessor, IncrementalProcessingError, 
    ProcessingProgress, ProcessingContext, IncrementalRegistry,
    register_processor, get_processor, list_processors
)

from tsap.incremental.splitter import (
    Splitter, SplitterError, ListSplitter, DictionarySplitter,
    TextSplitter, FileSplitter, JsonSplitter, DirectorySplitter,
    register_splitter, get_splitter, list_splitters,
    split_text, split_file, split_directory
)

from tsap.incremental.aggregator import (
    Aggregator, AggregationError, ListAggregator, DictionaryAggregator,
    CounterAggregator, StatisticsAggregator, GroupByAggregator,
    register_aggregator, get_aggregator, list_aggregators,
    aggregate_lists, aggregate_dictionaries, aggregate_counters, group_by
)

from tsap.incremental.streamer import (
    InputStream, OutputStream, StreamingError, ListInputStream, ListOutputStream,
    FileInputStream, FileOutputStream, JsonInputStream, JsonOutputStream,
    register_input_stream, register_output_stream,
    get_input_stream, get_output_stream,
    list_input_streams, list_output_streams,
    stream_pipeline, stream_json_file, stream_text_file
)


__all__ = [
    # Processor components
    'IncrementalProcessor', 'IncrementalProcessingError', 
    'ProcessingProgress', 'ProcessingContext', 'IncrementalRegistry',
    'register_processor', 'get_processor', 'list_processors',
    
    # Splitter components
    'Splitter', 'SplitterError', 'ListSplitter', 'DictionarySplitter',
    'TextSplitter', 'FileSplitter', 'JsonSplitter', 'DirectorySplitter',
    'register_splitter', 'get_splitter', 'list_splitters',
    'split_text', 'split_file', 'split_directory',
    
    # Aggregator components
    'Aggregator', 'AggregationError', 'ListAggregator', 'DictionaryAggregator',
    'CounterAggregator', 'StatisticsAggregator', 'GroupByAggregator',
    'register_aggregator', 'get_aggregator', 'list_aggregators',
    'aggregate_lists', 'aggregate_dictionaries', 'aggregate_counters', 'group_by',
    
    # Streamer components
    'InputStream', 'OutputStream', 'StreamingError', 'ListInputStream', 'ListOutputStream',
    'FileInputStream', 'FileOutputStream', 'JsonInputStream', 'JsonOutputStream',
    'register_input_stream', 'register_output_stream',
    'get_input_stream', 'get_output_stream',
    'list_input_streams', 'list_output_streams',
    'stream_pipeline', 'stream_json_file', 'stream_text_file'
]