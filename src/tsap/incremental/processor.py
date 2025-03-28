"""
TSAP Incremental Processor.

This module provides the core functionality for processing data incrementally,
enabling efficient processing of large datasets by dividing them into manageable chunks.
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, AsyncIterator, TypeVar, Generic, Callable, Tuple
from dataclasses import dataclass, field

from tsap.utils.logging import logger
from tsap.utils.errors import TSAPError
from tsap.performance_mode import get_parameter
from tsap.utils.async_utils import TaskManager


class IncrementalProcessingError(TSAPError):
    """Error raised when incremental processing fails."""
    
    def __init__(
        self,
        message: str,
        processor: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize an incremental processing error.
        
        Args:
            message: Error message
            processor: Processor name
            details: Additional error details
        """
        error_details = details or {}
        if processor:
            error_details["processor"] = processor
        
        super().__init__(message, "INCREMENTAL_PROCESSING_ERROR", error_details)


# Type variables for input and output
I = TypeVar('I')  # noqa: E741
O = TypeVar('O')  # noqa: E741
C = TypeVar('C')  # Chunk type


@dataclass
class ProcessingProgress:
    """Progress information for incremental processing."""
    
    total_items: int = 0
    processed_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    @property
    def completion_percentage(self) -> float:
        """Get the completion percentage.
        
        Returns:
            Completion percentage (0.0 to 100.0)
        """
        if self.total_items == 0:
            return 100.0
        
        return (self.processed_items / self.total_items) * 100
    
    @property
    def is_complete(self) -> bool:
        """Check if processing is complete.
        
        Returns:
            True if processing is complete, False otherwise
        """
        return self.processed_items >= self.total_items
    
    @property
    def elapsed_time(self) -> Optional[float]:
        """Get the elapsed time.
        
        Returns:
            Elapsed time in seconds or None if not started
        """
        if self.started_at is None:
            return None
        
        end_time = self.completed_at or time.time()
        return end_time - self.started_at
    
    @property
    def items_per_second(self) -> Optional[float]:
        """Get the processing rate.
        
        Returns:
            Items processed per second or None if not started or no time elapsed
        """
        elapsed = self.elapsed_time
        if elapsed is None or elapsed <= 0 or self.processed_items == 0:
            return None
        
        return self.processed_items / elapsed
    
    @property
    def estimated_time_remaining(self) -> Optional[float]:
        """Get the estimated time remaining.
        
        Returns:
            Estimated time remaining in seconds or None if not applicable
        """
        if self.is_complete or self.items_per_second is None:
            return None
        
        remaining_items = self.total_items - self.processed_items
        return remaining_items / self.items_per_second


@dataclass
class ProcessingContext:
    """Context for incremental processing operations."""
    
    params: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    progress: ProcessingProgress = field(default_factory=ProcessingProgress)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_progress(
        self,
        processed: int = 0,
        successful: int = 0,
        failed: int = 0,
    ) -> None:
        """Update the progress information.
        
        Args:
            processed: Number of newly processed items
            successful: Number of newly successful items
            failed: Number of newly failed items
        """
        self.progress.processed_items += processed
        self.progress.successful_items += successful
        self.progress.failed_items += failed
        
        # Check if processing is complete
        if self.progress.is_complete and self.progress.completed_at is None:
            self.progress.completed_at = time.time()


class IncrementalProcessor(Generic[I, O, C], ABC):
    """Base class for incremental processors.
    
    Incremental processors handle large datasets by processing them in chunks,
    which can be processed in parallel for improved performance.
    """
    
    def __init__(self, name: str):
        """Initialize an incremental processor.
        
        Args:
            name: Processor name
        """
        self.name = name
        self.statistics: Dict[str, Any] = {
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "total_items_processed": 0,
            "total_processing_time": 0.0,
        }
    
    @abstractmethod
    async def split(self, input_data: I) -> AsyncIterator[C]:
        """Split input data into chunks.
        
        Args:
            input_data: Input data
            
        Yields:
            Data chunks
        """
        pass
    
    @abstractmethod
    async def process_chunk(self, chunk: C, context: ProcessingContext) -> O:
        """Process a single chunk.
        
        Args:
            chunk: Data chunk
            context: Processing context
            
        Returns:
            Processed chunk result
        """
        pass
    
    @abstractmethod
    async def aggregate(self, results: List[O], context: ProcessingContext) -> Any:
        """Aggregate chunk results.
        
        Args:
            results: List of chunk results
            context: Processing context
            
        Returns:
            Aggregated result
        """
        pass
    
    async def preprocess(self, input_data: I, context: ProcessingContext) -> I:
        """Preprocess input data before splitting.
        
        Override this method to implement custom preprocessing.
        
        Args:
            input_data: Input data
            context: Processing context
            
        Returns:
            Preprocessed input data
        """
        return input_data
    
    async def postprocess(self, result: Any, context: ProcessingContext) -> Any:
        """Postprocess the aggregated result.
        
        Override this method to implement custom postprocessing.
        
        Args:
            result: Aggregated result
            context: Processing context
            
        Returns:
            Postprocessed result
        """
        return result
    
    async def process(
        self,
        input_data: I,
        params: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None,
    ) -> Any:
        """Process input data incrementally.
        
        Args:
            input_data: Input data
            params: Processing parameters
            metadata: Additional metadata
            progress_callback: Callback function for progress updates
            
        Returns:
            Processing result
            
        Raises:
            IncrementalProcessingError: If processing fails
        """
        # Create processing context
        context = ProcessingContext(
            params=params or {},
            metadata=metadata or {},
        )
        
        # Initialize progress tracking
        context.progress.started_at = time.time()
        
        # Start timing
        start_time = time.time()
        
        try:
            # Preprocess input data
            preprocessed_data = await self.preprocess(input_data, context)
            
            # Get performance mode parameters
            parallel_processing = get_parameter("parallel_processing", True)
            max_concurrency = get_parameter("max_concurrency", 5)
            batch_size = get_parameter("batch_size", 10)
            
            # Collect all chunks first to determine total items
            chunks = []
            async for chunk in self.split(preprocessed_data):
                chunks.append(chunk)
            
            # Update progress information
            context.progress.total_items = len(chunks)
            
            # Process chunks
            if parallel_processing and len(chunks) > 1:
                # Process chunks in parallel
                results = await self._process_chunks_parallel(chunks, context, max_concurrency, batch_size)
            else:
                # Process chunks sequentially
                results = await self._process_chunks_sequential(chunks, context)
            
            # Aggregate results
            aggregated_result = await self.aggregate(results, context)
            
            # Postprocess result
            final_result = await self.postprocess(aggregated_result, context)
            
            # Update statistics
            self.statistics["total_runs"] += 1
            self.statistics["successful_runs"] += 1
            self.statistics["total_items_processed"] += context.progress.processed_items
            
            return final_result
            
        except Exception as e:
            # Update statistics
            self.statistics["total_runs"] += 1
            self.statistics["failed_runs"] += 1
            
            # Wrap in IncrementalProcessingError if needed
            if not isinstance(e, IncrementalProcessingError):
                raise IncrementalProcessingError(
                    str(e),
                    processor=self.name,
                    details={"original_error": str(e)},
                ) from e
            
            raise
            
        finally:
            # Update statistics
            processing_time = time.time() - start_time
            self.statistics["total_processing_time"] += processing_time
            
            # Ensure completion time is set
            if context.progress.completed_at is None:
                context.progress.completed_at = time.time()
            
            # Call progress callback
            if progress_callback:
                progress_callback(context.progress)
    
    async def _process_chunks_sequential(
        self,
        chunks: List[C],
        context: ProcessingContext,
    ) -> List[O]:
        """Process chunks sequentially.
        
        Args:
            chunks: List of chunks
            context: Processing context
            
        Returns:
            List of chunk results
        """
        results = []
        
        for chunk in chunks:
            try:
                # Process chunk
                result = await self.process_chunk(chunk, context)
                
                # Store result
                results.append(result)
                
                # Update progress
                context.update_progress(processed=1, successful=1)
                
            except Exception as e:
                # Log error
                logger.error(f"Error processing chunk: {e}")
                
                # Update progress
                context.update_progress(processed=1, failed=1)
                
                # Re-raise if in strict mode
                strict_mode = context.params.get("strict_mode", False)
                if strict_mode:
                    raise
        
        return results
    
    async def _process_chunks_parallel(
        self,
        chunks: List[C],
        context: ProcessingContext,
        max_concurrency: int,
        batch_size: int,
    ) -> List[O]:
        """Process chunks in parallel.
        
        Args:
            chunks: List of chunks
            context: Processing context
            max_concurrency: Maximum number of concurrent tasks
            batch_size: Number of chunks to process in each batch
            
        Returns:
            List of chunk results
        """
        # Create a task manager
        task_manager = TaskManager()
        
        # Process chunks in batches
        results = []
        processed_count = 0  # noqa: F841
        
        # Define a wrapper function for processing a chunk
        async def process_chunk_wrapper(chunk: C) -> Tuple[O, bool]:
            try:
                # Create a copy of the context for this chunk
                chunk_context = ProcessingContext(
                    params=context.params,
                    metadata=context.metadata,
                )
                
                # Process chunk
                result = await self.process_chunk(chunk, chunk_context)
                
                return result, True  # Success
                
            except Exception as e:
                # Log error
                logger.error(f"Error processing chunk: {e}")
                
                return None, False  # Failure
        
        # Process chunks in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Create tasks for batch
            tasks = []
            for chunk in batch:
                task = task_manager.create_task(process_chunk_wrapper(chunk))
                tasks.append(task)
            
            # Wait for batch to complete
            batch_results, batch_exceptions = await task_manager.wait_all()
            
            # Process batch results
            for result, success in batch_results:
                if success:
                    # Store result
                    results.append(result)
                    
                    # Update progress
                    context.update_progress(processed=1, successful=1)
                else:
                    # Update progress
                    context.update_progress(processed=1, failed=1)
                    
                    # Re-raise if in strict mode
                    strict_mode = context.params.get("strict_mode", False)
                    if strict_mode:
                        raise IncrementalProcessingError(
                            "Processing failed in strict mode",
                            processor=self.name,
                        )
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processor statistics.
        
        Returns:
            Dictionary with processor statistics
        """
        return dict(self.statistics)
    
    def reset_statistics(self) -> None:
        """Reset processor statistics."""
        self.statistics = {
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "total_items_processed": 0,
            "total_processing_time": 0.0,
        }


class IncrementalRegistry:
    """Registry for incremental processors."""
    
    _processors: Dict[str, IncrementalProcessor] = {}
    
    @classmethod
    def register(cls, name: str, processor: IncrementalProcessor) -> None:
        """Register an incremental processor.
        
        Args:
            name: Processor name
            processor: Processor instance
        """
        cls._processors[name] = processor
        logger.debug(f"Registered incremental processor: {name}")
    
    @classmethod
    def get_processor(cls, name: str) -> Optional[IncrementalProcessor]:
        """Get an incremental processor by name.
        
        Args:
            name: Processor name
            
        Returns:
            Processor instance or None if not found
        """
        return cls._processors.get(name)
    
    @classmethod
    def list_processors(cls) -> List[str]:
        """Get a list of all registered processors.
        
        Returns:
            List of processor names
        """
        return list(cls._processors.keys())


def register_processor(processor: IncrementalProcessor) -> None:
    """Register an incremental processor.
    
    Args:
        processor: Processor to register
    """
    IncrementalRegistry.register(processor.name, processor)


# Convenience functions

def get_processor(name: str) -> Optional[IncrementalProcessor]:
    """Get an incremental processor by name.
    
    Args:
        name: Processor name
        
    Returns:
        Processor instance or None if not found
    """
    return IncrementalRegistry.get_processor(name)


def list_processors() -> List[str]:
    """Get a list of all registered incremental processors.
    
    Returns:
        List of processor names
    """
    return IncrementalRegistry.list_processors()