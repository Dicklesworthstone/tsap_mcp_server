"""
TSAP Asynchronous Utilities.

This module provides utilities for working with asynchronous operations,
including task management, semaphores, throttling, and batch processing.
"""

import time
import asyncio
import functools
import inspect
import contextlib
import uuid
from typing import (
    Dict, List, Any, Optional, Callable, TypeVar, Generic, 
    Awaitable, Union, Set, Tuple, AsyncIterator, Type, cast,
)
from dataclasses import dataclass, field

from tsap.utils.logging import logger

# Type variables
T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')
AsyncCallable = Callable[..., Awaitable[T]]
SyncOrAsyncCallable = Union[Callable[..., T], Callable[..., Awaitable[T]]]


def is_async_function(func: Callable) -> bool:
    """Check if a function is asynchronous.
    
    Args:
        func: Function to check
        
    Returns:
        True if the function is async, False otherwise
    """
    # Check if it's a coroutine function
    if inspect.iscoroutinefunction(func):
        return True
    
    # Check if it's a callable object with an async __call__ method
    if hasattr(func, "__call__") and not isinstance(func, type):
        if inspect.iscoroutinefunction(func.__call__):
            return True
    
    return False


async def call_async_safe(func: SyncOrAsyncCallable[T], *args: Any, **kwargs: Any) -> T:
    """Call a function asynchronously, whether it's async or not.
    
    Args:
        func: Function to call
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Function result
    """
    if is_async_function(func):
        # Function is already async
        return await func(*args, **kwargs)  # type: ignore
    else:
        # Function is synchronous, run in a thread
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: func(*args, **kwargs)  # type: ignore
        )


def ensure_async(func: SyncOrAsyncCallable[T]) -> AsyncCallable[T]:
    """Decorator to ensure a function is asynchronous.
    
    If the function is already async, it's returned as-is.
    If it's synchronous, it's wrapped to be called in a thread.
    
    Args:
        func: Function to wrap
        
    Returns:
        Async version of the function
    """
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        return await call_async_safe(func, *args, **kwargs)
    
    if is_async_function(func):
        return cast(AsyncCallable[T], func)
    else:
        return wrapper


class TaskManager:
    """Manages a group of related asynchronous tasks."""
    
    def __init__(self):
        """Initialize the task manager."""
        self.tasks: Set[asyncio.Task] = set()
        self._results: Dict[str, Any] = {}
        self._errors: Dict[str, Exception] = {}
        self._task_names: Dict[asyncio.Task, str] = {}
        self._lock = asyncio.Lock()
    
    async def create_task(
        self,
        coro: Awaitable[T],
        name: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> asyncio.Task:
        """Create and track a new task.
        
        Args:
            coro: Coroutine to run as a task
            name: Task name for tracking
            timeout: Timeout in seconds
            
        Returns:
            Created task
        """
        async with self._lock:
            if timeout is not None:
                # Wrap the coroutine with a timeout
                timed_coro = asyncio.wait_for(coro, timeout)
                task = asyncio.create_task(self._run_and_track(timed_coro, name))
            else:
                task = asyncio.create_task(self._run_and_track(coro, name))
            
            self.tasks.add(task)
            if name is not None:
                self._task_names[task] = name
            
            # Set up removal of the task when it's done
            task.add_done_callback(self._task_done)
            
            return task
    
    async def _run_and_track(self, coro: Awaitable[T], name: Optional[str]) -> T:
        """Run a coroutine and track its result or error.
        
        Args:
            coro: Coroutine to run
            name: Task name
            
        Returns:
            Coroutine result
            
        Raises:
            Any exception raised by the coroutine
        """
        try:
            result = await coro
            if name is not None:
                async with self._lock:
                    self._results[name] = result
            return result
        except Exception as e:
            if name is not None:
                async with self._lock:
                    self._errors[name] = e
            raise
    
    def _task_done(self, task: asyncio.Task) -> None:
        """Remove a completed task from tracking.
        
        Args:
            task: Completed task
        """
        self.tasks.discard(task)
        if task in self._task_names:
            del self._task_names[task]
    
    async def wait_all(self, timeout: Optional[float] = None) -> Tuple[List[T], List[Exception]]:
        """Wait for all tracked tasks to complete.
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            Tuple of (results, exceptions)
        """
        if not self.tasks:
            return [], []
        
        if timeout is not None:
            # Wait with timeout
            done, pending = await asyncio.wait(
                self.tasks, timeout=timeout, return_when=asyncio.ALL_COMPLETED
            )
            
            # Cancel any pending tasks
            for task in pending:
                task.cancel()
        else:
            # Wait indefinitely
            done = await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Collect results and exceptions
        results = []
        exceptions = []
        
        for task in done:
            try:
                if isinstance(task, asyncio.Task):
                    # Handle task
                    task_result = task.result()
                    results.append(task_result)
                elif isinstance(task, Exception):
                    # Handle exception directly
                    exceptions.append(task)
                else:
                    # Handle regular result
                    results.append(task)
            except Exception as e:
                exceptions.append(e)
        
        return results, exceptions
    
    async def wait_first(self, timeout: Optional[float] = None) -> Tuple[T, str]:
        """Wait for the first task to complete.
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            Tuple of (result, task_name)
            
        Raises:
            asyncio.TimeoutError: If timeout is reached
            ValueError: If no tasks are being tracked
        """
        if not self.tasks:
            raise ValueError("No tasks to wait for")
        
        # Wait for first task to complete
        done, _ = await asyncio.wait(
            self.tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
        )
        
        if not done:
            raise asyncio.TimeoutError("Timeout waiting for tasks")
        
        # Get the first completed task
        task = next(iter(done))
        task_name = self._task_names.get(task, "unnamed")
        
        # Get the result (or raise exception)
        return task.result(), task_name
    
    def get_task_count(self) -> int:
        """Get the number of active tasks.
        
        Returns:
            Number of active tasks
        """
        return len(self.tasks)
    
    def get_result(self, name: str) -> Any:
        """Get the result of a named task.
        
        Args:
            name: Task name
            
        Returns:
            Task result
            
        Raises:
            KeyError: If no result exists for the task
            Exception: If the task raised an exception
        """
        if name in self._errors:
            raise self._errors[name]
        
        if name not in self._results:
            raise KeyError(f"No result for task '{name}'")
        
        return self._results[name]
    
    def cancel_all(self) -> int:
        """Cancel all tracked tasks.
        
        Returns:
            Number of tasks cancelled
        """
        count = 0
        for task in self.tasks:
            if not task.done():
                task.cancel()
                count += 1
        
        return count
    
    async def cancel_and_wait(self) -> int:
        """Cancel all tasks and wait for them to complete.
        
        Returns:
            Number of tasks cancelled
        """
        count = self.cancel_all()
        
        if self.tasks:
            # Wait for all tasks to be cancelled
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        return count
    
    def __enter__(self) -> "TaskManager":
        """Context manager support.
        
        Returns:
            Self
        """
        return self
    
    async def __aenter__(self) -> "TaskManager":
        """Async context manager support.
        
        Returns:
            Self
        """
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - cancel all tasks.
        
        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
        """
        self.cancel_all()
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit - cancel all tasks and wait.
        
        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
        """
        await self.cancel_and_wait()


class AsyncSemaphore:
    """Enhanced semaphore with timeout and statistics."""
    
    def __init__(self, value: int = 1):
        """Initialize the semaphore.
        
        Args:
            value: Initial semaphore value (max concurrent operations)
        """
        self._semaphore = asyncio.Semaphore(value)
        self.max_concurrency = value
        self.acquired = 0
        self.wait_count = 0
        self.total_wait_time = 0.0
        self._lock = asyncio.Lock()
    
    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquire the semaphore.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if acquired, False if timeout
        """
        start_time = time.time()
        
        async with self._lock:
            self.wait_count += 1
        
        if timeout is not None:
            # Try to acquire with timeout
            try:
                acquired = await asyncio.wait_for(self._semaphore.acquire(), timeout)
            except asyncio.TimeoutError:
                # Timeout reached
                async with self._lock:
                    self.wait_count -= 1
                    self.total_wait_time += time.time() - start_time
                return False
        else:
            # Acquire without timeout
            await self._semaphore.acquire()
            acquired = True
        
        if acquired:
            # Update statistics
            wait_time = time.time() - start_time
            async with self._lock:
                self.acquired += 1
                self.wait_count -= 1
                self.total_wait_time += wait_time
        
        return acquired
    
    def release(self) -> None:
        """Release the semaphore."""
        self._semaphore.release()
        
        async def _update_stats() -> None:
            async with self._lock:
                self.acquired -= 1
        
        # Schedule statistics update in the background
        asyncio.create_task(_update_stats())
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get semaphore statistics.
        
        Returns:
            Dictionary with statistics
        """
        async with self._lock:
            return {
                "max_concurrency": self.max_concurrency,
                "current_acquired": self.acquired,
                "waiting": self.wait_count,
                "average_wait_time": self.total_wait_time / (self.wait_count + self.acquired) if (self.wait_count + self.acquired) > 0 else 0,
            }
    
    @contextlib.asynccontextmanager
    async def acquire_context(self, timeout: Optional[float] = None) -> AsyncIterator[bool]:
        """Context manager for acquiring the semaphore.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Yields:
            True if acquired, False if timeout
        """
        acquired = await self.acquire(timeout)
        try:
            yield acquired
        finally:
            if acquired:
                self.release()


class Throttler:
    """Rate limiter for asynchronous operations."""
    
    def __init__(self, rate_limit: float, period: float = 1.0):
        """Initialize the throttler.
        
        Args:
            rate_limit: Maximum number of operations per period
            period: Time period in seconds
        """
        self.rate_limit = rate_limit
        self.period = period
        self.tokens = rate_limit
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: float = 1.0, timeout: Optional[float] = None) -> bool:
        """Acquire tokens from the throttler.
        
        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if tokens acquired, False if timeout
        """
        if tokens > self.rate_limit:
            raise ValueError(f"Cannot acquire {tokens} tokens, maximum is {self.rate_limit}")
        
        start_time = time.time()
        
        while True:
            # Check if we have enough tokens
            async with self._lock:
                self._update_tokens()
                
                if self.tokens >= tokens:
                    # We have enough tokens, consume them
                    self.tokens -= tokens
                    return True
            
            # Calculate wait time for tokens to replenish
            required_tokens = tokens - self.tokens
            wait_time = (required_tokens / self.rate_limit) * self.period
            
            # Check if we've timed out
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed + wait_time > timeout:
                    return False
            
            # Wait for tokens to replenish
            await asyncio.sleep(wait_time)
    
    def _update_tokens(self) -> None:
        """Update available tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update
        
        # Add tokens based on elapsed time
        self.tokens = min(
            self.rate_limit,
            self.tokens + (elapsed / self.period) * self.rate_limit
        )
        
        self.last_update = now
    
    @contextlib.asynccontextmanager
    async def throttle(self, tokens: float = 1.0, timeout: Optional[float] = None) -> AsyncIterator[bool]:
        """Context manager for throttling operations.
        
        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait in seconds
            
        Yields:
            True if throttled, False if timeout
        """
        acquired = await self.acquire(tokens, timeout)
        try:
            yield acquired
        finally:
            pass  # No need to release tokens


class BatchProcessor(Generic[T, U]):
    """Process items in batches asynchronously."""
    
    def __init__(
        self,
        process_func: Callable[[List[T]], Awaitable[List[U]]],
        batch_size: int = 10,
        max_concurrency: int = 5,
    ):
        """Initialize the batch processor.
        
        Args:
            process_func: Function to process a batch of items
            batch_size: Maximum batch size
            max_concurrency: Maximum number of concurrent batches
        """
        self.process_func = process_func
        self.batch_size = batch_size
        self.semaphore = AsyncSemaphore(max_concurrency)
        self.stats = {
            "total_items": 0,
            "successful_items": 0,
            "failed_items": 0,
            "total_batches": 0,
            "successful_batches": 0,
            "failed_batches": 0,
        }
        self._lock = asyncio.Lock()
    
    async def process_items(
        self,
        items: List[T],
        batch_size: Optional[int] = None,
    ) -> List[U]:
        """Process a list of items in batches.
        
        Args:
            items: Items to process
            batch_size: Override default batch size
            
        Returns:
            List of results
        """
        actual_batch_size = batch_size or self.batch_size
        
        # Split items into batches
        batches = [
            items[i:i + actual_batch_size]
            for i in range(0, len(items), actual_batch_size)
        ]
        
        # Create tasks for each batch
        tasks = []
        for batch in batches:
            task = asyncio.create_task(self._process_batch(batch))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results, handling exceptions
        flattened_results = []
        for result in results:
            if isinstance(result, Exception):
                # Skip batches that failed
                continue
            flattened_results.extend(result)
        
        return flattened_results
    
    async def _process_batch(self, batch: List[T]) -> List[U]:
        """Process a single batch of items.
        
        Args:
            batch: Batch of items
            
        Returns:
            Batch results
            
        Raises:
            Exception: If batch processing fails
        """
        # Update stats for this batch
        async with self._lock:
            self.stats["total_items"] += len(batch)
            self.stats["total_batches"] += 1
        
        # Acquire semaphore
        async with self.semaphore.acquire_context() as acquired:
            if not acquired:
                # This should never happen with no timeout
                raise RuntimeError("Failed to acquire semaphore")
            
            try:
                # Process the batch
                results = await self.process_func(batch)
                
                # Update success stats
                async with self._lock:
                    self.stats["successful_items"] += len(batch)
                    self.stats["successful_batches"] += 1
                
                return results
            except Exception as e:
                # Update failure stats
                async with self._lock:
                    self.stats["failed_items"] += len(batch)
                    self.stats["failed_batches"] += 1
                
                raise e
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics.
        
        Returns:
            Dictionary with statistics
        """
        async with self._lock:
            # Get a copy of the stats
            stats_copy = dict(self.stats)
            
            # Add semaphore stats
            semaphore_stats = await self.semaphore.get_stats()
            stats_copy["semaphore"] = semaphore_stats
            
            return stats_copy


class RetryPolicy:
    """Policy for retrying failed operations."""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 0.1,
        max_delay: float = 10.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
    ):
        """Initialize the retry policy.
        
        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            backoff_factor: Factor by which the delay increases each retry
            jitter: Whether to add randomness to the delay
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """Calculate the delay for a retry attempt.
        
        Args:
            attempt: Retry attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        import random
        
        delay = min(
            self.max_delay,
            self.initial_delay * (self.backoff_factor ** attempt)
        )
        
        if self.jitter:
            # Add up to 25% jitter
            jitter_amount = delay * 0.25
            delay = delay + random.uniform(-jitter_amount, jitter_amount)
            
            # Ensure delay is not negative
            delay = max(0.001, delay)
        
        return delay


def retry(
    retry_policy: Optional[RetryPolicy] = None,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
) -> Callable[[AsyncCallable[T]], AsyncCallable[T]]:
    """Decorator to retry asynchronous functions.
    
    Args:
        retry_policy: Retry policy to use
        exceptions: Exception types to retry
        
    Returns:
        Decorator function
    """
    # Use default retry policy if not provided
    policy = retry_policy or RetryPolicy()
    
    def decorator(func: AsyncCallable[T]) -> AsyncCallable[T]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            attempt = 0
            last_error = None
            
            while attempt <= policy.max_retries:
                try:
                    # Attempt to call the function
                    return await func(*args, **kwargs)
                except exceptions as e:
                    # Check if we have retries left
                    attempt += 1
                    last_error = e
                    
                    if attempt > policy.max_retries:
                        # No more retries, re-raise the exception
                        raise
                    
                    # Calculate delay
                    delay = policy.get_delay(attempt - 1)
                    
                    # Log retry attempt
                    logger.warning(
                        f"Retrying '{func.__name__}' after error: {str(e)}. "
                        f"Attempt {attempt}/{policy.max_retries} (delay: {delay:.3f}s)"
                    )
                    
                    # Wait before retrying
                    await asyncio.sleep(delay)
            
            # This should never happen but keeps mypy happy
            assert last_error is not None
            raise last_error
        
        return wrapper
    
    return decorator


@dataclass
class Job(Generic[T]):
    """Represents an asynchronous job."""
    
    id: str
    status: str = "pending"
    result: Optional[T] = None
    error: Optional[Exception] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    progress: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[float]:
        """Get the job duration in seconds.
        
        Returns:
            Duration in seconds or None if not started/completed
        """
        if self.completed_at is not None and self.started_at is not None:
            return self.completed_at - self.started_at
        elif self.started_at is not None:
            return time.time() - self.started_at
        else:
            return None
    
    @property
    def is_complete(self) -> bool:
        """Check if the job is complete.
        
        Returns:
            True if complete, False otherwise
        """
        return self.status in ("completed", "failed", "cancelled")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the job to a dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration": self.duration,
            "progress": self.progress,
            "metadata": self.metadata,
            "error": str(self.error) if self.error else None,
        }


class JobManager(Generic[T]):
    """Manages asynchronous jobs."""
    
    def __init__(self, max_concurrent: int = 10, job_timeout: Optional[float] = None):
        """Initialize the job manager.
        
        Args:
            max_concurrent: Maximum concurrent jobs
            job_timeout: Default job timeout in seconds
        """
        self.jobs: Dict[str, Job[T]] = {}
        self.semaphore = AsyncSemaphore(max_concurrent)
        self.job_timeout = job_timeout
        self._lock = asyncio.Lock()
    
    async def submit(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        timeout: Optional[float] = None,
        job_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Submit a job for execution.
        
        Args:
            func: Async function to execute
            *args: Positional arguments
            timeout: Job timeout in seconds
            job_id: Custom job ID
            metadata: Job metadata
            **kwargs: Keyword arguments
            
        Returns:
            Job ID
        """
        # Generate job ID if not provided
        if job_id is None:
            job_id = generate_id("job-")
        
        # Create the job
        job = Job[T](
            id=job_id,
            status="pending",
            metadata=metadata or {},
        )
        
        # Store the job
        async with self._lock:
            self.jobs[job_id] = job
        
        # Create a task to run the job
        asyncio.create_task(self._run_job(job, func, args, kwargs, timeout))
        
        return job_id
    
    async def _run_job(
        self,
        job: Job[T],
        func: Callable[..., Awaitable[T]],
        args: Any,
        kwargs: Any,
        timeout: Optional[float],
    ) -> None:
        """Run a job.
        
        Args:
            job: Job to run
            func: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            timeout: Job timeout
        """
        # Update job status
        job.status = "waiting"
        
        # Acquire semaphore
        async with self.semaphore.acquire_context() as acquired:
            if not acquired:
                # This should never happen with no timeout
                job.status = "failed"
                job.error = RuntimeError("Failed to acquire semaphore")
                return
            
            # Update job status
            job.status = "running"
            job.started_at = time.time()
            
            try:
                # Run the function with timeout
                actual_timeout = timeout or self.job_timeout
                if actual_timeout is not None:
                    result = await asyncio.wait_for(func(*args, **kwargs), actual_timeout)
                else:
                    result = await func(*args, **kwargs)
                
                # Update job with result
                job.result = result
                job.status = "completed"
                job.completed_at = time.time()
                job.progress = 1.0
            except asyncio.TimeoutError:
                # Job timed out
                job.status = "timeout"
                job.error = asyncio.TimeoutError(f"Job timed out after {actual_timeout} seconds")
                job.completed_at = time.time()
            except asyncio.CancelledError:
                # Job was cancelled
                job.status = "cancelled"
                job.error = asyncio.CancelledError("Job was cancelled")
                job.completed_at = time.time()
            except Exception as e:
                # Job failed
                job.status = "failed"
                job.error = e
                job.completed_at = time.time()
    
    async def get_job(self, job_id: str) -> Optional[Job[T]]:
        """Get a job by ID.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job or None if not found
        """
        async with self._lock:
            return self.jobs.get(job_id)
    
    async def get_job_status(self, job_id: str) -> Optional[str]:
        """Get a job's status.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job status or None if not found
        """
        job = await self.get_job(job_id)
        return job.status if job else None
    
    async def get_job_result(self, job_id: str, wait: bool = False, timeout: Optional[float] = None) -> Optional[T]:
        """Get a job's result.
        
        Args:
            job_id: Job ID
            wait: Whether to wait for the job to complete
            timeout: Maximum time to wait
            
        Returns:
            Job result or None if not found or not completed
            
        Raises:
            TimeoutError: If wait is True and timeout is reached
            Exception: If the job failed
        """
        job = await self.get_job(job_id)
        
        if job is None:
            return None
        
        if wait and not job.is_complete:
            # Wait for the job to complete
            start_time = time.time()
            while not job.is_complete:
                # Check timeout
                if timeout is not None and time.time() - start_time > timeout:
                    raise TimeoutError(f"Timeout waiting for job {job_id}")
                
                # Wait a bit
                await asyncio.sleep(0.1)
                
                # Refresh job
                job = await self.get_job(job_id)
                if job is None:
                    return None
        
        # Check job status
        if job.status == "failed" and job.error is not None:
            raise job.error
        elif job.status == "timeout":
            raise TimeoutError(f"Job {job_id} timed out")
        elif job.status == "cancelled":
            raise asyncio.CancelledError(f"Job {job_id} was cancelled")
        elif job.status != "completed":
            return None
        
        return job.result
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job.
        
        Args:
            job_id: Job ID
            
        Returns:
            True if cancelled, False if not found or already complete
        """
        job = await self.get_job(job_id)
        
        if job is None or job.is_complete:
            return False
        
        # Update job status
        job.status = "cancelled"
        job.error = asyncio.CancelledError("Job was cancelled")
        job.completed_at = time.time()
        
        return True
    
    async def update_progress(self, job_id: str, progress: float) -> bool:
        """Update a job's progress.
        
        Args:
            job_id: Job ID
            progress: Progress value (0.0 to 1.0)
            
        Returns:
            True if updated, False if not found or already complete
        """
        job = await self.get_job(job_id)
        
        if job is None or job.is_complete:
            return False
        
        # Update job progress
        job.progress = max(0.0, min(1.0, progress))
        
        return True
    
    async def cleanup_jobs(self, max_age: float = 3600.0) -> int:
        """Clean up old completed jobs.
        
        Args:
            max_age: Maximum job age in seconds
            
        Returns:
            Number of jobs cleaned up
        """
        now = time.time()
        jobs_to_remove = []
        
        async with self._lock:
            for job_id, job in self.jobs.items():
                # Check if job is complete and old
                if job.is_complete and job.created_at < now - max_age:
                    jobs_to_remove.append(job_id)
            
            # Remove jobs
            for job_id in jobs_to_remove:
                del self.jobs[job_id]
        
        return len(jobs_to_remove)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get job manager statistics.
        
        Returns:
            Dictionary with statistics
        """
        async with self._lock:
            # Count jobs by status
            status_counts = {}
            for job in self.jobs.values():
                status_counts[job.status] = status_counts.get(job.status, 0) + 1
            
            # Get semaphore stats
            semaphore_stats = await self.semaphore.get_stats()
            
            return {
                "total_jobs": len(self.jobs),
                "status_counts": status_counts,
                "semaphore": semaphore_stats,
            }


# Global job manager
_job_manager = JobManager[Any]()


def get_job_manager() -> JobManager[Any]:
    """Get the global job manager.
    
    Returns:
        Global job manager
    """
    return _job_manager


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with an optional prefix.
    
    Args:
        prefix: Optional prefix for the ID
        
    Returns:
        Unique ID string
    """
    unique_id = str(uuid.uuid4())
    return f"{prefix}{unique_id}"