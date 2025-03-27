"""
Progress tracking and visualization for TSAP.

This module provides enhanced progress tracking capabilities with Rich,
supporting nested tasks, task groups, and dynamic progress updates.
"""
from typing import Dict, List, Optional, Any, Iterable, TypeVar
from contextlib import contextmanager
import time
from dataclasses import dataclass, field

from rich.progress import (
    Progress, BarColumn, TextColumn, TimeElapsedColumn,
    TimeRemainingColumn, SpinnerColumn
)
from rich.console import Console, Group
from rich.text import Text
from rich.live import Live

from .emojis import get_emoji

# TypeVar for generic progress tracking over iterables
T = TypeVar("T")

@dataclass
class ProgressContext:
    """Context object for tracking progress state."""
    total: int = 0
    completed: int = 0
    current_task: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    success_count: int = 0
    error_count: int = 0
    warning_count: int = 0
    skipped_count: int = 0
    tasks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    subtasks: Dict[str, List[str]] = field(default_factory=lambda: {})
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time
    
    @property
    def completion_percentage(self) -> float:
        """Get completion percentage."""
        if self.total == 0:
            return 0.0
        return (self.completed / self.total) * 100
    
    def add_task(
        self,
        name: str,
        total: int = 100,
        parent: Optional[str] = None,
        status: str = "pending"
    ) -> str:
        """Add a new task to the progress context.
        
        Args:
            name: Unique task name/identifier
            total: Total work units for this task
            parent: Optional parent task name
            status: Initial task status
            
        Returns:
            The task name
        """
        self.tasks[name] = {
            "total": total,
            "completed": 0,
            "status": status,
            "start_time": time.time(),
            "end_time": None,
            "parent": parent,
        }
        
        # Update parent-child relationships
        if parent:
            if parent not in self.subtasks:
                self.subtasks[parent] = []
            self.subtasks[parent].append(name)
            
        # Update global totals
        self.total += total
        
        return name
    
    def update_task(
        self,
        name: str,
        advance: int = 0,
        completed: Optional[int] = None,
        status: Optional[str] = None
    ) -> None:
        """Update a task's progress.
        
        Args:
            name: Task name/identifier
            advance: Increment completed by this amount
            completed: Set completed to this specific value
            status: Update task status
        """
        if name not in self.tasks:
            raise ValueError(f"Task '{name}' not found")
            
        task = self.tasks[name]
        
        # Update completion
        if completed is not None:
            # Calculate the delta for global completion
            delta = completed - task["completed"]
            task["completed"] = completed
            self.completed += delta
        elif advance > 0:
            task["completed"] += advance
            self.completed += advance
            
            # Cap at total
            if task["completed"] > task["total"]:
                # Adjust global completion for the cap
                self.completed -= (task["completed"] - task["total"])
                task["completed"] = task["total"]
        
        # Update status
        if status:
            old_status = task["status"]
            task["status"] = status
            
            # Update counters based on status transitions
            if status == "success" and old_status != "success":
                self.success_count += 1
            elif status == "error" and old_status != "error":
                self.error_count += 1
            elif status == "warning" and old_status != "warning":
                self.warning_count += 1
            elif status == "skipped" and old_status != "skipped":
                self.skipped_count += 1
                
            # Mark task as complete if terminal status
            if status in ("success", "error", "skipped"):
                if task["completed"] < task["total"]:
                    delta = task["total"] - task["completed"]
                    task["completed"] = task["total"]
                    self.completed += delta
                    
                # Set end time if not already set
                if task["end_time"] is None:
                    task["end_time"] = time.time()
    
    def complete_task(self, name: str, status: str = "success") -> None:
        """Mark a task as complete.
        
        Args:
            name: Task name/identifier
            status: Completion status (success, error, warning, skipped)
        """
        if name not in self.tasks:
            raise ValueError(f"Task '{name}' not found")
            
        task = self.tasks[name]
        
        # Calculate how many units are remaining
        remaining = task["total"] - task["completed"]
        
        # Update the task completion and status
        self.update_task(name, advance=remaining, status=status)
        
        # Recursively complete any subtasks
        if name in self.subtasks:
            for subtask in self.subtasks[name]:
                self.complete_task(subtask, status)
    
    def get_task_completion(self, name: str) -> float:
        """Get completion percentage for a specific task.
        
        Args:
            name: Task name/identifier
            
        Returns:
            Completion percentage (0-100)
        """
        if name not in self.tasks:
            raise ValueError(f"Task '{name}' not found")
            
        task = self.tasks[name]
        if task["total"] == 0:
            return 100.0  # Avoid division by zero
            
        return (task["completed"] / task["total"]) * 100
        
    def reset(self) -> None:
        """Reset all progress tracking."""
        self.total = 0
        self.completed = 0
        self.current_task = None
        self.start_time = time.time()
        self.success_count = 0
        self.error_count = 0
        self.warning_count = 0
        self.skipped_count = 0
        self.tasks.clear()
        self.subtasks.clear()

class TSAPProgress:
    """Enhanced progress tracker for TSAP operations."""
    
    def __init__(
        self,
        console: Optional[Console] = None,
        transient: bool = False,
        auto_refresh: bool = True,
        expand: bool = True,
    ):
        """Initialize the progress tracker.
        
        Args:
            console: Rich console to use
            transient: Whether progress display disappears after completion
            auto_refresh: Whether to automatically refresh the display
            expand: Whether to expand the progress display to full width
        """
        self.console = console or globals().get("console")
        self.context = ProgressContext()
        self.transient = transient
        self.auto_refresh = auto_refresh
        self.expand = expand
        self._live = None
        self._progress = None
        self._task_ids = {}  # Maps our task names to Rich TaskIDs
    
    def _create_progress(self) -> Progress:
        """Create the Rich Progress instance.
        
        Returns:
            Configured Progress instance
        """
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=self.transient,
            expand=self.expand,
        )
    
    def _render_summary(self) -> Group:
        """Render a summary of the current progress state.
        
        Returns:
            Rich Group containing the progress summary
        """
        # This will create a progress summary showing overall completion,
        # success/error counts, and elapsed time
        summary_text = Text()
        
        # Overall progress
        emoji = get_emoji("status", "running")
        summary_text.append(f"{emoji} Overall Progress: ")
        summary_text.append(
            f"{self.context.completion_percentage:.1f}% ",
            style="bold cyan"
        )
        summary_text.append(f"({self.context.completed}/{self.context.total})")
        
        # Status counts if any
        if any([
            self.context.success_count,
            self.context.error_count,
            self.context.warning_count,
            self.context.skipped_count
        ]):
            summary_text.append(" | ")
            
            if self.context.success_count:
                emoji = get_emoji("level", "success")
                summary_text.append(
                    f"{emoji} {self.context.success_count} ",
                    style="success"
                )
                
            if self.context.error_count:
                emoji = get_emoji("level", "error")
                summary_text.append(
                    f"{emoji} {self.context.error_count} ",
                    style="error"
                )
                
            if self.context.warning_count:
                emoji = get_emoji("level", "warning")
                summary_text.append(
                    f"{emoji} {self.context.warning_count} ",
                    style="warning"
                )
                
            if self.context.skipped_count:
                emoji = get_emoji("status", "skipped")
                summary_text.append(
                    f"{emoji} {self.context.skipped_count} ",
                    style="muted"
                )
        
        # Elapsed time
        summary_text.append(
            f" | ⏱️ {int(self.context.elapsed)}s",
            style="bright_black"
        )
        
        # Group with the summary text
        return Group(summary_text)
    
    def add_task(
        self,
        description: str,
        name: Optional[str] = None,
        total: int = 100,
        parent: Optional[str] = None,
        visible: bool = True,
    ) -> str:
        """Add a task to the progress tracker.
        
        Args:
            description: User-visible task description
            name: Unique task identifier (defaults to description if not provided)
            total: Total work units for this task
            parent: Optional parent task name
            visible: Whether this task is visible in the progress display
            
        Returns:
            Task name/identifier
        """
        # Use description as name if not provided
        name = name or description
        
        # Add to our context
        self.context.add_task(name, total, parent, status="pending")
        
        # Add to Rich progress if we're active
        if self._progress:
            task_id = self._progress.add_task(description, total=total, visible=visible)
            self._task_ids[name] = task_id
            
        return name
    
    def update_task(
        self,
        name: str,
        description: Optional[str] = None,
        advance: int = 0,
        completed: Optional[int] = None,
        total: Optional[int] = None,
        visible: Optional[bool] = None,
        status: Optional[str] = None,
    ) -> None:
        """Update a task's progress.
        
        Args:
            name: Task name/identifier
            description: Update the displayed description
            advance: Increment completed by this amount
            completed: Set completed to this specific value
            total: Update the total work units
            visible: Update task visibility
            status: Update task status
        """
        # Update our context
        self.context.update_task(name, advance, completed, status)
        
        # Update Rich progress if we're active and have this task
        if self._progress and name in self._task_ids:
            task_id = self._task_ids[name]
            update_args = {}
            
            if description:
                update_args["description"] = description
                
            if advance:
                update_args["advance"] = advance
                
            if completed is not None:
                update_args["completed"] = completed
                
            if total is not None:
                update_args["total"] = total
                
            if visible is not None:
                update_args["visible"] = visible
                
            self._progress.update(task_id, **update_args)
    
    def complete_task(
        self, name: str, status: str = "success", complete_subtasks: bool = True
    ) -> None:
        """Mark a task as complete.
        
        Args:
            name: Task name/identifier
            status: Completion status (success, error, warning, skipped)
            complete_subtasks: Whether to also complete all subtasks
        """
        if complete_subtasks:
            self.context.complete_task(name, status)
        else:
            # Just complete this specific task
            task = self.context.tasks.get(name)
            if task:
                remaining = task["total"] - task["completed"]
                self.context.update_task(name, advance=remaining, status=status)
        
        # Update Rich progress if we're active and have this task
        if self._progress and name in self._task_ids:
            task_id = self._task_ids[name]
            self._progress.update(task_id, completed=self.context.tasks[name]["total"])
    
    def start(self) -> "TSAPProgress":
        """Start displaying progress.
        
        Returns:
            Self for method chaining
        """
        if self._live is None:
            self._progress = self._create_progress()
            
            # Create an initial Live display that includes both the
            # progress tracker and our summary
            def renderable():
                return Group(
                    self._progress,
                    self._render_summary(),
                )
            
            self._live = Live(
                renderable(),
                console=self.console,
                refresh_per_second=10 if self.auto_refresh else 0,
                transient=self.transient,
            )
            self._live.start()
            
            # Add any existing tasks to the progress display
            for name, task in self.context.tasks.items():
                # Extract the description - if we don't have a better one, use the name
                description = name
                
                # Create task in the Rich progress
                task_id = self._progress.add_task(
                    description,
                    total=task["total"],
                    completed=task["completed"],
                )
                self._task_ids[name] = task_id
                
        return self
    
    def stop(self) -> None:
        """Stop displaying progress."""
        if self._live:
            self._live.stop()
            self._live = None
            self._progress = None
            self._task_ids.clear()
    
    def update(self) -> None:
        """Force an update of the progress display."""
        if self._live:
            # Update the live display with the latest progress and summary
            self._live.update(
                Group(
                    self._progress,
                    self._render_summary(),
                )
            )
    
    def reset(self) -> None:
        """Reset all progress tracking and restart the display."""
        was_active = self._live is not None
        
        if was_active:
            self.stop()
            
        self.context.reset()
        
        if was_active:
            self.start()
    
    @contextmanager
    def task(
        self,
        description: str,
        name: Optional[str] = None,
        total: int = 100,
        parent: Optional[str] = None,
        autostart: bool = True,
    ):
        """Context manager for a task with automatic start/complete.
        
        Args:
            description: User-visible task description
            name: Unique task identifier (defaults to description if not provided)
            total: Total work units for this task
            parent: Optional parent task name
            autostart: Whether to start progress tracking if not already started
            
        Yields:
            Task name
        """
        # Start progress tracking if not started and autostart is True
        if autostart and self._live is None:
            self.start()
            
        # Create the task
        task_name = self.add_task(description, name, total, parent)
        
        try:
            # Set as current task
            self.context.current_task = task_name
            
            # Yield control back with the task name
            yield task_name
            
            # Mark as success if no exception
            self.complete_task(task_name, status="success")
            
        except Exception:
            # Mark as error on exception
            self.complete_task(task_name, status="error")
            # Re-raise the exception
            raise
            
        finally:
            # Reset current task
            self.context.current_task = None
    
    def track(
        self,
        iterable: Iterable[T],
        description: str,
        name: Optional[str] = None,
        total: Optional[int] = None,
        parent: Optional[str] = None,
    ) -> Iterable[T]:
        """Track progress through an iterable.
        
        Args:
            iterable: The iterable to track
            description: User-visible task description
            name: Unique task identifier
            total: Total items (obtained from iterable if not provided)
            parent: Optional parent task name
            
        Yields:
            Items from the iterable
        """
        # Try to get length if total not provided
        if total is None:
            try:
                total = len(iterable)
            except (TypeError, AttributeError):
                # If we can't get the length, use 0 which means indeterminate
                total = 0
                
        # Create the task
        task_name = self.add_task(description, name, total, parent)
        
        # Start tracking if not already
        if self._live is None:
            self.start()
            
        # Yield items and update progress
        for item in iterable:
            yield item
            self.update_task(task_name, advance=1)
            
        # Ensure task is completed
        if task_name in self.context.tasks:
            task = self.context.tasks[task_name]
            if task["completed"] < task["total"]:
                self.complete_task(task_name)
    
    def __enter__(self) -> "TSAPProgress":
        """Start progress tracking when used as a context manager."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop progress tracking when exiting the context."""
        self.stop()

# Create a default global progress tracker
progress = TSAPProgress()

# Convenience function for tracking iterables with the global tracker
def track(
    iterable: Iterable[T],
    description: str,
    name: Optional[str] = None,
    total: Optional[int] = None,
    parent: Optional[str] = None,
) -> Iterable[T]:
    """Track progress through an iterable using the global progress tracker.
    
    Args:
        iterable: The iterable to track
        description: User-visible task description
        name: Unique task identifier
        total: Total items (obtained from iterable if not provided)
        parent: Optional parent task name
        
    Yields:
        Items from the iterable
    """
    return progress.track(iterable, description, name, total, parent)

@contextmanager
def task(
    description: str,
    name: Optional[str] = None,
    total: int = 100,
    parent: Optional[str] = None,
):
    """Context manager for a task with the global progress tracker.
    
    Args:
        description: User-visible task description
        name: Unique task identifier
        total: Total work units for this task
        parent: Optional parent task name
        
    Yields:
        Task name
    """
    with progress.task(description, name, total, parent) as task_name:
        yield task_name