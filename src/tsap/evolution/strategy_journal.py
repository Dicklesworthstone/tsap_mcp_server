"""
Strategy journal for recording and analyzing search and analysis strategies.

This module provides functionality for recording, retrieving, and analyzing
the execution history of search and analysis strategies, including their
effectiveness, execution times, and patterns of success and failure.
"""

import time
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple
import statistics
from collections import defaultdict

import sqlite3
from pathlib import Path

from tsap.utils.logging import logger
from tsap.config import get_config
from tsap.utils.errors import TSAPError
from tsap.mcp.models import (
    StrategyJournalParams, 
    StrategyJournalResult, 
    StrategyJournalEntry
)


class StrategyJournalError(TSAPError):
    """
    Exception for errors in strategy journal operations.
    
    Attributes:
        message: Error message
        operation: Journal operation that caused the error
        details: Additional error details
    """
    def __init__(
        self, 
        message: str, 
        operation: Optional[str] = None, 
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            message, 
            code=f"STRATEGY_JOURNAL_{operation.upper()}_ERROR" if operation else "STRATEGY_JOURNAL_ERROR",
            details=details
        )
        self.operation = operation


class StrategyJournal:
    """
    Journal for recording and analyzing strategy executions.
    """
    def __init__(self, db_path: Optional[str] = None) -> None:
        """
        Initialize the strategy journal.
        
        Args:
            db_path: Path to the journal database file
        """
        # Determine database path
        if db_path is None:
            config = get_config()
            data_dir = config.storage.data_dir
            db_path = str(Path(data_dir) / "strategy_journal.db")
        
        self.db_path = db_path
        self._conn = None
        
        # Initialize the database
        self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Initialize the journal database."""
        conn = self._get_connection()
        
        # Create tables if they don't exist
        cursor = conn.cursor()
        
        # Entries table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS journal_entries (
            entry_id TEXT PRIMARY KEY,
            strategy_id TEXT NOT NULL,
            execution_id TEXT,
            timestamp REAL NOT NULL,
            effectiveness REAL,
            execution_time REAL,
            context TEXT,
            notes TEXT,
            tags TEXT,
            details TEXT
        )
        ''')
        
        # Tags table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS tags (
            tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
            tag TEXT UNIQUE NOT NULL
        )
        ''')
        
        # Entry-tag relationship table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS entry_tags (
            entry_id TEXT,
            tag_id INTEGER,
            PRIMARY KEY (entry_id, tag_id),
            FOREIGN KEY (entry_id) REFERENCES journal_entries(entry_id) ON DELETE CASCADE,
            FOREIGN KEY (tag_id) REFERENCES tags(tag_id) ON DELETE CASCADE
        )
        ''')
        
        # Indices for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entries_strategy_id ON journal_entries(strategy_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entries_timestamp ON journal_entries(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entries_context ON journal_entries(context)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entry_tags_tag_id ON entry_tags(tag_id)')
        
        conn.commit()
    
    def _get_connection(self) -> sqlite3.Connection:
        """
        Get a connection to the database.
        
        Returns:
            SQLite connection
        """
        if self._conn is None:
            # Create parent directory if it doesn't exist
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Connect to the database
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        
        return self._conn
    
    async def add_entry(
        self,
        strategy_id: str,
        execution_id: Optional[str] = None,
        effectiveness: Optional[float] = None,
        execution_time: Optional[float] = None,
        context: Optional[str] = None,
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a journal entry.
        
        Args:
            strategy_id: ID of the strategy
            execution_id: ID of the execution (optional)
            effectiveness: Effectiveness score (0.0-1.0)
            execution_time: Execution time in seconds
            context: Context of the execution (e.g., "security_audit")
            notes: Additional notes
            tags: List of tags
            details: Additional details about the execution
            
        Returns:
            ID of the created entry
        """
        # Generate entry ID
        entry_id = str(uuid.uuid4())
        
        # Get current timestamp
        timestamp = time.time()
        
        # Convert details to JSON
        details_json = json.dumps(details) if details is not None else None
        
        # Convert tags to JSON
        tags_json = json.dumps(tags) if tags is not None else None
        
        # Insert the entry
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            '''
            INSERT INTO journal_entries 
            (entry_id, strategy_id, execution_id, timestamp, effectiveness, 
             execution_time, context, notes, tags, details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                entry_id, strategy_id, execution_id, timestamp, effectiveness,
                execution_time, context, notes, tags_json, details_json
            )
        )
        
        # Add tags if provided
        if tags:
            await self._add_tags(entry_id, tags)
        
        conn.commit()
        
        return entry_id
    
    async def _add_tags(self, entry_id: str, tags: List[str]) -> None:
        """
        Add tags to an entry.
        
        Args:
            entry_id: ID of the entry
            tags: List of tags
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Ensure each tag exists in the tags table
        for tag in tags:
            # Try to insert the tag, ignore if it already exists
            cursor.execute(
                'INSERT OR IGNORE INTO tags (tag) VALUES (?)',
                (tag,)
            )
            
            # Get the tag ID
            cursor.execute('SELECT tag_id FROM tags WHERE tag = ?', (tag,))
            tag_id = cursor.fetchone()[0]
            
            # Add the entry-tag relationship
            cursor.execute(
                'INSERT OR IGNORE INTO entry_tags (entry_id, tag_id) VALUES (?, ?)',
                (entry_id, tag_id)
            )
    
    async def get_entry(self, entry_id: str) -> Optional[StrategyJournalEntry]:
        """
        Get a journal entry by ID.
        
        Args:
            entry_id: ID of the entry
            
        Returns:
            Journal entry or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get the entry
        cursor.execute(
            'SELECT * FROM journal_entries WHERE entry_id = ?',
            (entry_id,)
        )
        
        row = cursor.fetchone()
        if row is None:
            return None
        
        # Convert row to dictionary
        entry_dict = {key: row[key] for key in row.keys()}
        
        # Parse JSON fields
        if entry_dict.get('details'):
            entry_dict['details'] = json.loads(entry_dict['details'])
        
        if entry_dict.get('tags'):
            entry_dict['tags'] = json.loads(entry_dict['tags'])
        
        # Create entry object
        return StrategyJournalEntry(
            entry_id=entry_dict['entry_id'],
            strategy_id=entry_dict['strategy_id'],
            execution_id=entry_dict['execution_id'],
            timestamp=entry_dict['timestamp'],
            effectiveness=entry_dict['effectiveness'],
            execution_time=entry_dict['execution_time'],
            context=entry_dict['context'],
            notes=entry_dict['notes'],
            tags=entry_dict['tags'],
            details=entry_dict['details']
        )
    
    async def update_entry(
        self,
        entry_id: str,
        effectiveness: Optional[float] = None,
        execution_time: Optional[float] = None,
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update a journal entry.
        
        Args:
            entry_id: ID of the entry
            effectiveness: New effectiveness score
            execution_time: New execution time
            notes: New notes
            tags: New tags
            details: New details
            
        Returns:
            True if the entry was updated, False otherwise
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Build update fields
        update_fields = []
        params = []
        
        if effectiveness is not None:
            update_fields.append('effectiveness = ?')
            params.append(effectiveness)
        
        if execution_time is not None:
            update_fields.append('execution_time = ?')
            params.append(execution_time)
        
        if notes is not None:
            update_fields.append('notes = ?')
            params.append(notes)
        
        if tags is not None:
            update_fields.append('tags = ?')
            params.append(json.dumps(tags))
            
            # Update tags
            await self._update_entry_tags(entry_id, tags)
        
        if details is not None:
            update_fields.append('details = ?')
            params.append(json.dumps(details))
        
        if not update_fields:
            # Nothing to update
            return False
        
        # Add entry ID to params
        params.append(entry_id)
        
        # Update the entry
        cursor.execute(
            f'''
            UPDATE journal_entries
            SET {', '.join(update_fields)}
            WHERE entry_id = ?
            ''',
            params
        )
        
        conn.commit()
        
        return cursor.rowcount > 0
    
    async def _update_entry_tags(self, entry_id: str, tags: List[str]) -> None:
        """
        Update the tags for an entry.
        
        Args:
            entry_id: ID of the entry
            tags: New tags
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Remove existing tags
        cursor.execute(
            'DELETE FROM entry_tags WHERE entry_id = ?',
            (entry_id,)
        )
        
        # Add new tags
        await self._add_tags(entry_id, tags)
    
    async def delete_entry(self, entry_id: str) -> bool:
        """
        Delete a journal entry.
        
        Args:
            entry_id: ID of the entry
            
        Returns:
            True if the entry was deleted, False otherwise
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Delete the entry
        cursor.execute(
            'DELETE FROM journal_entries WHERE entry_id = ?',
            (entry_id,)
        )
        
        conn.commit()
        
        return cursor.rowcount > 0
    
    async def search_entries(
        self,
        strategy_id: Optional[str] = None,
        date_range: Optional[Tuple[float, float]] = None,
        context: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_effectiveness: Optional[float] = None,
        max_entries: Optional[int] = None,
        sort_by: str = 'timestamp',
        sort_order: str = 'desc'
    ) -> List[StrategyJournalEntry]:
        """
        Search for journal entries.
        
        Args:
            strategy_id: Filter by strategy ID
            date_range: Filter by date range (start_timestamp, end_timestamp)
            context: Filter by context
            tags: Filter by tags (entries must have all listed tags)
            min_effectiveness: Filter by minimum effectiveness
            max_entries: Maximum number of entries to return
            sort_by: Field to sort by ('timestamp', 'effectiveness', 'execution_time')
            sort_order: Sort order ('asc' or 'desc')
            
        Returns:
            List of matching journal entries
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Build query
        query = 'SELECT * FROM journal_entries'
        where_clauses = []
        params = []
        
        if strategy_id is not None:
            where_clauses.append('strategy_id = ?')
            params.append(strategy_id)
        
        if date_range is not None:
            start_time, end_time = date_range
            where_clauses.append('timestamp BETWEEN ? AND ?')
            params.extend([start_time, end_time])
        
        if context is not None:
            where_clauses.append('context = ?')
            params.append(context)
        
        if min_effectiveness is not None:
            where_clauses.append('effectiveness >= ?')
            params.append(min_effectiveness)
        
        # Add WHERE clause if necessary
        if where_clauses:
            query += ' WHERE ' + ' AND '.join(where_clauses)
        
        # Add ORDER BY clause
        valid_sort_fields = {'timestamp', 'effectiveness', 'execution_time'}
        sort_field = sort_by if sort_by in valid_sort_fields else 'timestamp'
        
        valid_sort_orders = {'asc', 'desc'}
        order = sort_order.lower() if sort_order.lower() in valid_sort_orders else 'desc'
        
        query += f' ORDER BY {sort_field} {order.upper()}'
        
        # Add LIMIT clause if necessary
        if max_entries is not None:
            query += ' LIMIT ?'
            params.append(max_entries)
        
        # Execute query
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Convert rows to entries
        entries = []
        for row in rows:
            entry_dict = {key: row[key] for key in row.keys()}
            
            # Parse JSON fields
            if entry_dict.get('details'):
                entry_dict['details'] = json.loads(entry_dict['details'])
            
            if entry_dict.get('tags'):
                entry_dict['tags'] = json.loads(entry_dict['tags'])
            
            # Filter by tags if necessary
            if tags:
                entry_tags = set(entry_dict.get('tags', []))
                if not all(tag in entry_tags for tag in tags):
                    continue
            
            # Create entry object
            entry = StrategyJournalEntry(
                entry_id=entry_dict['entry_id'],
                strategy_id=entry_dict['strategy_id'],
                execution_id=entry_dict['execution_id'],
                timestamp=entry_dict['timestamp'],
                effectiveness=entry_dict['effectiveness'],
                execution_time=entry_dict['execution_time'],
                context=entry_dict['context'],
                notes=entry_dict['notes'],
                tags=entry_dict['tags'],
                details=entry_dict['details']
            )
            
            entries.append(entry)
        
        return entries
    
    async def get_strategy_history(
        self,
        strategy_id: str,
        max_entries: Optional[int] = None
    ) -> List[StrategyJournalEntry]:
        """
        Get the execution history for a strategy.
        
        Args:
            strategy_id: ID of the strategy
            max_entries: Maximum number of entries to return
            
        Returns:
            List of journal entries for the strategy
        """
        return await self.search_entries(
            strategy_id=strategy_id,
            max_entries=max_entries,
            sort_by='timestamp',
            sort_order='desc'
        )
    
    async def get_strategy_statistics(self, strategy_id: str) -> Dict[str, Any]:
        """
        Get statistics for a strategy.
        
        Args:
            strategy_id: ID of the strategy
            
        Returns:
            Dictionary with statistics
        """
        # Get all entries for the strategy
        entries = await self.get_strategy_history(strategy_id)
        
        if not entries:
            return {
                "strategy_id": strategy_id,
                "total_executions": 0,
                "average_effectiveness": None,
                "average_execution_time": None
            }
        
        # Calculate statistics
        effectiveness_values = [e.effectiveness for e in entries if e.effectiveness is not None]
        execution_time_values = [e.execution_time for e in entries if e.execution_time is not None]
        
        stats = {
            "strategy_id": strategy_id,
            "total_executions": len(entries),
            "first_execution": min(e.timestamp for e in entries),
            "last_execution": max(e.timestamp for e in entries)
        }
        
        if effectiveness_values:
            stats.update({
                "average_effectiveness": statistics.mean(effectiveness_values),
                "min_effectiveness": min(effectiveness_values),
                "max_effectiveness": max(effectiveness_values)
            })
            
            if len(effectiveness_values) > 1:
                stats["effectiveness_stdev"] = statistics.stdev(effectiveness_values)
        
        if execution_time_values:
            stats.update({
                "average_execution_time": statistics.mean(execution_time_values),
                "min_execution_time": min(execution_time_values),
                "max_execution_time": max(execution_time_values)
            })
            
            if len(execution_time_values) > 1:
                stats["execution_time_stdev"] = statistics.stdev(execution_time_values)
        
        # Count contexts
        context_counts = defaultdict(int)
        for entry in entries:
            if entry.context:
                context_counts[entry.context] += 1
        
        stats["contexts"] = dict(context_counts)
        
        # Count tags
        tag_counts = defaultdict(int)
        for entry in entries:
            if entry.tags:
                for tag in entry.tags:
                    tag_counts[tag] += 1
        
        stats["tags"] = dict(tag_counts)
        
        return stats
    
    async def get_execution_trends(
        self,
        strategy_id: Optional[str] = None,
        time_period: str = 'day',
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get execution trends over time.
        
        Args:
            strategy_id: Filter by strategy ID (optional)
            time_period: Time period for grouping ('hour', 'day', 'week', 'month')
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            
        Returns:
            Dictionary with trend data
        """
        # Define time period in seconds
        period_seconds = {
            'hour': 3600,
            'day': 86400,
            'week': 604800,
            'month': 2592000
        }.get(time_period, 86400)  # Default to day
        
        # Set default time range if not provided
        if end_time is None:
            end_time = time.time()
        
        if start_time is None:
            # Default to last 10 periods
            start_time = end_time - (10 * period_seconds)
        
        # Get entries within the time range
        where_clauses = ['timestamp BETWEEN ? AND ?']
        params = [start_time, end_time]
        
        if strategy_id is not None:
            where_clauses.append('strategy_id = ?')
            params.append(strategy_id)
        
        # Query for entries
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = f'''
        SELECT * FROM journal_entries
        WHERE {' AND '.join(where_clauses)}
        ORDER BY timestamp ASC
        '''
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Group entries by time period
        period_groups = defaultdict(list)
        
        for row in rows:
            entry_dict = {key: row[key] for key in row.keys()}
            
            # Parse JSON fields
            if entry_dict.get('details'):
                entry_dict['details'] = json.loads(entry_dict['details'])
            
            if entry_dict.get('tags'):
                entry_dict['tags'] = json.loads(entry_dict['tags'])
            
            # Create entry object
            entry = StrategyJournalEntry(
                entry_id=entry_dict['entry_id'],
                strategy_id=entry_dict['strategy_id'],
                execution_id=entry_dict['execution_id'],
                timestamp=entry_dict['timestamp'],
                effectiveness=entry_dict['effectiveness'],
                execution_time=entry_dict['execution_time'],
                context=entry_dict['context'],
                notes=entry_dict['notes'],
                tags=entry_dict['tags'],
                details=entry_dict['details']
            )
            
            # Determine period key
            period_key = int(entry.timestamp / period_seconds) * period_seconds
            period_groups[period_key].append(entry)
        
        # Calculate statistics for each period
        trend_data = []
        
        for period_key, entries in sorted(period_groups.items()):
            # Calculate statistics
            effectiveness_values = [e.effectiveness for e in entries if e.effectiveness is not None]
            execution_time_values = [e.execution_time for e in entries if e.execution_time is not None]
            
            period_stats = {
                "period_start": period_key,
                "period_end": period_key + period_seconds,
                "entry_count": len(entries)
            }
            
            if effectiveness_values:
                period_stats["average_effectiveness"] = statistics.mean(effectiveness_values)
            
            if execution_time_values:
                period_stats["average_execution_time"] = statistics.mean(execution_time_values)
            
            trend_data.append(period_stats)
        
        # Fill in missing periods
        all_periods = []
        current_period = start_time - (start_time % period_seconds)
        
        while current_period < end_time:
            existing = next((p for p in trend_data if p["period_start"] == current_period), None)
            
            if existing:
                all_periods.append(existing)
            else:
                all_periods.append({
                    "period_start": current_period,
                    "period_end": current_period + period_seconds,
                    "entry_count": 0,
                    "average_effectiveness": None,
                    "average_execution_time": None
                })
            
            current_period += period_seconds
        
        return {
            "strategy_id": strategy_id,
            "time_period": time_period,
            "start_time": start_time,
            "end_time": end_time,
            "trend_data": all_periods
        }
    
    async def analyze_strategy_comparison(
        self,
        strategy_ids: List[str],
        min_executions: int = 5
    ) -> Dict[str, Any]:
        """
        Analyze and compare multiple strategies.
        
        Args:
            strategy_ids: List of strategy IDs to compare
            min_executions: Minimum number of executions required for comparison
            
        Returns:
            Dictionary with comparison results
        """
        # Get statistics for each strategy
        strategy_stats = {}
        
        for strategy_id in strategy_ids:
            stats = await self.get_strategy_statistics(strategy_id)
            
            if stats["total_executions"] >= min_executions:
                strategy_stats[strategy_id] = stats
        
        if not strategy_stats:
            return {
                "strategy_ids": strategy_ids,
                "eligible_strategies": 0,
                "message": f"No strategies with at least {min_executions} executions"
            }
        
        # Compare effectiveness
        effectiveness_comparison = {}
        for strategy_id, stats in strategy_stats.items():
            if "average_effectiveness" in stats:
                effectiveness_comparison[strategy_id] = stats["average_effectiveness"]
        
        best_effectiveness = None
        best_strategy = None
        
        if effectiveness_comparison:
            best_strategy, best_effectiveness = max(
                effectiveness_comparison.items(), 
                key=lambda x: x[1]
            )
        
        # Compare execution time
        execution_time_comparison = {}
        for strategy_id, stats in strategy_stats.items():
            if "average_execution_time" in stats:
                execution_time_comparison[strategy_id] = stats["average_execution_time"]
        
        fastest_strategy = None
        fastest_time = None
        
        if execution_time_comparison:
            fastest_strategy, fastest_time = min(
                execution_time_comparison.items(), 
                key=lambda x: x[1]
            )
        
        # Overall comparison
        return {
            "strategy_ids": strategy_ids,
            "eligible_strategies": len(strategy_stats),
            "strategy_stats": strategy_stats,
            "effectiveness_comparison": effectiveness_comparison,
            "best_effectiveness": {
                "strategy_id": best_strategy,
                "value": best_effectiveness
            } if best_strategy else None,
            "execution_time_comparison": execution_time_comparison,
            "fastest_execution": {
                "strategy_id": fastest_strategy,
                "value": fastest_time
            } if fastest_strategy else None
        }
    
    async def get_all_tags(self) -> List[str]:
        """
        Get all tags used in the journal.
        
        Returns:
            List of all tags
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT tag FROM tags ORDER BY tag')
        rows = cursor.fetchall()
        
        return [row[0] for row in rows]
    
    async def get_tag_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about tag usage.
        
        Returns:
            Dictionary with tag statistics
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get tag counts
        cursor.execute('''
        SELECT t.tag, COUNT(et.entry_id) as entry_count
        FROM tags t
        JOIN entry_tags et ON t.tag_id = et.tag_id
        GROUP BY t.tag_id
        ORDER BY entry_count DESC
        ''')
        
        rows = cursor.fetchall()
        
        tag_counts = {row[0]: row[1] for row in rows}
        
        return {
            "total_tags": len(tag_counts),
            "tag_counts": tag_counts
        }
    
    async def get_context_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about context usage.
        
        Returns:
            Dictionary with context statistics
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get context counts
        cursor.execute('''
        SELECT context, COUNT(*) as entry_count
        FROM journal_entries
        WHERE context IS NOT NULL
        GROUP BY context
        ORDER BY entry_count DESC
        ''')
        
        rows = cursor.fetchall()
        
        context_counts = {row[0]: row[1] for row in rows}
        
        return {
            "total_contexts": len(context_counts),
            "context_counts": context_counts
        }
    
    async def get_journal_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the journal.
        
        Returns:
            Dictionary with journal summary
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get total entries
        cursor.execute('SELECT COUNT(*) FROM journal_entries')
        total_entries = cursor.fetchone()[0]
        
        # Get unique strategies
        cursor.execute('SELECT COUNT(DISTINCT strategy_id) FROM journal_entries')
        strategy_count = cursor.fetchone()[0]
        
        # Get date range
        cursor.execute('SELECT MIN(timestamp), MAX(timestamp) FROM journal_entries')
        min_timestamp, max_timestamp = cursor.fetchone()
        
        # Get total tags
        cursor.execute('SELECT COUNT(*) FROM tags')
        tag_count = cursor.fetchone()[0]
        
        return {
            "total_entries": total_entries,
            "strategy_count": strategy_count,
            "date_range": (min_timestamp, max_timestamp) if min_timestamp else None,
            "tag_count": tag_count
        }
    
    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


# Global instance
_strategy_journal = None


def get_strategy_journal() -> StrategyJournal:
    """
    Get or create the global StrategyJournal instance.
    
    Returns:
        StrategyJournal instance
    """
    global _strategy_journal
    if _strategy_journal is None:
        _strategy_journal = StrategyJournal()
    return _strategy_journal


async def record_journal_entry(params: StrategyJournalParams) -> StrategyJournalResult:
    """
    Record a strategy journal entry.
    
    Args:
        params: Journal entry parameters
            - strategy_id: ID of the strategy
            - execution_id: ID of the execution (optional)
            - effectiveness: Effectiveness score (0.0-1.0)
            - execution_time: Execution time in seconds
            - context: Context of the execution (e.g., "security_audit")
            - notes: Additional notes
            - tags: List of tags
            - details: Additional details about the execution
            
    Returns:
        Journal entry result
    """
    # Get strategy journal
    journal = get_strategy_journal()
    
    try:
        # Add the entry
        entry_id = await journal.add_entry(
            strategy_id=params.strategy_id,
            execution_id=params.execution_id,
            effectiveness=params.effectiveness,
            execution_time=params.execution_time,
            context=params.context,
            notes=params.notes,
            tags=params.tags,
            details=params.details
        )
        
        # Get the entry
        entry = await journal.get_entry(entry_id)
        
        # Return result
        return StrategyJournalResult(
            success=True,
            entry_id=entry_id,
            entry=entry,
            message="Journal entry recorded successfully",
            timestamp=time.time()
        )
        
    except Exception as e:
        # Log the error
        logger.error(f"Error recording journal entry: {str(e)}")
        
        # Return error result
        return StrategyJournalResult(
            success=False,
            message=f"Error recording journal entry: {str(e)}",
            timestamp=time.time()
        )


async def analyze_journal(
    strategy_id: Optional[str] = None,
    date_range: Optional[Tuple[float, float]] = None,
    tags: Optional[List[str]] = None,
    context: Optional[str] = None,
    analysis_type: str = "statistics"
) -> Dict[str, Any]:
    """
    Analyze strategy journal entries.
    
    Args:
        strategy_id: ID of the strategy to analyze (optional)
        date_range: Date range for analysis (start_time, end_time)
        tags: Filter by tags
        context: Filter by context
        analysis_type: Type of analysis to perform ("statistics", "trends", "comparison")
        
    Returns:
        Dictionary with analysis results
    """
    # Get strategy journal
    journal = get_strategy_journal()
    
    try:
        # Perform analysis based on type
        if analysis_type == "statistics":
            if strategy_id:
                # Get statistics for a specific strategy
                return await journal.get_strategy_statistics(strategy_id)
            else:
                # Get general journal summary
                return await journal.get_journal_summary()
                
        elif analysis_type == "trends":
            # Get execution trends
            return await journal.get_execution_trends(
                strategy_id=strategy_id,
                start_time=date_range[0] if date_range else None,
                end_time=date_range[1] if date_range else None
            )
            
        elif analysis_type == "comparison":
            # For comparison, we need multiple strategy IDs
            if not strategy_id:
                # Get all strategies with sufficient executions
                conn = journal._get_connection()
                cursor = conn.cursor()
                
                cursor.execute('''
                SELECT strategy_id, COUNT(*) as entry_count
                FROM journal_entries
                GROUP BY strategy_id
                HAVING entry_count >= 5
                ORDER BY entry_count DESC
                LIMIT 10
                ''')
                
                rows = cursor.fetchall()
                strategy_ids = [row[0] for row in rows]
            else:
                # Just compare the specified strategy with others
                strategy_ids = [strategy_id]
                
                # Find similar strategies
                if context:
                    # Find strategies used in the same context
                    conn = journal._get_connection()
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                    SELECT DISTINCT strategy_id
                    FROM journal_entries
                    WHERE context = ? AND strategy_id != ?
                    LIMIT 5
                    ''', (context, strategy_id))
                    
                    rows = cursor.fetchall()
                    strategy_ids.extend([row[0] for row in rows])
            
            # Perform comparison
            return await journal.analyze_strategy_comparison(strategy_ids)
            
        elif analysis_type == "tags":
            # Get tag statistics
            return await journal.get_tag_statistics()
            
        elif analysis_type == "contexts":
            # Get context statistics
            return await journal.get_context_statistics()
            
        else:
            raise StrategyJournalError(
                message=f"Unknown analysis type: {analysis_type}",
                operation="analyze"
            )
            
    except Exception as e:
        # Log the error
        logger.error(f"Error analyzing journal: {str(e)}")
        
        # Return error result
        return {
            "success": False,
            "message": f"Error analyzing journal: {str(e)}",
            "timestamp": time.time()
        }