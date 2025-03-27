"""
Pattern Library for TSAP.

This module provides functionality to store, retrieve, and manage search patterns,
building an institutional knowledge base of effective patterns.
"""
import os
import asyncio
import sqlite3
import uuid
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import re

from tsap.utils.logging import logger
from tsap.config import get_config
from tsap.evolution.pattern_analyzer import PatternStats


class PatternLibrary:
    """Library for storing and retrieving search patterns."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the pattern library.

        Args:
            db_path: Optional path to pattern library database
        """
        # Determine database path
        self.db_path = db_path
        if not self.db_path:
            config = get_config()
            storage_dir = Path(os.path.expanduser(config.storage_directory))
            storage_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(storage_dir / "pattern_library.db")

        # Initialize database
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Initialize the pattern library database."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create tables if they don't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS patterns (
                id TEXT PRIMARY KEY,
                pattern TEXT NOT NULL,
                description TEXT,
                tags TEXT,
                category TEXT,
                is_regex INTEGER NOT NULL,
                case_sensitive INTEGER NOT NULL,
                confidence REAL NOT NULL,
                effectiveness_score REAL NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                usage_count INTEGER NOT NULL DEFAULT 0,
                parent_pattern_id TEXT,
                FOREIGN KEY (parent_pattern_id) REFERENCES patterns (id)
            )
            ''')

            cursor.execute('''
            CREATE TABLE IF NOT EXISTS pattern_stats (
                id TEXT PRIMARY KEY,
                pattern_id TEXT NOT NULL,
                total_matches INTEGER NOT NULL,
                files_with_matches INTEGER NOT NULL,
                precision REAL NOT NULL,
                recall REAL NOT NULL,
                f1_score REAL NOT NULL,
                avg_context_relevance REAL NOT NULL,
                execution_time REAL NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (pattern_id) REFERENCES patterns (id)
            )
            ''')

            cursor.execute('''
            CREATE TABLE IF NOT EXISTS pattern_examples (
                id TEXT PRIMARY KEY,
                pattern_id TEXT NOT NULL,
                example_text TEXT NOT NULL,
                file_path TEXT,
                line_number INTEGER,
                is_positive INTEGER NOT NULL,
                added_at TEXT NOT NULL,
                FOREIGN KEY (pattern_id) REFERENCES patterns (id)
            )
            ''')

            cursor.execute('''
            CREATE TABLE IF NOT EXISTS tags (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                created_at TEXT NOT NULL
            )
            ''')

            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_patterns_tags ON patterns (tags)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_patterns_category ON patterns (category)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_pattern_stats_pattern_id ON pattern_stats (pattern_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_pattern_examples_pattern_id ON pattern_examples (pattern_id)')

            conn.commit()

            logger.debug(
                f"Pattern library database initialized: {self.db_path}",
                component="evolution",
                operation="init_pattern_library"
            )

        except sqlite3.Error as e:
            logger.error(
                f"Failed to initialize pattern library database: {str(e)}",
                component="evolution",
                operation="init_pattern_library",
                exception=e
            )
            raise

        finally:
            if conn:
                conn.close()

    async def add_pattern(
        self,
        pattern: str,
        description: str,
        is_regex: bool,
        case_sensitive: bool,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
        confidence: float = 1.0,
        effectiveness_score: float = 0.0,
        parent_pattern_id: Optional[str] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
        stats: Optional[PatternStats] = None,
    ) -> str:
        """Add a pattern to the library.

        Args:
            pattern: Pattern string
            description: Pattern description
            is_regex: Whether the pattern is a regex
            case_sensitive: Whether the pattern is case-sensitive
            tags: Optional tags for the pattern
            category: Optional pattern category
            confidence: Confidence in the pattern (0.0-1.0)
            effectiveness_score: Pattern effectiveness score (0.0-1.0)
            parent_pattern_id: Optional ID of parent pattern
            examples: Optional list of example matches
            stats: Optional pattern statistics

        Returns:
            Pattern ID
        """
        # Generate a new ID
        pattern_id = str(uuid.uuid4())

        # Generate timestamps
        now = datetime.utcnow().isoformat()

        # Process tags
        tags_str = None
        if tags:
            # Sort and join tags
            tags_str = ",".join(sorted(tags))

            # Ensure tags exist in the tags table
            await self._ensure_tags_exist(tags)

        # Execute in a thread pool
        loop = asyncio.get_event_loop()

        async def _add_pattern():
            conn = None
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                # Insert the pattern
                cursor.execute(
                    '''
                    INSERT INTO patterns (
                        id, pattern, description, tags, category,
                        is_regex, case_sensitive, confidence, effectiveness_score,
                        created_at, updated_at, parent_pattern_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        pattern_id, pattern, description, tags_str, category,
                        1 if is_regex else 0, 1 if case_sensitive else 0,
                        confidence, effectiveness_score,
                        now, None, parent_pattern_id
                    )
                )

                # Add examples if provided
                if examples:
                    for example in examples:
                        example_id = str(uuid.uuid4())
                        cursor.execute(
                            '''
                            INSERT INTO pattern_examples (
                                id, pattern_id, example_text, file_path, line_number,
                                is_positive, added_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            ''',
                            (
                                example_id, pattern_id,
                                example.get("text", ""),
                                example.get("file_path"),
                                example.get("line_number"),
                                1 if example.get("is_positive", True) else 0,
                                now
                            )
                        )

                # Add stats if provided
                if stats:
                    stats_id = str(uuid.uuid4())
                    cursor.execute(
                        '''
                        INSERT INTO pattern_stats (
                            id, pattern_id, total_matches, files_with_matches,
                            precision, recall, f1_score, avg_context_relevance,
                            execution_time, timestamp
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''',
                        (
                            stats_id, pattern_id,
                            stats.total_matches, stats.files_with_matches,
                            stats.precision, stats.recall, stats.f1_score,
                            stats.avg_context_relevance,
                            stats.execution_time, now
                        )
                    )

                conn.commit()

                logger.info(
                    f"Added pattern to library: {pattern}",
                    component="evolution",
                    operation="add_pattern",
                    context={
                        "pattern_id": pattern_id,
                        "is_regex": is_regex,
                    }
                )

                return pattern_id

            except sqlite3.Error as e:
                if conn:
                    conn.rollback()

                logger.error(
                    f"Failed to add pattern to library: {str(e)}",
                    component="evolution",
                    operation="add_pattern",
                    exception=e,
                    context={"pattern": pattern}
                )
                raise

            finally:
                if conn:
                    conn.close()

        return await loop.run_in_executor(None, _add_pattern)

    async def get_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """Get a pattern from the library.

        Args:
            pattern_id: Pattern ID

        Returns:
            Pattern information or None if not found
        """
        # Execute in a thread pool
        loop = asyncio.get_event_loop()

        async def _get_pattern():
            conn = None
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Get pattern
                cursor.execute(
                    '''
                    SELECT * FROM patterns WHERE id = ?
                    ''',
                    (pattern_id,)
                )

                pattern_row = cursor.fetchone()

                if not pattern_row:
                    return None

                # Convert to dictionary
                pattern_dict = dict(pattern_row)

                # Convert boolean fields
                pattern_dict["is_regex"] = bool(pattern_dict["is_regex"])
                pattern_dict["case_sensitive"] = bool(pattern_dict["case_sensitive"])

                # Convert tags
                if pattern_dict["tags"]:
                    pattern_dict["tags"] = pattern_dict["tags"].split(",")
                else:
                    pattern_dict["tags"] = []

                # Get examples
                cursor.execute(
                    '''
                    SELECT * FROM pattern_examples WHERE pattern_id = ?
                    ''',
                    (pattern_id,)
                )

                examples = [dict(row) for row in cursor.fetchall()]

                # Convert example boolean fields
                for example in examples:
                    example["is_positive"] = bool(example["is_positive"])

                pattern_dict["examples"] = examples

                # Get latest stats
                cursor.execute(
                    '''
                    SELECT * FROM pattern_stats
                    WHERE pattern_id = ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                    ''',
                    (pattern_id,)
                )

                stats_row = cursor.fetchone()

                if stats_row:
                    pattern_dict["stats"] = dict(stats_row)

                return pattern_dict

            except sqlite3.Error as e:
                logger.error(
                    f"Failed to get pattern from library: {str(e)}",
                    component="evolution",
                    operation="get_pattern",
                    exception=e,
                    context={"pattern_id": pattern_id}
                )
                raise

            finally:
                if conn:
                    conn.close()

        return await loop.run_in_executor(None, _get_pattern)

    async def update_pattern(
        self,
        pattern_id: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
        confidence: Optional[float] = None,
        effectiveness_score: Optional[float] = None,
    ) -> bool:
        """Update pattern metadata.

        Args:
            pattern_id: Pattern ID
            description: Optional new description
            tags: Optional new tags
            category: Optional new category
            confidence: Optional new confidence score
            effectiveness_score: Optional new effectiveness score

        Returns:
            Whether the update was successful
        """
        # Process tags
        tags_str = None
        if tags is not None:
            # Sort and join tags
            tags_str = ",".join(sorted(tags))

            # Ensure tags exist in the tags table
            await self._ensure_tags_exist(tags)

        # Generate updated timestamp
        now = datetime.utcnow().isoformat()

        # Execute in a thread pool
        loop = asyncio.get_event_loop()

        async def _update_pattern():
            conn = None
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                # Check if pattern exists
                cursor.execute(
                    '''
                    SELECT id FROM patterns WHERE id = ?
                    ''',
                    (pattern_id,)
                )

                if not cursor.fetchone():
                    logger.warning(
                        f"Pattern not found: {pattern_id}",
                        component="evolution",
                        operation="update_pattern"
                    )
                    return False

                # Build update query
                update_parts = []
                params = []

                if description is not None:
                    update_parts.append("description = ?")
                    params.append(description)

                if tags is not None:
                    update_parts.append("tags = ?")
                    params.append(tags_str)

                if category is not None:
                    update_parts.append("category = ?")
                    params.append(category)

                if confidence is not None:
                    update_parts.append("confidence = ?")
                    params.append(confidence)

                if effectiveness_score is not None:
                    update_parts.append("effectiveness_score = ?")
                    params.append(effectiveness_score)

                # Add updated timestamp
                update_parts.append("updated_at = ?")
                params.append(now)

                # Add pattern ID
                params.append(pattern_id)

                # Execute update if there are fields to update
                if update_parts:
                    query = f'''
                    UPDATE patterns
                    SET {", ".join(update_parts)}
                    WHERE id = ?
                    '''

                    cursor.execute(query, params)
                    conn.commit()

                    logger.info(
                        f"Updated pattern: {pattern_id}",
                        component="evolution",
                        operation="update_pattern",
                        context={
                            "fields_updated": len(update_parts) - 1, # Subtract 1 for updated_at
                        }
                    )

                    return True
                else:
                    # No fields to update
                    logger.info(
                        f"No fields to update for pattern: {pattern_id}",
                        component="evolution",
                        operation="update_pattern"
                    )
                    return True

            except sqlite3.Error as e:
                if conn:
                    conn.rollback()

                logger.error(
                    f"Failed to update pattern: {str(e)}",
                    component="evolution",
                    operation="update_pattern",
                    exception=e,
                    context={"pattern_id": pattern_id}
                )
                return False

            finally:
                if conn:
                    conn.close()

        return await loop.run_in_executor(None, _update_pattern)

    async def delete_pattern(self, pattern_id: str) -> bool:
        """Delete a pattern from the library.

        Args:
            pattern_id: Pattern ID

        Returns:
            Whether the deletion was successful
        """
        # Execute in a thread pool
        loop = asyncio.get_event_loop()

        async def _delete_pattern():
            conn = None
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                # Check if pattern exists
                cursor.execute(
                    '''
                    SELECT id FROM patterns WHERE id = ?
                    ''',
                    (pattern_id,)
                )

                if not cursor.fetchone():
                    logger.warning(
                        f"Pattern not found: {pattern_id}",
                        component="evolution",
                        operation="delete_pattern"
                    )
                    return False

                # Begin transaction
                conn.execute("BEGIN")

                # Delete examples
                cursor.execute(
                    '''
                    DELETE FROM pattern_examples WHERE pattern_id = ?
                    ''',
                    (pattern_id,)
                )

                # Delete stats
                cursor.execute(
                    '''
                    DELETE FROM pattern_stats WHERE pattern_id = ?
                    ''',
                    (pattern_id,)
                )

                # Delete pattern
                cursor.execute(
                    '''
                    DELETE FROM patterns WHERE id = ?
                    ''',
                    (pattern_id,)
                )

                # Commit transaction
                conn.commit()

                logger.info(
                    f"Deleted pattern: {pattern_id}",
                    component="evolution",
                    operation="delete_pattern"
                )

                return True

            except sqlite3.Error as e:
                if conn:
                    conn.rollback()

                logger.error(
                    f"Failed to delete pattern: {str(e)}",
                    component="evolution",
                    operation="delete_pattern",
                    exception=e,
                    context={"pattern_id": pattern_id}
                )
                return False

            finally:
                if conn:
                    conn.close()

        return await loop.run_in_executor(None, _delete_pattern)

    async def add_pattern_stats(
        self,
        pattern_id: str,
        stats: PatternStats
    ) -> bool:
        """Add statistics for a pattern.

        Args:
            pattern_id: Pattern ID
            stats: Pattern statistics

        Returns:
            Whether the addition was successful
        """
        # Check if pattern exists
        pattern = await self.get_pattern(pattern_id)
        if not pattern:
            logger.warning(
                f"Pattern not found: {pattern_id}",
                component="evolution",
                operation="add_pattern_stats"
            )
            return False

        # Generate timestamps
        now = datetime.utcnow().isoformat()

        # Execute in a thread pool
        loop = asyncio.get_event_loop()

        async def _add_stats():
            conn = None
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                # Add stats
                stats_id = str(uuid.uuid4())
                cursor.execute(
                    '''
                    INSERT INTO pattern_stats (
                        id, pattern_id, total_matches, files_with_matches,
                        precision, recall, f1_score, avg_context_relevance,
                        execution_time, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        stats_id, pattern_id,
                        stats.total_matches, stats.files_with_matches,
                        stats.precision, stats.recall, stats.f1_score,
                        stats.avg_context_relevance,
                        stats.execution_time, now
                    )
                )

                # Update effectiveness score based on stats
                effectiveness_score = (stats.precision + stats.recall + stats.f1_score) / 3

                cursor.execute(
                    '''
                    UPDATE patterns
                    SET effectiveness_score = ?, updated_at = ?
                    WHERE id = ?
                    ''',
                    (effectiveness_score, now, pattern_id)
                )

                conn.commit()

                logger.info(
                    f"Added stats for pattern: {pattern_id}",
                    component="evolution",
                    operation="add_pattern_stats",
                    context={
                        "stats_id": stats_id,
                        "total_matches": stats.total_matches,
                    }
                )

                return True

            except sqlite3.Error as e:
                if conn:
                    conn.rollback()

                logger.error(
                    f"Failed to add pattern stats: {str(e)}",
                    component="evolution",
                    operation="add_pattern_stats",
                    exception=e,
                    context={"pattern_id": pattern_id}
                )
                return False

            finally:
                if conn:
                    conn.close()

        return await loop.run_in_executor(None, _add_stats)

    async def add_pattern_example(
        self,
        pattern_id: str,
        example_text: str,
        file_path: Optional[str] = None,
        line_number: Optional[int] = None,
        is_positive: bool = True,
    ) -> bool:
        """Add an example match for a pattern.

        Args:
            pattern_id: Pattern ID
            example_text: Example text
            file_path: Optional file path
            line_number: Optional line number
            is_positive: Whether this is a positive example

        Returns:
            Whether the addition was successful
        """
        # Check if pattern exists
        pattern = await self.get_pattern(pattern_id)
        if not pattern:
            logger.warning(
                f"Pattern not found: {pattern_id}",
                component="evolution",
                operation="add_pattern_example"
            )
            return False

        # Generate timestamps
        now = datetime.utcnow().isoformat()

        # Execute in a thread pool
        loop = asyncio.get_event_loop()

        async def _add_example():
            conn = None
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                # Add example
                example_id = str(uuid.uuid4())
                cursor.execute(
                    '''
                    INSERT INTO pattern_examples (
                        id, pattern_id, example_text, file_path, line_number,
                        is_positive, added_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        example_id, pattern_id,
                        example_text, file_path, line_number,
                        1 if is_positive else 0,
                        now
                    )
                )

                conn.commit()

                logger.info(
                    f"Added example for pattern: {pattern_id}",
                    component="evolution",
                    operation="add_pattern_example",
                    context={
                        "example_id": example_id,
                        "is_positive": is_positive,
                    }
                )

                return True

            except sqlite3.Error as e:
                if conn:
                    conn.rollback()

                logger.error(
                    f"Failed to add pattern example: {str(e)}",
                    component="evolution",
                    operation="add_pattern_example",
                    exception=e,
                    context={"pattern_id": pattern_id}
                )
                return False

            finally:
                if conn:
                    conn.close()

        return await loop.run_in_executor(None, _add_example)

    async def search_patterns(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
        min_confidence: Optional[float] = None,
        min_effectiveness: Optional[float] = None,
        is_regex: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Search for patterns in the library.

        Args:
            query: Optional search query
            tags: Optional tags to filter by
            category: Optional category to filter by
            min_confidence: Optional minimum confidence score
            min_effectiveness: Optional minimum effectiveness score
            is_regex: Optional regex filter
            limit: Maximum number of results
            offset: Result offset

        Returns:
            Tuple of (patterns, total_count)
        """
        # Execute in a thread pool
        loop = asyncio.get_event_loop()

        async def _search_patterns():
            conn = None
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Build query conditions
                conditions = []
                params = []

                if query:
                    conditions.append("(pattern LIKE ? OR description LIKE ?)")
                    params.extend([f"%{query}%", f"%{query}%"])

                if tags:
                    # For each tag, check if it's in the comma-separated list
                    tag_conditions = []
                    for tag in tags:
                        tag_conditions.append("tags LIKE ?")
                        params.append(f"%{tag}%")
                    conditions.append(f"({' OR '.join(tag_conditions)})")

                if category:
                    conditions.append("category = ?")
                    params.append(category)

                if min_confidence is not None:
                    conditions.append("confidence >= ?")
                    params.append(min_confidence)

                if min_effectiveness is not None:
                    conditions.append("effectiveness_score >= ?")
                    params.append(min_effectiveness)

                if is_regex is not None:
                    conditions.append("is_regex = ?")
                    params.append(1 if is_regex else 0)

                # Combine conditions
                where_clause = ""
                if conditions:
                    where_clause = f"WHERE {' AND '.join(conditions)}"

                # Count total results
                count_query = f"SELECT COUNT(*) FROM patterns {where_clause}"
                cursor.execute(count_query, params)
                total_count = cursor.fetchone()[0]

                # Get paginated results
                query_sql = f"""
                SELECT * FROM patterns {where_clause}
                ORDER BY effectiveness_score DESC
                LIMIT ? OFFSET ?
                """
                params.extend([limit, offset])

                cursor.execute(query_sql, params)

                patterns = []
                for row in cursor.fetchall():
                    pattern_dict = dict(row)

                    # Convert boolean fields
                    pattern_dict["is_regex"] = bool(pattern_dict["is_regex"])
                    pattern_dict["case_sensitive"] = bool(pattern_dict["case_sensitive"])

                    # Convert tags
                    if pattern_dict["tags"]:
                        pattern_dict["tags"] = pattern_dict["tags"].split(",")
                    else:
                        pattern_dict["tags"] = []

                    patterns.append(pattern_dict)

                logger.info(
                    f"Searched patterns: {len(patterns)} results",
                    component="evolution",
                    operation="search_patterns",
                    context={
                        "total_count": total_count,
                        "limit": limit,
                        "offset": offset,
                    }
                )

                return patterns, total_count

            except sqlite3.Error as e:
                logger.error(
                    f"Failed to search patterns: {str(e)}",
                    component="evolution",
                    operation="search_patterns",
                    exception=e
                )
                return [], 0

            finally:
                if conn:
                    conn.close()

        return await loop.run_in_executor(None, _search_patterns)

    async def get_pattern_by_string(
        self,
        pattern_string: str,
        is_regex: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get a pattern by its string representation.

        Args:
            pattern_string: Pattern string
            is_regex: Optional regex filter

        Returns:
            Pattern information or None if not found
        """
        # Execute in a thread pool
        loop = asyncio.get_event_loop()

        async def _get_pattern():
            conn = None
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Build query
                query_sql = "SELECT * FROM patterns WHERE pattern = ?"
                params = [pattern_string]

                if is_regex is not None:
                    query_sql += " AND is_regex = ?"
                    params.append(1 if is_regex else 0)

                cursor.execute(query_sql, params)

                pattern_row = cursor.fetchone()

                if not pattern_row:
                    return None

                # Convert to dictionary
                pattern_dict = dict(pattern_row)

                # Convert boolean fields
                pattern_dict["is_regex"] = bool(pattern_dict["is_regex"])
                pattern_dict["case_sensitive"] = bool(pattern_dict["case_sensitive"])

                # Convert tags
                if pattern_dict["tags"]:
                    pattern_dict["tags"] = pattern_dict["tags"].split(",")
                else:
                    pattern_dict["tags"] = []

                return pattern_dict

            except sqlite3.Error as e:
                logger.error(
                    f"Failed to get pattern by string: {str(e)}",
                    component="evolution",
                    operation="get_pattern_by_string",
                    exception=e,
                    context={"pattern": pattern_string}
                )
                return None

            finally:
                if conn:
                    conn.close()

        return await loop.run_in_executor(None, _get_pattern)

    async def increment_usage_count(self, pattern_id: str) -> bool:
        """Increment the usage count for a pattern.

        Args:
            pattern_id: Pattern ID

        Returns:
            Whether the update was successful
        """
        # Execute in a thread pool
        loop = asyncio.get_event_loop()

        async def _increment_usage():
            conn = None
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                # Check if pattern exists
                cursor.execute(
                    '''
                    SELECT id FROM patterns WHERE id = ?
                    ''',
                    (pattern_id,)
                )

                if not cursor.fetchone():
                    logger.warning(
                        f"Pattern not found: {pattern_id}",
                        component="evolution",
                        operation="increment_usage_count"
                    )
                    return False

                # Increment usage count
                cursor.execute(
                    '''
                    UPDATE patterns
                    SET usage_count = usage_count + 1,
                        updated_at = ?
                    WHERE id = ?
                    ''',
                    (datetime.utcnow().isoformat(), pattern_id)
                )

                conn.commit()

                logger.debug(
                    f"Incremented usage count for pattern: {pattern_id}",
                    component="evolution",
                    operation="increment_usage_count"
                )

                return True

            except sqlite3.Error as e:
                if conn:
                    conn.rollback()

                logger.error(
                    f"Failed to increment usage count: {str(e)}",
                    component="evolution",
                    operation="increment_usage_count",
                    exception=e,
                    context={"pattern_id": pattern_id}
                )
                return False

            finally:
                if conn:
                    conn.close()

        return await loop.run_in_executor(None, _increment_usage)

    async def get_popular_patterns(
        self,
        limit: int = 10,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get the most popular patterns based on usage count.

        Args:
            limit: Maximum number of patterns to return
            category: Optional category filter
            tags: Optional tags filter

        Returns:
            List of popular patterns
        """
        # Execute in a thread pool
        loop = asyncio.get_event_loop()

        async def _get_popular():
            conn = None
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Build query conditions
                conditions = []
                params = []

                if category:
                    conditions.append("category = ?")
                    params.append(category)

                if tags:
                    # For each tag, check if it's in the comma-separated list
                    tag_conditions = []
                    for tag in tags:
                        tag_conditions.append("tags LIKE ?")
                        params.append(f"%{tag}%")
                    conditions.append(f"({' OR '.join(tag_conditions)})")

                # Combine conditions
                where_clause = ""
                if conditions:
                    where_clause = f"WHERE {' AND '.join(conditions)}"

                # Get popular patterns
                query_sql = f"""
                SELECT * FROM patterns {where_clause}
                ORDER BY usage_count DESC
                LIMIT ?
                """
                params.append(limit)

                cursor.execute(query_sql, params)

                patterns = []
                for row in cursor.fetchall():
                    pattern_dict = dict(row)

                    # Convert boolean fields
                    pattern_dict["is_regex"] = bool(pattern_dict["is_regex"])
                    pattern_dict["case_sensitive"] = bool(pattern_dict["case_sensitive"])

                    # Convert tags
                    if pattern_dict["tags"]:
                        pattern_dict["tags"] = pattern_dict["tags"].split(",")
                    else:
                        pattern_dict["tags"] = []

                    patterns.append(pattern_dict)

                logger.info(
                    f"Retrieved popular patterns: {len(patterns)} results",
                    component="evolution",
                    operation="get_popular_patterns",
                    context={"limit": limit}
                )

                return patterns

            except sqlite3.Error as e:
                logger.error(
                    f"Failed to get popular patterns: {str(e)}",
                    component="evolution",
                    operation="get_popular_patterns",
                    exception=e
                )
                return []

            finally:
                if conn:
                    conn.close()

        return await loop.run_in_executor(None, _get_popular)

    async def get_pattern_evolution_history(
        self,
        root_pattern_id: str
    ) -> List[Dict[str, Any]]:
        """Get the evolution history of a pattern.

        Args:
            root_pattern_id: Root pattern ID

        Returns:
            List of patterns in evolution history
        """
        # Execute in a thread pool
        loop = asyncio.get_event_loop()

        async def _get_history():
            conn = None
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Helper function to recursively find patterns with parent
                def find_children(parent_id, visited=None):
                    if visited is None:
                        visited = set()

                    if parent_id in visited:
                        return []

                    visited.add(parent_id)

                    cursor.execute(
                        '''
                        SELECT * FROM patterns WHERE parent_pattern_id = ?
                        ''',
                        (parent_id,)
                    )

                    children = []
                    for row in cursor.fetchall():
                        child = dict(row)

                        # Convert boolean fields
                        child["is_regex"] = bool(child["is_regex"])
                        child["case_sensitive"] = bool(child["case_sensitive"])

                        # Convert tags
                        if child["tags"]:
                            child["tags"] = child["tags"].split(",")
                        else:
                            child["tags"] = []

                        children.append(child)

                        # Recursively find grandchildren
                        grandchildren = find_children(child["id"], visited)
                        children.extend(grandchildren)

                    return children

                # Get the root pattern
                cursor.execute(
                    '''
                    SELECT * FROM patterns WHERE id = ?
                    ''',
                    (root_pattern_id,)
                )

                root_row = cursor.fetchone()

                if not root_row:
                    return []

                # Convert to dictionary
                root_dict = dict(root_row)

                # Convert boolean fields
                root_dict["is_regex"] = bool(root_dict["is_regex"])
                root_dict["case_sensitive"] = bool(root_dict["case_sensitive"])

                # Convert tags
                if root_dict["tags"]:
                    root_dict["tags"] = root_dict["tags"].split(",")
                else:
                    root_dict["tags"] = []

                # Get all children
                children = find_children(root_pattern_id)

                # Combine root and children
                patterns = [root_dict] + children

                # Sort by created_at
                patterns.sort(key=lambda p: p["created_at"])

                logger.info(
                    f"Retrieved pattern evolution history: {len(patterns)} patterns",
                    component="evolution",
                    operation="get_pattern_evolution_history",
                    context={"root_pattern_id": root_pattern_id}
                )

                return patterns

            except sqlite3.Error as e:
                logger.error(
                    f"Failed to get pattern evolution history: {str(e)}",
                    component="evolution",
                    operation="get_pattern_evolution_history",
                    exception=e,
                    context={"root_pattern_id": root_pattern_id}
                )
                return []

            finally:
                if conn:
                    conn.close()

        return await loop.run_in_executor(None, _get_history)

    async def _ensure_tags_exist(self, tags: List[str]) -> None:
        """Ensure tags exist in the tags table.

        Args:
            tags: List of tags
        """
        if not tags:
            return

        # Execute in a thread pool
        loop = asyncio.get_event_loop()

        async def _add_tags():
            conn = None
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                now = datetime.utcnow().isoformat()

                for tag in tags:
                    # Check if tag exists
                    cursor.execute(
                        '''
                        SELECT id FROM tags WHERE name = ?
                        ''',
                        (tag,)
                    )

                    if not cursor.fetchone():
                        # Add tag
                        tag_id = str(uuid.uuid4())
                        cursor.execute(
                            '''
                            INSERT INTO tags (id, name, created_at)
                            VALUES (?, ?, ?)
                            ''',
                            (tag_id, tag, now)
                        )

                conn.commit()

            except sqlite3.Error as e:
                if conn:
                    conn.rollback()

                logger.error(
                    f"Failed to ensure tags exist: {str(e)}",
                    component="evolution",
                    operation="ensure_tags_exist",
                    exception=e,
                    context={"tags": tags}
                )

            finally:
                if conn:
                    conn.close()

        await loop.run_in_executor(None, _add_tags)

    async def get_tags(self) -> List[Dict[str, Any]]:
        """Get all tags in the library.

        Returns:
            List of tags
        """
        # Execute in a thread pool
        loop = asyncio.get_event_loop()

        async def _get_tags():
            conn = None
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Get all tags
                cursor.execute(
                    '''
                    SELECT * FROM tags
                    ORDER BY name
                    '''
                )

                tags = [dict(row) for row in cursor.fetchall()]

                logger.debug(
                    f"Retrieved tags: {len(tags)} tags",
                    component="evolution",
                    operation="get_tags"
                )

                return tags

            except sqlite3.Error as e:
                logger.error(
                    f"Failed to get tags: {str(e)}",
                    component="evolution",
                    operation="get_tags",
                    exception=e
                )
                return []

            finally:
                if conn:
                    conn.close()

        return await loop.run_in_executor(None, _get_tags)

    async def find_similar_patterns(
        self,
        pattern: str,
        is_regex: bool,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find patterns similar to the given pattern.

        Args:
            pattern: Pattern string
            is_regex: Whether the pattern is a regex
            limit: Maximum number of patterns to return

        Returns:
            List of similar patterns
        """
        # Execute in a thread pool
        loop = asyncio.get_event_loop()

        async def _find_similar():
            conn = None
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Get all patterns of the same type
                query_sql = "SELECT * FROM patterns WHERE is_regex = ?"
                params = [1 if is_regex else 0]

                cursor.execute(query_sql, params)

                all_patterns = [dict(row) for row in cursor.fetchall()]

                # Calculate similarity scores
                similar_patterns = []

                for p in all_patterns:
                    # Skip exact match
                    if p["pattern"] == pattern:
                        continue

                    # Calculate similarity
                    similarity = self._calculate_pattern_similarity(
                        pattern,
                        p["pattern"],
                        is_regex
                    )

                    if similarity > 0.3: # Minimum similarity threshold
                        # Add similarity score
                        p["similarity"] = similarity

                        # Convert boolean fields
                        p["is_regex"] = bool(p["is_regex"])
                        p["case_sensitive"] = bool(p["case_sensitive"])

                        # Convert tags
                        if p["tags"]:
                            p["tags"] = p["tags"].split(",")
                        else:
                            p["tags"] = []

                        similar_patterns.append(p)

                # Sort by similarity
                similar_patterns.sort(key=lambda p: p["similarity"], reverse=True)

                # Limit results
                similar_patterns = similar_patterns[:limit]

                logger.info(
                    f"Found similar patterns: {len(similar_patterns)} results",
                    component="evolution",
                    operation="find_similar_patterns",
                    context={"pattern": pattern}
                )

                return similar_patterns

            except sqlite3.Error as e:
                logger.error(
                    f"Failed to find similar patterns: {str(e)}",
                    component="evolution",
                    operation="find_similar_patterns",
                    exception=e,
                    context={"pattern": pattern}
                )
                return []

            finally:
                if conn:
                    conn.close()

        return await loop.run_in_executor(None, _find_similar)

    def _calculate_pattern_similarity(
        self,
        pattern1: str,
        pattern2: str,
        is_regex: bool
    ) -> float:
        """Calculate similarity between two patterns.

        Args:
            pattern1: First pattern
            pattern2: Second pattern
            is_regex: Whether the patterns are regexes

        Returns:
            Similarity score (0.0-1.0)
        """
        if is_regex:
            # For regex patterns, use a more sophisticated comparison
            # This is a simplified implementation

            # Check for literal parts
            p1_literals = re.split(r'[\\[\](){}?*+^$.|]', pattern1)
            p2_literals = re.split(r'[\\[\](){}?*+^$.|]', pattern2)

            # Filter out empty strings
            p1_literals = [l for l in p1_literals if l]  # noqa: E741
            p2_literals = [l for l in p2_literals if l]  # noqa: E741

            if not p1_literals and not p2_literals:
                # Both patterns have no literals
                return 0.5

            if not p1_literals or not p2_literals:
                # One pattern has no literals
                return 0.0

            # Count common literals
            common_literals = set(p1_literals) & set(p2_literals)

            # Calculate Jaccard similarity
            jaccard_sim = len(common_literals) / len(set(p1_literals) | set(p2_literals))

            # Check for common structure
            structure_sim = 0.0

            # Extract pattern structure (regex special chars)
            p1_structure = ''.join(re.findall(r'[\\[\](){}?*+^$.|]', pattern1))
            p2_structure = ''.join(re.findall(r'[\\[\](){}?*+^$.|]', pattern2))

            if p1_structure and p2_structure:
                # Calculate structure similarity (simplified Levenshtein distance)
                max_len = max(len(p1_structure), len(p2_structure))
                if max_len > 0:
                    # Count matching characters
                    matches = sum(1 for a, b in zip(p1_structure, p2_structure) if a == b)
                    structure_sim = matches / max_len

            # Combine similarities
            return 0.7 * jaccard_sim + 0.3 * structure_sim
        else:
            # For literal patterns, use more direct comparison
            # This is a simplified implementation of string similarity

            # Tokenize patterns
            p1_tokens = pattern1.lower().split()
            p2_tokens = pattern2.lower().split()

            if not p1_tokens and not p2_tokens:
                # Both patterns are empty
                return 1.0

            if not p1_tokens or not p2_tokens:
                # One pattern is empty
                return 0.0

            # Count common tokens
            common_tokens = set(p1_tokens) & set(p2_tokens)

            # Calculate Jaccard similarity
            return len(common_tokens) / len(set(p1_tokens) | set(p2_tokens))


# Create a singleton instance
_pattern_library: Optional[PatternLibrary] = None


def get_pattern_library() -> PatternLibrary:
    """Get the singleton PatternLibrary instance.

    Returns:
        PatternLibrary instance
    """
    global _pattern_library

    if _pattern_library is None:
        try:
            _pattern_library = PatternLibrary()
        except Exception as e:
            logger.error(
                f"Failed to initialize PatternLibrary: {str(e)}",
                component="evolution",
                operation="init_pattern_library",
                exception=e
            )
            raise

    return _pattern_library


async def add_pattern_to_library(
    pattern: str,
    description: str,
    is_regex: bool,
    case_sensitive: bool,
    tags: Optional[List[str]] = None,
    category: Optional[str] = None,
    confidence: float = 1.0,
    effectiveness_score: float = 0.0,
    parent_pattern_id: Optional[str] = None,
    examples: Optional[List[Dict[str, Any]]] = None,
    stats: Optional[PatternStats] = None,
) -> str:
    """Add a pattern to the library.

    This is a convenience function that uses the singleton PatternLibrary.

    Args:
        pattern: Pattern string
        description: Pattern description
        is_regex: Whether the pattern is a regex
        case_sensitive: Whether the pattern is case-sensitive
        tags: Optional tags for the pattern
        category: Optional pattern category
        confidence: Confidence in the pattern (0.0-1.0)
        effectiveness_score: Pattern effectiveness score (0.0-1.0)
        parent_pattern_id: Optional ID of parent pattern
        examples: Optional list of example matches
        stats: Optional pattern statistics

    Returns:
        Pattern ID
    """
    library = get_pattern_library()

    return await library.add_pattern(
        pattern=pattern,
        description=description,
        is_regex=is_regex,
        case_sensitive=case_sensitive,
        tags=tags,
        category=category,
        confidence=confidence,
        effectiveness_score=effectiveness_score,
        parent_pattern_id=parent_pattern_id,
        examples=examples,
        stats=stats,
    )


async def search_patterns(
    query: Optional[str] = None,
    tags: Optional[List[str]] = None,
    category: Optional[str] = None,
    min_confidence: Optional[float] = None,
    min_effectiveness: Optional[float] = None,
    is_regex: Optional[bool] = None,
    limit: int = 100,
    offset: int = 0,
) -> Tuple[List[Dict[str, Any]], int]:
    """Search for patterns in the library.

    This is a convenience function that uses the singleton PatternLibrary.

    Args:
        query: Optional search query
        tags: Optional tags to filter by
        category: Optional category to filter by
        min_confidence: Optional minimum confidence score
        min_effectiveness: Optional minimum effectiveness score
        is_regex: Optional regex filter
        limit: Maximum number of results
        offset: Result offset

    Returns:
        Tuple of (patterns, total_count)
    """
    library = get_pattern_library()

    return await library.search_patterns(
        query=query,
        tags=tags,
        category=category,
        min_confidence=min_confidence,
        min_effectiveness=min_effectiveness,
        is_regex=is_regex,
        limit=limit,
        offset=offset,
    )


async def get_popular_patterns(
    limit: int = 10,
    category: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Get the most popular patterns based on usage count.

    This is a convenience function that uses the singleton PatternLibrary.

    Args:
        limit: Maximum number of patterns to return
        category: Optional category filter
        tags: Optional tags filter

    Returns:
        List of popular patterns
    """
    library = get_pattern_library()

    return await library.get_popular_patterns(
        limit=limit,
        category=category,
        tags=tags,
    )