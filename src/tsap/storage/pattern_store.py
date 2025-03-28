"""
Search pattern storage for TSAP.

This module provides persistent storage for search patterns and regex patterns,
allowing them to be saved, retrieved, and reused across different searches.
"""

import os
import time
import uuid
import threading
from typing import Dict, List, Any, Optional, Tuple

from tsap.utils.errors import TSAPError
from tsap.storage.database import Database, get_database, create_database


class PatternStoreError(TSAPError):
    """Exception raised for pattern storage errors."""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.details = details or {}


class PatternStore:
    """
    Persistent storage for search patterns.
    
    This class provides methods for saving, retrieving, and querying
    search patterns in a SQLite database.
    """
    
    def __init__(self, db: Database):
        """
        Initialize a pattern store.
        
        Args:
            db: Database instance
        """
        self.db = db
        self._lock = threading.RLock()
        self._initialize_schema()
    
    def _initialize_schema(self) -> None:
        """Initialize the database schema for pattern storage."""
        with self._lock:
            with self.db.transaction():
                # Create patterns table
                if not self.db.table_exists('patterns'):
                    self.db.create_table(
                        'patterns',
                        {
                            'pattern_id': 'TEXT NOT NULL',
                            'name': 'TEXT NOT NULL',
                            'description': 'TEXT',
                            'pattern': 'TEXT NOT NULL',
                            'is_regex': 'INTEGER NOT NULL DEFAULT 0',
                            'case_sensitive': 'INTEGER NOT NULL DEFAULT 0',
                            'category': 'TEXT',
                            'confidence': 'REAL',
                            'effectiveness_score': 'REAL',
                            'usage_count': 'INTEGER NOT NULL DEFAULT 0',
                            'created_at': 'REAL NOT NULL',
                            'updated_at': 'REAL NOT NULL',
                            'created_by': 'TEXT',
                            'parent_pattern_id': 'TEXT'
                        },
                        primary_key='pattern_id'
                    )
                    
                    # Create indices
                    self.db.create_index('idx_patterns_name', 'patterns', 'name')
                    self.db.create_index('idx_patterns_category', 'patterns', 'category')
                    self.db.create_index('idx_patterns_created_at', 'patterns', 'created_at')
                
                # Create pattern tags table
                if not self.db.table_exists('pattern_tags'):
                    self.db.create_table(
                        'pattern_tags',
                        {
                            'pattern_id': 'TEXT NOT NULL',
                            'tag': 'TEXT NOT NULL'
                        },
                        primary_key=['pattern_id', 'tag']
                    )
                    
                    # Create index on tag
                    self.db.create_index('idx_pt_tag', 'pattern_tags', 'tag')
                
                # Create pattern examples table
                if not self.db.table_exists('pattern_examples'):
                    self.db.create_table(
                        'pattern_examples',
                        {
                            'example_id': 'TEXT NOT NULL',
                            'pattern_id': 'TEXT NOT NULL',
                            'text': 'TEXT NOT NULL',
                            'file_path': 'TEXT',
                            'line_number': 'INTEGER',
                            'is_positive': 'INTEGER NOT NULL DEFAULT 1',
                            'created_at': 'REAL NOT NULL'
                        },
                        primary_key='example_id'
                    )
                    
                    # Create index on pattern_id
                    self.db.create_index('idx_pe_pattern_id', 'pattern_examples', 'pattern_id')
                
                # Create pattern statistics table
                if not self.db.table_exists('pattern_stats'):
                    self.db.create_table(
                        'pattern_stats',
                        {
                            'stat_id': 'TEXT NOT NULL',
                            'pattern_id': 'TEXT NOT NULL',
                            'matches': 'INTEGER',
                            'files_searched': 'INTEGER',
                            'true_positives': 'INTEGER',
                            'false_positives': 'INTEGER',
                            'precision': 'REAL',
                            'recall': 'REAL',
                            'f1_score': 'REAL',
                            'execution_time': 'REAL',
                            'recorded_at': 'REAL NOT NULL',
                            'context': 'TEXT'
                        },
                        primary_key='stat_id'
                    )
                    
                    # Create index on pattern_id
                    self.db.create_index('idx_ps_pattern_id', 'pattern_stats', 'pattern_id')
    
    def add_pattern(self, pattern: str, name: str, description: Optional[str] = None,
                  is_regex: bool = False, case_sensitive: bool = False,
                  category: Optional[str] = None, tags: Optional[List[str]] = None,
                  confidence: Optional[float] = None, 
                  effectiveness_score: Optional[float] = None,
                  created_by: Optional[str] = None,
                  parent_pattern_id: Optional[str] = None,
                  pattern_id: Optional[str] = None) -> str:
        """
        Add a pattern to the store.
        
        Args:
            pattern: The pattern string
            name: Pattern name
            description: Optional description
            is_regex: Whether the pattern is a regex
            case_sensitive: Whether the pattern is case sensitive
            category: Optional category
            tags: Optional tags
            confidence: Optional confidence score (0-1)
            effectiveness_score: Optional effectiveness score (0-1)
            created_by: Optional creator identifier
            parent_pattern_id: Optional ID of the parent pattern
            pattern_id: Optional pattern ID (generated if None)
            
        Returns:
            ID of the added pattern
        """
        with self._lock:
            with self.db.transaction():
                # Generate pattern ID if not provided
                if pattern_id is None:
                    pattern_id = str(uuid.uuid4())
                
                # Current time
                current_time = time.time()
                
                # Insert pattern
                self.db.insert(
                    'patterns',
                    {
                        'pattern_id': pattern_id,
                        'name': name,
                        'description': description,
                        'pattern': pattern,
                        'is_regex': 1 if is_regex else 0,
                        'case_sensitive': 1 if case_sensitive else 0,
                        'category': category,
                        'confidence': confidence,
                        'effectiveness_score': effectiveness_score,
                        'usage_count': 0,
                        'created_at': current_time,
                        'updated_at': current_time,
                        'created_by': created_by,
                        'parent_pattern_id': parent_pattern_id
                    }
                )
                
                # Add tags if provided
                if tags:
                    for tag in tags:
                        self.db.insert(
                            'pattern_tags',
                            {
                                'pattern_id': pattern_id,
                                'tag': tag
                            }
                        )
                
                return pattern_id
    
    def update_pattern(self, pattern_id: str, name: Optional[str] = None,
                     description: Optional[str] = None, pattern: Optional[str] = None,
                     is_regex: Optional[bool] = None, 
                     case_sensitive: Optional[bool] = None,
                     category: Optional[str] = None,
                     confidence: Optional[float] = None,
                     effectiveness_score: Optional[float] = None) -> bool:
        """
        Update a pattern in the store.
        
        Args:
            pattern_id: ID of the pattern
            name: New name
            description: New description
            pattern: New pattern string
            is_regex: Whether the pattern is a regex
            case_sensitive: Whether the pattern is case sensitive
            category: New category
            confidence: New confidence score
            effectiveness_score: New effectiveness score
            
        Returns:
            True if pattern was updated, False if not found
        """
        with self._lock:
            # Check if pattern exists
            existing = self.db.query_one(
                'SELECT 1 FROM patterns WHERE pattern_id = ?',
                (pattern_id,)
            )
            
            if not existing:
                return False
            
            # Build update data
            update_data = {
                'updated_at': time.time()
            }
            
            if name is not None:
                update_data['name'] = name
            
            if description is not None:
                update_data['description'] = description
            
            if pattern is not None:
                update_data['pattern'] = pattern
            
            if is_regex is not None:
                update_data['is_regex'] = 1 if is_regex else 0
            
            if case_sensitive is not None:
                update_data['case_sensitive'] = 1 if case_sensitive else 0
            
            if category is not None:
                update_data['category'] = category
            
            if confidence is not None:
                update_data['confidence'] = confidence
            
            if effectiveness_score is not None:
                update_data['effectiveness_score'] = effectiveness_score
            
            # Update pattern
            with self.db.transaction():
                self.db.update(
                    'patterns',
                    update_data,
                    'pattern_id = ?',
                    (pattern_id,)
                )
                
                return True
    
    def delete_pattern(self, pattern_id: str) -> bool:
        """
        Delete a pattern from the store.
        
        Args:
            pattern_id: ID of the pattern
            
        Returns:
            True if pattern was deleted, False if not found
        """
        with self._lock:
            # Check if pattern exists
            existing = self.db.query_one(
                'SELECT 1 FROM patterns WHERE pattern_id = ?',
                (pattern_id,)
            )
            
            if not existing:
                return False
            
            # Delete pattern and related data
            with self.db.transaction():
                # Delete tags
                self.db.delete('pattern_tags', 'pattern_id = ?', (pattern_id,))
                
                # Delete examples
                self.db.delete('pattern_examples', 'pattern_id = ?', (pattern_id,))
                
                # Delete stats
                self.db.delete('pattern_stats', 'pattern_id = ?', (pattern_id,))
                
                # Delete pattern
                self.db.delete('patterns', 'pattern_id = ?', (pattern_id,))
                
                return True
    
    def get_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a pattern from the store.
        
        Args:
            pattern_id: ID of the pattern
            
        Returns:
            Pattern data or None if not found
        """
        with self._lock:
            # Get pattern data
            pattern = self.db.query_one(
                'SELECT * FROM patterns WHERE pattern_id = ?',
                (pattern_id,)
            )
            
            if not pattern:
                return None
            
            # Convert boolean fields
            pattern['is_regex'] = bool(pattern['is_regex'])
            pattern['case_sensitive'] = bool(pattern['case_sensitive'])
            
            # Get tags
            tags = self.db.query(
                'SELECT tag FROM pattern_tags WHERE pattern_id = ?',
                (pattern_id,)
            )
            
            pattern['tags'] = [tag['tag'] for tag in tags]
            
            # Get examples
            examples = self.db.query(
                'SELECT * FROM pattern_examples WHERE pattern_id = ? ORDER BY created_at DESC',
                (pattern_id,)
            )
            
            pattern['examples'] = []
            for example in examples:
                example['is_positive'] = bool(example['is_positive'])
                pattern['examples'].append(example)
            
            # Get latest stats
            stats = self.db.query_one(
                'SELECT * FROM pattern_stats WHERE pattern_id = ? ORDER BY recorded_at DESC LIMIT 1',
                (pattern_id,)
            )
            
            if stats:
                # Parse context JSON if present
                if stats['context']:
                    stats['context'] = self.db.json_deserialize(stats['context'])
                
                pattern['stats'] = stats
            
            return pattern
    
    def get_pattern_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a pattern by name.
        
        Args:
            name: Pattern name
            
        Returns:
            Pattern data or None if not found
        """
        with self._lock:
            # Get pattern ID
            result = self.db.query_one(
                'SELECT pattern_id FROM patterns WHERE name = ? LIMIT 1',
                (name,)
            )
            
            if not result:
                return None
            
            # Get full pattern data
            return self.get_pattern(result['pattern_id'])
    
    def get_pattern_by_value(self, pattern: str, is_regex: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get a pattern by its value.
        
        Args:
            pattern: The pattern string
            is_regex: Whether the pattern is a regex
            
        Returns:
            Pattern data or None if not found
        """
        with self._lock:
            # Get pattern ID
            result = self.db.query_one(
                'SELECT pattern_id FROM patterns WHERE pattern = ? AND is_regex = ? LIMIT 1',
                (pattern, 1 if is_regex else 0)
            )
            
            if not result:
                return None
            
            # Get full pattern data
            return self.get_pattern(result['pattern_id'])
    
    def search_patterns(self, query: Optional[str] = None,
                       tags: Optional[List[str]] = None,
                       category: Optional[str] = None,
                       is_regex: Optional[bool] = None,
                       min_confidence: Optional[float] = None,
                       min_effectiveness: Optional[float] = None,
                       min_usage_count: Optional[int] = None,
                       limit: int = 100,
                       offset: int = 0,
                       sort_by: str = 'updated_at',
                       sort_order: str = 'desc') -> Tuple[List[Dict[str, Any]], int]:
        """
        Search for patterns in the store.
        
        Args:
            query: Text search query
            tags: Filter by tags (AND logic)
            category: Filter by category
            is_regex: Filter by regex flag
            min_confidence: Minimum confidence score
            min_effectiveness: Minimum effectiveness score
            min_usage_count: Minimum usage count
            limit: Maximum number of results to return
            offset: Offset for pagination
            sort_by: Field to sort by
            sort_order: Sort order ('asc' or 'desc')
            
        Returns:
            Tuple of (pattern list, total count)
        """
        with self._lock:
            # Build the query
            query_parts = ['SELECT p.* FROM patterns p']
            where_clauses = []
            params = []
            
            # Join with tags table if filtering by tags
            if tags and len(tags) > 0:
                for i, tag in enumerate(tags):
                    query_parts[0] += f' JOIN pattern_tags pt{i} ON p.pattern_id = pt{i}.pattern_id'
                    where_clauses.append(f'pt{i}.tag = ?')
                    params.append(tag)
            
            # Add filter conditions
            if query:
                # Search in name, description, and pattern
                where_clauses.append('(p.name LIKE ? OR p.description LIKE ? OR p.pattern LIKE ?)')
                search_term = f'%{query}%'
                params.extend([search_term, search_term, search_term])
            
            if category:
                where_clauses.append('p.category = ?')
                params.append(category)
            
            if is_regex is not None:
                where_clauses.append('p.is_regex = ?')
                params.append(1 if is_regex else 0)
            
            if min_confidence is not None:
                where_clauses.append('p.confidence >= ?')
                params.append(min_confidence)
            
            if min_effectiveness is not None:
                where_clauses.append('p.effectiveness_score >= ?')
                params.append(min_effectiveness)
            
            if min_usage_count is not None:
                where_clauses.append('p.usage_count >= ?')
                params.append(min_usage_count)
            
            # Combine WHERE clauses
            if where_clauses:
                query_parts.append('WHERE ' + ' AND '.join(where_clauses))
            
            # Build the full query for count
            count_query = ' '.join(['SELECT COUNT(*) as count FROM ('] + query_parts + [') as subquery'])
            
            # Get total count
            count_result = self.db.query_one(count_query, params)
            total_count = count_result['count'] if count_result else 0
            
            # Add sorting
            valid_sort_fields = {
                'name', 'created_at', 'updated_at', 'confidence',
                'effectiveness_score', 'usage_count'
            }
            if sort_by not in valid_sort_fields:
                sort_by = 'updated_at'
            
            sort_order = 'DESC' if sort_order.lower() == 'desc' else 'ASC'
            query_parts.append(f'ORDER BY p.{sort_by} {sort_order}')
            
            # Add pagination
            query_parts.append('LIMIT ? OFFSET ?')
            params.append(limit)
            params.append(offset)
            
            # Execute the query
            query = ' '.join(query_parts)
            patterns = self.db.query(query, params)
            
            # Process results
            result = []
            for pattern in patterns:
                # Convert boolean fields
                pattern['is_regex'] = bool(pattern['is_regex'])
                pattern['case_sensitive'] = bool(pattern['case_sensitive'])
                
                # Get tags
                tags = self.db.query(
                    'SELECT tag FROM pattern_tags WHERE pattern_id = ?',
                    (pattern['pattern_id'],)
                )
                
                pattern['tags'] = [tag['tag'] for tag in tags]
                
                result.append(pattern)
            
            return result, total_count
    
    def add_tag(self, pattern_id: str, tag: str) -> bool:
        """
        Add a tag to a pattern.
        
        Args:
            pattern_id: ID of the pattern
            tag: Tag to add
            
        Returns:
            True if tag was added, False if pattern not found
        """
        with self._lock:
            # Check if pattern exists
            existing = self.db.query_one(
                'SELECT 1 FROM patterns WHERE pattern_id = ?',
                (pattern_id,)
            )
            
            if not existing:
                return False
            
            # Check if tag already exists
            existing_tag = self.db.query_one(
                'SELECT 1 FROM pattern_tags WHERE pattern_id = ? AND tag = ?',
                (pattern_id, tag)
            )
            
            if existing_tag:
                return True  # Tag already exists
            
            # Add tag
            with self.db.transaction():
                self.db.insert(
                    'pattern_tags',
                    {
                        'pattern_id': pattern_id,
                        'tag': tag
                    }
                )
                
                # Update pattern updated_at
                self.db.update(
                    'patterns',
                    {'updated_at': time.time()},
                    'pattern_id = ?',
                    (pattern_id,)
                )
                
                return True
    
    def remove_tag(self, pattern_id: str, tag: str) -> bool:
        """
        Remove a tag from a pattern.
        
        Args:
            pattern_id: ID of the pattern
            tag: Tag to remove
            
        Returns:
            True if tag was removed, False if pattern or tag not found
        """
        with self._lock:
            with self.db.transaction():
                # Delete tag
                rows_affected = self.db.delete(
                    'pattern_tags',
                    'pattern_id = ? AND tag = ?',
                    (pattern_id, tag)
                )
                
                if rows_affected > 0:
                    # Update pattern updated_at
                    self.db.update(
                        'patterns',
                        {'updated_at': time.time()},
                        'pattern_id = ?',
                        (pattern_id,)
                    )
                    
                    return True
                
                return False
    
    def add_example(self, pattern_id: str, text: str, 
                  file_path: Optional[str] = None,
                  line_number: Optional[int] = None,
                  is_positive: bool = True) -> str:
        """
        Add an example for a pattern.
        
        Args:
            pattern_id: ID of the pattern
            text: Example text
            file_path: Optional file path
            line_number: Optional line number
            is_positive: Whether this is a positive example
            
        Returns:
            ID of the added example
            
        Raises:
            PatternStoreError: If pattern not found
        """
        with self._lock:
            # Check if pattern exists
            existing = self.db.query_one(
                'SELECT 1 FROM patterns WHERE pattern_id = ?',
                (pattern_id,)
            )
            
            if not existing:
                raise PatternStoreError(f"Pattern not found: {pattern_id}")
            
            # Generate example ID
            example_id = str(uuid.uuid4())
            
            # Current time
            current_time = time.time()
            
            # Add example
            with self.db.transaction():
                self.db.insert(
                    'pattern_examples',
                    {
                        'example_id': example_id,
                        'pattern_id': pattern_id,
                        'text': text,
                        'file_path': file_path,
                        'line_number': line_number,
                        'is_positive': 1 if is_positive else 0,
                        'created_at': current_time
                    }
                )
                
                # Update pattern updated_at
                self.db.update(
                    'patterns',
                    {'updated_at': current_time},
                    'pattern_id = ?',
                    (pattern_id,)
                )
                
                return example_id
    
    def get_examples(self, pattern_id: str, 
                    only_positive: bool = False) -> List[Dict[str, Any]]:
        """
        Get examples for a pattern.
        
        Args:
            pattern_id: ID of the pattern
            only_positive: Whether to return only positive examples
            
        Returns:
            List of examples
        """
        with self._lock:
            query = 'SELECT * FROM pattern_examples WHERE pattern_id = ?'
            params = [pattern_id]
            
            if only_positive:
                query += ' AND is_positive = 1'
            
            query += ' ORDER BY created_at DESC'
            
            examples = self.db.query(query, params)
            
            # Convert boolean fields
            for example in examples:
                example['is_positive'] = bool(example['is_positive'])
            
            return examples
    
    def add_stats(self, pattern_id: str, matches: int, files_searched: int,
                true_positives: Optional[int] = None, 
                false_positives: Optional[int] = None,
                precision: Optional[float] = None,
                recall: Optional[float] = None,
                f1_score: Optional[float] = None,
                execution_time: Optional[float] = None,
                context: Optional[Dict[str, Any]] = None) -> str:
        """
        Add statistics for a pattern.
        
        Args:
            pattern_id: ID of the pattern
            matches: Number of matches
            files_searched: Number of files searched
            true_positives: Number of true positives
            false_positives: Number of false positives
            precision: Precision score
            recall: Recall score
            f1_score: F1 score
            execution_time: Execution time in seconds
            context: Additional context information
            
        Returns:
            ID of the added stats
            
        Raises:
            PatternStoreError: If pattern not found
        """
        with self._lock:
            # Check if pattern exists
            existing = self.db.query_one(
                'SELECT 1 FROM patterns WHERE pattern_id = ?',
                (pattern_id,)
            )
            
            if not existing:
                raise PatternStoreError(f"Pattern not found: {pattern_id}")
            
            # Generate stat ID
            stat_id = str(uuid.uuid4())
            
            # Current time
            current_time = time.time()
            
            # Serialize context
            context_json = self.db.json_serialize(context) if context else None
            
            # Add stats
            with self.db.transaction():
                self.db.insert(
                    'pattern_stats',
                    {
                        'stat_id': stat_id,
                        'pattern_id': pattern_id,
                        'matches': matches,
                        'files_searched': files_searched,
                        'true_positives': true_positives,
                        'false_positives': false_positives,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1_score,
                        'execution_time': execution_time,
                        'recorded_at': current_time,
                        'context': context_json
                    }
                )
                
                # Update pattern effectiveness if F1 score provided
                if f1_score is not None:
                    self.db.update(
                        'patterns',
                        {
                            'effectiveness_score': f1_score,
                            'updated_at': current_time
                        },
                        'pattern_id = ?',
                        (pattern_id,)
                    )
                else:
                    # Just update the timestamp
                    self.db.update(
                        'patterns',
                        {'updated_at': current_time},
                        'pattern_id = ?',
                        (pattern_id,)
                    )
                
                return stat_id
    
    def increment_usage(self, pattern_id: str) -> bool:
        """
        Increment the usage count for a pattern.
        
        Args:
            pattern_id: ID of the pattern
            
        Returns:
            True if usage was incremented, False if pattern not found
        """
        with self._lock:
            with self.db.transaction():
                # Update usage count
                result = self.db.query_one(
                    'SELECT usage_count FROM patterns WHERE pattern_id = ?',
                    (pattern_id,)
                )
                
                if not result:
                    return False
                
                new_count = result['usage_count'] + 1
                
                self.db.update(
                    'patterns',
                    {
                        'usage_count': new_count,
                        'updated_at': time.time()
                    },
                    'pattern_id = ?',
                    (pattern_id,)
                )
                
                return True
    
    def get_popular_patterns(self, limit: int = 10, 
                           category: Optional[str] = None,
                           tag: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get the most popular patterns.
        
        Args:
            limit: Maximum number of patterns to return
            category: Optional category filter
            tag: Optional tag filter
            
        Returns:
            List of popular patterns
        """
        with self._lock:
            query = 'SELECT p.* FROM patterns p'
            where_clauses = []
            params = []
            
            # Join with tags table if filtering by tag
            if tag:
                query += ' JOIN pattern_tags pt ON p.pattern_id = pt.pattern_id'
                where_clauses.append('pt.tag = ?')
                params.append(tag)
            
            # Add category filter
            if category:
                where_clauses.append('p.category = ?')
                params.append(category)
            
            # Add where clause if needed
            if where_clauses:
                query += ' WHERE ' + ' AND '.join(where_clauses)
            
            # Add sorting and limit
            query += ' ORDER BY p.usage_count DESC LIMIT ?'
            params.append(limit)
            
            # Execute query
            patterns = self.db.query(query, params)
            
            # Process results
            result = []
            for pattern in patterns:
                # Convert boolean fields
                pattern['is_regex'] = bool(pattern['is_regex'])
                pattern['case_sensitive'] = bool(pattern['case_sensitive'])
                
                # Get tags
                tags = self.db.query(
                    'SELECT tag FROM pattern_tags WHERE pattern_id = ?',
                    (pattern['pattern_id'],)
                )
                
                pattern['tags'] = [tag['tag'] for tag in tags]
                
                result.append(pattern)
            
            return result
    
    def get_pattern_evolution(self, root_pattern_id: str) -> List[Dict[str, Any]]:
        """
        Get the evolution history of a pattern.
        
        Args:
            root_pattern_id: ID of the root pattern
            
        Returns:
            List of patterns in the evolution chain
        """
        with self._lock:
            # Get root pattern
            root_pattern = self.get_pattern(root_pattern_id)
            
            if not root_pattern:
                return []
            
            # Initialize result with root pattern
            result = [root_pattern]
            
            # Get all descendants
            descendants = []
            to_check = [root_pattern_id]
            
            while to_check:
                current_id = to_check.pop(0)
                
                children = self.db.query(
                    'SELECT * FROM patterns WHERE parent_pattern_id = ?',
                    (current_id,)
                )
                
                for child in children:
                    # Convert boolean fields
                    child['is_regex'] = bool(child['is_regex'])
                    child['case_sensitive'] = bool(child['case_sensitive'])
                    
                    # Get tags
                    tags = self.db.query(
                        'SELECT tag FROM pattern_tags WHERE pattern_id = ?',
                        (child['pattern_id'],)
                    )
                    
                    child['tags'] = [tag['tag'] for tag in tags]
                    
                    descendants.append(child)
                    to_check.append(child['pattern_id'])
            
            # Sort descendants by created_at
            descendants.sort(key=lambda p: p['created_at'])
            
            # Add to result
            result.extend(descendants)
            
            return result
    
    def find_similar_patterns(self, pattern: str, is_regex: bool,
                            max_distance: float = 0.3,
                            limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find patterns similar to a given pattern.
        
        Args:
            pattern: The pattern string
            is_regex: Whether the pattern is a regex
            max_distance: Maximum edit distance (as a proportion of pattern length)
            limit: Maximum number of patterns to return
            
        Returns:
            List of similar patterns
        """
        with self._lock:
            # Get all patterns of the same type
            patterns = self.db.query(
                'SELECT * FROM patterns WHERE is_regex = ?',
                (1 if is_regex else 0)
            )
            
            if not patterns:
                return []
            
            # Calculate similarity scores
            import difflib
            
            pattern_with_scores = []
            
            for p in patterns:
                similarity = difflib.SequenceMatcher(None, pattern, p['pattern']).ratio()
                
                # Consider patterns similar if similarity is high enough
                max_allowed_distance = max_distance
                
                if similarity >= (1.0 - max_allowed_distance):
                    pattern_with_scores.append((p, similarity))
            
            # Sort by similarity (highest first)
            pattern_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Take top results
            results = []
            for p, score in pattern_with_scores[:limit]:
                # Convert boolean fields
                p['is_regex'] = bool(p['is_regex'])
                p['case_sensitive'] = bool(p['case_sensitive'])
                
                # Get tags
                tags = self.db.query(
                    'SELECT tag FROM pattern_tags WHERE pattern_id = ?',
                    (p['pattern_id'],)
                )
                
                p['tags'] = [tag['tag'] for tag in tags]
                
                # Add similarity score
                p['similarity'] = score
                
                results.append(p)
            
            return results


# Global registry of pattern stores
_pattern_stores = {}
_pattern_lock = threading.RLock()


def get_pattern_store(db_name: str = 'patterns') -> PatternStore:
    """
    Get a pattern store.
    
    Args:
        db_name: Database name
        
    Returns:
        PatternStore instance
    """
    with _pattern_lock:
        if db_name in _pattern_stores:
            return _pattern_stores[db_name]
        
        # Get or create the database
        db = get_database(db_name)
        
        if db is None:
            # Create pattern database in data directory
            from tsap.config import get_config
            
            config = get_config()
            
            if hasattr(config, 'storage') and hasattr(config.storage, 'data_dir'):
                data_dir = config.storage.data_dir
            else:
                # Use system-dependent user data directory
                import appdirs  # Optional dependency
                data_dir = appdirs.user_data_dir("tsap", "tsap")
            
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, f"{db_name}.db")
            
            db = create_database(db_name, db_path)
        
        # Create pattern store
        store = PatternStore(db)
        _pattern_stores[db_name] = store
        
        return store


def add_pattern(pattern: str, name: str, description: Optional[str] = None,
              is_regex: bool = False, case_sensitive: bool = False,
              category: Optional[str] = None, tags: Optional[List[str]] = None,
              confidence: Optional[float] = None, 
              effectiveness_score: Optional[float] = None,
              created_by: Optional[str] = None,
              parent_pattern_id: Optional[str] = None) -> str:
    """
    Add a pattern to the store.
    
    Args:
        pattern: The pattern string
        name: Pattern name
        description: Optional description
        is_regex: Whether the pattern is a regex
        case_sensitive: Whether the pattern is case sensitive
        category: Optional category
        tags: Optional tags
        confidence: Optional confidence score (0-1)
        effectiveness_score: Optional effectiveness score (0-1)
        created_by: Optional creator identifier
        parent_pattern_id: Optional ID of the parent pattern
        
    Returns:
        ID of the added pattern
    """
    store = get_pattern_store()
    return store.add_pattern(
        pattern, name, description, is_regex, case_sensitive,
        category, tags, confidence, effectiveness_score,
        created_by, parent_pattern_id
    )


def search_patterns(query: Optional[str] = None,
                  tags: Optional[List[str]] = None,
                  category: Optional[str] = None,
                  is_regex: Optional[bool] = None,
                  min_confidence: Optional[float] = None,
                  min_effectiveness: Optional[float] = None,
                  min_usage_count: Optional[int] = None,
                  limit: int = 100,
                  offset: int = 0) -> Tuple[List[Dict[str, Any]], int]:
    """
    Search for patterns in the store.
    
    Args:
        query: Text search query
        tags: Filter by tags (AND logic)
        category: Filter by category
        is_regex: Filter by regex flag
        min_confidence: Minimum confidence score
        min_effectiveness: Minimum effectiveness score
        min_usage_count: Minimum usage count
        limit: Maximum number of results to return
        offset: Offset for pagination
        
    Returns:
        Tuple of (pattern list, total count)
    """
    store = get_pattern_store()
    return store.search_patterns(
        query, tags, category, is_regex, min_confidence,
        min_effectiveness, min_usage_count, limit, offset
    )


def get_pattern(pattern_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a pattern from the store.
    
    Args:
        pattern_id: ID of the pattern
        
    Returns:
        Pattern data or None if not found
    """
    store = get_pattern_store()
    return store.get_pattern(pattern_id)


def get_pattern_by_name(name: str) -> Optional[Dict[str, Any]]:
    """
    Get a pattern by name.
    
    Args:
        name: Pattern name
        
    Returns:
        Pattern data or None if not found
    """
    store = get_pattern_store()
    return store.get_pattern_by_name(name)


def get_popular_patterns(limit: int = 10, 
                       category: Optional[str] = None,
                       tag: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get the most popular patterns.
    
    Args:
        limit: Maximum number of patterns to return
        category: Optional category filter
        tag: Optional tag filter
        
    Returns:
        List of popular patterns
    """
    store = get_pattern_store()
    return store.get_popular_patterns(limit, category, tag)


def increment_pattern_usage(pattern_id: str) -> bool:
    """
    Increment the usage count for a pattern.
    
    Args:
        pattern_id: ID of the pattern
        
    Returns:
        True if usage was incremented, False if pattern not found
    """
    store = get_pattern_store()
    return store.increment_usage(pattern_id)


def add_pattern_example(pattern_id: str, text: str, 
                      file_path: Optional[str] = None,
                      line_number: Optional[int] = None,
                      is_positive: bool = True) -> str:
    """
    Add an example for a pattern.
    
    Args:
        pattern_id: ID of the pattern
        text: Example text
        file_path: Optional file path
        line_number: Optional line number
        is_positive: Whether this is a positive example
        
    Returns:
        ID of the added example
        
    Raises:
        PatternStoreError: If pattern not found
    """
    store = get_pattern_store()
    return store.add_example(pattern_id, text, file_path, line_number, is_positive)


def add_pattern_stats(pattern_id: str, matches: int, files_searched: int,
                    true_positives: Optional[int] = None, 
                    false_positives: Optional[int] = None,
                    precision: Optional[float] = None,
                    recall: Optional[float] = None,
                    f1_score: Optional[float] = None,
                    execution_time: Optional[float] = None,
                    context: Optional[Dict[str, Any]] = None) -> str:
    """
    Add statistics for a pattern.
    
    Args:
        pattern_id: ID of the pattern
        matches: Number of matches
        files_searched: Number of files searched
        true_positives: Number of true positives
        false_positives: Number of false positives
        precision: Precision score
        recall: Recall score
        f1_score: F1 score
        execution_time: Execution time in seconds
        context: Additional context information
        
    Returns:
        ID of the added stats
        
    Raises:
        PatternStoreError: If pattern not found
    """
    store = get_pattern_store()
    return store.add_stats(
        pattern_id, matches, files_searched, true_positives, false_positives,
        precision, recall, f1_score, execution_time, context
    )