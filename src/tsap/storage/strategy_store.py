"""
Strategy storage for TSAP.

This module provides persistent storage for search strategies and analysis strategies,
allowing them to be saved, retrieved, and reused across different sessions.
"""

import os
import time
import uuid
import threading
from typing import Dict, List, Any, Optional, Tuple

from tsap.utils.errors import TSAPError
from tsap.storage.database import Database, get_database, create_database


class StrategyStoreError(TSAPError):
    """Exception raised for strategy storage errors."""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.details = details or {}


class StrategyStore:
    """
    Persistent storage for search and analysis strategies.
    
    This class provides methods for saving, retrieving, and querying
    strategies in a SQLite database.
    """
    
    def __init__(self, db: Database):
        """
        Initialize a strategy store.
        
        Args:
            db: Database instance
        """
        self.db = db
        self._lock = threading.RLock()
        self._initialize_schema()
    
    def _initialize_schema(self) -> None:
        """Initialize the database schema for strategy storage."""
        with self._lock:
            with self.db.transaction():
                # Create strategies table
                if not self.db.table_exists('strategies'):
                    self.db.create_table(
                        'strategies',
                        {
                            'strategy_id': 'TEXT NOT NULL',
                            'name': 'TEXT NOT NULL',
                            'description': 'TEXT',
                            'strategy_type': 'TEXT NOT NULL',  # 'search', 'analysis', etc.
                            'objective': 'TEXT',
                            'strategy_data': 'TEXT NOT NULL',
                            'executable': 'INTEGER NOT NULL DEFAULT 1',
                            'complexity': 'INTEGER',
                            'confidence': 'REAL',
                            'effectiveness_score': 'REAL',
                            'average_execution_time': 'REAL',
                            'usage_count': 'INTEGER NOT NULL DEFAULT 0',
                            'created_at': 'REAL NOT NULL',
                            'updated_at': 'REAL NOT NULL',
                            'created_by': 'TEXT',
                            'parent_strategy_id': 'TEXT'
                        },
                        primary_key='strategy_id'
                    )
                    
                    # Create indices
                    self.db.create_index('idx_strategies_name', 'strategies', 'name')
                    self.db.create_index('idx_strategies_type', 'strategies', 'strategy_type')
                    self.db.create_index('idx_strategies_created_at', 'strategies', 'created_at')
                
                # Create strategy tags table
                if not self.db.table_exists('strategy_tags'):
                    self.db.create_table(
                        'strategy_tags',
                        {
                            'strategy_id': 'TEXT NOT NULL',
                            'tag': 'TEXT NOT NULL'
                        },
                        primary_key=['strategy_id', 'tag']
                    )
                    
                    # Create index on tag
                    self.db.create_index('idx_st_tag', 'strategy_tags', 'tag')
                
                # Create strategy executions table
                if not self.db.table_exists('strategy_executions'):
                    self.db.create_table(
                        'strategy_executions',
                        {
                            'execution_id': 'TEXT NOT NULL',
                            'strategy_id': 'TEXT NOT NULL',
                            'parameters': 'TEXT',
                            'result_summary': 'TEXT',
                            'success': 'INTEGER NOT NULL DEFAULT 0',
                            'execution_time': 'REAL',
                            'executed_at': 'REAL NOT NULL',
                            'target_data': 'TEXT',  # Description of the data the strategy was executed on
                            'error_message': 'TEXT'
                        },
                        primary_key='execution_id'
                    )
                    
                    # Create index on strategy_id
                    self.db.create_index('idx_se_strategy_id', 'strategy_executions', 'strategy_id')
                    self.db.create_index('idx_se_executed_at', 'strategy_executions', 'executed_at')
                
                # Create strategy operations table
                if not self.db.table_exists('strategy_operations'):
                    self.db.create_table(
                        'strategy_operations',
                        {
                            'operation_id': 'TEXT NOT NULL',
                            'strategy_id': 'TEXT NOT NULL',
                            'sequence': 'INTEGER NOT NULL',
                            'operation_type': 'TEXT NOT NULL',
                            'operation_name': 'TEXT NOT NULL',
                            'parameters': 'TEXT',
                            'description': 'TEXT',
                            'depends_on': 'TEXT'  # Comma-separated list of operation IDs this depends on
                        },
                        primary_key='operation_id'
                    )
                    
                    # Create index on strategy_id
                    self.db.create_index('idx_so_strategy_id', 'strategy_operations', 'strategy_id')
                    self.db.create_index('idx_so_strategy_sequence', 'strategy_operations', ['strategy_id', 'sequence'])
    
    def add_strategy(self, name: str, strategy_type: str, strategy_data: Dict[str, Any],
                   description: Optional[str] = None, objective: Optional[str] = None,
                   executable: bool = True, complexity: Optional[int] = None,
                   confidence: Optional[float] = None, 
                   tags: Optional[List[str]] = None,
                   created_by: Optional[str] = None,
                   parent_strategy_id: Optional[str] = None,
                   strategy_id: Optional[str] = None) -> str:
        """
        Add a strategy to the store.
        
        Args:
            name: Strategy name
            strategy_type: Strategy type ('search', 'analysis', etc.)
            strategy_data: Strategy data (structure depends on type)
            description: Optional description
            objective: Optional objective
            executable: Whether the strategy is executable
            complexity: Optional complexity score (higher for more complex)
            confidence: Optional confidence score (0-1)
            tags: Optional tags
            created_by: Optional creator identifier
            parent_strategy_id: Optional ID of the parent strategy
            strategy_id: Optional strategy ID (generated if None)
            
        Returns:
            ID of the added strategy
        """
        with self._lock:
            with self.db.transaction():
                # Generate strategy ID if not provided
                if strategy_id is None:
                    strategy_id = str(uuid.uuid4())
                
                # Current time
                current_time = time.time()
                
                # Serialize strategy data
                strategy_data_json = self.db.json_serialize(strategy_data)
                
                # Insert strategy
                self.db.insert(
                    'strategies',
                    {
                        'strategy_id': strategy_id,
                        'name': name,
                        'description': description,
                        'strategy_type': strategy_type,
                        'objective': objective,
                        'strategy_data': strategy_data_json,
                        'executable': 1 if executable else 0,
                        'complexity': complexity,
                        'confidence': confidence,
                        'effectiveness_score': None,
                        'average_execution_time': None,
                        'usage_count': 0,
                        'created_at': current_time,
                        'updated_at': current_time,
                        'created_by': created_by,
                        'parent_strategy_id': parent_strategy_id
                    }
                )
                
                # Add tags if provided
                if tags:
                    for tag in tags:
                        self.db.insert(
                            'strategy_tags',
                            {
                                'strategy_id': strategy_id,
                                'tag': tag
                            }
                        )
                
                # Add operations if provided in strategy data
                operations = strategy_data.get('operations', [])
                if operations:
                    for i, operation in enumerate(operations):
                        operation_id = operation.get('id') or str(uuid.uuid4())
                        depends_on = ','.join(operation.get('depends_on', [])) or None
                        
                        self.db.insert(
                            'strategy_operations',
                            {
                                'operation_id': operation_id,
                                'strategy_id': strategy_id,
                                'sequence': i,
                                'operation_type': operation.get('type', 'unknown'),
                                'operation_name': operation.get('name', f'Operation {i+1}'),
                                'parameters': self.db.json_serialize(operation.get('parameters', {})),
                                'description': operation.get('description'),
                                'depends_on': depends_on
                            }
                        )
                
                return strategy_id
    
    def update_strategy(self, strategy_id: str, name: Optional[str] = None,
                      description: Optional[str] = None, objective: Optional[str] = None,
                      strategy_data: Optional[Dict[str, Any]] = None,
                      executable: Optional[bool] = None,
                      complexity: Optional[int] = None,
                      confidence: Optional[float] = None) -> bool:
        """
        Update a strategy in the store.
        
        Args:
            strategy_id: ID of the strategy
            name: New name
            description: New description
            objective: New objective
            strategy_data: New strategy data
            executable: Whether the strategy is executable
            complexity: New complexity score
            confidence: New confidence score
            
        Returns:
            True if strategy was updated, False if not found
        """
        with self._lock:
            # Check if strategy exists
            existing = self.db.query_one(
                'SELECT 1 FROM strategies WHERE strategy_id = ?',
                (strategy_id,)
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
            
            if objective is not None:
                update_data['objective'] = objective
            
            if strategy_data is not None:
                update_data['strategy_data'] = self.db.json_serialize(strategy_data)
            
            if executable is not None:
                update_data['executable'] = 1 if executable else 0
            
            if complexity is not None:
                update_data['complexity'] = complexity
            
            if confidence is not None:
                update_data['confidence'] = confidence
            
            # Update strategy
            with self.db.transaction():
                self.db.update(
                    'strategies',
                    update_data,
                    'strategy_id = ?',
                    (strategy_id,)
                )
                
                # Update operations if strategy data provided
                if strategy_data is not None and 'operations' in strategy_data:
                    # Delete existing operations
                    self.db.delete('strategy_operations', 'strategy_id = ?', (strategy_id,))
                    
                    # Add new operations
                    operations = strategy_data.get('operations', [])
                    for i, operation in enumerate(operations):
                        operation_id = operation.get('id') or str(uuid.uuid4())
                        depends_on = ','.join(operation.get('depends_on', [])) or None
                        
                        self.db.insert(
                            'strategy_operations',
                            {
                                'operation_id': operation_id,
                                'strategy_id': strategy_id,
                                'sequence': i,
                                'operation_type': operation.get('type', 'unknown'),
                                'operation_name': operation.get('name', f'Operation {i+1}'),
                                'parameters': self.db.json_serialize(operation.get('parameters', {})),
                                'description': operation.get('description'),
                                'depends_on': depends_on
                            }
                        )
                
                return True
    
    def delete_strategy(self, strategy_id: str) -> bool:
        """
        Delete a strategy from the store.
        
        Args:
            strategy_id: ID of the strategy
            
        Returns:
            True if strategy was deleted, False if not found
        """
        with self._lock:
            # Check if strategy exists
            existing = self.db.query_one(
                'SELECT 1 FROM strategies WHERE strategy_id = ?',
                (strategy_id,)
            )
            
            if not existing:
                return False
            
            # Delete strategy and related data
            with self.db.transaction():
                # Delete tags
                self.db.delete('strategy_tags', 'strategy_id = ?', (strategy_id,))
                
                # Delete operations
                self.db.delete('strategy_operations', 'strategy_id = ?', (strategy_id,))
                
                # Delete executions
                self.db.delete('strategy_executions', 'strategy_id = ?', (strategy_id,))
                
                # Delete strategy
                self.db.delete('strategies', 'strategy_id = ?', (strategy_id,))
                
                return True
    
    def get_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a strategy from the store.
        
        Args:
            strategy_id: ID of the strategy
            
        Returns:
            Strategy data or None if not found
        """
        with self._lock:
            # Get strategy data
            strategy = self.db.query_one(
                'SELECT * FROM strategies WHERE strategy_id = ?',
                (strategy_id,)
            )
            
            if not strategy:
                return None
            
            # Convert boolean fields
            strategy['executable'] = bool(strategy['executable'])
            
            # Deserialize strategy data
            strategy['strategy_data'] = self.db.json_deserialize(strategy['strategy_data'])
            
            # Get tags
            tags = self.db.query(
                'SELECT tag FROM strategy_tags WHERE strategy_id = ?',
                (strategy_id,)
            )
            
            strategy['tags'] = [tag['tag'] for tag in tags]
            
            # Get operations
            operations = self.db.query(
                'SELECT * FROM strategy_operations WHERE strategy_id = ? ORDER BY sequence',
                (strategy_id,)
            )
            
            strategy_operations = []
            for operation in operations:
                operation_data = {
                    'id': operation['operation_id'],
                    'sequence': operation['sequence'],
                    'type': operation['operation_type'],
                    'name': operation['operation_name'],
                    'parameters': self.db.json_deserialize(operation['parameters']),
                    'description': operation['description']
                }
                
                if operation['depends_on']:
                    operation_data['depends_on'] = operation['depends_on'].split(',')
                else:
                    operation_data['depends_on'] = []
                
                strategy_operations.append(operation_data)
            
            strategy['operations'] = strategy_operations
            
            # Get recent executions
            executions = self.db.query(
                '''
                SELECT * FROM strategy_executions 
                WHERE strategy_id = ? 
                ORDER BY executed_at DESC LIMIT 5
                ''',
                (strategy_id,)
            )
            
            strategy_executions = []
            for execution in executions:
                execution_data = {
                    'execution_id': execution['execution_id'],
                    'parameters': self.db.json_deserialize(execution['parameters']),
                    'result_summary': self.db.json_deserialize(execution['result_summary']),
                    'success': bool(execution['success']),
                    'execution_time': execution['execution_time'],
                    'executed_at': execution['executed_at'],
                    'target_data': execution['target_data'],
                    'error_message': execution['error_message']
                }
                
                strategy_executions.append(execution_data)
            
            strategy['recent_executions'] = strategy_executions
            
            return strategy
    
    def get_strategy_by_name(self, name: str, strategy_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get a strategy by name.
        
        Args:
            name: Strategy name
            strategy_type: Optional strategy type filter
            
        Returns:
            Strategy data or None if not found
        """
        with self._lock:
            # Build query
            if strategy_type:
                query = '''
                SELECT strategy_id FROM strategies 
                WHERE name = ? AND strategy_type = ? 
                ORDER BY updated_at DESC LIMIT 1
                '''
                params = (name, strategy_type)
            else:
                query = '''
                SELECT strategy_id FROM strategies 
                WHERE name = ? 
                ORDER BY updated_at DESC LIMIT 1
                '''
                params = (name,)
            
            # Get strategy ID
            result = self.db.query_one(query, params)
            
            if not result:
                return None
            
            # Get full strategy data
            return self.get_strategy(result['strategy_id'])
    
    def search_strategies(self, query: Optional[str] = None,
                       strategy_type: Optional[str] = None,
                       tags: Optional[List[str]] = None,
                       executable_only: bool = False,
                       min_confidence: Optional[float] = None,
                       min_effectiveness: Optional[float] = None,
                       min_usage_count: Optional[int] = None,
                       limit: int = 100,
                       offset: int = 0,
                       sort_by: str = 'updated_at',
                       sort_order: str = 'desc') -> Tuple[List[Dict[str, Any]], int]:
        """
        Search for strategies in the store.
        
        Args:
            query: Text search query
            strategy_type: Filter by strategy type
            tags: Filter by tags (AND logic)
            executable_only: Whether to only return executable strategies
            min_confidence: Minimum confidence score
            min_effectiveness: Minimum effectiveness score
            min_usage_count: Minimum usage count
            limit: Maximum number of results to return
            offset: Offset for pagination
            sort_by: Field to sort by
            sort_order: Sort order ('asc' or 'desc')
            
        Returns:
            Tuple of (strategy list, total count)
        """
        with self._lock:
            # Build the query
            query_parts = ['SELECT s.* FROM strategies s']
            where_clauses = []
            params = []
            
            # Join with tags table if filtering by tags
            if tags and len(tags) > 0:
                for i, tag in enumerate(tags):
                    query_parts[0] += f' JOIN strategy_tags st{i} ON s.strategy_id = st{i}.strategy_id'
                    where_clauses.append(f'st{i}.tag = ?')
                    params.append(tag)
            
            # Add filter conditions
            if query:
                # Search in name, description, and objective
                where_clauses.append('(s.name LIKE ? OR s.description LIKE ? OR s.objective LIKE ?)')
                search_term = f'%{query}%'
                params.extend([search_term, search_term, search_term])
            
            if strategy_type:
                where_clauses.append('s.strategy_type = ?')
                params.append(strategy_type)
            
            if executable_only:
                where_clauses.append('s.executable = 1')
            
            if min_confidence is not None:
                where_clauses.append('s.confidence >= ?')
                params.append(min_confidence)
            
            if min_effectiveness is not None:
                where_clauses.append('s.effectiveness_score >= ?')
                params.append(min_effectiveness)
            
            if min_usage_count is not None:
                where_clauses.append('s.usage_count >= ?')
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
                'effectiveness_score', 'complexity', 'usage_count'
            }
            if sort_by not in valid_sort_fields:
                sort_by = 'updated_at'
            
            sort_order = 'DESC' if sort_order.lower() == 'desc' else 'ASC'
            query_parts.append(f'ORDER BY s.{sort_by} {sort_order}')
            
            # Add pagination
            query_parts.append('LIMIT ? OFFSET ?')
            params.append(limit)
            params.append(offset)
            
            # Execute the query
            query = ' '.join(query_parts)
            strategies = self.db.query(query, params)
            
            # Process results
            result = []
            for strategy in strategies:
                # Convert boolean fields
                strategy['executable'] = bool(strategy['executable'])
                
                # Deserialize strategy data
                strategy['strategy_data'] = self.db.json_deserialize(strategy['strategy_data'])
                
                # Get tags
                tags = self.db.query(
                    'SELECT tag FROM strategy_tags WHERE strategy_id = ?',
                    (strategy['strategy_id'],)
                )
                
                strategy['tags'] = [tag['tag'] for tag in tags]
                
                # Get operation count
                operations_count = self.db.query_one(
                    'SELECT COUNT(*) as count FROM strategy_operations WHERE strategy_id = ?',
                    (strategy['strategy_id'],)
                )
                
                strategy['operations_count'] = operations_count['count'] if operations_count else 0
                
                result.append(strategy)
            
            return result, total_count
    
    def add_tag(self, strategy_id: str, tag: str) -> bool:
        """
        Add a tag to a strategy.
        
        Args:
            strategy_id: ID of the strategy
            tag: Tag to add
            
        Returns:
            True if tag was added, False if strategy not found
        """
        with self._lock:
            # Check if strategy exists
            existing = self.db.query_one(
                'SELECT 1 FROM strategies WHERE strategy_id = ?',
                (strategy_id,)
            )
            
            if not existing:
                return False
            
            # Check if tag already exists
            existing_tag = self.db.query_one(
                'SELECT 1 FROM strategy_tags WHERE strategy_id = ? AND tag = ?',
                (strategy_id, tag)
            )
            
            if existing_tag:
                return True  # Tag already exists
            
            # Add tag
            with self.db.transaction():
                self.db.insert(
                    'strategy_tags',
                    {
                        'strategy_id': strategy_id,
                        'tag': tag
                    }
                )
                
                # Update strategy updated_at
                self.db.update(
                    'strategies',
                    {'updated_at': time.time()},
                    'strategy_id = ?',
                    (strategy_id,)
                )
                
                return True
    
    def remove_tag(self, strategy_id: str, tag: str) -> bool:
        """
        Remove a tag from a strategy.
        
        Args:
            strategy_id: ID of the strategy
            tag: Tag to remove
            
        Returns:
            True if tag was removed, False if strategy or tag not found
        """
        with self._lock:
            with self.db.transaction():
                # Delete tag
                rows_affected = self.db.delete(
                    'strategy_tags',
                    'strategy_id = ? AND tag = ?',
                    (strategy_id, tag)
                )
                
                if rows_affected > 0:
                    # Update strategy updated_at
                    self.db.update(
                        'strategies',
                        {'updated_at': time.time()},
                        'strategy_id = ?',
                        (strategy_id,)
                    )
                    
                    return True
                
                return False
    
    def record_execution(self, strategy_id: str, parameters: Optional[Dict[str, Any]] = None,
                       result_summary: Optional[Dict[str, Any]] = None,
                       success: bool = True, execution_time: Optional[float] = None,
                       target_data: Optional[str] = None,
                       error_message: Optional[str] = None) -> str:
        """
        Record a strategy execution.
        
        Args:
            strategy_id: ID of the strategy
            parameters: Execution parameters
            result_summary: Summary of execution results
            success: Whether the execution was successful
            execution_time: Execution time in seconds
            target_data: Description of the data the strategy was executed on
            error_message: Error message if execution failed
            
        Returns:
            ID of the recorded execution
            
        Raises:
            StrategyStoreError: If strategy not found
        """
        with self._lock:
            # Check if strategy exists
            existing = self.db.query_one(
                'SELECT 1 FROM strategies WHERE strategy_id = ?',
                (strategy_id,)
            )
            
            if not existing:
                raise StrategyStoreError(f"Strategy not found: {strategy_id}")
            
            # Generate execution ID
            execution_id = str(uuid.uuid4())
            
            # Current time
            current_time = time.time()
            
            # Serialize parameters and result summary
            parameters_json = self.db.json_serialize(parameters)
            result_summary_json = self.db.json_serialize(result_summary)
            
            # Add execution record
            with self.db.transaction():
                self.db.insert(
                    'strategy_executions',
                    {
                        'execution_id': execution_id,
                        'strategy_id': strategy_id,
                        'parameters': parameters_json,
                        'result_summary': result_summary_json,
                        'success': 1 if success else 0,
                        'execution_time': execution_time,
                        'executed_at': current_time,
                        'target_data': target_data,
                        'error_message': error_message
                    }
                )
                
                # Update strategy usage count and average execution time
                strategy = self.db.query_one(
                    'SELECT usage_count, average_execution_time FROM strategies WHERE strategy_id = ?',
                    (strategy_id,)
                )
                
                usage_count = strategy['usage_count'] + 1
                
                if execution_time is not None:
                    if strategy['average_execution_time'] is None:
                        avg_execution_time = execution_time
                    else:
                        # Weighted average based on usage count
                        avg_execution_time = (
                            (strategy['average_execution_time'] * (usage_count - 1)) + execution_time
                        ) / usage_count
                else:
                    avg_execution_time = strategy['average_execution_time']
                
                # Update effectiveness score if successful
                if success and result_summary:
                    # Extract effectiveness metrics from result summary if available
                    effectiveness_score = None
                    if isinstance(result_summary, dict):
                        effectiveness_score = result_summary.get('effectiveness_score')
                        if effectiveness_score is None:
                            metrics = result_summary.get('metrics', {})
                            if isinstance(metrics, dict):
                                effectiveness_score = metrics.get('effectiveness') or metrics.get('f1_score')
                    
                    if effectiveness_score is not None:
                        self.db.update(
                            'strategies',
                            {
                                'usage_count': usage_count,
                                'average_execution_time': avg_execution_time,
                                'effectiveness_score': effectiveness_score,
                                'updated_at': current_time
                            },
                            'strategy_id = ?',
                            (strategy_id,)
                        )
                    else:
                        self.db.update(
                            'strategies',
                            {
                                'usage_count': usage_count,
                                'average_execution_time': avg_execution_time,
                                'updated_at': current_time
                            },
                            'strategy_id = ?',
                            (strategy_id,)
                        )
                else:
                    self.db.update(
                        'strategies',
                        {
                            'usage_count': usage_count,
                            'average_execution_time': avg_execution_time,
                            'updated_at': current_time
                        },
                        'strategy_id = ?',
                        (strategy_id,)
                    )
                
                return execution_id
    
    def get_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a strategy execution record.
        
        Args:
            execution_id: ID of the execution
            
        Returns:
            Execution data or None if not found
        """
        with self._lock:
            # Get execution data
            execution = self.db.query_one(
                'SELECT * FROM strategy_executions WHERE execution_id = ?',
                (execution_id,)
            )
            
            if not execution:
                return None
            
            # Convert boolean fields
            execution['success'] = bool(execution['success'])
            
            # Deserialize fields
            execution['parameters'] = self.db.json_deserialize(execution['parameters'])
            execution['result_summary'] = self.db.json_deserialize(execution['result_summary'])
            
            return execution
    
    def get_strategy_executions(self, strategy_id: str, 
                             successful_only: bool = False,
                             limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get execution records for a strategy.
        
        Args:
            strategy_id: ID of the strategy
            successful_only: Whether to only return successful executions
            limit: Maximum number of records to return
            
        Returns:
            List of execution records
        """
        with self._lock:
            # Build query
            query = 'SELECT * FROM strategy_executions WHERE strategy_id = ?'
            params = [strategy_id]
            
            if successful_only:
                query += ' AND success = 1'
            
            query += ' ORDER BY executed_at DESC LIMIT ?'
            params.append(limit)
            
            # Get executions
            executions = self.db.query(query, params)
            
            # Process results
            result = []
            for execution in executions:
                # Convert boolean fields
                execution['success'] = bool(execution['success'])
                
                # Deserialize fields
                execution['parameters'] = self.db.json_deserialize(execution['parameters'])
                execution['result_summary'] = self.db.json_deserialize(execution['result_summary'])
                
                result.append(execution)
            
            return result
    
    def get_popular_strategies(self, limit: int = 10, 
                            strategy_type: Optional[str] = None,
                            tag: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get the most popular strategies.
        
        Args:
            limit: Maximum number of strategies to return
            strategy_type: Optional strategy type filter
            tag: Optional tag filter
            
        Returns:
            List of popular strategies
        """
        with self._lock:
            query = 'SELECT s.* FROM strategies s'
            where_clauses = []
            params = []
            
            # Join with tags table if filtering by tag
            if tag:
                query += ' JOIN strategy_tags st ON s.strategy_id = st.strategy_id'
                where_clauses.append('st.tag = ?')
                params.append(tag)
            
            # Add type filter
            if strategy_type:
                where_clauses.append('s.strategy_type = ?')
                params.append(strategy_type)
            
            # Add where clause if needed
            if where_clauses:
                query += ' WHERE ' + ' AND '.join(where_clauses)
            
            # Add sorting and limit
            query += ' ORDER BY s.usage_count DESC, s.effectiveness_score DESC LIMIT ?'
            params.append(limit)
            
            # Execute query
            strategies = self.db.query(query, params)
            
            # Process results
            result = []
            for strategy in strategies:
                # Convert boolean fields
                strategy['executable'] = bool(strategy['executable'])
                
                # Deserialize strategy data
                strategy['strategy_data'] = self.db.json_deserialize(strategy['strategy_data'])
                
                # Get tags
                tags = self.db.query(
                    'SELECT tag FROM strategy_tags WHERE strategy_id = ?',
                    (strategy['strategy_id'],)
                )
                
                strategy['tags'] = [tag['tag'] for tag in tags]
                
                result.append(strategy)
            
            return result
    
    def get_strategy_evolution(self, root_strategy_id: str) -> List[Dict[str, Any]]:
        """
        Get the evolution history of a strategy.
        
        Args:
            root_strategy_id: ID of the root strategy
            
        Returns:
            List of strategies in the evolution chain
        """
        with self._lock:
            # Get root strategy
            root_strategy = self.get_strategy(root_strategy_id)
            
            if not root_strategy:
                return []
            
            # Initialize result with root strategy
            result = [root_strategy]
            
            # Get all descendants
            descendants = []
            to_check = [root_strategy_id]
            
            while to_check:
                current_id = to_check.pop(0)
                
                children = self.db.query(
                    'SELECT * FROM strategies WHERE parent_strategy_id = ?',
                    (current_id,)
                )
                
                for child in children:
                    # Convert boolean fields
                    child['executable'] = bool(child['executable'])
                    
                    # Deserialize strategy data
                    child['strategy_data'] = self.db.json_deserialize(child['strategy_data'])
                    
                    # Get tags
                    tags = self.db.query(
                        'SELECT tag FROM strategy_tags WHERE strategy_id = ?',
                        (child['strategy_id'],)
                    )
                    
                    child['tags'] = [tag['tag'] for tag in tags]
                    
                    descendants.append(child)
                    to_check.append(child['strategy_id'])
            
            # Sort descendants by created_at
            descendants.sort(key=lambda s: s['created_at'])
            
            # Add to result
            result.extend(descendants)
            
            return result
    
    def find_similar_strategies(self, objective: str, strategy_type: Optional[str] = None,
                             max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Find strategies similar to a given objective.
        
        Args:
            objective: The objective to find similar strategies for
            strategy_type: Optional filter by strategy type
            max_results: Maximum number of results to return
            
        Returns:
            List of similar strategies
        """
        with self._lock:
            # Build query to get all strategies
            query = 'SELECT * FROM strategies WHERE objective IS NOT NULL'
            params = []
            
            if strategy_type:
                query += ' AND strategy_type = ?'
                params.append(strategy_type)
            
            # Get strategies
            strategies = self.db.query(query, params)
            
            # Calculate similarity scores
            strategies_with_scores = []
            
            for strategy in strategies:
                if not strategy['objective']:
                    continue
                
                # Calculate similarity between objectives
                similarity = self._calculate_text_similarity(objective, strategy['objective'])
                
                strategies_with_scores.append((strategy, similarity))
            
            # Sort by similarity (highest first)
            strategies_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Take top results
            results = []
            for strategy, similarity in strategies_with_scores[:max_results]:
                # Convert boolean fields
                strategy['executable'] = bool(strategy['executable'])
                
                # Deserialize strategy data
                strategy['strategy_data'] = self.db.json_deserialize(strategy['strategy_data'])
                
                # Get tags
                tags = self.db.query(
                    'SELECT tag FROM strategy_tags WHERE strategy_id = ?',
                    (strategy['strategy_id'],)
                )
                
                strategy['tags'] = [tag['tag'] for tag in tags]
                
                # Add similarity score
                strategy['similarity'] = similarity
                
                results.append(strategy)
            
            return results
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0.0 - 1.0)
        """
        # Extract words and normalize
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
            
        return intersection / union


# Global registry of strategy stores
_strategy_stores = {}
_strategy_lock = threading.RLock()


def get_strategy_store(db_name: str = 'strategies') -> StrategyStore:
    """
    Get a strategy store.
    
    Args:
        db_name: Database name
        
    Returns:
        StrategyStore instance
    """
    with _strategy_lock:
        if db_name in _strategy_stores:
            return _strategy_stores[db_name]
        
        # Get or create the database
        db = get_database(db_name)
        
        if db is None:
            # Create strategy database in data directory
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
        
        # Create strategy store
        store = StrategyStore(db)
        _strategy_stores[db_name] = store
        
        return store


def add_strategy(name: str, strategy_type: str, strategy_data: Dict[str, Any],
               description: Optional[str] = None, objective: Optional[str] = None,
               executable: bool = True, complexity: Optional[int] = None,
               confidence: Optional[float] = None, 
               tags: Optional[List[str]] = None,
               created_by: Optional[str] = None) -> str:
    """
    Add a strategy to the store.
    
    Args:
        name: Strategy name
        strategy_type: Strategy type ('search', 'analysis', etc.)
        strategy_data: Strategy data (structure depends on type)
        description: Optional description
        objective: Optional objective
        executable: Whether the strategy is executable
        complexity: Optional complexity score (higher for more complex)
        confidence: Optional confidence score (0-1)
        tags: Optional tags
        created_by: Optional creator identifier
        
    Returns:
        ID of the added strategy
    """
    store = get_strategy_store()
    return store.add_strategy(
        name, strategy_type, strategy_data, description, objective,
        executable, complexity, confidence, tags, created_by
    )


def get_strategy(strategy_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a strategy from the store.
    
    Args:
        strategy_id: ID of the strategy
        
    Returns:
        Strategy data or None if not found
    """
    store = get_strategy_store()
    return store.get_strategy(strategy_id)


def get_strategy_by_name(name: str, strategy_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Get a strategy by name.
    
    Args:
        name: Strategy name
        strategy_type: Optional strategy type filter
        
    Returns:
        Strategy data or None if not found
    """
    store = get_strategy_store()
    return store.get_strategy_by_name(name, strategy_type)


def search_strategies(query: Optional[str] = None,
                   strategy_type: Optional[str] = None,
                   tags: Optional[List[str]] = None,
                   executable_only: bool = False,
                   min_confidence: Optional[float] = None,
                   min_effectiveness: Optional[float] = None,
                   min_usage_count: Optional[int] = None,
                   limit: int = 100,
                   offset: int = 0) -> Tuple[List[Dict[str, Any]], int]:
    """
    Search for strategies in the store.
    
    Args:
        query: Text search query
        strategy_type: Filter by strategy type
        tags: Filter by tags (AND logic)
        executable_only: Whether to only return executable strategies
        min_confidence: Minimum confidence score
        min_effectiveness: Minimum effectiveness score
        min_usage_count: Minimum usage count
        limit: Maximum number of results to return
        offset: Offset for pagination
        
    Returns:
        Tuple of (strategy list, total count)
    """
    store = get_strategy_store()
    return store.search_strategies(
        query, strategy_type, tags, executable_only, min_confidence,
        min_effectiveness, min_usage_count, limit, offset
    )


def record_strategy_execution(strategy_id: str, parameters: Optional[Dict[str, Any]] = None,
                           result_summary: Optional[Dict[str, Any]] = None,
                           success: bool = True, execution_time: Optional[float] = None,
                           target_data: Optional[str] = None,
                           error_message: Optional[str] = None) -> str:
    """
    Record a strategy execution.
    
    Args:
        strategy_id: ID of the strategy
        parameters: Execution parameters
        result_summary: Summary of execution results
        success: Whether the execution was successful
        execution_time: Execution time in seconds
        target_data: Description of the data the strategy was executed on
        error_message: Error message if execution failed
        
    Returns:
        ID of the recorded execution
        
    Raises:
        StrategyStoreError: If strategy not found
    """
    store = get_strategy_store()
    return store.record_execution(
        strategy_id, parameters, result_summary, success,
        execution_time, target_data, error_message
    )


def get_popular_strategies(limit: int = 10, 
                        strategy_type: Optional[str] = None,
                        tag: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get the most popular strategies.
    
    Args:
        limit: Maximum number of strategies to return
        strategy_type: Optional strategy type filter
        tag: Optional tag filter
        
    Returns:
        List of popular strategies
    """
    store = get_strategy_store()
    return store.get_popular_strategies(limit, strategy_type, tag)


def find_similar_strategies(objective: str, strategy_type: Optional[str] = None,
                         max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Find strategies similar to a given objective.
    
    Args:
        objective: The objective to find similar strategies for
        strategy_type: Optional filter by strategy type
        max_results: Maximum number of results to return
        
    Returns:
        List of similar strategies
    """
    store = get_strategy_store()
    return store.find_similar_strategies(objective, strategy_type, max_results)