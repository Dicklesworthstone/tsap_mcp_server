"""
Pydantic models for the evolution system API endpoints.

This module defines the request and response models for evolutionary features,
including strategy evolution, pattern optimization, learning, and analysis.
"""

from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum

from tsap.mcp.models import (
    StrategyEvolutionParams, StrategyEvolutionResult,
    PatternLibraryParams, StrategyJournalParams, StrategyJournalEntry, RuntimeLearningParams, RuntimeLearningResult,
    OfflineLearningParams, OfflineLearningResult
)


class SelectionMethod(str, Enum):
    """Selection methods for genetic algorithms."""
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"
    RANDOM = "random"
    BEST = "best"


class CrossoverMethod(str, Enum):
    """Crossover methods for genetic algorithms."""
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    BLEND = "blend"
    ADAPTIVE = "adaptive"


class MutationMethod(str, Enum):
    """Mutation methods for genetic algorithms."""
    RANDOM = "random"
    SWAP = "swap"
    INVERSION = "inversion"
    SCRAMBLE = "scramble"
    ADAPTIVE = "adaptive"


class EvolutionConfigRequest(BaseModel):
    """Configuration for an evolutionary algorithm."""
    population_size: int = Field(20, description="Size of the population")
    generations: int = Field(10, description="Number of generations to evolve")
    mutation_rate: float = Field(0.1, description="Probability of mutation")
    crossover_rate: float = Field(0.7, description="Probability of crossover")
    elitism: int = Field(2, description="Number of top individuals to preserve unchanged")
    selection_method: SelectionMethod = Field(SelectionMethod.TOURNAMENT, description="Method for selecting parents")
    crossover_method: CrossoverMethod = Field(CrossoverMethod.SINGLE_POINT, description="Method for crossing over genomes")
    mutation_method: MutationMethod = Field(MutationMethod.RANDOM, description="Method for mutating genomes")
    fitness_target: Optional[float] = Field(None, description="Target fitness to stop evolution")
    max_runtime: Optional[int] = Field(None, description="Maximum runtime in seconds")
    parallelism: int = Field(4, description="Number of parallel evaluations")
    
    class Config:
        json_schema_extra = {
            "example": {
                "population_size": 30,
                "generations": 20,
                "mutation_rate": 0.15,
                "crossover_rate": 0.8,
                "elitism": 2,
                "selection_method": "tournament",
                "crossover_method": "two_point",
                "mutation_method": "random",
                "fitness_target": 0.95,
                "max_runtime": 300,
                "parallelism": 4
            }
        }


class EvolutionRequest(BaseModel):
    """Base class for evolution-related requests."""
    config: Optional[EvolutionConfigRequest] = Field(None, description="Evolution configuration")
    async_execution: bool = Field(False, description="Whether to execute the evolution asynchronously")
    performance_mode: Optional[str] = Field(None, description="Performance mode for this specific request")
    

class PatternEvolutionRequest(EvolutionRequest):
    """Request for evolving regex patterns."""
    params: Union[Dict[str, Any], PatternLibraryParams] = Field(..., description="Pattern evolution parameters")
    positive_examples: List[str] = Field(..., description="Strings that should match the pattern")
    negative_examples: Optional[List[str]] = Field(None, description="Strings that should not match the pattern")
    initial_patterns: Optional[List[str]] = Field(None, description="Initial patterns to start with")
    
    class Config:
        json_schema_extra = {
            "example": {
                "params": {
                    "description": "Email pattern",
                    "pattern_type": "regex"
                },
                "positive_examples": [
                    "user@example.com",
                    "john.doe@company.co.uk",
                    "info+spam@site.net"
                ],
                "negative_examples": [
                    "not an email",
                    "missing@domain",
                    "@incomplete.com"
                ],
                "config": {
                    "population_size": 30,
                    "generations": 20
                },
                "async_execution": True
            }
        }


class StrategyEvolutionRequest(EvolutionRequest):
    """Request for evolving search and analysis strategies."""
    params: Union[Dict[str, Any], StrategyEvolutionParams] = Field(..., description="Strategy evolution parameters")
    target_documents: List[str] = Field(..., description="Documents to use for strategy evaluation")
    training_queries: List[Dict[str, Any]] = Field(..., description="Queries with expected results for training")
    
    class Config:
        json_schema_extra = {
            "example": {
                "params": {
                    "objective": "Find security vulnerabilities in code",
                    "strategy_type": "search"
                },
                "target_documents": [
                    "./src/main.py",
                    "./src/utils.py"
                ],
                "training_queries": [
                    {
                        "query": "SQL injection",
                        "expected_matches": ["./src/main.py:27", "./src/utils.py:115"]
                    }
                ],
                "config": {
                    "population_size": 20,
                    "generations": 15
                },
                "async_execution": True
            }
        }


class JournalEntryRequest(BaseModel):
    """Request for recording a strategy journal entry."""
    params: Union[Dict[str, Any], StrategyJournalParams] = Field(..., description="Journal entry parameters")
    
    class Config:
        json_schema_extra = {
            "example": {
                "params": {
                    "strategy_id": "s123456",
                    "execution_id": "e789012",
                    "effectiveness": 0.85,
                    "execution_time": 2.5,
                    "context": "security_audit",
                    "notes": "Strategy worked well for detecting XSS vulnerabilities",
                    "tags": ["security", "web", "javascript"]
                }
            }
        }


class JournalAnalysisRequest(BaseModel):
    """Request for analyzing strategy journal entries."""
    strategy_id: Optional[str] = Field(None, description="Specific strategy to analyze")
    date_range: Optional[List[str]] = Field(None, description="Date range [start, end] for filtering")
    tags: Optional[List[str]] = Field(None, description="Tags to filter by")
    context: Optional[str] = Field(None, description="Context to filter by")
    min_effectiveness: Optional[float] = Field(None, description="Minimum effectiveness score")
    
    class Config:
        json_schema_extra = {
            "example": {
                "strategy_id": "s123456",
                "date_range": ["2023-01-01", "2023-06-30"],
                "tags": ["security"],
                "min_effectiveness": 0.7
            }
        }


class RuntimeLearningRequest(BaseModel):
    """Request for runtime learning of patterns and strategies."""
    params: Union[Dict[str, Any], RuntimeLearningParams] = Field(..., description="Runtime learning parameters")
    
    class Config:
        json_schema_extra = {
            "example": {
                "params": {
                    "learning_type": "pattern_optimization",
                    "target_pattern_id": "p123456",
                    "feedback": [
                        {"match": "./src/main.py:27", "relevant": True},
                        {"match": "./src/utils.py:115", "relevant": False}
                    ],
                    "learning_rate": 0.1
                }
            }
        }


class OfflineLearningRequest(BaseModel):
    """Request for offline learning from historical data."""
    params: Union[Dict[str, Any], OfflineLearningParams] = Field(..., description="Offline learning parameters")
    
    class Config:
        json_schema_extra = {
            "example": {
                "params": {
                    "learning_type": "strategy_optimization",
                    "data_source": "journal",
                    "date_range": ["2023-01-01", "2023-06-30"],
                    "tags": ["security"],
                    "optimization_goal": "effectiveness"
                }
            }
        }


class AsyncJobResponse(BaseModel):
    """Response model for asynchronous job submission."""
    job_id: str = Field(..., description="ID of the submitted job")
    status: str = Field(..., description="Initial status of the job")
    submitted_at: datetime = Field(..., description="Timestamp when the job was submitted")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    
    @validator('status')
    def validate_status(cls, v):
        """Validate that the status is a known value."""
        allowed_statuses = {'pending', 'running', 'completed', 'failed', 'canceled'}
        if v not in allowed_statuses:
            raise ValueError(f"Status must be one of: {', '.join(allowed_statuses)}")
        return v


class PatternEvolutionResult(BaseModel):
    """Result model for pattern evolution."""
    pattern: str = Field(..., description="Evolved pattern")
    fitness: float = Field(..., description="Fitness score of the pattern")
    generations: int = Field(..., description="Number of generations evolved")
    positive_matches: int = Field(..., description="Number of positive examples matched")
    negative_matches: int = Field(..., description="Number of negative examples matched")
    execution_time: float = Field(..., description="Total execution time in seconds")
    evolution_history: List[Dict[str, Any]] = Field(..., description="History of evolution by generation")
    alternative_patterns: List[Dict[str, Any]] = Field(..., description="Alternative patterns with fitness scores")


class PatternEvolutionResponse(BaseModel):
    """Response model for pattern evolution API endpoint."""
    result: PatternEvolutionResult
    timestamp: datetime = Field(..., description="Timestamp when the result was generated")
    performance_mode: str = Field(..., description="Performance mode used for the execution")


class StrategyEvolutionResponse(BaseModel):
    """Response model for strategy evolution API endpoint."""
    result: StrategyEvolutionResult
    timestamp: datetime = Field(..., description="Timestamp when the result was generated")
    performance_mode: str = Field(..., description="Performance mode used for the execution")


class JournalEntryResponse(BaseModel):
    """Response model for recording a strategy journal entry."""
    entry_id: str = Field(..., description="ID of the recorded journal entry")
    timestamp: datetime = Field(..., description="Timestamp when the entry was recorded")
    strategy_id: str = Field(..., description="ID of the strategy")
    effectiveness: float = Field(..., description="Effectiveness score")


class JournalAnalysisResponse(BaseModel):
    """Response model for analyzing strategy journal entries."""
    entries: List[StrategyJournalEntry] = Field(..., description="Journal entries matching the criteria")
    summary: Dict[str, Any] = Field(..., description="Statistical summary of the entries")
    trends: Dict[str, Any] = Field(..., description="Identified trends in the data")
    recommendations: List[Dict[str, Any]] = Field(..., description="Strategy improvement recommendations")
    timestamp: datetime = Field(..., description="Timestamp when the analysis was performed")


class RuntimeLearningResponse(BaseModel):
    """Response model for runtime learning."""
    result: RuntimeLearningResult
    timestamp: datetime = Field(..., description="Timestamp when the learning was performed")
    performance_mode: str = Field(..., description="Performance mode used for the execution")


class OfflineLearningResponse(BaseModel):
    """Response model for offline learning."""
    result: OfflineLearningResult
    timestamp: datetime = Field(..., description="Timestamp when the learning was performed")
    performance_mode: str = Field(..., description="Performance mode used for the execution")


class EvolutionJobStatusResponse(BaseModel):
    """Response model for evolution job status API endpoint."""
    job_id: str = Field(..., description="ID of the job")
    status: str = Field(..., description="Current status of the job")
    job_type: str = Field(..., description="Type of evolution job")
    submitted_at: datetime = Field(..., description="Timestamp when the job was submitted")
    started_at: Optional[datetime] = Field(None, description="Timestamp when the job started execution")
    completed_at: Optional[datetime] = Field(None, description="Timestamp when the job completed")
    current_generation: Optional[int] = Field(None, description="Current generation in the evolution process")
    total_generations: Optional[int] = Field(None, description="Total number of generations to evolve")
    best_fitness: Optional[float] = Field(None, description="Best fitness score so far")
    progress: Optional[float] = Field(None, description="Job progress as a percentage")
    message: Optional[str] = Field(None, description="Additional status message")
    
    @validator('status')
    def validate_status(cls, v):
        """Validate that the status is a known value."""
        allowed_statuses = {'pending', 'running', 'completed', 'failed', 'canceled'}
        if v not in allowed_statuses:
            raise ValueError(f"Status must be one of: {', '.join(allowed_statuses)}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
                "type": "pattern_evolution",
                "status": "running",
                "submitted_at": "2023-10-27T10:00:00Z",
                "started_at": "2023-10-27T10:01:00Z",
                "completed_at": None,
                "progress": 50.0,
                "current_generation": 5,
                "total_generations": 10,
                "best_fitness": 0.85,
                "message": "Evolved 5 of 10 generations. Best fitness: 0.8500",
                "result": None,
                "error": None,
                "execution_time": None
            }
        }