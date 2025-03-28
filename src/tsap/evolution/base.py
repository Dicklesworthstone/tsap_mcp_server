"""
Base classes and utilities for evolutionary algorithms.

This module defines the core abstractions for evolutionary algorithms,
including population management, fitness evaluation, selection, crossover,
mutation, and generational progression.
"""

import time
import asyncio
import random
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, TypeVar, Generic, Type
from dataclasses import dataclass, field
from enum import Enum

from tsap.utils.logging import logger
from tsap.utils.errors import TSAPError

# Type variables for genome and fitness
G = TypeVar('G')  # Genome type
F = TypeVar('F')  # Fitness type (usually float, but could be more complex)


class EvolutionError(TSAPError):
    """
    Exception raised for errors in evolutionary algorithms.
    
    Attributes:
        message: Error message
        algorithm: Name of the algorithm that caused the error
        details: Additional error details
    """
    def __init__(self, message: str, algorithm: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, code=f"EVOLUTION_{algorithm.upper()}_ERROR" if algorithm else "EVOLUTION_ERROR", details=details)
        self.algorithm = algorithm


class SelectionMethod(str, Enum):
    """Selection methods for choosing parents in genetic algorithms."""
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"
    RANDOM = "random"
    BEST = "best"


class CrossoverMethod(str, Enum):
    """Crossover methods for combining parent genomes."""
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    BLEND = "blend"
    ADAPTIVE = "adaptive"


class MutationMethod(str, Enum):
    """Mutation methods for introducing variation in genomes."""
    RANDOM = "random"
    SWAP = "swap"
    INVERSION = "inversion"
    SCRAMBLE = "scramble"
    ADAPTIVE = "adaptive"


@dataclass
class EvolutionConfig:
    """Configuration for an evolutionary algorithm."""
    population_size: int = 20
    generations: int = 10
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elitism: int = 2
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    crossover_method: CrossoverMethod = CrossoverMethod.SINGLE_POINT
    mutation_method: MutationMethod = MutationMethod.RANDOM
    fitness_target: Optional[float] = None
    max_runtime: Optional[int] = None
    parallelism: int = 4
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.population_size < 2:
            raise ValueError("Population size must be at least 2")
        if self.generations < 1:
            raise ValueError("Number of generations must be at least 1")
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError("Mutation rate must be between 0 and 1")
        if not 0 <= self.crossover_rate <= 1:
            raise ValueError("Crossover rate must be between 0 and 1")
        if self.elitism < 0:
            raise ValueError("Elitism must be non-negative")
        if self.elitism >= self.population_size:
            raise ValueError("Elitism must be less than population size")
        if self.parallelism < 1:
            raise ValueError("Parallelism must be at least 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return {
            "population_size": self.population_size,
            "generations": self.generations,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "elitism": self.elitism,
            "selection_method": self.selection_method,
            "crossover_method": self.crossover_method,
            "mutation_method": self.mutation_method,
            "fitness_target": self.fitness_target,
            "max_runtime": self.max_runtime,
            "parallelism": self.parallelism
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvolutionConfig':
        """Create configuration from a dictionary."""
        # Process enum values
        if "selection_method" in data and not isinstance(data["selection_method"], SelectionMethod):
            data["selection_method"] = SelectionMethod(data["selection_method"])
        if "crossover_method" in data and not isinstance(data["crossover_method"], CrossoverMethod):
            data["crossover_method"] = CrossoverMethod(data["crossover_method"])
        if "mutation_method" in data and not isinstance(data["mutation_method"], MutationMethod):
            data["mutation_method"] = MutationMethod(data["mutation_method"])
        
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class Individual(Generic[G, F]):
    """
    An individual in a population for evolutionary algorithms.
    
    Attributes:
        genome: The genetic representation of the individual
        fitness: The fitness score of the individual (higher is better)
        id: Unique identifier for the individual
        generation: Generation number in which this individual was created
        parent_ids: IDs of the parent individuals (if any)
        metadata: Additional metadata about the individual
    """
    genome: G
    fitness: Optional[F] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert individual to a dictionary."""
        return {
            "id": self.id,
            "genome": self.genome,
            "fitness": self.fitness,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Individual':
        """Create individual from a dictionary."""
        return cls(
            genome=data["genome"],
            fitness=data.get("fitness"),
            id=data.get("id", str(uuid.uuid4())),
            generation=data.get("generation", 0),
            parent_ids=data.get("parent_ids", []),
            metadata=data.get("metadata", {})
        )
    
    def __eq__(self, other) -> bool:
        """Check if two individuals are equal based on their ID."""
        if not isinstance(other, Individual):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        """Hash individual based on ID."""
        return hash(self.id)


@dataclass
class Population(Generic[G, F]):
    """
    A population of individuals for evolutionary algorithms.
    
    Attributes:
        individuals: List of individuals in the population
        generation: Current generation number
        metadata: Additional metadata about the population
    """
    individuals: List[Individual[G, F]] = field(default_factory=list)
    generation: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def size(self) -> int:
        """Get the size of the population."""
        return len(self.individuals)
    
    @property
    def best_individual(self) -> Optional[Individual[G, F]]:
        """Get the individual with the highest fitness."""
        if not self.individuals:
            return None
        return max(self.individuals, key=lambda i: i.fitness if i.fitness is not None else float('-inf'))
    
    @property
    def average_fitness(self) -> float:
        """Get the average fitness of the population."""
        if not self.individuals or all(i.fitness is None for i in self.individuals):
            return 0.0
        valid_fitnesses = [i.fitness for i in self.individuals if i.fitness is not None]
        return sum(valid_fitnesses) / len(valid_fitnesses) if valid_fitnesses else 0.0
    
    @property
    def diversity(self) -> float:
        """
        Get a measure of genetic diversity in the population.
        
        This is a simple placeholder implementation. Actual diversity calculation
        would depend on the specific genome representation.
        """
        if len(self.individuals) <= 1:
            return 0.0
        
        # Placeholder: Return a random diversity value
        return random.uniform(0.0, 1.0)
    
    def add_individual(self, individual: Individual[G, F]) -> None:
        """Add an individual to the population."""
        self.individuals.append(individual)
    
    def remove_individual(self, individual: Individual[G, F]) -> None:
        """Remove an individual from the population."""
        self.individuals.remove(individual)
    
    def sort_by_fitness(self, reverse: bool = True) -> None:
        """
        Sort individuals by fitness.
        
        Args:
            reverse: If True, sort in descending order (highest fitness first).
        """
        self.individuals.sort(
            key=lambda i: i.fitness if i.fitness is not None else float('-inf'),
            reverse=reverse
        )
    
    def get_individual_by_id(self, individual_id: str) -> Optional[Individual[G, F]]:
        """Get an individual by ID."""
        for individual in self.individuals:
            if individual.id == individual_id:
                return individual
        return None
    
    def get_fittest(self, count: int = 1) -> List[Individual[G, F]]:
        """Get the top n fittest individuals."""
        sorted_individuals = sorted(
            [i for i in self.individuals if i.fitness is not None],
            key=lambda i: i.fitness,
            reverse=True
        )
        return sorted_individuals[:count]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert population to a dictionary."""
        return {
            "individuals": [i.to_dict() for i in self.individuals],
            "generation": self.generation,
            "metadata": self.metadata,
            "statistics": {
                "size": self.size,
                "best_fitness": self.best_individual.fitness if self.best_individual else None,
                "average_fitness": self.average_fitness,
                "diversity": self.diversity
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Population':
        """Create population from a dictionary."""
        population = cls(
            individuals=[Individual.from_dict(i) for i in data.get("individuals", [])],
            generation=data.get("generation", 0),
            metadata=data.get("metadata", {})
        )
        return population


@dataclass
class EvolutionResult(Generic[G, F]):
    """
    Result of an evolutionary algorithm.
    
    Attributes:
        best_individual: The individual with the highest fitness
        final_population: The final population after evolution
        generations_history: History of populations for each generation
        statistics: Statistics about the evolution process
        config: Configuration used for the evolution
        runtime: Total runtime in seconds
    """
    best_individual: Individual[G, F]
    final_population: Population[G, F]
    generations_history: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    config: EvolutionConfig
    runtime: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert evolution result to a dictionary."""
        return {
            "best_individual": self.best_individual.to_dict(),
            "final_population": self.final_population.to_dict(),
            "generations_history": self.generations_history,
            "statistics": self.statistics,
            "config": self.config.to_dict(),
            "runtime": self.runtime
        }


class EvolutionAlgorithm(Generic[G, F], ABC):
    """
    Abstract base class for evolutionary algorithms.
    
    Attributes:
        name: Name of the algorithm
        config: Configuration for the algorithm
    """
    def __init__(self, name: str, config: Optional[EvolutionConfig] = None) -> None:
        """
        Initialize a new evolutionary algorithm.
        
        Args:
            name: Name of the algorithm
            config: Configuration for the algorithm
        """
        self.name = name
        self.config = config or EvolutionConfig()
        self._stats = {
            "executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_runtime": 0.0,
            "average_runtime": 0.0,
            "last_runtime": None,
            "last_executed_at": None,
            "best_fitness_achieved": None
        }
    
    @abstractmethod
    async def initialize_population(self) -> Population[G, F]:
        """
        Initialize the initial population.
        
        Returns:
            Initial population
        """
        pass
    
    @abstractmethod
    async def evaluate_fitness(self, individual: Individual[G, F]) -> F:
        """
        Evaluate the fitness of an individual.
        
        Args:
            individual: Individual to evaluate
            
        Returns:
            Fitness score
        """
        pass
    
    @abstractmethod
    async def select_parent(self, population: Population[G, F]) -> Individual[G, F]:
        """
        Select a parent individual for reproduction.
        
        Args:
            population: Population to select from
            
        Returns:
            Selected parent individual
        """
        pass
    
    @abstractmethod
    async def crossover(self, parent1: Individual[G, F], parent2: Individual[G, F]) -> Individual[G, F]:
        """
        Perform crossover between two parent individuals.
        
        Args:
            parent1: First parent individual
            parent2: Second parent individual
            
        Returns:
            Child individual
        """
        pass
    
    @abstractmethod
    async def mutate(self, individual: Individual[G, F]) -> Individual[G, F]:
        """
        Mutate an individual.
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Mutated individual
        """
        pass
    
    async def evolve(self, progress_callback: Optional[Callable[[int, int, F], None]] = None) -> EvolutionResult[G, F]:
        """
        Execute the evolutionary algorithm.
        
        Args:
            progress_callback: Optional callback function for reporting progress
                Takes generation number, total generations, and current best fitness
                
        Returns:
            Result of the evolution
        
        Raises:
            EvolutionError: If the algorithm fails
        """
        start_time = time.time()
        self._stats["executions"] += 1
        
        try:
            # Initialize population
            population = await self.initialize_population()
            
            # Evaluate initial population
            await self._evaluate_population(population)
            
            # Initialize generations history
            generations_history = [{
                "generation": 0,
                "best_fitness": population.best_individual.fitness if population.best_individual else None,
                "average_fitness": population.average_fitness,
                "diversity": population.diversity,
                "timestamp": time.time()
            }]
            
            # Report initial progress
            if progress_callback:
                await progress_callback(
                    0, 
                    self.config.generations,
                    population.best_individual.fitness if population.best_individual else 0.0
                )
            
            # Evolution loop
            for generation in range(1, self.config.generations + 1):
                # Check if maximum runtime has been reached
                if self.config.max_runtime and time.time() - start_time > self.config.max_runtime:
                    logger.info(f"Maximum runtime reached after {generation-1} generations")
                    break
                
                # Create next generation
                population = await self._create_next_generation(population)
                population.generation = generation
                
                # Evaluate population
                await self._evaluate_population(population)
                
                # Record generation history
                generations_history.append({
                    "generation": generation,
                    "best_fitness": population.best_individual.fitness if population.best_individual else None,
                    "average_fitness": population.average_fitness,
                    "diversity": population.diversity,
                    "timestamp": time.time()
                })
                
                # Report progress
                if progress_callback:
                    await progress_callback(
                        generation, 
                        self.config.generations,
                        population.best_individual.fitness if population.best_individual else 0.0
                    )
                
                # Check if target fitness has been reached
                if (self.config.fitness_target is not None and 
                    population.best_individual and 
                    population.best_individual.fitness >= self.config.fitness_target):
                    logger.info(f"Target fitness reached after {generation} generations")
                    break
            
            # Get best individual
            best_individual = population.best_individual
            
            # Compile statistics
            runtime = time.time() - start_time
            statistics = {
                "generations": population.generation,
                "runtime": runtime,
                "initial_best_fitness": generations_history[0]["best_fitness"],
                "final_best_fitness": generations_history[-1]["best_fitness"],
                "fitness_improvement": (generations_history[-1]["best_fitness"] - generations_history[0]["best_fitness"]) if generations_history[0]["best_fitness"] is not None and generations_history[-1]["best_fitness"] is not None else None,
                "average_generation_time": runtime / population.generation if population.generation > 0 else 0.0,
                "population_size": population.size
            }
            
            # Update algorithm statistics
            self._stats["successful_executions"] += 1
            self._stats["total_runtime"] += runtime
            self._stats["average_runtime"] = self._stats["total_runtime"] / self._stats["successful_executions"]
            self._stats["last_runtime"] = runtime
            self._stats["last_executed_at"] = time.time()
            
            if best_individual and best_individual.fitness is not None:
                if (self._stats["best_fitness_achieved"] is None or 
                    best_individual.fitness > self._stats["best_fitness_achieved"]):
                    self._stats["best_fitness_achieved"] = best_individual.fitness
            
            # Create and return result
            return EvolutionResult(
                best_individual=best_individual,
                final_population=population,
                generations_history=generations_history,
                statistics=statistics,
                config=self.config,
                runtime=runtime
            )
            
        except Exception as e:
            # Update failure statistics
            self._stats["failed_executions"] += 1
            self._stats["last_executed_at"] = time.time()
            
            # Re-raise as EvolutionError if it's not already one
            if not isinstance(e, EvolutionError):
                raise EvolutionError(
                    message=str(e),
                    algorithm=self.name,
                    details={"original_error": str(e), "error_type": type(e).__name__}
                ) from e
            raise
    
    async def _evaluate_population(self, population: Population[G, F]) -> None:
        """
        Evaluate the fitness of all individuals in a population.
        
        Args:
            population: Population to evaluate
        """
        # Get parallelism level based on configuration
        parallelism = self.config.parallelism
        
        # If parallelism is disabled or only one individual, evaluate sequentially
        if parallelism <= 1 or len(population.individuals) <= 1:
            for individual in population.individuals:
                if individual.fitness is None:
                    individual.fitness = await self.evaluate_fitness(individual)
        else:
            # Otherwise, evaluate in parallel using asyncio.gather
            tasks = []
            for individual in population.individuals:
                if individual.fitness is None:
                    tasks.append(self._evaluate_individual(individual))
            
            # Use semaphore to limit concurrency
            semaphore = asyncio.Semaphore(parallelism)
            
            async def _evaluate_with_semaphore(individual):
                async with semaphore:
                    await self._evaluate_individual(individual)
            
            # Start all tasks and wait for them to complete
            await asyncio.gather(*[_evaluate_with_semaphore(individual) for individual in population.individuals if individual.fitness is None])
    
    async def _evaluate_individual(self, individual: Individual[G, F]) -> None:
        """
        Evaluate the fitness of an individual and update its fitness.
        
        Args:
            individual: Individual to evaluate
        """
        individual.fitness = await self.evaluate_fitness(individual)
    
    async def _select_parents(self, population: Population[G, F], count: int = 2) -> List[Individual[G, F]]:
        """
        Select multiple parent individuals for reproduction.
        
        Args:
            population: Population to select from
            count: Number of parents to select
            
        Returns:
            List of selected parent individuals
        """
        return [await self.select_parent(population) for _ in range(count)]
    
    async def _create_next_generation(self, population: Population[G, F]) -> Population[G, F]:
        """
        Create the next generation of individuals.
        
        Args:
            population: Current population
            
        Returns:
            Next generation population
        """
        # Sort current population by fitness
        population.sort_by_fitness()
        
        # Create new population
        new_population = Population[G, F](generation=population.generation + 1)
        
        # Add elite individuals to new population
        elite_count = min(self.config.elitism, len(population.individuals))
        for i in range(elite_count):
            elite = population.individuals[i]
            new_population.add_individual(elite)
        
        # Fill the rest of the population with offspring
        while len(new_population.individuals) < self.config.population_size:
            # Determine if we'll perform crossover based on crossover rate
            if random.random() < self.config.crossover_rate and len(population.individuals) >= 2:
                # Select parents
                parents = await self._select_parents(population, 2)
                
                # Perform crossover
                child = await self.crossover(parents[0], parents[1])
                
                # Set child's generation and parent IDs
                child.generation = new_population.generation
                child.parent_ids = [parents[0].id, parents[1].id]
            else:
                # No crossover, just select an individual to copy
                parent = await self.select_parent(population)
                
                # Create a copy of the parent
                child = Individual[G, F](
                    genome=parent.genome,
                    generation=new_population.generation,
                    parent_ids=[parent.id]
                )
            
            # Determine if we'll perform mutation based on mutation rate
            if random.random() < self.config.mutation_rate:
                child = await self.mutate(child)
            
            # Add child to new population
            new_population.add_individual(child)
        
        return new_population
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the algorithm's execution.
        
        Returns:
            Dictionary of statistics
        """
        return self._stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset all execution statistics."""
        self._stats = {
            "executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_runtime": 0.0,
            "average_runtime": 0.0,
            "last_runtime": None,
            "last_executed_at": None,
            "best_fitness_achieved": None
        }


class EvolutionRegistry:
    """
    Registry for evolution algorithms.
    
    Maintains a global registry of all available evolution algorithms.
    """
    _algorithms: Dict[str, Type[EvolutionAlgorithm]] = {}
    _instances: Dict[str, EvolutionAlgorithm] = {}
    
    @classmethod
    def register(cls, name: str, algorithm_class: Type[EvolutionAlgorithm]) -> None:
        """
        Register an evolution algorithm class.
        
        Args:
            name: Unique identifier for the algorithm
            algorithm_class: Class implementing the algorithm
        """
        cls._algorithms[name] = algorithm_class
        logger.debug(f"Registered evolution algorithm: {name}")
    
    @classmethod
    def get_algorithm_class(cls, name: str) -> Optional[Type[EvolutionAlgorithm]]:
        """
        Get an evolution algorithm class by name.
        
        Args:
            name: Unique identifier for the algorithm
            
        Returns:
            The algorithm class, or None if not found
        """
        return cls._algorithms.get(name)
    
    @classmethod
    def get_algorithm(cls, name: str, config: Optional[EvolutionConfig] = None) -> Optional[EvolutionAlgorithm]:
        """
        Get or create an instance of an evolution algorithm.
        
        Args:
            name: Unique identifier for the algorithm
            config: Configuration for the algorithm
            
        Returns:
            An instance of the algorithm, or None if not found
        """
        # Create a new instance (ignore existing instances)
        algorithm_class = cls.get_algorithm_class(name)
        if algorithm_class:
            instance = algorithm_class(name, config)
            cls._instances[name] = instance
            return instance
        
        return None
    
    @classmethod
    def list_algorithms(cls) -> List[str]:
        """
        List all registered algorithms.
        
        Returns:
            List of algorithm names
        """
        return list(cls._algorithms.keys())


def register_algorithm(name: str) -> Callable[[Type[EvolutionAlgorithm]], Type[EvolutionAlgorithm]]:
    """
    Decorator to register an evolution algorithm class.
    
    Args:
        name: Unique identifier for the algorithm
        
    Returns:
        Decorator function
    """
    def decorator(algorithm_class: Type[EvolutionAlgorithm]) -> Type[EvolutionAlgorithm]:
        EvolutionRegistry.register(name, algorithm_class)
        return algorithm_class
    return decorator


def get_algorithm(name: str, config: Optional[EvolutionConfig] = None) -> Optional[EvolutionAlgorithm]:
    """
    Get an evolution algorithm instance by name.
    
    Args:
        name: Unique identifier for the algorithm
        config: Configuration for the algorithm
        
    Returns:
        An instance of the algorithm, or None if not found
    """
    return EvolutionRegistry.get_algorithm(name, config)


def list_algorithms() -> List[str]:
    """
    List all registered evolution algorithms.
    
    Returns:
        List of algorithm names
    """
    return EvolutionRegistry.list_algorithms()