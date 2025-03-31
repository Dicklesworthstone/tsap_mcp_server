"""
Genetic algorithm implementation for the TSAP MCP Server.

This module implements genetic algorithms specifically for evolving
search patterns or strategies. It provides a flexible framework for
defining fitness functions, mutation/crossover operations, and
selection strategies.
"""

import random
import re
import string
import time
import uuid
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from tsap.utils.logging import logger


class SelectionMethod(str, Enum):
    """Selection methods for genetic algorithms."""
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"
    ELITE = "elite"


class CrossoverMethod(str, Enum):
    """Crossover methods for genetic algorithms."""
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"


class MutationMethod(str, Enum):
    """Mutation methods for genetic algorithms."""
    POINT = "point"
    SWAP = "swap"
    SCRAMBLE = "scramble"
    INVERSION = "inversion"


@dataclass
class Individual:
    """
    Represents an individual in a genetic algorithm population.
    
    An individual can represent a search pattern, a sequence of operations,
    or any other evolvable entity.
    """
    genome: Any
    fitness: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize after creation."""
        # Ensure parents is a list
        if not isinstance(self.parent_ids, list):
            self.parent_ids = [self.parent_ids]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "genome": self.genome,
            "fitness": self.fitness,
            "metadata": self.metadata,
            "generation": self.generation,
            "parent_ids": self.parent_ids
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Individual':
        """
        Create from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Individual instance
        """
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            genome=data["genome"],
            fitness=data.get("fitness", 0.0),
            metadata=data.get("metadata", {}),
            generation=data.get("generation", 0),
            parent_ids=data.get("parent_ids", [])
        )
    
    def __eq__(self, other):
        """Compare equality by genome."""
        if not isinstance(other, Individual):
            return False
        return self.genome == other.genome
    
    def __hash__(self):
        """Hash by genome."""
        return hash(str(self.genome))


@dataclass
class Population:
    """
    Represents a population of individuals in a genetic algorithm.
    """
    individuals: List[Individual] = field(default_factory=list)
    generation: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def size(self) -> int:
        """Get population size."""
        return len(self.individuals)
    
    @property
    def best_individual(self) -> Optional[Individual]:
        """Get the best individual in the population."""
        if not self.individuals:
            return None
        return max(self.individuals, key=lambda ind: ind.fitness)
    
    @property
    def average_fitness(self) -> float:
        """Get the average fitness of the population."""
        if not self.individuals:
            return 0.0
        return sum(ind.fitness for ind in self.individuals) / len(self.individuals)
    
    @property
    def diversity(self) -> float:
        """
        Calculate population diversity.
        
        Higher values indicate more diverse populations.
        Scale 0.0 to 1.0.
        """
        if not self.individuals or len(self.individuals) <= 1:
            return 0.0
        
        # Use unique genomes as a simple measure
        unique_genomes = len(set(str(ind.genome) for ind in self.individuals))
        return unique_genomes / len(self.individuals)
    
    def add_individual(self, individual: Individual) -> None:
        """
        Add an individual to the population.
        
        Args:
            individual: Individual to add
        """
        self.individuals.append(individual)
    
    def remove_individual(self, individual: Individual) -> None:
        """
        Remove an individual from the population.
        
        Args:
            individual: Individual to remove
        """
        self.individuals.remove(individual)
    
    def sort_by_fitness(self, reverse: bool = True) -> None:
        """
        Sort the population by fitness.
        
        Args:
            reverse: Whether to sort in descending order
        """
        self.individuals.sort(key=lambda ind: ind.fitness, reverse=reverse)
    
    def get_individual_by_id(self, individual_id: str) -> Optional[Individual]:
        """
        Get an individual by ID.
        
        Args:
            individual_id: Individual ID
            
        Returns:
            Individual or None if not found
        """
        for ind in self.individuals:
            if ind.id == individual_id:
                return ind
        return None
    
    def get_fittest(self, count: int = 1) -> List[Individual]:
        """
        Get the fittest individuals.
        
        Args:
            count: Number of individuals to return
            
        Returns:
            List of fittest individuals
        """
        sorted_inds = sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)
        return sorted_inds[:count]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "individuals": [ind.to_dict() for ind in self.individuals],
            "generation": self.generation,
            "metadata": self.metadata,
            "stats": {
                "size": self.size,
                "average_fitness": self.average_fitness,
                "best_fitness": self.best_individual.fitness if self.best_individual else 0.0,
                "diversity": self.diversity
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Population':
        """
        Create from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Population instance
        """
        individuals = [Individual.from_dict(ind) for ind in data.get("individuals", [])]
        
        return cls(
            individuals=individuals,
            generation=data.get("generation", 0),
            metadata=data.get("metadata", {})
        )


@dataclass
class EvolutionConfig:
    """
    Configuration for a genetic algorithm.
    """
    population_size: int = 50
    generations: int = 20
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    elitism_count: int = 2
    tournament_size: int = 3
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    crossover_method: CrossoverMethod = CrossoverMethod.SINGLE_POINT
    mutation_method: MutationMethod = MutationMethod.POINT
    fitness_target: Optional[float] = None  # Target fitness to stop early
    max_stagnant_generations: int = 5  # Stop if no improvement for this many generations
    
    def __post_init__(self):
        """Initialize after creation."""
        # Ensure enums are properly typed
        if isinstance(self.selection_method, str):
            self.selection_method = SelectionMethod(self.selection_method)
        
        if isinstance(self.crossover_method, str):
            self.crossover_method = CrossoverMethod(self.crossover_method)
        
        if isinstance(self.mutation_method, str):
            self.mutation_method = MutationMethod(self.mutation_method)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "population_size": self.population_size,
            "generations": self.generations,
            "crossover_rate": self.crossover_rate,
            "mutation_rate": self.mutation_rate,
            "elitism_count": self.elitism_count,
            "tournament_size": self.tournament_size,
            "selection_method": self.selection_method.value,
            "crossover_method": self.crossover_method.value,
            "mutation_method": self.mutation_method.value,
            "fitness_target": self.fitness_target,
            "max_stagnant_generations": self.max_stagnant_generations
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvolutionConfig':
        """
        Create from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            EvolutionConfig instance
        """
        return cls(
            population_size=data.get("population_size", 50),
            generations=data.get("generations", 20),
            crossover_rate=data.get("crossover_rate", 0.8),
            mutation_rate=data.get("mutation_rate", 0.2),
            elitism_count=data.get("elitism_count", 2),
            tournament_size=data.get("tournament_size", 3),
            selection_method=data.get("selection_method", SelectionMethod.TOURNAMENT),
            crossover_method=data.get("crossover_method", CrossoverMethod.SINGLE_POINT),
            mutation_method=data.get("mutation_method", MutationMethod.POINT),
            fitness_target=data.get("fitness_target"),
            max_stagnant_generations=data.get("max_stagnant_generations", 5)
        )


@dataclass
class EvolutionResult:
    """
    Results from running a genetic algorithm.
    """
    best_individual: Individual
    final_population: Population
    all_populations: List[Population]
    config: EvolutionConfig
    execution_time: float
    early_stopped: bool = False
    generations_executed: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "best_individual": self.best_individual.to_dict(),
            "final_population": self.final_population.to_dict(),
            "config": self.config.to_dict(),
            "execution_time": self.execution_time,
            "early_stopped": self.early_stopped,
            "generations_executed": self.generations_executed,
            "fitness_history": [
                pop.best_individual.fitness if pop.best_individual else 0.0
                for pop in self.all_populations
            ],
            "diversity_history": [pop.diversity for pop in self.all_populations]
        }


class GeneticAlgorithm:
    """
    Implements a genetic algorithm for evolving search patterns or strategies.
    """
    
    def __init__(
        self,
        config: EvolutionConfig,
        genome_generator: Callable[[], Any],
        fitness_function: Callable[[Any], float],
        crossover_function: Optional[Callable[[Any, Any], Any]] = None,
        mutation_function: Optional[Callable[[Any], Any]] = None
    ):
        """
        Initialize the genetic algorithm.
        
        Args:
            config: Evolution configuration
            genome_generator: Function to generate a random genome
            fitness_function: Function to evaluate genome fitness
            crossover_function: Function to perform crossover (optional)
            mutation_function: Function to perform mutation (optional)
        """
        self.config = config
        self.genome_generator = genome_generator
        self.fitness_function = fitness_function
        self.crossover_function = crossover_function
        self.mutation_function = mutation_function
        
        # Tracking
        self.current_generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.diversity_history = []
        self.populations = []
    
    def _initialize_population(self) -> Population:
        """
        Initialize a random population.
        
        Returns:
            Initial population
        """
        individuals = []
        
        for _ in range(self.config.population_size):
            genome = self.genome_generator()
            fitness = self.fitness_function(genome)
            
            individuals.append(Individual(
                genome=genome,
                fitness=fitness,
                generation=0
            ))
        
        return Population(individuals=individuals, generation=0)
    
    def _evaluate_population(self, population: Population) -> None:
        """
        Evaluate fitness for all individuals in the population.
        
        Args:
            population: Population to evaluate
        """
        for individual in population.individuals:
            individual.fitness = self.fitness_function(individual.genome)
    
    def _select_individual(self, population: Population) -> Individual:
        """
        Select an individual using the configured selection method.
        
        Args:
            population: Population to select from
            
        Returns:
            Selected individual
        """
        method = self.config.selection_method
        
        if method == SelectionMethod.TOURNAMENT:
            # Tournament selection
            tournament = random.sample(population.individuals, min(self.config.tournament_size, len(population.individuals)))
            return max(tournament, key=lambda ind: ind.fitness)
        
        elif method == SelectionMethod.ROULETTE:
            # Roulette wheel selection
            total_fitness = sum(ind.fitness for ind in population.individuals)
            if total_fitness <= 0:
                return random.choice(population.individuals)
            
            pick = random.uniform(0, total_fitness)
            current = 0
            for individual in population.individuals:
                current += individual.fitness
                if current >= pick:
                    return individual
            
            return population.individuals[-1]
        
        elif method == SelectionMethod.RANK:
            # Rank selection
            sorted_individuals = sorted(population.individuals, key=lambda ind: ind.fitness, reverse=True)
            ranks = list(range(1, len(sorted_individuals) + 1))
            total_rank = sum(ranks)
            
            pick = random.uniform(0, total_rank)
            current = 0
            for i, individual in enumerate(sorted_individuals):
                current += ranks[i]
                if current >= pick:
                    return individual
            
            return sorted_individuals[-1]
        
        elif method == SelectionMethod.ELITE:
            # Simply pick one of the top individuals
            sorted_individuals = sorted(population.individuals, key=lambda ind: ind.fitness, reverse=True)
            elite_size = max(1, int(len(sorted_individuals) * 0.1))  # Top 10%
            return random.choice(sorted_individuals[:elite_size])
        
        # Default to random selection
        return random.choice(population.individuals)
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """
        Perform crossover between two parents.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Offspring individual
        """
        # Use custom crossover function if provided
        if self.crossover_function:
            child_genome = self.crossover_function(parent1.genome, parent2.genome)
        else:
            # Default crossover implementation based on genome type
            genome_type = type(parent1.genome)  # noqa: F841
            
            if isinstance(parent1.genome, str):
                child_genome = self._crossover_string(parent1.genome, parent2.genome)
            elif isinstance(parent1.genome, list):
                child_genome = self._crossover_list(parent1.genome, parent2.genome)
            elif isinstance(parent1.genome, dict):
                child_genome = self._crossover_dict(parent1.genome, parent2.genome)
            else:
                # For unsupported types, just pick one parent's genome
                child_genome = random.choice([parent1.genome, parent2.genome])
        
        # Create child individual
        child = Individual(
            genome=child_genome,
            generation=self.current_generation,
            parent_ids=[parent1.id, parent2.id]
        )
        
        return child
    
    def _crossover_string(self, genome1: str, genome2: str) -> str:
        """
        Perform crossover on string genomes.
        
        Args:
            genome1: First parent genome
            genome2: Second parent genome
            
        Returns:
            Child genome
        """
        method = self.config.crossover_method
        
        if method == CrossoverMethod.SINGLE_POINT:
            # Single point crossover
            point = random.randint(0, min(len(genome1), len(genome2)))
            return genome1[:point] + genome2[point:]
        
        elif method == CrossoverMethod.TWO_POINT:
            # Two point crossover
            length = min(len(genome1), len(genome2))
            point1 = random.randint(0, length)
            point2 = random.randint(point1, length)
            return genome1[:point1] + genome2[point1:point2] + genome1[point2:]
        
        elif method == CrossoverMethod.UNIFORM:
            # Uniform crossover
            length = min(len(genome1), len(genome2))
            result = ""
            for i in range(length):
                result += genome1[i] if random.random() < 0.5 else genome2[i]
            return result
        
        # Default to single point
        point = random.randint(0, min(len(genome1), len(genome2)))
        return genome1[:point] + genome2[point:]
    
    def _crossover_list(self, genome1: List[Any], genome2: List[Any]) -> List[Any]:
        """
        Perform crossover on list genomes.
        
        Args:
            genome1: First parent genome
            genome2: Second parent genome
            
        Returns:
            Child genome
        """
        method = self.config.crossover_method
        
        if method == CrossoverMethod.SINGLE_POINT:
            # Single point crossover
            point = random.randint(0, min(len(genome1), len(genome2)))
            return genome1[:point] + genome2[point:]
        
        elif method == CrossoverMethod.TWO_POINT:
            # Two point crossover
            length = min(len(genome1), len(genome2))
            point1 = random.randint(0, length)
            point2 = random.randint(point1, length)
            return genome1[:point1] + genome2[point1:point2] + genome1[point2:]
        
        elif method == CrossoverMethod.UNIFORM:
            # Uniform crossover
            length = min(len(genome1), len(genome2))
            result = []
            for i in range(length):
                result.append(genome1[i] if random.random() < 0.5 else genome2[i])
            return result
        
        # Default to single point
        point = random.randint(0, min(len(genome1), len(genome2)))
        return genome1[:point] + genome2[point:]
    
    def _crossover_dict(self, genome1: Dict[str, Any], genome2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform crossover on dict genomes.
        
        Args:
            genome1: First parent genome
            genome2: Second parent genome
            
        Returns:
            Child genome
        """
        result = {}
        all_keys = set(genome1.keys()) | set(genome2.keys())
        
        for key in all_keys:
            if key in genome1 and key in genome2:
                # Both parents have this key
                if random.random() < 0.5:
                    result[key] = genome1[key]
                else:
                    result[key] = genome2[key]
            elif key in genome1:
                # Only parent1 has this key
                if random.random() < 0.5:
                    result[key] = genome1[key]
            else:
                # Only parent2 has this key
                if random.random() < 0.5:
                    result[key] = genome2[key]
        
        return result
    
    def _mutate(self, individual: Individual) -> Individual:
        """
        Mutate an individual.
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Mutated individual (or the same if no mutation occurred)
        """
        # Only mutate with probability mutation_rate
        if random.random() > self.config.mutation_rate:
            return individual
        
        # Use custom mutation function if provided
        if self.mutation_function:
            mutated_genome = self.mutation_function(individual.genome)
            
            # Create mutated individual
            return Individual(
                genome=mutated_genome,
                generation=self.current_generation,
                parent_ids=[individual.id]
            )
        
        # Default mutation implementation based on genome type
        mutated_genome = None
        
        if isinstance(individual.genome, str):
            mutated_genome = self._mutate_string(individual.genome)
        elif isinstance(individual.genome, list):
            mutated_genome = self._mutate_list(individual.genome)
        elif isinstance(individual.genome, dict):
            mutated_genome = self._mutate_dict(individual.genome)
        else:
            # For unsupported types, don't mutate
            return individual
        
        # Create mutated individual
        mutated = Individual(
            genome=mutated_genome,
            generation=self.current_generation,
            parent_ids=[individual.id]
        )
        
        return mutated
    
    def _mutate_string(self, genome: str) -> str:
        """
        Perform mutation on a string genome.
        
        Args:
            genome: Genome to mutate
            
        Returns:
            Mutated genome
        """
        method = self.config.mutation_method
        
        if method == MutationMethod.POINT:
            # Point mutation - change, insert, or delete a character
            if not genome:
                return genome
            
            chars = list(genome)
            mutation_type = random.choice(["change", "insert", "delete"])
            
            if mutation_type == "change" and chars:
                pos = random.randint(0, len(chars) - 1)
                chars[pos] = random.choice(string.ascii_letters + string.digits + ".*+?[](){}|^$")
            elif mutation_type == "insert":
                pos = random.randint(0, len(chars))
                chars.insert(pos, random.choice(string.ascii_letters + string.digits + ".*+?[](){}|^$"))
            elif mutation_type == "delete" and chars:
                pos = random.randint(0, len(chars) - 1)
                chars.pop(pos)
            
            return "".join(chars)
        
        elif method == MutationMethod.SWAP:
            # Swap mutation - swap two characters
            if len(genome) < 2:
                return genome
            
            chars = list(genome)
            pos1, pos2 = random.sample(range(len(chars)), 2)
            chars[pos1], chars[pos2] = chars[pos2], chars[pos1]
            
            return "".join(chars)
        
        elif method == MutationMethod.SCRAMBLE:
            # Scramble mutation - scramble a substring
            if len(genome) < 3:
                return genome
            
            chars = list(genome)
            length = random.randint(2, min(5, len(chars) - 1))
            start = random.randint(0, len(chars) - length)
            
            substring = chars[start:start+length]
            random.shuffle(substring)
            
            for i in range(length):
                chars[start + i] = substring[i]
            
            return "".join(chars)
        
        elif method == MutationMethod.INVERSION:
            # Inversion mutation - reverse a substring
            if len(genome) < 3:
                return genome
            
            chars = list(genome)
            length = random.randint(2, min(5, len(chars) - 1))
            start = random.randint(0, len(chars) - length)
            
            substring = chars[start:start+length]
            substring.reverse()
            
            for i in range(length):
                chars[start + i] = substring[i]
            
            return "".join(chars)
        
        # Default to point mutation
        return self._mutate_string(genome)
    
    def _mutate_list(self, genome: List[Any]) -> List[Any]:
        """
        Perform mutation on a list genome.
        
        Args:
            genome: Genome to mutate
            
        Returns:
            Mutated genome
        """
        method = self.config.mutation_method
        
        # Create a copy of the genome
        result = genome.copy()
        
        if not result:
            return result
        
        if method == MutationMethod.POINT:
            # Point mutation - change, insert, or delete an element
            mutation_type = random.choice(["change", "insert", "delete"])
            
            if mutation_type == "change" and result:
                pos = random.randint(0, len(result) - 1)
                # For simplicity, we'll just use the genome generator to create a new element
                if self.genome_generator:
                    new_element = self.genome_generator()
                    # If genome_generator creates a full genome, just take the first element
                    if isinstance(new_element, list) and new_element:
                        result[pos] = new_element[0]
                    else:
                        result[pos] = new_element
            elif mutation_type == "insert":
                pos = random.randint(0, len(result))
                if self.genome_generator:
                    new_element = self.genome_generator()
                    # If genome_generator creates a full genome, just take the first element
                    if isinstance(new_element, list) and new_element:
                        result.insert(pos, new_element[0])
                    else:
                        result.insert(pos, new_element)
            elif mutation_type == "delete" and result:
                pos = random.randint(0, len(result) - 1)
                result.pop(pos)
        
        elif method == MutationMethod.SWAP:
            # Swap mutation - swap two elements
            if len(result) < 2:
                return result
            
            pos1, pos2 = random.sample(range(len(result)), 2)
            result[pos1], result[pos2] = result[pos2], result[pos1]
        
        elif method == MutationMethod.SCRAMBLE:
            # Scramble mutation - scramble a sublist
            if len(result) < 3:
                return result
            
            length = random.randint(2, min(5, len(result) - 1))
            start = random.randint(0, len(result) - length)
            
            sublist = result[start:start+length]
            random.shuffle(sublist)
            
            for i in range(length):
                result[start + i] = sublist[i]
        
        elif method == MutationMethod.INVERSION:
            # Inversion mutation - reverse a sublist
            if len(result) < 3:
                return result
            
            length = random.randint(2, min(5, len(result) - 1))
            start = random.randint(0, len(result) - length)
            
            result[start:start+length] = reversed(result[start:start+length])
        
        return result
    
    def _mutate_dict(self, genome: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform mutation on a dict genome.
        
        Args:
            genome: Genome to mutate
            
        Returns:
            Mutated genome
        """
        # Create a copy of the genome
        result = genome.copy()
        
        if not result:
            return result
        
        # Choose a mutation type
        mutation_type = random.choice(["change", "add", "remove"])
        
        if mutation_type == "change" and result:
            # Change a value
            key = random.choice(list(result.keys()))
            # For simplicity, generate a random value of the same type
            value_type = type(result[key])
            if value_type is int:
                result[key] = random.randint(0, 100)
            elif value_type is float:
                result[key] = random.random() * 100
            elif value_type is bool:
                result[key] = random.choice([True, False])
            elif value_type is str:
                result[key] = "".join(random.choice(string.ascii_letters) for _ in range(5))
            elif value_type is list:
                result[key] = [random.randint(0, 100) for _ in range(3)]
        
        elif mutation_type == "add":
            # Add a new key-value pair
            key = "".join(random.choice(string.ascii_lowercase) for _ in range(5))
            while key in result:
                key = "".join(random.choice(string.ascii_lowercase) for _ in range(5))
            
            # Generate a random value
            value_type = random.choice([int, float, bool, str])
            if value_type is int:
                result[key] = random.randint(0, 100)
            elif value_type is float:
                result[key] = random.random() * 100
            elif value_type is bool:
                result[key] = random.choice([True, False])
            elif value_type is str:
                result[key] = "".join(random.choice(string.ascii_letters) for _ in range(5))
        
        elif mutation_type == "remove" and result:
            # Remove a key
            key = random.choice(list(result.keys()))
            del result[key]
        
        return result
    
    def _create_next_generation(self, population: Population) -> Population:
        """
        Create the next generation from the current population.
        
        Args:
            population: Current population
            
        Returns:
            Next generation population
        """
        new_individuals = []
        
        # Sort by fitness for elitism
        population.sort_by_fitness()
        
        # Keep the best individuals (elitism)
        for i in range(min(self.config.elitism_count, len(population.individuals))):
            new_individuals.append(population.individuals[i])
        
        # Fill the rest with offspring
        while len(new_individuals) < self.config.population_size:
            # Select parents
            parent1 = self._select_individual(population)
            parent2 = self._select_individual(population)
            
            # Crossover with probability crossover_rate
            if random.random() < self.config.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                # No crossover, just copy parent1
                child = Individual(
                    genome=parent1.genome,
                    generation=self.current_generation,
                    parent_ids=[parent1.id]
                )
            
            # Mutate
            child = self._mutate(child)
            
            # Evaluate fitness
            child.fitness = self.fitness_function(child.genome)
            
            # Add to new population
            new_individuals.append(child)
        
        # Create next generation population
        next_population = Population(
            individuals=new_individuals,
            generation=self.current_generation
        )
        
        return next_population
    
    def evolve(self) -> EvolutionResult:
        """
        Run the genetic algorithm.
        
        Returns:
            Evolution result
        """
        start_time = time.time()
        
        # Initialize population
        population = self._initialize_population()
        self.populations = [population]
        
        # Evaluate initial population
        self._evaluate_population(population)
        
        # Store initial stats
        best_individual = population.best_individual
        self.best_fitness_history.append(best_individual.fitness if best_individual else 0.0)
        self.avg_fitness_history.append(population.average_fitness)
        self.diversity_history.append(population.diversity)
        
        # Track generations without improvement
        stagnant_generations = 0
        
        # Main evolution loop
        for generation in range(1, self.config.generations + 1):
            self.current_generation = generation
            
            # Create next generation
            population = self._create_next_generation(population)
            self.populations.append(population)
            
            # Update stats
            best_individual = population.best_individual
            self.best_fitness_history.append(best_individual.fitness if best_individual else 0.0)
            self.avg_fitness_history.append(population.average_fitness)
            self.diversity_history.append(population.diversity)
            
            # Log progress
            logger.info(
                f"Generation {generation}/{self.config.generations}: Best fitness = {best_individual.fitness if best_individual else 0.0:.6f}, "
                f"Avg fitness = {population.average_fitness:.6f}, Diversity = {population.diversity:.2f}",
                component="evolution",
                operation="genetic_algorithm"
            )
            
            # Check early stopping conditions
            if self.config.fitness_target is not None and best_individual.fitness >= self.config.fitness_target:
                logger.info(
                    f"Target fitness reached: {best_individual.fitness:.6f} >= {self.config.fitness_target:.6f}",
                    component="evolution",
                    operation="genetic_algorithm"
                )
                break
            
            # Check for improvement
            if generation > 1:
                if self.best_fitness_history[-1] <= self.best_fitness_history[-2]:
                    stagnant_generations += 1
                else:
                    stagnant_generations = 0
            
            if stagnant_generations >= self.config.max_stagnant_generations:
                logger.info(
                    f"No improvement for {stagnant_generations} generations, stopping early",
                    component="evolution",
                    operation="genetic_algorithm"
                )
                break
        
        # Get final results
        final_population = self.populations[-1]
        final_population.sort_by_fitness()
        best_individual = final_population.best_individual
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Create result object
        result = EvolutionResult(
            best_individual=best_individual,
            final_population=final_population,
            all_populations=self.populations,
            config=self.config,
            execution_time=execution_time,
            early_stopped=self.current_generation < self.config.generations,
            generations_executed=self.current_generation
        )
        
        return result


# Convenience function for string pattern evolution
async def evolve_regex_pattern(
    positive_examples: List[str],
    negative_examples: Optional[List[str]] = None,
    config: Optional[EvolutionConfig] = None,
    initial_patterns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Evolve a regex pattern using a genetic algorithm.
    
    Args:
        positive_examples: Strings that should match the pattern
        negative_examples: Strings that should not match the pattern
        config: Evolution configuration
        initial_patterns: Initial patterns to include in the population
        
    Returns:
        Dictionary with the best regex pattern and evolution statistics
    """
    # Use default config if not provided
    if config is None:
        config = EvolutionConfig(
            population_size=30,
            generations=15,
            crossover_rate=0.8,
            mutation_rate=0.3,
            elitism_count=2,
            tournament_size=3
        )
    
    # Define genome generator
    def generate_regex() -> str:
        """Generate a random regex pattern."""
        if initial_patterns and random.random() < 0.3:
            # Use one of the initial patterns
            return random.choice(initial_patterns)
        
        # Generate a pattern based on examples
        patterns = [
            r"\w+",
            r"\d+",
            r"[a-z]+",
            r"[A-Z]+",
            r".*",
            r"\S+",
            r"[^\s]+",
            r"^[a-zA-Z]\w*$"
        ]
        
        if positive_examples:
            # Extract characters that appear in examples
            chars = set()
            for ex in positive_examples:
                chars.update(ex)
            
            # Create character classes
            if chars:
                char_class = "[" + "".join(sorted(chars)) + "]"
                patterns.append(f"{char_class}+")
                patterns.append(f"^{char_class}+$")
        
        return random.choice(patterns)
    
    # Define fitness function
    def evaluate_fitness(regex: str) -> float:
        """Evaluate the fitness of a regex pattern."""
        try:
            pattern = re.compile(regex)
        except re.error:
            return 0.0
        
        # Check positive examples
        matches = sum(1 for ex in positive_examples if pattern.search(ex))
        positive_score = matches / len(positive_examples) if positive_examples else 0
        
        # Check negative examples
        negative_score = 1.0
        if negative_examples:
            false_positives = sum(1 for ex in negative_examples if pattern.search(ex))
            negative_score = 1.0 - (false_positives / len(negative_examples))
        
        # Combine scores (weight positive matches more)
        score = (0.7 * positive_score) + (0.3 * negative_score)
        
        # Penalize long patterns
        length_penalty = min(1.0, max(0.0, 1.0 - (len(regex) / 100)))
        score *= (0.9 + (0.1 * length_penalty))
        
        return score
    
    # Define crossover function
    def crossover(regex1: str, regex2: str) -> str:
        """Crossover two regex patterns."""
        # Simple implementation using random crossover point
        if not regex1 or not regex2:
            return regex1 or regex2
        
        point = random.randint(0, min(len(regex1), len(regex2)))
        return regex1[:point] + regex2[point:]
    
    # Define mutation function
    def mutate(regex: str) -> str:
        """Mutate a regex pattern."""
        # Possible mutations
        mutations = [
            # Add a quantifier
            lambda r: r + random.choice(["*", "+", "?"]),
            # Add a character class
            lambda r: r + random.choice([r"\d", r"\w", r"."]),
            # Add a boundary
            lambda r: random.choice(["^", ""]) + r + random.choice(["$", ""]),
            # Replace a character with a character class
            lambda r: r.replace(random.choice(r) if r else "", random.choice([r"\d", r"\w", r"."]))
        ]
        
        if not regex:
            return ".*"
        
        # Apply a random mutation
        mutation = random.choice(mutations)
        mutated = mutation(regex)
        
        # Ensure the result is a valid regex
        try:
            re.compile(mutated)
            return mutated
        except re.error:
            return regex
    
    # Create genetic algorithm
    ga = GeneticAlgorithm(
        config=config,
        genome_generator=generate_regex,
        fitness_function=evaluate_fitness,
        crossover_function=crossover,
        mutation_function=mutate
    )
    
    # Run evolution
    result = ga.evolve()
    
    # Build result dictionary
    return {
        "best_pattern": result.best_individual.genome,
        "fitness": result.best_individual.fitness,
        "generations": result.generations_executed,
        "execution_time": result.execution_time,
        "population_size": config.population_size,
        "early_stopped": result.early_stopped,
        "fitness_history": [pop.best_individual.fitness if pop.best_individual else 0.0 for pop in result.all_populations],
        "diversity_history": [pop.diversity for pop in result.all_populations]
    }


# Convenience function for evolving strategies
async def evolve_search_strategy(
    target_documents: List[str],
    training_queries: List[Dict[str, Any]],
    config: Optional[EvolutionConfig] = None
) -> Dict[str, Any]:
    """
    Evolve a search strategy using a genetic algorithm.
    
    Args:
        target_documents: Documents to search in
        training_queries: Training queries with expected results
        config: Evolution configuration
        
    Returns:
        Dictionary with the best strategy and evolution statistics
    """
    # Placeholder implementation - would need to be expanded
    # with actual strategy definition and evaluation logic
    
    return {
        "message": "Strategy evolution not fully implemented",
        "status": "placeholder",
        "target_documents": len(target_documents),
        "training_queries": len(training_queries)
    }