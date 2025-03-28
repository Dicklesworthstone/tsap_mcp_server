"""
Evolution package marker for TSAP MCP Server.

This package provides functionality for evolutionary algorithms and learning systems
that enable search patterns, extraction strategies, and analysis approaches to
improve over time based on results and feedback.
"""

from tsap.evolution.genetic import (
    GeneticAlgorithm, Individual, Population, 
    EvolutionConfig, EvolutionResult,
    evolve_regex_pattern
)

from tsap.evolution.metrics import (
    SearchMetrics, ExtractionMetrics, StrategyMetrics,
    MetricsCalculator, evaluate_search_results,
    evaluate_extraction_results, evaluate_strategy
)

from tsap.evolution.pattern_analyzer import (
    PatternAnalyzer, PatternStats, PatternVariant,
    get_pattern_analyzer, analyze_pattern, evolve_pattern
)

from tsap.evolution.pattern_library import (
    PatternLibrary, get_pattern_library, add_pattern_to_library,
    search_patterns, get_popular_patterns
)

# These modules would be imported once implemented
# from tsap.evolution.strategy_evolution import (
#     StrategyEvolution, get_strategy_evolution,
#     evolve_strategy, evaluate_strategy_fitness
# )

# from tsap.evolution.strategy_journal import (
#     StrategyJournal, get_strategy_journal,
#     record_strategy_execution, get_strategy_history
# )

# from tsap.evolution.runtime_learning import (
#     RuntimeLearner, get_runtime_learner,
#     optimize_pattern, adapt_strategy
# )

# from tsap.evolution.offline_learning import (
#     OfflineLearner, get_offline_learner,
#     train_on_history, generate_optimal_strategies
# )


__all__ = [
    # Genetic algorithm components
    'GeneticAlgorithm', 'Individual', 'Population', 
    'EvolutionConfig', 'EvolutionResult', 'evolve_regex_pattern',
    
    # Metrics components
    'SearchMetrics', 'ExtractionMetrics', 'StrategyMetrics',
    'MetricsCalculator', 'evaluate_search_results',
    'evaluate_extraction_results', 'evaluate_strategy',
    
    # Pattern analyzer components
    'PatternAnalyzer', 'PatternStats', 'PatternVariant',
    'get_pattern_analyzer', 'analyze_pattern', 'evolve_pattern',
    
    # Pattern library components
    'PatternLibrary', 'get_pattern_library', 'add_pattern_to_library',
    'search_patterns', 'get_popular_patterns',
    
    # Once implemented, these would be added to __all__
    # 'StrategyEvolution', 'get_strategy_evolution',
    # 'evolve_strategy', 'evaluate_strategy_fitness',
    # 'StrategyJournal', 'get_strategy_journal',
    # 'record_strategy_execution', 'get_strategy_history',
    # 'RuntimeLearner', 'get_runtime_learner',
    # 'optimize_pattern', 'adapt_strategy',
    # 'OfflineLearner', 'get_offline_learner',
    # 'train_on_history', 'generate_optimal_strategies'
]