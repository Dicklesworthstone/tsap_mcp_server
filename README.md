# TSAP MCP Server

**Text Search and Processing Model Context Protocol Server**

## Overview

TSAP MCP Server is a comprehensive implementation of the Model Context Protocol designed to give large language models like Claude 3.7 powerful capabilities for searching, analyzing, and transforming text data across various domains. By integrating best-in-class command-line utilities through a standardized interface, TSAP enables AI assistants to perform sophisticated text operations without requiring users to have specialized knowledge of complex CLI tools.

## Core Rationale

1. **Leverage Existing Tools**: Many powerful text processing tools (ripgrep, awk, jq) already exist but have steep learning curves and complex syntax.

2. **Minimize Round Trips**: By enabling parallel execution and batch operations, TSAP reduces the need for multiple LLM interactions, making text analysis workflows more efficient.

3. **Progressive Abstraction**: The layered architecture lets models work at the appropriate level of abstraction for different tasks, from low-level text operations to high-level semantic analysis.

4. **Pattern Discovery**: Rather than requiring pre-defined schemas, TSAP includes tools for discovering patterns in data and automatically generating extraction rules.

## Architecture Overview

TSAP is structured in three distinct layers, each building on the capabilities of the layers below:

### Layer 1: Core Tools

Minimal, non-overlapping wrappers around high-performance CLI utilities with standardized interfaces:

- **ripgrep_tool**: High-performance text searching with regex support
- **awk_tool**: Advanced text transformation and field processing
- **jq_tool**: JSON data processing and transformation
- **sqlite_tool**: Relational queries on extracted or transformed data
- **html_processor**: Intelligently parse and search HTML content with structure awareness
- **pdf_extractor**: Native PDF text extraction that works with complex layouts
- **table_processor**: Intelligent processing of tabular data across formats

### Layer 2: Composite Operations

Pre-configured combinations of Layer 1 tools that perform common multi-step operations efficiently:

- **parallel_search**: Run multiple search patterns simultaneously
- **recursive_refinement**: Iteratively narrow search scope
- **context_extractor**: Extract meaningful code/text units
- **pattern_analyzer**: Identify and count patterns in text data
- **filename_pattern_discoverer**: Discover naming conventions in filenames
- **content_structure_analyzer**: Discover document structure patterns
- **structure_search**: Search based on structural position, not just content
- **diff_generator**: Find meaningful changes across document versions
- **regex_generator**: Generate optimal regular expressions for pattern matching
- **document_profiler**: Create structural fingerprints of documents

### Layer 3: Intelligent Analysis Tools

High-level semantic tools that leverage Layers 1 & 2 and work iteratively with the LLM to perform complex analysis tasks:

- **code_analyzer**: Comprehensive code analysis across repositories
- **document_explorer**: Analyze document collections and extract structured information
- **adaptive_metadata_extractor**: Progressively build metadata models
- **corpus_cartographer**: Map relationships between documents
- **counterfactual_analyzer**: Find what's missing or unusual in documents
- **strategy_compiler**: Create optimized execution plans for complex searches
- **resource_allocator**: Intelligently allocate computational resources

### Enhanced Systems

#### Evolution & Learning Systems

- **strategy_evolution**: Evolve search strategies based on results effectiveness
- **pattern_analyzer**: Build institutional knowledge about effective search strategies
- **strategy_journal**: Record and analyze search effectiveness over time
- **pattern_library**: Build a reusable library of effective search patterns
- **runtime_learning**: Real-time pattern optimization
- **offline_learning**: Deep analysis of historical performance

#### LLM-powered Pattern Generation

- **LLM Pattern Evolution**: Connect to an existing LLM gateway to generate superior regex patterns
- **Environment Variable Control**: Easily toggle between rule-based and LLM-based pattern generation
- **Fallback Mechanism**: Gracefully fall back to rule-based generation if LLM generation fails
- **Pattern Validation**: Automatically validate LLM-generated patterns before using them
- **Source Tracking**: Clearly identify whether patterns came from rules or an LLM

To enable LLM-powered pattern generation:

```bash
# Enable LLM pattern generation
export USE_LLM_PATTERN_GENERATION=true

# Set the LLM gateway URL (default: http://localhost:8013)
export LLM_MCP_SERVER_URL=http://localhost:8013
```

See `examples/llm_pattern_evolution.py` for a complete demonstration.

#### Plugin Architecture

TSAP implements a true plugin architecture that allows:

- Simpler core system with extensibility
- Third-party contributions for specialized domains
- Easier maintenance and individual component updates

#### Progressive Performance Levels

Distinct operational modes to balance speed vs depth:

- **Fast Mode**: Quick searches with minimal processing
- **Standard Mode**: Balanced approach
- **Deep Mode**: Exhaustive analysis with all evolutionary features enabled

#### Task Templates

Pre-configured workflows for common tasks:

- Code security audits
- Regulatory change detection
- Document corpus exploration
- And more...

#### Smart Caching & Diagnostics

- Transparent caching with intelligent invalidation
- Exceptional logging using Rich with colors and emojis
- Comprehensive diagnostics and performance profiling
- Result confidence scoring

## Key Implementation Concepts

### Parallel Execution

TSAP minimizes round-trips by executing operations in parallel:

- **Batch Processing**: Generate multiple search patterns in a single LLM call
- **Fan-out Execution**: Run the same operation across many files/directories
- **Hypothesis Testing**: Try multiple approaches at once and return the most successful

### Iterative Refinement

For complex tasks, TSAP implements multi-stage refinement:

- **Progressive Filtering**: Start broad, iteratively narrow the focus
- **Feedback Loops**: Use initial results to inform subsequent operations
- **Confidence Scoring**: Assign confidence levels to results to guide refinement

### Pattern Discovery

Rather than requiring pre-defined knowledge about document structure:

- **Adaptive Learning**: Sample data and discover patterns automatically
- **Schema Generation**: Build extraction rules based on discovered patterns
- **Self-Improvement**: Refine models through multiple iterations

## Getting Started

### Prerequisites

- Python 3.13+
- ripgrep, jq, gawk, and sqlite3 (installable via the provided scripts)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/tsap-mcp-server.git
cd tsap-mcp-server
```

2. Install dependencies:

```bash
bash scripts/install_dependencies.sh
```

3. Create and activate a virtual environment:

```bash
uv venv --python=3.13 .venv
source .venv/bin/activate
```

4. Install the package:

```bash
uv pip install -e '.[dev,docs,performance,database,visualization,analysis,cloud,semantic,js_rendering]'      
```

### GPU Support

By default, TSAP installs with faiss-cpu for vector operations and semantic search support. To enable GPU acceleration:

1. Make sure you have CUDA properly installed on your system
2. Install faiss-gpu manually with a Python version that has compatible wheels:

```bash
# Only works with Python versions that have compatible wheels (not Python 3.13 yet)
pip install faiss-gpu
```

The system will automatically detect and use GPU acceleration if the faiss-gpu package is installed correctly. If installation fails or if GPU initialization fails at runtime, the system will automatically fall back to CPU mode.

### Usage

Run the server:

```bash
tsap server
```

Or use it from Python:

```python
from tsap.core import ripgrep_tool
from tsap.composite import parallel_search

# Search for multiple patterns
results = parallel_search(
    patterns=[
        {"pattern": "api_key", "description": "Generic API key"},
        {"pattern": "password", "description": "Password"}
    ],
    file_types=["py", "js"]
)
```

## Examples

See the `examples/` directory for practical use cases:

- Code security audits
- Legal document analysis
- Document profiling
- Pattern evolution
- And more...

## Detailed Project Breakdown by Section


Here is the revised breakdown of each subfolder within `src/tsap/`:

1.  `analysis/`
    *   **What it does:** Based on its `__init__.py`, this directory contains high-level analysis tools. It defines base classes (`BaseAnalysisTool`, `AnalysisRegistry`, `AnalysisContext` in `base.py`) for creating and managing analysis tools. Specific tools like `CorpusCartographer` (`cartographer.py`) for mapping document relationships, `CodeAnalyzer` (`code.py`) for analyzing code structure/patterns/dependencies/quality, and `DocumentExplorer` (`documents.py`) for exploring document collections are implemented here. Other files like `counterfactual.py`, `llm.py`, `metadata.py`, `resource_allocator.py`, and `strategy_compiler.py` define corresponding classes/functions for those specific analysis tasks. The package docstring confirms these tools build on core and composite operations for complex tasks.
    *   **Fit & Interaction:** This layer provides the most advanced capabilities, directly implementing complex analysis features. It uses components from `tsap.composite` and `tsap.core` (as indicated by the docstring and the nature of analysis tools) and likely interacts with `tsap.storage` to persist analysis results or models. The analysis tools are exposed via the `tsap.api.routes.analysis` module.
    *   **Necessity:** This layer is essential for providing the server's intelligent analysis capabilities beyond basic search and transformation, directly addressing sophisticated user goals like understanding code quality, exploring document relationships, or compiling complex search strategies.

2.  `api/`
    *   **What it does:** This directory implements the server's REST API using FastAPI. `app.py` sets up the main FastAPI application instance. `dependencies.py` defines reusable dependencies for API routes, such as validating API keys (`get_api_key`) and handling performance modes (`performance_mode_dependency`). The `middleware/` subfolder contains modules for specific middleware functions: `error.py` sets up global exception handlers, `logging.py` implements request/response logging (`RequestLoggingMiddleware`), and `auth.py` (based on its content, *not* its name) actually defines API routes for core tools (`ripgrep_search`, `awk_process`, etc.). The `models/` subfolder defines Pydantic models (`analysis.py`, `composite.py`, etc.) used for request validation and response serialization in the API layer. The `routes/` subfolder contains the API endpoint definitions, organized by functionality (`core.py`, `composite.py`, `analysis.py`, `evolution.py`, `plugins.py`), which expose the server's capabilities over HTTP.
    *   **Fit & Interaction:** This layer serves as the network interface for the TSAP server. Routes defined in `routes/` call functions and methods from the `core`, `composite`, `analysis`, `evolution`, and `plugins` packages. It uses models from `api/models` for data validation/serialization and relies on middleware and dependencies for handling requests.
    *   **Necessity:** This package is required to expose the server's functionality via a standard REST API, allowing external clients (like LLMs or other applications) to interact with TSAP.

3.  `cache/`
    *   **What it does:** This package implements caching functionality. `manager.py` defines the `CacheManager` class, responsible for get/set operations, coordinating in-memory and disk caching, and managing cache size/TTL. It uses a `ThreadPoolExecutor` for non-blocking disk operations. `invalidation.py` defines various `InvalidationStrategy` classes (TTL, LRU, LFU, FIFO, ContentAware, Hybrid) and a `InvalidationPolicyManager` to control cache eviction. `persistence.py` defines the `CachePersistenceProvider` interface and concrete implementations like `FilePersistenceProvider` and `SqlitePersistenceProvider` for storing cache data persistently. `metrics.py` defines `CacheMetricsSnapshot` and `CacheMetricsCollector` for tracking cache performance (hit rate, size, latency). The `__init__.py` exports key functions for interacting with the cache manager.
    *   **Fit & Interaction:** This is a utility layer used by performance-sensitive operations in other packages (like `core`, `composite`, `analysis`) to store and retrieve computed results, thereby reducing redundant work. It uses `tsap.utils.metrics` for metrics collection and interacts with the filesystem or databases via its persistence providers.
    *   **Necessity:** Caching is vital for the server's performance and efficiency, preventing repeated computations for identical requests and speeding up responses.

4.  `composite/`
    *   **What it does:** This package provides composite operations (Layer 2), which combine core functionalities. The `__init__.py` exports functions like `parallel_search`, `extract_context`, `structure_search`, `generate_diff`, `generate_regex`, `discover_filename_patterns`, and `profile_document(s)` implemented in their respective modules (e.g., `parallel.py`, `context.py`). `base.py` defines the `CompositeOperation` base class, a `CompositeRegistry` for managing these operations, decorators (`register_operation`, `operation`) for registration, and utilities like `ConfidenceLevel` and `calculate_confidence` for scoring results, as seen in `confidence.py`.
    *   **Fit & Interaction:** This layer acts as an intermediate abstraction, building upon `tsap.core` tools and providing more complex, reusable operations for the `tsap.analysis` layer or direct API exposure. It interacts with `tsap.cache` and `tsap.utils`.
    *   **Necessity:** This layer encapsulates common multi-step tasks, offering optimized and standardized implementations that simplify the development of higher-level analysis tools and ensure consistent behavior for common workflows.

5.  `core/`
    *   **What it does:** This directory contains the foundational tools (Layer 1). The `__init__.py` shows it provides wrappers and interfaces for external tools like `ripgrep` (`ripgrep.py`), `awk` (`awk.py`), `jq` (`jq.py`), and `sqlite` (`sqlite.py`), as well as internal processors for HTML (`html_processor.py`), PDF (`pdf_extractor.py`), and tables (`table_processor.py`). `base.py` defines the `BaseCoreTool` and `ToolRegistry`. `process.py` provides utilities (`run_process`, `run_pipeline`) for executing external commands asynchronously. `validation.py` offers functions for validating various inputs like paths, regex, JSON, types, and sanitizing arguments.
    *   **Fit & Interaction:** This is the lowest layer interacting directly with external CLI tools or performing basic file processing. Its components are heavily used by `tsap.composite` and `tsap.analysis`. It relies on `tsap.utils` for logging and error handling.
    *   **Necessity:** Provides the essential, high-performance building blocks for text search, transformation, and extraction, forming the bedrock of the entire server's capabilities.

6.  `evolution/`
    *   **What it does:** This package implements evolutionary algorithms and learning systems. `base.py` defines core concepts like `EvolutionAlgorithm`, `Individual`, `Population`, and `EvolutionConfig`. `genetic.py` provides a `GeneticAlgorithm` implementation, including functions for evolving regex patterns (`evolve_regex_pattern`). `metrics.py` defines data structures (`SearchMetrics`, `ExtractionMetrics`, `StrategyMetrics`) and functions (`evaluate_search_results`, etc.) for evaluating the performance of patterns and strategies. `pattern_analyzer.py` contains the `PatternAnalyzer` for evaluating and generating variants of search patterns. `pattern_library.py` defines the `PatternLibrary` for storing and managing patterns, interacting with `tsap.storage.pattern_store`. `strategy_journal.py` implements the `StrategyJournal` for recording strategy execution history, interacting with `tsap.storage.strategy_store`. `runtime_learning.py` and `offline_learning.py` implement mechanisms for real-time and batch learning/optimization, respectively.
    *   **Fit & Interaction:** This advanced layer enables the system to adapt and improve. It interacts with `tsap.storage` to persist learned patterns and strategies, uses `tsap.core` and `tsap.composite` tools for evaluation, and provides optimized patterns/strategies back to the `analysis` and `composite` layers.
    *   **Necessity:** This allows the server to automatically optimize its performance and effectiveness over time by learning from past executions and feedback, making it more robust and intelligent.

7.  `incremental/`
    *   **What it does:** This package provides tools for processing data incrementally. `splitter.py` defines the `Splitter` interface and concrete implementations (`ListSplitter`, `TextSplitter`, `FileSplitter`, `JsonSplitter`, `DirectorySplitter`) for dividing input data into chunks. `processor.py` defines the `IncrementalProcessor` base class and registry for managing chunk processing. `aggregator.py` defines the `Aggregator` interface and implementations (`ListAggregator`, `DictionaryAggregator`, `CounterAggregator`, `StatisticsAggregator`, `GroupByAggregator`) for combining results from processed chunks. `streamer.py` defines `InputStream` and `OutputStream` interfaces and implementations (`FileInputStream`, `JsonOutputStream`, etc.) for handling data streams.
    *   **Fit & Interaction:** This layer provides mechanisms to handle large datasets or continuous data streams, making operations from `core`, `composite`, and `analysis` scalable. It defines the workflow: split -> process -> aggregate.
    *   **Necessity:** Enables the server to handle data that exceeds memory limits or arrives continuously, crucial for processing large files, directories, or real-time data feeds.

8.  `mcp/`
    *   **What it does:** This package defines and handles the Model Context Protocol (MCP). `protocol.py` defines the core request (`MCPRequest`), response (`MCPResponse`), and error (`MCPError`) structures, along with command types (`MCPCommandType`). `models.py` defines numerous Pydantic models specifying the structure of arguments (`*Params`) and results (`*Result`) for various operations exposed via MCP. `handler.py` contains the central `handle_request` function which routes incoming `MCPRequest` objects to registered handler functions (decorated with `@command_handler`), executes the corresponding TSAP operations (calling functions from `core`, `composite`, `analysis`, etc.), and uses helper functions (`create_success_response`, `create_error_response`) from `protocol.py` to format the output.
    *   **Fit & Interaction:** This is the specific protocol interface designed for AI model interaction. The `handler` acts as the main dispatcher, translating MCP commands into internal TSAP function calls and packaging results back into MCP responses. It relies heavily on the schemas defined in `models.py`.
    *   **Necessity:** Provides the structured, standardized communication protocol required for AI models like Claude to reliably interact with and utilize the server's capabilities.

9.  `output/`:
    *   **What it does:** This package is responsible for formatting TSAP results into different output representations. `formatter.py` defines the `OutputFormatter` base class and basic `JsonFormatter` and `TextFormatter`. `json_output.py` provides an `EnhancedJsonEncoder` (handling types like `datetime`, `UUID`, `set`) and formatters for standard JSON (`EnhancedJsonFormatter`) and JSON Lines (`JsonLinesFormatter`). `csv_output.py` provides a `CsvFormatter`. `terminal.py` implements `TerminalFormatter` using the Rich library for styled console output, including tables and panels. `reporting.py` defines structures (`Report`, `ReportSection`) and a `ReportGenerator` for creating structured reports in various formats (Text, Markdown, HTML, JSON, CSV). `visualization.py` defines `Visualization` base class and implementations like `BarChart`, `LineChart`, `NetworkGraph` (using Matplotlib and NetworkX where available) for creating visual representations of results. The `__init__.py` provides factory functions (`create_formatter`, `format_output`, `save_output`) to easily use these formatters.
    *   **Fit & Interaction:** This layer takes data generated by other TSAP components (e.g., `analysis`, `composite` results) and transforms it for presentation or consumption by users or other systems. It's used by the `api` layer for generating HTTP responses and potentially by the `cli`.
    *   **Necessity:** Ensures results are presented in a clear, usable, and appropriate format depending on the context (e.g., human-readable terminal output, machine-readable JSON, structured reports).

10. `plugins/`:
    *   **What it does:** This package implements the plugin system. `interface.py` defines the base `Plugin` class and specific subtypes (`CoreToolPlugin`, `CompositePlugin`, etc.), `PluginType` and `PluginCapability` enums, and decorators (`@plugin_info`, `@plugin_capabilities`, `@plugin_dependencies`) for defining plugin metadata. `loader.py` (`PluginLoader`) handles discovering plugins in specified paths (including built-in) and loading their code. `manager.py` (`PluginManager`) provides a high-level interface to manage the plugin lifecycle (initialize, register, enable/disable, reload). `registry.py` (`PluginRegistry`) acts as a central store for components (tools, operations, functions) registered by active plugins. The `builtin/` subdirectory contains default plugins like `ExamplePlugin` (`example.py`).
    *   **Fit & Interaction:** This is a foundational architectural component. The `manager` orchestrates loading and registration. The `loader` interacts with the filesystem and Python's import system. The `registry` provides a central lookup for components used by other parts of TSAP, such as the `mcp.handler` or `api` routes.
    *   **Necessity:** Provides the core extensibility mechanism, allowing TSAP's functionality to be expanded with new tools, analyses, and integrations without modifying the base code.

11. `project/`:
    *   **What it does:** This package manages the concept of distinct projects. `context.py` defines `ProjectContext` to hold project-specific state (like directories, settings, registered files, active analyses/operations) and `ProjectRegistry` to manage multiple project contexts. `history.py` defines `CommandEntry` and `CommandHistory` (using `tsap.storage`) to record and query commands executed within a project. `profile.py` defines `ProjectProfile` for storing configurations/preferences and `ProfileManager` (using `tsap.storage`) to manage multiple profiles per project. `transfer.py` defines `ProjectExport` and `ProjectImport` classes for exporting/importing project data (context, profiles, history, results, files) often using ZIP archives.
    *   **Fit & Interaction:** This layer provides organizational structure and state persistence for user workflows. It relies heavily on `tsap.storage` for saving project-related data. The `ProjectContext` can be used by other layers (e.g., `analysis`, `composite`) to scope operations or access project-specific settings.
    *   **Necessity:** Enables users to manage multiple distinct work environments, each with its own configuration, data, and history, making TSAP suitable for managing complex or concurrent tasks.

12. `storage/`:
    *   **What it does:** This is the data persistence layer. `database.py` provides a `Database` class (currently focused on SQLite) with utilities for connection management, transactions, and common operations, along with a `DatabaseRegistry`. Specific stores are built on this base: `history_store.py` (`HistoryStore`) saves command history, `pattern_store.py` (`PatternStore`) saves search patterns and their metadata/stats/examples, `profile_store.py` (`ProfileStore`) saves project/document profiles and settings, and `strategy_store.py` (`StrategyStore`) saves search/analysis strategies and their execution records.
    *   **Fit & Interaction:** This layer provides the mechanism for saving and retrieving application state and learned data. It's used by `project` (for context, history, profiles), `evolution` (for patterns, strategies, journal entries), and potentially `cache` (for persistent caching via `SqlitePersistenceProvider`).
    *   **Necessity:** Ensures that project configurations, command history, learned patterns/strategies, and user profiles are saved persistently across server restarts, enabling stateful operation and long-term learning.

13. `templates/`:
    *   **What it does:** This package contains predefined, reusable workflows called templates. `base.py` defines the `Template` base class, `TemplateParameter` for defining inputs, `TemplateResult` for outputs, and a `TemplateRunner` (managed by `registry.py`) for executing templates. Specific templates like `SecurityAuditTemplate` (`security_audit.py`), `RegulatoryAnalysisTemplate` (`regulatory_analysis.py`), and `CorpusExplorationTemplate` (`corpus_exploration.py`) implement concrete workflows by orchestrating calls to various core, composite, and analysis tools. `custom.py` serves as a placeholder for user-defined templates.
    *   **Fit & Interaction:** Templates represent high-level workflows built using components from the `analysis`, `composite`, and `core` layers. They provide a simplified interface for common, complex tasks, exposed via the `api` or `cli`.
    *   **Necessity:** Offers users pre-built solutions for common tasks, reducing the complexity of interacting with the server and ensuring best practices are followed for specific workflows like security audits or regulatory analysis.

14. `utils/`:
    *   **What it does:** This package is a collection of shared utility modules. `async_utils.py` provides helpers like `TaskManager`, `AsyncSemaphore`, `Throttler`, and `retry` decorators for managing asynchronous code. `caching.py` contains caching utilities like `MemoryCache`, `DiskCache`, `CacheManager`, and caching decorators (`@cached`, `@async_cached`). `errors.py` defines custom `TSAPError` subclasses for specific error conditions. `filesystem.py` offers functions for file operations (hashing, mime type detection, async read/write, finding files). `helpers.py` contains miscellaneous functions like `generate_id`, `truncate_string`, `format_timestamp`. `metrics.py` provides a `MetricsCollector` and registry for tracking general application metrics. `optimization.py` includes functions for estimating resource usage, memoization, and optimizing parameters like chunk/batch sizes. `security.py` provides functions for sanitizing inputs (filenames, paths, commands) and secure operations (password hashing, token generation). The `diagnostics/` sub-package contains tools for system analysis (`analyzer.py`), performance profiling (`profiler.py`), report generation (`reporter.py`), and data visualization (`visualizer.py`). The `logging/` sub-package implements the entire rich logging system (console, logger, formatters, handlers, panels, progress tracking, themes, emojis).
    *   **Fit & Interaction:** This is a foundational support layer providing common, reusable code leveraged by nearly all other packages in the `tsap` application for tasks ranging from logging and error handling to async management and security.
    *   **Necessity:** Contains essential cross-cutting utilities that ensure code consistency, maintainability, security, and observability throughout the entire application.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
