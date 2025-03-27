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
uv pip install -e '.[dev,docs,performance,database,visualization,analysis,cloud]'         
```

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.
