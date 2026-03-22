# Changelog

All notable changes to **TSAP MCP Server** are documented in this file.

This project has no tagged releases. All entries correspond to commits on `main`, grouped into logical milestones and organized by capability area. Each entry links to the exact commit on GitHub.

- Repository: <https://github.com/Dicklesworthstone/tsap_mcp_server>
- Package version: `0.1.0` (since initial commit)
- License: MIT with OpenAI/Anthropic Rider (since 2026-02-21)

---

## 2026-02-21 / 2026-02-22 -- License Update and Repository Metadata

### Licensing

- Replace the plain MIT license with **MIT + OpenAI/Anthropic Rider**, restricting use by OpenAI, Anthropic, and their affiliates without express written permission from Jeffrey Emanuel
  ([`16e02de`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/16e02de56c58c0b8dcca308b9cc10657e4a1135a))
- Update all README license references to reflect the new license
  ([`0da5a45`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/0da5a4548e8c22bfdc378a62dfe74cec72acb29a))

### Repository Metadata

- Add GitHub social preview image (1280x640, `gh_og_share_image.png`) for consistent link previews
  ([`c329c4b`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/c329c4b0a0ab10d30ace9e39063a0b3f06607369))

---

## 2026-02-16 -- Cache Subsystem Removal

### Refactoring (Breaking)

- **Remove the entire `tsap.cache` package** -- 2,977 lines across 5 files deleted:
  - `CacheManager` with memory and disk backends, size-based eviction (`manager.py`, 620 lines)
  - `InvalidationStrategy` hierarchy: TTL, LRU, LFU, size-based, composite (`invalidation.py`, 703 lines)
  - `CacheMetrics` with hit/miss/eviction tracking and reporting (`metrics.py`, 425 lines)
  - `CachePersistence` with SQLite and filesystem serialization (`persistence.py`, 1,205 lines)
  - Package init (`__init__.py`, 24 lines)
- **Rationale**: over-engineered for current needs; carried significant maintenance surface without being utilized in the active request pipeline
  ([`8d51198`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/8d511980bfa41dade357439f8e2598486dabaaee))

---

## 2025-04-07 -- Architecture Overhaul: Three-Component Split

The single largest commit in the project (225 files changed, +35,581 / -6,245 lines). Restructures the codebase from a monolithic `tsap.mcp` subpackage into a proper three-component architecture: `tsap` (core), `toolapi` (internal API), and `tsap_mcp` (MCP adapter).

### MCP Adapter Layer (`tsap_mcp`)

- **Introduce `src/tsap_mcp/` as a standalone MCP adapter package**, replacing the former `tsap.mcp` subpackage:
  - `server.py` -- MCP server implementation
  - `adapter.py` -- bridge between TSAP core and MCP protocol
  - `cli.py` -- command-line interface (`tsap-mcp run`, `tsap-mcp info`, `tsap-mcp install --desktop`)
  - `lifespan.py` -- server lifecycle management
  - `tool_map.py` -- tool name/handler mapping
- **MCP tools** (`tsap_mcp/tools/`): search, processing, analysis, visualization, composite
- **MCP resources** (`tsap_mcp/resources/`): files, project, config, semantic, analysis, composite, processing, search, visualization
- **MCP prompts** (`tsap_mcp/prompts/`): code analysis, composite, processing, search, visualization
- **Protocol adapters** (`tsap_mcp/adapters/tool_adapters.py`) -- 681 lines
- **Utility modules** (`tsap_mcp/utils/`): analysis, composite, context, processing, search, visualization

### ToolAPI Extraction (`tsap.toolapi`)

- **Extract `toolapi` from the old `tsap.mcp`** as a standalone internal API layer:
  - `protocol.py` -- protocol definition (310 lines)
  - `handler.py` -- tool request handling (1,300 lines)
  - `models.py` -- data models (1,711 lines)
  - `client/base.py` -- programmatic client library (585 lines)
- Delete the old `tsap.mcp` subpackage entirely

### MCP Infrastructure

- Add `mcp[cli]>=0.1.0` as a core dependency
- Include full MCP protocol schema (`mcp_protocol_schema_2025-03-25_version.json`)
- Add MCP Python library docs reference (`mcp_python_lib_docs.md`)
- Add helper scripts: `run_mcp_server.py`, `run_dual_servers.py`, `run_parallel.py`, `add_mcp_dependency.py`, `check_mcp_completeness.py`

### Examples

- Create `mcp_examples/` with 33 dedicated MCP client demos covering all tool categories
- Key infrastructure examples: `mcp_stdio_server.py`, `mcp_proxy.py`, `test_stdio_server.py`
- Diagnostic utilities: `debug_tools.py`, `verify_tool_call.py`, `simple_mcp_test.py`
- Add `run_complete_battery_of_demonstration_scripts.py` and full demo output (2,655 lines)
- Refactor existing `examples/` to use `toolapi` imports instead of old `mcp` imports

### Project Metadata

- Rename project from `tsap_mcp_server` to `tsap-mcp` in `pyproject.toml`
- Rewrite README with three-component architecture documentation
- Add `src/tsap/api/models/auth.py` for authentication models
- Enable `spacy==3.7.2` in dependencies
- Add `isort>=5.12.0` to dev dependencies
- Remove scaffold script `create_project_structure.sh`

### Test Suite

- Add MCP test modules: `test_integration.py`, `test_prompts.py`, `test_resources.py`, `test_server.py`, `test_tools.py`

### Bug Fixes

- Broad import path corrections across all `src/tsap/` modules to resolve circular imports and broken references after the restructuring
- Add `src/tsap/version.py` with package, protocol, and API version constants

([`5baff2d`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/5baff2d0bb0d641db493924112943a8605a21246))

---

## 2025-04-01 -- MCP Client, Structure Search Improvements, Pattern Bootstrap

### Search and Pattern Capabilities

- **MCP client library** (`src/tsap/mcp/client/base.py`, 596 lines) -- programmatic `MCPClient` class for server interaction via `httpx`
- **Bootstrap pattern system** (`src/tsap/composite/bootstrap_patterns.py`) -- seed the pattern library with initial security, code quality, and documentation patterns
- Enhance `src/tsap/composite/structure_search.py` -- improved structural matching algorithms
- Expand `src/tsap/evolution/pattern_analyzer.py` -- deeper pattern effectiveness analysis with LLM integration

### Server and Handler

- Major expansion of `src/tsap/mcp/handler.py` (+482 lines) -- register additional tool handlers
- Expand `src/tsap/server.py` (+356 lines) -- richer server configuration and endpoints

### Examples and Data

- `advanced_document_profiler_demo.py`, `advanced_patterns_demo.py`, `advanced_structure_search_demo.py`
- `llm_pattern_evolution.py`, `log_pattern_evolution.py`, `simple_structure_search_test.py`, `test_llm_patterns.py`
- Sample data for structure search (Python, HTML, Markdown) and document analysis (code, strategic thinking, technical spec)

([`9ecd381`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/9ecd3815b55e57d8f00ee32e1c32f924bcc5e3c2))

---

## 2025-03-31 -- HTML Processor, SQLite Tool Expansion, Application Entry Point

### Core Tool Improvements

- **HTML processor expansion** -- `src/tsap/core/html_processor.py` gains +482 lines for structure-aware parsing, element extraction, and content normalization
- **SQLite tool expansion** -- `src/tsap/core/sqlite.py` gains +364 lines for richer query capabilities and result formatting
- **Application entry point** -- `src/tsap/main.py` (128 lines) provides a unified startup path

### MCP Handler

- Register HTML processor and SQLite tools as MCP-callable handlers
- Expand MCP client example with new tool invocations

### Examples and Data

- `advanced_html_processor_demo.py` (1,220 lines) -- comprehensive HTML parsing and extraction
- `advanced_sqlite_demo.py` (960 lines) -- relational queries on extracted data
- Add HTML example data: SEC 10-Q filing, Wikipedia algebraic topology article
- Reorganize PDF example data into `tsap_example_data/pdfs/` subdirectory
- Add Jupyter notebook stubs for future tutorials

### Minor Fixes

- Fix `src/tsap/cache/persistence.py` edge cases
- Fix `src/tsap/evolution/genetic.py` -- use `is` instead of `==` for type comparisons, remove unused imports

([`7b98219`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/7b98219a945d59accdff5c4d7365ba077fc00b8b),
[`f3581f6`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/f3581f6f403b2c0636df995c7bee83635efd54f8))

---

## 2025-03-30 -- PDF, Table, AWK, and JQ Tool Fixes and Demos

### Core Tool Fixes

- **Table processor rewrite** -- `src/tsap/core/table_processor.py` significantly refactored (374 lines changed) for correct multi-format table handling
  ([`46e0217`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/46e021744df568e2b1c01f3bafd587f192af83d4))
- **AWK tool fixes** -- `src/tsap/core/awk.py` and `src/tsap/core/jq.py` corrected for proper subprocess invocation and output parsing
  ([`832a6f9`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/832a6f97531fbaf0baad81cbbbfa214d93b3fc71))
- **PDF extractor fixes** -- correct MCP handler registration and tool invocation for PDF operations
  ([`52ecb3b`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/52ecb3b7e57ff389b047e17f5494830e3cd32d86))

### MCP Integration

- Register table processor, AWK, and JQ tools in MCP handler
- Fix MCP model definitions for table and PDF operations

### Examples

- `advanced_table_processor_demo.py` (602 lines) -- CSV, TSV, and JSON table manipulation
- `advanced_awk_demo.py` (395 lines) -- field processing, pattern matching, aggregation
- `advanced_jq_demo.py` (445 lines) -- JSON path queries, filtering, transformation
- `advanced_pdf_extractor_demo.py` (483 lines) -- multi-page PDF text extraction

### Example Data

- CSV tables: `sales_data.csv`, `products.csv`, `inventory.csv`, `orders.tsv`
- JSON data: `sensor_data.json`, `logs.jsonl`, `nested_data.json`, `users.json`
- Regional CSV reports: `report_east.csv`, `report_west.csv`, `report_south.csv`
- PDF test documents: Durnovo presentation slides, Lamport Paxos paper

---

## 2025-03-29 -- Semantic Search and Server Hardening

### Semantic Search (New Capability)

- **Semantic search pipeline** using FAISS vector indexing and Nomic AI embeddings:
  - `src/tsap/core/semantic_search_tool.py` (372 lines) -- embedding generation, FAISS index management, similarity search
  - `src/tsap/composite/semantic_search.py` (80 lines) -- composite semantic search orchestration
- Add `faiss-cpu`, `nomic`, and `sentence-transformers` to `pyproject.toml` dependencies
  ([`cdfa8b9`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/cdfa8b9c905d8305e2186d966aec05bf1b53b60e))

### Server and Logging Improvements

- Expand `src/tsap/server.py` with semantic search integration, aggressive recursive log truncation, and `JSONResponse` error handling (+179 lines)
  ([`76ad93f`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/76ad93fafd492e559c6a225c0f88a871504fb5f3))
- Refactor `src/tsap/composite/document_profiler.py` for more robust profiling
- Simplify API layer (`src/tsap/api/app.py`) -- mount API router, reduce boilerplate
- Overhaul logging subsystem: fix formatter, console, and logger modules
  ([`b28e7b4`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/b28e7b411f9aa2c8c0faec80840207f4d57578e2))

### Bug Fixes

- Fix MCP protocol model definitions for response serialization
  ([`1c131a6`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/1c131a6156a6427c226573785dcab790edb7a696))
- Fix logging emoji handling and logger initialization
  ([`76ad93f`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/76ad93fafd492e559c6a225c0f88a871504fb5f3))
- Fix semantic search tool result handling and MCP handler dispatch
  ([`b28e7b4`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/b28e7b411f9aa2c8c0faec80840207f4d57578e2))

### Examples and Data

- `semantic_search_demo.py` (319 lines, later expanded to 510 lines)
- `advanced_ripgrep_demo.py` (424 lines)
- `mcp_client_example.py` (249 lines)
- `performance_modes.py` (193 lines)
- `plugin_development.py` (459 lines)
- `search_documents.py` (106 lines)
- `standalone_mcp_server.py` (77 lines)
- `debug_server.py` (64 lines)
- Add `uv.lock` for reproducible dependency resolution
- Add `tsap_example_data/code/llm_tournament.py` (1,644 lines)
- Add reference docs: `browser_use_docs.md`, `mcp_python_lib_docs.md`, `sample_input_document_as_markdown__durnovo_memo.md`
- MCP protocol test (`tests/test_mcp/test_protocol.py`, 93 lines)

---

## 2025-03-27 / 2025-03-28 -- Initial Implementation

These four commits establish the full TSAP system -- from project scaffold to working core, analysis, composite, evolution, storage, and infrastructure layers.

### Core Search and Processing Tools (`src/tsap/core/`)

- **ripgrep wrapper** (`ripgrep.py`, 454 lines) -- high-performance regex search with path filtering, case-sensitivity, context lines, and result formatting
- **AWK wrapper** (`awk.py`, 312 lines) -- text transformation, field processing, and pattern-action rules
- **JQ wrapper** (`jq.py`, 341 lines) -- JSON processing, path queries, filtering, and transformation
- **SQLite wrapper** (`sqlite.py`, 336 lines) -- relational queries on extracted or structured data
- **HTML processor** (`html_processor.py`, 849 lines) -- structure-aware HTML parsing, element extraction, and content normalization
- **PDF extractor** (`pdf_extractor.py`, 803 lines) -- text extraction from complex PDF layouts with page-level control
- **Table processor** (`table_processor.py`, 382 lines, added in second commit) -- intelligent processing of tabular data across CSV, TSV, and JSON formats
- **Subprocess manager** (`process.py`, 436 lines) -- safe subprocess execution with timeout and output capture
- **Input validation** (`validation.py`, 424 lines) -- parameter validation for all tool inputs
- **Base classes** (`base.py`, 219 lines) -- abstract base for all core tool implementations

### Analysis System (`src/tsap/analysis/`)

- **Code analysis** (`code.py`, 828 lines) -- code quality metrics, structure analysis, complexity assessment
- **Document analysis** (`documents.py`, 690 lines) -- document content analysis, topic extraction, readability scoring
- **Metadata extraction** (`metadata.py`, 516 lines) -- file and content metadata extraction
- **Strategy compiler** (`strategy_compiler.py`, 529 lines) -- execution plan optimization for multi-tool workflows
- **LLM integration** (`llm.py`, 368 lines) -- LLM-assisted analysis capabilities
- **Resource allocator** (`resource_allocator.py`, 319 lines) -- computational resource management
- **Counterfactual analysis** (`counterfactual.py`, 245 lines) -- detect missing or unusual content
- **Cartographer** (`cartographer.py`, 129 lines) -- document relationship mapping

### Composite Operations (`src/tsap/composite/`)

- **Structure search** (`structure_search.py`, 913 lines) -- search based on structural document position, not just content
- **Context extractor** (`context.py`, 905 lines) -- extract meaningful code/text units around matches
- **Document profiler** (`document_profiler.py`, 851 lines) -- create structural fingerprints of documents
- **Pattern matching** (`patterns.py`, 981 lines) -- identify, count, and categorize patterns across text
- **Confidence scoring** (`confidence.py`, 1,006 lines) -- confidence-weighted result ranking
- **Regex generator** (`regex_generator.py`, 589 lines) -- automatically generate regular expressions from examples
- **Filename discovery** (`filenames.py`, 566 lines) -- discover naming conventions in project files
- **Diff generator** (`diff_generator.py`, 512 lines) -- cross-version change detection and analysis
- **Structure analyzer** (`structure.py`, 641 lines) -- discover document structure patterns
- **Parallel search** (`parallel.py`, 390 lines) -- run multiple search patterns simultaneously
- **Refinement** (`refinement.py`, 405 lines) -- iteratively narrow search scope
- **Composite base** (`base.py`, 417 lines) -- base class for multi-step composite operations

### Evolutionary Search Engine (`src/tsap/evolution/`)

- **Genetic algorithm** (`genetic.py`, 1,218 lines) -- genetic algorithm with crossover, mutation, and selection strategies for search optimization
- **Pattern library** (`pattern_library.py`, 1,718 lines) -- persistent, categorized library of reusable search patterns
- **Pattern analyzer** (`pattern_analyzer.py`, 1,101 lines) -- analyze pattern effectiveness across corpora
- **Strategy evolution** (`strategy_evolution.py`, 3,055 lines) -- evolve multi-tool search strategies over time
- **Strategy journal** (`strategy_journal.py`, 1,108 lines) -- track strategy effectiveness and decisions
- **Offline learning** (`offline_learning.py`, 1,116 lines) -- batch pattern learning from historical data
- **Runtime learning** (`runtime_learning.py`, 626 lines) -- real-time strategy optimization during execution
- **Evolution base** (`base.py`, 781 lines) -- foundational types and interfaces
- **Evolution metrics** (`metrics.py`, 707 lines) -- fitness tracking and population statistics

### Storage Layer (`src/tsap/storage/`)

- **Database** (`database.py`, 991 lines) -- SQLite-backed persistence with migration support
- **History store** (`history_store.py`, 773 lines) -- search history storage and retrieval
- **Pattern store** (`pattern_store.py`, 1,250 lines) -- persistent pattern catalog
- **Profile store** (`profile_store.py`, 1,215 lines) -- project profile persistence
- **Strategy store** (`strategy_store.py`, 1,313 lines) -- evolved strategy persistence

### Project Management (`src/tsap/project/`)

- **Project context** (`context.py`, 866 lines) -- track active project state and file organization
- **Project history** (`history.py`, 979 lines) -- record and query project-level analysis history
- **Project profiling** (`profile.py`, 1,177 lines) -- comprehensive project characterization
- **Transfer learning** (`transfer.py`, 1,041 lines) -- apply insights from one project to another

### Incremental Processing (`src/tsap/incremental/`)

- **Aggregator** (`aggregator.py`, 729 lines) -- combine results from incremental operations
- **Processor** (`processor.py`, 545 lines) -- process large inputs in manageable chunks
- **Splitter** (`splitter.py`, 686 lines) -- intelligent document/data splitting
- **Streamer** (`streamer.py`, 831 lines) -- stream-process large files without full memory loading

### Plugin System (`src/tsap/plugins/`)

- **Plugin interface** (`interface.py`, 583 lines) -- plugin contract definition
- **Plugin loader** (`loader.py`, 1,134 lines) -- dynamic plugin discovery and loading
- **Plugin manager** (`manager.py`, 649 lines) -- lifecycle management
- **Plugin registry** (`registry.py`, 776 lines) -- registration and lookup
- **Example plugin** (`builtin/example.py`, 154 lines) -- reference implementation

### Template System (`src/tsap/templates/`)

- **Security audit template** (`security_audit.py`, 508 lines) -- pre-built code security audit workflow
- **Corpus exploration template** (`corpus_exploration.py`, 856 lines) -- document corpus analysis workflow
- **Regulatory analysis template** (`regulatory_analysis.py`, 713 lines) -- regulatory document analysis
- **Custom template** (`custom.py`, 99 lines) -- user-defined template base
- **Template registry** (`registry.py`, 429 lines) -- template discovery and management
- **Template base** (`base.py`, 539 lines) -- foundational template types

### Output and Visualization (`src/tsap/output/`)

- **Visualization** (`visualization.py`, 1,094 lines) -- matplotlib/networkx data and network visualization
- **Reporting** (`reporting.py`, 907 lines) -- structured report generation
- **Terminal** (`terminal.py`, 683 lines) -- rich terminal output with formatting
- **JSON output** (`json_output.py`, 270 lines) -- structured JSON result formatting
- **CSV output** (`csv_output.py`, 155 lines) -- tabular CSV result formatting
- **Formatter** (`formatter.py`, 194 lines) -- output format dispatch

### Cache Subsystem (`src/tsap/cache/`)

- **Cache manager** (`manager.py`, 592 lines) -- memory/disk backends with size-based eviction
- **Invalidation** (`invalidation.py`, 703 lines) -- TTL, LRU, LFU, size-based, and composite invalidation strategies
- **Metrics** (`metrics.py`, 425 lines) -- cache hit/miss/eviction tracking
- **Persistence** (`persistence.py`, 1,205 lines) -- SQLite and filesystem cache serialization

*Note: The cache subsystem was later removed in the 2026-02-16 refactoring.*

### Utilities (`src/tsap/utils/`)

- **Async utilities** (`async_utils.py`, 1,121 lines) -- async helpers, task management, concurrency control
- **Caching** (`caching.py`, 1,056 lines) -- function-level memoization
- **Diagnostics** (`diagnostics/`) -- system analyzer (584 lines), profiler (710 lines), reporter (963 lines), visualizer (1,403 lines)
- **Logging** (`logging/`) -- structured logger, Rich console handler, dashboard (867 lines), log handler (609 lines)
- **Optimization** (`optimization.py`, 677 lines) -- performance tuning utilities
- **Security** (`security.py`, 518 lines) -- input sanitization and security utilities
- **Metrics** (`metrics.py`, 329 lines) -- system-level metric collection
- **Helpers** (`helpers.py`, 230 lines) -- common utility functions
- **Errors** (`errors.py`, 145 lines) -- custom exception hierarchy

### MCP Protocol Layer (`src/tsap/mcp/`)

- **Protocol** (`protocol.py`) -- MCP request/response definitions
- **Handler** (`handler.py`) -- tool request routing and dispatch
- **Models** (`models.py`) -- MCP data models

*Note: This original `tsap.mcp` subpackage was later replaced by `tsap_mcp` (standalone adapter) and `tsap.toolapi` (internal API) in the 2025-04-07 architecture overhaul.*

### API Layer (`src/tsap/api/`)

- **FastAPI application** (`app.py`, 133 lines) -- REST API with CORS, auth, error, and logging middleware
- **Dependencies** (`dependencies.py`, 123 lines) -- FastAPI dependency injection
- **Middleware**: authentication (`auth.py`, 351 lines), error handling (`error.py`, 182 lines), logging (`logging.py`, 101 lines)
- **Route stubs**: analysis, composite (added in later commits: core, evolution, plugins)
- **Model stubs**: analysis (155 lines), composite (236 lines)

### CLI (`src/tsap/cli.py`)

- Full command-line interface (957 lines) for server management and tool invocation

### Configuration (`src/tsap/config.py`)

- Environment-variable-driven configuration (367 lines) supporting `TSAP_HOST`, `TSAP_PORT`, `TSAP_LOG_LEVEL`, `TSAP_PERFORMANCE_MODE`, `TSAP_CACHE_ENABLED`, `TSAP_DEBUG`

### Infrastructure

- `Dockerfile` (87 lines) and `docker-compose.yml` for containerized deployment
- `pyproject.toml` (243 lines) with hatchling build system, full dependency specification, and optional extras (`[all]`, `[dev]`)
- `install_dependencies.sh` (177 lines) -- system dependency installer
- `create_project_structure.sh` (512 lines) -- project scaffold generator (later removed)
- `.gitignore`, `.dockerignore`, `.env.example`
- GitHub Actions workflow stubs: CI, dependency review, release
- MkDocs documentation structure (`docs/`)
- Test directory structure for all subsystems (empty test files)

### Examples

- `api_client.py`, `code_security_audit.py`, `document_dna_profiling.py`, `evolving_search.py`, `incremental_processing.py`
- Example data: `tsap_example_data/code/main.py`, `utils.py`; `documents/report_v1.txt`, `report_v2.txt`

([`8700377`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/8700377038e809032dedac8bbdbbc1d1d2440b84),
[`5653506`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/56535063cf987c555d73c2ed144cd0d32a40e63d),
[`6ea92ce`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/6ea92ce4fd14ae7c0f2f6ee54fb5fdd541019934),
[`e016725`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/e0167254841e13f4bfe40b19cd9531310695c88b))

---

## Commit Index

| Date | Hash | Summary |
|------|------|---------|
| 2026-02-22 | [`0da5a45`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/0da5a4548e8c22bfdc378a62dfe74cec72acb29a) | docs: update README license references to MIT + OpenAI/Anthropic Rider |
| 2026-02-21 | [`16e02de`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/16e02de56c58c0b8dcca308b9cc10657e4a1135a) | chore: update license to MIT with OpenAI/Anthropic Rider |
| 2026-02-21 | [`c329c4b`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/c329c4b0a0ab10d30ace9e39063a0b3f06607369) | chore: add GitHub social preview image (1280x640) |
| 2026-02-16 | [`8d51198`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/8d511980bfa41dade357439f8e2598486dabaaee) | refactor: remove cache subsystem |
| 2025-04-07 | [`5baff2d`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/5baff2d0bb0d641db493924112943a8605a21246) | Architecture overhaul: MCP adapter layer + ToolAPI extraction |
| 2025-04-01 | [`9ecd381`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/9ecd3815b55e57d8f00ee32e1c32f924bcc5e3c2) | MCP client library, structure search, pattern bootstrap |
| 2025-03-31 | [`f3581f6`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/f3581f6f403b2c0636df995c7bee83635efd54f8) | Tweaks to cache persistence and genetic evolution |
| 2025-03-31 | [`7b98219`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/7b98219a945d59accdff5c4d7365ba077fc00b8b) | HTML processor, SQLite tool, entry point |
| 2025-03-30 | [`52ecb3b`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/52ecb3b7e57ff389b047e17f5494830e3cd32d86) | Fix PDF extraction demo |
| 2025-03-30 | [`46e0217`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/46e021744df568e2b1c01f3bafd587f192af83d4) | Fix table processing |
| 2025-03-30 | [`832a6f9`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/832a6f97531fbaf0baad81cbbbfa214d93b3fc71) | Fix AWK/JQ demos and tool implementations |
| 2025-03-29 | [`b28e7b4`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/b28e7b411f9aa2c8c0faec80840207f4d57578e2) | Fixes and improvements (logging, API, server, document profiler) |
| 2025-03-29 | [`76ad93f`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/76ad93fafd492e559c6a225c0f88a871504fb5f3) | Fixes (semantic search, server, logging) |
| 2025-03-29 | [`cdfa8b9`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/cdfa8b9c905d8305e2186d966aec05bf1b53b60e) | Add semantic search (FAISS + Nomic embeddings) |
| 2025-03-28 | [`1c131a6`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/1c131a6156a6427c226573785dcab790edb7a696) | Fixes (MCP protocol, logging) |
| 2025-03-28 | [`e016725`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/e0167254841e13f4bfe40b19cd9531310695c88b) | Examples and example data |
| 2025-03-28 | [`6ea92ce`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/6ea92ce4fd14ae7c0f2f6ee54fb5fdd541019934) | Evolution, storage, project, cache, diagnostics |
| 2025-03-27 | [`5653506`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/56535063cf987c555d73c2ed144cd0d32a40e63d) | Output, plugins, templates, incremental processing, utilities |
| 2025-03-27 | [`8700377`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/8700377038e809032dedac8bbdbbc1d1d2440b84) | Initial commit |
