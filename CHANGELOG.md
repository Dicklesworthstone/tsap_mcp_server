# Changelog

All notable changes to the TSAP MCP Server are documented in this file.

This project has no tagged releases yet. All entries below correspond to individual commits on `main`, organized chronologically from newest to oldest. Each entry links to the exact commit on GitHub.

Repository: <https://github.com/Dicklesworthstone/tsap_mcp_server>

---

## 2026-02-22 -- License and Metadata Updates

### Documentation

- Update README license references to reflect MIT + OpenAI/Anthropic Rider
  ([`0da5a45`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/0da5a4548e8c22bfdc378a62dfe74cec72acb29a))

### Licensing

- Replace plain MIT license with MIT + OpenAI/Anthropic Rider, restricting use by OpenAI, Anthropic, and their affiliates without express written permission from Jeffrey Emanuel
  ([`16e02de`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/16e02de56c58c0b8dcca308b9cc10657e4a1135a))

### Repository Metadata

- Add GitHub social preview image (1280x640) for consistent social media link previews
  ([`c329c4b`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/c329c4b0a0ab10d30ace9e39063a0b3f06607369))

---

## 2026-02-16 -- Remove Cache Subsystem

### Refactoring (Breaking)

- **Remove the entire `tsap.cache` package** (2,977 lines across 5 files), which contained:
  - `CacheManager` with memory/disk backends and size-based eviction
  - `InvalidationStrategy` hierarchy (TTL, LRU, LFU, size-based, composite)
  - `CacheMetrics` with hit/miss/eviction tracking and reporting
  - `CachePersistence` with SQLite and filesystem serialization
- Rationale: over-engineered for current needs and carried significant maintenance surface without being actively utilized in the request pipeline
  ([`8d51198`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/8d511980bfa41dade357439f8e2598486dabaaee))

**Files removed:** `src/tsap/cache/__init__.py`, `invalidation.py`, `manager.py`, `metrics.py`, `persistence.py`

---

## 2025-04-07 -- Architecture Overhaul: MCP Adapter Layer and ToolAPI Extraction

This is the largest single commit in the project's history (218 files changed, 31,458 insertions, 2,122 deletions). It restructures the entire project from a monolithic `tsap.mcp` subpackage into a proper three-component architecture.

### Architecture (Breaking)

- **Introduce `tsap_mcp` as a standalone MCP adapter package** (`src/tsap_mcp/`), replacing the old `tsap.mcp` subpackage. The new package contains:
  - `server.py` -- MCP server implementation
  - `adapter.py` -- bridge between TSAP core and MCP protocol
  - `cli.py` -- command-line interface (`tsap-mcp run`, `tsap-mcp info`, `tsap-mcp install`)
  - `lifespan.py` -- server lifecycle management
  - `tools/` -- MCP tool implementations (search, processing, analysis, visualization, composite)
  - `resources/` -- MCP resource handlers (files, project, config, semantic, analysis, composite, processing, search, visualization)
  - `prompts/` -- MCP prompt templates (code analysis, composite, processing, search, visualization)
  - `adapters/` -- protocol adapters (tool adapters)
  - `utils/` -- utility modules (analysis, composite, context, processing, search, visualization)
- **Extract `toolapi` from `tsap.mcp`** -- rename and expand the old `tsap.mcp` internals into `tsap.toolapi`, a standalone internal API layer with:
  - `protocol.py` -- protocol definition (310 lines)
  - `handler.py` -- tool request handling (1,300 lines)
  - `models.py` -- data models (1,711 lines)
  - `client/base.py` -- client library (585 lines)
- Delete old `tsap.mcp` subpackage entirely (`__init__.py`, `handler.py`, `models.py`, `protocol.py`, `client/`)

### MCP Integration

- Add `mcp[cli]>=0.1.0` as a core dependency in `pyproject.toml`
- Add helper scripts: `run_mcp_server.py`, `run_dual_servers.py`, `run_parallel.py`, `add_mcp_dependency.py`, `check_mcp_completeness.py`
- Include full MCP protocol schema (`mcp_protocol_schema_2025-03-25_version.json`)

### Examples

- Create `mcp_examples/` directory with 33 dedicated MCP client examples, including:
  - All tool demos rewritten for MCP protocol (awk, jq, ripgrep, sqlite, pdf, html, table, patterns, structure search, semantic search, document profiler)
  - `mcp_stdio_server.py`, `mcp_proxy.py` -- server infrastructure examples
  - `debug_tools.py`, `verify_tool_call.py`, `simple_mcp_test.py` -- diagnostic utilities
  - `run_mcp_test.py`, `test_stdio_server.py` -- test harnesses
- Refactor existing `examples/` to use `toolapi` imports instead of old `mcp` imports
- Add `run_complete_battery_of_demonstration_scripts.py` with full output capture
- Add `output_of_complete_battery_of_demos.txt` (2,655 lines of demo output)

### Project Metadata

- Rename project from `tsap_mcp_server` to `tsap-mcp` in `pyproject.toml`
- Rewrite README with three-component architecture documentation
- Add `src/tsap/api/models/auth.py` for authentication models
- Remove scaffold script `create_project_structure.sh`
- Enable `spacy==3.7.2` in dependencies (previously commented out)
- Add `isort>=5.12.0` to dev dependencies

### Bug Fixes

- Broad import path fixes across all `src/tsap/` modules to resolve circular imports and broken references after the restructuring

([`5baff2d`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/5baff2d0bb0d641db493924112943a8605a21246))

---

## 2025-04-01 -- MCP Client Library, Structure Search, and Pattern Bootstrap

### New Features

- **MCP client library** (`src/tsap/mcp/client/base.py`, 596 lines) -- programmatic client for interacting with the MCP server
- **Bootstrap pattern system** (`src/tsap/composite/bootstrap_patterns.py`) -- seed the pattern library with initial high-quality patterns
- **Advanced document profiler demo** (`examples/advanced_document_profiler_demo.py`)
- **Advanced patterns demo** (`examples/advanced_patterns_demo.py`)
- **Advanced structure search demo** (`examples/advanced_structure_search_demo.py`)
- **LLM pattern evolution example** (`examples/llm_pattern_evolution.py`)
- **Log pattern evolution example** (`examples/log_pattern_evolution.py`)

### Improvements

- Major expansion of `src/tsap/mcp/handler.py` (+482 lines) -- more tool handlers registered
- Expand `src/tsap/server.py` (+356 lines) -- richer server capabilities
- Improve `src/tsap/composite/structure_search.py` -- better structural matching
- Improve `src/tsap/evolution/pattern_analyzer.py` -- deeper pattern analysis with LLM integration
- Add sample data files for structure search testing (Python, HTML, Markdown)
- Add sample data for document analysis (code samples, strategic thinking text, technical spec)

([`9ecd381`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/9ecd3815b55e57d8f00ee32e1c32f924bcc5e3c2))

---

## 2025-03-31 -- HTML Processor, SQLite Tool, and Entry Point

### New Features

- **HTML processor demo** (`examples/advanced_html_processor_demo.py`, 1,220 lines) -- comprehensive HTML parsing and extraction
- **SQLite demo** (`examples/advanced_sqlite_demo.py`, 960 lines) -- relational queries on extracted data
- **Application entry point** (`src/tsap/main.py`, 128 lines) -- unified main entry point

### Improvements

- Major expansion of `src/tsap/core/html_processor.py` (+482 lines) -- structure-aware HTML parsing
- Major expansion of `src/tsap/core/sqlite.py` (+364 lines) -- richer query capabilities
- Expand MCP handler with HTML and SQLite tool registrations
- Add HTML example data: SEC 10-Q filing, Wikipedia algebraic topology article
- Reorganize PDF example data into `tsap_example_data/pdfs/` subdirectory

### Tweaks

- Fix `src/tsap/cache/persistence.py` minor issues
- Fix `src/tsap/evolution/genetic.py` parameter handling

([`7b98219`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/7b98219a945d59accdff5c4d7365ba077fc00b8b),
[`f3581f6`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/f3581f6f403b2c0636df995c7bee83635efd54f8))

---

## 2025-03-30 -- PDF Extraction, Table Processing, AWK/JQ Demos

### Bug Fixes

- **Fix PDF extraction demo** -- correct PDF extractor tool invocation and MCP handler registration for PDF tools
  ([`52ecb3b`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/52ecb3b7e57ff389b047e17f5494830e3cd32d86))
- **Fix table processing** -- rewrite `src/tsap/core/table_processor.py` (374 lines changed), fix MCP model definitions for table operations
  ([`46e0217`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/46e021744df568e2b1c01f3bafd587f192af83d4))
- **Fix AWK demo** -- fix `src/tsap/core/awk.py` and `src/tsap/core/jq.py` tool implementations
  ([`832a6f9`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/832a6f97531fbaf0baad81cbbbfa214d93b3fc71))

### New Features

- **Advanced table processor demo** (`examples/advanced_table_processor_demo.py`, 602 lines)
- **Advanced AWK demo** (`examples/advanced_awk_demo.py`, 395 lines)
- **Advanced JQ demo** (`examples/advanced_jq_demo.py`, 445 lines)
- **Advanced PDF extractor demo** (`examples/advanced_pdf_extractor_demo.py`, 483 lines)

### Example Data

- Add CSV table data: `sales_data.csv`, `products.csv`, `inventory.csv`, `orders.tsv`, `sensor_data.json`
- Add CSV report data: `report_east.csv`, `report_west.csv`, `report_south.csv`
- Add JSON example data: `logs.jsonl`, `nested_data.json`, `users.json`
- Add PDF test documents: Durnovo presentation slides, Lamport Paxos paper

---

## 2025-03-29 -- Semantic Search, Server Hardening, Logging Fixes

### New Features

- **Semantic search** -- full vector search pipeline using FAISS and Nomic embeddings:
  - `src/tsap/core/semantic_search_tool.py` (372 lines) -- core semantic search with embedding generation, index management, and similarity search
  - `src/tsap/composite/semantic_search.py` (80 lines) -- composite semantic search operations
  - `examples/semantic_search_demo.py` (319 lines, later rewritten to 510 lines)
  - Add FAISS, Nomic, and sentence-transformers to `pyproject.toml` dependencies
  ([`cdfa8b9`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/cdfa8b9c905d8305e2186d966aec05bf1b53b60e))

### New Examples

- `examples/advanced_ripgrep_demo.py` (424 lines) -- comprehensive ripgrep usage patterns
- `examples/mcp_client_example.py` (249 lines) -- MCP client interaction example
- `examples/performance_modes.py` (193 lines) -- speed vs accuracy tradeoffs
- `examples/plugin_development.py` (459 lines) -- plugin authoring guide
- `examples/search_documents.py` (106 lines) -- document search patterns
- `examples/standalone_mcp_server.py` (77 lines) -- minimal server setup
- `examples/debug_server.py` (64 lines) -- server debugging utilities
- Add `uv.lock` for reproducible dependency resolution

### Improvements

- Expand `src/tsap/server.py` with semantic search server integration and better error handling (+179 lines)
- Refactor `src/tsap/composite/document_profiler.py` for more robust profiling
- Simplify API layer (`src/tsap/api/app.py`) and logging subsystem
- Fix logging formatter and console output modules

### Bug Fixes

- Fix MCP protocol model definitions
  ([`1c131a6`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/1c131a6156a6427c226573785dcab790edb7a696))
- Fix logging emoji handling and logger initialization
  ([`76ad93f`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/76ad93fafd492e559c6a225c0f88a871504fb5f3))
- Fix semantic search tool result handling
  ([`b28e7b4`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/b28e7b411f9aa2c8c0faec80840207f4d57578e2))

### Example Data

- Add `tsap_example_data/code/llm_tournament.py` (1,644 lines) -- LLM tournament simulation
- Add `tsap_example_data/documents/browser_use_docs.md` (1,217 lines)
- Add `tsap_example_data/documents/mcp_python_lib_docs.md` (649 lines)
- Add `tsap_example_data/documents/sample_input_document_as_markdown__durnovo_memo.md` (208 lines)

---

## 2025-03-28 -- Examples, CLI, and Core Implementations

### New Features

- **CLI** (`src/tsap/cli.py`, 957 lines) -- full command-line interface
- **Evolution system**:
  - `src/tsap/evolution/base.py` (781 lines) -- evolutionary algorithm base
  - `src/tsap/evolution/strategy_evolution.py` (3,055 lines) -- search strategy evolution
  - `src/tsap/evolution/strategy_journal.py` (1,108 lines) -- strategy effectiveness tracking
  - `src/tsap/evolution/offline_learning.py` (1,116 lines) -- offline pattern learning
  - `src/tsap/evolution/runtime_learning.py` (626 lines) -- real-time optimization
- **Storage layer**:
  - `src/tsap/storage/database.py` (991 lines) -- SQLite-backed persistence
  - `src/tsap/storage/history_store.py` (773 lines) -- search history
  - `src/tsap/storage/pattern_store.py` (1,250 lines) -- pattern storage
  - `src/tsap/storage/profile_store.py` (1,215 lines) -- profile storage
  - `src/tsap/storage/strategy_store.py` (1,313 lines) -- strategy storage
- **Project management**:
  - `src/tsap/project/context.py` (866 lines) -- project context tracking
  - `src/tsap/project/history.py` (979 lines) -- project history
  - `src/tsap/project/profile.py` (1,177 lines) -- project profiling
  - `src/tsap/project/transfer.py` (1,041 lines) -- cross-project transfer learning
- **Cache subsystem**:
  - `src/tsap/cache/metrics.py` (425 lines) -- cache hit/miss metrics
  - `src/tsap/cache/persistence.py` (1,205 lines) -- cache serialization
- **Composite operations**:
  - `src/tsap/composite/base.py` (417 lines) -- composite operation base
  - `src/tsap/composite/confidence.py` (1,006 lines) -- confidence scoring
  - `src/tsap/composite/patterns.py` (981 lines) -- pattern matching
  - `src/tsap/composite/refinement.py` (405 lines) -- iterative refinement
- **API routes**:
  - `src/tsap/api/routes/core.py` (880 lines) -- core API endpoints
  - `src/tsap/api/routes/evolution.py` (570 lines) -- evolution endpoints
  - `src/tsap/api/routes/plugins.py` (717 lines) -- plugin endpoints
  - `src/tsap/api/models/core.py` (273 lines), `evolution.py` (321 lines), `plugins.py` (221 lines)
- **Template system**:
  - `src/tsap/templates/corpus_exploration.py` (856 lines)
  - `src/tsap/templates/regulatory_analysis.py` (713 lines)
  - `src/tsap/templates/custom.py` (99 lines)
  - `src/tsap/templates/registry.py` (429 lines)
- **Diagnostics**:
  - `src/tsap/utils/diagnostics/analyzer.py` (584 lines) -- system diagnostics
  - `src/tsap/utils/diagnostics/profiler.py` (710 lines) -- performance profiling
  - `src/tsap/utils/diagnostics/reporter.py` (963 lines) -- diagnostic reports
  - `src/tsap/utils/diagnostics/visualizer.py` (1,403 lines) -- diagnostic visualization
- **Utilities**:
  - `src/tsap/utils/logging/dashboard.py` (867 lines) -- logging dashboard
  - `src/tsap/utils/logging/handler.py` (609 lines) -- log handlers
  - `src/tsap/utils/metrics.py` (329 lines) -- system metrics
  - `src/tsap/utils/optimization.py` (677 lines) -- performance optimization
  - `src/tsap/utils/security.py` (518 lines) -- security utilities
- **Incremental processing**:
  - `src/tsap/incremental/aggregator.py` (729 lines)
  - `src/tsap/incremental/processor.py` (545 lines)
  - `src/tsap/incremental/splitter.py` (686 lines)
  - `src/tsap/incremental/streamer.py` (831 lines)
- **Table processor** (`src/tsap/core/table_processor.py`, 382 lines)

### Examples

- Add working examples: `api_client.py`, `code_security_audit.py`, `document_dna_profiling.py`, `evolving_search.py`, `incremental_processing.py`
- Add example data: `tsap_example_data/code/main.py`, `utils.py`, `documents/report_v1.txt`, `report_v2.txt`

### Bug Fixes

- Fix MCP protocol and logging initialization
  ([`1c131a6`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/1c131a6156a6427c226573785dcab790edb7a696))

([`6ea92ce`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/6ea92ce4fd14ae7c0f2f6ee54fb5fdd541019934),
[`e016725`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/e0167254841e13f4bfe40b19cd9531310695c88b),
[`5653506`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/56535063cf987c555d73c2ed144cd0d32a40e63d))

---

## 2025-03-27 -- Initial Commit

### Project Foundation

- **Core tool wrappers** (`src/tsap/core/`):
  - `ripgrep.py` (455 lines) -- high-performance regex search via ripgrep
  - `awk.py` (313 lines) -- text transformation and field processing
  - `jq.py` (342 lines) -- JSON processing and transformation
  - `sqlite.py` (337 lines) -- relational queries on extracted data
  - `html_processor.py` (850 lines) -- structure-aware HTML parsing
  - `pdf_extractor.py` (804 lines) -- PDF text extraction for complex layouts
  - `process.py` (436 lines) -- subprocess management
  - `validation.py` (424 lines) -- input validation
  - `base.py` (219 lines) -- abstract base classes for all core tools
- **Analysis tools** (`src/tsap/analysis/`):
  - `code.py` (829 lines) -- code quality and structure analysis
  - `documents.py` (691 lines) -- document content analysis
  - `metadata.py` (516 lines) -- metadata extraction
  - `strategy_compiler.py` (529 lines) -- execution plan optimization
  - `llm.py` (368 lines) -- LLM integration for analysis
  - `resource_allocator.py` (319 lines) -- computational resource allocation
  - `counterfactual.py` (245 lines) -- missing/unusual content detection
  - `cartographer.py` (129 lines) -- document relationship mapping
- **Composite operations** (`src/tsap/composite/`):
  - `structure_search.py` (914 lines) -- structural position-based search
  - `context.py` (905 lines) -- meaningful code/text unit extraction
  - `document_profiler.py` (851 lines) -- document fingerprinting
  - `regex_generator.py` (589 lines) -- automatic regex generation
  - `filenames.py` (566 lines) -- filename convention discovery
  - `diff_generator.py` (512 lines) -- cross-version change detection
  - `structure.py` (641 lines) -- document structure analysis
  - `parallel.py` (390 lines) -- parallel multi-pattern search
- **Evolution engine** (`src/tsap/evolution/`):
  - `pattern_library.py` (1,719 lines) -- reusable pattern library
  - `genetic.py` (1,219 lines) -- genetic algorithm for strategy evolution
  - `pattern_analyzer.py` (1,101 lines) -- pattern effectiveness analysis
  - `metrics.py` (707 lines) -- evolution metrics
- **Cache subsystem** (`src/tsap/cache/`):
  - `manager.py` (592 lines) -- cache manager with memory/disk backends
  - `invalidation.py` (703 lines) -- TTL/LRU/LFU invalidation strategies
- **Configuration** (`src/tsap/config.py`, 367 lines) -- environment-variable-driven configuration
- **API layer** (`src/tsap/api/`) -- FastAPI-based REST API with middleware (auth, error handling, logging) and route stubs
- **Infrastructure**: Dockerfile (87 lines), docker-compose.yml, pyproject.toml (243 lines), .gitignore, .dockerignore
- **Documentation stubs**: README.md, docs/ directory with MkDocs structure, empty tutorial and reference files
- **Test directory structure**: empty test files for all subsystems

([`8700377`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/8700377038e809032dedac8bbdbbc1d1d2440b84))

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
| 2025-03-30 | [`832a6f9`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/832a6f97531fbaf0baad81cbbbfa214d93b3fc71) | Fix AWK/JQ demos |
| 2025-03-29 | [`b28e7b4`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/b28e7b411f9aa2c8c0faec80840207f4d57578e2) | Fixes and improvements (logging, API, server) |
| 2025-03-29 | [`76ad93f`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/76ad93fafd492e559c6a225c0f88a871504fb5f3) | Fixes (semantic search, server, logging) |
| 2025-03-29 | [`cdfa8b9`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/cdfa8b9c905d8305e2186d966aec05bf1b53b60e) | Add semantic search (FAISS + Nomic embeddings) |
| 2025-03-28 | [`1c131a6`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/1c131a6156a6427c226573785dcab790edb7a696) | Fixes (MCP protocol, logging) |
| 2025-03-28 | [`e016725`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/e0167254841e13f4bfe40b19cd9531310695c88b) | Examples and example data |
| 2025-03-28 | [`6ea92ce`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/6ea92ce4fd14ae7c0f2f6ee54fb5fdd541019934) | Evolution, storage, project, diagnostics, incremental processing |
| 2025-03-27 | [`5653506`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/56535063cf987c555d73c2ed144cd0d32a40e63d) | Output, plugins, templates, incremental processing, utilities |
| 2025-03-27 | [`8700377`](https://github.com/Dicklesworthstone/tsap_mcp_server/commit/8700377038e809032dedac8bbdbbc1d1d2440b84) | Initial commit |
