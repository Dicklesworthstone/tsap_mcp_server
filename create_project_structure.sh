#!/bin/bash

# TSAP MCP Server - Project Structure Generator
# This script creates the entire folder and file structure for the project

set -e  # Exit immediately if a command exits with a non-zero status

# Print with colors for better user experience
print_step() {
  echo -e "\033[1;34m=> $1\033[0m"
}

print_success() {
  echo -e "\033[1;32m✓ $1\033[0m"
}

# Main directories creation
create_directories() {
  print_step "Creating main directory structure..."
  
  # GitHub workflows
  mkdir -p .github/workflows
  
  # Source code structure
  mkdir -p src/tsap/{core,composite,analysis,evolution,plugins/builtin,templates,cache,incremental,output,project,api/routes,api/models,api/middleware,mcp,storage,utils/logging,utils/diagnostics}
  
  # Tests structure
  mkdir -p tests/{test_fixtures/{code_samples,documents,pdf_samples,html_samples,mock_responses},test_core,test_composite,test_analysis,test_evolution,test_plugins,test_templates,test_cache,test_incremental,test_output,test_project,test_storage,test_api/test_routes,test_api/test_middleware,test_mcp,test_utils/test_logging,test_utils/test_diagnostics}
  
  # Examples and notebooks
  mkdir -p examples/notebooks
  
  # Documentation
  mkdir -p docs/{tutorials,reference,development}
  
  # Scripts
  mkdir -p scripts
  
  print_success "Directory structure created"
}

# Create empty files
create_files() {
  print_step "Creating project files..."
  
  # Root level files
  touch .gitignore
  touch LICENSE
  touch README.md
  touch pyproject.toml
  touch .pre-commit-config.yaml
  touch .ruff.toml
  touch .env.example
  touch Dockerfile
  touch docker-compose.yml
  touch .dockerignore
  
  # GitHub workflows
  touch .github/workflows/ci.yml
  touch .github/workflows/release.yml
  touch .github/workflows/dependency-review.yml
  
  # Scripts
  touch scripts/install_dependencies.sh
  touch scripts/setup_dev.sh
  touch scripts/update_docs.py
  touch scripts/generate_openapi.py
  touch scripts/create_plugin.py
  
  print_success "Base project files created"
  
  print_step "Creating source files..."
  
  # Main package files
  touch src/tsap/__init__.py
  touch src/tsap/__main__.py
  touch src/tsap/cli.py
  touch src/tsap/config.py
  touch src/tsap/version.py
  touch src/tsap/constants.py
  touch src/tsap/server.py
  touch src/tsap/dependencies.py
  touch src/tsap/performance_mode.py
  
  # Core module
  touch src/tsap/core/__init__.py
  touch src/tsap/core/base.py
  touch src/tsap/core/ripgrep.py
  touch src/tsap/core/awk.py
  touch src/tsap/core/jq.py
  touch src/tsap/core/sqlite.py
  touch src/tsap/core/process.py
  touch src/tsap/core/validation.py
  touch src/tsap/core/html_processor.py
  touch src/tsap/core/pdf_extractor.py
  touch src/tsap/core/table_processor.py
  
  # Composite module
  touch src/tsap/composite/__init__.py
  touch src/tsap/composite/base.py
  touch src/tsap/composite/parallel.py
  touch src/tsap/composite/refinement.py
  touch src/tsap/composite/context.py
  touch src/tsap/composite/patterns.py
  touch src/tsap/composite/filenames.py
  touch src/tsap/composite/structure.py
  touch src/tsap/composite/structure_search.py
  touch src/tsap/composite/diff_generator.py
  touch src/tsap/composite/regex_generator.py
  touch src/tsap/composite/document_profiler.py
  touch src/tsap/composite/confidence.py
  
  # Analysis module
  touch src/tsap/analysis/__init__.py
  touch src/tsap/analysis/base.py
  touch src/tsap/analysis/code.py
  touch src/tsap/analysis/documents.py
  touch src/tsap/analysis/metadata.py
  touch src/tsap/analysis/cartographer.py
  touch src/tsap/analysis/llm.py
  touch src/tsap/analysis/counterfactual.py
  touch src/tsap/analysis/strategy_compiler.py
  touch src/tsap/analysis/resource_allocator.py
  
  # Evolution module
  touch src/tsap/evolution/__init__.py
  touch src/tsap/evolution/base.py
  touch src/tsap/evolution/strategy_evolution.py
  touch src/tsap/evolution/pattern_analyzer.py
  touch src/tsap/evolution/strategy_journal.py
  touch src/tsap/evolution/pattern_library.py
  touch src/tsap/evolution/genetic.py
  touch src/tsap/evolution/metrics.py
  touch src/tsap/evolution/runtime_learning.py
  touch src/tsap/evolution/offline_learning.py
  
  # Plugins module
  touch src/tsap/plugins/__init__.py
  touch src/tsap/plugins/registry.py
  touch src/tsap/plugins/manager.py
  touch src/tsap/plugins/loader.py
  touch src/tsap/plugins/interface.py
  touch src/tsap/plugins/builtin/__init__.py
  touch src/tsap/plugins/builtin/example.py
  
  # Templates module
  touch src/tsap/templates/__init__.py
  touch src/tsap/templates/base.py
  touch src/tsap/templates/registry.py
  touch src/tsap/templates/security_audit.py
  touch src/tsap/templates/regulatory_analysis.py
  touch src/tsap/templates/corpus_exploration.py
  touch src/tsap/templates/custom.py
  
  # Cache module
  touch src/tsap/cache/__init__.py
  touch src/tsap/cache/manager.py
  touch src/tsap/cache/invalidation.py
  touch src/tsap/cache/persistence.py
  touch src/tsap/cache/metrics.py
  
  # Incremental module
  touch src/tsap/incremental/__init__.py
  touch src/tsap/incremental/processor.py
  touch src/tsap/incremental/splitter.py
  touch src/tsap/incremental/aggregator.py
  touch src/tsap/incremental/streamer.py
  
  # Output module
  touch src/tsap/output/__init__.py
  touch src/tsap/output/formatter.py
  touch src/tsap/output/json_output.py
  touch src/tsap/output/csv_output.py
  touch src/tsap/output/terminal.py
  touch src/tsap/output/visualization.py
  touch src/tsap/output/reporting.py
  
  # Project module
  touch src/tsap/project/__init__.py
  touch src/tsap/project/profile.py
  touch src/tsap/project/context.py
  touch src/tsap/project/transfer.py
  touch src/tsap/project/history.py
  
  # API module
  touch src/tsap/api/__init__.py
  touch src/tsap/api/app.py
  touch src/tsap/api/dependencies.py
  touch src/tsap/api/routes/__init__.py
  touch src/tsap/api/routes/core.py
  touch src/tsap/api/routes/composite.py
  touch src/tsap/api/routes/analysis.py
  touch src/tsap/api/routes/evolution.py
  touch src/tsap/api/routes/plugins.py
  touch src/tsap/api/models/__init__.py
  touch src/tsap/api/models/core.py
  touch src/tsap/api/models/composite.py
  touch src/tsap/api/models/analysis.py
  touch src/tsap/api/models/evolution.py
  touch src/tsap/api/models/plugins.py
  touch src/tsap/api/middleware/__init__.py
  touch src/tsap/api/middleware/auth.py
  touch src/tsap/api/middleware/logging.py
  touch src/tsap/api/middleware/error.py
  
  # MCP module
  touch src/tsap/mcp/__init__.py
  touch src/tsap/mcp/protocol.py
  touch src/tsap/mcp/handler.py
  touch src/tsap/mcp/models.py
  
  # Storage module
  touch src/tsap/storage/__init__.py
  touch src/tsap/storage/database.py
  touch src/tsap/storage/pattern_store.py
  touch src/tsap/storage/profile_store.py
  touch src/tsap/storage/history_store.py
  touch src/tsap/storage/strategy_store.py
  
  # Utils module
  touch src/tsap/utils/__init__.py
  touch src/tsap/utils/errors.py
  touch src/tsap/utils/helpers.py
  touch src/tsap/utils/security.py
  touch src/tsap/utils/async_utils.py
  touch src/tsap/utils/filesystem.py
  touch src/tsap/utils/caching.py
  touch src/tsap/utils/metrics.py
  touch src/tsap/utils/optimization.py
  
  # Logging module
  touch src/tsap/utils/logging/__init__.py
  touch src/tsap/utils/logging/formatter.py
  touch src/tsap/utils/logging/handler.py
  touch src/tsap/utils/logging/logger.py
  touch src/tsap/utils/logging/console.py
  touch src/tsap/utils/logging/progress.py
  touch src/tsap/utils/logging/panels.py
  touch src/tsap/utils/logging/themes.py
  touch src/tsap/utils/logging/emojis.py
  touch src/tsap/utils/logging/dashboard.py
  
  # Diagnostics module
  touch src/tsap/utils/diagnostics/__init__.py
  touch src/tsap/utils/diagnostics/profiler.py
  touch src/tsap/utils/diagnostics/analyzer.py
  touch src/tsap/utils/diagnostics/reporter.py
  touch src/tsap/utils/diagnostics/visualizer.py
  
  print_success "Source files created"
  
  print_step "Creating test files..."
  
  # Test initialization
  touch tests/__init__.py
  touch tests/conftest.py
  
  # Test fixtures
  touch tests/test_fixtures/__init__.py
  
  # Core tests
  touch tests/test_core/__init__.py
  touch tests/test_core/test_ripgrep.py
  touch tests/test_core/test_awk.py
  touch tests/test_core/test_jq.py
  touch tests/test_core/test_sqlite.py
  touch tests/test_core/test_process.py
  touch tests/test_core/test_html_processor.py
  touch tests/test_core/test_pdf_extractor.py
  touch tests/test_core/test_table_processor.py
  
  # Composite tests
  touch tests/test_composite/__init__.py
  touch tests/test_composite/test_parallel.py
  touch tests/test_composite/test_refinement.py
  touch tests/test_composite/test_context.py
  touch tests/test_composite/test_patterns.py
  touch tests/test_composite/test_filenames.py
  touch tests/test_composite/test_structure.py
  touch tests/test_composite/test_structure_search.py
  touch tests/test_composite/test_diff_generator.py
  touch tests/test_composite/test_regex_generator.py
  touch tests/test_composite/test_document_profiler.py
  touch tests/test_composite/test_confidence.py
  
  # Analysis tests
  touch tests/test_analysis/__init__.py
  touch tests/test_analysis/test_code.py
  touch tests/test_analysis/test_documents.py
  touch tests/test_analysis/test_metadata.py
  touch tests/test_analysis/test_cartographer.py
  touch tests/test_analysis/test_llm.py
  touch tests/test_analysis/test_counterfactual.py
  touch tests/test_analysis/test_strategy_compiler.py
  touch tests/test_analysis/test_resource_allocator.py
  
  # Evolution tests
  touch tests/test_evolution/__init__.py
  touch tests/test_evolution/test_strategy_evolution.py
  touch tests/test_evolution/test_pattern_analyzer.py
  touch tests/test_evolution/test_strategy_journal.py
  touch tests/test_evolution/test_pattern_library.py
  touch tests/test_evolution/test_genetic.py
  touch tests/test_evolution/test_metrics.py
  touch tests/test_evolution/test_runtime_learning.py
  touch tests/test_evolution/test_offline_learning.py
  
  # Plugin tests
  touch tests/test_plugins/__init__.py
  touch tests/test_plugins/test_registry.py
  touch tests/test_plugins/test_manager.py
  touch tests/test_plugins/test_loader.py
  touch tests/test_plugins/test_interface.py
  
  # Template tests
  touch tests/test_templates/__init__.py
  touch tests/test_templates/test_registry.py
  touch tests/test_templates/test_security_audit.py
  touch tests/test_templates/test_regulatory_analysis.py
  touch tests/test_templates/test_custom.py
  
  # Cache tests
  touch tests/test_cache/__init__.py
  touch tests/test_cache/test_manager.py
  touch tests/test_cache/test_invalidation.py
  touch tests/test_cache/test_metrics.py
  
  # Incremental tests
  touch tests/test_incremental/__init__.py
  touch tests/test_incremental/test_processor.py
  touch tests/test_incremental/test_splitter.py
  touch tests/test_incremental/test_aggregator.py
  
  # Output tests
  touch tests/test_output/__init__.py
  touch tests/test_output/test_formatter.py
  touch tests/test_output/test_json_output.py
  touch tests/test_output/test_terminal.py
  touch tests/test_output/test_visualization.py
  
  # Project tests
  touch tests/test_project/__init__.py
  touch tests/test_project/test_profile.py
  touch tests/test_project/test_context.py
  touch tests/test_project/test_transfer.py
  
  # Storage tests
  touch tests/test_storage/__init__.py
  touch tests/test_storage/test_database.py
  touch tests/test_storage/test_pattern_store.py
  touch tests/test_storage/test_profile_store.py
  touch tests/test_storage/test_history_store.py
  touch tests/test_storage/test_strategy_store.py
  
  # API tests
  touch tests/test_api/__init__.py
  touch tests/test_api/test_app.py
  touch tests/test_api/test_routes/__init__.py
  touch tests/test_api/test_routes/test_core.py
  touch tests/test_api/test_routes/test_composite.py
  touch tests/test_api/test_routes/test_analysis.py
  touch tests/test_api/test_routes/test_evolution.py
  touch tests/test_api/test_routes/test_plugins.py
  touch tests/test_api/test_middleware/__init__.py
  touch tests/test_api/test_middleware/test_auth.py
  touch tests/test_api/test_middleware/test_error.py
  
  # MCP tests
  touch tests/test_mcp/__init__.py
  touch tests/test_mcp/test_protocol.py
  touch tests/test_mcp/test_handler.py
  
  # Utils tests
  touch tests/test_utils/__init__.py
  touch tests/test_utils/test_helpers.py
  touch tests/test_utils/test_filesystem.py
  touch tests/test_utils/test_optimization.py
  
  # Logging tests
  touch tests/test_utils/test_logging/__init__.py
  touch tests/test_utils/test_logging/test_logger.py
  touch tests/test_utils/test_logging/test_console.py
  touch tests/test_utils/test_logging/test_progress.py
  touch tests/test_utils/test_logging/test_dashboard.py
  
  # Diagnostics tests
  touch tests/test_utils/test_diagnostics/__init__.py
  touch tests/test_utils/test_diagnostics/test_profiler.py
  touch tests/test_utils/test_diagnostics/test_analyzer.py
  touch tests/test_utils/test_diagnostics/test_reporter.py
  
  print_success "Test files created"
  
  print_step "Creating examples and documentation..."
  
  # Examples
  touch examples/code_security_audit.py
  touch examples/sec_filing_analysis.py
  touch examples/api_client.py
  touch examples/document_dna_profiling.py
  touch examples/evolving_search.py
  touch examples/regulatory_changes.py
  touch examples/plugin_development.py
  touch examples/performance_modes.py
  touch examples/incremental_processing.py
  touch examples/project_profiles.py
  
  # Example notebooks
  touch examples/notebooks/code_analysis_example.ipynb
  touch examples/notebooks/document_analysis_example.ipynb
  touch examples/notebooks/pdf_extraction_example.ipynb
  touch examples/notebooks/pattern_evolution_example.ipynb
  touch examples/notebooks/document_comparison_example.ipynb
  touch examples/notebooks/plugin_tutorial.ipynb
  touch examples/notebooks/performance_tuning.ipynb
  touch examples/notebooks/rich_logging_visualization.ipynb
  
  # Documentation
  touch docs/index.md
  touch docs/installation.md
  touch docs/api.md
  touch docs/architecture.md
  touch docs/deployment.md
  touch docs/performance_modes.md
  touch docs/plugin_system.md
  
  # Documentation tutorials
  touch docs/tutorials/getting_started.md
  touch docs/tutorials/code_analysis.md
  touch docs/tutorials/document_analysis.md
  touch docs/tutorials/pattern_evolution.md
  touch docs/tutorials/document_intelligence.md
  touch docs/tutorials/pdf_processing.md
  touch docs/tutorials/creating_plugins.md
  touch docs/tutorials/task_templates.md
  touch docs/tutorials/project_profiles.md
  touch docs/tutorials/rich_logging.md
  
  # Documentation reference
  touch docs/reference/core_tools.md
  touch docs/reference/composite_operations.md
  touch docs/reference/analysis_tools.md
  touch docs/reference/evolutionary_systems.md
  touch docs/reference/plugin_interfaces.md
  touch docs/reference/task_templates.md
  touch docs/reference/api_reference.md
  
  # Documentation development
  touch docs/development/contributing.md
  touch docs/development/testing.md
  touch docs/development/plugins.md
  touch docs/development/performance_tuning.md
  touch docs/development/releasing.md
  
  # MkDocs configuration
  touch docs/_mkdocs.yml
  
  print_success "Examples and documentation created"
}

# Make the necessary directories executable
make_executable() {
  print_step "Setting executable permissions on scripts..."
  
  chmod +x scripts/install_dependencies.sh
  chmod +x scripts/setup_dev.sh
  chmod +x scripts/update_docs.py
  chmod +x scripts/generate_openapi.py
  chmod +x scripts/create_plugin.py
  chmod +x src/tsap/__main__.py
  
  print_success "Executable permissions set"
}

# Main function
main() {
  echo "===================================================================="
  echo "   TSAP MCP Server - Project Structure Generator"
  echo "===================================================================="
  echo ""
  echo "This script will create the entire project structure in the current"
  echo "directory ($(pwd)). Make sure you're in the 'tsap_mcp_server' directory."
  echo ""
  read -p "Continue? (y/n): " -n 1 -r
  echo ""
  
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Operation canceled."
    exit 1
  fi
  
  # Create everything
  create_directories
  create_files
  make_executable
  
  echo ""
  echo "===================================================================="
  print_success "Project structure created successfully!"
  echo "Next steps:"
  echo "1. Initialize git repository: 'git init'"
  echo "2. Install dependencies: 'bash scripts/install_dependencies.sh'"
  echo "3. Create virtual environment: 'uv venv --python=3.13 .venv'"
  echo "4. Activate virtual environment: 'source .venv/bin/activate'"
  echo "5. Install project in development mode: 'uv pip install -e \".[dev]\"'"
  echo "===================================================================="
}

# Run the main function
main
