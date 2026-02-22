# TSAP MCP Server

A Model Context Protocol (MCP) implementation of Text Search and Analysis Processing (TSAP) server for code intelligence and text analysis.

## Table of Contents

- [TSAP MCP Server](#tsap-mcp-server)
  - [Overview](#overview)
  - [Project Components](#project-components)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Architecture](#architecture)
  - [Development](#development)
  - [Contributing](#contributing)
  - [License](#license)

## Overview

TSAP MCP Server is an implementation of the Model Context Protocol (MCP) standard that provides advanced text search, processing, and analysis capabilities. It operates as a standalone server exposing a wide range of functionality through the MCP interface, making it compatible with Claude Desktop and other MCP-compatible clients.

Built on the MCP Python SDK and utilizing a diverse set of specialized libraries, this server empowers AI assistants with rich capabilities for working with text, code, and document analysis through standardized interfaces.

## Project Components

This project consists of three major components that work together:

1. **`tsap`**: The core Text Search and Analysis Processing system providing fundamental capabilities for:
   - Code and text search
   - Document analysis
   - Pattern extraction
   - Data processing
   - Evolutionary algorithms
   - Storage and caching
   
   This component implements the core functionality and business logic of the system.

2. **`toolapi`**: An internal API system within `tsap` that:
   - Defines the tool interaction protocol
   - Handles tool request/response lifecycle
   - Manages tool registration and discovery
   - Provides client libraries for tool consumption
   
   The `toolapi` serves as the original API layer for programmatic access to TSAP capabilities.

3. **`tsap_mcp`**: The Model Context Protocol adapter layer that:
   - Wraps the core TSAP functionality in MCP-compatible interfaces
   - Maps TSAP tools to MCP tool functions
   - Implements resource handlers for MCP URI patterns
   - Provides MCP prompt templates
   - Serves as a bridge between TSAP and Claude Desktop (or other MCP clients)

Together, these components create a powerful system where the core TSAP functionality is exposed through a standardized MCP interface, making it seamlessly integratable with Claude Desktop and other AI assistants that support the Model Context Protocol.

## Features

- **MCP Protocol Implementation**: Fully compliant with the Model Context Protocol standard for AI tooling.

- **Search Capabilities**:
  - Text search with regex support, case sensitivity options, and path filtering
  - Code search using ripgrep with pattern matching and file filtering
  - Semantic search using embedding models from Nomic and sentence-transformers

- **Processing Tools**:
  - Text normalization and transformation
  - Pattern-based data extraction
  - Document structure analysis 

- **Analysis Tools**:
  - Code quality and structure analysis
  - Text content analysis
  - Document profiling and metadata extraction

- **Visualization Tools**:
  - Data visualization using matplotlib
  - Network visualization with networkx

- **Resource Access**:
  - File content access with path-based retrieval
  - Project structure information
  - Configuration management
  - Semantic corpus management

- **Claude Desktop Integration**:
  - Seamless integration with Claude Desktop through the MCP protocol

## Installation

### Prerequisites

- Python 3.13 or higher
- ripgrep installed on your system (for code search functionality)

### From Source

```bash
# Clone the repository
git clone https://github.com/your-org/tsap-mcp.git
cd tsap-mcp

# Install the package
pip install -e .

# For all optional dependencies (visualization, semantic search, etc.)
pip install -e ".[all]"
```

### Claude Desktop Integration

To install the server for use with Claude Desktop:

```bash
tsap-mcp install --desktop
```

This will register the server with Claude Desktop, making it available for use in conversations.

## Usage

### Running the Server

```bash
# Run with default settings (localhost:8000)
tsap-mcp run

# Specify host and port
tsap-mcp run --host 0.0.0.0 --port 9000

# Enable auto-reload for development
tsap-mcp run --reload

# Disable adapter layer
tsap-mcp run --no-adapter
```

### Server Information

```bash
# Show basic server information
tsap-mcp info

# Show all registered components (tools, resources, prompts)
tsap-mcp info --components
```

### Command-Line Options

The server supports various command-line options:

```
tsap-mcp run [-H HOST] [-p PORT] [--reload] [--no-adapter]
tsap-mcp info [--components]
tsap-mcp test [--compatibility]
tsap-mcp install [--desktop] [--system]
```

### Environment Variables

The server behavior can be configured using the following environment variables:

- `TSAP_HOST`: Host to bind (default: 127.0.0.1)
- `TSAP_PORT`: Port to bind (default: 8000)
- `TSAP_LOG_LEVEL`: Logging level (default: INFO)
- `TSAP_PERFORMANCE_MODE`: Performance mode (balanced, speed, accuracy)
- `TSAP_CACHE_ENABLED`: Enable cache (default: true)
- `TSAP_DEBUG`: Enable debug mode (default: false)

## Architecture

TSAP MCP Server follows a modular architecture organized around the core components:

```
src/
├── tsap/                  # Core TSAP implementation
│   ├── __init__.py        # Package initialization
│   ├── server.py          # Main server implementation
│   ├── cli.py             # Command-line interface
│   ├── config.py          # Configuration management
│   ├── main.py            # Main entry point
│   ├── api/               # API layer 
│   ├── analysis/          # Text and code analysis
│   ├── core/              # Core functionality
│   ├── project/           # Project management
│   ├── storage/           # Storage utilities
│   ├── utils/             # Utility functions
│   ├── toolapi/           # Tool API protocol implementation
│   │   ├── __init__.py    # Package initialization
│   │   ├── handler.py     # Tool request handling
│   │   ├── models.py      # Data models
│   │   ├── protocol.py    # Protocol definition
│   │   └── client/        # Client libraries
│   └── ...                # Other TSAP components
│
├── tsap_mcp/              # MCP adapter layer
│   ├── __init__.py        # Package initialization
│   ├── server.py          # MCP server implementation
│   ├── cli.py             # Command-line interface
│   ├── adapter.py         # Adapter for original TSAP
│   ├── tool_map.py        # Tool mapping utilities
│   ├── lifespan.py        # Server lifecycle management
│   ├── tools/             # MCP tool implementations
│   │   ├── search.py      # Search tools
│   │   ├── processing.py  # Text processing tools
│   │   ├── analysis.py    # Analysis tools
│   │   ├── visualization.py # Visualization tools
│   │   └── composite.py   # Composite tools
│   ├── resources/         # MCP resource implementations
│   │   ├── files.py       # File access resources
│   │   ├── project.py     # Project structure resources
│   │   ├── config.py      # Configuration resources
│   │   └── semantic.py    # Semantic corpus resources
│   ├── prompts/           # MCP prompt implementations
│   └── adapters/          # Protocol adapters
│
├── scripts/               # Utility scripts
└── tests/                 # Test suite
```

### Data Flow

The data flow in the system follows this pattern:

1. An MCP client (such as Claude Desktop) makes a request to the `tsap_mcp` server
2. The `tsap_mcp` adapter layer translates the MCP request to the appropriate `tsap` operation
3. The `tsap` core system processes the request, potentially using its `toolapi` subsystem
4. Results are transformed back into MCP-compatible responses by `tsap_mcp`
5. The MCP client receives and processes the response

### MCP Components

The server implements three MCP component types:

1. **Tools**: Functions that can be called to perform specific actions
2. **Resources**: Data access patterns through URI-based interfaces
3. **Prompts**: Reusable templates for common interactions

## Development

### Development Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-org/tsap-mcp.git
cd tsap-mcp

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

### Running in Development Mode

```bash
# Run with auto-reload
tsap-mcp run --reload
```

### MCP Inspector

For development and debugging, you can use the MCP Inspector:

```bash
pip install mcp[cli]
mcp dev src/tsap_mcp/server.py
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License (with OpenAI/Anthropic Rider) - see the [LICENSE](LICENSE) file for details.
