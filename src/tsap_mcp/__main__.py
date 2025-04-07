#!/usr/bin/env python3
"""
Main entry point for the TSAP MCP server package.

This module allows running the package as a module:
python -m tsap_mcp
"""
import sys
from tsap_mcp.cli import main

if __name__ == "__main__":
    sys.exit(main()) 