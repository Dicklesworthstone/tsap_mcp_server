#!/usr/bin/env python3
"""
Command-line interface for the TSAP MCP server.

This module provides a CLI for running the TSAP MCP server with
various configuration options.
"""
import os
import sys
import logging
import argparse
import importlib.metadata
from pathlib import Path
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=os.environ.get("TSAP_LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("tsap_mcp.cli")


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set up logging based on verbosity
    if args.verbose:
        logging.getLogger("tsap_mcp").setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Run the appropriate command
    if args.command == "run":
        run_server(args)
    elif args.command == "info":
        show_server_info(args)
    elif args.command == "test":
        run_tests(args)
    elif args.command == "install":
        install_server(args)
    else:
        parser.print_help()


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="TSAP MCP Server - Text Search and Analysis Processing with Model Context Protocol",
    )
    
    # Get version
    try:
        version = importlib.metadata.version("tsap-mcp")
    except importlib.metadata.PackageNotFoundError:
        version = "0.1.0"  # Development version
    
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--version", action="version",
        version=f"TSAP MCP Server {version}"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run the MCP server")
    run_parser.add_argument(
        "-H", "--host", type=str, default=os.environ.get("TSAP_HOST", "127.0.0.1"),
        help="Host to bind the server to (default: 127.0.0.1 or TSAP_HOST env var)"
    )
    run_parser.add_argument(
        "-p", "--port", type=int, default=int(os.environ.get("TSAP_PORT", "8000")),
        help="Port to bind the server to (default: 8000 or TSAP_PORT env var)"
    )
    run_parser.add_argument(
        "--reload", action="store_true",
        help="Enable auto-reload for development"
    )
    run_parser.add_argument(
        "--no-adapter", action="store_true",
        help="Disable the adapter layer for the original TSAP implementation"
    )
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show server information")
    info_parser.add_argument(
        "--components", action="store_true",
        help="Show registered components (tools, resources, prompts)"
    )
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run server tests")
    test_parser.add_argument(
        "--compatibility", action="store_true",
        help="Run compatibility tests with original implementation"
    )
    
    # Install command
    install_parser = subparsers.add_parser("install", help="Install the server")
    install_parser.add_argument(
        "--desktop", action="store_true",
        help="Install for Claude Desktop"
    )
    install_parser.add_argument(
        "--system", action="store_true",
        help="Install system-wide (requires admin/root permissions)"
    )
    
    return parser


def run_server(args: argparse.Namespace) -> None:
    """Run the MCP server.
    
    Args:
        args: Parsed command line arguments
    """
    logger.info(f"Starting TSAP MCP Server on {args.host}:{args.port}")
    
    # Import the server module
    try:
        from tsap_mcp.server import mcp
    except ImportError as e:
        logger.error(f"Failed to import server module: {e}")
        sys.exit(1)
    
    # Set up adapter if requested
    if not args.no_adapter:
        try:
            logger.info("Initializing adapter layer for original TSAP implementation")
            from tsap_mcp.adapter import initialize_adapter
            initialize_adapter(mcp)
        except ImportError:
            logger.warning("Original TSAP implementation not found, adapter disabled")
        except Exception as e:
            logger.error(f"Failed to initialize adapter: {e}")
    
    # Run the server
    try:
        # Import asyncio and uvicorn
        import asyncio
        import uvicorn
        
        # Configure uvicorn server
        config = uvicorn.Config(
            app=mcp.get_app(),
            host=args.host,
            port=args.port,
            log_level="debug" if args.verbose else "info",
            reload=args.reload,
        )
        
        # Run the server
        server = uvicorn.Server(config)
        asyncio.run(server.serve())
        
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        logger.error("Please install with: pip install uvicorn[standard]")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)


def show_server_info(args: argparse.Namespace) -> None:
    """Show server information.
    
    Args:
        args: Parsed command line arguments
    """
    logger.info("Gathering TSAP MCP Server information")
    
    # Import the server module
    try:
        from tsap_mcp.server import mcp
    except ImportError as e:
        logger.error(f"Failed to import server module: {e}")
        sys.exit(1)
    
    # Basic server info
    try:
        version = importlib.metadata.version("tsap-mcp")
    except importlib.metadata.PackageNotFoundError:
        version = "0.1.0"  # Development version
    
    print(f"TSAP MCP Server {version}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"MCP SDK: {importlib.metadata.version('mcp')}")
    
    # Show registered components
    if args.components:
        tools = mcp.list_tool_functions()
        print(f"\nRegistered tools: {len(tools)}")
        for tool in tools:
            print(f"  - {tool.__name__}")
        
        resource_pattern_methods = [m for m in dir(mcp) if callable(getattr(mcp, m)) and hasattr(getattr(mcp, m), "__mcp_resource__")]
        print(f"\nRegistered resources: {len(resource_pattern_methods)}")
        for method in resource_pattern_methods:
            resource = getattr(mcp, method)
            if hasattr(resource, "__mcp_resource__"):
                pattern = getattr(resource, "__mcp_resource__", "unknown")
                print(f"  - {pattern}")
        
        prompt_methods = [m for m in dir(mcp) if callable(getattr(mcp, m)) and hasattr(getattr(mcp, m), "__mcp_prompt__")]
        print(f"\nRegistered prompts: {len(prompt_methods)}")
        for method in prompt_methods:
            prompt = getattr(mcp, method)
            if hasattr(prompt, "__mcp_prompt__"):
                name = prompt.__name__
                print(f"  - {name}")


def run_tests(args: argparse.Namespace) -> None:
    """Run server tests.
    
    Args:
        args: Parsed command line arguments
    """
    logger.info("Running TSAP MCP Server tests")
    
    try:
        # Run MCP integration tests
        from tsap_mcp.tests import run_tests as run_mcp_tests
        run_mcp_tests()
        
        # Run compatibility tests if requested
        if args.compatibility:
            try:
                from tsap_mcp.tests import run_compatibility_tests
                run_compatibility_tests()
            except ImportError:
                logger.error("Original TSAP implementation not found, compatibility tests skipped")
            except Exception as e:
                logger.error(f"Compatibility tests failed: {e}")
                sys.exit(1)
                
    except ImportError as e:
        logger.error(f"Failed to import test modules: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Tests failed: {e}")
        sys.exit(1)


def install_server(args: argparse.Namespace) -> None:
    """Install the server.
    
    Args:
        args: Parsed command line arguments
    """
    logger.info("Installing TSAP MCP Server")
    
    if args.desktop:
        # Install for Claude Desktop
        try:
            install_for_claude_desktop()
        except Exception as e:
            logger.error(f"Failed to install for Claude Desktop: {e}")
            sys.exit(1)
    elif args.system:
        # System-wide installation
        try:
            install_system_wide()
        except Exception as e:
            logger.error(f"Failed to install system-wide: {e}")
            sys.exit(1)
    else:
        # Regular installation
        try:
            install_regular()
        except Exception as e:
            logger.error(f"Failed to install: {e}")
            sys.exit(1)


def install_for_claude_desktop() -> None:
    """Install the server for Claude Desktop."""
    import json
    import shutil
    import subprocess
    
    logger.info("Installing TSAP MCP Server for Claude Desktop")
    
    # Get Claude Desktop MCP directory
    home_dir = Path.home()
    
    if sys.platform == "win32":
        claude_mcp_dir = home_dir / "AppData" / "Local" / "Claude" / "mcps"
    elif sys.platform == "darwin":
        claude_mcp_dir = home_dir / "Library" / "Application Support" / "Claude" / "mcps"
    else:  # Linux and others
        claude_mcp_dir = home_dir / ".config" / "Claude" / "mcps"
    
    # Create directory if it doesn't exist
    claude_mcp_dir.mkdir(parents=True, exist_ok=True)
    
    # Get package directory
    try:
        package_dir = Path(importlib.metadata.files("tsap-mcp")[0].locate()).parent
    except (importlib.metadata.PackageNotFoundError, IndexError):
        # Development mode, use current source directory
        package_dir = Path(__file__).parent.parent
    
    # Create manifest
    manifest = {
        "name": "TSAP MCP Server",
        "id": "tsap-mcp",
        "description": "Text Search and Analysis Processing with Model Context Protocol",
        "version": "0.1.0",
        "command": [sys.executable, "-m", "tsap_mcp.cli", "run"],
        "mcp_schema_version": "0.1.0",
    }
    
    # Write manifest
    manifest_path = claude_mcp_dir / "tsap-mcp.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Installed manifest at {manifest_path}")
    logger.info("TSAP MCP Server is now available in Claude Desktop")


def install_system_wide() -> None:
    """Install the server system-wide."""
    import subprocess
    
    logger.info("Installing TSAP MCP Server system-wide")
    
    # Check if we have appropriate permissions
    if os.geteuid() != 0 and sys.platform != "win32":
        logger.error("System-wide installation requires root/admin permissions")
        logger.error("Please run with sudo or as administrator")
        sys.exit(1)
    
    # Install package
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            check=True,
        )
        logger.info("TSAP MCP Server installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Installation failed: {e}")
        sys.exit(1)


def install_regular() -> None:
    """Install the server for the current user."""
    import subprocess
    
    logger.info("Installing TSAP MCP Server for current user")
    
    # Install package
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", ".", "--user"],
            check=True,
        )
        logger.info("TSAP MCP Server installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Installation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 