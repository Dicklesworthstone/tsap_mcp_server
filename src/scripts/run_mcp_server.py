#!/usr/bin/env python3
"""
Run TSAP MCP Server

This script provides a command-line interface to run the TSAP ToolAPI server.
"""
import os
import sys
import argparse
import logging
import traceback
from importlib.metadata import version, PackageNotFoundError

# Add parent directory to path if running as script
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(script_dir)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

# Import the server
from tsap_mcp.server import mcp, run_server


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run TSAP MCP Server")
    
    # Server configuration
    parser.add_argument(
        "--host", 
        default=os.environ.get("TSAP_HOST", "127.0.0.1"),
        help="Host to bind to (default: 127.0.0.1 or TSAP_HOST env var)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=int(os.environ.get("TSAP_PORT", "8000")),
        help="Port to listen on (default: 8000 or TSAP_PORT env var)"
    )
    
    # Logging configuration
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=os.environ.get("TSAP_LOG_LEVEL", "INFO"),
        help="Logging level (default: INFO or TSAP_LOG_LEVEL env var)"
    )
    
    # Performance configuration
    parser.add_argument(
        "--performance-mode",
        choices=["fast", "balanced", "quality"],
        default=os.environ.get("TSAP_PERFORMANCE_MODE", "balanced"),
        help="Performance mode (default: balanced or TSAP_PERFORMANCE_MODE env var)"
    )
    
    # Debugging options
    parser.add_argument(
        "--debug", 
        action="store_true",
        default=os.environ.get("TSAP_DEBUG", "").lower() == "true",
        help="Enable debug mode (default: false or TSAP_DEBUG env var)"
    )
    
    # Display version
    parser.add_argument(
        "--version", 
        action="store_true",
        help="Display version information and exit"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the server."""
    args = parse_args()
    
    # Show version and exit if requested
    if args.version:
        try:
            pkg_version = version("tsap_mcp")
        except PackageNotFoundError:
            pkg_version = "0.1.0 (development)"
        
        print(f"TSAP MCP Server version: {pkg_version}")
        print(f"Python version: {sys.version}")
        return 0
    
    # Configure logging
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=getattr(logging, args.log_level), format=log_format)
    logger = logging.getLogger("tsap_mcp")
    
    # Set environment variables for server config
    os.environ["TSAP_HOST"] = args.host
    os.environ["TSAP_PORT"] = str(args.port)
    os.environ["TSAP_LOG_LEVEL"] = args.log_level
    os.environ["TSAP_PERFORMANCE_MODE"] = args.performance_mode
    os.environ["TSAP_DEBUG"] = str(args.debug).lower()
    
    # Log startup information
    logger.info(f"Starting TSAP MCP Server on {args.host}:{args.port}")
    logger.info(f"Performance mode: {args.performance_mode}")
    logger.info(f"Debug mode: {'Enabled' if args.debug else 'Disabled'}")
    
    try:
        # Run the server
        run_server()
        return 0
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Error running server: {e}")
        if args.debug:
            logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main()) 