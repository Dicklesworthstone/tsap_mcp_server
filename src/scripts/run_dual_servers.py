"""
Run both original and ToolAPI servers in parallel.

This script launches both the original TSAP server and the MCP-compliant
version in parallel, allowing for side-by-side testing and comparison.
"""
import asyncio
import os
import sys
import signal
import subprocess
import threading
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def run_original_server():
    """Run the original TSAP server."""
    from src.tsap.__main__ import app
    print("Starting original TSAP server...")
    # Run on a different port than the ToolAPI server
    os.environ["TSAP_PORT"] = "8000"
    app()


def run_mcp_server():
    """Run the ToolAPI server."""
    from src.tsap_mcp.__main__ import app
    print("Starting TSAP ToolAPI server...")
    # Run the ToolAPI server with the CLI
    app()


def thread_wrapper(func):
    """Wrapper to handle exceptions in threads."""
    try:
        func()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in {func.__name__}: {e}")


def main():
    """Run both servers in parallel."""
    # Start the original server in a separate thread
    original_thread = threading.Thread(target=lambda: thread_wrapper(run_original_server))
    original_thread.daemon = True
    original_thread.start()
    
    # Start the ToolAPI server in the main thread
    try:
        run_mcp_server()
    except KeyboardInterrupt:
        print("Shutting down servers...")
    
    # Wait for the original server thread to finish
    if original_thread.is_alive():
        original_thread.join(timeout=5)


if __name__ == "__main__":
    main() 