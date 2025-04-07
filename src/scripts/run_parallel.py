#!/usr/bin/env python3
"""
Run both TSAP servers in parallel.

This script runs both the original TSAP server and the new MCP-native server
in parallel, allowing for side-by-side testing and comparison.
"""
import os
import sys
import signal
import multiprocessing
import time
import argparse
from pathlib import Path

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def run_original_server(host: str, port: int):
    """Run the original TSAP server.
    
    Args:
        host: Host to bind to
        port: Port to listen on
    """
    from tsap.server import start_server
    print(f"Starting original TSAP server on {host}:{port}...")
    start_server(host=host, port=port)


def run_mcp_native_server():
    """Run the MCP-native TSAP server."""
    import uvicorn
    print("Starting MCP-native TSAP server...")
    uvicorn.run(
        "tsap_mcp.server:get_mcp_app()",
        host="127.0.0.1",
        port=8022,
        factory=True
    )


def run_proxy_server(original_port: int, mcp_native_port: int, proxy_port: int):
    """Run a proxy server that can route requests to either server.
    
    Args:
        original_port: Port of the original server
        mcp_native_port: Port of the MCP-native server
        proxy_port: Port for the proxy server to listen on
    """
    from fastapi import FastAPI, Request, Response
    import uvicorn
    import httpx
    import json
    
    app = FastAPI(title="TSAP Proxy Server")
    
    # Configure server URLs
    original_url = f"http://127.0.0.1:{original_port}/mcp/"
    mcp_native_url = f"http://127.0.0.1:{mcp_native_port}/"
    
    # Commands to route to MCP native (gradually expand this list)
    mcp_native_commands = set([
        "ripgrep_search",
        "jq_query",
        "html_process",
    ])
    
    @app.post("/mcp/")
    async def proxy_request(request: Request):
        """Proxy requests to the appropriate server based on command."""
        # Parse request body
        body = await request.json()
        command = body.get("command")
        
        # Determine target server
        target_url = mcp_native_url if command in mcp_native_commands else original_url
        
        # Log the routing decision
        print(f"Routing command '{command}' to {'MCP-native' if command in mcp_native_commands else 'original'} server")
        
        # Forward request to target server
        async with httpx.AsyncClient() as client:
            response = await client.post(
                target_url,
                json=body,
                headers={"Content-Type": "application/json"}
            )
            
            # Return response from target server
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers)
            )
    
    # Run the proxy server
    print(f"Starting proxy server on port {proxy_port}...")
    uvicorn.run(app, host="127.0.0.1", port=proxy_port)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run both TSAP servers in parallel")
    parser.add_argument("--original-port", type=int, default=8021, help="Port for original server")
    parser.add_argument("--mcp-port", type=int, default=8022, help="Port for MCP-native server")
    parser.add_argument("--proxy-port", type=int, default=8023, help="Port for proxy server")
    parser.add_argument("--proxy", action="store_true", help="Run proxy server")
    args = parser.parse_args()
    
    # Start the original server
    original_process = multiprocessing.Process(
        target=run_original_server,
        args=("127.0.0.1", args.original_port)
    )
    original_process.start()
    
    # Start the MCP-native server
    mcp_process = multiprocessing.Process(
        target=run_mcp_native_server
    )
    mcp_process.start()
    
    # Start the proxy server if requested
    proxy_process = None
    if args.proxy:
        proxy_process = multiprocessing.Process(
            target=run_proxy_server,
            args=(args.original_port, args.mcp_port, args.proxy_port)
        )
        proxy_process.start()
    
    # Print information
    print("\nServers started successfully!")
    print(f"Original TSAP server:   http://127.0.0.1:{args.original_port}/mcp/")
    print(f"MCP-native TSAP server: http://127.0.0.1:{args.mcp_port}/")
    if args.proxy:
        print(f"Proxy server:           http://127.0.0.1:{args.proxy_port}/mcp/")
    print("\nPress Ctrl+C to stop all servers\n")
    
    # Wait for termination
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        
        # Terminate all processes
        original_process.terminate()
        mcp_process.terminate()
        if proxy_process:
            proxy_process.terminate()
        
        # Wait for processes to terminate
        original_process.join()
        mcp_process.join()
        if proxy_process:
            proxy_process.join()
        
        print("All servers stopped.")


if __name__ == "__main__":
    # Ignore SIGINT in the main process and let it be handled by the child processes
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    main() 