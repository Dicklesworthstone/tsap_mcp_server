#!/usr/bin/env python3
"""
Run MCP Test with Proxy

This script runs the MCP proxy in debug mode and captures the output.
"""
import os
import subprocess
import time

def main():
    """Run the test with the proxy in debug mode."""
    # Set debug mode for the proxy
    env = os.environ.copy()
    env["MCP_DEBUG"] = "1"
    
    # Clear previous log
    if os.path.exists("mcp_proxy_debug.log"):
        os.unlink("mcp_proxy_debug.log")
    
    print("Running MCP test with debug enabled...")
    
    # Start the proxy script
    proxy_process = subprocess.Popen(
        ["python", "mcp_examples/mcp_proxy.py"],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    # Send a sample initialize request
    initialize_request = {
        "jsonrpc": "2.0",
        "id": "test-1",
        "method": "initialize",
        "params": {
            "protocolVersion": 1,
            "clientInfo": {
                "name": "MCP Test Runner",
                "version": "1.0.0"
            },
            "capabilities": {}
        }
    }
    
    # Convert to JSON and send
    import json
    proxy_process.stdin.write(json.dumps(initialize_request) + "\n")
    proxy_process.stdin.flush()
    
    # Wait for response
    print("Waiting for response...")
    time.sleep(1)
    
    # Send a listTools request
    list_tools_request = {
        "jsonrpc": "2.0",
        "id": "test-2",
        "method": "listTools"
    }
    
    # Convert to JSON and send
    proxy_process.stdin.write(json.dumps(list_tools_request) + "\n")
    proxy_process.stdin.flush()
    
    # Wait for response
    print("Waiting for tools response...")
    time.sleep(1)
    
    # Close the process
    proxy_process.stdin.close()
    proxy_process.terminate()
    
    # Display the debug log
    print("\nDebug log contents:")
    with open("mcp_proxy_debug.log", "r") as f:
        print(f.read())
    
    # Display stdout
    stdout_data = proxy_process.stdout.read()
    print("\nStdout from proxy:")
    print(stdout_data)
    
    # Display stderr if any
    stderr_data = proxy_process.stderr.read()
    if stderr_data:
        print("\nStderr from proxy:")
        print(stderr_data)

if __name__ == "__main__":
    main() 