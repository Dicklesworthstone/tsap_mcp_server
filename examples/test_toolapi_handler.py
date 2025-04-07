#!/usr/bin/env python3
"""
Test script for ToolAPI handler system.

This script directly tests the ToolAPI handler system without going through the server.
"""
import asyncio
import json

from tsap.toolapi.protocol import ToolAPIRequest, ToolAPICommandType
from tsap.toolapi.handler import handle_request, initialize_handlers, _command_handlers

async def main():
    """Test the ToolAPI handler system."""
    # Initialize handlers
    print("Initializing ToolAPI handlers...")
    initialize_handlers()
    
    # Check registered handlers
    print(f"\nRegistered handlers: {len(_command_handlers)}")
    for cmd, handler in _command_handlers.items():
        print(f"  - {cmd}: {handler.__name__}")
    
    # Create a test request
    print("\nCreating test request...")
    request = ToolAPIRequest(
        command="test",
        args={"param": "value"}
    )
    print(f"Request ID: {request.request_id}")
    
    # Handle the request
    print("\nHandling request...")
    try:
        response = await handle_request(request)
        print("\nResponse:")
        print(f"Status: {response.status}")
        print(f"Data: {json.dumps(response.data, indent=2)}")
    except Exception as e:
        print(f"Error handling request: {e}")
        import traceback
        traceback.print_exc()
    
    # Try an INFO request
    print("\n\nCreating INFO request...")
    info_request = ToolAPIRequest(
        command=ToolAPICommandType.INFO,
        args={}
    )
    print(f"Request ID: {info_request.request_id}")
    
    # Handle the INFO request
    print("\nHandling INFO request...")
    try:
        info_response = await handle_request(info_request)
        print("\nResponse:")
        print(f"Status: {info_response.status}")
        print(f"Data: {json.dumps(info_response.data, indent=2)}")
    except Exception as e:
        print(f"Error handling INFO request: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 