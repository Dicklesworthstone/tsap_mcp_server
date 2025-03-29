#!/usr/bin/env python3
"""
Standalone MCP Server for TSAP

This is a minimal FastAPI server that uses the TSAP MCP handler system.
"""
import os
import sys
import json
import uvicorn
import traceback
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import TSAP MCP components
from tsap.mcp.protocol import MCPRequest
from tsap.mcp.handler import handle_request, initialize_handlers, _command_handlers

# Create FastAPI app
app = FastAPI(title="TSAP MCP Standalone Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MCP handlers
print("Initializing MCP handlers...")
initialize_handlers()

# Print registered handlers
print(f"Registered handlers: {len(_command_handlers)}")
for cmd, handler in _command_handlers.items():
    print(f"  - {cmd}: {handler.__name__}")

@app.post("/mcp/")
async def mcp_endpoint(request: MCPRequest):
    """MCP endpoint that uses the TSAP handler system."""
    print(f"\nREQUEST: {request.model_dump_json()}")
    
    try:
        # Handle the request using the TSAP handler
        response = await handle_request(request)
        
        # Log the response
        print(f"RESPONSE STATUS: {response.status}")
        if response.data:
            print(f"RESPONSE DATA: {json.dumps(response.data, default=str)[:200]}...")
        if response.error:
            print(f"RESPONSE ERROR: {response.error}")
            
        # Return the response
        return response
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": traceback.format_exc()}
        )

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}

if __name__ == "__main__":
    print("Starting standalone MCP server on http://127.0.0.1:8021")
    uvicorn.run(app, host="127.0.0.1", port=8021) 