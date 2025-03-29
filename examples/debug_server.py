#!/usr/bin/env python3
"""
Debug server for TSAP MCP.

This is a simple HTTP server that dumps all requests for debugging.
"""
import json
import uvicorn
from fastapi import FastAPI, Request

app = FastAPI()

@app.post("/mcp/")
async def mcp_endpoint(request: Request):
    """Simple MCP endpoint that just dumps the request."""
    # Get raw request
    raw_body = await request.body()
    
    # Print headers
    print("\n--- REQUEST HEADERS ---")
    for header, value in request.headers.items():
        print(f"{header}: {value}")
    
    # Print body
    print("\n--- REQUEST BODY ---")
    body_str = raw_body.decode('utf-8')
    print(body_str)
    
    # Try to parse as JSON
    try:
        body_json = json.loads(body_str)
        print("\n--- PARSED JSON ---")
        print(json.dumps(body_json, indent=2))
    except Exception:
        print("Failed to parse body as JSON")
    
    # Return a success response
    req_id = None
    command = None
    try:
        req_id = body_json.get("request_id", "unknown")
        command = body_json.get("command", "unknown")
    except Exception:
        pass
    
    # Return a success response
    return {
        "request_id": req_id,
        "status": "success",
        "command": command,
        "data": {
            "message": "Debug server received request",
            "received": body_json
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}

if __name__ == "__main__":
    print("Starting debug server on http://127.0.0.1:8021")
    uvicorn.run(app, host="127.0.0.1", port=8021) 