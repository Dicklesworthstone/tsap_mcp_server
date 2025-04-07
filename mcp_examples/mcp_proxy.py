#!/usr/bin/env python3
"""
MCP Proxy Script

Acts as a bridge between stdio MCP client and the HTTP MCP server.
"""
import asyncio
import json
import sys
import httpx
import os

# Enable debug logging
DEBUG = os.environ.get("MCP_DEBUG", "0") == "1"

def debug_log(msg, data=None):
    """Log debug information to a file."""
    if not DEBUG:
        return
    
    with open("mcp_proxy_debug.log", "a") as f:
        f.write(f"{msg}\n")
        if data:
            f.write(f"DATA: {json.dumps(data, indent=2)}\n")
        f.write("-" * 40 + "\n")

async def main():
    """Run the MCP proxy."""
    server_url = "http://localhost:8021/toolapi/"
    debug_log(f"Starting MCP proxy to server: {server_url}")
    
    async with httpx.AsyncClient(base_url=server_url, timeout=30.0) as client:
        while True:
            # Read a line from stdin
            line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
            if not line:
                debug_log("End of input, exiting")
                break
            
            debug_log("Received request from client", line)
                
            try:
                # Parse the JSON-RPC request
                request = json.loads(line)
                
                # Special handling for initialize request
                if request.get("method") == "initialize":
                    debug_log("Handling initialize request")
                    
                    # Extract protocol version from the client request
                    client_protocol_version = request.get("params", {}).get("protocolVersion", "2024-11-05")
                    debug_log(f"Client requested protocol version: {client_protocol_version}")
                    
                    # Create a proper MCP initialization response using the client's protocol version
                    jsonrpc_response = {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": {
                            "protocolVersion": client_protocol_version,
                            "serverInfo": {
                                "name": "TSAP MCP Server",
                                "version": "0.1.0"
                            },
                            "capabilities": {
                                "prompts": {"listChanged": True},
                                "resources": {"subscribe": True, "listChanged": True},
                                "tools": {"listChanged": True},
                                "logging": {},
                                "completions": {}
                            },
                            "instructions": "TSAP MCP Server provides tools for text search and analysis."
                        }
                    }
                    debug_log("Sending initialize response", jsonrpc_response)
                    sys.stdout.write(json.dumps(jsonrpc_response) + "\n")
                    sys.stdout.flush()
                    continue
                
                # Special handling for initialized notification
                if request.get("method") == "notifications/initialized":
                    debug_log("Received initialized notification, no response needed")
                    continue  # No response needed for notifications
                
                # Special handling for tools/list request
                if request.get("method") == "tools/list":
                    debug_log("Handling tools list request")
                    
                    # Forward to the TSAP server to get actual tools
                    tsap_request = {
                        "request_id": str(request.get("id", "unknown")),
                        "command": "list_tools",
                        "args": {}
                    }
                    
                    try:
                        response = await client.post("", json=tsap_request)
                        result = response.json()
                        
                        debug_log("Got list_tools response from server", result)
                        
                        # Extract tools from the response
                        tools_data = []
                        
                        # The server returns tools in categories
                        categories = result.get("data", {})
                        for category_name, category_tools in categories.items():
                            # Skip any non-list items
                            if not isinstance(category_tools, list):
                                continue
                            # Add tools from this category
                            tools_data.extend(category_tools)
                        
                        tool_count = len(tools_data)
                        debug_log(f"Found {tool_count} tools")
                        
                        # Format tools for MCP response according to the schema
                        mcp_tools = []
                        
                        # Add each tool with proper schema format
                        for tool in tools_data:
                            tool_name = tool.get("name", "")
                            tool_desc = tool.get("description", "")
                            
                            # Create proper inputSchema following the JSON schema spec
                            input_schema = {
                                "type": "object",
                                "properties": {},
                                "required": []
                            }
                            
                            # Handle arguments based on the tool
                            tool_args = tool.get("arguments", [])
                            for arg in tool_args:
                                arg_name = arg.get("name", "")
                                arg_desc = arg.get("description", "")
                                arg_required = arg.get("required", False)
                                arg_type = arg.get("type", "string")
                                
                                # Add to input schema properties
                                input_schema["properties"][arg_name] = {
                                    "type": arg_type,
                                    "description": arg_desc
                                }
                                
                                # Add to required list if needed
                                if arg_required:
                                    input_schema["required"].append(arg_name)
                            
                            # If no arguments defined, add a default query argument
                            if not tool_args:
                                input_schema["properties"]["query"] = {
                                    "type": "string",
                                    "description": f"Input for {tool_name}"
                                }
                                input_schema["required"] = ["query"]
                            
                            # Create tool definition with proper schema
                            mcp_tool = {
                                "name": tool_name,
                                "description": tool_desc,
                                "inputSchema": input_schema,
                                "annotations": {
                                    "title": tool_name,
                                    "readOnlyHint": True
                                }
                            }
                            
                            mcp_tools.append(mcp_tool)
                        
                        debug_log(f"Formatted {len(mcp_tools)} tools for response")
                        
                        # Create response with properly formatted tools list
                        jsonrpc_response = {
                            "jsonrpc": "2.0",
                            "id": request.get("id"),
                            "result": {
                                "tools": mcp_tools
                            }
                        }
                    except Exception as e:
                        debug_log(f"Error getting tools from server: {str(e)}")
                        # Error response
                        jsonrpc_response = {
                            "jsonrpc": "2.0",
                            "id": request.get("id"),
                            "error": {
                                "code": -32603,
                                "message": f"Server error: {str(e)}"
                            }
                        }
                    
                    # Send the response
                    debug_log("Sending tools list response", jsonrpc_response)
                    sys.stdout.write(json.dumps(jsonrpc_response) + "\n")
                    sys.stdout.flush()
                    continue
                
                # Handle tool calls
                if request.get("method") == "tools/call":
                    debug_log("Handling tool call request")
                    
                    # Extract tool name and arguments
                    params = request.get("params", {})
                    tool_name = params.get("name", "")
                    arguments = params.get("arguments", {})
                    
                    debug_log(f"Calling tool {tool_name} with arguments", arguments)
                    
                    # Forward to the TSAP server
                    tsap_request = {
                        "request_id": str(request.get("id", "unknown")),
                        "command": tool_name,
                        "args": arguments
                    }
                    
                    try:
                        response = await client.post("", json=tsap_request)
                        result = response.json()
                        
                        # Format the tool call response according to schema
                        tool_result = result.get("data", {})
                        
                        # Format as textContent
                        content_text = json.dumps(tool_result) if isinstance(tool_result, (dict, list)) else str(tool_result)
                        
                        jsonrpc_response = {
                            "jsonrpc": "2.0",
                            "id": request.get("id"),
                            "result": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": content_text
                                    }
                                ],
                                "isError": result.get("status") != "success"
                            }
                        }
                    except Exception as e:
                        debug_log(f"Error calling tool: {str(e)}")
                        jsonrpc_response = {
                            "jsonrpc": "2.0",
                            "id": request.get("id"),
                            "error": {
                                "code": -32603,
                                "message": f"Error calling tool {tool_name}: {str(e)}"
                            }
                        }
                    
                    debug_log("Sending tool call response", jsonrpc_response)
                    sys.stdout.write(json.dumps(jsonrpc_response) + "\n")
                    sys.stdout.flush()
                    continue
                
                # Handle ping request
                if request.get("method") == "ping":
                    debug_log("Handling ping request")
                    jsonrpc_response = {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": {}
                    }
                    sys.stdout.write(json.dumps(jsonrpc_response) + "\n")
                    sys.stdout.flush()
                    continue
                
                # Handle other requests
                debug_log(f"Unhandled method: {request.get('method')}")
                error_response = {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "error": {
                        "code": -32601,
                        "message": f"Method not implemented: {request.get('method')}"
                    }
                }
                sys.stdout.write(json.dumps(error_response) + "\n")
                sys.stdout.flush()
                
            except Exception as e:
                debug_log(f"Error processing request: {str(e)}")
                # Create an error response
                error_response = {
                    "jsonrpc": "2.0",
                    "id": request.get("id") if "request" in locals() else None,
                    "error": {
                        "code": -32603,
                        "message": f"Internal error: {str(e)}"
                    }
                }
                sys.stdout.write(json.dumps(error_response) + "\n")
                sys.stdout.flush()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        debug_log(f"Unhandled exception: {str(e)}")
        import traceback
        debug_log(traceback.format_exc())
