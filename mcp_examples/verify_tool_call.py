#!/usr/bin/env python3
"""
Simple script to verify that the MCP proxy works by making a concrete tool call.
"""
import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    """Test making a concrete tool call via the MCP proxy."""
    # Path to the proxy script
    proxy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_proxy.py")
    
    print(f"Connecting to MCP proxy: {proxy_path}")
    
    # Configure server parameters for stdio connection to our proxy
    server_params = StdioServerParameters(
        command="python",
        args=[proxy_path],
        env={"MCP_DEBUG": "1"}  # Enable debug logging
    )
    
    try:
        async with stdio_client(server_params) as (read, write):
            print("Connected to proxy, initializing session...")
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()
                print("Session initialized successfully!")
                
                # Get the list of available tools
                print("Fetching tool list...")
                tools_result = await session.list_tools()
                # Extract the tools list from the result
                tools = tools_result.tools
                print(f"Found {len(tools)} tools")
                
                # Pick a simple tool that should work without complex arguments
                info_tool = next((t for t in tools if t.name == "info"), None)
                if not info_tool:
                    print("Could not find 'info' tool, using first available tool instead")
                    if tools:
                        test_tool = tools[0]
                        print(f"Testing with tool: {test_tool.name}")
                    else:
                        raise ValueError("No tools available to test")
                else:
                    test_tool = info_tool
                    print(f"Testing with 'info' tool: {test_tool.description}")
                
                # Make an actual tool call
                print(f"Making tool call to '{test_tool.name}'...")
                tool_result = await session.call_tool(test_tool.name, arguments={})
                
                # Extract the result content as text
                result_text = None
                for content in tool_result.content:
                    if content.type == "text":
                        result_text = content.text
                        break
                
                # Display the result
                print("\n=== TOOL CALL RESULT ===")
                if result_text:
                    print(result_text)
                else:
                    print(f"Raw result: {tool_result}")
                print("========================\n")
                
                # Test a second tool if info worked
                print("Testing another tool call...")
                ripgrep_tool = next((t for t in tools if t.name == "ripgrep_search"), None)
                if ripgrep_tool:
                    print(f"Making call to '{ripgrep_tool.name}'...")
                    # Set up basic ripgrep parameters
                    args = {
                        "pattern": "import",
                        "paths": ["."]
                    }
                    try:
                        rg_result = await session.call_tool(ripgrep_tool.name, arguments=args)
                        
                        # Extract result content
                        rg_text = None
                        for content in rg_result.content:
                            if content.type == "text":
                                rg_text = content.text
                                break
                        
                        print("\n=== RIPGREP RESULT ===")
                        if rg_text:
                            # Show just a few lines if result is too long
                            if len(rg_text) > 500:
                                print(f"{rg_text[:500]}...\n[truncated - {len(rg_text)} chars total]")
                            else:
                                print(rg_text)
                        else:
                            print(f"Raw result: {rg_result}")
                        print("======================\n")
                    except Exception as e:
                        print(f"Error with ripgrep call: {e}")
                
                print("MCP proxy verification complete!")
                return True
                    
    except Exception as e:
        print(f"Error during MCP proxy test: {e}")
        import traceback
        traceback.print_exc()
        return False

# Run with a timeout
async def run_with_timeout():
    """Run the main function with a timeout."""
    try:
        result = await asyncio.wait_for(main(), timeout=20)
        if result:
            print("✅ SUCCESS: MCP proxy is working correctly!")
        else:
            print("❌ FAILURE: MCP proxy test encountered errors.")
    except asyncio.TimeoutError:
        print("❌ TIMEOUT: MCP proxy test took too long to complete.")

if __name__ == "__main__":
    asyncio.run(run_with_timeout()) 