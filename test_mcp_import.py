#!/usr/bin/env python3
"""
Test imports and basic functionality of the TSAP MCP server.

This script verifies that all modules can be imported and the
server can be instantiated.
"""
import sys
from pathlib import Path

# Add the src directory to Python path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

# Try to import the required modules
try:
    print("Importing core modules...")
    from mcp.server.fastmcp import FastMCP
    print("✓ MCP SDK imported successfully")
except ImportError as e:
    print(f"❌ Failed to import MCP SDK: {e}")
    print("   Please install it with: pip install mcp[cli]")
    sys.exit(1)

try:
    print("\nImporting TSAP MCP components...")
    from tsap_mcp.server import mcp
    print("✓ TSAP MCP server module imported successfully")
    
    print("\nVerifying component registration...")
    # Check tools
    tools = mcp.list_tool_functions()
    print(f"✓ {len(tools)} tools registered")
    for tool in tools[:5]:  # Show first 5
        print(f"  - {tool.__name__}")
    if len(tools) > 5:
        print(f"  - ... ({len(tools) - 5} more)")
    
    # Check resources
    resource_pattern_methods = [m for m in dir(mcp) if callable(getattr(mcp, m)) and hasattr(getattr(mcp, m), "__mcp_resource__")]
    resources = []
    for method in resource_pattern_methods:
        resource = getattr(mcp, method)
        if hasattr(resource, "__mcp_resource__"):
            resources.append(resource)
    
    print(f"✓ {len(resources)} resources registered")
    for i, resource in enumerate(resources[:5]):  # Show first 5
        pattern = getattr(resource, "__mcp_resource__", "unknown")
        print(f"  - {pattern}")
    if len(resources) > 5:
        print(f"  - ... ({len(resources) - 5} more)")
    
    # Check prompts
    prompt_methods = [m for m in dir(mcp) if callable(getattr(mcp, m)) and hasattr(getattr(mcp, m), "__mcp_prompt__")]
    prompts = []
    for method in prompt_methods:
        prompt = getattr(mcp, method)
        if hasattr(prompt, "__mcp_prompt__"):
            prompts.append(prompt)
    
    print(f"✓ {len(prompts)} prompts registered")
    for i, prompt in enumerate(prompts[:5]):  # Show first 5
        name = getattr(prompt, "__name__", "unknown")
        print(f"  - {name}")
    if len(prompts) > 5:
        print(f"  - ... ({len(prompts) - 5} more)")
    
    print("\nTSAP MCP server verification complete!")
    
except ImportError as e:
    print(f"❌ Failed to import TSAP MCP components: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error during verification: {e}")
    sys.exit(1) 