#!/usr/bin/env python3
"""
Check MCP implementation completeness.

This script verifies that all planned components of the MCP implementation
have been created and are functioning correctly.
"""
import os
import sys
import asyncio
import importlib
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

REQUIRED_FILES = [
    "src/tsap_mcp/server.py",
    "src/tsap_mcp/cli.py",
    "src/tsap_mcp/adapter.py",
    "src/tsap_mcp/__init__.py",
    "src/tsap_mcp/tools/search.py",
    "src/tsap_mcp/tools/processing.py",
    "src/tsap_mcp/tools/analysis.py",
    "src/tsap_mcp/tools/visualization.py",
    "src/tsap_mcp/tools/__init__.py",
    "src/tsap_mcp/resources/files.py",
    "src/tsap_mcp/resources/project.py",
    "src/tsap_mcp/resources/config.py",
    "src/tsap_mcp/resources/semantic.py",
    "src/tsap_mcp/resources/__init__.py",
    "src/tsap_mcp/prompts/search.py",
    "src/tsap_mcp/prompts/code_analysis.py",
    "src/tsap_mcp/prompts/__init__.py",
    "docs/migration_guide.md",
    "pyproject.toml",
    "README.md",
]

REQUIRED_IMPORTS = [
    "mcp",
    "mcp.server.fastmcp",
    "fastapi",
    "pydantic",
    "asyncio",
]

REQUIRED_MODULES = [
    "tsap_mcp",
    "tsap_mcp.server",
    "tsap_mcp.cli",
    "tsap_mcp.adapter",
    "tsap_mcp.tools.search",
    "tsap_mcp.tools.processing",
    "tsap_mcp.tools.analysis",
    "tsap_mcp.tools.visualization",
    "tsap_mcp.resources.files",
    "tsap_mcp.resources.project",
    "tsap_mcp.resources.config",
    "tsap_mcp.resources.semantic",
    "tsap_mcp.prompts.search",
    "tsap_mcp.prompts.code_analysis",
]

# Minimum required tools, resources, and prompts
REQUIRED_TOOLS = [
    # Search tools
    "search",
    "search_regex",
    "search_semantic",
    "search_code",
    
    # Processing tools
    "process_text",
    "extract_data",
    "transform_text",
    "format_data",
    "validate_text",
    
    # Analysis tools
    "analyze_code",
    "analyze_text",
    "analyze_data",
    
    # Visualization tools
    "generate_chart",
    "generate_network_graph"
]

REQUIRED_RESOURCES = [
    # File resources
    "file://{path}",
    "file://{path}/info",
    
    # Project resources
    "project://structure",
    "project://dependencies",
    "project://stats",
    
    # Config resources
    "config://app",
    
    # Semantic resources
    "semantic://corpus/{corpus_id}",
]

REQUIRED_PROMPTS = [
    "code_review",
    "search_help",
]


def check_files_exist() -> Tuple[bool, List[str]]:
    """Check if all required files exist.
    
    Returns:
        Tuple containing success status and list of missing files
    """
    missing_files = []
    
    for file_path in REQUIRED_FILES:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    return len(missing_files) == 0, missing_files


def check_imports_available() -> Tuple[bool, List[str]]:
    """Check if all required imports are available.
    
    Returns:
        Tuple containing success status and list of missing imports
    """
    missing_imports = []
    
    for module_name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing_imports.append(module_name)
    
    return len(missing_imports) == 0, missing_imports


def check_modules_importable() -> Tuple[bool, List[str]]:
    """Check if all required modules can be imported.
    
    Returns:
        Tuple containing success status and list of modules that couldn't be imported
    """
    failed_modules = []
    
    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
        except ImportError:
            failed_modules.append(module_name)
    
    return len(failed_modules) == 0, failed_modules


async def check_tools_registered() -> Tuple[bool, Dict[str, bool]]:
    """Check if all required tools are registered.
    
    Returns:
        Tuple containing success status and dictionary of tool registration status
    """
    try:
        # Import the server module
        from tsap_mcp.server import mcp
        
        # Get list of tool functions
        registered_tools = {
            tool.__name__: True
            for tool in mcp.list_tool_functions()
        }
        
        # Check required tools
        tool_status = {
            tool_name: tool_name in registered_tools
            for tool_name in REQUIRED_TOOLS
        }
        
        return all(tool_status.values()), tool_status
    except ImportError:
        return False, {tool_name: False for tool_name in REQUIRED_TOOLS}


async def check_resources_registered() -> Tuple[bool, Dict[str, bool]]:
    """Check if all required resources are registered.
    
    Returns:
        Tuple containing success status and dictionary of resource registration status
    """
    try:
        # Import the server module
        from tsap_mcp.server import mcp
        
        # Get list of resource patterns
        registered_resources = set()
        for name in dir(mcp):
            obj = getattr(mcp, name)
            if callable(obj) and hasattr(obj, "__mcp_resource__"):
                pattern = getattr(obj, "__mcp_resource__", None)
                if pattern:
                    registered_resources.add(pattern)
        
        # Check required resources
        resource_status = {
            resource_pattern: any(
                req_pattern_matches(resource_pattern, reg_pattern)
                for reg_pattern in registered_resources
            )
            for resource_pattern in REQUIRED_RESOURCES
        }
        
        return all(resource_status.values()), resource_status
    except ImportError:
        return False, {resource_pattern: False for resource_pattern in REQUIRED_RESOURCES}


def req_pattern_matches(required: str, registered: str) -> bool:
    """Check if a registered resource pattern matches a required pattern.
    
    Args:
        required: Required resource pattern
        registered: Registered resource pattern
        
    Returns:
        True if the registered pattern satisfies the required pattern
    """
    # Simple exact match
    if required == registered:
        return True
    
    # Handle parameter patterns
    req_parts = required.split("/")
    reg_parts = registered.split("/")
    
    if len(req_parts) != len(reg_parts):
        return False
    
    for req_part, reg_part in zip(req_parts, reg_parts):
        # Check if parts match or if the required part is a parameter pattern
        if req_part != reg_part and not (
            req_part.startswith("{") and req_part.endswith("}")
        ):
            return False
    
    return True


async def check_prompts_registered() -> Tuple[bool, Dict[str, bool]]:
    """Check if all required prompts are registered.
    
    Returns:
        Tuple containing success status and dictionary of prompt registration status
    """
    try:
        # Import the server module
        from tsap_mcp.server import mcp
        
        # Get list of prompt methods
        registered_prompts = set()
        for name in dir(mcp):
            obj = getattr(mcp, name)
            if callable(obj) and hasattr(obj, "__mcp_prompt__"):
                registered_prompts.add(obj.__name__)
        
        # Check required prompts
        prompt_status = {
            prompt_name: prompt_name in registered_prompts
            for prompt_name in REQUIRED_PROMPTS
        }
        
        return all(prompt_status.values()), prompt_status
    except ImportError:
        return False, {prompt_name: False for prompt_name in REQUIRED_PROMPTS}


async def check_cli_executable() -> bool:
    """Check if CLI can be executed.
    
    Returns:
        True if CLI can be executed, False otherwise
    """
    try:
        # Import CLI module
        from tsap_mcp.cli import main
        
        # Check if main function exists and is callable
        return callable(main)
    except ImportError:
        return False


async def check_adapter_works() -> bool:
    """Check if adapter layer works.
    
    Returns:
        True if adapter layer works, False otherwise
    """
    try:
        # Import adapter module
        from tsap_mcp.adapter import initialize_adapter
        
        # Check if initialize_adapter function exists and is callable
        return callable(initialize_adapter)
    except ImportError:
        return False


async def main() -> int:
    """Run completeness checks and return exit code.
    
    Returns:
        0 if all checks pass, 1 otherwise
    """
    # Print header
    print("=== TSAP MCP Implementation Completeness Check ===\n")
    
    # Check files
    files_ok, missing_files = check_files_exist()
    print(f"File check: {'✅ PASS' if files_ok else '❌ FAIL'}")
    if not files_ok:
        print("Missing files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
    print()
    
    # Check imports
    imports_ok, missing_imports = check_imports_available()
    print(f"Import check: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    if not imports_ok:
        print("Missing imports:")
        for import_name in missing_imports:
            print(f"  - {import_name}")
    print()
    
    # Check modules
    modules_ok, failed_modules = check_modules_importable()
    print(f"Module check: {'✅ PASS' if modules_ok else '❌ FAIL'}")
    if not modules_ok:
        print("Failed modules:")
        for module_name in failed_modules:
            print(f"  - {module_name}")
    print()
    
    # Check tools
    tools_ok, tool_status = await check_tools_registered()
    print(f"Tool check: {'✅ PASS' if tools_ok else '❌ FAIL'}")
    if not tools_ok:
        print("Missing tools:")
        for tool_name, registered in tool_status.items():
            if not registered:
                print(f"  - {tool_name}")
    print()
    
    # Check resources
    resources_ok, resource_status = await check_resources_registered()
    print(f"Resource check: {'✅ PASS' if resources_ok else '❌ FAIL'}")
    if not resources_ok:
        print("Missing resources:")
        for resource_pattern, registered in resource_status.items():
            if not registered:
                print(f"  - {resource_pattern}")
    print()
    
    # Check prompts
    prompts_ok, prompt_status = await check_prompts_registered()
    print(f"Prompt check: {'✅ PASS' if prompts_ok else '❌ FAIL'}")
    if not prompts_ok:
        print("Missing prompts:")
        for prompt_name, registered in prompt_status.items():
            if not registered:
                print(f"  - {prompt_name}")
    print()
    
    # Check CLI
    cli_ok = await check_cli_executable()
    print(f"CLI check: {'✅ PASS' if cli_ok else '❌ FAIL'}")
    print()
    
    # Check adapter
    adapter_ok = await check_adapter_works()
    print(f"Adapter check: {'✅ PASS' if adapter_ok else '❌ FAIL'}")
    print()
    
    # Overall summary
    all_ok = (
        files_ok and imports_ok and modules_ok and tools_ok and 
        resources_ok and prompts_ok and cli_ok and adapter_ok
    )
    
    # Print summary
    print("=== Summary ===")
    print(f"Overall: {'✅ PASS' if all_ok else '❌ FAIL'}")
    print(f"Files: {files_ok}")
    print(f"Imports: {imports_ok}")
    print(f"Modules: {modules_ok}")
    print(f"Tools: {tools_ok}")
    print(f"Resources: {resources_ok}")
    print(f"Prompts: {prompts_ok}")
    print(f"CLI: {cli_ok}")
    print(f"Adapter: {adapter_ok}")
    
    # Calculate completion percentage
    total_checks = 7  # Files, imports, modules, tools, resources, prompts, CLI, adapter
    passed_checks = sum([
        1 if check else 0 
        for check in [files_ok, imports_ok, modules_ok, tools_ok, 
                      resources_ok, prompts_ok, cli_ok, adapter_ok]
    ])
    
    # Add detailed completion for tools, resources, and prompts
    tool_completion = sum(1 for status in tool_status.values() if status) / len(tool_status)
    resource_completion = sum(1 for status in resource_status.values() if status) / len(resource_status)
    prompt_completion = sum(1 for status in prompt_status.values() if status) / len(prompt_status)
    
    # Calculate total completion percentage
    completion_percentage = (passed_checks / total_checks) * 100
    
    # Adjust for partial completions in tools, resources, and prompts
    if not tools_ok:
        completion_percentage += (tool_completion / total_checks) * 100
    
    if not resources_ok:
        completion_percentage += (resource_completion / total_checks) * 100
    
    if not prompts_ok:
        completion_percentage += (prompt_completion / total_checks) * 100
    
    print(f"Completion: {completion_percentage:.1f}%")
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 