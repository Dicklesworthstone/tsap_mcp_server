import pytest
import inspect

from mcp.server.fastmcp import FastMCP, Context
from tsap_mcp.server import mcp


@pytest.fixture
def server():
    """Create a test server instance."""
    return FastMCP("Test Server")


def test_server_initialization():
    """Test server initialization."""
    # Test server exists with proper name
    assert mcp.name == "TSAP MCP Server"
    
    # Test server has proper version
    assert hasattr(mcp, "version")
    assert isinstance(mcp.version, str)
    assert mcp.version.count(".") >= 2  # Ensure it has at least major.minor.patch format


def test_tool_registration():
    """Test that tools are properly registered."""
    # Get all registered tools
    tools = mcp.list_tool_functions()
    
    # Verify that tools are registered and properly formatted
    assert len(tools) > 0
    
    # Check that various tool categories are represented
    tool_names = [tool.__name__ for tool in tools]
    
    # Check for search tools
    assert any(name.startswith('search') for name in tool_names)
    
    # Check for processing tools
    assert any(name.startswith('process') or name.startswith('extract') or name.startswith('transform') for name in tool_names)
    
    # Check for analysis tools
    assert any(name.startswith('analyze') for name in tool_names)
    
    # Check for visualization tools
    assert any(name.startswith('generate') for name in tool_names)
    
    # Check a specific tool to ensure its signature is correct
    search_tool = next((t for t in tools if t.__name__ == 'search'), None)
    if search_tool:
        sig = inspect.signature(search_tool)
        params = sig.parameters
        assert 'query' in params
        assert 'path' in params
        assert 'max_results' in params


def test_resource_registration():
    """Test that resources are properly registered."""
    # Get all registered resource patterns
    resource_patterns = set()
    for name in dir(mcp):
        obj = getattr(mcp, name)
        if callable(obj) and hasattr(obj, "__mcp_resource__"):
            pattern = getattr(obj, "__mcp_resource__", None)
            if pattern:
                resource_patterns.add(pattern)
    
    # Verify that resources are registered
    assert len(resource_patterns) > 0
    
    # Check for various resource categories
    
    # File resources
    assert any(pattern.startswith('file://') for pattern in resource_patterns)
    
    # Project resources
    assert any(pattern.startswith('project://') for pattern in resource_patterns)
    
    # Config resources
    assert any(pattern.startswith('config://') for pattern in resource_patterns)
    
    # Semantic resources
    assert any(pattern.startswith('semantic://') for pattern in resource_patterns)


def test_prompt_registration():
    """Test that prompts are properly registered."""
    # Get all registered prompts
    prompt_functions = set()
    for name in dir(mcp):
        obj = getattr(mcp, name)
        if callable(obj) and hasattr(obj, "__mcp_prompt__"):
            prompt_functions.add(obj.__name__)
    
    # Verify that prompts are registered
    assert len(prompt_functions) > 0
    
    # Check for code analysis prompts
    assert any(name.startswith('code_') for name in prompt_functions)
    
    # Check for search prompts
    assert any(name.startswith('search_') for name in prompt_functions)


@pytest.mark.asyncio
async def test_lifespan_context():
    """Test that lifespan context is properly initialized."""
    # Test server with lifespan
    try:
        # Import lifespan to avoid circular imports
        from tsap_mcp.lifespan import lifespan_context
        
        # Create a test server with the lifespan
        test_server = FastMCP("Test Server", lifespan=lifespan_context)
        
        # Mock an async context manager to simulate server startup/shutdown
        class MockContext:
            async def __aenter__(self):
                # This simulates server startup and lifespan initialization
                ctx = Context()
                ctx.request_context.lifespan_context = await test_server.lifespan(test_server)
                return ctx
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                # This simulates server shutdown
                pass
        
        # Use the context manager to test lifespan
        async with MockContext() as ctx:
            # Verify required components are in the lifespan context
            assert "config" in ctx.request_context.lifespan_context
            assert "server_info" in ctx.request_context.lifespan_context
            
            # Check config contains required sections
            config = ctx.request_context.lifespan_context["config"]
            assert "server" in config
            assert "search" in config
            
            # Check server info contains required fields
            server_info = ctx.request_context.lifespan_context["server_info"]
            assert "name" in server_info
            assert "version" in server_info
            assert server_info["name"] == "TSAP MCP Server"
    
    except ImportError:
        pytest.skip("Lifespan module not found")


@pytest.mark.asyncio
async def test_server_capabilities():
    """Test that server has required capabilities enabled."""
    # Check server capabilities
    capabilities = mcp.get_capabilities()
    
    # Verify that tools capability is enabled
    assert capabilities["tools"] is not None
    
    # Verify that resources capability is enabled
    assert capabilities["resources"] is not None
    
    # Verify that prompts capability is enabled
    assert capabilities["prompts"] is not None
    
    # Verify that list_changed notifications are enabled
    assert capabilities["tools"].get("listChanged") is True
    assert capabilities["resources"].get("listChanged") is True
    assert capabilities["prompts"].get("listChanged") is True 