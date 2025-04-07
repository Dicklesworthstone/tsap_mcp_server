import pytest
import asyncio
from unittest.mock import patch, MagicMock
import tempfile
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from tsap_mcp.resources.files import get_file_content, get_file_info
from tsap_mcp.resources.project import get_project_structure, get_project_dependencies, get_project_stats
from tsap_mcp.resources.config import get_config
from tsap_mcp.resources.semantic import get_semantic_corpus_info


@pytest.fixture
def mock_file_content():
    """Create a temporary file with content for testing."""
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp:
        temp.write("This is test file content\nLine 2\nLine 3")
        temp_path = temp.name
    
    yield temp_path
    
    # Clean up
    os.unlink(temp_path)


@pytest.mark.asyncio
async def test_file_content_resource(mock_file_content):
    """Test file content resource."""
    # Create relative path from temp file
    file_path = mock_file_content
    
    # Test with context object
    context = MagicMock()
    content = await get_file_content(file_path, context)
    
    assert isinstance(content, str)
    assert "This is test file content" in content
    assert "Line 2" in content
    assert "Line 3" in content


@pytest.mark.asyncio
async def test_file_info_resource(mock_file_content):
    """Test file info resource."""
    # Create relative path from temp file
    file_path = mock_file_content
    
    # Test with context object
    context = MagicMock()
    info = await get_file_info(file_path, context)
    
    assert isinstance(info, dict)
    assert "size" in info
    assert "modified" in info
    assert "mime_type" in info
    assert info["size"] > 0
    assert "text/plain" in info["mime_type"]


@pytest.mark.asyncio
async def test_project_structure_resource():
    """Test project structure resource."""
    mock_structure = {
        "name": "test-project",
        "type": "directory",
        "children": [
            {
                "name": "src",
                "type": "directory",
                "children": [
                    {"name": "main.py", "type": "file"},
                    {"name": "utils.py", "type": "file"}
                ]
            },
            {
                "name": "tests",
                "type": "directory",
                "children": [
                    {"name": "test_main.py", "type": "file"}
                ]
            },
            {"name": "README.md", "type": "file"}
        ]
    }
    
    with patch('tsap_mcp.resources.project._get_directory_structure', return_value=mock_structure):
        # Test with context object
        context = MagicMock()
        structure = await get_project_structure(context)
        
        assert isinstance(structure, dict)
        assert structure["name"] == "test-project"
        assert len(structure["children"]) == 3
        assert structure["children"][0]["name"] == "src"
        assert len(structure["children"][0]["children"]) == 2


@pytest.mark.asyncio
async def test_project_dependencies_resource():
    """Test project dependencies resource."""
    mock_dependencies = {
        "dependencies": [
            {"name": "mcp", "version": "0.1.0"},
            {"name": "fastapi", "version": "0.110.0"},
            {"name": "pydantic", "version": "2.5.0"}
        ],
        "dev_dependencies": [
            {"name": "pytest", "version": "7.4.3"},
            {"name": "black", "version": "24.2.0"}
        ]
    }
    
    with patch('tsap_mcp.resources.project._get_project_dependencies', return_value=mock_dependencies):
        # Test with context object
        context = MagicMock()
        dependencies = await get_project_dependencies(context)
        
        assert isinstance(dependencies, dict)
        assert len(dependencies["dependencies"]) == 3
        assert dependencies["dependencies"][0]["name"] == "mcp"
        assert len(dependencies["dev_dependencies"]) == 2


@pytest.mark.asyncio
async def test_project_stats_resource():
    """Test project stats resource."""
    mock_stats = {
        "files": 42,
        "lines": 9876,
        "languages": {
            "Python": {"files": 30, "lines": 8000},
            "Markdown": {"files": 10, "lines": 1500},
            "JSON": {"files": 2, "lines": 376}
        }
    }
    
    with patch('tsap_mcp.resources.project._get_project_stats', return_value=mock_stats):
        # Test with context object
        context = MagicMock()
        stats = await get_project_stats(context)
        
        assert isinstance(stats, dict)
        assert stats["files"] == 42
        assert stats["lines"] == 9876
        assert len(stats["languages"]) == 3
        assert stats["languages"]["Python"]["files"] == 30


@pytest.mark.asyncio
async def test_config_resource():
    """Test configuration resource."""
    mock_config = {
        "name": "TSAP MCP Server",
        "version": "0.1.0",
        "settings": {
            "max_search_results": 100,
            "enable_semantic_search": True,
            "default_language": "python"
        }
    }
    
    with patch('tsap_mcp.resources.config._load_config', return_value=mock_config):
        # Test with context object
        context = MagicMock()
        config = await get_config(context)
        
        assert isinstance(config, dict)
        assert config["name"] == "TSAP MCP Server"
        assert config["version"] == "0.1.0"
        assert config["settings"]["max_search_results"] == 100
        assert config["settings"]["enable_semantic_search"] is True


@pytest.mark.asyncio
async def test_semantic_corpus_resource():
    """Test semantic corpus resource."""
    mock_corpus_info = {
        "id": "code",
        "description": "Code corpus for semantic search",
        "embeddings": {
            "model": "nomic-embed-text-v2.0",
            "dimensions": 768,
            "count": 5432
        },
        "last_updated": "2023-07-15T10:30:00Z"
    }
    
    with patch('tsap_mcp.resources.semantic._get_corpus_info', return_value=mock_corpus_info):
        # Test with context object
        context = MagicMock()
        corpus_info = await get_semantic_corpus_info("code", context)
        
        assert isinstance(corpus_info, dict)
        assert corpus_info["id"] == "code"
        assert corpus_info["description"] == "Code corpus for semantic search"
        assert corpus_info["embeddings"]["model"] == "nomic-embed-text-v2.0"
        assert corpus_info["embeddings"]["count"] == 5432 