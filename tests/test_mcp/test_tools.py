import pytest
import asyncio
from unittest.mock import patch, MagicMock

from tsap_mcp.tools.search import search, search_regex, search_semantic, search_code
from tsap_mcp.tools.processing import process_text, extract_data, transform_text, format_data
from tsap_mcp.tools.analysis import analyze_code, analyze_text, analyze_data
from tsap_mcp.tools.visualization import generate_chart, generate_network_graph


@pytest.mark.asyncio
async def test_search_tool():
    """Test basic search functionality."""
    # Test basic search
    with patch('tsap_mcp.tools.search._perform_search', return_value=["result1", "result2"]):
        result = await search(query="test", path="src/", max_results=5)
        assert isinstance(result, list)
        assert len(result) == 2
        assert "result1" in result


@pytest.mark.asyncio
async def test_search_regex_tool():
    """Test regex search functionality."""
    # Test regex search
    with patch('tsap_mcp.tools.search._perform_regex_search', return_value=["match1", "match2"]):
        result = await search_regex(pattern=r"test\w+", path="src/", case_sensitive=False)
        assert isinstance(result, list)
        assert len(result) == 2
        assert "match1" in result


@pytest.mark.asyncio
async def test_search_semantic_tool():
    """Test semantic search functionality."""
    mock_results = [
        {"text": "This is a test", "score": 0.95},
        {"text": "Another test", "score": 0.85},
    ]
    
    with patch('tsap_mcp.tools.search._perform_semantic_search', return_value=mock_results):
        result = await search_semantic(query="test concept", corpus="code", top_k=2)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["score"] > result[1]["score"]
        assert "This is a test" == result[0]["text"]


@pytest.mark.asyncio
async def test_search_code_tool():
    """Test code search functionality."""
    mock_results = [
        {"path": "file1.py", "line": 10, "code": "def test_func():"},
        {"path": "file2.py", "line": 20, "code": "class TestClass:"},
    ]
    
    with patch('tsap_mcp.tools.search._perform_code_search', return_value=mock_results):
        result = await search_code(query="test", language="python", path="src/")
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["path"] == "file1.py"
        assert result[1]["code"] == "class TestClass:"


@pytest.mark.asyncio
async def test_process_text_tool():
    """Test text processing functionality."""
    with patch('tsap_mcp.tools.processing._perform_text_processing', return_value="Processed text"):
        result = await process_text(text="Raw text", operation="clean")
        assert isinstance(result, str)
        assert result == "Processed text"


@pytest.mark.asyncio
async def test_extract_data_tool():
    """Test data extraction functionality."""
    mock_data = {"name": "John", "age": 30}
    
    with patch('tsap_mcp.tools.processing._perform_data_extraction', return_value=mock_data):
        result = await extract_data(text="Name: John, Age: 30", pattern=r"Name: (\w+), Age: (\d+)")
        assert isinstance(result, dict)
        assert result["name"] == "John"
        assert result["age"] == 30


@pytest.mark.asyncio
async def test_analyze_code_tool():
    """Test code analysis functionality."""
    mock_analysis = {
        "complexity": 5,
        "issues": ["Missing docstring", "Too many arguments"],
        "quality_score": 0.75
    }
    
    with patch('tsap_mcp.tools.analysis._perform_code_analysis', return_value=mock_analysis):
        result = await analyze_code(code="def test(): pass", language="python")
        assert isinstance(result, dict)
        assert result["complexity"] == 5
        assert len(result["issues"]) == 2
        assert result["quality_score"] == 0.75


@pytest.mark.asyncio
async def test_generate_chart_tool():
    """Test chart generation functionality."""
    mock_chart_data = {
        "chart_type": "bar",
        "data": [1, 2, 3, 4, 5],
        "labels": ["A", "B", "C", "D", "E"],
        "image_data": "base64_encoded_image"
    }
    
    with patch('tsap_mcp.tools.visualization._generate_chart_image', return_value=mock_chart_data):
        result = await generate_chart(
            data=[1, 2, 3, 4, 5],
            labels=["A", "B", "C", "D", "E"],
            chart_type="bar",
            title="Test Chart"
        )
        assert isinstance(result, dict)
        assert result["chart_type"] == "bar"
        assert "image_data" in result
        assert len(result["data"]) == 5


@pytest.mark.asyncio
async def test_generate_network_graph_tool():
    """Test network graph generation functionality."""
    mock_graph_data = {
        "nodes": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
        "edges": [{"source": "A", "target": "B"}, {"source": "B", "target": "C"}],
        "image_data": "base64_encoded_image"
    }
    
    with patch('tsap_mcp.tools.visualization._generate_network_graph_image', return_value=mock_graph_data):
        result = await generate_network_graph(
            nodes=[{"id": "A"}, {"id": "B"}, {"id": "C"}],
            edges=[{"source": "A", "target": "B"}, {"source": "B", "target": "C"}],
            title="Test Graph"
        )
        assert isinstance(result, dict)
        assert len(result["nodes"]) == 3
        assert len(result["edges"]) == 2
        assert "image_data" in result 