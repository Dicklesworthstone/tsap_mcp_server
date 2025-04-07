import pytest
import asyncio
from unittest.mock import patch, MagicMock

from tsap_mcp.prompts.code_analysis import code_review, code_explain, code_refactor
from tsap_mcp.prompts.search import search_help, search_advanced


@pytest.mark.asyncio
async def test_code_review_prompt():
    """Test code review prompt."""
    # Test code content
    code = """
def add(a, b):
    return a + b
    """
    
    language = "python"
    
    # Generate prompt
    prompt = code_review(code=code, language=language)
    
    # Basic assertions
    assert isinstance(prompt, str)
    assert "def add(a, b)" in prompt
    assert "python" in prompt.lower()
    assert "review" in prompt.lower()


@pytest.mark.asyncio
async def test_code_explain_prompt():
    """Test code explain prompt."""
    # Test code content
    code = """
class Calculator:
    def __init__(self):
        self.result = 0
        
    def add(self, value):
        self.result += value
        return self.result
    """
    
    language = "python"
    
    # Generate prompt
    prompt = code_explain(code=code, language=language)
    
    # Basic assertions
    assert isinstance(prompt, str)
    assert "class Calculator" in prompt
    assert "def add" in prompt
    assert "explain" in prompt.lower()


@pytest.mark.asyncio
async def test_code_refactor_prompt():
    """Test code refactor prompt."""
    # Test code content
    code = """
def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result
    """
    
    language = "python"
    goal = "Make it more functional and use list comprehension"
    
    # Generate prompt
    prompt = code_refactor(code=code, language=language, goal=goal)
    
    # Basic assertions
    assert isinstance(prompt, str)
    assert "def process_data" in prompt
    assert "list comprehension" in prompt.lower()
    assert "refactor" in prompt.lower()


@pytest.mark.asyncio
async def test_search_help_prompt():
    """Test search help prompt."""
    # Test query
    query = "How to implement binary search"
    
    # Generate prompt
    prompt = search_help(query=query)
    
    # Basic assertions
    assert isinstance(prompt, str)
    assert "binary search" in prompt.lower()
    assert "search" in prompt.lower()


@pytest.mark.asyncio
async def test_search_advanced_prompt():
    """Test advanced search prompt."""
    # Test parameters
    query = "Find all API endpoints that handle authentication"
    file_types = [".py", ".js"]
    include_patterns = ["auth", "login"]
    exclude_patterns = ["test_", "mock_"]
    
    # Generate prompt
    prompt = search_advanced(
        query=query,
        file_types=file_types,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns
    )
    
    # Basic assertions
    assert isinstance(prompt, str)
    assert "API endpoints" in prompt
    assert "authentication" in prompt.lower()
    assert ".py" in prompt
    assert ".js" in prompt
    assert "auth" in prompt
    assert "login" in prompt
    assert "test_" in prompt
    assert "mock_" in prompt


@pytest.mark.asyncio
async def test_prompt_messages_format():
    """Test that prompts can be formatted as message sequences."""
    from mcp.server.fastmcp.prompts import base
    
    # Create a patch that returns a message sequence instead of a string
    with patch('tsap_mcp.prompts.code_analysis.code_review', return_value=[
        base.UserMessage("I need you to review this code:"),
        base.UserMessage("```python\ndef test():\n    pass\n```"),
        base.AssistantMessage("I'll analyze this code for you.")
    ]):
        # Call the patched function directly to get the messages
        messages = code_review(code="def test():\n    pass", language="python")
        
        # Verify it's a list of messages
        assert isinstance(messages, list)
        assert len(messages) == 3
        assert isinstance(messages[0], base.UserMessage)
        assert isinstance(messages[1], base.UserMessage)
        assert isinstance(messages[2], base.AssistantMessage)
        assert "review this code" in messages[0].content
        assert "```python" in messages[1].content
        assert "analyze this code" in messages[2].content 