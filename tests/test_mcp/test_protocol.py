import pytest
import httpx
from typing import Dict, Any

from tsap.mcp.protocol import MCPRequest, MCPCommandType

# Test client that communicates directly with the server
class TestMCPClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8021"):
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
        }
        self._client = httpx.AsyncClient(
            base_url=self.base_url, 
            headers=self.headers, 
            timeout=30.0
        )

    async def close(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def send_request(self, request: MCPRequest) -> Dict[str, Any]:
        """Sends an MCPRequest to the server."""
        try:
            response = await self._client.post(
                "/mcp/", 
                json=request.model_dump()
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
            return {"error": {"code": f"HTTP_{e.response.status_code}", "message": e.response.text}}
        except Exception as e:
            print(f"Unexpected Error: {e}")
            return {"error": {"code": "CLIENT_ERROR", "message": str(e)}}

@pytest.mark.asyncio
async def test_mcp_server_health():
    """Test that server responds to basic requests."""
    async with TestMCPClient() as client:
        # Create a simple info request
        request = MCPRequest(
            command=MCPCommandType.INFO,
            args={}
        )
        
        response = await client.send_request(request)
        print(f"Response: {response}")
        
        # Basic validation - should have the same request_id
        assert "request_id" in response
        assert response["request_id"] == request.request_id
        assert response["command"] == MCPCommandType.INFO
        assert response["status"] == "success"

@pytest.mark.asyncio
async def test_ripgrep_search():
    """Test ripgrep search functionality."""
    async with TestMCPClient() as client:
        # Create a ripgrep search request
        request = MCPRequest(
            command=MCPCommandType.RIPGREP_SEARCH,
            args={
                "pattern": "TSAP",  # Search for the TSAP acronym
                "paths": ["src/"],  # Look in the src directory
                "case_sensitive": False,
                "file_patterns": ["*.py"],
                "context_lines": 1
            }
        )
        
        response = await client.send_request(request)
        print(f"Ripgrep response: {response}")
        
        # Basic validation
        assert "request_id" in response
        assert response["request_id"] == request.request_id
        assert response["command"] == MCPCommandType.RIPGREP_SEARCH
        assert response["status"] == "success"
        
        # Check response data
        assert "data" in response
        assert "matches" in response["data"]
        # We should find at least one match for "TSAP" in the codebase
        assert len(response["data"]["matches"]) > 0
