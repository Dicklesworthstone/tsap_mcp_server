"""
Tests for the integration between the original TSAP server and the new ToolAPI server.
"""

import pytest
import uuid
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import FastAPI

from src.tsap.toolapi_integration import ToolAPIRequest, mount_toolapi_server, execute_toolapi_command
from src.tsap.toolapi_mount import ToolAPIServerProcess, ToolAPIServerProxy, execute_toolapi_command


class TestToolAPIServerProcess:
    """Test ToolAPI server process handling."""
    
    def test_init(self):
        """Test process initialization."""
        process = ToolAPIServerProcess()
        assert process.process is None
        assert process.running is False
    
    @pytest.mark.asyncio
    @patch('subprocess.Popen')
    @patch('asyncio.sleep')
    async def test_start_success(self, mock_sleep, mock_popen):
        """Test successful start of the process."""
        # Setup mock
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Process still running
        mock_popen.return_value = mock_process
        
        # Start the process
        process = ToolAPIServerProcess()
        await process.start()
        
        # Assertions
        mock_popen.assert_called_once()
        mock_sleep.assert_called_once()
        assert process.running is True
        assert process.process is mock_process


class TestToolAPIServerProxy:
    """Test ToolAPI server proxy."""
    
    def test_init(self):
        """Test proxy initialization."""
        proxy = ToolAPIServerProxy()
        assert isinstance(proxy.server_process, ToolAPIServerProcess)
        assert proxy._session is None
    
    @pytest.mark.asyncio
    @patch('src.tsap.toolapi_mount.ToolAPIServerProcess.start')
    async def test_start(self, mock_start):
        """Test proxy start."""
        proxy = ToolAPIServerProxy()
        await proxy.start()
        mock_start.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('src.tsap.toolapi_mount.ToolAPIServerProcess.stop')
    async def test_stop(self, mock_stop):
        """Test proxy stop."""
        proxy = ToolAPIServerProxy()
        await proxy.stop()
        mock_stop.assert_called_once()


@pytest.mark.asyncio
async def test_execute_toolapi_command():
    """Test command execution function."""
    with patch('src.tsap.toolapi_mount._toolapi_server') as mock_toolapi_server:
        mock_toolapi_server.execute = AsyncMock(return_value={"result": "test"})
        
        result = await execute_toolapi_command("test_command", {"arg": "value"})
        
        mock_toolapi_server.execute.assert_called_once_with("test_command", {"arg": "value"})
        assert result == {"result": "test"}


@pytest.mark.asyncio
async def test_execute_toolapi_command():
    """Test native command execution endpoint."""
    with patch('src.tsap.toolapi_integration.execute_toolapi_command') as mock_execute:
        mock_execute.return_value = {"result": "test"}
        
        request = ToolAPIRequest(
            command="test_command",
            args={"arg": "value"},
            request_id=str(uuid.uuid4())
        )
        
        response = await execute_toolapi_command(request)
        
        mock_execute.assert_called_once_with("test_command", {"arg": "value"})
        assert response["status"] == "success"
        assert response["data"] == {"result": "test"}
        assert response["error"] is None
        assert response["request_id"] == request.request_id


def test_mount_toolapi_server():
    """Test mounting ToolAPI server to FastAPI."""
    app = FastAPI()
    
    with patch('src.tsap.toolapi_integration.toolapi_router') as mock_router:
        with patch('src.tsap.toolapi_integration.get_toolapi_startup_handler') as mock_startup:
            with patch('src.tsap.toolapi_integration.get_toolapi_shutdown_handler') as mock_shutdown:
                mock_startup.return_value = "startup_handler"
                mock_shutdown.return_value = "shutdown_handler"
                
                mount_toolapi_server(app)
                
                # Check that router was included
                app.include_router.assert_called_once_with(mock_router)
                
                # Check that event handlers were added
                app.add_event_handler.assert_any_call("startup", "startup_handler")
                app.add_event_handler.assert_any_call("shutdown", "shutdown_handler") 