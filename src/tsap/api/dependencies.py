"""
Dependencies for FastAPI routes in the TSAP MCP Server API.
"""

from fastapi import Depends, HTTPException, Header, Request, status
from typing import Optional
from contextlib import asynccontextmanager

from tsap.utils.logging import logger
from tsap.config import get_config
from tsap.performance_mode import get_performance_mode, set_performance_mode


async def get_api_key(api_key: str = Header(None)) -> str:
    """
    Validate the API key from the header.
    
    Args:
        api_key: API key from X-API-Key header
        
    Returns:
        Validated API key
        
    Raises:
        HTTPException: If API key is missing or invalid
    """
    config = get_config()
    
    # If auth is not enabled, check the global config extra field
    if not config.extra.get("api_auth_enabled", False):
        return None
    
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is required",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    # In a real implementation, this would validate against a database or configuration
    # For now, we just check that something was provided
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    return api_key


async def get_performance_mode_from_request(
    request: Request,
    mode: Optional[str] = None
) -> str:
    """
    Get performance mode from query parameters or use the default.
    
    Args:
        request: FastAPI request object
        mode: Performance mode from query parameters
        
    Returns:
        Valid performance mode string
    """
    config = get_config()  # noqa: F841
    
    # If mode is provided in query parameters, use it
    if mode is not None:
        valid_modes = ["fast", "standard", "deep"]
        if mode.lower() not in valid_modes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid performance mode. Must be one of: {', '.join(valid_modes)}"
            )
        return mode.lower()
    
    # Otherwise use the current mode
    return get_performance_mode()


@asynccontextmanager
async def with_performance_mode(request: Request, mode: Optional[str] = None):
    """
    Context manager that sets the performance mode for the duration of the request.
    
    Args:
        request: FastAPI request object
        mode: Performance mode to use
    """
    mode = await get_performance_mode_from_request(request, mode)
    original_mode = get_performance_mode()
    
    try:
        set_performance_mode(mode)
        logger.debug(f"Set performance mode to {mode} for request", component="api")
        yield mode
    finally:
        set_performance_mode(original_mode)
        logger.debug(f"Restored performance mode to {original_mode}", component="api")


def performance_mode_dependency(mode: Optional[str] = None):
    """
    Dependency that sets the performance mode for the duration of the request.
    
    Args:
        mode: Performance mode to use (from query parameters)
        
    Returns:
        Performance mode that was set
    """
    async def dependency(request: Request):
        async with with_performance_mode(request, mode) as current_mode:
            return current_mode
            
    return Depends(dependency)


# Common dependencies
performance_mode = performance_mode_dependency()
api_key_dependency = Depends(get_api_key)