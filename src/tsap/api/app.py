"""
Defines the main API router for the TSAP ToolAPI Server API.
"""

from fastapi import APIRouter

from tsap.utils.logging import logger
from tsap.version import get_version_info

# Import all router modules directly - we know they exist
from tsap.api.routes import core, composite, analysis, evolution, plugins

# Create main API router instance
api_router = APIRouter()

# Health check endpoint
@api_router.get("/health", tags=["System"])
async def health_check():
    return {"status": "ok"}

# Version endpoint
@api_router.get("/version", tags=["System"])
async def version():
    return get_version_info()

# Include all sub-routers directly
api_router.include_router(core.router, prefix="/core", tags=["Core Tools"])
api_router.include_router(composite.router, prefix="/composite", tags=["Composite Operations"])
api_router.include_router(analysis.router, prefix="/analysis", tags=["Analysis Tools"])
api_router.include_router(evolution.router, prefix="/evolution", tags=["Evolution"])
api_router.include_router(plugins.router, prefix="/plugins", tags=["Plugins"])

logger.info("API router created and configured.", component="api")