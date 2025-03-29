"""
Defines the main API router for the TSAP MCP Server API.
"""

from fastapi import APIRouter

from tsap.utils.logging import logger
from tsap.version import get_version_info

# Helper functions to create sub-routers (moved from create_app)
def _create_core_router() -> APIRouter:
    """Create and return the core tools router."""
    try:
        from tsap.api.routes.core import router as core_router
        return core_router
    except ImportError:
        logger.warning("Core router not found, using empty router", component="api")
        return APIRouter()

def _create_composite_router() -> APIRouter:
    """Create and return the composite operations router."""
    try:
        from tsap.api.routes.composite import router as composite_router
        return composite_router
    except ImportError:
        logger.warning("Composite router not found, using empty router", component="api")
        return APIRouter()

def _create_analysis_router() -> APIRouter:
    """Create and return the analysis tools router."""
    try:
        from tsap.api.routes.analysis import router as analysis_router
        return analysis_router
    except ImportError:
        logger.warning("Analysis router not found, using empty router", component="api")
        return APIRouter()

def _create_evolution_router() -> APIRouter:
    """Create and return the evolution features router."""
    try:
        from tsap.api.routes.evolution import router as evolution_router
        return evolution_router
    except ImportError:
        logger.warning("Evolution router not found, using empty router", component="api")
        return APIRouter()

def _create_plugins_router() -> APIRouter:
    """Create and return the plugins management router."""
    try:
        from tsap.api.routes.plugins import router as plugins_router
        return plugins_router
    except ImportError:
        logger.warning("Plugins router not found, using empty router", component="api")
        return APIRouter()

# --- Main API Router Definition ---

# Create main API router instance
# Note: Prefix is applied when including this router in the main app (server.py)
api_router = APIRouter()

# Health check endpoint
@api_router.get("/health", tags=["System"])
async def health_check():
    return {"status": "ok"}

# Version endpoint
@api_router.get("/version", tags=["System"])
async def version():
    return get_version_info()

# Include all sub-route modules
api_router.include_router(_create_core_router(), prefix="/core", tags=["Core Tools"])
api_router.include_router(_create_composite_router(), prefix="/composite", tags=["Composite Operations"])
api_router.include_router(_create_analysis_router(), prefix="/analysis", tags=["Analysis Tools"])
api_router.include_router(_create_evolution_router(), prefix="/evolution", tags=["Evolution"])
api_router.include_router(_create_plugins_router(), prefix="/plugins", tags=["Plugins"])

logger.info("API router created and configured.", component="api")