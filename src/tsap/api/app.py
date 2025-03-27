"""
Main FastAPI application for the TSAP MCP Server API.
"""

from fastapi import FastAPI, APIRouter, Depends, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
from typing import Dict, Any, Optional, List

from tsap.utils.logging import logger
from tsap.config import get_config
from tsap.version import get_version_info, API_VERSION
from tsap.api.middleware.auth import verify_api_key
from tsap.api.middleware.error import add_error_handlers
from tsap.api.middleware.logging import add_logging_middleware


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application
    """
    config = get_config()
    
    # Create FastAPI app with metadata
    app = FastAPI(
        title="TSAP MCP Server API",
        description="REST API for Text Search and Processing Model Context Protocol Server",
        version=API_VERSION,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add custom middleware
    add_error_handlers(app)
    add_logging_middleware(app)
    
    # Create main API router
    api_router = APIRouter(prefix="/api")
    
    # Health check endpoint
    @api_router.get("/health", tags=["System"])
    async def health_check():
        return {"status": "ok"}
    
    # Version endpoint
    @api_router.get("/version", tags=["System"])
    async def version():
        return get_version_info()
    
    # Include all route modules
    api_router.include_router(_create_core_router(), prefix="/core", tags=["Core Tools"])
    api_router.include_router(_create_composite_router(), prefix="/composite", tags=["Composite Operations"])
    api_router.include_router(_create_analysis_router(), prefix="/analysis", tags=["Analysis Tools"])
    api_router.include_router(_create_evolution_router(), prefix="/evolution", tags=["Evolution"])
    api_router.include_router(_create_plugins_router(), prefix="/plugins", tags=["Plugins"])
    
    # Add API router to app
    app.include_router(api_router)
    
    logger.info(
        f"API initialized with {len(config.server.cors_origins)} allowed origins",
        component="api"
    )
    
    return app


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


# Global FastAPI app instance
app = create_app()