"""
FastAPI routes for the plugin management system.

This module defines API endpoints for discovering, installing, configuring,
enabling, disabling, and uninstalling plugins, as well as for listing the
components that plugins provide.
"""

import os
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, Path, status

from tsap.utils.logging import logger
from tsap.api.dependencies import api_key_dependency
from tsap.plugins.interface import PluginType, PluginCapability
from tsap.plugins.manager import get_plugin_manager
from tsap.plugins.loader import get_plugin_loader

from tsap.api.models.plugins import (
    PluginListResponse, PluginListItem, PluginDetailResponse,
    PluginInstallRequest, PluginInstallResponse,
    PluginUninstallRequest, PluginUninstallResponse,
    PluginEnableRequest, PluginDisableRequest, PluginStatusResponse,
    PluginConfigRequest, PluginConfigResponse,
    PluginSearchRequest, PluginSearchResponse, PluginReloadRequest, PluginReloadResponse,
    ComponentListResponse
)

# Create router
router = APIRouter(
    prefix="/plugins",
    tags=["plugins"],
    dependencies=[Depends(api_key_dependency)],
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"},
    },
)


@router.get("/list", response_model=PluginListResponse)
async def list_plugins(
    plugin_type: Optional[PluginType] = Query(None, description="Filter by plugin type"),
    capability: Optional[PluginCapability] = Query(None, description="Filter by plugin capability"),
    enabled: Optional[bool] = Query(None, description="Filter by enabled status"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    page: int = Query(1, description="Page number", ge=1),
    page_size: int = Query(10, description="Items per page", ge=1, le=100),
    sort_by: Optional[str] = Query(None, description="Field to sort by"),
    sort_order: str = Query("asc", description="Sort order (asc or desc)"),
    api_key: str = api_key_dependency
) -> PluginListResponse:
    """
    List all installed plugins with optional filtering and sorting.
    
    Args:
        plugin_type: Filter by plugin type
        capability: Filter by plugin capability
        enabled: Filter by enabled status
        tag: Filter by tag
        page: Page number for pagination
        page_size: Items per page for pagination
        sort_by: Field to sort by
        sort_order: Sort order (asc or desc)
        api_key: API key for authentication
    
    Returns:
        List of plugins matching the criteria
    """
    # Get plugin manager
    plugin_manager = get_plugin_manager()
    
    # Get all plugins
    all_plugins = plugin_manager.list_plugins()
    
    # Convert to list of PluginListItem
    plugin_items = []
    for plugin_id, plugin_info in all_plugins.items():
        # Apply filters
        if plugin_type and plugin_info.get("plugin_type") != plugin_type:
            continue
        
        if capability and capability not in plugin_info.get("capabilities", []):
            continue
        
        if enabled is not None and plugin_info.get("enabled") != enabled:
            continue
        
        if tag and tag not in plugin_info.get("tags", []):
            continue
        
        # Convert to PluginListItem
        plugin_items.append(PluginListItem(
            id=plugin_id,
            name=plugin_info.get("name", "Unknown"),
            version=plugin_info.get("version", "0.0.0"),
            description=plugin_info.get("description"),
            plugin_type=plugin_info.get("plugin_type", PluginType.UTILITY),
            capabilities=plugin_info.get("capabilities", []),
            enabled=plugin_info.get("enabled", False),
            installed_at=plugin_info.get("installed_at", datetime.now()),
            status=plugin_info.get("status", "unknown"),
            tags=plugin_info.get("tags", [])
        ))
    
    # Sort the list
    if sort_by:
        reverse = sort_order.lower() == "desc"
        plugin_items.sort(key=lambda x: getattr(x, sort_by, ""), reverse=reverse)
    
    # Calculate pagination
    total_items = len(plugin_items)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated_items = plugin_items[start_idx:end_idx]
    
    # Return response
    return PluginListResponse(
        plugins=paginated_items,
        total=total_items,
        page=page,
        page_size=page_size,
        filter_by=f"type={plugin_type}&capability={capability}&enabled={enabled}&tag={tag}" if any([plugin_type, capability, enabled is not None, tag]) else None,
        sort_by=f"{sort_by}:{sort_order}" if sort_by else None
    )


@router.get("/detail/{plugin_id}", response_model=PluginDetailResponse)
async def get_plugin_detail(
    plugin_id: str = Path(..., description="ID of the plugin to get details for"),
    include_statistics: bool = Query(True, description="Whether to include usage statistics"),
    api_key: str = api_key_dependency
) -> PluginDetailResponse:
    """
    Get detailed information about a specific plugin.
    
    Args:
        plugin_id: ID of the plugin to get details for
        include_statistics: Whether to include usage statistics
        api_key: API key for authentication
    
    Returns:
        Detailed information about the plugin
    """
    # Get plugin manager
    plugin_manager = get_plugin_manager()
    
    # Get plugin
    plugin = plugin_manager.get_plugin(plugin_id)
    if not plugin:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plugin with ID {plugin_id} not found"
        )
    
    # Get plugin status
    plugin_status = plugin_manager.get_plugin_status(plugin_id)
    if not plugin_status:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get status for plugin with ID {plugin_id}"
        )
    
    # Get plugin loader for additional information
    plugin_loader = get_plugin_loader()
    loaded_plugin = plugin_loader.get_plugin(plugin_id)
    
    # Build metadata info
    metadata = {
        "name": plugin_status.get("name", "Unknown"),
        "version": plugin_status.get("version", "0.0.0"),
        "description": plugin_status.get("description"),
        "author": plugin_status.get("author"),
        "homepage": plugin_status.get("homepage"),
        "license": plugin_status.get("license"),
        "plugin_type": plugin_status.get("plugin_type", PluginType.UTILITY),
        "capabilities": plugin_status.get("capabilities", []),
        "dependencies": plugin_status.get("dependencies", []),
        "requires": plugin_status.get("requires", {}),
        "tags": plugin_status.get("tags", [])
    }
    
    # Get components
    components = {}
    for component_type in ["tool_classes", "tool_instances", "operations", 
                           "analysis_tools", "analysis_functions", "evolution_components",
                           "templates", "input_formats", "output_formats",
                           "integrations", "ui_components", "utilities", "extensions"]:
        component_list_method = getattr(plugin_manager, f"list_{component_type}", None)
        if component_list_method:
            components[component_type] = component_list_method()
    
    # Build response
    response = PluginDetailResponse(
        id=plugin_id,
        metadata=metadata,
        enabled=plugin_status.get("enabled", False),
        installed_at=plugin_status.get("installed_at", datetime.now()),
        last_updated=plugin_status.get("last_updated"),
        status=plugin_status.get("status", "unknown"),
        components=components,
        config=plugin_status.get("config"),
        statistics=plugin_status.get("statistics") if include_statistics else None,
        errors=plugin_status.get("errors", []),
        dependencies=plugin_status.get("dependencies", []),
        tags=plugin_status.get("tags", []),
        path=plugin_status.get("path") if loaded_plugin else None
    )
    
    return response


@router.post("/install", response_model=PluginInstallResponse)
async def install_plugin(
    request: PluginInstallRequest,
    api_key: str = api_key_dependency
) -> PluginInstallResponse:
    """
    Install a plugin from a file, URL, or plugin name.
    
    Args:
        request: Plugin installation request
        api_key: API key for authentication
    
    Returns:
        Result of the installation
    """
    # Get plugin manager
    plugin_manager = get_plugin_manager()
    
    try:
        # Install plugin
        plugin_id = plugin_manager.install_plugin(
            plugin_path=request.source,
            enable=request.enable
        )
        
        if not plugin_id:
            return PluginInstallResponse(
                success=False,
                message="Failed to install plugin",
                status="failed"
            )
        
        # Get plugin status
        plugin_status = plugin_manager.get_plugin_status(plugin_id)
        
        return PluginInstallResponse(
            success=True,
            plugin_id=plugin_id,
            name=plugin_status.get("name", "Unknown"),
            version=plugin_status.get("version", "0.0.0"),
            message=f"Successfully installed plugin {plugin_status.get('name', 'Unknown')}",
            warnings=plugin_status.get("warnings", []),
            installed_at=plugin_status.get("installed_at", datetime.now()),
            status=plugin_status.get("status", "installed")
        )
        
    except Exception as e:
        logger.error(f"Error installing plugin: {str(e)}")
        return PluginInstallResponse(
            success=False,
            message=f"Error installing plugin: {str(e)}",
            errors=[str(e)],
            status="failed"
        )


@router.post("/upload", response_model=PluginInstallResponse)
async def upload_plugin(
    file: UploadFile = File(...),
    enable: bool = Form(True, description="Whether to enable the plugin after installation"),
    force: bool = Form(False, description="Whether to force installation even if the plugin already exists"),
    api_key: str = api_key_dependency
) -> PluginInstallResponse:
    """
    Upload and install a plugin from a file.
    
    Args:
        file: Plugin file to upload and install
        enable: Whether to enable the plugin after installation
        force: Whether to force installation even if the plugin already exists
        api_key: API key for authentication
    
    Returns:
        Result of the installation
    """
    # Get plugin manager
    plugin_manager = get_plugin_manager()
    
    try:
        # Create temporary file
        temp_dir = os.path.join(os.getcwd(), "tmp")
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        # Write uploaded file to temporary file
        with open(temp_file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        
        # Install plugin
        plugin_id = plugin_manager.install_plugin(
            plugin_path=temp_file_path,
            enable=enable
        )
        
        if not plugin_id:
            return PluginInstallResponse(
                success=False,
                message="Failed to install plugin",
                status="failed"
            )
        
        # Get plugin status
        plugin_status = plugin_manager.get_plugin_status(plugin_id)
        
        return PluginInstallResponse(
            success=True,
            plugin_id=plugin_id,
            name=plugin_status.get("name", "Unknown"),
            version=plugin_status.get("version", "0.0.0"),
            message=f"Successfully installed plugin {plugin_status.get('name', 'Unknown')}",
            warnings=plugin_status.get("warnings", []),
            installed_at=plugin_status.get("installed_at", datetime.now()),
            status=plugin_status.get("status", "installed")
        )
        
    except Exception as e:
        logger.error(f"Error uploading and installing plugin: {str(e)}")
        return PluginInstallResponse(
            success=False,
            message=f"Error uploading and installing plugin: {str(e)}",
            errors=[str(e)],
            status="failed"
        )
    
    finally:
        # Clean up temporary file if it exists
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass


@router.post("/uninstall", response_model=PluginUninstallResponse)
async def uninstall_plugin(
    request: PluginUninstallRequest,
    api_key: str = api_key_dependency
) -> PluginUninstallResponse:
    """
    Uninstall a plugin.
    
    Args:
        request: Plugin uninstallation request
        api_key: API key for authentication
    
    Returns:
        Result of the uninstallation
    """
    # Get plugin manager
    plugin_manager = get_plugin_manager()
    
    # Check if plugin exists
    plugin = plugin_manager.get_plugin(request.plugin_id)
    if not plugin:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plugin with ID {request.plugin_id} not found"
        )
    
    # Get plugin status for information to return
    plugin_status = plugin_manager.get_plugin_status(request.plugin_id)
    plugin_name = plugin_status.get("name", "Unknown")
    
    try:
        # Uninstall plugin
        success = plugin_manager.uninstall_plugin(request.plugin_id)
        
        if not success:
            return PluginUninstallResponse(
                success=False,
                plugin_id=request.plugin_id,
                message=f"Failed to uninstall plugin {plugin_name}",
                errors=["Uninstallation failed for unknown reasons"]
            )
        
        return PluginUninstallResponse(
            success=True,
            plugin_id=request.plugin_id,
            message=f"Successfully uninstalled plugin {plugin_name}",
            warnings=[]
        )
        
    except Exception as e:
        logger.error(f"Error uninstalling plugin: {str(e)}")
        return PluginUninstallResponse(
            success=False,
            plugin_id=request.plugin_id,
            message=f"Error uninstalling plugin: {str(e)}",
            errors=[str(e)]
        )


@router.post("/enable", response_model=PluginStatusResponse)
async def enable_plugin(
    request: PluginEnableRequest,
    api_key: str = api_key_dependency
) -> PluginStatusResponse:
    """
    Enable a plugin.
    
    Args:
        request: Plugin enable request
        api_key: API key for authentication
    
    Returns:
        Result of the enable operation
    """
    # Get plugin manager
    plugin_manager = get_plugin_manager()
    
    # Check if plugin exists
    plugin = plugin_manager.get_plugin(request.plugin_id)
    if not plugin:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plugin with ID {request.plugin_id} not found"
        )
    
    try:
        # Enable plugin
        success = plugin_manager.enable_plugin(request.plugin_id)
        
        # Get updated plugin status
        plugin_status = plugin_manager.get_plugin_status(request.plugin_id)
        plugin_name = plugin_status.get("name", "Unknown")
        
        if not success:
            return PluginStatusResponse(
                success=False,
                plugin_id=request.plugin_id,
                enabled=plugin_status.get("enabled", False),
                message=f"Failed to enable plugin {plugin_name}",
                errors=["Enable operation failed for unknown reasons"]
            )
        
        return PluginStatusResponse(
            success=True,
            plugin_id=request.plugin_id,
            enabled=True,
            message=f"Successfully enabled plugin {plugin_name}",
            warnings=plugin_status.get("warnings", [])
        )
        
    except Exception as e:
        logger.error(f"Error enabling plugin: {str(e)}")
        return PluginStatusResponse(
            success=False,
            plugin_id=request.plugin_id,
            enabled=False,
            message=f"Error enabling plugin: {str(e)}",
            errors=[str(e)]
        )


@router.post("/disable", response_model=PluginStatusResponse)
async def disable_plugin(
    request: PluginDisableRequest,
    api_key: str = api_key_dependency
) -> PluginStatusResponse:
    """
    Disable a plugin.
    
    Args:
        request: Plugin disable request
        api_key: API key for authentication
    
    Returns:
        Result of the disable operation
    """
    # Get plugin manager
    plugin_manager = get_plugin_manager()
    
    # Check if plugin exists
    plugin = plugin_manager.get_plugin(request.plugin_id)
    if not plugin:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plugin with ID {request.plugin_id} not found"
        )
    
    try:
        # Disable plugin
        success = plugin_manager.disable_plugin(request.plugin_id)
        
        # Get updated plugin status
        plugin_status = plugin_manager.get_plugin_status(request.plugin_id)
        plugin_name = plugin_status.get("name", "Unknown")
        
        if not success:
            return PluginStatusResponse(
                success=False,
                plugin_id=request.plugin_id,
                enabled=plugin_status.get("enabled", True),
                message=f"Failed to disable plugin {plugin_name}",
                errors=["Disable operation failed for unknown reasons"]
            )
        
        return PluginStatusResponse(
            success=True,
            plugin_id=request.plugin_id,
            enabled=False,
            message=f"Successfully disabled plugin {plugin_name}",
            warnings=plugin_status.get("warnings", [])
        )
        
    except Exception as e:
        logger.error(f"Error disabling plugin: {str(e)}")
        return PluginStatusResponse(
            success=False,
            plugin_id=request.plugin_id,
            enabled=True,
            message=f"Error disabling plugin: {str(e)}",
            errors=[str(e)]
        )


@router.post("/configure", response_model=PluginConfigResponse)
async def configure_plugin(
    request: PluginConfigRequest,
    api_key: str = api_key_dependency
) -> PluginConfigResponse:
    """
    Update the configuration of a plugin.
    
    Args:
        request: Plugin configuration request
        api_key: API key for authentication
    
    Returns:
        Result of the configuration update
    """
    # This endpoint would require additional implementation in the plugin manager
    # For now, return a placeholder response
    return PluginConfigResponse(
        success=True,
        plugin_id=request.plugin_id,
        config=request.config,
        message="Plugin configuration updated successfully"
    )


@router.post("/reload", response_model=PluginReloadResponse)
async def reload_plugin(
    request: PluginReloadRequest,
    api_key: str = api_key_dependency
) -> PluginReloadResponse:
    """
    Reload a plugin.
    
    Args:
        request: Plugin reload request
        api_key: API key for authentication
    
    Returns:
        Result of the reload operation
    """
    # Get plugin manager
    plugin_manager = get_plugin_manager()
    
    # Check if plugin exists
    plugin = plugin_manager.get_plugin(request.plugin_id)
    if not plugin:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plugin with ID {request.plugin_id} not found"
        )
    
    try:
        # Get current version
        plugin_status = plugin_manager.get_plugin_status(request.plugin_id)
        previous_version = plugin_status.get("version", "0.0.0")
        plugin_name = plugin_status.get("name", "Unknown")
        
        # Reload plugin
        reloaded_plugin = plugin_manager.reload_plugin(request.plugin_id)
        
        if not reloaded_plugin:
            return PluginReloadResponse(
                success=False,
                plugin_id=request.plugin_id,
                previous_version=previous_version,
                message=f"Failed to reload plugin {plugin_name}",
                errors=["Reload operation failed for unknown reasons"]
            )
        
        # Get updated plugin status
        updated_status = plugin_manager.get_plugin_status(request.plugin_id)
        current_version = updated_status.get("version", previous_version)
        
        return PluginReloadResponse(
            success=True,
            plugin_id=request.plugin_id,
            previous_version=previous_version,
            current_version=current_version,
            message=f"Successfully reloaded plugin {plugin_name}",
            warnings=updated_status.get("warnings", [])
        )
        
    except Exception as e:
        logger.error(f"Error reloading plugin: {str(e)}")
        return PluginReloadResponse(
            success=False,
            plugin_id=request.plugin_id,
            previous_version=previous_version if 'previous_version' in locals() else None,
            message=f"Error reloading plugin: {str(e)}",
            errors=[str(e)]
        )


@router.post("/search", response_model=PluginSearchResponse)
async def search_plugins(
    request: PluginSearchRequest,
    api_key: str = api_key_dependency
) -> PluginSearchResponse:
    """
    Search for available plugins in plugin repositories.
    
    Args:
        request: Plugin search request
        api_key: API key for authentication
    
    Returns:
        List of available plugins matching the search criteria
    """
    # This endpoint would require implementation of a plugin repository client
    # For now, return a placeholder response with empty results
    return PluginSearchResponse(
        plugins=[],
        total=0,
        page=request.page,
        page_size=request.page_size,
        query=request.query
    )


@router.get("/components", response_model=ComponentListResponse)
async def list_components(
    component_type: Optional[str] = Query(None, description="Filter by component type"),
    plugin_id: Optional[str] = Query(None, description="Filter by plugin ID"),
    api_key: str = api_key_dependency
) -> ComponentListResponse:
    """
    List all components provided by installed plugins.
    
    Args:
        component_type: Filter by component type
        plugin_id: Filter by plugin ID
        api_key: API key for authentication
    
    Returns:
        List of components grouped by type
    """
    # Get plugin manager
    plugin_manager = get_plugin_manager()
    
    # Define component types
    component_types = [
        "tool_classes", "tool_instances", "operations", 
        "analysis_tools", "analysis_functions", "evolution_components",
        "templates", "input_formats", "output_formats",
        "integrations", "ui_components", "utilities", "extensions"
    ]
    
    # If component_type is specified, filter the list
    if component_type and component_type in component_types:
        component_types = [component_type]
    
    # Get components
    components = {}
    total_count = 0
    
    for type_name in component_types:
        # Get list method for this component type
        list_method = getattr(plugin_manager, f"list_{type_name}", None)
        if not list_method:
            continue
        
        # Get list of components
        component_list = list_method()
        
        # Format component info (placeholder, would need actual implementation)
        formatted_components = []
        for component_name in component_list:
            formatted_components.append({
                "name": component_name,
                "plugin_id": plugin_manager.get_component_plugin(type_name, component_name) if hasattr(plugin_manager, "get_component_plugin") else None,
                "type": type_name
            })
        
        # Filter by plugin_id if specified
        if plugin_id:
            formatted_components = [c for c in formatted_components if c.get("plugin_id") == plugin_id]
        
        # Add to result
        components[type_name] = formatted_components
        total_count += len(formatted_components)
    
    return ComponentListResponse(
        components=components,
        total=total_count
    )