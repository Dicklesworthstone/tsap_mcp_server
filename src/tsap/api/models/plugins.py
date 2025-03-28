"""
Pydantic models for the plugin system API endpoints.

This module defines the request and response models for the plugin management API,
allowing for discovery, installation, configuration, and registration of plugins.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime

from tsap.plugins.interface import PluginType, PluginCapability


class PluginMetadataInfo(BaseModel):
    """Model representing plugin metadata information."""
    name: str = Field(..., description="Name of the plugin")
    version: str = Field(..., description="Version of the plugin")
    description: Optional[str] = Field(None, description="Description of the plugin")
    author: Optional[str] = Field(None, description="Author of the plugin")
    homepage: Optional[str] = Field(None, description="Homepage URL for the plugin")
    license: Optional[str] = Field(None, description="License of the plugin")
    plugin_type: PluginType = Field(..., description="Type of the plugin")
    capabilities: List[PluginCapability] = Field(default_factory=list, description="Capabilities provided by the plugin")
    dependencies: List[str] = Field(default_factory=list, description="Plugin dependencies")
    requires: Dict[str, str] = Field(default_factory=dict, description="Required package dependencies")
    tags: List[str] = Field(default_factory=list, description="Tags associated with the plugin")


class PluginListItem(BaseModel):
    """Model representing a plugin in a list."""
    id: str = Field(..., description="Unique identifier of the plugin")
    name: str = Field(..., description="Name of the plugin")
    version: str = Field(..., description="Version of the plugin")
    description: Optional[str] = Field(None, description="Description of the plugin")
    plugin_type: PluginType = Field(..., description="Type of the plugin")
    capabilities: List[PluginCapability] = Field(default_factory=list, description="Capabilities provided by the plugin")
    enabled: bool = Field(False, description="Whether the plugin is enabled")
    installed_at: datetime = Field(..., description="Timestamp when the plugin was installed")
    status: str = Field(..., description="Status of the plugin (installed, registered, initialized, etc.)")
    tags: List[str] = Field(default_factory=list, description="Tags associated with the plugin")


class PluginDetailResponse(BaseModel):
    """Model representing detailed information about a plugin."""
    id: str = Field(..., description="Unique identifier of the plugin")
    metadata: PluginMetadataInfo = Field(..., description="Plugin metadata")
    enabled: bool = Field(False, description="Whether the plugin is enabled")
    installed_at: datetime = Field(..., description="Timestamp when the plugin was installed")
    last_updated: Optional[datetime] = Field(None, description="Timestamp when the plugin was last updated")
    status: str = Field(..., description="Status of the plugin (installed, registered, initialized, etc.)")
    components: Dict[str, List[str]] = Field(default_factory=dict, description="Components provided by the plugin")
    config: Optional[Dict[str, Any]] = Field(None, description="Plugin configuration")
    statistics: Optional[Dict[str, Any]] = Field(None, description="Plugin usage statistics")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Errors associated with the plugin")
    dependencies: List[str] = Field(default_factory=list, description="Plugin dependencies")
    tags: List[str] = Field(default_factory=list, description="Tags associated with the plugin")
    path: Optional[str] = Field(None, description="File path to the plugin")


class PluginListResponse(BaseModel):
    """Model representing a list of plugins."""
    plugins: List[PluginListItem] = Field(..., description="List of plugins")
    total: int = Field(..., description="Total number of plugins")
    page: int = Field(1, description="Current page number")
    page_size: int = Field(..., description="Number of plugins per page")
    filter_by: Optional[str] = Field(None, description="Filter applied to the list")
    sort_by: Optional[str] = Field(None, description="Sort applied to the list")


class PluginInstallRequest(BaseModel):
    """Model representing a request to install a plugin."""
    source: str = Field(..., description="Source of the plugin (file path, URL, or plugin name)")
    source_type: str = Field("auto", description="Type of the source (file, url, name, git)")
    enable: bool = Field(True, description="Whether to enable the plugin after installation")
    force: bool = Field(False, description="Whether to force installation even if the plugin already exists")
    config: Optional[Dict[str, Any]] = Field(None, description="Initial configuration for the plugin")
    
    @validator('source_type')
    def validate_source_type(cls, v):
        """Validate that the source type is valid."""
        allowed_types = {'file', 'url', 'name', 'git', 'auto'}
        if v not in allowed_types:
            raise ValueError(f"Source type must be one of: {', '.join(allowed_types)}")
        return v


class PluginInstallResponse(BaseModel):
    """Model representing the response to a plugin installation request."""
    success: bool = Field(..., description="Whether the installation was successful")
    plugin_id: Optional[str] = Field(None, description="ID of the installed plugin if successful")
    name: Optional[str] = Field(None, description="Name of the installed plugin if successful")
    version: Optional[str] = Field(None, description="Version of the installed plugin if successful")
    message: str = Field(..., description="Message describing the result of the installation")
    warnings: List[str] = Field(default_factory=list, description="Warnings that occurred during installation")
    errors: List[str] = Field(default_factory=list, description="Errors that occurred during installation")
    installed_at: Optional[datetime] = Field(None, description="Timestamp when the plugin was installed")
    status: str = Field(..., description="Status of the plugin after installation")


class PluginUninstallRequest(BaseModel):
    """Model representing a request to uninstall a plugin."""
    plugin_id: str = Field(..., description="ID of the plugin to uninstall")
    force: bool = Field(False, description="Whether to force uninstallation even if other plugins depend on it")
    remove_data: bool = Field(False, description="Whether to remove all data associated with the plugin")


class PluginUninstallResponse(BaseModel):
    """Model representing the response to a plugin uninstallation request."""
    success: bool = Field(..., description="Whether the uninstallation was successful")
    plugin_id: str = Field(..., description="ID of the uninstalled plugin")
    message: str = Field(..., description="Message describing the result of the uninstallation")
    warnings: List[str] = Field(default_factory=list, description="Warnings that occurred during uninstallation")
    errors: List[str] = Field(default_factory=list, description="Errors that occurred during uninstallation")
    dependent_plugins: List[str] = Field(default_factory=list, description="Plugins that depend on the uninstalled plugin")


class PluginEnableRequest(BaseModel):
    """Model representing a request to enable a plugin."""
    plugin_id: str = Field(..., description="ID of the plugin to enable")


class PluginDisableRequest(BaseModel):
    """Model representing a request to disable a plugin."""
    plugin_id: str = Field(..., description="ID of the plugin to disable")


class PluginStatusResponse(BaseModel):
    """Model representing the status of a plugin enable/disable operation."""
    success: bool = Field(..., description="Whether the operation was successful")
    plugin_id: str = Field(..., description="ID of the plugin")
    enabled: bool = Field(..., description="Whether the plugin is enabled")
    message: str = Field(..., description="Message describing the result of the operation")
    warnings: List[str] = Field(default_factory=list, description="Warnings that occurred during the operation")
    errors: List[str] = Field(default_factory=list, description="Errors that occurred during the operation")


class PluginConfigRequest(BaseModel):
    """Model representing a request to update plugin configuration."""
    plugin_id: str = Field(..., description="ID of the plugin to configure")
    config: Dict[str, Any] = Field(..., description="Configuration to apply to the plugin")
    merge: bool = Field(True, description="Whether to merge with existing configuration or replace it entirely")


class PluginConfigResponse(BaseModel):
    """Model representing the response to a plugin configuration update request."""
    success: bool = Field(..., description="Whether the configuration update was successful")
    plugin_id: str = Field(..., description="ID of the plugin")
    config: Dict[str, Any] = Field(..., description="Current configuration of the plugin")
    message: str = Field(..., description="Message describing the result of the operation")
    warnings: List[str] = Field(default_factory=list, description="Warnings that occurred during the operation")
    errors: List[str] = Field(default_factory=list, description="Errors that occurred during the operation")


class PluginSearchRequest(BaseModel):
    """Model representing a request to search for available plugins."""
    query: Optional[str] = Field(None, description="Search query")
    plugin_type: Optional[PluginType] = Field(None, description="Filter by plugin type")
    capability: Optional[PluginCapability] = Field(None, description="Filter by plugin capability")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    page: int = Field(1, description="Page number")
    page_size: int = Field(10, description="Number of plugins per page")
    sort_by: Optional[str] = Field(None, description="Field to sort by")
    sort_order: str = Field("asc", description="Sort order (asc or desc)")
    
    @validator('sort_order')
    def validate_sort_order(cls, v):
        """Validate that the sort order is valid."""
        if v not in {'asc', 'desc'}:
            raise ValueError("Sort order must be either 'asc' or 'desc'")
        return v


class AvailablePlugin(BaseModel):
    """Model representing an available plugin in a registry or repository."""
    name: str = Field(..., description="Name of the plugin")
    version: str = Field(..., description="Version of the plugin")
    description: Optional[str] = Field(None, description="Description of the plugin")
    author: Optional[str] = Field(None, description="Author of the plugin")
    homepage: Optional[str] = Field(None, description="Homepage URL for the plugin")
    license: Optional[str] = Field(None, description="License of the plugin")
    plugin_type: PluginType = Field(..., description="Type of the plugin")
    capabilities: List[PluginCapability] = Field(default_factory=list, description="Capabilities provided by the plugin")
    tags: List[str] = Field(default_factory=list, description="Tags associated with the plugin")
    downloads: Optional[int] = Field(None, description="Number of downloads")
    rating: Optional[float] = Field(None, description="Rating of the plugin")
    published_at: Optional[datetime] = Field(None, description="Timestamp when the plugin was published")
    repository_url: Optional[str] = Field(None, description="URL to the plugin repository")
    is_installed: bool = Field(False, description="Whether the plugin is already installed")
    installed_version: Optional[str] = Field(None, description="Version of the plugin that is installed")


class PluginSearchResponse(BaseModel):
    """Model representing the response to a plugin search request."""
    plugins: List[AvailablePlugin] = Field(..., description="List of available plugins")
    total: int = Field(..., description="Total number of plugins matching the search")
    page: int = Field(1, description="Current page number")
    page_size: int = Field(..., description="Number of plugins per page")
    query: Optional[str] = Field(None, description="Search query used")


class PluginReloadRequest(BaseModel):
    """Model representing a request to reload a plugin."""
    plugin_id: str = Field(..., description="ID of the plugin to reload")


class PluginReloadResponse(BaseModel):
    """Model representing the response to a plugin reload request."""
    success: bool = Field(..., description="Whether the reload was successful")
    plugin_id: str = Field(..., description="ID of the plugin")
    previous_version: Optional[str] = Field(None, description="Previous version of the plugin")
    current_version: Optional[str] = Field(None, description="Current version of the plugin")
    message: str = Field(..., description="Message describing the result of the reload")
    warnings: List[str] = Field(default_factory=list, description="Warnings that occurred during reload")
    errors: List[str] = Field(default_factory=list, description="Errors that occurred during reload")


class ComponentListResponse(BaseModel):
    """Model representing a list of components provided by plugins."""
    components: Dict[str, List[Dict[str, Any]]] = Field(..., description="Components grouped by type")
    total: int = Field(..., description="Total number of components")