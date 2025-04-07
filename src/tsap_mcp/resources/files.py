"""
File resources for TSAP MCP Server.

This module provides MCP resource implementations for accessing file content
and related data.
"""
import os
from typing import Optional
from datetime import datetime
from mcp.server.fastmcp import FastMCP


def register_file_resources(mcp: FastMCP) -> None:
    """Register all file-related resources with the MCP server.
    
    Args:
        mcp: FastMCP server instance
    """
    
    @mcp.resource("file://{path}")
    async def get_file_content(path: str) -> str:
        """Get the content of a file.
        
        This resource provides access to file content, with intelligent
        handling of different file types and encoding.
        
        Args:
            path: Path to the file
            
        Returns:
            File content as string
        """
        # Check if file exists
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        # Read the file
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            # Try alternative encodings
            encodings = ["latin-1", "cp1252", "iso-8859-1"]
            for encoding in encodings:
                try:
                    with open(path, "r", encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            
            # If we get here, all encodings failed
            raise ValueError(f"Could not decode file {path} with any supported encoding")
    
    @mcp.resource("file://{path}/info")
    async def get_file_info(path: str) -> str:
        """Get metadata about a file.
        
        This resource provides information about a file, including its
        size, modification time, and other attributes.
        
        Args:
            path: Path to the file
            
        Returns:
            File metadata as JSON string
        """
        import json
        import stat
        
        # Check if file exists
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        # Get file stats
        stats = os.stat(path)
        
        # Determine file type
        file_ext = os.path.splitext(path)[1].lower()[1:]  # Remove the dot
        
        # Create metadata dictionary
        metadata = {
            "path": path,
            "filename": os.path.basename(path),
            "directory": os.path.dirname(path),
            "size": stats.st_size,
            "size_human": _format_size(stats.st_size),
            "created": datetime.fromtimestamp(stats.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stats.st_mtime).isoformat(),
            "accessed": datetime.fromtimestamp(stats.st_atime).isoformat(),
            "extension": file_ext,
            "is_executable": bool(stats.st_mode & stat.S_IXUSR),
            "is_readable": bool(stats.st_mode & stat.S_IRUSR),
            "is_writeable": bool(stats.st_mode & stat.S_IWUSR),
        }
        
        return json.dumps(metadata, indent=2)
    
    @mcp.resource("directory://{path}")
    async def get_directory_listing(path: str) -> str:
        """Get a listing of files in a directory.
        
        This resource provides a listing of files and subdirectories
        within a specified directory.
        
        Args:
            path: Path to the directory
            
        Returns:
            Directory listing as JSON string
        """
        import json
        
        # Check if directory exists
        if not os.path.isdir(path):
            raise NotADirectoryError(f"Directory not found: {path}")
        
        # Get directory contents
        entries = []
        for entry in os.listdir(path):
            entry_path = os.path.join(path, entry)
            entry_type = "directory" if os.path.isdir(entry_path) else "file"
            
            # Get basic metadata
            try:
                stats = os.stat(entry_path)
                size = stats.st_size if entry_type == "file" else None
                modified = datetime.fromtimestamp(stats.st_mtime).isoformat()
            except (FileNotFoundError, PermissionError):
                size = None
                modified = None
            
            entries.append({
                "name": entry,
                "path": entry_path,
                "type": entry_type,
                "size": size,
                "size_human": _format_size(size) if size is not None else None,
                "modified": modified,
            })
        
        # Sort entries: directories first, then files, both alphabetically
        entries.sort(key=lambda e: (0 if e["type"] == "directory" else 1, e["name"].lower()))
        
        result = {
            "path": path,
            "entries": entries,
            "total_files": sum(1 for e in entries if e["type"] == "file"),
            "total_directories": sum(1 for e in entries if e["type"] == "directory"),
        }
        
        return json.dumps(result, indent=2)


def _format_size(size: Optional[int]) -> Optional[str]:
    """Format file size in a human-readable format.
    
    Args:
        size: File size in bytes
        
    Returns:
        Human-readable size string
    """
    if size is None:
        return None
    
    # Define size units
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size_float = float(size)
    unit_index = 0
    
    # Find appropriate unit
    while size_float >= 1024.0 and unit_index < len(units) - 1:
        size_float /= 1024.0
        unit_index += 1
    
    # Format with appropriate precision
    if unit_index == 0:
        return f"{size_float:.0f} {units[unit_index]}"
    else:
        return f"{size_float:.2f} {units[unit_index]}" 