"""
Project resources for TSAP MCP Server.

This module provides MCP resource implementations for accessing project-related
data and metadata.
"""
import os
import json
import subprocess
from typing import Optional, Dict, Any
from pathlib import Path
from mcp.server.fastmcp import FastMCP


def register_project_resources(mcp: FastMCP) -> None:
    """Register all project-related resources with the MCP server.
    
    Args:
        mcp: FastMCP server instance
    """
    
    @mcp.resource("project://structure/{depth}")
    async def get_project_structure(depth: int = 3) -> str:
        """Get the structure of the project.
        
        This resource provides a hierarchical representation of the project's
        directory structure, with configurable depth.
        
        Args:
            depth: Maximum directory depth to traverse
            
        Returns:
            Project structure as JSON string
        """
        def scan_directory(directory: Path, current_depth: int = 0) -> Dict[str, Any]:
            """Recursively scan a directory to build structure representation."""
            if current_depth > depth:
                return {"name": directory.name, "type": "directory", "truncated": True}
            
            result = {
                "name": directory.name,
                "type": "directory", 
                "path": str(directory),
                "items": []
            }
            
            try:
                # List contents
                for item in sorted(directory.iterdir()):
                    if item.is_dir() and not item.name.startswith('.') and item.name != '__pycache__':
                        # Recursively process subdirectory
                        result["items"].append(scan_directory(item, current_depth + 1))
                    elif item.is_file() and not item.name.startswith('.'):
                        # Add file info
                        file_info = {
                            "name": item.name,
                            "type": "file",
                            "path": str(item),
                            "extension": item.suffix[1:] if item.suffix else None,
                            "size": item.stat().st_size,
                        }
                        result["items"].append(file_info)
            except (PermissionError, OSError):
                # Handle access errors
                result["error"] = "Access denied"
                
            return result
        
        # Get the project root directory
        try:
            # Try to get Git root directory first
            git_root = subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"], 
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()
            root_dir = Path(git_root)
        except (subprocess.SubprocessError, FileNotFoundError):
            # Fall back to current directory
            root_dir = Path.cwd()
        
        # Scan the directory structure
        structure = scan_directory(root_dir)
        
        # Add metadata
        structure["total_dirs"] = sum(1 for _ in root_dir.glob('**/')  if not any(p.startswith('.') for p in _.parts))
        structure["total_files"] = sum(1 for _ in root_dir.glob('**/*') if _.is_file() and not any(p.startswith('.') for p in _.parts))
        
        # Convert to JSON
        return json.dumps(structure, indent=2)
    
    # Also add a default structure resource without depth parameter
    @mcp.resource("project://structure")
    async def get_default_project_structure() -> str:
        """Get the structure of the project with default depth.
        
        This is a convenience resource that uses the default depth value.
        
        Returns:
            Project structure as JSON string
        """
        return await get_project_structure(depth=3)
    
    @mcp.resource("project://dependencies/{include_dev}")
    async def get_project_dependencies(include_dev: bool = False) -> str:
        """Get the project's dependencies.
        
        This resource provides information about the project's dependencies,
        including versions and package metadata.
        
        Args:
            include_dev: Whether to include development dependencies
            
        Returns:
            Dependencies as JSON string
        """
        dependencies = {"production": [], "development": []}
        
        # Check for various dependency files
        root_dir = Path.cwd()
        
        # Python dependencies from requirements.txt
        req_txt = root_dir / "requirements.txt"
        if req_txt.exists():
            dependencies["format"] = "requirements.txt"
            with open(req_txt, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        dependencies["production"].append({"name": line})
        
        # Python dependencies from pyproject.toml
        pyproject = root_dir / "pyproject.toml"
        if pyproject.exists():
            dependencies["format"] = "pyproject.toml"
            try:
                import tomli
                with open(pyproject, "rb") as f:
                    pyproject_data = tomli.load(f)
                
                # Get project dependencies
                if "project" in pyproject_data and "dependencies" in pyproject_data["project"]:
                    dependencies["production"] = [{"name": dep} for dep in pyproject_data["project"]["dependencies"]]
                
                # Get optional dependencies
                if include_dev and "project" in pyproject_data and "optional-dependencies" in pyproject_data["project"]:
                    for category, deps in pyproject_data["project"]["optional-dependencies"].items():
                        if category.lower() in ("dev", "development", "test", "testing"):
                            dependencies["development"].extend([{"name": dep} for dep in deps])
            except (ImportError, Exception) as e:
                dependencies["error"] = f"Failed to parse pyproject.toml: {str(e)}"
        
        # Node.js dependencies from package.json
        package_json = root_dir / "package.json"
        if package_json.exists():
            dependencies["format"] = "package.json"
            try:
                with open(package_json, "r") as f:
                    package_data = json.load(f)
                
                # Get production dependencies
                if "dependencies" in package_data:
                    dependencies["production"] = [
                        {"name": name, "version": version} 
                        for name, version in package_data["dependencies"].items()
                    ]
                
                # Get development dependencies
                if include_dev and "devDependencies" in package_data:
                    dependencies["development"] = [
                        {"name": name, "version": version} 
                        for name, version in package_data["devDependencies"].items()
                    ]
            except Exception as e:
                dependencies["error"] = f"Failed to parse package.json: {str(e)}"
        
        # Add metadata
        dependencies["total_production"] = len(dependencies["production"])
        dependencies["total_development"] = len(dependencies["development"]) if include_dev else 0
        
        # Convert to JSON
        return json.dumps(dependencies, indent=2)
    
    @mcp.resource("project://stats")
    async def get_project_stats(path: Optional[str] = None) -> str:
        """Get statistics about the project.
        
        This resource provides detailed statistics about the project,
        including line counts, file types, and other metrics.
        
        Args:
            path: Optional subdirectory to analyze (defaults to project root)
            
        Returns:
            Project statistics as JSON string
        """
        stats = {
            "file_counts": {},
            "line_counts": {},
            "size_by_type": {},
            "total_files": 0,
            "total_lines": 0,
            "total_size": 0,
        }
        
        # Determine the directory to analyze
        if path:
            directory = Path(path)
        else:
            try:
                # Try to get Git root directory first
                git_root = subprocess.check_output(
                    ["git", "rev-parse", "--show-toplevel"], 
                    stderr=subprocess.DEVNULL,
                    text=True
                ).strip()
                directory = Path(git_root)
            except (subprocess.SubprocessError, FileNotFoundError):
                # Fall back to current directory
                directory = Path.cwd()
        
        # Walk the directory and collect stats
        for root, _, files in os.walk(directory):
            root_path = Path(root)
            
            # Skip hidden directories
            if any(part.startswith('.') for part in root_path.parts):
                continue
            
            for file in files:
                # Skip hidden files
                if file.startswith('.'):
                    continue
                
                file_path = root_path / file
                file_ext = file_path.suffix.lower()[1:] if file_path.suffix else "no_extension"
                
                try:
                    # Get file size
                    file_size = file_path.stat().st_size
                    stats["total_size"] += file_size
                    stats["size_by_type"].setdefault(file_ext, 0)
                    stats["size_by_type"][file_ext] += file_size
                    
                    # Count file by type
                    stats["file_counts"].setdefault(file_ext, 0)
                    stats["file_counts"][file_ext] += 1
                    stats["total_files"] += 1
                    
                    # Count lines (for text files only)
                    if file_ext in TEXT_FILE_EXTENSIONS:
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                line_count = sum(1 for _ in f)
                                stats["line_counts"].setdefault(file_ext, 0)
                                stats["line_counts"][file_ext] += line_count
                                stats["total_lines"] += line_count
                        except Exception:
                            # Skip if can't count lines
                            pass
                except (PermissionError, OSError):
                    # Skip if can't access file
                    continue
        
        # Add human-readable sizes
        stats["total_size_human"] = format_size(stats["total_size"])
        stats["size_by_type_human"] = {ext: format_size(size) for ext, size in stats["size_by_type"].items()}
        
        # Additional project metadata
        stats["metadata"] = {}
        
        # Git info if available
        try:
            stats["metadata"]["git_branch"] = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()
            
            stats["metadata"]["git_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()
            
            stats["metadata"]["git_commit_count"] = int(subprocess.check_output(
                ["git", "rev-list", "--count", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip())
        except (subprocess.SubprocessError, FileNotFoundError):
            # No Git info available
            pass
        
        # Convert to JSON
        return json.dumps(stats, indent=2)


# File extensions that are typically text files (for line counting)
TEXT_FILE_EXTENSIONS = {
    "py", "js", "ts", "java", "c", "cpp", "h", "hpp", "cs", "go", "rs", "rb", "php", 
    "html", "htm", "css", "scss", "less", "xml", "json", "yaml", "yml", "md", "rst",
    "txt", "sql", "sh", "bash", "ps1", "bat", "cmd", "toml", "ini", "cfg", "conf"
}


def format_size(size_bytes: int) -> str:
    """Format file size in a human-readable format.
    
    Args:
        size_bytes: File size in bytes
        
    Returns:
        Human-readable size string
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    
    # Convert to appropriate unit
    units = ["KB", "MB", "GB", "TB"]
    size = size_bytes
    unit_index = 0
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    # Format with appropriate precision
    if size < 10:
        return f"{size:.2f} {units[unit_index]}"
    elif size < 100:
        return f"{size:.1f} {units[unit_index]}"
    else:
        return f"{int(size)} {units[unit_index]}" 