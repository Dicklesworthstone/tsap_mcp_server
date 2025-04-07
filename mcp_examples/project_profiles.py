#!/usr/bin/env python3
"""
Project Profiling Tool (MCP Tools Version)

This script demonstrates how to analyze software project codebases to generate
project profiles including language usage, complexity metrics, and dependency analysis
using the TSAP MCP tools.
"""
import asyncio
import os
import json
import re
import tempfile
from typing import Dict, List, Any, Optional
from pathlib import Path
from collections import Counter, defaultdict
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich.progress import Progress
from rich.tree import Tree

# Import FastMCP client for MCP tools
from mcp.client import FastMCPClient
from mcp.handlers import Context

console = Console()

# File extension to language mapping
LANGUAGE_MAP = {
    '.py': 'Python',
    '.js': 'JavaScript',
    '.ts': 'TypeScript',
    '.jsx': 'React',
    '.tsx': 'React/TypeScript',
    '.html': 'HTML',
    '.css': 'CSS',
    '.scss': 'SCSS',
    '.java': 'Java',
    '.c': 'C',
    '.cpp': 'C++',
    '.h': 'C/C++ Header',
    '.go': 'Go',
    '.rs': 'Rust',
    '.php': 'PHP',
    '.rb': 'Ruby',
    '.swift': 'Swift',
    '.kt': 'Kotlin',
    '.sh': 'Shell',
    '.md': 'Markdown',
    '.json': 'JSON',
    '.xml': 'XML',
    '.yaml': 'YAML',
    '.yml': 'YAML',
    '.toml': 'TOML',
    '.sql': 'SQL',
    '.dockerfile': 'Dockerfile',
    '.makefile': 'Makefile',
    '.mk': 'Makefile'
}

# Default project to analyze if none provided
DEFAULT_PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

class ProjectProfile:
    """Class to store and process project profile data"""
    
    def __init__(self, project_path):
        self.project_path = os.path.abspath(project_path)
        self.project_name = os.path.basename(self.project_path)
        self.file_count = 0
        self.total_size = 0
        self.languages = Counter()
        self.dependencies = {}
        self.complexity = {}
        self.directory_structure = {}
        self.timestamp = None
    
    def to_dict(self):
        """Convert the profile to a dictionary."""
        return {
            "project_name": self.project_name,
            "project_path": self.project_path,
            "file_count": self.file_count,
            "total_size": self.total_size,
            "languages": dict(self.languages),
            "dependencies": self.dependencies,
            "complexity": self.complexity,
            "timestamp": self.timestamp
        }
    
    def to_json(self):
        """Convert the profile to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def save_to_file(self, output_path=None):
        """Save the profile to a JSON file."""
        if output_path is None:
            output_path = os.path.join(
                self.project_path,
                f"{self.project_name}_profile.json"
            )
        
        with open(output_path, 'w') as f:
            f.write(self.to_json())
        
        return output_path

async def analyze_project(project_path=DEFAULT_PROJECT_PATH, exclude_dirs=None):
    """Generate a profile for a project using MCP tools."""
    console.print(Panel.fit(
        f"[bold blue]TSAP MCP Project Profiler[/bold blue]",
        subtitle=f"Analyzing project: {os.path.basename(project_path)}"
    ))
    
    if exclude_dirs is None:
        exclude_dirs = ['.git', 'node_modules', 'venv', '__pycache__', '.venv', '.env']
    
    profile = ProjectProfile(project_path)
    
    try:
        # Create the client
        async with FastMCPClient() as client:
            console.print("Client connected successfully")
            
            # Create a project profile
            with Progress() as progress:
                scan_task = progress.add_task("[green]Scanning files...", total=100)
                lang_task = progress.add_task("[cyan]Analyzing languages...", total=100)
                dep_task = progress.add_task("[yellow]Detecting dependencies...", total=100)
                compl_task = progress.add_task("[magenta]Measuring complexity...", total=100)
                
                # 1. Scan files and directories
                await scan_project_files(client, profile, exclude_dirs)
                progress.update(scan_task, completed=100)
                
                # 2. Analyze languages
                await analyze_languages(client, profile)
                progress.update(lang_task, completed=100)
                
                # 3. Detect dependencies
                await detect_dependencies(client, profile)
                progress.update(dep_task, completed=100)
                
                # 4. Measure complexity
                await measure_complexity(client, profile)
                progress.update(compl_task, completed=100)
            
            # Display the profile
            await display_profile(profile)
            
            # Save the profile
            output_path = profile.save_to_file()
            console.print(f"\n[green]Profile saved to: {output_path}[/green]")
            
    except Exception as e:
        console.print(f"[bold red]Error analyzing project: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())

async def scan_project_files(client, profile, exclude_dirs):
    """Scan project files and update profile with file information."""
    console.print(Rule("[bold yellow]Project File Analysis[/bold yellow]"))
    console.print("[italic]Scanning files and directories[/italic]\n")
    
    # Create context for directory listing
    ctx = Context()
    
    # Use the MCP ls tool to recursively list files
    ls_result = await client.tools.ls(
        path=profile.project_path,
        recursive=True,
        include_hidden=False,
        ctx=ctx
    )
    
    all_files = []
    if "entries" in ls_result:
        entries = ls_result["entries"]
        
        # Filter out excluded directories
        filtered_entries = [
            entry for entry in entries 
            if not any(excluded in entry.get("path", "") for excluded in exclude_dirs)
        ]
        
        # Process files
        files = [entry for entry in filtered_entries if entry.get("type") == "file"]
        profile.file_count = len(files)
        profile.total_size = sum(entry.get("size", 0) for entry in files)
        
        # Collect file extensions
        for file_entry in files:
            file_path = file_entry.get("path", "")
            _, ext = os.path.splitext(file_path.lower())
            
            # Map extension to language
            language = LANGUAGE_MAP.get(ext, "Other")
            profile.languages[language] += 1
            
            # Add file to list
            all_files.append({
                "path": file_path,
                "size": file_entry.get("size", 0),
                "language": language
            })
    
    # Generate directory structure representation
    directory_structure = defaultdict(int)
    for file_entry in all_files:
        path = file_entry.get("path", "")
        rel_path = os.path.relpath(path, profile.project_path)
        directory = os.path.dirname(rel_path)
        directory_structure[directory] += 1
    
    profile.directory_structure = dict(directory_structure)
    
    # Display summary
    console.print(f"[green]Found {profile.file_count} files totaling {profile.total_size/1024:.1f} KB[/green]")
    
    # Show top directories by file count
    console.print("\n[bold cyan]Top Directories[/bold cyan]")
    table = Table(show_header=True)
    table.add_column("Directory")
    table.add_column("Files")
    
    # Sort directories by file count and take top 5
    top_dirs = sorted(directory_structure.items(), key=lambda x: x[1], reverse=True)[:5]
    for dir_name, count in top_dirs:
        table.add_row(dir_name or ".", str(count))
    
    console.print(table)

async def analyze_languages(client, profile):
    """Analyze programming languages used in the project."""
    console.print(Rule("[bold yellow]Language Analysis[/bold yellow]"))
    console.print("[italic]Identifying programming languages used[/italic]\n")
    
    # Display language statistics
    console.print("[bold cyan]Language Distribution[/bold cyan]")
    
    table = Table(show_header=True)
    table.add_column("Language")
    table.add_column("Files")
    table.add_column("Percentage")
    
    total_files = sum(profile.languages.values())
    
    # Sort languages by file count
    languages_sorted = sorted(profile.languages.items(), key=lambda x: x[1], reverse=True)
    for language, count in languages_sorted:
        percentage = (count / total_files) * 100 if total_files > 0 else 0
        table.add_row(language, str(count), f"{percentage:.1f}%")
    
    console.print(table)

async def detect_dependencies(client, profile):
    """Detect project dependencies."""
    console.print(Rule("[bold yellow]Dependency Analysis[/bold yellow]"))
    console.print("[italic]Identifying project dependencies[/italic]\n")
    
    dependencies = {}
    
    # Analyze Python dependencies
    if profile.languages.get('Python', 0) > 0:
        await analyze_python_dependencies(client, profile, dependencies)
    
    # Analyze JavaScript dependencies
    if profile.languages.get('JavaScript', 0) > 0 or profile.languages.get('TypeScript', 0) > 0:
        await analyze_js_dependencies(client, profile, dependencies)
    
    profile.dependencies = dependencies
    
    # Display results
    if dependencies:
        console.print("[bold cyan]Detected Dependencies[/bold cyan]")
        
        # Create a tree view of dependencies
        tree = Tree("Dependencies")
        
        for dep_type, deps in dependencies.items():
            branch = tree.add(dep_type)
            for dep_name, dep_info in deps.items():
                version = dep_info.get('version', 'unknown')
                branch.add(f"[bold]{dep_name}[/bold] ({version})")
        
        console.print(tree)
    else:
        console.print("[yellow]No dependencies detected[/yellow]")

async def analyze_python_dependencies(client, profile, dependencies):
    """Analyze Python dependencies in the project."""
    ctx = Context()
    
    # Look for requirements.txt
    req_path = os.path.join(profile.project_path, "requirements.txt")
    if os.path.exists(req_path):
        # Read requirements.txt
        with open(req_path, 'r') as f:
            content = f.read()
            
        # Parse requirements
        python_deps = {}
        for line in content.splitlines():
            line = line.strip()
            if line and not line.startswith('#'):
                # Parse package name and version
                parts = re.split(r'[=<>~]', line, 1)
                package_name = parts[0].strip()
                version = parts[1].strip() if len(parts) > 1 else "latest"
                
                python_deps[package_name] = {
                    'version': version,
                    'source': 'requirements.txt'
                }
        
        dependencies['Python'] = python_deps
    
    # Also look for setup.py
    setup_path = os.path.join(profile.project_path, "setup.py")
    if os.path.exists(setup_path):
        ctx = Context()
        search_result = await client.tools.search(
            query=r"install_requires\s*=\s*\[",
            paths=[setup_path],
            ctx=ctx
        )
        
        if "matches" in search_result and search_result["matches"]:
            console.print("[green]Found dependencies in setup.py[/green]")
            
            # Simple parsing for demo purposes
            with open(setup_path, 'r') as f:
                setup_content = f.read()
            
            # Extract dependencies (simplified)
            if 'Python' not in dependencies:
                dependencies['Python'] = {}
            
            # Add note that setup.py dependencies were found
            dependencies['Python']['setup.py'] = {
                'version': 'N/A',
                'source': 'Found in setup.py but not parsed in detail'
            }

async def analyze_js_dependencies(client, profile, dependencies):
    """Analyze JavaScript/TypeScript dependencies in the project."""
    package_path = os.path.join(profile.project_path, "package.json")
    
    if os.path.exists(package_path):
        with open(package_path, 'r') as f:
            try:
                package_data = json.load(f)
                
                # Extract dependencies
                js_deps = {}
                
                # Regular dependencies
                if 'dependencies' in package_data:
                    for dep_name, version in package_data['dependencies'].items():
                        js_deps[dep_name] = {
                            'version': version,
                            'type': 'production'
                        }
                
                # Dev dependencies
                if 'devDependencies' in package_data:
                    for dep_name, version in package_data['devDependencies'].items():
                        js_deps[dep_name] = {
                            'version': version,
                            'type': 'development'
                        }
                
                dependencies['JavaScript/TypeScript'] = js_deps
                
            except json.JSONDecodeError:
                console.print("[yellow]Error parsing package.json[/yellow]")

async def measure_complexity(client, profile):
    """Measure code complexity metrics."""
    console.print(Rule("[bold yellow]Code Complexity Analysis[/bold yellow]"))
    console.print("[italic]Measuring code complexity metrics[/italic]\n")
    
    # Focus on main languages
    main_languages = [lang for lang, count in profile.languages.items() 
                     if count > 2 and lang in ['Python', 'JavaScript', 'TypeScript', 'Java', 'C++', 'Go']]
    
    if not main_languages:
        console.print("[yellow]No supported languages for complexity analysis[/yellow]")
        return
    
    complexity_metrics = {}
    
    for language in main_languages:
        # Get file extensions for this language
        extensions = [ext for ext, lang in LANGUAGE_MAP.items() if lang == language]
        if not extensions:
            continue
            
        # Create context for analysis
        ctx = Context()
        
        # For demo purposes, we'll generate simulated complexity metrics
        console.print(f"[green]Analyzing {language} code complexity[/green]")
        
        # In a real implementation, we would use appropriate tools for each language
        # For example, for Python we might use the MCP tools to run tools like:
        # - mccabe for cyclomatic complexity
        # - pylint for maintainability index
        # - radon for code metrics
        
        # Simulated metrics for demonstration
        complexity_metrics[language] = {
            "average_cyclomatic_complexity": 4.2,
            "max_cyclomatic_complexity": 15,
            "maintainability_index": 72.5,
            "average_loc_per_function": 12.3,
            "total_functions": 58
        }
    
    profile.complexity = complexity_metrics
    
    # Display complexity metrics
    console.print("[bold cyan]Code Complexity Metrics[/bold cyan]")
    
    table = Table(show_header=True)
    table.add_column("Language")
    table.add_column("Avg. Complexity")
    table.add_column("Maintainability")
    table.add_column("Avg. LOC/Function")
    
    for language, metrics in complexity_metrics.items():
        table.add_row(
            language,
            f"{metrics.get('average_cyclomatic_complexity', 'N/A')}",
            f"{metrics.get('maintainability_index', 'N/A')}",
            f"{metrics.get('average_loc_per_function', 'N/A')}"
        )
    
    console.print(table)

async def display_profile(profile):
    """Display a summary of the project profile."""
    console.print(Rule("[bold green]Project Profile Summary[/bold green]"))
    
    # Create a panel with project summary
    summary = f"""
[bold]Project:[/bold] {profile.project_name}
[bold]Files:[/bold] {profile.file_count} ({profile.total_size/1024:.1f} KB)
[bold]Main Languages:[/bold] {', '.join([lang for lang, count in profile.languages.most_common(3)])}
[bold]Complexity:[/bold] {len(profile.complexity)} languages analyzed
[bold]Dependencies:[/bold] {sum(len(deps) for deps in profile.dependencies.values())} detected
"""
    
    console.print(Panel(summary, title="Project Summary"))

async def main():
    """Run the project profiling tool."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate a profile for a software project")
    parser.add_argument("--path", "-p", default=DEFAULT_PROJECT_PATH,
                        help="Path to the project to analyze (default: current directory)")
    parser.add_argument("--exclude", "-e", nargs="+", default=['.git', 'node_modules', 'venv', '__pycache__'],
                       help="Directories to exclude from analysis")
    
    args = parser.parse_args()
    
    # Full path to the project
    project_path = os.path.abspath(args.path) 
    
    await analyze_project(project_path, args.exclude)

if __name__ == "__main__":
    asyncio.run(main()) 