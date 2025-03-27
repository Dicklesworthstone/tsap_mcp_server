"""
Code analysis tools for TSAP.

This module provides functionality for analyzing code across different programming
languages, focusing on structure, patterns, dependencies, and quality metrics.
"""
import os
import re
import asyncio
from typing import Dict, List, Any, Optional
import tempfile

from tsap.utils.logging import logger
from tsap.core.ripgrep import ripgrep_search
from tsap.composite.parallel import parallel_search
from tsap.mcp.models import (
    CodeAnalyzerParams, CodeAnalyzerResult,
    RipgrepSearchParams, ParallelSearchParams, SearchPattern
)
from tsap.analysis.base import BaseAnalysisTool, register_analysis_tool, AnalysisContext


# Language patterns and configurations
LANGUAGE_EXTENSIONS = {
    "python": [".py", ".pyi", ".pyx", ".pyw"],
    "javascript": [".js", ".jsx", ".mjs", ".cjs"],
    "typescript": [".ts", ".tsx"],
    "java": [".java"],
    "c": [".c", ".h"],
    "cpp": [".cpp", ".hpp", ".cc", ".hh", ".cxx", ".hxx"],
    "csharp": [".cs"],
    "go": [".go"],
    "rust": [".rs"],
    "ruby": [".rb"],
    "php": [".php"],
    "swift": [".swift"],
    "kotlin": [".kt", ".kts"],
    "scala": [".scala"],
    "shell": [".sh", ".bash", ".zsh"],
    "html": [".html", ".htm"],
    "css": [".css", ".scss", ".sass", ".less"],
    "sql": [".sql"],
    "yaml": [".yaml", ".yml"],
    "json": [".json"],
    "markdown": [".md", ".markdown"],
}

# Language-specific patterns and rules
LANGUAGE_PATTERNS = {
    "python": {
        "imports": {
            "pattern": r"^\s*(import|from)\s+(\S+)",
            "description": "Python import statement"
        },
        "classes": {
            "pattern": r"^\s*class\s+(\w+)",
            "description": "Python class definition"
        },
        "functions": {
            "pattern": r"^\s*def\s+(\w+)",
            "description": "Python function definition"
        },
        "async_functions": {
            "pattern": r"^\s*async\s+def\s+(\w+)",
            "description": "Python async function definition"
        },
        "variables": {
            "pattern": r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=",
            "description": "Python variable assignment"
        },
        "constants": {
            "pattern": r"^\s*([A-Z][A-Z0-9_]*)\s*=",
            "description": "Python constant definition"
        }
    },
    "javascript": {
        "imports": {
            "pattern": r"^\s*(import|require)\s+(\S+)",
            "description": "JavaScript import/require statement"
        },
        "classes": {
            "pattern": r"^\s*class\s+(\w+)",
            "description": "JavaScript class definition"
        },
        "functions": {
            "pattern": r"^\s*(function)\s+(\w+)|^\s*const\s+(\w+)\s*=\s*(?:async\s*)?\(.*\)\s*=>",
            "description": "JavaScript function definition"
        },
        "async_functions": {
            "pattern": r"^\s*async\s+function\s+(\w+)|^\s*const\s+(\w+)\s*=\s*async\s*\(.*\)\s*=>",
            "description": "JavaScript async function definition"
        },
        "variables": {
            "pattern": r"^\s*(let|var|const)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=",
            "description": "JavaScript variable declaration"
        }
    }
    # More languages would be defined here
}

# Complexity metrics patterns
COMPLEXITY_PATTERNS = {
    "conditionals": {
        "pattern": r"^\s*(if|else|switch|case|while|for|do)",
        "description": "Conditional or loop statement"
    },
    "nested_conditionals": {
        "pattern": r"^\s+\s+(if|else|switch|case|while|for|do)",
        "description": "Nested conditional or loop"
    },
    "long_lines": {
        "pattern": r"^.{120,}$",
        "description": "Line with more than 120 characters"
    }
}

# Security patterns
SECURITY_PATTERNS = {
    "passwords": {
        "pattern": r"(password|passwd|pwd)\s*=\s*['\"][^'\"]+['\"]",
        "description": "Hardcoded password"
    },
    "api_keys": {
        "pattern": r"(api_key|apikey|api_token|secret_key|api[-_]?key)[_]?[[:alnum:]]*\s*=\s*['\"][[:alnum:]]{16,}['\"]",
        "description": "Hardcoded API key"
    },
    "sql_injection": {
        "pattern": r"(SELECT|INSERT|UPDATE|DELETE|DROP).*\+\s*.*\+",
        "description": "Potential SQL injection vulnerability"
    },
    "command_injection": {
        "pattern": r"(exec|eval|system|subprocess|popen|os\.system|spawn)\s*\(.*\$.*\)",
        "description": "Potential command injection vulnerability"
    }
}


@register_analysis_tool("code_analyzer")
class CodeAnalyzer(BaseAnalysisTool):
    """Code analyzer for various programming languages."""
    
    def __init__(self, name: str = "code_analyzer"):
        """Initialize the code analyzer.
        
        Args:
            name: Analyzer name
        """
        super().__init__(name)
    
    def _detect_language(self, file_path: str) -> Optional[str]:
        """Detect the programming language of a file based on extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected language or None if not recognized
        """
        extension = os.path.splitext(file_path.lower())[1]
        
        for language, extensions in LANGUAGE_EXTENSIONS.items():
            if extension in extensions:
                return language
                
        return None
    
    def _get_language_file_patterns(self, language: Optional[str] = None) -> List[str]:
        """Get file patterns for a language or all languages.
        
        Args:
            language: Specific language or None for all
            
        Returns:
            List of file extension patterns (e.g., "*.py")
        """
        patterns = []
        
        if language:
            # Get patterns for specific language
            extensions = LANGUAGE_EXTENSIONS.get(language, [])
            patterns = [f"*{ext}" for ext in extensions]
        else:
            # Get patterns for all languages
            for extensions in LANGUAGE_EXTENSIONS.values():
                for ext in extensions:
                    pattern = f"*{ext}"
                    if pattern not in patterns:
                        patterns.append(pattern)
                        
        return patterns
    
    async def _find_code_files(
        self, 
        paths: List[str], 
        language: Optional[str] = None
    ) -> List[str]:
        """Find code files in the given paths.
        
        Args:
            paths: Paths to search
            language: Optional language filter
            
        Returns:
            List of file paths
        """
        file_patterns = self._get_language_file_patterns(language)
        
        # Create ripgrep search parameters to find all files
        params = RipgrepSearchParams(
            pattern="",  # Empty pattern to match all files
            paths=paths,
            file_patterns=file_patterns,
            max_total_matches=None,  # No limit
            invert_match=True,  # Invert to match file names, not content
        )
        
        # Execute the search
        result = await ripgrep_search(params)
        
        # Extract file paths from matches
        files = list(set(match.path for match in result.matches))
        
        logger.info(
            f"Found {len(files)} code files",
            component="analysis",
            operation="find_code_files",
            context={
                "language": language or "all",
                "file_count": len(files),
                "file_patterns": file_patterns,
            }
        )
        
        return files
    
    async def _analyze_structure(
        self, 
        files: List[str], 
        context: AnalysisContext
    ) -> Dict[str, Any]:
        """Analyze code structure (classes, functions, imports).
        
        Args:
            files: List of file paths
            context: Analysis context
            
        Returns:
            Structure analysis results
        """
        logger.info(
            "Analyzing code structure",
            component="analysis",
            operation="analyze_structure"
        )
        
        # Group files by language
        files_by_language = {}
        for file in files:
            language = self._detect_language(file)
            if language:
                if language not in files_by_language:
                    files_by_language[language] = []
                files_by_language[language].append(file)
        
        # Analyze each language
        language_results = {}
        for language, language_files in files_by_language.items():
            # Get patterns for this language
            language_patterns = LANGUAGE_PATTERNS.get(language)
            if not language_patterns:
                logger.debug(
                    f"No patterns defined for language: {language}",
                    component="analysis",
                    operation="analyze_structure"
                )
                continue
            
            # Create search patterns
            search_patterns = []
            for category, pattern_info in language_patterns.items():
                search_patterns.append(SearchPattern(
                    pattern=pattern_info["pattern"],
                    description=pattern_info["description"],
                    category=category,
                    regex=True,
                    case_sensitive=True,
                    tags=[language, category],
                ))
            
            # Create parallel search parameters
            params = ParallelSearchParams(
                patterns=search_patterns,
                paths=language_files,
                consolidate_overlapping=False,
                context_lines=2,
            )
            
            # Execute parallel search
            result = await parallel_search(params)
            
            # Process results
            structure_data = {
                "file_count": len(language_files),
                "categories": {},
            }
            
            # Group matches by category
            for match in result.matches:
                category = match.pattern_category
                if category not in structure_data["categories"]:
                    structure_data["categories"][category] = []
                    
                structure_data["categories"][category].append({
                    "file": match.path,
                    "line": match.line_number,
                    "text": match.line_text.strip(),
                    "match": match.match_text,
                })
            
            # Calculate statistics
            for category, items in structure_data["categories"].items():
                # Count items per file
                files_with_category = set(item["file"] for item in items)
                
                structure_data["categories"][category] = {
                    "items": items,
                    "count": len(items),
                    "files_with_items": len(files_with_category),
                    "average_per_file": len(items) / len(language_files) if language_files else 0,
                }
            
            language_results[language] = structure_data
        
        # Summary statistics
        summary = {
            "total_files": len(files),
            "languages": list(files_by_language.keys()),
            "files_by_language": {lang: len(lang_files) for lang, lang_files in files_by_language.items()},
        }
        
        return {
            "summary": summary,
            "languages": language_results,
        }
    
    async def _analyze_dependencies(
        self, 
        files: List[str], 
        context: AnalysisContext
    ) -> Dict[str, Any]:
        """Analyze code dependencies.
        
        Args:
            files: List of file paths
            context: Analysis context
            
        Returns:
            Dependency analysis results
        """
        logger.info(
            "Analyzing code dependencies",
            component="analysis",
            operation="analyze_dependencies"
        )
        
        # Group files by language
        files_by_language = {}
        for file in files:
            language = self._detect_language(file)
            if language:
                if language not in files_by_language:
                    files_by_language[language] = []
                files_by_language[language].append(file)
        
        # Analyze each language
        language_results = {}
        for language, language_files in files_by_language.items():
            # Get patterns for this language
            language_patterns = LANGUAGE_PATTERNS.get(language)
            if not language_patterns or "imports" not in language_patterns:
                continue
            
            # Create search pattern for imports
            import_pattern = language_patterns["imports"]
            search_pattern = SearchPattern(
                pattern=import_pattern["pattern"],
                description=import_pattern["description"],
                category="imports",
                regex=True,
                case_sensitive=True,
                tags=[language, "imports"],
            )
            
            # Create ripgrep search parameters
            params = RipgrepSearchParams(
                pattern=search_pattern.pattern,
                paths=language_files,
                regex=True,
                case_sensitive=True,
            )
            
            # Execute search
            result = await ripgrep_search(params)
            
            # Process results
            dependencies = {}
            imports_by_file = {}
            
            for match in result.matches:
                file_path = match.path
                import_text = match.line_text.strip()
                
                # Add to imports by file
                if file_path not in imports_by_file:
                    imports_by_file[file_path] = []
                    
                imports_by_file[file_path].append(import_text)
                
                # Extract module name (language-specific logic)
                module_name = None
                if language == "python":
                    if import_text.startswith("import "):
                        module_name = import_text.split("import ")[1].split(" as ")[0].strip().split(",")[0].strip()
                    elif import_text.startswith("from "):
                        parts = import_text.split("from ")[1].split(" import ")
                        if len(parts) > 0:
                            module_name = parts[0].strip()
                elif language in ["javascript", "typescript"]:
                    if "import " in import_text:
                        # Extract from "import X from Y" or similar
                        parts = import_text.split("from ")
                        if len(parts) > 1:
                            module_name = parts[1].strip().strip('"\'').strip(";")
                    elif "require(" in import_text:
                        # Extract from require('module')
                        match = re.search(r"require\(['\"]([^'\"]+)['\"]", import_text)
                        if match:
                            module_name = match.group(1)
                
                if module_name:
                    # Add to dependencies count
                    if module_name not in dependencies:
                        dependencies[module_name] = {
                            "count": 0,
                            "files": set(),
                        }
                        
                    dependencies[module_name]["count"] += 1
                    dependencies[module_name]["files"].add(file_path)
            
            # Finalize dependencies
            for module, data in dependencies.items():
                data["files"] = list(data["files"])
                data["usage_percentage"] = (len(data["files"]) / len(language_files)) * 100
            
            # Sort dependencies by count
            sorted_dependencies = dict(sorted(
                dependencies.items(), 
                key=lambda x: x[1]["count"], 
                reverse=True
            ))
            
            language_results[language] = {
                "dependencies": sorted_dependencies,
                "imports_by_file": imports_by_file,
                "total_imports": sum(len(imports) for imports in imports_by_file.values()),
                "files_with_imports": len(imports_by_file),
                "avg_imports_per_file": sum(len(imports) for imports in imports_by_file.values()) / len(language_files) if language_files else 0,
            }
        
        return language_results
    
    async def _analyze_complexity(
        self, 
        files: List[str], 
        context: AnalysisContext
    ) -> Dict[str, Any]:
        """Analyze code complexity metrics.
        
        Args:
            files: List of file paths
            context: Analysis context
            
        Returns:
            Complexity analysis results
        """
        logger.info(
            "Analyzing code complexity",
            component="analysis",
            operation="analyze_complexity"
        )
        
        # Create search patterns
        search_patterns = []
        for category, pattern_info in COMPLEXITY_PATTERNS.items():
            search_patterns.append(SearchPattern(
                pattern=pattern_info["pattern"],
                description=pattern_info["description"],
                category=category,
                regex=True,
                case_sensitive=False,
                tags=["complexity", category],
            ))
        
        # Create parallel search parameters
        params = ParallelSearchParams(
            patterns=search_patterns,
            paths=files,
            consolidate_overlapping=False,
            context_lines=0,
        )
        
        # Execute parallel search
        result = await parallel_search(params)
        
        # Process results
        complexity_data = {
            "metrics_by_file": {},
            "overall_metrics": {
                "conditionals": 0,
                "nested_conditionals": 0,
                "long_lines": 0,
            },
        }
        
        # Process matches
        for match in result.matches:
            file_path = match.path
            category = match.pattern_category
            
            # Initialize file data if needed
            if file_path not in complexity_data["metrics_by_file"]:
                complexity_data["metrics_by_file"][file_path] = {
                    "conditionals": 0,
                    "nested_conditionals": 0,
                    "long_lines": 0,
                    "total_complexity": 0,
                }
                
            # Increment counters
            complexity_data["metrics_by_file"][file_path][category] += 1
            complexity_data["overall_metrics"][category] += 1
        
        # Calculate total complexity scores
        for file_path, metrics in complexity_data["metrics_by_file"].items():
            # Simple weighted complexity score
            metrics["total_complexity"] = (
                metrics["conditionals"] + 
                metrics["nested_conditionals"] * 2 + 
                metrics["long_lines"] * 0.5
            )
            
            # Determine language
            language = self._detect_language(file_path)
            if language:
                metrics["language"] = language
        
        # Calculate statistics
        total_files = len(files)
        complexity_data["summary"] = {
            "avg_conditionals_per_file": complexity_data["overall_metrics"]["conditionals"] / total_files if total_files else 0,
            "avg_nested_conditionals_per_file": complexity_data["overall_metrics"]["nested_conditionals"] / total_files if total_files else 0,
            "avg_long_lines_per_file": complexity_data["overall_metrics"]["long_lines"] / total_files if total_files else 0,
            "total_files": total_files,
        }
        
        # Identify most complex files
        sorted_files = sorted(
            complexity_data["metrics_by_file"].items(),
            key=lambda x: x[1]["total_complexity"],
            reverse=True
        )
        
        complexity_data["most_complex_files"] = [
            {"file": file, "metrics": metrics}
            for file, metrics in sorted_files[:10]  # Top 10 most complex files
        ]
        
        return complexity_data
    
    async def _analyze_security(
        self, 
        files: List[str], 
        context: AnalysisContext
    ) -> Dict[str, Any]:
        """Analyze code for security issues.
        
        Args:
            files: List of file paths
            context: Analysis context
            
        Returns:
            Security analysis results
        """
        logger.info(
            "Analyzing code for security issues",
            component="analysis",
            operation="analyze_security"
        )
        
        # Create search patterns
        search_patterns = []
        for category, pattern_info in SECURITY_PATTERNS.items():
            search_patterns.append(SearchPattern(
                pattern=pattern_info["pattern"],
                description=pattern_info["description"],
                category=category,
                regex=True,
                case_sensitive=False,
                tags=["security", category],
            ))
        
        # Create parallel search parameters
        params = ParallelSearchParams(
            patterns=search_patterns,
            paths=files,
            consolidate_overlapping=False,
            context_lines=1,
        )
        
        # Execute parallel search
        result = await parallel_search(params)
        
        # Process results
        security_data = {
            "issues_by_category": {},
            "issues_by_file": {},
            "overall_count": 0,
        }
        
        # Initialize categories
        for category in SECURITY_PATTERNS.keys():
            security_data["issues_by_category"][category] = []
        
        # Process matches
        for match in result.matches:
            file_path = match.path
            category = match.pattern_category
            
            # Add to issues by category
            security_data["issues_by_category"][category].append({
                "file": file_path,
                "line": match.line_number,
                "text": match.line_text.strip(),
                "context_before": match.before_context,
                "context_after": match.after_context,
            })
            
            # Add to issues by file
            if file_path not in security_data["issues_by_file"]:
                security_data["issues_by_file"][file_path] = []
                
            security_data["issues_by_file"][file_path].append({
                "category": category,
                "line": match.line_number,
                "text": match.line_text.strip(),
                "description": match.pattern_description,
            })
            
            # Increment overall count
            security_data["overall_count"] += 1
        
        # Calculate statistics
        security_data["summary"] = {
            "total_issues": security_data["overall_count"],
            "files_with_issues": len(security_data["issues_by_file"]),
            "issues_by_type": {
                category: len(issues)
                for category, issues in security_data["issues_by_category"].items()
            },
        }
        
        # Prioritize issues
        high_priority_categories = ["passwords", "api_keys"]
        medium_priority_categories = ["sql_injection", "command_injection"]
        
        security_data["high_priority_issues"] = []
        security_data["medium_priority_issues"] = []
        security_data["low_priority_issues"] = []
        
        for category, issues in security_data["issues_by_category"].items():
            for issue in issues:
                issue_with_category = issue.copy()
                issue_with_category["category"] = category
                issue_with_category["description"] = SECURITY_PATTERNS[category]["description"]
                
                if category in high_priority_categories:
                    security_data["high_priority_issues"].append(issue_with_category)
                elif category in medium_priority_categories:
                    security_data["medium_priority_issues"].append(issue_with_category)
                else:
                    security_data["low_priority_issues"].append(issue_with_category)
        
        return security_data
    
    async def analyze(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code with specified parameters.
        
        Args:
            params: Analysis parameters (CodeAnalyzerParams)
            
        Returns:
            Analysis results
        """
        async with self._measure_execution_time():
            start_time = asyncio.get_event_loop().time()
            
            # Convert dict to CodeAnalyzerParams if needed
            if not isinstance(params, CodeAnalyzerParams):
                params = CodeAnalyzerParams(**params)
                
            # Create analysis context
            context = AnalysisContext()
            
            # Determine input source
            if params.code:
                # Single code string - create temporary file
                with tempfile.NamedTemporaryFile(
                    mode="w", 
                    suffix=".txt", 
                    delete=False
                ) as temp_file:
                    temp_file.write(params.code)
                    temp_file_path = temp_file.name
                    
                # Use temporary file for analysis
                file_paths = [temp_file_path]
                try:
                    # Set language-specific extension if known
                    if params.language:
                        extensions = LANGUAGE_EXTENSIONS.get(params.language, [".txt"])
                        if extensions:
                            new_path = f"{temp_file_path}{extensions[0]}"
                            os.rename(temp_file_path, new_path)
                            file_paths = [new_path]
                except Exception as e:
                    logger.warning(
                        f"Failed to rename temporary file: {str(e)}",
                        component="analysis",
                        operation="analyze_code"
                    )
            elif params.file_paths:
                # Explicit file paths
                file_paths = params.file_paths
            elif params.repository_path:
                # Repository path - find all code files
                file_paths = await self._find_code_files(
                    [params.repository_path], 
                    params.language
                )
            else:
                raise ValueError("No code input provided")
                
            # Log analysis start
            logger.info(
                f"Starting code analysis: {len(file_paths)} files",
                component="analysis",
                operation="analyze_code",
                context={
                    "file_count": len(file_paths),
                    "language": params.language,
                    "analysis_types": params.analysis_types,
                }
            )
            
            # Execute selected analysis types
            results = {}
            
            if "structure" in params.analysis_types:
                results["structure"] = await self._analyze_structure(file_paths, context)
                
            if "dependencies" in params.analysis_types:
                results["dependencies"] = await self._analyze_dependencies(file_paths, context)
                
            if "complexity" in params.analysis_types:
                results["complexity"] = await self._analyze_complexity(file_paths, context)
                
            if "security" in params.analysis_types:
                results["security"] = await self._analyze_security(file_paths, context)
                
            # Create summary
            summary = {
                "file_count": len(file_paths),
                "language": params.language,
                "analysis_types": params.analysis_types,
                "execution_time": asyncio.get_event_loop().time() - start_time,
            }
            
            # Log completion
            logger.success(
                f"Code analysis completed: {len(file_paths)} files",
                component="analysis",
                operation="analyze_code",
                context={
                    "file_count": len(file_paths),
                    "execution_time": summary["execution_time"],
                }
            )
            
            # Clean up temporary file if created
            if params.code and "temp_file_path" in locals():
                try:
                    os.unlink(temp_file_path)
                except Exception:
                    pass
            
            # Create and return result
            return CodeAnalyzerResult(
                **results,
                summary=summary,
                execution_time=summary["execution_time"],
            ).dict()


# Convenience function to use the code analyzer
async def analyze_code(params: CodeAnalyzerParams) -> CodeAnalyzerResult:
    """Analyze code with specified parameters.
    
    This is a convenience function that uses the CodeAnalyzer class.
    
    Args:
        params: Analysis parameters
        
    Returns:
        Analysis results
    """
    analyzer = CodeAnalyzer()
    result = await analyzer.analyze(params)
    return CodeAnalyzerResult(**result)