"""
Analysis utilities for TSAP MCP Server.

This module provides utilities for code analysis operations,
structure parsing, and metric calculations.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import re
import os


def parse_function_signature(signature: str) -> Dict[str, Any]:
    """Parse a function signature into its components.
    
    Args:
        signature: Function signature string
        
    Returns:
        Dictionary with function name, parameters, return type, etc.
    """
    # Initialize result
    result = {
        "name": "",
        "parameters": [],
        "return_type": None,
        "decorators": [],
        "is_async": False,
    }
    
    # Check for async
    if signature.startswith("async "):
        result["is_async"] = True
        signature = signature[6:].strip()
    
    # Extract function name and parameters
    match = re.match(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)(?:\s*->\s*([^:]+))?:", signature)
    if not match:
        return result
    
    result["name"] = match.group(1)
    
    # Parse parameters
    params_str = match.group(2).strip()
    if params_str:
        params = []
        for param in params_str.split(","):
            param = param.strip()
            if not param:
                continue
                
            # Check for default value
            if "=" in param:
                param_name, default_value = param.split("=", 1)
                params.append({
                    "name": param_name.strip(),
                    "default": default_value.strip(),
                    "type": None
                })
            # Check for type annotation
            elif ":" in param:
                param_name, param_type = param.split(":", 1)
                params.append({
                    "name": param_name.strip(),
                    "type": param_type.strip(),
                    "default": None
                })
            else:
                params.append({
                    "name": param,
                    "type": None,
                    "default": None
                })
        
        result["parameters"] = params
    
    # Extract return type
    if match.group(3):
        result["return_type"] = match.group(3).strip()
    
    return result


def extract_class_info(code: str) -> List[Dict[str, Any]]:
    """Extract information about classes from code.
    
    Args:
        code: Source code to analyze
        
    Returns:
        List of dictionaries with class information
    """
    classes = []
    
    # Find class definitions
    class_pattern = r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\((.*?)\))?\s*:"
    for match in re.finditer(class_pattern, code):
        class_name = match.group(1)
        class_info = {
            "name": class_name,
            "methods": [],
            "attributes": [],
            "base_classes": [],
        }
        
        # Extract base classes
        if match.group(2):
            bases = match.group(2).split(",")
            class_info["base_classes"] = [base.strip() for base in bases]
        
        # Extract class body by finding the block at the same indentation level
        pos = match.end()
        # Skip until line end
        while pos < len(code) and code[pos] != '\n':
            pos += 1
        if pos < len(code):
            pos += 1  # Skip newline
            
        # Find the indentation of the first line of the class body
        indent_match = re.match(r"(\s+)", code[pos:])
        if indent_match:
            indent = indent_match.group(1)
            # Extract methods using the indentation to find the class body
            method_pattern = fr"^{indent}def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
            for method_match in re.finditer(method_pattern, code[pos:], re.MULTILINE):
                method_start = pos + method_match.start()
                # Extract method signature
                line_end = code.find('\n', method_start)
                if line_end == -1:
                    line_end = len(code)
                signature_line = code[method_start:line_end]
                
                # Parse signature
                method_info = parse_function_signature(signature_line.strip())
                class_info["methods"].append(method_info)
            
            # Extract attributes (using type annotations)
            attr_pattern = fr"^{indent}([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*([^=\n]+)(?:\s*=\s*([^\n]+))?"
            for attr_match in re.finditer(attr_pattern, code[pos:], re.MULTILINE):
                name = attr_match.group(1)
                type_annot = attr_match.group(2).strip() if attr_match.group(2) else None
                default = attr_match.group(3).strip() if attr_match.group(3) else None
                
                # Skip if it looks like a method definition
                if name == "def":
                    continue
                    
                class_info["attributes"].append({
                    "name": name,
                    "type": type_annot,
                    "default": default
                })
        
        classes.append(class_info)
    
    return classes


def calculate_code_complexity(code: str) -> Dict[str, Any]:
    """Calculate code complexity metrics.
    
    Args:
        code: Source code to analyze
        
    Returns:
        Dictionary with complexity metrics
    """
    complexity = {
        "cyclomatic_complexity": 0,
        "cognitive_complexity": 0,
        "line_count": 0,
        "comment_ratio": 0.0,
        "maintainability_index": 0,
    }
    
    # Count lines
    lines = code.splitlines()
    complexity["line_count"] = len(lines)
    
    # Count branches (if, else, for, while, etc.)
    branch_patterns = [
        r"\bif\b",
        r"\belse\b",
        r"\belif\b",
        r"\bfor\b",
        r"\bwhile\b",
        r"\btry\b",
        r"\bexcept\b",
        r"\bwith\b",
    ]
    branch_count = 0
    for pattern in branch_patterns:
        branch_count += len(re.findall(pattern, code))
    
    # Count logical operators
    operator_patterns = [
        r"\band\b",
        r"\bor\b",
        r"\bnot\b",
    ]
    operator_count = 0
    for pattern in operator_patterns:
        operator_count += len(re.findall(pattern, code))
    
    # Count comments
    comment_count = len(re.findall(r"#.*$", code, re.MULTILINE))
    comment_lines = len(re.findall(r"^\s*#.*$", code, re.MULTILINE))
    docstring_count = len(re.findall(r'""".*?"""', code, re.DOTALL))
    complexity["comment_ratio"] = (comment_lines + docstring_count * 3) / complexity["line_count"] if complexity["line_count"] > 0 else 0
    
    # Calculate cyclomatic complexity
    complexity["cyclomatic_complexity"] = 1 + branch_count + operator_count
    
    # Calculate cognitive complexity (simplified)
    complexity["cognitive_complexity"] = branch_count + operator_count * 0.5
    
    # Calculate maintainability index (simplified)
    volume = complexity["line_count"] * (1 - complexity["comment_ratio"])
    complexity["maintainability_index"] = max(0, min(100, 100 - volume * 0.2 - complexity["cyclomatic_complexity"] * 0.4))
    
    return complexity


def format_security_findings(findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format security findings for MCP compatibility.
    
    Args:
        findings: Raw security findings from analysis
        
    Returns:
        Formatted security findings
    """
    formatted_findings = []
    
    for finding in findings:
        # Create a standard format for security findings
        formatted_finding = {
            "severity": finding.get("severity", "unknown"),
            "category": finding.get("category", "unknown"),
            "description": finding.get("description", ""),
            "location": {
                "file": finding.get("file", ""),
                "line": finding.get("line", 0),
                "column": finding.get("column", 0),
            },
            "code": finding.get("code", ""),
            "recommendation": finding.get("recommendation", ""),
        }
        
        # Add optional fields if present
        if "cwe" in finding:
            formatted_finding["cwe"] = finding["cwe"]
        if "confidence" in finding:
            formatted_finding["confidence"] = finding["confidence"]
        
        formatted_findings.append(formatted_finding)
    
    return formatted_findings


def extract_docstring(code: str) -> Optional[str]:
    """Extract docstring from code.
    
    Args:
        code: Function or class code to extract docstring from
        
    Returns:
        Docstring string or None if not found
    """
    # Match triple-quoted docstrings
    match = re.search(r'"""(.*?)"""', code, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Try single-quoted docstrings
    match = re.search(r"'''(.*?)'''", code, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return None


def analyze_imports(code: str) -> Dict[str, List[str]]:
    """Analyze imports in Python code.
    
    Args:
        code: Python code to analyze
        
    Returns:
        Dictionary with import information
    """
    imports = {
        "standard_lib": [],
        "third_party": [],
        "local": [],
        "all_imports": [],
    }
    
    # Standard library modules
    std_libs = [
        "os", "sys", "re", "math", "random", "datetime", "time", "json", 
        "csv", "collections", "itertools", "functools", "typing", "pathlib",
        "logging", "io", "subprocess", "argparse", "unittest", "tempfile"
    ]
    
    # Match import statements
    import_pattern = r"^\s*import\s+([^#\n]+)"
    from_pattern = r"^\s*from\s+([^#\n]+)\s+import\s+([^#\n]+)"
    
    # Process "import x" statements
    for match in re.finditer(import_pattern, code, re.MULTILINE):
        modules = [m.strip() for m in match.group(1).split(",")]
        for module in modules:
            base_module = module.split(".")[0]
            imports["all_imports"].append(module)
            
            if base_module in std_libs:
                imports["standard_lib"].append(module)
            elif base_module.startswith((".", "_")):
                imports["local"].append(module)
            else:
                imports["third_party"].append(module)
    
    # Process "from x import y" statements
    for match in re.finditer(from_pattern, code, re.MULTILINE):
        module = match.group(1).strip()
        base_module = module.split(".")[0]
        imported_names = [name.strip() for name in match.group(2).split(",")]
        
        # Record the imported module
        full_import = f"{module}: {', '.join(imported_names)}"
        imports["all_imports"].append(full_import)
        
        if base_module in std_libs:
            imports["standard_lib"].append(full_import)
        elif base_module.startswith((".", "_")):
            imports["local"].append(full_import)
        else:
            imports["third_party"].append(full_import)
    
    return imports


def find_dependencies(file_path: str) -> Dict[str, Any]:
    """Find dependencies of a Python file.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        Dictionary with dependency information
    """
    dependencies = {
        "imports": [],
        "local_dependencies": [],
        "external_dependencies": [],
        "standard_lib_dependencies": [],
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Analyze imports
        import_info = analyze_imports(code)
        dependencies["imports"] = import_info["all_imports"]
        dependencies["local_dependencies"] = import_info["local"]
        dependencies["external_dependencies"] = import_info["third_party"]
        dependencies["standard_lib_dependencies"] = import_info["standard_lib"]
        
    except Exception as e:
        dependencies["error"] = str(e)
    
    return dependencies 