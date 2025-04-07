"""
Analysis tools for TSAP MCP Server.

This module provides MCP tool implementations for various analysis functions,
including code structure analysis, metadata extraction, and pattern recognition.
"""
import logging
from typing import Dict, List, Any, Optional

from mcp.server.fastmcp import FastMCP, Context

# Import original implementations
# Import code structure analysis tools
from tsap.core.structure import StructureAnalyzer, StructureParams
# Import security audit tools
from tsap.core.security import SecurityAnalyzer, SecurityParams
# Import dependency analysis tools
from tsap.core.dependencies import DependencyAnalyzer, DependencyParams
# Import other analysis tools as needed

logger = logging.getLogger("tsap_mcp.tools.analysis")


def register_analysis_tools(mcp: FastMCP) -> None:
    """Register all analysis-related tools with the MCP server.
    
    Args:
        mcp: FastMCP server instance
    """
    
    @mcp.tool()
    async def analyze_code_structure(
        code: str,
        language: str,
        include_imports: bool = True,
        include_comments: bool = False,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """Analyze the structure of code.
        
        This tool parses and analyzes code to extract its structure,
        such as functions, classes, imports, and other declarations.
        
        Args:
            code: Code to analyze
            language: Programming language of the code
            include_imports: Whether to include import statements
            include_comments: Whether to include comments
            ctx: MCP context
            
        Returns:
            Code structure analysis results
        """
        if ctx:
            ctx.info(f"Analyzing {language} code structure")
        
        # Use original implementation
        params = StructureParams(
            code=code,
            language=language,
            options={
                "include_imports": include_imports,
                "include_comments": include_comments,
            },
        )
        
        analyzer = StructureAnalyzer()
        result = await analyzer.analyze(params)
        
        if hasattr(result, 'structure'):
            return result.structure
        return {}
    
    @mcp.tool()
    async def extract_dependencies(
        code: str,
        language: str,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """Extract dependencies from code.
        
        This tool analyzes code to extract dependencies like imports,
        includes, and references to external resources.
        
        Args:
            code: Code to analyze
            language: Programming language of the code
            ctx: MCP context
            
        Returns:
            Extracted dependencies
        """
        if ctx:
            ctx.info(f"Extracting dependencies from {language} code")
        
        # Use original implementation
        params = DependencyParams(
            code=code,
            language=language,
        )
        
        analyzer = DependencyAnalyzer()
        result = await analyzer.analyze(params)
        
        if hasattr(result, 'dependencies'):
            return result.dependencies
        return {}
    
    @mcp.tool()
    async def generate_class_diagram(
        code: str,
        language: str,
        format: str = "mermaid",
        options: Optional[Dict[str, Any]] = None,
        ctx: Optional[Context] = None,
    ) -> str:
        """Generate a class diagram from code.
        
        This tool creates a textual class diagram representation of code
        structure, in formats like Mermaid, PlantUML, or DOT.
        
        Args:
            code: Code to diagram
            language: Programming language of the code
            format: Diagram format (mermaid, plantuml, dot)
            options: Additional diagram options
            ctx: MCP context
            
        Returns:
            Generated diagram text
        """
        if ctx:
            ctx.info(f"Generating {format} class diagram for {language} code")
        
        # Use original implementation
        # First get the structure
        structure_params = StructureParams(
            code=code,
            language=language,
            options={
                "include_relationships": True,
                "include_attributes": True,
            },
        )
        
        analyzer = StructureAnalyzer()
        structure_result = await analyzer.analyze(structure_params)
        
        # Then generate the diagram
        params = StructureParams(
            structure=structure_result.structure if hasattr(structure_result, 'structure') else {},
            format=format,
            options=options or {},
        )
        
        diagram_result = await analyzer.generate_diagram(params)
        
        if hasattr(diagram_result, 'diagram'):
            return diagram_result.diagram
        return ""
    
    @mcp.tool()
    async def code_security_audit(
        file_paths: List[str],
        language: Optional[str] = None,
        severity_level: str = "low",
        include_metrics: bool = False,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """Perform security audit on code.
        
        This tool scans code for security vulnerabilities, bad practices,
        and potential issues across multiple files.
        
        Args:
            file_paths: Paths to files to audit
            language: Programming language (auto-detected if None)
            severity_level: Minimum severity level to report (low, medium, high, critical)
            include_metrics: Whether to include security metrics in the result
            ctx: MCP context
            
        Returns:
            Security audit results
        """
        if ctx:
            ctx.info(f"Performing security audit on {len(file_paths)} files")
        
        # Use original implementation
        params = SecurityParams(
            file_paths=file_paths,
            language=language,
            options={
                "severity_level": severity_level,
                "include_metrics": include_metrics,
            },
        )
        
        analyzer = SecurityAnalyzer()
        result = await analyzer.analyze_security(params)
        
        if hasattr(result, 'findings'):
            return result.findings
        return {}
    
    # Register more analysis tools as needed


# Helper functions for diagram generation

def _generate_mermaid_diagram(structure: Dict[str, Any], options: Dict[str, Any]) -> str:
    """Generate a Mermaid class diagram from code structure.
    
    Args:
        structure: Code structure from analyze_code_structure
        options: Diagram generation options
        
    Returns:
        Mermaid diagram text
    """
    show_methods = options.get("show_methods", True)
    show_fields = options.get("show_fields", True)
    
    # Start diagram
    diagram = ["classDiagram"]
    
    # Add classes
    for cls in structure.get("classes", []):
        class_name = cls.get("name", "")
        
        # Add class definition
        diagram.append(f"    class {class_name}")
        
        # Add inheritance
        extends = cls.get("extends")
        if extends:
            diagram.append(f"    {extends} <|-- {class_name}")
        
        # Add interfaces (Java)
        implements = cls.get("implements", [])
        for interface in implements:
            if interface:
                diagram.append(f"    {class_name} ..|> {interface}")
        
        # Add methods
        if show_methods:
            for method in cls.get("methods", []):
                method_name = method.get("name", "")
                if method_name:
                    params = method.get("params", [])
                    param_str = ", ".join(params)
                    diagram.append(f"    {class_name} : {method_name}({param_str})")
        
        # Add fields
        if show_fields and "fields" in cls:
            for field in cls.get("fields", []):
                field_name = field.get("name", "")
                field_type = field.get("type", "")
                if field_name:
                    diagram.append(f"    {class_name} : {field_type} {field_name}")
    
    # Return diagram
    return "\n".join(diagram)


def _generate_plantuml_diagram(structure: Dict[str, Any], options: Dict[str, Any]) -> str:
    """Generate a PlantUML class diagram from code structure.
    
    Args:
        structure: Code structure from analyze_code_structure
        options: Diagram generation options
        
    Returns:
        PlantUML diagram text
    """
    show_methods = options.get("show_methods", True)
    show_fields = options.get("show_fields", True)
    
    # Start diagram
    diagram = ["@startuml", ""]
    
    # Add classes
    for cls in structure.get("classes", []):
        class_name = cls.get("name", "")
        
        # Add class definition
        diagram.append(f"class {class_name} {{")
        
        # Add fields
        if show_fields and "fields" in cls:
            for field in cls.get("fields", []):
                field_name = field.get("name", "")
                field_type = field.get("type", "")
                if field_name:
                    diagram.append(f"    {field_type} {field_name}")
        
        # Add methods
        if show_methods:
            for method in cls.get("methods", []):
                method_name = method.get("name", "")
                if method_name:
                    params = method.get("params", [])
                    param_str = ", ".join(params)
                    diagram.append(f"    {method_name}({param_str})")
        
        diagram.append("}")
        diagram.append("")
    
    # Add relationships
    for cls in structure.get("classes", []):
        class_name = cls.get("name", "")
        
        # Add inheritance
        extends = cls.get("extends")
        if extends:
            diagram.append(f"{extends} <|-- {class_name}")
        
        # Add interfaces (Java)
        implements = cls.get("implements", [])
        for interface in implements:
            if interface:
                diagram.append(f"{class_name} ..|> {interface}")
    
    # End diagram
    diagram.append("@enduml")
    
    # Return diagram
    return "\n".join(diagram)


def _generate_dot_diagram(structure: Dict[str, Any], options: Dict[str, Any]) -> str:
    """Generate a DOT (Graphviz) class diagram from code structure.
    
    Args:
        structure: Code structure from analyze_code_structure
        options: Diagram generation options
        
    Returns:
        DOT diagram text
    """
    show_methods = options.get("show_methods", True)
    show_fields = options.get("show_fields", True)
    
    # Start diagram
    diagram = ["digraph ClassDiagram {", "    node [shape=record];"]
    
    # Add classes
    for cls in structure.get("classes", []):
        class_name = cls.get("name", "")
        
        # Build label with fields and methods
        label_parts = [f"{class_name}"]
        
        # Add fields
        if show_fields and "fields" in cls:
            fields = cls.get("fields", [])
            if fields:
                field_labels = []
                for field in fields:
                    field_name = field.get("name", "")
                    field_type = field.get("type", "")
                    if field_name:
                        field_labels.append(f"{field_type} {field_name}")
                
                if field_labels:
                    label_parts.append("|" + "\\l".join(field_labels) + "\\l")
        
        # Add methods
        if show_methods:
            methods = cls.get("methods", [])
            if methods:
                method_labels = []
                for method in methods:
                    method_name = method.get("name", "")
                    if method_name:
                        params = method.get("params", [])
                        param_str = ", ".join(params)
                        method_labels.append(f"{method_name}({param_str})")
                
                if method_labels:
                    label_parts.append("|" + "\\l".join(method_labels) + "\\l")
        
        # Add class definition
        label = "{" + "".join(label_parts) + "}"
        diagram.append(f'    {class_name} [label="{label}"];')
    
    # Add relationships
    for cls in structure.get("classes", []):
        class_name = cls.get("name", "")
        
        # Add inheritance
        extends = cls.get("extends")
        if extends:
            diagram.append(f'    {extends} -> {class_name} [arrowhead=empty];')
        
        # Add interfaces (Java)
        implements = cls.get("implements", [])
        for interface in implements:
            if interface:
                diagram.append(f'    {interface} -> {class_name} [arrowhead=empty, style=dashed];')
    
    # End diagram
    diagram.append("}")
    
    # Return diagram
    return "\n".join(diagram) 