"""
Template for regulatory document analysis.

This module provides a template for analyzing regulatory documents, including
finding changes between versions, extracting requirements, and identifying
dependencies between regulatory documents.
"""

import os
import re
import time
from typing import Dict, List, Any, Optional

from tsap.utils.logging import logger
from tsap.templates.base import (
    Template, 
    TemplateResult, 
    TemplateError, 
    template_decorator
)
from tsap.composite.parallel import parallel_search, ParallelSearchParams, SearchPattern
from tsap.composite.diff_generator import generate_diff, DiffGeneratorParams
from tsap.composite.context import extract_context, ContextExtractParams
from tsap.mcp.models import DiffChunk

@template_decorator(
    name="regulatory_analysis",
    description="Analyze regulatory documents to identify changes, requirements, and dependencies",
    category="document_analysis",
    version="1.0.0"
)
class RegulatoryAnalysisTemplate(Template):
    """
    Template for regulatory document analysis.
    
    This template provides functionality to:
    - Compare different versions of regulatory documents to identify changes
    - Extract regulatory requirements from documents
    - Identify dependencies between regulatory documents
    - Track changes in regulatory requirements over time
    """
    
    def __init__(self) -> None:
        """Initialize the regulatory analysis template."""
        super().__init__()
        self._requirement_patterns = self._load_requirement_patterns()
        self._section_patterns = self._load_section_patterns()
        self._reference_patterns = self._load_reference_patterns()
    
    def _load_requirement_patterns(self) -> List[Dict[str, Any]]:
        """
        Load patterns for identifying regulatory requirements.
        
        Returns:
            List of requirement patterns
        """
        # Common patterns for regulatory requirements
        return [
            {
                "pattern": r"\b(shall|must|required|requirement)\b",
                "description": "Mandatory requirement",
                "priority": "high",
                "type": "obligation"
            },
            {
                "pattern": r"\b(should|recommended|recommendation)\b",
                "description": "Recommended practice",
                "priority": "medium",
                "type": "recommendation"
            },
            {
                "pattern": r"\b(may|optional|permitted|allowable)\b",
                "description": "Optional permission",
                "priority": "low",
                "type": "permission"
            },
            {
                "pattern": r"\b(will|can|possible)\b",
                "description": "Statement of capability or intent",
                "priority": "low",
                "type": "statement"
            },
            {
                "pattern": r"\b(prohibited|disallowed|not permitted|shall not|must not)\b",
                "description": "Prohibition",
                "priority": "high",
                "type": "prohibition"
            }
        ]
    
    def _load_section_patterns(self) -> List[Dict[str, Any]]:
        """
        Load patterns for identifying document sections.
        
        Returns:
            List of section patterns
        """
        # Common patterns for regulatory document sections
        return [
            {
                "pattern": r"^\s*(?:Section|SECTION)\s+(\d+(?:\.\d+)*)",
                "description": "Section header",
                "type": "section"
            },
            {
                "pattern": r"^\s*(?:Article|ARTICLE)\s+(\d+)",
                "description": "Article header",
                "type": "article"
            },
            {
                "pattern": r"^\s*(?:Chapter|CHAPTER)\s+(\d+)",
                "description": "Chapter header",
                "type": "chapter"
            },
            {
                "pattern": r"^\s*(?:Appendix|APPENDIX)\s+([A-Z])",
                "description": "Appendix header",
                "type": "appendix"
            },
            {
                "pattern": r"^\s*(?:\d+(?:\.\d+)*)\s+(.+)$",
                "description": "Numbered section with title",
                "type": "numbered_section"
            }
        ]
    
    def _load_reference_patterns(self) -> List[Dict[str, Any]]:
        """
        Load patterns for identifying references to other regulatory documents.
        
        Returns:
            List of reference patterns
        """
        # Common patterns for references to other documents
        return [
            {
                "pattern": r"(?:refer(?:red)?\s+to|see|reference[sd]?|as\s+specified\s+in|according\s+to)\s+([^\.]+)",
                "description": "Document reference",
                "type": "reference"
            },
            {
                "pattern": r"(?:compliance|complies|conform[s]?)\s+(?:with|to)\s+([^\.]+)",
                "description": "Compliance reference",
                "type": "compliance"
            },
            {
                "pattern": r"(?:ISO|IEC|ANSI|NIST|CFR|Directive)\s+(\d+(?:[-:]\d+)?(?:[-:]\d+)?)",
                "description": "Standard reference",
                "type": "standard"
            }
        ]
    
    async def _identify_document_versions(self, document_paths: List[str]) -> Dict[str, List[str]]:
        """
        Group documents by their base name to identify different versions.
        
        Args:
            document_paths: List of document paths
            
        Returns:
            Dictionary mapping base document names to lists of versions
        """
        # Group documents by base name (removing version information)
        document_groups: Dict[str, List[str]] = {}
        
        for path in document_paths:
            # Extract filename
            filename = os.path.basename(path)
            
            # Try to identify base name by removing version information
            # Common patterns: filename_v1.0.pdf, filename-2021.pdf, filename_2021-05-01.pdf
            base_name = re.sub(r'[-_]v?\d+(?:\.\d+)*(?:[-_]\d+)*(?:[-_]\d{4}(?:[-_]\d{2})?(?:[-_]\d{2})?)?(?=\.[^.]+$)', '', filename)
            base_name = re.sub(r'[-_]\d{4}(?:[-_]\d{2})?(?:[-_]\d{2})?(?=\.[^.]+$)', '', base_name)
            
            # Add to the appropriate group
            if base_name not in document_groups:
                document_groups[base_name] = []
            document_groups[base_name].append(path)
        
        # Sort versions within each group (assuming version information is in the filename)
        for base_name, versions in document_groups.items():
            versions.sort()
        
        return document_groups
    
    async def _extract_version_info(self, document_path: str) -> Dict[str, Any]:
        """
        Extract version information from a document.
        
        Args:
            document_path: Path to the document
            
        Returns:
            Dictionary with version information
        """
        # Extract filename and look for version patterns
        filename = os.path.basename(document_path)
        
        # Look for version number pattern
        version_match = re.search(r'v?(\d+(?:\.\d+)*)', filename)
        version = version_match.group(1) if version_match else None
        
        # Look for date pattern
        date_match = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})', filename)
        date = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}" if date_match else None
        
        # Simple algorithm to extract version information from the filename
        # More sophisticated approaches would examine the document content
        return {
            "path": document_path,
            "filename": filename,
            "version": version,
            "date": date,
            # Add more version information as needed
        }
    
    async def _compare_document_versions(self, older_version: str, newer_version: str) -> Dict[str, Any]:
        """
        Compare two versions of a document to identify changes.
        
        Args:
            older_version: Path to the older version of the document
            newer_version: Path to the newer version of the document
            
        Returns:
            Dictionary with comparison results
        """
        # Use the diff_generator to generate differences
        diff_params = DiffGeneratorParams(
            source_path=older_version,
            target_path=newer_version,
            context_lines=2,
            # Add more parameters as needed
        )
        
        diff_result = await generate_diff(diff_params)
        
        # Classify changes
        return await self._classify_changes(diff_result.chunks)
    
    async def _classify_changes(self, diff_chunks: List[DiffChunk]) -> Dict[str, Any]:
        """
        Classify the changes in the diff chunks.
        
        Args:
            diff_chunks: List of diff chunks
            
        Returns:
            Dictionary with classified changes
        """
        # Initialize counters for different types of changes
        results = {
            "total_chunks": len(diff_chunks),
            "added_chunks": 0,
            "removed_chunks": 0,
            "modified_chunks": 0,
            "requirement_changes": {
                "added": 0,
                "removed": 0,
                "modified": 0
            },
            "section_changes": {
                "added": 0,
                "removed": 0,
                "modified": 0
            },
            "significant_changes": [],
            "change_summary": {}
        }
        
        # Analyze each diff chunk
        for chunk in diff_chunks:
            # Determine chunk type
            if chunk.chunk_type == "added":
                results["added_chunks"] += 1
                
                # Check if the added chunk contains requirements
                for pattern in self._requirement_patterns:
                    if re.search(pattern["pattern"], chunk.content, re.IGNORECASE):
                        results["requirement_changes"]["added"] += 1
                        if pattern["priority"] == "high":
                            results["significant_changes"].append({
                                "type": "added_requirement",
                                "content": chunk.content,
                                "pattern": pattern["pattern"],
                                "line": chunk.target_start
                            })
                        break
            
            elif chunk.chunk_type == "removed":
                results["removed_chunks"] += 1
                
                # Check if the removed chunk contains requirements
                for pattern in self._requirement_patterns:
                    if re.search(pattern["pattern"], chunk.content, re.IGNORECASE):
                        results["requirement_changes"]["removed"] += 1
                        if pattern["priority"] == "high":
                            results["significant_changes"].append({
                                "type": "removed_requirement",
                                "content": chunk.content,
                                "pattern": pattern["pattern"],
                                "line": chunk.source_start
                            })
                        break
            
            elif chunk.chunk_type == "modified":
                results["modified_chunks"] += 1
                
                # Check if the modified chunk contains requirements
                for pattern in self._requirement_patterns:
                    if (re.search(pattern["pattern"], chunk.source_content, re.IGNORECASE) or 
                        re.search(pattern["pattern"], chunk.target_content, re.IGNORECASE)):
                        results["requirement_changes"]["modified"] += 1
                        if pattern["priority"] == "high":
                            results["significant_changes"].append({
                                "type": "modified_requirement",
                                "source_content": chunk.source_content,
                                "target_content": chunk.target_content,
                                "pattern": pattern["pattern"],
                                "source_line": chunk.source_start,
                                "target_line": chunk.target_start
                            })
                        break
        
        # Generate a summary of the changes
        results["change_summary"] = {
            "total_chunks": results["total_chunks"],
            "added_chunks": results["added_chunks"],
            "removed_chunks": results["removed_chunks"],
            "modified_chunks": results["modified_chunks"],
            "added_requirements": results["requirement_changes"]["added"],
            "removed_requirements": results["requirement_changes"]["removed"],
            "modified_requirements": results["requirement_changes"]["modified"],
            "significant_changes": len(results["significant_changes"])
        }
        
        return results
    
    async def _extract_requirements(self, document_path: str) -> Dict[str, Any]:
        """
        Extract regulatory requirements from a document.
        
        Args:
            document_path: Path to the document
            
        Returns:
            Dictionary with extracted requirements
        """
        # Prepare search patterns
        search_patterns = []
        for pattern in self._requirement_patterns:
            search_patterns.append(SearchPattern(
                pattern=pattern["pattern"],
                description=pattern["description"],
                is_regex=True,
                case_sensitive=False
            ))
        
        # Perform parallel search for all requirement patterns
        search_params = ParallelSearchParams(
            patterns=search_patterns,
            paths=[document_path],
            context_lines=2,
            consolidate_matches=True
        )
        
        search_result = await parallel_search(search_params)
        
        # Process matches to extract requirements
        requirements = []
        for match in search_result.matches:
            # Extract context for each match
            context_params = ContextExtractParams(
                match={
                    "file": match.file,
                    "line": match.line,
                    "content": match.content
                },
                context_type="paragraph"
            )
            context_result = await extract_context(context_params)
            
            # Determine the section of the document
            section = await self._determine_section(document_path, match.line)
            
            # Add the requirement
            requirements.append({
                "file": match.file,
                "line": match.line,
                "pattern": match.pattern,
                "pattern_description": match.pattern_description,
                "content": match.content,
                "context": context_result.context[0].content if context_result.context else match.content,
                "section": section,
                "requirement_type": next((p["type"] for p in self._requirement_patterns if p["pattern"] == match.pattern), "unknown"),
                "priority": next((p["priority"] for p in self._requirement_patterns if p["pattern"] == match.pattern), "medium")
            })
        
        # Group requirements by section
        requirements_by_section = {}
        for req in requirements:
            section = req["section"] or "unknown"
            if section not in requirements_by_section:
                requirements_by_section[section] = []
            requirements_by_section[section].append(req)
        
        return {
            "total_requirements": len(requirements),
            "requirements": requirements,
            "requirements_by_section": requirements_by_section,
            "document_path": document_path
        }
    
    async def _determine_section(self, document_path: str, line_number: int) -> Optional[str]:
        """
        Determine the section of the document for a given line number.
        
        Args:
            document_path: Path to the document
            line_number: Line number
            
        Returns:
            Section title or None if not found
        """
        # Prepare search patterns
        search_patterns = []
        for pattern in self._section_patterns:
            search_patterns.append(SearchPattern(
                pattern=pattern["pattern"],
                description=pattern["description"],
                is_regex=True,
                case_sensitive=False
            ))
        
        # Perform parallel search for section patterns
        search_params = ParallelSearchParams(
            patterns=search_patterns,
            paths=[document_path],
            context_lines=0
        )
        
        search_result = await parallel_search(search_params)
        
        # Find the most recent section header before the given line number
        section_matches = sorted(search_result.matches, key=lambda m: m.line)
        current_section = None
        
        for match in section_matches:
            if match.line > line_number:
                break
            current_section = match.content.strip()
        
        return current_section
    
    async def _identify_references(self, document_path: str) -> Dict[str, Any]:
        """
        Identify references to other regulatory documents.
        
        Args:
            document_path: Path to the document
            
        Returns:
            Dictionary with identified references
        """
        # Prepare search patterns
        search_patterns = []
        for pattern in self._reference_patterns:
            search_patterns.append(SearchPattern(
                pattern=pattern["pattern"],
                description=pattern["description"],
                is_regex=True,
                case_sensitive=False
            ))
        
        # Perform parallel search for reference patterns
        search_params = ParallelSearchParams(
            patterns=search_patterns,
            paths=[document_path],
            context_lines=1
        )
        
        search_result = await parallel_search(search_params)
        
        # Process matches to extract references
        references = []
        for match in search_result.matches:
            # Try to extract the referenced document
            ref_match = re.search(match.pattern, match.content, re.IGNORECASE)
            if ref_match and len(ref_match.groups()) > 0:
                referenced_doc = ref_match.group(1).strip()
                
                # Add the reference
                references.append({
                    "file": match.file,
                    "line": match.line,
                    "pattern": match.pattern,
                    "pattern_description": match.pattern_description,
                    "content": match.content,
                    "referenced_document": referenced_doc,
                    "reference_type": next((p["type"] for p in self._reference_patterns if p["pattern"] == match.pattern), "unknown")
                })
        
        # Group references by type
        references_by_type = {}
        for ref in references:
            ref_type = ref["reference_type"]
            if ref_type not in references_by_type:
                references_by_type[ref_type] = []
            references_by_type[ref_type].append(ref)
        
        # Count unique referenced documents
        unique_references = set(ref["referenced_document"] for ref in references)
        
        return {
            "total_references": len(references),
            "unique_references": len(unique_references),
            "references": references,
            "references_by_type": references_by_type,
            "document_path": document_path
        }
    
    async def _analyze_requirement_dependencies(self, document_paths: List[str]) -> Dict[str, Any]:
        """
        Analyze dependencies between requirements in different documents.
        
        Args:
            document_paths: List of document paths
            
        Returns:
            Dictionary with requirement dependencies
        """
        # Extract requirements from all documents
        requirements_by_document = {}
        for path in document_paths:
            requirements_result = await self._extract_requirements(path)
            requirements_by_document[path] = requirements_result
        
        # Identify references in all documents
        references_by_document = {}
        for path in document_paths:
            references_result = await self._identify_references(path)
            references_by_document[path] = references_result
        
        # Build a graph of dependencies
        dependencies = []
        
        # Simple algorithm to identify potential dependencies
        # A more sophisticated approach would use NLP or other techniques
        for doc_path, references in references_by_document.items():
            for ref in references["references"]:
                # Try to find a document that matches the reference
                matching_docs = [p for p in document_paths if os.path.basename(p).lower() in ref["referenced_document"].lower()]
                
                if matching_docs:
                    for matching_doc in matching_docs:
                        dependencies.append({
                            "source_document": doc_path,
                            "target_document": matching_doc,
                            "reference_line": ref["line"],
                            "reference_content": ref["content"],
                            "reference_type": ref["reference_type"]
                        })
        
        return {
            "total_dependencies": len(dependencies),
            "dependencies": dependencies,
            "requirements_by_document": {path: result["total_requirements"] for path, result in requirements_by_document.items()},
            "references_by_document": {path: result["total_references"] for path, result in references_by_document.items()}
        }
    
    async def execute(self, params: Dict[str, Any]) -> TemplateResult:
        """
        Execute the regulatory analysis template.
        
        Args:
            params: Parameters for the analysis
                - document_paths: List of document paths
                - analysis_type: Type of analysis to perform
                  ("version_comparison", "requirement_extraction", "reference_identification", "dependency_analysis")
                - older_version: Path to the older version (for version comparison)
                - newer_version: Path to the newer version (for version comparison)
            
        Returns:
            Result of the analysis
        
        Raises:
            TemplateError: If the analysis fails
        """
        start_time = time.time()
        
        try:
            # Get parameters
            document_paths = params.get("document_paths", [])
            analysis_type = params.get("analysis_type", "requirement_extraction")
            older_version = params.get("older_version")
            newer_version = params.get("newer_version")
            
            # Validate parameters
            if not document_paths and not (older_version and newer_version):
                raise TemplateError("No documents provided for analysis")
            
            # Perform the requested analysis
            results = {}
            
            if analysis_type == "version_comparison":
                # Ensure older and newer versions are provided
                if not older_version or not newer_version:
                    raise TemplateError("Both older_version and newer_version must be provided for version comparison")
                
                # Compare document versions
                results = await self._compare_document_versions(older_version, newer_version)
                
                # Add version information
                older_info = await self._extract_version_info(older_version)
                newer_info = await self._extract_version_info(newer_version)
                
                results["older_version"] = older_info
                results["newer_version"] = newer_info
                
            elif analysis_type == "requirement_extraction":
                # Extract requirements from each document
                requirements_by_document = {}
                for path in document_paths:
                    requirements_result = await self._extract_requirements(path)
                    requirements_by_document[path] = requirements_result
                
                # Summarize results
                total_requirements = sum(result["total_requirements"] for result in requirements_by_document.values())
                
                results = {
                    "total_documents": len(document_paths),
                    "total_requirements": total_requirements,
                    "requirements_by_document": requirements_by_document
                }
                
            elif analysis_type == "reference_identification":
                # Identify references in each document
                references_by_document = {}
                for path in document_paths:
                    references_result = await self._identify_references(path)
                    references_by_document[path] = references_result
                
                # Summarize results
                total_references = sum(result["total_references"] for result in references_by_document.values())
                
                results = {
                    "total_documents": len(document_paths),
                    "total_references": total_references,
                    "references_by_document": references_by_document
                }
                
            elif analysis_type == "dependency_analysis":
                # Analyze requirement dependencies
                results = await self._analyze_requirement_dependencies(document_paths)
                
                # Add document version information
                document_versions = {}
                for path in document_paths:
                    version_info = await self._extract_version_info(path)
                    document_versions[path] = version_info
                
                results["document_versions"] = document_versions
                
            else:
                raise TemplateError(f"Unknown analysis type: {analysis_type}")
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Create template result
            return TemplateResult(
                success=True,
                template_name="regulatory_analysis",
                template_version="1.0.0",
                execution_time=execution_time,
                results=results,
                summary={
                    "analysis_type": analysis_type,
                    "document_count": len(document_paths) if document_paths else 2,
                    "execution_time": execution_time
                }
            )
            
        except Exception as e:
            # Log the error
            logger.error(f"Error in regulatory analysis: {str(e)}")
            
            # Create error result
            return TemplateResult(
                success=False,
                template_name="regulatory_analysis",
                template_version="1.0.0",
                execution_time=time.time() - start_time,
                error=str(e),
                results={},
                summary={
                    "error": str(e)
                }
            )


async def run_regulatory_analysis(params: Dict[str, Any]) -> TemplateResult:
    """
    Run the regulatory analysis template.
    
    Args:
        params: Parameters for the analysis
    
    Returns:
        Result of the analysis
    """
    template = RegulatoryAnalysisTemplate()
    return await template.execute(params)