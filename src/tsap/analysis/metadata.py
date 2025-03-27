"""
Metadata analysis tools for the TSAP MCP Server.

This module provides tools for adaptively extracting metadata from documents
and generating schemas based on discovered patterns.
"""

import re
from typing import Dict, List, Any, Optional
from collections import defaultdict

from tsap.utils.logging import logger
from tsap.analysis.base import BaseAnalysisTool, register_analysis_tool, AnalysisContext
from tsap.mcp.models import MetadataExtractorParams, MetadataExtractorResult, ExtractedMetadata
from tsap.composite.parallel import parallel_search


@register_analysis_tool("adaptive_metadata_extractor")
class AdaptiveMetadataExtractor(BaseAnalysisTool):
    """
    Analyze documents to adaptively extract metadata and build schemas.
    
    The AdaptiveMetadataExtractor progressively builds metadata models
    by sampling documents, discovering patterns, and refining extraction rules.
    """
    
    def __init__(self, name: str = "adaptive_metadata_extractor"):
        """Initialize the metadata extractor."""
        super().__init__(name)
    
    async def _discover_patterns(
        self, 
        documents: List[str],
        context: AnalysisContext
    ) -> Dict[str, Any]:
        """
        Discover potential metadata patterns in documents.
        
        Args:
            documents: List of document paths to analyze
            context: Analysis context
            
        Returns:
            Dictionary of discovered patterns
        """
        # Common metadata patterns to search for
        common_patterns = [
            {"pattern": r"title[:\s]+([^\n]+)", "description": "Document title"},
            {"pattern": r"author[s]?[:\s]+([^\n]+)", "description": "Author"},
            {"pattern": r"date[:\s]+([^\n]+)", "description": "Date"},
            {"pattern": r"version[:\s]+([^\n]+)", "description": "Version"},
            {"pattern": r"subject[:\s]+([^\n]+)", "description": "Subject"},
            {"pattern": r"keywords?[:\s]+([^\n]+)", "description": "Keywords"},
            {"pattern": r"abstract[:\s]+([^\n]+)", "description": "Abstract"},
            {"pattern": r"summary[:\s]+([^\n]+)", "description": "Summary"},
            {"pattern": r"copyright[:\s]+([^\n]+)", "description": "Copyright"},
            {"pattern": r"license[:\s]+([^\n]+)", "description": "License"}
        ]
        
        # Special patterns for common document types
        document_type_patterns = {
            "academic": [
                {"pattern": r"doi[:\s]+([^\n]+)", "description": "DOI"},
                {"pattern": r"issn[:\s]+([^\n]+)", "description": "ISSN"},
                {"pattern": r"citation[:\s]+([^\n]+)", "description": "Citation"}
            ],
            "code": [
                {"pattern": r"@version\s+([^\n]+)", "description": "Code version"},
                {"pattern": r"@author\s+([^\n]+)", "description": "Code author"},
                {"pattern": r"package\s+([a-z0-9_.]+)", "description": "Package name"}
            ],
            "technical": [
                {"pattern": r"specification[:\s]+([^\n]+)", "description": "Specification"},
                {"pattern": r"requirement[s]?[:\s]+([^\n]+)", "description": "Requirements"},
                {"pattern": r"reference[s]?[:\s]+([^\n]+)", "description": "References"}
            ]
        }
        
        # Search for common patterns
        search_result = await parallel_search({
            "patterns": common_patterns,
            "paths": documents,
            "file_patterns": None,
            "exclude_patterns": None,
            "recursive": True,
            "case_sensitive": False,
            "max_matches_per_pattern": 100,
            "context_lines": 1,
            "consolidate_matches": True,
            "consolidation_threshold": 5
        })
        
        # Detect document types based on file extensions and content patterns
        document_types = []
        if any(doc.endswith(('.tex', '.bib', '.pdf')) for doc in documents):
            document_types.append("academic")
        if any(doc.endswith(('.py', '.js', '.java', '.c', '.cpp', '.h', '.go', '.rs')) for doc in documents):
            document_types.append("code")
        if any(doc.endswith(('.md', '.txt', '.docx', '.html', '.rst')) for doc in documents):
            document_types.append("technical")
        
        # Add document-specific patterns based on detected types
        additional_patterns = []
        for doc_type in document_types:
            if doc_type in document_type_patterns:
                additional_patterns.extend(document_type_patterns[doc_type])
        
        # Search for document-type specific patterns if any were added
        if additional_patterns:
            type_search_result = await parallel_search({
                "patterns": additional_patterns,
                "paths": documents,
                "file_patterns": None,
                "exclude_patterns": None,
                "recursive": True,
                "case_sensitive": False,
                "max_matches_per_pattern": 100,
                "context_lines": 1,
                "consolidate_matches": True,
                "consolidation_threshold": 5
            })
            
            # Merge the results
            search_result.matches.extend(type_search_result.matches)
        
        # Process results to identify good metadata fields
        field_matches = defaultdict(list)
        
        for match in search_result.matches:
            pattern_desc = match.pattern_info.get("description", "Unknown")
            
            # Extract the field value using regex
            pattern = match.pattern_info.get("pattern", "")
            regex = re.compile(pattern, re.IGNORECASE)
            
            for occurrence in match.occurrences:
                m = regex.search(occurrence.get("line_text", ""))
                if m and m.groups():
                    field_value = m.group(1).strip()
                    field_matches[pattern_desc].append({
                        "value": field_value,
                        "file": occurrence.get("file_path", ""),
                        "line": occurrence.get("line_number", 0)
                    })
        
        # Count field occurrences to identify common fields
        field_counts = {field: len(matches) for field, matches in field_matches.items()}
        
        # Calculate frequency distribution
        total_count = sum(field_counts.values())
        field_frequencies = {
            field: count / total_count if total_count > 0 else 0
            for field, count in field_counts.items()
        }
        
        # Calculate confidence for each field based on consistency
        field_confidence = {}
        for field, matches in field_matches.items():
            if not matches:
                field_confidence[field] = 0
                continue
            
            # Calculate value diversity
            values = [m["value"] for m in matches]
            unique_values = set(values)
            value_diversity = len(unique_values) / len(values)
            
            # Higher diversity means lower confidence
            consistency = 1 - value_diversity
            
            # Confidence is a combination of frequency and consistency
            frequency = field_frequencies[field]
            field_confidence[field] = (0.7 * frequency + 0.3 * consistency)
        
        # Build the patterns structure
        patterns = {
            "common_fields": [
                {
                    "field": field,
                    "pattern": next((p["pattern"] for p in common_patterns if p["description"] == field), ""),
                    "frequency": field_frequencies.get(field, 0),
                    "confidence": field_confidence.get(field, 0),
                    "sample_values": [m["value"] for m in matches[:5]]
                }
                for field, matches in field_matches.items()
                if field_confidence.get(field, 0) > 0.2  # Only include fields with reasonable confidence
            ]
        }
        
        context.add_statistic("pattern_count", len(patterns["common_fields"]))
        return patterns
    
    async def _build_extraction_schema(
        self,
        patterns: Dict[str, Any],
        existing_schema: Optional[Dict[str, Any]],
        context: AnalysisContext
    ) -> Dict[str, Any]:
        """
        Build a metadata extraction schema based on discovered patterns.
        
        Args:
            patterns: Discovered patterns
            existing_schema: Optional existing schema to extend
            context: Analysis context
            
        Returns:
            Metadata extraction schema
        """
        # Start with existing schema or create a new one
        schema = existing_schema or {
            "fields": [],
            "version": 1,
            "created": context.get_parameter("timestamp", "")
        }
        
        # Add/update fields from patterns
        existing_field_names = {f["name"] for f in schema.get("fields", [])}
        
        for field in patterns.get("common_fields", []):
            field_name = field["field"].lower().replace(" ", "_")
            
            # Define field type based on sample values
            field_type = "string"  # Default type
            
            if field_name in ["date", "created", "modified", "published"]:
                field_type = "date"
            elif field_name in ["version", "revision"]:
                field_type = "version"
            elif field_name in ["keywords", "tags", "categories"]:
                field_type = "list"
            
            # Create field definition
            field_def = {
                "name": field_name,
                "display_name": field["field"],
                "type": field_type,
                "pattern": field["pattern"],
                "confidence": field["confidence"],
                "required": field["confidence"] > 0.8,
                "multi_value": field_type == "list"
            }
            
            # Add to schema if not already present
            if field_name not in existing_field_names:
                schema["fields"].append(field_def)
                existing_field_names.add(field_name)
            else:
                # Update existing field
                for i, existing_field in enumerate(schema["fields"]):
                    if existing_field["name"] == field_name:
                        # Merge definitions, keep highest confidence
                        schema["fields"][i]["confidence"] = max(
                            existing_field["confidence"],
                            field_def["confidence"]
                        )
                        schema["fields"][i]["required"] = schema["fields"][i]["confidence"] > 0.8
                        break
        
        # Sort fields by confidence
        schema["fields"].sort(key=lambda f: f["confidence"], reverse=True)
        
        # Update schema version and timestamp
        schema["version"] += 1
        schema["last_updated"] = context.get_parameter("timestamp", "")
        
        context.add_statistic("schema_fields", len(schema["fields"]))
        return schema
    
    async def _extract_metadata(
        self,
        documents: List[str],
        schema: Dict[str, Any],
        context: AnalysisContext
    ) -> List[ExtractedMetadata]:
        """
        Extract metadata from documents using the schema.
        
        Args:
            documents: List of document paths
            schema: Metadata extraction schema
            context: Analysis context
            
        Returns:
            List of extracted metadata
        """
        results = []
        
        # Process each document
        for document in documents:
            # Create patterns for each field in the schema
            field_patterns = []
            for field in schema.get("fields", []):
                if "pattern" in field and field["pattern"]:
                    field_patterns.append({
                        "pattern": field["pattern"],
                        "description": field["name"],
                        "is_regex": True,
                        "case_sensitive": False
                    })
            
            # Skip if no patterns
            if not field_patterns:
                continue
            
            # Search for all patterns in the document
            search_result = await parallel_search({
                "patterns": field_patterns,
                "paths": [document],
                "file_patterns": None,
                "exclude_patterns": None,
                "recursive": False,
                "case_sensitive": False,
                "max_matches_per_pattern": 5,
                "context_lines": 0,
                "consolidate_matches": False
            })
            
            # Extract metadata values
            metadata_values = {}
            
            for match in search_result.matches:
                field_name = match.pattern_info.get("description", "unknown")
                
                # Get field definition from schema
                field_def = next(
                    (f for f in schema.get("fields", []) if f["name"] == field_name),
                    None
                )
                
                if not field_def:
                    continue
                
                # Extract the field value using regex
                pattern = field_def.get("pattern", "")
                regex = re.compile(pattern, re.IGNORECASE)
                
                values = []
                for occurrence in match.occurrences:
                    m = regex.search(occurrence.get("line_text", ""))
                    if m and m.groups():
                        field_value = m.group(1).strip()
                        values.append(field_value)
                
                if values:
                    # Store as list or single value based on field definition
                    if field_def.get("multi_value", False):
                        metadata_values[field_name] = values
                    else:
                        # Use the first value (highest confidence)
                        metadata_values[field_name] = values[0]
            
            # Create metadata object
            metadata = ExtractedMetadata(
                document_path=document,
                metadata=metadata_values,
                schema_version=schema.get("version", 1),
                confidence=self._calculate_metadata_confidence(metadata_values, schema),
                missing_required_fields=self._get_missing_required_fields(metadata_values, schema)
            )
            
            results.append(metadata)
        
        context.add_statistic("documents_processed", len(results))
        return results
    
    def _calculate_metadata_confidence(
        self,
        metadata_values: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> float:
        """
        Calculate overall confidence for extracted metadata.
        
        Args:
            metadata_values: Extracted metadata values
            schema: Metadata schema
            
        Returns:
            Confidence score (0-1)
        """
        if not schema.get("fields"):
            return 0.0
        
        # Get all required fields
        required_fields = [
            f["name"] for f in schema.get("fields", [])
            if f.get("required", False)
        ]
        
        # Calculate required fields coverage
        if required_fields:
            required_coverage = sum(
                1 for field in required_fields if field in metadata_values
            ) / len(required_fields)
        else:
            required_coverage = 1.0
        
        # Calculate overall field coverage
        all_fields = [f["name"] for f in schema.get("fields", [])]
        if all_fields:
            all_coverage = sum(
                1 for field in all_fields if field in metadata_values
            ) / len(all_fields)
        else:
            all_coverage = 1.0
        
        # Calculate weighted confidence score
        confidence = (0.7 * required_coverage) + (0.3 * all_coverage)
        
        return confidence
    
    def _get_missing_required_fields(
        self,
        metadata_values: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> List[str]:
        """
        Get list of missing required fields.
        
        Args:
            metadata_values: Extracted metadata values
            schema: Metadata schema
            
        Returns:
            List of missing required field names
        """
        required_fields = [
            f["name"] for f in schema.get("fields", [])
            if f.get("required", False)
        ]
        
        return [field for field in required_fields if field not in metadata_values]
    
    async def analyze(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from documents and build schemas.
        
        Args:
            params: MetadataExtractorParams as a dictionary
            
        Returns:
            MetadataExtractorResult as a dictionary
        """
        params = MetadataExtractorParams.parse_obj(params)
        context = AnalysisContext()
        
        # Add timestamp for schema versioning
        import datetime
        context.add_parameter("timestamp", datetime.datetime.now().isoformat())
        
        try:
            async with self._measure_execution_time():
                # Discover metadata patterns
                if params.discover_patterns:
                    patterns = await self._discover_patterns(
                        params.documents,
                        context
                    )
                    context.add_result("patterns", patterns)
                else:
                    patterns = params.patterns or {"common_fields": []}
                
                # Build extraction schema
                schema = await self._build_extraction_schema(
                    patterns,
                    params.existing_schema,
                    context
                )
                context.add_result("schema", schema)
                
                # Extract metadata using schema
                extracted_metadata = await self._extract_metadata(
                    params.documents,
                    schema,
                    context
                )
                
                # Calculate overall statistics
                total_confidence = sum(meta.confidence for meta in extracted_metadata)
                avg_confidence = total_confidence / len(extracted_metadata) if extracted_metadata else 0
                
                # Create the result
                result = MetadataExtractorResult(
                    extracted_metadata=extracted_metadata,
                    schema=schema,
                    patterns=patterns if params.discover_patterns else None,
                    document_count=len(params.documents),
                    total_fields_discovered=len(schema.get("fields", [])),
                    average_confidence=avg_confidence,
                    execution_stats=self.get_statistics()
                )
                
                return result.dict()
        except Exception as e:
            logger.error(
                f"Error during metadata extraction: {str(e)}",
                component="analysis",
                operation="metadata_extract"
            )
            raise


async def extract_metadata(params: MetadataExtractorParams) -> MetadataExtractorResult:
    """
    Convenience function to extract metadata from documents.
    
    Args:
        params: Parameters for metadata extraction
        
    Returns:
        Results of metadata extraction
    """
    extractor = AdaptiveMetadataExtractor()
    result = await extractor.analyze(params.dict())
    return MetadataExtractorResult.parse_obj(result)