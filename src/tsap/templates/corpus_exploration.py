"""
Template for exploring a document corpus.

This module provides a template for exploring a corpus of documents, including
discovering patterns, clustering documents, extracting key information, and
visualizing the corpus structure.
"""

import os
import re
import time
from typing import Dict, List, Any, Optional
import random
from collections import Counter, defaultdict

from tsap.utils.logging import logger
from tsap.templates.base import (
    Template, 
    TemplateResult, 
    TemplateError, 
    template_decorator
)
from tsap.composite.parallel import parallel_search, ParallelSearchParams, SearchPattern
from tsap.composite.document_profiler import profile_documents
from tsap.composite.filenames import discover_filename_patterns
from tsap.analysis.documents import explore_documents


@template_decorator(
    name="corpus_exploration",
    description="Explore a corpus of documents to discover patterns, clusters, and key information",
    category="document_analysis",
    version="1.0.0"
)
class CorpusExplorationTemplate(Template):
    """
    Template for exploring a document corpus.
    
    This template provides functionality to:
    - Discover and profile documents within a directory structure
    - Identify patterns and themes across documents
    - Cluster documents based on content similarity
    - Extract key information and visualize the corpus structure
    """
    
    def __init__(self) -> None:
        """Initialize the corpus exploration template."""
        super().__init__()
        self._entity_patterns = self._load_entity_patterns()
        self._content_type_patterns = self._load_content_type_patterns()
    
    def _load_entity_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load patterns for identifying named entities.
        
        Returns:
            Dictionary mapping entity types to lists of patterns
        """
        # Common patterns for named entities
        return {
            "person": [
                {
                    "pattern": r"(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+",
                    "description": "Person with title"
                },
                {
                    "pattern": r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}",
                    "description": "Name (first and last)"
                }
            ],
            "organization": [
                {
                    "pattern": r"(?:Inc\.|Corp\.|Ltd\.|LLC|LLP)(?:\.|\b)",
                    "description": "Organization with legal suffix"
                },
                {
                    "pattern": r"(?:Company|Corporation|University|Institute|Association|Foundation)",
                    "description": "Organization type"
                }
            ],
            "location": [
                {
                    "pattern": r"(?:St\.|Ave\.|Rd\.|Blvd\.|Ln\.|Dr\.)\s+(?:N|S|E|W|NE|NW|SE|SW)?(?:\.|\b)",
                    "description": "Street address"
                },
                {
                    "pattern": r"[A-Z][a-z]+(?:,\s+[A-Z]{2})?(?:,\s+\d{5}(?:-\d{4})?)?",
                    "description": "City, state, ZIP"
                }
            ],
            "date": [
                {
                    "pattern": r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}",
                    "description": "Date (MM/DD/YYYY)"
                },
                {
                    "pattern": r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}",
                    "description": "Date (Month DD, YYYY)"
                }
            ],
            "email": [
                {
                    "pattern": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
                    "description": "Email address"
                }
            ],
            "phone": [
                {
                    "pattern": r"(?:\+\d{1,3}\s*)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}",
                    "description": "Phone number"
                }
            ],
            "url": [
                {
                    "pattern": r"(?:https?://)?(?:www\.)?[a-zA-Z0-9-]+(?:\.[a-zA-Z]{2,})+(?:/[a-zA-Z0-9-._~:/?#[\]@!$&'()*+,;=]*)?",
                    "description": "URL"
                }
            ]
        }
    
    def _load_content_type_patterns(self) -> List[Dict[str, Any]]:
        """
        Load patterns for identifying content types.
        
        Returns:
            List of content type patterns
        """
        # Common patterns for document content types
        return [
            {
                "pattern": r"(?:Introduction|Abstract|Summary|Overview)",
                "description": "Introductory content",
                "type": "introduction"
            },
            {
                "pattern": r"(?:Conclusion|Results|Discussion|Summary)",
                "description": "Concluding content",
                "type": "conclusion"
            },
            {
                "pattern": r"(?:References|Bibliography|Works Cited|Sources|Citations)",
                "description": "References section",
                "type": "references"
            },
            {
                "pattern": r"(?:Appendix|Appendices|Supplementary|Additional|Extra)",
                "description": "Appendix section",
                "type": "appendix"
            },
            {
                "pattern": r"(?:Figure|Fig\.|Table|Chart|Graph)\s+\d+",
                "description": "Visual content",
                "type": "visual"
            },
            {
                "pattern": r"(?:Methodology|Method|Procedure|Process|Steps)",
                "description": "Methodology content",
                "type": "methodology"
            },
            {
                "pattern": r"(?:Data|Variables|Dataset|Sample)",
                "description": "Data description",
                "type": "data"
            }
        ]
    
    async def _discover_documents(self, directory: str, file_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Discover documents within a directory structure.
        
        Args:
            directory: Directory path
            file_patterns: Optional list of file patterns to match
            
        Returns:
            Dictionary with discovered documents
        """
        # Use the DocumentExplorer to discover documents
        explorer_params = {
            "directory_path": directory,
            "file_patterns": file_patterns,
            "recursive": True,
            "max_depth": None,
            "extract_metadata": True,
            "extract_full_text": False,
            "extract_summaries": True,
            "categorize": True
        }
        
        explorer_result = await explore_documents(explorer_params)
        
        # Analyze file naming patterns
        filename_pattern_result = await discover_filename_patterns({
            "directory_path": directory,
            "recursive": True,
            "file_types": file_patterns
        })
        
        return {
            "total_documents": len(explorer_result.documents),
            "documents": explorer_result.documents,
            "categories": explorer_result.categories,
            "directory": directory,
            "filename_patterns": filename_pattern_result.patterns,
            "extensions": filename_pattern_result.extensions,
            "directory_structure": filename_pattern_result.directories
        }
    
    async def _profile_documents(self, documents: List[str]) -> Dict[str, Any]:
        """
        Create profiles for a list of documents.
        
        Args:
            documents: List of document paths
            
        Returns:
            Dictionary with document profiles
        """
        # Use the document profiler to profile all documents
        profiler_result = await profile_documents({
            "document_paths": documents,
            "include_content_features": True,
            "compare_documents": True
        })
        
        return profiler_result
    
    async def _extract_entities(self, documents: List[str]) -> Dict[str, Any]:
        """
        Extract named entities from documents.
        
        Args:
            documents: List of document paths
            
        Returns:
            Dictionary with extracted entities
        """
        entities_by_type = {}
        
        # For each entity type, create a parallel search for all documents
        for entity_type, patterns in self._entity_patterns.items():
            # Convert patterns to SearchPattern objects
            search_patterns = [
                SearchPattern(
                    pattern=p["pattern"],
                    description=p["description"],
                    is_regex=True,
                    case_sensitive=False
                )
                for p in patterns
            ]
            
            # Perform parallel search
            search_params = ParallelSearchParams(
                patterns=search_patterns,
                paths=documents,
                context_lines=1,
                max_matches=5000  # Limit number of matches
            )
            
            search_result = await parallel_search(search_params)
            
            # Extract entities from matches
            entities = []
            for match in search_result.matches:
                # Extract the actual entity from the match content
                entity_match = re.search(match.pattern, match.content, re.IGNORECASE)
                if entity_match:
                    entity_text = entity_match.group(0)
                    entities.append({
                        "text": entity_text,
                        "file": match.file,
                        "line": match.line,
                        "pattern": match.pattern,
                        "pattern_description": match.pattern_description,
                        "context": match.context
                    })
            
            # Remove duplicates while preserving order
            unique_entities = []
            seen_texts = set()
            for entity in entities:
                if entity["text"] not in seen_texts:
                    seen_texts.add(entity["text"])
                    unique_entities.append(entity)
            
            # Store entities for this type
            entities_by_type[entity_type] = unique_entities
        
        # Count entities by type
        entity_counts = {entity_type: len(entities) for entity_type, entities in entities_by_type.items()}
        total_entities = sum(entity_counts.values())
        
        # Count mentions by entity
        entity_mentions = {}
        for entity_type, entities in entities_by_type.items():
            entity_texts = [entity["text"] for entity in entities]
            mention_counter = Counter(entity_texts)
            entity_mentions[entity_type] = [
                {"text": text, "count": count}
                for text, count in mention_counter.most_common(50)  # Top 50 entities by mentions
            ]
        
        return {
            "total_entities": total_entities,
            "entity_counts": entity_counts,
            "entities_by_type": entities_by_type,
            "entity_mentions": entity_mentions
        }
    
    async def _identify_themes(self, documents: List[str]) -> Dict[str, Any]:
        """
        Identify themes and topics across documents.
        
        Args:
            documents: List of document paths
            
        Returns:
            Dictionary with identified themes
        """
        # This would ideally use NLP techniques like topic modeling
        # For now, use a simplified keyword-based approach
        
        # Predefined themes with associated keywords
        themes = {
            "technology": [
                "technology", "digital", "software", "hardware", "computer", "internet", "innovation",
                "data", "cloud", "application", "system", "platform", "technical", "network"
            ],
            "business": [
                "business", "company", "corporation", "market", "industry", "strategy", "customer",
                "client", "revenue", "profit", "cost", "budget", "financial", "investment"
            ],
            "science": [
                "science", "scientific", "research", "experiment", "hypothesis", "theory", "laboratory",
                "biology", "chemistry", "physics", "mathematics", "analysis", "methodology", "observation"
            ],
            "healthcare": [
                "health", "medical", "patient", "doctor", "hospital", "treatment", "diagnosis", "pharmacy",
                "medication", "disease", "symptoms", "therapy", "clinical", "healthcare"
            ],
            "education": [
                "education", "school", "university", "college", "student", "teacher", "learning", "academic",
                "course", "curriculum", "degree", "study", "teaching", "classroom"
            ],
            "legal": [
                "legal", "law", "court", "judge", "attorney", "regulation", "compliance", "contract",
                "legislation", "rights", "regulatory", "statute", "jurisdiction", "settlement"
            ],
            "environmental": [
                "environment", "sustainability", "climate", "pollution", "renewable", "conservation", "ecological",
                "energy", "green", "emission", "waste", "biodiversity", "sustainable", "carbon"
            ]
        }
        
        # Prepare search patterns for each theme
        theme_results = {}
        
        for theme_name, keywords in themes.items():
            # Create a regex pattern for the theme keywords
            pattern = r'\b(?:' + '|'.join(keywords) + r')\b'
            
            # Create search pattern
            search_pattern = SearchPattern(
                pattern=pattern,
                description=f"{theme_name} theme",
                is_regex=True,
                case_sensitive=False
            )
            
            # Perform search
            search_params = ParallelSearchParams(
                patterns=[search_pattern],
                paths=documents,
                max_matches=10000  # Limit number of matches
            )
            
            search_result = await parallel_search(search_params)
            
            # Count matches by document
            matches_by_document = defaultdict(int)
            for match in search_result.matches:
                matches_by_document[match.file] += 1
            
            # Threshold for considering a document to belong to a theme
            # (at least 5 mentions of theme keywords)
            threshold = 5
            documents_with_theme = [
                {"document": doc, "matches": count}
                for doc, count in matches_by_document.items()
                if count >= threshold
            ]
            
            # Store theme results
            theme_results[theme_name] = {
                "total_matches": len(search_result.matches),
                "documents_with_theme": documents_with_theme,
                "document_count": len(documents_with_theme),
                "average_matches_per_document": (len(search_result.matches) / len(documents) 
                                                if documents else 0)
            }
        
        # Determine primary theme for each document
        document_themes = {}
        for document in documents:
            document_themes[document] = []
            
            for theme_name, theme_data in theme_results.items():
                doc_matches = next(
                    (item["matches"] for item in theme_data["documents_with_theme"] 
                     if item["document"] == document), 
                    0
                )
                
                if doc_matches >= 5:  # Threshold for including a theme
                    document_themes[document].append({
                        "theme": theme_name,
                        "matches": doc_matches
                    })
            
            # Sort themes by match count
            document_themes[document].sort(key=lambda x: x["matches"], reverse=True)
        
        # Overall theme distribution
        theme_distribution = {
            theme_name: theme_data["document_count"] 
            for theme_name, theme_data in theme_results.items()
        }
        
        return {
            "themes": theme_results,
            "document_themes": document_themes,
            "theme_distribution": theme_distribution
        }
    
    async def _cluster_documents(self, documents: List[str], profile_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cluster documents based on content similarity.
        
        Args:
            documents: List of document paths
            profile_result: Result from _profile_documents
            
        Returns:
            Dictionary with document clusters
        """
        # Simplified document clustering based on similarity matrix
        # A more sophisticated approach would use proper clustering algorithms
        
        # Extract similarity matrix from profile result
        similarity_matrix = profile_result.get("similarity_matrix", {})
        
        # Create clusters
        clusters = []
        remaining_documents = set(documents)
        
        # Simple greedy clustering algorithm
        while remaining_documents:
            # Select a random document as the cluster center
            center = random.choice(list(remaining_documents))
            remaining_documents.remove(center)
            
            # Find similar documents
            cluster_documents = [center]
            
            for doc in list(remaining_documents):
                similarity_key = f"{center}:{doc}"
                alt_key = f"{doc}:{center}"
                
                # Check similarity against the center document
                similarity = similarity_matrix.get(similarity_key, similarity_matrix.get(alt_key, 0))
                
                # If similarity is above threshold, add to cluster
                if similarity >= 0.5:  # Threshold for similarity
                    cluster_documents.append(doc)
                    remaining_documents.remove(doc)
            
            # Add cluster if it has at least one document
            if cluster_documents:
                # Calculate average similarity within cluster
                internal_similarities = []
                for i, doc1 in enumerate(cluster_documents):
                    for doc2 in cluster_documents[i+1:]:
                        similarity_key = f"{doc1}:{doc2}"
                        alt_key = f"{doc2}:{doc1}"
                        similarity = similarity_matrix.get(similarity_key, similarity_matrix.get(alt_key, 0))
                        internal_similarities.append(similarity)
                
                avg_similarity = sum(internal_similarities) / len(internal_similarities) if internal_similarities else 0
                
                clusters.append({
                    "center": center,
                    "documents": cluster_documents,
                    "size": len(cluster_documents),
                    "average_similarity": avg_similarity
                })
        
        # Sort clusters by size
        clusters.sort(key=lambda c: c["size"], reverse=True)
        
        # Generate suggested names for clusters based on document names
        for cluster in clusters:
            # Extract base names of documents
            base_names = [os.path.splitext(os.path.basename(doc))[0] for doc in cluster["documents"]]
            
            # Extract words from base names
            words = []
            for name in base_names:
                words.extend(re.findall(r'\w+', name.lower()))
            
            # Count word frequency
            word_counts = Counter(words)
            
            # Filter out common words and short words
            filtered_words = [word for word, count in word_counts.items() 
                             if count > 1 and len(word) > 3 and word not in ["document", "file"]]
            
            # Generate suggested name
            if filtered_words:
                cluster["suggested_name"] = " ".join(filtered_words[:3]).title()
            else:
                cluster["suggested_name"] = f"Cluster {clusters.index(cluster) + 1}"
        
        return {
            "total_clusters": len(clusters),
            "clusters": clusters,
            "similarity_matrix": similarity_matrix
        }
    
    async def _analyze_content_types(self, documents: List[str]) -> Dict[str, Any]:
        """
        Analyze content types within documents.
        
        Args:
            documents: List of document paths
            
        Returns:
            Dictionary with content type analysis
        """
        # Prepare search patterns for content types
        search_patterns = [
            SearchPattern(
                pattern=p["pattern"],
                description=p["description"],
                is_regex=True,
                case_sensitive=False
            )
            for p in self._content_type_patterns
        ]
        
        # Perform parallel search
        search_params = ParallelSearchParams(
            patterns=search_patterns,
            paths=documents,
            context_lines=2
        )
        
        search_result = await parallel_search(search_params)
        
        # Group matches by content type
        content_types = {}
        for p in self._content_type_patterns:
            content_type = p["type"]
            content_types[content_type] = []
        
        for match in search_result.matches:
            # Find the content type for this match
            for p in self._content_type_patterns:
                if p["description"] == match.pattern_description:
                    content_type = p["type"]
                    content_types[content_type].append({
                        "file": match.file,
                        "line": match.line,
                        "content": match.content,
                        "context": match.context
                    })
                    break
        
        # Count content types by document
        content_types_by_document = defaultdict(lambda: defaultdict(int))
        for content_type, matches in content_types.items():
            for match in matches:
                content_types_by_document[match["file"]][content_type] += 1
        
        # Determine predominant content type for each document
        document_content_types = {}
        for document, type_counts in content_types_by_document.items():
            if type_counts:
                predominant_type = max(type_counts.items(), key=lambda x: x[1])
                document_content_types[document] = {
                    "predominant_type": predominant_type[0],
                    "count": predominant_type[1],
                    "all_types": [{"type": t, "count": c} for t, c in type_counts.items()]
                }
        
        return {
            "content_types": content_types,
            "document_content_types": document_content_types
        }
    
    async def _generate_corpus_map(self, documents: List[str], profile_result: Dict[str, Any], cluster_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a map of the document corpus.
        
        Args:
            documents: List of document paths
            profile_result: Result from _profile_documents
            cluster_result: Result from _cluster_documents
            
        Returns:
            Dictionary with corpus map
        """
        # Build a graph representation of the corpus
        nodes = []
        edges = []
        
        # Add document nodes
        for doc in documents:
            # Get document name
            doc_name = os.path.basename(doc)
            
            # Get document cluster
            doc_cluster = None
            for i, cluster in enumerate(cluster_result["clusters"]):
                if doc in cluster["documents"]:
                    doc_cluster = i
                    break
            
            # Get document size (if available)
            doc_size = 0
            if "document_details" in profile_result:
                for detail in profile_result["document_details"]:
                    if detail["document_path"] == doc:
                        doc_size = detail.get("size", 0)
                        break
            
            # Add node
            nodes.append({
                "id": doc,
                "name": doc_name,
                "type": "document",
                "cluster": doc_cluster,
                "size": doc_size
            })
        
        # Add cluster nodes
        for i, cluster in enumerate(cluster_result["clusters"]):
            nodes.append({
                "id": f"cluster_{i}",
                "name": cluster["suggested_name"],
                "type": "cluster",
                "size": len(cluster["documents"])
            })
            
            # Add edges from cluster to documents
            for doc in cluster["documents"]:
                edges.append({
                    "source": f"cluster_{i}",
                    "target": doc,
                    "type": "contains"
                })
        
        # Add similarity edges between documents
        similarity_matrix = cluster_result["similarity_matrix"]
        for doc1 in documents:
            for doc2 in documents:
                if doc1 != doc2:
                    similarity_key = f"{doc1}:{doc2}"
                    if similarity_key in similarity_matrix:
                        similarity = similarity_matrix[similarity_key]
                        if similarity >= 0.7:  # Only show high similarities
                            edges.append({
                                "source": doc1,
                                "target": doc2,
                                "type": "similar",
                                "weight": similarity
                            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "total_nodes": len(nodes),
            "total_edges": len(edges)
        }
    
    async def execute(self, params: Dict[str, Any]) -> TemplateResult:
        """
        Execute the corpus exploration template.
        
        Args:
            params: Parameters for the exploration
                - directory: Directory path containing the document corpus
                - file_patterns: Optional list of file patterns to match
                - analysis_types: List of analysis types to perform
                  ("discovery", "profiling", "entities", "themes", "clustering", "content_types", "mapping")
                - documents: Optional explicit list of document paths (if not using directory)
            
        Returns:
            Result of the exploration
        
        Raises:
            TemplateError: If the exploration fails
        """
        start_time = time.time()
        
        try:
            # Get parameters
            directory = params.get("directory")
            file_patterns = params.get("file_patterns")
            analysis_types = params.get("analysis_types", ["discovery", "profiling"])
            documents = params.get("documents", [])
            
            # Validate parameters
            if not directory and not documents:
                raise TemplateError("Either directory or documents must be provided")
            
            # Perform document discovery if needed
            discovered_documents = []
            discovery_result = {}
            
            if "discovery" in analysis_types or not documents:
                if directory:
                    # Discover documents
                    discovery_result = await self._discover_documents(directory, file_patterns)
                    discovered_documents = [doc.file_path for doc in discovery_result.get("documents", [])]
                else:
                    # Use provided documents
                    discovery_result = {
                        "total_documents": len(documents),
                        "documents": [],  # Would normally contain DocumentInfo objects
                        "directory": None
                    }
                    discovered_documents = documents
            else:
                # Use provided documents
                discovered_documents = documents
            
            # Use discovered documents for subsequent analyses
            if not discovered_documents:
                logger.warning("No documents found for corpus exploration")
                
                # Return result with discovery information only
                return TemplateResult(
                    success=True,
                    template_name="corpus_exploration",
                    template_version="1.0.0",
                    execution_time=time.time() - start_time,
                    results={
                        "discovery": discovery_result
                    },
                    summary={
                        "total_documents": 0,
                        "analyses_performed": ["discovery"],
                        "execution_time": time.time() - start_time
                    }
                )
            
            # Results dictionary
            results = {
                "discovery": discovery_result
            }
            
            # Perform document profiling
            profile_result = {}
            if "profiling" in analysis_types:
                profile_result = await self._profile_documents(discovered_documents)
                results["profiling"] = profile_result
            
            # Extract entities
            if "entities" in analysis_types:
                entity_result = await self._extract_entities(discovered_documents)
                results["entities"] = entity_result
            
            # Identify themes
            if "themes" in analysis_types:
                theme_result = await self._identify_themes(discovered_documents)
                results["themes"] = theme_result
            
            # Cluster documents
            cluster_result = {}
            if "clustering" in analysis_types:
                if "profiling" not in analysis_types:
                    # Need profiles for clustering
                    profile_result = await self._profile_documents(discovered_documents)
                    results["profiling"] = profile_result
                
                cluster_result = await self._cluster_documents(discovered_documents, profile_result)
                results["clustering"] = cluster_result
            
            # Analyze content types
            if "content_types" in analysis_types:
                content_type_result = await self._analyze_content_types(discovered_documents)
                results["content_types"] = content_type_result
            
            # Generate corpus map
            if "mapping" in analysis_types:
                if "profiling" not in analysis_types:
                    # Need profiles for mapping
                    profile_result = await self._profile_documents(discovered_documents)
                    results["profiling"] = profile_result
                    
                if "clustering" not in analysis_types:
                    # Need clusters for mapping
                    cluster_result = await self._cluster_documents(discovered_documents, profile_result)
                    results["clustering"] = cluster_result
                
                map_result = await self._generate_corpus_map(discovered_documents, profile_result, cluster_result)
                results["mapping"] = map_result
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Create template result
            return TemplateResult(
                success=True,
                template_name="corpus_exploration",
                template_version="1.0.0",
                execution_time=execution_time,
                results=results,
                summary={
                    "total_documents": len(discovered_documents),
                    "analyses_performed": analysis_types,
                    "execution_time": execution_time
                }
            )
            
        except Exception as e:
            # Log the error
            logger.error(f"Error in corpus exploration: {str(e)}")
            
            # Create error result
            return TemplateResult(
                success=False,
                template_name="corpus_exploration",
                template_version="1.0.0",
                execution_time=time.time() - start_time,
                error=str(e),
                results={},
                summary={
                    "error": str(e)
                }
            )


async def run_corpus_exploration(params: Dict[str, Any]) -> TemplateResult:
    """
    Run the corpus exploration template.
    
    Args:
        params: Parameters for the exploration
    
    Returns:
        Result of the exploration
    """
    template = CorpusExplorationTemplate()
    return await template.execute(params)