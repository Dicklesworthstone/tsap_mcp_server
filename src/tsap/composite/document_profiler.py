"""
Document Profiler for the TSAP MCP Server.

This module implements a tool for creating structural and content-based 
fingerprints of documents for comparison, classification, or search.
"""

import os
import hashlib
import re
from typing import Dict, List, Any
from collections import Counter
import asyncio

from tsap.utils.logging import logger
from tsap.core.pdf_extractor import extract_pdf_text
from tsap.core.html_processor import extract_html_text
from tsap.mcp.models import DocumentProfilerParams, DocumentProfilerResult


class DocumentProfile:
    """
    Contains structural and content-based profile information for a document.
    """
    
    def __init__(self, document_path: str):
        """
        Initialize a document profile.
        
        Args:
            document_path: Path to the document
        """
        self.document_path = document_path
        self.file_name = os.path.basename(document_path)
        self.file_extension = os.path.splitext(document_path)[1].lower()
        
        # Basic properties
        self.file_size = 0
        self.creation_time = 0
        self.modification_time = 0
        self.hash = ""
        
        # Content metrics
        self.line_count = 0
        self.word_count = 0
        self.char_count = 0
        
        # Structure metrics
        self.section_count = 0
        self.paragraph_count = 0
        self.heading_count = 0
        self.list_count = 0
        self.table_count = 0
        
        # Language features
        self.language = ""
        self.language_confidence = 0.0
        
        # Content features
        self.top_terms = []
        self.ngrams = []
        self.term_frequencies = {}
        self.readability_score = 0.0
        
        # Structure features
        self.section_structure = []
        self.heading_hierarchy = []
        self.named_entities = []
        
        # Special features
        self.content_type_features = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert profile to dictionary.
        
        Returns:
            Dictionary representation of the profile
        """
        return {
            "document_path": self.document_path,
            "file_name": self.file_name,
            "file_extension": self.file_extension,
            "basic_properties": {
                "file_size": self.file_size,
                "creation_time": self.creation_time,
                "modification_time": self.modification_time,
                "hash": self.hash
            },
            "content_metrics": {
                "line_count": self.line_count,
                "word_count": self.word_count,
                "char_count": self.char_count
            },
            "structure_metrics": {
                "section_count": self.section_count,
                "paragraph_count": self.paragraph_count,
                "heading_count": self.heading_count,
                "list_count": self.list_count,
                "table_count": self.table_count
            },
            "language_features": {
                "language": self.language,
                "language_confidence": self.language_confidence
            },
            "content_features": {
                "top_terms": self.top_terms,
                "ngrams": self.ngrams,
                "term_frequencies": self.term_frequencies,
                "readability_score": self.readability_score
            },
            "structure_features": {
                "section_structure": self.section_structure,
                "heading_hierarchy": self.heading_hierarchy,
                "named_entities": self.named_entities
            },
            "content_type_features": self.content_type_features
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentProfile':
        """
        Create profile from dictionary.
        
        Args:
            data: Dictionary representation of profile
            
        Returns:
            DocumentProfile instance
        """
        profile = cls(data["document_path"])
        
        # Basic properties
        profile.file_name = data["file_name"]
        profile.file_extension = data["file_extension"]
        profile.file_size = data["basic_properties"]["file_size"]
        profile.creation_time = data["basic_properties"]["creation_time"]
        profile.modification_time = data["basic_properties"]["modification_time"]
        profile.hash = data["basic_properties"]["hash"]
        
        # Content metrics
        profile.line_count = data["content_metrics"]["line_count"]
        profile.word_count = data["content_metrics"]["word_count"]
        profile.char_count = data["content_metrics"]["char_count"]
        
        # Structure metrics
        profile.section_count = data["structure_metrics"]["section_count"]
        profile.paragraph_count = data["structure_metrics"]["paragraph_count"]
        profile.heading_count = data["structure_metrics"]["heading_count"]
        profile.list_count = data["structure_metrics"]["list_count"]
        profile.table_count = data["structure_metrics"]["table_count"]
        
        # Language features
        profile.language = data["language_features"]["language"]
        profile.language_confidence = data["language_features"]["language_confidence"]
        
        # Content features
        profile.top_terms = data["content_features"]["top_terms"]
        profile.ngrams = data["content_features"]["ngrams"]
        profile.term_frequencies = data["content_features"]["term_frequencies"]
        profile.readability_score = data["content_features"]["readability_score"]
        
        # Structure features
        profile.section_structure = data["structure_features"]["section_structure"]
        profile.heading_hierarchy = data["structure_features"]["heading_hierarchy"]
        profile.named_entities = data["structure_features"]["named_entities"]
        
        # Special features
        profile.content_type_features = data["content_type_features"]
        
        return profile
    
    def similarity_to(self, other: 'DocumentProfile') -> float:
        """
        Calculate similarity to another document profile.
        
        Args:
            other: Another document profile
            
        Returns:
            Similarity score (0-1)
        """
        # Calculate structural similarity
        structural_sim = self._calculate_structural_similarity(other)
        
        # Calculate content similarity
        content_sim = self._calculate_content_similarity(other)
        
        # Combine similarities (weighted average)
        return 0.4 * structural_sim + 0.6 * content_sim
    
    def _calculate_structural_similarity(self, other: 'DocumentProfile') -> float:
        """
        Calculate structural similarity to another document profile.
        
        Args:
            other: Another document profile
            
        Returns:
            Structural similarity score (0-1)
        """
        # Compare section counts and heading hierarchy
        section_similarity = 0.0
        if self.section_count > 0 and other.section_count > 0:
            section_similarity = 1.0 - abs(self.section_count - other.section_count) / max(self.section_count, other.section_count)
        
        # Compare heading hierarchy
        heading_similarity = 0.0
        if self.heading_hierarchy and other.heading_hierarchy:
            # Calculate Jaccard similarity of heading hierarchies
            a_set = set(tuple(h) for h in self.heading_hierarchy)
            b_set = set(tuple(h) for h in other.heading_hierarchy)
            
            if a_set or b_set:
                intersection = len(a_set.intersection(b_set))
                union = len(a_set.union(b_set))
                heading_similarity = intersection / union
        
        # Combine similarities
        return 0.5 * section_similarity + 0.5 * heading_similarity
    
    def _calculate_content_similarity(self, other: 'DocumentProfile') -> float:
        """
        Calculate content similarity to another document profile.
        
        Args:
            other: Another document profile
            
        Returns:
            Content similarity score (0-1)
        """
        # Compare term frequencies
        term_similarity = 0.0
        if self.term_frequencies and other.term_frequencies:
            # Get common terms
            common_terms = set(self.term_frequencies.keys()) & set(other.term_frequencies.keys())
            all_terms = set(self.term_frequencies.keys()) | set(other.term_frequencies.keys())
            
            if all_terms:
                # Calculate cosine similarity for common terms
                dot_product = sum(self.term_frequencies.get(term, 0) * other.term_frequencies.get(term, 0) for term in common_terms)
                magnitude_a = sum(freq**2 for freq in self.term_frequencies.values()) ** 0.5
                magnitude_b = sum(freq**2 for freq in other.term_frequencies.values()) ** 0.5
                
                if magnitude_a > 0 and magnitude_b > 0:
                    term_similarity = dot_product / (magnitude_a * magnitude_b)
        
        # Compare ngrams
        ngram_similarity = 0.0
        if self.ngrams and other.ngrams:
            a_set = set(tuple(ng) for ng in self.ngrams)
            b_set = set(tuple(ng) for ng in other.ngrams)
            
            if a_set or b_set:
                intersection = len(a_set.intersection(b_set))
                union = len(a_set.union(b_set))
                ngram_similarity = intersection / union
        
        # Combine similarities
        return 0.7 * term_similarity + 0.3 * ngram_similarity


async def _extract_document_text(document_path: str) -> str:
    """
    Extract text from a document based on its file type.
    
    Args:
        document_path: Path to the document
        
    Returns:
        Extracted text
    """
    # Get file extension
    ext = os.path.splitext(document_path)[1].lower()
    
    try:
        # Extract text based on file type
        if ext == '.pdf':
            logger.debug(f"Extracting text from PDF: {document_path}", component="composite", operation="document_profiler")
            return await extract_pdf_text(document_path, None, None) or ""
        elif ext in ['.html', '.htm']:
            logger.debug(f"Extracting text from HTML: {document_path}", component="composite", operation="document_profiler")
            return await extract_html_text(None, None, document_path) or ""
        else:
            # Attempt to read as plain text
            logger.debug(f"Reading as plain text: {document_path}", component="composite", operation="document_profiler")
            try:
                with open(document_path, 'r', encoding='utf-8', errors='replace') as f:
                    text = f.read()
                logger.debug(f"Successfully read {len(text)} characters from {document_path}", component="composite", operation="document_profiler")
                return text
            except UnicodeDecodeError:
                # Try binary mode as a fallback
                logger.warning(f"Unicode decode error, trying binary mode: {document_path}", component="composite", operation="document_profiler")
                with open(document_path, 'rb') as f:
                    binary_data = f.read()
                return binary_data.decode('utf-8', errors='replace')
    except Exception as e:
        logger.error(
            f"Error extracting text from {document_path}: {str(e)}",
            component="composite", 
            operation="document_profiler",
            exc_info=True  # Include full traceback
        )
        # Re-raise the exception to ensure it's handled properly
        raise


async def _compute_basic_properties(document_path: str) -> Dict[str, Any]:
    """
    Compute basic file properties.
    
    Args:
        document_path: Path to the document
        
    Returns:
        Dictionary of basic properties
    """
    try:
        # Get file stats
        stats = os.stat(document_path)
        
        # Compute file hash
        hash_md5 = hashlib.md5()
        with open(document_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        
        return {
            "file_size": stats.st_size,
            "creation_time": stats.st_ctime,
            "modification_time": stats.st_mtime,
            "hash": hash_md5.hexdigest()
        }
    except Exception as e:
        logger.error(
            f"Error computing basic properties for {document_path}: {str(e)}",
            component="composite",
            operation="document_profiler"
        )
        return {
            "file_size": 0,
            "creation_time": 0,
            "modification_time": 0,
            "hash": ""
        }


async def _compute_content_metrics(text: str) -> Dict[str, Any]:
    """
    Compute content metrics from text.
    
    Args:
        text: Document text
        
    Returns:
        Dictionary of content metrics
    """
    # Split into lines and words
    lines = text.split('\n')
    words = re.findall(r'\b\w+\b', text.lower())
    
    return {
        "line_count": len(lines),
        "word_count": len(words),
        "char_count": len(text)
    }


async def _detect_language(text: str) -> Dict[str, Any]:
    """
    Detect document language.
    
    Args:
        text: Document text
        
    Returns:
        Dictionary with language and confidence
    """
    # Simple language detection based on common words
    # A real implementation would use a proper language detection library
    
    # Sample words for a few languages
    language_words = {
        "english": ["the", "and", "to", "of", "in", "is", "it", "that", "for", "you"],
        "spanish": ["el", "la", "de", "en", "y", "es", "un", "que", "por", "los"],
        "french": ["le", "la", "de", "et", "est", "en", "un", "une", "pour", "dans"],
        "german": ["der", "die", "das", "und", "ist", "in", "zu", "den", "mit", "auf"]
    }
    
    # Count occurrences of language-specific words
    words = re.findall(r'\b\w+\b', text.lower())
    word_count = Counter(words)
    
    # Calculate scores for each language
    language_scores = {}
    for language, markers in language_words.items():
        score = sum(word_count.get(word, 0) for word in markers)
        language_scores[language] = score
    
    # Find language with highest score
    if language_scores:
        best_language = max(language_scores.items(), key=lambda x: x[1])
        total_score = sum(language_scores.values())
        
        if total_score > 0:
            confidence = best_language[1] / total_score
            return {
                "language": best_language[0],
                "language_confidence": confidence
            }
    
    # Default if no language detected
    return {
        "language": "unknown",
        "language_confidence": 0.0
    }


async def _compute_content_features(text: str, max_terms: int = 20) -> Dict[str, Any]:
    """
    Compute content features from text.
    
    Args:
        text: Document text
        max_terms: Maximum number of top terms to include
        
    Returns:
        Dictionary of content features
    """
    # Calculate word frequencies
    words = re.findall(r'\b\w+\b', text.lower())
    word_count = Counter(words)
    
    # Remove common stop words
    stop_words = {"the", "and", "to", "of", "in", "is", "it", "that", "for", "you", "a", "an"}
    for word in stop_words:
        if word in word_count:
            del word_count[word]
    
    # Get top terms
    top_terms = word_count.most_common(max_terms)
    
    # Calculate ngrams (bigrams and trigrams)
    bigrams = []
    trigrams = []
    
    for i in range(len(words) - 1):
        bigrams.append((words[i], words[i+1]))
    
    for i in range(len(words) - 2):
        trigrams.append((words[i], words[i+1], words[i+2]))
    
    # Calculate term frequencies (normalized)
    total_words = len(words)
    term_frequencies = {word: count / total_words for word, count in word_count.items()}
    
    # Simple readability score (average word length)
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    readability_score = max(0, min(1, (avg_word_length - 3) / 7))  # Normalize to 0-1
    
    return {
        "top_terms": top_terms,
        "ngrams": [bigrams[:50], trigrams[:30]],
        "term_frequencies": term_frequencies,
        "readability_score": readability_score
    }


async def _analyze_structure(text: str) -> Dict[str, Any]:
    """
    Analyze document structure.
    
    Args:
        text: Document text
        
    Returns:
        Dictionary of structure metrics and features
    """
    # Split into paragraphs
    paragraphs = [p for p in re.split(r'\n\s*\n', text) if p.strip()]
    
    # Find potential headings
    heading_patterns = [
        r'^#+\s+(.+?)$',                   # Markdown headings
        r'^([A-Z][A-Za-z0-9 ]+)$',         # All caps line
        r'^(\d+\.(?:\d+)*)\s+(.+?)$',      # Numbered headings
        r'^([A-Z][A-Za-z0-9 ]+:)',         # Heading with colon
    ]
    
    headings = []
    for pattern in heading_patterns:
        headings.extend(re.findall(pattern, text, re.MULTILINE))
    
    # Find lists
    list_patterns = [
        r'^\s*[-*•]\s+(.+?)$',             # Bullet lists
        r'^\s*\d+\.\s+(.+?)$',             # Numbered lists
    ]
    
    lists = []
    for pattern in list_patterns:
        lists.extend(re.findall(pattern, text, re.MULTILINE))
    
    # Find tables (simple heuristic)
    table_patterns = [
        r'\|\s*([^|]+\s*\|\s*)+',          # Markdown tables
        r'[-+]{3,}\n',                      # ASCII tables
    ]
    
    tables = []
    for pattern in table_patterns:
        tables.extend(re.findall(pattern, text, re.MULTILINE))
    
    # Analyze heading hierarchy
    heading_hierarchy = []  # noqa: F841
    current_section = []
    
    # Simple section detection (based on empty lines)
    sections = []
    current_section = []
    current_section_text = ""
    
    for line in text.split('\n'):
        if not line.strip() and current_section:
            if current_section_text:
                sections.append(current_section_text)
                current_section = []
                current_section_text = ""
        else:
            current_section.append(line)
            current_section_text += line + "\n"
    
    # Add the last section
    if current_section_text:
        sections.append(current_section_text)
    
    # Find named entities (simple pattern-based approach)
    # This would use NER in a real implementation
    entity_patterns = {
        "person": r'([A-Z][a-z]+ [A-Z][a-z]+)',
        "organization": r'([A-Z][A-Za-z]+ (?:Inc|LLC|Corp|Corporation|Company))',
        "location": r'([A-Z][a-z]+ (?:Street|Road|Avenue|Blvd|Boulevard|Place|Square))'
    }
    
    named_entities = []
    for entity_type, pattern in entity_patterns.items():
        matches = re.findall(pattern, text)
        for match in matches:
            named_entities.append({"text": match, "type": entity_type})
    
    # Return structure metrics and features
    return {
        "metrics": {
            "section_count": len(sections),
            "paragraph_count": len(paragraphs),
            "heading_count": len(headings),
            "list_count": len(lists),
            "table_count": len(tables)
        },
        "features": {
            "section_structure": [s[:100] + "..." if len(s) > 100 else s for s in sections],
            "heading_hierarchy": headings,
            "named_entities": named_entities[:50]
        }
    }


async def _analyze_content_type(document_path: str, text: str) -> Dict[str, Any]:
    """
    Analyze content type-specific features.
    
    Args:
        document_path: Path to the document
        text: Document text
        
    Returns:
        Dictionary of content type-specific features
    """
    # Get file extension
    ext = os.path.splitext(document_path)[1].lower()
    
    features = {}
    
    if ext == '.py':
        # Python code analysis
        features = await _analyze_python_code(text)
    elif ext in ['.js', '.ts']:
        # JavaScript/TypeScript analysis
        features = await _analyze_js_code(text)
    elif ext in ['.java', '.kt']:
        # Java/Kotlin analysis
        features = await _analyze_java_code(text)
    elif ext in ['.c', '.cpp', '.h', '.hpp']:
        # C/C++ analysis
        features = await _analyze_cpp_code(text)
    elif ext == '.md':
        # Markdown analysis
        features = await _analyze_markdown(text)
    elif ext in ['.json', '.yml', '.yaml']:
        # Data file analysis
        features = await _analyze_data_file(text, ext)
    elif ext == '.pdf':
        # PDF-specific analysis (though most is done in text extraction)
        features = {"format": "pdf"}
    elif ext in ['.html', '.htm']:
        # HTML-specific analysis
        features = await _analyze_html(text)
    
    return features


async def _analyze_python_code(text: str) -> Dict[str, Any]:
    """
    Analyze Python code features.
    
    Args:
        text: Python code text
        
    Returns:
        Dictionary of Python code features
    """
    # Find classes, functions, imports
    classes = re.findall(r'class\s+(\w+)', text)
    functions = re.findall(r'def\s+(\w+)', text)
    imports = re.findall(r'import\s+(\w+)|from\s+(\w+)', text)
    
    # Flatten imports list
    imports = [imp[0] or imp[1] for imp in imports]
    
    # Count comments
    comments = re.findall(r'^\s*#.*$', text, re.MULTILINE)
    docstrings = re.findall(r'""".*?"""', text, re.DOTALL)
    
    return {
        "code_type": "python",
        "classes": classes,
        "functions": functions,
        "imports": imports,
        "comment_count": len(comments),
        "docstring_count": len(docstrings),
        "loc": len(text.split('\n')),
        "complexity_indicators": {
            "nested_loops": len(re.findall(r'for.*\n.*for', text)),
            "conditionals": len(re.findall(r'if\s+|elif\s+|else:', text))
        }
    }


async def _analyze_js_code(text: str) -> Dict[str, Any]:
    """
    Analyze JavaScript/TypeScript code features.
    
    Args:
        text: JS/TS code text
        
    Returns:
        Dictionary of JS/TS code features
    """
    # Simplified analysis - a real implementation would use a proper parser
    functions = re.findall(r'function\s+(\w+)|(\w+)\s*=\s*function', text)
    classes = re.findall(r'class\s+(\w+)', text)
    imports = re.findall(r'import\s+.*?from\s+[\'"](.+?)[\'"]', text)
    
    # Flatten functions list
    functions = [func[0] or func[1] for func in functions]
    
    # Count comments
    comments = re.findall(r'^\s*//.*$', text, re.MULTILINE)
    block_comments = re.findall(r'/\*.*?\*/', text, re.DOTALL)
    
    return {
        "code_type": "javascript",
        "functions": functions,
        "classes": classes,
        "imports": imports,
        "comment_count": len(comments) + len(block_comments),
        "loc": len(text.split('\n')),
        "complexity_indicators": {
            "nested_loops": len(re.findall(r'for.*\n.*for', text)),
            "conditionals": len(re.findall(r'if\s*\(|else\s*|switch\s*\(', text))
        }
    }


async def _analyze_java_code(text: str) -> Dict[str, Any]:
    """Placeholder for Java/Kotlin code analysis."""
    return {"code_type": "java"}


async def _analyze_cpp_code(text: str) -> Dict[str, Any]:
    """Placeholder for C/C++ code analysis."""
    return {"code_type": "cpp"}


async def _analyze_markdown(text: str) -> Dict[str, Any]:
    """Placeholder for Markdown analysis."""
    return {"format": "markdown"}


async def _analyze_data_file(text: str, ext: str) -> Dict[str, Any]:
    """Placeholder for data file analysis."""
    return {"format": ext[1:]}  # Remove leading dot


async def _analyze_html(text: str) -> Dict[str, Any]:
    """Placeholder for HTML analysis."""
    return {"format": "html"}


async def create_document_profile(document_path: str, include_content: bool = True) -> DocumentProfile:
    """
    Create a comprehensive profile for a single document.

    Args:
        document_path: Path to the document file.
        include_content: Whether to include detailed content features (can be slower).

    Returns:
        A DocumentProfile object.
        
    Raises:
        FileNotFoundError: If the document path does not exist.
        IOError: If there's an issue reading the document.
        Exception: For other underlying analysis errors.
    """
    logger.info(f"Starting profile creation for: {document_path}", component="composite", operation="create_profile")
    
    if not os.path.exists(document_path):
        logger.error(f"Document not found: {document_path}", component="composite", operation="create_profile")
        raise FileNotFoundError(f"Document not found: {document_path}")

    profile = DocumentProfile(document_path)
    
    try:
        # 1. Extract Text
        logger.debug(f"Extracting text from: {document_path}", component="composite", operation="extract_text")
        text = await _extract_document_text(document_path)
        logger.debug(f"Text extracted successfully ({len(text)} chars).", component="composite", operation="extract_text")
        
        # 2. Basic Properties
        logger.debug("Computing basic properties...", component="composite", operation="compute_basic_props")
        basic_props = await _compute_basic_properties(document_path)
        profile.file_size = basic_props["file_size"]
        profile.creation_time = basic_props["creation_time"]
        profile.modification_time = basic_props["modification_time"]
        profile.hash = basic_props["hash"]
        logger.debug("Basic properties computed.", component="composite", operation="compute_basic_props")

        # 3. Content Metrics
        logger.debug("Computing content metrics...", component="composite", operation="compute_content_metrics")
        content_metrics = await _compute_content_metrics(text)
        profile.line_count = content_metrics["line_count"]
        profile.word_count = content_metrics["word_count"]
        profile.char_count = content_metrics["char_count"]
        logger.debug("Content metrics computed.", component="composite", operation="compute_content_metrics")

        # 4. Language Detection
        logger.debug("Detecting language...", component="composite", operation="detect_language")
        lang_features = await _detect_language(text)
        profile.language = lang_features["language"]
        profile.language_confidence = lang_features["language_confidence"]
        logger.debug(f"Language detected: {profile.language} (Confidence: {profile.language_confidence:.2f})", component="composite", operation="detect_language")

        # 5. Content Features (Optional)
        if include_content:
            logger.debug("Computing content features...", component="composite", operation="compute_content_features")
            content_features = await _compute_content_features(text)
            profile.top_terms = content_features["top_terms"]
            profile.ngrams = content_features["ngrams"]
            profile.term_frequencies = content_features["term_frequencies"]
            profile.readability_score = content_features["readability_score"]
            logger.debug("Content features computed.", component="composite", operation="compute_content_features")

        # 6. Structure Analysis
        logger.debug("Analyzing structure...", component="composite", operation="analyze_structure")
        structure_features = await _analyze_structure(text)
        profile.section_count = structure_features["metrics"]["section_count"]
        profile.paragraph_count = structure_features["metrics"]["paragraph_count"]
        profile.heading_count = structure_features["metrics"]["heading_count"]
        profile.list_count = structure_features["metrics"]["list_count"]
        profile.table_count = structure_features["metrics"]["table_count"]
        profile.section_structure = structure_features["features"]["section_structure"]
        profile.heading_hierarchy = structure_features["features"]["heading_hierarchy"]
        profile.named_entities = structure_features["features"]["named_entities"] # Assuming NER is part of structure
        logger.debug("Structure analysis completed.", component="composite", operation="analyze_structure")

        # 7. Content-Type Specific Analysis
        logger.debug("Analyzing content type specific features...", component="composite", operation="analyze_content_type")
        profile.content_type_features = await _analyze_content_type(document_path, text)
        logger.debug("Content type analysis completed.", component="composite", operation="analyze_content_type")

        logger.info(f"Finished profile creation for: {document_path}", component="composite", operation="create_profile")
        return profile

    except Exception as e:
        logger.error(
            f"Error creating profile for {document_path}: {str(e)}", 
            component="composite", 
            operation="create_profile",
            exc_info=True # Include traceback in log
        )
        # Re-raise the exception so the caller knows something went wrong
        raise e


async def profile_document(params: DocumentProfilerParams) -> DocumentProfilerResult:
    """
    Profile a document and return the results.
    
    Args:
        params: Document profiler parameters
        
    Returns:
        Document profiler result
    """
    # Create profile
    profile = await create_document_profile(
        params.document_path,
        params.include_content_features
    )
    
    # Compare to reference profiles if provided
    comparisons = {}
    if params.reference_profiles:
        for ref_path, ref_data in params.reference_profiles.items():
            # Create reference profile
            ref_profile = DocumentProfile.from_dict(ref_data)
            
            # Calculate similarity
            similarity = profile.similarity_to(ref_profile)
            
            # Add to comparisons
            comparisons[ref_path] = {
                "similarity": similarity,
                "similar_features": []  # Would identify specific similar features in a real implementation
            }
    
    # Create result
    result = DocumentProfilerResult(
        document_path=params.document_path,
        profile=profile.to_dict(),
        comparisons=comparisons,
        error=None
    )
    
    return result


async def profile_documents(
    document_paths: List[str],
    include_content_features: bool = True,
    compare_documents: bool = True
) -> Dict[str, Any]:
    """
    Create profiles for multiple documents and optionally compare them.

    Args:
        document_paths: List of paths to the documents.
        include_content_features: Whether to compute detailed content features.
        compare_documents: Whether to calculate similarity between all pairs of documents.

    Returns:
        A dictionary containing:
            - profiles: A dictionary mapping document paths to their profile data.
            - comparisons: A dictionary mapping document pairs (path1 <-> path2) 
                           to their similarity score, if compare_documents is True.
    """
    logger.info(f"Starting batch document profiling for {len(document_paths)} documents.", component="composite", operation="batch_profile")
    
    profiles: Dict[str, DocumentProfile] = {}
    errors: Dict[str, str] = {}

    # Create profiles concurrently
    tasks = []
    for path in document_paths:
        tasks.append(create_document_profile(path, include_content_features))
        
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for path, result in zip(document_paths, results):
        if isinstance(result, DocumentProfile):
            profiles[path] = result
        elif isinstance(result, Exception):
            errors[path] = str(result)
            logger.warning(f"Failed to profile {path}: {str(result)}", component="composite", operation="batch_profile")

    # Optionally compare profiles
    comparisons: Dict[str, Dict[str, float]] = {}
    if compare_documents and len(profiles) > 1:
        logger.info("Starting document comparison phase.", component="composite", operation="batch_profile_compare")
        profile_items = list(profiles.items())
        for i in range(len(profile_items)):
            for j in range(i + 1, len(profile_items)):
                path1, profile1 = profile_items[i]
                path2, profile2 = profile_items[j]
                try:
                    similarity = profile1.similarity_to(profile2)
                    pair_key = f"{path1} <-> {path2}"
                    comparisons[pair_key] = {"similarity": similarity}
                    logger.debug(f"Compared {os.path.basename(path1)} <-> {os.path.basename(path2)}: Similarity {similarity:.3f}", component="composite", operation="batch_profile_compare")
                except Exception as e:
                    logger.error(
                        f"Error comparing {path1} and {path2}: {str(e)}", 
                        exception=e,
                        component="composite", 
                        operation="batch_profile_compare"
                    )
                    errors[f"{path1} <-> {path2}"] = str(e)
        logger.info(f"Completed document comparison. Found {len(comparisons)} comparison pairs.", component="composite", operation="batch_profile_compare")

    # Prepare results
    profile_dicts = {path: profile.to_dict() for path, profile in profiles.items()}
    
    final_result = {
        "profiles": profile_dicts,
        "comparisons": comparisons,
        "errors": errors # Include errors in the result
    }

    logger.info(f"Finished batch document profiling. Generated {len(profiles)} profiles.", component="composite", operation="batch_profile")
    return final_result