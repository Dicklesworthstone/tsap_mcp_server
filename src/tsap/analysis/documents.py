"""
Document exploration and analysis tools for TSAP.

This module provides functionality for exploring, analyzing, and extracting
information from document collections across various file formats.
"""
import os
import re
import asyncio
from typing import Dict, List, Any, Optional
import datetime
import mimetypes
from dataclasses import dataclass

from tsap.utils.logging import logger
from tsap.core.ripgrep import ripgrep_search
from tsap.core.pdf_extractor import extract_pdf_text
from tsap.mcp.models import (
    DocumentExplorerParams, DocumentExplorerResult, DocumentInfo,
    RipgrepSearchParams
)
from tsap.analysis.base import BaseAnalysisTool, register_analysis_tool, AnalysisContext


# Document format configurations
DOCUMENT_EXTENSIONS = {
    "pdf": [".pdf"],
    "word": [".doc", ".docx", ".odt"],
    "excel": [".xls", ".xlsx", ".ods"],
    "powerpoint": [".ppt", ".pptx", ".odp"],
    "text": [".txt", ".md", ".markdown", ".rst"],
    "html": [".html", ".htm", ".xhtml"],
    "xml": [".xml"],
    "json": [".json"],
    "yaml": [".yaml", ".yml"],
    "csv": [".csv"],
    "rtf": [".rtf"],
    "images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"],
}

# Common document categories
DOCUMENT_CATEGORIES = [
    "report", "article", "paper", "memo", "letter", "email", "invoice",
    "contract", "agreement", "policy", "manual", "guide", "specification",
    "presentation", "spreadsheet", "form", "template", "brochure", "newsletter",
]

# Category detection patterns
CATEGORY_PATTERNS = {
    "report": r"(report|analysis|study|findings|assessment|evaluation|survey|review)",
    "article": r"(article|publication|journal|paper|proceedings|conference)",
    "paper": r"(paper|manuscript|thesis|dissertation|research|study)",
    "memo": r"(memo|memorandum|note|notification|advisory)",
    "letter": r"(letter|correspondence|communication)",
    "email": r"(email|e-mail|message|inquiry|request)",
    "invoice": r"(invoice|bill|receipt|payment|statement|account)",
    "contract": r"(contract|agreement|covenant|arrangement|deal|commitment)",
    "policy": r"(policy|procedure|guideline|regulation|compliance|rule)",
    "manual": r"(manual|guide|handbook|instruction|reference|tutorial)",
    "specification": r"(specification|requirement|standard|protocol|schema)",
    "presentation": r"(presentation|slides|deck|briefing|overview)",
}


@dataclass
class DocumentMetadata:
    """Document metadata container."""
    title: Optional[str] = None
    author: Optional[str] = None
    date: Optional[datetime.datetime] = None
    subject: Optional[str] = None
    keywords: List[str] = None
    language: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    custom: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.custom is None:
            self.custom = {}


@register_analysis_tool("document_explorer")
class DocumentExplorer(BaseAnalysisTool):
    """Document exploration and analysis tool."""
    
    def __init__(self, name: str = "document_explorer"):
        """Initialize the document explorer.
        
        Args:
            name: Explorer name
        """
        super().__init__(name)
    
    def _detect_document_type(self, file_path: str) -> Optional[str]:
        """Detect the document type based on file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected document type or None if not recognized
        """
        extension = os.path.splitext(file_path.lower())[1]
        
        for doc_type, extensions in DOCUMENT_EXTENSIONS.items():
            if extension in extensions:
                return doc_type
                
        return None
    
    def _get_document_file_patterns(self, doc_type: Optional[str] = None) -> List[str]:
        """Get file patterns for a document type or all types.
        
        Args:
            doc_type: Specific document type or None for all
            
        Returns:
            List of file extension patterns (e.g., "*.pdf")
        """
        patterns = []
        
        if doc_type:
            # Get patterns for specific document type
            extensions = DOCUMENT_EXTENSIONS.get(doc_type, [])
            patterns = [f"*{ext}" for ext in extensions]
        else:
            # Get patterns for all document types
            for extensions in DOCUMENT_EXTENSIONS.values():
                for ext in extensions:
                    pattern = f"*{ext}"
                    if pattern not in patterns:
                        patterns.append(pattern)
                        
        return patterns
    
    async def _find_documents(
        self, 
        directory_path: str, 
        file_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        recursive: bool = True,
        max_depth: Optional[int] = None,
        max_files: Optional[int] = None,
    ) -> List[str]:
        """Find documents in a directory.
        
        Args:
            directory_path: Directory to search
            file_patterns: Optional file patterns to include
            exclude_patterns: Optional file patterns to exclude
            recursive: Whether to search recursively
            max_depth: Maximum directory depth
            max_files: Maximum number of files to return
            
        Returns:
            List of document file paths
        """
        logger.info(
            f"Finding documents in: {directory_path}",
            component="analysis",
            operation="find_documents",
            context={
                "directory": directory_path,
                "recursive": recursive,
                "max_depth": max_depth,
            }
        )
        
        # If no patterns specified, use all document patterns
        if not file_patterns:
            file_patterns = self._get_document_file_patterns()
        
        # Create ripgrep search parameters to find all files
        params = RipgrepSearchParams(
            pattern="",  # Empty pattern to match all files
            paths=[directory_path],
            file_patterns=file_patterns,
            exclude_patterns=exclude_patterns,
            max_depth=max_depth if recursive else 0,
            max_total_matches=max_files,
            invert_match=True,  # Invert to match file names, not content
        )
        
        # Execute the search
        result = await ripgrep_search(params)
        
        # Extract file paths from matches
        files = list(set(match.path for match in result.matches))
        
        logger.info(
            f"Found {len(files)} documents",
            component="analysis",
            operation="find_documents",
            context={
                "file_count": len(files),
                "file_patterns": file_patterns,
            }
        )
        
        return files
    
    async def _extract_metadata(self, file_path: str) -> DocumentMetadata:
        """Extract metadata from a document.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Extracted metadata
        """
        metadata = DocumentMetadata()
        
        # Get document type
        doc_type = self._detect_document_type(file_path)
        
        # Extract metadata based on document type
        if doc_type == "pdf":
            # For PDF, use pdf_extractor module
            metadata_dict = await extract_pdf_metadata(file_path)
            
            # Map common metadata fields
            metadata.title = metadata_dict.get("Title")
            metadata.author = metadata_dict.get("Author")
            
            # Parse date if available
            date_str = metadata_dict.get("CreationDate")
            if date_str:
                try:
                    # Handle different date formats
                    # This is simplified and would need more robust parsing in production
                    if re.match(r"D:\d{14}", date_str):
                        # PDF format D:YYYYMMDDHHmmSS
                        date_str = date_str[2:]  # Remove D:
                        metadata.date = datetime.datetime.strptime(date_str[:14], "%Y%m%d%H%M%S")
                    else:
                        # Try common formats
                        for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"]:
                            try:
                                metadata.date = datetime.datetime.strptime(date_str, fmt)
                                break
                            except ValueError:
                                continue
                except Exception:
                    logger.debug(
                        f"Failed to parse PDF date: {date_str}",
                        component="analysis",
                        operation="extract_metadata"
                    )
            
            metadata.subject = metadata_dict.get("Subject")
            keywords_str = metadata_dict.get("Keywords")
            if keywords_str:
                metadata.keywords = [k.strip() for k in keywords_str.split(",")]
            
            metadata.creator = metadata_dict.get("Creator")
            metadata.producer = metadata_dict.get("Producer")
            
            # Add remaining fields to custom
            for key, value in metadata_dict.items():
                if key not in ["Title", "Author", "CreationDate", "Subject", "Keywords", "Creator", "Producer"]:
                    metadata.custom[key] = value
                    
        elif doc_type in ["word", "excel", "powerpoint"]:
            # Office documents would need specialized extraction
            # This is a placeholder for actual implementation
            # Could use libraries like python-docx, openpyxl, etc.
            pass
            
        # For other document types, use basic file information
        if not metadata.title:
            metadata.title = os.path.basename(file_path)
            
        if not metadata.date:
            # Use file modification time
            mtime = os.path.getmtime(file_path)
            metadata.date = datetime.datetime.fromtimestamp(mtime)
        
        return metadata
    
    async def _categorize_document(
        self, 
        file_path: str,
        content: Optional[str] = None
    ) -> List[str]:
        """Categorize a document based on content and metadata.
        
        Args:
            file_path: Path to the document
            content: Optional document content (to avoid reloading)
            
        Returns:
            List of document categories
        """
        categories = []
        
        # Get document type as base category
        doc_type = self._detect_document_type(file_path)
        if doc_type:
            categories.append(doc_type)
        
        # If content is provided, scan for category patterns
        if content:
            for category, pattern in CATEGORY_PATTERNS.items():
                if re.search(pattern, content, re.IGNORECASE):
                    categories.append(category)
        
        return categories
    
    async def _generate_summary(
        self, 
        file_path: str,
        content: Optional[str] = None,
        metadata: Optional[DocumentMetadata] = None
    ) -> str:
        """Generate a summary for a document.
        
        Args:
            file_path: Path to the document
            content: Optional document content (to avoid reloading)
            metadata: Optional document metadata
            
        Returns:
            Document summary
        """
        # This is a simplified implementation
        # In a real system, this would use more sophisticated techniques
        # or leverage LLM capabilities
        
        doc_type = self._detect_document_type(file_path)
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        
        # Format file size for display
        if file_size < 1024:
            size_str = f"{file_size} bytes"
        elif file_size < 1024 * 1024:
            size_str = f"{file_size / 1024:.1f} KB"
        else:
            size_str = f"{file_size / (1024 * 1024):.1f} MB"
        
        # Use metadata if available
        title = file_name
        author = "Unknown"
        date = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
        
        if metadata:
            if metadata.title:
                title = metadata.title
            if metadata.author:
                author = metadata.author
            if metadata.date:
                date = metadata.date
        
        # Format date for display
        date_str = date.strftime("%Y-%m-%d")
        
        # Create a basic summary
        summary = f"{title} ({doc_type}). {size_str}, last modified on {date_str}"
        
        if author != "Unknown":
            summary += f", by {author}"
            
        return summary
    
    async def _extract_text(self, file_path: str) -> Optional[str]:
        """Extract text from a document.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Extracted text or None if extraction failed
        """
        # Get document type
        doc_type = self._detect_document_type(file_path)
        
        # Extract text based on document type
        if doc_type == "pdf":
            # For PDF, use pdf_extractor module
            text = await extract_pdf_text(file_path)
            return text
        elif doc_type == "text":
            # Plain text - read directly
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            except UnicodeDecodeError:
                # Try different encodings
                try:
                    with open(file_path, "r", encoding="latin-1") as f:
                        return f.read()
                except Exception as e:
                    logger.warning(
                        f"Failed to read text file: {str(e)}",
                        component="analysis",
                        operation="extract_text",
                        context={"file": file_path}
                    )
                    return None
        elif doc_type in ["word", "excel", "powerpoint", "html"]:
            # These would need specialized extraction
            # This is a placeholder for actual implementation
            pass
            
        return None
    
    async def _process_document(
        self, 
        file_path: str,
        extract_metadata: bool = True,
        extract_full_text: bool = False,
        extract_summaries: bool = True,
        categorize: bool = True
    ) -> DocumentInfo:
        """Process a single document to extract information.
        
        Args:
            file_path: Path to the document
            extract_metadata: Whether to extract metadata
            extract_full_text: Whether to extract full text
            extract_summaries: Whether to generate summaries
            categorize: Whether to categorize the document
            
        Returns:
            Document information
        """
        # Basic file information
        file_name = os.path.basename(file_path)
        file_extension = os.path.splitext(file_path)[1].lower()
        file_size = os.path.getsize(file_path)
        modified_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
        
        try:
            created_time = datetime.datetime.fromtimestamp(os.path.getctime(file_path))
        except Exception:
            created_time = None
            
        # Get MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        
        # Extract text if needed for categorization or summarization
        text = None
        if extract_full_text or (categorize and not extract_metadata) or extract_summaries:
            text = await self._extract_text(file_path)
        
        # Extract metadata
        metadata_dict = {}
        if extract_metadata:
            metadata = await self._extract_metadata(file_path)
            
            # Convert to dictionary
            metadata_dict = {
                "title": metadata.title,
                "author": metadata.author,
                "date": metadata.date.isoformat() if metadata.date else None,
                "subject": metadata.subject,
                "keywords": metadata.keywords,
                "language": metadata.language,
                "creator": metadata.creator,
                "producer": metadata.producer,
            }
            
            # Add custom fields
            for key, value in metadata.custom.items():
                metadata_dict[key] = value
        
        # Categorize document
        categories = []
        if categorize:
            categories = await self._categorize_document(file_path, text)
        
        # Generate summary
        summary = None
        if extract_summaries:
            summary = await self._generate_summary(file_path, text, 
                                                 metadata if extract_metadata else None)
        
        # Create document info
        doc_info = DocumentInfo(
            path=file_path,
            file_name=file_name,
            file_extension=file_extension,
            file_size=file_size,
            modified_time=modified_time,
            created_time=created_time,
            mime_type=mime_type,
            metadata=metadata_dict,
            summary=summary,
            categories=categories,
            extracted_text=text if extract_full_text else None,
        )
        
        return doc_info
    
    async def explore(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Explore documents with specified parameters.
        
        Args:
            params: Exploration parameters (DocumentExplorerParams)
            
        Returns:
            Exploration results
        """
        async with self._measure_execution_time():
            start_time = asyncio.get_event_loop().time()
            
            # Convert dict to DocumentExplorerParams if needed
            if not isinstance(params, DocumentExplorerParams):
                params = DocumentExplorerParams(**params)
                
            # Create analysis context
            context = AnalysisContext()  # noqa: F841
            
            # Find documents
            document_paths = await self._find_documents(
                directory_path=params.directory_path,
                file_patterns=params.file_patterns,
                exclude_patterns=params.exclude_patterns,
                recursive=params.recursive,
                max_depth=params.max_depth,
                max_files=params.max_files,
            )
            
            # Log exploration start
            logger.info(
                f"Starting document exploration: {len(document_paths)} documents",
                component="analysis",
                operation="explore_documents",
                context={
                    "document_count": len(document_paths),
                    "directory": params.directory_path,
                }
            )
            
            # Process documents
            documents = []
            for file_path in document_paths:
                try:
                    doc_info = await self._process_document(
                        file_path=file_path,
                        extract_metadata=params.extract_metadata,
                        extract_full_text=params.extract_full_text,
                        extract_summaries=params.extract_summaries,
                        categorize=params.categorize,
                    )
                    documents.append(doc_info)
                except Exception as e:
                    logger.warning(
                        f"Failed to process document: {str(e)}",
                        component="analysis",
                        operation="explore_documents",
                        exception=e,
                        context={"file": file_path}
                    )
            
            # Calculate statistics
            categories_count = {}
            file_types_count = {}
            size_stats = {
                "total_size": 0,
                "min_size": float('inf'),
                "max_size": 0,
                "avg_size": 0,
            }
            date_stats = {
                "oldest": datetime.datetime.now(),
                "newest": datetime.datetime.min,
            }
            
            # Process each document for statistics
            for doc in documents:
                # Categories
                for category in doc.categories:
                    categories_count[category] = categories_count.get(category, 0) + 1
                
                # File types
                file_type = doc.file_extension
                file_types_count[file_type] = file_types_count.get(file_type, 0) + 1
                
                # Size statistics
                size_stats["total_size"] += doc.file_size
                size_stats["min_size"] = min(size_stats["min_size"], doc.file_size)
                size_stats["max_size"] = max(size_stats["max_size"], doc.file_size)
                
                # Date statistics
                if doc.modified_time:
                    if doc.modified_time < date_stats["oldest"]:
                        date_stats["oldest"] = doc.modified_time
                    if doc.modified_time > date_stats["newest"]:
                        date_stats["newest"] = doc.modified_time
            
            # Calculate average size
            if documents:
                size_stats["avg_size"] = size_stats["total_size"] / len(documents)
            else:
                size_stats["min_size"] = 0
            
            # Convert datetime to string for JSON serialization
            date_stats["oldest"] = date_stats["oldest"].isoformat() if date_stats["oldest"] != datetime.datetime.now() else None
            date_stats["newest"] = date_stats["newest"].isoformat() if date_stats["newest"] != datetime.datetime.min else None
            
            # Index for search if requested
            index_id = None
            if params.index_for_search:
                # This would need actual implementation
                # Create search index
                index_id = "doc_index_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                
                logger.info(
                    f"Created search index: {index_id}",
                    component="analysis",
                    operation="explore_documents"
                )
            
            # Calculate execution time
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Log completion
            logger.success(
                f"Document exploration completed: {len(documents)} documents",
                component="analysis",
                operation="explore_documents",
                context={
                    "document_count": len(documents),
                    "execution_time": execution_time,
                }
            )
            
            # Create and return result
            return DocumentExplorerResult(
                documents=documents,
                count=len(documents),
                categories=categories_count,
                file_types=file_types_count,
                size_stats=size_stats,
                date_stats=date_stats,
                index_id=index_id,
                execution_time=execution_time,
            ).model_dump()

async def extract_pdf_metadata(file_path: str) -> Dict[str, Any]:
    """Extract metadata from a PDF file.
    
    This is a placeholder function for the actual PDF metadata extraction.
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Extracted metadata
    """
    # Placeholder - just indicate this is a simulation
    logger.debug(
        f"Simulating PDF metadata extraction: {file_path}",
        component="analysis",
        operation="extract_pdf_metadata"
    )
    
    # Return placeholder metadata
    file_name = os.path.basename(file_path)
    file_base = os.path.splitext(file_name)[0]
    
    return {
        "Title": file_base,
        "Author": "Unknown Author",
        "CreationDate": datetime.datetime.now().strftime("%Y-%m-%d"),
        "Subject": "Simulated PDF",
        "Keywords": "simulated, pdf, metadata",
        "Creator": "TSAP PDF Extractor",
        "Producer": "TSAP",
    }


# Convenience function to use the document explorer
async def explore_documents(params: DocumentExplorerParams) -> DocumentExplorerResult:
    """Explore documents with specified parameters.
    
    This is a convenience function that uses the DocumentExplorer class.
    
    Args:
        params: Exploration parameters
        
    Returns:
        Exploration results
    """
    explorer = DocumentExplorer()
    result = await explorer.explore(params)
    return DocumentExplorerResult(**result)