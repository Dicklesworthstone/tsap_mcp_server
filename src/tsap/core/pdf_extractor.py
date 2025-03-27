"""
PDF text extraction for TSAP.

This module provides functionality to extract text and metadata from PDF files,
with support for complex layouts, tables, and structured content extraction.
"""
import os
import asyncio
import tempfile
from typing import Dict, List, Any, Optional, Tuple, BinaryIO, Union

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LAParams, LTTextContainer, LTTextBox, LTTextLine, LTPage
except ImportError:
    pdfminer_extract_text = None
    extract_pages = None

from tsap.utils.logging import logger
from tsap.config import get_config
from tsap.performance_mode import get_parameter
from tsap.core.base import BaseCoreTool, register_tool
from tsap.mcp.models import PdfExtractParams, PdfExtractResult


@register_tool("pdf_extractor")
class PdfExtractor(BaseCoreTool):
    """Tool for extracting text and metadata from PDF files."""
    
    def __init__(self):
        """Initialize the PDF extractor tool."""
        super().__init__("pdf_extractor")
        
        # Check available PDF libraries
        self.has_pymupdf = fitz is not None
        self.has_pdfminer = pdfminer_extract_text is not None
        
        if not self.has_pymupdf and not self.has_pdfminer:
            logger.warning(
                "No PDF extraction libraries found. Install PyMuPDF or pdfminer.six for PDF support.",
                component="core",
                operation="init_pdf_extractor"
            )
    
    async def extract_text(self, params: PdfExtractParams) -> PdfExtractResult:
        """Extract text from a PDF file.
        
        Args:
            params: PDF extraction parameters
            
        Returns:
            Extraction results
            
        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            RuntimeError: If no PDF extraction libraries are available
        """
        async with self._measure_execution_time():
            start_time = asyncio.get_event_loop().time()
            
            # Validate PDF path
            if not os.path.isfile(params.pdf_path):
                raise FileNotFoundError(f"PDF file not found: {params.pdf_path}")
                
            # Check if any PDF libraries are available
            if not self.has_pymupdf and not self.has_pdfminer:
                raise RuntimeError(
                    "No PDF extraction libraries available. Install PyMuPDF or pdfminer.six."
                )
            
            # Get timeout from performance mode
            timeout = get_parameter("timeout", 60.0)
            
            # Log extraction start
            logger.info(
                f"Extracting text from PDF: {params.pdf_path}",
                component="core",
                operation="pdf_extract",
                context={
                    "file": params.pdf_path,
                    "pages": params.pages,
                    "extract_text": params.extract_text,
                    "extract_tables": params.extract_tables,
                    "extract_images": params.extract_images,
                    "extract_metadata": params.extract_metadata,
                }
            )
            
            # Extract based on requested functionality
            result = PdfExtractResult(
                page_count=0,
                execution_time=0.0,
            )
            
            try:
                # Get page count and basic info first
                page_count = await self._get_page_count(params.pdf_path, params.password)
                result.page_count = page_count
                
                # Extract metadata if requested
                if params.extract_metadata:
                    metadata = await self._extract_metadata(params.pdf_path, params.password)
                    result.metadata = metadata
                
                # Extract text if requested
                if params.extract_text:
                    # Process pages parameter
                    pages_to_extract = await self._parse_pages_param(params.pages, page_count)
                    
                    # Extract text
                    if pages_to_extract == "all":
                        # Extract all pages as a single string
                        text = await self._extract_all_text(params.pdf_path, params.password)
                        result.text = text
                    else:
                        # Extract specified pages as a dictionary
                        text_by_page = await self._extract_pages_text(
                            params.pdf_path, pages_to_extract, params.password
                        )
                        result.text = text_by_page
                
                # Extract tables if requested
                if params.extract_tables:
                    # Process pages parameter
                    pages_to_extract = await self._parse_pages_param(params.pages, page_count)
                    
                    # Extract tables
                    tables = await self._extract_tables(params.pdf_path, pages_to_extract, params.password)
                    result.tables = tables
                
                # Extract images if requested
                if params.extract_images:
                    # Process pages parameter
                    pages_to_extract = await self._parse_pages_param(params.pages, page_count)
                    
                    # Extract images (info only by default)
                    images = await self._extract_images(
                        params.pdf_path, pages_to_extract, params.password
                    )
                    result.images = images
                
                # Calculate execution time
                result.execution_time = asyncio.get_event_loop().time() - start_time
                
                # Log extraction completion
                logger.success(
                    f"Extracted content from PDF: {params.pdf_path}",
                    component="core",
                    operation="pdf_extract",
                    context={
                        "file": params.pdf_path,
                        "page_count": page_count,
                        "execution_time": result.execution_time,
                    }
                )
                
                return result
                
            except asyncio.TimeoutError:
                # Log the timeout
                logger.warning(
                    f"PDF extraction timed out after {timeout}s: {params.pdf_path}",
                    component="core",
                    operation="pdf_extract",
                    context={"file": params.pdf_path, "timeout": timeout}
                )
                
                # Return partial results if any
                result.execution_time = timeout
                return result
                
            except Exception as e:
                # Log the error
                logger.error(
                    f"PDF extraction failed: {str(e)}",
                    component="core",
                    operation="pdf_extract",
                    exception=e,
                    context={"file": params.pdf_path}
                )
                
                # Re-raise the exception
                raise
    
    async def _get_page_count(self, pdf_path: str, password: Optional[str] = None) -> int:
        """Get the number of pages in a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            password: Optional password for encrypted PDFs
            
        Returns:
            Number of pages
        """
        if self.has_pymupdf:
            return await self._get_page_count_pymupdf(pdf_path, password)
        else:
            return await self._get_page_count_pdfminer(pdf_path, password)
    
    async def _get_page_count_pymupdf(self, pdf_path: str, password: Optional[str] = None) -> int:
        """Get page count using PyMuPDF.
        
        Args:
            pdf_path: Path to the PDF file
            password: Optional password for encrypted PDFs
            
        Returns:
            Number of pages
        """
        # Use asyncio.to_thread in Python 3.9+
        loop = asyncio.get_event_loop()
        
        def count_pages():
            doc = fitz.open(pdf_path)
            if password:
                doc.authenticate(password)
            count = doc.page_count
            doc.close()
            return count
        
        return await loop.run_in_executor(None, count_pages)
    
    async def _get_page_count_pdfminer(self, pdf_path: str, password: Optional[str] = None) -> int:
        """Get page count using pdfminer.
        
        Args:
            pdf_path: Path to the PDF file
            password: Optional password for encrypted PDFs
            
        Returns:
            Number of pages
        """
        if not self.has_pdfminer:
            return 0
            
        # Use asyncio.to_thread in Python 3.9+
        loop = asyncio.get_event_loop()
        
        def count_pages():
            try:
                with open(pdf_path, 'rb') as f:
                    pages = list(extract_pages(f, password=password))
                    return len(pages)
            except Exception:
                return 0
        
        return await loop.run_in_executor(None, count_pages)
    
    async def _parse_pages_param(
        self, pages: Optional[Union[List[int], str]], page_count: int
    ) -> Union[List[int], str]:
        """Parse and validate the pages parameter.
        
        Args:
            pages: Pages parameter (list of page numbers or range like '1-5')
            page_count: Total number of pages in the document
            
        Returns:
            Validated list of page numbers or 'all'
        """
        if pages is None:
            return "all"
            
        if isinstance(pages, str):
            # Parse page range (e.g., "1-5")
            try:
                if pages.lower() == "all":
                    return "all"
                    
                if "-" in pages:
                    start, end = pages.split("-", 1)
                    start_page = int(start.strip())
                    end_page = int(end.strip())
                    
                    # Validate range
                    if start_page < 1:
                        start_page = 1
                    if end_page > page_count:
                        end_page = page_count
                        
                    return list(range(start_page, end_page + 1))
                else:
                    # Single page number
                    page_num = int(pages.strip())
                    if 1 <= page_num <= page_count:
                        return [page_num]
                    else:
                        return "all"
            except ValueError:
                # Invalid format, extract all pages
                return "all"
        elif isinstance(pages, list):
            # List of page numbers
            valid_pages = [p for p in pages if 1 <= p <= page_count]
            return valid_pages if valid_pages else "all"
        else:
            # Invalid type, extract all pages
            return "all"
    
    async def _extract_metadata(
        self, pdf_path: str, password: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract metadata from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            password: Optional password for encrypted PDFs
            
        Returns:
            Dictionary of metadata
        """
        if self.has_pymupdf:
            return await self._extract_metadata_pymupdf(pdf_path, password)
        else:
            return await self._extract_metadata_pdfminer(pdf_path, password)
    
    async def _extract_metadata_pymupdf(
        self, pdf_path: str, password: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract metadata using PyMuPDF.
        
        Args:
            pdf_path: Path to the PDF file
            password: Optional password for encrypted PDFs
            
        Returns:
            Dictionary of metadata
        """
        # Use asyncio.to_thread in Python 3.9+
        loop = asyncio.get_event_loop()
        
        def get_metadata():
            doc = fitz.open(pdf_path)
            if password:
                doc.authenticate(password)
                
            # Extract standard metadata
            metadata = doc.metadata
            
            # Add additional metadata
            result = {
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "keywords": metadata.get("keywords", ""),
                "creator": metadata.get("creator", ""),
                "producer": metadata.get("producer", ""),
                "creation_date": metadata.get("creationDate", ""),
                "modification_date": metadata.get("modDate", ""),
                "page_count": doc.page_count,
                "file_size": os.path.getsize(pdf_path),
                "format": "PDF " + metadata.get("format", ""),
                "encryption": doc.is_encrypted,
                "has_xfa": doc.xfa,
                "has_xml": doc.xml,
                "is_reflowable": doc.is_reflowable,
                "form_fields": doc.has_form_fields,
                "embedded_files": len(doc.embfile_names()),
            }
            
            # Extract page sizes for first 3 pages (as sample)
            page_sizes = []
            for i in range(min(3, doc.page_count)):
                page = doc[i]
                page_sizes.append({
                    "page": i + 1,
                    "width": page.rect.width,
                    "height": page.rect.height,
                    "rotation": page.rotation,
                })
            result["page_sizes"] = page_sizes
            
            doc.close()
            return result
        
        return await loop.run_in_executor(None, get_metadata)
    
    async def _extract_metadata_pdfminer(
        self, pdf_path: str, password: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract metadata using pdfminer.
        
        Args:
            pdf_path: Path to the PDF file
            password: Optional password for encrypted PDFs
            
        Returns:
            Dictionary of metadata
        """
        if not self.has_pdfminer:
            return {}
            
        # Use asyncio.to_thread in Python 3.9+
        loop = asyncio.get_event_loop()
        
        def get_metadata():
            try:
                # Basic file info
                result = {
                    "file_size": os.path.getsize(pdf_path),
                    "page_count": 0,
                }
                
                # Try to get page count and basic info
                with open(pdf_path, 'rb') as f:
                    pages = list(extract_pages(f, password=password))
                    result["page_count"] = len(pages)
                    
                    # Extract page sizes for first 3 pages (as sample)
                    page_sizes = []
                    for i, page in enumerate(pages[:3]):
                        if isinstance(page, LTPage):
                            page_sizes.append({
                                "page": i + 1,
                                "width": page.width,
                                "height": page.height,
                                "rotation": page.rotate if hasattr(page, 'rotate') else 0,
                            })
                    result["page_sizes"] = page_sizes
                    
                return result
            except Exception:
                return {
                    "file_size": os.path.getsize(pdf_path),
                    "error": "Failed to extract metadata",
                }
        
        return await loop.run_in_executor(None, get_metadata)
    
    async def _extract_all_text(
        self, pdf_path: str, password: Optional[str] = None
    ) -> str:
        """Extract text from all pages of a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            password: Optional password for encrypted PDFs
            
        Returns:
            Extracted text
        """
        if self.has_pymupdf:
            return await self._extract_all_text_pymupdf(pdf_path, password)
        else:
            return await self._extract_all_text_pdfminer(pdf_path, password)
    
    async def _extract_all_text_pymupdf(
        self, pdf_path: str, password: Optional[str] = None
    ) -> str:
        """Extract all text using PyMuPDF.
        
        Args:
            pdf_path: Path to the PDF file
            password: Optional password for encrypted PDFs
            
        Returns:
            Extracted text
        """
        # Use asyncio.to_thread in Python 3.9+
        loop = asyncio.get_event_loop()
        
        def extract_text():
            doc = fitz.open(pdf_path)
            if password:
                doc.authenticate(password)
                
            text = ""
            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text = page.get_text()
                text += page_text + "\n\n"
                
            doc.close()
            return text
        
        return await loop.run_in_executor(None, extract_text)
    
    async def _extract_all_text_pdfminer(
        self, pdf_path: str, password: Optional[str] = None
    ) -> str:
        """Extract all text using pdfminer.
        
        Args:
            pdf_path: Path to the PDF file
            password: Optional password for encrypted PDFs
            
        Returns:
            Extracted text
        """
        if not self.has_pdfminer:
            return ""
            
        # Use asyncio.to_thread in Python 3.9+
        loop = asyncio.get_event_loop()
        
        def extract_text():
            return pdfminer_extract_text(pdf_path, password=password)
        
        return await loop.run_in_executor(None, extract_text)
    
    async def _extract_pages_text(
        self, pdf_path: str, pages: List[int], password: Optional[str] = None
    ) -> Dict[int, str]:
        """Extract text from specific pages of a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            pages: List of page numbers to extract (1-based)
            password: Optional password for encrypted PDFs
            
        Returns:
            Dictionary mapping page numbers to extracted text
        """
        if self.has_pymupdf:
            return await self._extract_pages_text_pymupdf(pdf_path, pages, password)
        else:
            return await self._extract_pages_text_pdfminer(pdf_path, pages, password)
    
    async def _extract_pages_text_pymupdf(
        self, pdf_path: str, pages: List[int], password: Optional[str] = None
    ) -> Dict[int, str]:
        """Extract text from specific pages using PyMuPDF.
        
        Args:
            pdf_path: Path to the PDF file
            pages: List of page numbers to extract (1-based)
            password: Optional password for encrypted PDFs
            
        Returns:
            Dictionary mapping page numbers to extracted text
        """
        # Use asyncio.to_thread in Python 3.9+
        loop = asyncio.get_event_loop()
        
        def extract_text():
            doc = fitz.open(pdf_path)
            if password:
                doc.authenticate(password)
                
            result = {}
            for page_num in pages:
                if 1 <= page_num <= doc.page_count:
                    # Adjust for 0-based indexing
                    page = doc[page_num - 1]
                    result[page_num] = page.get_text()
                
            doc.close()
            return result
        
        return await loop.run_in_executor(None, extract_text)
    
    async def _extract_pages_text_pdfminer(
        self, pdf_path: str, pages: List[int], password: Optional[str] = None
    ) -> Dict[int, str]:
        """Extract text from specific pages using pdfminer.
        
        Args:
            pdf_path: Path to the PDF file
            pages: List of page numbers to extract (1-based)
            password: Optional password for encrypted PDFs
            
        Returns:
            Dictionary mapping page numbers to extracted text
        """
        if not self.has_pdfminer:
            return {}
            
        # Use asyncio.to_thread in Python 3.9+
        loop = asyncio.get_event_loop()
        
        def extract_text():
            result = {}
            
            with open(pdf_path, 'rb') as f:
                page_iter = extract_pages(f, password=password)
                
                for i, page in enumerate(page_iter, 1):
                    if i in pages:
                        # Extract text from this page
                        page_text = ""
                        
                        # Get all text elements from the page
                        for element in page:
                            if isinstance(element, LTTextContainer):
                                page_text += element.get_text() + "\n"
                                
                        result[i] = page_text.strip()
                
            return result
        
        return await loop.run_in_executor(None, extract_text)
    
    async def _extract_tables(
        self, pdf_path: str, pages: Union[List[int], str], password: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Extract tables from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            pages: List of page numbers to extract from or 'all'
            password: Optional password for encrypted PDFs
            
        Returns:
            List of extracted tables with metadata
        """
        # This is a placeholder for table extraction
        # In a real implementation, you might want to use libraries like
        # camelot-py, tabula-py, or another table extraction library
        
        # For now, we'll return an empty list but log that table extraction
        # is not yet implemented
        logger.warning(
            "PDF table extraction is not fully implemented yet",
            component="core",
            operation="extract_tables"
        )
        
        return []
    
    async def _extract_images(
        self, pdf_path: str, pages: Union[List[int], str], password: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Extract image information from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            pages: List of page numbers to extract from or 'all'
            password: Optional password for encrypted PDFs
            
        Returns:
            List of image metadata (without actual image data by default)
        """
        if not self.has_pymupdf:
            logger.warning(
                "PDF image extraction requires PyMuPDF",
                component="core",
                operation="extract_images"
            )
            return []
            
        # Use asyncio.to_thread in Python 3.9+
        loop = asyncio.get_event_loop()
        
        def extract_image_info():
            doc = fitz.open(pdf_path)
            if password:
                doc.authenticate(password)
                
            image_list = []
            
            # Determine which pages to process
            page_numbers = []
            if pages == "all":
                page_numbers = range(doc.page_count)
            else:
                # Convert 1-based page numbers to 0-based indices
                page_numbers = [p - 1 for p in pages if 1 <= p <= doc.page_count]
            
            # Extract images from each page
            for page_idx in page_numbers:
                page = doc[page_idx]
                
                # Get image info
                image_dict = page.get_images(full=True)
                
                for img_idx, img_info in enumerate(image_dict):
                    xref = img_info[0]  # Image reference number
                    
                    # Basic image properties
                    width = img_info[2]
                    height = img_info[3]
                    
                    # Add to the list
                    image_list.append({
                        "page": page_idx + 1,  # Convert back to 1-based
                        "index": img_idx + 1,
                        "width": width,
                        "height": height,
                        "xref": xref,
                        "type": img_info[1],
                        "size_bytes": img_info[4],
                        "colorspace": str(img_info[5]),
                        "extraction_supported": True,
                    })
            
            doc.close()
            return image_list
        
        return await loop.run_in_executor(None, extract_image_info)


# Singleton instance
_pdf_extractor: Optional[PdfExtractor] = None


def get_pdf_extractor() -> PdfExtractor:
    """Get the singleton PdfExtractor instance.
    
    Returns:
        PdfExtractor instance
    """
    global _pdf_extractor
    
    if _pdf_extractor is None:
        try:
            _pdf_extractor = PdfExtractor()
        except Exception as e:
            logger.error(
                f"Failed to initialize PdfExtractor: {str(e)}",
                component="core",
                operation="init_pdf_extractor",
                exception=e
            )
            raise
            
    return _pdf_extractor


async def extract_pdf_text(
    pdf_path: str, 
    pages: Optional[Union[List[int], str]] = None,
    password: Optional[str] = None
) -> Optional[Union[str, Dict[int, str]]]:
    """Extract text from a PDF file.
    
    This is a convenience function that uses the PdfExtractor.
    
    Args:
        pdf_path: Path to the PDF file
        pages: Optional pages to extract (None for all)
        password: Optional password for encrypted PDFs
        
    Returns:
        Extracted text as string or dict mapping page numbers to text
    """
    extractor = get_pdf_extractor()
    
    params = PdfExtractParams(
        pdf_path=pdf_path,
        pages=pages,
        extract_text=True,
        extract_tables=False,
        extract_images=False,
        extract_metadata=False,
        password=password,
    )
    
    try:
        result = await extractor.extract_text(params)
        return result.text
    except Exception as e:
        logger.error(
            f"Failed to extract PDF text: {str(e)}",
            component="core",
            operation="extract_pdf_text",
            exception=e,
            context={"file": pdf_path}
        )
        return None


async def extract_pdf_metadata(
    pdf_path: str, password: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Extract metadata from a PDF file.
    
    This is a convenience function that uses the PdfExtractor.
    
    Args:
        pdf_path: Path to the PDF file
        password: Optional password for encrypted PDFs
        
    Returns:
        Dictionary of metadata
    """
    extractor = get_pdf_extractor()
    
    params = PdfExtractParams(
        pdf_path=pdf_path,
        extract_text=False,
        extract_tables=False,
        extract_images=False,
        extract_metadata=True,
        password=password,
    )
    
    try:
        result = await extractor.extract_text(params)
        return result.metadata
    except Exception as e:
        logger.error(
            f"Failed to extract PDF metadata: {str(e)}",
            component="core",
            operation="extract_pdf_metadata",
            exception=e,
            context={"file": pdf_path}
        )
        return None