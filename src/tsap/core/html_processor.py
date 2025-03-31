"""
HTML processing for TSAP.

This module provides functionality to parse, search and extract content from HTML,
with structure-aware processing and advanced selector support.
"""
import asyncio
import importlib.util
import sys
import subprocess
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urljoin
import re

try:
    from bs4 import BeautifulSoup, Tag, NavigableString
    from bs4.element import ResultSet
except ImportError:
    BeautifulSoup = None
    Tag = None
    NavigableString = None
    ResultSet = None

try:
    import httpx
except ImportError:
    httpx = None

# Dynamically check if Playwright is available
playwright_available = importlib.util.find_spec("playwright") is not None

from tsap.utils.logging import logger  # noqa: E402
from tsap.performance_mode import get_parameter  # noqa: E402
from tsap.core.base import BaseCoreTool, register_tool  # noqa: E402
from tsap.mcp.models import HtmlProcessParams, HtmlProcessResult  # noqa: E402

# Constants
PLAYWRIGHT_INSTALL_CMD = ["python", "-m", "playwright", "install", "chromium"]

@dataclass
class HtmlElement:
    """Represents an HTML element extracted from a document."""
    
    tag_name: str
    attributes: Dict[str, str]
    text: str
    html: str
    xpath: str
    css_path: str
    

@register_tool("html_processor")
class HtmlProcessor(BaseCoreTool):
    """Tool for processing and extracting content from HTML."""
    
    def __init__(self):
        """Initialize the HTML processor tool."""
        super().__init__("html_processor")
        
        # Check if BeautifulSoup is available
        self.has_bs4 = BeautifulSoup is not None
        
        if not self.has_bs4:
            logger.warning(
                "BeautifulSoup library not found. Install beautifulsoup4 for HTML processing.",
                component="core",
                operation="init_html_processor"
            )
        
        # Check if httpx is available for URL fetching
        self.has_httpx = httpx is not None
        
        if not self.has_httpx:
            logger.warning(
                "httpx library not found. Install httpx for URL fetching.",
                component="core",
                operation="init_html_processor"
            )
        
        # Check if Playwright is available for JavaScript rendering
        self.has_playwright = playwright_available
        
        if not self.has_playwright:
            logger.warning(
                "Playwright library not found. Install playwright for JavaScript rendering.",
                component="core",
                operation="init_html_processor"
            )
    
    async def process(self, params: HtmlProcessParams) -> HtmlProcessResult:
        """Process HTML content based on parameters.
        
        Args:
            params: HTML processing parameters
            
        Returns:
            HTML processing results
            
        Raises:
            RuntimeError: If BeautifulSoup is not available
            ValueError: If no HTML content source is provided
        """
        async with self._measure_execution_time():
            start_time = asyncio.get_event_loop().time()
            
            # Check if BeautifulSoup is available
            if not self.has_bs4:
                raise RuntimeError(
                    "BeautifulSoup is required for HTML processing. Install beautifulsoup4."
                )
                
            # Get HTML content from the appropriate source
            html_content, base_url = await self._get_html_content(params)
            
            if not html_content:
                raise ValueError("No HTML content provided or could not be retrieved")
                
            # Get timeout from performance mode
            timeout = get_parameter("timeout", 30.0)
            
            # Log processing start
            logger.info(
                "Processing HTML content",
                component="core",
                operation="html_process",
                context={
                    "content_length": len(html_content),
                    "source": "url" if params.url else "file" if params.file_path else "direct",
                    "selector": params.selector,
                    "xpath": params.xpath,
                    "render_js": params.render_js
                }
            )
            
            # Initialize result
            result = HtmlProcessResult(
                execution_time=0.0,
            )
            
            try:
                # Parse HTML
                soup = await self._parse_html(html_content)
                
                # Process based on requested operations
                result_tasks = []
                
                # Extract elements if selector or xpath is provided
                if params.selector or params.xpath:
                    result_tasks.append(self._extract_elements(
                        soup, params.selector, params.xpath, base_url
                    ))
                
                # Extract tables if requested
                if params.extract_tables:
                    result_tasks.append(self._extract_tables(soup, base_url))
                
                # Extract links if requested
                if params.extract_links:
                    result_tasks.append(self._extract_links(soup, base_url))
                
                # Extract text if requested
                if params.extract_text:
                    result_tasks.append(self._extract_text(soup))
                
                # Extract metadata if requested
                if params.extract_metadata:
                    result_tasks.append(self._extract_metadata(soup, html_content, base_url))
                
                # Wait for all processing tasks with timeout
                processing_results = await asyncio.wait_for(
                    asyncio.gather(*result_tasks),
                    timeout=timeout
                )
                
                # Assign results to the result object
                if params.selector or params.xpath:
                    result.elements = processing_results[0]
                    processing_results = processing_results[1:]
                    
                if params.extract_tables:
                    result.tables = processing_results[0]
                    processing_results = processing_results[1:]
                    
                if params.extract_links:
                    result.links = processing_results[0]
                    processing_results = processing_results[1:]
                    
                if params.extract_text:
                    result.text = processing_results[0]
                    processing_results = processing_results[1:]
                    
                if params.extract_metadata:
                    result.metadata = processing_results[0]
                
                # Calculate execution time
                result.execution_time = asyncio.get_event_loop().time() - start_time
                
                # Log processing completion
                logger.success(
                    "HTML processing completed",
                    component="core",
                    operation="html_process",
                    context={
                        "execution_time": result.execution_time,
                        "elements_count": len(result.elements) if result.elements else 0,
                        "tables_count": len(result.tables) if result.tables else 0,
                        "links_count": len(result.links) if result.links else 0,
                        "text_length": len(result.text) if result.text else 0,
                    }
                )
                
                return result
                
            except asyncio.TimeoutError:
                # Log the timeout
                logger.warning(
                    f"HTML processing timed out after {timeout}s",
                    component="core",
                    operation="html_process",
                    context={"timeout": timeout}
                )
                
                # Return partial results if any
                result.execution_time = timeout
                return result
                
            except Exception as e:
                # Log the error
                logger.error(
                    f"HTML processing failed: {str(e)}",
                    component="core",
                    operation="html_process",
                    exception=e
                )
                
                # Re-raise the exception
                raise
    
    async def _get_html_content(
        self, params: HtmlProcessParams
    ) -> Tuple[Optional[str], Optional[str]]:
        """Get HTML content from the specified source.
        
        Args:
            params: HTML processing parameters
            
        Returns:
            Tuple of (html_content, base_url)
        """
        # Check sources in priority order
        if params.html:
            # Direct HTML content
            if params.render_js and params.url:
                # If we have both HTML and URL, and JS rendering is requested,
                # use the URL to render and the original HTML is ignored
                logger.info(
                    "Ignoring provided HTML content for JS rendering from URL",
                    component="core",
                    operation="get_html_content"
                )
                return await self._render_javascript(params.url, params.js_timeout)
            return params.html, None
            
        if params.url:
            # Fetch from URL
            if params.render_js and self.has_playwright:
                # Render with JavaScript
                return await self._render_javascript(params.url, params.js_timeout)
            elif self.has_httpx:
                # Standard fetch without JS
                return await self._fetch_url(params.url)
            else:
                logger.error(
                    "httpx library not available for URL fetching",
                    component="core",
                    operation="get_html_content"
                )
                return None, params.url
            
        if params.file_path:
            # Read from file
            content = await self._read_file(params.file_path)
            if content and params.render_js and self.has_playwright:
                # Convert file path to file URL and render
                file_url = f"file://{os.path.abspath(params.file_path)}"
                return await self._render_javascript(file_url, params.js_timeout)
            return content, None
            
        # No valid source
        return None, None
    
    async def _render_javascript(
        self, url: str, timeout: Optional[int] = 30, interactive_actions: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Optional[str], str]:
        """Render a page with JavaScript using Playwright.
        
        Args:
            url: URL to render
            timeout: Timeout in seconds
            interactive_actions: List of interactive actions to perform
            
        Returns:
            Tuple of (html_content, base_url)
        """
        if not self.has_playwright:
            # Attempt to install Playwright
            installed = await self._ensure_playwright_installed()
            if not installed:
                logger.error(
                    "Cannot render JavaScript: Playwright not available and installation failed",
                    component="core",
                    operation="html_processor"
                )
                return None, url
                
        # Run rendering in Playwright
        from playwright.async_api import async_playwright
        
        try:
            async with async_playwright() as p:
                # Launch browser
                browser = await p.chromium.launch()
                
                try:
                    # Create a new page
                    page = await browser.new_page()
                    
                    # Go to the URL
                    await page.goto(url, wait_until="networkidle", timeout=timeout * 1000)
                    
                    # Wait a bit for any final JavaScript to execute
                    await asyncio.sleep(1)
                    
                    # If interactive actions are provided, perform them
                    if interactive_actions:
                        for action in interactive_actions:
                            action_type = action.get("action", "")
                            
                            if action_type == "click":
                                selector = action.get("selector", "")
                                if selector:
                                    await page.click(selector)
                            
                            elif action_type == "fill":
                                selector = action.get("selector", "")
                                text = action.get("text", "")
                                if selector and text:
                                    await page.fill(selector, text)
                            
                            elif action_type == "wait_for_navigation":
                                await page.wait_for_navigation()
                            
                            elif action_type == "wait_for_selector":
                                selector = action.get("selector", "")
                                if selector:
                                    await page.wait_for_selector(selector)
                            
                            elif action_type == "wait_for_load_state":
                                state = action.get("state", "load")
                                await page.wait_for_load_state(state)
                            
                            # Add more actions as needed
                    
                    # Get the rendered HTML content
                    content = await page.content()
                    
                    # Get the current URL (may have changed due to redirects or interactive actions)
                    current_url = page.url
                    
                    return content, current_url
                finally:
                    await browser.close()
        except Exception as e:
            logger.error(
                f"Error rendering JavaScript: {str(e)}",
                component="core",
                operation="html_processor",
                exception=e
            )
            return None, url
    
    async def _ensure_playwright_installed(self) -> bool:
        """Ensure Playwright is installed with at least one browser."""
        try:
            # First check if module is importable
            from playwright.async_api import async_playwright
            
            # Try to initialize playwright to check if system dependencies are present
            try:
                async with async_playwright() as p:
                    # Try to launch browser to check if system deps are present
                    try:
                        browser = await p.chromium.launch()
                        await browser.close()
                        logger.info("Playwright is fully operational", component="core", operation="html_processor")
                        return True  # Everything works!
                    except Exception as e:
                        error_str = str(e)
                        if "Host system is missing dependencies" in error_str:
                            # Try to install dependencies automatically
                            deps_installed = await self._ensure_playwright_system_deps()
                            if deps_installed:
                                # Try again after installing deps
                                try:
                                    browser = await p.chromium.launch()
                                    await browser.close()
                                    return True
                                except Exception as launch_e:
                                    logger.error(f"Browser launch failed after installing deps: {str(launch_e)}", 
                                               component="core", operation="html_processor")
                            # Fall back to no JS rendering if installation failed
                            return False
                        else:
                            logger.error(f"Error launching browser: {error_str}",
                                        component="core", operation="html_processor")
                            return False
            except Exception as e:
                logger.error(f"Error initializing Playwright: {str(e)}",
                            component="core", operation="html_processor")
                return False
            
            # Check if browser is already installed
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "playwright", "install", "--help",
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            _, stderr = await proc.communicate()
            
            # If already installed or help succeeded, we're good
            if b"already installed" in stderr or proc.returncode == 0:
                return True
            
            # Install Chromium automatically
            logger.info("Installing Chromium browser for JavaScript rendering...",
                      component="core", operation="html_processor")
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "playwright", "install", "chromium"
            )
            await proc.communicate()
            return proc.returncode == 0
        except ImportError:
            logger.warning("Playwright not available. Cannot render JavaScript.",
                         component="core", operation="html_processor")
            return False

    async def _ensure_playwright_system_deps(self) -> bool:
        """Attempt to install Playwright system dependencies automatically."""
        try:
            logger.info("Attempting to install Playwright system dependencies...", 
                       component="core", operation="html_processor")
            
            # First try without sudo (in case user has permissions already)
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "playwright", "install-deps",
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            # Check if it worked
            if proc.returncode == 0:
                logger.info("Successfully installed Playwright dependencies", 
                           component="core", operation="html_processor")
                return True
            
            # If it failed, try with sudo (will prompt for password)
            logger.info("Attempting with sudo (may require password)...", 
                       component="core", operation="html_processor")
            
            # Using sudo requires terminal interaction, so inform the user
            print("\n------------------------")
            print("TSAP needs to install system dependencies for JavaScript rendering.")
            print("You may be prompted for your password.")
            print("------------------------\n")
            
            # Use pkexec or sudo depending on what's available
            for cmd in ["pkexec", "sudo"]:
                try:
                    proc = await asyncio.create_subprocess_exec(
                        cmd, sys.executable, "-m", "playwright", "install-deps",
                        stdout=None, stderr=None  # Don't capture output so user can see prompts
                    )
                    await proc.wait()
                    
                    if proc.returncode == 0:
                        logger.info(f"Successfully installed Playwright dependencies using {cmd}", 
                                   component="core", operation="html_processor")
                        return True
                except FileNotFoundError:
                    continue  # Try next command if this one isn't available
            
            # If we got here, both methods failed
            logger.warning(
                "Failed to install Playwright dependencies automatically. Manual install required.",
                component="core", operation="html_processor"
            )
            
            # Try to parse the exact packages needed
            try:
                # Run again to capture the error message
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, "-m", "playwright", "install-deps",
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()
                stderr_text = stderr.decode('utf-8', errors='replace') if stderr else ""
                
                # Extract apt packages if mentioned
                required_packages = []
                if "apt-get install" in stderr_text:
                    match = re.search(r"sudo apt-get install\s+(.+)", stderr_text)
                    if match:
                        required_packages = match.group(1).strip().split()
                        
                if required_packages:
                    logger.info(
                        f"Required packages: {' '.join(required_packages)}. Please install manually with sudo.",
                        component="core", operation="html_processor"
                    )
                    print("\nTo manually install required packages, run:")
                    print(f"sudo apt-get install {' '.join(required_packages)}")
            except Exception as extract_e:
                logger.error(f"Error extracting required packages: {str(extract_e)}",
                            component="core", operation="html_processor")
            
            return False
            
        except Exception as e:
            logger.error(f"Error installing Playwright dependencies: {str(e)}", 
                        component="core", operation="html_processor")
            return False
    
    async def _fetch_url(self, url: str) -> Tuple[Optional[str], str]:
        """Fetch HTML content from a URL.
        
        Args:
            url: URL to fetch
            
        Returns:
            Tuple of (html_content, base_url)
        """
        # Use asyncio.to_thread in Python 3.9+
        loop = asyncio.get_event_loop()  # noqa: F841
        
        async def fetch():
            try:
                async with httpx.AsyncClient(
                    follow_redirects=True,
                    timeout=get_parameter("timeout", 30.0)
                ) as client:
                    response = await client.get(url)
                    response.raise_for_status()
                    return response.text, url
            except Exception as e:
                logger.error(
                    f"Failed to fetch URL: {str(e)}",
                    component="core",
                    operation="fetch_url",
                    exception=e,
                    context={"url": url}
                )
                return None, url
        
        return await fetch()
    
    async def _read_file(self, file_path: str) -> Optional[str]:
        """Read HTML content from a file.
        
        Args:
            file_path: Path to HTML file
            
        Returns:
            HTML content or None if file cannot be read
        """
        # Use asyncio.to_thread in Python 3.9+
        loop = asyncio.get_event_loop()
        
        def read_file():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except UnicodeDecodeError:
                # Try with a different encoding
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        return f.read()
                except Exception as e:
                    logger.error(
                        f"Failed to read file with latin-1 encoding: {str(e)}",
                        component="core",
                        operation="read_file",
                        exception=e,
                        context={"file": file_path}
                    )
                    return None
            except Exception as e:
                logger.error(
                    f"Failed to read file: {str(e)}",
                    component="core",
                    operation="read_file",
                    exception=e,
                    context={"file": file_path}
                )
                return None
        
        return await loop.run_in_executor(None, read_file)
    
    async def _parse_html(self, html_content: str) -> Any:
        """Parse HTML content with BeautifulSoup.
        
        Args:
            html_content: HTML content to parse
            
        Returns:
            BeautifulSoup object
        """
        # Use asyncio.to_thread in Python 3.9+
        loop = asyncio.get_event_loop()
        
        def parse():
            # Try different parsers in order of preference
            for parser in ['html.parser', 'lxml', 'html5lib']:
                try:
                    return BeautifulSoup(html_content, parser)
                except Exception:
                    continue
                    
            # Fallback to basic parser
            return BeautifulSoup(html_content, 'html.parser')
        
        return await loop.run_in_executor(None, parse)
    
    async def _extract_elements(
        self, soup: Any, css_selector: Optional[str], xpath: Optional[str], base_url: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Extract elements from HTML based on CSS selector or XPath.
        
        Args:
            soup: BeautifulSoup object
            css_selector: CSS selector
            xpath: XPath expression
            base_url: Base URL for resolving relative URLs
            
        Returns:
            List of extracted elements
        """
        # Use asyncio.to_thread in Python 3.9+
        loop = asyncio.get_event_loop()
        
        def extract():
            elements = []
            
            # Process CSS selector if provided
            if css_selector:
                try:
                    selected = soup.select(css_selector)
                    for element in selected:
                        elements.append(self._process_element(element, base_url))
                except Exception as e:
                    logger.error(
                        f"CSS selector error: {str(e)}",
                        component="core",
                        operation="extract_elements",
                        exception=e,
                        context={"selector": css_selector}
                    )
            
            # Process XPath if provided
            if xpath:
                try:
                    # BeautifulSoup doesn't have direct XPath support
                    # For proper XPath, we'd use lxml directly, but for simplicity
                    # we'll use a very basic approximation here
                    if xpath.startswith('//'):
                        tag_name = xpath.split('//')[-1].split('[')[0]
                        selected = soup.find_all(tag_name)
                        for element in selected:
                            elements.append(self._process_element(element, base_url))
                except Exception as e:
                    logger.error(
                        f"XPath error: {str(e)}",
                        component="core",
                        operation="extract_elements",
                        exception=e,
                        context={"xpath": xpath}
                    )
            
            return elements
        
        return await loop.run_in_executor(None, extract)
    
    def _process_element(self, element: Any, base_url: Optional[str]) -> Dict[str, Any]:
        """Process a BeautifulSoup element into a dictionary.
        
        Args:
            element: BeautifulSoup element
            base_url: Base URL for resolving relative URLs
            
        Returns:
            Dictionary with element information
        """
        # Get element attributes
        attributes = {}
        for attr, value in element.attrs.items():
            # Convert to string representation
            if isinstance(value, list):
                attributes[attr] = ' '.join(value)
            else:
                attributes[attr] = str(value)
                
        # Resolve URLs in href and src attributes
        if base_url:
            for attr in ['href', 'src']:
                if attr in attributes and not attributes[attr].startswith(('http://', 'https://')):
                    try:
                        attributes[attr + '_resolved'] = urljoin(base_url, attributes[attr])
                    except Exception:
                        attributes[attr + '_resolved'] = attributes[attr]
        
        # Get element text
        text = element.get_text(strip=True)
        
        # Get element HTML
        html = str(element)
        
        # Simplified XPath (not true XPath, just a representation)
        xpath = self._generate_simple_xpath(element)
        
        # Simplified CSS path
        css_path = self._generate_simple_css_path(element)
        
        return {
            "tag_name": element.name,
            "attributes": attributes,
            "text": text,
            "html": html,
            "xpath": xpath,
            "css_path": css_path,
        }
    
    def _generate_simple_xpath(self, element: Any) -> str:
        """Generate a simplified XPath representation for an element.
        
        Args:
            element: BeautifulSoup element
            
        Returns:
            Simplified XPath string
        """
        parts = []
        current = element
        
        while current and current.name:
            # Get position among siblings of same type
            siblings = current.find_previous_siblings(current.name)
            position = len(siblings) + 1
            
            # Add to path
            parts.append(f"{current.name}[{position}]")
            
            # Move to parent
            current = current.parent
            
        # Reverse and join with slashes
        return '//' + '/'.join(reversed(parts))
    
    def _generate_simple_css_path(self, element: Any) -> str:
        """Generate a simplified CSS selector path for an element.
        
        Args:
            element: BeautifulSoup element
            
        Returns:
            Simplified CSS selector path
        """
        parts = []
        current = element
        
        while current and current.name:
            # Get position among siblings of same type
            siblings = current.find_previous_siblings(current.name)
            position = len(siblings) + 1
            
            # Use ID if available
            if 'id' in current.attrs:
                parts.append(f"{current.name}#{current['id']}")
                break
                
            # Use class if available
            elif 'class' in current.attrs and current['class']:
                classes = '.'.join(current['class'])
                parts.append(f"{current.name}.{classes}")
            else:
                # Use nth-of-type if no ID or class
                parts.append(f"{current.name}:nth-of-type({position})")
                
            # Move to parent
            current = current.parent
            
        # Reverse and join with spaces
        return ' > '.join(reversed(parts))
    
    async def _extract_tables(
        self, soup: Any, base_url: Optional[str]
    ) -> List[List[List[str]]]:
        """Extract tables from HTML content.
        
        Args:
            soup: BeautifulSoup object
            base_url: Base URL for resolving relative URLs
            
        Returns:
            List of tables, each as a list of rows, each row as a list of cells
        """
        # Use asyncio.to_thread in Python 3.9+
        loop = asyncio.get_event_loop()
        
        def extract():
            tables = []
            
            # Find all table elements
            html_tables = soup.find_all('table')
            
            for table in html_tables:
                parsed_table = []
                
                # Get table rows
                rows = table.find_all('tr')
                
                for row in rows:
                    parsed_row = []
                    
                    # Get header cells and data cells
                    cells = row.find_all(['th', 'td'])
                    
                    for cell in cells:
                        # Extract cell text
                        cell_text = cell.get_text(strip=True)
                        parsed_row.append(cell_text)
                        
                    # Add row to table
                    if parsed_row:
                        parsed_table.append(parsed_row)
                
                # Add table to results if it has content
                if parsed_table and any(row for row in parsed_table):
                    tables.append(parsed_table)
            
            return tables
        
        return await loop.run_in_executor(None, extract)
    
    async def _extract_links(
        self, soup: Any, base_url: Optional[str]
    ) -> List[Dict[str, str]]:
        """Extract links from HTML content.
        
        Args:
            soup: BeautifulSoup object
            base_url: Base URL for resolving relative URLs
            
        Returns:
            List of dictionaries with link information
        """
        # Use asyncio.to_thread in Python 3.9+
        loop = asyncio.get_event_loop()
        
        def extract():
            links = []
            
            # Find all link elements
            for anchor in soup.find_all('a', href=True):
                href = anchor['href']
                text = anchor.get_text(strip=True)
                
                link_info = {
                    "href": href,
                    "text": text,
                }
                
                # Resolve relative URLs if base URL is provided
                if base_url and not href.startswith(('http://', 'https://', 'mailto:', 'tel:')):
                    try:
                        link_info["href_resolved"] = urljoin(base_url, href)
                    except Exception:
                        link_info["href_resolved"] = href
                
                # Extract additional attributes
                for attr in ['title', 'rel', 'target', 'class', 'id']:
                    if attr in anchor.attrs:
                        # Convert lists (like multi-value class attributes) to strings
                        value = anchor.attrs[attr]
                        if isinstance(value, list):
                            link_info[attr] = ' '.join(value)
                        else:
                            link_info[attr] = str(value)
                
                links.append(link_info)
            
            return links
        
        return await loop.run_in_executor(None, extract)
    
    async def _extract_text(self, soup: Any) -> str:
        """Extract clean text from HTML content.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Extracted text
        """
        # Use asyncio.to_thread in Python 3.9+
        loop = asyncio.get_event_loop()
        
        def extract():
            # Extract text with minimal formatting
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()    # Remove these elements
                
            # Get text
            text = soup.get_text(separator='\n')
            
            # Break into lines and remove leading/trailing space on each
            lines = (line.strip() for line in text.splitlines())
            
            # Break multi-headlines into a single line
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            
            # Join the lines, removing empty ones
            return '\n'.join(chunk for chunk in chunks if chunk)
        
        return await loop.run_in_executor(None, extract)
    
    async def _extract_metadata(
        self, soup: Any, html_content: str, base_url: Optional[str]
    ) -> Dict[str, Any]:
        """Extract metadata from HTML content.
        
        Args:
            soup: BeautifulSoup object
            html_content: Original HTML content
            base_url: Base URL for resolving relative URLs
            
        Returns:
            Dictionary with metadata
        """
        # Use asyncio.to_thread in Python 3.9+
        loop = asyncio.get_event_loop()
        
        def extract():
            metadata = {}
            
            # Extract title
            title_tag = soup.find('title')
            if title_tag:
                metadata['title'] = title_tag.get_text(strip=True)
            
            # Extract meta tags
            for meta in soup.find_all('meta'):
                # Process standard metadata
                if 'name' in meta.attrs and 'content' in meta.attrs:
                    metadata[meta['name']] = meta['content']
                    
                # Process Open Graph metadata
                if 'property' in meta.attrs and 'content' in meta.attrs:
                    if meta['property'].startswith('og:'):
                        og_property = meta['property'][3:]  # Remove 'og:' prefix
                        if 'og' not in metadata:
                            metadata['og'] = {}
                        metadata['og'][og_property] = meta['content']
                        
                # Process Twitter card metadata
                if 'name' in meta.attrs and meta['name'].startswith('twitter:'):
                    twitter_property = meta['name'][8:]  # Remove 'twitter:' prefix
                    if 'twitter' not in metadata:
                        metadata['twitter'] = {}
                    metadata['twitter'][twitter_property] = meta['content']
            
            # Extract links
            link_metadata = {}
            for link in soup.find_all('link'):
                if 'rel' in link.attrs and 'href' in link.attrs:
                    rel = link['rel'][0] if isinstance(link['rel'], list) else link['rel']
                    href = link['href']
                    
                    # Resolve relative URLs
                    if base_url and not href.startswith(('http://', 'https://')):
                        try:
                            href = urljoin(base_url, href)
                        except Exception:
                            pass
                            
                    link_metadata[rel] = href
            
            if link_metadata:
                metadata['links'] = link_metadata
            
            # Extract main heading
            h1 = soup.find('h1')
            if h1:
                metadata['h1'] = h1.get_text(strip=True)
            
            # Get basic page stats
            metadata['stats'] = {
                'html_size': len(html_content),
                'images': len(soup.find_all('img')),
                'links': len(soup.find_all('a', href=True)),
                'headings': len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])),
                'paragraphs': len(soup.find_all('p')),
                'lists': len(soup.find_all(['ul', 'ol'])),
                'tables': len(soup.find_all('table')),
                'forms': len(soup.find_all('form')),
                'scripts': len(soup.find_all('script')),
                'styles': len(soup.find_all('style')),
            }
            
            return metadata
        
        return await loop.run_in_executor(None, extract)

    async def _extract_computed_styles(self, soup: Any, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract computed styles for elements.
        
        This requires JavaScript rendering to be enabled and uses JavaScript to 
        extract the computed CSS styles for each element.
        
        Args:
            soup: BeautifulSoup object (not used in this method but kept for consistency)
            elements: List of element dictionaries to enhance with computed styles
            
        Returns:
            Updated element dictionaries with computed styles
        """
        if not self.has_playwright or not hasattr(self, "_page"):
            logger.warning(
                "Cannot extract computed styles: Playwright not available or page not rendered",
                component="core",
                operation="html_processor"
            )
            return elements
            
        # For each element, extract its computed styles using JavaScript
        for i, elem in enumerate(elements):
            try:
                xpath = elem.get("xpath")
                if not xpath:
                    continue
                    
                # Use JavaScript to get computed styles
                computed_styles = await self._page.evaluate(f"""() => {{
                    const element = document.evaluate(
                        "{xpath}", 
                        document, 
                        null, 
                        XPathResult.FIRST_ORDERED_NODE_TYPE, 
                        null
                    ).singleNodeValue;
                    
                    if (!element) return null;
                    
                    const styles = window.getComputedStyle(element);
                    const result = {{}};
                    
                    // Extract common useful properties
                    ['color', 'background-color', 'font-family', 'font-size', 
                     'margin', 'padding', 'display', 'position', 'width', 'height']
                    .forEach(prop => {{
                        result[prop] = styles.getPropertyValue(prop);
                    }});
                    
                    return result;
                }}""")
                
                if computed_styles:
                    elem["computed_styles"] = computed_styles
            except Exception as e:
                logger.error(
                    f"Error extracting computed styles for element {i}: {str(e)}",
                    component="core",
                    operation="html_processor",
                    exception=e
                )
                
        return elements
            
    async def process_html(
        self,
        html: Optional[str] = None,
        url: Optional[str] = None,
        file_path: Optional[str] = None,
        selector: Optional[str] = None,
        xpath: Optional[str] = None,
        extract_tables: bool = False,
        extract_links: bool = False,
        extract_text: bool = False,
        extract_metadata: bool = False,
        render_js: bool = False,
        js_timeout: int = 30,
        interactive_actions: Optional[List[Dict[str, Any]]] = None,
        extract_computed_styles: bool = False
    ) -> Dict[str, Any]:
        """Process HTML content.
        
        Args:
            html: HTML content to process
            url: URL to fetch HTML from
            file_path: Path to HTML file
            selector: CSS selector for elements to extract
            xpath: XPath for elements to extract
            extract_tables: Whether to extract tables
            extract_links: Whether to extract links
            extract_text: Whether to extract plain text
            extract_metadata: Whether to extract metadata
            render_js: Whether to render JavaScript
            js_timeout: Timeout for JavaScript rendering
            interactive_actions: List of interactive actions for JavaScript rendering
            extract_computed_styles: Whether to extract computed CSS styles
            
        Returns:
            Dictionary of processed results
        """
        logger.info(
            "Processing HTML content",
            component="core",
            operation="html_process"
        )
        
        result = {
            "elements": None,
            "tables": None,
            "links": None,
            "text": None,
            "metadata": None
        }
        
        # Source priority: HTML > URL > File
        html_content = html
        base_url = url
        
        # 1. Get HTML content from source
        if html_content is None and url is not None:
            if render_js:
                html_content, base_url = await self._render_javascript(
                    url, 
                    timeout=js_timeout,
                    interactive_actions=interactive_actions
                )
                # Store the page for potential computed style extraction
                self._current_url = base_url
            else:
                html_content, base_url = await self._fetch_url(url)
        
        if html_content is None and file_path is not None:
            html_content = await self._read_file(file_path)
            base_url = None
            
        if html_content is None:
            raise ValueError("Failed to obtain HTML content from any source")
            
        # 2. Parse HTML
        soup = await self._parse_html(html_content)
        if not soup:
            raise ValueError("Failed to parse HTML content")
            
        # 3. Process according to parameters
        if selector:
            result["elements"] = await self._extract_elements(soup, selector, "css")
        elif xpath:
            result["elements"] = await self._extract_elements(soup, xpath, "xpath")
            
        if extract_tables:
            result["tables"] = await self._extract_tables(soup)
            
        if extract_links:
            result["links"] = await self._extract_links(soup, base_url)
            
        if extract_text:
            result["text"] = await self._extract_text(soup)
            
        if extract_metadata:
            result["metadata"] = await self._extract_metadata(soup)
            
        # 4. Extract computed styles if requested
        if extract_computed_styles and render_js and result["elements"]:
            result["elements"] = await self._extract_computed_styles(soup, result["elements"])
            
        logger.info(
            "HTML processing completed",
            component="core",
            operation="html_process"
        )
        
        return result


# Singleton instance
_html_processor: Optional[HtmlProcessor] = None


def get_html_processor() -> HtmlProcessor:
    """Get the singleton HtmlProcessor instance.
    
    Returns:
        HtmlProcessor instance
    """
    global _html_processor
    
    if _html_processor is None:
        try:
            _html_processor = HtmlProcessor()
        except Exception as e:
            logger.error(
                f"Failed to initialize HtmlProcessor: {str(e)}",
                component="core",
                operation="init_html_processor",
                exception=e
            )
            raise
            
    return _html_processor


async def process_html(params: HtmlProcessParams) -> HtmlProcessResult:
    """Process HTML content based on parameters.
    
    This is a convenience function that uses the HtmlProcessor.
    
    Args:
        params: HTML processing parameters
        
    Returns:
        HTML processing results
    """
    processor = get_html_processor()
    return await processor.process(params)


async def extract_html_text(
    html: Optional[str] = None,
    url: Optional[str] = None,
    file_path: Optional[str] = None
) -> Optional[str]:
    """Extract clean text from HTML content.
    
    This is a convenience function that uses the HtmlProcessor.
    
    Args:
        html: Optional HTML content
        url: Optional URL to fetch HTML from
        file_path: Optional file path to read HTML from
        
    Returns:
        Extracted text
    """
    processor = get_html_processor()
    
    params = HtmlProcessParams(
        html=html,
        url=url,
        file_path=file_path,
        extract_text=True,
    )
    
    try:
        result = await processor.process(params)
        return result.text
    except Exception as e:
        logger.error(
            f"Failed to extract HTML text: {str(e)}",
            component="core",
            operation="extract_html_text",
            exception=e
        )
        return None


async def extract_html_tables(
    html: Optional[str] = None,
    url: Optional[str] = None,
    file_path: Optional[str] = None
) -> Optional[List[List[List[str]]]]:
    """Extract tables from HTML content.
    
    This is a convenience function that uses the HtmlProcessor.
    
    Args:
        html: Optional HTML content
        url: Optional URL to fetch HTML from
        file_path: Optional file path to read HTML from
        
    Returns:
        List of tables
    """
    processor = get_html_processor()
    
    params = HtmlProcessParams(
        html=html,
        url=url,
        file_path=file_path,
        extract_tables=True,
    )
    
    try:
        result = await processor.process(params)
        return result.tables
    except Exception as e:
        logger.error(
            f"Failed to extract HTML tables: {str(e)}",
            component="core",
            operation="extract_html_tables",
            exception=e
        )
        return None