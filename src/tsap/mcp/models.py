"""
Data models for MCP protocol operations.

This module defines Pydantic models for the different MCP operations 
and their parameters and results.
"""
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator, root_validator, model_validator


# ==================== Core Tools Models ====================

class RipgrepMatch(BaseModel):
    """A match found by ripgrep."""
    
    path: str = Field(..., description="Path to the file containing the match")
    line_number: int = Field(..., description="Line number of the match (1-based)")
    column_number: Optional[int] = Field(None, description="Column number of the match (1-based)")
    match_text: str = Field(..., description="The matched text")
    line_text: str = Field(..., description="The entire line containing the match")
    before_context: List[str] = Field(default_factory=list, description="Lines before the match")
    after_context: List[str] = Field(default_factory=list, description="Lines after the match")
    submatches: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Information about capturing groups if regex used"
    )


class RipgrepSearchParams(BaseModel):
    """Parameters for ripgrep search."""
    
    pattern: str = Field(..., description="Pattern to search for")
    paths: List[str] = Field(..., description="Paths to search in")
    case_sensitive: bool = Field(False, description="Use case-sensitive matching")
    whole_word: bool = Field(False, description="Match whole words only")
    regex: bool = Field(True, description="Treat pattern as regex")
    invert_match: bool = Field(False, description="Invert matching (find non-matching lines)")
    file_patterns: Optional[List[str]] = Field(
        None, 
        description="Only search files matching these glob patterns"
    )
    exclude_patterns: Optional[List[str]] = Field(
        None, 
        description="Exclude files matching these glob patterns"
    )
    max_depth: Optional[int] = Field(None, description="Maximum directory depth to search")
    max_count: Optional[int] = Field(None, description="Maximum matches per file")
    max_total_matches: Optional[int] = Field(None, description="Maximum total matches")
    context_lines: int = Field(2, description="Number of context lines before and after match")
    before_context: Optional[int] = Field(None, description="Number of context lines before match")
    after_context: Optional[int] = Field(None, description="Number of context lines after match")
    no_ignore: bool = Field(False, description="Don't respect .gitignore etc.")
    follow_symlinks: bool = Field(False, description="Follow symbolic links")
    hidden: bool = Field(False, description="Search hidden files and directories")
    encoding: Optional[str] = Field(None, description="Specify text encoding of files")
    binary: bool = Field(False, description="Search binary files")


class RipgrepSearchResult(BaseModel):
    """Result of a ripgrep search operation."""
    
    matches: List[RipgrepMatch] = Field(default_factory=list, description="List of matches found")
    stats: Dict[str, Any] = Field(default_factory=dict, description="Search statistics")
    truncated: bool = Field(False, description="Whether the results were truncated")
    command: Optional[str] = Field(None, description="Actual command executed")
    execution_time: float = Field(..., description="Execution time in seconds")


class AwkProcessParams(BaseModel):
    """Parameters for AWK processing."""
    
    script: str = Field(..., description="AWK script to execute")
    input_text: Optional[str] = Field(None, description="Input text to process")
    input_files: Optional[List[str]] = Field(None, description="Input files to process")
    field_separator: Optional[str] = Field(None, description="Field separator")
    output_field_separator: Optional[str] = Field(None, description="Output field separator")
    variables: Optional[Dict[str, str]] = Field(None, description="AWK variables to set")
    

class AwkProcessResult(BaseModel):
    """Result of an AWK processing operation."""
    
    output: str = Field(..., description="Output from AWK")
    exit_code: int = Field(0, description="AWK process exit code")
    command: Optional[str] = Field(None, description="Actual command executed")
    execution_time: float = Field(..., description="Execution time in seconds")


class JqQueryParams(BaseModel):
    """Parameters for jq JSON query."""
    
    query: str = Field(..., description="jq query string")
    input_json: Optional[str] = Field(None, description="Input JSON to query")
    input_files: Optional[List[str]] = Field(None, description="Input JSON files to query")
    raw_output: bool = Field(False, description="Output raw strings, not JSON texts")
    compact_output: bool = Field(False, description="Output compact JSON")
    monochrome_output: bool = Field(True, description="Disable colored output")
    
    @root_validator(skip_on_failure=True)
    def validate_input(cls, values):
        """Validate that at least one input source is provided."""
        input_json = values.get("input_json")
        input_files = values.get("input_files")
        
        if input_json is None and (input_files is None or len(input_files) == 0):
            raise ValueError("Either input_json or input_files must be provided")
            
        return values


class JqQueryResult(BaseModel):
    """Result of a jq JSON query operation."""
    
    output: Union[str, List[Any], Dict[str, Any]] = Field(
        ..., 
        description="Query result (string or parsed JSON)"
    )
    parsed: bool = Field(True, description="Whether the output was parsed as JSON")
    exit_code: int = Field(0, description="jq process exit code")
    command: Optional[str] = Field(None, description="Actual command executed")
    execution_time: float = Field(..., description="Execution time in seconds")


class SqliteQueryParams(BaseModel):
    """Parameters for SQLite query."""
    
    query: str = Field(..., description="SQL query to execute")
    database: str = Field(..., description="Path to SQLite database file")
    params: Optional[List[Any]] = Field(None, description="Query parameters")
    headers: bool = Field(True, description="Include column headers in result")
    mode: str = Field("list", description="Result mode: list, dict, or table")
    
    @validator("mode")
    def validate_mode(cls, v):
        """Validate mode value."""
        allowed = ["list", "dict", "table"]
        if v not in allowed:
            raise ValueError(f"Mode must be one of: {', '.join(allowed)}")
        return v


class SqliteQueryResult(BaseModel):
    """Result of a SQLite query operation."""
    
    rows: List[Union[List[Any], Dict[str, Any]]] = Field(
        default_factory=list, 
        description="Query result rows"
    )
    columns: Optional[List[str]] = Field(None, description="Column names")
    row_count: int = Field(0, description="Number of rows returned")
    execution_time: float = Field(..., description="Execution time in seconds")


class HtmlProcessParams(BaseModel):
    """Parameters for HTML processing."""
    
    html: Optional[str] = Field(None, description="HTML content to process")
    url: Optional[str] = Field(None, description="URL to fetch HTML from")
    file_path: Optional[str] = Field(None, description="File path to read HTML from")
    selector: Optional[str] = Field(None, description="CSS selector to extract elements")
    xpath: Optional[str] = Field(None, description="XPath to extract elements")
    extract_tables: bool = Field(False, description="Extract tables from HTML")
    extract_links: bool = Field(False, description="Extract links from HTML")
    extract_text: bool = Field(False, description="Extract plain text from HTML")
    extract_metadata: bool = Field(False, description="Extract metadata from HTML")
    render_js: bool = Field(False, description="Render JavaScript before processing HTML")
    js_timeout: int = Field(30, description="Timeout in seconds for JavaScript rendering")
    interactive_actions: Optional[List[Dict[str, Any]]] = Field(None, description="List of interactive actions to perform when rendering with JavaScript")
    extract_computed_styles: bool = Field(False, description="Extract computed CSS styles for elements (requires render_js)")
    
    @model_validator(mode='after')
    def check_input_source(self) -> 'HtmlProcessParams':
        """Ensure at least one input source is provided."""
        html = self.html
        url = self.url
        file_path = self.file_path
        
        if not any([html, url, file_path]):
            raise ValueError("At least one of 'html', 'url', or 'file_path' must be provided")
        
        return self


class HtmlProcessResult(BaseModel):
    """Result of an HTML processing operation."""
    
    elements: Optional[List[Dict[str, Any]]] = Field(
        None, 
        description="Elements extracted by selector or XPath"
    )
    tables: Optional[List[List[List[str]]]] = Field(
        None, 
        description="Tables extracted from HTML"
    )
    links: Optional[List[Dict[str, str]]] = Field(
        None, 
        description="Links extracted from HTML"
    )
    text: Optional[str] = Field(None, description="Plain text extracted from HTML")
    metadata: Optional[Dict[str, Any]] = Field(
        None, 
        description="Metadata extracted from HTML"
    )
    execution_time: float = Field(..., description="Execution time in seconds")


class PdfExtractParams(BaseModel):
    """Parameters for PDF extraction."""
    
    pdf_path: str = Field(..., description="Path to PDF file")
    pages: Optional[Union[List[int], str]] = Field(
        None, 
        description="Pages to extract (list of page numbers or range like '1-5')"
    )
    extract_text: bool = Field(True, description="Extract text from PDF")
    extract_tables: bool = Field(False, description="Extract tables from PDF")
    extract_images: bool = Field(False, description="Extract images from PDF")
    extract_metadata: bool = Field(False, description="Extract metadata from PDF")
    password: Optional[str] = Field(None, description="Password for encrypted PDF")


class PdfExtractResult(BaseModel):
    """Result of a PDF extraction operation."""
    
    text: Optional[Union[str, Dict[int, str]]] = Field(
        None, 
        description="Extracted text (overall or by page)"
    )
    tables: Optional[List[Dict[str, Any]]] = Field(
        None, 
        description="Tables extracted from PDF"
    )
    images: Optional[List[Dict[str, Any]]] = Field(
        None, 
        description="Images extracted from PDF"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, 
        description="Metadata extracted from PDF"
    )
    page_count: int = Field(0, description="Total number of pages in PDF")
    execution_time: float = Field(..., description="Execution time in seconds")


class TableTransformParams(BaseModel):
    """Parameters specific to data transformation within TableProcessor."""
    filter_expr: Optional[str] = Field(None, description="Python expression to filter rows (e.g., 'row[\"value\"] > 10')")
    columns: Optional[List[str]] = Field(None, description="List of columns to select/keep")
    sort_by: Optional[str] = Field(None, description="Column name to sort by")
    sort_desc: bool = Field(False, description="Sort in descending order")
    computed_columns: Optional[Dict[str, str]] = Field(
        None,
        description="Dict {new_col: expr} to compute columns (e.g., {'total': 'row[\"a\"] + row[\"b\"]'})"
    )


class TableProcessParams(BaseModel):
    """Parameters for table processing."""
    # Input sources
    file_path: Optional[str] = Field(None, description="Path to table file (CSV, TSV, Excel, Parquet, JSON)")
    input_data: Optional[List[Dict[str, Any]]] = Field(None, description="Direct data input as list of dictionaries")

    # Input format and options
    input_format: Optional[str] = Field(None, description="Input format (e.g., csv, tsv, json). Inferred if None.")
    sheet_name: Optional[str] = Field(None, description="Sheet name for Excel files")
    delimiter: Optional[str] = Field(None, description="Delimiter for CSV/TSV files")
    quotechar: Optional[str] = Field(None, description="Quote character for CSV/TSV files")
    has_header: bool = Field(True, description="Input has a header row")
    encoding: Optional[str] = Field(None, description="Input file encoding")
    skiprows: Optional[int] = Field(None, description="Number of rows to skip at the beginning")

    # Transformation options (nested model)
    transform: Optional[TableTransformParams] = Field(None, description="Data transformation parameters")

    # Analysis options
    analyze: bool = Field(False, description="Perform data analysis")

    # Output options
    output_format: Optional[str] = Field(None, description="Output format (e.g., csv, json)")
    output_path: Optional[str] = Field(None, description="Path for output file (temp file if None)")
    json_orient: Optional[str] = Field("records", description="JSON output orientation")

    # Response options
    max_rows_return: Optional[int] = Field(1000, description="Max rows in response data (0 for all)")

    @root_validator(skip_on_failure=True)
    def validate_input_source(cls, values):
        if values.get("file_path") is None and values.get("input_data") is None:
            raise ValueError("Either file_path or input_data must be provided")
        return values


class TableProcessResult(BaseModel):
    """Result of a table processing operation."""
    
    success: bool = Field(..., description="Whether the operation was successful")
    result: List[Dict[str, Any]] = Field(default_factory=list, description="Processed table data (limited by max_rows_return)")
    columns: List[str] = Field(default_factory=list, description="List of column names in the result")
    row_count: int = Field(..., description="Total number of rows after processing")
    column_count: int = Field(..., description="Number of columns in the result")
    analysis: Optional[Dict[str, Any]] = Field(None, description="Analysis results if requested")
    output_path: Optional[str] = Field(None, description="Path to the output file if generated")
    execution_time: float = Field(..., description="Execution time for this specific call in seconds")


# ==================== Composite Operations Models ====================

class SearchPattern(BaseModel):
    """Search pattern with metadata for parallel search."""
    
    pattern: str = Field(..., description="Pattern to search for")
    description: Optional[str] = Field(None, description="Description of what this pattern matches")
    category: Optional[str] = Field(None, description="Category for this pattern")
    regex: bool = Field(True, description="Whether pattern is a regex")
    case_sensitive: bool = Field(False, description="Use case-sensitive matching")
    whole_word: bool = Field(False, description="Match whole words only")
    
    # Additional fields for advanced pattern metadata
    tags: List[str] = Field(default_factory=list, description="Tags for this pattern")
    priority: int = Field(1, description="Priority of this pattern (higher = more important)")
    confidence: float = Field(1.0, description="Confidence in this pattern (0.0-1.0)")


class ParallelSearchParams(BaseModel):
    """Parameters for parallel search operation."""
    
    patterns: List[SearchPattern] = Field(
        ..., 
        description="Patterns to search for in parallel"
    )
    paths: List[str] = Field(..., description="Paths to search in")
    file_patterns: Optional[List[str]] = Field(
        None, 
        description="Only search files matching these glob patterns"
    )
    exclude_patterns: Optional[List[str]] = Field(
        None, 
        description="Exclude files matching these glob patterns"
    )
    max_matches_per_pattern: Optional[int] = Field(
        None, 
        description="Maximum matches per pattern"
    )
    max_total_matches: Optional[int] = Field(
        None, 
        description="Maximum total matches across all patterns"
    )
    context_lines: int = Field(2, description="Number of context lines around matches")
    consolidate_overlapping: bool = Field(
        True, 
        description="Consolidate overlapping matches from different patterns"
    )
    min_confidence: Optional[float] = Field(
        None, 
        description="Minimum pattern confidence to include"
    )


class ParallelSearchMatch(BaseModel):
    """A match from parallel search, including pattern metadata."""
    
    path: str = Field(..., description="Path to the file containing the match")
    line_number: int = Field(..., description="Line number of the match (1-based)")
    match_text: str = Field(..., description="The matched text")
    line_text: str = Field(..., description="The entire line containing the match")
    pattern: str = Field(..., description="Pattern that matched")
    pattern_description: Optional[str] = Field(
        None, 
        description="Description of the matched pattern"
    )
    pattern_category: Optional[str] = Field(
        None, 
        description="Category of the matched pattern"
    )
    before_context: List[str] = Field(
        default_factory=list, 
        description="Lines before the match"
    )
    after_context: List[str] = Field(
        default_factory=list, 
        description="Lines after the match"
    )
    tags: List[str] = Field(default_factory=list, description="Tags from the pattern")
    confidence: float = Field(1.0, description="Confidence of this match (0.0-1.0)")


class ParallelSearchResult(BaseModel):
    """Result of a parallel search operation."""
    
    matches: List[ParallelSearchMatch] = Field(
        default_factory=list, 
        description="Matches found"
    )
    pattern_stats: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Statistics per pattern"
    )
    total_patterns: int = Field(0, description="Total number of patterns searched")
    total_files_searched: int = Field(0, description="Total number of files searched")
    total_matches: int = Field(0, description="Total number of matches found")
    truncated: bool = Field(False, description="Whether results were truncated")
    execution_time: float = Field(..., description="Execution time in seconds")


class RecursiveRefinementParams(BaseModel):
    """Parameters for recursive refinement operation."""
    
    initial_pattern: str = Field(..., description="Initial pattern to search for")
    paths: List[str] = Field(..., description="Paths to search in")
    refinement_steps: List[Dict[str, Any]] = Field(
        ..., 
        description="Steps for recursive refinement"
    )
    max_iterations: int = Field(5, description="Maximum number of refinement iterations")
    max_matches_per_iteration: Optional[int] = Field(
        None, 
        description="Maximum matches per iteration"
    )
    context_lines: int = Field(2, description="Number of context lines around matches")
    


class RecursiveRefinementResult(BaseModel):
    """Result of a recursive refinement operation."""
    
    final_matches: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Final matches after refinement"
    )
    iteration_results: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Results from each iteration"
    )
    iterations_completed: int = Field(0, description="Number of iterations completed")
    refinement_path: List[str] = Field(
        default_factory=list, 
        description="Sequence of refinement patterns used"
    )
    execution_time: float = Field(..., description="Execution time in seconds")


class ContextExtractParams(BaseModel):
    """Parameters for context extraction operation."""
    
    matches: List[Dict[str, Any]] = Field(
        ..., 
        description="Matches to extract context around"
    )
    context_type: str = Field(..., description="Type of context to extract")
    context_lines: Optional[int] = Field(None, description="Lines of context to extract")
    include_file_context: bool = Field(
        False, 
        description="Include file-level context (imports, namespace, etc.)"
    )
    consolidate_contexts: bool = Field(
        True, 
        description="Consolidate overlapping contexts"
    )
    
    @validator("context_type")
    def validate_context_type(cls, v):
        """Validate context type."""
        allowed = [
            "lines", "function", "class", "block", "paragraph", "section", 
            "file", "unit", "custom"
        ]
        if v not in allowed:
            raise ValueError(f"Context type must be one of: {', '.join(allowed)}")
        return v


class ContextExtractResult(BaseModel):
    """Result of a context extraction operation."""
    
    contexts: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Extracted contexts"
    )
    consolidated_count: int = Field(
        0, 
        description="Number of contexts consolidated (if applicable)"
    )
    execution_time: float = Field(..., description="Execution time in seconds")


class PatternAnalyzerParams(BaseModel):
    """Parameters for pattern analysis operation."""
    
    text: Optional[str] = Field(None, description="Text to analyze for patterns")
    file_paths: Optional[List[str]] = Field(None, description="Files to analyze for patterns")
    pattern_types: List[str] = Field(
        ..., 
        description="Types of patterns to analyze for"
    )
    min_occurrences: int = Field(2, description="Minimum occurrences to consider a pattern")
    max_patterns_per_type: Optional[int] = Field(
        None, 
        description="Maximum patterns to return per type"
    )
    
    @root_validator(skip_on_failure=True)
    def validate_input(cls, values):
        """Validate that at least one input source is provided."""
        text = values.get("text")
        file_paths = values.get("file_paths")
        
        if text is None and (file_paths is None or len(file_paths) == 0):
            raise ValueError("Either text or file_paths must be provided")
            
        return values
    
    @validator("pattern_types")
    def validate_pattern_types(cls, v):
        """Validate pattern types."""
        allowed = [
            "regex", "string", "number", "date", "email", "url", "path", 
            "identifier", "keyword", "symbol", "structure"
        ]
        for pattern_type in v:
            if pattern_type not in allowed:
                raise ValueError(
                    f"Pattern type '{pattern_type}' not recognized. "
                    f"Allowed types: {', '.join(allowed)}"
                )
        return v


class PatternAnalyzerResult(BaseModel):
    """Result of a pattern analysis operation."""
    
    patterns: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict, 
        description="Discovered patterns grouped by type"
    )
    pattern_counts: Dict[str, int] = Field(
        default_factory=dict, 
        description="Count of patterns found by type"
    )
    common_occurrences: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Most common pattern occurrences"
    )
    suggested_regex: Optional[Dict[str, str]] = Field(
        None, 
        description="Suggested regex patterns for identified structures"
    )
    execution_time: float = Field(..., description="Execution time in seconds")


# Add more composite operation models as needed...


# ==================== Analysis Tools Models ====================

class CodeAnalyzerParams(BaseModel):
    """Parameters for code analysis operation."""
    
    code: Optional[str] = Field(None, description="Code to analyze")
    file_paths: Optional[List[str]] = Field(None, description="Files to analyze")
    repository_path: Optional[str] = Field(None, description="Repository to analyze")
    language: Optional[str] = Field(None, description="Programming language")
    analysis_types: List[str] = Field(
        ..., 
        description="Types of analysis to perform"
    )
    include_metrics: bool = Field(False, description="Include code metrics in analysis")
    
    @root_validator(skip_on_failure=True)
    def validate_input(cls, values):
        """Validate that at least one input source is provided."""
        code = values.get("code")
        file_paths = values.get("file_paths")
        repository_path = values.get("repository_path")
        
        if (code is None and 
            (file_paths is None or len(file_paths) == 0) and 
            repository_path is None):
            raise ValueError(
                "At least one of code, file_paths, or repository_path must be provided"
            )
            
        return values
    
    @validator("analysis_types")
    def validate_analysis_types(cls, v):
        """Validate analysis types."""
        allowed = [
            "structure", "dependencies", "complexity", "patterns", 
            "security", "style", "documentation", "performance"
        ]
        for analysis_type in v:
            if analysis_type not in allowed:
                raise ValueError(
                    f"Analysis type '{analysis_type}' not recognized. "
                    f"Allowed types: {', '.join(allowed)}"
                )
        return v


class CodeAnalyzerResult(BaseModel):
    """Result of a code analysis operation."""
    
    structure: Optional[Dict[str, Any]] = Field(
        None, 
        description="Code structure analysis"
    )
    dependencies: Optional[Dict[str, Any]] = Field(
        None, 
        description="Dependency analysis"
    )
    complexity: Optional[Dict[str, Any]] = Field(
        None, 
        description="Complexity metrics"
    )
    patterns: Optional[Dict[str, Any]] = Field(
        None, 
        description="Pattern analysis"
    )
    security: Optional[Dict[str, Any]] = Field(
        None, 
        description="Security analysis"
    )
    style: Optional[Dict[str, Any]] = Field(
        None, 
        description="Style analysis"
    )
    documentation: Optional[Dict[str, Any]] = Field(
        None, 
        description="Documentation analysis"
    )
    performance: Optional[Dict[str, Any]] = Field(
        None, 
        description="Performance analysis"
    )
    summary: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Summary of analysis results"
    )
    execution_time: float = Field(..., description="Execution time in seconds")


class DocumentExplorerParams(BaseModel):
    """Parameters for document exploration operation."""
    
    directory_path: str = Field(..., description="Directory containing documents to explore")
    file_patterns: Optional[List[str]] = Field(
        None, 
        description="File patterns to include (e.g., *.pdf, *.docx)"
    )
    exclude_patterns: Optional[List[str]] = Field(
        None, 
        description="File patterns to exclude"
    )
    recursive: bool = Field(True, description="Explore directories recursively")
    max_depth: Optional[int] = Field(None, description="Maximum directory depth")
    max_files: Optional[int] = Field(None, description="Maximum number of files to explore")
    extract_metadata: bool = Field(True, description="Extract document metadata")
    extract_full_text: bool = Field(False, description="Extract full text from documents")
    extract_summaries: bool = Field(True, description="Generate document summaries")
    categorize: bool = Field(True, description="Categorize documents")
    index_for_search: bool = Field(False, description="Index documents for later search")


class DocumentInfo(BaseModel):
    """Information about a document."""
    
    path: str = Field(..., description="Path to the document")
    file_name: str = Field(..., description="File name")
    file_extension: str = Field(..., description="File extension")
    file_size: int = Field(..., description="File size in bytes")
    modified_time: datetime = Field(..., description="Last modified time")
    created_time: Optional[datetime] = Field(None, description="Creation time")
    mime_type: Optional[str] = Field(None, description="MIME type")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    summary: Optional[str] = Field(None, description="Document summary")
    categories: List[str] = Field(default_factory=list, description="Document categories")
    extracted_text: Optional[str] = Field(None, description="Extracted text if requested")


class DocumentExplorerResult(BaseModel):
    """Result of a document exploration operation."""
    
    documents: List[DocumentInfo] = Field(
        default_factory=list, 
        description="Information about explored documents"
    )
    count: int = Field(0, description="Number of documents explored")
    categories: Dict[str, int] = Field(
        default_factory=dict, 
        description="Count of documents by category"
    )
    file_types: Dict[str, int] = Field(
        default_factory=dict, 
        description="Count of documents by file type"
    )
    size_stats: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Size statistics of documents"
    )
    date_stats: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Date statistics of documents"
    )
    index_id: Optional[str] = Field(
        None, 
        description="ID of created search index if requested"
    )
    execution_time: float = Field(..., description="Execution time in seconds")


class MetadataExtractorParams(BaseModel):
    """Parameters for metadata extraction operation."""
    
    file_paths: List[str] = Field(..., description="Paths to files for metadata extraction")
    extraction_schema: Optional[Dict[str, Any]] = Field(
        None, 
        description="Schema defining metadata to extract"
    )
    adaptive: bool = Field(
        True, 
        description="Adaptively discover metadata fields beyond schema"
    )
    extraction_depth: str = Field(
        "standard", 
        description="Depth of metadata extraction (basic, standard, deep)"
    )
    custom_extractors: Optional[List[Dict[str, Any]]] = Field(
        None, 
        description="Custom metadata extractors"
    )
    
    @validator("extraction_depth")
    def validate_extraction_depth(cls, v):
        """Validate extraction depth."""
        allowed = ["basic", "standard", "deep"]
        if v.lower() not in allowed:
            raise ValueError(f"Extraction depth must be one of: {', '.join(allowed)}")
        return v.lower()


class ExtractedMetadata(BaseModel):
    """Metadata extracted from a file."""
    
    file_path: str = Field(..., description="Path to the file")
    file_type: str = Field(..., description="File type")
    standard_metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Standard metadata fields"
    )
    content_metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Content-based metadata"
    )
    custom_metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Custom extracted metadata"
    )
    discovered_fields: List[str] = Field(
        default_factory=list, 
        description="Adaptively discovered metadata fields"
    )
    confidence_scores: Dict[str, float] = Field(
        default_factory=dict, 
        description="Confidence scores for extracted fields (0.0-1.0)"
    )


class MetadataExtractorResult(BaseModel):
    """Result of a metadata extraction operation."""
    
    metadata: List[ExtractedMetadata] = Field(
        default_factory=list, 
        description="Extracted metadata for each file"
    )
    common_fields: List[str] = Field(
        default_factory=list, 
        description="Fields found in all files"
    )
    field_statistics: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Statistics for each metadata field"
    )
    extraction_schema: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Schema used for extraction (including discovered fields)"
    )
    execution_time: float = Field(..., description="Execution time in seconds")


class CorpusCartographerParams(BaseModel):
    """Parameters for corpus mapping operation."""
    
    corpus_paths: List[str] = Field(..., description="Paths to corpus documents")
    relationship_types: List[str] = Field(
        ..., 
        description="Types of relationships to map"
    )
    max_documents: Optional[int] = Field(None, description="Maximum documents to process")
    min_relationship_strength: float = Field(
        0.3, 
        description="Minimum relationship strength to include (0.0-1.0)"
    )
    include_document_summaries: bool = Field(
        True, 
        description="Include document summaries in nodes"
    )
    cluster_documents: bool = Field(
        True, 
        description="Cluster documents by relationships"
    )
    visualization_format: Optional[str] = Field(
        None, 
        description="Format for relationship visualization"
    )
    
    @validator("relationship_types")
    def validate_relationship_types(cls, v):
        """Validate relationship types."""
        allowed = [
            "content_similarity", "references", "temporal", "authorship", 
            "keywords", "topics", "citations", "dependencies", "custom"
        ]
        for rel_type in v:
            if rel_type not in allowed:
                raise ValueError(
                    f"Relationship type '{rel_type}' not recognized. "
                    f"Allowed types: {', '.join(allowed)}"
                )
        return v


class DocumentNode(BaseModel):
    """Node representing a document in the corpus map."""
    
    id: str = Field(..., description="Unique identifier for the document")
    path: str = Field(..., description="Path to the document")
    title: Optional[str] = Field(None, description="Document title")
    type: str = Field(..., description="Document type")
    summary: Optional[str] = Field(None, description="Document summary")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    clusters: List[str] = Field(default_factory=list, description="Clusters the document belongs to")
    importance_score: float = Field(
        0.0, 
        description="Document importance in the corpus (0.0-1.0)"
    )


class DocumentRelationship(BaseModel):
    """Relationship between documents in the corpus map."""
    
    source_id: str = Field(..., description="ID of source document")
    target_id: str = Field(..., description="ID of target document")
    type: str = Field(..., description="Relationship type")
    strength: float = Field(..., description="Relationship strength (0.0-1.0)")
    bidirectional: bool = Field(False, description="Whether relationship is bidirectional")
    properties: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional relationship properties"
    )


class CorpusCluster(BaseModel):
    """Cluster of related documents in the corpus map."""
    
    id: str = Field(..., description="Cluster identifier")
    name: str = Field(..., description="Cluster name")
    document_ids: List[str] = Field(..., description="IDs of documents in the cluster")
    theme: Optional[str] = Field(None, description="Central theme of the cluster")
    keywords: List[str] = Field(default_factory=list, description="Keywords for the cluster")
    cohesion_score: float = Field(
        0.0, 
        description="Internal cohesion of the cluster (0.0-1.0)"
    )


class CorpusCartographerResult(BaseModel):
    """Result of a corpus mapping operation."""
    
    nodes: List[DocumentNode] = Field(
        default_factory=list, 
        description="Document nodes"
    )
    relationships: List[DocumentRelationship] = Field(
        default_factory=list, 
        description="Relationships between documents"
    )
    clusters: List[CorpusCluster] = Field(
        default_factory=list, 
        description="Document clusters"
    )
    central_documents: List[str] = Field(
        default_factory=list, 
        description="IDs of central documents in the corpus"
    )
    isolated_documents: List[str] = Field(
        default_factory=list, 
        description="IDs of isolated documents with few relationships"
    )
    relationship_stats: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Statistics for each relationship type"
    )
    visualization_data: Optional[Dict[str, Any]] = Field(
        None, 
        description="Data for visualization if requested"
    )
    execution_time: float = Field(..., description="Execution time in seconds")


class CounterfactualAnalyzerParams(BaseModel):
    """Parameters for counterfactual analysis operation."""
    
    document_paths: List[str] = Field(..., description="Paths to documents to analyze")
    reference_paths: Optional[List[str]] = Field(
        None, 
        description="Paths to reference documents"
    )
    analysis_types: List[str] = Field(
        ..., 
        description="Types of counterfactual analysis to perform"
    )
    analysis_schema: Optional[Dict[str, Any]] = Field(
        None, 
        description="Schema for structured counterfactual analysis"
    )
    max_counterfactuals: Optional[int] = Field(
        None, 
        description="Maximum number of counterfactuals to generate"
    )
    min_confidence: float = Field(
        0.7, 
        description="Minimum confidence threshold for counterfactuals (0.0-1.0)"
    )
    
    @validator("analysis_types")
    def validate_analysis_types(cls, v):
        """Validate analysis types."""
        allowed = [
            "missing_content", "outliers", "inconsistencies", "contradictions", 
            "gaps", "expectations", "patterns", "anomalies", "custom"
        ]
        for analysis_type in v:
            if analysis_type not in allowed:
                raise ValueError(
                    f"Analysis type '{analysis_type}' not recognized. "
                    f"Allowed types: {', '.join(allowed)}"
                )
        return v
        
    def model_dump(self) -> Dict[str, Any]:
        """Make the model JSON serializable for cache operations."""
        return {
            "document_paths": self.document_paths,
            "reference_paths": self.reference_paths,
            "analysis_types": self.analysis_types,
            "analysis_schema": self.analysis_schema,
            "max_counterfactuals": self.max_counterfactuals,
            "min_confidence": self.min_confidence
        }


class Counterfactual(BaseModel):
    """A counterfactual finding from analysis."""
    
    type: str = Field(..., description="Type of counterfactual")
    description: str = Field(..., description="Description of the counterfactual")
    document_path: str = Field(..., description="Path to the document")
    location: Optional[Dict[str, Any]] = Field(
        None, 
        description="Location in document (if applicable)"
    )
    evidence: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Evidence supporting the counterfactual"
    )
    expected: Optional[Any] = Field(None, description="Expected content or pattern")
    actual: Optional[Any] = Field(None, description="Actual content found")
    confidence: float = Field(..., description="Confidence in the counterfactual (0.0-1.0)")
    severity: Optional[float] = Field(
        None, 
        description="Severity of the counterfactual (0.0-1.0)"
    )
    suggestions: List[str] = Field(
        default_factory=list, 
        description="Suggestions related to the counterfactual"
    )


class CounterfactualAnalyzerResult(BaseModel):
    """Result of a counterfactual analysis operation."""
    
    counterfactuals: List[Counterfactual] = Field(
        default_factory=list, 
        description="Identified counterfactuals"
    )
    summary: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Summary of counterfactual analysis"
    )
    document_stats: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Statistics for each analyzed document"
    )
    patterns: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Patterns identified in the analysis"
    )
    execution_time: float = Field(..., description="Execution time in seconds")


class StrategyCompilerParams(BaseModel):
    """Parameters for strategy compilation operation."""
    
    target_paths: List[str] = Field(..., description="Paths to target content")
    objective: str = Field(..., description="Search/analysis objective description")
    strategy_type: str = Field(..., description="Type of strategy to compile")
    constraints: Optional[Dict[str, Any]] = Field(
        None, 
        description="Constraints on the strategy"
    )
    optimization_criteria: List[str] = Field(
        default_factory=list, 
        description="Criteria to optimize for"
    )
    max_operations: Optional[int] = Field(
        None, 
        description="Maximum number of operations in strategy"
    )
    performance_mode: Optional[str] = Field(
        None, 
        description="Performance mode for strategy execution"
    )
    
    @validator("strategy_type")
    def validate_strategy_type(cls, v):
        """Validate strategy type."""
        allowed = [
            "search", "analysis", "extraction", "transformation", 
            "comparison", "mapping", "custom"
        ]
        if v not in allowed:
            raise ValueError(f"Strategy type must be one of: {', '.join(allowed)}")
        return v


class StrategyOperation(BaseModel):
    """An operation within a compiled search/analysis strategy."""
    
    id: str = Field(..., description="Operation identifier")
    type: str = Field(..., description="Operation type")
    command: str = Field(..., description="Command to execute")
    args: Dict[str, Any] = Field(default_factory=dict, description="Command arguments")
    dependencies: List[str] = Field(
        default_factory=list, 
        description="IDs of operations this depends on"
    )
    description: str = Field(..., description="Human-readable description")
    expected_output: Optional[Dict[str, Any]] = Field(
        None, 
        description="Expected output schema"
    )
    contingency: Optional[Dict[str, Any]] = Field(
        None, 
        description="Contingency plan if operation fails"
    )


class CompiledStrategy(BaseModel):
    """A compiled search/analysis strategy."""
    
    id: str = Field(..., description="Strategy identifier")
    name: str = Field(..., description="Strategy name")
    objective: str = Field(..., description="Strategy objective")
    operations: List[StrategyOperation] = Field(
        default_factory=list, 
        description="Operations in the strategy"
    )
    execution_graph: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Execution dependency graph"
    )
    estimated_runtime: float = Field(
        0.0, 
        description="Estimated runtime in seconds"
    )
    estimated_complexity: str = Field(
        "medium", 
        description="Estimated complexity (low, medium, high)"
    )
    performance_mode: str = Field(
        "standard", 
        description="Performance mode for execution"
    )


class StrategyCompilerResult(BaseModel):
    """Result of a strategy compilation operation."""
    
    strategy: CompiledStrategy = Field(..., description="Compiled strategy")
    alternatives: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Alternative strategies considered"
    )
    reasoning: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Reasoning behind the strategy compilation"
    )
    execution_time: float = Field(..., description="Execution time in seconds")


# ==================== Evolution Models ====================

class StrategyEvolutionParams(BaseModel):
    """Parameters for strategy evolution operation."""
    
    base_strategy: CompiledStrategy = Field(..., description="Base strategy to evolve")
    target_paths: List[str] = Field(..., description="Paths to target content")
    objective: str = Field(..., description="Evolution objective")
    success_criteria: Dict[str, Any] = Field(
        ..., 
        description="Criteria to measure success"
    )
    generations: int = Field(5, description="Number of evolution generations")
    population_size: int = Field(10, description="Population size per generation")
    mutation_rate: float = Field(0.2, description="Mutation rate (0.0-1.0)")
    crossover_rate: float = Field(0.3, description="Crossover rate (0.0-1.0)")
    max_runtime: Optional[float] = Field(
        None, 
        description="Maximum runtime for evolution in seconds"
    )
    preserve_elite: bool = Field(
        True, 
        description="Preserve elite strategies across generations"
    )


class EvolutionGeneration(BaseModel):
    """A generation in the strategy evolution process."""
    
    generation_number: int = Field(..., description="Generation number")
    strategies: List[Dict[str, Any]] = Field(..., description="Strategies in this generation")
    best_fitness: float = Field(..., description="Best fitness score in the generation")
    average_fitness: float = Field(..., description="Average fitness score")
    best_strategy_id: str = Field(..., description="ID of the best strategy")
    mutations: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Mutations applied in this generation"
    )
    crossovers: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Crossovers applied in this generation"
    )


class StrategyEvolutionResult(BaseModel):
    """Result of a strategy evolution operation."""
    
    final_strategy: CompiledStrategy = Field(..., description="Final evolved strategy")
    generations: List[EvolutionGeneration] = Field(
        default_factory=list, 
        description="Details of each generation"
    )
    fitness_history: List[float] = Field(
        default_factory=list, 
        description="Best fitness score per generation"
    )
    improvement: float = Field(
        0.0, 
        description="Improvement percentage over base strategy"
    )
    generations_completed: int = Field(0, description="Number of generations completed")
    evolution_path: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Path of evolution from base to final strategy"
    )
    execution_time: float = Field(..., description="Execution time in seconds")


class PatternLibraryParams(BaseModel):
    """Parameters for pattern library operations."""
    
    operation: str = Field(..., description="Operation to perform on pattern library")
    patterns: Optional[List[Dict[str, Any]]] = Field(
        None, 
        description="Patterns to add or update"
    )
    pattern_ids: Optional[List[str]] = Field(
        None, 
        description="IDs of patterns to retrieve or delete"
    )
    categories: Optional[List[str]] = Field(
        None, 
        description="Categories to filter by"
    )
    tags: Optional[List[str]] = Field(None, description="Tags to filter by")
    min_confidence: Optional[float] = Field(
        None, 
        description="Minimum confidence score to filter by"
    )
    max_patterns: Optional[int] = Field(
        None, 
        description="Maximum patterns to return"
    )
    
    @validator("operation")
    def validate_operation(cls, v):
        """Validate operation."""
        allowed = ["add", "update", "get", "delete", "list", "search"]
        if v not in allowed:
            raise ValueError(f"Operation must be one of: {', '.join(allowed)}")
        return v
    
    @root_validator(skip_on_failure=True)
    def validate_params(cls, values):
        """Validate parameters based on operation."""
        operation = values.get("operation")
        patterns = values.get("patterns")
        pattern_ids = values.get("pattern_ids")
        
        if operation in ["add", "update"] and not patterns:
            raise ValueError(f"Operation '{operation}' requires patterns")
            
        if operation in ["get", "delete"] and not pattern_ids:
            raise ValueError(f"Operation '{operation}' requires pattern_ids")
            
        return values


class PatternDefinition(BaseModel):
    """Definition of a pattern in the pattern library."""
    
    id: Optional[str] = Field(None, description="Pattern identifier")
    name: str = Field(..., description="Pattern name")
    pattern: str = Field(..., description="Pattern definition (regex, etc.)")
    description: str = Field(..., description="Pattern description")
    type: str = Field(..., description="Pattern type")
    category: str = Field(..., description="Pattern category")
    tags: List[str] = Field(default_factory=list, description="Pattern tags")
    confidence: float = Field(
        1.0, 
        description="Confidence score for the pattern (0.0-1.0)"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, 
        description="Creation timestamp"
    )
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    examples: List[str] = Field(default_factory=list, description="Example matches")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class PatternLibraryResult(BaseModel):
    """Result of a pattern library operation."""
    
    operation: str = Field(..., description="Operation performed")
    status: str = Field(..., description="Operation status")
    patterns: List[PatternDefinition] = Field(
        default_factory=list, 
        description="Patterns returned (if applicable)"
    )
    count: int = Field(0, description="Number of patterns affected/returned")
    execution_time: float = Field(..., description="Execution time in seconds")


class StrategyJournalParams(BaseModel):
    """Parameters for strategy journal operations."""
    
    operation: str = Field(..., description="Operation to perform on strategy journal")
    strategies: Optional[List[Dict[str, Any]]] = Field(
        None, 
        description="Strategies to record or update"
    )
    strategy_ids: Optional[List[str]] = Field(
        None, 
        description="IDs of strategies to retrieve"
    )
    effectiveness_data: Optional[List[Dict[str, Any]]] = Field(
        None, 
        description="Effectiveness data to record"
    )
    query_filters: Optional[Dict[str, Any]] = Field(
        None, 
        description="Filters for querying strategies"
    )
    max_results: Optional[int] = Field(
        None, 
        description="Maximum results to return"
    )
    
    @validator("operation")
    def validate_operation(cls, v):
        """Validate operation."""
        allowed = [
            "record", "update", "get", "list", "search", 
            "analyze", "record_effectiveness"
        ]
        if v not in allowed:
            raise ValueError(f"Operation must be one of: {', '.join(allowed)}")
        return v


class StrategyJournalEntry(BaseModel):
    """An entry in the strategy journal."""
    
    id: str = Field(..., description="Strategy identifier")
    name: str = Field(..., description="Strategy name")
    objective: str = Field(..., description="Strategy objective")
    strategy_type: str = Field(..., description="Strategy type")
    created_at: datetime = Field(..., description="Creation timestamp")
    operations: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Operations in the strategy"
    )
    performance_metrics: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Performance metrics"
    )
    effectiveness_score: float = Field(
        0.0, 
        description="Overall effectiveness score (0.0-1.0)"
    )
    execution_count: int = Field(0, description="Number of times executed")
    tags: List[str] = Field(default_factory=list, description="Strategy tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class StrategyJournalResult(BaseModel):
    """Result of a strategy journal operation."""
    
    operation: str = Field(..., description="Operation performed")
    status: str = Field(..., description="Operation status")
    entries: List[StrategyJournalEntry] = Field(
        default_factory=list, 
        description="Journal entries returned (if applicable)"
    )
    analysis: Optional[Dict[str, Any]] = Field(
        None, 
        description="Analysis results (if applicable)"
    )
    count: int = Field(0, description="Number of entries affected/returned")
    execution_time: float = Field(..., description="Execution time in seconds")


class StructureSearchParams(BaseModel):
    """Parameters for structure search operation."""
    
    paths: List[str] = Field(..., description="Paths to search in")
    structure_type: str = Field(..., description="Type of structure to search for")
    structure_pattern: str = Field(..., description="Pattern describing the structure")
    file_patterns: Optional[List[str]] = Field(
        None, 
        description="Only search files matching these glob patterns"
    )
    exclude_patterns: Optional[List[str]] = Field(
        None, 
        description="Exclude files matching these glob patterns"
    )
    max_matches: Optional[int] = Field(None, description="Maximum matches to return")
    min_confidence: float = Field(0.5, description="Minimum confidence score (0.0-1.0)")
    include_context: bool = Field(True, description="Include surrounding context")
    context_lines: int = Field(2, description="Number of context lines to include")
    
    @validator("structure_type")
    def validate_structure_type(cls, v):
        """Validate structure type."""
        allowed = [
            "function", "class", "method", "block", "section", "nested", "custom",
            "conditional", "loop", "declaration", "definition", "import"
        ]
        if v not in allowed:
            raise ValueError(f"Structure type must be one of: {', '.join(allowed)}")
        return v


class StructureSearchResult(BaseModel):
    """Result of a structure search operation."""
    
    matches: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Matches found"
    )
    structures: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Structures found"
    )
    stats: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Search statistics"
    )
    truncated: bool = Field(False, description="Whether results were truncated")
    execution_time: float = Field(..., description="Execution time in seconds")


class DiffChunk(BaseModel):
    """Chunk of a diff between files or texts."""
    
    type: str = Field(..., description="Chunk type (added, removed, changed, unchanged)")
    old_start: Optional[int] = Field(None, description="Start line in old content (1-based)")
    old_count: Optional[int] = Field(None, description="Line count in old content")
    new_start: Optional[int] = Field(None, description="Start line in new content (1-based)")
    new_count: Optional[int] = Field(None, description="Line count in new content")
    content: List[str] = Field(default_factory=list, description="Lines in the chunk")
    changes: Optional[List[Dict[str, Any]]] = Field(
        None, 
        description="Detailed changes within lines"
    )


class DiffGeneratorParams(BaseModel):
    """Parameters for diff generation operation."""
    
    old_content: Optional[str] = Field(None, description="Old content as string")
    new_content: Optional[str] = Field(None, description="New content as string")
    old_file: Optional[str] = Field(None, description="Path to old file")
    new_file: Optional[str] = Field(None, description="Path to new file")
    context_lines: int = Field(3, description="Number of context lines around changes")
    ignore_whitespace: bool = Field(False, description="Ignore whitespace changes")
    ignore_case: bool = Field(False, description="Ignore case changes")
    output_format: str = Field("unified", description="Output format (unified, context, html, json)")
    
    @root_validator(skip_on_failure=True)
    def validate_input(cls, values):
        """Validate that at least one input source is provided."""
        old_content = values.get("old_content")
        new_content = values.get("new_content")
        old_file = values.get("old_file")
        new_file = values.get("new_file")
        
        if old_content is None and old_file is None:
            raise ValueError("Either old_content or old_file must be provided")
        
        if new_content is None and new_file is None:
            raise ValueError("Either new_content or new_file must be provided")
            
        return values
    
    @validator("output_format")
    def validate_output_format(cls, v):
        """Validate output format."""
        allowed = ["unified", "context", "html", "json"]
        if v not in allowed:
            raise ValueError(f"Output format must be one of: {', '.join(allowed)}")
        return v


class DiffGeneratorResult(BaseModel):
    """Result of a diff generation operation."""
    
    diff_text: Optional[str] = Field(None, description="Diff as text")
    diff_html: Optional[str] = Field(None, description="Diff as HTML")
    chunks: List[DiffChunk] = Field(default_factory=list, description="Diff chunks")
    stats: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Diff statistics"
    )
    execution_time: float = Field(..., description="Execution time in seconds")


class RegexGeneratorParams(BaseModel):
    """Parameters for regex pattern generation operation."""
    
    examples: List[str] = Field(..., description="Example strings to match")
    negative_examples: Optional[List[str]] = Field(
        None, 
        description="Example strings that should not match"
    )
    pattern_complexity: str = Field(
        "medium", 
        description="Desired complexity level (simple, medium, complex)"
    )
    capture_groups: bool = Field(True, description="Include capture groups in pattern")
    pattern_type: str = Field("standard", description="Pattern type (standard, extended, pcre)")
    test_pattern: bool = Field(True, description="Test the generated pattern against examples")
    
    @validator("pattern_complexity")
    def validate_pattern_complexity(cls, v):
        """Validate pattern complexity."""
        allowed = ["simple", "medium", "complex"]
        if v not in allowed:
            raise ValueError(f"Pattern complexity must be one of: {', '.join(allowed)}")
        return v
    
    @validator("pattern_type")
    def validate_pattern_type(cls, v):
        """Validate pattern type."""
        allowed = ["standard", "extended", "pcre"]
        if v not in allowed:
            raise ValueError(f"Pattern type must be one of: {', '.join(allowed)}")
        return v


class RegexGeneratorResult(BaseModel):
    """Result of a regex pattern generation operation."""
    
    pattern: str = Field(..., description="Generated regex pattern")
    capture_groups: Optional[Dict[str, str]] = Field(
        None, 
        description="Description of capture groups"
    )
    matches_all_examples: bool = Field(True, description="Whether pattern matches all examples")
    avoids_all_negative: bool = Field(True, description="Whether pattern avoids all negative examples")
    test_results: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Pattern test results"
    )
    alternative_patterns: Optional[List[str]] = Field(
        None, 
        description="Alternative pattern suggestions"
    )
    execution_time: float = Field(..., description="Execution time in seconds")


class FilenamePatternParams(BaseModel):
    """Parameters for filename pattern discovery operation."""
    
    directory: str = Field(..., description="Directory to analyze")
    recursive: bool = Field(True, description="Recursively analyze subdirectories")
    include_patterns: Optional[List[str]] = Field(
        None, 
        description="Only include files matching these glob patterns"
    )
    exclude_patterns: Optional[List[str]] = Field(
        None, 
        description="Exclude files matching these glob patterns"
    )
    min_pattern_frequency: int = Field(3, description="Minimum frequency to consider a pattern")
    max_patterns: Optional[int] = Field(None, description="Maximum patterns to discover")
    analyze_extensions: bool = Field(True, description="Analyze file extensions")
    analyze_prefixes: bool = Field(True, description="Analyze filename prefixes")
    analyze_suffixes: bool = Field(True, description="Analyze filename suffixes")
    analyze_segments: bool = Field(True, description="Analyze filename segments")
    analyze_numbering: bool = Field(True, description="Analyze numbering patterns")
    generate_regexes: bool = Field(True, description="Generate regex patterns for discovered patterns")


class FilenamePatternResult(BaseModel):
    """Result of a filename pattern discovery operation."""
    
    patterns: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict, 
        description="Discovered filename patterns by type"
    )
    file_count: int = Field(0, description="Total number of files analyzed")
    extensions: Dict[str, int] = Field(
        default_factory=dict, 
        description="File extensions and their frequencies"
    )
    common_prefixes: Dict[str, int] = Field(
        default_factory=dict, 
        description="Common filename prefixes and their frequencies"
    )
    common_suffixes: Dict[str, int] = Field(
        default_factory=dict, 
        description="Common filename suffixes and their frequencies"
    )
    numbering_patterns: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Detected numbering patterns"
    )
    regex_patterns: Optional[Dict[str, str]] = Field(
        None, 
        description="Generated regex patterns for discovered patterns"
    )
    execution_time: float = Field(..., description="Execution time in seconds")


class DocumentProfilerParams(BaseModel):
    """Parameters for document profiling operation."""
    
    file_paths: List[str] = Field(..., description="Paths to documents to profile")
    extract_text: bool = Field(True, description="Extract and analyze text content")
    analyze_structure: bool = Field(True, description="Analyze document structure")
    analyze_metadata: bool = Field(True, description="Analyze document metadata")
    calculate_statistics: bool = Field(True, description="Calculate text statistics")
    detect_language: bool = Field(True, description="Detect document language")
    extract_entities: bool = Field(False, description="Extract named entities from text")
    extract_keywords: bool = Field(True, description="Extract keywords from text")
    generate_summary: bool = Field(False, description="Generate document summary")
    max_keywords: int = Field(20, description="Maximum keywords to extract")
    max_summary_length: Optional[int] = Field(None, description="Maximum summary length in characters")
    custom_analyzers: Optional[List[Dict[str, Any]]] = Field(
        None, 
        description="Custom document analyzers to use"
    )


class DocumentProfilerResult(BaseModel):
    """Result of a document profiling operation."""
    
    profiles: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Detailed profiles for each document"
    )
    summary: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Summary of all documents profiled"
    )
    common_entities: Dict[str, int] = Field(
        default_factory=dict, 
        description="Common entities across documents"
    )
    common_keywords: Dict[str, int] = Field(
        default_factory=dict, 
        description="Common keywords across documents"
    )
    content_similarities: Optional[List[Dict[str, Any]]] = Field(
        None, 
        description="Similarities between documents"
    )
    execution_time: float = Field(..., description="Execution time in seconds")


class RuntimeLearningParams(BaseModel):
    """Parameters for runtime learning operation."""
    
    learning_target: str = Field(..., description="Target of runtime learning")
    input_data: Dict[str, Any] = Field(..., description="Input data for learning")
    feedback_data: Optional[Dict[str, Any]] = Field(
        None, 
        description="Feedback data for learning"
    )
    learning_mode: str = Field(
        "incremental", 
        description="Learning mode (incremental, batch)"
    )
    max_iterations: Optional[int] = Field(None, description="Maximum learning iterations")
    convergence_threshold: Optional[float] = Field(
        None, 
        description="Convergence threshold"
    )
    
    @validator("learning_target")
    def validate_learning_target(cls, v):
        """Validate learning target."""
        allowed = [
            "pattern_recognition", "ranking_function", "relevance_model",
            "confidence_scoring", "structure_detection", "custom"
        ]
        if v not in allowed:
            raise ValueError(f"Learning target must be one of: {', '.join(allowed)}")
        return v
    
    @validator("learning_mode")
    def validate_learning_mode(cls, v):
        """Validate learning mode."""
        allowed = ["incremental", "batch"]
        if v not in allowed:
            raise ValueError(f"Learning mode must be one of: {', '.join(allowed)}")
        return v


class RuntimeLearningResult(BaseModel):
    """Result of a runtime learning operation."""
    
    learning_target: str = Field(..., description="Target of runtime learning")
    iterations: int = Field(0, description="Number of learning iterations performed")
    initial_state: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Initial state before learning"
    )
    final_state: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Final state after learning"
    )
    improvement_metrics: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Metrics showing improvement"
    )
    convergence_achieved: bool = Field(
        False, 
        description="Whether convergence was achieved"
    )
    execution_time: float = Field(..., description="Execution time in seconds")


class OfflineLearningParams(BaseModel):
    """Parameters for offline learning operation."""
    
    learning_target: str = Field(..., description="Target of offline learning")
    training_data_paths: List[str] = Field(..., description="Paths to training data")
    model_type: str = Field(..., description="Type of model to train")
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Model hyperparameters"
    )
    validation_split: float = Field(
        0.2, 
        description="Fraction of data to use for validation"
    )
    max_epochs: Optional[int] = Field(None, description="Maximum training epochs")
    early_stopping: bool = Field(True, description="Use early stopping")
    save_model: bool = Field(True, description="Save the trained model")
    model_output_path: Optional[str] = Field(None, description="Path to save model")
    
    @validator("learning_target")
    def validate_learning_target(cls, v):
        """Validate learning target."""
        allowed = [
            "pattern_generator", "relevance_ranker", "document_classifier",
            "metadata_extractor", "search_optimizer", "strategy_generator"
        ]
        if v not in allowed:
            raise ValueError(f"Learning target must be one of: {', '.join(allowed)}")
        return v


class OfflineLearningResult(BaseModel):
    """Result of an offline learning operation."""
    
    learning_target: str = Field(..., description="Target of offline learning")
    model_type: str = Field(..., description="Type of model trained")
    training_metrics: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Training metrics"
    )
    validation_metrics: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Validation metrics"
    )
    epochs_completed: int = Field(0, description="Number of training epochs completed")
    model_parameters: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Model parameters"
    )
    model_path: Optional[str] = Field(None, description="Path to saved model")
    execution_time: float = Field(..., description="Execution time in seconds")