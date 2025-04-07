"""
Processing prompts for TSAP MCP Server.

This module provides MCP prompt implementations for data processing
operations and related tasks.
"""
from typing import Optional, List, Dict, Any
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base


def register_processing_prompts(mcp: FastMCP) -> None:
    """Register all processing-related prompts with the MCP server.
    
    Args:
        mcp: FastMCP server instance
    """
    
    @mcp.prompt()
    def text_processing(
        text: str, 
        processing_goal: str,
        example_output: Optional[str] = None
    ) -> list[base.Message]:
        """Generate a prompt for text processing operations.
        
        This prompt helps users process and transform text data using
        various operations like cleaning, normalization, and extraction.
        
        Args:
            text: Sample text to process
            processing_goal: What the user wants to accomplish with processing
            example_output: Optional example of desired output format
            
        Returns:
            List of messages forming the prompt
        """
        example_context = f"\n\nHere's an example of the output format I'm looking for:\n```\n{example_output}\n```" if example_output else ""
        
        return [
            base.SystemMessage(
                """You are an expert in text processing and data transformation with deep knowledge of
                text manipulation techniques, regex patterns, and natural language processing.
                You help users transform and extract information from text data efficiently."""
            ),
            base.UserMessage(f"I need to process this text to {processing_goal}:"),
            base.UserMessage(f"```\n{text}\n```{example_context}"),
            base.UserMessage(
                """Please help me with:
                1. The most appropriate approach for this text processing task
                2. Any patterns or transformations I should apply
                3. How to handle edge cases or exceptions
                4. Implementation suggestions (tools, libraries, or code)
                5. Validation strategies to ensure correct processing"""
            ),
            base.AssistantMessage(
                """I'll help you process this text effectively:
                
                ## Text Analysis
                
                First, let me analyze your text and processing goal:
                
                (I'll examine the text structure and requirements)
                
                ## Processing Approach
                
                To achieve your goal of {processing_goal}, I recommend:
                
                (I'll suggest specific processing approaches)
                
                ## Patterns & Transformations
                
                Apply these specific patterns or transformations:
                
                (I'll provide detailed pattern/transformation recommendations)
                
                ## Edge Case Handling
                
                Be aware of these potential edge cases:
                
                (I'll identify potential issues and solutions)
                
                ## Implementation Guidance
                
                Here's how you can implement this processing:
                
                (I'll suggest specific implementation details)"""
            ),
        ]
    
    @mcp.prompt()
    def data_extraction(
        source_text: str,
        extraction_targets: str,
        format: Optional[str] = None
    ) -> list[base.Message]:
        """Generate a prompt for extracting structured data from text.
        
        This prompt helps users extract specific data elements from
        unstructured or semi-structured text.
        
        Args:
            source_text: Text to extract data from
            extraction_targets: Description of what to extract
            format: Desired output format (JSON, CSV, etc.)
            
        Returns:
            List of messages forming the prompt
        """
        format_info = f" in {format} format" if format else ""
        
        return [
            base.SystemMessage(
                """You are an expert in data extraction and text mining with deep experience
                identifying and extracting structured information from unstructured text.
                You excel at creating patterns and rules to consistently extract valuable data."""
            ),
            base.UserMessage(f"I need to extract {extraction_targets} from this text{format_info}:"),
            base.UserMessage(f"```\n{source_text}\n```"),
            base.UserMessage(
                """Please help me with:
                1. How to identify and extract each target data element
                2. Pattern recognition strategies for consistent extraction
                3. How to handle variations or missing data
                4. Structuring the extracted data appropriately
                5. Validation to ensure extraction accuracy"""
            ),
            base.AssistantMessage(
                """I'll help you extract this data effectively:
                
                ## Source Analysis
                
                First, let me analyze the text and extraction targets:
                
                (I'll examine the text structure and data to extract)
                
                ## Extraction Strategy
                
                To extract the {extraction_targets}, I recommend:
                
                (I'll suggest specific extraction strategies)
                
                ## Pattern Recognition
                
                Use these patterns to identify each element:
                
                (I'll provide specific patterns for each data element)
                
                ## Handling Variations
                
                For variations or missing data, consider:
                
                (I'll provide strategies for edge cases)
                
                ## Output Structure
                
                Structure your extracted data like this:
                
                (I'll provide a recommended data structure)"""
            ),
        ]
    
    @mcp.prompt()
    def html_processing(
        html_sample: str,
        extraction_goal: str
    ) -> list[base.Message]:
        """Generate a prompt for HTML processing and extraction.
        
        This prompt helps users extract data from HTML content and
        transform it into structured formats.
        
        Args:
            html_sample: Sample HTML to process
            extraction_goal: What the user wants to extract from the HTML
            
        Returns:
            List of messages forming the prompt
        """
        return [
            base.SystemMessage(
                """You are an expert in HTML parsing and web scraping with deep knowledge of
                CSS selectors, DOM traversal, and structured data extraction from web pages.
                You help users efficiently extract and process data from HTML content."""
            ),
            base.UserMessage(f"I need to extract {extraction_goal} from this HTML:"),
            base.UserMessage(f"```html\n{html_sample}\n```"),
            base.UserMessage(
                """Please help me with:
                1. The most effective selectors to target the desired elements
                2. How to extract and clean the data properly
                3. Handling variations in the HTML structure
                4. Structuring the extracted data appropriately
                5. Implementation suggestions using common libraries"""
            ),
            base.AssistantMessage(
                """I'll help you extract data from this HTML:
                
                ## HTML Structure Analysis
                
                Let me analyze the HTML structure:
                
                (I'll identify key elements and the document structure)
                
                ## Selector Strategy
                
                To extract {extraction_goal}, use these selectors:
                
                (I'll provide specific CSS selectors)
                
                ## Extraction Process
                
                Follow this process to extract and clean the data:
                
                (I'll outline a step-by-step process)
                
                ## Handling Variations
                
                To handle potential HTML variations:
                
                (I'll suggest robust extraction techniques)
                
                ## Implementation Approach
                
                Here's how you can implement this extraction:
                
                (I'll provide implementation recommendations)"""
            ),
        ]
    
    @mcp.prompt()
    def table_processing(
        table_data: str,
        operation: str,
        format: Optional[str] = None
    ) -> list[base.Message]:
        """Generate a prompt for processing tabular data.
        
        This prompt helps users transform, filter, aggregate, and analyze
        data in tabular formats like CSV, markdown tables, or HTML tables.
        
        Args:
            table_data: Sample table data to process
            operation: Operation to perform (filter, sort, aggregate, etc.)
            format: Format of the table data (CSV, markdown, HTML)
            
        Returns:
            List of messages forming the prompt
        """
        format_info = f" in {format} format" if format else ""
        
        return [
            base.SystemMessage(
                """You are an expert in tabular data processing with deep experience manipulating,
                transforming, and analyzing structured data in various formats. You excel at
                data cleaning, filtering, joining, and aggregation operations."""
            ),
            base.UserMessage(f"I need to {operation} this table data{format_info}:"),
            base.UserMessage(f"```\n{table_data}\n```"),
            base.UserMessage(
                """Please help me with:
                1. How to parse this table data correctly
                2. The specific steps to perform this operation
                3. How to handle edge cases or errors
                4. The most efficient approach for this task
                5. How to validate the results"""
            ),
            base.AssistantMessage(
                f"""I'll help you {operation} this table data:
                
                ## Table Structure Analysis
                
                First, let me analyze your table data:
                
                (I'll examine the table structure and content)
                
                ## Parsing Approach
                
                To correctly parse this table:
                
                (I'll suggest a parsing strategy)
                
                ## {operation.title()} Process
                
                Follow these steps to {operation} the data:
                
                (I'll provide a step-by-step process)
                
                ## Edge Case Handling
                
                Be aware of these potential issues:
                
                (I'll identify edge cases and solutions)
                
                ## Implementation Strategy
                
                Here's an efficient way to implement this:
                
                (I'll suggest specific implementation details)"""
            ),
        ]
    
    @mcp.prompt()
    def json_processing(
        json_data: str,
        operation: str,
        criteria: Optional[str] = None
    ) -> list[base.Message]:
        """Generate a prompt for processing JSON data.
        
        This prompt helps users query, transform, filter, and manipulate
        JSON data structures efficiently.
        
        Args:
            json_data: Sample JSON data to process
            operation: Operation to perform (query, transform, filter, etc.)
            criteria: Specific criteria for the operation (if applicable)
            
        Returns:
            List of messages forming the prompt
        """
        criteria_info = f" based on {criteria}" if criteria else ""
        
        return [
            base.SystemMessage(
                """You are an expert in JSON data processing with deep knowledge of
                JSON query languages, transformation techniques, and data manipulation.
                You help users efficiently extract, transform, and analyze JSON data."""
            ),
            base.UserMessage(f"I need to {operation} this JSON data{criteria_info}:"),
            base.UserMessage(f"```json\n{json_data}\n```"),
            base.UserMessage(
                """Please help me with:
                1. The most effective approach for this JSON operation
                2. Query expressions or transformation logic needed
                3. How to handle nested structures and edge cases
                4. The most efficient implementation method
                5. Validation strategies for the results"""
            ),
            base.AssistantMessage(
                f"""I'll help you {operation} this JSON data:
                
                ## JSON Structure Analysis
                
                Let me analyze your JSON structure:
                
                (I'll examine the JSON structure and content)
                
                ## {operation.title()} Approach
                
                To {operation} this JSON{criteria_info}, I recommend:
                
                (I'll suggest specific approaches)
                
                ## Query/Transformation Logic
                
                Use these expressions or logic:
                
                (I'll provide specific query/transformation details)
                
                ## Handling Nested Structures
                
                For complex or nested elements:
                
                (I'll provide strategies for nested structures)
                
                ## Implementation Strategy
                
                Here's how you can implement this efficiently:
                
                (I'll suggest implementation details)"""
            ),
        ]
    
    @mcp.prompt()
    def data_validation(
        data_sample: str,
        validation_rules: str,
        format: Optional[str] = None
    ) -> list[base.Message]:
        """Generate a prompt for data validation.
        
        This prompt helps users validate data against rules, schemas, 
        or requirements and identify issues.
        
        Args:
            data_sample: Sample data to validate
            validation_rules: Description of validation requirements
            format: Format of the data (JSON, CSV, etc.)
            
        Returns:
            List of messages forming the prompt
        """
        format_info = f" in {format} format" if format else ""
        
        return [
            base.SystemMessage(
                """You are an expert in data validation and quality assurance with deep knowledge of
                schema validation, data integrity rules, and quality control processes. You help users
                ensure their data meets required standards and identify potential issues."""
            ),
            base.UserMessage(f"I need to validate this data{format_info} against these rules:"),
            base.UserMessage(f"Validation rules: {validation_rules}"),
            base.UserMessage(f"```\n{data_sample}\n```"),
            base.UserMessage(
                """Please help me with:
                1. How to interpret and implement these validation rules
                2. Identifying any validation issues in the sample data
                3. How to handle edge cases or special conditions
                4. Implementation strategies for thorough validation
                5. Recommendations for fixing any identified issues"""
            ),
            base.AssistantMessage(
                """I'll help you validate this data:
                
                ## Validation Rule Analysis
                
                First, let me analyze the validation requirements:
                
                (I'll interpret and clarify the validation rules)
                
                ## Data Structure Assessment
                
                Examining your data structure:
                
                (I'll analyze the data format and structure)
                
                ## Validation Results
                
                Based on the rules, I've identified these issues:
                
                (I'll list validation results and issues)
                
                ## Edge Case Considerations
                
                Be aware of these special cases:
                
                (I'll identify edge cases and how to handle them)
                
                ## Implementation Strategy
                
                To implement these validations effectively:
                
                (I'll suggest implementation approaches)"""
            ),
        ] 