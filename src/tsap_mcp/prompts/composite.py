"""
Composite prompts for TSAP MCP Server.

This module provides MCP prompt implementations for composite operations
combining multiple core functionalities.
"""
from typing import Optional, List, Dict, Any
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base


def register_composite_prompts(mcp: FastMCP) -> None:
    """Register all composite-related prompts with the MCP server.
    
    Args:
        mcp: FastMCP server instance
    """
    
    @mcp.prompt()
    def document_profiling(
        document_description: str,
        profile_goal: str = "comprehensive analysis"
    ) -> list[base.Message]:
        """Generate a prompt for document profiling.
        
        This prompt helps users analyze documents to extract key characteristics,
        structures, patterns, and insights.
        
        Args:
            document_description: Description of the document to profile
            profile_goal: Goal of the profiling operation
            
        Returns:
            List of messages forming the prompt
        """
        return [
            base.SystemMessage(
                """You are an expert document analyst with deep knowledge of text analytics,
                content structure recognition, and information architecture. You help users
                understand document characteristics, patterns, and opportunities for optimization."""
            ),
            base.UserMessage(f"I need to profile this document for {profile_goal}:"),
            base.UserMessage(document_description),
            base.UserMessage(
                """Please help me with:
                1. Identifying key document characteristics and structures
                2. Recognizing patterns and organization principles
                3. Analyzing content distribution and information flow
                4. Identifying opportunities for improvement or optimization
                5. Recommendations for further analysis"""
            ),
            base.AssistantMessage(
                """I'll help you profile this document:
                
                ## Document Structure Analysis
                
                Let me analyze the document structure:
                
                (I'll identify document sections, formatting, and organization)
                
                ## Content Characteristics
                
                Key characteristics of this document include:
                
                (I'll identify content types, patterns, and distribution)
                
                ## Information Architecture
                
                The document's information is structured as follows:
                
                (I'll analyze how information is organized)
                
                ## Improvement Opportunities
                
                Consider these opportunities for optimization:
                
                (I'll suggest potential improvements)
                
                ## Additional Analysis Recommendations
                
                For deeper insights, I recommend:
                
                (I'll suggest additional analyses)"""
            ),
        ]
    
    @mcp.prompt()
    def pattern_matching(
        text_sample: str,
        pattern_types: Optional[List[str]] = None
    ) -> list[base.Message]:
        """Generate a prompt for identifying patterns in text.
        
        This prompt helps users identify, extract, and analyze patterns
        in text using various pattern recognition techniques.
        
        Args:
            text_sample: Sample text to analyze for patterns
            pattern_types: Types of patterns to look for (optional)
            
        Returns:
            List of messages forming the prompt
        """
        pattern_focus = f" focusing on {', '.join(pattern_types)}" if pattern_types else ""
        
        return [
            base.SystemMessage(
                """You are an expert in pattern recognition and text analysis with deep knowledge of
                regex, natural language patterns, and structured text extraction. You help users
                identify and leverage patterns in their text data for various applications."""
            ),
            base.UserMessage(f"I need to identify patterns in this text{pattern_focus}:"),
            base.UserMessage(f"```\n{text_sample}\n```"),
            base.UserMessage(
                """Please help me with:
                1. Identifying recurring patterns in the text
                2. Describing each pattern's structure and frequency
                3. Suggesting efficient ways to extract or match these patterns
                4. Handling variations and edge cases for each pattern
                5. Potential applications or uses for the identified patterns"""
            ),
            base.AssistantMessage(
                """I'll help you identify patterns in this text:
                
                ## Pattern Identification
                
                I've identified these key patterns:
                
                (I'll list and describe discovered patterns)
                
                ## Pattern Structure Analysis
                
                Each pattern is structured as follows:
                
                (I'll analyze pattern structures and variations)
                
                ## Extraction Strategies
                
                To effectively extract these patterns:
                
                (I'll provide pattern matching strategies)
                
                ## Handling Variations
                
                For pattern variations, consider:
                
                (I'll suggest approaches for variations)
                
                ## Pattern Applications
                
                These patterns could be useful for:
                
                (I'll suggest applications for the patterns)"""
            ),
        ]
    
    @mcp.prompt()
    def context_extraction(
        text: str,
        query: str,
        context_scope: str = "surrounding text"
    ) -> list[base.Message]:
        """Generate a prompt for extracting context around key information.
        
        This prompt helps users identify relevant context around
        specific information points in text.
        
        Args:
            text: Text to extract context from
            query: Query or point of interest to find context for
            context_scope: Type of context to extract (surrounding text, relevant sections, etc.)
            
        Returns:
            List of messages forming the prompt
        """
        return [
            base.SystemMessage(
                """You are an expert in contextual analysis and information extraction with
                deep knowledge of text semantics, relevance determination, and content relationships.
                You help users identify meaningful context around key information points."""
            ),
            base.UserMessage(f"I need to extract {context_scope} context around '{query}' in this text:"),
            base.UserMessage(f"```\n{text}\n```"),
            base.UserMessage(
                """Please help me with:
                1. Identifying the most relevant context surrounding this query
                2. Determining appropriate context boundaries
                3. Extracting the context with proper formatting
                4. Highlighting key relationships within the context
                5. Suggesting how to use this context effectively"""
            ),
            base.AssistantMessage(
                """I'll help you extract meaningful context:
                
                ## Query Analysis
                
                Let me analyze your query point:
                
                (I'll identify the query's significance)
                
                ## Context Identification
                
                The most relevant context includes:
                
                (I'll identify appropriate context)
                
                ## Extracted Context
                
                Here's the extracted context with the query highlighted:
                
                (I'll present formatted context)
                
                ## Relationship Analysis
                
                Within this context, note these key relationships:
                
                (I'll identify relationships to the query)
                
                ## Context Utilization
                
                This context can be effectively used to:
                
                (I'll suggest applications for the context)"""
            ),
        ]
    
    @mcp.prompt()
    def diff_generation(
        original_description: str,
        modified_description: str,
        diff_goal: str = "identify changes"
    ) -> list[base.Message]:
        """Generate a prompt for creating and analyzing differences.
        
        This prompt helps users understand differences between versions
        of text, code, or structured data.
        
        Args:
            original_description: Description of the original content
            modified_description: Description of the modified content
            diff_goal: Goal of the diff analysis
            
        Returns:
            List of messages forming the prompt
        """
        return [
            base.SystemMessage(
                """You are an expert in version comparison and change analysis with deep
                knowledge of diff algorithms, change visualization, and impact assessment.
                You help users understand differences between content versions and their significance."""
            ),
            base.UserMessage(f"I need to {diff_goal} between these versions:"),
            base.UserMessage(f"Original: {original_description}"),
            base.UserMessage(f"Modified: {modified_description}"),
            base.UserMessage(
                """Please help me with:
                1. Identifying all significant changes between versions
                2. Categorizing the types of changes (additions, deletions, modifications)
                3. Assessing the potential impact of these changes
                4. Visualizing the differences effectively
                5. Recommendations based on the changes"""
            ),
            base.AssistantMessage(
                """I'll help you analyze the differences between versions:
                
                ## Change Identification
                
                I've identified these key changes:
                
                (I'll list and describe all changes)
                
                ## Change Categorization
                
                The changes can be categorized as:
                
                (I'll categorize the types of changes)
                
                ## Impact Assessment
                
                These changes have the following potential impact:
                
                (I'll assess the significance of changes)
                
                ## Visualization
                
                Here's a visualization of the differences:
                
                (I'll provide a clear representation of changes)
                
                ## Recommendations
                
                Based on these changes, I recommend:
                
                (I'll suggest actions based on the changes)"""
            ),
        ]
    
    @mcp.prompt()
    def regex_generation(
        example_matches: str,
        non_matches: Optional[str] = None,
        complexity: str = "balanced"
    ) -> list[base.Message]:
        """Generate a prompt for creating regex patterns from examples.
        
        This prompt helps users develop effective regex patterns that
        match desired strings and exclude unwanted matches.
        
        Args:
            example_matches: Examples of strings to match
            non_matches: Examples of strings to not match (optional)
            complexity: Desired regex complexity (simple, balanced, advanced)
            
        Returns:
            List of messages forming the prompt
        """
        non_matches_info = f"\n\nThese strings should NOT match:\n{non_matches}" if non_matches else ""
        
        return [
            base.SystemMessage(
                f"""You are an expert in regular expressions with deep knowledge of pattern syntax,
                optimization, and testing. You specialize in creating {complexity} regex patterns
                that accurately match desired strings while avoiding false positives."""
            ),
            base.UserMessage(f"I need a regex pattern that matches these strings:\n{example_matches}{non_matches_info}"),
            base.UserMessage(
                """Please help me with:
                1. Creating an effective regex pattern for these examples
                2. Explaining how the pattern works
                3. Highlighting any assumptions or limitations
                4. Testing the pattern against the examples
                5. Suggestions for pattern refinement if needed"""
            ),
            base.AssistantMessage(
                """I'll help you create an effective regex pattern:
                
                ## Pattern Analysis
                
                Let me analyze what we need to match and exclude:
                
                (I'll analyze the common patterns in match examples)
                
                ## Recommended Regex
                
                Based on your examples, here's the regex pattern:
                
                ```regex
                (I'll provide the regex pattern here)
                ```
                
                ## Pattern Explanation
                
                Here's how this pattern works:
                
                (I'll explain each component of the pattern)
                
                ## Pattern Testing
                
                Testing against your examples:
                
                (I'll show how the pattern matches/doesn't match)
                
                ## Refinement Suggestions
                
                To make this pattern more robust:
                
                (I'll suggest potential improvements)"""
            ),
        ]
    
    @mcp.prompt()
    def semantic_search_assistance(
        corpus_description: str,
        query_examples: str,
        search_goal: str = "find relevant content"
    ) -> list[base.Message]:
        """Generate a prompt for effective semantic searching.
        
        This prompt helps users craft effective semantic search strategies
        to find relevant content beyond exact keyword matching.
        
        Args:
            corpus_description: Description of the content to search
            query_examples: Examples of search queries
            search_goal: Goal of the semantic search
            
        Returns:
            List of messages forming the prompt
        """
        return [
            base.SystemMessage(
                """You are an expert in semantic search and information retrieval with deep
                knowledge of embedding models, similarity measures, and search optimization.
                You help users craft effective search strategies that find conceptually
                relevant content beyond simple keyword matching."""
            ),
            base.UserMessage(f"I need to {search_goal} using semantic search in this content:"),
            base.UserMessage(f"Content description: {corpus_description}"),
            base.UserMessage(f"Example queries: {query_examples}"),
            base.UserMessage(
                """Please help me with:
                1. Crafting effective semantic search queries
                2. Choosing appropriate search parameters
                3. Strategies for improving search relevance
                4. Interpreting search results effectively
                5. Refining searches to narrow or broaden results"""
            ),
            base.AssistantMessage(
                """I'll help you develop effective semantic search strategies:
                
                ## Content Analysis
                
                First, let me analyze your content domain:
                
                (I'll analyze the content for search considerations)
                
                ## Query Optimization
                
                For your example queries, I recommend these modifications:
                
                (I'll suggest improved query formulations)
                
                ## Search Parameters
                
                I recommend these search parameters:
                
                (I'll suggest similarity thresholds, model choice, etc.)
                
                ## Result Interpretation
                
                When interpreting results, focus on:
                
                (I'll provide guidance on assessing relevance)
                
                ## Refinement Strategies
                
                To improve your results further:
                
                (I'll suggest iterative refinement techniques)"""
            ),
        ]
    
    @mcp.prompt()
    def filename_analysis(
        filename_examples: str,
        analysis_goal: str,
        suggested_patterns: Optional[str] = None
    ) -> list[base.Message]:
        """Generate a prompt for analyzing filename patterns.
        
        This prompt helps users analyze filename collections to identify
        patterns, suggest organization schemes, or extract information.
        
        Args:
            filename_examples: Examples of filenames to analyze
            analysis_goal: Goal of the filename analysis
            suggested_patterns: Patterns user suspects might exist (optional)
            
        Returns:
            List of messages forming the prompt
        """
        pattern_hint = f"\n\nI suspect these patterns might exist: {suggested_patterns}" if suggested_patterns else ""
        
        return [
            base.SystemMessage(
                """You are an expert in file organization and naming convention analysis with deep
                knowledge of pattern recognition, metadata extraction, and naming best practices.
                You help users understand and optimize their file organization systems."""
            ),
            base.UserMessage(f"I need to {analysis_goal} for these filenames:{pattern_hint}"),
            base.UserMessage(f"```\n{filename_examples}\n```"),
            base.UserMessage(
                """Please help me with:
                1. Identifying patterns in these filenames
                2. Extracting consistent components or metadata
                3. Analyzing strengths and weaknesses of the current naming scheme
                4. Suggesting improvements or standardization approaches
                5. Proposing tools or methods to work with these filenames effectively"""
            ),
            base.AssistantMessage(
                """I'll help you analyze these filenames:
                
                ## Pattern Identification
                
                I've identified these patterns in your filenames:
                
                (I'll list and describe filename patterns)
                
                ## Component Analysis
                
                The filenames contain these key components:
                
                (I'll identify filename parts and their meaning)
                
                ## Naming Scheme Evaluation
                
                Strengths and weaknesses of the current naming approach:
                
                (I'll assess the effectiveness of the naming scheme)
                
                ## Improvement Suggestions
                
                Consider these improvements to your naming convention:
                
                (I'll suggest standardization or improvements)
                
                ## Implementation Approach
                
                To work effectively with these filenames:
                
                (I'll suggest tools or techniques for processing)"""
            ),
        ]
    
    @mcp.prompt()
    def structure_search_assistance(
        structure_description: str,
        search_criteria: str,
        structure_type: Optional[str] = None
    ) -> list[base.Message]:
        """Generate a prompt for searching within structured data.
        
        This prompt helps users effectively search within structured data
        like JSON, XML, code, or other hierarchical formats.
        
        Args:
            structure_description: Description of the structured data
            search_criteria: What to search for in the structure
            structure_type: Type of structure (JSON, XML, code, etc.)
            
        Returns:
            List of messages forming the prompt
        """
        structure_info = f" in {structure_type}" if structure_type else ""
        
        return [
            base.SystemMessage(
                """You are an expert in structured data navigation and query with deep knowledge
                of path expressions, query languages, and hierarchical data traversal. You help
                users effectively search and extract information from complex structured data."""
            ),
            base.UserMessage(f"I need to search for {search_criteria}{structure_info} structured data:"),
            base.UserMessage(structure_description),
            base.UserMessage(
                """Please help me with:
                1. The most effective query approach for this structure
                2. How to navigate to the relevant parts of the structure
                3. Query expressions or patterns to use
                4. How to handle variations or edge cases
                5. Interpreting and using the search results"""
            ),
            base.AssistantMessage(
                f"""I'll help you search this structured data:
                
                ## Structure Analysis
                
                First, let me analyze the structure:
                
                (I'll analyze the structure type and organization)
                
                ## Search Approach
                
                To find {search_criteria}, I recommend:
                
                (I'll suggest an appropriate search approach)
                
                ## Query Expressions
                
                Use these specific queries or expressions:
                
                (I'll provide query syntax appropriate for the structure)
                
                ## Edge Case Handling
                
                Be aware of these potential variations:
                
                (I'll identify potential edge cases)
                
                ## Result Utilization
                
                Once you've found the results:
                
                (I'll suggest ways to use the search results)"""
            ),
        ] 