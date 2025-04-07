"""
Search prompts for TSAP MCP Server.

This module provides MCP prompt implementations for search operations
and related tasks.
"""
from typing import Optional, List
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base


def register_search_prompts(mcp: FastMCP) -> None:
    """Register all search-related prompts with the MCP server.
    
    Args:
        mcp: FastMCP server instance
    """
    
    @mcp.prompt()
    def search_help(file_types: Optional[List[str]] = None, search_type: str = "general") -> list[base.Message]:
        """Generate a prompt for getting search assistance.
        
        This prompt helps users understand how to effectively search through code
        or documents, with tailored guidance for different file types and search types.
        
        Args:
            file_types: List of file types to search (e.g., ["py", "js"])
            search_type: Type of search (general, code, documentation, pattern)
            
        Returns:
            List of messages forming the prompt
        """
        file_types_str = ", ".join(file_types) if file_types else "all files"
        
        return [
            base.SystemMessage(
                f"""You are an expert search assistant specializing in helping users find information
                in {file_types_str} through {search_type} searches. You provide guidance on crafting
                effective search queries and interpreting results."""
            ),
            base.UserMessage(f"I need help searching through {file_types_str} using {search_type} search techniques."),
            base.UserMessage("What are the most effective strategies, and what common pitfalls should I avoid?"),
            base.AssistantMessage(
                f"""I'll help you search effectively through {file_types_str} using {search_type} search techniques.
                
                Let me provide some guidance:"""
            ),
        ]
    
    @mcp.prompt()
    def regex_pattern_builder(
        target_text: str, 
        file_type: Optional[str] = None,
        complexity: str = "moderate"
    ) -> list[base.Message]:
        """Generate a prompt for building regex search patterns.
        
        This prompt helps users create effective regular expression patterns
        for searching text, with explanations and examples.
        
        Args:
            target_text: Description of the text to search for
            file_type: Type of file to search in (optional)
            complexity: Desired regex complexity (simple, moderate, complex)
            
        Returns:
            List of messages forming the prompt
        """
        file_context = f" in {file_type} files" if file_type else ""
        
        return [
            base.SystemMessage(
                f"""You are an expert in regular expressions with deep knowledge of pattern matching
                across different programming languages and text formats. You specialize in creating
                {complexity} regex patterns that are efficient, accurate, and maintainable."""
            ),
            base.UserMessage(f"I need to create a regex pattern to find {target_text}{file_context}."),
            base.UserMessage(f"Please create a {complexity} regex pattern with explanation."),
            base.AssistantMessage(
                """I'll create an effective regex pattern for your search need. Let me break this down:
                
                ## Understanding the Target
                
                First, let me understand exactly what you're trying to match.
                
                ## Pattern Development
                
                Based on your requirements, here's the regex pattern I recommend:
                
                ```regex
                (I'll provide the pattern here)
                ```
                
                ## Explanation
                
                Let me explain how this pattern works:
                
                - (I'll explain each component)
                
                ## Examples
                
                This pattern will match:
                - (I'll provide example matches)
                
                But won't match:
                - (I'll provide non-matching examples)
                
                ## Usage Tips
                
                When using this pattern:
                - (I'll provide usage recommendations)"""
            ),
        ]
    
    @mcp.prompt()
    def pattern_discovery(
        sample_text: str,
        extraction_goal: str,
        file_type: Optional[str] = None
    ) -> list[base.Message]:
        """Generate a prompt for discovering patterns in text.
        
        This prompt helps users identify patterns in text data for extraction
        or analysis purposes.
        
        Args:
            sample_text: Example text containing the patterns
            extraction_goal: What the user wants to extract or identify
            file_type: Type of file the text comes from (optional)
            
        Returns:
            List of messages forming the prompt
        """
        file_context = f" from {file_type} files" if file_type else ""
        
        return [
            base.SystemMessage(
                f"""You are an expert in pattern analysis and text processing with deep knowledge
                of natural language processing, regular expressions, and data extraction techniques.
                You excel at identifying patterns in text and creating extraction strategies."""
            ),
            base.UserMessage(f"I need to extract {extraction_goal}{file_context}. Here's a sample of the text:"),
            base.UserMessage(f"```\n{sample_text}\n```"),
            base.UserMessage("Please help me identify patterns and create an extraction strategy."),
            base.AssistantMessage(
                """I'll analyze this text and help you develop a pattern extraction strategy:
                
                ## Pattern Analysis
                
                Looking at your sample text, I can identify these patterns:
                
                - (I'll identify key patterns)
                
                ## Extraction Strategy
                
                Based on the patterns I've identified, here's my recommended approach:
                
                1. (I'll outline the extraction strategy)
                
                ## Implementation
                
                Here's how you can implement this extraction:
                
                ```
                (I'll provide implementation details based on the context)
                ```
                
                ## Validation
                
                To ensure accuracy, validate your extraction with these checks:
                
                - (I'll suggest validation methods)"""
            ),
        ]
    
    @mcp.prompt()
    def search_strategy(
        search_objective: str,
        search_scope: str,
        priority: str = "accuracy"
    ) -> list[base.Message]:
        """Generate a prompt for developing a search strategy.
        
        This prompt helps users develop comprehensive search strategies
        for complex search objectives across large codebases or document sets.
        
        Args:
            search_objective: What the user is trying to find
            search_scope: Where to search (e.g., "entire codebase", "docs directory")
            priority: Search priority (accuracy, speed, comprehensiveness)
            
        Returns:
            List of messages forming the prompt
        """
        return [
            base.SystemMessage(
                f"""You are an expert in information retrieval and search optimization with deep
                understanding of how to efficiently locate information in large codebases and
                document collections. You excel at creating search strategies that prioritize
                {priority} while minimizing search time and effort."""
            ),
            base.UserMessage(f"I need to find {search_objective} within {search_scope}."),
            base.UserMessage(f"Please develop a search strategy that prioritizes {priority}."),
            base.AssistantMessage(
                f"""I'll develop a comprehensive search strategy to help you find {search_objective} in {search_scope},
                with a focus on {priority}:
                
                ## Search Objective Analysis
                
                First, let me break down what you're looking for:
                
                - (I'll analyze the search objective)
                
                ## Recommended Search Strategy
                
                Here's my recommended approach:
                
                1. **Initial Broad Search**
                   - (I'll suggest initial queries)
                
                2. **Refinement**
                   - (I'll suggest refinement techniques)
                
                3. **Targeted Deep Dive**
                   - (I'll suggest focused search areas)
                
                4. **Verification**
                   - (I'll suggest verification methods)
                
                ## Search Tools and Commands
                
                I recommend using these specific tools and commands:
                
                - (I'll suggest appropriate tools)
                
                ## Expected Results
                
                Here's what you should expect to find:
                
                - (I'll describe expected results)"""
            ),
        ]

    @mcp.prompt()
    def code_search(pattern: str, path: str = ".") -> list[base.Message]:
        """Create a prompt for searching code with a pattern.
        
        This prompt helps the LLM search code with a specific pattern
        and analyze the results.
        
        Args:
            pattern: Regex pattern to search for
            path: Path to search in
            
        Returns:
            List of messages forming the prompt
        """
        return [
            base.UserMessage(f"I need to search for this pattern in the code: `{pattern}`"),
            base.UserMessage(f"Please search in the path: {path}"),
            base.UserMessage("For each match you find, explain what the code is doing in that context."),
            base.AssistantMessage("I'll search for that pattern and analyze each match for you.")
        ]
    
    @mcp.prompt()
    def pattern_exploration(pattern: str, examples: int = 3) -> str:
        """Create a prompt for exploring pattern usage in the codebase.
        
        This prompt helps the LLM understand how a pattern is used throughout
        the codebase by examining examples.
        
        Args:
            pattern: Pattern to explore
            examples: Number of examples to request
            
        Returns:
            Prompt text
        """
        return f"""
I'd like to explore how the pattern `{pattern}` is used in the codebase.

Please:
1. Find {examples} distinct examples of this pattern being used
2. For each example, explain:
   - The context/file where it's used
   - The purpose of the pattern in that context
   - Any variations you observe in how it's implemented

Then, provide a summary of your findings, highlighting any patterns, best practices, or potential issues
you noticed in how this pattern is used across the codebase.
"""
    
    @mcp.prompt()
    def find_all_occurrences(text: str, file_types: str = None) -> list[base.Message]:
        """Create a prompt to find all occurrences of text in the codebase.
        
        This prompt helps the LLM search for specific text and return all occurrences.
        
        Args:
            text: Text to search for
            file_types: Optional file types to limit search (e.g., "*.py,*.js")
            
        Returns:
            List of messages forming the prompt
        """
        messages = [
            base.UserMessage(f"I need to find all occurrences of the following text in the codebase:"),
            base.UserMessage(f"```\n{text}\n```"),
        ]
        
        if file_types:
            messages.append(base.UserMessage(f"Please limit the search to these file types: {file_types}"))
        
        messages.append(base.UserMessage("List all the files and line numbers where this text occurs."))
        messages.append(base.AssistantMessage("I'll search for all occurrences and list them for you."))
        
        return messages 