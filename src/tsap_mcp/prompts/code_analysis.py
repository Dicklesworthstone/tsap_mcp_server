"""
Code analysis prompts for TSAP MCP Server.

This module provides MCP prompt implementations for code analysis
and related tasks.
"""
from typing import Optional
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base


def register_code_analysis_prompts(mcp: FastMCP) -> None:
    """Register all code analysis-related prompts with the MCP server.
    
    Args:
        mcp: FastMCP server instance
    """
    
    @mcp.prompt()
    def code_review(code: str, language: Optional[str] = None) -> list[base.Message]:
        """Generate a prompt for reviewing code.
        
        This prompt helps analyze code quality, readability, and best practices,
        providing constructive feedback and suggestions for improvement.
        
        Args:
            code: The code to review
            language: Programming language (optional, will be inferred if not provided)
            
        Returns:
            List of messages forming the prompt
        """
        language_info = f" {language}" if language else ""
        
        return [
            base.SystemMessage(
                """You are an expert code reviewer with deep knowledge of software engineering principles, design patterns, 
                and best practices across many programming languages. You provide thorough, constructive, and actionable feedback 
                to help developers improve their code quality, maintainability, and performance."""
            ),
            base.UserMessage(f"Please review the following{language_info} code:"),
            base.UserMessage(f"```{language if language else ''}\n{code}\n```"),
            base.UserMessage(
                """Focus your analysis on:
                1. Code quality and readability
                2. Potential bugs or edge cases
                3. Performance considerations
                4. Security vulnerabilities
                5. Best practices and design patterns
                6. Architectural improvements
                
                Please be specific and provide actionable suggestions where possible."""
            ),
            base.AssistantMessage(
                """I'll review this code thoroughly. Let me analyze it section by section:
                
                ## Summary
                
                (I'll first provide a brief overview of the code and its apparent purpose)
                
                ## Strengths
                
                (I'll highlight what's done well in the code)
                
                ## Areas for Improvement
                
                (I'll discuss issues categorized by type, with specific line references and suggestions)
                
                Let me begin my detailed analysis..."""
            ),
        ]
    
    @mcp.prompt()
    def bug_analysis(code: str, error_message: Optional[str] = None, language: Optional[str] = None) -> list[base.Message]:
        """Generate a prompt for analyzing and fixing bugs in code.
        
        This prompt helps identify and resolve bugs, providing explanations
        of the root cause and suggested fixes.
        
        Args:
            code: The buggy code
            error_message: Error message or exception details (optional)
            language: Programming language (optional, will be inferred if not provided)
            
        Returns:
            List of messages forming the prompt
        """
        language_info = f" {language}" if language else ""
        error_context = f"\nError message:\n```\n{error_message}\n```" if error_message else ""
        
        return [
            base.SystemMessage(
                """You are an expert debugging assistant with deep experience troubleshooting complex software issues.
                You can identify bugs, analyze their root causes, and suggest efficient fixes, even with limited information.
                You understand common programming errors, edge cases, and language-specific pitfalls."""
            ),
            base.UserMessage(f"Please help me debug this{language_info} code:{error_context}"),
            base.UserMessage(f"```{language if language else ''}\n{code}\n```"),
            base.UserMessage(
                """Please:
                1. Identify the bug(s) in the code
                2. Explain the root cause of each issue
                3. Provide a working fix for each problem
                4. Suggest best practices to prevent similar issues"""
            ),
            base.AssistantMessage(
                """I'll help you debug this code. Let me analyze the issues:
                
                ## Bug Identification
                
                (I'll identify the bugs or issues in the code)
                
                ## Root Cause Analysis
                
                (I'll explain what's causing each issue)
                
                ## Fixes
                
                (I'll provide corrected code for each issue)
                
                ## Prevention
                
                (I'll suggest best practices to avoid these issues in the future)
                
                Now, let me start my detailed analysis..."""
            ),
        ]
    
    @mcp.prompt()
    def code_explanation(code: str, language: Optional[str] = None, complexity_level: str = "intermediate") -> list[base.Message]:
        """Generate a prompt for explaining code functionality.
        
        This prompt helps understand complex code by providing clear explanations
        of how it works and what it does.
        
        Args:
            code: The code to explain
            language: Programming language (optional, will be inferred if not provided)
            complexity_level: Level of detail (basic, intermediate, advanced)
            
        Returns:
            List of messages forming the prompt
        """
        language_info = f" {language}" if language else ""
        
        return [
            base.SystemMessage(
                f"""You are an expert programming educator who excels at explaining code clearly at various levels of detail.
                You're currently explaining code to someone at the {complexity_level} level of expertise.
                You break down complex concepts into understandable components and provide meaningful context."""
            ),
            base.UserMessage(f"Please explain how this{language_info} code works:"),
            base.UserMessage(f"```{language if language else ''}\n{code}\n```"),
            base.UserMessage(
                """Please include:
                1. A high-level overview of what the code does
                2. Line-by-line or block-by-block explanation
                3. Key functions, variables, or classes and their purpose
                4. The overall logic and flow of execution
                5. Any important algorithms, patterns, or techniques used"""
            ),
            base.AssistantMessage(
                """I'll explain this code clearly. Let me break it down:
                
                ## Overview
                
                (I'll provide a high-level summary of the code's purpose)
                
                ## Detailed Explanation
                
                (I'll explain the code section by section)
                
                ## Key Components
                
                (I'll highlight important functions, variables, or classes)
                
                ## Execution Flow
                
                (I'll describe how the code executes)
                
                Let me start with the detailed analysis..."""
            ),
        ]
    
    @mcp.prompt()
    def code_optimization(code: str, language: Optional[str] = None, optimization_goals: str = "performance") -> list[base.Message]:
        """Generate a prompt for optimizing code.
        
        This prompt helps improve code efficiency, identifying optimization 
        opportunities and suggesting improvements.
        
        Args:
            code: The code to optimize
            language: Programming language (optional, will be inferred if not provided)
            optimization_goals: What to optimize for (performance, memory, readability)
            
        Returns:
            List of messages forming the prompt
        """
        language_info = f" {language}" if language else ""
        
        return [
            base.SystemMessage(
                f"""You are an expert code optimizer specializing in improving {optimization_goals}.
                You have deep knowledge of algorithmic efficiency, language-specific optimizations,
                and can identify bottlenecks and optimization opportunities in code."""
            ),
            base.UserMessage(f"Please optimize this{language_info} code for {optimization_goals}:"),
            base.UserMessage(f"```{language if language else ''}\n{code}\n```"),
            base.UserMessage(
                f"""Please:
                1. Identify inefficient parts of the code related to {optimization_goals}
                2. Explain why they are inefficient
                3. Provide optimized alternatives
                4. Estimate the improvement in {optimization_goals}
                5. Consider any tradeoffs your optimizations might introduce"""
            ),
            base.AssistantMessage(
                f"""I'll help optimize this code for {optimization_goals}. Here's my analysis:
                
                ## Inefficiencies
                
                (I'll identify inefficient parts of the code)
                
                ## Optimized Solution
                
                (I'll provide an optimized version of the code)
                
                ## Expected Improvements
                
                (I'll explain the expected improvements in {optimization_goals})
                
                ## Tradeoffs
                
                (I'll discuss any tradeoffs introduced by the optimizations)
                
                Let me begin with identifying the inefficiencies..."""
            ),
        ]
    
    @mcp.prompt()
    def dependency_analysis(code: str, language: Optional[str] = None) -> list[base.Message]:
        """Generate a prompt for analyzing code dependencies.
        
        This prompt helps understand the dependencies within code, identifying
        external libraries, internal modules, and suggesting improvements.
        
        Args:
            code: The code to analyze
            language: Programming language (optional, will be inferred if not provided)
            
        Returns:
            List of messages forming the prompt
        """
        language_info = f" {language}" if language else ""
        
        return [
            base.SystemMessage(
                """You are an expert in software architecture and dependency management.
                You excel at analyzing how code components depend on each other and on external libraries.
                You can identify dependency issues, vulnerabilities, and opportunities for improvement."""
            ),
            base.UserMessage(f"Please analyze the dependencies in this{language_info} code:"),
            base.UserMessage(f"```{language if language else ''}\n{code}\n```"),
            base.UserMessage(
                """Please provide:
                1. A list of all external libraries/packages used
                2. A list of internal modules/components referenced
                3. Identification of any missing, outdated, or unnecessary dependencies
                4. Suggestions for dependency optimization or security improvements
                5. Analysis of the dependency structure and coupling between components"""
            ),
            base.AssistantMessage(
                """I'll analyze the dependencies in this code:
                
                ## External Dependencies
                
                (I'll list all external libraries/packages)
                
                ## Internal Dependencies
                
                (I'll identify internal modules and components)
                
                ## Dependency Issues
                
                (I'll highlight any problematic dependencies)
                
                ## Recommendations
                
                (I'll suggest improvements to the dependency structure)
                
                Let me start by identifying all the dependencies..."""
            ),
        ] 