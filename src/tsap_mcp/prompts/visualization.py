"""
Visualization prompts for TSAP MCP Server.

This module provides MCP prompt implementations for visualization operations
and related tasks.
"""
from typing import Optional, List, Dict, Any
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base


def register_visualization_prompts(mcp: FastMCP) -> None:
    """Register all visualization-related prompts with the MCP server.
    
    Args:
        mcp: FastMCP server instance
    """
    
    @mcp.prompt()
    def chart_generator(
        data_description: str, 
        chart_type: Optional[str] = None, 
        goal: Optional[str] = None
    ) -> list[base.Message]:
        """Generate a prompt for choosing and creating appropriate charts.
        
        This prompt helps users select and generate appropriate chart types
        for different kinds of data and visualization goals.
        
        Args:
            data_description: Description of the data to visualize
            chart_type: Optional specific chart type to use (if known)
            goal: Purpose or goal of the visualization
            
        Returns:
            List of messages forming the prompt
        """
        chart_specifics = f" using a {chart_type} chart" if chart_type else ""
        goal_statement = f" with the goal of {goal}" if goal else ""
        
        return [
            base.SystemMessage(
                """You are an expert data visualization specialist with deep knowledge of chart types,
                data presentation principles, and visualization best practices. You help users determine
                the most effective ways to visualize their data for maximum clarity and impact."""
            ),
            base.UserMessage(f"I need to visualize this data{chart_specifics}{goal_statement}:"),
            base.UserMessage(data_description),
            base.UserMessage(
                """Please help me with:
                1. The most appropriate chart type(s) for this data
                2. The key elements to include in the visualization
                3. Suggestions for layout, color scheme, and formatting
                4. Best practices for this particular visualization"""
            ),
            base.AssistantMessage(
                """I'll help you create an effective visualization for your data:
                
                ## Data Analysis
                
                First, let me understand your data and its characteristics:
                
                (I'll analyze the structure, dimensions, and relationships in your data)
                
                ## Recommended Visualization
                
                Based on your data and goals, I recommend:
                
                (I'll recommend specific chart types and explain why they're appropriate)
                
                ## Design Recommendations
                
                For this visualization, consider these design elements:
                
                (I'll suggest specific design elements like axis labels, legends, colors, etc.)
                
                ## Implementation Guidance
                
                To create this visualization effectively:
                
                (I'll provide specific guidance on implementing the visualization)"""
            ),
        ]
    
    @mcp.prompt()
    def graph_visualization(
        graph_description: str, 
        node_types: Optional[List[str]] = None, 
        layout_type: Optional[str] = None
    ) -> list[base.Message]:
        """Generate a prompt for visualizing graph and network data.
        
        This prompt helps users create effective graph visualizations for
        showing relationships, networks, hierarchies, and flows.
        
        Args:
            graph_description: Description of the graph data to visualize
            node_types: Types of nodes in the graph (if known)
            layout_type: Preferred layout type (force, hierarchical, etc.)
            
        Returns:
            List of messages forming the prompt
        """
        nodes_info = f" with node types: {', '.join(node_types)}" if node_types else ""
        layout_info = f" using a {layout_type} layout" if layout_type else ""
        
        return [
            base.SystemMessage(
                """You are an expert in graph theory and network visualization with deep knowledge of
                graph layout algorithms, interactive visualizations, and relationship representations.
                You help users create clear and insightful visualizations of complex interconnected data."""
            ),
            base.UserMessage(f"I need to visualize this graph data{nodes_info}{layout_info}:"),
            base.UserMessage(graph_description),
            base.UserMessage(
                """Please help me with:
                1. The most appropriate graph visualization technique
                2. Node and edge representation recommendations
                3. Layout algorithm selection
                4. Visual encoding strategies for attributes
                5. Interaction and exploration features to include"""
            ),
            base.AssistantMessage(
                """I'll help you create an effective graph visualization:
                
                ## Graph Structure Analysis
                
                Let me analyze your graph data:
                
                (I'll examine the graph's structure, scale, and key properties)
                
                ## Visualization Approach
                
                Based on your graph, I recommend:
                
                (I'll recommend specific visualization approaches)
                
                ## Node & Edge Design
                
                For optimal clarity, consider these design elements:
                
                (I'll suggest node and edge visual encodings)
                
                ## Layout Strategy
                
                The most effective layout for your graph would be:
                
                (I'll recommend layout algorithms and explain why)
                
                ## Interactive Features
                
                To enhance exploration, consider adding:
                
                (I'll suggest interactive features appropriate for your graph)"""
            ),
        ]
    
    @mcp.prompt()
    def code_visualization(
        code_description: str, 
        language: Optional[str] = None, 
        visualization_goal: str = "understand structure"
    ) -> list[base.Message]:
        """Generate a prompt for visualizing code structure and relationships.
        
        This prompt helps users create visualizations to understand code
        structure, dependencies, call flows, and architecture.
        
        Args:
            code_description: Description of the code to visualize
            language: Programming language of the code
            visualization_goal: What the user wants to understand about the code
            
        Returns:
            List of messages forming the prompt
        """
        language_info = f" in {language}" if language else ""
        
        return [
            base.SystemMessage(
                """You are an expert in software visualization and architecture diagrams with deep knowledge
                of code structure representation, dependency visualization, and software metrics visualization.
                You help users create clear visual models of complex codebases to improve understanding."""
            ),
            base.UserMessage(f"I need to visualize code{language_info} to help {visualization_goal}:"),
            base.UserMessage(code_description),
            base.UserMessage(
                """Please help me with:
                1. The most appropriate visualization technique for this goal
                2. Key elements of the code to highlight
                3. Level of abstraction to use
                4. Notation or diagramming approach to follow
                5. Tools or libraries that could help create this visualization"""
            ),
            base.AssistantMessage(
                """I'll help you create an effective code visualization:
                
                ## Code Analysis
                
                Based on your description, I understand you're working with:
                
                (I'll analyze the code characteristics and structure)
                
                ## Recommended Visualization Approach
                
                For your goal of understanding {visualization_goal}, I recommend:
                
                (I'll recommend specific visualization approaches)
                
                ## Elements to Highlight
                
                Focus on these key elements in your visualization:
                
                (I'll suggest specific code elements to emphasize)
                
                ## Diagramming Approach
                
                Consider using this notation/approach:
                
                (I'll recommend appropriate notation systems or diagramming techniques)
                
                ## Implementation Suggestions
                
                Here are some tools that can help create this visualization:
                
                (I'll suggest specific tools or libraries)"""
            ),
        ]
    
    @mcp.prompt()
    def data_storytelling(
        data_findings: str, 
        audience: str = "technical", 
        visualization_types: Optional[List[str]] = None
    ) -> list[base.Message]:
        """Generate a prompt for creating data visualizations that tell a story.
        
        This prompt helps users craft visualizations that effectively
        communicate narratives and insights from data.
        
        Args:
            data_findings: Description of the data findings to communicate
            audience: Target audience for the visualization
            visualization_types: Types of visualizations to consider
            
        Returns:
            List of messages forming the prompt
        """
        viz_options = f" using {', '.join(visualization_types)}" if visualization_types else ""
        
        return [
            base.SystemMessage(
                f"""You are an expert in data storytelling and visual communication with deep experience
                creating compelling visual narratives for {audience} audiences. You help users transform
                raw data findings into clear, impactful visual stories that drive understanding and action."""
            ),
            base.UserMessage(f"I need to create visualizations{viz_options} to tell a story about these findings:"),
            base.UserMessage(data_findings),
            base.UserMessage(f"My target audience is: {audience}"),
            base.UserMessage(
                """Please help me with:
                1. A narrative structure for my data story
                2. The sequence of visualizations to use
                3. Key elements to emphasize visually
                4. Design principles for my target audience
                5. How to ensure my visualizations support my key messages"""
            ),
            base.AssistantMessage(
                f"""I'll help you craft compelling visual storytelling for your {audience} audience:
                
                ## Narrative Analysis
                
                Based on your findings, here's the story emerging from your data:
                
                (I'll identify the core narrative in your findings)
                
                ## Visual Storytelling Approach
                
                To effectively communicate with your {audience} audience, I recommend:
                
                (I'll suggest a specific visual storytelling approach)
                
                ## Visualization Sequence
                
                Structure your story with these visualizations in sequence:
                
                (I'll recommend a logical flow of specific visualizations)
                
                ## Design Focus
                
                For each visualization, emphasize these elements:
                
                (I'll suggest specific visual elements to emphasize)
                
                ## Connecting to Key Messages
                
                To ensure your visualizations support your narrative:
                
                (I'll provide specific guidance on connecting visuals to messages)"""
            ),
        ]
    
    @mcp.prompt()
    def visualization_critique(
        visualization_description: str,
        goal: str,
        audience: Optional[str] = None
    ) -> list[base.Message]:
        """Generate a prompt for critiquing and improving visualizations.
        
        This prompt helps users assess existing visualizations and identify
        ways to improve their clarity, impact, and effectiveness.
        
        Args:
            visualization_description: Description of the current visualization
            goal: What the visualization is trying to accomplish
            audience: Target audience for the visualization
            
        Returns:
            List of messages forming the prompt
        """
        audience_info = f" for a {audience} audience" if audience else ""
        
        return [
            base.SystemMessage(
                """You are an expert in data visualization critique with deep knowledge of perception,
                cognitive principles, and design best practices. You provide constructive feedback
                to help users improve their visualizations for maximum clarity and impact."""
            ),
            base.UserMessage(f"Please critique this visualization that aims to {goal}{audience_info}:"),
            base.UserMessage(visualization_description),
            base.UserMessage(
                """Please provide:
                1. An assessment of how well the visualization achieves its goal
                2. Identification of strengths in the current approach
                3. Specific areas that could be improved
                4. Alternative approaches to consider
                5. Recommendations prioritized by potential impact"""
            ),
            base.AssistantMessage(
                """I'll provide a constructive critique of your visualization:
                
                ## Goal Alignment
                
                Analyzing how well your visualization achieves its goal of {goal}:
                
                (I'll assess how effectively the visualization meets its stated purpose)
                
                ## Strengths
                
                Your visualization has these effective elements:
                
                (I'll identify what's working well)
                
                ## Improvement Opportunities
                
                Consider enhancing these aspects:
                
                (I'll suggest specific improvements)
                
                ## Alternative Approaches
                
                You might also consider these different approaches:
                
                (I'll suggest alternative visualization strategies)
                
                ## Prioritized Recommendations
                
                For maximum impact, focus on these changes first:
                
                (I'll prioritize the most important improvements)"""
            ),
        ] 