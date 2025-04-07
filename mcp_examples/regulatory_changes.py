#!/usr/bin/env python3
"""
Regulatory Changes Analysis (MCP Tools Version)

This script demonstrates how to analyze regulatory documents to identify changes
between versions, extract requirements, and track regulatory compliance using
the TSAP MCP tools.
"""
import asyncio
import os
import re
import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule

# Import proper MCP client modules
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

console = Console()

# Example documents
SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")
os.makedirs(SAMPLE_DIR, exist_ok=True)

# Sample regulatory document file paths
REGULATION_V1 = os.path.join(SAMPLE_DIR, "regulation_v1.md")
REGULATION_V2 = os.path.join(SAMPLE_DIR, "regulation_v2.md")

# Patterns for regulatory analysis
REQUIREMENT_PATTERNS = [
    {
        "pattern": r"\b(shall|must|required|requirement)\b",
        "description": "Mandatory requirement",
        "priority": "high",
        "type": "obligation"
    },
    {
        "pattern": r"\b(should|recommended|recommendation)\b",
        "description": "Recommended practice",
        "priority": "medium",
        "type": "recommendation"
    },
    {
        "pattern": r"\b(may|optional|permitted|allowable)\b",
        "description": "Optional permission",
        "priority": "low",
        "type": "permission"
    },
    {
        "pattern": r"\b(prohibited|disallowed|not permitted|shall not|must not)\b",
        "description": "Prohibition",
        "priority": "high",
        "type": "prohibition"
    }
]

async def create_sample_regulations():
    """Create sample regulatory document files for demonstration."""
    console.print("[bold]Creating sample regulatory documents...[/bold]")
    
    # Create first version of the regulation
    regulation_v1_content = """# Data Protection Regulation v1.0
    
## Section 1: General Provisions

### 1.1 Purpose
This regulation establishes requirements for the protection of personal data.

### 1.2 Scope
This regulation applies to all organizations processing personal data.

## Section 2: Data Protection Requirements

### 2.1 Data Collection
Organizations shall obtain explicit consent before collecting personal data.

### 2.2 Data Storage
Personal data should be stored securely with encryption.

### 2.3 Data Retention
Organizations may retain personal data for up to 5 years.

### 2.4 Data Sharing
Sharing personal data with third parties is prohibited without explicit consent.

## Section 3: Compliance

### 3.1 Compliance Timeline
Organizations must comply with this regulation within 12 months of publication.

### 3.2 Penalties
Non-compliance may result in penalties up to $10,000.
"""
    
    # Create updated version of the regulation with changes
    regulation_v2_content = """# Data Protection Regulation v2.0
    
## Section 1: General Provisions

### 1.1 Purpose
This regulation establishes requirements for the protection of personal data.

### 1.2 Scope
This regulation applies to all organizations processing personal data of citizens.

## Section 2: Data Protection Requirements

### 2.1 Data Collection
Organizations shall obtain explicit consent before collecting personal data.
Organizations shall maintain records of all consent obtained.

### 2.2 Data Storage
Personal data shall be stored securely with encryption.
Encryption keys must be rotated every 90 days.

### 2.3 Data Retention
Organizations may retain personal data for up to 3 years.
After retention period, data must be permanently deleted.

### 2.4 Data Sharing
Sharing personal data with third parties is prohibited without explicit consent.
All data sharing activities must be logged and reported quarterly.

## Section 3: Compliance

### 3.1 Compliance Timeline
Organizations must comply with this regulation within 6 months of publication.

### 3.2 Penalties
Non-compliance may result in penalties up to $50,000.

### 3.3 Auditing
Organizations shall perform annual data protection audits.
"""
    
    # Write files
    with open(REGULATION_V1, "w") as f:
        f.write(regulation_v1_content)
    
    with open(REGULATION_V2, "w") as f:
        f.write(regulation_v2_content)
    
    console.print("[green]Sample regulatory documents created successfully![/green]")
    return REGULATION_V1, REGULATION_V2

async def analyze_regulatory_changes():
    """Demonstrate regulatory document analysis using MCP tools."""
    console.print(Panel.fit(
        "[bold blue]TSAP MCP Regulatory Changes Analysis[/bold blue]",
        subtitle="Analyzing changes in regulatory requirements between document versions"
    ))
    
    # Path to the proxy script
    proxy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_proxy.py")
    console.print(f"Using proxy script: {proxy_path}")
    
    # Configure server parameters for stdio connection to our proxy
    server_params = StdioServerParameters(
        command="python",
        args=[proxy_path],
        env={"MCP_DEBUG": "0"}  # Disable debug logging by default
    )
    
    try:
        # Connect to the MCP server via proxy
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()
                
                # Get the list of available tools
                tools_result = await session.list_tools()
                tools = tools_result.tools
                
                # Find required tools
                diff_tool = next((t for t in tools if t.name == "diff"), None)
                search_tool = next((t for t in tools if t.name == "search"), None)
                semantic_search_tool = next((t for t in tools if t.name == "semantic_search"), None)
                
                if not diff_tool or not search_tool:
                    console.print("[bold red]Error: Required tools not found![/bold red]")
                    return
                
                console.print("Client connected successfully")
                
                # Create sample regulatory documents
                old_regulation, new_regulation = await create_sample_regulations()
                
                # Display the documents
                console.print("\n[bold]Sample Regulatory Documents Created:[/bold]")
                console.print(f"Version 1: {old_regulation}")
                console.print(f"Version 2: {new_regulation}")
                
                # Generate a diff between document versions
                await compare_document_versions(session, diff_tool, old_regulation, new_regulation)
                
                # Extract and analyze requirements
                await extract_requirements(session, search_tool, new_regulation)
                
                # Find dependencies between regulations (simulated analysis)
                await analyze_dependencies(session, semantic_search_tool, new_regulation)
                
    except Exception as e:
        console.print(f"[bold red]Error running regulatory analysis: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())

async def compare_document_versions(session, diff_tool, old_version, new_version):
    """Compare two versions of a regulatory document to identify changes."""
    console.print(Rule("[bold yellow]Regulatory Document Comparison[/bold yellow]"))
    console.print("[italic]Comparing versions to identify significant changes[/italic]\n")
    
    # Generate diff between versions using MCP diff tool
    console.print("\n[bold cyan]Generating Diff Between Versions[/bold cyan]")
    
    try:
        # Call the diff tool
        result = await session.call_tool(
            diff_tool.name,
            arguments={
                "source_path": old_version,
                "target_path": new_version,
                "context_lines": 2
            }
        )
        
        # Extract the text content
        result_text = None
        for content in result.content:
            if content.type == "text":
                result_text = content.text
                break
        
        if result_text:
            try:
                # Parse the JSON response
                diff_result = json.loads(result_text)
                
                # Display the diff
                if diff_result.get("chunks"):
                    chunks = diff_result.get("chunks", [])
                    console.print(f"[green]Found {len(chunks)} changed sections[/green]")
                    
                    # Extract and classify changes
                    changes = await classify_changes(chunks)
                    
                    # Display a summary of changes
                    console.print("\n[bold cyan]Regulatory Change Summary[/bold cyan]")
                    table = Table()
                    table.add_column("Change Type")
                    table.add_column("Count")
                    table.add_column("Impact")
                    
                    table.add_row(
                        "New Requirements", 
                        str(changes.get("requirement_changes", {}).get("added", 0)),
                        "High" if changes.get("requirement_changes", {}).get("added", 0) > 0 else "Low"
                    )
                    table.add_row(
                        "Modified Requirements", 
                        str(changes.get("requirement_changes", {}).get("modified", 0)),
                        "Medium" if changes.get("requirement_changes", {}).get("modified", 0) > 0 else "Low"
                    )
                    table.add_row(
                        "Removed Requirements", 
                        str(changes.get("requirement_changes", {}).get("removed", 0)),
                        "High" if changes.get("requirement_changes", {}).get("removed", 0) > 0 else "Low"
                    )
                    
                    console.print(table)
                    
                    # Display significant changes
                    if changes.get("significant_changes"):
                        console.print("\n[bold cyan]Significant Regulatory Changes[/bold cyan]")
                        for i, change in enumerate(changes.get("significant_changes", [])[:5]):  # Show top 5
                            console.print(f"[bold]{i+1}. {change.get('type')}:[/bold]")
                            if change.get('type') == "modified_requirement":
                                console.print(f"  From: {change.get('source_content')}")
                                console.print(f"  To:   {change.get('target_content')}")
                            else:
                                console.print(f"  {change.get('content')}")
                            console.print("")
                else:
                    console.print("[yellow]No changes found between document versions[/yellow]")
            except json.JSONDecodeError:
                console.print("[bold red]Failed to parse diff response as JSON[/bold red]")
                console.print(f"Raw response: {result_text[:200]}...")
        else:
            console.print("[bold red]No text content in diff response[/bold red]")
            
    except Exception as e:
        console.print(f"[bold red]Error comparing documents: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())

async def classify_changes(diff_chunks):
    """Classify the changes in the diff chunks."""
    # Initialize counters for different types of changes
    results = {
        "total_chunks": len(diff_chunks),
        "added_chunks": 0,
        "removed_chunks": 0,
        "modified_chunks": 0,
        "requirement_changes": {
            "added": 0,
            "removed": 0,
            "modified": 0
        },
        "significant_changes": []
    }
    
    # Analyze each diff chunk
    for chunk in diff_chunks:
        # Determine chunk type
        if chunk.get("type") == "added":
            results["added_chunks"] += 1
            
            # Check if the added chunk contains requirements
            for pattern in REQUIREMENT_PATTERNS:
                if re.search(pattern["pattern"], chunk.get("content", ""), re.IGNORECASE):
                    results["requirement_changes"]["added"] += 1
                    if pattern["priority"] == "high":
                        results["significant_changes"].append({
                            "type": "added_requirement",
                            "content": chunk.get("content", ""),
                            "pattern": pattern["pattern"],
                            "line": chunk.get("target_start")
                        })
                    break
        
        elif chunk.get("type") == "removed":
            results["removed_chunks"] += 1
            
            # Check if the removed chunk contains requirements
            for pattern in REQUIREMENT_PATTERNS:
                if re.search(pattern["pattern"], chunk.get("content", ""), re.IGNORECASE):
                    results["requirement_changes"]["removed"] += 1
                    if pattern["priority"] == "high":
                        results["significant_changes"].append({
                            "type": "removed_requirement",
                            "content": chunk.get("content", ""),
                            "pattern": pattern["pattern"],
                            "line": chunk.get("source_start")
                        })
                    break
        
        elif chunk.get("type") == "modified":
            results["modified_chunks"] += 1
            
            # Check if the modified chunk contains requirements
            for pattern in REQUIREMENT_PATTERNS:
                if (re.search(pattern["pattern"], chunk.get("source_content", ""), re.IGNORECASE) or 
                    re.search(pattern["pattern"], chunk.get("target_content", ""), re.IGNORECASE)):
                    results["requirement_changes"]["modified"] += 1
                    if pattern["priority"] == "high":
                        results["significant_changes"].append({
                            "type": "modified_requirement",
                            "source_content": chunk.get("source_content", ""),
                            "target_content": chunk.get("target_content", ""),
                            "pattern": pattern["pattern"],
                            "source_line": chunk.get("source_start"),
                            "target_line": chunk.get("target_start")
                        })
                    break
    
    return results

async def extract_requirements(session, search_tool, regulation_file):
    """Extract regulatory requirements from a document."""
    console.print(Rule("[bold yellow]Regulatory Requirements Extraction[/bold yellow]"))
    console.print("[italic]Extracting and classifying requirements from the regulation[/italic]\n")
    
    # Read the file content
    with open(regulation_file, 'r') as f:
        content = f.read()
    
    # Use the MCP search tool to find requirements
    console.print("[bold cyan]Extracting Regulatory Requirements[/bold cyan]")
    
    requirements = []
    for pattern in REQUIREMENT_PATTERNS:
        try:
            # Call the search tool for each pattern
            result = await session.call_tool(
                search_tool.name,
                arguments={
                    "query": pattern["pattern"],
                    "content": content,
                    "case_sensitive": False
                }
            )
            
            # Extract the text content
            result_text = None
            for content_item in result.content:
                if content_item.type == "text":
                    result_text = content_item.text
                    break
            
            if result_text:
                try:
                    # Parse the JSON response
                    search_result = json.loads(result_text)
                    
                    if "matches" in search_result:
                        matches = search_result["matches"]
                        console.print(f"[green]Found {len(matches)} {pattern['description']} statements[/green]")
                        
                        # Process each match
                        for match in matches:
                            # Extract the matching line and some context
                            match_line = match.get("line_text", "")
                            
                            requirements.append({
                                "text": match_line,
                                "type": pattern["type"],
                                "priority": pattern["priority"],
                                "line": match.get("line_number")
                            })
                except json.JSONDecodeError:
                    console.print(f"[bold red]Failed to parse search response for pattern '{pattern['pattern']}' as JSON[/bold red]")
            else:
                console.print(f"[bold red]No text content in search response for pattern '{pattern['pattern']}'[/bold red]")
                
        except Exception as e:
            console.print(f"[bold red]Error searching for pattern '{pattern['pattern']}': {str(e)}[/bold red]")
    
    # Display requirements
    if requirements:
        console.print("\n[bold cyan]Extracted Regulatory Requirements[/bold cyan]")
        
        table = Table()
        table.add_column("Type")
        table.add_column("Priority")
        table.add_column("Requirement")
        
        # Sort by priority (high -> medium -> low)
        priority_order = {"high": 0, "medium": 1, "low": 2}
        requirements.sort(key=lambda x: priority_order.get(x["priority"], 3))
        
        # Show the first 10 requirements
        for req in requirements[:10]:
            table.add_row(
                req["type"].capitalize(),
                req["priority"].capitalize(),
                req["text"]
            )
        
        console.print(table)
        console.print(f"Showing 10 of {len(requirements)} requirements")
    else:
        console.print("[yellow]No requirements found in the document[/yellow]")

async def analyze_dependencies(session, semantic_search_tool, regulation_file):
    """Analyze dependencies between regulations (simulated)."""
    console.print(Rule("[bold yellow]Regulatory Dependencies Analysis[/bold yellow]"))
    console.print("[italic]Identifying references and dependencies to other regulations[/italic]\n")
    
    # Read the file content
    with open(regulation_file, 'r') as f:
        content = f.read()
    
    # Use semantic search if available
    console.print("[bold cyan]Searching for Regulatory References[/bold cyan]")
    search_query = "reference to other regulation or standard"
    
    if semantic_search_tool:
        try:
            # Call the semantic search tool
            result = await session.call_tool(  # noqa: F841
                semantic_search_tool.name,
                arguments={
                    "query": search_query,
                    "content": content
                }
            )
            
            # For the demo, we'll simulate finding some dependencies
            console.print("[green]Analysis complete - simulated results[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Semantic search failed: {str(e)}. Using simulated results.[/yellow]")
    else:
        console.print("[yellow]Semantic search tool not available. Using simulated results.[/yellow]")
    
    # Create a table of simulated dependencies
    table = Table()
    table.add_column("Referenced Regulation")
    table.add_column("Relevance")
    table.add_column("Impact")
    
    # Add some sample data
    table.add_row("GDPR Article 5", "High", "Direct compliance required")
    table.add_row("ISO 27001", "Medium", "Security standard alignment")
    table.add_row("NIST 800-53", "Low", "Optional reference")
    
    console.print(table)

async def main():
    """Run the regulatory changes analysis demo."""
    await analyze_regulatory_changes()

if __name__ == "__main__":
    asyncio.run(main()) 