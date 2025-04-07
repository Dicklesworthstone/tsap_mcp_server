#!/usr/bin/env python3
"""
Advanced SQLite Demo (MCP Tools Version)

This script demonstrates the comprehensive features of the SQLite Query
tool in TSAP MCP, including various query types, schema inspection,
and data manipulation.
"""
import asyncio
import json
import os
import sqlite3
import tempfile
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from typing import Any, List, Optional

# Import proper MCP client modules
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

console = Console()

# Add some basic debugging
DEBUG = False

def debug_print(msg):
    """Print debug message if DEBUG is True."""
    if DEBUG:
        console.print(f"[dim][DEBUG] {msg}[/dim]")

# Data directory for our example databases
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Path to our example database
EXAMPLE_DB = os.path.join(DATA_DIR, "example.db")

# Path to our document database
DOCUMENTS_DB = os.path.join(DATA_DIR, "documents.db")

async def create_example_database():
    """Create an example database with sample data for demonstrations."""
    # Create a fresh database
    if os.path.exists(EXAMPLE_DB):
        os.remove(EXAMPLE_DB)
        
    console.print("[bold]Creating example database...[/bold]")
    
    # Create database with Python's sqlite3
    conn = sqlite3.connect(EXAMPLE_DB)
    cursor = conn.cursor()
    
    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON")
    
    # Create tables with some meaningful example data
    cursor.execute("""
    CREATE TABLE users (
        id INTEGER PRIMARY KEY,
        username TEXT NOT NULL UNIQUE,
        email TEXT NOT NULL UNIQUE,
        date_joined TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        active BOOLEAN DEFAULT TRUE
    )
    """)
    
    cursor.execute("""
    CREATE TABLE posts (
        id INTEGER PRIMARY KEY,
        user_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        views INTEGER DEFAULT 0,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """)
    
    cursor.execute("""
    CREATE TABLE comments (
        id INTEGER PRIMARY KEY,
        post_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (post_id) REFERENCES posts(id),
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """)
    
    # Create an index on posts for better query performance
    cursor.execute("CREATE INDEX idx_posts_user_id ON posts(user_id)")
    cursor.execute("CREATE INDEX idx_comments_post_id ON comments(post_id)")
    
    # Insert sample data for users
    users = [
        (1, "johndoe", "john.doe@example.com", "2022-01-01 10:00:00", True),
        (2, "janedoe", "jane.doe@example.com", "2022-01-02 11:30:00", True),
        (3, "bobsmith", "bob.smith@example.com", "2022-02-15 09:15:00", True),
        (4, "alicejones", "alice.jones@example.com", "2022-03-10 14:45:00", True),
        (5, "charliebrown", "charlie.brown@example.com", "2022-04-05 16:20:00", False),
        (6, "emilydavis", "emily.davis@example.com", "2022-04-20 13:10:00", True),
        (7, "michaelwilson", "michael.wilson@example.com", "2022-05-12 10:30:00", True),
        (8, "oliviamartin", "olivia.martin@example.com", "2022-06-01 09:00:00", True)
    ]
    
    cursor.executemany(
        "INSERT INTO users (id, username, email, date_joined, active) VALUES (?, ?, ?, ?, ?)",
        users
    )
    
    # Insert sample data for posts
    posts = [
        (1, 1, "First Post", "This is my first post content.", "2022-01-05 12:00:00", "2022-01-05 12:00:00", 45),
        (2, 1, "SQLite Tips", "Here are some tips for using SQLite...", "2022-01-10 14:30:00", "2022-01-15 16:45:00", 120),
        (3, 2, "Hello World", "Introduction to programming languages.", "2022-01-15 09:20:00", "2022-01-15 09:20:00", 75),
        (4, 3, "Database Design", "Best practices for designing databases.", "2022-02-20 11:00:00", "2022-03-01 13:15:00", 210),
        (5, 4, "Python Tutorial", "Learn Python programming basics.", "2022-03-15 10:30:00", "2022-03-20 15:45:00", 350),
        (6, 6, "Data Analysis", "Using SQL for data analysis.", "2022-04-25 14:00:00", "2022-05-01 11:30:00", 180),
        (7, 7, "Mobile Development", "Introduction to app development.", "2022-05-15 09:45:00", "2022-05-20 16:15:00", 90),
        (8, 8, "Web Design Principles", "Modern web design techniques.", "2022-06-05 13:30:00", "2022-06-10 10:15:00", 135),
        (9, 1, "Advanced SQLite", "Deep dive into SQLite features.", "2022-02-01 15:20:00", "2022-02-10 09:30:00", 200),
        (10, 2, "Programming Tips", "Useful programming tips and tricks.", "2022-03-05 11:40:00", "2022-03-10 14:00:00", 110)
    ]
    
    cursor.executemany(
        "INSERT INTO posts (id, user_id, title, content, created_at, updated_at, views) VALUES (?, ?, ?, ?, ?, ?, ?)",
        posts
    )
    
    # Insert sample data for comments
    comments = [
        (1, 1, 2, "Great first post!", "2022-01-06 09:15:00"),
        (2, 1, 3, "Welcome to the community!", "2022-01-06 10:30:00"),
        (3, 2, 1, "I learned a lot from this, thanks!", "2022-01-11 14:00:00"),
        (4, 2, 3, "Very helpful tips, especially for beginners.", "2022-01-12 16:45:00"),
        (5, 3, 4, "Nice introduction post.", "2022-01-16 11:20:00"),
        (6, 4, 5, "This database design post is excellent!", "2022-02-21 13:10:00"),
        (7, 5, 6, "Great Python tutorial for beginners.", "2022-03-16 15:30:00"),
        (8, 6, 7, "Data analysis with SQL is so powerful.", "2022-04-26 10:15:00"),
        (9, 7, 8, "Looking forward to more mobile dev content.", "2022-05-16 14:20:00"),
        (10, 8, 1, "Web design principles are changing fast!", "2022-06-06 09:45:00"),
        (11, 9, 2, "This advanced SQLite post is amazing.", "2022-02-05 16:00:00"),
        (12, 10, 3, "These programming tips saved me hours!", "2022-03-07 11:30:00"),
        (13, 9, 4, "I'm going to use these SQLite features.", "2022-02-07 13:45:00"),
        (14, 5, 5, "Python is my favorite language now!", "2022-03-18 10:00:00"),
        (15, 4, 6, "Database design is crucial - great post.", "2022-02-25 15:30:00")
    ]
    
    cursor.executemany(
        "INSERT INTO comments (id, post_id, user_id, content, created_at) VALUES (?, ?, ?, ?, ?)",
        comments
    )
    
    # Create a view for easy access to post stats
    cursor.execute("""
    CREATE VIEW post_stats AS
    SELECT 
        p.id as post_id, 
        p.title, 
        u.username as author,
        p.views,
        COUNT(c.id) as comment_count
    FROM posts p
    JOIN users u ON p.user_id = u.id
    LEFT JOIN comments c ON p.id = c.post_id
    GROUP BY p.id
    """)
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    console.print("[green]Example database created successfully![/green]")
    return EXAMPLE_DB

async def create_documents_database():
    """Create a database to store document data for search demonstrations."""
    # Create a fresh database
    if os.path.exists(DOCUMENTS_DB):
        os.remove(DOCUMENTS_DB)
        
    console.print("[bold]Creating documents database...[/bold]")
    
    # Create database with Python's sqlite3
    conn = sqlite3.connect(DOCUMENTS_DB)
    cursor = conn.cursor()
    
    # Create tables for documents and content
    cursor.execute("""
    CREATE TABLE documents (
        id INTEGER PRIMARY KEY,
        filename TEXT NOT NULL,
        title TEXT,
        author TEXT,
        date_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        date_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        file_type TEXT,
        file_size INTEGER,
        tags TEXT
    )
    """)
    
    cursor.execute("""
    CREATE TABLE document_content (
        id INTEGER PRIMARY KEY,
        document_id INTEGER NOT NULL,
        section_name TEXT,
        content TEXT,
        FOREIGN KEY (document_id) REFERENCES documents(id)
    )
    """)
    
    # Create table for search results from ripgrep
    cursor.execute("""
    CREATE TABLE search_results (
        id INTEGER PRIMARY KEY,
        document_id INTEGER NOT NULL,
        search_term TEXT NOT NULL,
        line_number INTEGER,
        line_text TEXT,
        search_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (document_id) REFERENCES documents(id)
    )
    """)
    
    # Sample document data
    documents = [
        (1, "project_proposal.md", "Project Proposal: TSAP Integration", "John Smith", "2023-01-15 10:30:00", "2023-02-01 14:45:00", "markdown", 15240, "proposal,project,planning"),
        (2, "database_schema.sql", "Database Schema", "Jane Doe", "2023-01-20 11:15:00", "2023-01-25 16:30:00", "sql", 8456, "database,schema,sql"),
        (3, "api_documentation.md", "API Documentation", "Bob Johnson", "2023-02-05 09:45:00", "2023-03-10 13:20:00", "markdown", 24680, "api,docs,reference"),
        (4, "user_guide.txt", "User Guide", "Alice Williams", "2023-02-15 15:00:00", "2023-02-28 10:10:00", "text", 32450, "user,guide,manual"),
        (5, "config_template.json", "Configuration Template", "Charlie Davis", "2023-03-01 13:30:00", "2023-03-05 11:45:00", "json", 4230, "config,template,settings"),
        (6, "presentation.txt", "Project Presentation", "Emily Wilson", "2023-03-15 14:00:00", "2023-03-20 16:15:00", "text", 18690, "presentation,slides"),
        (7, "meeting_notes.md", "Team Meeting Notes", "Michael Brown", "2023-03-25 10:00:00", "2023-03-25 11:30:00", "markdown", 5780, "meeting,notes,team"),
        (8, "requirements.txt", "Software Requirements", "Olivia Martinez", "2023-04-01 09:15:00", "2023-04-10 14:40:00", "text", 12340, "requirements,software"),
        (9, "changelog.md", "Change Log", "David Thompson", "2023-04-15 11:20:00", "2023-04-30 15:50:00", "markdown", 9870, "changelog,history,versions"),
        (10, "deployment_guide.txt", "Deployment Guide", "Sophia Rodriguez", "2023-05-01 13:45:00", "2023-05-15 10:30:00", "text", 28760, "deployment,guide,setup")
    ]
    
    cursor.executemany(
        "INSERT INTO documents (id, filename, title, author, date_created, date_modified, file_type, file_size, tags) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        documents
    )
    
    # Sample document content
    document_content = [
        (1, 1, "Introduction", "This document outlines the proposal for implementing TSAP integration with our current systems..."),
        (2, 1, "Objectives", "The main objectives of this project are to enhance data analysis capabilities and improve search performance..."),
        (3, 1, "Timeline", "Phase 1 (January): Initial assessment and planning\nPhase 2 (February): Development of core components\nPhase 3 (March): Integration and testing\nPhase 4 (April): Deployment and user training"),
        (4, 2, "Users Table", "CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT NOT NULL, email TEXT NOT NULL UNIQUE);"),
        (5, 2, "Documents Table", "CREATE TABLE documents (id INTEGER PRIMARY KEY, title TEXT, content TEXT, user_id INTEGER, FOREIGN KEY (user_id) REFERENCES users(id));"),
        (6, 3, "Authentication", "## Authentication\n\nThis API uses OAuth2 for authentication. To authenticate, send a POST request to `/auth/token` with your client credentials..."),
        (7, 3, "Endpoints", "## Endpoints\n\n### GET /api/documents\nReturns a list of all documents accessible to the authenticated user.\n\n### POST /api/documents\nCreates a new document. Request body must include title and content fields."),
        (8, 4, "Getting Started", "# Getting Started\n\nTo begin using the application, first install the required dependencies by running:\n```\npip install -r requirements.txt\n```"),
        (9, 4, "Configuration", "# Configuration\n\nEdit the `config.ini` file to set up your environment variables and database connection parameters..."),
        (10, 5, "Basic Configuration", "{\n  \"server\": {\n    \"host\": \"localhost\",\n    \"port\": 8080\n  },\n  \"database\": {\n    \"host\": \"db.example.com\",\n    \"user\": \"username\",\n    \"password\": \"password\"\n  }\n}"),
        (11, 6, "Introduction Slide", "# TSAP Integration Project\n\nPresented by: Emily Wilson\nDate: March 20, 2023"),
        (12, 6, "Features Slide", "## Key Features\n\n- Enhanced search capabilities\n- Integration with existing systems\n- Improved data analysis\n- User-friendly interface"),
        (13, 7, "Attendees", "# Team Meeting - March 25, 2023\n\nAttendees: John, Jane, Bob, Alice, Charlie"),
        (14, 7, "Action Items", "## Action Items\n\n1. John to complete the API documentation by April 1\n2. Jane to finalize database schema by March 28\n3. Bob to set up the testing environment by March 30"),
        (15, 8, "Python Packages", "numpy==1.21.0\npandas==1.3.0\nmatplotlib==3.4.2\nrequests==2.26.0\nsqlite3==2.6.0"),
        (16, 9, "Version 1.0.0", "## Version 1.0.0 (2023-01-15)\n\n- Initial release\n- Basic search functionality\n- User authentication"),
        (17, 9, "Version 1.1.0", "## Version 1.1.0 (2023-02-20)\n\n- Added advanced search options\n- Improved performance\n- Fixed user login issues"),
        (18, 10, "Prerequisites", "# Prerequisites\n\n- Python 3.8 or higher\n- Docker\n- Access to the production server"),
        (19, 10, "Deployment Steps", "# Deployment Steps\n\n1. Build the Docker image: `docker build -t tsap-app .`\n2. Push the image to the registry: `docker push registry.example.com/tsap-app:latest`\n3. SSH into the production server\n4. Pull and run the Docker container: `docker pull registry.example.com/tsap-app:latest && docker run -d -p 8080:8080 tsap-app`")
    ]
    
    cursor.executemany(
        "INSERT INTO document_content (id, document_id, section_name, content) VALUES (?, ?, ?, ?)",
        document_content
    )
    
    # Create a virtual table for full-text search
    cursor.execute("CREATE VIRTUAL TABLE content_fts USING fts5(document_id, section_name, content, content='document_content', content_rowid='id')")
    cursor.execute("INSERT INTO content_fts(content_fts) VALUES('rebuild')")
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    console.print("[green]Documents database created successfully![/green]")
    return DOCUMENTS_DB

async def sqlite_demo():
    """Demonstrate SQLite's advanced features with example databases."""
    console.print(Panel.fit(
        "[bold blue]TSAP MCP SQLite Advanced Features Demo[/bold blue]",
        subtitle="Exploring SQLite capabilities and integration with search tools"
    ))
    
    # Path to the proxy script
    proxy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_proxy.py")
    debug_print(f"Proxy path: {proxy_path}")
    
    # Verify the proxy script exists
    if not os.path.exists(proxy_path):
        console.print(f"[bold red]Error: Proxy script not found at {proxy_path}[/bold red]")
        return
    
    # Configure server parameters for stdio connection to our proxy
    server_params = StdioServerParameters(
        command="python",
        args=[proxy_path],
        env={"MCP_DEBUG": "1"}  # Enable debug logging
    )
    
    # Create our client
    try:
        debug_print("Creating MCP client session...")
        
        # Create example databases first (this doesn't require the MCP client)
        example_db = await create_example_database()
        documents_db = await create_documents_database()
        
        # Test a direct SQLite query to verify database creation
        console.print("[bold]Testing direct SQLite query...[/bold]")
        conn = sqlite3.connect(example_db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        conn.close()
        console.print(f"[green]Tables in database: {tables}[/green]")
        
        # Connect to the MCP server via proxy
        console.print("[bold]Connecting to MCP proxy...[/bold]")
        async with stdio_client(server_params) as (read, write):
            debug_print("Connected to MCP proxy")
            async with ClientSession(read, write) as session:
                # Initialize the connection
                console.print("Initializing session...")
                init_result = await session.initialize()
                console.print(f"[green]Successfully initialized connection to {init_result.serverInfo.name} {init_result.serverInfo.version}[/green]")
                debug_print("Session initialized successfully")
                
                # Get the list of available tools
                console.print("Getting available tools...")
                tools_result = await session.list_tools()
                tools = tools_result.tools
                console.print(f"[green]Found {len(tools)} tools[/green]")
                debug_print(f"Found {len(tools)} tools")
                
                # Find the database tool
                database_tool = next((t for t in tools if t.name == "database" or t.name == "sqlite_query"), None)
                search_tool = next((t for t in tools if t.name == "search" or t.name == "ripgrep_search"), None)
                
                if not database_tool:
                    console.print("[bold red]Error: database/sqlite_query tool not found![/bold red]")
                    tool_names = [t.name for t in tools]
                    console.print(f"Available tools: {', '.join(tool_names)}")
                    return
                
                console.print(f"[green]Using database tool: {database_tool.name}[/green]")
                if search_tool:
                    console.print(f"[green]Using search tool: {search_tool.name}[/green]")
                else:
                    console.print("[yellow]Search tool not found, search integration demo will be skipped[/yellow]")
                
                # --- Add initial info check ---
                console.print("Checking server info...")
                
                # Get info tool
                info_tool = next((t for t in tools if t.name == "info"), None)
                if info_tool:
                    console.print(f"Using info tool: {info_tool.name}")
                    info_result = await session.call_tool(info_tool.name, arguments={})
                    
                    # Extract info text
                    info_text = None
                    for content in info_result.content:
                        if content.type == "text":
                            info_text = content.text
                            break
                    
                    if info_text:
                        try:
                            info_data = json.loads(info_text)  # noqa: F841
                            console.print("[green]Initial server info check successful.[/green]")
                        except json.JSONDecodeError:
                            console.print(f"[yellow]Info response is not JSON: {info_text[:100]}...[/yellow]")
                    else:
                        console.print("[yellow]No text content in info response[/yellow]")
                else:
                    console.print("[yellow]Info tool not found, continuing anyway...[/yellow]")
                # ------------------------------
                
                # Let's limit our demos for debugging
                if DEBUG:
                    debug_print("DEBUG mode: Only running database creation")
                    
                    # Test basic query function directly using MCP tools
                    console.print("[bold]Testing basic SQLite query via MCP client...[/bold]")
                    test_query = "SELECT name FROM sqlite_master WHERE type='table'"
                    
                    # Call the database tool with MCP protocol
                    console.print(f"Calling {database_tool.name} tool with query: {test_query}")
                    test_result = await session.call_tool(database_tool.name, arguments={
                        "query": test_query,
                        "database": example_db
                    })
                    
                    # Extract the text content
                    test_text = None
                    for content in test_result.content:
                        if content.type == "text":
                            test_text = content.text
                            break
                    
                    if test_text:
                        try:
                            test_data = json.loads(test_text)
                            debug_print(f"Test query response: {test_data}")
                            console.print(f"[green]Test query successful: {test_data}[/green]")
                        except json.JSONDecodeError:
                            console.print(f"[yellow]Test response is not JSON: {test_text[:100]}...[/yellow]")
                    else:
                        console.print("[yellow]No text content in test response[/yellow]")
                    
                    return
                
                # Add your SQLite demo functions here
                await run_basic_query_demo(session, database_tool, example_db)
                await run_advanced_query_demo(session, database_tool, example_db)
                await run_transaction_demo(session, database_tool, example_db)
                await run_result_modes_demo(session, database_tool, example_db)
                if search_tool:
                    await run_ripgrep_integration_demo(session, database_tool, search_tool, documents_db)
                await run_performance_optimization_demo(session, database_tool, example_db)
                
    except Exception as e:
        console.print(f"[bold red]Error running demo: {str(e)}[/bold red]")

async def run_basic_query_demo(session, database_tool, database):
    """Run a basic SQLite query demo."""
    console.print(Rule("[bold yellow]Basic Query Demo[/bold yellow]"))
    console.print("[italic]Demonstrates simple SELECT queries with different clauses[/italic]\n")
    
    # Demo 1: Simple SELECT query
    console.print("[bold cyan]Simple SELECT Query[/bold cyan]")
    await run_demo(
        session,
        database_tool,
        None,
        None,
        query="SELECT * FROM users LIMIT 5",
        database=database,
        show_title=False
    )
    
    # Demo 2: SELECT with WHERE clause
    console.print("\n[bold cyan]SELECT with WHERE Clause[/bold cyan]")
    await run_demo(
        session,
        database_tool,
        None,
        None,
        query="SELECT username, email FROM users WHERE active = ?",
        params=[True],
        database=database,
        show_title=False
    )
    
    # Demo 3: SELECT with ORDER BY
    console.print("\n[bold cyan]SELECT with ORDER BY[/bold cyan]")
    await run_demo(
        session,
        database_tool,
        None,
        None,
        query="SELECT id, title, views FROM posts ORDER BY views DESC LIMIT 5",
        database=database,
        show_title=False
    )

async def run_advanced_query_demo(session, database_tool, database):
    """Run advanced SQLite query demos."""
    console.print(Rule("[bold yellow]Advanced Query Demo[/bold yellow]"))
    console.print("[italic]Demonstrates JOINs, aggregate functions, and complex queries[/italic]\n")
    
    # Demo 1: JOIN query
    console.print("[bold cyan]JOIN Query[/bold cyan]")
    await run_demo(
        session,
        database_tool,
        None,
        None,
        query="""
        SELECT posts.title, users.username, posts.views
        FROM posts
        JOIN users ON posts.user_id = users.id
        ORDER BY posts.views DESC
        LIMIT 5
        """,
        database=database,
        show_title=False
    )
    
    # Demo 2: Aggregate functions
    console.print("\n[bold cyan]Aggregate Functions[/bold cyan]")
    await run_demo(
        session,
        database_tool,
        None,
        None,
        query="""
        SELECT 
            users.username,
            COUNT(posts.id) as post_count,
            SUM(posts.views) as total_views,
            AVG(posts.views) as avg_views
        FROM users
        LEFT JOIN posts ON users.id = posts.user_id
        GROUP BY users.id
        ORDER BY total_views DESC
        """,
        database=database,
        show_title=False
    )
    
    # Demo 3: Subqueries
    console.print("\n[bold cyan]Subqueries[/bold cyan]")
    await run_demo(
        session,
        database_tool,
        None,
        None,
        query="""
        SELECT username, email
        FROM users
        WHERE id IN (
            SELECT user_id
            FROM posts
            GROUP BY user_id
            HAVING COUNT(*) > 1
        )
        """,
        database=database,
        show_title=False
    )

async def run_transaction_demo(session, database_tool, database):
    """Run a transaction and schema demo."""
    console.print(Rule("[bold yellow]Transaction and Schema Demo[/bold yellow]"))
    console.print("[italic]Demonstrates SQLite's transaction support and schema operations[/italic]\n")
    
    # Demo 1: Query to show current schema
    console.print("[bold cyan]Schema Information[/bold cyan]")
    await run_demo(
        session,
        database_tool,
        None,
        None,
        query="SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
        database=database,
        show_title=False
    )
    
    # Demo 2: Transaction with multiple operations
    console.print("\n[bold cyan]Multi-Statement Transaction[/bold cyan]")
    await run_demo(
        session,
        database_tool,
        None,
        None,
        query="""
        BEGIN TRANSACTION;
        
        CREATE TEMPORARY TABLE temp_stats AS
        SELECT 
            u.username,
            COUNT(p.id) as post_count,
            COUNT(c.id) as comment_count
        FROM users u
        LEFT JOIN posts p ON u.id = p.user_id
        LEFT JOIN comments c ON u.id = c.user_id
        GROUP BY u.id;
        
        SELECT * FROM temp_stats ORDER BY post_count DESC;
        
        DROP TABLE temp_stats;
        
        COMMIT;
        """,
        database=database,
        show_title=False
    )
    
    # Demo 3: View information
    console.print("\n[bold cyan]Database Views[/bold cyan]")
    await run_demo(
        session,
        database_tool,
        None,
        None,
        query="""
        SELECT name, sql FROM sqlite_master WHERE type='view';
        
        SELECT * FROM post_stats ORDER BY views DESC LIMIT 5;
        """,
        database=database,
        show_title=False
    )

async def run_result_modes_demo(session, database_tool, database):
    """Demonstrate different result modes (list, dict, table)."""
    console.print(Rule("[bold yellow]Result Modes Demo[/bold yellow]"))
    console.print("[italic]Demonstrates the different result format modes available[/italic]\n")
    
    # Query to use for all demos
    query = "SELECT users.username, posts.title, posts.views FROM users JOIN posts ON users.id = posts.user_id LIMIT 5"
    
    # Demo 1: List mode (default)
    console.print("[bold cyan]List Mode Results[/bold cyan]")
    await run_demo(
        session,
        database_tool,
        None,
        None,
        query=query,
        database=database,
        mode="list",
        show_title=False
    )
    
    # Demo 2: Dict mode
    console.print("\n[bold cyan]Dict Mode Results[/bold cyan]")
    await run_demo(
        session,
        database_tool,
        None,
        None,
        query=query,
        database=database,
        mode="dict",
        show_title=False
    )
    
    # Demo 3: Table mode 
    console.print("\n[bold cyan]Table Mode Results[/bold cyan]")
    await run_demo(
        session,
        database_tool,
        None,
        None,
        query=query,
        database=database,
        mode="table",  # Table mode is handled on client side
        show_title=False
    )

async def run_ripgrep_integration_demo(session, database_tool, search_tool, database):
    """Demonstrate integration between search and SQLite."""
    console.print(Rule("[bold yellow]Search + SQLite Integration Demo[/bold yellow]"))
    console.print("[italic]Demonstrates how search tools and SQLite can work together for powerful data analysis[/italic]\n")
    
    # Demo 1: Use search tool (ripgrep) to search files
    console.print("[bold cyan]Step 1: Using Search Tool[/bold cyan]")
    search_term = "auth"
    console.print(f"Searching for '{search_term}' in document content...")
    
    try:
        # Call the search tool using MCP protocol
        search_result = await session.call_tool(search_tool.name, arguments={
            "query": search_term,
            "paths": ["."],  # Search current directory
            "include_pattern": "*.md,*.txt"  # Only search markdown and text files
        })
        
        # Extract the text content
        search_text = None
        for content in search_result.content:
            if content.type == "text":
                search_text = content.text
                break
        
        if not search_text:
            console.print("[yellow]No text content in search response[/yellow]")
            console.print("Skipping search integration demo")
            return
        
        # Try to parse as JSON first
        search_data = None
        try:
            debug_print(f"Search response text: {search_text[:100]}...")
            search_data = json.loads(search_text)
            debug_print(f"Parsed search data: {str(search_data)[:200]}...")
            
            # Handle different search result formats
            matches = []
            
            # Format 1: Content with 'matches' array
            if isinstance(search_data, dict) and 'matches' in search_data:
                matches = search_data['matches']
            # Format 2: Direct array of matches
            elif isinstance(search_data, list):
                matches = search_data
            # Format 3: Status success format
            elif isinstance(search_data, dict) and 'status' in search_data and search_data['status'] == 'success':
                if 'data' in search_data and 'matches' in search_data['data']:
                    matches = search_data['data']['matches']
                elif 'data' in search_data and isinstance(search_data['data'], list):
                    matches = search_data['data']
            
            if not matches:
                console.print("[yellow]No matches found in search results[/yellow]")
                console.print("Skipping search integration demo")
                return
                
            console.print(f"[green]Found {len(matches)} matches for '{search_term}'[/green]")
            
            # For demonstration, we'll just show a few matches
            for i, match in enumerate(matches[:3]):
                if isinstance(match, dict):
                    # Try to extract info from structured format
                    filename = match.get('file', match.get('filename', 'unknown'))
                    line_num = match.get('line_number', match.get('line', 0))
                    line_text = match.get('line_text', match.get('text', match.get('content', '')))
                    console.print(f"{filename}:{line_num}: {line_text}")
                else:
                    # Just print the match as-is
                    console.print(f"Match {i+1}: {match}")
            
            if len(matches) > 3:
                console.print(f"... and {len(matches) - 3} more matches")
            
            # Demonstrate inserting search results into SQLite
            console.print("\n[bold cyan]Step 2: Storing Search Results in SQLite[/bold cyan]")
            console.print("Storing search results in the database...")
            
            # Create an insert statement for our search results
            insert_query = """
            INSERT INTO search_results (document_id, search_term, line_number, line_text)
            VALUES (?, ?, ?, ?)
            """
            
            # For demonstration, we'll insert a subset of the results
            # This is just a mock demonstration - in reality, we'd need to correlate
            # filenames with document_ids from our documents table
            
            params = []
            for i, match in enumerate(matches[:5]):
                # Mock document ID (1-5) based on index
                doc_id = (i % 5) + 1
                if isinstance(match, dict):
                    line_num = match.get('line_number', match.get('line', 0))
                    line_text = match.get('line_text', match.get('text', match.get('content', '')))
                    params.append((doc_id, search_term, line_num, line_text[:100]))
                else:
                    # Just use the match string directly
                    params.append((doc_id, search_term, 0, str(match)[:100]))
            
            # Use SQLite to insert the data (do this directly for simplicity)
            conn = sqlite3.connect(database)
            cursor = conn.cursor()
            cursor.executemany(insert_query, params)
            conn.commit()
            
            # Now query the stored search results
            console.print("\n[bold cyan]Step 3: Analyzing Search Results with SQL[/bold cyan]")
            
            # Use the MCP database tool for the query
            await run_demo(
                session,
                database_tool,
                None,
                None,
                query="""
                SELECT d.title, sr.line_number, sr.line_text
                FROM search_results sr
                JOIN documents d ON sr.document_id = d.id
                WHERE sr.search_term = ?
                ORDER BY d.title
                """,
                database=database,
                params=[search_term],
                show_title=False
            )
            
        except json.JSONDecodeError:
            console.print(f"[yellow]Failed to parse search response as JSON: {search_text[:100]}...[/yellow]")
            
            # Try to handle as plain text
            if search_text and len(search_text) > 0:
                console.print("[yellow]Treating response as plain text[/yellow]")
                
                # Split the text into lines
                lines = search_text.splitlines()
                
                # Filter to keep only lines that contain our search term
                matches = [line for line in lines if search_term in line]
                
                if matches:
                    console.print(f"[green]Found {len(matches)} text matches for '{search_term}'[/green]")
                    
                    # Show a few matches
                    for i, match in enumerate(matches[:3]):
                        console.print(f"Match {i+1}: {match}")
                    
                    if len(matches) > 3:
                        console.print(f"... and {len(matches) - 3} more matches")
                else:
                    console.print(f"[yellow]No plain text matches found for '{search_term}'[/yellow]")
                    console.print("Skipping search integration demo")
            else:
                console.print("[yellow]Empty or invalid search response[/yellow]")
                console.print("Skipping search integration demo")
    except Exception as e:
        console.print(f"[bold red]Error in search integration demo: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())

async def run_performance_optimization_demo(session, database_tool, database):
    """Demonstrate SQLite performance optimizations."""
    console.print(Rule("[bold yellow]Performance Optimization Demo[/bold yellow]"))
    console.print("[italic]Demonstrates SQLite performance tuning techniques[/italic]\n")
    
    # Demo 1: EXPLAIN QUERY PLAN
    console.print("[bold cyan]EXPLAIN QUERY PLAN[/bold cyan]")
    await run_demo(
        session,
        database_tool,
        None,
        None,
        query="EXPLAIN QUERY PLAN SELECT * FROM posts WHERE user_id = 1",
        database=database,
        show_title=False
    )
    
    # Demo 2: Pragmas for performance tuning
    console.print("\n[bold cyan]SQLite PRAGMAs for Performance[/bold cyan]")
    await run_demo(
        session,
        database_tool,
        None,
        None,
        query="""
        -- Check current journal mode
        PRAGMA journal_mode;
        
        -- Check synchronous setting
        PRAGMA synchronous;
        
        -- Check page size
        PRAGMA page_size;
        
        -- Check cache size
        PRAGMA cache_size;
        """,
        database=database,
        show_title=False
    )
    
    # Demo 3: Benchmarking queries
    console.print("\n[bold cyan]Query Benchmarking[/bold cyan]")
    
    # Simple benchmark query
    await run_demo(
        session,
        database_tool,
        None, 
        None,
        query="""
        BEGIN TRANSACTION;
        
        -- Create a temporary table for benchmarking
        CREATE TEMPORARY TABLE benchmark_results (
            query_id INTEGER PRIMARY KEY,
            query_desc TEXT,
            execution_time REAL
        );
        
        -- Run benchmark for query 1 (with index)
        INSERT INTO benchmark_results (query_id, query_desc, execution_time)
        SELECT 1, 'With index', (julianday('now') - julianday('now', 'localtime'))*86400.0;
        
        UPDATE benchmark_results 
        SET execution_time = (julianday('now') - julianday('now', 'start'))*86400.0
        WHERE query_id = 1;
        
        -- Display results
        SELECT * FROM benchmark_results;
        
        -- Clean up
        DROP TABLE benchmark_results;
        
        COMMIT;
        """,
        database=database,
        show_title=False
    )

async def run_demo(
    session,
    database_tool,
    title: Optional[str] = None,
    description: Optional[str] = None,
    query: str = None,
    database: str = None,
    params: Optional[List[Any]] = None,
    mode: Optional[str] = None,
    show_title: bool = True,
    **kwargs
):
    """Execute a SQL query and display the results.
    
    Args:
        session: The ClientSession instance
        database_tool: The database tool to use
        title: Title for the demo section (optional)
        description: Description text for the demo (optional)
        query: SQL query to execute
        database: Path to the database file
        params: Optional parameters for the query
        mode: Optional result format mode (list, dict, table)
        show_title: Whether to show the title/description
        **kwargs: Additional parameters for the query
    """
    if show_title and title:
        console.print(f"[bold]{title}[/bold]")
    if show_title and description:
        console.print(f"[italic]{description}[/italic]")
    
    try:
        # Execute the query using MCP tools database tool with proper arguments
        arguments = {
            "query": query,
            "database": database
        }
        if params:
            arguments["parameters"] = params
        for key, value in kwargs.items():
            arguments[key] = value
            
        # Add debug output of the query
        debug_print(f"Executing query via {database_tool.name}: {query}")
        result = await session.call_tool(database_tool.name, arguments=arguments)
        
        # Extract the text content from the result
        response_text = None
        for content in result.content:
            if content.type == "text":
                response_text = content.text
                break
        
        if not response_text:
            console.print("[bold red]No text content in response[/bold red]")
            return
            
        # Print debug info
        debug_print(f"Response text: {response_text[:100]}...")
            
        # Parse the JSON response
        try:
            response = json.loads(response_text)
            debug_print(f"Parsed JSON response: {str(response)[:200]}...")
            
            # The response format may be different than expected
            # Try multiple approaches to extract data:
            
            # Approach 1: Check for explicit status field
            if "status" in response:
                if response.get("status") != "success":
                    error = response.get("error", {})
                    error_msg = error.get("message", "Unknown error")
                    console.print(f"[bold red]Error executing query: {error_msg}[/bold red]")
                    return
                
                # Success with status field - get data
                data = response.get("data", {})
            # Approach 2: Direct data format
            elif "rows" in response or "columns" in response:
                # Assume this is a direct data response
                data = response
            # Approach 3: Unknown format
            else:
                # Just print what we got
                console.print(f"[yellow]Response format not recognized: {response_text[:100]}...[/yellow]")
                console.print(response)
                return
            
            # Display the results based on mode
            if mode == "table" and "columns" in data and "rows" in data:
                # Create a Rich table
                table = Table()
                for col in data["columns"]:
                    table.add_column(str(col))
                
                for row in data["rows"]:
                    table.add_row(*[str(val) for val in row])
                
                console.print(table)
            else:
                # Default display
                if "columns" in data and "rows" in data:
                    columns = data["columns"]  # noqa: F841
                    rows = data["rows"]
                    
                    console.print(f"[bold]Results: {len(rows)} rows[/bold]")
                    for i, row in enumerate(rows):
                        if i < 10:  # Limit display to first 10 rows
                            # Format as a simple table structure
                            console.print(f"Row {i+1}: {row}")
                    if len(rows) > 10:
                        console.print(f"... and {len(rows) - 10} more rows")
                else:
                    # Just print the raw data
                    console.print(data)
                    
            # Display execution time if available
            if "execution_time" in data:
                console.print(f"[dim]Execution time: {data['execution_time']:.4f}s[/dim]")
        except json.JSONDecodeError as e:
            console.print(f"[bold red]Failed to parse response as JSON: {response_text[:100]}...[/bold red]")
            console.print(f"[red]JSON error: {str(e)}[/red]")
    
    except Exception as e:
        console.print(f"[bold red]Error in run_demo: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())

async def main():
    """Run the SQLite demo."""
    # Parse command line arguments
    if "--debug" in sys.argv:
        global DEBUG
        DEBUG = True
        debug_print("Debug mode enabled")
    
    # Show the banner
    console.print(Panel("[bold blue]TSAP MCP SQLite Query Advanced Demo[/bold blue]", 
                        subtitle="Demonstrating SQLite database interactions"))
    
    # Create a temporary database for testing
    db_path = os.path.join(tempfile.gettempdir(), "tsap_sqlite_demo.db")
    console.print(f"Using database path: {db_path}")
    
    # Create a simple database file if it doesn't exist
    if not os.path.exists(db_path):
        console.print(f"Creating new SQLite database at {db_path}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, name TEXT)")
        cursor.execute("INSERT INTO test (name) VALUES ('test_entry')")
        conn.commit()
        conn.close()
    
    # Run the main SQLite demo
    await sqlite_demo()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Unhandled error: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc()) 