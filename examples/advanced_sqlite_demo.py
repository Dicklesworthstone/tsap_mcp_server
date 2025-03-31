#!/usr/bin/env python3
"""
Advanced SQLite Demo

This script demonstrates the comprehensive features of the SQLite integration
in TSAP by creating, querying, and analyzing databases in various ways.
It also shows how SQLite can be used synergistically with ripgrep for more
powerful data analysis workflows.
"""
import asyncio
import os
import sys
import json
import sqlite3
from datetime import datetime
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.console import Console
from rich.rule import Rule

# Assuming the MCPClient is in the examples directory
from mcp_client_example import MCPClient

# Add the SQLite client method to MCPClient
async def sqlite_query(
    self,
    query: str,
    database: str,
    params: list = None,
    mode: str = "dict",
    headers: bool = True,
) -> dict:
    """Execute a SQLite query.
    
    Args:
        query: The SQL query to execute
        database: Path to the SQLite database file
        params: Optional list of query parameters (for parameterized queries)
        mode: Result mode - "list", "dict", or "table"
        headers: Whether to include column headers in the result
        
    Returns:
        Query results
    """
    args = {
        "query": query,
        "database": database,
        "mode": mode,
        "headers": headers
    }
    
    if params is not None:
        args["params"] = params
        
    return await self.send_request("sqlite_query", args)

# Add the method to the MCPClient class
MCPClient.sqlite_query = sqlite_query

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
    
    # Insert sample data for comments - fixed to ensure valid foreign keys
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
        "[bold blue]TSAP SQLite Advanced Features Demo[/bold blue]",
        subtitle="Exploring SQLite capabilities and integration with ripgrep"
    ))
    
    # Create our client
    try:
        debug_print("Creating MCPClient...")
        
        # Check if MCP server is running
        try:
            import httpx
            response = await httpx.AsyncClient().get("http://localhost:8021/health")
            debug_print(f"MCP server health check response: {response.status_code}")
            if response.status_code != 200:
                console.print("[bold red]MCP server is not responding correctly. Make sure it's running.[/bold red]")
                return
        except Exception as e:
            console.print(f"[bold red]Error connecting to MCP server: {str(e)}[/bold red]")
            console.print("[yellow]Make sure the MCP server is running on http://localhost:8021[/yellow]")
            return
        
        # Create the client
        async with MCPClient() as client:
            debug_print(f"Client created with base_url: {client.base_url}")
            
            # --- Add initial info check ---
            console.print(f"Attempting to get server info from {client.base_url}...")
            info = await client.info()
            debug_print(f"Info response: {info}")
            
            if info.get("status") != "success" or info.get("error") is not None:
                console.print(f"[bold red]Error during initial client.info() check:[/bold red] {info.get('error', 'Status was not success')}")
                return # Exit if server check fails
            else:
               console.print("[green]Initial client.info() check successful.[/green]")
            # ------------------------------
            
            # Create example databases
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
            
            # Let's limit our demos for debugging
            if DEBUG:
                debug_print("DEBUG mode: Only running database creation")
                
                # Test basic query function directly
                console.print("[bold]Testing basic SQLite query via MCP client...[/bold]")
                test_query = "SELECT name FROM sqlite_master WHERE type='table'"
                test_response = await client.sqlite_query(
                    query=test_query,
                    database=example_db
                )
                debug_print(f"Test query response: {test_response}")
                
                return
            
            # Add your SQLite demo functions here
            await run_basic_query_demo(client, example_db)
            await run_advanced_query_demo(client, example_db)
            await run_transaction_demo(client, example_db)
            await run_result_modes_demo(client, example_db)
            await run_ripgrep_integration_demo(client, documents_db)
            await run_performance_optimization_demo(client, example_db)
            
    except Exception as e:
        console.print(f"[bold red]Error running demo: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())

async def run_basic_query_demo(client, database):
    """Run a basic SQLite query demo."""
    console.print(Rule("[bold yellow]Basic Query Demo[/bold yellow]"))
    console.print("[italic]Demonstrates simple SELECT queries with different clauses[/italic]\n")
    
    # Demo 1: Simple SELECT query
    console.print("[bold cyan]Simple SELECT Query[/bold cyan]")
    await run_demo(
        client,
        None,
        None,
        query="SELECT * FROM users LIMIT 5",
        database=database,
        show_title=False
    )
    
    # Demo 2: SELECT with WHERE clause
    console.print("\n[bold cyan]SELECT with WHERE Clause[/bold cyan]")
    await run_demo(
        client,
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
        client,
        None,
        None,
        query="SELECT id, title, views FROM posts ORDER BY views DESC LIMIT 5",
        database=database,
        show_title=False
    )

async def run_advanced_query_demo(client, database):
    """Run advanced SQLite query demos."""
    console.print(Rule("[bold yellow]Advanced Query Demo[/bold yellow]"))
    console.print("[italic]Demonstrates JOINs, aggregate functions, and complex queries[/italic]\n")
    
    # Demo 1: JOIN query
    console.print("[bold cyan]JOIN Query[/bold cyan]")
    await run_demo(
        client,
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
        client,
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
        client,
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

async def run_transaction_demo(client, database):
    """Run a transaction and schema demo."""
    console.print(Rule("[bold yellow]Transaction and Schema Demo[/bold yellow]"))
    console.print("[italic]Demonstrates SQLite's transaction support and schema operations[/italic]\n")
    
    # Demo 1: Query to show current schema
    console.print("[bold cyan]Schema Information[/bold cyan]")
    await run_demo(
        client,
        None,
        None,
        query="SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
        database=database,
        show_title=False
    )
    
    # Demo 2: Transaction with multiple operations
    console.print("\n[bold cyan]Multi-Statement Transaction[/bold cyan]")
    await run_demo(
        client,
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
        client,
        None,
        None,
        query="""
        SELECT name, sql FROM sqlite_master WHERE type='view';
        
        SELECT * FROM post_stats ORDER BY views DESC LIMIT 5;
        """,
        database=database,
        show_title=False
    )

async def run_result_modes_demo(client, database):
    """Demonstrate different result modes (list, dict, table)."""
    console.print(Rule("[bold yellow]Result Modes Demo[/bold yellow]"))
    console.print("[italic]Demonstrates the different result format modes available[/italic]\n")
    
    # Query to use for all demos
    query = "SELECT users.username, posts.title, posts.views FROM users JOIN posts ON users.id = posts.user_id LIMIT 5"
    
    # Demo 1: List mode (default)
    console.print("[bold cyan]List Mode Results[/bold cyan]")
    await run_demo(
        client,
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
        client,
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
        client,
        None,
        None,
        query=query,
        database=database,
        mode="table",  # Table mode is handled on client side
        show_title=False
    )

async def run_ripgrep_integration_demo(client, database):
    """Demonstrate integration between ripgrep and SQLite."""
    console.print(Rule("[bold yellow]Ripgrep + SQLite Integration Demo[/bold yellow]"))
    console.print("[italic]Demonstrates how ripgrep and SQLite can work together for powerful data analysis[/italic]\n")
    
    # Demo 1: Use ripgrep to search files
    console.print("[bold cyan]Step 1: Using Ripgrep to Search[/bold cyan]")
    search_term = "auth"
    console.print(f"Searching for '{search_term}' in document content...")
    
    # First perform the ripgrep search
    ripgrep_result = await client.ripgrep_search(
        pattern=search_term,
        paths=["."],  # Search current directory
        case_sensitive=False,
        context_lines=0
    )
    
    if "data" in ripgrep_result and "matches" in ripgrep_result["data"]:
        matches = ripgrep_result["data"]["matches"]
        console.print(f"[green]Found {len(matches)} matches with ripgrep[/green]")
        
        # Store results in the database
        await run_demo(
            client,
            None,
            None,
            query="""
            BEGIN TRANSACTION;
            
            -- First, clear previous search results for this term
            DELETE FROM search_results WHERE search_term = ?;
            
            COMMIT;
            """,
            params=[search_term],
            database=database,
            show_title=False
        )
        
        # Insert search results into the database (simplified, in real app would insert from ripgrep results)
        console.print("\n[bold cyan]Step 2: Storing Search Results in SQLite[/bold cyan]")
        await run_demo(
            client,
            None,
            None,
            query="""
            BEGIN TRANSACTION;
            
            -- Insert some sample search results
            INSERT INTO search_results (document_id, search_term, line_number, line_text)
            VALUES
                (3, ?, 2, 'This API uses OAuth2 for authentication...'),
                (4, ?, 15, 'The authentication module provides security features...'),
                (8, ?, 8, 'Authentication requires proper credentials setup...');
                
            COMMIT;
            """,
            params=[search_term, search_term, search_term],
            database=database,
            show_title=False
        )
        
        # Demo 3: Join search results with document metadata for analysis
        console.print("\n[bold cyan]Step 3: Analyzing Search Results with SQL[/bold cyan]")
        await run_demo(
            client,
            None,
            None,
            query="""
            SELECT 
                d.title,
                d.author,
                d.file_type,
                sr.line_number,
                sr.line_text
            FROM search_results sr
            JOIN documents d ON sr.document_id = d.id
            WHERE sr.search_term = ?
            ORDER BY d.date_modified DESC
            """,
            params=[search_term],
            database=database,
            show_title=False
        )
        
        # Demo 4: Generate summary statistics from search results
        console.print("\n[bold cyan]Step 4: Generating Summary Statistics[/bold cyan]")
        await run_demo(
            client,
            None,
            None,
            query="""
            SELECT 
                d.file_type,
                COUNT(*) as match_count,
                AVG(d.file_size) as avg_file_size
            FROM search_results sr
            JOIN documents d ON sr.document_id = d.id
            WHERE sr.search_term = ?
            GROUP BY d.file_type
            """,
            params=[search_term],
            database=database,
            show_title=False
        )
    else:
        console.print("[yellow]No ripgrep matches found[/yellow]")

async def run_performance_optimization_demo(client, database):
    """Demonstrate SQLite performance optimizations."""
    console.print(Rule("[bold yellow]Performance Optimization Demo[/bold yellow]"))
    console.print("[italic]Demonstrates SQLite performance tuning techniques[/italic]\n")
    
    # Demo 1: EXPLAIN QUERY PLAN
    console.print("[bold cyan]EXPLAIN QUERY PLAN[/bold cyan]")
    await run_demo(
        client,
        None,
        None,
        query="EXPLAIN QUERY PLAN SELECT * FROM posts WHERE user_id = 1",
        database=database,
        show_title=False
    )
    
    # Demo 2: Pragmas for performance tuning
    console.print("\n[bold cyan]SQLite PRAGMAs for Performance[/bold cyan]")
    await run_demo(
        client,
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
        client,
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
        
        -- Show results
        SELECT * FROM benchmark_results;
        
        -- Clean up
        DROP TABLE benchmark_results;
        
        COMMIT;
        """,
        database=database,
        show_title=False
    )
    
    # Demo 4: WAL mode and other pragmas
    console.print("\n[bold cyan]Write-Ahead Logging (WAL) Mode[/bold cyan]")
    console.print("[yellow]Note: In a real application, these PRAGMAs would be set at database open time[/yellow]")
    
    # Show the optimized PRAGMA settings
    await run_demo(
        client,
        None,
        None,
        query="""
        -- Example PRAGMA settings for high-performance applications
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        PRAGMA cache_size = -32000; -- 32MB cache
        PRAGMA temp_store = MEMORY;
        PRAGMA mmap_size = 30000000000; -- Use memory-mapped I/O
        PRAGMA busy_timeout = 5000; -- 5 second timeout
        """,
        database=database,
        show_title=False
    )

async def run_demo(client, title, description, **query_params):
    """Run a SQLite query demo with the given parameters."""
    # Whether to show the title (defaults to True)
    show_title = query_params.pop("show_title", True)
    
    if show_title and title:
        console.print(Rule(f"[bold yellow]{title}[/bold yellow]"))
        
    if description:
        console.print(f"[italic]{description}[/italic]\n")
    
    # Show the query parameters
    console.print("[bold cyan]Query Parameters:[/bold cyan]")
    params_table = Table(show_header=False, box=None)
    params_table.add_column("Parameter", style="green")
    params_table.add_column("Value", style="white")
    
    # Extract params if present to handle them specially
    params = query_params.pop("params", None)
    
    for key, value in query_params.items():
        # Skip our internal parameter
        if key == "show_title":
            continue
        
        # Format the value for display
        if key == "query":
            # Show query over multiple lines with syntax highlighting
            console.print("[bold cyan]SQL Query:[/bold cyan]")
            console.print(Syntax(value, "sql", theme="monokai"))
            continue
            
        params_table.add_row(key, str(value))
    
    # Add params back for display
    if params:
        params_table.add_row("params", str(params))
    
    console.print(params_table)
    console.print()
    
    # Add params back to query_params for the API call
    if params:
        query_params["params"] = params
    
    # Execute the query
    start_time = datetime.now()
    console.print("[bold]Executing query...[/bold]")
    
    try:
        response = await client.sqlite_query(**query_params)
    except Exception as e:
        console.print(f"[bold red]Error during client.sqlite_query call: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        response = {"error": {"code": "CLIENT_SIDE_ERROR", "message": str(e)}}
    
    # Calculate execution time
    execution_time = (datetime.now() - start_time).total_seconds()
    
    # Debug the response
    debug_print(f"Response structure: {str(response.keys())}")
    
    # Process and display results
    if response.get("status") == "success" and "data" in response:
        data = response["data"]
        # Show summary statistics
        console.print(f"[green]Query completed in {execution_time:.3f} seconds[/green]")
        
        if "rows" in data:
            rows = data["rows"]
            row_count = len(rows)
            
            # Show the command that was executed
            if "columns" in data and data["columns"]:
                columns = data["columns"]
                
                # Display results in a table
                if row_count > 0:
                    results_table = Table(title=f"Query Results ({row_count} rows)")
                    
                    # Add columns
                    for column in columns:
                        results_table.add_column(column, style="cyan")
                    
                    # Add rows to the table (max 15 for readability)
                    max_display = min(15, row_count)
                    
                    # Handle different result formats
                    mode = query_params.get("mode", "dict")  # noqa: F841
                    
                    for i in range(max_display):
                        if i >= len(rows):
                            break
                            
                        row = rows[i]
                        if isinstance(row, dict):
                            # Dict mode
                            results_table.add_row(*[str(row.get(col, "")) for col in columns])
                        elif isinstance(row, (list, tuple)):
                            # List mode
                            results_table.add_row(*[str(val) for val in row])
                        else:
                            # Unknown format
                            results_table.add_row(str(row))
                    
                    console.print(results_table)
                    
                    if row_count > 15:
                        console.print(f"[dim]... and {row_count - 15} more rows not shown[/dim]")
                else:
                    console.print("[yellow]Query returned 0 rows[/yellow]")
            else:
                # No columns returned, just show raw data
                console.print("[bold cyan]Raw Results:[/bold cyan]")
                console.print(Syntax(json.dumps(rows, indent=2), "json", theme="monokai"))
        else:
            console.print("[yellow]Query successful but no rows returned[/yellow]")
    else:
        if "error" in response:
            console.print(f"[bold red]Query failed with error: {response.get('error', {}).get('message', 'Unknown error')}[/bold red]")
        else:
            console.print("[bold red]Query failed or returned no data[/bold red]")
            debug_print(f"Full response: {response}")
    
    console.print("\n")  # Add space between demos

if __name__ == "__main__":
    if "--debug" in sys.argv:
        DEBUG = True
        debug_print("Debug mode enabled")
    
    try:
        asyncio.run(sqlite_demo())
    except KeyboardInterrupt:
        console.print("[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Unhandled error: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
