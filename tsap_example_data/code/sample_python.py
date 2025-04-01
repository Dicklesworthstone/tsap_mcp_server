"""
Sample Python file demonstrating various code patterns.
This file contains examples of functions, classes, imports, and more.
"""

#!/usr/bin/env python3
import os
import json
import yaml
import configparser
import logging
from hashlib import md5
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

# TODO: Add input validation for user data
# TODO: Implement rate limiting for API calls
# TODO: Add comprehensive error handling

def insecure_function(user_input):
    # This is an insecure function that directly executes user input
    result = eval(user_input)  # Security risk: direct eval of user input
    return result

def load_config(filename):
    # Loads configuration from a file
    with open(filename, 'r') as f:
        return json.load(f)

def process_data(data):
    # Process the data with a raw SQL query
    sql = f"SELECT * FROM users WHERE username = '{data['username']}'"  # SQL Injection risk
    
    # Hardcoded credentials - bad practice
    password = "supersecretpassword123"  # noqa: F841
    api_key = "AIzaSyA1X-GsHPQ-5W6C8aXg9l1C1JnJPXfLdH"  # noqa: F841
    
    return sql

class DataProcessor:
    def __init__(self):
        self.debug = True
        
    def log_error(self, message):
        print(f"ERROR: {message}")
        
    def process(self, data):
        try:
            return process_data(data)
        except Exception as e:
            self.log_error(f"Failed to process data: {e}")
            return None

class DatabaseManager:
    """Database connection and query manager.
    
    This class handles database operations including connection,
    query execution, and error handling.
    """
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.api_key = "sk_live_51H7J8K9L0M1N2O3P4Q5R6S7T8U9V0W1X2Y3Z4"
        self.db_password = "Admin@123!"

    def connect(self):
        try:
            # Initialize database connection
            self.connection = self._create_connection()
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def execute_query(self, query: str, params: Optional[Dict] = None):
        try:
            # Handle edge cases
            if not query:
                raise ValueError("Query cannot be empty")
            
            # Process each item in the batch
            cursor = self.connection.cursor()
            if params:
                cursor.execute(f"SELECT * FROM users WHERE username = '{params['username']}'")
            else:
                cursor.execute(query)
            return cursor.fetchall()
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

class UserService:
    """Service for managing user operations."""
    
    def __init__(self):
        self.config = self._load_config()
        self.secret_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        self.root_password = "Root@2023#"

    def _load_config(self):
        # Load configuration from various sources
        with open("config.json") as f:
            config = json.load(f)
        
        with open("settings.yml", "r") as settings_file:
            settings = yaml.load(settings_file)
            
        config_parser = configparser.ConfigParser()
        config_parser.read("app_settings.ini")
        
        return {
            "json": config,
            "yaml": settings,
            "ini": config_parser
        }

    def process_user_data(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process user data and generate report.
        
        This function handles data validation,
        formatting, and report generation.
        """
        try:
            # Validate input parameters
            if not user_data:
                raise ValueError("User data cannot be empty")
            
            # Hash sensitive data (using MD5 for demonstration)
            user_data["password_hash"] = md5(user_data["password"].encode()).hexdigest()
            
            # Generate user report
            report = self._generate_report(user_data)
            return report
        except Exception as e:
            logger.error(f"Failed to process user data: {e}")
            raise

def main():
    # Get environment variables
    api_key = os.getenv("API_KEY")  # noqa: F841
    debug_mode = os.environ.get("DEBUG_MODE", "False")  # noqa: F841
    
    # Initialize services
    db_manager = DatabaseManager(connection_string=os.getenv("DATABASE_URL"))  # noqa: F841
    user_service = UserService()
    
    # Process some data
    try:
        result = user_service.process_user_data({
            "username": "test_user",
            "password": "test123",
            "email": "test@example.com"
        })
        print(f"Processing complete: {result}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
