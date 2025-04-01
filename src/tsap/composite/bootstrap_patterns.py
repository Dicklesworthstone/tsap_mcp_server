"""
Bootstrap patterns module.

This module provides functions to bootstrap the pattern library with
sample patterns for demos and testing.
"""
import uuid
from typing import List

from tsap.composite.patterns import PatternDefinition, PatternCategory, PatternPriority, get_pattern_library

def bootstrap_pattern_library() -> List[str]:
    """
    Bootstrap the pattern library with sample patterns.
    
    Returns:
        List of pattern IDs added
    """
    library = get_pattern_library()
    pattern_ids = []
    
    # Skip if the library already has patterns
    if library.list_patterns():
        return []
    
    # Add some security patterns
    security_patterns = [
        PatternDefinition(
            id=str(uuid.uuid4()),
            name="SQL Injection Risk",
            pattern=r"SELECT \* FROM users WHERE username = '{",
            description="Detects potential SQL injection attack patterns in code",
            category=PatternCategory.SECURITY,
            is_regex=True,
            case_sensitive=False,
            multiline=False,
            priority=PatternPriority.HIGH,
            tags=["security", "sql-injection", "vulnerability"],
            confidence=0.9,
            author="TSAP Demo",
            source="Bootstrap"
        ),
        PatternDefinition(
            id=str(uuid.uuid4()),
            name="Hardcoded Credentials",
            pattern=r"password = \"supersecretpassword123\"",
            description="Detects hardcoded credentials in source code",
            category=PatternCategory.SECURITY,
            is_regex=True,
            case_sensitive=False,
            multiline=False,
            priority=PatternPriority.HIGH,
            tags=["security", "credentials", "hardcoded"],
            confidence=0.85,
            author="TSAP Demo",
            source="Bootstrap" 
        ),
        PatternDefinition(
            id=str(uuid.uuid4()),
            name="API Key",
            pattern=r"api_key = \"AIzaSyA1X-GsHPQ-5W6C8aXg9l1C1JnJPXfLdH\"",
            description="Detects hardcoded API keys in source code",
            category=PatternCategory.SECURITY,
            is_regex=True,
            case_sensitive=False,
            multiline=False,
            priority=PatternPriority.HIGH,
            tags=["security", "api-key", "hardcoded"],
            confidence=0.9,
            author="TSAP Demo",
            source="Bootstrap"
        )
    ]
    
    # Add some code quality patterns
    code_patterns = [
        PatternDefinition(
            id=str(uuid.uuid4()),
            name="TODO Comment",
            pattern=r"# TODO:",
            description="Finds TODO comments in code",
            category=PatternCategory.CODE,
            is_regex=True,
            case_sensitive=False,
            multiline=False,
            priority=PatternPriority.MEDIUM,
            tags=["code", "todo", "comment"],
            confidence=0.95,
            author="TSAP Demo", 
            source="Bootstrap"
        ),
        PatternDefinition(
            id=str(uuid.uuid4()),
            name="FIXME Comment",
            pattern=r"# FIXME:",
            description="Finds FIXME comments in code",
            category=PatternCategory.CODE,
            is_regex=True,
            case_sensitive=False,
            multiline=False,
            priority=PatternPriority.MEDIUM,
            tags=["code", "fixme", "comment"],
            confidence=0.95,
            author="TSAP Demo",
            source="Bootstrap"
        )
    ]
    
    # Add some configuration patterns
    config_patterns = [
        PatternDefinition(
            id=str(uuid.uuid4()),
            name="Debug Mode Enabled",
            pattern=r"debug: true",
            description="Finds debug mode enabled in configuration files",
            category=PatternCategory.CONFIGURATION,
            is_regex=True,
            case_sensitive=False,
            multiline=False,
            priority=PatternPriority.MEDIUM,
            tags=["configuration", "debug", "security"],
            confidence=0.85,
            author="TSAP Demo",
            source="Bootstrap"
        ),
        PatternDefinition(
            id=str(uuid.uuid4()),
            name="Hardcoded Database Password",
            pattern=r"password: secret_password",
            description="Detects hardcoded database credentials in config files",
            category=PatternCategory.CONFIGURATION,
            is_regex=True,
            case_sensitive=False,
            multiline=False,
            priority=PatternPriority.HIGH,
            tags=["configuration", "credentials", "security"],
            confidence=0.9,
            author="TSAP Demo",
            source="Bootstrap"
        )
    ]
    
    # Add some log patterns
    log_patterns = [
        PatternDefinition(
            id=str(uuid.uuid4()),
            name="Error Log Entry",
            pattern=r"\[ERROR\]",
            description="Finds error entries in log files",
            category=PatternCategory.LOG,
            is_regex=True,
            case_sensitive=True,
            multiline=False,
            priority=PatternPriority.HIGH,
            tags=["log", "error", "monitoring"],
            confidence=0.95,
            author="TSAP Demo",
            source="Bootstrap"
        ),
        PatternDefinition(
            id=str(uuid.uuid4()),
            name="Authentication Failure",
            pattern=r"Failed to authenticate user",
            description="Detects authentication failures in log files",
            category=PatternCategory.LOG,
            is_regex=True,
            case_sensitive=False,
            multiline=False,
            priority=PatternPriority.HIGH,
            tags=["log", "security", "authentication"],
            confidence=0.95,
            author="TSAP Demo",
            source="Bootstrap"
        ),
        PatternDefinition(
            id=str(uuid.uuid4()),
            name="Database Connection Error",
            pattern=r"Unable to connect to database",
            description="Detects database connection errors in log files",
            category=PatternCategory.LOG,
            is_regex=True,
            case_sensitive=False,
            multiline=False,
            priority=PatternPriority.HIGH,
            tags=["log", "database", "error"],
            confidence=0.9,
            author="TSAP Demo",
            source="Bootstrap"
        )
    ]
    
    # Add all patterns to the library
    all_patterns = security_patterns + code_patterns + config_patterns + log_patterns
    
    for pattern in all_patterns:
        library._add_pattern(pattern)
        pattern_ids.append(pattern.id)
    
    return pattern_ids 