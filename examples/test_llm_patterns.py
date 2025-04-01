#!/usr/bin/env python3
"""
Simple test for LLM pattern generation

This is a minimal script to verify the LLM pattern generation feature
"""
import os
import asyncio
import sys

# Set environment variables before importing
os.environ["USE_LLM_PATTERN_GENERATION"] = "true"
os.environ["LLM_MCP_SERVER_URL"] = "http://localhost:8013"

# Add the src directory to Python path
sys.path.insert(0, './src')

async def test_llm_generation():
    """Test the LLM pattern generation functionality directly."""
    print("Testing LLM pattern generation...")
    
    # Import inside the function to ensure environment variables are set first
    from tsap.evolution.pattern_analyzer import generate_patterns_with_llm
    
    # Test data
    positive_examples = [
        "2024-07-30 10:02:05 ERROR [APIService] Failed processing request: Invalid token",
        "2024-07-30 10:04:10 ERROR [APIService] Internal server error in /api/v1/users: Database connection timeout",
        "2024-07-30 10:07:50 ERROR [APIService] Invalid input parameters for request ID 83749: Missing required field",
    ]
    
    negative_examples = [
        "2024-07-30 10:00:01 INFO [AuthService] User 'admin' logged in",
        "2024-07-30 10:01:15 WARN [APIService] Request rate limit approaching for client 12345",
        "2024-07-30 10:02:30 INFO [DatabaseService] Connected to database",
        "2024-07-30 10:03:45 DEBUG [APIService] Request parameters: {'id': 123, 'action': 'get'}",
        "2024-07-30 10:05:20 INFO [CacheService] Cache invalidated for user 456",
        "2024-07-30 10:06:35 ERROR [AuthService] Authentication failed for user 'test': Invalid credentials",
    ]
    
    reference_set = {
        "positive": positive_examples,
        "negative": negative_examples
    }
    
    # Initial pattern
    pattern = r"ERROR.*\[APIService\].*"
    description = "Evolve pattern to find APIService errors in logs"
    
    print(f"Initial pattern: {pattern}")
    
    # Try to generate LLM patterns directly
    print("Calling LLM pattern generation...")
    try:
        llm_patterns = await generate_patterns_with_llm(
            pattern=pattern,
            description=description,
            reference_set=reference_set,
            num_variants=3
        )
        print(f"LLM-generated patterns: {llm_patterns}")
    except Exception as e:
        print(f"Error during LLM pattern generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_llm_generation()) 