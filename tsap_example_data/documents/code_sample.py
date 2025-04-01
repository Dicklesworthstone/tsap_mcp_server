#!/usr/bin/env python3
# Data Processing Module
# This module implements various data processing utilities

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataProcessor:
    """A utility class for processing and transforming data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data processor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.transformations = []
        self.processed_count = 0
        logger.info("DataProcessor initialized with config: %s", self.config)
    
    def register_transformation(self, transformation_func):
        """
        Register a transformation function.
        
        Args:
            transformation_func: A function that takes data and returns transformed data
        """
        self.transformations.append(transformation_func)
        logger.debug("Registered transformation: %s", transformation_func.__name__)
        return self
    
    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a list of data items by applying all registered transformations.
        
        Args:
            data: List of data items to process
            
        Returns:
            Processed data items
        """
        result = data
        for transform in self.transformations:
            result = [transform(item) for item in result]
        
        self.processed_count += len(result)
        logger.info("Processed %d items (total: %d)", len(result), self.processed_count)
        return result

def load_json_data(filepath: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        List of data items
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        else:
            return [data]
    except Exception as e:
        logger.error("Error loading JSON data: %s", str(e))
        return []

def save_json_data(data: List[Dict[str, Any]], filepath: str) -> bool:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        filepath: Path to the output file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error("Error saving JSON data: %s", str(e))
        return False

# Common transformations

def add_timestamp(item: Dict[str, Any]) -> Dict[str, Any]:
    """Add a processed_at timestamp to the item."""
    item['processed_at'] = datetime.now().isoformat()
    return item

def convert_to_uppercase(item: Dict[str, Any]) -> Dict[str, Any]:
    """Convert all string values to uppercase."""
    return {k: v.upper() if isinstance(v, str) else v for k, v in item.items()}

def remove_nulls(item: Dict[str, Any]) -> Dict[str, Any]:
    """Remove all keys with null values."""
    return {k: v for k, v in item.items() if v is not None}

def main():
    """Example usage of the DataProcessor."""
    # Example data
    data = [
        {"id": 1, "name": "Alpha", "value": 100, "valid": True},
        {"id": 2, "name": "Beta", "value": None, "valid": False},
        {"id": 3, "name": "Gamma", "value": 300, "valid": True}
    ]
    
    # Create processor with custom configuration
    processor = DataProcessor({"debug": True})
    
    # Register transformations
    processor.register_transformation(add_timestamp)
    processor.register_transformation(remove_nulls)
    
    # Process the data
    processed_data = processor.process(data)
    
    # Print results
    print("Processed data:")
    for item in processed_data:
        print(f"- {item}")
    
    return processed_data

if __name__ == "__main__":
    main()
