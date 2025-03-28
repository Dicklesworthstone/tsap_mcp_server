"""
TSAP Incremental Splitter.

This module provides utilities for splitting input data into processable chunks
for incremental processing. It supports various data types and splitting strategies.
"""

import os
import re
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, AsyncIterator, TypeVar, Generic

from tsap.utils.logging import logger
from tsap.utils.errors import TSAPError


# Type variables for input and chunk types
I = TypeVar('I')  # noqa: E741
C = TypeVar('C')


class SplitterError(TSAPError):
    """Error raised when data splitting fails."""
    
    def __init__(
        self,
        message: str,
        splitter: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a splitter error.
        
        Args:
            message: Error message
            splitter: Splitter name
            details: Additional error details
        """
        error_details = details or {}
        if splitter:
            error_details["splitter"] = splitter
        
        super().__init__(message, "SPLITTER_ERROR", error_details)


class Splitter(Generic[I, C], ABC):
    """Base class for data splitters.
    
    Data splitters divide input data into processable chunks for
    incremental processing.
    """
    
    def __init__(self, name: str):
        """Initialize a splitter.
        
        Args:
            name: Splitter name
        """
        self.name = name
    
    @abstractmethod
    async def split(self, input_data: I) -> AsyncIterator[C]:
        """Split input data into chunks.
        
        Args:
            input_data: Input data
            
        Yields:
            Data chunks
        """
        pass
    
    async def estimate_chunks(self, input_data: I) -> int:
        """Estimate the number of chunks that will be produced.
        
        Override this method to provide accurate estimates.
        
        Args:
            input_data: Input data
            
        Returns:
            Estimated number of chunks
        """
        return 0


class ListSplitter(Splitter[List[Any], List[Any]]):
    """Splitter for lists.
    
    Splits a list into smaller lists of the specified size.
    """
    
    def __init__(self, chunk_size: int = 100, name: str = "list_splitter"):
        """Initialize a list splitter.
        
        Args:
            chunk_size: Maximum number of items per chunk
            name: Splitter name
        """
        super().__init__(name)
        self.chunk_size = max(1, chunk_size)
    
    async def split(self, input_data: List[Any]) -> AsyncIterator[List[Any]]:
        """Split a list into chunks.
        
        Args:
            input_data: Input list
            
        Yields:
            List chunks
        """
        for i in range(0, len(input_data), self.chunk_size):
            yield input_data[i:i + self.chunk_size]
    
    async def estimate_chunks(self, input_data: List[Any]) -> int:
        """Estimate the number of chunks.
        
        Args:
            input_data: Input list
            
        Returns:
            Estimated number of chunks
        """
        return (len(input_data) + self.chunk_size - 1) // self.chunk_size


class DictionarySplitter(Splitter[Dict[Any, Any], Dict[Any, Any]]):
    """Splitter for dictionaries.
    
    Splits a dictionary into smaller dictionaries of the specified size.
    """
    
    def __init__(self, chunk_size: int = 100, name: str = "dict_splitter"):
        """Initialize a dictionary splitter.
        
        Args:
            chunk_size: Maximum number of items per chunk
            name: Splitter name
        """
        super().__init__(name)
        self.chunk_size = max(1, chunk_size)
    
    async def split(self, input_data: Dict[Any, Any]) -> AsyncIterator[Dict[Any, Any]]:
        """Split a dictionary into chunks.
        
        Args:
            input_data: Input dictionary
            
        Yields:
            Dictionary chunks
        """
        items = list(input_data.items())
        
        for i in range(0, len(items), self.chunk_size):
            chunk_items = items[i:i + self.chunk_size]
            yield dict(chunk_items)
    
    async def estimate_chunks(self, input_data: Dict[Any, Any]) -> int:
        """Estimate the number of chunks.
        
        Args:
            input_data: Input dictionary
            
        Returns:
            Estimated number of chunks
        """
        return (len(input_data) + self.chunk_size - 1) // self.chunk_size


class TextSplitter(Splitter[str, str]):
    """Splitter for text.
    
    Splits text into chunks based on different strategies.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 0,
        split_strategy: str = "chars",
        name: str = "text_splitter",
    ):
        """Initialize a text splitter.
        
        Args:
            chunk_size: Maximum chunk size
            overlap: Overlap between chunks
            split_strategy: Splitting strategy ('chars', 'words', 'lines', 'paragraphs', 'sentences')
            name: Splitter name
        """
        super().__init__(name)
        self.chunk_size = max(1, chunk_size)
        self.overlap = max(0, min(overlap, chunk_size - 1))
        self.split_strategy = split_strategy
    
    async def split(self, input_data: str) -> AsyncIterator[str]:
        """Split text into chunks.
        
        Args:
            input_data: Input text
            
        Yields:
            Text chunks
        """
        if not input_data:
            return
        
        if self.split_strategy == "chars":
            # Split by characters
            for i in range(0, len(input_data), self.chunk_size - self.overlap):
                if i > 0:
                    i -= self.overlap
                yield input_data[i:i + self.chunk_size]
                
        elif self.split_strategy == "words":
            # Split by words
            words = input_data.split()
            for i in range(0, len(words), self.chunk_size - self.overlap):
                if i > 0:
                    i -= self.overlap
                yield " ".join(words[i:i + self.chunk_size])
                
        elif self.split_strategy == "lines":
            # Split by lines
            lines = input_data.splitlines()
            for i in range(0, len(lines), self.chunk_size - self.overlap):
                if i > 0:
                    i -= self.overlap
                yield "\n".join(lines[i:i + self.chunk_size])
                
        elif self.split_strategy == "paragraphs":
            # Split by paragraphs (double newlines)
            paragraphs = re.split(r"\n\s*\n", input_data)
            for i in range(0, len(paragraphs), self.chunk_size - self.overlap):
                if i > 0:
                    i -= self.overlap
                yield "\n\n".join(paragraphs[i:i + self.chunk_size])
                
        elif self.split_strategy == "sentences":
            # Split by sentences
            # Simple sentence splitting for demonstration purposes
            sentences = re.split(r"(?<=[.!?])\s+", input_data)
            for i in range(0, len(sentences), self.chunk_size - self.overlap):
                if i > 0:
                    i -= self.overlap
                yield " ".join(sentences[i:i + self.chunk_size])
                
        else:
            # Default to character splitting
            for i in range(0, len(input_data), self.chunk_size - self.overlap):
                if i > 0:
                    i -= self.overlap
                yield input_data[i:i + self.chunk_size]
    
    async def estimate_chunks(self, input_data: str) -> int:
        """Estimate the number of chunks.
        
        Args:
            input_data: Input text
            
        Returns:
            Estimated number of chunks
        """
        if not input_data:
            return 0
        
        if self.split_strategy == "chars":
            length = len(input_data)
        elif self.split_strategy == "words":
            length = len(input_data.split())
        elif self.split_strategy == "lines":
            length = len(input_data.splitlines())
        elif self.split_strategy == "paragraphs":
            length = len(re.split(r"\n\s*\n", input_data))
        elif self.split_strategy == "sentences":
            length = len(re.split(r"(?<=[.!?])\s+", input_data))
        else:
            length = len(input_data)
        
        effective_chunk_size = self.chunk_size - self.overlap
        if effective_chunk_size <= 0:
            effective_chunk_size = 1
            
        return (length + effective_chunk_size - 1) // effective_chunk_size


class FileSplitter(Splitter[str, bytes]):
    """Splitter for files.
    
    Splits a file into chunks of the specified size.
    """
    
    def __init__(
        self,
        chunk_size: int = 1024 * 1024,  # 1 MB
        name: str = "file_splitter",
    ):
        """Initialize a file splitter.
        
        Args:
            chunk_size: Maximum chunk size in bytes
            name: Splitter name
        """
        super().__init__(name)
        self.chunk_size = max(1, chunk_size)
    
    async def split(self, input_data: str) -> AsyncIterator[bytes]:
        """Split a file into chunks.
        
        Args:
            input_data: File path
            
        Yields:
            File chunks
        """
        if not os.path.exists(input_data):
            raise SplitterError(f"File not found: {input_data}", splitter=self.name)
        
        file_size = os.path.getsize(input_data)
        
        with open(input_data, "rb") as f:
            for _ in range(0, file_size, self.chunk_size):
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                yield chunk
    
    async def estimate_chunks(self, input_data: str) -> int:
        """Estimate the number of chunks.
        
        Args:
            input_data: File path
            
        Returns:
            Estimated number of chunks
        """
        if not os.path.exists(input_data):
            raise SplitterError(f"File not found: {input_data}", splitter=self.name)
        
        file_size = os.path.getsize(input_data)
        return (file_size + self.chunk_size - 1) // self.chunk_size


class JsonSplitter(Splitter[Union[str, Dict[str, Any]], Dict[str, Any]]):
    """Splitter for JSON data.
    
    Splits JSON data into chunks based on different strategies.
    """
    
    def __init__(
        self,
        chunk_size: int = 100,
        flatten_arrays: bool = True,
        name: str = "json_splitter",
    ):
        """Initialize a JSON splitter.
        
        Args:
            chunk_size: Maximum items per chunk
            flatten_arrays: Whether to flatten nested arrays
            name: Splitter name
        """
        super().__init__(name)
        self.chunk_size = max(1, chunk_size)
        self.flatten_arrays = flatten_arrays
    
    async def split(self, input_data: Union[str, Dict[str, Any]]) -> AsyncIterator[Dict[str, Any]]:
        """Split JSON data into chunks.
        
        Args:
            input_data: JSON data (string or dictionary)
            
        Yields:
            JSON chunks (dictionaries)
        """
        # Parse JSON if it's a string
        if isinstance(input_data, str):
            try:
                data = json.loads(input_data)
            except json.JSONDecodeError as e:
                raise SplitterError(f"Invalid JSON: {e}", splitter=self.name)
        else:
            data = input_data
        
        # Handle different JSON structures
        if isinstance(data, dict):
            # Split dictionary
            dict_splitter = DictionarySplitter(self.chunk_size)
            async for chunk in dict_splitter.split(data):
                yield chunk
                
        elif isinstance(data, list):
            # Split array
            if self.flatten_arrays:
                # Flatten nested arrays
                flat_items = self._flatten_array(data)
                list_splitter = ListSplitter(self.chunk_size)
                async for items_chunk in list_splitter.split(flat_items):
                    yield {"items": items_chunk}
            else:
                # Split array without flattening
                list_splitter = ListSplitter(self.chunk_size)
                async for items_chunk in list_splitter.split(data):
                    yield {"items": items_chunk}
                    
        else:
            # Single value, wrap in a dictionary
            yield {"value": data}
    
    def _flatten_array(self, arr: List[Any]) -> List[Any]:
        """Flatten a nested array.
        
        Args:
            arr: Nested array
            
        Returns:
            Flattened array
        """
        result = []
        
        for item in arr:
            if isinstance(item, list):
                result.extend(self._flatten_array(item))
            else:
                result.append(item)
                
        return result
    
    async def estimate_chunks(self, input_data: Union[str, Dict[str, Any]]) -> int:
        """Estimate the number of chunks.
        
        Args:
            input_data: JSON data
            
        Returns:
            Estimated number of chunks
        """
        # Parse JSON if it's a string
        if isinstance(input_data, str):
            try:
                data = json.loads(input_data)
            except json.JSONDecodeError:
                return 0
        else:
            data = input_data
        
        # Estimate based on data type
        if isinstance(data, dict):
            return (len(data) + self.chunk_size - 1) // self.chunk_size
        elif isinstance(data, list):
            if self.flatten_arrays:
                # Flatten nested arrays
                flat_items = self._flatten_array(data)
                return (len(flat_items) + self.chunk_size - 1) // self.chunk_size
            else:
                return (len(data) + self.chunk_size - 1) // self.chunk_size
        else:
            # Single value
            return 1


class DirectorySplitter(Splitter[str, str]):
    """Splitter for directories.
    
    Splits a directory into individual files.
    """
    
    def __init__(
        self,
        recursive: bool = True,
        file_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        name: str = "directory_splitter",
    ):
        """Initialize a directory splitter.
        
        Args:
            recursive: Whether to recursively scan subdirectories
            file_patterns: Patterns of files to include
            exclude_patterns: Patterns of files to exclude
            name: Splitter name
        """
        super().__init__(name)
        self.recursive = recursive
        self.file_patterns = file_patterns
        self.exclude_patterns = exclude_patterns
    
    async def split(self, input_data: str) -> AsyncIterator[str]:
        """Split a directory into files.
        
        Args:
            input_data: Directory path
            
        Yields:
            File paths
        """
        if not os.path.isdir(input_data):
            raise SplitterError(
                f"Not a directory: {input_data}",
                splitter=self.name,
            )
        
        # Get list of files
        for root, dirs, files in os.walk(input_data):
            # Skip subdirectories if not recursive
            if not self.recursive and root != input_data:
                continue
            
            for file in files:
                file_path = os.path.join(root, file)
                
                # Check if file matches include patterns
                if self.file_patterns:
                    if not any(self._match_pattern(file, pattern) for pattern in self.file_patterns):
                        continue
                
                # Check if file matches exclude patterns
                if self.exclude_patterns:
                    if any(self._match_pattern(file, pattern) for pattern in self.exclude_patterns):
                        continue
                
                yield file_path
    
    def _match_pattern(self, file: str, pattern: str) -> bool:
        """Check if a file matches a pattern.
        
        Args:
            file: File name
            pattern: Pattern to match
            
        Returns:
            True if the file matches the pattern, False otherwise
        """
        # Simple wildcard matching
        if pattern.startswith("*."):
            # Match file extension
            return file.endswith(pattern[1:])
        elif pattern.startswith("*"):
            # Match file suffix
            return file.endswith(pattern[1:])
        elif pattern.endswith("*"):
            # Match file prefix
            return file.startswith(pattern[:-1])
        else:
            # Exact match
            return file == pattern
    
    async def estimate_chunks(self, input_data: str) -> int:
        """Estimate the number of chunks (files).
        
        This is an expensive operation as it requires scanning the directory.
        
        Args:
            input_data: Directory path
            
        Returns:
            Estimated number of files
        """
        if not os.path.isdir(input_data):
            return 0
        
        count = 0
        async for _ in self.split(input_data):
            count += 1
        
        return count


# Splitter registry

_splitters: Dict[str, Splitter] = {
    "list": ListSplitter(),
    "dict": DictionarySplitter(),
    "text": TextSplitter(),
    "file": FileSplitter(),
    "json": JsonSplitter(),
    "directory": DirectorySplitter(),
}


def register_splitter(name: str, splitter: Splitter) -> None:
    """Register a splitter.
    
    Args:
        name: Splitter name
        splitter: Splitter instance
    """
    _splitters[name] = splitter
    logger.debug(f"Registered splitter: {name}")


def get_splitter(name: str) -> Optional[Splitter]:
    """Get a splitter by name.
    
    Args:
        name: Splitter name
        
    Returns:
        Splitter instance or None if not found
    """
    return _splitters.get(name)


def list_splitters() -> List[str]:
    """Get a list of all registered splitters.
    
    Returns:
        List of splitter names
    """
    return list(_splitters.keys())


# Convenience functions

async def split_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 0,
    strategy: str = "chars",
) -> List[str]:
    """Split text into chunks.
    
    Args:
        text: Text to split
        chunk_size: Maximum chunk size
        overlap: Overlap between chunks
        strategy: Splitting strategy
        
    Returns:
        List of text chunks
    """
    splitter = TextSplitter(chunk_size, overlap, strategy)
    
    chunks = []
    async for chunk in splitter.split(text):
        chunks.append(chunk)
    
    return chunks


async def split_file(
    file_path: str,
    chunk_size: int = 1024 * 1024,
) -> List[bytes]:
    """Split a file into chunks.
    
    Args:
        file_path: Path to the file
        chunk_size: Maximum chunk size in bytes
        
    Returns:
        List of file chunks
    """
    splitter = FileSplitter(chunk_size)
    
    chunks = []
    async for chunk in splitter.split(file_path):
        chunks.append(chunk)
    
    return chunks


async def split_directory(
    directory_path: str,
    recursive: bool = True,
    file_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> List[str]:
    """Split a directory into files.
    
    Args:
        directory_path: Path to the directory
        recursive: Whether to recursively scan subdirectories
        file_patterns: Patterns of files to include
        exclude_patterns: Patterns of files to exclude
        
    Returns:
        List of file paths
    """
    splitter = DirectorySplitter(recursive, file_patterns, exclude_patterns)
    
    file_paths = []
    async for file_path in splitter.split(directory_path):
        file_paths.append(file_path)
    
    return file_paths