"""
TSAP Incremental Streamer.

This module provides utilities for streaming input and output data in incremental
processing, supporting various streaming formats and protocols.
"""

import os
import json
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Any, Optional, Union, AsyncIterator, Type, TypeVar, Generic, BinaryIO, TextIO

from tsap.utils.logging import logger
from tsap.utils.errors import TSAPError


# Type variables for input and output
I = TypeVar('I')  # noqa: E741
O = TypeVar('O')  # noqa: E741


class StreamingError(TSAPError):
    """Error raised when streaming operations fail."""
    
    def __init__(
        self,
        message: str,
        streamer: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a streaming error.
        
        Args:
            message: Error message
            streamer: Streamer name
            details: Additional error details
        """
        error_details = details or {}
        if streamer:
            error_details["streamer"] = streamer
        
        super().__init__(message, "STREAMING_ERROR", error_details)


class InputStream(Generic[I], ABC):
    """Base class for input streams.
    
    Input streams provide a standardized interface for reading data from various sources.
    """
    
    def __init__(self, name: str):
        """Initialize an input stream.
        
        Args:
            name: Stream name
        """
        self.name = name
        self.closed = False
    
    @abstractmethod
    async def read(self) -> Optional[I]:
        """Read the next item from the stream.
        
        Returns:
            The next item or None if the stream is exhausted
        """
        pass
    
    @abstractmethod
    async def read_all(self) -> List[I]:
        """Read all remaining items from the stream.
        
        Returns:
            List of all remaining items
        """
        pass
    
    async def aiter(self) -> AsyncIterator[I]:
        """Get an asynchronous iterator for the stream.
        
        Yields:
            Items from the stream
        """
        while True:
            item = await self.read()
            if item is None:
                break
            yield item
    
    async def close(self) -> None:
        """Close the stream.
        
        This method should be overridden by subclasses to perform cleanup.
        """
        self.closed = True
    
    async def __aenter__(self) -> "InputStream[I]":
        """Enter the context manager.
        
        Returns:
            Self
        """
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager.
        
        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
        """
        await self.close()


class OutputStream(Generic[O], ABC):
    """Base class for output streams.
    
    Output streams provide a standardized interface for writing data to various destinations.
    """
    
    def __init__(self, name: str):
        """Initialize an output stream.
        
        Args:
            name: Stream name
        """
        self.name = name
        self.closed = False
    
    @abstractmethod
    async def write(self, item: O) -> None:
        """Write an item to the stream.
        
        Args:
            item: Item to write
        """
        pass
    
    @abstractmethod
    async def write_all(self, items: List[O]) -> None:
        """Write multiple items to the stream.
        
        Args:
            items: Items to write
        """
        pass
    
    async def flush(self) -> None:
        """Flush the stream.
        
        This method should be overridden by subclasses to perform flushing.
        """
        pass
    
    async def close(self) -> None:
        """Close the stream.
        
        This method should be overridden by subclasses to perform cleanup.
        """
        await self.flush()
        self.closed = True
    
    async def __aenter__(self) -> "OutputStream[O]":
        """Enter the context manager.
        
        Returns:
            Self
        """
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager.
        
        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
        """
        await self.close()


class ListInputStream(InputStream[I]):
    """Input stream from a list.
    
    Reads items from a pre-existing list.
    """
    
    def __init__(self, items: List[I], name: str = "list_input_stream"):
        """Initialize a list input stream.
        
        Args:
            items: List of items to read
            name: Stream name
        """
        super().__init__(name)
        self.items = items
        self.position = 0
    
    async def read(self) -> Optional[I]:
        """Read the next item from the stream.
        
        Returns:
            The next item or None if the stream is exhausted
        """
        if self.closed:
            raise StreamingError("Stream is closed", streamer=self.name)
        
        if self.position >= len(self.items):
            return None
        
        item = self.items[self.position]
        self.position += 1
        return item
    
    async def read_all(self) -> List[I]:
        """Read all remaining items from the stream.
        
        Returns:
            List of all remaining items
        """
        if self.closed:
            raise StreamingError("Stream is closed", streamer=self.name)
        
        remaining = self.items[self.position:]
        self.position = len(self.items)
        return remaining


class ListOutputStream(OutputStream[O]):
    """Output stream to a list.
    
    Writes items to a list that can be accessed later.
    """
    
    def __init__(self, name: str = "list_output_stream"):
        """Initialize a list output stream.
        
        Args:
            name: Stream name
        """
        super().__init__(name)
        self.items: List[O] = []
    
    async def write(self, item: O) -> None:
        """Write an item to the stream.
        
        Args:
            item: Item to write
        """
        if self.closed:
            raise StreamingError("Stream is closed", streamer=self.name)
        
        self.items.append(item)
    
    async def write_all(self, items: List[O]) -> None:
        """Write multiple items to the stream.
        
        Args:
            items: Items to write
        """
        if self.closed:
            raise StreamingError("Stream is closed", streamer=self.name)
        
        self.items.extend(items)
    
    def get_items(self) -> List[O]:
        """Get all written items.
        
        Returns:
            List of all written items
        """
        return self.items.copy()


class FileInputStream(InputStream[Union[str, bytes]]):
    """Input stream from a file.
    
    Reads lines or chunks from a file.
    """
    
    def __init__(
        self,
        file_path: str,
        binary: bool = False,
        chunk_size: int = 4096,
        read_mode: str = "line",
        name: str = "file_input_stream",
    ):
        """Initialize a file input stream.
        
        Args:
            file_path: Path to the file
            binary: Whether to read in binary mode
            chunk_size: Size of chunks to read in chunk mode
            read_mode: Reading mode ('line' or 'chunk')
            name: Stream name
        """
        super().__init__(name)
        self.file_path = file_path
        self.binary = binary
        self.chunk_size = chunk_size
        self.read_mode = read_mode
        self.file: Optional[Union[TextIO, BinaryIO]] = None
    
    async def _open(self) -> None:
        """Open the file if not already open."""
        if self.file is None:
            mode = "rb" if self.binary else "r"
            self.file = open(self.file_path, mode)
    
    async def read(self) -> Optional[Union[str, bytes]]:
        """Read the next line or chunk from the file.
        
        Returns:
            The next line/chunk or None if the end of file is reached
        """
        if self.closed:
            raise StreamingError("Stream is closed", streamer=self.name)
        
        await self._open()
        
        if self.read_mode == "line" and not self.binary:
            # Read a line
            line = self.file.readline()  # type: ignore
            return line if line else None
        else:
            # Read a chunk
            chunk = self.file.read(self.chunk_size)  # type: ignore
            return chunk if chunk else None
    
    async def read_all(self) -> List[Union[str, bytes]]:
        """Read all remaining lines or chunks from the file.
        
        Returns:
            List of all remaining lines/chunks
        """
        if self.closed:
            raise StreamingError("Stream is closed", streamer=self.name)
        
        await self._open()
        
        if self.read_mode == "line" and not self.binary:
            # Read all lines
            lines = self.file.readlines()  # type: ignore
            return lines
        else:
            # Read all remaining content as a single chunk
            content = self.file.read()  # type: ignore
            return [content] if content else []
    
    async def close(self) -> None:
        """Close the file."""
        if self.file is not None:
            self.file.close()
            self.file = None
        
        await super().close()


class FileOutputStream(OutputStream[Union[str, bytes]]):
    """Output stream to a file.
    
    Writes lines or chunks to a file.
    """
    
    def __init__(
        self,
        file_path: str,
        binary: bool = False,
        append: bool = False,
        name: str = "file_output_stream",
    ):
        """Initialize a file output stream.
        
        Args:
            file_path: Path to the file
            binary: Whether to write in binary mode
            append: Whether to append to an existing file
            name: Stream name
        """
        super().__init__(name)
        self.file_path = file_path
        self.binary = binary
        self.append = append
        self.file: Optional[Union[TextIO, BinaryIO]] = None
    
    async def _open(self) -> None:
        """Open the file if not already open."""
        if self.file is None:
            mode = "ab" if self.binary and self.append else "wb" if self.binary else "a" if self.append else "w"
            self.file = open(self.file_path, mode)
    
    async def write(self, item: Union[str, bytes]) -> None:
        """Write an item to the file.
        
        Args:
            item: Item to write
        """
        if self.closed:
            raise StreamingError("Stream is closed", streamer=self.name)
        
        await self._open()
        
        if isinstance(item, str) and self.binary:
            # Convert string to bytes in binary mode
            self.file.write(item.encode())  # type: ignore
        elif isinstance(item, bytes) and not self.binary:
            # Convert bytes to string in text mode
            self.file.write(item.decode())  # type: ignore
        else:
            # Write directly
            self.file.write(item)  # type: ignore
    
    async def write_all(self, items: List[Union[str, bytes]]) -> None:
        """Write multiple items to the file.
        
        Args:
            items: Items to write
        """
        if self.closed:
            raise StreamingError("Stream is closed", streamer=self.name)
        
        for item in items:
            await self.write(item)
    
    async def flush(self) -> None:
        """Flush the file."""
        if self.file is not None:
            self.file.flush()
    
    async def close(self) -> None:
        """Close the file."""
        if self.file is not None:
            self.file.close()
            self.file = None
        
        await super().close()


class JsonInputStream(InputStream[Any]):
    """Input stream for JSON data.
    
    Reads JSON data from various sources.
    """
    
    def __init__(
        self,
        source: Union[str, List[Dict[str, Any]], Dict[str, Any]],
        source_type: str = "auto",
        name: str = "json_input_stream",
    ):
        """Initialize a JSON input stream.
        
        Args:
            source: JSON data source (file path, JSON string, or parsed JSON data)
            source_type: Source type ('file', 'string', 'data', or 'auto')
            name: Stream name
        """
        super().__init__(name)
        self.source = source
        self.source_type = source_type
        self.data: Optional[Union[List[Any], Dict[str, Any]]] = None
        self.position = 0
    
    async def _load_data(self) -> None:
        """Load JSON data from the source."""
        if self.data is not None:
            return
        
        source_type = self.source_type
        
        # Auto-detect source type if not specified
        if source_type == "auto":
            if isinstance(self.source, str):
                if os.path.exists(self.source):
                    source_type = "file"
                else:
                    source_type = "string"
            else:
                source_type = "data"
        
        # Load data based on source type
        if source_type == "file":
            # Load from file
            with open(self.source, "r") as f:  # type: ignore
                self.data = json.load(f)
        elif source_type == "string":
            # Parse JSON string
            self.data = json.loads(self.source)  # type: ignore
        elif source_type == "data":
            # Use pre-parsed data
            self.data = self.source  # type: ignore
        else:
            raise StreamingError(
                f"Invalid source type: {source_type}",
                streamer=self.name,
            )
        
        # Convert single object to list if necessary
        if isinstance(self.data, dict):
            self.data = [self.data]
    
    async def read(self) -> Optional[Any]:
        """Read the next JSON object from the stream.
        
        Returns:
            The next JSON object or None if the stream is exhausted
        """
        if self.closed:
            raise StreamingError("Stream is closed", streamer=self.name)
        
        await self._load_data()
        
        if self.position >= len(self.data):  # type: ignore
            return None
        
        item = self.data[self.position]  # type: ignore
        self.position += 1
        return item
    
    async def read_all(self) -> List[Any]:
        """Read all remaining JSON objects from the stream.
        
        Returns:
            List of all remaining JSON objects
        """
        if self.closed:
            raise StreamingError("Stream is closed", streamer=self.name)
        
        await self._load_data()
        
        remaining = self.data[self.position:] if self.data else []  # type: ignore
        self.position = len(self.data) if self.data else 0  # type: ignore
        return remaining


class JsonOutputStream(OutputStream[Any]):
    """Output stream for JSON data.
    
    Writes JSON data to various destinations.
    """
    
    def __init__(
        self,
        destination: Optional[str] = None,
        format: str = "array",
        pretty: bool = True,
        name: str = "json_output_stream",
    ):
        """Initialize a JSON output stream.
        
        Args:
            destination: Output destination (file path or None for in-memory)
            format: Output format ('array', 'object', or 'lines')
            pretty: Whether to format with indentation
            name: Stream name
        """
        super().__init__(name)
        self.destination = destination
        self.format = format
        self.pretty = pretty
        self.items: List[Any] = []
        self.file: Optional[TextIO] = None
    
    async def _open_file(self) -> None:
        """Open the output file if not already open."""
        if self.destination and not self.file:
            self.file = open(self.destination, "w")
            
            # Write opening bracket for array format
            if self.format == "array":
                self.file.write("[\n" if self.pretty else "[")
    
    async def write(self, item: Any) -> None:
        """Write a JSON object to the stream.
        
        Args:
            item: JSON object to write
        """
        if self.closed:
            raise StreamingError("Stream is closed", streamer=self.name)
        
        if self.destination:
            # Write to file
            await self._open_file()
            
            if self.format == "array":
                # Write item as array element
                prefix = ",\n" if self.items else ""
                indent = "  " if self.pretty else ""
                json_str = json.dumps(item, indent=2 if self.pretty else None)
                indented_json = "\n".join(f"{indent}{line}" for line in json_str.splitlines())
                self.file.write(f"{prefix}{indented_json}")  # type: ignore
            elif self.format == "lines":
                # Write item as separate line
                json_str = json.dumps(item)
                self.file.write(f"{json_str}\n")  # type: ignore
            else:
                # Not supported for file output
                raise StreamingError(
                    f"Format '{self.format}' not supported for file output",
                    streamer=self.name,
                )
        
        # Store item in memory
        self.items.append(item)
    
    async def write_all(self, items: List[Any]) -> None:
        """Write multiple JSON objects to the stream.
        
        Args:
            items: JSON objects to write
        """
        if self.closed:
            raise StreamingError("Stream is closed", streamer=self.name)
        
        for item in items:
            await self.write(item)
    
    async def flush(self) -> None:
        """Flush the output file."""
        if self.file:
            self.file.flush()
    
    async def close(self) -> None:
        """Close the stream and finalize output."""
        if self.file:
            # Write closing bracket for array format
            if self.format == "array":
                self.file.write("\n]" if self.pretty else "]")
            
            # Close the file
            self.file.close()
            self.file = None
        
        await super().close()
    
    def get_result(self) -> Any:
        """Get the final result.
        
        Returns:
            JSON result based on the specified format
        """
        if self.format == "array":
            return self.items
        elif self.format == "object":
            # Merge objects
            result = {}
            for item in self.items:
                if isinstance(item, dict):
                    result.update(item)
            return result
        elif self.format == "lines":
            return self.items
        else:
            return self.items


# Stream registry

_input_streams: Dict[str, Type[InputStream]] = {
    "list": ListInputStream,
    "file": FileInputStream,
    "json": JsonInputStream,
}

_output_streams: Dict[str, Type[OutputStream]] = {
    "list": ListOutputStream,
    "file": FileOutputStream,
    "json": JsonOutputStream,
}


def register_input_stream(name: str, stream_class: Type[InputStream]) -> None:
    """Register an input stream class.
    
    Args:
        name: Stream name
        stream_class: Stream class
    """
    _input_streams[name] = stream_class
    logger.debug(f"Registered input stream: {name}")


def register_output_stream(name: str, stream_class: Type[OutputStream]) -> None:
    """Register an output stream class.
    
    Args:
        name: Stream name
        stream_class: Stream class
    """
    _output_streams[name] = stream_class
    logger.debug(f"Registered output stream: {name}")


def get_input_stream(
    name: str,
    *args: Any,
    **kwargs: Any,
) -> Optional[InputStream]:
    """Get an input stream by name.
    
    Args:
        name: Stream name
        *args: Stream constructor positional arguments
        **kwargs: Stream constructor keyword arguments
        
    Returns:
        Stream instance or None if not found
    """
    stream_class = _input_streams.get(name)
    if not stream_class:
        return None
    
    return stream_class(*args, **kwargs)


def get_output_stream(
    name: str,
    *args: Any,
    **kwargs: Any,
) -> Optional[OutputStream]:
    """Get an output stream by name.
    
    Args:
        name: Stream name
        *args: Stream constructor positional arguments
        **kwargs: Stream constructor keyword arguments
        
    Returns:
        Stream instance or None if not found
    """
    stream_class = _output_streams.get(name)
    if not stream_class:
        return None
    
    return stream_class(*args, **kwargs)


def list_input_streams() -> List[str]:
    """Get a list of all registered input streams.
    
    Returns:
        List of input stream names
    """
    return list(_input_streams.keys())


def list_output_streams() -> List[str]:
    """Get a list of all registered output streams.
    
    Returns:
        List of output stream names
    """
    return list(_output_streams.keys())


# Convenience functions

async def stream_pipeline(
    input_stream: InputStream[I],
    output_stream: OutputStream[O],
    transform_func: Optional[Callable[[I], O]] = None,
) -> int:
    """Process a stream pipeline.
    
    Reads items from an input stream, optionally transforms them, and writes them to an output stream.
    
    Args:
        input_stream: Input stream
        output_stream: Output stream
        transform_func: Function to transform input items to output items
        
    Returns:
        Number of items processed
    """
    count = 0
    
    async for item in input_stream.aiter():
        # Transform item if a transform function is provided
        if transform_func:
            output_item = transform_func(item)
        else:
            output_item = item  # type: ignore
        
        # Write to output stream
        await output_stream.write(output_item)
        count += 1
    
    return count


async def stream_json_file(
    input_path: str,
    output_path: str,
    transform_func: Optional[Callable[[Any], Any]] = None,
) -> int:
    """Stream JSON data from a file to another file.
    
    Args:
        input_path: Input file path
        output_path: Output file path
        transform_func: Function to transform JSON objects
        
    Returns:
        Number of JSON objects processed
    """
    async with JsonInputStream(input_path, "file") as input_stream:
        async with JsonOutputStream(output_path) as output_stream:
            return await stream_pipeline(input_stream, output_stream, transform_func)


async def stream_text_file(
    input_path: str,
    output_path: str,
    transform_func: Optional[Callable[[str], str]] = None,
) -> int:
    """Stream lines from a text file to another text file.
    
    Args:
        input_path: Input file path
        output_path: Output file path
        transform_func: Function to transform lines
        
    Returns:
        Number of lines processed
    """
    async with FileInputStream(input_path, binary=False, read_mode="line") as input_stream:
        async with FileOutputStream(output_path, binary=False) as output_stream:
            return await stream_pipeline(input_stream, output_stream, transform_func)