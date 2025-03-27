"""
Filesystem utilities for TSAP.

This module provides utility functions for working with files and directories,
including file type detection, path normalization, and content operations.
"""
import os
import re
import tarfile
import tempfile
import hashlib
import mimetypes
import zipfile
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
import fnmatch



# Initialize mime types
mimetypes.init()

# File type groups
TEXT_EXTENSIONS = {
    '.txt', '.md', '.markdown', '.rst', '.csv', '.tsv',
    '.json', '.yaml', '.yml', '.xml', '.html', '.htm',
    '.css', '.js', '.py', '.java', '.c', '.cpp', '.h',
    '.rb', '.php', '.go', '.ts', '.sh', '.bat', '.ps1',
    '.sql', '.conf', '.ini', '.cfg', '.properties',
}

BINARY_EXTENSIONS = {
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.zip', '.tar', '.gz', '.bz2', '.xz', '.7z', '.rar',
    '.exe', '.dll', '.so', '.dylib', '.bin', '.dat',
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp',
    '.mp3', '.mp4', '.wav', '.avi', '.mkv', '.mov',
    '.db', '.sqlite', '.sqlite3',
}


def is_text_file(path: str) -> bool:
    """Check if a file is a text file.
    
    This uses a combination of extension checks and content sampling.
    
    Args:
        path: Path to the file
        
    Returns:
        Whether the file is a text file
    """
    # Check extension first
    ext = os.path.splitext(path)[1].lower()
    
    if ext in TEXT_EXTENSIONS:
        return True
    
    if ext in BINARY_EXTENSIONS:
        return False
    
    # Try to read as text
    try:
        with open(path, 'r', encoding='utf-8') as f:
            # Read a small sample
            sample = f.read(4096)
            
            # Check for common binary characters
            if b'\x00' in sample.encode('utf-8'):
                return False
                
            # If we got here, it's probably text
            return True
    except (UnicodeDecodeError, IsADirectoryError, PermissionError):
        return False


def detect_mime_type(path: str) -> Tuple[str, bool]:
    """Detect the MIME type of a file.
    
    Args:
        path: Path to the file
        
    Returns:
        Tuple of (mime_type, is_text)
    """
    # Use mimetypes to get mime type
    mime_type, _ = mimetypes.guess_type(path)
    
    # Default to text/plain for unknown types
    if not mime_type:
        mime_type = 'application/octet-stream'
        if is_text_file(path):
            mime_type = 'text/plain'
    
    # Determine if it's a text type
    is_text = mime_type.startswith('text/') or mime_type in [
        'application/json',
        'application/xml',
        'application/javascript',
        'application/x-python',
        'application/x-ruby',
        'application/x-sh',
        'application/x-php',
    ]
    
    return mime_type, is_text


def hash_file(path: str, algorithm: str = 'sha256') -> str:
    """Generate a hash for a file.
    
    Args:
        path: Path to the file
        algorithm: Hash algorithm (md5, sha1, sha256, etc.)
        
    Returns:
        Hex digest of the hash
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If algorithm is invalid
    """
    hash_func = getattr(hashlib, algorithm, None)
    
    if not hash_func:
        raise ValueError(f"Invalid hash algorithm: {algorithm}")
    
    h = hash_func()
    
    with open(path, 'rb') as f:
        # Read and update in chunks to avoid memory issues
        for chunk in iter(lambda: f.read(16384), b''):
            h.update(chunk)
    
    return h.hexdigest()


def get_file_info(path: str) -> Dict[str, Any]:
    """Get comprehensive information about a file.
    
    Args:
        path: Path to the file
        
    Returns:
        Dictionary with file information
        
    Raises:
        FileNotFoundError: If the file does not exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    # Get basic file stats
    stat = os.stat(path)
    
    # Get mime type
    mime_type, is_text = detect_mime_type(path)
    
    # Calculate hash for small files
    file_hash = None
    if stat.st_size < 10 * 1024 * 1024:  # 10 MB
        try:
            file_hash = hash_file(path)
        except Exception:
            pass
    
    # Return file info
    return {
        'path': path,
        'name': os.path.basename(path),
        'directory': os.path.dirname(path),
        'size': stat.st_size,
        'modified_time': stat.st_mtime,
        'created_time': stat.st_ctime,
        'is_directory': os.path.isdir(path),
        'is_file': os.path.isfile(path),
        'is_symlink': os.path.islink(path),
        'is_text': is_text,
        'mime_type': mime_type,
        'extension': os.path.splitext(path)[1].lower(),
        'permissions': stat.st_mode & 0o777,
        'hash': file_hash,
    }


def find_files(
    directory: str,
    patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    recursive: bool = True,
    follow_symlinks: bool = False,
    max_depth: Optional[int] = None,
) -> List[str]:
    """Find files matching patterns in a directory.
    
    Args:
        directory: Directory to search
        patterns: Optional list of patterns to match (glob style)
        exclude_patterns: Optional list of patterns to exclude
        recursive: Whether to search recursively
        follow_symlinks: Whether to follow symbolic links
        max_depth: Maximum directory depth to search
        
    Returns:
        List of matching file paths
        
    Raises:
        FileNotFoundError: If the directory does not exist
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    # Normalize directory path
    directory = os.path.abspath(directory)
    
    # Default patterns
    if patterns is None:
        patterns = ['*']
    
    if exclude_patterns is None:
        exclude_patterns = []
    
    matches = []
    
    # Convert patterns to compiled regexes
    compiled_patterns = [fnmatch.translate(p) for p in patterns]
    compiled_excludes = [fnmatch.translate(p) for p in exclude_patterns]
    
    pattern_regexes = [re.compile(p) for p in compiled_patterns]
    exclude_regexes = [re.compile(p) for p in compiled_excludes]
    
    # Walk directory
    for root, dirs, files in os.walk(directory, topdown=True, followlinks=follow_symlinks):
        # Check depth
        if max_depth is not None:
            rel_path = os.path.relpath(root, directory)
            depth = len(rel_path.split(os.sep)) if rel_path != '.' else 0
            if depth > max_depth:
                dirs.clear()  # Don't descend any further
                continue
        
        # Process files
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, directory)
            
            # Check patterns
            if any(r.match(file) for r in pattern_regexes) or \
               any(r.match(rel_path) for r in pattern_regexes):
                # Check exclude patterns
                if not any(r.match(file) for r in exclude_regexes) and \
                   not any(r.match(rel_path) for r in exclude_regexes):
                    matches.append(full_path)
        
        # Stop recursion if not recursive
        if not recursive:
            dirs.clear()
    
    return matches


async def read_file_async(
    path: str, 
    encoding: Optional[str] = None,
    mode: str = 'r',
) -> Union[str, bytes]:
    """Read a file asynchronously.
    
    Args:
        path: Path to the file
        encoding: Optional encoding (defaults to utf-8 for text mode)
        mode: File mode ('r' for text, 'rb' for binary)
        
    Returns:
        File contents as string or bytes
        
    Raises:
        FileNotFoundError: If the file does not exist
        UnicodeDecodeError: If the file cannot be decoded
    """
    if 'b' in mode:
        # Binary mode
        return await _read_binary_file_async(path)
    else:
        # Text mode
        return await _read_text_file_async(path, encoding or 'utf-8')


async def _read_text_file_async(path: str, encoding: str = 'utf-8') -> str:
    """Read a text file asynchronously.
    
    Args:
        path: Path to the file
        encoding: File encoding
        
    Returns:
        File contents as string
        
    Raises:
        FileNotFoundError: If the file does not exist
        UnicodeDecodeError: If the file cannot be decoded
    """
    # Use asyncio.to_thread in Python 3.9+
    loop = asyncio.get_event_loop()
    
    def read_file():
        with open(path, 'r', encoding=encoding) as f:
            return f.read()
    
    return await loop.run_in_executor(None, read_file)


async def _read_binary_file_async(path: str) -> bytes:
    """Read a binary file asynchronously.
    
    Args:
        path: Path to the file
        
    Returns:
        File contents as bytes
        
    Raises:
        FileNotFoundError: If the file does not exist
    """
    # Use asyncio.to_thread in Python 3.9+
    loop = asyncio.get_event_loop()
    
    def read_file():
        with open(path, 'rb') as f:
            return f.read()
    
    return await loop.run_in_executor(None, read_file)


async def write_file_async(
    path: str,
    content: Union[str, bytes],
    encoding: Optional[str] = None,
    mode: str = 'w',
) -> int:
    """Write to a file asynchronously.
    
    Args:
        path: Path to the file
        content: Content to write
        encoding: Optional encoding (defaults to utf-8 for text mode)
        mode: File mode ('w' for text, 'wb' for binary)
        
    Returns:
        Number of bytes written
        
    Raises:
        PermissionError: If the file cannot be written
    """
    # Make sure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    if isinstance(content, str) and 'b' in mode:
        # Convert string to bytes for binary mode
        content_bytes = content.encode(encoding or 'utf-8')
    elif isinstance(content, bytes) and 'b' not in mode:
        # Convert bytes to string for text mode
        content_str = content.decode(encoding or 'utf-8')
        content = content_str
    elif isinstance(content, bytes):
        content_bytes = content
    
    # Use asyncio.to_thread in Python 3.9+
    loop = asyncio.get_event_loop()
    
    def write_file():
        with open(path, mode) as f:
            if isinstance(content, str):
                return f.write(content)
            else:
                return f.write(content_bytes)
    
    return await loop.run_in_executor(None, write_file)


def create_temp_dir() -> str:
    """Create a temporary directory.
    
    Returns:
        Path to the temporary directory
    """
    return tempfile.mkdtemp(prefix="tsap_")


def create_temp_file(
    content: Optional[Union[str, bytes]] = None,
    suffix: str = '',
    prefix: str = 'tsap_',
    delete: bool = False,
) -> str:
    """Create a temporary file with optional content.
    
    Args:
        content: Optional content to write to the file
        suffix: File suffix
        prefix: File prefix
        delete: Whether to delete the file when closed
        
    Returns:
        Path to the temporary file
    """
    if content is None:
        # Just create an empty temp file
        fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
        os.close(fd)
        return path
    
    # Create with content
    if isinstance(content, str):
        mode = 'w'
    else:
        mode = 'wb'
    
    with tempfile.NamedTemporaryFile(mode=mode, suffix=suffix, 
                                     prefix=prefix, delete=delete) as f:
        f.write(content)
        f.flush()
        return f.name


def extract_archive(
    archive_path: str,
    extract_dir: Optional[str] = None,
) -> str:
    """Extract an archive file.
    
    Args:
        archive_path: Path to the archive file
        extract_dir: Optional directory to extract to
        
    Returns:
        Path to the extraction directory
        
    Raises:
        ValueError: If the archive format is not supported
        FileNotFoundError: If the archive file does not exist
    """
    if not os.path.exists(archive_path):
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    
    # Create extraction directory if not specified
    if extract_dir is None:
        extract_dir = create_temp_dir()
    else:
        os.makedirs(extract_dir, exist_ok=True)
    
    # Determine archive type
    lower_path = archive_path.lower()
    
    if lower_path.endswith('.zip'):
        # Extract ZIP archive
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    elif lower_path.endswith(('.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2')):
        # Extract TAR archive
        mode = 'r'
        if lower_path.endswith(('.tar.gz', '.tgz')):
            mode = 'r:gz'
        elif lower_path.endswith(('.tar.bz2', '.tbz2')):
            mode = 'r:bz2'
        
        with tarfile.open(archive_path, mode) as tar_ref:
            tar_ref.extractall(extract_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")
    
    return extract_dir