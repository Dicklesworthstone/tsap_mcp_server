"""
Security-related utilities for the TSAP MCP Server.

This module provides functions for input sanitization, secure file handling,
permission checks, and other security-related tasks.
"""

import os
import re
import stat
import base64
import hashlib
import secrets
import ipaddress
import subprocess
from typing import List, Dict, Optional, Tuple

from tsap.utils.logging import debug, warning
from tsap.utils.errors import TSAPError
from tsap.utils.filesystem import is_text_file
from tsap.config import get_config


class SecurityError(TSAPError):
    """Exception raised for security-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.details = details or {}


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to remove potentially dangerous characters.
    
    Args:
        filename: The filename to sanitize
        
    Returns:
        A sanitized version of the filename
    """
    # Replace directory traversal sequences
    sanitized = re.sub(r'[\\/*?:"<>|]', '_', filename)
    
    # Replace any consecutive dots (which might be used to hide file extensions)
    sanitized = re.sub(r'\.{2,}', '.', sanitized)
    
    # Ensure the filename doesn't start with a dot (hidden file in Unix)
    if sanitized.startswith('.'):
        sanitized = '_' + sanitized
        
    # Ensure the filename isn't empty after sanitization
    if not sanitized:
        sanitized = "unnamed_file"
    
    return sanitized


def sanitize_path(path: str, allow_relative: bool = False) -> str:
    """
    Sanitize a file path to remove potentially dangerous elements.
    
    Args:
        path: The path to sanitize
        allow_relative: Whether to allow relative paths
        
    Returns:
        A sanitized version of the path
    """
    # Convert to absolute path if needed
    if not allow_relative and not os.path.isabs(path):
        path = os.path.abspath(path)
    
    # Split into components and sanitize each part
    parts = path.split(os.sep)
    sanitized_parts = [sanitize_filename(part) for part in parts if part]
    
    # Handle relative path component
    if os.path.isabs(path):
        # Preserve the leading slash for absolute paths
        result = os.sep + os.path.join(*sanitized_parts)
    else:
        result = os.path.join(*sanitized_parts)
    
    return result


def is_path_allowed(path: str, allowed_dirs: Optional[List[str]] = None) -> bool:
    """
    Check if a path is within allowed directories.
    
    Args:
        path: The path to check
        allowed_dirs: List of allowed directories. If None, uses configured safe directories.
        
    Returns:
        True if the path is allowed, False otherwise
    """
    if allowed_dirs is None:
        # Use configured safe directories
        config = get_config()
        allowed_dirs = config.tools.safe_directories
    
    # Resolve to absolute path
    abs_path = os.path.abspath(path)
    
    # Check against each allowed directory
    for allowed_dir in allowed_dirs:
        allowed_abs = os.path.abspath(allowed_dir)
        
        # Check if the path is within the allowed directory
        if abs_path.startswith(allowed_abs + os.sep) or abs_path == allowed_abs:
            return True
            
    return False


def check_file_permissions(path: str, require_readable: bool = True, 
                          require_writable: bool = False, 
                          require_executable: bool = False) -> bool:
    """
    Check if a file has the required permissions.
    
    Args:
        path: The path to check
        require_readable: Whether the file must be readable
        require_writable: Whether the file must be writable
        require_executable: Whether the file must be executable
        
    Returns:
        True if the file has the required permissions, False otherwise
    """
    if not os.path.exists(path):
        return False
    
    # Check read permission
    if require_readable and not os.access(path, os.R_OK):
        return False
    
    # Check write permission
    if require_writable and not os.access(path, os.W_OK):
        return False
    
    # Check execute permission
    if require_executable and not os.access(path, os.X_OK):
        return False
    
    return True


def secure_delete_file(path: str, passes: int = 1) -> bool:
    """
    Securely delete a file by overwriting its contents before deletion.
    
    Args:
        path: The path to the file to delete
        passes: Number of overwrite passes
        
    Returns:
        True if the file was successfully deleted, False otherwise
    """
    if not os.path.isfile(path):
        warning(f"Cannot secure delete non-existent file: {path}", component="security")
        return False
    
    try:
        file_size = os.path.getsize(path)
        
        # Only perform secure deletion on reasonable file sizes
        if file_size > 100 * 1024 * 1024:  # 100 MB
            warning(f"File too large for secure deletion: {path}", component="security")
            os.remove(path)
            return True
        
        # Overwrite the file with random data
        with open(path, "rb+") as f:
            for _ in range(passes):
                f.seek(0)
                random_data = secrets.token_bytes(file_size)
                f.write(random_data)
                f.flush()
                os.fsync(f.fileno())
            
            # Final pass with zeros
            f.seek(0)
            f.write(b'\x00' * file_size)
            f.flush()
            os.fsync(f.fileno())
        
        # Delete the file
        os.remove(path)
        debug(f"Securely deleted file: {path}", component="security")
        return True
        
    except Exception as e:
        warning(f"Failed to securely delete file {path}: {str(e)}", component="security")
        # Attempt regular deletion as fallback
        try:
            os.remove(path)
            return True
        except Exception:
            return False


def is_safe_regex(pattern: str, timeout_seconds: int = 1) -> bool:
    """
    Check if a regular expression is safe to use (not vulnerable to ReDoS).
    
    Args:
        pattern: The regex pattern to check
        timeout_seconds: Maximum time allowed for regex compilation and test
        
    Returns:
        True if the regex is considered safe, False otherwise
    """
    # List of potentially problematic regex patterns
    dangerous_patterns = [
        r'(a+)+',  # Nested repetition
        r'([a-zA-Z0-9])+\1+',  # Backreferences with repetition
        r'(a|a?)+',  # Repetition of optional groups
        r'(.*a){10}',  # Large repetition of unbounded match
        r'(a|b|c)*\w+'  # Multiple alternation with repetition
    ]
    
    # Check for known problematic patterns
    for dangerous in dangerous_patterns:
        if re.search(dangerous, pattern):
            warning(f"Potentially unsafe regex pattern detected: {pattern}", component="security")
            return False
    
    # Test the regex compilation and matching with timeout
    try:
        # Compile the regex
        compiled = re.compile(pattern)  # noqa: F841
        
        # Test against sample strings
        test_strings = [
            "",
            "a" * 100,
            "aaaaaaaaaaaaabbbbbbbbbbbb",
            "a" * 1000 + "b" * 1000
        ]
        
        for test_str in test_strings:
            # Test with timeout using subprocess to prevent blocking
            cmd = [
                "python", "-c", 
                f"import re; import sys; print(bool(re.search(r'''{pattern}''', '''{test_str}''')))"
            ]
            
            process = subprocess.run(
                cmd,
                timeout=timeout_seconds,
                capture_output=True,
                text=True
            )
            
            if process.returncode != 0:
                warning(f"Regex pattern failed testing: {pattern}", component="security")
                return False
        
        return True
        
    except (re.error, subprocess.TimeoutExpired) as e:
        warning(f"Regex pattern {pattern} is unsafe: {str(e)}", component="security")
        return False


def sanitize_command_args(args: List[str], allow_shell_operators: bool = False) -> List[str]:
    """
    Sanitize command-line arguments to prevent command injection.
    
    Args:
        args: List of command arguments to sanitize
        allow_shell_operators: Whether to allow shell operators
        
    Returns:
        Sanitized list of command arguments
    """
    if not args:
        return []
    
    sanitized = []
    shell_operators = ['|', '&', ';', '$', '`', '>', '<', '(', ')', '{', '}', '[', ']', '&&', '||', ';;', '$?']
    
    for arg in args:
        if isinstance(arg, str):
            # Replace null bytes
            clean_arg = arg.replace('\0', '')
            
            # Check for shell operators
            if not allow_shell_operators:
                for op in shell_operators:
                    if op in clean_arg:
                        raise SecurityError(f"Forbidden shell operator in command: {op}", 
                                          {"argument": arg})
            
            sanitized.append(clean_arg)
        else:
            # Convert non-string arguments to strings
            sanitized.append(str(arg))
    
    return sanitized


def generate_secure_token(length: int = 32) -> str:
    """
    Generate a cryptographically secure random token.
    
    Args:
        length: Length of the token in bytes
        
    Returns:
        A secure token as a URL-safe base64 string
    """
    # Generate random bytes
    random_bytes = secrets.token_bytes(length)
    
    # Encode as URL-safe base64
    token = base64.urlsafe_b64encode(random_bytes).decode('ascii')
    
    # Remove padding characters
    token = token.rstrip('=')
    
    return token


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """
    Hash a password using a secure method (Argon2 if available, else PBKDF2).
    
    Args:
        password: The password to hash
        salt: Optional salt (generated if not provided)
        
    Returns:
        Tuple of (hash, salt)
    """
    # Generate salt if not provided
    if salt is None:
        salt = secrets.token_hex(16)
    
    # Try to use Argon2 if available
    try:
        from argon2 import PasswordHasher
        ph = PasswordHasher()
        hash_value = ph.hash(password + salt)
        return hash_value, salt
    except ImportError:
        # Fall back to PBKDF2 with SHA-256
        dk = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            iterations=100000
        )
        hash_value = base64.b64encode(dk).decode('ascii')
        return hash_value, salt


def is_safe_input(input_str: str, max_length: int = 10000) -> bool:
    """
    Check if an input string is safe (not too long and contains no control chars).
    
    Args:
        input_str: The input string to check
        max_length: Maximum allowed length
        
    Returns:
        True if the input is safe, False otherwise
    """
    # Check length
    if len(input_str) > max_length:
        return False
    
    # Check for control characters (except common whitespace)
    allowed_control = {'\t', '\n', '\r'}
    control_chars = {c for c in input_str if c < ' ' and c not in allowed_control}
    
    return len(control_chars) == 0


def scan_file_for_security_issues(file_path: str, scan_types: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
    """
    Scan a file for potential security issues.
    
    Args:
        file_path: Path to the file to scan
        scan_types: List of scan types to perform (e.g., ['secrets', 'malware', 'backdoors'])
        
    Returns:
        Dictionary of found issues grouped by type
    """
    if scan_types is None:
        scan_types = ['secrets']
    
    results = {scan_type: [] for scan_type in scan_types}
    
    if not os.path.isfile(file_path):
        return results
    
    # Only scan text files
    if not is_text_file(file_path):
        debug(f"Skipping scan of non-text file: {file_path}", component="security")
        return results
    
    try:
        with open(file_path, 'r', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        warning(f"Could not read file for security scan: {file_path} - {str(e)}", component="security")
        return results
    
    # Check for secrets if requested
    if 'secrets' in scan_types:
        # Define patterns to search for
        secret_patterns = {
            'api_key': r'(?i)(?:api[_-]?key|access[_-]?key|secret[_-]?key)[\'"\s]*(?:=|:)[\'"\s]*([a-zA-Z0-9_\-\.]{16,64})',
            'aws_key': r'(?i)(?:aws[_-]?access[_-]?key|aws[_-]?secret)[\'"\s]*(?:=|:)[\'"\s]*([a-zA-Z0-9/+]{20,40})',
            'password': r'(?i)(?:password|passwd|pwd)[\'"\s]*(?:=|:)[\'"\s]*([^\'"\s]{8,})',
            'private_key': r'-----BEGIN (?:RSA|DSA|EC|OPENSSH) PRIVATE KEY-----',
            'token': r'(?i)(?:token|auth[_-]?token|jwt)[\'"\s]*(?:=|:)[\'"\s]*([a-zA-Z0-9_\-\.]{16,})',
        }
        
        for secret_type, pattern in secret_patterns.items():
            matches = re.findall(pattern, content)
            for match in matches:
                # Don't include the actual secret in the results
                results['secrets'].append({
                    'type': secret_type,
                    'file': file_path,
                    'risk': 'high',
                })
                # Only report one instance per file per type
                break
    
    # Additional scan types could be implemented here
    
    return results


def is_valid_ip(ip_str: str) -> bool:
    """
    Check if a string is a valid IP address (IPv4 or IPv6).
    
    Args:
        ip_str: String to check
        
    Returns:
        True if the string is a valid IP address, False otherwise
    """
    try:
        ipaddress.ip_address(ip_str)
        return True
    except ValueError:
        return False


def make_file_private(path: str) -> bool:
    """
    Make a file readable and writable only by the owner.
    
    Args:
        path: Path to the file
        
    Returns:
        True if the permissions were set successfully, False otherwise
    """
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        return True
    except Exception as e:
        warning(f"Failed to set private permissions on {path}: {str(e)}", component="security")
        return False


def check_file_hash(file_path: str, expected_hash: str, hash_type: str = 'sha256') -> bool:
    """
    Check if a file matches an expected hash.
    
    Args:
        file_path: Path to the file
        expected_hash: Expected hash value
        hash_type: Hash algorithm to use ('md5', 'sha1', 'sha256', 'sha512')
        
    Returns:
        True if the file's hash matches the expected hash, False otherwise
    """
    if not os.path.isfile(file_path):
        return False
    
    hash_funcs = {
        'md5': hashlib.md5,
        'sha1': hashlib.sha1,
        'sha256': hashlib.sha256,
        'sha512': hashlib.sha512
    }
    
    if hash_type not in hash_funcs:
        raise ValueError(f"Unsupported hash type: {hash_type}")
    
    hash_func = hash_funcs[hash_type]
    
    try:
        with open(file_path, 'rb') as f:
            file_hash = hash_func()
            # Read in chunks to handle large files
            chunk = f.read(8192)
            while chunk:
                file_hash.update(chunk)
                chunk = f.read(8192)
        
        calculated_hash = file_hash.hexdigest()
        return calculated_hash.lower() == expected_hash.lower()
        
    except Exception as e:
        warning(f"Failed to check file hash for {file_path}: {str(e)}", component="security")
        return False