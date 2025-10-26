#!/usr/bin/env python3
"""
File utilities for cross-platform path handling and file validation.

Author: Ian Philip Eglin  
Date: 2025-10-25
"""

import os
import hashlib
from pathlib import Path
from typing import Union, Optional, Dict, Any

from .error_handling import PathResolutionError, DataNotFoundError


def resolve_platform_path(base_path: Union[str, Path], relative_path: str) -> Path:
    """
    Resolve relative path against base path with platform awareness.
    
    Args:
        base_path: Base directory path
        relative_path: Relative path to resolve
        
    Returns:
        Resolved Path object
        
    Raises:
        PathResolutionError: If path resolution fails
    """
    try:
        base = Path(base_path)
        resolved = base / relative_path
        return resolved.resolve()
    except Exception as e:
        raise PathResolutionError(f"Failed to resolve path '{relative_path}' against '{base_path}': {e}")


def check_file_exists(file_path: Union[str, Path]) -> bool:
    """
    Check if file exists with proper error handling.
    
    Args:
        file_path: Path to check
        
    Returns:
        True if file exists and is readable
    """
    try:
        path = Path(file_path)
        return path.exists() and path.is_file()
    except (OSError, PermissionError):
        return False


def check_file_integrity(file_path: Union[str, Path], expected_size: Optional[int] = None) -> Dict[str, Any]:
    """
    Check file integrity and gather basic metadata.
    
    Args:
        file_path: Path to file
        expected_size: Expected file size in bytes (optional)
        
    Returns:
        Dictionary with integrity information
        
    Raises:
        DataNotFoundError: If file doesn't exist
    """
    path = Path(file_path)
    
    if not path.exists():
        raise DataNotFoundError(f"File not found: {path}")
    
    if not path.is_file():
        raise DataNotFoundError(f"Path is not a file: {path}")
    
    try:
        stat = path.stat()
        file_size = stat.st_size
        
        integrity_info = {
            'path': str(path),
            'exists': True,
            'size_bytes': file_size,
            'size_mb': round(file_size / (1024 * 1024), 2),
            'readable': os.access(path, os.R_OK),
            'modified_time': stat.st_mtime
        }
        
        # Check expected size if provided
        if expected_size is not None:
            integrity_info['size_match'] = (file_size == expected_size)
            integrity_info['expected_size'] = expected_size
        
        # Basic file format validation for common types
        suffix = path.suffix.lower()
        if suffix == '.h5':
            integrity_info['format'] = 'hdf5'
        elif suffix == '.csv':
            integrity_info['format'] = 'csv'
        elif suffix == '.tsv':
            integrity_info['format'] = 'tsv'
        elif suffix == '.json':
            integrity_info['format'] = 'json'
        else:
            integrity_info['format'] = 'unknown'
        
        return integrity_info
        
    except (OSError, PermissionError) as e:
        raise DataNotFoundError(f"Cannot access file {path}: {e}")


def get_file_hash(file_path: Union[str, Path], algorithm: str = 'md5') -> str:
    """
    Calculate file hash for integrity verification.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
        
    Returns:
        Hexadecimal hash string
        
    Raises:
        DataNotFoundError: If file doesn't exist or can't be read
    """
    path = Path(file_path)
    
    if not path.exists():
        raise DataNotFoundError(f"File not found: {path}")
    
    # Choose hash algorithm
    if algorithm == 'md5':
        hasher = hashlib.md5()
    elif algorithm == 'sha1':
        hasher = hashlib.sha1()
    elif algorithm == 'sha256':
        hasher = hashlib.sha256()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    try:
        with open(path, 'rb') as f:
            # Read file in chunks to handle large files
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except (OSError, PermissionError) as e:
        raise DataNotFoundError(f"Cannot read file {path}: {e}")


def safe_path_join(*parts: Union[str, Path]) -> Path:
    """
    Safely join path components with platform handling.
    
    Args:
        *parts: Path components to join
        
    Returns:
        Joined Path object
    """
    if not parts:
        return Path()
    
    result = Path(parts[0])
    for part in parts[1:]:
        result = result / part
    
    return result