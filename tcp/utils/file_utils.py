#!/usr/bin/env python3
"""
File utilities for cross-platform path handling, file validation, and git-annex support.

This module consolidates file utility functions used across preprocessing and processing pipelines.
Handles git-annex symlinks and pointer files on different platforms, especially Windows where
git-annex uses text pointer files instead of symlinks.

Author: Ian Philip Eglin
Date: 2025-12-17
"""

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

# ============================================================================
# Git-Annex Detection Utilities
# ============================================================================

def is_git_annex_pointer(file_path: Union[str, Path]) -> bool:
    """
    Check if a file is a git-annex pointer file or symlink.

    Cross-platform detection that works on:
    - Unix/macOS: Detects symlinks pointing to .git/annex/objects
    - Windows: Detects text pointer files containing annex object paths

    Args:
        file_path: Path to the file to check

    Returns:
        True if the file is a git-annex pointer/symlink, False otherwise
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return False

    # Check if it's a symlink (Unix/macOS/Linux)
    if file_path.is_symlink():
        try:
            link_target = str(file_path.readlink())
            if '.git/annex/objects' in link_target or '/annex/objects' in link_target:
                return True
        except (OSError, RuntimeError):
            # If we can't read the symlink, assume it's not an annex pointer
            pass

    # On Windows, git-annex may create text pointer files
    # These are small files containing just the annex path
    try:
        file_size = file_path.stat().st_size
        if file_size < 200:  # Pointer files are very small
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                # Check if content is a git-annex object path
                if '/annex/objects/' in content and content.endswith(file_path.suffix):
                    return True
    except (OSError, UnicodeDecodeError):
        pass

    return False


def is_actual_file(file_path: Union[str, Path]) -> bool:
    """
    Check if a file is actually downloaded (not a git-annex symlink/pointer).

    This is the inverse of is_git_annex_pointer() with additional size validation.
    Works cross-platform (Windows, macOS, Linux) by checking:
    1. File exists
    2. If it's a symlink, verify the target exists and is accessible
    3. File has actual content (size > 1KB for data files)

    Args:
        file_path: Path to check

    Returns:
        True if file is actually available for reading, False if it's a git-annex stub
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return False

    # Check if it's a symlink
    if file_path.is_symlink():
        try:
            # Check if symlink target exists and is accessible
            resolved = file_path.resolve(strict=True)
            # Verify it's not just a symlink to an annex object path that doesn't exist
            if '.git/annex/objects' in str(resolved):
                # This is a git-annex symlink, check if target actually exists
                if not resolved.exists():
                    return False
        except (OSError, RuntimeError):
            # Can't resolve symlink = not an actual file
            return False

    # Check file size - git-annex pointers/symlinks are very small (<1KB)
    # Real data files (H5, CSV, TSV) should be much larger
    try:
        size = file_path.stat().st_size
        if size < 1024:  # Less than 1KB = likely a pointer or empty
            return False
    except OSError:
        return False

    return True


def resolve_git_annex_path(file_path: Union[str, Path], dataset_root: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """
    Resolve a git-annex pointer to the actual file path.

    Args:
        file_path: Path to the pointer file
        dataset_root: Root directory of the git-annex repository
                     If None, will try to auto-detect from file_path

    Returns:
        Path to the actual annex object, or None if not found
    """
    file_path = Path(file_path)

    if not is_git_annex_pointer(file_path):
        # If it's not a pointer, return the original path
        return file_path

    # Find dataset root if not provided
    if dataset_root is None:
        # Walk up the directory tree to find .git directory
        current = file_path.parent
        while current != current.parent:
            git_dir = current / '.git'
            if git_dir.exists():
                dataset_root = current
                break
            current = current.parent

        if dataset_root is None:
            raise ValueError(f"Could not find git repository root for {file_path}")

    dataset_root = Path(dataset_root)

    # Read the pointer content
    if file_path.is_symlink():
        # On Unix-like systems, follow the symlink
        annex_relative_path = file_path.readlink()
    else:
        # On Windows, read the pointer file
        with open(file_path, 'r', encoding='utf-8') as f:
            annex_relative_path = f.read().strip()

    # Construct the full path to the annex object
    # The pointer contains a path like: /annex/objects/SHA256E-s23524--hash.tsv
    # We need to prepend .git to get: .git/annex/objects/SHA256E-s23524--hash.tsv
    annex_path_str = str(annex_relative_path).lstrip('/')
    if not annex_path_str.startswith('.git'):
        annex_path_str = '.git/' + annex_path_str

    full_annex_path = dataset_root / annex_path_str

    if full_annex_path.exists():
        return full_annex_path
    else:
        # File hasn't been fetched yet
        return None


def read_tsv_with_annex_support(file_path: Union[str, Path],
                                 dataset_root: Optional[Union[str, Path]] = None,
                                 **pandas_kwargs) -> pd.DataFrame:
    """
    Read a TSV file with automatic git-annex pointer resolution.

    This function automatically detects and resolves git-annex pointers,
    making it safe to use on both fetched and unfetched files across all platforms.

    Args:
        file_path: Path to the TSV file (may be a git-annex pointer)
        dataset_root: Root directory of the git-annex repository
        **pandas_kwargs: Additional arguments to pass to pd.read_csv

    Returns:
        DataFrame with the TSV contents

    Raises:
        FileNotFoundError: If file doesn't exist or hasn't been fetched from git-annex
        ValueError: If the file is a pointer but the actual data hasn't been fetched
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")

    # Check if it's a git-annex pointer
    if is_git_annex_pointer(file_path):
        # Resolve to actual annex object
        actual_path = resolve_git_annex_path(file_path, dataset_root)

        if actual_path is None or not actual_path.exists():
            raise ValueError(
                f"Git-annex file has not been fetched: {file_path}\n"
                f"Please run 'datalad get {file_path}' to fetch the file."
            )

        file_path = actual_path

    # Set default pandas arguments for TSV reading
    default_kwargs = {
        'sep': '\t',
        'encoding': 'utf-8'
    }

    # Merge with user-provided kwargs (user kwargs take precedence)
    read_kwargs = {**default_kwargs, **pandas_kwargs}

    # Try to read with UTF-8 first, fallback to latin-1 if needed
    try:
        return pd.read_csv(file_path, **read_kwargs)
    except UnicodeDecodeError:
        # Fallback to latin-1 encoding
        read_kwargs['encoding'] = 'latin-1'
        return pd.read_csv(file_path, **read_kwargs)


def read_csv_with_annex_support(file_path: Union[str, Path],
                                 dataset_root: Optional[Union[str, Path]] = None,
                                 **pandas_kwargs) -> pd.DataFrame:
    """
    Read a CSV file with automatic git-annex pointer resolution.

    Wrapper around read_tsv_with_annex_support for CSV files.

    Args:
        file_path: Path to the CSV file (may be a git-annex pointer)
        dataset_root: Root directory of the git-annex repository
        **pandas_kwargs: Additional arguments to pass to pd.read_csv

    Returns:
        DataFrame with the CSV contents
    """
    # Set default separator for CSV
    if 'sep' not in pandas_kwargs:
        pandas_kwargs['sep'] = ','

    return read_tsv_with_annex_support(file_path, dataset_root, **pandas_kwargs)


# ============================================================================
# General File Utilities
# ============================================================================

def resolve_platform_path(base_path: Union[str, Path], relative_path: str) -> Path:
    """
    Resolve relative path against base path with platform awareness.

    Args:
        base_path: Base directory path
        relative_path: Relative path to resolve

    Returns:
        Resolved Path object

    Raises:
        RuntimeError: If path resolution fails
    """
    try:
        base = Path(base_path)
        resolved = base / relative_path
        return resolved.resolve()
    except Exception as e:
        raise RuntimeError(f"Failed to resolve path '{relative_path}' against '{base_path}': {e}")


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
        FileNotFoundError: If file doesn't exist or is not accessible
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise FileNotFoundError(f"Path is not a file: {path}")

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
        raise FileNotFoundError(f"Cannot access file {path}: {e}")


def get_file_hash(file_path: Union[str, Path], algorithm: str = 'md5') -> str:
    """
    Calculate file hash for integrity verification.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')

    Returns:
        Hexadecimal hash string

    Raises:
        FileNotFoundError: If file doesn't exist or can't be read
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

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
        raise FileNotFoundError(f"Cannot read file {path}: {e}")


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
