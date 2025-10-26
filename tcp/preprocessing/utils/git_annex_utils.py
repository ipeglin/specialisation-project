#!/usr/bin/env python3
"""
Git-Annex Utilities for Cross-Platform File Access

Handles git-annex symlinks and pointer files on different platforms,
especially Windows where git-annex uses text pointer files instead of symlinks.

Author: Ian Philip Eglin
Date: 2025-10-26
"""

from pathlib import Path
from typing import Union, Optional
import pandas as pd


def is_git_annex_pointer(file_path: Union[str, Path]) -> bool:
    """
    Check if a file is a git-annex pointer file (common on Windows).

    Git-annex on Windows often creates text files containing the path to the
    annex object instead of symlinks. This function detects such files.

    Args:
        file_path: Path to the file to check

    Returns:
        True if the file is a git-annex pointer, False otherwise
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return False

    # Check if it's a symlink
    if file_path.is_symlink():
        link_target = str(file_path.readlink())
        if '.git/annex/objects' in link_target or '/annex/objects' in link_target:
            return True

    # On Windows, git-annex may create text pointer files
    # These are small files containing just the annex path
    try:
        if file_path.stat().st_size < 200:  # Pointer files are very small
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                # Check if content is a git-annex object path
                if '/annex/objects/' in content and content.endswith(file_path.suffix):
                    return True
    except (OSError, UnicodeDecodeError):
        pass

    return False


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


if __name__ == "__main__":
    # Test the utilities
    import sys

    if len(sys.argv) > 1:
        test_file = Path(sys.argv[1])

        print(f"Testing file: {test_file}")
        print(f"Is git-annex pointer: {is_git_annex_pointer(test_file)}")

        if is_git_annex_pointer(test_file):
            resolved = resolve_git_annex_path(test_file)
            print(f"Resolved path: {resolved}")

            if resolved and resolved.exists():
                print("\nTrying to read as TSV...")
                try:
                    df = read_tsv_with_annex_support(test_file)
                    print(f"Successfully read {len(df)} rows, {len(df.columns)} columns")
                    print(f"Columns: {list(df.columns[:10])}")
                except Exception as e:
                    print(f"Error reading file: {e}")
            else:
                print("File has not been fetched from git-annex yet.")
    else:
        print("Usage: python git_annex_utils.py <file_path>")
        print("Example: python git_annex_utils.py /path/to/dataset/phenotype/shaps01.tsv")
