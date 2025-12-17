#!/usr/bin/env python3
"""
Git-Annex Utilities for Cross-Platform File Access

DEPRECATED: This module has been moved to tcp.utils.file_utils
Please update your imports to use:
    from tcp.utils.file_utils import is_git_annex_pointer, resolve_git_annex_path, etc.

This file is kept temporarily for backward compatibility.

Author: Ian Philip Eglin
Date: 2025-10-26
"""

# Import from new location for backward compatibility
from tcp.utils.file_utils import (
    is_git_annex_pointer,
    read_csv_with_annex_support,
    read_tsv_with_annex_support,
    resolve_git_annex_path,
)

__all__ = [
    'is_git_annex_pointer',
    'resolve_git_annex_path',
    'read_tsv_with_annex_support',
    'read_csv_with_annex_support'
]
