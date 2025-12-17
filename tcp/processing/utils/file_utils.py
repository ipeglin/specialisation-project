#!/usr/bin/env python3
"""
File utilities for cross-platform path handling and file validation.

DEPRECATED: General file utilities have been moved to tcp.utils.file_utils
Please update your imports to use:
    from tcp.utils.file_utils import resolve_platform_path, check_file_exists, etc.

This file is kept temporarily for backward compatibility.

Author: Ian Philip Eglin
Date: 2025-10-25
"""

# Import from new location for backward compatibility
from tcp.utils.file_utils import (
    check_file_exists,
    check_file_integrity,
    get_file_hash,
    is_actual_file,
    is_git_annex_pointer,
    resolve_platform_path,
    safe_path_join,
)

__all__ = [
    'resolve_platform_path',
    'check_file_exists',
    'check_file_integrity',
    'get_file_hash',
    'safe_path_join',
    'is_git_annex_pointer',
    'is_actual_file'
]
