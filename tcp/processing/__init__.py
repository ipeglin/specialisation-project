#!/usr/bin/env python3
"""
TCP Processing Pipeline

Data infrastructure for neuroimaging analysis. Provides standardized data loading
and subject management without implementing analysis algorithms.

Author: Ian Philip Eglin
Date: 2025-10-25
"""

from .data_loader import DataLoader
from .subject_manager import SubjectManager
from .utils.validation import validate_manifest, validate_file_paths
from .config.processing_config import ProcessingConfig

__all__ = [
    'DataLoader',
    'SubjectManager', 
    'validate_manifest',
    'validate_file_paths',
    'ProcessingConfig'
]

__version__ = "1.0.0"