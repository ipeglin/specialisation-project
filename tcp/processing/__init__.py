#!/usr/bin/env python3
"""
TCP Processing Pipeline

Data infrastructure for neuroimaging analysis. Provides standardized data loading
and subject management without implementing analysis algorithms.

Author: Ian Philip Eglin
Date: 2025-10-25
"""

# Import configuration first (no pandas dependency)
from .config.processing_config import ProcessingConfig
from .utils.validation import validate_manifest, validate_file_paths

# Try to import pandas-dependent components
try:
    from .data_loader import DataLoader
    from .subject_manager import SubjectManager
    _PANDAS_AVAILABLE = True
except ImportError as e:
    if "pandas" in str(e):
        # Create placeholder classes for graceful degradation
        class DataLoader:
            def __init__(self, *args, **kwargs):
                raise ImportError("DataLoader requires pandas. Please install pandas or activate conda environment.")
        
        class SubjectManager:
            def __init__(self, *args, **kwargs):
                raise ImportError("SubjectManager requires pandas. Please install pandas or activate conda environment.")
        
        _PANDAS_AVAILABLE = False
    else:
        raise

__all__ = [
    'DataLoader',
    'SubjectManager', 
    'validate_manifest',
    'validate_file_paths',
    'ProcessingConfig'
]

__version__ = "1.0.0"