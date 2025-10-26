#!/usr/bin/env python3
"""
Error handling classes for TCP processing pipeline.

Author: Ian Philip Eglin
Date: 2025-10-25
"""

class ProcessingError(Exception):
    """Base exception for processing pipeline errors"""
    pass

class DataNotFoundError(ProcessingError):
    """Raised when required data files are not found"""
    pass

class ValidationError(ProcessingError):
    """Raised when data validation fails"""
    pass

class ManifestError(ProcessingError):
    """Raised when manifest file is invalid or corrupt"""
    pass

class PathResolutionError(ProcessingError):
    """Raised when cross-platform path resolution fails"""
    pass