#!/usr/bin/env python3
"""
ROI extraction module for TCP processing pipeline.

Provides modular ROI extraction capabilities with dependency injection
for different atlas types.

Author: Ian Philip Eglin
Date: 2025-10-28
"""

from .atlas_lookup_interface import AtlasLookupInterface
from .cortical_atlas_lookup import CorticalAtlasLookup
from .roi_extraction_service import ROIExtractionService

__all__ = [
    'AtlasLookupInterface',
    'CorticalAtlasLookup', 
    'ROIExtractionService'
]