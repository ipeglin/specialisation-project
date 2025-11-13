#!/usr/bin/env python3
"""
TCP Processing Configuration

Configuration management for the processing pipeline.
"""

from .analysis_config import (
    AtlasConfig,
    OutputConfig,
    PlottingConfig,
    SignalProcessingConfig,
)
from .analysis_config import ProcessingConfig as AnalysisProcessingConfig
from .processing_config import ProcessingConfig

__all__ = [
    'ProcessingConfig',
    'AtlasConfig',
    'SignalProcessingConfig',
    'OutputConfig',
    'PlottingConfig',
    'AnalysisProcessingConfig'
]
