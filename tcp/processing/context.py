"""
Processing Context for Dependency Injection.

This module defines the ProcessingContext class which serves as a dependency
injection container for the subject processing pipeline.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tcp.processing import DataLoader, SubjectManager
    from tcp.processing.config.analysis_config import (
        ProcessingConfig as AnalysisProcessingConfig,
    )
    from tcp.processing.roi import ROIExtractionService


@dataclass
class ProcessingContext:
    """Dependency injection container for subject processing.

    This class aggregates all dependencies needed to process a subject,
    reducing function parameter lists and making the code more testable.

    Attributes:
        manager: SubjectManager for subject metadata and filtering
        loader: DataLoader for loading fMRI data files
        cortical_extractor: ROI extraction service for cortical atlas
        subcortical_extractor: ROI extraction service for subcortical atlas
        config: Processing configuration with all settings

    Example:
        >>> context = ProcessingContext(
        ...     manager=subject_manager,
        ...     loader=data_loader,
        ...     cortical_extractor=cortical_roi_extractor,
        ...     subcortical_extractor=subcortical_roi_extractor,
        ...     config=processing_config
        ... )
        >>> result = process_subject('sub-123', context)
    """

    manager: 'SubjectManager'
    loader: 'DataLoader'
    cortical_extractor: 'ROIExtractionService'
    subcortical_extractor: 'ROIExtractionService'
    config: 'AnalysisProcessingConfig'

    @property
    def verbose(self) -> bool:
        """Convenience accessor for verbose flag from config."""
        return self.config.verbose

    @property
    def cortical_rois(self):
        """Convenience accessor for cortical ROI list from config."""
        return self.config.atlas_config.cortical_rois

    @property
    def subcortical_rois(self):
        """Convenience accessor for subcortical ROI list from config."""
        return self.config.atlas_config.subcortical_rois
