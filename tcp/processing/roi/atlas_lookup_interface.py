#!/usr/bin/env python3
"""
Abstract interface for atlas lookup operations.

Defines the contract for ROI atlas lookup implementations using dependency injection.
This allows for flexible atlas types (cortical, subcortical, cerebellum) while
maintaining consistent interface.

Author: Ian Philip Eglin
Date: 2025-10-28
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class AtlasLookupInterface(ABC):
    """
    Abstract base class for atlas-based ROI lookup operations.
    
    Defines the contract that all atlas lookup implementations must follow,
    enabling dependency injection and modular design.
    """
    
    @abstractmethod
    def get_roi_indices(self, roi_names: List[str]) -> Dict[str, List[int]]:
        """
        Get parcel indices for specified ROI names.
        
        Returns 0-based indices ready for array indexing, regardless of
        the underlying atlas indexing scheme.
        
        Args:
            roi_names: List of ROI identifiers to look up
            
        Returns:
            Dictionary mapping ROI names to lists of 0-based parcel indices
            
        Raises:
            ValueError: If ROI names are not found in atlas
        """
        pass
    
    @abstractmethod
    def get_available_rois(self) -> Set[str]:
        """
        Get all available ROI names in this atlas.
        
        Returns:
            Set of available ROI identifiers
        """
        pass
    
    @abstractmethod
    def validate_rois(self, roi_names: List[str]) -> Tuple[List[str], List[str]]:
        """
        Validate which ROI names exist in the atlas.
        
        Args:
            roi_names: List of ROI identifiers to validate
            
        Returns:
            Tuple of (valid_rois, invalid_rois)
        """
        pass
    
    @abstractmethod
    def get_roi_metadata(self, roi_name: str) -> Optional[Dict[str, any]]:
        """
        Get metadata for a specific ROI.
        
        Args:
            roi_name: ROI identifier
            
        Returns:
            Dictionary with ROI metadata (hemisphere, network, etc.) or None
        """
        pass
    
    @property
    @abstractmethod
    def atlas_name(self) -> str:
        """Return the name of this atlas."""
        pass
    
    @property
    @abstractmethod
    def total_parcels(self) -> int:
        """Return the total number of parcels in this atlas."""
        pass
    
    @property
    @abstractmethod
    def uses_zero_based_indexing(self) -> bool:
        """Return True if the underlying atlas uses 0-based indexing, False for 1-based."""
        pass