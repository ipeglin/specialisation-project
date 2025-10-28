#!/usr/bin/env python3
"""
ROI extraction service with dependency injection for atlas lookups.

Provides modular timeseries extraction functionality that can work with
different atlas types through the AtlasLookupInterface. Handles indexing
conversion transparently through the interface.

Author: Ian Philip Eglin
Date: 2025-10-28
"""

from typing import Dict, List, Optional

import numpy as np

from .atlas_lookup_interface import AtlasLookupInterface


class ROIExtractionService:
    """
    Service for extracting ROI timeseries data using pluggable atlas lookups.
    
    Uses dependency injection to work with different atlas types while
    maintaining consistent extraction logic. The atlas lookup handles
    indexing conversion transparently.
    """
    
    def __init__(self, atlas_lookup: AtlasLookupInterface):
        """
        Initialize extraction service with atlas lookup implementation.
        
        Args:
            atlas_lookup: Atlas lookup implementation following interface
        """
        self.atlas_lookup = atlas_lookup
    
    def extract_roi_timeseries(self, 
                             timeseries_data: np.ndarray, 
                             roi_names: List[str],
                             aggregation_method: str = 'all') -> Dict[str, np.ndarray]:
        """
        Extract timeseries data for specified ROIs.
        
        Args:
            timeseries_data: 2D array of shape (n_parcels, n_timepoints)
            roi_names: List of ROI identifiers to extract
            aggregation_method: How to aggregate multiple parcels per ROI 
                              ('mean', 'median', 'first', 'all')
            
        Returns:
            Dictionary mapping ROI names to extracted timeseries arrays
            
        Raises:
            ValueError: If ROI names not found or invalid aggregation method
        """
        # Validate inputs
        if not isinstance(timeseries_data, np.ndarray) or timeseries_data.ndim != 2:
            raise ValueError("timeseries_data must be 2D numpy array (parcels x timepoints)")
        
        if aggregation_method not in ['mean', 'median', 'first', 'all']:
            raise ValueError(f"aggregation_method must be one of: mean, median, first, all")
        
        # Get 0-based parcel indices for requested ROIs (interface handles conversion)
        roi_indices = self.atlas_lookup.get_roi_indices(roi_names)
        
        # Extract timeseries for each ROI
        extracted_timeseries = {}
        
        for roi_name, parcel_indices in roi_indices.items():
            # Validate indices are within bounds (should be 0-based from interface)
            max_index = max(parcel_indices)
            if max_index >= timeseries_data.shape[0]:
                raise ValueError(
                    f"ROI {roi_name} requires parcel index {max_index} "
                    f"but data only has {timeseries_data.shape[0]} parcels"
                )
            
            # Extract timeseries for this ROI's parcels
            roi_timeseries = timeseries_data[parcel_indices, :]
            
            # Apply aggregation method
            if aggregation_method == 'mean':
                extracted_timeseries[roi_name] = np.mean(roi_timeseries, axis=0)
            elif aggregation_method == 'median':
                extracted_timeseries[roi_name] = np.median(roi_timeseries, axis=0)
            elif aggregation_method == 'first':
                extracted_timeseries[roi_name] = roi_timeseries[0, :]
            elif aggregation_method == 'all':
                extracted_timeseries[roi_name] = roi_timeseries  # Keep all parcels
        
        return extracted_timeseries
    
    def get_roi_metadata(self, roi_names: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Get metadata for ROIs.
        
        Args:
            roi_names: List of ROI names, or None for all available ROIs
            
        Returns:
            Dictionary mapping ROI names to their metadata
        """
        if roi_names is None:
            roi_names = list(self.atlas_lookup.get_available_rois())
        
        metadata = {}
        for roi_name in roi_names:
            roi_metadata = self.atlas_lookup.get_roi_metadata(roi_name)
            if roi_metadata:
                metadata[roi_name] = roi_metadata
        
        return metadata
    
    def validate_roi_coverage(self, 
                            timeseries_data: np.ndarray, 
                            roi_names: List[str]) -> Dict[str, any]:
        """
        Validate that requested ROIs can be extracted from the data.
        
        Args:
            timeseries_data: 2D array of shape (n_parcels, n_timepoints)
            roi_names: List of ROI identifiers to validate
            
        Returns:
            Dictionary with validation results
        """
        # Validate ROI names exist in atlas
        valid_rois, invalid_rois = self.atlas_lookup.validate_rois(roi_names)
        
        # Check data coverage for valid ROIs
        coverage_issues = []
        if valid_rois:
            try:
                roi_indices = self.atlas_lookup.get_roi_indices(valid_rois)
                for roi_name, parcel_indices in roi_indices.items():
                    max_required_index = max(parcel_indices)
                    if max_required_index >= timeseries_data.shape[0]:
                        coverage_issues.append({
                            'roi': roi_name,
                            'required_max_index': max_required_index,
                            'available_parcels': timeseries_data.shape[0]
                        })
            except ValueError:
                pass  # Already handled in invalid_rois
        
        return {
            'valid_rois': valid_rois,
            'invalid_rois': invalid_rois,
            'coverage_issues': coverage_issues,
            'data_shape': timeseries_data.shape,
            'atlas_info': {
                'name': self.atlas_lookup.atlas_name,
                'total_parcels': self.atlas_lookup.total_parcels,
                'uses_zero_based_indexing': self.atlas_lookup.uses_zero_based_indexing
            }
        }
    
    def get_extraction_summary(self, 
                             roi_names: List[str], 
                             extracted_data: Dict[str, np.ndarray]) -> Dict[str, any]:
        """
        Generate summary of extraction results.
        
        Args:
            roi_names: Original ROI names requested
            extracted_data: Results from extract_roi_timeseries
            
        Returns:
            Dictionary with extraction summary
        """
        summary = {
            'requested_rois': roi_names,
            'extracted_rois': list(extracted_data.keys()),
            'extraction_count': len(extracted_data),
            'atlas_name': self.atlas_lookup.atlas_name,
            'atlas_indexing': '0-based' if self.atlas_lookup.uses_zero_based_indexing else '1-based'
        }
        
        # Add details for each extracted ROI
        roi_details = {}
        for roi_name, timeseries in extracted_data.items():
            roi_metadata = self.atlas_lookup.get_roi_metadata(roi_name)
            roi_details[roi_name] = {
                'timeseries_shape': timeseries.shape,
                'parcel_count': roi_metadata['parcel_count'] if roi_metadata else 'unknown',
                'hemispheres': roi_metadata['hemispheres'] if roi_metadata else 'unknown',
                'networks': roi_metadata['networks'] if roi_metadata else 'unknown'
            }
        
        summary['roi_details'] = roi_details
        return summary
    
    def extract_roi_timeseries_by_network(self, 
                                        timeseries_data: np.ndarray, 
                                        roi_names: List[str],
                                        networks: Optional[List[str]] = None,
                                        aggregation_method: str = 'all') -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract timeseries data for ROIs broken down by network (if supported by atlas).
        
        Args:
            timeseries_data: 2D array of shape (n_parcels, n_timepoints)
            roi_names: List of ROI identifiers to extract
            networks: Optional list of networks to filter by
            aggregation_method: How to aggregate multiple parcels per ROI-network combination
            
        Returns:
            Dictionary mapping ROI names to network-specific timeseries:
            {'PFCm': {'DefaultA': timeseries_array, 'LimbicB': timeseries_array}, ...}
            
        Raises:
            ValueError: If atlas doesn't support network queries or other validation errors
            AttributeError: If atlas doesn't have network methods
        """
        # Check if atlas supports network queries
        if not hasattr(self.atlas_lookup, 'get_roi_indices_by_network'):
            raise ValueError(f"Atlas {self.atlas_lookup.atlas_name} does not support network-specific queries")
        
        # Validate inputs
        if not isinstance(timeseries_data, np.ndarray) or timeseries_data.ndim != 2:
            raise ValueError("timeseries_data must be 2D numpy array (parcels x timepoints)")
        
        if aggregation_method not in ['mean', 'median', 'first', 'all']:
            raise ValueError(f"aggregation_method must be one of: mean, median, first, all")
        
        # Get network-specific parcel indices
        roi_network_indices = self.atlas_lookup.get_roi_indices_by_network(roi_names, networks)
        
        # Extract timeseries for each ROI-network combination
        extracted_timeseries = {}
        
        for roi_name, network_data in roi_network_indices.items():
            extracted_timeseries[roi_name] = {}
            
            for network, parcel_indices in network_data.items():
                if not parcel_indices:  # Skip empty networks
                    continue
                    
                # Validate indices are within bounds
                max_index = max(parcel_indices)
                if max_index >= timeseries_data.shape[0]:
                    raise ValueError(
                        f"ROI {roi_name} network {network} requires parcel index {max_index} "
                        f"but data only has {timeseries_data.shape[0]} parcels"
                    )
                
                # Extract timeseries for this ROI-network combination
                network_timeseries = timeseries_data[parcel_indices, :]
                
                # Apply aggregation method
                if aggregation_method == 'mean':
                    extracted_timeseries[roi_name][network] = np.mean(network_timeseries, axis=0)
                elif aggregation_method == 'median':
                    extracted_timeseries[roi_name][network] = np.median(network_timeseries, axis=0)
                elif aggregation_method == 'first':
                    extracted_timeseries[roi_name][network] = network_timeseries[0, :]
                elif aggregation_method == 'all':
                    extracted_timeseries[roi_name][network] = network_timeseries
        
        return extracted_timeseries
    
    def get_network_breakdown_summary(self, roi_names: List[str]) -> Optional[Dict[str, Dict[str, Dict]]]:
        """
        Get network breakdown summary for ROIs (if supported by atlas).
        
        Args:
            roi_names: List of ROI names to analyze
            
        Returns:
            Dictionary with network breakdown information or None if not supported
        """
        if not hasattr(self.atlas_lookup, 'get_network_breakdown'):
            return None
        
        breakdown_summary = {}
        for roi_name in roi_names:
            breakdown = self.atlas_lookup.get_network_breakdown(roi_name)
            if breakdown:
                breakdown_summary[roi_name] = breakdown
        
        return breakdown_summary if breakdown_summary else None
    
    def supports_network_queries(self) -> bool:
        """
        Check if the current atlas supports network-specific queries.
        
        Returns:
            True if network queries are supported, False otherwise
        """
        return hasattr(self.atlas_lookup, 'get_roi_indices_by_network')