#!/usr/bin/env python3
"""
Subcortical atlas lookup implementation for Tian Scale II atlas (Subcortex-only).

Handles parsing of the 32-parcel atlas label file and provides ROI-to-parcel 
index mapping for subcortical regions with hierarchical ROI matching.

Author: Ian Philip Eglin
Date: 2025-10-29
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .atlas_lookup_interface import AtlasLookupInterface


class SubCorticalAtlasLookup(AtlasLookupInterface):
    """
    Lookup implementation for Tian Scale II subcortical atlas.
    
    Supports hierarchical ROI queries:
    - Base structure (e.g., "HIP"): All subdivisions across hemispheres
    - Structure + hemisphere (e.g., "HIP-lh"): All subdivisions in hemisphere
    - Specific subdivision (e.g., "aHIP-lh"): Exact ROI only
    - Subdivision across hemispheres (e.g., "aHIP"): That subdivision in both hemispheres
    """
    
    def __init__(self, label_file_path: Path):
        """
        Initialize subcortical atlas lookup.
        
        Args:
            label_file_path: Path to the Tian Scale II label file (.txt)
            
        Raises:
            FileNotFoundError: If label file doesn't exist
            ValueError: If label file format is invalid
        """
        self.label_file_path = Path(label_file_path)
        if not self.label_file_path.exists():
            raise FileNotFoundError(f"Label file not found: {self.label_file_path}")
        
        self._roi_to_indices: Dict[str, List[int]] = {}
        self._roi_metadata: Dict[str, Dict[str, str]] = {}
        self._available_rois: Set[str] = set()
        self._structure_to_rois: Dict[str, Dict[str, List[int]]] = {}  # structure -> roi -> indices
        self._available_structures: Set[str] = set()
        
        self._parse_label_file()
    
    def _parse_label_file(self) -> None:
        """Parse the label file and build ROI mappings."""
        with open(self.label_file_path, 'r') as file:
            lines = file.readlines()
        
        # Process each line - line number corresponds to 0-based array index
        for index, line in enumerate(lines):
            roi_name = line.strip()
            if not roi_name:
                continue
            
            # Parse ROI information
            roi_info = self._parse_roi_name(roi_name)
            if not roi_info:
                continue
            
            structure = roi_info['structure']
            subdivision = roi_info['subdivision']
            hemisphere = roi_info['hemisphere']
            
            # Build hierarchical mappings
            
            # 1. Specific ROI (exact match)
            if roi_name not in self._roi_to_indices:
                self._roi_to_indices[roi_name] = []
            self._roi_to_indices[roi_name].append(index)
            self._available_rois.add(roi_name)
            
            # 2. Base structure (all subdivisions, all hemispheres)
            if structure not in self._roi_to_indices:
                self._roi_to_indices[structure] = []
            self._roi_to_indices[structure].append(index)
            self._available_rois.add(structure)
            
            # 3. Structure + hemisphere (all subdivisions in hemisphere)
            structure_hemi = f"{structure}-{hemisphere}"
            if structure_hemi not in self._roi_to_indices:
                self._roi_to_indices[structure_hemi] = []
            self._roi_to_indices[structure_hemi].append(index)
            self._available_rois.add(structure_hemi)
            
            # 4. Subdivision across hemispheres (if subdivision exists)
            if subdivision:
                subdivision_both = f"{subdivision}{structure}"
                if subdivision_both not in self._roi_to_indices:
                    self._roi_to_indices[subdivision_both] = []
                self._roi_to_indices[subdivision_both].append(index)
                self._available_rois.add(subdivision_both)
            
            # Store structure-specific mappings for additional queries
            if structure not in self._structure_to_rois:
                self._structure_to_rois[structure] = {}
            
            if roi_name not in self._structure_to_rois[structure]:
                self._structure_to_rois[structure][roi_name] = []
            self._structure_to_rois[structure][roi_name].append(index)
            
            # Store metadata
            self._roi_metadata[roi_name] = roi_info
            self._available_structures.add(structure)
    
    def _parse_roi_name(self, roi_name: str) -> Optional[Dict[str, str]]:
        """
        Parse ROI name into components.
        
        Format: {subdivision}{structure}-{hemisphere}
        Examples: aHIP-rh, THA-DP-lh, NAc-shell-rh
        
        Args:
            roi_name: ROI name from label file
            
        Returns:
            Dictionary with parsed components or None if invalid
        """
        # Pattern to match Tian naming convention
        # Examples: aHIP-rh, pHIP-lh, THA-DP-rh, NAc-shell-lh
        pattern = r'^(.+?)([A-Z][A-Za-z]*)-([lr]h)$'
        match = re.match(pattern, roi_name)
        
        if not match:
            return None
        
        subdivision_raw = match.group(1)
        structure = match.group(2)
        hemisphere = match.group(3)
        
        # Clean up subdivision (remove any trailing hyphens)
        subdivision = subdivision_raw.rstrip('-') if subdivision_raw else None
        
        # Special handling for compound structures like THA-DP, NAc-shell
        if '-' in structure:
            parts = structure.split('-')
            actual_structure = parts[0]
            additional_subdivision = '-'.join(parts[1:])
            if subdivision:
                subdivision = f"{subdivision}-{additional_subdivision}"
            else:
                subdivision = additional_subdivision
            structure = actual_structure
        
        return {
            'structure': structure,
            'subdivision': subdivision,
            'hemisphere': hemisphere,
            'full_name': roi_name
        }
    
    def get_roi_indices(self, roi_names: List[str]) -> Dict[str, List[int]]:
        """
        Get parcel indices for specified ROI names.
        
        Supports hierarchical matching:
        - Exact ROI name (e.g., 'aHIP-rh')
        - Base structure (e.g., 'HIP' -> all HIP subdivisions)
        - Structure + hemisphere (e.g., 'HIP-lh' -> left HIP subdivisions)
        - Subdivision across hemispheres (e.g., 'aHIP' -> anterior HIP both hemispheres)
        
        Args:
            roi_names: List of ROI identifiers to look up
            
        Returns:
            Dictionary mapping ROI names to lists of 0-based parcel indices
            
        Raises:
            ValueError: If any ROI names are not found in atlas
        """
        result = {}
        missing_rois = []
        
        for roi_name in roi_names:
            if roi_name in self._roi_to_indices:
                # Indices are already 0-based (line numbers from label file)
                indices = sorted(self._roi_to_indices[roi_name])
                result[roi_name] = indices
            else:
                missing_rois.append(roi_name)
        
        if missing_rois:
            available = sorted(self._available_rois)
            raise ValueError(
                f"ROI(s) not found in atlas: {missing_rois}. "
                f"Available ROIs: {available}"
            )
        
        return result
    
    def get_available_rois(self) -> Set[str]:
        """
        Get all available ROI names in this atlas.
        
        Returns:
            Set of available ROI identifiers (includes hierarchical variants)
        """
        return self._available_rois.copy()
    
    def validate_rois(self, roi_names: List[str]) -> Tuple[List[str], List[str]]:
        """
        Validate which ROI names exist in the atlas.
        
        Args:
            roi_names: List of ROI identifiers to validate
            
        Returns:
            Tuple of (valid_rois, invalid_rois)
        """
        valid_rois = []
        invalid_rois = []
        
        for roi_name in roi_names:
            if roi_name in self._available_rois:
                valid_rois.append(roi_name)
            else:
                invalid_rois.append(roi_name)
        
        return valid_rois, invalid_rois
    
    def get_roi_metadata(self, roi_name: str) -> Optional[Dict[str, any]]:
        """
        Get metadata for a specific ROI.
        
        Args:
            roi_name: ROI identifier
            
        Returns:
            Dictionary with ROI metadata or None if not found
        """
        if roi_name not in self._available_rois:
            return None
        
        # For hierarchical ROIs, collect metadata from all constituent parts
        if roi_name in self._roi_to_indices:
            indices = sorted(self._roi_to_indices[roi_name])
            
            # Collect metadata from individual ROIs
            constituent_rois = []
            hemispheres = set()
            structures = set()
            subdivisions = set()
            
            for roi_full_name, metadata in self._roi_metadata.items():
                # Check if this specific ROI contributes to the requested ROI
                roi_indices = self._roi_to_indices.get(roi_full_name, [])
                if any(idx in indices for idx in roi_indices):
                    constituent_rois.append(metadata)
                    hemispheres.add(metadata['hemisphere'])
                    structures.add(metadata['structure'])
                    if metadata['subdivision']:
                        subdivisions.add(metadata['subdivision'])
            
            return {
                'roi_name': roi_name,
                'parcel_count': len(indices),
                'parcel_indices': indices,  # Already 0-based
                'hemispheres': sorted(hemispheres),
                'structures': sorted(structures),
                'subdivisions': sorted(subdivisions) if subdivisions else None,
                'constituent_rois': constituent_rois
            }
        
        return None
    
    @property
    def atlas_name(self) -> str:
        """Return the name of this atlas."""
        return "Tian_Subcortex_S2_3T_32Parcels"
    
    @property
    def total_parcels(self) -> int:
        """Return the total number of parcels in this atlas."""
        return 32
    
    @property
    def uses_zero_based_indexing(self) -> bool:
        """Return True if the underlying atlas uses 0-based indexing, False for 1-based."""
        return True  # Label file line numbers correspond to 0-based array indices