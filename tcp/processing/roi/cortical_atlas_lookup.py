#!/usr/bin/env python3
"""
Cortical atlas lookup implementation for Yeo17 400-parcel atlas.

Handles parsing of the hMRF atlas LUT file format and provides ROI-to-parcel
index mapping for cortical regions.

Author: Ian Philip Eglin
Date: 2025-10-28
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .atlas_lookup_interface import AtlasLookupInterface


class CorticalAtlasLookup(AtlasLookupInterface):
    """
    Lookup implementation for Yeo17 400-parcel cortical atlas.

    Parses the LUT file format where each ROI is defined by two lines:
    1. ROI name: 17networks_{LH|RH}_{network}_{region}[_subarea]
    2. Parcel data: <index> <red> <green> <blue> <alpha>
    """

    def __init__(self, lut_file_path: Path):
        """
        Initialize cortical atlas lookup.

        Args:
            lut_file_path: Path to the Yeo17 LUT file

        Raises:
            FileNotFoundError: If LUT file doesn't exist
            ValueError: If LUT file format is invalid
        """
        self.lut_file_path = Path(lut_file_path)
        if not self.lut_file_path.exists():
            raise FileNotFoundError(f"LUT file not found: {self.lut_file_path}")

        self._roi_to_indices: Dict[str, List[int]] = {}
        self._roi_metadata: Dict[str, Dict[str, str]] = {}
        self._available_rois: Set[str] = set()
        self._network_to_rois: Dict[str, Dict[str, List[int]]] = {}  # network -> roi -> indices
        self._available_networks: Set[str] = set()
        self._hemisphere_to_rois: Dict[str, Dict[str, List[int]]] = {}  # hemisphere -> roi -> indices
        self._index_to_parcel_name: Dict[int, str] = {}  # parcel index (0-based) -> full parcel name

        self._parse_lut_file()

    def _parse_lut_file(self) -> None:
        """Parse the LUT file and build ROI mappings."""
        with open(self.lut_file_path, 'r') as file:
            lines = file.readlines()

        # Process pairs of lines (ROI name, parcel data)
        for i in range(0, len(lines), 2):
            if i + 1 >= len(lines):
                break

            roi_line = lines[i].strip()
            parcel_line = lines[i + 1].strip()

            if not roi_line or not parcel_line:
                continue

            # Parse ROI information
            roi_info = self._parse_roi_name(roi_line)
            if not roi_info:
                continue

            # Parse parcel index
            parcel_parts = parcel_line.split()
            if len(parcel_parts) < 5:
                continue

            try:
                parcel_index = int(parcel_parts[0])
            except ValueError:
                continue

            # Store mapping from 0-based index to full parcel name
            zero_based_index = parcel_index - 1
            self._index_to_parcel_name[zero_based_index] = roi_line

            # Extract base ROI name and specific subarea name
            base_roi = roi_info['region']
            specific_roi = base_roi
            if roi_info['subarea']:
                specific_roi += f"_{roi_info['subarea']}"

            network = roi_info['network']
            hemisphere = roi_info['hemisphere']

            # Store mapping for both base ROI (aggregated) and specific subarea
            # Base ROI aggregates all subareas
            if base_roi not in self._roi_to_indices:
                self._roi_to_indices[base_roi] = []
            self._roi_to_indices[base_roi].append(parcel_index)

            # Specific subarea (if exists)
            if roi_info['subarea']:
                if specific_roi not in self._roi_to_indices:
                    self._roi_to_indices[specific_roi] = []
                self._roi_to_indices[specific_roi].append(parcel_index)
                self._available_rois.add(specific_roi)

            # Store network-specific mappings
            if network not in self._network_to_rois:
                self._network_to_rois[network] = {}

            # Base ROI by network
            if base_roi not in self._network_to_rois[network]:
                self._network_to_rois[network][base_roi] = []
            self._network_to_rois[network][base_roi].append(parcel_index)

            # Specific subarea by network (if exists)
            if roi_info['subarea']:
                if specific_roi not in self._network_to_rois[network]:
                    self._network_to_rois[network][specific_roi] = []
                self._network_to_rois[network][specific_roi].append(parcel_index)

            # Store hemisphere-specific mappings
            if hemisphere not in self._hemisphere_to_rois:
                self._hemisphere_to_rois[hemisphere] = {}

            # Base ROI by hemisphere
            if base_roi not in self._hemisphere_to_rois[hemisphere]:
                self._hemisphere_to_rois[hemisphere][base_roi] = []
            self._hemisphere_to_rois[hemisphere][base_roi].append(parcel_index)

            # Specific subarea by hemisphere (if exists)
            if roi_info['subarea']:
                if specific_roi not in self._hemisphere_to_rois[hemisphere]:
                    self._hemisphere_to_rois[hemisphere][specific_roi] = []
                self._hemisphere_to_rois[hemisphere][specific_roi].append(parcel_index)

            # Store metadata for full ROI name
            full_roi_name = f"{roi_info['hemisphere']}_{roi_info['network']}_{specific_roi}"
            self._roi_metadata[full_roi_name] = roi_info

            # Add base ROI to available ROIs and networks
            self._available_rois.add(base_roi)
            self._available_networks.add(network)

    def _parse_roi_name(self, roi_line: str) -> Optional[Dict[str, str]]:
        """
        Parse ROI name line into components.

        Format: 17networks_{LH|RH}_{network}_{region}[_subarea_number]

        Args:
            roi_line: ROI name line from LUT file

        Returns:
            Dictionary with parsed components or None if invalid
        """
        # Pattern to match ROI naming convention
        pattern = r'17networks_([LR]H)_([^_]+)_([^_]+)(?:_(\d+))?'
        match = re.match(pattern, roi_line)

        if not match:
            return None

        hemisphere = match.group(1)
        network = match.group(2)
        region = match.group(3)
        subarea = match.group(4) if match.group(4) else None

        return {
            'hemisphere': hemisphere,
            'network': network,
            'region': region,
            'subarea': subarea,
            'full_name': roi_line
        }

    def get_roi_indices(self, roi_names: List[str]) -> Dict[str, List[int]]:
        """
        Get parcel indices for specified ROI names.

        Returns 0-based indices ready for array indexing, converted from
        the 1-based indexing used in the Yeo17 LUT file.

        Args:
            roi_names: List of ROI identifiers (e.g., ['PFCm', 'PFCv'])

        Returns:
            Dictionary mapping ROI names to lists of 0-based parcel indices

        Raises:
            ValueError: If any ROI names are not found in atlas
        """
        result = {}
        missing_rois = []

        for roi_name in roi_names:
            if roi_name in self._roi_to_indices:
                # Convert from 1-based (LUT file) to 0-based (array indexing)
                one_based_indices = sorted(self._roi_to_indices[roi_name])
                zero_based_indices = [idx - 1 for idx in one_based_indices]
                result[roi_name] = zero_based_indices
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
            Set of available ROI identifiers
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
            Dictionary with aggregated metadata from all instances or None
        """
        if roi_name not in self._available_rois:
            return None

        # Collect metadata from all instances of this ROI
        instances = []
        hemispheres = set()
        networks = set()

        for full_name, metadata in self._roi_metadata.items():
            # Check if this metadata entry belongs to our ROI
            metadata_roi_base = metadata['region']
            metadata_roi_specific = metadata_roi_base
            if metadata['subarea']:
                metadata_roi_specific += f"_{metadata['subarea']}"

            # Match either base ROI (all subareas) or specific subarea
            if metadata_roi_base == roi_name or metadata_roi_specific == roi_name:
                instances.append(metadata)
                hemispheres.add(metadata['hemisphere'])
                networks.add(metadata['network'])

        # Get 0-based indices for metadata
        one_based_indices = sorted(self._roi_to_indices[roi_name])
        zero_based_indices = [idx - 1 for idx in one_based_indices]

        return {
            'roi_name': roi_name,
            'parcel_count': len(self._roi_to_indices[roi_name]),
            'parcel_indices': zero_based_indices,  # Return 0-based indices
            'parcel_indices_original': one_based_indices,  # Keep original for reference
            'hemispheres': sorted(hemispheres),
            'networks': sorted(networks),
            'instances': instances
        }

    @property
    def atlas_name(self) -> str:
        """Return the name of this atlas."""
        return "Yeo2011_17Networks_400Parcels"

    @property
    def total_parcels(self) -> int:
        """Return the total number of parcels in this atlas."""
        return 400

    @property
    def uses_zero_based_indexing(self) -> bool:
        """Return True if the underlying atlas uses 0-based indexing, False for 1-based."""
        return False  # Yeo17 LUT uses 1-based indexing

    def get_roi_indices_by_network(self, roi_names: List[str], networks: Optional[List[str]] = None) -> Dict[str, Dict[str, List[int]]]:
        """
        Get parcel indices for ROIs broken down by network (Yeo17-specific).

        Args:
            roi_names: List of ROI identifiers to look up
            networks: Optional list of networks to filter by

        Returns:
            Dictionary mapping ROI names to network-specific 0-based parcel indices:
            {'PFCm': {'DefaultA': [0,1,2], 'LimbicB': [3,4]}, ...}

        Raises:
            ValueError: If ROI names are not found in atlas
        """
        result = {}
        missing_rois = []

        # Filter networks if specified
        target_networks = set(networks) if networks else self._available_networks

        for roi_name in roi_names:
            if roi_name not in self._available_rois:
                missing_rois.append(roi_name)
                continue

            result[roi_name] = {}

            # Find this ROI in each network
            for network in target_networks:
                if network in self._network_to_rois and roi_name in self._network_to_rois[network]:
                    # Convert from 1-based to 0-based indexing
                    one_based_indices = sorted(self._network_to_rois[network][roi_name])
                    zero_based_indices = [idx - 1 for idx in one_based_indices]
                    result[roi_name][network] = zero_based_indices

        if missing_rois:
            available = sorted(self._available_rois)
            raise ValueError(
                f"ROI(s) not found in atlas: {missing_rois}. "
                f"Available ROIs: {available}"
            )

        return result

    def get_available_networks(self) -> Set[str]:
        """
        Get all available network names in this atlas (Yeo17-specific).

        Returns:
            Set of available network identifiers
        """
        return self._available_networks.copy()

    def get_network_breakdown(self, roi_name: str) -> Optional[Dict[str, Dict[str, any]]]:
        """
        Get detailed breakdown of an ROI across networks (Yeo17-specific).

        Args:
            roi_name: ROI identifier

        Returns:
            Dictionary with network breakdown or None if ROI not found
        """
        if roi_name not in self._available_rois:
            return None

        breakdown = {}

        for network in self._available_networks:
            if network in self._network_to_rois and roi_name in self._network_to_rois[network]:
                one_based_indices = sorted(self._network_to_rois[network][roi_name])
                zero_based_indices = [idx - 1 for idx in one_based_indices]

                breakdown[network] = {
                    'parcel_count': len(one_based_indices),
                    'parcel_indices': zero_based_indices,
                    'parcel_indices_original': one_based_indices
                }

        return breakdown

    def get_roi_indices_by_hemisphere(self, roi_names: List[str], hemisphere: str) -> Dict[str, List[int]]:
        """
        Get parcel indices for ROIs filtered by hemisphere (Yeo17-specific).

        Args:
            roi_names: List of ROI identifiers to look up
            hemisphere: Hemisphere to filter by ('LH' or 'RH')

        Returns:
            Dictionary mapping ROI names to hemisphere-specific 0-based parcel indices

        Raises:
            ValueError: If ROI names are not found or invalid hemisphere specified
        """
        if hemisphere not in {'LH', 'RH'}:
            raise ValueError(f"Invalid hemisphere '{hemisphere}'. Must be 'LH' or 'RH'")

        result = {}
        missing_rois = []

        for roi_name in roi_names:
            if roi_name not in self._available_rois:
                missing_rois.append(roi_name)
                continue

            # Use precomputed hemisphere mapping
            if hemisphere in self._hemisphere_to_rois and roi_name in self._hemisphere_to_rois[hemisphere]:
                # Convert from 1-based to 0-based indexing
                one_based_indices = sorted(self._hemisphere_to_rois[hemisphere][roi_name])
                zero_based_indices = [idx - 1 for idx in one_based_indices]
                result[roi_name] = zero_based_indices
            else:
                result[roi_name] = []

        if missing_rois:
            available = sorted(self._available_rois)
            raise ValueError(
                f"ROI(s) not found in atlas: {missing_rois}. "
                f"Available ROIs: {available}"
            )

        return result

    def get_available_hemispheres(self) -> Set[str]:
        """
        Get all available hemisphere identifiers in this atlas (Yeo17-specific).

        Returns:
            Set of available hemisphere identifiers ('LH', 'RH')
        """
        hemispheres = set()
        for metadata in self._roi_metadata.values():
            hemispheres.add(metadata['hemisphere'])
        return hemispheres

    def get_parcel_name(self, parcel_index: int) -> Optional[str]:
        """
        Get the full parcel name for a given 0-based parcel index.

        Args:
            parcel_index: 0-based parcel index

        Returns:
            Full parcel name (e.g., '17networks_LH_DefaultA_PFCm_1') or None if index not found
        """
        return self._index_to_parcel_name.get(parcel_index)
