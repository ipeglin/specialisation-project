"""
Hemisphere Extraction Service for ROI timeseries.

This service handles hemisphere-specific ROI extraction and label creation,
eliminating code duplication between cortical and subcortical extraction.
"""

from typing import Dict, List, Tuple

import numpy as np


class HemisphereExtractionService:
    """Service for extracting hemisphere-specific ROI timeseries with labels.

    This service consolidates the duplicated hemisphere extraction logic
    that was previously repeated for cortical and subcortical ROIs.
    """

    def extract_with_labels(
        self,
        full_timeseries: np.ndarray,
        roi_extractor,
        atlas,
        rois: List[str],
        verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Extract hemisphere-specific timeseries with proper labels.

        This method consolidates the hemisphere extraction logic, eliminating
        the duplication between cortical and subcortical extraction.

        Args:
            full_timeseries: Full timeseries array (all parcels x timepoints)
                           For cortical: 400 parcels, for subcortical: 32 parcels
            roi_extractor: ROI extraction service instance
            atlas: Atlas lookup instance (cortical or subcortical)
            rois: List of ROI names to extract
            verbose: Whether to print detailed information

        Returns:
            Tuple of (right_channels, left_channels, right_labels, left_labels):
                - right_channels: Right hemisphere timeseries array
                - left_channels: Left hemisphere timeseries array
                - right_labels: List of channel labels for right hemisphere
                - left_labels: List of channel labels for left hemisphere

        Raises:
            ValueError: If ROI validation fails
        """
        # Validate ROIs using full timeseries
        validation_result = roi_extractor.validate_roi_coverage(
            full_timeseries,
            rois
        )

        if validation_result['invalid_rois']:
            invalid = validation_result['invalid_rois']
            raise ValueError(f"Invalid ROIs: {invalid}")

        # Extract right hemisphere from full timeseries
        right_timeseries, right_labels = self._extract_hemisphere(
            full_timeseries,
            roi_extractor,
            atlas,
            rois,
            hemisphere='RH',
            verbose=verbose
        )

        # Extract left hemisphere from full timeseries
        left_timeseries, left_labels = self._extract_hemisphere(
            full_timeseries,
            roi_extractor,
            atlas,
            rois,
            hemisphere='LH',
            verbose=verbose
        )

        return right_timeseries, left_timeseries, right_labels, left_labels

    def _extract_hemisphere(
        self,
        timeseries: np.ndarray,
        roi_extractor,
        atlas,
        rois: List[str],
        hemisphere: str,
        verbose: bool = False
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract timeseries for one hemisphere with labels.

        Args:
            timeseries: Timeseries array for this hemisphere
            roi_extractor: ROI extraction service
            atlas: Atlas lookup instance
            rois: List of ROI names
            hemisphere: 'RH' or 'LH'
            verbose: Print details

        Returns:
            Tuple of (channel_timeseries, channel_labels)
        """
        # Determine hemisphere keys based on atlas type
        # Cortical atlases use 'RH'/'LH', subcortical use 'rh'/'lh'
        atlas_name = atlas.atlas_name if hasattr(atlas, 'atlas_name') else ''
        is_subcortical = 'subcortex' in atlas_name.lower() or 'tian' in atlas_name.lower()

        if hemisphere == 'RH':
            hemisphere_key = 'rh' if is_subcortical else 'RH'
        else:  # LH
            hemisphere_key = 'lh' if is_subcortical else 'LH'

        # Extract timeseries
        extracted = roi_extractor.extract_roi_timeseries_by_hemisphere(timeseries, rois, hemisphere_key)

        if verbose:
            print(f"{hemisphere} hemisphere extraction results:")
            for roi_name, ts in extracted.items():
                if ts.size > 0:
                    print(f"  {roi_name}: shape {ts.shape}")

        # Build parcel labels
        parcel_labels = {}
        for roi_name in rois:
            roi_indices_dict = atlas.get_roi_indices_by_hemisphere([roi_name], hemisphere_key)
            hemisphere_indices = roi_indices_dict.get(roi_name, [])

            roi_parcel_labels = []
            for idx in hemisphere_indices:
                parcel_info = atlas.get_parcel_name(idx)
                if parcel_info:
                    # Parse parcel info: "17Networks_RH_DefaultA_1"
                    parts = parcel_info.split('_')
                    if len(parts) >= 3:
                        network = parts[2] if len(parts) > 2 else 'Unknown'
                        parcel_num = parts[3] if len(parts) > 3 else ''

                        # Create descriptive label
                        if parcel_num:
                            label = f"{roi_name}_{hemisphere_key}_{network}_p{parcel_num}"
                        else:
                            label = f"{roi_name}_{hemisphere_key}_{network}"
                    else:
                        # Fallback for simpler naming (subcortical)
                        label = f"{parcel_info}-{hemisphere_key}"

                    roi_parcel_labels.append(label)

            parcel_labels[roi_name] = {hemisphere_key: roi_parcel_labels}

        # Collect all channels and labels in order
        all_channels = []
        all_labels = []

        for roi_name in rois:
            roi_timeseries = extracted.get(roi_name)
            if roi_timeseries is not None and roi_timeseries.size > 0:
                # Add each channel
                if roi_timeseries.ndim == 1:
                    all_channels.append(roi_timeseries)
                    labels_for_roi = parcel_labels.get(roi_name, {}).get(hemisphere_key, [roi_name])
                    all_labels.append(labels_for_roi[0] if labels_for_roi else roi_name)
                else:
                    for ch_idx in range(roi_timeseries.shape[0]):
                        all_channels.append(roi_timeseries[ch_idx])
                        labels_for_roi = parcel_labels.get(roi_name, {}).get(hemisphere_key, [])
                        if ch_idx < len(labels_for_roi):
                            all_labels.append(labels_for_roi[ch_idx])
                        else:
                            all_labels.append(f"{roi_name}_{hemisphere_key}_ch{ch_idx}")

        channel_timeseries = np.vstack(all_channels) if all_channels else np.array([])

        return channel_timeseries, all_labels
