"""
Data Loading Service for fMRI timeseries data.

This service handles loading and segmenting timeseries data from HDF5 files.
"""

from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np


class DataLoadingService:
    """Service for loading and segmenting fMRI timeseries data.

    This service encapsulates the logic for:
    - Loading timeseries from HDF5 files
    - Segmenting data into cortical, subcortical, and cerebellum regions
    - Splitting cortical data into hemispheres and homotopic pairs
    """

    def load_and_segment_timeseries(self, file_path: Path, verbose: bool = False) -> Dict[str, np.ndarray]:
        """Load HDF5 file and segment into anatomical regions.

        Args:
            file_path: Path to the HDF5 timeseries file
            verbose: Whether to print detailed information

        Returns:
            Dictionary containing segmented timeseries:
                - 'cortical': Full cortical timeseries (400 parcels)
                - 'cortical_L': Left hemisphere cortical (200 parcels)
                - 'cortical_R': Right hemisphere cortical (200 parcels)
                - 'cortical_homotopic_pairs': Homotopic pairs (200 pairs x 2)
                - 'subcortical': Subcortical timeseries (32 parcels)
                - 'cerebellum': Cerebellum timeseries (2 parcels)

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If data format is unexpected
        """
        # Read data from HDF5 file
        with h5py.File(file_path, 'r') as file:
            a_group_key = list(file.keys())[0]
            data = np.asarray(file[a_group_key])

        if verbose:
            print(f"Found data with shape: {data.shape}")

        # Segment into anatomical regions (fixed indices for this dataset)
        cortical_timeseries = data[:400]
        cortical_R, cortical_L = cortical_timeseries[:200], cortical_timeseries[200:]
        cortical_homotopic_pairs = np.asarray(list(zip(cortical_L, cortical_R)))
        subcortical_timeseries = data[400:432]
        cerebellum_timeseries = data[432:]

        if verbose:
            print("Found parcels:")
            print(f"Cortical: {cortical_timeseries.shape}")
            print(f"\tLEFT Hemisphere: {cortical_L.shape}")
            print(f"\tRIGHT Hemisphere: {cortical_R.shape}")
            print(f"\tHomotopic Pairs: {cortical_homotopic_pairs.shape}")
            print(f"Subcortical: {subcortical_timeseries.shape}")
            print(f"Cerebellum: {cerebellum_timeseries.shape}")

        return {
            'cortical': cortical_timeseries,
            'cortical_L': cortical_L,
            'cortical_R': cortical_R,
            'cortical_homotopic_pairs': cortical_homotopic_pairs,
            'subcortical': subcortical_timeseries,
            'cerebellum': cerebellum_timeseries
        }
