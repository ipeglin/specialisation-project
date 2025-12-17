#!/usr/bin/env python3
"""
Functional Connectivity Analysis Script

This script performs STATIC functional connectivity analysis on TCP dataset timeseries.
Static FC computes Pearson correlations between ROI timeseries across the entire session.

TODO: Dynamic FC analysis (binned/windowed correlations) is not yet implemented.

Author: Ian Philip Eglin
"""

import sys
from pathlib import Path

# Add project root to path (same fix as test script)
project_root = Path(__file__).parent.parent.parent

# Clear any conflicting paths and ensure clean import environment
paths_to_remove = [
    str(Path.cwd()),
    str(Path(__file__).parent.parent),  # tcp directory
    str(Path(__file__).parent),         # tcp/processing directory#
]

for path in paths_to_remove:
    while path in sys.path:
        sys.path.remove(path)

# Insert project root at the beginning
sys.path.insert(0, str(project_root))

import random
from itertools import islice

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, stats

from config.paths import get_analysis_path
from tcp.processing import DataLoader, SubjectManager
from tcp.processing.lib.csv_export import (
    export_group_averaged_fc_to_csv,
    export_static_fc_results_to_csv,
)
from tcp.processing.lib.fisher import fisher_r_to_z, fisher_z_to_r
from tcp.processing.lib.logging import write_analysis_log
from tcp.processing.lib.mvmd import MVMD
from tcp.processing.lib.plot import (
    plot_fc_results,
    plot_marginal_spectrum_per_mode,
    plot_multivariate_hilbert_spectrum,
    plot_roi_timeseries_result,
    plot_signal_decomposition,
)
from tcp.processing.lib.slow_band import get_frequency_range
from tcp.processing.lib.subject_filtering import get_accessible_subjects_from_file
from tcp.processing.roi import (
    CorticalAtlasLookup,
    ROIExtractionService,
    SubCorticalAtlasLookup,
)
from tcp.processing.utils.lists import chunks

# ===== SIGNAL ACQUISITION PARAMETERS =====
TR = 800e-3  # Repetition Time [seconds]
SAMPLING_RATE = 1 / TR  # 1.25 Hz
NYQUIST_FREQUENCY = 0.5 * SAMPLING_RATE  # 0.625 Hz

# ===== STATISTICAL TESTING PARAMETERS =====
PERMUTATION_COUNT = 3
PERMUTATION_GROUP_COUNT = 3


def compute_fc_matrix(timeseries_dict, roi_names=None):
    """
    Compute functional connectivity (Pearson correlation) matrix between ROI timeseries.

    Uses Fisher r-to-z transformation for statistically valid p-value computation,
    which is necessary for non-stationary nonlinear fMRI BOLD signals.

    Args:
        timeseries_dict: Dictionary mapping ROI names to timeseries arrays
        roi_names: Optional list to specify order of ROIs in matrix

    Returns:
        tuple: (correlation_matrix, roi_labels, p_values_matrix)
    """
    if roi_names is None:
        roi_names = list(timeseries_dict.keys())

    # Collect all timeseries
    timeseries_list = []
    roi_labels = []

    for roi_name in roi_names:
        if roi_name in timeseries_dict:
            ts = timeseries_dict[roi_name]
            if ts.size > 0:
                timeseries_list.append(ts)
                roi_labels.append(roi_name)

    if len(timeseries_list) < 2:
        print(f"[WARNING] Need at least 2 ROI timeseries for FC computation, got {len(timeseries_list)}")
        return None, roi_labels, None

    # Stack timeseries for correlation computation
    stacked_timeseries = np.vstack(timeseries_list)

    # Compute Pearson correlation matrix
    corr_matrix = np.corrcoef(stacked_timeseries)

    # Compute p-values using Fisher r-to-z transformation
    n_rois = len(roi_labels)
    n_samples = timeseries_list[0].shape[0]
    p_values = np.ones((n_rois, n_rois))

    for i in range(n_rois):
        for j in range(i+1, n_rois):
            r = corr_matrix[i, j]

            if np.isnan(r):
                p_values[i, j] = 1.0
                p_values[j, i] = 1.0
                continue

            # Apply Fisher r-to-z transformation
            z = fisher_r_to_z(r)

            # Under H0, z ~ N(0, 1/sqrt(n-3))
            se_z = 1.0 / np.sqrt(n_samples - 3)
            z_stat = z / se_z

            # Two-tailed p-value from standard normal
            p_val = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

            p_values[i, j] = p_val
            p_values[j, i] = p_val

    return corr_matrix, roi_labels, p_values

def compute_fc_per_mode(time_modes, channel_labels, verbose=False):
    """
    Compute functional connectivity matrix for each MVMD mode independently.

    This function addresses the statistical error of summing modes in the time domain
    before computing correlations. Instead, it computes FC for each mode separately,
    preserving mode-specific phase and connectivity information.

    Args:
        time_modes: MVMD modes array, shape (n_modes, n_channels, n_samples)
                   Output from MVMD decomposition
        channel_labels: List of channel label strings (length n_channels)
        verbose: If True, print progress information

    Returns:
        dict: {
            'mode_fc_matrices': np.ndarray, shape (n_modes, n_channels, n_channels)
                               Correlation matrices in r-scale for each mode
            'mode_fc_z_matrices': np.ndarray, shape (n_modes, n_channels, n_channels)
                                 Fisher Z-transformed matrices for each mode
            'mode_fc_pvalues': np.ndarray, shape (n_modes, n_channels, n_channels)
                              P-values for each mode's correlations
            'fc_labels': list, channel labels (length n_channels)
            'n_modes': int, number of modes
            'n_channels': int, number of channels
        }

    Example:
        >>> time_modes = mvmd_output  # shape: (10, 28, 500)
        >>> labels = ['vmPFC_RH_ch0', 'vmPFC_RH_ch1', ...]
        >>> mode_fc_data = compute_fc_per_mode(time_modes, labels)
        >>> # mode_fc_data['mode_fc_matrices'][0] is FC matrix for mode 0
    """
    # Input validation
    if time_modes.ndim != 3:
        raise ValueError(f"time_modes must be 3D array (modes, channels, samples), got shape {time_modes.shape}")

    n_modes, n_channels, n_samples = time_modes.shape

    if len(channel_labels) != n_channels:
        raise ValueError(f"channel_labels length ({len(channel_labels)}) must match n_channels ({n_channels})")

    if verbose:
        print(f"\n{'='*80}")
        print("COMPUTING MODE-LEVEL FUNCTIONAL CONNECTIVITY")
        print(f"{'='*80}")
        print(f"  Number of modes: {n_modes}")
        print(f"  Number of channels: {n_channels}")
        print(f"  Samples per mode: {n_samples}")

    # Initialize 3D arrays to store results for all modes
    mode_fc_matrices = np.zeros((n_modes, n_channels, n_channels))
    mode_fc_z_matrices = np.zeros((n_modes, n_channels, n_channels))
    mode_fc_pvalues = np.ones((n_modes, n_channels, n_channels))  # Initialize with 1.0

    # Compute FC for each mode independently
    for mode_idx in range(n_modes):
        # Extract signal for this mode: shape (n_channels, n_samples)
        mode_signal = time_modes[mode_idx, :, :]

        # Create timeseries dictionary for compute_fc_matrix()
        # Format: {channel_label: timeseries_array}
        mode_timeseries = {
            channel_labels[ch_idx]: mode_signal[ch_idx, :]
            for ch_idx in range(n_channels)
        }

        # Compute FC matrix using existing function
        fc_matrix, fc_labels, fc_pvalues = compute_fc_matrix(
            mode_timeseries,
            roi_names=channel_labels
        )

        # Handle case where compute_fc_matrix returns None (shouldn't happen with valid input)
        if fc_matrix is None:
            if verbose:
                print(f"  [WARNING] Mode {mode_idx}: FC computation returned None, skipping")
            mode_fc_matrices[mode_idx, :, :] = np.nan
            mode_fc_z_matrices[mode_idx, :, :] = np.nan
            mode_fc_pvalues[mode_idx, :, :] = np.nan
            continue

        # Store r-values (correlation scale)
        mode_fc_matrices[mode_idx, :, :] = fc_matrix

        # Apply Fisher Z-transformation immediately
        mode_fc_z_matrices[mode_idx, :, :] = fisher_r_to_z(fc_matrix)

        # Store p-values
        if fc_pvalues is not None:
            mode_fc_pvalues[mode_idx, :, :] = fc_pvalues
        else:
            mode_fc_pvalues[mode_idx, :, :] = np.nan

        if verbose and (mode_idx + 1) % 5 == 0:
            print(f"  Processed {mode_idx + 1}/{n_modes} modes...")

    if verbose:
        print(f"  Completed FC computation for all {n_modes} modes")
        print(f"  Output shapes:")
        print(f"    - mode_fc_matrices: {mode_fc_matrices.shape}")
        print(f"    - mode_fc_z_matrices: {mode_fc_z_matrices.shape}")
        print(f"    - mode_fc_pvalues: {mode_fc_pvalues.shape}")

    return {
        'mode_fc_matrices': mode_fc_matrices,
        'mode_fc_z_matrices': mode_fc_z_matrices,
        'mode_fc_pvalues': mode_fc_pvalues,
        'fc_labels': channel_labels,
        'n_modes': n_modes,
        'n_channels': n_channels
    }


def aggregate_mode_fc_to_bands(mode_fc_data, mode_frequencies, verbose=False):
    """
    Aggregate mode-level FC matrices into slow-band FC matrices.

    Uses Fisher Z-transformation for statistically valid averaging of correlations.
    This addresses the fundamental statistical error: you cannot average correlations
    in r-scale (they have a skewed distribution), but Fisher Z-values are approximately
    normal and can be averaged using standard arithmetic mean.

    Slow-band definitions:
        Slow-6: 0.000-0.010 Hz
        Slow-5: 0.010-0.027 Hz
        Slow-4: 0.027-0.073 Hz
        Slow-3: 0.073-0.198 Hz
        Slow-2: 0.198-0.500 Hz
        Slow-1: 0.500-0.750 Hz

    Args:
        mode_fc_data: dict, output from compute_fc_per_mode() containing:
                     - 'mode_fc_z_matrices': Z-transformed FC matrices per mode
                     - 'mode_fc_matrices': r-scale FC matrices per mode (for edge cases)
                     - 'mode_fc_pvalues': p-values per mode
                     - 'fc_labels': channel labels
        mode_frequencies: np.ndarray, center frequency for each mode (length n_modes)
        verbose: If True, print progress information

    Returns:
        dict: {
            'slow-6': band_fc_dict,
            'slow-5': band_fc_dict,
            'slow-4': band_fc_dict,
            'slow-3': band_fc_dict,
            'slow-2': band_fc_dict,
            'slow-1': band_fc_dict
        }

        Each band_fc_dict contains:
            'fc_matrix': np.ndarray (n_channels, n_channels) - Averaged r-scale matrix
            'fc_z_matrix': np.ndarray (n_channels, n_channels) - Averaged Z-scale matrix
            'fc_pvalues': np.ndarray (n_channels, n_channels) - P-values (placeholder)
            'fc_labels': list - Channel labels
            'mode_indices': list - Which modes contributed to this band
            'mode_frequencies': list - Center frequencies of contributing modes
            'n_modes_in_band': int - Number of modes in this band

    Example:
        >>> mode_fc_data = compute_fc_per_mode(time_modes, labels)
        >>> center_freqs = np.array([0.015, 0.035, 0.055, ...])  # from MVMD
        >>> band_fc = aggregate_mode_fc_to_bands(mode_fc_data, center_freqs)
        >>> # band_fc['slow-5']['fc_matrix'] is the averaged FC for Slow-5 band
    """
    # Extract data from input
    mode_fc_z_matrices = mode_fc_data['mode_fc_z_matrices']
    mode_fc_matrices = mode_fc_data['mode_fc_matrices']
    mode_fc_pvalues = mode_fc_data['mode_fc_pvalues']
    fc_labels = mode_fc_data['fc_labels']
    n_modes = mode_fc_data['n_modes']
    n_channels = mode_fc_data['n_channels']

    # Input validation
    if len(mode_frequencies) != n_modes:
        raise ValueError(f"mode_frequencies length ({len(mode_frequencies)}) must match n_modes ({n_modes})")

    if verbose:
        print(f"\n{'='*80}")
        print("AGGREGATING MODE-LEVEL FC INTO SLOW-BANDS")
        print(f"{'='*80}")
        print(f"  Total modes to process: {n_modes}")
        print(f"  Mode frequencies: {mode_frequencies}")

    # Define slow-band frequency ranges
    def get_band_number(frequency):
        """Determine which slow-band a frequency belongs to."""
        if 0.0 < frequency <= 0.01:
            return "6"
        elif 0.01 < frequency <= 0.027:
            return "5"
        elif 0.027 < frequency <= 0.073:
            return "4"
        elif 0.073 < frequency <= 0.198:
            return "3"
        elif 0.198 < frequency <= 0.5:
            return "2"
        elif 0.5 < frequency <= 0.75:
            return "1"
        else:
            return None  # Outside slow-band range

    # Group modes by slow-band
    band_mode_groups = {
        '6': {'indices': [], 'frequencies': []},
        '5': {'indices': [], 'frequencies': []},
        '4': {'indices': [], 'frequencies': []},
        '3': {'indices': [], 'frequencies': []},
        '2': {'indices': [], 'frequencies': []},
        '1': {'indices': [], 'frequencies': []}
    }

    for mode_idx, freq in enumerate(mode_frequencies):
        band_num = get_band_number(freq)
        if band_num is not None:
            band_mode_groups[band_num]['indices'].append(mode_idx)
            band_mode_groups[band_num]['frequencies'].append(freq)

    if verbose:
        print(f"\n  Mode distribution across bands:")
        for band_num in ['6', '5', '4', '3', '2', '1']:
            n_modes_in_band = len(band_mode_groups[band_num]['indices'])
            freqs = band_mode_groups[band_num]['frequencies']
            if n_modes_in_band > 0:
                print(f"    Slow-{band_num}: {n_modes_in_band} modes, freqs={freqs}")
            else:
                print(f"    Slow-{band_num}: 0 modes (empty band)")

    # Aggregate FC matrices for each band
    slow_band_fc_results = {}

    for band_num in ['6', '5', '4', '3', '2', '1']:
        mode_indices = band_mode_groups[band_num]['indices']
        mode_freqs = band_mode_groups[band_num]['frequencies']
        n_modes_in_band = len(mode_indices)

        if n_modes_in_band == 0:
            # Skip empty bands
            if verbose:
                print(f"\n  Slow-{band_num}: Skipping (no modes in this band)")
            continue

        elif n_modes_in_band == 1:
            # Edge case: Only 1 mode in band
            # No averaging needed - use the single mode's values directly
            mode_idx = mode_indices[0]
            fc_matrix = mode_fc_matrices[mode_idx, :, :]
            fc_z_matrix = mode_fc_z_matrices[mode_idx, :, :]
            fc_pvalues = mode_fc_pvalues[mode_idx, :, :]

            if verbose:
                print(f"\n  Slow-{band_num}: Using single mode (mode {mode_idx}, freq={mode_freqs[0]:.4f} Hz)")

        else:
            # Multiple modes in band: Average using Fisher Z-transformation
            # Extract Z-matrices for all modes in this band
            z_matrices_in_band = mode_fc_z_matrices[mode_indices, :, :]  # shape: (n_modes_in_band, n_channels, n_channels)

            # Compute mean Z-matrix (handles NaN values)
            avg_z_matrix = np.nanmean(z_matrices_in_band, axis=0)  # shape: (n_channels, n_channels)

            # Convert back to r-scale
            avg_r_matrix = fisher_z_to_r(avg_z_matrix)

            # P-value computation: Placeholder for now
            # NOTE: This will be updated later with t-test or permutation testing
            # For now, set all p-values to NaN to indicate they need computation
            fc_pvalues = np.full((n_channels, n_channels), np.nan)

            fc_matrix = avg_r_matrix
            fc_z_matrix = avg_z_matrix

            if verbose:
                print(f"\n  Slow-{band_num}: Averaged {n_modes_in_band} modes")
                print(f"    Mode indices: {mode_indices}")
                print(f"    Frequencies: {[f'{f:.4f}' for f in mode_freqs]}")
                print(f"    Avg Z-matrix shape: {avg_z_matrix.shape}")
                print(f"    Avg r-matrix range: [{np.nanmin(avg_r_matrix):.3f}, {np.nanmax(avg_r_matrix):.3f}]")

        # Store results for this band
        slow_band_fc_results[f'slow-{band_num}'] = {
            'fc_matrix': fc_matrix,
            'fc_z_matrix': fc_z_matrix,
            'fc_pvalues': fc_pvalues,
            'fc_labels': fc_labels,
            'mode_indices': mode_indices,
            'mode_frequencies': mode_freqs,
            'n_modes_in_band': n_modes_in_band
        }

    if verbose:
        print(f"\n  Completed aggregation for {len(slow_band_fc_results)} bands")
        print(f"  Band keys: {list(slow_band_fc_results.keys())}")

    return slow_band_fc_results


def compute_group_averaged_fc(subject_results, group_subjects, group_name, fc_type='static', band_key=None, verbose=False):
    """
    Compute group-averaged functional connectivity matrices.

    Args:
        subject_results: Dictionary of all subject results
        group_subjects: List of subject IDs belonging to this group
        group_name: Name of the group (e.g., 'non-anhedonic', 'Low Anhedonic', 'High Anhedonic')
        fc_type: Type of FC to average ('static' or 'slow_band')
        band_key: If fc_type='slow_band', specify which band (e.g., 'slow-5')
        verbose: Print progress messages

    Returns:
        dict: Group-averaged FC data containing:
            - avg_fc_matrix: Averaged correlation matrix (NaN values excluded from averaging)
            - avg_fc_labels: ROI labels
            - n_subjects: Number of subjects included
            - group_name: Group identifier
            - fc_type: Type of FC
            - band_key: Band identifier (if slow-band)
            - subject_ids: List of included subject IDs
    """
    if verbose:
        fc_desc = f"{band_key} " if band_key else "whole-signal "
        print(f"\nComputing group-averaged {fc_desc}FC for {group_name}...")

    # Collect FC matrices from all subjects in this group
    fc_matrices = []
    fc_labels = None
    included_subjects = []

    for subject_id in group_subjects:
        if subject_id not in subject_results:
            continue

        result = subject_results[subject_id]
        if not result.get('success'):
            continue

        # Extract FC data based on type
        if fc_type == 'static':
            fc_data = result.get('static_functional_connectivity')
            if fc_data and fc_data.get('static_fc_matrix') is not None:
                fc_matrix = fc_data['static_fc_matrix']
                if fc_labels is None:
                    fc_labels = fc_data['static_fc_labels']
                fc_matrices.append(fc_matrix)
                included_subjects.append(subject_id)

        elif fc_type == 'slow_band' and band_key:
            slow_band_data = result.get('slow_band_fc', {}).get(band_key)
            if slow_band_data and slow_band_data.get('fc_matrix') is not None:
                fc_matrix = slow_band_data['fc_matrix']
                if fc_labels is None:
                    fc_labels = slow_band_data['fc_labels']
                fc_matrices.append(fc_matrix)
                included_subjects.append(subject_id)

    if len(fc_matrices) == 0:
        if verbose:
            print(f"  No valid FC matrices found for {group_name}")
        return None

    # Stack matrices and compute mean (ignoring NaN values)
    # Shape: (n_subjects, n_rois, n_rois)
    # Stack matrices (n_subjects, n_rois, n_rois)
    stacked_r_matrices = np.stack(fc_matrices, axis=0)

    # STEP 1: Apply Fisher r-to-z transformation to each subject's matrix
    # This is REQUIRED for statistically valid averaging of correlations
    stacked_z_matrices = np.zeros_like(stacked_r_matrices)
    for subj_idx in range(len(fc_matrices)):
        stacked_z_matrices[subj_idx] = fisher_r_to_z(stacked_r_matrices[subj_idx])

    # STEP 2: Compute mean across subjects on Z-transformed values (ignoring NaN)
    avg_z_matrix = np.nanmean(stacked_z_matrices, axis=0)

    # STEP 3: Apply inverse Fisher transformation to get back to correlation scale
    avg_fc_matrix = fisher_z_to_r(avg_z_matrix)

    # STEP 4: Compute p-values using one-sample t-test on Fisher z-transformed values
    # This tests if the group-average correlation is significantly different from 0
    from scipy import stats
    n_rois = avg_fc_matrix.shape[0]
    p_values = np.ones((n_rois, n_rois))  # Initialize with 1.0 (non-significant)

    for i in range(n_rois):
        for j in range(n_rois):
            # Get z-transformed values across subjects for this pair
            z_values = stacked_z_matrices[:, i, j]

            # Only test if we have valid (non-NaN) values
            valid_z = z_values[~np.isnan(z_values)]

            if len(valid_z) >= 3:  # Need at least 3 subjects for reliable t-test
                # One-sample t-test on z-values: test if mean z != 0
                t_stat, p_val = stats.ttest_1samp(valid_z, 0.0)
                p_values[i, j] = p_val
            else:
                # Not enough data, mark as non-significant
                p_values[i, j] = 1.0

    if verbose:
        print(f"  Averaged {len(fc_matrices)} subjects")
        print(f"  Matrix shape: {avg_fc_matrix.shape}")
        nan_count = np.sum(np.isnan(avg_fc_matrix))
        total_elements = avg_fc_matrix.size
        print(f"  NaN values: {nan_count}/{total_elements} ({nan_count/total_elements*100:.1f}%)")

        # Count significant correlations (upper triangle only, excluding diagonal)
        significant_mask = np.triu(p_values < 0.05, k=1)
        n_significant = np.sum(significant_mask)
        n_total = (n_rois * (n_rois - 1)) // 2
        print(f"  Significant correlations: {n_significant}/{n_total} ({n_significant/n_total*100:.1f}%)")

    return {
        'avg_fc_matrix': avg_fc_matrix,
        'avg_fc_labels': fc_labels,
        'avg_fc_pvalues': p_values,
        'n_subjects': len(fc_matrices),
        'group_name': group_name,
        'fc_type': fc_type,
        'band_key': band_key,
        'subject_ids': included_subjects
    }


def analyze_connectivity_patterns(corr_matrix, roi_labels, p_values=None, alpha=0.05):
    """
    Extract and analyze specific connectivity patterns from correlation matrix.

    Args:
        corr_matrix: Correlation matrix
        roi_labels: ROI labels corresponding to matrix rows/columns
        p_values: Optional p-values matrix for significance testing
        alpha: Significance threshold

    Returns:
        dict: Dictionary with different connectivity pattern results.
              Each pattern type has 'pairs' (dict of connections) and 'stats' (metadata/statistics)
    """
    results = {
        'interhemispheric': {'pairs': {}, 'stats': {}},
        'cross_regional': {'pairs': {}, 'stats': {}},
        'ipsilateral': {'pairs': {}, 'stats': {}},
        'contralateral': {'pairs': {}, 'stats': {}},
        'all_pairwise': {'pairs': {}, 'stats': {}}
    }

    n_rois = len(roi_labels)

    # Extract all pairwise correlations
    for i in range(n_rois):
        for j in range(i+1, n_rois):
            roi1, roi2 = roi_labels[i], roi_labels[j]
            corr_val = corr_matrix[i, j]

            # Skip if correlation is NaN (unavailable channels)
            if np.isnan(corr_val):
                continue

            p_val = p_values[i, j] if p_values is not None else None
            is_significant = p_val < alpha if p_val is not None else None

            pair_key = f"{roi1}_{roi2}"
            results['all_pairwise']['pairs'][pair_key] = {
                'correlation': corr_val,
                'p_value': p_val,
                'significant': is_significant
            }

            # Categorize by connectivity pattern

            # Extract hemisphere and region info
            roi1_parts = roi1.split('_')
            roi2_parts = roi2.split('_')

            if len(roi1_parts) >= 2 and len(roi2_parts) >= 2:
                roi1_region = roi1_parts[0]
                roi1_hemi = roi1_parts[1] if len(roi1_parts) > 1 else 'unknown'
                roi2_region = roi2_parts[0]
                roi2_hemi = roi2_parts[1] if len(roi2_parts) > 1 else 'unknown'

                # Interhemispheric (same region, different hemispheres)
                if roi1_region == roi2_region and roi1_hemi != roi2_hemi:
                    # Extract network information (3rd field in label)
                    roi1_network = roi1_parts[2] if len(roi1_parts) > 2 else None
                    roi2_network = roi2_parts[2] if len(roi2_parts) > 2 else None

                    results['interhemispheric']['pairs'][pair_key] = {
                        'correlation': corr_val,
                        'p_value': p_val,
                        'significant': is_significant,
                        'region': roi1_region,
                        'network': roi1_network,  # NEW: Store network info
                        'roi1_index': i,  # NEW: Store indices for later use
                        'roi2_index': j
                    }

                # Cross-regional (different regions)
                elif roi1_region != roi2_region:
                    results['cross_regional']['pairs'][pair_key] = {
                        'correlation': corr_val,
                        'p_value': p_val,
                        'significant': is_significant,
                        'regions': f"{roi1_region}_{roi2_region}"
                    }

                    # Ipsilateral (same hemisphere, different regions)
                    if roi1_hemi == roi2_hemi:
                        results['ipsilateral']['pairs'][pair_key] = {
                            'correlation': corr_val,
                            'p_value': p_val,
                            'significant': is_significant,
                            'hemisphere': roi1_hemi,
                            'regions': f"{roi1_region}_{roi2_region}"
                        }

                    # Contralateral (different hemisphere, different regions)
                    elif roi1_hemi != roi2_hemi:
                        results['contralateral']['pairs'][pair_key] = {
                            'correlation': corr_val,
                            'p_value': p_val,
                            'significant': is_significant,
                            'hemispheres': f"{roi1_hemi}_{roi2_hemi}",
                            'regions': f"{roi1_region}_{roi2_region}"
                        }

    # Compute statistics for each connection type
    for connection_type in results.keys():
        pairs = results[connection_type]['pairs']
        pair_count = len(pairs)

        if pair_count > 0:
            # Handle case where 'significant' might be None (when p_values is None)
            significant_count = sum(1 for pair in pairs.values() if pair.get('significant') is True)
            significance_pct = significant_count / pair_count
        else:
            significant_count = 0
            significance_pct = 0.0

        # Store statistics separately from pair data
        results[connection_type]['stats'] = {
            'total_pairs': pair_count,
            'significant_pairs': significant_count,
            'significance_percentage': significance_pct
        }

    # NEW: Group interhemispheric pairs by region+network and compute mean Fisher-Z coherence
    interhemi_pairs = results['interhemispheric']['pairs']
    network_pairs = {}
    network_stats = {}

    for pair_key, pair_data in interhemi_pairs.items():
        region = pair_data.get('region')
        network = pair_data.get('network')

        # Skip if network info is missing
        if region is None or network is None:
            continue

        # Create network-level key (e.g., 'PFCm_DefaultA', 'AMY_lAMY')
        network_key = f"{region}_{network}"

        # Initialize network group if first time seeing this network
        if network_key not in network_pairs:
            network_pairs[network_key] = {
                'correlations': [],
                'fisher_z_values': [],
                'p_values': [],
                'parcel_pairs': [],
                'lh_indices': [],
                'rh_indices': []
            }

        # Add this pair's data to the network group
        network_pairs[network_key]['correlations'].append(pair_data['correlation'])
        network_pairs[network_key]['p_values'].append(pair_data.get('p_value'))
        network_pairs[network_key]['parcel_pairs'].append(pair_key)
        # Note: We don't know which is LH vs RH from just the pair, but we store indices
        network_pairs[network_key]['lh_indices'].append(pair_data.get('roi1_index'))
        network_pairs[network_key]['rh_indices'].append(pair_data.get('roi2_index'))

    # Compute mean Fisher-Z coherence for each network
    for network_key, network_data in network_pairs.items():
        correlations = np.array(network_data['correlations'])

        # Apply Fisher Z-transformation
        fisher_z = fisher_r_to_z(correlations)
        network_data['fisher_z_values'] = fisher_z.tolist()

        # Compute mean coherence (in Fisher-Z space)
        mean_fisher_z = np.nanmean(fisher_z)

        # Store network-level statistics
        p_values_array = np.array([pv for pv in network_data['p_values'] if pv is not None])
        network_stats[network_key] = {
            'mean_fisher_z': float(mean_fisher_z),
            'mean_correlation': float(np.nanmean(correlations)),  # For interpretation
            'n_parcel_pairs': len(correlations),
            'n_significant': int(np.sum(p_values_array < alpha)) if len(p_values_array) > 0 else 0,
            'all_fisher_z_values': fisher_z.tolist()  # For later analysis
        }

    # Add network-level results to interhemispheric category
    results['interhemispheric']['network_pairs'] = network_pairs
    results['interhemispheric']['network_stats'] = network_stats

    return results

# def combine_slow_band_components(time_modes, center_freqs, verbose=True):
#     """
#     Combine all signal components within specific slow bands into multi-component signals.

#     Args:
#         time_modes: Time signals from which the original is a superposition of (modes x channels x samples)
#         center_freqs: Center frequencies of all modes, used to decide which modes to combine

#     Bands:
#         Slow-6: 0–0.01Hz
#         Slow-5: 0.01–0.027Hz
#         Slow-4: 0.027–0.073Hz
#         Slow-3: 0.073–0.198Hz
#         Slow-2: 0.198–0.5Hz
#         Slow-1: 0.5-0.75Hz

#     Returns:
#         dict: Dictionary with band names as keys, each containing:
#             - 'band_signal': Combined signal for the band (sum of all components)
#             - 'components': List of individual mode signals in this band
#             - 'indeces': List of mode indices that belong to this band
#     """
#     # Initialize separate dictionaries for each band to avoid shared reference issue
#     band_signals = {
#         '1': {'band_signal': None, 'components': [], 'indeces': [], 'center_freqs': []},
#         '2': {'band_signal': None, 'components': [], 'indeces': [], 'center_freqs': []},
#         '3': {'band_signal': None, 'components': [], 'indeces': [], 'center_freqs': []},
#         '4': {'band_signal': None, 'components': [], 'indeces': [], 'center_freqs': []},
#         '5': {'band_signal': None, 'components': [], 'indeces': [], 'center_freqs': []},
#         '6': {'band_signal': None, 'components': [], 'indeces': [], 'center_freqs': []},
#         'excluded': {'band_signal': None, 'components': [], 'indeces': [], 'center_freqs': []},
#     }

#     def get_band_number(frequency):
#         if 0.0 < frequency <= 0.01:
#             return "6"
#         elif 0.01 < frequency <= 0.027:
#             return "5"
#         elif 0.027 < frequency <= 0.073:
#             return "4"
#         elif 0.073 < frequency <= 0.198:
#             return "3"
#         elif 0.198 < frequency <= 0.5:
#             return "2"
#         elif 0.5 < frequency <= 0.75:
#             return "1"
#         else:
#             return 'excluded'

#     for idx, (center_frequency, mode_signal) in enumerate(zip(center_freqs, time_modes)):
#         band_number = get_band_number(center_frequency)

#         band_signals[band_number]['indeces'].append(idx)
#         band_signals[band_number]['components'].append(mode_signal)
#         band_signals[band_number]['center_freqs'].append(center_frequency)

#     # Combine components within each band and convert to arrays
#     for key, val in band_signals.items():
#         if len(val['components']) > 0:
#             comps_array = np.array(val['components'])
#             idcs_array = np.array(val['indeces'])
#             freqs_array = np.array(val['center_freqs'])

#             band_signals[key]['components'] = comps_array
#             band_signals[key]['indeces'] = idcs_array
#             band_signals[key]['center_freqs'] = freqs_array

#             # Only sum components for actual slow bands, not for excluded frequencies
#             if key != 'excluded':
#                 # Sum all components in this band to create the band signal
#                 band_signal = np.sum(comps_array, axis=0)
#                 band_signals[key]['band_signal'] = band_signal

#                 if verbose:
#                     print(f'Band Slow-{key}: components={comps_array.shape}, indeces={idcs_array}, center_freqs={freqs_array}, band_signal={band_signal.shape}')
#             else:
#                 # Keep excluded components separate (do not sum them)
#                 band_signals[key]['band_signal'] = None

#                 if verbose:
#                     print(f'Outside of bands: components={comps_array.shape}, indeces={idcs_array}, center_freqs={freqs_array}, signals kept separate (not summed)')
#         elif verbose:
#             print(f'Band Slow-{key}: no components in this band')

#     return band_signals

# def detect_available_channels(band_signal, threshold=1e-10):
#     """
#     Detect which channels have valid timeseries for a given slow-band.

#     Uses RMS (root mean square) energy to identify channels with meaningful
#     signal content. Channels with RMS below threshold are marked unavailable.

#     Args:
#         band_signal: ndarray, shape (n_channels, n_samples)
#             Reconstructed slow-band signal for all channels
#         threshold: float, default=1e-10
#             RMS threshold for channel availability

#     Returns:
#         available_mask: ndarray, shape (n_channels,)
#             Boolean mask where True indicates channel is available
#     """
#     n_channels = band_signal.shape[0]
#     available_mask = np.zeros(n_channels, dtype=bool)

#     for ch_idx in range(n_channels):
#         rms = np.sqrt(np.mean(band_signal[ch_idx, :]**2))
#         available_mask[ch_idx] = (rms >= threshold)

#     return available_mask

def compute_hilbert_transform_per_mode(time_modes, sampling_rate):
    """
    Apply Hilbert Transform to each MVMD mode to extract instantaneous properties.

    This is the correct approach for multi-channel signals:
    1. MVMD decomposes multi-channel signal into K mono-component modes
    2. Hilbert Transform is applied to each mode separately
    3. Instantaneous frequency and amplitude are extracted per mode

    Args:
        time_modes: MVMD time-domain modes with shape (modes, channels, samples)
        sampling_rate: Sampling frequency in Hz (e.g., 1.25 Hz for TR=800ms)

    Returns:
        dict: Dictionary containing per-mode instantaneous properties:
            - 'modes_data': List of dicts, one per mode, each containing:
                - 'mode_idx': Mode index (0-based)
                - 'instantaneous_frequency': (channels, samples)
                - 'instantaneous_amplitude': (channels, samples)
                - 'analytic_signal': Complex analytic signal (channels, samples)
            - 'sampling_rate': Sampling rate in Hz
            - 'n_modes': Number of modes
            - 'n_channels': Number of channels
            - 'n_samples': Number of time samples
    """
    n_modes, n_channels, n_samples = time_modes.shape

    modes_data = []

    # Process each mode independently
    for mode_idx in range(n_modes):
        mode_signal = time_modes[mode_idx, :, :]  # Shape: (channels, samples)

        # Initialize arrays for this mode
        inst_freq = np.zeros((n_channels, n_samples))
        inst_amp = np.zeros((n_channels, n_samples))
        analytic_signals = np.zeros((n_channels, n_samples), dtype=complex)

        # Apply Hilbert transform to each channel of this mode
        for ch_idx in range(n_channels):
            # Get analytic signal using Hilbert transform
            analytic_signal = signal.hilbert(mode_signal[ch_idx, :])
            analytic_signals[ch_idx, :] = analytic_signal

            # Extract instantaneous amplitude (envelope)
            inst_amp[ch_idx, :] = np.abs(analytic_signal)

            # Extract instantaneous phase and unwrap to avoid discontinuities
            inst_phase = np.unwrap(np.angle(analytic_signal))

            # Compute instantaneous frequency from phase derivative
            # f(t) = (1/2π) * dφ/dt * fs
            inst_freq[ch_idx, :] = np.gradient(inst_phase) * sampling_rate / (2 * np.pi)

        # Store data for this mode
        modes_data.append({
            'mode_idx': mode_idx,
            'instantaneous_frequency': inst_freq,
            'instantaneous_amplitude': inst_amp,
            'analytic_signal': analytic_signals
        })

    return {
        'modes_data': modes_data,
        'sampling_rate': sampling_rate,
        'n_modes': n_modes,
        'n_channels': n_channels,
        'n_samples': n_samples
    }


# def compute_hilbert_transform_per_band(band_signals, sampling_rate):
#     """
#     Apply Hilbert Transform to each slow-band signal to extract instantaneous properties.

#     Similar to compute_hilbert_transform_per_mode, but operates on reconstructed band signals
#     instead of individual MVMD modes. Each band signal is the sum of multiple modes.

#     Args:
#         band_signals: Dictionary from combine_slow_band_components(), containing band signals
#         sampling_rate: Sampling frequency in Hz (e.g., 1.25 Hz for TR=800ms)

#     Returns:
#         dict: Dictionary containing per-band instantaneous properties:
#             - 'bands_data': List of dicts, one per band, each containing:
#                 - 'band_key': Band identifier (e.g., '1', '2', '3', '4', '5', '6')
#                 - 'band_name': Full band name (e.g., 'Slow-1', 'Slow-2')
#                 - 'instantaneous_frequency': (channels, samples)
#                 - 'instantaneous_amplitude': (channels, samples)
#                 - 'analytic_signal': Complex analytic signal (channels, samples)
#             - 'sampling_rate': Sampling rate in Hz
#             - 'n_bands': Number of bands processed
#             - 'n_channels': Number of channels
#             - 'n_samples': Number of time samples
#     """
#     bands_data = []

#     # Define band names mapping
#     band_names = {
#         '1': 'Slow-1 (0.5-0.75 Hz)',
#         '2': 'Slow-2 (0.198-0.5 Hz)',
#         '3': 'Slow-3 (0.073-0.198 Hz)',
#         '4': 'Slow-4 (0.027-0.073 Hz)',
#         '5': 'Slow-5 (0.01-0.027 Hz)',
#         '6': 'Slow-6 (0-0.01 Hz)'
#     }

#     n_channels = None
#     n_samples = None

#     # Process each band (skip 'excluded')
#     for band_key in ['1', '2', '3', '4', '5', '6']:
#         if band_key not in band_signals:
#             continue

#         band_data = band_signals[band_key]
#         band_signal = band_data.get('band_signal')

#         # Skip if no signal for this band
#         if band_signal is None or len(band_data['components']) == 0:
#             continue

#         # band_signal shape: (channels, samples)
#         if n_channels is None:
#             n_channels = band_signal.shape[0]
#             n_samples = band_signal.shape[1]

#         # Initialize arrays for this band
#         inst_freq = np.zeros((n_channels, n_samples))
#         inst_amp = np.zeros((n_channels, n_samples))
#         analytic_signals = np.zeros((n_channels, n_samples), dtype=complex)

#         # Apply Hilbert transform to each channel of this band signal
#         for ch_idx in range(n_channels):
#             # Get analytic signal using Hilbert transform
#             analytic_signal = signal.hilbert(band_signal[ch_idx, :])
#             analytic_signals[ch_idx, :] = analytic_signal

#             # Extract instantaneous amplitude (envelope)
#             inst_amp[ch_idx, :] = np.abs(analytic_signal)

#             # Extract instantaneous phase and unwrap to avoid discontinuities
#             inst_phase = np.unwrap(np.angle(analytic_signal))

#             # Compute instantaneous frequency from phase derivative
#             inst_freq[ch_idx, :] = np.gradient(inst_phase) * sampling_rate / (2 * np.pi)

#         # Store data for this band
#         bands_data.append({
#             'band_key': band_key,
#             'band_name': band_names.get(band_key, f'Slow-{band_key}'),
#             'instantaneous_frequency': inst_freq,
#             'instantaneous_amplitude': inst_amp,
#             'analytic_signal': analytic_signals
#         })

#     return {
#         'bands_data': bands_data,
#         'sampling_rate': sampling_rate,
#         'n_bands': len(bands_data),
#         'n_channels': n_channels,
#         'n_samples': n_samples
#     }

def compare_fc_between_groups(group1_fc_results, group2_fc_results, group1_name="Group1", group2_name="Group2"):
    """
    Compare functional connectivity patterns between two groups.

    Args:
        group1_fc_results: List of FC results dictionaries for group 1
        group2_fc_results: List of FC results dictionaries for group 2
        group1_name: Name for group 1 (e.g., "Anhedonic")
        group2_name: Name for group 2 (e.g., "Non-anhedonic")

    Returns:
        dict: Statistical comparison results
    """
    from scipy import stats as scipy_stats

    # Extract connectivity patterns for each group
    def extract_group_connectivity(fc_results_list):
        group_patterns = {
            'interhemispheric': [],
            'cross_regional': [],
            'ipsilateral': [],
            'contralateral': [],
            'all_pairwise': []
        }

        for fc_result in fc_results_list:
            if fc_result and 'static_connectivity_patterns' in fc_result:
                patterns = fc_result['static_connectivity_patterns']
                for pattern_type in group_patterns.keys():
                    if pattern_type in patterns and 'pairs' in patterns[pattern_type]:
                        # Extract correlations from pairs dictionary
                        correlations = [v['correlation'] for v in patterns[pattern_type]['pairs'].values()]
                        group_patterns[pattern_type].extend(correlations)

        return group_patterns

    group1_patterns = extract_group_connectivity(group1_fc_results)
    group2_patterns = extract_group_connectivity(group2_fc_results)

    # Perform statistical comparisons
    comparison_results = {}

    for pattern_type in group1_patterns.keys():
        g1_values = group1_patterns[pattern_type]
        g2_values = group2_patterns[pattern_type]

        if len(g1_values) > 0 and len(g2_values) > 0:
            # Perform t-test
            t_stat, p_val = scipy_stats.ttest_ind(g1_values, g2_values)

            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(g1_values)-1)*np.var(g1_values) + (len(g2_values)-1)*np.var(g2_values)) / (len(g1_values)+len(g2_values)-2))
            cohens_d = (np.mean(g1_values) - np.mean(g2_values)) / pooled_std if pooled_std > 0 else 0

            comparison_results[pattern_type] = {
                'group1_mean': np.mean(g1_values),
                'group1_std': np.std(g1_values),
                'group1_n': len(g1_values),
                'group2_mean': np.mean(g2_values),
                'group2_std': np.std(g2_values),
                'group2_n': len(g2_values),
                'ttest_statistic': t_stat,
                'ttest_pvalue': p_val,
                'cohens_d': cohens_d,
                'significant': p_val < 0.05
            }
        else:
            comparison_results[pattern_type] = {
                'group1_mean': np.mean(g1_values) if g1_values else np.nan,
                'group1_std': np.std(g1_values) if g1_values else np.nan,
                'group1_n': len(g1_values),
                'group2_mean': np.mean(g2_values) if g2_values else np.nan,
                'group2_std': np.std(g2_values) if g2_values else np.nan,
                'group2_n': len(g2_values),
                'ttest_statistic': np.nan,
                'ttest_pvalue': np.nan,
                'cohens_d': np.nan,
                'significant': False,
                'note': 'Insufficient data for comparison'
            }

    return {
        'group1_name': group1_name,
        'group2_name': group2_name,
        'comparisons': comparison_results,
        'group1_patterns': group1_patterns,
        'group2_patterns': group2_patterns
    }


def process_subject(subject_id, manager, loader, cortical_atlas, subcortical_atlas,
                   cortical_roi_extractor, subcortical_roi_extractor, cortical_ROIs, subcortical_ROIs,
                   verbose=True):
    """
    Process a single subject for ROI extraction and functional connectivity analysis.

    Args:
        subject_id: Subject identifier
        manager: SubjectManager instance
        loader: DataLoader instance
        cortical_atlas: CorticalAtlasLookup instance
        subcortical_atlas: SubCorticalAtlasLookup instance
        cortical_roi_extractor: ROIExtractionService for cortical data
        subcortical_roi_extractor: ROIExtractionService for subcortical data
        cortical_ROIs: List of cortical ROI names
        subcortical_ROIs: List of subcortical ROI names
        verbose: Whether to print detailed output

    Returns:
        dict: Subject analysis results
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing Subject: {subject_id}")
        print(f"{'='*60}")

    try:
        # Get subject files
        hammer_files = manager.get_subject_files_by_task(subject_id, 'timeseries', 'hammer')
        if not hammer_files:
            return {
                'subject_id': subject_id,
                'error': 'No hammer task files found',
                'success': False
            }

        subject_file = loader.resolve_file_path(hammer_files[0])

        # Read data from .h5 file
        data = None
        with h5py.File(subject_file, 'r') as file:
            a_group_key = list(file.keys())[0]
            data = np.asarray(file[a_group_key])

        if verbose:
            print(f"Found data with shape: {data.shape}")

        # Segment into timeseries groups
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

        # ROI Validation and Extraction
        cortical_validation_result = cortical_roi_extractor.validate_roi_coverage(cortical_timeseries, cortical_ROIs)
        subcortical_validation_result = subcortical_roi_extractor.validate_roi_coverage(subcortical_timeseries, subcortical_ROIs)

        if verbose:
            print(f"\nCORTICAL ROI Validation Results:")
            print(f"  Valid ROIs: {cortical_validation_result['valid_rois']}")
            print(f"  Invalid ROIs: {cortical_validation_result['invalid_rois']}")

            print(f"\nSUBCORTICAL ROI Validation Results:")
            print(f"  Valid ROIs: {subcortical_validation_result['valid_rois']}")
            print(f"  Invalid ROIs: {subcortical_validation_result['invalid_rois']}")

        # Extract ROI timeseries
        cortical_roi_timeseries = None
        if cortical_validation_result['valid_rois'] and not cortical_validation_result['coverage_issues']:
            cortical_roi_timeseries = cortical_roi_extractor.extract_roi_timeseries(
                cortical_timeseries,
                cortical_ROIs,
                aggregation_method='all'
            )

        subcortical_roi_timeseries = None
        if subcortical_validation_result['valid_rois'] and not subcortical_validation_result['coverage_issues']:
            subcortical_roi_timeseries = subcortical_roi_extractor.extract_roi_timeseries(
                subcortical_timeseries,
                subcortical_ROIs,
                aggregation_method='all'
            )

        # Hemisphere-specific extraction
        cortical_right_timeseries = None
        cortical_left_timeseries = None
        subcortical_right_timeseries = None
        subcortical_left_timeseries = None
        # individual channel timeseries with label mapping
        vmPFC_right_channels = None
        vmPFC_left_channels = None
        amy_right_channels = None
        amy_left_channels = None
        channel_label_map = None

        cortical_valid_rois = cortical_validation_result['valid_rois']
        cortical_parcel_labels = {}  # Maps ROI -> hemisphere -> list of parcel labels

        if cortical_roi_extractor.supports_hemisphere_queries() and cortical_valid_rois:
            if verbose:
                print(f"\n=== HEMISPHERE-SPECIFIC CORTICAL EXTRACTION ===")

            cortical_right_timeseries = cortical_roi_extractor.extract_roi_timeseries_by_hemisphere(
                cortical_timeseries,
                cortical_valid_rois,
                hemisphere='RH',
                aggregation_method='all'
            )

            cortical_left_timeseries = cortical_roi_extractor.extract_roi_timeseries_by_hemisphere(
                cortical_timeseries,
                cortical_valid_rois,
                hemisphere='LH',
                aggregation_method='all'
            )

            # Create parcel labels for each ROI and hemisphere
            for roi_name in cortical_valid_rois:
                if roi_name not in cortical_parcel_labels:
                    cortical_parcel_labels[roi_name] = {'RH': [], 'LH': []}

                # Right hemisphere labels - get actual parcel names with network info
                if roi_name in cortical_right_timeseries:
                    rh_indices_dict = cortical_atlas.get_roi_indices_by_hemisphere([roi_name], hemisphere='RH')
                    rh_indices = rh_indices_dict.get(roi_name, [])
                    rh_parcel_labels = []

                    for idx in rh_indices:
                        # Get full parcel name (e.g., '17networks_RH_DefaultA_PFCm_1')
                        full_name = cortical_atlas.get_parcel_name(idx)
                        if full_name:
                            # Extract network and create compact label
                            # Format: 17networks_{hemi}_{network}_{region}_{subarea}
                            parts = full_name.split('_')
                            if len(parts) >= 4:
                                # parts: ['17networks', 'RH', 'DefaultA', 'PFCm', '1']
                                hemi = parts[1]
                                network = parts[2]
                                region = parts[3]
                                subarea = parts[4] if len(parts) > 4 else ''
                                # Create label: PFCm_RH_DefaultA_p1 (includes network, p = parcel)
                                label = f'{region}_{hemi}_{network}'
                                if subarea:
                                    label += f'_p{subarea}'
                                rh_parcel_labels.append(label)
                            else:
                                # Fallback if parsing fails
                                rh_parcel_labels.append(f'{roi_name}_RH_parcel{len(rh_parcel_labels)+1}')
                        else:
                            rh_parcel_labels.append(f'{roi_name}_RH_parcel{len(rh_parcel_labels)+1}')

                    cortical_parcel_labels[roi_name]['RH'] = rh_parcel_labels

                # Left hemisphere labels
                if roi_name in cortical_left_timeseries:
                    lh_indices_dict = cortical_atlas.get_roi_indices_by_hemisphere([roi_name], hemisphere='LH')
                    lh_indices = lh_indices_dict.get(roi_name, [])
                    lh_parcel_labels = []

                    for idx in lh_indices:
                        full_name = cortical_atlas.get_parcel_name(idx)
                        if full_name:
                            parts = full_name.split('_')
                            if len(parts) >= 4:
                                hemi = parts[1]
                                network = parts[2]
                                region = parts[3]
                                subarea = parts[4] if len(parts) > 4 else ''
                                label = f'{region}_{hemi}_{network}'
                                if subarea:
                                    label += f'_p{subarea}'
                                lh_parcel_labels.append(label)
                            else:
                                lh_parcel_labels.append(f'{roi_name}_LH_parcel{len(lh_parcel_labels)+1}')
                        else:
                            lh_parcel_labels.append(f'{roi_name}_LH_parcel{len(lh_parcel_labels)+1}')

                    cortical_parcel_labels[roi_name]['LH'] = lh_parcel_labels

            if verbose:
                print(f"RIGHT hemisphere extraction results:")
                for roi_name, timeseries in cortical_right_timeseries.items():
                    if timeseries.size > 0:
                        print(f"  {roi_name}: shape {timeseries.shape}")

                print(f"LEFT hemisphere extraction results:")
                for roi_name, timeseries in cortical_left_timeseries.items():
                    if timeseries.size > 0:
                        print(f"  {roi_name}: shape {timeseries.shape}")

            # Extract individual vmPFC channels with proper labeling (only 'all' aggregation)
            # Assume 'all' aggregation: each ROI contains multiple parcels as 2D arrays
            vmPFC_right_channels = np.vstack([cortical_right_timeseries['PFCm'],
                                            cortical_right_timeseries['PFCv']])
            vmPFC_left_channels = np.vstack([cortical_left_timeseries['PFCm'],
                                           cortical_left_timeseries['PFCv']])

            # Create channel labels directly from parcel labels (no need for mapping)
            # Build list of all channel labels in the order they appear in vmPFC_right/left_channels
            cortical_channel_labels = []

            # Right hemisphere labels (PFCm first, then PFCv)
            for roi_name in ['PFCm', 'PFCv']:
                rh_labels = cortical_parcel_labels.get(roi_name, {}).get('RH', [])
                cortical_channel_labels.extend(rh_labels)

            # Left hemisphere labels (PFCm first, then PFCv)
            for roi_name in ['PFCm', 'PFCv']:
                lh_labels = cortical_parcel_labels.get(roi_name, {}).get('LH', [])
                cortical_channel_labels.extend(lh_labels)

            # Channel label map is now identity (label -> label) for display purposes
            channel_label_map = {label: label for label in cortical_channel_labels}

            if verbose:
                print(f"Individual cortical (PFCm + PFCv) channel extraction results:")
                print(f"  RIGHT channels: shape {vmPFC_right_channels.shape}")
                print(f"  LEFT channels: shape {vmPFC_left_channels.shape}")
                print(f"  Cortical channel labels: {len(cortical_channel_labels)} channels")

        subcortical_valid_rois = subcortical_validation_result['valid_rois']
        subcortical_parcel_labels = {}  # Maps ROI -> hemisphere -> list of parcel labels

        if subcortical_roi_extractor.supports_hemisphere_queries() and subcortical_valid_rois:
            if verbose:
                print(f"\n=== HEMISPHERE-SPECIFIC SUBCORTICAL EXTRACTION ===")

            subcortical_right_timeseries = subcortical_roi_extractor.extract_roi_timeseries_by_hemisphere(
                subcortical_timeseries,
                subcortical_valid_rois,
                hemisphere='rh',
                aggregation_method='all'
            )

            subcortical_left_timeseries = subcortical_roi_extractor.extract_roi_timeseries_by_hemisphere(
                subcortical_timeseries,
                subcortical_valid_rois,
                hemisphere='lh',
                aggregation_method='all'
            )

            # Create parcel labels for each ROI and hemisphere
            # Normalize subcortical labels to match cortical format: ROI_HEMI_subdivision
            for roi_name in subcortical_valid_rois:
                if roi_name not in subcortical_parcel_labels:
                    subcortical_parcel_labels[roi_name] = {'rh': [], 'lh': []}

                # Right hemisphere labels - normalize to match cortical format
                if roi_name in subcortical_right_timeseries:
                    # Get parcel indices for this ROI in right hemisphere
                    rh_indices_dict = subcortical_atlas.get_roi_indices_by_hemisphere([roi_name], hemisphere='rh')
                    rh_indices = rh_indices_dict.get(roi_name, [])
                    rh_parcel_names = []

                    for idx in rh_indices:
                        # Get the actual parcel name from the atlas (e.g., 'lAMY-rh', 'mAMY-rh')
                        parcel_name = subcortical_atlas.get_parcel_name(idx)
                        if parcel_name:
                            # Normalize format: AMY_RH_lAMY (consistent with cortical labels)
                            # Remove hemisphere suffix from atlas name and convert to standard format
                            subdivision = parcel_name.replace('-rh', '').replace('-lh', '')
                            normalized_label = f'{roi_name}_RH_{subdivision}'
                            rh_parcel_names.append(normalized_label)
                        else:
                            # Fallback if parcel name not available
                            rh_parcel_names.append(f'{roi_name}_RH_parcel{len(rh_parcel_names)+1}')

                    subcortical_parcel_labels[roi_name]['rh'] = rh_parcel_names

                # Left hemisphere labels - normalize to match cortical format
                if roi_name in subcortical_left_timeseries:
                    lh_indices_dict = subcortical_atlas.get_roi_indices_by_hemisphere([roi_name], hemisphere='lh')
                    lh_indices = lh_indices_dict.get(roi_name, [])
                    lh_parcel_names = []

                    for idx in lh_indices:
                        parcel_name = subcortical_atlas.get_parcel_name(idx)
                        if parcel_name:
                            # Normalize format: AMY_LH_mAMY (consistent with cortical labels)
                            subdivision = parcel_name.replace('-rh', '').replace('-lh', '')
                            normalized_label = f'{roi_name}_LH_{subdivision}'
                            lh_parcel_names.append(normalized_label)
                        else:
                            lh_parcel_names.append(f'{roi_name}_LH_parcel{len(lh_parcel_names)+1}')

                    subcortical_parcel_labels[roi_name]['lh'] = lh_parcel_names

            if verbose:
                print(f"RIGHT hemisphere extraction results:")
                for roi_name, timeseries in subcortical_right_timeseries.items():
                    if timeseries.size > 0:
                        print(f"  {roi_name}: shape {timeseries.shape}")

                print(f"LEFT hemisphere extraction results:")
                for roi_name, timeseries in subcortical_left_timeseries.items():
                    if timeseries.size > 0:
                        print(f"  {roi_name}: shape {timeseries.shape}")

            # Extract individual AMY channels with proper labeling (only 'all' aggregation)
            # Assume 'all' aggregation: AMY contains multiple parcels as 2D arrays
            amy_right_channels = subcortical_right_timeseries['AMY']
            amy_left_channels = subcortical_left_timeseries['AMY']

            # Build subcortical channel labels using actual parcel names from atlas (lAMY, mAMY)
            subcortical_channel_labels = []
            amy_rh_labels = subcortical_parcel_labels.get('AMY', {}).get('rh', [])
            amy_lh_labels = subcortical_parcel_labels.get('AMY', {}).get('lh', [])

            subcortical_channel_labels.extend(amy_rh_labels)
            subcortical_channel_labels.extend(amy_lh_labels)

            # Add subcortical labels to channel_label_map (identity mapping)
            for label in subcortical_channel_labels:
                channel_label_map[label] = label

            if verbose:
                print(f"Individual subcortical (AMY) channel extraction results:")
                print(f"  RIGHT channels: shape {amy_right_channels.shape}")
                print(f"  LEFT channels: shape {amy_left_channels.shape}")
                print(f"  Subcortical channel labels: {len(subcortical_channel_labels)} channels")
                print(f"  Total channel mapping: {len(channel_label_map)} channels")
                print(f"  Sample channel labels: {list(channel_label_map.keys())[:8]}{'...' if len(channel_label_map) > 8 else ''}")

        # Require all timeseries to perform analyses
        required_timeseries = [cortical_right_timeseries, cortical_left_timeseries,
                                                     subcortical_right_timeseries, subcortical_left_timeseries,
                                                     vmPFC_right_channels, vmPFC_left_channels,
                                                     amy_right_channels, amy_left_channels,
                                                     channel_label_map]
        is_missing_timeseries = any([v is None for v in required_timeseries])

        if is_missing_timeseries:
            formatted_list = '\n\t- '.join(required_timeseries)
            raise ValueError(f'Not all timeseries are present. Expected channels:\n\t{formatted_list}')

        # Keeping track of all original channel labels from atlases
        all_channel_labels = cortical_channel_labels + subcortical_channel_labels

        # Combine into multivariate signal (multi-channel),
        # and get analytical signals through Hilbert transform (HT)
        # Combine all channel timeseries in order
        all_channels = np.vstack([
            vmPFC_right_channels,
            vmPFC_left_channels,
            amy_right_channels,
            amy_left_channels
        ])

        if verbose:
            print(f"Combined all channels into a multivariate signal shape: {all_channels.shape}")

        # Perform Hilbert transform on each timeseries
        hilbert_transforms = signal.hilbert(all_channels)

        # Derive Analytic signals z(t) = x(t) + j * H{x(t)}
        analytic_timeseries = all_channels + 1j * hilbert_transforms

        # Activity analysis
        activity_results = None

        if verbose:
            print(f"\n=== ACTIVITY ANALYSIS ===")

        # Create Activity timeseries dictionary using actual parcel labels
        activity_timeseries = {}

        # Map each channel timeseries to its actual parcel label
        for i, channel_label in enumerate(all_channel_labels):
            activity_timeseries[channel_label] = all_channels[i]

        # Compute envelope of analytic signal
        analytic_envelope = np.abs(analytic_timeseries) # Activity

        # Apply low-pass filter for smoothing envelope
        # Filter properties
        filter_order = 2
        cutoff_frequency = 0.2  # Frequency at which signal starts to attenuate
                                # Digital filter critical frequencies must be 0 < Wn < 1
        normalized_cutoff = cutoff_frequency / NYQUIST_FREQUENCY
        b, a = signal.butter(filter_order, normalized_cutoff, btype='low', analog=False)

        filtered_timeseries = signal.lfilter(b, a, analytic_timeseries)
        filtered_envelope = np.abs(filtered_timeseries)

        if verbose:
            print(f"All Channels shape: {all_channels.shape}")
            print(f"Hilbert Transformed channels shape: {hilbert_transforms.shape}")
            print(f"Analytic channels shape: {analytic_timeseries.shape}")

        activity_results = {
            'all_channels': all_channels,
            'analytic_signal': analytic_timeseries,
            'hilbert_transforms': hilbert_transforms,
            'analytic_envelope': analytic_envelope, # Activity
            'smoothed_envelope': filtered_envelope, # LP-Filtered acitivty
            'timeseries_used': activity_timeseries,
            'filtered_timeseries': filtered_timeseries,
            'filter_used': (b, a),
            'channel_label_map': channel_label_map,
            'channel_labels': all_channel_labels
        }


        # Static functional connectivity analysis
        static_fc_results = None

        if verbose:
            print(f"\n=== FUNCTIONAL CONNECTIVITY ANALYSIS ===")

        # Create FC timeseries dictionary using actual parcel labels
        fc_timeseries = {}

        # Map each channel timeseries to its actual parcel label
        for i, channel_label in enumerate(all_channel_labels):
            fc_timeseries[channel_label] = all_channels[i]

        fc_matrix, fc_labels, fc_pvalues = compute_fc_matrix(fc_timeseries)

        if fc_matrix is not None:
            if verbose:
                fc_labels_ordered = '\n\t- '.join(fc_labels)
                print(f"FC Matrix shape: {fc_matrix.shape}")
                print(f"ROI labels (same order): \n\t- {fc_labels_ordered}")

            connectivity_patterns = analyze_connectivity_patterns(fc_matrix, fc_labels, fc_pvalues)

            # NEW: Extract network-level interhemispheric coherence
            static_interhemi_networks = connectivity_patterns['interhemispheric'].get('network_stats', {})

            if verbose:
                pattern_labels = '\n\t- '.join(connectivity_patterns['interhemispheric']['pairs'].keys())
                print(f"\nInterhemispheric connections (same order): \n\t- {pattern_labels}")

                # NEW: Print network-level coherence summary
                if static_interhemi_networks:
                    print(f"\nInterhemispheric Network Coherence (Fisher-Z):")
                    for network_key, network_data in static_interhemi_networks.items():
                        print(f"  {network_key}: mean_z = {network_data['mean_fisher_z']:.3f}, "
                              f"n_pairs = {network_data['n_parcel_pairs']}")

            if verbose:
                print(f"\nConnectivity Pattern Analysis:")
                print(f"  Total pairwise connections: {connectivity_patterns['all_pairwise']['stats']['total_pairs']}")
                print(f"  Interhemispheric connections: {connectivity_patterns['interhemispheric']['stats']['total_pairs']}")
                print(f"  Cross-regional connections: {connectivity_patterns['cross_regional']['stats']['total_pairs']}")

            static_fc_results = {
                'static_fc_matrix': fc_matrix,
                'static_fc_labels': fc_labels,
                'static_fc_pvalues': fc_pvalues,
                'static_connectivity_patterns': connectivity_patterns,
                'interhemispheric_network_coherence': static_interhemi_networks,  # NEW
                'timeseries_used': fc_timeseries,
                'channel_label_map': channel_label_map
            }

        # TODO: Dynamic functional connectivity analysis

        # TODO: Multiscale analysis
        # 1. Static FC
        # 2. Dynamic FC

        # Multiscale functional connectivity analysis
        mvmd_config = None
        mvmd = MVMD(config=mvmd_config)
        mvmd_result = mvmd.decompose(all_channels, num_modes=10)

        time_modes = mvmd_result['time_modes']
        center_freqs = mvmd_result['center_freqs'][-1, :]

        reconstructed_timeseries = np.sum(time_modes, axis=0)
        reconstruction_error = np.linalg.norm(all_channels - reconstructed_timeseries) / np.linalg.norm(analytic_timeseries)

        # Create index-based channel label map for MVMD plots
        mvmd_channel_label_map = {idx: label for idx, label in enumerate(all_channel_labels)}

        mvmd_result = {
            **mvmd_result,
            'ts_reconstruction': reconstructed_timeseries,
            'reconstruction_error': reconstruction_error,
            'channel_label_map': mvmd_channel_label_map,
        }

        if verbose:
            print(f"\nExtracted centre frequencies: {center_freqs} Hz")
            print(f"Modes shape: {time_modes.shape}")
            print(f"Signal reconstruction error: {reconstruction_error:.4f}")

        # ===== HILBERT SPECTRAL ANALYSIS =====
        if verbose:
            print(f"\n{'='*80}")
            print("HILBERT SPECTRAL ANALYSIS")
            print(f"{'='*80}")

        # Apply Hilbert Transform to each mode to extract instantaneous properties
        if verbose:
            print("\nComputing Hilbert Transform per mode...")

        hsa_data = compute_hilbert_transform_per_mode(
            time_modes=time_modes,
            sampling_rate=SAMPLING_RATE
        )

        if verbose:
            print(f"  Processed {hsa_data['n_modes']} modes")
            print(f"  Channels per mode: {hsa_data['n_channels']}")
            print(f"  Time samples: {hsa_data['n_samples']}")

        # Store HSA results in mvmd_result
        mvmd_result['hilbert_spectral_analysis'] = hsa_data

        if verbose:
            print(f"\n{'='*80}")
            print("HILBERT SPECTRAL ANALYSIS COMPLETE")
            print(f"{'='*80}")

        # Compute mode-wise FC and aggregate to slow-bands
        slow_band_fc_results = {}
        if mvmd_result['success']:
            # ===== MODE-LEVEL FUNCTIONAL CONNECTIVITY =====
            if verbose:
                print(f"\n{'='*80}")
                print("MODE-LEVEL FUNCTIONAL CONNECTIVITY ANALYSIS")
                print(f"{'='*80}")

            # STEP 1: Compute FC for each mode independently
            mode_fc_data = compute_fc_per_mode(
                time_modes=time_modes,
                channel_labels=all_channel_labels,
                verbose=verbose
            )

            # STEP 2: Aggregate mode-level FC into slow-bands using Fisher Z-transformation
            if verbose:
                print(f"\n{'='*80}")
                print("AGGREGATING MODE-LEVEL FC TO SLOW-BANDS")
                print(f"{'='*80}")

            slow_band_fc_results = aggregate_mode_fc_to_bands(
                mode_fc_data=mode_fc_data,
                mode_frequencies=center_freqs,
                verbose=verbose
            )

            # STEP 3: Enrich band FC results with metadata and connectivity patterns
            if verbose:
                print(f"\n{'='*80}")
                print("ENRICHING SLOW-BAND FC RESULTS")
                print(f"{'='*80}")

            for band_key in ['6', '5', '4', '3', '2', '1']:
                band_fc = slow_band_fc_results.get(f'slow-{band_key}')

                # Skip if no modes in this band
                if band_fc is None:
                    if verbose:
                        print(f"\nSlow-{band_key}: No modes in this band, skipping")
                    continue

                mode_indices = band_fc['mode_indices']
                n_channels = mode_fc_data['n_channels']

                if verbose:
                    print(f"\nSlow-{band_key}:")
                    print(f"  Modes in band: {band_fc['n_modes_in_band']}")
                    print(f"  Mode indices: {mode_indices}")
                    print(f"  Center frequencies: {[f'{f:.4f}' for f in band_fc['mode_frequencies']]}")

                # Analyze connectivity patterns (operates on FC matrix directly)
                band_connectivity_patterns = analyze_connectivity_patterns(
                    band_fc['fc_matrix'],
                    band_fc['fc_labels'],
                    band_fc['fc_pvalues'],
                    alpha=0.05
                )

                # NEW: Extract network-level interhemispheric coherence for this band
                band_interhemi_networks = band_connectivity_patterns['interhemispheric'].get('network_stats', {})

                # Add metadata to results
                band_fc['connectivity_patterns'] = band_connectivity_patterns
                band_fc['interhemispheric_network_coherence'] = band_interhemi_networks  # NEW
                band_fc['components_used'] = band_fc['mode_indices']  # Alias for backward compatibility
                band_fc['center_freqs'] = band_fc['mode_frequencies']  # Alias for backward compatibility
                band_fc['channel_label_map'] = mvmd_channel_label_map
                band_fc['frequency_range'] = get_frequency_range(band_key)

            if verbose:
                print(f"\n{'='*80}")
                print("SLOW-BAND FC ANALYSIS COMPLETE")
                print(f"{'='*80}")
                print(f"  Total bands with results: {len(slow_band_fc_results)}")
                print(f"  Bands: {list(slow_band_fc_results.keys())}")

        return {
            'subject_id': subject_id,
            'success': True,
            'data_shape': data.shape,
            'roi_extraction_results': {
                'cortical': {
                    'atlas_name': cortical_atlas.atlas_name,
                    'roi_timeseries': cortical_roi_timeseries,
                    'requested_rois': cortical_ROIs,
                    'extraction_successful': cortical_roi_timeseries is not None,
                    'hemisphere_specific': {
                        'right_hemisphere': cortical_right_timeseries,
                        'left_hemisphere': cortical_left_timeseries,
                        'supports_hemisphere_queries': cortical_roi_extractor.supports_hemisphere_queries()
                    },
                    'parcel_labels': cortical_parcel_labels
                },
                'subcortical': {
                    'atlas_name': subcortical_atlas.atlas_name,
                    'roi_timeseries': subcortical_roi_timeseries,
                    'requested_rois': subcortical_ROIs,
                    'extraction_successful': subcortical_roi_timeseries is not None,
                    'hemisphere_specific': {
                        'right_hemisphere': subcortical_right_timeseries,
                        'left_hemisphere': subcortical_left_timeseries,
                        'supports_hemisphere_queries': subcortical_roi_extractor.supports_hemisphere_queries()
                    },
                    'parcel_labels': subcortical_parcel_labels
                }
            },
            'activity': activity_results,
            'static_functional_connectivity': static_fc_results,
            'channel_signals': {
                'vmPFC_right_channels': vmPFC_right_channels,
                'vmPFC_left_channels': vmPFC_left_channels,
                'amy_right_channels': amy_right_channels,
                'amy_left_channels': amy_left_channels,
                'channel_label_map': channel_label_map
            },
            'mvmd': mvmd_result,
            'slow_band_fc': slow_band_fc_results,
        }

    except Exception as e:
        if verbose:
            print(f"[ERROR] Failed to process subject {subject_id}: {str(e)}")
        return {
            'subject_id': subject_id,
            'error': str(e),
            'success': False
        }


def main(mask_diagonal=False, mask_nonsignificant=False, create_plots=True, show_plots=True, save_figures=False, verbose=True, subjects_per_group=None):
    """Main function for FC MVP analysis"""
    from datetime import datetime

    print("=== Functional Connectivity MVP ===")

    # Create timestamped parent folder for this analysis run only if saving figures
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if save_figures:
        run_parent_dir = get_analysis_path(f'analysis_runs/run_{run_timestamp}')
        run_parent_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nAnalysis Run Directory: {run_parent_dir}")
        print(f"Timestamp: {run_timestamp}")
        print(f"All outputs for this run will be saved in this directory\n")
    else:
        run_parent_dir = None
        print(f"\nRunning in no-save mode - figures will not be saved to disk\n")

    # ===== CONFIGURATION FOR MULTI-SUBJECT ANALYSIS =====
    LIMIT_SUBJECTS = subjects_per_group is not None  # Enable limiting if subjects_per_group is specified
    MAX_SUBJECTS_PER_GROUP = subjects_per_group if subjects_per_group is not None else 8  # Use specified limit or default

    print(f"Configuration:")
    print(f"  Subject limiting: {'ENABLED' if LIMIT_SUBJECTS else 'DISABLED'}")
    if LIMIT_SUBJECTS:
        print(f"  Max subjects per group: {MAX_SUBJECTS_PER_GROUP}")
    print(f"  Individual plots: {'ENABLED' if show_plots else 'DISABLED'}")
    print(f"  Verbose output: {'ENABLED' if verbose else 'DISABLED'}")
    print()

    # Initialize data infrastructure
    loader = DataLoader()
    manager = SubjectManager(data_loader=loader)

    print(f"[OK] Loaded manifest with {len(loader.get_all_subject_ids())} subjects")

    # Get available analysis groups
    groups = loader.get_analysis_groups()
    print(f"Available groups: {list(groups.keys())}")

    # Show data availability summary
    availability = manager.get_subjects_availability_summary()
    print(f"\nData Availability Summary:")
    print(f"  Total subjects in manifest: {availability['total_subjects']}")
    print(f"  Downloaded locally: {availability['downloaded_subjects']}")
    print(f"  With timeseries metadata: {availability['with_timeseries_data']}")
    print(f"  Ready for processing: {availability['breakdown']['ready_for_processing']}")

    # Decide on processing mode
    use_downloaded_only = availability['downloaded_subjects'] > 0
    if use_downloaded_only:
        print(f"\nUsing DOWNLOADED-ONLY mode ({availability['downloaded_subjects']} subjects)")
        print("  This ensures all subjects have locally available data files")
    else:
        print(f"\nUsing ALL-AVAILABLE mode ({availability['with_timeseries_data']} subjects)")
        print("  Warning: Some subjects may not have locally downloaded data")

    # Use a valid group name from the manifest
    # The manifest shows: anhedonic_vs_non_anhedonic, anhedonic_patients_vs_controls, etc.
    group_name = 'anhedonic_vs_non_anhedonic'  # Use actual group from manifest

    # Get subjects for analysis
    low_anhedonic_subjects = manager.filter_subjects(
        groups=[group_name],
        classifications={'anhedonic_status': 'anhedonic',
                         'anhedonia_class': 'low-anhedonic'},
        data_requirements=['timeseries'],
        downloaded_only=use_downloaded_only,
    )
    high_anhedonic_subjects = manager.filter_subjects(
        groups=[group_name],
        classifications={'anhedonic_status': 'anhedonic',
                         'anhedonia_class': 'high-anhedonic'},
        data_requirements=['timeseries'],
        downloaded_only=use_downloaded_only,
    )
    anhedonic_subjects = low_anhedonic_subjects + high_anhedonic_subjects

    non_anhedonic_subjects = manager.filter_subjects(  # Fixed typo
        groups=[group_name],
        classifications={'anhedonic_status': 'non-anhedonic'},
        data_requirements=['timeseries'],
        downloaded_only=use_downloaded_only,
    )

    print(f"\nSubject Selection:")
    print(f"  Anhedonic subjects: {len(anhedonic_subjects)}")
    print(f"\tLOW: {len(low_anhedonic_subjects)}")
    print(f"\tHIGH: {len(high_anhedonic_subjects)}")
    print(f"  Non-anhedonic subjects: {len(non_anhedonic_subjects)}")

    # Validate file access for processing
    print(f"\nValidating file access:")
    accessible_anhedonic = get_accessible_subjects_from_file(
        subjects=anhedonic_subjects,
        subject_manager=manager,
        file_loader=loader,
        task='hammer'
    )
    accessible_non_anhedonic = get_accessible_subjects_from_file(
        subjects=non_anhedonic_subjects,
        subject_manager=manager,
        file_loader=loader,
        task='hammer'
    )

    # Report final accessible counts
    print(f"\nFinal Processing Summary:")
    contact_str = '\n - '
    print(f"  Anhedonic subjects (accessible): {len(accessible_anhedonic)}")
    print(f"  Non-anhedonic subjects (accessible): {len(accessible_non_anhedonic)}")
    print(f"  Total ready for FC analysis (hammer task only): {len(accessible_anhedonic) + len(accessible_non_anhedonic)}")

    # Check if we have any accessible subjects
    if len(accessible_anhedonic) == 0 and len(accessible_non_anhedonic) == 0:
        print(f"\n[ERROR] No subjects have actually downloaded timeseries files!")
        print(f"   The files exist as git-annex symlinks but haven't been fetched yet.")
        print(f"\n   To download timeseries data, run:")
        print(f"   cd {loader.base_path}")
        print(f"   datalad get fMRI_timeseries_clean_denoised_GSR_parcellated/")
        print(f"\n   Or use the preprocessing pipeline with timeseries data type enabled.")
        return {
            'error': 'No downloaded timeseries files',
            'anhedonic_subjects': [],
            'non_anhedonic_subjects': [],
            'processing_mode': 'downloaded_only' if use_downloaded_only else 'all_available',
        }

    # Show example hammer task file paths for first few accessible subjects
    if accessible_anhedonic:
        print(f"\nExample accessible hammer task file paths:")
        for subject_id in accessible_anhedonic[:2]:  # Show first 2
            print(f"\n  Subject: {subject_id}")
            try:
                # Get only hammer task files
                hammer_files = manager.get_subject_files_by_task(subject_id, 'timeseries', 'hammer')
                for file_path in hammer_files[:2]:  # Show first 2 hammer files
                    full_path = loader.resolve_file_path(file_path)
                    print(f"    {full_path}")
            except Exception as e:
                print(f"    Error: {e}")

    # ===== ATLAS INITIALIZATION =====
    print(f"\n{'='*50}")
    print(f"INITIALIZING ATLAS SYSTEMS")
    print(f"{'='*50}")

    # Initialize modular ROI extraction system
    cortical_lut_file = Path(__file__).parent / 'parcellations/cortical/yeo17/400Parcels_Yeo2011_17Networks_info.txt'
    subcortical_lut_file = Path(__file__).parent / 'parcellations/subcortical/tian/Tian_Subcortex_S2_3T_label.txt'
    cortical_atlas = CorticalAtlasLookup(cortical_lut_file)
    subcortical_atlas = SubCorticalAtlasLookup(subcortical_lut_file)
    cortical_roi_extractor = ROIExtractionService(cortical_atlas)
    subcortical_roi_extractor = ROIExtractionService(subcortical_atlas)

    # Define ROIs of interest
    cortical_ROIs = [
        'PFCm',  # medial PFC
        'PFCv',  # ventral PFC
    ]

    subcortical_ROIs = [
        'AMY',  # whole amygdala
    ]

    print(f"Initialized atlases:")
    print(f"  Cortical: {cortical_atlas.atlas_name} ({cortical_atlas.total_parcels} parcels)")
    print(f"  Subcortical: {subcortical_atlas.atlas_name} ({subcortical_atlas.total_parcels} parcels)")
    print(f"ROIs of interest:")
    print(f"  Cortical: {cortical_ROIs}")
    print(f"  Subcortical: {subcortical_ROIs}")

    # ===== MULTI-SUBJECT PROCESSING =====
    print(f"\n{'='*80}")
    print(f"STARTING MULTI-SUBJECT ANALYSIS")
    print(f"{'='*80}")

    # Apply subject limiting if enabled
    non_anhedonic_subjects_to_process = []
    low_anhedonic_to_process = []
    high_anhedonic_to_process = []

    if LIMIT_SUBJECTS:
        # Separate low-anhedonic and high-anhedonic subjects for proper sampling
        low_anhedonic_to_process = low_anhedonic_subjects[:MAX_SUBJECTS_PER_GROUP]
        high_anhedonic_to_process = high_anhedonic_subjects[:MAX_SUBJECTS_PER_GROUP]
        non_anhedonic_subjects_to_process = accessible_non_anhedonic[:MAX_SUBJECTS_PER_GROUP]

        # Combine anhedonic subjects after sampling
        anhedonic_subjects_to_process = low_anhedonic_to_process + high_anhedonic_to_process

        print(f"LIMITING ENABLED: Processing {len(low_anhedonic_to_process)} low-anhedonic + {len(high_anhedonic_to_process)} high-anhedonic + {len(non_anhedonic_subjects_to_process)} non-anhedonic subjects")
        print(f"  Total: {len(anhedonic_subjects_to_process)} anhedonic + {len(non_anhedonic_subjects_to_process)} non-anhedonic = {len(anhedonic_subjects_to_process) + len(non_anhedonic_subjects_to_process)} subjects")
    else:
        anhedonic_subjects_to_process = accessible_anhedonic
        non_anhedonic_subjects_to_process = accessible_non_anhedonic
        print(f"FULL ANALYSIS: Processing {len(anhedonic_subjects_to_process)} anhedonic + {len(non_anhedonic_subjects_to_process)} non-anhedonic subjects")

    # Process all subjects
    anhedonic_results = {}
    non_anhedonic_results = {}

    # Process anhedonic subjects
    print(f"\n{'='*50}")
    print(f"PROCESSING ANHEDONIC SUBJECTS ({len(anhedonic_subjects_to_process)})")
    print(f"{'='*50}")

    for i, subject_id in enumerate(anhedonic_subjects_to_process, 1):
        print(f"\n[{i}/{len(anhedonic_subjects_to_process)}] Processing anhedonic subject: {subject_id}")

        subject_result = process_subject(
            subject_id, manager, loader, cortical_atlas, subcortical_atlas,
            cortical_roi_extractor, subcortical_roi_extractor, cortical_ROIs, subcortical_ROIs,
            verbose=verbose
        )

        anhedonic_results[subject_id] = subject_result

        if subject_result['success']:
            print(f"    ✅ Success: {subject_id}")
        else:
            print(f"    ❌ Failed: {subject_id} - {subject_result.get('error', 'Unknown error')}")

    # Process non-anhedonic subjects
    print(f"\n{'='*50}")
    print(f"PROCESSING NON-ANHEDONIC SUBJECTS ({len(non_anhedonic_subjects_to_process)})")
    print(f"{'='*50}")

    for i, subject_id in enumerate(non_anhedonic_subjects_to_process, 1):
        print(f"\n[{i}/{len(non_anhedonic_subjects_to_process)}] Processing non-anhedonic subject: {subject_id}")

        subject_result = process_subject(
            subject_id, manager, loader, cortical_atlas, subcortical_atlas,
            cortical_roi_extractor, subcortical_roi_extractor, cortical_ROIs, subcortical_ROIs,
            verbose=verbose
        )

        non_anhedonic_results[subject_id] = subject_result

        if subject_result['success']:
            print(f"    ✅ Success: {subject_id}")
        else:
            print(f"    ❌ Failed: {subject_id} - {subject_result.get('error', 'Unknown error')}")

    # ===== RESULTS SUMMARY =====
    print(f"\n{'='*80}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*80}")

    anhedonic_success = sum(1 for r in anhedonic_results.values() if r['success'])
    non_anhedonic_success = sum(1 for r in non_anhedonic_results.values() if r['success'])
    total_success = anhedonic_success + non_anhedonic_success
    total_processed = len(anhedonic_results) + len(non_anhedonic_results)

    print(f"Successfully processed: {total_success}/{total_processed} subjects")
    print(f"  Anhedonic: {anhedonic_success}/{len(anhedonic_results)}")
    print(f"  Non-anhedonic: {non_anhedonic_success}/{len(non_anhedonic_results)}")

    # Collect FC results for group comparison
    anhedonic_fc_results = []
    non_anhedonic_fc_results = []

    for subject_id, result in anhedonic_results.items():
        if result['success'] and result.get('static_functional_connectivity'):
            anhedonic_fc_results.append(result['static_functional_connectivity'])

    for subject_id, result in non_anhedonic_results.items():
        if result['success'] and result.get('static_functional_connectivity'):
            non_anhedonic_fc_results.append(result['static_functional_connectivity'])

    print(f"FC analysis available:")
    print(f"  Anhedonic: {len(anhedonic_fc_results)} subjects")
    print(f"  Non-anhedonic: {len(non_anhedonic_fc_results)} subjects")

    # ===== NEW: AGGREGATE INTERHEMISPHERIC NETWORK COHERENCE BY GROUP =====
    print(f"\n{'='*80}")
    print(f"AGGREGATING INTERHEMISPHERIC NETWORK COHERENCE")
    print(f"{'='*80}")

    grouped_interhemi_coherence = {
        'non-anhedonic': {},
        'low-anhedonic': {},
        'high-anhedonic': {}
    }

    # Process non-anhedonic (control) group
    for subject_id, result in non_anhedonic_results.items():
        if not result.get('success'):
            continue

        static_fc = result.get('static_functional_connectivity', {})
        network_coherence = static_fc.get('interhemispheric_network_coherence', {})

        for network_key, network_stats in network_coherence.items():
            if network_key not in grouped_interhemi_coherence['non-anhedonic']:
                grouped_interhemi_coherence['non-anhedonic'][network_key] = {
                    'subject_ids': [],
                    'mean_fisher_z_values': [],
                    'n_parcel_pairs_per_subject': []
                }

            grouped_interhemi_coherence['non-anhedonic'][network_key]['subject_ids'].append(subject_id)
            grouped_interhemi_coherence['non-anhedonic'][network_key]['mean_fisher_z_values'].append(
                network_stats['mean_fisher_z']
            )
            grouped_interhemi_coherence['non-anhedonic'][network_key]['n_parcel_pairs_per_subject'].append(
                network_stats['n_parcel_pairs']
            )

    # Process anhedonic subjects (split by low/high)
    for subject_id, result in anhedonic_results.items():
        if not result.get('success'):
            continue

        # Determine if low or high anhedonic
        if subject_id in low_anhedonic_subjects:
            group_name = 'low-anhedonic'
        elif subject_id in high_anhedonic_subjects:
            group_name = 'high-anhedonic'
        else:
            print(f"Warning: {subject_id} in anhedonic_results but not in low/high lists, skipping")
            continue

        static_fc = result.get('static_functional_connectivity', {})
        network_coherence = static_fc.get('interhemispheric_network_coherence', {})

        for network_key, network_stats in network_coherence.items():
            if network_key not in grouped_interhemi_coherence[group_name]:
                grouped_interhemi_coherence[group_name][network_key] = {
                    'subject_ids': [],
                    'mean_fisher_z_values': [],
                    'n_parcel_pairs_per_subject': []
                }

            grouped_interhemi_coherence[group_name][network_key]['subject_ids'].append(subject_id)
            grouped_interhemi_coherence[group_name][network_key]['mean_fisher_z_values'].append(
                network_stats['mean_fisher_z']
            )
            grouped_interhemi_coherence[group_name][network_key]['n_parcel_pairs_per_subject'].append(
                network_stats['n_parcel_pairs']
            )

    # Filter out NaN values and track excluded subjects
    for group_name, networks in grouped_interhemi_coherence.items():
        for network_key, network_data in networks.items():
            values = np.array(network_data['mean_fisher_z_values'])

            # Remove NaN values (subjects with no valid pairs for this network)
            valid_mask = ~np.isnan(values)

            network_data['valid_subject_ids'] = [
                sid for sid, valid in zip(network_data['subject_ids'], valid_mask) if valid
            ]
            network_data['valid_fisher_z_values'] = values[valid_mask].tolist()
            network_data['n_valid_subjects'] = int(np.sum(valid_mask))
            network_data['n_excluded_subjects'] = int(np.sum(~valid_mask))

    print(f"Network coherence aggregation complete")
    print(f"  Groups processed: {list(grouped_interhemi_coherence.keys())}")

    # ===== COMPUTE OBSERVED TEST STATISTICS =====
    print(f"\n{'='*80}")
    print(f"COMPUTING OBSERVED TEST STATISTICS")
    print(f"{'='*80}")

    from scipy import stats as scipy_stats

    observed_test_statistics = {}

    # Get all unique networks
    all_networks_for_stats = set()
    for group_data in grouped_interhemi_coherence.values():
        all_networks_for_stats.update(group_data.keys())

    for network_key in sorted(all_networks_for_stats):
        # Collect valid data for each group
        non_anhedonic_values = []
        low_anhedonic_values = []
        high_anhedonic_values = []

        if network_key in grouped_interhemi_coherence['non-anhedonic']:
            non_anhedonic_values = np.array(
                grouped_interhemi_coherence['non-anhedonic'][network_key]['valid_fisher_z_values']
            )

        if network_key in grouped_interhemi_coherence['low-anhedonic']:
            low_anhedonic_values = np.array(
                grouped_interhemi_coherence['low-anhedonic'][network_key]['valid_fisher_z_values']
            )

        if network_key in grouped_interhemi_coherence['high-anhedonic']:
            high_anhedonic_values = np.array(
                grouped_interhemi_coherence['high-anhedonic'][network_key]['valid_fisher_z_values']
            )

        # Initialize results for this network
        observed_test_statistics[network_key] = {
            'group_sizes': {
                'non-anhedonic': len(non_anhedonic_values),
                'low-anhedonic': len(low_anhedonic_values),
                'high-anhedonic': len(high_anhedonic_values)
            },
            'group_means': {
                'non-anhedonic': float(np.mean(non_anhedonic_values)) if len(non_anhedonic_values) > 0 else np.nan,
                'low-anhedonic': float(np.mean(low_anhedonic_values)) if len(low_anhedonic_values) > 0 else np.nan,
                'high-anhedonic': float(np.mean(high_anhedonic_values)) if len(high_anhedonic_values) > 0 else np.nan
            },
            'group_sds': {
                'non-anhedonic': float(np.std(non_anhedonic_values, ddof=1)) if len(non_anhedonic_values) > 1 else np.nan,
                'low-anhedonic': float(np.std(low_anhedonic_values, ddof=1)) if len(low_anhedonic_values) > 1 else np.nan,
                'high-anhedonic': float(np.std(high_anhedonic_values, ddof=1)) if len(high_anhedonic_values) > 1 else np.nan
            }
        }

        # One-way ANOVA across all three groups
        all_groups = [non_anhedonic_values, low_anhedonic_values, high_anhedonic_values]
        valid_groups = [g for g in all_groups if len(g) > 0]

        if len(valid_groups) >= 2:
            # Perform one-way ANOVA
            f_stat, p_val = scipy_stats.f_oneway(*valid_groups)
            observed_test_statistics[network_key]['anova'] = {
                'F_statistic': float(f_stat),
                'p_value': float(p_val),
                'n_groups_compared': len(valid_groups)
            }
        else:
            observed_test_statistics[network_key]['anova'] = {
                'F_statistic': np.nan,
                'p_value': np.nan,
                'n_groups_compared': len(valid_groups),
                'note': 'Insufficient groups for ANOVA'
            }

        # Pairwise comparisons (independent t-tests)
        pairwise_comparisons = {}

        # non-anhedonic vs low-anhedonic
        if len(non_anhedonic_values) > 0 and len(low_anhedonic_values) > 0:
            t_stat, p_val = scipy_stats.ttest_ind(non_anhedonic_values, low_anhedonic_values)
            pairwise_comparisons['non-anhedonic_vs_low-anhedonic'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_val),
                'mean_diff': float(np.mean(non_anhedonic_values) - np.mean(low_anhedonic_values))
            }

        # non-anhedonic vs high-anhedonic
        if len(non_anhedonic_values) > 0 and len(high_anhedonic_values) > 0:
            t_stat, p_val = scipy_stats.ttest_ind(non_anhedonic_values, high_anhedonic_values)
            pairwise_comparisons['non-anhedonic_vs_high-anhedonic'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_val),
                'mean_diff': float(np.mean(non_anhedonic_values) - np.mean(high_anhedonic_values))
            }

        # low-anhedonic vs high-anhedonic
        if len(low_anhedonic_values) > 0 and len(high_anhedonic_values) > 0:
            t_stat, p_val = scipy_stats.ttest_ind(low_anhedonic_values, high_anhedonic_values)
            pairwise_comparisons['low-anhedonic_vs_high-anhedonic'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_val),
                'mean_diff': float(np.mean(low_anhedonic_values) - np.mean(high_anhedonic_values))
            }

        observed_test_statistics[network_key]['pairwise'] = pairwise_comparisons

    print(f"Observed test statistics computed for {len(observed_test_statistics)} networks")

    # Print summary of significant results (uncorrected)
    print(f"\nNetworks with p < 0.05 (ANOVA, uncorrected):")
    significant_networks = []
    for network_key, stats in observed_test_statistics.items():
        if 'anova' in stats and stats['anova'].get('p_value', 1.0) < 0.05:
            significant_networks.append((network_key, stats['anova']['p_value']))

    if significant_networks:
        for network_key, p_val in sorted(significant_networks, key=lambda x: x[1]):
            print(f"  {network_key}: F = {observed_test_statistics[network_key]['anova']['F_statistic']:.3f}, p = {p_val:.4f}")
    else:
        print(f"  None (no networks show p < 0.05)")

    # ===== GROUP COMPARISON ANALYSIS =====
    group_comparison_results = None
    if len(anhedonic_fc_results) > 0 and len(non_anhedonic_fc_results) > 0:
        print(f"\n{'='*80}")
        print(f"GROUP COMPARISON ANALYSIS")
        print(f"{'='*80}")

        group_comparison_results = compare_fc_between_groups(
            anhedonic_fc_results,
            non_anhedonic_fc_results,
            group1_name="Anhedonic",
            group2_name="Non-anhedonic"
        )

        print(f"Statistical Comparisons (Anhedonic vs Non-anhedonic):")
        for pattern_type, comparison in group_comparison_results['comparisons'].items():
            if not comparison.get('note'):  # Skip patterns with insufficient data
                print(f"\n{pattern_type.upper()}:")
                print(f"  Anhedonic: M={comparison['group1_mean']:.3f}, SD={comparison['group1_std']:.3f}, N={comparison['group1_n']}")
                print(f"  Non-anhedonic: M={comparison['group2_mean']:.3f}, SD={comparison['group2_std']:.3f}, N={comparison['group2_n']}")
                print(f"  t({comparison['group1_n']+comparison['group2_n']-2})={comparison['ttest_statistic']:.3f}, p={comparison['ttest_pvalue']:.3f}")
                print(f"  Cohen's d={comparison['cohens_d']:.3f}, Significant={'Yes' if comparison['significant'] else 'No'}")

    else:
        print(f"\n[WARNING] Insufficient FC data for group comparison")
        print(f"  Need at least 1 subject per group with successful FC analysis")

    # ===== EXPORT FC RESULTS TO CSV =====
    if save_figures:
        print(f"\n{'='*80}")
        print(f"EXPORTING STATIC FC RESULTS TO CSV")
        print(f"{'='*80}")

        # Create output directory for FC CSV files within run folder
        fc_output_dir = run_parent_dir / 'static_fc'
        fc_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {fc_output_dir}")

        csv_export_count = 0
        all_results = {**anhedonic_results, **non_anhedonic_results}

        for subject_id, result in all_results.items():
            if not result['success']:
                continue

            static_fc_data = result.get('static_functional_connectivity')
            if static_fc_data:
                print(f"\nExporting static FC results for {subject_id}...")
                exported_files = export_static_fc_results_to_csv(static_fc_data, subject_id, fc_output_dir)
                if exported_files:
                    csv_export_count += 1

        print(f"\n✓ Exported static FC results for {csv_export_count}/{total_success} subjects")
        print(f"  Files saved to: {fc_output_dir}")

    # ===== GROUP-AVERAGED FC ANALYSIS =====
    print(f"\n{'='*80}")
    print(f"COMPUTING GROUP-AVERAGED FUNCTIONAL CONNECTIVITY")
    print(f"{'='*80}")

    # Define groups
    groups_config = [
        ('non-anhedonic', non_anhedonic_subjects),
        ('Low Anhedonic', low_anhedonic_subjects),
        ('High Anhedonic', high_anhedonic_subjects)
    ]

    # Combine all results for easier access
    all_results = {**anhedonic_results, **non_anhedonic_results}

    # Storage for group-averaged results
    group_averaged_fc = {
        'static': {},
        'slow_bands': {}
    }

    # 1. Compute group-averaged whole-signal FC
    print(f"\n--- Whole-Signal Static FC ---")
    for group_name, group_subjects in groups_config:
        avg_fc = compute_group_averaged_fc(
            all_results,
            group_subjects,
            group_name,
            fc_type='static',
            verbose=True
        )
        if avg_fc:
            group_averaged_fc['static'][group_name] = avg_fc

    # 2. Compute group-averaged slow-band FC
    print(f"\n--- Slow-Band FC ---")
    slow_bands = ['slow-5', 'slow-4', 'slow-3', 'slow-2']

    for band_key in slow_bands:
        print(f"\n  {band_key.upper()}:")
        group_averaged_fc['slow_bands'][band_key] = {}

        for group_name, group_subjects in groups_config:
            avg_fc = compute_group_averaged_fc(
                all_results,
                group_subjects,
                group_name,
                fc_type='slow_band',
                band_key=band_key,
                verbose=True
            )
            if avg_fc:
                group_averaged_fc['slow_bands'][band_key][group_name] = avg_fc

    print(f"\n✓ Group averaging complete")
    print(f"  Static FC: {len([g for g in group_averaged_fc['static'].values() if g])} groups")
    total_slow_band_groups = sum(len([g for g in band_groups.values() if g])
                                  for band_groups in group_averaged_fc['slow_bands'].values())
    print(f"  Slow-band FC: {total_slow_band_groups} group×band combinations")

    # Export group-averaged FC to CSV within run folder
    if save_figures:
        print(f"\n--- Exporting Group-Averaged FC ---")
        group_avg_output_dir = run_parent_dir / 'group_averages'
        group_avg_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {group_avg_output_dir}")

        exported_count = 0

        # Export static FC
        print(f"\n  Static FC:")
        for group_name, avg_data in group_averaged_fc['static'].items():
            if avg_data:
                export_group_averaged_fc_to_csv(avg_data, group_avg_output_dir, verbose=True)
                exported_count += 1

        # Export slow-band FC
        print(f"\n  Slow-Band FC:")
        for band_key, band_groups in group_averaged_fc['slow_bands'].items():
            if band_groups:
                print(f"    {band_key}:")
                for group_name, avg_data in band_groups.items():
                    if avg_data:
                        export_group_averaged_fc_to_csv(avg_data, group_avg_output_dir, verbose=True)
                        exported_count += 1

        print(f"\n✓ Exported {exported_count} group-averaged FC results")
        print(f"  Files saved to: {group_avg_output_dir}")

    # ===== PERMUTATION TESTING ANALYSIS =====
    print(f"\n{'='*50}")
    print(f"GENERATING {PERMUTATION_COUNT} PERMUTATIONS FOR {PERMUTATION_GROUP_COUNT} GROUPS")
    print(f"\n{'='*50}")

    all_subjects_to_process = anhedonic_subjects_to_process + non_anhedonic_subjects_to_process
    subject_ids_shuffled = all_subjects_to_process.copy()

    group_subject_counts = {
        'non-anhedonic': len(non_anhedonic_subjects_to_process),
        'low-anhedonic': len(low_anhedonic_to_process),
        'high-anhedonic': len(high_anhedonic_to_process),
    }

    print(f"Group Counts: {group_subject_counts.values()}")

    # Preallocate for the number of permutations
    permutation_groups = [None] * PERMUTATION_COUNT

    for permutation_idx in range(PERMUTATION_COUNT):
        random.shuffle(subject_ids_shuffled)
        # Create a fresh iterator for islice to consume
        it = iter(subject_ids_shuffled)
        permutation_groups[permutation_idx] = [
                list(islice(it, size)) for size in group_subject_counts.values()
            ]
    print(f"Permutations available: {len(permutation_groups)}:\n{permutation_groups}")

    # ===== INDIVIDUAL SUBJECT PLOTS =====
    individual_plots_created = 0
    figures_saved_count = 0

    # Create output directories for different analysis types
    figures_base_dir = run_parent_dir / 'figures' if save_figures and create_plots else None
    fc_subject_dir = None
    fc_group_dir = None
    mvmd_figures_dir = None
    hsa_figures_dir = None
    marginal_hsa_figures_dir = None
    roi_figures_dir = None

    if save_figures and create_plots:
        # FC directories
        fc_subject_dir = figures_base_dir / 'fc_subject'
        fc_group_dir = figures_base_dir / 'fc_group'

        # MVMD decomposition directory
        mvmd_figures_dir = figures_base_dir / 'mvmd_analysis'

        # Hilbert Spectral Analysis directories
        hsa_figures_dir = figures_base_dir / 'hilbert_spectral_analysis'
        marginal_hsa_figures_dir = figures_base_dir / 'marginal_hilbert_spectral_analysis'

        # ROI extraction directory
        roi_figures_dir = figures_base_dir / 'roi_extraction'

        # Create base directories
        fc_subject_dir.mkdir(parents=True, exist_ok=True)
        fc_group_dir.mkdir(parents=True, exist_ok=True)
        mvmd_figures_dir.mkdir(parents=True, exist_ok=True)
        hsa_figures_dir.mkdir(parents=True, exist_ok=True)
        marginal_hsa_figures_dir.mkdir(parents=True, exist_ok=True)
        roi_figures_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[INFO] Figures will be saved to:")
        print(f"  FC (subject-level): {fc_subject_dir}")
        print(f"  FC (group-level): {fc_group_dir}")
        print(f"  MVMD decomposition: {mvmd_figures_dir}")
        print(f"  Hilbert Spectral Analysis: {hsa_figures_dir}")
        print(f"  Marginal Hilbert Spectral Analysis: {marginal_hsa_figures_dir}")
        print(f"  ROI extraction: {roi_figures_dir}")

    if create_plots and total_success > 0:
        print(f"\n{'='*80}")
        print(f"CREATING INDIVIDUAL SUBJECT PLOTS")
        print(f"{'='*80}")

        if show_plots:
            print(f"[INFO] Figures will be displayed in batches by analysis type")
            print(f"       Close all figures in each batch to proceed to the next batch")

        all_results = {**anhedonic_results, **non_anhedonic_results}

        # Organize plotting by type to display in batches
        # This prevents overwhelming the system with 177+ figures at once
        plot_batches = {
            'roi_cortical': [],
            'roi_subcortical': [],
            'fc_static': [],
            'fc_slow_bands': [],
            'fc_group_avg': [],
            'mvmd_modes': [],
            'hsa_multivariate': [],
            'hsa_marginal': [],
        }
        plot_batch_count = len(plot_batches.keys())

        for subject_id, result in all_results.items():
            if not result['success']:
                continue

            # Determine which group this subject belongs to
            if subject_id in low_anhedonic_subjects:
                subject_group = "Low Anhedonic"
            elif subject_id in high_anhedonic_subjects:
                subject_group = "High Anhedonic"
            elif subject_id in non_anhedonic_subjects:
                subject_group = "Non-anhedonic"
            else:
                subject_group = None

            plots_for_subject = 0

            # 1. Prepare cortical ROI timeseries plot
            roi_results = result.get('roi_extraction_results', {})
            if roi_results.get('cortical'):
                cortical_data = roi_results['cortical']
                if cortical_data.get('extraction_successful'):
                    plot_batches['roi_cortical'].append({
                        'subject_id': subject_id,
                        'data': cortical_data,
                        'save_dir': roi_figures_dir if save_figures and roi_figures_dir else None
                    })
                    plots_for_subject += 1

            # 2. Prepare subcortical ROI timeseries plot
            if roi_results.get('subcortical'):
                subcortical_data = roi_results['subcortical']
                if subcortical_data.get('extraction_successful'):
                    plot_batches['roi_subcortical'].append({
                        'subject_id': subject_id,
                        'data': subcortical_data,
                        'save_dir': roi_figures_dir if save_figures and roi_figures_dir else None
                    })
                    plots_for_subject += 1


            # 3. Prepare static functional connectivity analysis plot
            if result.get('static_functional_connectivity'):
                static_fc_data = result['static_functional_connectivity']
                if static_fc_data.get('static_fc_matrix') is not None:
                    plot_batches['fc_static'].append({
                        'subject_id': subject_id,
                        'subject_group': subject_group,
                        'data': static_fc_data,
                        'mask_diagonal': mask_diagonal,
                        'mask_nonsignificant': mask_nonsignificant,
                        'save_dir': fc_subject_dir if save_figures and fc_subject_dir else None
                    })
                    plots_for_subject += 1

            # 3b. Prepare slow-band FC plots
            if result.get('slow_band_fc'):
                slow_band_fc_data = result['slow_band_fc']
                for band_key, band_fc_data in slow_band_fc_data.items():
                    if band_fc_data.get('fc_matrix') is not None:
                        plot_batches['fc_slow_bands'].append({
                            'subject_id': subject_id,
                            'subject_group': subject_group,
                            'band_key': band_key,
                            'data': band_fc_data,
                            'mask_diagonal': mask_diagonal,
                            'mask_nonsignificant': mask_nonsignificant,
                            'save_dir': fc_subject_dir if save_figures and fc_subject_dir else None
                        })
                        plots_for_subject += 1

            # 4. Prepare MVMD decomposition plots
            if result.get('mvmd'):
                mvmd_data = result['mvmd']
                if mvmd_data.get('time_modes') is not None:
                    center_freqs = mvmd_data['center_freqs'][-1, :] if mvmd_data.get('center_freqs') is not None else None
                    channel_label_map = mvmd_data.get('channel_label_map')

                    plot_batches['mvmd_modes'].append({
                        'subject_id': subject_id,
                        'mvmd_data': mvmd_data,
                        'center_freqs': center_freqs,
                        'channel_label_map': channel_label_map,
                        'save_dir': mvmd_figures_dir if save_figures and mvmd_figures_dir else None
                    })

                    # Estimate channel count for progress tracking
                    channel_count = mvmd_data['original'].shape[0] if 'original' in mvmd_data else 0
                    plots_for_subject += channel_count

            # 5. MVMD slow-band signal plots: REMOVED
            # These plots were based on time-domain summing of modes, which is statistically invalid.
            # We now only work at the mode level and aggregate FC matrices (not time-domain signals).

            # 6. Prepare Multivariate Hilbert Spectrum plots
            if result.get('mvmd'):
                mvmd_data = result['mvmd']
                if mvmd_data.get('hilbert_spectral_analysis') is not None:
                    hsa_data = mvmd_data['hilbert_spectral_analysis']
                    center_freqs = mvmd_data['center_freqs'][-1, :] if mvmd_data.get('center_freqs') is not None else None
                    channel_label_map = mvmd_data.get('channel_label_map', {})
                    # Convert channel_label_map to list of labels
                    channel_labels = [channel_label_map.get(i, f'Ch{i+1}') for i in range(len(channel_label_map))]

                    plot_batches['hsa_multivariate'].append({
                        'subject_id': subject_id,
                        'hsa_data': hsa_data,
                        'center_freqs': center_freqs,
                        'channel_labels': channel_labels,
                        'save_dir': hsa_figures_dir if save_figures and hsa_figures_dir else None
                    })
                    plots_for_subject += 1

            # 7. Prepare Marginal Spectrum per Mode plots
            if result.get('mvmd'):
                mvmd_data = result['mvmd']
                if mvmd_data.get('hilbert_spectral_analysis') is not None:
                    hsa_data = mvmd_data['hilbert_spectral_analysis']
                    center_freqs = mvmd_data['center_freqs'][-1, :] if mvmd_data.get('center_freqs') is not None else None
                    channel_label_map = mvmd_data.get('channel_label_map', {})
                    # Convert channel_label_map to list of labels
                    channel_labels = [channel_label_map.get(i, f'Ch{i+1}') for i in range(len(channel_label_map))]

                    plot_batches['hsa_marginal'].append({
                        'subject_id': subject_id,
                        'hsa_data': hsa_data,
                        'center_freqs': center_freqs,
                        'channel_labels': channel_labels,
                        'save_dir': marginal_hsa_figures_dir if save_figures and marginal_hsa_figures_dir else None
                    })
                    plots_for_subject += 1

            if plots_for_subject > 0:
                individual_plots_created += 1
                print(f"  ✓ Prepared {plots_for_subject} plots for {subject_id}")

        print(f"\nPrepared plots for {individual_plots_created} subjects")

        # Prepare group-averaged FC plots
        print(f"\nPreparing group-averaged FC plots...")

        # Add static FC group averages
        for group_name, avg_data in group_averaged_fc['static'].items():
            if avg_data:
                group_name_clean = group_name.replace(' ', '_').replace('-', '_').lower()
                plot_batches['fc_group_avg'].append({
                    'group_name': group_name,
                    'data': avg_data,
                    'mask_diagonal': mask_diagonal,
                    'mask_nonsignificant': False,  # No p-values for group averages
                    'save_path': fc_group_dir / f'group_avg_{group_name_clean}_static_fc.svg' if save_figures and fc_group_dir else None,
                    'is_slow_band': False
                })

        # Add slow-band FC group averages
        for band_key, band_groups in group_averaged_fc['slow_bands'].items():
            for group_name, avg_data in band_groups.items():
                if avg_data:
                    group_name_clean = group_name.replace(' ', '_').replace('-', '_').lower()
                    plot_batches['fc_group_avg'].append({
                        'group_name': group_name,
                        'data': avg_data,
                        'mask_diagonal': mask_diagonal,
                        'mask_nonsignificant': False,
                        'save_path': fc_group_dir / f'group_avg_{group_name_clean}_{band_key}_fc.svg' if save_figures and fc_group_dir else None,
                        'is_slow_band': True,
                        'band_key': band_key
                    })

        print(f"  ✓ Prepared {len(plot_batches['fc_group_avg'])} group-averaged FC plots")

        # Now create and display plots in batches by type
        print(f"\n{'='*80}")
        print(f"CREATING AND DISPLAYING PLOTS BY TYPE")
        print(f"{'='*80}")

        current_plot_batch = 0

        # Batch 1: ROI Cortical Timeseries
        if plot_batches['roi_cortical']:
            current_plot_batch += 1
            print(f"\n[Batch {current_plot_batch}/{plot_batch_count}] Creating {len(plot_batches['roi_cortical'])} cortical ROI timeseries plots...")
            for plot_info in plot_batches['roi_cortical']:
                figures = plot_roi_timeseries_result(plot_info['data'], subject_id=plot_info['subject_id'], atlas_type='Cortical')
                # plot_roi_timeseries_result now returns a list of figures (one per ROI)
                for fig in figures:
                    if plot_info['save_dir']:
                        # Create subject directory
                        subject_roi_dir = plot_info['save_dir'] / plot_info['subject_id']
                        subject_roi_dir.mkdir(parents=True, exist_ok=True)

                        # Extract ROI name from figure title
                        fig_title = fig._suptitle.get_text() if fig._suptitle else ''
                        roi_name = 'unknown'
                        if 'PFCm' in fig_title:
                            roi_name = 'PFCm'
                        elif 'PFCv' in fig_title:
                            roi_name = 'PFCv'

                        # Create ROI-specific filename in subject directory
                        fig_path = subject_roi_dir / f'{roi_name}_roi_timeseries_cortical.svg'
                        fig.savefig(fig_path, format='svg', bbox_inches='tight', dpi=300)
                        figures_saved_count += 1
                    if not show_plots:
                        plt.close(fig)
            if show_plots:
                print(f"  Displaying {len(plot_batches['roi_cortical'])} cortical ROI plots. Close all figures to continue...")
                plt.show()

        # Batch 2: ROI Subcortical Timeseries
        if plot_batches['roi_subcortical']:
            current_plot_batch += 1
            print(f"\n[Batch {current_plot_batch}/{plot_batch_count}] Creating {len(plot_batches['roi_subcortical'])} subcortical ROI timeseries plots...")
            for plot_info in plot_batches['roi_subcortical']:
                figures = plot_roi_timeseries_result(plot_info['data'], subject_id=plot_info['subject_id'], atlas_type='Subcortical')
                # plot_roi_timeseries_result now returns a list of figures (one per ROI)
                for fig in figures:
                    if plot_info['save_dir']:
                        # Create subject directory
                        subject_roi_dir = plot_info['save_dir'] / plot_info['subject_id']
                        subject_roi_dir.mkdir(parents=True, exist_ok=True)

                        # Extract ROI name from figure title
                        fig_title = fig._suptitle.get_text() if fig._suptitle else ''
                        roi_name = 'unknown'
                        if 'AMY' in fig_title:
                            roi_name = 'AMY'

                        # Create ROI-specific filename in subject directory
                        fig_path = subject_roi_dir / f'{roi_name}_roi_timeseries_subcortical.svg'
                        fig.savefig(fig_path, format='svg', bbox_inches='tight', dpi=300)
                        figures_saved_count += 1
                    if not show_plots:
                        plt.close(fig)
            if show_plots:
                print(f"  Displaying {len(plot_batches['roi_subcortical'])} subcortical ROI plots. Close all figures to continue...")
                plt.show()

        # Batch 3: Multivariate Hilbert Spectrum
        if plot_batches['hsa_multivariate']:
            current_plot_batch += 1
            print(f"\n[Batch {current_plot_batch}/{plot_batch_count}] Creating {len(plot_batches['hsa_multivariate'])} Multivariate Hilbert Spectrum plots...")

            for plot_info in plot_batches['hsa_multivariate']:
                # Create multivariate HS plot (returns list of tuples: (figure, region_network_key))
                hs_results = plot_multivariate_hilbert_spectrum(
                    hsa_data=plot_info['hsa_data'],
                    subject_id=plot_info['subject_id'],
                    center_freqs=plot_info['center_freqs'],
                    channel_labels=plot_info['channel_labels']
                )

                # Save figures if enabled
                if plot_info['save_dir']:
                    # Create subject/composite/ directory structure
                    subject_composite_dir = plot_info['save_dir'] / plot_info['subject_id'] / 'composite'
                    subject_composite_dir.mkdir(parents=True, exist_ok=True)

                    for hs_fig, region_network_key in hs_results:
                        # Use region+network+hemisphere info in filename
                        fig_path = subject_composite_dir / f'{region_network_key}_multivariate_hs.svg'
                        hs_fig.savefig(fig_path, format='svg', bbox_inches='tight', dpi=300)
                        figures_saved_count += 1

                if not show_plots:
                    for hs_fig, _ in hs_results:
                        plt.close(hs_fig)

            if show_plots:
                print(f"  Displaying {len(plot_batches['hsa_multivariate'])} Multivariate Hilbert Spectrum plots. Close all figures to continue...")
                plt.show()

        # Batch 4: Marginal Spectrum per Mode
        if plot_batches['hsa_marginal']:
            current_plot_batch += 1
            print(f"\n[Batch {current_plot_batch}/{plot_batch_count}] Creating {len(plot_batches['hsa_marginal'])} Marginal Spectrum per Mode plots...")

            for plot_info in plot_batches['hsa_marginal']:
                # Create marginal spectrum plot (returns list of tuples: (figure, region_network_key))
                mhs_results = plot_marginal_spectrum_per_mode(
                    hsa_data=plot_info['hsa_data'],
                    subject_id=plot_info['subject_id'],
                    center_freqs=plot_info['center_freqs'],
                    channel_labels=plot_info['channel_labels']
                )

                # Save figures if enabled
                if plot_info['save_dir']:
                    # Create subject/composite/ directory structure
                    subject_composite_dir = plot_info['save_dir'] / plot_info['subject_id'] / 'composite'
                    subject_composite_dir.mkdir(parents=True, exist_ok=True)

                    for mhs_fig, region_network_key in mhs_results:
                        # Use region+network+hemisphere info in filename
                        fig_path = subject_composite_dir / f'{region_network_key}_marginal_spectrum.svg'
                        mhs_fig.savefig(fig_path, format='svg', bbox_inches='tight', dpi=300)
                        figures_saved_count += 1

                if not show_plots:
                    for mhs_fig, _ in mhs_results:
                        plt.close(mhs_fig)

            if show_plots:
                print(f"  Displaying {len(plot_batches['hsa_marginal'])} Marginal Spectrum plots. Close all figures to continue...")
                plt.show()

        # Batch 5: Static FC Analysis
        if plot_batches['fc_static']:
            current_plot_batch += 1
            print(f"\n[Batch {current_plot_batch}/{plot_batch_count}] Creating {len(plot_batches['fc_static'])} static FC plots...")
            for plot_info in plot_batches['fc_static']:
                # Use fc_output_dir for CSV exports (fc_analysis/static_fc/), not figures directory
                csv_output_dir = fc_output_dir if save_figures else None

                fc_fig_inter, fc_fig_ipsi = plot_fc_results(
                    plot_info['data']['static_fc_matrix'],
                    plot_info['data']['static_fc_labels'],
                    plot_info['data']['static_fc_pvalues'],
                    plot_info['data']['static_connectivity_patterns'],
                    plot_info['data'].get('channel_label_map'),
                    mask_diagonal=plot_info['mask_diagonal'],
                    mask_nonsignificant=plot_info['mask_nonsignificant'],
                    subject_group=plot_info['subject_group'],
                    subject_id=plot_info['subject_id'],
                    output_dir=csv_output_dir,
                    verbose=verbose
                )
                fc_fig_inter.suptitle(f'FC Analysis (Interhemispheric) - {plot_info["subject_id"]}', fontsize=16, fontweight='bold')
                fc_fig_ipsi.suptitle(f'FC Analysis (Ipsilateral) - {plot_info["subject_id"]}', fontsize=16, fontweight='bold')

                if plot_info['save_dir']:
                    # Create subject directory
                    subject_fc_dir = plot_info['save_dir'] / plot_info['subject_id']
                    subject_fc_dir.mkdir(parents=True, exist_ok=True)

                    # Save interhemispheric figure
                    save_path_inter = subject_fc_dir / 'static_fc_interhemispheric.svg'
                    fc_fig_inter.savefig(save_path_inter, format='svg', bbox_inches='tight', dpi=300)
                    figures_saved_count += 1

                    # Save ipsilateral figure
                    save_path_ipsi = subject_fc_dir / 'static_fc_ipsilateral.svg'
                    fc_fig_ipsi.savefig(save_path_ipsi, format='svg', bbox_inches='tight', dpi=300)
                    figures_saved_count += 1

                if not show_plots:
                    plt.close(fc_fig_inter)
                    plt.close(fc_fig_ipsi)
            if show_plots:
                print(f"  Displaying {len(plot_batches['fc_static'])} FC plots. Close all figures to continue...")
                plt.show()

        # Batch 6: Slow-Band FC Analysis
        if plot_batches['fc_slow_bands']:
            current_plot_batch += 1
            print(f"\n[Batch {current_plot_batch}/{plot_batch_count}] Creating {len(plot_batches['fc_slow_bands'])} slow-band FC plots...")
            for plot_info in plot_batches['fc_slow_bands']:
                # Use fc_output_dir for CSV exports (fc_analysis/static_fc/), not figures directory
                csv_output_dir = fc_output_dir if save_figures else None

                # Extract band number from band_key (e.g., "slow-5" -> "5")
                band_key = plot_info['band_key']
                band_number = band_key.split('-')[1] if '-' in band_key else band_key

                fc_fig_inter, fc_fig_ipsi = plot_fc_results(
                    plot_info['data']['fc_matrix'],
                    plot_info['data']['fc_labels'],
                    plot_info['data']['fc_pvalues'],
                    plot_info['data']['connectivity_patterns'],
                    plot_info['data'].get('channel_label_map'),
                    mask_diagonal=plot_info['mask_diagonal'],
                    mask_nonsignificant=plot_info['mask_nonsignificant'],
                    subject_group=plot_info['subject_group'],
                    subject_id=plot_info['subject_id'],
                    output_dir=csv_output_dir,
                    verbose=verbose,
                    band_name=f'Slow-{band_number}',
                    frequency_range=plot_info['data'].get('frequency_range'),
                    n_available_channels=plot_info['data'].get('n_available_channels')
                )
                fc_fig_inter.suptitle(f'Slow-{band_number} FC (Interhemispheric) - {plot_info["subject_id"]}', fontsize=16, fontweight='bold')
                fc_fig_ipsi.suptitle(f'Slow-{band_number} FC (Ipsilateral) - {plot_info["subject_id"]}', fontsize=16, fontweight='bold')

                if plot_info['save_dir']:
                    # Create subject directory
                    subject_fc_dir = plot_info['save_dir'] / plot_info['subject_id']
                    subject_fc_dir.mkdir(parents=True, exist_ok=True)

                    # Save interhemispheric figure
                    save_path_inter = subject_fc_dir / f'slow_{band_number}_fc_interhemispheric.svg'
                    fc_fig_inter.savefig(save_path_inter, format='svg', bbox_inches='tight', dpi=300)
                    figures_saved_count += 1

                    # Save ipsilateral figure
                    save_path_ipsi = subject_fc_dir / f'slow_{band_number}_fc_ipsilateral.svg'
                    fc_fig_ipsi.savefig(save_path_ipsi, format='svg', bbox_inches='tight', dpi=300)
                    figures_saved_count += 1

                if not show_plots:
                    plt.close(fc_fig_inter)
                    plt.close(fc_fig_ipsi)
            if show_plots:
                print(f"  Displaying {len(plot_batches['fc_slow_bands'])} slow-band FC plots. Close all figures to continue...")
                plt.show()

        # Batch 7: Group-Averaged FC Analysis
        if plot_batches['fc_group_avg']:
            current_plot_batch += 1
            print(f"\n[Batch {current_plot_batch}/{plot_batch_count}] Creating {len(plot_batches['fc_group_avg'])} group-averaged FC plots...")
            for plot_info in plot_batches['fc_group_avg']:
                # Compute connectivity patterns for group-averaged matrix
                # Use computed p-values from statistical testing
                connectivity_patterns = analyze_connectivity_patterns(
                    plot_info['data']['avg_fc_matrix'],
                    plot_info['data']['avg_fc_labels'],
                    p_values=plot_info['data'].get('avg_fc_pvalues'),
                    alpha=0.05
                )

                if plot_info['is_slow_band']:
                    # Slow-band group average
                    band_key = plot_info['band_key']
                    band_number = band_key.split('-')[1] if '-' in band_key else band_key
                    subject_label = f"Group_{plot_info['group_name'].replace(' ', '_')}_{band_key}"

                    fc_fig_inter, fc_fig_ipsi = plot_fc_results(
                        plot_info['data']['avg_fc_matrix'],
                        plot_info['data']['avg_fc_labels'],
                        p_values=plot_info['data'].get('avg_fc_pvalues'),
                        connectivity_patterns=connectivity_patterns,
                        channel_label_map=None,
                        mask_diagonal=plot_info['mask_diagonal'],
                        mask_nonsignificant=plot_info['mask_nonsignificant'],
                        subject_group=None,
                        subject_id=subject_label,
                        output_dir=None,
                        verbose=verbose,
                        band_name=f'Slow-{band_number}',
                        frequency_range=get_frequency_range(band_number),
                        n_available_channels=None
                    )
                    title_inter = f"Group Average: {plot_info['group_name']} - Slow-{band_number} FC (Interhemispheric, n={plot_info['data']['n_subjects']})"
                    title_ipsi = f"Group Average: {plot_info['group_name']} - Slow-{band_number} FC (Ipsilateral, n={plot_info['data']['n_subjects']})"
                else:
                    # Static FC group average
                    subject_label = f"Group_{plot_info['group_name'].replace(' ', '_')}_static"

                    fc_fig_inter, fc_fig_ipsi = plot_fc_results(
                        plot_info['data']['avg_fc_matrix'],
                        plot_info['data']['avg_fc_labels'],
                        p_values=plot_info['data'].get('avg_fc_pvalues'),
                        connectivity_patterns=connectivity_patterns,
                        channel_label_map=None,
                        mask_diagonal=plot_info['mask_diagonal'],
                        mask_nonsignificant=plot_info['mask_nonsignificant'],
                        subject_group=None,
                        subject_id=subject_label,
                        output_dir=None,
                        verbose=verbose
                    )
                    title_inter = f"Group Average: {plot_info['group_name']} - Static FC (Interhemispheric, n={plot_info['data']['n_subjects']})"
                    title_ipsi = f"Group Average: {plot_info['group_name']} - Static FC (Ipsilateral, n={plot_info['data']['n_subjects']})"

                fc_fig_inter.suptitle(title_inter, fontsize=16, fontweight='bold')
                fc_fig_ipsi.suptitle(title_ipsi, fontsize=16, fontweight='bold')

                if plot_info['save_path']:
                    # Save interhemispheric figure
                    save_path_inter = plot_info['save_path'].parent / f"{plot_info['save_path'].stem}_interhemispheric{plot_info['save_path'].suffix}"
                    fc_fig_inter.savefig(save_path_inter, format='svg', bbox_inches='tight', dpi=300)
                    figures_saved_count += 1

                    # Save ipsilateral figure
                    save_path_ipsi = plot_info['save_path'].parent / f"{plot_info['save_path'].stem}_ipsilateral{plot_info['save_path'].suffix}"
                    fc_fig_ipsi.savefig(save_path_ipsi, format='svg', bbox_inches='tight', dpi=300)
                    figures_saved_count += 1

                if not show_plots:
                    plt.close(fc_fig_inter)
                    plt.close(fc_fig_ipsi)
            if show_plots:
                print(f"  Displaying {len(plot_batches['fc_group_avg'])} group-averaged FC plots. Close all figures to continue...")
                plt.show()

        # Batch 8: MVMD Mode Decomposition
        if plot_batches['mvmd_modes']:
            total_mode_figs = sum(p['mvmd_data']['original'].shape[0] for p in plot_batches['mvmd_modes'])
            current_plot_batch += 1
            print(f"\n[Batch {current_plot_batch}/{plot_batch_count}] Creating {total_mode_figs} MVMD mode decomposition plots ({len(plot_batches['mvmd_modes'])} subjects)...")

            # Process in sub-batches of 28 figures to avoid memory issues
            MAX_FIGS_PER_BATCH = 28
            current_batch_figs = []

            for plot_info in plot_batches['mvmd_modes']:
                mvmd_figure_generator = plot_signal_decomposition(
                    plot_info['mvmd_data']['original'],
                    plot_info['mvmd_data']['time_modes'],
                    subject_id=plot_info['subject_id'],
                    channel_label_map=plot_info['channel_label_map'],
                    center_freqs=plot_info['center_freqs'],
                    max_figures_per_batch=MAX_FIGS_PER_BATCH
                )

                # Process each batch of figures from the generator
                channel_idx_base = 0
                for mvmd_figures in mvmd_figure_generator:
                    if plot_info['save_dir']:
                        subject_mvmd_dir = plot_info['save_dir'] / plot_info['subject_id']
                        subject_mvmd_dir.mkdir(parents=True, exist_ok=True)

                        for fig_idx, fig in enumerate(mvmd_figures):
                            channel_idx = channel_idx_base + fig_idx
                            if plot_info['channel_label_map'] is not None:
                                channel_label = plot_info['channel_label_map'].get(channel_idx, f'ch{channel_idx}')
                                channel_label_clean = channel_label.replace('/', '_').replace(' ', '_')
                            else:
                                channel_label_clean = f'ch{channel_idx}'

                            fig_path = subject_mvmd_dir / f'mvmd_modes_decomposition_{channel_label_clean}.svg'
                            fig.savefig(fig_path, format='svg', bbox_inches='tight', dpi=300)
                            figures_saved_count += 1

                    if show_plots:
                        current_batch_figs.extend(mvmd_figures)

                        # Display and clear batch when reaching limit
                        if len(current_batch_figs) >= MAX_FIGS_PER_BATCH:
                            print(f"  Displaying {len(current_batch_figs)} MVMD mode plots. Close all figures to continue...")
                            plt.show()
                            current_batch_figs = []
                    else:
                        for fig in mvmd_figures:
                            plt.close(fig)

                    channel_idx_base += len(mvmd_figures)

            # Display remaining figures if any
            if show_plots and current_batch_figs:
                print(f"  Displaying {len(current_batch_figs)} MVMD mode plots. Close all figures to continue...")
                plt.show()


        print(f"\n{'='*80}")
        print(f"PLOTTING COMPLETE")
        print(f"{'='*80}")

        # Summary of saved figures
        if save_figures and figures_saved_count > 0:
            print(f"✓ Saved {figures_saved_count} figures across multiple directories:")
            print(f"  FC (subject-level): {fc_subject_dir}")
            print(f"  FC (group-level): {fc_group_dir}")
            print(f"  MVMD analysis: {mvmd_figures_dir}")
            print(f"  Hilbert Spectral Analysis: {hsa_figures_dir}")
            print(f"  Marginal Hilbert Spectral Analysis: {marginal_hsa_figures_dir}")
            print(f"  ROI extraction: {roi_figures_dir}")
    elif not create_plots:
        print(f"\n[INFO] Plot creation disabled (CREATE_PLOTS=False)")
    else:
        print(f"\n[INFO] No plots created (no successful subjects)")

    # ===== INTERHEMISPHERIC NETWORK COHERENCE VALIDATION SUMMARY =====
    print(f"\n{'='*80}")
    print(f"INTERHEMISPHERIC NETWORK COHERENCE EXTRACTION SUMMARY")
    print(f"{'='*80}")

    # Get all unique networks across all groups
    all_networks = set()
    for group_data in grouped_interhemi_coherence.values():
        all_networks.update(group_data.keys())
    all_networks = sorted(all_networks)

    print(f"\nTotal Networks Detected: {len(all_networks)}")
    if all_networks:
        print(f"Networks: {', '.join(all_networks)}")

    # Show detailed stats for each network
    for network_key in all_networks:
        print(f"\n{'─'*80}")
        print(f"Network: {network_key}")
        print(f"{'─'*80}")

        for group_name in ['non-anhedonic', 'low-anhedonic', 'high-anhedonic']:
            if network_key in grouped_interhemi_coherence[group_name]:
                network_data = grouped_interhemi_coherence[group_name][network_key]
                values = np.array(network_data['valid_fisher_z_values'])

                if len(values) > 0:
                    mean_z = np.mean(values)
                    std_z = np.std(values, ddof=1) if len(values) > 1 else 0.0
                    min_z = np.min(values)
                    max_z = np.max(values)
                    n_valid = network_data['n_valid_subjects']
                    n_excluded = network_data['n_excluded_subjects']

                    print(f"  {group_name:20s}: Mean = {mean_z:7.3f}, SD = {std_z:6.3f}, "
                          f"Range = [{min_z:7.3f}, {max_z:7.3f}], "
                          f"N = {n_valid:3d} (excluded: {n_excluded})")
                else:
                    print(f"  {group_name:20s}: NO VALID SUBJECTS")
            else:
                print(f"  {group_name:20s}: NETWORK NOT PRESENT")

    # Test statistics detailed summary
    print(f"\n{'='*80}")
    print(f"OBSERVED TEST STATISTICS SUMMARY")
    print(f"{'='*80}")

    for network_key in all_networks:
        if network_key in observed_test_statistics:
            stats = observed_test_statistics[network_key]

            print(f"\n{network_key}:")
            print(f"  Group Means (Fisher-Z):")
            for group_name in ['non-anhedonic', 'low-anhedonic', 'high-anhedonic']:
                mean_val = stats['group_means'].get(group_name, np.nan)
                sd_val = stats['group_sds'].get(group_name, np.nan)
                n_val = stats['group_sizes'].get(group_name, 0)
                if not np.isnan(mean_val):
                    print(f"    {group_name:20s}: M = {mean_val:7.3f}, SD = {sd_val:6.3f}, N = {n_val:3d}")
                else:
                    print(f"    {group_name:20s}: No data")

            # ANOVA results
            if 'anova' in stats:
                anova = stats['anova']
                if not np.isnan(anova.get('F_statistic', np.nan)):
                    print(f"  ANOVA: F({anova['n_groups_compared']-1},{sum(stats['group_sizes'].values())-anova['n_groups_compared']}) = "
                          f"{anova['F_statistic']:.3f}, p = {anova['p_value']:.4f} "
                          f"{'***' if anova['p_value'] < 0.001 else '**' if anova['p_value'] < 0.01 else '*' if anova['p_value'] < 0.05 else 'ns'}")
                else:
                    print(f"  ANOVA: {anova.get('note', 'Not computed')}")

            # Pairwise comparisons
            if 'pairwise' in stats and len(stats['pairwise']) > 0:
                print(f"  Pairwise comparisons:")
                for comparison_name, comparison_data in stats['pairwise'].items():
                    t_stat = comparison_data['t_statistic']
                    p_val = comparison_data['p_value']
                    mean_diff = comparison_data['mean_diff']
                    sig_marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                    print(f"    {comparison_name:45s}: t = {t_stat:7.3f}, p = {p_val:.4f} {sig_marker:3s} (Δ = {mean_diff:7.3f})")

    # Validation warnings
    print(f"\n{'='*80}")
    print(f"VALIDATION CHECKS")
    print(f"{'='*80}")

    warnings_found = False
    for group_name, networks in grouped_interhemi_coherence.items():
        for network_key, network_data in networks.items():
            values = np.array(network_data['valid_fisher_z_values'])

            # Check for extreme values (|z| > 2.0 unusual but valid)
            if len(values) > 0:
                extreme_mask = np.abs(values) > 2.0
                if np.any(extreme_mask):
                    warnings_found = True
                    n_extreme = np.sum(extreme_mask)
                    extreme_vals = values[extreme_mask]
                    print(f"⚠ {group_name}/{network_key}: {n_extreme} subjects with |Fisher-Z| > 2.0")
                    print(f"  Extreme values: {extreme_vals}")

            # Check for networks with few subjects
            if len(values) < 5 and len(values) > 0:
                warnings_found = True
                print(f"⚠ {group_name}/{network_key}: Only {len(values)} subjects (statistical power may be low)")

            # Check for excluded subjects
            if network_data['n_excluded_subjects'] > 0:
                warnings_found = True
                print(f"ℹ {group_name}/{network_key}: {network_data['n_excluded_subjects']} subjects excluded (no valid parcel pairs)")

    if not warnings_found:
        print("✓ No validation warnings - all values within expected ranges")

    # Save detailed results to JSON
    import json
    from datetime import datetime

    if save_figures:
        # Combine coherence data and test statistics
        results_to_save = {
            'grouped_interhemi_coherence': grouped_interhemi_coherence,
            'observed_test_statistics': observed_test_statistics,
            'metadata': {
                'analysis_date': run_timestamp,
                'n_subjects': {
                    'non-anhedonic': len(non_anhedonic_results),
                    'low-anhedonic': len([sid for sid in anhedonic_results.keys() if sid in low_anhedonic_subjects]),
                    'high-anhedonic': len([sid for sid in anhedonic_results.keys() if sid in high_anhedonic_subjects])
                },
                'n_networks': len(all_networks)
            }
        }

        output_file = run_parent_dir / f"interhemispheric_network_coherence_{run_timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)

        print(f"\n✓ Detailed results saved to: {output_file}")
        print(f"  Includes: coherence values, test statistics, and metadata")

    print(f"\n{'='*80}")
    print(f"ANALYSIS HALTED - AWAITING USER CONFIRMATION")
    print(f"{'='*80}")
    print(f"\nNext Steps (NOT IMPLEMENTED YET):")
    print(f"  1. Permutation-based ANOVA across groups")
    print(f"  2. Post-hoc pairwise comparisons")
    print(f"  3. Multiple comparison correction (FDR/Bonferroni)")
    print(f"\n{'='*80}\n")

    # ===== WRITE ANALYSIS LOG =====
    print(f"\n{'='*80}")
    print(f"WRITING ANALYSIS LOG")
    print(f"{'='*80}")

    if save_figures:
        log_file = write_analysis_log(
            output_dir=run_parent_dir,
            groups_config=groups_config,
            all_results=all_results,
            low_anhedonic_subjects=low_anhedonic_subjects,
            high_anhedonic_subjects=high_anhedonic_subjects,
            timestamp=run_timestamp
        )

        print(f"Analysis log saved to: {log_file}")

    # ===== FINAL SUMMARY =====
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}")
    if save_figures:
        print(f"All outputs saved to: {run_parent_dir}")
        print(f"  - Static FC CSVs: {fc_output_dir}")
        print(f"  - Group averages: {group_avg_output_dir}")
        if figures_saved_count > 0:
            print(f"  - Figures ({figures_saved_count} total):")
            print(f"    - FC (subject-level): {fc_subject_dir}")
            print(f"    - FC (group-level): {fc_group_dir}")
            print(f"    - MVMD analysis: {mvmd_figures_dir}")
            print(f"    - Hilbert Spectral Analysis: {hsa_figures_dir}")
            print(f"    - Marginal Hilbert Spectral Analysis: {marginal_hsa_figures_dir}")
            print(f"    - ROI extraction: {roi_figures_dir}")
        print(f"  - Analysis log: {log_file}")
        print(f"\nRun ID: {run_timestamp}")
    else:
        print(f"No files saved (--no-save mode)")
        print(f"Analysis completed successfully")

    # ===== RETURN MULTI-SUBJECT RESULTS =====
    return {
        'anhedonic_subjects': anhedonic_subjects_to_process,
        'non_anhedonic_subjects': non_anhedonic_subjects_to_process,
        'processing_mode': 'downloaded_only' if use_downloaded_only else 'all_available',
        'configuration': {
            'limit_subjects': LIMIT_SUBJECTS,
            'max_subjects_per_group': MAX_SUBJECTS_PER_GROUP if LIMIT_SUBJECTS else None,
            'show_plots': show_plots,
            'verbose_subject_output': verbose
        },
        'summary': {
            'total_processed': total_processed,
            'total_successful': total_success,
            'anhedonic_processed': len(anhedonic_results),
            'anhedonic_successful': anhedonic_success,
            'non_anhedonic_processed': len(non_anhedonic_results),
            'non_anhedonic_successful': non_anhedonic_success,
            'individual_plots_created': individual_plots_created
        },
        'subject_results': {
            'anhedonic': anhedonic_results,
            'non_anhedonic': non_anhedonic_results
        },
        'group_analysis': {
            'anhedonic_fc_count': len(anhedonic_fc_results),
            'non_anhedonic_fc_count': len(non_anhedonic_fc_results),
            'group_comparison': group_comparison_results
        }
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Functional Connectivity MVP Analysis - Processes fMRI data to compute '
                   'static and dynamic functional connectivity matrices with MVMD decomposition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python fc_mvp.py                           # Process all subjects with default settings
  python fc_mvp.py --subjects-per-group 5   # Limit to 5 subjects per group
  python fc_mvp.py --verbose --show-plots   # Enable verbose output and show plots
  python fc_mvp.py --subjects-per-group 3 --no-save  # Limit subjects and don't save figures

Output:
  - Static FC matrices and pairwise connectivity CSV files
  - Group-averaged FC analysis with statistical testing
  - MVMD decomposition plots (mode and slow-band)
  - Comprehensive analysis logs
        '''
    )
    parser.add_argument('--subjects-per-group', type=int, default=None, metavar='N',
                       help='Limit the number of subjects processed per group. '
                           'When specified, enables subject limiting. When not specified, '
                           'processes all available subjects (default: no limit)')
    parser.add_argument('--verbose', action='store_true', default=False,
                       help='Enable verbose output including detailed subject-level information '
                           'and processing status updates')
    parser.add_argument('--show-plots', action='store_true', default=False,
                       help='Display plots interactively in batches of 20. User must close '
                           'each batch to continue processing')
    parser.add_argument('--no-save', action='store_true', default=False,
                       help='Skip saving figures to disk. Figures are still created for '
                           'display if --show-plots is enabled')
    parser.add_argument('--skip-plots', action='store_true', default=False,
                       help='Skip creating plots entirely. Overrides --show-plots and --no-save. '
                           'Use this to run analysis without any visualization overhead')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Display configuration
    VERBOSE_OUTPUT = args.verbose
    CREATE_PLOTS = not args.skip_plots  # Whether to create plots (required for both displaying and saving)
    SHOW_PLOTS = args.show_plots and CREATE_PLOTS  # Whether to display plots interactively (requires CREATE_PLOTS=True)
    SAVE_FIGURES = not args.no_save and CREATE_PLOTS  # Whether to save figures to disk as SVG files (requires CREATE_PLOTS=True)

    # FC Matrix display mode:
    # - False: Show all correlations, mark non-significant with asterisks
    # - True: Hide non-significant correlations (masked)
    MASK_NONSIGNIFICANT = False
    MASK_DIAGONAL = False

    # Reproducibility
    random.seed(args.random_seed)

    main(
        mask_diagonal=MASK_DIAGONAL,
        mask_nonsignificant=MASK_NONSIGNIFICANT,
        create_plots=CREATE_PLOTS,
        show_plots=SHOW_PLOTS,
        save_figures=SAVE_FIGURES,
        verbose=VERBOSE_OUTPUT,
        subjects_per_group=args.subjects_per_group
    )

    # Note: plt.show() is now called within each batch in main(), not here
