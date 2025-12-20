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

import logging
import random
import warnings
from itertools import islice

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
from scipy import signal, stats
from statsmodels.stats.multitest import fdrcorrection

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
    plot_interhemispheric_intra_network_violin,
    plot_ipsilateral_intra_network_violin,
    plot_marginal_spectrum_per_mode,
    plot_multivariate_hilbert_spectrum,
    plot_roi_timeseries_result,
    plot_signal_decomposition,
)
from tcp.processing.lib.slow_band import get_band_number, get_frequency_range
from tcp.processing.lib.subject_filtering import get_accessible_subjects_from_file
from tcp.processing.roi import (
    CorticalAtlasLookup,
    ROIExtractionService,
    SubCorticalAtlasLookup,
)
from tcp.processing.utils.lists import chunks, split_by_sizes

logger = logging.getLogger(__name__)

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

    # Fisher-transform static FC
    z_corr_matrix = fisher_r_to_z(corr_matrix)

    # Compute p-values using Fisher r-to-z transformation
    rois_count = len(roi_labels)
    samples_count = timeseries_list[0].shape[0]
    p_values = np.ones((rois_count, rois_count))

    for i in range(rois_count):
        for j in range(i+1, rois_count):
            r = corr_matrix[i, j]

            if np.isnan(r):
                p_values[i, j] = 1.0
                p_values[j, i] = 1.0
                continue

            # Apply Fisher r-to-z transformation
            z = fisher_r_to_z(r)

            # Under H0, z ~ N(0, 1/sqrt(n-3))
            se_z = 1.0 / np.sqrt(samples_count - 3)
            z_stat = z / se_z

            # Two-tailed p-value from standard normal
            p_val = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

            p_values[i, j] = p_val
            p_values[j, i] = p_val

    return corr_matrix, z_corr_matrix, roi_labels, p_values

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

    modes_count, channels_count, samples_count = time_modes.shape

    if len(channel_labels) != channels_count:
        raise ValueError(f"channel_labels length ({len(channel_labels)}) must match channels_count ({channels_count})")

    if verbose:
        print(f"\n{'='*80}")
        print("COMPUTING MODE-LEVEL FUNCTIONAL CONNECTIVITY")
        print(f"{'='*80}")
        print(f"  Number of modes: {modes_count}")
        print(f"  Number of channels: {channels_count}")
        print(f"  Samples per mode: {samples_count}")

    # Initialize 3D arrays to store results for all modes
    mode_fc_matrices = np.zeros((modes_count, channels_count, channels_count))
    mode_fc_z_matrices = np.zeros((modes_count, channels_count, channels_count))
    mode_fc_pvalues = np.ones((modes_count, channels_count, channels_count))  # Initialize with 1.0

    # Compute FC for each mode independently
    for mode_idx in range(modes_count):
        # Extract signal for this mode: shape (channels_count, samples_count)
        mode_signal = time_modes[mode_idx, :, :]

        # Create timeseries dictionary for compute_fc_matrix()
        # Format: {channel_label: timeseries_array}
        mode_timeseries = {
            channel_labels[ch_idx]: mode_signal[ch_idx, :]
            for ch_idx in range(channels_count)
        }

        # Compute FC matrix using existing function
        fc_matrix, z_fc_matrix, fc_labels, fc_pvalues = compute_fc_matrix(
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
        mode_fc_z_matrices[mode_idx, :, :] = z_fc_matrix

        # Store p-values
        if fc_pvalues is not None:
            mode_fc_pvalues[mode_idx, :, :] = fc_pvalues
        else:
            mode_fc_pvalues[mode_idx, :, :] = np.nan

        if verbose and (mode_idx + 1) % 5 == 0:
            print(f"  Processed {mode_idx + 1}/{modes_count} modes...")

    if verbose:
        print(f"  Completed FC computation for all {modes_count} modes")
        print(f"  Output shapes:")
        print(f"    - mode_fc_matrices: {mode_fc_matrices.shape}")
        print(f"    - mode_fc_z_matrices: {mode_fc_z_matrices.shape}")
        print(f"    - mode_fc_pvalues: {mode_fc_pvalues.shape}")

    return {
        'mode_fc_matrices': mode_fc_matrices,
        'mode_fc_z_matrices': mode_fc_z_matrices,
        'mode_fc_pvalues': mode_fc_pvalues,
        'fc_labels': channel_labels,
        'n_modes': modes_count,
        'n_channels': channels_count
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
        mode_frequencies: np.ndarray, center frequency for each mode (length modes_count)
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
            'fc_matrix': np.ndarray (channels_count, channels_count) - Averaged r-scale matrix
            'fc_z_matrix': np.ndarray (channels_count, channels_count) - Averaged Z-scale matrix
            'fc_pvalues': np.ndarray (channels_count, channels_count) - P-values (placeholder)
            'fc_labels': list - Channel labels
            'mode_indices': list - Which modes contributed to this band
            'mode_frequencies': list - Center frequencies of contributing modes
            'modes_in_band_count': int - Number of modes in this band

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
    modes_count = mode_fc_data['n_modes']
    channels_count = mode_fc_data['n_channels']

    # Input validation
    if len(mode_frequencies) != modes_count:
        raise ValueError(f"mode_frequencies length ({len(mode_frequencies)}) must match modes_count ({modes_count})")

    if verbose:
        print(f"\n{'='*80}")
        print("AGGREGATING MODE-LEVEL FC INTO SLOW-BANDS")
        print(f"{'='*80}")
        print(f"  Total modes to process: {modes_count}")
        print(f"  Mode frequencies: {mode_frequencies}")

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
            modes_in_band_count = len(band_mode_groups[band_num]['indices'])
            freqs = band_mode_groups[band_num]['frequencies']
            if modes_in_band_count > 0:
                print(f"    Slow-{band_num}: {modes_in_band_count} modes, freqs={freqs}")
            else:
                print(f"    Slow-{band_num}: 0 modes (empty band)")

    # Aggregate FC matrices for each band
    slow_band_fc_results = {}

    for band_num in ['6', '5', '4', '3', '2', '1']:
        mode_indices = band_mode_groups[band_num]['indices']
        mode_freqs = band_mode_groups[band_num]['frequencies']
        modes_in_band_count = len(mode_indices)

        if modes_in_band_count == 0:
            # Skip empty bands
            if verbose:
                print(f"\n  Slow-{band_num}: Skipping (no modes in this band)")
            continue

        elif modes_in_band_count == 1:
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
            z_matrices_in_band = mode_fc_z_matrices[mode_indices, :, :]  # shape: (modes_in_band_count, channels_count, channels_count)

            # Compute mean Z-matrix (handles NaN values)
            avg_z_matrix = np.nanmean(z_matrices_in_band, axis=0)  # shape: (channels_count, channels_count)

            # Convert back to r-scale
            avg_r_matrix = fisher_z_to_r(avg_z_matrix)

            # P-value computation: Placeholder for now
            # TODO: This will be updated later with t-test or permutation testing
            # For now, set all p-values to NaN to indicate they need computation
            fc_pvalues = np.full((channels_count, channels_count), np.nan)

            fc_matrix = avg_r_matrix
            fc_z_matrix = avg_z_matrix

            if verbose:
                print(f"\n  Slow-{band_num}: Averaged {modes_in_band_count} modes")
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
            'modes_in_band_count': modes_in_band_count
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

        # Debug: Check how many subjects have the required data
        successful_count = sum(1 for sid in group_subjects if sid in subject_results and subject_results[sid].get('success'))
        total_count = len(group_subjects)
        print(f"  Debug: {successful_count}/{total_count} subjects marked as successful")

    # Collect FC matrices from all subjects in this group
    fc_matrices = []
    fc_labels = None
    included_subjects = []

    for subject_id in group_subjects:
        if subject_id not in subject_results:
            continue

        result = subject_results[subject_id]

        # Check if subject has required data, even if marked as failed
        has_static_fc = False
        has_slow_band_fc = False

        if fc_type == 'static':
            fc_data = result.get('fc_static')
            has_static_fc = fc_data and fc_data.get('r_fc') is not None
        else:
            slow_band_data = result.get('slow_band_fc')
            has_slow_band_fc = slow_band_data and band_key and band_key in slow_band_data

        # Skip if no relevant data (regardless of success flag)
        if fc_type == 'static' and not has_static_fc:
            continue
        elif fc_type == 'slow_band' and not has_slow_band_fc:
            continue

        # Extract FC data based on type
        if fc_type == 'static':
            fc_data = result.get('fc_static')
            if fc_data and fc_data.get('r_fc') is not None:
                fc_matrix = fc_data['r_fc']
                if fc_labels is None:
                    fc_labels = fc_data.get('labels')
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
            # Debug: Check what data subjects actually have
            subjects_with_data = 0
            for subject_id in group_subjects:
                if subject_id in subject_results:
                    result = subject_results[subject_id]
                    has_static = result.get('static_functional_connectivity', {}).get('static_fc_matrix') is not None
                    has_slow_band = bool(result.get('slow_band_fc', {}))
                    success = result.get('success', False)
                    if has_static or has_slow_band:
                        subjects_with_data += 1
                        print(f"    {subject_id}: success={success}, has_static={has_static}, has_slow_band={has_slow_band}")

            if subjects_with_data == 0:
                print(f"    No subjects have any FC data at all")
            else:
                print(f"    {subjects_with_data} subjects have some FC data but weren't included")
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
    rois_count = avg_fc_matrix.shape[0]
    p_values = np.ones((rois_count, rois_count))  # Initialize with 1.0 (non-significant)

    for i in range(rois_count):
        for j in range(rois_count):
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
        significant_count = np.sum(significant_mask)
        total_count = (rois_count * (rois_count - 1)) // 2
        print(f"  Significant correlations: {significant_count}/{total_count} ({significant_count/total_count*100:.1f}%)")

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

    rois_count = len(roi_labels)

    # Extract all pairwise correlations
    for i in range(rois_count):
        for j in range(i+1, rois_count):
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
                        'network': roi1_network,  # Store network info for roi1
                        'network2': roi2_network,  # NEW: Also store network for roi2
                        'roi1_index': i,  # Store indices for later use
                        'roi2_index': j,
                        'is_intra_network': (roi1_network == roi2_network and roi1_network is not None)  # NEW: Flag for filtering
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
    # NOTE: This grouping is ONLY for intra-network pairs (for statistical testing)
    interhemi_pairs = results['interhemispheric']['pairs']
    network_pairs = {}
    network_stats = {}

    for pair_key, pair_data in interhemi_pairs.items():
        # FILTER: Only include intra-network pairs for statistical analysis
        if not pair_data.get('is_intra_network', False):
            continue

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


def extract_interhemispheric_network_coherence_per_mode(fc_matrix, roi_labels):
    """
    Extract intra-network interhemispheric coherence for a SINGLE mode's FC matrix.

    This implements Level 1 Reduction (Spatial): For each network, extract all
    intra-network interhemispheric pairs, apply Fisher-Z transformation, and compute
    the arithmetic mean → ONE scalar per network for this mode.

    Args:
        fc_matrix: np.ndarray (n_channels, n_channels) - Correlation matrix for ONE mode
        roi_labels: list of str - Channel labels

    Returns:
        dict: {network_key: fisher_z_scalar}
              e.g., {'PFCm_DefaultA': 0.42, 'PFCv_LimbicB': 0.35, 'AMY_lAMY': 0.28}
              Returns {} if no valid interhemispheric pairs found
    """
    network_coherence = {}
    rois_count = len(roi_labels)

    # Collect interhemispheric intra-network pairs
    network_pairs = {}  # {network_key: [list of r-values]}

    for i in range(rois_count):
        for j in range(i+1, rois_count):
            roi1, roi2 = roi_labels[i], roi_labels[j]
            corr_val = fc_matrix[i, j]

            # Skip NaN values
            if np.isnan(corr_val):
                continue

            # Parse labels
            roi1_parts = roi1.split('_')
            roi2_parts = roi2.split('_')

            if len(roi1_parts) < 2 or len(roi2_parts) < 2:
                continue

            roi1_region = roi1_parts[0]
            roi1_hemi = roi1_parts[1]
            roi1_network = roi1_parts[2] if len(roi1_parts) > 2 else None

            roi2_region = roi2_parts[0]
            roi2_hemi = roi2_parts[1]
            roi2_network = roi2_parts[2] if len(roi2_parts) > 2 else None

            # Check if interhemispheric (same region, different hemisphere)
            if roi1_region != roi2_region or roi1_hemi == roi2_hemi:
                continue

            # Check if intra-network (same network across hemispheres)
            if roi1_network != roi2_network or roi1_network is None:
                continue

            # This is a valid intra-network interhemispheric pair
            network_key = f"{roi1_region}_{roi1_network}"

            if network_key not in network_pairs:
                network_pairs[network_key] = []

            network_pairs[network_key].append(corr_val)

    # Level 1 Reduction: Spatial averaging within each network
    for network_key, r_values in network_pairs.items():
        r_array = np.array(r_values)

        # Apply Fisher Z-transformation
        z_values = fisher_r_to_z(r_array)

        # Compute arithmetic mean (spatial average)
        mean_z = np.nanmean(z_values)

        network_coherence[network_key] = float(mean_z)

    return network_coherence


def compute_band_specific_coherence(mode_fc_data, mode_frequencies, verbose=False):
    """
    Compute band-specific interhemispheric network coherence using nested averaging.

    This implements the correct two-level reduction:
    Level 1 (Spatial): For each mode, extract intra-network interhemispheric pairs,
                       Fisher-Z transform, and average → ONE scalar per mode per network
    Level 2 (Spectral): For each slow-band, average the scalars from all modes in that band
                        → ONE scalar per band per network

    Args:
        mode_fc_data: dict from compute_fc_per_mode() containing:
                     - 'mode_fc_matrices': r-scale FC matrices per mode (n_modes, n_channels, n_channels)
                     - 'fc_labels': channel labels
                     - 'n_modes': number of modes
        mode_frequencies: np.ndarray, center frequency for each mode
        verbose: If True, print progress information

    Returns:
        dict: {
            'slow-6': {network_key: fisher_z_scalar, ...},
            'slow-5': {network_key: fisher_z_scalar, ...},
            ...
        }
        Each band contains {network_key: mean_fisher_z} pairs
    """
    mode_fc_matrices = mode_fc_data['mode_fc_matrices']
    fc_labels = mode_fc_data['fc_labels']
    modes_count = mode_fc_data['n_modes']

    if len(mode_frequencies) != modes_count:
        raise ValueError(f"mode_frequencies length ({len(mode_frequencies)}) must match modes_count ({modes_count})")

    if verbose:
        print(f"\n{'='*80}")
        print("COMPUTING BAND-SPECIFIC INTERHEMISPHERIC COHERENCE (NESTED AVERAGING)")
        print(f"{'='*80}")
        print(f"  Total modes: {modes_count}")
        print(f"  Mode frequencies: {mode_frequencies}")

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
            modes_in_band_count = len(band_mode_groups[band_num]['indices'])
            freqs = band_mode_groups[band_num]['frequencies']
            if modes_in_band_count > 0:
                print(f"    Slow-{band_num}: {modes_in_band_count} modes, freqs={freqs}")

    # Level 1: Extract coherence per mode (spatial reduction)
    mode_coherence_by_band = {}
    for band_num in ['6', '5', '4', '3', '2', '1']:
        mode_coherence_by_band[band_num] = []  # List of dicts, one per mode

    if verbose:
        print(f"\n  Level 1 (Spatial): Extracting coherence per mode...")

    for mode_idx in range(modes_count):
        fc_matrix = mode_fc_matrices[mode_idx, :, :]
        freq = mode_frequencies[mode_idx]
        band_num = get_band_number(freq)

        if band_num is None:
            continue

        # Extract network coherence for this single mode
        mode_network_coherence = extract_interhemispheric_network_coherence_per_mode(fc_matrix, fc_labels)

        mode_coherence_by_band[band_num].append(mode_network_coherence)

        if verbose and len(mode_network_coherence) > 0:
            print(f"    Mode {mode_idx} (freq={freq:.4f} Hz, band=Slow-{band_num}): {len(mode_network_coherence)} networks")

    # Level 2: Average across modes within each band (spectral reduction)
    band_specific_coherence = {}

    if verbose:
        print(f"\n  Level 2 (Spectral): Averaging across modes per band...")

    for band_num in ['6', '5', '4', '3', '2', '1']:
        band_name = f'slow-{band_num}'
        mode_coherence_list = mode_coherence_by_band[band_num]
        modes_in_band_count = len(mode_coherence_list)

        if modes_in_band_count == 0:
            if verbose:
                print(f"    {band_name}: No modes in this band, skipping")
            continue

        # Collect all networks across all modes in this band
        all_networks = set()
        for mode_coherence in mode_coherence_list:
            all_networks.update(mode_coherence.keys())

        # For each network, average Fisher-Z values across modes
        band_coherence = {}
        for network_key in all_networks:
            z_values = []
            for mode_coherence in mode_coherence_list:
                if network_key in mode_coherence:
                    z_values.append(mode_coherence[network_key])

            if len(z_values) > 0:
                # Spectral average: arithmetic mean of Fisher-Z values
                mean_z = np.nanmean(z_values)
                band_coherence[network_key] = float(mean_z)

        band_specific_coherence[band_name] = band_coherence

        if verbose:
            print(f"    {band_name}: {len(band_coherence)} networks, {modes_in_band_count} modes averaged")

    return band_specific_coherence


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
    modes_count, channels_count, samples_count = time_modes.shape

    modes_data = []

    # Process each mode independently
    for mode_idx in range(modes_count):
        mode_signal = time_modes[mode_idx, :, :]  # Shape: (channels, samples)

        # Initialize arrays for this mode
        inst_freq = np.zeros((channels_count, samples_count))
        inst_amp = np.zeros((channels_count, samples_count))
        analytic_signals = np.zeros((channels_count, samples_count), dtype=complex)

        # Apply Hilbert transform to each channel of this mode
        for ch_idx in range(channels_count):
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
        'n_modes': modes_count,
        'n_channels': channels_count,
        'n_samples': samples_count
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
    Stage 1: Pure data extraction and basic computation for a single subject.

    This function ONLY performs:
    1. Data loading and ROI extraction
    2. Static FC computation (correlation matrix)
    3. MVMD decomposition
    4. Per-mode FC computation
    5. Hilbert transform for spectral analysis

    NO analysis, aggregation, or statistics are computed here.

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
        dict: Pure computation results containing:
            - subject_id: str
            - success: bool
            - z_fc_static: np.ndarray (Fisher-transformed static FC)
            - z_fc_modes: np.ndarray (Fisher-transformed FC per mode)
            - analytic_signal_modes: np.ndarray (Hilbert transform output)
            - channel_labels: dict (Index to standardized channel names)
            - mvmd_metadata: dict (Center frequencies, sampling rate, etc.)
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing Subject: {subject_id}")
        print(f"{'='*60}")

    try:
        # ===== 1. DATA LOADING =====
        hammer_files = manager.get_subject_files_by_task(subject_id, 'timeseries', 'hammer')
        if not hammer_files:
            return {
                'subject_id': subject_id,
                'error': 'No hammer task files found',
                'success': False
            }

        subject_file = loader.resolve_file_path(hammer_files[0])

        # ===== 2. DATA LOADING & SEGMENTATION =====
        with h5py.File(subject_file, 'r') as file:
            a_group_key = list(file.keys())[0]
            data = np.asarray(file[a_group_key])

        if verbose:
            print(f"Loaded data with shape: {data.shape}")

        # Segment into anatomical groups
        cortical_timeseries = data[:400]
        subcortical_timeseries = data[400:432]
        # Note: cerebellum data (432:) ignored for this analysis

        # ===== 3. ROI EXTRACTION =====
        if verbose:
            print(f"\n=== ROI EXTRACTION ===")

        # Extract hemisphere-specific ROI timeseries
        cortical_right_timeseries = cortical_roi_extractor.extract_roi_timeseries_by_hemisphere(
            cortical_timeseries, cortical_ROIs, hemisphere='RH', aggregation_method='all'
        )
        cortical_left_timeseries = cortical_roi_extractor.extract_roi_timeseries_by_hemisphere(
            cortical_timeseries, cortical_ROIs, hemisphere='LH', aggregation_method='all'
        )
        subcortical_right_timeseries = subcortical_roi_extractor.extract_roi_timeseries_by_hemisphere(
            subcortical_timeseries, subcortical_ROIs, hemisphere='rh', aggregation_method='all'
        )
        subcortical_left_timeseries = subcortical_roi_extractor.extract_roi_timeseries_by_hemisphere(
            subcortical_timeseries, subcortical_ROIs, hemisphere='lh', aggregation_method='all'
        )

        # Extract combined ROI timeseries (for plotting)
        cortical_roi_timeseries = cortical_roi_extractor.extract_roi_timeseries(
            cortical_timeseries, cortical_ROIs, aggregation_method='all'
        )
        subcortical_roi_timeseries = subcortical_roi_extractor.extract_roi_timeseries(
            subcortical_timeseries, subcortical_ROIs, aggregation_method='all'
        )

        # Create standardized channel labels
        all_channel_labels = []
        all_channels_list = []

        # Process cortical channels
        for roi_name in cortical_ROIs:
            for hemi, hemi_data in [('RH', cortical_right_timeseries), ('LH', cortical_left_timeseries)]:
                if roi_name in hemi_data:
                    roi_timeseries = hemi_data[roi_name]
                    # Get atlas indices to create proper labels
                    indices_dict = cortical_atlas.get_roi_indices_by_hemisphere([roi_name], hemisphere=hemi)
                    indices = indices_dict.get(roi_name, [])

                    for parcel_idx, atlas_idx in enumerate(indices):
                        full_name = cortical_atlas.get_parcel_name(atlas_idx)
                        if full_name:
                            # Expected format: '17networks_RH_DefaultA_PFCm_1'
                            parts = full_name.split('_')
                            if len(parts) >= 4:
                                # parts: ['17networks', 'RH', 'DefaultA', 'PFCm', '1']
                                hemisphere = parts[1]
                                network = parts[2]
                                region = parts[3]
                                subarea = parts[4] if len(parts) > 4 else ''
                                # Create label: PFCm_RH_DefaultA_p1 (includes network, p = parcel)
                                label = f'{region}_{hemisphere}_{network}'
                                if subarea:
                                    label += f'_p{subarea}'
                            else:
                                # Fallback if parsing fails
                                label = f'{roi_name}_{hemi}_parcel{parcel_idx+1}'
                        else:
                            label = f'{roi_name}_{hemi}_parcel{parcel_idx+1}'

                        all_channel_labels.append(label)
                        all_channels_list.append(roi_timeseries[parcel_idx, :])

        # Process subcortical channels with proper parcel naming
        # Create parcel labels for each ROI and hemisphere using atlas information
        subcortical_parcel_labels = {}  # Maps ROI -> hemisphere -> list of parcel labels

        for roi_name in subcortical_ROIs:
            if roi_name not in subcortical_parcel_labels:
                subcortical_parcel_labels[roi_name] = {'rh': [], 'lh': []}

            # Right hemisphere labels - get actual anatomical names from atlas
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

                # Add timeseries for right hemisphere
                roi_timeseries = subcortical_right_timeseries[roi_name]
                for parcel_idx, label in enumerate(rh_parcel_names):
                    all_channel_labels.append(label)
                    all_channels_list.append(roi_timeseries[parcel_idx, :])

            # Left hemisphere labels - get actual anatomical names from atlas
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

                # Add timeseries for left hemisphere
                roi_timeseries = subcortical_left_timeseries[roi_name]
                for parcel_idx, label in enumerate(lh_parcel_names):
                    all_channel_labels.append(label)
                    all_channels_list.append(roi_timeseries[parcel_idx, :])

        # Stack all timeseries
        all_channels = np.stack(all_channels_list, axis=0)

        if verbose:
            print(f"Extracted {all_channels.shape[0]} channels from {len(cortical_ROIs) + len(subcortical_ROIs)} ROIs")

        # ===== 4. STATIC FUNCTIONAL CONNECTIVITY =====
        if verbose:
            print(f"\n=== STATIC FUNCTIONAL CONNECTIVITY ===")

        # Create timeseries dictionary for FC computation
        fc_timeseries = {label: all_channels[i, :] for i, label in enumerate(all_channel_labels)}
        fc_matrix, z_fc_static, fc_labels, fc_pvalues = compute_fc_matrix(fc_timeseries)

        if verbose and fc_matrix is not None:
            print(f"Static FC matrix shape: {fc_matrix.shape}")

        # ===== 5. MVMD DECOMPOSITION =====
        if verbose:
            print(f"\n=== MVMD DECOMPOSITION ===")

        mvmd = MVMD(config=None)
        mvmd_result = mvmd.decompose(all_channels, num_modes=10)
        time_modes = mvmd_result['time_modes']
        center_freqs = mvmd_result['center_freqs'][-1, :]

        if verbose:
            print(f"MVMD modes shape: {time_modes.shape}")
            print(f"Center frequencies: {center_freqs}")

        # ===== 6. PER-MODE FUNCTIONAL CONNECTIVITY =====
        if verbose:
            print(f"\n=== PER-MODE FUNCTIONAL CONNECTIVITY ===")

        mode_fc_data = compute_fc_per_mode(time_modes, all_channel_labels, verbose=verbose)
        z_fc_modes = mode_fc_data.get('mode_fc_z_matrices') if mode_fc_data else None

        # ===== 7. AGGREGATE MODE FC INTO SLOW-BAND FC =====
        if verbose:
            print(f"\n=== SLOW-BAND AGGREGATION OF MODE FC ===")

        band_fc_data = aggregate_mode_fc_to_bands(mode_fc_data, center_freqs, verbose=verbose)


        # ===== 8. HILBERT SPECTRAL ANALYSIS =====
        if verbose:
            print(f"\n=== HILBERT SPECTRAL ANALYSIS ===")

        hsa_data = compute_hilbert_transform_per_mode(time_modes, SAMPLING_RATE)

        # Extract analytic signals: shape (n_modes, n_channels, n_samples)
        analytic_signal_modes = None
        modes_count = 0
        if hsa_data and hsa_data.get('modes_data'):
            modes_data = hsa_data['modes_data']
            modes_count = len(modes_data)
            if modes_count > 0:
                channels_count, samples_count = modes_data[0]['analytic_signal'].shape
                analytic_signal_modes = np.zeros((modes_count, channels_count, samples_count), dtype=complex)
                for i, mode_data in enumerate(modes_data):
                    analytic_signal_modes[i] = mode_data['analytic_signal']

        if verbose:
            print(f"Hilbert analysis complete: {modes_count} modes processed")

        # ===== 8. PREPARE OUTPUTS =====
        # Standardized channel labels mapping
        channel_labels = {idx: label for idx, label in enumerate(all_channel_labels)}

        # MVMD metadata
        mvmd_metadata = {
            'center_freqs': center_freqs,
            'sampling_rate': SAMPLING_RATE,
            'n_modes': time_modes.shape[0],
            'n_channels': all_channels.shape[0],
            'reconstruction_error': mvmd_result.get('reconstruction_error', None)
        }

        return {
            'subject_id': subject_id,
            'success': True,
            'channel_labels': channel_labels,
            'fc_static': {
                'r_fc': fc_matrix,
                'z_fc': z_fc_static,
                'labels': fc_labels,
            },
            'mvmd': {
                **mvmd_result,
                'metadata': mvmd_metadata,
            },
            'fc_modes': {
                'r_fc': mode_fc_data.get('mode_fc_matrices', None),
                'z_fc': mode_fc_data.get('mode_fc_z_matrices', None),
                'labels': mode_fc_data.get('fc_labels', None),
                'n_modes': mode_fc_data.get('n_modes', 0),
                'n_channels': mode_fc_data.get('n_channels', 0),
            },
            'fc_bands': {
                'r_fc': band_fc_data.get('fc_matrix', None),
                'z_fc': band_fc_data.get('fc_z_matrix', None),
                'labels': band_fc_data.get('fc_labels', None),
                'mode_indeces': band_fc_data.get('mode_indeces', []),
                'mode_frequencies': band_fc_data.get('mode_freqs', []),
                'n_modes_in_band': band_fc_data.get('modes_in_band_count', 0),
            },
            'hsa': analytic_signal_modes, # Hilbert Spectral Analysis
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
                    }
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
                    }
                }
            },

            """ LEGACY UNDERNEATH """
            'z_fc_static': z_fc_static, # LEGACY
            'static_fc_pvalues': fc_pvalues, # LEGACY
            'z_fc_modes': z_fc_modes, # LEGACY
            'mode_fc_pvalues': mode_fc_data.get('mode_fc_pvalues') if mode_fc_data else None, # LEGACY
            'analytic_signal_modes': analytic_signal_modes, # LEGACY
            'mvmd_metadata': mvmd_metadata, # LEGACY
        }

    except Exception as e:
        if verbose:
            print(f"[ERROR] Failed to process subject {subject_id}: {str(e)}")
        return {
            'subject_id': subject_id,
            'error': str(e),
            'success': False
        }

def create_connectivity_mappings(all_subject_results, verbose=True):
    """
    Stage 2: Create standardized connectivity pair mappings from all subject results.

    This function runs once after collecting all subject data from Stage 1.
    It standardizes channel names and creates index lists for different types
    of connectivity analysis.

    Args:
        all_subject_results: Dict of subject_id -> process_subject results
        verbose: Whether to print detailed output

    Returns:
        dict: Connectivity mappings containing:
            - channel_index_map: Dict mapping standardized names to indices
            - interhemispheric_pairs: Dict of (LH_idx, RH_idx) pairs separated by network type
            - ipsilateral_pairs: Dict of within-hemisphere pairs
            - intra_network_pairs: Dict for statistical analysis (same region+network across hemispheres)
    """
    if verbose:
        print(f"\n{'='*80}")
        print("STAGE 2: CREATING CONNECTIVITY MAPPINGS")
        print(f"{'='*80}")

    # Collect all unique channel labels across subjects
    all_channel_labels = set()
    for subject_id, result in all_subject_results.items():
        if result.get('success') and result.get('channel_labels'):
            all_channel_labels.update(result['channel_labels'].values())

    # Create standardized channel-to-index mapping
    sorted_labels = sorted(all_channel_labels)
    channel_index_map = {label: idx for idx, label in enumerate(sorted_labels)}

    if verbose:
        print(f"Found {len(sorted_labels)} unique channel labels across all subjects")
        print(f"Sample labels: {sorted_labels[:5]}...")

    # Separate cortical and subcortical channels
    cortical_channels = []
    subcortical_channels = []

    for label in sorted_labels:
        if any(region in label for region in ['PFCm', 'PFCv']):
            cortical_channels.append(label)
        elif 'AMY' in label:
            subcortical_channels.append(label)

    # Create interhemispheric pairs (LH - RH)
    interhemispheric_pairs = {
        'indiscriminate': [], # Network independent
        'intra_network': [],  # Same region + same network across hemispheres
        'inter_network': []   # Same region + different network across hemispheres
    }

    # Create ipsilateral pairs (within hemisphere)
    ipsilateral_pairs = {
        'left_hemisphere': [],
        'right_hemisphere': []
    }

    # Parse cortical channels for interhemispheric pairing
    # Expected format: "PFCm_RH_DefaultA_p1" -> region_hemi_network_parcel
    lh_channels = [ch for ch in cortical_channels if '_LH_' in ch]
    rh_channels = [ch for ch in cortical_channels if '_RH_' in ch]

    for lh_ch in lh_channels:
        lh_parts = lh_ch.split('_')
        if len(lh_parts) >= 3:
            lh_region = lh_parts[0]  # e.g., "PFCm"
            lh_network = lh_parts[2] # e.g., "DefaultA"

            # Find corresponding RH channel with same region
            for rh_ch in rh_channels:
                rh_parts = rh_ch.split('_')
                if len(rh_parts) >= 3:
                    rh_region = rh_parts[0]
                    rh_network = rh_parts[2]

                    if lh_region == rh_region:  # Same region across hemispheres
                        lh_idx = channel_index_map[lh_ch]
                        rh_idx = channel_index_map[rh_ch]

                        interhemispheric_pairs['indiscriminate'].append({
                            'lh_idx': lh_idx, 'rh_idx': rh_idx,
                            'lh_label': lh_ch, 'rh_label': rh_ch,
                            'region': lh_region,
                            'lh_network': lh_network, 'rh_network': rh_network
                        })

                        if lh_network == rh_network:  # Same network
                            interhemispheric_pairs['intra_network'].append({
                                'lh_idx': lh_idx, 'rh_idx': rh_idx,
                                'lh_label': lh_ch, 'rh_label': rh_ch,
                                'region': lh_region, 'network': lh_network
                            })
                        else:  # Different network
                            interhemispheric_pairs['inter_network'].append({
                                'lh_idx': lh_idx, 'rh_idx': rh_idx,
                                'lh_label': lh_ch, 'rh_label': rh_ch,
                                'region': lh_region,
                                'lh_network': lh_network, 'rh_network': rh_network
                            })

    # Parse subcortical channels for interhemispheric pairing
    # Expected format: "AMY_RH_lAMY" -> region_hemi_subregion
    # For subcortical regions, we treat each subregion as its own "network" for intra-network analysis
    lh_subcortical = [ch for ch in subcortical_channels if '_LH_' in ch]
    rh_subcortical = [ch for ch in subcortical_channels if '_RH_' in ch]

    for lh_ch in lh_subcortical:
        lh_parts = lh_ch.split('_')
        if len(lh_parts) >= 3:
            lh_region = lh_parts[0]  # e.g., "AMY"
            lh_subregion = lh_parts[2]  # e.g., "lAMY" or "parcel1" (fallback)

            # Find corresponding RH channel with same region and subregion
            for rh_ch in rh_subcortical:
                rh_parts = rh_ch.split('_')
                if len(rh_parts) >= 3:
                    rh_region = rh_parts[0]
                    rh_subregion = rh_parts[2]

                    if lh_region == rh_region:
                        lh_idx = channel_index_map[lh_ch]
                        rh_idx = channel_index_map[rh_ch]

                        # For subcortical regions, use subregion as "network" identifier
                        lh_network = lh_subregion
                        rh_network = rh_subregion
                        interhemispheric_pairs['indiscriminate'].append({
                            'lh_idx': lh_idx, 'rh_idx': rh_idx,
                            'lh_label': lh_ch, 'rh_label': rh_ch,
                            'region': lh_region,
                            'lh_network': lh_network, 'rh_network': rh_network
                        })

                        if lh_network == rh_network:  # Same region and subregion
                            interhemispheric_pairs['intra_network'].append({
                                'lh_idx': lh_idx, 'rh_idx': rh_idx,
                                'lh_label': lh_ch, 'rh_label': rh_ch,
                                'region': lh_region, 'network': lh_network
                            })

    # Create ipsilateral pairs within each hemisphere and group by region/network combinations
    def parse_region_network(label):
        parts = label.split('_')
        region = parts[0] if len(parts) > 0 else 'Unknown'
        network = parts[2] if len(parts) > 2 else 'Unknown'
        return region, network

    ipsilateral_pair_groups = {'LH': {}, 'RH': {}}
    for hemisphere, channels in [('LH', [ch for ch in sorted_labels if '_LH_' in ch]),
                                 ('RH', [ch for ch in sorted_labels if '_RH_' in ch])]:
        for i, ch1 in enumerate(channels):
            for ch2 in channels[i+1:]:
                idx1 = channel_index_map[ch1]
                idx2 = channel_index_map[ch2]
                region1, network1 = parse_region_network(ch1)
                region2, network2 = parse_region_network(ch2)
                key_parts = sorted([f"{region1}_{network1}", f"{region2}_{network2}"])
                conn_key = f"{hemisphere}:{key_parts[0]}__{key_parts[1]}"
                if conn_key not in ipsilateral_pair_groups[hemisphere]:
                    ipsilateral_pair_groups[hemisphere][conn_key] = {
                        'pairs': [],
                        'regions_networks': key_parts
                    }
                ipsilateral_pair_groups[hemisphere][conn_key]['pairs'].append({
                    'idx1': idx1, 'idx2': idx2,
                    'label1': ch1, 'label2': ch2,
                    'region1': region1, 'network1': network1,
                    'region2': region2, 'network2': network2,
                    'hemisphere': hemisphere
                })

    # Create intra-network pairs for statistics (only same region + same network across hemispheres)
    intra_network_pairs = {}
    for pair in interhemispheric_pairs['intra_network']:
        network_key = f"{pair['region']}_{pair['network']}"
        if network_key not in intra_network_pairs:
            intra_network_pairs[network_key] = []
        intra_network_pairs[network_key].append(pair)

    if verbose:
        print(f"Connectivity mapping results:")
        print(f"  Interhemispheric intra-network pairs: {len(interhemispheric_pairs['intra_network'])}")
        print(f"  Interhemispheric inter-network pairs: {len(interhemispheric_pairs['inter_network'])}")
        print(f"  Left hemisphere pairs: {len(ipsilateral_pairs['left_hemisphere'])}")
        print(f"  Right hemisphere pairs: {len(ipsilateral_pairs['right_hemisphere'])}")
        print(f"  Intra-network groups for statistics: {len(intra_network_pairs)}")

        if intra_network_pairs:
            print(f"  Network groups: {list(intra_network_pairs.keys())}")

    return {
        'channel_index_map': channel_index_map,
        'interhemispheric_pairs': interhemispheric_pairs,
        'ipsilateral_pairs': ipsilateral_pairs,
        'ipsilateral_pair_groups': ipsilateral_pair_groups,
        'intra_network_pairs': intra_network_pairs,
        'cortical_channels': cortical_channels,
        'subcortical_channels': subcortical_channels
    }


# ===== STAGE 3: GROUPED AGGREGATION =====

def aggregate_group_connectivity(all_subject_results, connectivity_mappings, anhedonia_groups, verbose=True):
    """
    Stage 3: Group subjects by anhedonia type and aggregate FC matrices.

    Args:
        all_subject_results: Dict of subject_id -> process_subject results
        connectivity_mappings: Results from create_connectivity_mappings()
        anhedonia_groups: Dict with 'low-anhedonic', 'high-anhedonic', 'non-anhedonic' subject lists
        verbose: Whether to print detailed output

    Returns:
        dict: Group-level aggregated results containing:
            - static_fc_by_group: Group-averaged static FC matrices
            - slow_band_fc_by_group: Group-averaged FC by frequency band and group
    """
    if verbose:
        print(f"\n{'='*80}")
        print("STAGE 3: AGGREGATING GROUP CONNECTIVITY")
        print(f"{'='*80}")

    # Initialize results structure
    group_results = {
        'static_fc_by_group': {},
        'slow_band_fc_by_group': {}
    }

    # Process each group
    for group_name, subject_ids in anhedonia_groups.items():
        if verbose:
            print(f"\nProcessing {group_name} group ({len(subject_ids)} subjects)...")

        # Collect static FC matrices for this group
        static_matrices = []
        valid_subjects = []

        for subject_id in subject_ids:
            result = all_subject_results.get(subject_id)
            if result and result.get('success') and result.get('z_fc_static') is not None:
                static_matrices.append(result['z_fc_static'])
                valid_subjects.append(subject_id)

        if static_matrices:
            # Compute group average in Fisher-Z space
            stacked_matrices = np.stack(static_matrices, axis=0)
            avg_z_matrix = np.nanmean(stacked_matrices, axis=0)

            # Convert back to correlation coefficients for plotting
            avg_r_matrix = fisher_z_to_r(avg_z_matrix)

            group_results['static_fc_by_group'][group_name] = {
                'avg_z_matrix': avg_z_matrix,      # For computation/statistics
                'avg_r_matrix': avg_r_matrix,      # For plotting
                'n_subjects': len(valid_subjects),
                'subject_ids': valid_subjects
            }

            if verbose:
                print(f"  Static FC: {len(valid_subjects)} valid subjects")

        # Collect slow-band FC matrices for this group
        slow_band_matrices = {}  # {band_name: [list of Z matrices]}

        for subject_id in subject_ids:
            result = all_subject_results.get(subject_id)
            if not (result and result.get('success')):
                continue

            z_fc_modes = result.get('z_fc_modes')
            center_freqs = result.get('mvmd_metadata', {}).get('center_freqs')

            if z_fc_modes is None or center_freqs is None:
                continue

            # Group this subject's modes by bands
            for mode_idx, freq in enumerate(center_freqs):
                band_num = get_band_number(freq)
                if band_num is not None:
                    band_name = f'slow-{band_num}'

                    if band_name not in slow_band_matrices:
                        slow_band_matrices[band_name] = []

                    slow_band_matrices[band_name].append(z_fc_modes[mode_idx])

        # Aggregate slow-band matrices within this group
        group_results['slow_band_fc_by_group'][group_name] = {}
        for band_name, z_matrices_list in slow_band_matrices.items():
            if len(z_matrices_list) == 0:
                continue

            # Stack and average all Z-matrices for this band within this group
            stacked_z = np.stack(z_matrices_list, axis=0)
            avg_z_matrix = np.nanmean(stacked_z, axis=0)

            # Convert to R-scale for plotting
            avg_r_matrix = fisher_z_to_r(avg_z_matrix)

            group_results['slow_band_fc_by_group'][group_name][band_name] = {
                'avg_z_matrix': avg_z_matrix,      # For computation/statistics
                'avg_r_matrix': avg_r_matrix,      # For plotting
                'n_matrices': len(z_matrices_list),
                'description': f'{group_name} group {band_name} average ({len(z_matrices_list)} mode matrices)'
            }

        if verbose and group_results['slow_band_fc_by_group'][group_name]:
            slow_band_summary = ", ".join([f"{band}: {data['n_matrices']} matrices"
                                         for band, data in group_results['slow_band_fc_by_group'][group_name].items()])
            print(f"  Slow-band FC: {slow_band_summary}")

    if verbose:
        print(f"\nGroup aggregation completed:")
        for group_name, data in group_results['static_fc_by_group'].items():
            print(f"  {group_name}: {data['n_subjects']} subjects")

    return group_results


# ===== HELPER FUNCTIONS FOR STAGE 4 =====

def compute_band_specific_coherence_from_modes(z_fc_modes, center_freqs, network_pairs, network_key, subject_channel_labels=None, verbose=False):
    """
    Compute band-specific interhemispheric coherence from mode FC matrices.

    This function implements the nested averaging approach:
    1. Bin modes by frequency band
    2. Average Z-transformed FC matrices within each band
    3. Extract network pairs and compute spatial average

    Args:
        z_fc_modes: Fisher-transformed FC matrices per mode (n_modes, n_channels, n_channels)
        center_freqs: Center frequencies for each mode
        network_pairs: List of interhemispheric pairs for this network
        network_key: Network identifier for logging
        subject_channel_labels: Dict mapping subject indices to channel labels (for correct indexing)
        verbose: Whether to print detailed output

    Returns:
        dict: Band-specific coherence values {band_name: coherence_value}
    """
    from tcp.processing.lib.slow_band import get_frequency_range

    # Group modes by slow-band
    band_modes = {}
    for mode_idx, freq in enumerate(center_freqs):
        band_num = get_band_number(freq)
        if band_num is not None:
            band_name = f'slow-{band_num}'
            if band_name not in band_modes:
                band_modes[band_name] = []
            band_modes[band_name].append(mode_idx)

    # Compute coherence for each band
    band_coherence = {}
    for band_name, mode_indices in band_modes.items():
        if not mode_indices:
            continue

        # Average Z-matrices across modes in this band
        band_z_matrices = z_fc_modes[mode_indices]  # Shape: (modes_in_band_count, channels_count, channels_count)
        avg_band_z_matrix = np.nanmean(band_z_matrices, axis=0)  # Shape: (channels_count, channels_count)

        # Extract network pairs and compute spatial average
        network_z_values = []
        for pair in network_pairs:
            if subject_channel_labels:
                # Convert from standardized indices to channel labels, then to subject indices
                lh_label = pair['lh_label']
                rh_label = pair['rh_label']
                subject_label_to_idx = {label: idx for idx, label in subject_channel_labels.items()}

                if lh_label in subject_label_to_idx and rh_label in subject_label_to_idx:
                    subj_lh_idx = subject_label_to_idx[lh_label]
                    subj_rh_idx = subject_label_to_idx[rh_label]
                    z_value = avg_band_z_matrix[subj_lh_idx, subj_rh_idx]
                    if not np.isnan(z_value):
                        network_z_values.append(z_value)
            else:
                # Fallback to standardized indices (may be incorrect)
                lh_idx = pair['lh_idx']
                rh_idx = pair['rh_idx']
                z_value = avg_band_z_matrix[lh_idx, rh_idx]
                if not np.isnan(z_value):
                    network_z_values.append(z_value)

        if network_z_values:
            # Spatial averaging: collapse network nodes into single scalar
            band_coherence[band_name] = np.mean(network_z_values)

            if verbose:
                print(f"    {band_name}: {len(mode_indices)} modes, coherence = {band_coherence[band_name]:.4f}")

    return band_coherence


def compute_band_specific_coherence_from_modes_ipsi(z_fc_modes, center_freqs, pair_group, subject_channel_labels=None, verbose=False):
    """
    Compute band-specific ipsilateral coherence (within hemisphere) from mode FC matrices.

    Args:
        z_fc_modes: Fisher-transformed FC matrices per mode (n_modes, n_channels, n_channels)
        center_freqs: Center frequencies for each mode
        pair_group: Dict with keys 'pairs' (list of pair dicts with label1/label2) and 'regions_networks'
        subject_channel_labels: Dict mapping subject indices to channel labels
        verbose: Whether to print detailed output

    Returns:
        dict: Band-specific coherence values {band_name: coherence_value}
    """
    # Group modes by slow-band
    band_modes = {}
    for mode_idx, freq in enumerate(center_freqs):
        band_num = get_band_number(freq)
        if band_num is not None:
            band_name = f'slow-{band_num}'
            band_modes.setdefault(band_name, []).append(mode_idx)

    band_coherence = {}
    for band_name, mode_indices in band_modes.items():
        if not mode_indices:
            continue

        band_z_matrices = z_fc_modes[mode_indices]
        avg_band_z_matrix = np.nanmean(band_z_matrices, axis=0)

        network_z_values = []
        for pair in pair_group['pairs']:
            if subject_channel_labels:
                subject_label_to_idx = {label: idx for idx, label in subject_channel_labels.items()}
                l1 = pair['label1']
                l2 = pair['label2']
                if l1 in subject_label_to_idx and l2 in subject_label_to_idx:
                    idx1 = subject_label_to_idx[l1]
                    idx2 = subject_label_to_idx[l2]
                    z_val = avg_band_z_matrix[idx1, idx2]
                    if not np.isnan(z_val):
                        network_z_values.append(z_val)
            else:
                idx1 = pair['idx1']
                idx2 = pair['idx2']
                z_val = avg_band_z_matrix[idx1, idx2]
                if not np.isnan(z_val):
                    network_z_values.append(z_val)

        if network_z_values:
            band_coherence[band_name] = np.mean(network_z_values)

    return band_coherence

def get_group_permutations(all_subject_results, groups, verbose=False):
    results = {
        'group_permutations': [],
        'n_permutations': 0,
        'n_groups': 0,
        'group_sizes': []
    }

    # Make group permutations for permutation tests
    group_subject_counts = [len(subjects) for subjects in groups.values()]
    if all(counts >= PERMUTATION_GROUP_COUNT for counts in group_subject_counts):
        all_ids = [sid for sid in all_subject_results.keys()]
        subject_permutations = np.array([np.random.permutation(all_ids) for _ in range(PERMUTATION_COUNT)])
        grouped_chunks = split_by_sizes(subject_permutations, group_subject_counts, axis=1)
        formatted_permutations = []
        for i in range(PERMUTATION_COUNT):
            # Create a dictionary for generating "Group_1", "Group_2", etc.
            iteration_dict = {
                f"Group_{idx + 1}": chunk[i]
                for idx, chunk in enumerate(grouped_chunks)
            }
            formatted_permutations.append(iteration_dict)

        if verbose:
            logger.info(f"Created {len(grouped_chunks)} permutations:\n{formatted_permutations}")
        else:
            logger.info(f"Created {len(grouped_chunks)} permutations")

        results['group_permutations'] = formatted_permutations
        results['group_sizes'] = group_subject_counts
        results['n_permutations'] = len(formatted_permutations)
        results['n_groups'] = len(formatted_permutations[0].keys())
    else:
        if verbose:
            logger.info(f"Skipped group permutations. One or more groups have less than {PERMUTATION_GROUP_COUNT} subjects")

    return results


# ===== STAGE 4: STATISTICS PREPARATION =====

def prepare_statistics_data(all_subject_results, connectivity_mappings, groups, verbose=True):
    """
    Stage 4: Prepare intra-network interhemispheric coherence data for statistical testing.

    Filters for interhemispheric pairs that share the same region AND network,
    computes spatial and spectral averaging to prepare data for PERMANOVA.

    Args:
        all_subject_results: Dict of subject_id -> process_subject results
        connectivity_mappings: Results from create_connectivity_mappings()
        anhedonia_groups: Dict with 'low-anhedonic', 'high-anhedonic', 'non-anhedonic' subject lists
        verbose: Whether to print detailed output

    Returns:
        dict: Statistical test data containing:
            - static_coherence_by_group: Observed test statistics per group (static FC)
            - slow_band_coherence_by_group: Observed test statistics per group per frequency band
    """
    if verbose:
        print(f"\n{'='*80}")
        print("STAGE 4: PREPARING STATISTICS DATA")
        print(f"{'='*80}")

    # Get intra-network pairs for statistics (same region + same network across hemispheres)
    intra_network_pairs = connectivity_mappings['intra_network_pairs']

    if not intra_network_pairs:
        if verbose:
            print("No intra-network pairs found for statistical analysis")
        return {'static_coherence_by_group': {}, 'slow_band_coherence_by_group': {}}

    if verbose:
        print(f"Found {len(intra_network_pairs)} intra-network groups:")
        for network_key, pairs in intra_network_pairs.items():
            print(f"  {network_key}: {len(pairs)} pairs")

    # Initialize results
    stats_results = {
        'static_coherence_by_group': {},
        'slow_band_coherence_by_group': {},
        'ipsi_static_coherence_by_group': {},
        'ipsi_slow_band_coherence_by_group': {},
    }

    # Process each anhedonia group
    for group_name, subject_ids in groups.items():
        if verbose:
            print(f"\nProcessing {group_name} group for statistics...")

        # Initialize group data
        stats_results['static_coherence_by_group'][group_name] = {}
        stats_results['slow_band_coherence_by_group'][group_name] = {}
        stats_results['ipsi_static_coherence_by_group'][group_name] = {}
        stats_results['ipsi_slow_band_coherence_by_group'][group_name] = {}
        stats_results['ipsi_static_coherence_by_group'][group_name] = {}
        stats_results['ipsi_slow_band_coherence_by_group'][group_name] = {}

        valid_subjects = 0

        # For each network group, collect coherence values across subjects
        for network_key, pairs in intra_network_pairs.items():
            static_coherence_values = []
            slow_band_coherence_values = {}  # Dict by band
            slow_band_coherence_subjects = {}  # Dict by band

            for subject_id in subject_ids:
                result = all_subject_results.get(subject_id)
                if not (result and result.get('success')):
                    continue

                # ===== STATIC FC ANALYSIS =====
                # Use new structured format
                fc_static_data = result.get('fc_static', {})
                z_fc_static = fc_static_data.get('z_fc')
                subject_channel_labels = result.get('channel_labels', {})

                # DEBUG: Print debug info for first subject and network (only for real anhedonia groups)
                if verbose and (subject_id == subject_ids[0] and network_key == list(intra_network_pairs.keys())[0] and
                    any(name in ['non-anhedonic', 'low-anhedonic', 'high-anhedonic'] for name in groups.keys())):
                    logger.info(f"DEBUG - Subject {subject_id}, Network {network_key}:")
                    logger.info(f"  fc_static_data keys: {list(fc_static_data.keys()) if fc_static_data else 'None'}")
                    logger.info(f"  z_fc_static shape: {z_fc_static.shape if z_fc_static is not None else 'None'}")
                    logger.info(f"  subject_channel_labels count: {len(subject_channel_labels)}")
                    logger.info(f"  pairs count: {len(pairs)}")
                    if pairs:
                        sample_pair = pairs[0]
                        logger.info(f"  sample pair: {sample_pair}")

                if z_fc_static is not None and subject_channel_labels:
                    # Create mapping from subject indices to channel labels
                    subject_idx_to_label = {idx: label for idx, label in subject_channel_labels.items()}
                    subject_label_to_idx = {label: idx for idx, label in subject_channel_labels.items()}

                    # Get Fisher-Z values for all pairs in this network
                    network_z_values = []
                    for pair in pairs:
                        # Convert from standardized indices to channel labels, then to subject indices
                        lh_label = pair['lh_label']  # Use labels instead of standardized indices
                        rh_label = pair['rh_label']

                        # Get subject-specific indices for these labels
                        if lh_label in subject_label_to_idx and rh_label in subject_label_to_idx:
                            subj_lh_idx = subject_label_to_idx[lh_label]
                            subj_rh_idx = subject_label_to_idx[rh_label]

                            # Extract correlation from Fisher-Z matrix using subject indices
                            z_value = z_fc_static[subj_lh_idx, subj_rh_idx]
                            if not np.isnan(z_value):
                                network_z_values.append(z_value)
                        elif verbose and (subject_id == subject_ids[0] and network_key == list(intra_network_pairs.keys())[0] and
                              any(name in ['non-anhedonic', 'low-anhedonic', 'high-anhedonic'] for name in groups.keys())):
                            logger.info(f"  DEBUG - Missing labels: {lh_label} in subject: {lh_label in subject_label_to_idx}, {rh_label} in subject: {rh_label in subject_label_to_idx}")
                            if lh_label not in subject_label_to_idx:
                                logger.info(f"    Available labels sample: {list(subject_label_to_idx.keys())[:5]}")

                    if network_z_values:
                        # Spatial averaging: collapse network nodes into single scalar
                        mean_network_coherence = np.mean(network_z_values)
                        static_coherence_values.append(mean_network_coherence)
                else:
                    # DEBUG: Why is static FC missing?
                    if verbose and (subject_id == subject_ids[0] and network_key == list(intra_network_pairs.keys())[0] and
                        any(name in ['non-anhedonic', 'low-anhedonic', 'high-anhedonic'] for name in groups.keys())):
                        logger.info(f"DEBUG - Static FC missing: z_fc_static={z_fc_static is not None}, channel_labels={len(subject_channel_labels)}")

                # ===== SLOW-BAND FC ANALYSIS =====
                # Use new structured format
                fc_modes_data = result.get('fc_modes', {})
                z_fc_modes = fc_modes_data.get('z_fc')
                mvmd_data = result.get('mvmd', {})
                mvmd_metadata = mvmd_data.get('metadata', {})
                center_freqs = mvmd_metadata.get('center_freqs')

                if z_fc_modes is not None and center_freqs is not None:
                    # Compute band-specific coherence for this subject
                    subject_band_coherence = compute_band_specific_coherence_from_modes(
                        z_fc_modes, center_freqs, pairs, network_key, subject_channel_labels, verbose=False
                    )

                    # Store coherence values by band
                    for band_name, coherence_value in subject_band_coherence.items():
                        if band_name not in slow_band_coherence_values:
                            slow_band_coherence_values[band_name] = []
                            slow_band_coherence_subjects[band_name] = []
                        slow_band_coherence_values[band_name].append(coherence_value)
                        slow_band_coherence_subjects[band_name].append(subject_id)


                if len(static_coherence_values) > 0 or slow_band_coherence_values:
                    valid_subjects += 1

            # Store observed test statistics for this network - STATIC
            if static_coherence_values:
                stats_results['static_coherence_by_group'][group_name][network_key] = {
                    'observed_values': static_coherence_values,
                    'n_subjects': len(static_coherence_values),
                    'mean_coherence': np.mean(static_coherence_values),
                    'std_coherence': np.std(static_coherence_values),
                }

            # Store observed test statistics for this network - SLOW-BANDS
            if slow_band_coherence_values:
                stats_results['slow_band_coherence_by_group'][group_name][network_key] = {}
                for band_name, coherence_values in slow_band_coherence_values.items():
                    if coherence_values:
                        stats_results['slow_band_coherence_by_group'][group_name][network_key][band_name] = {
                            'observed_values': coherence_values,
                            'n_subjects': len(coherence_values),
                            'mean_coherence': np.mean(coherence_values),
                            'std_coherence': np.std(coherence_values),
                            'subject_ids': slow_band_coherence_subjects[band_name]
                        }

        # IPSILATERAL COHERENCE (within hemisphere connections)
        ipsi_groups = connectivity_mappings.get('ipsilateral_pair_groups', {})
        for hemisphere, conn_groups in ipsi_groups.items():
            for conn_key, conn_data in conn_groups.items():
                static_values = []
                slow_band_values = {}
                slow_band_subjects = {}

                for subject_id in subject_ids:
                    result = all_subject_results.get(subject_id)
                    if not (result and result.get('success')):
                        continue

                    fc_static_data = result.get('fc_static', {})
                    z_fc_static = fc_static_data.get('z_fc')
                    subject_channel_labels = result.get('channel_labels', {})

                    if z_fc_static is not None and subject_channel_labels:
                        subject_label_to_idx = {label: idx for idx, label in subject_channel_labels.items()}
                        z_vals = []
                        for pair in conn_data['pairs']:
                            l1 = pair['label1']
                            l2 = pair['label2']
                            if l1 in subject_label_to_idx and l2 in subject_label_to_idx:
                                idx1 = subject_label_to_idx[l1]
                                idx2 = subject_label_to_idx[l2]
                                z_val = z_fc_static[idx1, idx2]
                                if not np.isnan(z_val):
                                    z_vals.append(z_val)
                        if z_vals:
                            static_values.append(np.mean(z_vals))

                    # Slow-band from modes
                    fc_modes_data = result.get('fc_modes', {})
                    z_fc_modes = fc_modes_data.get('z_fc')
                    mvmd_data = result.get('mvmd', {})
                    mvmd_metadata = mvmd_data.get('metadata', {})
                    center_freqs = mvmd_metadata.get('center_freqs')

                    if z_fc_modes is not None and center_freqs is not None and subject_channel_labels:
                        band_coh = compute_band_specific_coherence_from_modes_ipsi(
                            z_fc_modes, center_freqs, conn_data, subject_channel_labels, verbose=False
                        )
                        for band_name, val in band_coh.items():
                            slow_band_values.setdefault(band_name, []).append(val)
                            slow_band_subjects.setdefault(band_name, []).append(subject_id)

                if static_values:
                    stats_results['ipsi_static_coherence_by_group'][group_name][f"{conn_key}"] = {
                        'observed_values': static_values,
                        'n_subjects': len(static_values),
                        'mean_coherence': np.mean(static_values),
                        'std_coherence': np.std(static_values),
                        'hemisphere': hemisphere,
                        'regions_networks': conn_data['regions_networks']
                    }

                if slow_band_values:
                    stats_results['ipsi_slow_band_coherence_by_group'][group_name][f"{conn_key}"] = {}
                    for band_name, vals in slow_band_values.items():
                        stats_results['ipsi_slow_band_coherence_by_group'][group_name][f"{conn_key}"][band_name] = {
                            'observed_values': vals,
                            'n_subjects': len(vals),
                            'mean_coherence': np.mean(vals),
                            'std_coherence': np.std(vals),
                            'subject_ids': slow_band_subjects.get(band_name, []),
                            'hemisphere': hemisphere,
                            'regions_networks': conn_data['regions_networks']
                        }

        if verbose:
            print(f"  Processed {valid_subjects} valid subjects")
            static_networks_count = len(stats_results['static_coherence_by_group'][group_name])
            slow_band_networks_count = len(stats_results['slow_band_coherence_by_group'][group_name])
            print(f"  Generated static statistics for {static_networks_count} network groups")
            print(f"  Generated slow-band statistics for {slow_band_networks_count} network groups")

    if verbose:
        print(f"\nStatistics preparation completed:")
        for group_name in groups.keys():
            networks_count = len(stats_results['static_coherence_by_group'].get(group_name, {}))
            print(f"  {group_name}: {networks_count} network groups prepared")

    return stats_results

# ===== HELPER FUNCTIONS FOR STATISTICAL TESTING =====

def _extract_group_data_for_anova(stat_results_by_group, network_key, band_name=None):
    """
    Extract group data for ANOVA testing from prepared statistics structure.

    Args:
        stat_results_by_group: Group statistics from prepare_statistics_data()
        network_key: Network identifier (e.g., "PFCm_DefaultA")
        band_name: Optional frequency band name for slow-band analysis

    Returns:
        tuple: (group_data_dict, all_values_list)
            - group_data_dict: {group_name: [observed_values]}
            - all_values_list: flattened list of all values for overall tests
    """
    group_data = {}
    all_values = []

    for group_name, group_networks in stat_results_by_group.items():
        if network_key not in group_networks:
            continue

        if band_name is None:
            # Static FC analysis
            network_data = group_networks[network_key]
            values = network_data.get('observed_values', [])
        else:
            # Slow-band FC analysis
            if band_name not in group_networks[network_key]:
                continue
            band_data = group_networks[network_key][band_name]
            values = band_data.get('observed_values', [])

        if values:
            group_data[group_name] = values
            all_values.extend(values)

    return group_data, all_values


def _perform_levene_test(group_data, verbose=False):
    """
    Perform Levene test for homogeneity of variance across groups.

    Args:
        group_data: Dict of {group_name: [values]} from _extract_group_data_for_anova()
        verbose: Whether to print test details

    Returns:
        dict: Levene test results with statistic, p-value, and assumption met flag
    """
    if len(group_data) < 2:
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'homogeneity_assumption_met': False,
            'note': 'Insufficient groups for variance testing'
        }

    group_values = list(group_data.values())

    # Check if all groups have sufficient data
    min_group_size = min(len(vals) for vals in group_values)
    if min_group_size < 2:
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'homogeneity_assumption_met': False,
            'note': 'Insufficient subjects in one or more groups'
        }

    try:
        # Perform Levene test (using median, more robust than mean)
        levene_stat, levene_p = stats.levene(*group_values, center='median')

        # Conventional alpha = 0.05 for variance homogeneity
        assumption_met = levene_p > 0.05

        if verbose:
            print(f"      Levene test: W = {levene_stat:.4f}, p = {levene_p:.4f} "
                  f"({'assumption met' if assumption_met else 'assumption violated'})")

        return {
            'statistic': levene_stat,
            'p_value': levene_p,
            'homogeneity_assumption_met': assumption_met,
            'note': 'Test completed successfully'
        }

    except Exception as e:
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'homogeneity_assumption_met': False,
            'note': f'Levene test failed: {str(e)}'
        }


def _perform_anova_or_welch(group_data, homogeneity_met, verbose=False):
    """
    Perform appropriate ANOVA test based on variance homogeneity.

    Args:
        group_data: Dict of {group_name: [values]} from _extract_group_data_for_anova()
        homogeneity_met: Boolean from Levene test results
        verbose: Whether to print test details

    Returns:
        dict: ANOVA test results with test type, statistics, and degrees of freedom
    """
    if len(group_data) < 2:
        return {
            'test_type': 'none',
            'F_statistic': np.nan,
            'p_value': np.nan,
            'df_between': np.nan,
            'df_within': np.nan,
            'note': 'Insufficient groups for ANOVA'
        }

    group_values = list(group_data.values())
    group_names = list(group_data.keys())

    # Calculate degrees of freedom
    n_groups = len(group_values)
    group_sizes = [len(vals) for vals in group_values]
    total_n = sum(group_sizes)
    df_between = n_groups - 1
    df_within = total_n - n_groups

    if df_within <= 0:
        return {
            'test_type': 'none',
            'F_statistic': np.nan,
            'p_value': np.nan,
            'df_between': df_between,
            'df_within': df_within,
            'note': 'Insufficient degrees of freedom'
        }

    try:
        if homogeneity_met:
            # Standard one-way ANOVA (assumes equal variances)
            f_stat, p_value = stats.f_oneway(*group_values)
            test_type = 'standard_anova'

            if verbose:
                print(f"      Standard ANOVA: F({df_between},{df_within}) = {f_stat:.4f}, p = {p_value:.4f}")

        else:
            # Welch ANOVA (does not assume equal variances)
            # NOTE: scipy doesn't have built-in Welch ANOVA, so we implement it
            f_stat, p_value = _welch_anova_new(group_values)
            test_type = 'welch_anova'

            if verbose:
                print(f"      Welch ANOVA: F = {f_stat:.4f}, p = {p_value:.4f} (unequal variances)")

        return {
            'test_type': test_type,
            'F_statistic': f_stat,
            'p_value': p_value,
            'df_between': df_between,
            'df_within': df_within,
            'note': 'Test completed successfully'
        }

    except Exception as e:
        return {
            'test_type': 'failed',
            'F_statistic': np.nan,
            'p_value': np.nan,
            'df_between': df_between,
            'df_within': df_within,
            'note': f'ANOVA failed: {str(e)}'
        }


def _welch_anova(group_values):
    """
    Perform Welch ANOVA (one-way ANOVA without equal variance assumption).

    Args:
        group_values: List of arrays, one per group

    Returns:
        tuple: (F_statistic, p_value)
    """
    # Calculate group statistics
    group_means = [np.mean(vals) for vals in group_values]
    # Calculate sample variance with warning suppression for edge cases
    import warnings
    group_vars = []
    for vals in group_values:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            var = np.var(vals, ddof=1)
            group_vars.append(var if not np.isnan(var) else 0.0)
    group_ns = [len(vals) for vals in group_values]

    # Check for zero or near-zero variances (problematic for Welch ANOVA)
    min_var_threshold = 1e-10
    zero_var_groups = [i for i, var in enumerate(group_vars) if var < min_var_threshold]
    if zero_var_groups:
        # If any group has zero variance, Welch ANOVA is not appropriate
        return np.nan, np.nan

    # Check for sufficient sample sizes
    if any(n <= 1 for n in group_ns):
        return np.nan, np.nan

    # Weighted mean
    try:
        weights = [n/var for n, var in zip(group_ns, group_vars)]
        total_weight = sum(weights)
        if total_weight == 0:
            return np.nan, np.nan
        weighted_mean = sum(w*mean for w, mean in zip(weights, group_means)) / total_weight
    except (ZeroDivisionError, ValueError):
        return np.nan, np.nan

    # Welch F-statistic
    numerator = sum(w * (mean - weighted_mean)**2 for w, mean in zip(weights, group_means))
    k = len(group_values)  # number of groups

    # Denominator correction factor
    denominator = 2 * (k - 2) / (k**2 - 1) * sum((1 - w/sum(weights))**2 / (n - 1)
                                                  for w, n in zip(weights, group_ns))

    if denominator == 0:
        return np.nan, np.nan

    f_statistic = numerator / (k - 1) / (1 + denominator)

    # Approximate degrees of freedom for p-value calculation
    df1 = k - 1
    df2 = (k**2 - 1) / (3 * sum((1 - w/sum(weights))**2 / (n - 1)
                                for w, n in zip(weights, group_ns)))

    # Calculate p-value
    p_value = 1 - stats.f.cdf(f_statistic, df1, df2)

    return f_statistic, p_value

def _welch_anova_new(group_values):
    """
    Perform Welch ANOVA, gracefully handling empty groups or groups with n=1.
    """
    # 1. Filter out invalid groups (must have at least 2 points for variance)
    # This prevents the RuntimeWarning entirely
    valid_groups = [np.asarray(g) for g in group_values if g is not None and len(g) > 1]

    # 2. Check if we have enough groups left to compare (k >= 2)
    k = len(valid_groups)
    if k < 2:
        return np.nan, np.nan

    # 3. Calculate group statistics
    group_ns = np.array([len(g) for g in valid_groups])
    group_means = np.array([np.mean(g) for g in valid_groups])
    group_vars = np.array([np.var(g, ddof=1) for g in valid_groups])

    # 4. Check for zero variance across the remaining groups
    # Welch's ANOVA fails if variance is 0 because it's used in the denominator
    min_var_threshold = 1e-10
    if np.any(group_vars < min_var_threshold):
        return np.nan, np.nan

    # 5. Calculate Weights (w = n / var)
    weights = group_ns / group_vars
    sum_w = np.sum(weights)

    # 6. Weighted mean and Welch components
    weighted_mean = np.sum(weights * group_means) / sum_w

    # Numerator of the F-statistic
    numerator = np.sum(weights * (group_means - weighted_mean)**2) / (k - 1)

    # Denominator adjustment (Aspin-Welch)
    lambdas = (1 - weights / sum_w)**2 / (group_ns - 1)
    sum_lambdas = np.sum(lambdas)

    v_adj = (2 * (k - 2) / (k**2 - 1)) * sum_lambdas
    f_statistic = numerator / (1 + v_adj)

    # 7. Degrees of Freedom
    df1 = k - 1
    df2 = (k**2 - 1) / (3 * sum_lambdas)

    # 8. P-value using the survival function (sf)
    p_value = stats.f.sf(f_statistic, df1, df2)

    return f_statistic, p_value

def _compute_group_summary_stats(group_data):
    """
    Compute summary statistics for each group.

    Args:
        group_data: Dict of {group_name: [values]}

    Returns:
        dict: Summary statistics by group
    """
    summary = {
        'group_sizes': {},
        'group_means': {},
        'group_stds': {},
        'group_counts': {}
    }

    for group_name, values in group_data.items():
        if values:
            summary['group_sizes'][group_name] = len(values)
            summary['group_means'][group_name] = np.mean(values)
            summary['group_stds'][group_name] = np.std(values, ddof=1)  # Sample std
            summary['group_counts'][group_name] = len(values)
        else:
            summary['group_sizes'][group_name] = 0
            summary['group_means'][group_name] = np.nan
            summary['group_stds'][group_name] = np.nan
            summary['group_counts'][group_name] = 0

    return summary


# ===== WELCH ANOVA AND GAMES-HOWELL HELPER FUNCTIONS =====
#
# STATISTICAL TESTING APPROACH IMPLEMENTED:
# - Welch one-way ANOVA as default omnibus test (robust to unequal group sizes/heteroscedasticity)
# - Games-Howell post-hoc tests for significant omnibus results (uncorrected p < 0.05)
# - FDR correction applied to omnibus ANOVA p-values per band (not to post-hoc p-values)
# - Fisher-Z transformed FC values used for all statistical inference
# - Post-hoc results saved to CSV files for interpretation
# - TODO: Covariate analysis (age, sex, education, motion/FD) via GLM/ANCOVA framework
#

def run_welch_anova(z_values_by_group):
    """
    Perform Welch one-way ANOVA on Fisher-Z transformed FC values.

    Args:
        z_values_by_group: Dict mapping group_name -> list[float] for a single ROI-ROI connection

    Returns:
        float: Omnibus p-value from Welch ANOVA (uncorrected)
    """
    # Build DataFrame for pingouin
    data_rows = []
    for group_name, z_values in z_values_by_group.items():
        for z_val in z_values:
            data_rows.append({'fc': z_val, 'group': group_name})

    if len(data_rows) == 0:
        return np.nan

    df = pd.DataFrame(data_rows)

    # Perform Welch ANOVA
    try:
        anova_result = pg.welch_anova(dv='fc', between='group', data=df)
        return anova_result['p-unc'].iloc[0]
    except Exception as e:
        logging.warning(f"Welch ANOVA failed: {e}")
        return np.nan


def run_games_howell(z_values_by_group):
    """
    Perform Games-Howell post-hoc test on Fisher-Z transformed FC values.

    Args:
        z_values_by_group: Dict mapping group_name -> list[float] for a single ROI-ROI connection

    Returns:
        pandas.DataFrame: Games-Howell post-hoc results
    """
    # Build DataFrame for pingouin
    data_rows = []
    for group_name, z_values in z_values_by_group.items():
        for z_val in z_values:
            data_rows.append({'fc': z_val, 'group': group_name})

    if len(data_rows) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(data_rows)

    # Perform Games-Howell post-hoc test
    try:
        posthoc_result = pg.pairwise_gameshowell(dv='fc', between='group', data=df)
        return posthoc_result
    except Exception as e:
        logging.warning(f"Games-Howell post-hoc failed: {e}")
        return pd.DataFrame()


# ===== MAIN STATISTICAL TESTING FUNCTION =====

def coherence_anova_test(stat_results, verbose=True):
    """
    Stage 4: Welch one-way ANOVA analysis with Games-Howell post-hoc for interhemispheric
    intra-network connectivity.

    Statistical Logic (applied per network AND per connection):
    1. Extract Fisher z-transformed FC values per subject by group
    2. Perform Welch one-way ANOVA (robust to unequal group sizes and heteroscedasticity)
    3. Apply FDR correction to omnibus ANOVA p-values per band
    4. Run Games-Howell post-hoc tests for significant omnibus results (uncorrected p < 0.05)

    TODO: Load covariates (age, sex, education, motion/FD) and include them in a GLM/ANCOVA framework.
    TODO: If covariates are added, switch from Welch ANOVA to an appropriate regression/GLM with
          robust (HC3) standard errors or permutation tests.

    Args:
        stat_results: Output from prepare_statistics_data() containing:
            - static_coherence_by_group: Static FC statistics by group
            - slow_band_coherence_by_group: Slow-band FC statistics by group
        verbose: Whether to print detailed progress information

    Returns:
        dict: Complete statistical results structure:
            - static_results: {network_key: {omnibus_p, fdr_corrected_p, post_hoc_results}}
            - slow_band_results: {network_key: {band_name: {omnibus_p, fdr_corrected_p, post_hoc_results}}}
            - post_hoc_collection: List of all Games-Howell results for significant tests
    """

    if verbose:
        print(f"\n{'='*80}")
        print("STAGE 4: WELCH ANOVA + GAMES-HOWELL TESTING")
        print(f"{'='*80}")

    results = {
        'static_results': {},
        'slow_band_results': {},
        'ipsi_static_results': {},
        'ipsi_slow_band_results': {},
        'post_hoc_collection': []
    }

    static_groups = stat_results.get('static_coherence_by_group', {})
    slow_band_groups = stat_results.get('slow_band_coherence_by_group', {})
    ipsi_static_groups = stat_results.get('ipsi_static_coherence_by_group', {})
    ipsi_slow_groups = stat_results.get('ipsi_slow_band_coherence_by_group', {})

    # ===== STATIC FC WELCH ANOVA TESTING =====
    if static_groups:
        if verbose:
            print(f"\nTesting static FC across {len(static_groups)} groups...")

        # Find all network keys across groups
        all_network_keys = set()
        for group_data in static_groups.values():
            all_network_keys.update(group_data.keys())

        # Process each network
        omnibus_p_values = []
        network_omnibus_results = {}

        for network_key in all_network_keys:
            if verbose:
                print(f"  Processing network: {network_key}")

            # Extract group data for this network
            group_data, all_values = _extract_group_data_for_anova(
                static_groups, network_key, band_name=None
            )

            if len(group_data) < 2:
                if verbose:
                    print(f"    Skipping: insufficient groups with data")
                continue

            # Log group sizes
            group_sizes = {name: len(values) for name, values in group_data.items()}
            group_size_str = ", ".join([f"{name}={size}" for name, size in group_sizes.items()])
            logging.info(f"Network: {network_key}")
            logging.info(f"Groups: {group_size_str}")
            logging.info(f"Running Welch ANOVA on static FC connectivity")

            # Perform Welch ANOVA
            omnibus_p = run_welch_anova(group_data)
            omnibus_p_values.append(omnibus_p)

            # Store omnibus result
            network_omnibus_results[network_key] = {
                'omnibus_p': omnibus_p,
                'group_sizes': group_sizes,
                'post_hoc_results': None
            }

            # Run post-hoc if omnibus is significant (uncorrected)
            if not np.isnan(omnibus_p) and omnibus_p < 0.05:
                post_hoc_df = run_games_howell(group_data)
                if not post_hoc_df.empty:
                    network_omnibus_results[network_key]['post_hoc_results'] = post_hoc_df
                    results['post_hoc_collection'].append({
                        'network': network_key,
                        'band': 'static',
                        'post_hoc': post_hoc_df
                    })

            if verbose:
                total_subjects = sum(group_sizes.values())
                significant = omnibus_p < 0.05 if not np.isnan(omnibus_p) else False
                print(f"    Result: {total_subjects} subjects, "
                      f"p = {omnibus_p:.4f} "
                      f"{'***' if omnibus_p < 0.001 else '**' if omnibus_p < 0.01 else '*' if significant else 'ns'}")

        # Apply FDR correction to static FC omnibus p-values
        if omnibus_p_values:
            valid_p_mask = ~np.isnan(omnibus_p_values)
            if valid_p_mask.sum() > 0:
                fdr_rejected, fdr_corrected_p = fdrcorrection(
                    np.array(omnibus_p_values)[valid_p_mask]
                )

                # Map corrected p-values back to networks
                fdr_idx = 0
                for network_key in network_omnibus_results:
                    if not np.isnan(network_omnibus_results[network_key]['omnibus_p']):
                        network_omnibus_results[network_key]['fdr_corrected_p'] = fdr_corrected_p[fdr_idx]
                        network_omnibus_results[network_key]['fdr_significant'] = fdr_rejected[fdr_idx]
                        fdr_idx += 1
                    else:
                        network_omnibus_results[network_key]['fdr_corrected_p'] = np.nan
                        network_omnibus_results[network_key]['fdr_significant'] = False

        results['static_results'] = network_omnibus_results

    # ===== SLOW-BAND FC WELCH ANOVA TESTING =====
    if slow_band_groups:
        if verbose:
            print(f"\nTesting slow-band FC across {len(slow_band_groups)} groups...")

        # Find all network keys and band combinations
        all_network_keys = set()
        all_band_names = set()
        for group_data in slow_band_groups.values():
            for network_key, network_bands in group_data.items():
                all_network_keys.add(network_key)
                all_band_names.update(network_bands.keys())

        # Process each band separately for FDR correction
        for band_name in all_band_names:
            if verbose:
                print(f"\n  Processing band: {band_name}")

            # Log band processing
            logging.info(f"Band: {band_name}")

            omnibus_p_values_band = []
            band_omnibus_results = {}

            for network_key in all_network_keys:
                if verbose:
                    print(f"    Processing network: {network_key}")

                # Extract group data for this network and band
                group_data, all_values = _extract_group_data_for_anova(
                    slow_band_groups, network_key, band_name=band_name
                )

                if len(group_data) < 2:
                    if verbose:
                        print(f"      Skipping: insufficient groups with data")
                    continue

                # Log group sizes
                group_sizes = {name: len(values) for name, values in group_data.items()}
                group_size_str = ", ".join([f"{name}={size}" for name, size in group_sizes.items()])
                logging.info(f"Groups: {group_size_str}")
                logging.info(f"Running Welch ANOVA on ROI–ROI connectivity")

                # Perform Welch ANOVA
                omnibus_p = run_welch_anova(group_data)
                omnibus_p_values_band.append(omnibus_p)

                # Store omnibus result
                band_omnibus_results[network_key] = {
                    'omnibus_p': omnibus_p,
                    'group_sizes': group_sizes,
                    'post_hoc_results': None
                }

                # Run post-hoc if omnibus is significant (uncorrected)
                if not np.isnan(omnibus_p) and omnibus_p < 0.05:
                    post_hoc_df = run_games_howell(group_data)
                    if not post_hoc_df.empty:
                        band_omnibus_results[network_key]['post_hoc_results'] = post_hoc_df
                        results['post_hoc_collection'].append({
                            'network': network_key,
                            'band': band_name,
                            'post_hoc': post_hoc_df
                        })

                if verbose:
                    total_subjects = sum(group_sizes.values())
                    significant = omnibus_p < 0.05 if not np.isnan(omnibus_p) else False
                    print(f"      Result: {total_subjects} subjects, "
                          f"p = {omnibus_p:.4f} "
                          f"{'***' if omnibus_p < 0.001 else '**' if omnibus_p < 0.01 else '*' if significant else 'ns'}")

            # Apply FDR correction to this band's omnibus p-values
            if omnibus_p_values_band:
                valid_p_mask = ~np.isnan(omnibus_p_values_band)
                if valid_p_mask.sum() > 0:
                    fdr_rejected, fdr_corrected_p = fdrcorrection(
                        np.array(omnibus_p_values_band)[valid_p_mask]
                    )

                    # Map corrected p-values back to networks
                    fdr_idx = 0
                    for network_key in band_omnibus_results:
                        if not np.isnan(band_omnibus_results[network_key]['omnibus_p']):
                            band_omnibus_results[network_key]['fdr_corrected_p'] = fdr_corrected_p[fdr_idx]
                            band_omnibus_results[network_key]['fdr_significant'] = fdr_rejected[fdr_idx]
                            fdr_idx += 1
                        else:
                            band_omnibus_results[network_key]['fdr_corrected_p'] = np.nan
                            band_omnibus_results[network_key]['fdr_significant'] = False

            # Initialize slow_band_results structure if needed
            if band_name not in results['slow_band_results']:
                results['slow_band_results'][band_name] = {}

            results['slow_band_results'][band_name] = band_omnibus_results

    # ===== IPSILATERAL STATIC FC WELCH ANOVA =====
    if ipsi_static_groups:
        if verbose:
            print(f"\nTesting IPSILATERAL static FC across {len(ipsi_static_groups)} groups...")

        all_conn_keys = set()
        for group_data in ipsi_static_groups.values():
            all_conn_keys.update(group_data.keys())

        omnibus_p_values_ipsi = []
        ipsi_static_results = {}

        for conn_key in all_conn_keys:
            group_data = {}
            for group_name, group_conn in ipsi_static_groups.items():
                if conn_key in group_conn:
                    vals = group_conn[conn_key]['observed_values']
                    if len(vals) > 0:
                        group_data[group_name] = vals

            if len(group_data) < 2:
                continue

            group_sizes = {name: len(values) for name, values in group_data.items()}
            logging.info(f"IPSILATERAL Connection: {conn_key}")
            logging.info(f"Groups: {group_sizes}")
            logging.info(f"Running Welch ANOVA on ipsilateral FC connectivity")

            omnibus_p = run_welch_anova(group_data)
            omnibus_p_values_ipsi.append(omnibus_p)

            ipsi_static_results[conn_key] = {
                'omnibus_p': omnibus_p,
                'group_sizes': group_sizes,
                'post_hoc_results': None
            }

            if not np.isnan(omnibus_p) and omnibus_p < 0.05:
                post_hoc_df = run_games_howell(group_data)
                if not post_hoc_df.empty:
                    ipsi_static_results[conn_key]['post_hoc_results'] = post_hoc_df
                    results['post_hoc_collection'].append({
                        'network': conn_key,
                        'band': 'static_ipsi',
                        'post_hoc': post_hoc_df
                    })

            if verbose:
                total_subjects = sum(group_sizes.values())
                significant = omnibus_p < 0.05 if not np.isnan(omnibus_p) else False
                print(f"    Result: {total_subjects} subjects, "
                      f"p = {omnibus_p:.4f} "
                      f"{'***' if omnibus_p < 0.001 else '**' if omnibus_p < 0.01 else '*' if significant else 'ns'}")

        # Apply FDR correction
        if omnibus_p_values_ipsi:
            valid_p_mask = ~np.isnan(omnibus_p_values_ipsi)
            if valid_p_mask.sum() > 0:
                fdr_rejected, fdr_corrected_p = fdrcorrection(
                    np.array(omnibus_p_values_ipsi)[valid_p_mask]
                )
                fdr_idx = 0
                for conn_key in ipsi_static_results:
                    if not np.isnan(ipsi_static_results[conn_key]['omnibus_p']):
                        ipsi_static_results[conn_key]['fdr_corrected_p'] = fdr_corrected_p[fdr_idx]
                        ipsi_static_results[conn_key]['fdr_significant'] = fdr_rejected[fdr_idx]
                        fdr_idx += 1
                    else:
                        ipsi_static_results[conn_key]['fdr_corrected_p'] = np.nan
                        ipsi_static_results[conn_key]['fdr_significant'] = False

        results['ipsi_static_results'] = ipsi_static_results

    # ===== IPSILATERAL SLOW-BAND WELCH ANOVA =====
    if ipsi_slow_groups:
        if verbose:
            print(f"\nTesting IPSILATERAL slow-band FC across {len(ipsi_slow_groups)} groups...")

        all_conn_keys = set()
        all_band_names = set()
        for group_data in ipsi_slow_groups.values():
            for conn_key, bands in group_data.items():
                all_conn_keys.add(conn_key)
                all_band_names.update(bands.keys())

        ipsi_slow_results = {}

        for band_name in all_band_names:
            omnibus_p_values_band = []
            band_results = {}

            for conn_key in all_conn_keys:
                group_data = {}
                for group_name, group_conn in ipsi_slow_groups.items():
                    if conn_key in group_conn and band_name in group_conn[conn_key]:
                        vals = group_conn[conn_key][band_name]['observed_values']
                        if len(vals) > 0:
                            group_data[group_name] = vals

                if len(group_data) < 2:
                    continue

                group_sizes = {name: len(values) for name, values in group_data.items()}
                logging.info(f"IPSILATERAL Band: {band_name}, Connection: {conn_key}")
                logging.info(f"Groups: {group_sizes}")
                logging.info(f"Running Welch ANOVA on ipsilateral slow-band FC")

                omnibus_p = run_welch_anova(group_data)
                omnibus_p_values_band.append(omnibus_p)

                band_results[conn_key] = {
                    'omnibus_p': omnibus_p,
                    'group_sizes': group_sizes,
                    'post_hoc_results': None
                }

                if not np.isnan(omnibus_p) and omnibus_p < 0.05:
                    post_hoc_df = run_games_howell(group_data)
                    if not post_hoc_df.empty:
                        band_results[conn_key]['post_hoc_results'] = post_hoc_df
                        results['post_hoc_collection'].append({
                            'network': conn_key,
                            'band': f'{band_name}_ipsi',
                            'post_hoc': post_hoc_df
                        })

                if verbose:
                    total_subjects = sum(group_sizes.values())
                    significant = omnibus_p < 0.05 if not np.isnan(omnibus_p) else False
                    print(f"    Result: {total_subjects} subjects, "
                          f"p = {omnibus_p:.4f} "
                          f"{'***' if omnibus_p < 0.001 else '**' if omnibus_p < 0.01 else '*' if significant else 'ns'}")

            # FDR correction for this band
            if omnibus_p_values_band:
                valid_p_mask = ~np.isnan(omnibus_p_values_band)
                if valid_p_mask.sum() > 0:
                    fdr_rejected, fdr_corrected_p = fdrcorrection(
                        np.array(omnibus_p_values_band)[valid_p_mask]
                    )
                    fdr_idx = 0
                    for conn_key in band_results:
                        if not np.isnan(band_results[conn_key]['omnibus_p']):
                            band_results[conn_key]['fdr_corrected_p'] = fdr_corrected_p[fdr_idx]
                            band_results[conn_key]['fdr_significant'] = fdr_rejected[fdr_idx]
                            fdr_idx += 1
                        else:
                            band_results[conn_key]['fdr_corrected_p'] = np.nan
                            band_results[conn_key]['fdr_significant'] = False

            results['ipsi_slow_band_results'][band_name] = band_results

    if verbose:
        static_networks = len(results['static_results'])
        slow_band_tests = sum(len(band_data) for band_data in results['slow_band_results'].values())
        ipsi_static_networks = len(results['ipsi_static_results'])
        ipsi_slow_tests = sum(len(band_data) for band_data in results['ipsi_slow_band_results'].values())
        post_hoc_count = len(results['post_hoc_collection'])
        print(f"\nWelch ANOVA testing completed:")
        print(f"  Static FC: {static_networks} networks tested")
        print(f"  Slow-band FC: {slow_band_tests} network-band combinations tested")
        print(f"  Ipsilateral static FC: {ipsi_static_networks} connections tested")
        print(f"  Ipsilateral slow-band FC: {ipsi_slow_tests} connection-band combinations tested")
        print(f"  Games-Howell post-hoc tests: {post_hoc_count} performed")

    return results


# ===== STAGE 5: VISUALIZATION HELPERS =====

def prepare_plotting_data(all_subject_results, verbose=True):
    """
    Stage 5: Prepare data structures for plotting by converting Z-values to R-values.

    Takes the pure computation results from the pipeline and converts them into
    the format expected by the existing plotting functions.

    Args:
        all_subject_results: Dict of subject_id -> process_subject results
        verbose: Whether to print detailed output

    Returns:
        dict: Plotting-ready data with both individual and group results
    """
    plotting_data = {}

    for subject_id, result in all_subject_results.items():
        if not result.get('success'):
            continue

        # Convert individual subject Z-matrices to R-matrices for plotting
        subject_plotting_data = {}

        # Prepare channel labels once (used across static and mode-level plots)
        channel_labels = result.get('channel_labels', {})
        labels_list = []
        if channel_labels:
            # Convert index-based labels to list
            labels_list = [channel_labels.get(i, f'Ch{i+1}') for i in range(len(channel_labels))]
        subject_plotting_data['static_fc_labels'] = labels_list

        # Static FC (new pipeline structure)
        fc_static = result.get('fc_static')
        if fc_static and fc_static.get('r_fc') is not None:
            # Prefer labels from fc_static if available
            if not labels_list and fc_static.get('labels'):
                labels_list = list(fc_static['labels'])
                subject_plotting_data['static_fc_labels'] = labels_list

            r_fc_static = fc_static['r_fc']
            subject_plotting_data['static_fc_matrix'] = r_fc_static

            # Add p-values and compute connectivity patterns for plotting
            static_fc_pvalues = fc_static.get('p_values')
            subject_plotting_data['static_fc_pvalues'] = static_fc_pvalues
            subject_plotting_data['static_connectivity_patterns'] = analyze_connectivity_patterns(
                r_fc_static, labels_list, p_values=static_fc_pvalues
            )

        # Mode-wise FC (new pipeline structure)
        fc_modes = result.get('fc_modes')
        if fc_modes and fc_modes.get('r_fc') is not None:
            r_fc_modes = fc_modes['r_fc']
            subject_plotting_data['r_fc_modes'] = r_fc_modes

            # Aggregate modes into slow-bands for plotting
            if result.get('mvmd_metadata', {}).get('center_freqs') is not None:
                center_freqs = result['mvmd_metadata']['center_freqs']
                subject_plotting_data['slow_band_fc_modes'] = aggregate_modes_to_slow_bands_for_plotting(
                    fc_modes.get('z_fc'), center_freqs, mode_fc_pvalues=result.get('mode_fc_pvalues'), verbose=False
                )
                subject_plotting_data['center_freqs'] = center_freqs

        plotting_data[subject_id] = subject_plotting_data

        if verbose and subject_plotting_data:
            print(f"  Prepared plotting data for {subject_id}")

    if verbose:
        subjects_count = len([p for p in plotting_data.values() if 'static_fc_matrix' in p])
        print(f"\nVisualization data prepared for {subjects_count} subjects")

    return plotting_data


def aggregate_modes_to_slow_bands_for_plotting(z_fc_modes, center_freqs, mode_fc_pvalues=None, verbose=False):
    """
    Aggregate mode-level Z-transformed FC matrices into slow-band FC matrices for individual subject plotting.

    Args:
        z_fc_modes: Z-transformed FC matrices per mode (n_modes, n_channels, n_channels)
        center_freqs: Center frequencies for each mode
        mode_fc_pvalues: Optional p-values per mode (n_modes, n_channels, n_channels)
        verbose: Whether to print verbose output

    Returns:
        dict: {band_name: r_fc_matrix} - R-transformed matrices for plotting
    """

    # Group modes by frequency bands
    band_mode_groups = {}
    for mode_idx, freq in enumerate(center_freqs):
        band_num = get_band_number(freq)
        band_name = f'slow-{band_num}'

        if band_name not in band_mode_groups:
            band_mode_groups[band_name] = {'indices': [], 'frequencies': []}

        band_mode_groups[band_name]['indices'].append(mode_idx)
        band_mode_groups[band_name]['frequencies'].append(freq)

    # Aggregate modes within each band
    slow_band_fc = {}
    for band_name, band_data in band_mode_groups.items():
        mode_indices = band_data['indices']

        if len(mode_indices) == 0:
            continue

        # Average Z-matrices across modes in this band
        band_z_matrices = z_fc_modes[mode_indices]
        avg_band_z_matrix = np.nanmean(band_z_matrices, axis=0)

        band_pvalues = None
        if mode_fc_pvalues is not None:
            # Conservative combine: take the minimum p-value across contributing modes
            band_mode_pvalues = mode_fc_pvalues[mode_indices]
            band_pvalues = np.nanmin(band_mode_pvalues, axis=0)

        # Convert to R-scale for plotting
        avg_band_r_matrix = fisher_z_to_r(avg_band_z_matrix)

        slow_band_fc[band_name] = {
            'fc_matrix': avg_band_r_matrix,
            'n_modes': len(mode_indices),
            'frequencies': band_data['frequencies'],
            'fc_pvalues': band_pvalues
        }

        if verbose:
            print(f"  {band_name}: {len(mode_indices)} modes aggregated")

    return slow_band_fc


# def aggregate_cross_subject_slow_bands_by_groups(all_subject_results, anhedonia_groups, verbose=False):
#     """
#     Aggregate Z-transformed FC matrices across subjects within each anhedonia group for modes within same slow-band frequency ranges.

#     Args:
#         all_subject_results: Dict of subject results containing z_fc_modes and center_freqs
#         anhedonia_groups: Dict with 'low-anhedonic', 'high-anhedonic', 'non-anhedonic' subject lists
#         verbose: Whether to print verbose output

#     Returns:
#         dict: {group_name: {band_name: data}} - Group-specific aggregated R-matrices for plotting
#     """
#     if verbose:
#         print(f"\nAggregating FC matrices by slow-bands within anhedonia groups...")

#     group_slow_bands = {}

#     # Process each anhedonia group separately
#     for group_name, subject_ids in anhedonia_groups.items():
#         if verbose:
#             print(f"\nProcessing {group_name} group ({len(subject_ids)} subjects)...")

#         # Collect all Z-matrices by slow-band for this group
#         band_z_matrices = {}  # {band_name: [list of Z matrices]}

#         for subject_id in subject_ids:
#             result = all_subject_results.get(subject_id)
#             if not (result and result.get('success')):
#                 continue

#             z_fc_modes = result.get('z_fc_modes')
#             center_freqs = result.get('mvmd_metadata', {}).get('center_freqs')

#             if z_fc_modes is None or center_freqs is None:
#                 continue

#             # Group this subject's modes by bands
#             for mode_idx, freq in enumerate(center_freqs):
#                 band_num = get_band_number(freq)
#                 band_name = f'slow-{band_num}'

#                 if band_name not in band_z_matrices:
#                     band_z_matrices[band_name] = []

#                 band_z_matrices[band_name].append(z_fc_modes[mode_idx])

#         # Aggregate within this group and convert to R-scale
#         group_slow_bands[group_name] = {}
#         for band_name, z_matrices_list in band_z_matrices.items():
#             if len(z_matrices_list) == 0:
#                 continue

#             # Stack and average all Z-matrices for this band within this group
#             stacked_z = np.stack(z_matrices_list, axis=0)
#             avg_z_matrix = np.nanmean(stacked_z, axis=0)

#             # Convert to R-scale for plotting
#             avg_r_matrix = fisher_z_to_r(avg_z_matrix)

#             group_slow_bands[group_name][band_name] = {
#                 'fc_matrix': avg_r_matrix,
#                 'n_matrices': len(z_matrices_list),
#                 'description': f'{group_name} group average ({len(z_matrices_list)} mode matrices)'
#             }

#             if verbose:
#                 print(f"  {band_name}: {len(z_matrices_list)} mode matrices aggregated for {group_name}")

#     return group_slow_bands


def main(mask_diagonal=False, mask_nonsignificant=False, create_plots=True, show_plots=True, save_figures=False, verbose=True, subjects_per_group=None):
    """Main function for FC MVP analysis"""
    from datetime import datetime

    print("[Init] Functional Connectivity MVP")

    # Create timestamped parent folder for this analysis run only if saving figures
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if save_figures:
        run_parent_dir = get_analysis_path(f'analysis_runs/run_{run_timestamp}')
        run_parent_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Init] Analysis run directory: {run_parent_dir}")
        print(f"[Init] Timestamp: {run_timestamp}")
        print(f"[Init] All outputs for this run will be saved in this directory")
    else:
        run_parent_dir = None
        print(f"[Init] Running in no-save mode - figures will not be saved to disk")

    # ===== CONFIGURATION FOR MULTI-SUBJECT ANALYSIS =====
    LIMIT_SUBJECTS = subjects_per_group is not None  # Enable limiting if subjects_per_group is specified
    MAX_SUBJECTS_PER_GROUP = subjects_per_group if subjects_per_group is not None else 8  # Use specified limit or default

    print(f"[Config] Subject limiting: {'ENABLED' if LIMIT_SUBJECTS else 'DISABLED'}")
    if LIMIT_SUBJECTS:
        print(f"[Config] Max subjects per group: {MAX_SUBJECTS_PER_GROUP}")
    print(f"[Config] Individual plots: {'ENABLED' if show_plots else 'DISABLED'}")
    print(f"[Config] Verbose output: {'ENABLED' if verbose else 'DISABLED'}")

    # Configure logging for statistical testing and log statistical approach
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format='[STATS] %(message)s'
    )
    # Suppress precision-loss warnings from scipy/pingouin when groups have nearly identical values
    warnings.filterwarnings(
        'ignore',
        message='Precision loss occurred in moment calculation due to catastrophic cancellation',
        category=RuntimeWarning
    )
    logging.info("Statistical Analysis Configuration:")
    logging.info("Omnibus test: Welch one-way ANOVA (default)")
    logging.info("Post-hoc test: Games–Howell")
    logging.info("Multiple comparisons: FDR correction applied to omnibus p-values only")
    logging.info("Rationale: robustness to unequal group sizes and heteroscedasticity")
    print()

    # Initialize data infrastructure
    loader = DataLoader()
    manager = SubjectManager(data_loader=loader)

    print(f"[Init] Loaded manifest with {len(loader.get_all_subject_ids())} subjects")

    # Get available analysis groups
    groups = loader.get_analysis_groups()
    print(f"[Init] Available groups: {list(groups.keys())}")

    # Show data availability summary
    availability = manager.get_subjects_availability_summary()
    print(f"[Init] Data availability summary: total={availability['total_subjects']}, "
          f"downloaded={availability['downloaded_subjects']}, "
          f"with_timeseries={availability['with_timeseries_data']}, "
          f"ready_for_processing={availability['breakdown']['ready_for_processing']}")

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
    print(f"[Stage 1] Processing individual subjects...")

    # Apply subject limiting if enabled
    non_anhedonic_to_process = []
    low_anhedonic_to_process = []
    high_anhedonic_to_process = []

    if LIMIT_SUBJECTS:
        # Separate low-anhedonic and high-anhedonic subjects for proper sampling
        low_anhedonic_to_process = low_anhedonic_subjects[:MAX_SUBJECTS_PER_GROUP]
        high_anhedonic_to_process = high_anhedonic_subjects[:MAX_SUBJECTS_PER_GROUP]
        non_anhedonic_to_process = accessible_non_anhedonic[:MAX_SUBJECTS_PER_GROUP]

        # Combine anhedonic subjects after sampling
        anhedonic_subjects_to_process = low_anhedonic_to_process + high_anhedonic_to_process

        print(f"LIMITING ENABLED: Processing {len(low_anhedonic_to_process)} low-anhedonic + {len(high_anhedonic_to_process)} high-anhedonic + {len(non_anhedonic_to_process)} non-anhedonic subjects")
        print(f"  Total: {len(anhedonic_subjects_to_process)} anhedonic + {len(non_anhedonic_to_process)} non-anhedonic = {len(anhedonic_subjects_to_process) + len(non_anhedonic_to_process)} subjects")
    else:
        anhedonic_subjects_to_process = accessible_anhedonic
        non_anhedonic_to_process = accessible_non_anhedonic
        print(f"FULL ANALYSIS: Processing {len(anhedonic_subjects_to_process)} anhedonic + {len(non_anhedonic_to_process)} non-anhedonic subjects")

    # ===== NEW 5-STAGE PIPELINE =====

    # STAGE 1: Process all subjects (pure computation)
    # (Subject-level progress printed below)

    all_subject_results = {}
    total_subjects = len(anhedonic_subjects_to_process) + len(non_anhedonic_to_process)

    # Process all subjects (anhedonic + non-anhedonic)
    all_subjects_to_process = anhedonic_subjects_to_process + non_anhedonic_to_process

    for i, subject_id in enumerate(all_subjects_to_process, 1):
        print_progress = True  # Always print per-subject status (success/fail)
        if print_progress:
            print(f"\n[{i}/{total_subjects}] Processing subject: {subject_id}")
        subject_result = process_subject(
            subject_id, manager, loader, cortical_atlas, subcortical_atlas,
            cortical_roi_extractor, subcortical_roi_extractor, cortical_ROIs, subcortical_ROIs,
            verbose=verbose
        )
        all_subject_results[subject_id] = subject_result
        if subject_result['success']:
            print(f"    ✅ Success")
        else:
            print(f"    ❌ Failed: {subject_id} - {subject_result.get('error', 'Unknown error')}")

    # Quick Stage 1 summary
    successful_subjects = sum(1 for r in all_subject_results.values() if r.get('success'))
    print(f"\n[Stage 1] Completed subject processing: {successful_subjects}/{total_subjects} successful")

    # Split results for backward compatibility
    anhedonic_results = {sid: all_subject_results[sid] for sid in anhedonic_subjects_to_process if sid in all_subject_results}
    non_anhedonic_results = {sid: all_subject_results[sid] for sid in non_anhedonic_to_process if sid in all_subject_results}

    # STAGE 2: Create connectivity mappings
    print(f"[Stage 2] Building connectivity mappings...")
    connectivity_mappings = create_connectivity_mappings(all_subject_results, verbose=verbose)

    # STAGE 3: Group aggregation
    print(f"[Stage 3] Aggregating group connectivity...")
    anhedonia_groups = {
        'non-anhedonic': non_anhedonic_to_process,
        'low-anhedonic': low_anhedonic_to_process,
        'high-anhedonic': high_anhedonic_to_process
    }

    group_aggregations = aggregate_group_connectivity(
        all_subject_results, connectivity_mappings, anhedonia_groups, verbose=verbose
    )

    # STAGE 4: Statistics preparation
    print(f"[Stage 4] Preparing statistics data...")
    anhedonic_stat_data = prepare_statistics_data(
        all_subject_results, connectivity_mappings, anhedonia_groups, verbose=verbose
    )

    print(f"[Stage 4] Creating group permutations...")
    permutation_result = get_group_permutations(all_subject_results, anhedonia_groups, verbose=verbose)
    permutation_count = permutation_result.get('n_permutations', 0)
    permutation_stat_data = [None] * permutation_count
    for perm_idx in range(permutation_count):
        group_permutation = permutation_result['group_permutations'][perm_idx]
        group_perm_stat_data = prepare_statistics_data(
            all_subject_results, connectivity_mappings, group_permutation, verbose=False
        )
        permutation_stat_data.append(group_perm_stat_data)

    # Stage 4: Statistical testing - perform Welch ANOVA analysis on prepared statistics
    # Tests each interhemispheric intra-network connection for group differences
    # using Welch ANOVA + Games-Howell post-hoc approach
    print(f"[Stage 4] Running Welch ANOVA + Games-Howell tests...")
    anhedonic_anova_results = coherence_anova_test(anhedonic_stat_data, verbose=verbose)

    # STAGE 5: Compute group-averaged FC
    print(f"[Stage 5] Computing group-averaged FC...")

    group_averaged_fc = {
        'static': {},
        'slow_bands': {}
    }

    for group_name, group_subjects in anhedonia_groups.items():
        # Compute static FC group average
        static_avg = compute_group_averaged_fc(
            all_subject_results, group_subjects, group_name, fc_type='static', verbose=verbose
        )
        if static_avg:
            group_averaged_fc['static'][group_name] = static_avg

    # Compute slow-band FC group averages (organized by band, then group)
    for band_num in range(1, 7):  # Slow bands 1-6
        band_key = f'slow-{band_num}'
        group_averaged_fc['slow_bands'][band_key] = {}

        for group_name, group_subjects in anhedonia_groups.items():
            band_avg = compute_group_averaged_fc(
                all_subject_results, group_subjects, group_name, fc_type='slow_band', band_key=band_key, verbose=verbose
            )
            if band_avg:
                group_averaged_fc['slow_bands'][band_key][group_name] = band_avg

    # STAGE 6: Prepare visualization data (Z->R conversion)
    print(f"[Stage 6] Preparing plotting data (Z->R conversions)...")
    plotting_data = prepare_plotting_data(all_subject_results, verbose=verbose)

    # STAGE 6b: Prepare cross-subject slow-band aggregated FC matrices
    print(f"[Stage 6B] Preparing cross-subject slow-band FC matrices...")

    # Use the slow-band aggregations from the main group aggregation (already organized by groups)
    cross_subject_slow_bands = group_aggregations['slow_band_fc_by_group']

    # ===== RESULTS SUMMARY =====
    print(f"\n[Summary] Processing results")

    anhedonic_success = sum(1 for r in anhedonic_results.values() if r['success'])
    non_anhedonic_success = sum(1 for r in non_anhedonic_results.values() if r['success'])
    total_success = anhedonic_success + non_anhedonic_success
    total_processed = len(anhedonic_results) + len(non_anhedonic_results)

    print(f"[Summary] Successfully processed: {total_success}/{total_processed} subjects "
          f"(anhedonic={anhedonic_success}/{len(anhedonic_results)}, "
          f"non-anhedonic={non_anhedonic_success}/{len(non_anhedonic_results)})")

    # Note: STAGE 5 (Visualization) is handled by the existing plotting code below
    # The plotting functions are now updated to work with new data structures

    # ===== DISPLAY PIPELINE RESULTS =====
    print(f"\n{'='*80}")
    print("PIPELINE RESULTS SUMMARY")
    print(f"{'='*80}")

    # Show connectivity mappings summary
    interhemi_intra_count = len(connectivity_mappings['interhemispheric_pairs']['intra_network'])
    interhemi_inter_count = len(connectivity_mappings['interhemispheric_pairs']['inter_network'])
    intra_network_groups_count = len(connectivity_mappings['intra_network_pairs'])

    print(f"\nConnectivity Analysis:")
    print(f"  Interhemispheric intra-network pairs: {interhemi_intra_count}")
    print(f"  Interhemispheric inter-network pairs: {interhemi_inter_count}")
    print(f"  Network groups for statistics: {intra_network_groups_count}")

    # Show group aggregation summary
    print(f"\nGroup Aggregations:")
    for group_name, data in group_aggregations['static_fc_by_group'].items():
        print(f"  {group_name}: {data['n_subjects']} subjects with static FC")

    # Show statistics data summary
    print(f"\nStatistics Data:")
    for group_name in anhedonia_groups.keys():
        static_nets = len(anhedonic_stat_data['static_coherence_by_group'].get(group_name, {}))
        slow_nets = len(anhedonic_stat_data['slow_band_coherence_by_group'].get(group_name, {}))
        print(f"  {group_name}: {static_nets} static networks, {slow_nets} slow-band networks")

    # Show plotting data summary
    plotting_subjects = sum(1 for p in plotting_data.values() if 'static_fc_matrix' in p)
    print(f"\nVisualization Data: {plotting_subjects} subjects ready for plotting")

    # ===== ANALYSIS USING NEW PIPELINE RESULTS =====

    # Display detailed statistics data
    if verbose and anhedonic_stat_data:
        print(f"\n{'='*80}")
        print("DETAILED STATISTICS RESULTS")
        print(f"{'='*80}")

        for group_name in anhedonia_groups.keys():
            print(f"\n{group_name.upper()} GROUP:")

            # Static coherence
            static_data = anhedonic_stat_data['static_coherence_by_group'].get(group_name, {})
            if static_data:
                print(f"  Static FC Networks ({len(static_data)}):")
                for network, stats in static_data.items():
                    mean_coh = stats['mean_coherence']
                    subjects_count = stats['n_subjects']
                    print(f"    {network}: mean_z={mean_coh:.4f} (n={subjects_count})")

            # Slow-band coherence
            slow_data = anhedonic_stat_data['slow_band_coherence_by_group'].get(group_name, {})
            if slow_data:
                print(f"  Slow-Band Networks ({len(slow_data)}):")
                for network, band_data in slow_data.items():
                    print(f"    {network}:")
                    for band, stats in band_data.items():
                        mean_coh = stats['mean_coherence']
                        subjects_count = stats['n_subjects']
                        print(f"      {band}: mean_z={mean_coh:.4f} (n={subjects_count})")

    # CSV Export using new pipeline data
    if save_figures:
        print(f"\n{'='*80}")
        print("EXPORTING CSV DATA FROM NEW PIPELINE")
        print(f"{'='*80}")

        # Create CSV export directory
        csv_export_dir = run_parent_dir / 'csv_exports'
        csv_export_dir.mkdir(parents=True, exist_ok=True)

        # Create FC analysis output directory for CSV exports
        fc_output_dir = csv_export_dir / 'fc_analysis'
        fc_output_dir.mkdir(parents=True, exist_ok=True)

        # Create group averages output directory
        group_avg_output_dir = csv_export_dir / 'group_averages'
        group_avg_output_dir.mkdir(parents=True, exist_ok=True)

        # Create ANOVA/Post-hoc output directory
        anova_output_dir = csv_export_dir / 'anova_results'
        anova_output_dir.mkdir(parents=True, exist_ok=True)
        post_hoc_dir = csv_export_dir / 'post_hoc_results'
        post_hoc_dir.mkdir(parents=True, exist_ok=True)

        # Export connectivity mappings
        mappings_file = csv_export_dir / 'connectivity_mappings.csv'
        print(f"Exporting connectivity mappings to: {mappings_file}")

        # Export Welch ANOVA summary (static)
        static_rows = []
        static_results = anhedonic_anova_results.get('static_results', {})
        static_stats = anhedonic_stat_data.get('static_coherence_by_group', {})

        def _get_group_stats(group_dict, network_key, group_name):
            net_data = group_dict.get(group_name, {}).get(network_key, {})
            return (
                net_data.get('mean_coherence', np.nan),
                net_data.get('std_coherence', np.nan),
                net_data.get('n_subjects', 0)
            )

        for network_key, anova_data in static_results.items():
            row = {
                'band': 'static',
                'network': network_key,
                'omnibus_p': anova_data.get('omnibus_p'),
                'fdr_corrected_p': anova_data.get('fdr_corrected_p'),
                'fdr_significant': anova_data.get('fdr_significant', False),
            }
            for group_name in ['non-anhedonic', 'low-anhedonic', 'high-anhedonic']:
                mean_val, sd_val, n_val = _get_group_stats(static_stats, network_key, group_name)
                prefix = group_name.replace('-', '_')
                row[f'{prefix}_mean_z'] = mean_val
                row[f'{prefix}_sd_z'] = sd_val
                row[f'{prefix}_n'] = n_val
            static_rows.append(row)

        static_anova_path = anova_output_dir / 'welch_anova_static.csv'
        pd.DataFrame(static_rows).to_csv(static_anova_path, index=False)
        print(f"Exported static Welch ANOVA results to: {static_anova_path}")

        # Export Welch ANOVA summary (slow bands)
        slow_rows = []
        slow_results = anhedonic_anova_results.get('slow_band_results', {})
        slow_stats = anhedonic_stat_data.get('slow_band_coherence_by_group', {})

        for band_name, band_data in slow_results.items():
            for network_key, anova_data in band_data.items():
                row = {
                    'band': band_name,
                    'network': network_key,
                    'omnibus_p': anova_data.get('omnibus_p'),
                    'fdr_corrected_p': anova_data.get('fdr_corrected_p'),
                    'fdr_significant': anova_data.get('fdr_significant', False),
                }
                for group_name in ['non-anhedonic', 'low-anhedonic', 'high-anhedonic']:
                    net_data = slow_stats.get(group_name, {}).get(network_key, {}).get(band_name, {})
                    mean_val = net_data.get('mean_coherence', np.nan)
                    sd_val = net_data.get('std_coherence', np.nan)
                    n_val = net_data.get('n_subjects', 0)
                    prefix = group_name.replace('-', '_')
                    row[f'{prefix}_mean_z'] = mean_val
                    row[f'{prefix}_sd_z'] = sd_val
                    row[f'{prefix}_n'] = n_val
                slow_rows.append(row)

        slow_anova_path = anova_output_dir / 'welch_anova_slow_bands.csv'
        pd.DataFrame(slow_rows).to_csv(slow_anova_path, index=False)
        print(f"Exported slow-band Welch ANOVA results to: {slow_anova_path}")

        # Export IPSILATERAL Welch ANOVA summary (static)
        ipsi_static_rows = []
        ipsi_static_results = anhedonic_anova_results.get('ipsi_static_results', {})
        ipsi_static_stats = anhedonic_stat_data.get('ipsi_static_coherence_by_group', {})

        def _parse_conn_key(conn_key):
            # conn_key format: "LH:RegionNet__RegionNet"
            hemi, rest = conn_key.split(':', 1) if ':' in conn_key else ('NA', conn_key)
            parts = rest.split('__')
            p1 = parts[0] if len(parts) > 0 else 'NA'
            p2 = parts[1] if len(parts) > 1 else 'NA'
            return hemi, p1, p2

        for conn_key, anova_data in ipsi_static_results.items():
            hemi, rn1, rn2 = _parse_conn_key(conn_key)
            row = {
                'band': 'static_ipsi',
                'connection_key': conn_key,
                'hemisphere': hemi,
                'region_network_1': rn1,
                'region_network_2': rn2,
                'omnibus_p': anova_data.get('omnibus_p'),
                'fdr_corrected_p': anova_data.get('fdr_corrected_p'),
                'fdr_significant': anova_data.get('fdr_significant', False),
            }
            for group_name in ['non-anhedonic', 'low-anhedonic', 'high-anhedonic']:
                conn_stats = ipsi_static_stats.get(group_name, {}).get(conn_key, {})
                mean_val = conn_stats.get('mean_coherence', np.nan)
                sd_val = conn_stats.get('std_coherence', np.nan)
                n_val = conn_stats.get('n_subjects', 0)
                prefix = group_name.replace('-', '_')
                row[f'{prefix}_mean_z'] = mean_val
                row[f'{prefix}_sd_z'] = sd_val
                row[f'{prefix}_n'] = n_val
            ipsi_static_rows.append(row)

        ipsi_static_anova_path = anova_output_dir / 'welch_anova_static_ipsilateral.csv'
        pd.DataFrame(ipsi_static_rows).to_csv(ipsi_static_anova_path, index=False)
        print(f"Exported ipsilateral static Welch ANOVA results to: {ipsi_static_anova_path}")

        # Export IPSILATERAL Welch ANOVA summary (slow bands)
        ipsi_slow_rows = []
        ipsi_slow_results = anhedonic_anova_results.get('ipsi_slow_band_results', {})
        ipsi_slow_stats = anhedonic_stat_data.get('ipsi_slow_band_coherence_by_group', {})

        for band_name, band_data in ipsi_slow_results.items():
            for conn_key, anova_data in band_data.items():
                hemi, rn1, rn2 = _parse_conn_key(conn_key)
                row = {
                    'band': f'{band_name}_ipsi',
                    'connection_key': conn_key,
                    'hemisphere': hemi,
                    'region_network_1': rn1,
                    'region_network_2': rn2,
                    'omnibus_p': anova_data.get('omnibus_p'),
                    'fdr_corrected_p': anova_data.get('fdr_corrected_p'),
                    'fdr_significant': anova_data.get('fdr_significant', False),
                }
                for group_name in ['non-anhedonic', 'low-anhedonic', 'high-anhedonic']:
                    band_stats = ipsi_slow_stats.get(group_name, {}).get(conn_key, {}).get(band_name, {})
                    mean_val = band_stats.get('mean_coherence', np.nan)
                    sd_val = band_stats.get('std_coherence', np.nan)
                    n_val = band_stats.get('n_subjects', 0)
                    prefix = group_name.replace('-', '_')
                    row[f'{prefix}_mean_z'] = mean_val
                    row[f'{prefix}_sd_z'] = sd_val
                    row[f'{prefix}_n'] = n_val
                ipsi_slow_rows.append(row)

        ipsi_slow_anova_path = anova_output_dir / 'welch_anova_slow_bands_ipsilateral.csv'
        pd.DataFrame(ipsi_slow_rows).to_csv(ipsi_slow_anova_path, index=False)
        print(f"Exported ipsilateral slow-band Welch ANOVA results to: {ipsi_slow_anova_path}")

        # Export Games-Howell post-hoc results
        post_hoc_collection = anhedonic_anova_results.get('post_hoc_collection', [])
        for post_hoc_data in post_hoc_collection:
            network = post_hoc_data['network']
            band = post_hoc_data['band']
            post_hoc_df = post_hoc_data['post_hoc']

            # Clean network and band names for filename
            safe_network = network.replace('/', '_').replace(' ', '_')
            safe_band = band.replace('/', '_').replace(' ', '_')

            filename = f"games_howell_{safe_network}_{safe_band}.csv"
            filepath = post_hoc_dir / filename

            # Save with additional metadata
            with open(filepath, 'w') as f:
                f.write(f"# Games-Howell Post-hoc Test Results\n")
                f.write(f"# Network: {network}\n")
                f.write(f"# Band: {band}\n")
                f.write(f"# Test performed only because omnibus Welch ANOVA p < 0.05 (uncorrected)\n")
                f.write(f"# Post-hoc p-values are NOT corrected for multiple comparisons\n")
                f.write(f"#\n")
                post_hoc_df.to_csv(f, index=False)

        if post_hoc_collection:
            print(f"Exported {len(post_hoc_collection)} Games-Howell files to: {post_hoc_dir}")
    # ===== INDIVIDUAL SUBJECT PLOTS =====
    individual_plots_created = 0
    figures_saved_count = 0

    # Create output directories for different analysis types
    figures_base_dir = run_parent_dir / 'figures' if save_figures and create_plots else None
    fc_subject_dir = None
    fc_group_dir = None
    welch_interhemi_dir = None
    welch_ipsi_dir = None
    mvmd_figures_dir = None
    hsa_figures_dir = None
    marginal_hsa_figures_dir = None
    roi_figures_dir = None

    if save_figures and create_plots:
        # FC directories
        fc_subject_dir = figures_base_dir / 'fc_subject'
        fc_group_dir = figures_base_dir / 'fc_group'

        # Welch ANOVA interhemispheric intra-network directory
        welch_interhemi_dir = figures_base_dir / 'welch_interhemispheric_violin'
        welch_ipsi_dir = figures_base_dir / 'welch_ipsilateral_violin'

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
        welch_interhemi_dir.mkdir(parents=True, exist_ok=True)
        welch_ipsi_dir.mkdir(parents=True, exist_ok=True)
        mvmd_figures_dir.mkdir(parents=True, exist_ok=True)
        hsa_figures_dir.mkdir(parents=True, exist_ok=True)
        marginal_hsa_figures_dir.mkdir(parents=True, exist_ok=True)
        roi_figures_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[INFO] Figures will be saved to:")
        print(f"  FC (subject-level): {fc_subject_dir}")
        print(f"  FC (group-level): {fc_group_dir}")
        print(f"  Welch Interhemispheric Violin: {welch_interhemi_dir}")
        print(f"  Welch Ipsilateral Violin: {welch_ipsi_dir}")
        print(f"  MVMD decomposition: {mvmd_figures_dir}")
        print(f"  Hilbert Spectral Analysis: {hsa_figures_dir}")
        print(f"  Marginal Hilbert Spectral Analysis: {marginal_hsa_figures_dir}")
        print(f"  ROI extraction: {roi_figures_dir}")

    if create_plots:
        print(f"\n{'='*80}")
        print(f"CREATING PLOTS")
        print(f"{'='*80}")

        if show_plots:
            print(f"[INFO] Figures will be displayed in batches by analysis type")
            print(f"       Close all figures in each batch to proceed to the next batch")

        # Organize plotting by type to display in batches
        # This prevents overwhelming the system with 177+ figures at once
        plot_batches = {
            'roi_cortical': [],
            'roi_subcortical': [],
            'fc_static': [],
            'fc_per_mode': [],
            'fc_slow_bands_individual': [],
            'fc_slow_bands': [],
            'fc_slow_bands_cross_subject': [],
            'fc_group_avg': [],
            'interhemispheric_violin': [],
            'ipsilateral_violin': [],
            'mvmd_modes': [],
            'hsa_multivariate': [],
            'hsa_marginal': [],
        }
        plot_batch_count = len(plot_batches.keys())

        # ===== INDIVIDUAL SUBJECT PLOTS =====
        print(f"\nPreparing individual subject plots...")
        individual_plots_created = 0
        debug_print = print if verbose else (lambda *args, **kwargs: None)

        # Debug: Check subject success status
        total_subjects = len(all_subject_results)
        successful_subjects = sum(1 for r in all_subject_results.values() if r.get('success', False))
        debug_print(f"  Debug: {successful_subjects}/{total_subjects} subjects marked as successful")

        # Debug group membership
        debug_print(f"  Debug group lists:")
        debug_print(f"    low_anhedonic_to_process: {low_anhedonic_to_process}")
        debug_print(f"    high_anhedonic_to_process: {high_anhedonic_to_process}")
        debug_print(f"    non_anhedonic_subjects_to_process: {non_anhedonic_to_process}")

        if successful_subjects == 0:
            debug_print(f"  Debug: No successful subjects found. Checking first few subjects:")
            for i, (subject_id, result) in enumerate(list(all_subject_results.items())[:3]):
                success_status = result.get('success', False)
                error_msg = result.get('error', 'No error')
                debug_print(f"    {subject_id}: success={success_status}, error='{error_msg}'")

        # Use new pipeline results instead of legacy all_results
        subjects_entered_loop = 0
        for subject_id, result in all_subject_results.items():
            subjects_entered_loop += 1
            debug_print(f"    Checking {subject_id}: success={result.get('success', False)}")
            if not result.get('success', False):
                print(f"      Skipping {subject_id} - not successful")
                continue

            # Determine which group this subject belongs to
            if subject_id in low_anhedonic_to_process:
                subject_group = "Low Anhedonic"
            elif subject_id in high_anhedonic_to_process:
                subject_group = "High Anhedonic"
            elif subject_id in non_anhedonic_to_process:
                subject_group = "Non-anhedonic"
            else:
                subject_group = None

            debug_print(f"      Processing {subject_id} for plot preparation...")
            debug_print(f"        plotting_data keys: {list(plotting_data.keys())}")
            debug_print(f"        {subject_id} in plotting_data: {subject_id in plotting_data}")
            plots_for_subject = 0

            # 1. Prepare cortical ROI timeseries plot
            roi_results = result.get('roi_extraction_results', {})
            debug_print(f"        ROI results present: {roi_results is not None}")
            debug_print(f"        Cortical data present: {roi_results.get('cortical') is not None}")
            if roi_results.get('cortical'):
                cortical_data = roi_results['cortical']
                debug_print(f"        Cortical extraction successful: {cortical_data.get('extraction_successful')}")
                if cortical_data.get('extraction_successful'):
                    debug_print(f"          Adding cortical ROI plot for {subject_id}")
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


            # 3. Prepare static functional connectivity analysis plot - UPDATED FOR NEW PIPELINE
            if subject_id in plotting_data:
                static_fc_data = plotting_data[subject_id]
                debug_print(f"        Static FC data keys: {list(static_fc_data.keys())}")
                debug_print(f"        Has static_fc_matrix: {'static_fc_matrix' in static_fc_data}")
                if static_fc_data.get('static_fc_matrix') is not None:
                    debug_print(f"          Adding static FC plot for {subject_id}")
                    plot_batches['fc_static'].append({
                        'subject_id': subject_id,
                        'subject_group': subject_group,
                        'data': static_fc_data,  # Now contains R-values for proper visualization
                        'mask_diagonal': mask_diagonal,
                        'mask_nonsignificant': mask_nonsignificant,
                        'save_dir': fc_subject_dir if save_figures and fc_subject_dir else None
                    })
                    plots_for_subject += 1
                    debug_print(f"          plots_for_subject now: {plots_for_subject}")

            # 3a. Prepare per-mode FC plots (individual modes)
            if subject_id in plotting_data:
                subject_data = plotting_data[subject_id]
                if 'r_fc_modes' in subject_data and 'center_freqs' in subject_data:
                    r_fc_modes = subject_data['r_fc_modes']
                    center_freqs = subject_data['center_freqs']
                    fc_labels = subject_data.get('static_fc_labels', [])

                    # Get mode p-values from result
                    mode_fc_pvalues = result.get('mode_fc_pvalues')

                    # Create plots for each mode
                    for mode_idx in range(r_fc_modes.shape[0]):
                        # Compute connectivity patterns for this mode
                        mode_pvalues = mode_fc_pvalues[mode_idx] if mode_fc_pvalues is not None else None
                        mode_connectivity_patterns = analyze_connectivity_patterns(
                            r_fc_modes[mode_idx],
                            fc_labels,
                            p_values=mode_pvalues,
                            alpha=0.05
                        )

                        plot_batches['fc_per_mode'].append({
                            'subject_id': subject_id,
                            'subject_group': subject_group,
                            'mode_idx': mode_idx,
                            'frequency': center_freqs[mode_idx],
                            'data': {
                                'fc_matrix': r_fc_modes[mode_idx],
                                'fc_labels': fc_labels,
                                'fc_pvalues': mode_pvalues,
                                'connectivity_patterns': mode_connectivity_patterns
                            },
                            'mask_diagonal': mask_diagonal,
                            'mask_nonsignificant': mask_nonsignificant,
                            'save_dir': fc_subject_dir if save_figures and fc_subject_dir else None
                        })
                        plots_for_subject += 1

            # 3b. Prepare individual subject slow-band FC plots (aggregated within-subject)
            if subject_id in plotting_data:
                subject_data = plotting_data[subject_id]
                if 'slow_band_fc_modes' in subject_data:
                    slow_band_fc_data = subject_data['slow_band_fc_modes']
                    fc_labels = subject_data.get('static_fc_labels', [])

                    for band_name, band_data in slow_band_fc_data.items():
                        # Compute connectivity patterns for this slow-band
                        # Note: No p-values for slow-band aggregations (would need statistical testing)
                        band_connectivity_patterns = analyze_connectivity_patterns(
                            band_data['fc_matrix'],
                            fc_labels,
                            p_values=None,  # No p-values for aggregated slow-bands
                            alpha=0.05
                        )

                    plot_batches['fc_slow_bands_individual'].append({
                        'subject_id': subject_id,
                        'subject_group': subject_group,
                        'band_name': band_name,
                        'data': {
                            'fc_matrix': band_data['fc_matrix'],
                            'fc_labels': fc_labels,
                            'fc_pvalues': band_data.get('fc_pvalues'),
                            'connectivity_patterns': band_connectivity_patterns,
                            'n_modes': band_data['n_modes'],
                            'frequencies': band_data['frequencies']
                        },
                        'mask_diagonal': mask_diagonal,
                        'mask_nonsignificant': mask_nonsignificant,
                        'save_dir': fc_subject_dir if save_figures and fc_subject_dir else None
                    })
                    plots_for_subject += 1

            debug_print(f"        About to check slow-band FC plots...")
            # 3c. Prepare slow-band FC plots (legacy)
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

                debug_print(f"        Final plots_for_subject count: {plots_for_subject}")
                if plots_for_subject > 0:
                    individual_plots_created += 1
                    debug_print(f"      ✓ Prepared {plots_for_subject} plots for {subject_id}")
                else:
                    debug_print(f"      ✗ No plots prepared for {subject_id}")
        else:
            debug_print(f"  No successful subjects found for individual plots (checked {subjects_entered_loop} subjects total)")

        print(f"\nPrepared plots for {individual_plots_created} subjects")

        # Prepare cross-subject slow-band FC plots (grouped by anhedonia status)
        print(f"\nPreparing group-specific slow-band FC plots...")
        total_group_slow_band_plots = 0

        for group_name, group_bands in cross_subject_slow_bands.items():
            for band_name, band_data in group_bands.items():
                # Extract channel labels from any successful subject
                fc_labels = []
                for subject_data in plotting_data.values():
                    if 'static_fc_labels' in subject_data:
                        fc_labels = subject_data['static_fc_labels']
                        break

                # Compute connectivity patterns for this group slow-band
                band_connectivity_patterns = analyze_connectivity_patterns(
                    band_data['avg_r_matrix'],
                    fc_labels,
                    p_values=None,  # No p-values for aggregated slow-bands
                    alpha=0.05
                )

                plot_batches['fc_slow_bands_cross_subject'].append({
                    'band_name': f'{group_name}_{band_name}',
                    'data': {
                        'fc_matrix': band_data['avg_r_matrix'],
                        'fc_labels': fc_labels,
                        'n_matrices': band_data['n_matrices'],
                        'description': band_data['description'],
                        'connectivity_patterns': band_connectivity_patterns
                    },
                    'mask_diagonal': mask_diagonal,
                    'mask_nonsignificant': False,  # No p-values for cross-subject averages
                    'save_path': fc_group_dir / f'{group_name}_{band_name}_fc.svg' if save_figures and fc_group_dir else None
                })
                total_group_slow_band_plots += 1

        print(f"✓ Prepared {total_group_slow_band_plots} group-specific slow-band FC plots")

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

        # Prepare interhemispheric intra-network violin plots
        print(f"\nPreparing interhemispheric intra-network violin plots...")
        from tcp.processing.lib.plot import plot_interhemispheric_intra_network_violin

        interhemi_figures = plot_interhemispheric_intra_network_violin(
            stat_data=anhedonic_stat_data,
            anova_results=anhedonic_anova_results
        )

        if interhemi_figures:
            for fig, metadata in interhemi_figures:
                plot_batches['interhemispheric_violin'].append({
                    'figure': fig,
                    'metadata': metadata,
                    'save_dir': welch_interhemi_dir
                })

        print(f"  ✓ Prepared {len(plot_batches['interhemispheric_violin'])} interhemispheric violin plots")

        # Prepare ipsilateral connectivity violin plots
        ipsi_figures = plot_ipsilateral_intra_network_violin(
            stat_data=anhedonic_stat_data,
            anova_results=anhedonic_anova_results
        )

        if ipsi_figures:
            for fig, metadata in ipsi_figures:
                plot_batches['ipsilateral_violin'].append({
                    'figure': fig,
                    'metadata': metadata,
                    'save_dir': welch_ipsi_dir
                })

        print(f"  ✓ Prepared {len(plot_batches['ipsilateral_violin'])} ipsilateral violin plots")

        # Now create and display plots in batches by type
        # Recompute batch count based on non-empty queues
        non_empty_batches = [name for name, items in plot_batches.items() if items]
        plot_batch_count = len(non_empty_batches) if non_empty_batches else 0

        print(f"\n{'='*80}")
        print(f"CREATING AND DISPLAYING PLOTS BY TYPE")
        print(f"{'='*80}")

        current_plot_batch = 0

        # Batch 1: ROI Cortical Timeseries
        if plot_batches['roi_cortical']:
            current_plot_batch += 1
            print(f"[Batch {current_plot_batch}/{plot_batch_count}] Creating {len(plot_batches['roi_cortical'])} cortical ROI timeseries plots...")
            for plot_info in plot_batches['roi_cortical']:
                figures = plot_roi_timeseries_result(plot_info['data'], subject_id=plot_info['subject_id'], atlas_type='Cortical')
                # Handle both single figure and list of figures return types
                if not isinstance(figures, list):
                    figures = [figures]
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
            print(f"[Batch {current_plot_batch}/{plot_batch_count}] Creating {len(plot_batches['roi_subcortical'])} subcortical ROI timeseries plots...")
            for plot_info in plot_batches['roi_subcortical']:
                figures = plot_roi_timeseries_result(plot_info['data'], subject_id=plot_info['subject_id'], atlas_type='Subcortical')
                # Handle both single figure and list of figures return types
                if not isinstance(figures, list):
                    figures = [figures]
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
            print(f"[Batch {current_plot_batch}/{plot_batch_count}] Creating {len(plot_batches['hsa_multivariate'])} Multivariate Hilbert Spectrum plots...")

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
            print(f"[Batch {current_plot_batch}/{plot_batch_count}] Creating {len(plot_batches['hsa_marginal'])} Marginal Spectrum per Mode plots...")

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
            print(f"[Batch {current_plot_batch}/{plot_batch_count}] Creating {len(plot_batches['fc_static'])} static FC plots...")
            for plot_info in plot_batches['fc_static']:
                # Use fc_output_dir for CSV exports (fc_analysis/static_fc/), not figures directory
                csv_output_dir = fc_output_dir if save_figures else None

                fc_fig_inter, fc_fig_ipsi = plot_fc_results(
                    plot_info['data']['static_fc_matrix'],
                    plot_info['data']['static_fc_labels'],
                    p_values=plot_info['data']['static_fc_pvalues'],
                    connectivity_patterns=plot_info['data']['static_connectivity_patterns'],
                    channel_label_map=plot_info['data'].get('channel_label_map'),
                    mask_diagonal=plot_info['mask_diagonal'],
                    mask_nonsignificant=plot_info['mask_nonsignificant'],
                    subject_group=plot_info['subject_group'],
                    subject_id=plot_info['subject_id'],
                    output_dir=csv_output_dir,
                    verbose=verbose
                )
                if fc_fig_inter is None or fc_fig_ipsi is None:
                    print(f"  Skipping static FC plot for {plot_info['subject_id']} due to missing data (empty matrix/labels)")
                    continue
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
            if show_plots and len(plot_batches['fc_static']) >= 20:
                print(f"  Skipping display of {len(plot_batches['fc_static'])} FC plots. Will be too memory exhaustive...")
            elif show_plots:
                print(f"  Displaying {len(plot_batches['fc_static'])} FC plots. Close all figures to continue...")
                plt.show()

        # Batch 6: Slow-Band FC Analysis
        if plot_batches['fc_slow_bands']:
            current_plot_batch += 1
            print(f"[Batch {current_plot_batch}/{plot_batch_count}] Creating {len(plot_batches['fc_slow_bands'])} slow-band FC plots...")
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
                if fc_fig_inter is None or fc_fig_ipsi is None:
                    print(f"  Skipping slow-band FC plot (Slow-{band_number}) for {plot_info['subject_id']} due to missing data (empty matrix/labels)")
                    continue
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
            if show_plots and len(plot_batches['fc_slow_bands']) >= 20:
                print(f"  Skipping display of {len(plot_batches['fc_slow_bands'])} FC plots. Will be too memory exhaustive...")
            elif show_plots:
                print(f"  Displaying {len(plot_batches['fc_slow_bands'])} slow-band FC plots. Close all figures to continue...")
                plt.show()

        # Batch 6a: Per-Mode FC Analysis
        if plot_batches['fc_per_mode']:
            current_plot_batch += 1
            print(f"[Batch {current_plot_batch}/{plot_batch_count}] Creating {len(plot_batches['fc_per_mode'])} per-mode FC plots...")
            for plot_info in plot_batches['fc_per_mode']:
                csv_output_dir = fc_output_dir if save_figures else None

                fc_fig_inter, fc_fig_ipsi = plot_fc_results(
                    plot_info['data']['fc_matrix'],
                    plot_info['data']['fc_labels'],
                    p_values=plot_info['data'].get('fc_pvalues'),
                    connectivity_patterns=plot_info['data']['connectivity_patterns'],
                    channel_label_map=None,
                    mask_diagonal=plot_info['mask_diagonal'],
                    mask_nonsignificant=plot_info['mask_nonsignificant'],
                    subject_group=plot_info['subject_group'],
                    subject_id=plot_info['subject_id'],
                    output_dir=csv_output_dir,
                    verbose=verbose
                )
                if fc_fig_inter is None or fc_fig_ipsi is None:
                    print(f"  Skipping per-mode FC plot (mode {plot_info['mode_idx']}) for {plot_info['subject_id']} due to missing data (empty matrix/labels)")
                    continue

                mode_idx = plot_info['mode_idx']
                frequency = plot_info['frequency']
                fc_fig_inter.suptitle(f'Mode {mode_idx} FC (Interhemispheric) - {plot_info["subject_id"]} ({frequency:.4f} Hz)', fontsize=12, fontweight='bold')
                fc_fig_ipsi.suptitle(f'Mode {mode_idx} FC (Ipsilateral) - {plot_info["subject_id"]} ({frequency:.4f} Hz)', fontsize=12, fontweight='bold')

                if plot_info['save_dir']:
                    subject_fc_dir = plot_info['save_dir'] / plot_info['subject_id']
                    subject_fc_dir.mkdir(parents=True, exist_ok=True)

                    save_path_inter = subject_fc_dir / f'mode_{mode_idx:02d}_fc_interhemispheric_{frequency:.4f}Hz.svg'
                    save_path_ipsi = subject_fc_dir / f'mode_{mode_idx:02d}_fc_ipsilateral_{frequency:.4f}Hz.svg'

                    fc_fig_inter.savefig(save_path_inter, format='svg', bbox_inches='tight', dpi=300)
                    fc_fig_ipsi.savefig(save_path_ipsi, format='svg', bbox_inches='tight', dpi=300)
                    figures_saved_count += 2

                if not show_plots:
                    plt.close(fc_fig_inter)
                    plt.close(fc_fig_ipsi)

            if show_plots and len(plot_batches['fc_per_mode']) >= 20:
                print(f"  Skipping display of {len(plot_batches['fc_per_mode'])} FC plots. Will be too memory exhaustive...")
            elif show_plots:
                print(f"  Displaying {len(plot_batches['fc_per_mode'])} per-mode FC plots. Close all figures to continue...")
                plt.show()

        # Batch 6b: Individual Subject Slow-Band FC Analysis
        if plot_batches['fc_slow_bands_individual']:
            current_plot_batch += 1
            print(f"[Batch {current_plot_batch}/{plot_batch_count}] Creating {len(plot_batches['fc_slow_bands_individual'])} individual subject slow-band FC plots...")
            for plot_info in plot_batches['fc_slow_bands_individual']:
                csv_output_dir = fc_output_dir if save_figures else None

                fc_fig_inter, fc_fig_ipsi = plot_fc_results(
                    plot_info['data']['fc_matrix'],
                    plot_info['data']['fc_labels'],
                    p_values=None,  # No p-values for aggregated slow-bands
                    connectivity_patterns=plot_info['data'].get('connectivity_patterns'),
                    channel_label_map=None,
                    mask_diagonal=plot_info['mask_diagonal'],
                    mask_nonsignificant=plot_info['mask_nonsignificant'],
                    subject_group=plot_info['subject_group'],
                    subject_id=plot_info['subject_id'],
                    output_dir=csv_output_dir,
                    verbose=verbose,
                    band_name=plot_info['band_name']
                )
                if fc_fig_inter is None or fc_fig_ipsi is None:
                    print(f"  Skipping individual slow-band FC plot ({plot_info['band_name']}) for {plot_info['subject_id']} due to missing data (empty matrix/labels)")
                    continue

                band_name = plot_info['band_name']
                n_modes = plot_info['data']['n_modes']
                fc_fig_inter.suptitle(f'{band_name} FC (Interhemispheric) - {plot_info["subject_id"]} ({n_modes} modes)', fontsize=12, fontweight='bold')
                fc_fig_ipsi.suptitle(f'{band_name} FC (Ipsilateral) - {plot_info["subject_id"]} ({n_modes} modes)', fontsize=12, fontweight='bold')

                if plot_info['save_dir']:
                    subject_fc_dir = plot_info['save_dir'] / plot_info['subject_id']
                    subject_fc_dir.mkdir(parents=True, exist_ok=True)

                    save_path_inter = subject_fc_dir / f'{band_name}_fc_interhemispheric.svg'
                    save_path_ipsi = subject_fc_dir / f'{band_name}_fc_ipsilateral.svg'

                    fc_fig_inter.savefig(save_path_inter, format='svg', bbox_inches='tight', dpi=300)
                    fc_fig_ipsi.savefig(save_path_ipsi, format='svg', bbox_inches='tight', dpi=300)
                    figures_saved_count += 2

                if not show_plots:
                    plt.close(fc_fig_inter)
                    plt.close(fc_fig_ipsi)

            if show_plots and len(plot_batches['fc_slow_bands_individual']) >= 20:
                print(f"  Skipping display of {len(plot_batches['fc_slow_bands_individual'])} FC plots. Will be too memory exhaustive...")
            elif show_plots:
                print(f"  Displaying {len(plot_batches['fc_slow_bands_individual'])} individual subject slow-band FC plots. Close all figures to continue...")
                plt.show()

        # Batch 6c: Cross-Subject Slow-Band FC Analysis
        if plot_batches['fc_slow_bands_cross_subject']:
            current_plot_batch += 1
            print(f"[Batch {current_plot_batch}/{plot_batch_count}] Creating {len(plot_batches['fc_slow_bands_cross_subject'])} cross-subject slow-band FC plots...")
            for plot_info in plot_batches['fc_slow_bands_cross_subject']:

                fc_fig_inter, fc_fig_ipsi = plot_fc_results(
                    plot_info['data']['fc_matrix'],
                    plot_info['data']['fc_labels'],
                    p_values=None,
                    connectivity_patterns=plot_info['data'].get('connectivity_patterns'),
                    channel_label_map=None,
                    mask_diagonal=plot_info['mask_diagonal'],
                    mask_nonsignificant=plot_info['mask_nonsignificant'],
                    subject_group=None,
                    subject_id=f"Group {plot_info['band_name']}",
                    output_dir=None,
                    verbose=verbose,
                    band_name=plot_info['band_name']
                )
                if fc_fig_inter is None or fc_fig_ipsi is None:
                    print(f"  Skipping cross-subject slow-band FC plot ({plot_info['band_name']}) due to missing data (empty matrix/labels)")
                    continue

                band_name = plot_info['band_name']
                description = plot_info['data']['description']

                fc_fig_inter.suptitle(f'{band_name} FC (Interhemispheric) - {description}', fontsize=12, fontweight='bold')
                fc_fig_ipsi.suptitle(f'{band_name} FC (Ipsilateral) - {description}', fontsize=12, fontweight='bold')

                if plot_info['save_path']:
                    save_path_inter = plot_info['save_path'].parent / f'{plot_info["save_path"].stem}_interhemispheric.svg'
                    save_path_ipsi = plot_info['save_path'].parent / f'{plot_info["save_path"].stem}_ipsilateral.svg'

                    fc_fig_inter.savefig(save_path_inter, format='svg', bbox_inches='tight', dpi=300)
                    fc_fig_ipsi.savefig(save_path_ipsi, format='svg', bbox_inches='tight', dpi=300)
                    figures_saved_count += 2

                if not show_plots:
                    plt.close(fc_fig_inter)
                    plt.close(fc_fig_ipsi)

            if show_plots and len(plot_batches['fc_slow_bands_cross_subject']) >= 20:
                print(f"  Skipping display of {len(plot_batches['fc_slow_bands_cross_subject'])} FC plots. Will be too memory exhaustive...")
            elif show_plots:
                print(f"  Displaying {len(plot_batches['fc_slow_bands_cross_subject'])} cross-subject slow-band FC plots. Close all figures to continue...")
                plt.show()

        # Batch 7: Group-Averaged FC Analysis
        if plot_batches['fc_group_avg']:
            current_plot_batch += 1
            print(f"[Batch {current_plot_batch}/{plot_batch_count}] Creating {len(plot_batches['fc_group_avg'])} group-averaged FC plots...")
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
                        p_values=None,  # Omit p-values for group averages
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
                    plot_desc = f"Group {plot_info['group_name']} Slow-{band_number}"
                    title_inter = f"Group Average: {plot_info['group_name']} - Slow-{band_number} FC (Interhemispheric, n={plot_info['data']['n_subjects']})"
                    title_ipsi = f"Group Average: {plot_info['group_name']} - Slow-{band_number} FC (Ipsilateral, n={plot_info['data']['n_subjects']})"
                else:
                    # Static FC group average
                    subject_label = f"Group_{plot_info['group_name'].replace(' ', '_')}_static"

                    fc_fig_inter, fc_fig_ipsi = plot_fc_results(
                        plot_info['data']['avg_fc_matrix'],
                        plot_info['data']['avg_fc_labels'],
                        p_values=None,  # Omit p-values for group averages
                        connectivity_patterns=connectivity_patterns,
                        channel_label_map=None,
                        mask_diagonal=plot_info['mask_diagonal'],
                        mask_nonsignificant=plot_info['mask_nonsignificant'],
                        subject_group=None,
                        subject_id=subject_label,
                        output_dir=None,
                        verbose=verbose
                    )
                    plot_desc = f"Group {plot_info['group_name']} Static"
                    title_inter = f"Group Average: {plot_info['group_name']} - Static FC (Interhemispheric, n={plot_info['data']['n_subjects']})"
                    title_ipsi = f"Group Average: {plot_info['group_name']} - Static FC (Ipsilateral, n={plot_info['data']['n_subjects']})"

                if fc_fig_inter is None or fc_fig_ipsi is None:
                    print(f"  Skipping group-averaged FC plot for {plot_desc} due to missing data (empty matrix/labels)")
                    continue

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

        # Batch 8: Interhemispheric Intra-Network Violin Plots
        if plot_batches['interhemispheric_violin']:
            current_plot_batch += 1
            print(f"[Batch {current_plot_batch}/{plot_batch_count}] Creating {len(plot_batches['interhemispheric_violin'])} interhemispheric intra-network violin plots...")

            # Suppress seaborn warnings about identical ylims
            warnings.filterwarnings('ignore', message='Attempting to set identical low and high ylims')

            for plot_info in plot_batches['interhemispheric_violin']:
                fig = plot_info['figure']
                metadata = plot_info['metadata']
                save_dir = plot_info['save_dir']

                if save_dir:
                    band_name = metadata['band_name']
                    safe_network = metadata['safe_network']

                    # Create band-specific directory
                    band_dir = save_dir / band_name
                    band_dir.mkdir(parents=True, exist_ok=True)

                    # Save figure
                    filename = f'{safe_network}_violin.png'
                    filepath = band_dir / filename
                    fig.savefig(filepath, dpi=300, bbox_inches='tight')
                    figures_saved_count += 1

                # Close figure if not showing
                if not show_plots:
                    plt.close(fig)

            if show_plots:
                print(f"  Displaying {len(plot_batches['interhemispheric_violin'])} interhemispheric violin plots. Close all figures to continue...")
                plt.show()

        # Batch 9: Ipsilateral Intra-Network Violin Plots
        if plot_batches['ipsilateral_violin']:
            current_plot_batch += 1
            print(f"[Batch {current_plot_batch}/{plot_batch_count}] Creating {len(plot_batches['ipsilateral_violin'])} ipsilateral intra-network violin plots...")

            for plot_info in plot_batches['ipsilateral_violin']:
                fig = plot_info['figure']
                metadata = plot_info['metadata']
                save_dir = plot_info['save_dir']

                if save_dir:
                    band_name = metadata['band_name']
                    safe_network = metadata['safe_network']

                    # Create band-specific directory
                    band_dir = save_dir / band_name
                    band_dir.mkdir(parents=True, exist_ok=True)

                    # Save figure
                    filename = f'{safe_network}_violin.png'
                    filepath = band_dir / filename
                    fig.savefig(filepath, dpi=300, bbox_inches='tight')
                    figures_saved_count += 1

                # Close figure if not showing
                if not show_plots:
                    plt.close(fig)

            if show_plots:
                print(f"  Displaying {len(plot_batches['ipsilateral_violin'])} ipsilateral violin plots. Close all figures to continue...")
                plt.show()

        # Batch 9: MVMD Mode Decomposition
        if plot_batches['mvmd_modes']:
            total_mode_figs = sum(p['mvmd_data']['original'].shape[0] for p in plot_batches['mvmd_modes'])
            current_plot_batch += 1
            print(f"[Batch {current_plot_batch}/{plot_batch_count}] Creating {total_mode_figs} MVMD mode decomposition plots ({len(plot_batches['mvmd_modes'])} subjects)...")

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

    # Extract grouped coherence data from statistics_data
    grouped_interhemi_coherence = {
        'static': anhedonic_stat_data['static_coherence_by_group'],
        'slow_band': anhedonic_stat_data['slow_band_coherence_by_group']
    }

    # Process each FC type separately
    for fc_type in grouped_interhemi_coherence.keys():
        print(f"\n{'─'*80}")
        print(f"FC Type: {fc_type.upper()}")
        print(f"{'─'*80}")

        if fc_type == 'static':
            # For static FC, show normal aggregated summary
            # Get all unique networks for this FC type
            all_networks = set()
            for group_data in grouped_interhemi_coherence[fc_type].values():
                all_networks.update(group_data.keys())
            all_networks = sorted(all_networks)

            print(f"Total Networks Detected: {len(all_networks)}")
            if all_networks:
                print(f"Networks: {', '.join(all_networks)}")

            # Show detailed stats for each network
            for network_key in all_networks:
                print(f"\n  Network: {network_key}")

                for group_name in ['non-anhedonic', 'low-anhedonic', 'high-anhedonic']:
                    if network_key in grouped_interhemi_coherence[fc_type][group_name]:
                        network_data = grouped_interhemi_coherence[fc_type][group_name][network_key]

                        # Static FC: direct access
                        values = np.array(network_data['observed_values'])
                        subjects_count = network_data['n_subjects']
                        mean_coh = network_data['mean_coherence']
                        std_coh = network_data['std_coherence']

                        if len(values) > 0:
                            min_z = np.min(values)
                            max_z = np.max(values)
                            print(f"    {group_name:20s}: Mean = {mean_coh:7.3f}, SD = {std_coh:6.3f}, "
                                  f"Range = [{min_z:7.3f}, {max_z:7.3f}], N = {subjects_count:3d}")
                        else:
                            print(f"    {group_name:20s}: NO VALID SUBJECTS")
                    else:
                        print(f"    {group_name:20s}: NETWORK NOT PRESENT")
        else:
            # For slow-band FC, show per-band breakdown
            # Get all unique networks and bands for this FC type
            all_networks = set()
            all_bands = set()
            for group_data in grouped_interhemi_coherence[fc_type].values():
                for network_key, network_data in group_data.items():
                    all_networks.add(network_key)
                    all_bands.update(network_data.keys())

            all_networks = sorted(all_networks)
            all_bands = sorted(all_bands)

            print(f"Total Networks Detected: {len(all_networks)}")
            if all_networks:
                print(f"Networks: {', '.join(all_networks)}")
            print(f"Slow Bands Available: {', '.join(all_bands)}")

            # Show detailed stats for each network, broken down by band
            for network_key in all_networks:
                print(f"\n  Network: {network_key}")

                for band_name in all_bands:
                    print(f"    {band_name}:")

                    for group_name in ['non-anhedonic', 'low-anhedonic', 'high-anhedonic']:
                        if (network_key in grouped_interhemi_coherence[fc_type][group_name] and
                            band_name in grouped_interhemi_coherence[fc_type][group_name][network_key]):

                            band_data = grouped_interhemi_coherence[fc_type][group_name][network_key][band_name]
                            values = np.array(band_data['observed_values'])
                            subjects_count = band_data['n_subjects']
                            mean_coh = band_data['mean_coherence']
                            std_coh = band_data['std_coherence']

                            if len(values) > 0:
                                min_z = np.min(values)
                                max_z = np.max(values)
                                print(f"      {group_name:18s}: Mean = {mean_coh:7.3f}, SD = {std_coh:6.3f}, "
                                      f"Range = [{min_z:7.3f}, {max_z:7.3f}], N = {subjects_count:3d}")
                            else:
                                print(f"      {group_name:18s}: NO VALID SUBJECTS")
                        else:
                            print(f"      {group_name:18s}: NO DATA FOR THIS BAND")

    # Compute basic statistics from grouped coherence data
    observed_test_statistics = {}
    for fc_type in grouped_interhemi_coherence.keys():
        observed_test_statistics[fc_type] = {}

        # Get all networks for this FC type
        all_networks = set()
        for group_data in grouped_interhemi_coherence[fc_type].values():
            all_networks.update(group_data.keys())

        for network_key in all_networks:
            # Collect values per group for this network
            group_means = {}
            group_sds = {}
            group_sizes = {}

            for group_name in ['non-anhedonic', 'low-anhedonic', 'high-anhedonic']:
                if network_key in grouped_interhemi_coherence[fc_type][group_name]:
                    network_data = grouped_interhemi_coherence[fc_type][group_name][network_key]

                    if fc_type == 'static':
                        values = np.array(network_data['observed_values'])
                    else:
                        # Aggregate slow-band values
                        values = []
                        for band_data in network_data.values():
                            values.extend(band_data['observed_values'])
                        values = np.array(values)

                    if len(values) > 0:
                        group_means[group_name] = np.mean(values)
                        group_sds[group_name] = np.std(values, ddof=1) if len(values) > 1 else 0.0
                        group_sizes[group_name] = len(values)
                    else:
                        group_means[group_name] = np.nan
                        group_sds[group_name] = np.nan
                        group_sizes[group_name] = 0
                else:
                    group_means[group_name] = np.nan
                    group_sds[group_name] = np.nan
                    group_sizes[group_name] = 0

            observed_test_statistics[fc_type][network_key] = {
                'group_means': group_means,
                'group_sds': group_sds,
                'group_sizes': group_sizes,
                'anova': {'note': 'Not computed'},
                'pairwise': {}
            }

    # Test statistics detailed summary
    print(f"\n{'='*80}")
    print(f"OBSERVED TEST STATISTICS SUMMARY")
    print(f"{'='*80}")

    for fc_type in observed_test_statistics.keys():
        print(f"\n{'─'*80}")
        print(f"FC Type: {fc_type.upper()}")
        print(f"{'─'*80}")

        if fc_type == 'slow_band':
            # For slow-band, show per-band breakdown instead of aggregated
            # Get all networks and bands from the original data structure
            slow_band_data = grouped_interhemi_coherence['slow_band']
            all_networks_fc = set()
            all_bands = set()

            for group_data in slow_band_data.values():
                for network_key, network_data in group_data.items():
                    all_networks_fc.add(network_key)
                    all_bands.update(network_data.keys())

            all_networks_fc = sorted(all_networks_fc)
            all_bands = sorted(all_bands)

            print(f"Showing per-band breakdown for {len(all_bands)} bands: {', '.join(all_bands)}")

            for network_key in all_networks_fc:
                print(f"\n  {network_key}:")

                for band_name in all_bands:
                    print(f"    {band_name}:")

                    for group_name in ['non-anhedonic', 'low-anhedonic', 'high-anhedonic']:
                        if (network_key in slow_band_data[group_name] and
                            band_name in slow_band_data[group_name][network_key]):

                            band_data = slow_band_data[group_name][network_key][band_name]
                            values = np.array(band_data['observed_values'])

                            if len(values) > 0:
                                mean_val = np.mean(values)
                                sd_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
                                val_count = len(values)
                                print(f"      {group_name:20s}: M = {mean_val:7.3f}, SD = {sd_val:6.3f}, N = {val_count:3d}")
                            else:
                                print(f"      {group_name:20s}: No data")
                        else:
                            print(f"      {group_name:20s}: No data")
        else:
            # For static FC, show normal aggregated stats
            all_networks_fc = sorted(observed_test_statistics[fc_type].keys())

            for network_key in all_networks_fc:
                stats = observed_test_statistics[fc_type][network_key]

                print(f"\n  {network_key}:")
                print(f"    Group Means (Fisher-Z):")
                for group_name in ['non-anhedonic', 'low-anhedonic', 'high-anhedonic']:
                    mean_val = stats['group_means'].get(group_name, np.nan)
                    sd_val = stats['group_sds'].get(group_name, np.nan)
                    val_count = stats['group_sizes'].get(group_name, 0)
                    if not np.isnan(mean_val):
                        print(f"      {group_name:20s}: M = {mean_val:7.3f}, SD = {sd_val:6.3f}, N = {val_count:3d}")
                    else:
                        print(f"      {group_name:20s}: No data")

                # ANOVA results
                if 'anova' in stats:
                    anova = stats['anova']
                    if not np.isnan(anova.get('F_statistic', np.nan)):
                        print(f"    ANOVA: F({anova['n_groups_compared']-1},{sum(stats['group_sizes'].values())-anova['n_groups_compared']}) = "
                              f"{anova['F_statistic']:.3f}, p = {anova['p_value']:.4f} "
                              f"{'***' if anova['p_value'] < 0.001 else '**' if anova['p_value'] < 0.01 else '*' if anova['p_value'] < 0.05 else 'ns'}")
                    else:
                        print(f"    ANOVA: {anova.get('note', 'Not computed')}")

                # Pairwise comparisons
                if 'pairwise' in stats and len(stats['pairwise']) > 0:
                    print(f"    Pairwise comparisons:")
                    for comparison_name, comparison_data in stats['pairwise'].items():
                        t_stat = comparison_data['t_statistic']
                        p_val = comparison_data['p_value']
                        mean_diff = comparison_data['mean_diff']
                        sig_marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                        print(f"      {comparison_name:45s}: t = {t_stat:7.3f}, p = {p_val:.4f} {sig_marker:3s} (Δ = {mean_diff:7.3f})")

    # Validation warnings
    print(f"\n{'='*80}")
    print(f"VALIDATION CHECKS")
    print(f"{'='*80}")

    warnings_found = False
    for fc_type in grouped_interhemi_coherence.keys():
        for group_name, networks in grouped_interhemi_coherence[fc_type].items():
            for network_key, network_data in networks.items():
                if fc_type == 'static':
                    values = np.array(network_data['observed_values'])
                else:
                    # Slow-band: aggregate all band values
                    values = []
                    for band_data in network_data.values():
                        values.extend(band_data['observed_values'])
                    values = np.array(values)

                # Check for extreme values (|z| > 2.0 unusual but valid)
                if len(values) > 0:
                    extreme_mask = np.abs(values) > 2.0
                    if np.any(extreme_mask):
                        warnings_found = True
                        extreme_count = np.sum(extreme_mask)
                        extreme_vals = values[extreme_mask]
                        print(f"⚠ {fc_type}/{group_name}/{network_key}: {extreme_count} subjects with |Fisher-Z| > 2.0")
                        print(f"  Extreme values: {extreme_vals}")

                # Check for networks with few subjects
                if len(values) < 5 and len(values) > 0:
                    warnings_found = True
                    print(f"⚠ {fc_type}/{group_name}/{network_key}: Only {len(values)} subjects (statistical power may be low)")

                # Check for excluded subjects (if tracking is available)
                if 'n_excluded_subjects' in network_data and network_data['n_excluded_subjects'] > 0:
                    warnings_found = True
                    print(f"ℹ {fc_type}/{group_name}/{network_key}: {network_data['n_excluded_subjects']} subjects excluded (no valid parcel pairs)")

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
                'fc_types': list(grouped_interhemi_coherence.keys()),
                'n_fc_types': len(grouped_interhemi_coherence)
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
        # Create groups configuration for logging
        groups_config = [
            ('Low Anhedonic', low_anhedonic_to_process),
            ('High Anhedonic', high_anhedonic_to_process),
            ('Non-anhedonic', non_anhedonic_to_process)
        ]

        log_file = write_analysis_log(
            output_dir=run_parent_dir,
            groups_config=groups_config,
            all_results=all_subject_results,
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

    # ===== EXTRACT FC RESULTS FOR GROUP ANALYSIS =====
    # Extract successful FC results for each group
    anhedonic_fc_results = [result for result in anhedonic_results.values() if result.get('success', False)]
    non_anhedonic_fc_results = [result for result in non_anhedonic_results.values() if result.get('success', False)]

    # Group comparison results using ANOVA testing
    group_comparison_results = {
        'anova_testing': anhedonic_anova_results,
        'statistics_data': anhedonic_stat_data,
        'group_averaged_fc': group_averaged_fc
    }

    # ===== RETURN MULTI-SUBJECT RESULTS =====
    return {
        'anhedonic_subjects': anhedonic_subjects_to_process,
        'non_anhedonic_subjects': non_anhedonic_to_process,
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
