#!/usr/bin/env python3
"""
Functional Connectivity Analysis Script

This script performs STATIC functional connectivity analysis on TCP dataset timeseries.
Static FC computes Pearson correlations between ROI timeseries across the entire session.

TODO: Dynamic FC analysis (binned/windowed correlations) is not yet implemented.

Author: Ian Philip Eglin
"""

import csv
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

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import signal, stats

from config.paths import get_analysis_path
from tcp.processing import DataLoader, SubjectManager
from tcp.processing.roi import (
    CorticalAtlasLookup,
    ROIExtractionService,
    SubCorticalAtlasLookup,
)
from tcp.processing.signal_processing.mvmd import MVMD


def is_actual_file(file_path: Path) -> bool:
    """
    Check if a file is actually downloaded (not a git-annex symlink).

    Works cross-platform (Windows, macOS, Linux) by checking:
    1. File exists
    2. If it's a symlink, verify the target exists
    3. File has actual content (size > 1KB, symlinks are tiny)

    Args:
        file_path: Path to check

    Returns:
        True if file is actually available for reading, False if it's a symlink stub
    """
    if not file_path.exists():
        return False

    # Check if it's a symlink
    if file_path.is_symlink():
        # On Windows, symlinks might point to git-annex objects
        try:
            # Check if symlink target exists and is accessible
            resolved = file_path.resolve(strict=True)
            # Verify it's not just a symlink to an annex object path
            if '.git/annex/objects' in str(resolved):
                # This is a git-annex symlink, check if target actually exists
                if not resolved.exists():
                    return False
        except (OSError, RuntimeError):
            return False

    # Check file size - git-annex symlinks are very small (<1KB)
    # Real H5 files should be much larger
    try:
        size = file_path.stat().st_size
        if size < 1024:  # Less than 1KB = likely a symlink or empty
            return False
    except OSError:
        return False

    return True


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


def fisher_r_to_z(r_matrix):
    """
    Apply Fisher r-to-z transformation to correlation matrix.

    The Fisher transformation normalizes the sampling distribution of correlation
    coefficients, making them suitable for averaging and statistical testing.

    Formula: z = 0.5 * ln((1 + r) / (1 - r)) = arctanh(r)

    Args:
        r_matrix: Correlation matrix (Pearson r values)

    Returns:
        z_matrix: Fisher z-transformed matrix
    """
    # Clip values to avoid infinity at r = ±1
    # Use conservative bounds to maintain numerical stability
    r_clipped = np.clip(r_matrix, -0.9999, 0.9999)

    # Apply Fisher transformation (arctanh is more numerically stable than the log form)
    z_matrix = np.arctanh(r_clipped)

    return z_matrix


def fisher_z_to_r(z_matrix):
    """
    Apply inverse Fisher transformation to convert z-scores back to correlations.

    Formula: r = (exp(2z) - 1) / (exp(2z) + 1) = tanh(z)

    Args:
        z_matrix: Fisher z-transformed matrix

    Returns:
        r_matrix: Correlation matrix (Pearson r values)
    """
    # Apply inverse Fisher transformation (tanh is more numerically stable)
    r_matrix = np.tanh(z_matrix)

    return r_matrix


def export_static_fc_results_to_csv(static_fc_results, subject_id, output_dir):
    """
    Export STATIC functional connectivity results to CSV files for later analysis.

    Creates three CSV files per subject:
    1. {subject_id}_static_fc_matrix.csv - Full correlation matrix
    2. {subject_id}_static_fc_pvalues.csv - P-values matrix
    3. {subject_id}_static_fc_pairwise.csv - Pairwise connections with metadata

    NOTE: This exports STATIC FC (whole-session correlations).
    Dynamic FC (time-windowed) is not yet implemented.

    Args:
        static_fc_results: Static FC results dictionary from process_subject
        subject_id: Subject identifier
        output_dir: Directory to save CSV files (Path object or string)

    Returns:
        dict: Paths to created CSV files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not static_fc_results or 'static_fc_matrix' not in static_fc_results:
        print(f"[WARNING] No static FC results to export for {subject_id}")
        return None

    fc_matrix = static_fc_results['static_fc_matrix']
    fc_labels = static_fc_results['static_fc_labels']
    fc_pvalues = static_fc_results.get('static_fc_pvalues')
    connectivity_patterns = static_fc_results.get('static_connectivity_patterns', {})
    channel_label_map = static_fc_results.get('channel_label_map', {})

    created_files = {}

    # 1. Export static FC correlation matrix
    matrix_file = output_dir / f'{subject_id}_static_fc_matrix.csv'
    with open(matrix_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header row with channel labels
        display_labels = [channel_label_map.get(label, label) for label in fc_labels]
        writer.writerow([''] + display_labels)

        # Data rows
        for i, row_label in enumerate(display_labels):
            writer.writerow([row_label] + list(fc_matrix[i, :]))

    created_files['fc_matrix'] = matrix_file
    print(f"  Exported FC matrix to: {matrix_file}")

    # 2. Export p-values matrix (if available)
    if fc_pvalues is not None:
        pvalues_file = output_dir / f'{subject_id}_static_fc_pvalues.csv'
        with open(pvalues_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header row
            display_labels = [channel_label_map.get(label, label) for label in fc_labels]
            writer.writerow([''] + display_labels)

            # Data rows
            for i, row_label in enumerate(display_labels):
                writer.writerow([row_label] + list(fc_pvalues[i, :]))

        created_files['fc_pvalues'] = pvalues_file
        print(f"  Exported p-values to: {pvalues_file}")

    # 3. Export pairwise connections with metadata
    pairwise_file = output_dir / f'{subject_id}_static_fc_pairwise.csv'
    with open(pairwise_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'ROI_1', 'ROI_2',
            'ROI_1_label', 'ROI_2_label',
            'Correlation', 'P_value', 'Significant',
            'Connection_type', 'Regions', 'Hemispheres'
        ])

        # Extract all pairwise data (using new 'pairs' structure)
        all_pairwise = connectivity_patterns.get('all_pairwise', {}).get('pairs', {})

        for pair_key, pair_data in all_pairwise.items():
            # Parse pair key (e.g., 'vmPFC_RH_ch0_vmPFC_LH_ch2')
            parts = pair_key.split('_')

            # Try to reconstruct ROI names
            if len(parts) >= 6:
                roi1_key = '_'.join(parts[:3])
                roi2_key = '_'.join(parts[3:])
            else:
                roi1_key = parts[0] if len(parts) > 0 else 'unknown'
                roi2_key = parts[1] if len(parts) > 1 else 'unknown'

            # Get descriptive labels
            roi1_label = channel_label_map.get(roi1_key, roi1_key)
            roi2_label = channel_label_map.get(roi2_key, roi2_key)

            # Determine connection type
            connection_type = 'other'
            regions = ''
            hemispheres = ''

            # Check in connectivity patterns (using new 'pairs' structure)
            if pair_key in connectivity_patterns.get('interhemispheric', {}).get('pairs', {}):
                connection_type = 'interhemispheric'
                regions = connectivity_patterns['interhemispheric']['pairs'][pair_key].get('region', '')
            elif pair_key in connectivity_patterns.get('cross_regional', {}).get('pairs', {}):
                connection_type = 'cross_regional'
                regions = connectivity_patterns['cross_regional']['pairs'][pair_key].get('regions', '')
                if pair_key in connectivity_patterns.get('ipsilateral', {}).get('pairs', {}):
                    connection_type = 'ipsilateral_cross_regional'
                    hemispheres = connectivity_patterns['ipsilateral']['pairs'][pair_key].get('hemisphere', '')
                elif pair_key in connectivity_patterns.get('contralateral', {}).get('pairs', {}):
                    connection_type = 'contralateral_cross_regional'
                    hemispheres = connectivity_patterns['contralateral']['pairs'][pair_key].get('hemispheres', '')

            # Write row
            writer.writerow([
                roi1_key, roi2_key,
                roi1_label, roi2_label,
                pair_data['correlation'],
                pair_data.get('p_value', ''),
                pair_data.get('significant', ''),
                connection_type,
                regions,
                hemispheres
            ])

    created_files['fc_pairwise'] = pairwise_file
    print(f"  Exported pairwise connections to: {pairwise_file}")

    return created_files


def compute_group_averaged_fc(subject_results, group_subjects, group_name, fc_type='static', band_key=None, verbose=False):
    """
    Compute group-averaged functional connectivity matrices.

    Args:
        subject_results: Dictionary of all subject results
        group_subjects: List of subject IDs belonging to this group
        group_name: Name of the group (e.g., 'Non-anhedonic', 'Low Anhedonic', 'High Anhedonic')
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


def export_group_averaged_fc_to_csv(group_avg_data, output_dir, verbose=False):
    """
    Export group-averaged FC matrices and interhemispheric correlations to CSV.

    Args:
        group_avg_data: Group-averaged FC data from compute_group_averaged_fc()
        output_dir: Directory to save CSV files
        verbose: Print progress messages

    Returns:
        dict: Paths to created CSV files
    """
    from pathlib import Path

    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    group_name = group_avg_data['group_name']
    fc_type = group_avg_data['fc_type']
    band_key = group_avg_data.get('band_key', '')

    # Create filename prefix
    group_name_clean = group_name.replace(' ', '_').replace('-', '_').lower()
    if band_key:
        filename_prefix = f"group_avg_{group_name_clean}_{band_key}"
    else:
        filename_prefix = f"group_avg_{group_name_clean}_static"

    created_files = {}

    # 1. Export averaged correlation matrix
    matrix_file = output_dir / f"{filename_prefix}_fc_matrix.csv"
    matrix_df = pd.DataFrame(
        group_avg_data['avg_fc_matrix'],
        index=group_avg_data['avg_fc_labels'],
        columns=group_avg_data['avg_fc_labels']
    )
    matrix_df.to_csv(matrix_file)
    created_files['matrix'] = matrix_file

    if verbose:
        print(f"    Exported: {matrix_file.name}")

    # 2. Export metadata
    metadata_file = output_dir / f"{filename_prefix}_metadata.csv"
    metadata_df = pd.DataFrame({
        'group_name': [group_name],
        'n_subjects': [group_avg_data['n_subjects']],
        'fc_type': [fc_type],
        'band_key': [band_key if band_key else 'N/A'],
        'subject_ids': [','.join(group_avg_data['subject_ids'])]
    })
    metadata_df.to_csv(metadata_file, index=False)
    created_files['metadata'] = metadata_file

    if verbose:
        print(f"    Exported: {metadata_file.name}")

    return created_files


def write_analysis_log(output_dir, groups_config, all_results, low_anhedonic_subjects, high_anhedonic_subjects, timestamp=None):
    """
    Write comprehensive analysis log file tracking all analyzed subjects organized by groups.

    Args:
        output_dir: Directory to save log file
        groups_config: List of tuples (group_name, subject_ids)
        all_results: Dictionary mapping subject_id to processing results
        low_anhedonic_subjects: List of low anhedonic subject IDs
        high_anhedonic_subjects: List of high anhedonic subject IDs
        timestamp: Optional timestamp string for log filename

    Returns:
        Path to created log file
    """
    from datetime import datetime
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    log_file = output_dir / f'analysis_log_{timestamp}.txt'

    with open(log_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FUNCTIONAL CONNECTIVITY ANALYSIS LOG\n")
        f.write("="*80 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Log File: {log_file.name}\n\n")

        total_attempted = len(all_results)
        total_success = sum(1 for r in all_results.values() if r.get('success'))
        total_failed = total_attempted - total_success

        f.write("="*80 + "\n")
        f.write("OVERALL SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total subjects attempted: {total_attempted}\n")
        f.write(f"Successfully processed: {total_success}\n")
        f.write(f"Failed: {total_failed}\n")
        f.write(f"Success rate: {total_success/total_attempted*100:.1f}%\n\n")

        f.write("="*80 + "\n")
        f.write("SUBJECTS BY GROUP\n")
        f.write("="*80 + "\n\n")

        for group_name, group_subjects in groups_config:
            successful_in_group = [sid for sid in group_subjects if all_results.get(sid, {}).get('success')]
            failed_in_group = [sid for sid in group_subjects if sid in all_results and not all_results[sid].get('success')]

            f.write(f"{group_name.upper()}\n")
            f.write(f"{'-'*80}\n")
            f.write(f"Total in group: {len(group_subjects)}\n")
            f.write(f"Successfully processed: {len(successful_in_group)}\n")
            f.write(f"Failed: {len(failed_in_group)}\n\n")

            if successful_in_group:
                f.write(f"Successfully processed subjects ({len(successful_in_group)}):\n")
                for sid in sorted(successful_in_group):
                    subgroup = ""
                    if sid in low_anhedonic_subjects:
                        subgroup = " [LOW-ANHEDONIC]"
                    elif sid in high_anhedonic_subjects:
                        subgroup = " [HIGH-ANHEDONIC]"
                    f.write(f"  - {sid}{subgroup}\n")
                f.write("\n")

            if failed_in_group:
                f.write(f"Failed subjects ({len(failed_in_group)}):\n")
                for sid in sorted(failed_in_group):
                    error_msg = all_results[sid].get('error', 'Unknown error')
                    f.write(f"  - {sid}: {error_msg}\n")
                f.write("\n")

            f.write("\n")

        f.write("="*80 + "\n")
        f.write("GROUP AVERAGES - INCLUDED SUBJECTS\n")
        f.write("="*80 + "\n\n")

        for group_name, group_subjects in groups_config:
            successful_in_group = [sid for sid in group_subjects if all_results.get(sid, {}).get('success')]

            f.write(f"{group_name.upper()} (N={len(successful_in_group)})\n")
            f.write(f"{'-'*80}\n")

            if successful_in_group:
                for sid in sorted(successful_in_group):
                    subgroup = ""
                    if sid in low_anhedonic_subjects:
                        subgroup = " [LOW-ANHEDONIC]"
                    elif sid in high_anhedonic_subjects:
                        subgroup = " [HIGH-ANHEDONIC]"
                    f.write(f"  {sid}{subgroup}\n")
            else:
                f.write("  None\n")

            f.write("\n")

        f.write("="*80 + "\n")
        f.write("END OF LOG\n")
        f.write("="*80 + "\n")

    return log_file


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
                    results['interhemispheric']['pairs'][pair_key] = {
                        'correlation': corr_val,
                        'p_value': p_val,
                        'significant': is_significant,
                        'region': roi1_region
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

    return results

def combine_slow_band_components(time_modes, center_freqs, verbose=True):
    """
    Combine all signal components within specific slow bands into multi-component signals.

    Args:
        time_modes: Time signals from which the original is a superposition of (modes x channels x samples)
        center_freqs: Center frequencies of all modes, used to decide which modes to combine

    Bands:
        Slow-6: 0–0.01Hz
        Slow-5: 0.01–0.027Hz
        Slow-4: 0.027–0.073Hz
        Slow-3: 0.073–0.198Hz
        Slow-2: 0.198–0.25Hz

    Returns:
        dict: Dictionary with band names as keys, each containing:
            - 'band_signal': Combined signal for the band (sum of all components)
            - 'components': List of individual mode signals in this band
            - 'indeces': List of mode indices that belong to this band
    """
    # Initialize separate dictionaries for each band to avoid shared reference issue
    band_signals = {
        '2': {'band_signal': None, 'components': [], 'indeces': [], 'center_freqs': []},
        '3': {'band_signal': None, 'components': [], 'indeces': [], 'center_freqs': []},
        '4': {'band_signal': None, 'components': [], 'indeces': [], 'center_freqs': []},
        '5': {'band_signal': None, 'components': [], 'indeces': [], 'center_freqs': []},
        '6': {'band_signal': None, 'components': [], 'indeces': [], 'center_freqs': []},
        'excluded': {'band_signal': None, 'components': [], 'indeces': [], 'center_freqs': []},
    }

    def get_band_number(frequency):
        if 0.0 < frequency <= 0.01:
            return "6"
        elif 0.01 < frequency <= 0.027:
            return "5"
        elif 0.027 < frequency <= 0.073:
            return "4"
        elif 0.073 < frequency <= 0.198:
            return "3"
        elif 0.198 < frequency <= 0.25:
            return "2"
        else:
            return 'excluded'

    for idx, (center_frequency, mode_signal) in enumerate(zip(center_freqs, time_modes)):
        band_number = get_band_number(center_frequency)

        band_signals[band_number]['indeces'].append(idx)
        band_signals[band_number]['components'].append(mode_signal)
        band_signals[band_number]['center_freqs'].append(center_frequency)

    # Combine components within each band and convert to arrays
    for key, val in band_signals.items():
        if len(val['components']) > 0:
            comps_array = np.array(val['components'])
            idcs_array = np.array(val['indeces'])
            freqs_array = np.array(val['center_freqs'])

            band_signals[key]['components'] = comps_array
            band_signals[key]['indeces'] = idcs_array
            band_signals[key]['center_freqs'] = freqs_array

            # Only sum components for actual slow bands, not for excluded frequencies
            if key != 'excluded':
                # Sum all components in this band to create the band signal
                band_signal = np.sum(comps_array, axis=0)
                band_signals[key]['band_signal'] = band_signal

                if verbose:
                    print(f'Band Slow-{key}: components={comps_array.shape}, indeces={idcs_array}, center_freqs={freqs_array}, band_signal={band_signal.shape}')
            else:
                # Keep excluded components separate (do not sum them)
                band_signals[key]['band_signal'] = None

                if verbose:
                    print(f'Outside of bands: components={comps_array.shape}, indeces={idcs_array}, center_freqs={freqs_array}, signals kept separate (not summed)')
        elif verbose:
            print(f'Band Slow-{key}: no components in this band')

    return band_signals


def get_frequency_range(band_key):
    """
    Return frequency range string for slow-band visualization.

    Args:
        band_key: str, band identifier ('2', '3', '4', '5', or '6')

    Returns:
        str: Frequency range in Hz (e.g., '0.027-0.073 Hz')
    """
    ranges = {
        '6': '0.000-0.010 Hz',
        '5': '0.010-0.027 Hz',
        '4': '0.027-0.073 Hz',
        '3': '0.073-0.198 Hz',
        '2': '0.198-0.250 Hz',
    }
    return ranges.get(band_key, 'Unknown')


def detect_available_channels(band_signal, threshold=1e-10):
    """
    Detect which channels have valid timeseries for a given slow-band.

    Uses RMS (root mean square) energy to identify channels with meaningful
    signal content. Channels with RMS below threshold are marked unavailable.

    Args:
        band_signal: ndarray, shape (n_channels, n_samples)
            Reconstructed slow-band signal for all channels
        threshold: float, default=1e-10
            RMS threshold for channel availability

    Returns:
        available_mask: ndarray, shape (n_channels,)
            Boolean mask where True indicates channel is available
    """
    n_channels = band_signal.shape[0]
    available_mask = np.zeros(n_channels, dtype=bool)

    for ch_idx in range(n_channels):
        rms = np.sqrt(np.mean(band_signal[ch_idx, :]**2))
        available_mask[ch_idx] = (rms >= threshold)

    return available_mask


def plot_roi_timeseries_result(roi_extraction_results, subject_id=None, atlas_type=''):
    """
    Create visualization for ROI timeseries from the TCP dataset.
    Plots all individual channels separately, organized by region and hemisphere.

    Args:
        roi_extraction_results: Results from cortical, or subcortical roi extraction from the process_subject() function
        subject_id: Optional subject identifier to add to figure title
        atlas_type: Optional string to specify atlas type ('Cortical' or 'Subcortical')

    Example:
        cortical': {
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
    """

    # Something wrong happened. Return an empty figure
    if not roi_extraction_results.get('extraction_successful'):
        fig = plt.figure(figsize=(18, 11))
        return fig

    roi_timeseries = roi_extraction_results.get('roi_timeseries')
    requested_rois = roi_extraction_results.get('requested_rois')
    atlas_name = roi_extraction_results.get('atlas_name', 'Unknown Atlas')
    parcel_labels = roi_extraction_results.get('parcel_labels', {})

    right_hemisphere_ts = roi_extraction_results.get('hemisphere_specific', {}).get('right_hemisphere')
    left_hemisphere_ts = roi_extraction_results.get('hemisphere_specific', {}).get('left_hemisphere')
    supports_hemisphere = roi_extraction_results.get('hemisphere_specific', {}).get('supports_hemisphere_queries', False)

    # Only show hemisphere-specific data for cleaner visualization
    has_hemisphere_data = supports_hemisphere and right_hemisphere_ts and left_hemisphere_ts

    if not has_hemisphere_data:
        fig = plt.figure(figsize=(16, 6))
        fig.text(0.5, 0.5, 'No hemisphere-specific data available',
                ha='center', va='center', fontsize=14, color='gray')
        return fig

    # Create separate figures for each ROI (region) to keep plots organized
    figures = []

    for roi_name in requested_rois:
        if roi_name not in right_hemisphere_ts or roi_name not in left_hemisphere_ts:
            continue

        ts_right = right_hemisphere_ts[roi_name]
        ts_left = left_hemisphere_ts[roi_name]

        # Ensure 2D format (channels x timepoints)
        if ts_right.ndim == 1:
            ts_right = ts_right.reshape(1, -1)
        if ts_left.ndim == 1:
            ts_left = ts_left.reshape(1, -1)

        n_channels_right = ts_right.shape[0]
        n_channels_left = ts_left.shape[0]

        # Get parcel labels for this ROI (if available)
        roi_parcel_labels = parcel_labels.get(roi_name, {})
        right_labels = roi_parcel_labels.get('RH', roi_parcel_labels.get('rh', []))
        left_labels = roi_parcel_labels.get('LH', roi_parcel_labels.get('lh', []))

        # Create figure with 2 columns (right and left hemisphere)
        n_rows = max(n_channels_right, n_channels_left)
        fig, axes = plt.subplots(n_rows, 2, figsize=(16, 2.5 * n_rows), sharex=True)

        # Handle single row case
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        # Plot right hemisphere channels
        for ch_idx in range(n_channels_right):
            ax = axes[ch_idx, 0]
            ax.plot(ts_right[ch_idx], color='#d62728', linewidth=1.5, alpha=0.85)
            ax.set_ylabel('Signal', fontsize=10)

            # Use descriptive label if available, otherwise fall back to generic label
            if ch_idx < len(right_labels):
                title = f'{right_labels[ch_idx]} (Ch {ch_idx+1})'
            else:
                title = f'{roi_name} RH - Channel {ch_idx+1}'

            ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Hide unused right hemisphere subplots
        for ch_idx in range(n_channels_right, n_rows):
            axes[ch_idx, 0].axis('off')

        # Plot left hemisphere channels
        for ch_idx in range(n_channels_left):
            ax = axes[ch_idx, 1]
            ax.plot(ts_left[ch_idx], color='#1f77b4', linewidth=1.5, alpha=0.85)
            ax.set_ylabel('Signal', fontsize=10)

            # Use descriptive label if available, otherwise fall back to generic label
            if ch_idx < len(left_labels):
                title = f'{left_labels[ch_idx]} (Ch {ch_idx+1})'
            else:
                title = f'{roi_name} LH - Channel {ch_idx+1}'

            ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Hide unused left hemisphere subplots
        for ch_idx in range(n_channels_left, n_rows):
            axes[ch_idx, 1].axis('off')

        # Set x-label on the bottom plots
        axes[-1, 0].set_xlabel('Time (volumes)', fontsize=12)
        axes[-1, 1].set_xlabel('Time (volumes)', fontsize=12)

        # Add column titles with more spacing to prevent overlap with subplot titles
        axes[0, 0].text(0.5, 1.25, 'Right Hemisphere', transform=axes[0, 0].transAxes,
                       ha='center', va='bottom', fontsize=13, fontweight='bold')
        axes[0, 1].text(0.5, 1.25, 'Left Hemisphere', transform=axes[0, 1].transAxes,
                       ha='center', va='bottom', fontsize=13, fontweight='bold')

        # Add subject ID and atlas type to figure suptitle
        title_parts = []
        if atlas_type:
            title_parts.append(f'{atlas_type} ROI')
        title_parts.append(f'{roi_name} Timeseries')
        if subject_id:
            title_parts.append(f'({subject_id})')

        # Reduce suptitle y position to bring it closer to subplot titles (reduce gap by ~50%)
        fig.suptitle(' - '.join(title_parts), fontsize=16, fontweight='bold', y=0.98)

        # Use tight_layout with less space reserved for suptitle (reduce gap by ~37.5%)
        plt.tight_layout(pad=2.0, h_pad=2.5, rect=[0, 0, 1, 0.97])
        figures.append(fig)

    # Return all figures (or create empty if none)
    if figures:
        return figures
    else:
        fig = plt.figure(figsize=(16, 6))
        fig.text(0.5, 0.5, 'No matching ROIs found',
                ha='center', va='center', fontsize=14, color='gray')
        return [fig]


def plot_timeseries_with_envelopes(analytic_signal, analytic_envelope, smoothed_envelope, channel_labels, subject_id=None, envelope_type='raw'):
    """
    Plot analytic signal with their envelopes (either raw or LP-filtered).

    Args:
        analytic_signal: Array of analytic (complex-valued) timeseries data (n_channels x n_timepoints)
        analytic_envelope: Raw envelope from Hilbert transform
        smoothed_envelope: LP-filtered envelope
        channel_labels: List of channel label strings
        subject_id: Optional subject identifier
        envelope_type: 'raw' for analytic envelope, 'filtered' for LP-filtered envelope

    Returns:
        matplotlib.figure.Figure: Figure with analytic signal and envelope plots
    """
    n_channels = analytic_signal.shape[0]
    n_cols = 2  # Plot 2 channels per row
    n_rows = int(np.ceil(n_channels / n_cols))

    # Choose which envelope to plot
    envelope_to_plot = analytic_envelope if envelope_type == 'raw' else smoothed_envelope
    envelope_label = 'Raw Envelope' if envelope_type == 'raw' else 'LP-Filtered Envelope'

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows), sharex=True)

    # Handle single row case
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for ch_idx in range(n_channels):
        row = ch_idx // n_cols
        col = ch_idx % n_cols
        ax = axes[row, col]

        # Get channel label
        if ch_idx < len(channel_labels):
            label = channel_labels[ch_idx]
        else:
            label = f'Channel {ch_idx + 1}'

        # Plot real part of analytic signal
        ax.plot(np.real(analytic_signal[ch_idx]), color='steelblue', linewidth=0.8, alpha=0.7, label='Real(Analytic Signal)')

        # Plot envelope
        ax.plot(envelope_to_plot[ch_idx], color='crimson', linewidth=1.5, alpha=0.8, label=envelope_label)
        ax.plot(-envelope_to_plot[ch_idx], color='crimson', linewidth=1.5, alpha=0.8)

        ax.set_ylabel('Amplitude', fontsize=9)
        ax.set_title(f'{label}', fontsize=10, fontweight='bold', pad=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide unused subplots
    for ch_idx in range(n_channels, n_rows * n_cols):
        row = ch_idx // n_cols
        col = ch_idx % n_cols
        axes[row, col].axis('off')

    # Set x-label on bottom plots
    for col in range(n_cols):
        if n_rows * n_cols - n_cols + col < n_channels:
            axes[-1, col].set_xlabel('Time (volumes)', fontsize=11)

    # Add figure title
    title_parts = ['Timeseries with', envelope_label]
    if subject_id:
        title_parts.append(f'({subject_id})')

    fig.suptitle(' - '.join(title_parts), fontsize=16, fontweight='bold', y=0.995)

    # Use tight_layout with padding
    plt.tight_layout(pad=2.0, h_pad=2.5, rect=[0, 0, 1, 0.99])

    return fig


def plot_fc_results(corr_matrix, roi_labels, p_values=None, connectivity_patterns=None, channel_label_map=None, alpha=0.05, mask_diagonal=False, mask_nonsignificant=False, subject_group=None, subject_id=None, output_dir=None, verbose=False, band_name=None, frequency_range=None, n_available_channels=None):
    """
    Create two separate visualizations:
    1. FC matrix with interhemispheric connectivity
    2. FC matrix with ipsilateral connectivity

    Args:
        corr_matrix: Correlation matrix
        roi_labels: ROI labels (channel keys)
        p_values: Optional p-values matrix
        connectivity_patterns: Optional results from analyze_connectivity_patterns
        channel_label_map: Dictionary mapping channel keys to descriptive labels
        alpha: Significance threshold for marking significant correlations
        mask_nonsignificant: If True, hide non-significant correlations. If False, mark them with asterisks.
        subject_group: Optional group name for the subject (e.g., "Anhedonic", "Non-anhedonic")
        subject_id: Optional subject identifier for CSV export
        output_dir: Optional directory to save raw data CSV files
        verbose: If True, print detailed information about reordering
        band_name: Optional slow-band name (e.g., "Slow-4") for plot title
        frequency_range: Optional frequency range string (e.g., "0.027-0.073 Hz") for subtitle
        n_available_channels: Optional number of available channels for slow-band FC

    Returns:
        tuple: (fig_interhemispheric, fig_ipsilateral) - Two separate figures
    """

    # ===== FIGURE 1: FC Matrix + Interhemispheric Connectivity =====
    fig_inter = plt.figure(figsize=(16, 8))

    # 1. Static FC Matrix (left side)
    ax1 = plt.subplot(1, 2, 1)

    # Use descriptive labels if channel_label_map is provided
    display_labels = roi_labels
    if channel_label_map:
        display_labels = [channel_label_map.get(label, label) for label in roi_labels]

    # === HIERARCHICAL REORDERING OF CORRELATION MATRIX ===
    # Parse labels and create hierarchical sorting index
    # Expected label formats:
    #   Cortical: {region}_{hemisphere}_{network}_p{subarea} (e.g., PFCm_RH_DefaultA_p1)
    #   Subcortical: {region}_{hemisphere}_{subdivision} (e.g., AMY_RH_lAMY)

    def parse_channel_label(label):
        """
        Parse channel label to extract network, region, hemisphere, and parcel info.

        Returns:
            tuple: (region_network_key, region, hemisphere, parcel_id)
                - region_network_key: matches the bar plot grouping (e.g., "PFCm_DefaultA", "AMY_lAMY")
                - region: anatomical region (e.g., "PFCm", "AMY")
                - hemisphere: "RH" or "LH"
                - parcel_id: parcel identifier for stable sorting
        """
        parts = label.split('_')

        if len(parts) >= 4 and parts[3].startswith('p'):
            # Cortical format: region_hemi_network_pSubarea (e.g., PFCm_RH_DefaultA_p1)
            region = parts[0]
            hemisphere = parts[1]
            network = parts[2]
            parcel_id = parts[3] if len(parts) > 3 else ''
            # Create region_network_key matching bar plot format
            region_network_key = f"{region}_{network}"
        elif len(parts) >= 3:
            # Subcortical format: region_hemi_subdivision (e.g., AMY_RH_lAMY)
            region = parts[0]
            hemisphere = parts[1]
            subdivision = parts[2]
            parcel_id = subdivision
            # Create region_network_key matching bar plot format
            region_network_key = f"{region}_{subdivision}"
        else:
            # Fallback for unexpected formats
            region = parts[0] if len(parts) > 0 else 'Unknown'
            hemisphere = parts[1] if len(parts) > 1 else 'Unknown'
            region_network_key = 'Unknown'
            parcel_id = ''

        return (region_network_key, region, hemisphere, parcel_id)

    # Extract network/region ordering from interhemispheric connectivity patterns (if available)
    # The bar plot shows interhemispheric pairs in the order they appear in the dictionary,
    # which corresponds to the order they're discovered when iterating through the upper triangle
    # of the original correlation matrix
    network_region_order = []
    if connectivity_patterns and 'interhemispheric' in connectivity_patterns:
        inter_data = connectivity_patterns['interhemispheric']['pairs']
        seen_network_regions = set()

        # Iterate through pairs in insertion order (dict preserves order in Python 3.7+)
        for pair_key in inter_data.keys():
            parts = pair_key.split('_')

            # Parse pair_key to extract region_network_key (same logic as bar plot)
            if len(parts) >= 8:  # Cortical with network
                region = parts[0]
                network = parts[2]
                region_network_key = f"{region}_{network}"
            elif len(parts) >= 6:  # Subcortical without network
                region = parts[0]
                subdivision = parts[2]
                region_network_key = f"{region}_{subdivision}"
            else:
                region_network_key = parts[0] if parts else 'Unknown'

            # Preserve insertion order (first occurrence)
            if region_network_key not in seen_network_regions:
                network_region_order.append(region_network_key)
                seen_network_regions.add(region_network_key)

    # If no connectivity patterns available, fall back to preserving original order
    if not network_region_order:
        # Extract all unique region_network_keys in the order they appear
        seen = set()
        for label in display_labels:
            region_network_key, _, _, _ = parse_channel_label(label)
            if region_network_key not in seen:
                network_region_order.append(region_network_key)
                seen.add(region_network_key)

    # Create list of (label, sort_key, original_index) tuples
    label_sort_data = []
    for idx, label in enumerate(display_labels):
        region_network_key, region, hemisphere, parcel_id = parse_channel_label(label)

        # Create hierarchical sort key:
        # 1. Primary: Network/region group order (matches bar plot discovery order)
        # 2. Secondary: Hemisphere (RH=0 comes before LH=1)
        # 3. Tertiary: Parcel ID for stable sorting within hemisphere
        try:
            network_order_idx = network_region_order.index(region_network_key)
        except ValueError:
            # If region_network_key not in ordering list, place at end
            network_order_idx = len(network_region_order)

        hemi_order = 0 if hemisphere == 'RH' else 1
        sort_key = (network_order_idx, hemi_order, parcel_id)

        label_sort_data.append((label, sort_key, idx))

    # Sort by hierarchical key
    label_sort_data.sort(key=lambda x: x[1])

    # Extract new order indices
    new_order = [item[2] for item in label_sort_data]
    reordered_labels = [item[0] for item in label_sort_data]

    # Reorder correlation matrix (both rows and columns)
    corr_matrix_reordered = corr_matrix[np.ix_(new_order, new_order)]

    # Reorder p-values matrix if provided
    p_values_reordered = None
    if p_values is not None:
        p_values_reordered = p_values[np.ix_(new_order, new_order)]

    # Use reordered data for plotting
    corr_matrix = corr_matrix_reordered
    p_values = p_values_reordered
    display_labels = reordered_labels

    # Verbose output: Print reordered axis labels
    if verbose:
        print(f"\n{'='*80}")
        print(f"CORRELATION MATRIX REORDERING (Subject: {subject_id if subject_id else 'Unknown'})")
        print(f"{'='*80}")
        print(f"Reordered axis labels (matrix rows/columns):")
        for idx, label in enumerate(display_labels):
            print(f"  [{idx:2d}] {label}")
        print(f"\nNetwork/Region unique groups (for matrix ordering):")
        for idx, network_region in enumerate(network_region_order):
            print(f"  [{idx:2d}] {network_region}")
        print(f"\n(Interhemispheric pairs will be printed in sorted order below)")
        print(f"{'='*80}\n")

    # Export raw reordered data to CSV files (if output_dir and subject_id provided)
    if output_dir and subject_id:
        from pathlib import Path
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Export reordered correlation matrix
        reordered_matrix_file = output_dir / f'{subject_id}_fc_matrix_reordered.csv'
        with open(reordered_matrix_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header row with reordered labels
            writer.writerow([''] + display_labels)
            # Data rows
            for i, row_label in enumerate(display_labels):
                writer.writerow([row_label] + list(corr_matrix[i, :]))

        if verbose:
            print(f"Exported reordered correlation matrix to: {reordered_matrix_file}")

        # 2. Export reordered p-values matrix (if available)
        if p_values is not None:
            reordered_pvalues_file = output_dir / f'{subject_id}_fc_pvalues_reordered.csv'
            with open(reordered_pvalues_file, 'w', newline='') as f:
                writer = csv.writer(f)
                # Header row
                writer.writerow([''] + display_labels)
                # Data rows
                for i, row_label in enumerate(display_labels):
                    writer.writerow([row_label] + list(p_values[i, :]))

            if verbose:
                print(f"Exported reordered p-values to: {reordered_pvalues_file}")

        # 3. Export interhemispheric correlations (matching bar plot order)
        if connectivity_patterns and 'interhemispheric' in connectivity_patterns:
            inter_data = connectivity_patterns['interhemispheric']['pairs']
            if inter_data:
                interhemispheric_file = output_dir / f'{subject_id}_interhemispheric_correlations.csv'
                with open(interhemispheric_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    # Header
                    writer.writerow([
                        'Pair_Index', 'ROI_1', 'ROI_2',
                        'Region_Network_Key', 'Correlation', 'P_value', 'Significant'
                    ])

                    # Write pairs in the exact order they appear (bar plot order)
                    for pair_idx, (pair_key, pair_data) in enumerate(inter_data.items()):
                        # Parse pair to extract region_network_key
                        parts = pair_key.split('_')
                        if len(parts) >= 8:  # Cortical
                            region = parts[0]
                            network = parts[2]
                            region_network_key = f"{region}_{network}"
                        elif len(parts) >= 6:  # Subcortical
                            region = parts[0]
                            subdivision = parts[2]
                            region_network_key = f"{region}_{subdivision}"
                        else:
                            region_network_key = 'Unknown'

                        # Split pair_key into ROI_1 and ROI_2
                        roi_parts = pair_key.split('_')
                        if len(roi_parts) >= 8:  # Cortical
                            roi1 = '_'.join(roi_parts[:4])
                            roi2 = '_'.join(roi_parts[4:])
                        elif len(roi_parts) >= 6:  # Subcortical
                            roi1 = '_'.join(roi_parts[:3])
                            roi2 = '_'.join(roi_parts[3:])
                        else:
                            roi1 = roi_parts[0] if len(roi_parts) > 0 else 'Unknown'
                            roi2 = roi_parts[1] if len(roi_parts) > 1 else 'Unknown'

                        writer.writerow([
                            pair_idx,
                            roi1,
                            roi2,
                            region_network_key,
                            pair_data['correlation'],
                            pair_data.get('p_value', ''),
                            pair_data.get('significant', '')
                        ])

                if verbose:
                    print(f"Exported interhemispheric correlations to: {interhemispheric_file}")
                    print(f"  Total interhemispheric pairs: {len(inter_data)}")

    # Create mask
    # Check for unavailable channels (NaN values in correlation matrix)
    mask_unavailable = np.isnan(corr_matrix)
    has_unavailable = np.any(mask_unavailable)

    # Start with no masking
    mask_combined = None

    if mask_diagonal:
        # Mask the diagonal
        mask_combined = np.eye(corr_matrix.shape[0], dtype=bool)

    # Optionally mask non-significant correlations
    if mask_nonsignificant and p_values is not None:
        # Create non-significant mask, but exclude the diagonal and unavailable from this check
        nonsig_mask = (p_values >= alpha) & ~mask_unavailable
        # Ensure diagonal is NOT masked by the nonsignificant filter (only by mask_diagonal flag)
        np.fill_diagonal(nonsig_mask, False)

        if mask_combined is not None:
            # Combine diagonal mask with non-significant mask
            mask_combined = mask_combined | nonsig_mask
        else:
            # Only mask non-significant correlations
            mask_combined = nonsig_mask

    # Add unavailable mask
    if mask_combined is not None:
        mask_combined = mask_combined | mask_unavailable
    else:
        mask_combined = mask_unavailable

    # Display correlations with appropriate masking
    # Remove annotations (annot=False) for cleaner visualization with large matrices
    sns.heatmap(corr_matrix,
                annot=False,
                xticklabels=display_labels,
                yticklabels=display_labels,
                center=0,
                cmap='RdBu_r',
                vmin=-1, vmax=1,
                mask=mask_combined,
                ax=ax1,
                cbar_kws={'label': 'Pearson Correlation'},
                square=True)

    # Add asterisk to non-significant correlations (only if not masking them)
    if not mask_nonsignificant and p_values is not None:
        n_rois = corr_matrix.shape[0]
        for i in range(n_rois):
            for j in range(n_rois):
                # Check if correlation is non-significant (and not on diagonal)
                if i != j and p_values[i, j] >= alpha:
                    # Add small black asterisk in center of non-significant cells
                    ax1.text(j + 0.5, i + 0.5, '*',
                            ha='center', va='center',
                            color='black', fontsize=7, fontweight='bold')

    # Add note about asterisk marker and availability in the title
    # Main title
    if band_name and frequency_range:
        title_text = f'Static Functional Connectivity - {band_name}'
    else:
        title_text = 'Static Functional Connectivity Matrix'

    # Subtitle with availability and significance info
    subtitle_parts = []
    if not mask_nonsignificant and p_values is not None:
        subtitle_parts.append(f'$*$ non-significant, p ≥ {alpha}')
    # if n_available_channels is not None and has_unavailable:
    #     n_total = corr_matrix.shape[0]
    #     subtitle_parts.append(f'{n_available_channels}/{n_total} channels available')
    # if frequency_range:
    #     subtitle_parts.append(frequency_range)

    if subtitle_parts:
        title_text += '\n' + ' | '.join(subtitle_parts)

    ax1.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
    ax1.tick_params(axis='x', rotation=90, labelsize=7)
    ax1.tick_params(axis='y', rotation=0, labelsize=7)

    # Add colored rectangles highlighting interhemispheric connectivity blocks
    # This requires the interhemispheric pairs to be processed first
    # We'll add rectangles after both plots are created to access the color map

    # 2. Interhemispheric Connectivity Analysis (right side)
    ax2_inter = plt.subplot(1, 2, 2)

    if connectivity_patterns and 'interhemispheric' in connectivity_patterns:
        inter_data = connectivity_patterns['interhemispheric']['pairs']

        if inter_data:
            # Sort pairs: Group by region+RH_network (using matrix order), then same-network pairs, then cross-network
            # Order: Matrix_Region_Network_Order -> (same-network block, then cross-network block)
            def get_lh_sort_key(pair_key):
                parts = pair_key.split('_')
                # Find LH start (look for 'LH' marker)
                lh_idx = None
                for i in range(3, len(parts)):
                    if parts[i] == 'LH':
                        lh_idx = i - 1
                        break

                if lh_idx is not None and lh_idx + 2 < len(parts):
                    region = parts[0]  # Same for both RH and LH
                    network_rh = parts[2]
                    network_lh = parts[lh_idx + 2]
                    rh_parcel = parts[3] if len(parts) > 3 else ''
                    lh_parcel = parts[lh_idx + 3] if lh_idx + 3 < len(parts) else ''

                    is_same_network = (network_rh == network_lh)

                    # Use the matrix region/network order instead of alphabetical sorting
                    # This ensures AMY appears at the end to match the FC matrix axes
                    region_network_key_rh = f"{region}_{network_rh}"
                    try:
                        matrix_order = network_region_order.index(region_network_key_rh)
                    except (ValueError, NameError):
                        # Fallback to alphabetical if not in network_region_order
                        matrix_order = 999

                    # Sort by: (matrix_order, is_cross_network, LH_network, RH_parcel, LH_parcel)
                    # Groups all same-region+network pairs together, then cross-network, following matrix axis order
                    return (matrix_order, not is_same_network, network_lh, rh_parcel, lh_parcel)
                else:
                    return (999, True, '', '', '')

            # Sort pairs
            sorted_pairs = sorted(inter_data.items(), key=lambda x: get_lh_sort_key(x[0]))

            # Extract sorted data
            inter_pairs = [pair[0] for pair in sorted_pairs]
            inter_corrs = [pair[1]['correlation'] for pair in sorted_pairs]
            # Rebuild inter_data as OrderedDict to maintain sorted order
            from collections import OrderedDict
            inter_data = OrderedDict(sorted_pairs)

            # Verbose: Print sorted interhemispheric pairs (actual bar plot order)
            if verbose:
                print(f"\nInterhemispheric pairs (sorted bar plot order):")
                for idx, pair_key in enumerate(inter_pairs):
                    print(f"  [{idx:3d}] {pair_key}")
                print(f"\n  Total interhemispheric pairs: {len(inter_pairs)}")
                print(f"{'='*80}\n")

            # Helper function to extract region/network information from pair labels
            def parse_region_network(pair_key):
                """
                Parse pair key to extract region and network information for BOTH hemispheres.
                Returns (region_network_key, display_label, is_cortical, is_same_network)

                Cortical format: {region}_{hemi}_{network}_p{subarea}_{region}_{hemi}_{network}_p{subarea}
                Subcortical format: {region}_{hemi}_{subdivision}_{region}_{hemi}_{subdivision}
                """
                parts = pair_key.split('_')

                # Try to identify if this is cortical (has network info) or subcortical
                # Cortical labels have format: PFCm_RH_DefaultA_p1_PFCm_LH_DefaultA_p2
                # Subcortical labels have format: AMY_RH_lAMY_AMY_LH_lAMY

                if len(parts) >= 7:  # Cortical with network (may or may not have subarea)
                    # Extract RH (first) ROI components
                    region_rh = parts[0]
                    network_rh = parts[2]

                    # Find where LH ROI starts (look for 'LH' in parts)
                    lh_start_idx = None
                    for i in range(3, len(parts)):
                        if parts[i] == 'LH':
                            lh_start_idx = i - 1  # Region is before 'LH'
                            break

                    if lh_start_idx and lh_start_idx + 2 < len(parts):
                        region_lh = parts[lh_start_idx]
                        network_lh = parts[lh_start_idx + 2]
                    else:
                        # Fallback parsing
                        region_lh = parts[4] if len(parts) > 4 else 'Unknown'
                        network_lh = parts[6] if len(parts) > 6 else 'Unknown'

                    is_same_network = (network_rh == network_lh)

                    if is_same_network:
                        # Same network: "PFCm (DefaultA)"
                        region_network_key = f"{region_rh}_{network_rh}"
                        display_label = f"{region_rh} ({network_rh})"
                    else:
                        # Cross-network: "PFCm DefaultA (RH) - LimbicB (LH)"
                        region_network_key = f"{region_rh}_{network_rh}_x_{network_lh}"
                        display_label = f"{region_rh} {network_rh} (RH) - {network_lh} (LH)"

                    is_cortical = True

                elif len(parts) >= 6:  # Subcortical without network
                    # Extract region and subdivision for both hemispheres
                    region_rh = parts[0]
                    subdivision_rh = parts[2]
                    region_lh = parts[3]
                    subdivision_lh = parts[5]

                    is_same_network = (subdivision_rh == subdivision_lh)

                    if is_same_network:
                        # Same subdivision: "AMY (lAMY)"
                        region_network_key = f"{region_rh}_{subdivision_rh}"
                        display_label = f"{region_rh} ({subdivision_rh})"
                    else:
                        # Cross-subdivision: "AMY lAMY (RH) - mAMY (LH)"
                        region_network_key = f"{region_rh}_{subdivision_rh}_x_{subdivision_lh}"
                        display_label = f"{region_rh} {subdivision_rh} (RH) - {subdivision_lh} (LH)"

                    is_cortical = False

                else:
                    # Fallback for unexpected formats
                    region_network_key = parts[0] if parts else 'Unknown'
                    display_label = region_network_key
                    is_cortical = False
                    is_same_network = True

                return region_network_key, display_label, is_cortical, is_same_network

            # Assign colors to unique region/network combinations
            unique_region_networks = {}
            pair_region_networks = []

            for pair_key in inter_pairs:
                region_network_key, display_label, is_cortical, is_same_network = parse_region_network(pair_key)
                pair_region_networks.append((region_network_key, display_label, is_cortical, is_same_network))

                if region_network_key not in unique_region_networks:
                    unique_region_networks[region_network_key] = {
                        'display_label': display_label,
                        'is_cortical': is_cortical,
                        'is_same_network': is_same_network,
                        'indices': []
                    }
                unique_region_networks[region_network_key]['indices'].append(len(pair_region_networks) - 1)

            # Define distinct color palette with saturated, medium-to-dark colors
            # Optimized for visibility against RdBu_r colormap (red-blue diverging)
            # Avoiding very light colors (poor contrast) and very dark colors (hide black hatches)
            base_colors = [
                '#1f77b4',  # Blue
                '#ff7f0e',  # Orange
                '#2ca02c',  # Green
                '#d62728',  # Red
                '#9467bd',  # Purple
                '#8c564b',  # Brown
                '#e6d800',  # Bright yellow
                '#00CED1',  # Dark turquoise
                '#FF1493',  # Deep pink
                '#32CD32',  # Lime green
                '#FF4500',  # Orange-red
                '#9932CC',  # Dark orchid
                '#DAA520',  # Goldenrod
                '#DC143C',  # Crimson
                '#4169E1',  # Royal blue
                '#FF6347',  # Tomato
                '#20B2AA',  # Light sea green
                '#8B4789',  # Medium purple
                '#CD853F',  # Peru
                '#FF8C00',  # Dark orange
                '#BA55D3',  # Medium orchid
                '#3CB371',  # Medium sea green
                '#FF69B4',  # Hot pink
                '#4682B4',  # Steel blue
                '#DB7093',  # Pale violet red
                '#CD5C5C',  # Indian red
            ]

            # Assign colors to each unique region/network
            color_map = {}
            for idx, (region_network_key, info) in enumerate(unique_region_networks.items()):
                color_map[region_network_key] = base_colors[idx % len(base_colors)]

            # Prepare bar colors and hatching
            bar_colors = []
            bar_hatches = []
            for i, (corr_val, pair_data) in enumerate(zip(inter_corrs, inter_data.values())):
                region_network_key, _, _, is_same_network = pair_region_networks[i]
                bar_colors.append(color_map[region_network_key])

                # Only use hatching for non-significant correlations
                if p_values is not None and not pair_data.get('significant', False):
                    bar_hatches.append('//////')  # Dense diagonal lines
                else:
                    bar_hatches.append(None)

            # Create all bars at once
            bars = ax2_inter.bar(range(len(inter_corrs)), inter_corrs, color=bar_colors,
                          edgecolor='black', alpha=0.8, width=0.8, linewidth=0.8)

            # Apply hatching to non-significant bars
            for bar, hatch in zip(bars, bar_hatches):
                if hatch:
                    bar.set_hatch(hatch)
                    bar.set_alpha(0.5)

            # Remove x-axis tick labels (too cluttered with many connections)
            ax2_inter.set_xticks([])
            ax2_inter.set_xlabel(f'{len(inter_corrs)} interhemispheric connections', fontsize=11)
            ax2_inter.set_ylabel('Pearson Correlation', fontsize=12)
            ax2_inter.set_title('Interhemispheric Connectivity\n(Same Region, Different Hemispheres)',
                         fontsize=12, fontweight='bold', pad=20)
            ax2_inter.set_ylim(-1, 1)
            ax2_inter.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2_inter.grid(True, alpha=0.3, axis='y')

            # Calculate significance statistics
            total_pairs = len(inter_data)
            significant_pairs = sum(1 for pair_data in inter_data.values()
                                   if pair_data.get('significant', False))
            significance_pct = (significant_pairs / total_pairs * 100) if total_pairs > 0 else 0

            # Add group label in upper left if provided
            if subject_group:
                ax2_inter.text(0.02, 0.98, f'Group: {subject_group}',
                        transform=ax2_inter.transAxes, fontsize=10, verticalalignment='top',
                        fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, edgecolor='gray'))

            # Add significance percentage text in bottom left
            ax2_inter.text(0.02, 0.02, f'{significance_pct:.1f}% significant pairs\n({significant_pairs}/{total_pairs})',
                    transform=ax2_inter.transAxes, fontsize=9, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

            # Create legend with color-coded region/network pairs
            from matplotlib.patches import Patch
            legend_elements = []

            # Add color patches for each region/network
            for region_network_key, info in unique_region_networks.items():
                color = color_map[region_network_key]
                legend_elements.append(
                    Patch(facecolor=color, edgecolor='black', linewidth=0.5,
                         alpha=0.9, label=info['display_label'])
                )

            # Add hatching pattern explanation
            legend_elements.append(
                Patch(facecolor='gray', edgecolor='black', hatch='//////',
                     alpha=0.5, label='Non-significant (p ≥ 0.05)')
            )

            # Place legend outside plot area to the right to avoid long labels overflowing
            # This keeps the legend from covering data and handles long network names
            ax2_inter.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
                      fontsize=7, framealpha=0.95, edgecolor='black',
                      handlelength=1.5, handleheight=1.0)

            # Add colored rectangles to correlation matrix (ax1) highlighting interhemispheric blocks
            # For each region/network group, find RH and LH parcel indices and draw rectangle
            from matplotlib.patches import Rectangle

            for region_network_key, info in unique_region_networks.items():
                # Only process same-network pairs (not cross-network)
                if not info['is_same_network']:
                    continue

                # Extract RH and LH parcel labels for this region/network
                rh_parcels = []
                lh_parcels = []

                # Parse region/network key to get base pattern
                # Format: "PFCm_DefaultA" or "AMY_lAMY"
                key_parts = region_network_key.split('_')
                if len(key_parts) >= 2:
                    region = key_parts[0]
                    network_or_subdivision = key_parts[1]

                    # Search through reordered labels to find matching RH and LH parcels
                    for idx, label in enumerate(reordered_labels):
                        label_parts = label.split('_')
                        if len(label_parts) >= 3:
                            label_region = label_parts[0]
                            label_hemi = label_parts[1]
                            label_network = label_parts[2]

                            # Check if this label matches the region and network/subdivision
                            if label_region == region and label_network == network_or_subdivision:
                                if label_hemi == 'RH':
                                    rh_parcels.append(idx)
                                elif label_hemi == 'LH':
                                    lh_parcels.append(idx)

                # Draw rectangle if we have both RH and LH parcels
                if rh_parcels and lh_parcels:
                    # Rectangle spans: rows = LH parcels, cols = RH parcels
                    row_start = min(lh_parcels)
                    row_end = max(lh_parcels) + 1
                    col_start = min(rh_parcels)
                    col_end = max(rh_parcels) + 1

                    color = color_map[region_network_key]

                    # Draw rectangle outline (no fill, just border)
                    rect = Rectangle(
                        (col_start, row_start),  # (x, y) - bottom-left corner
                        col_end - col_start,     # width
                        row_end - row_start,     # height
                        linewidth=1.5,
                        edgecolor=color,
                        facecolor='none',
                        clip_on=False,
                        zorder=10
                    )
                    ax1.add_patch(rect)

        else:
            ax2_inter.text(0.5, 0.5, 'No interhemispheric\nconnections found',
                    ha='center', va='center', transform=ax2_inter.transAxes, fontsize=14, color='gray')
            ax2_inter.set_title('Interhemispheric Connectivity\n(Same Region, Different Hemispheres)',
                         fontsize=13, fontweight='bold', pad=20)
            ax2_inter.set_xlim(0, 1)
            ax2_inter.set_ylim(0, 1)
    else:
        ax2_inter.text(0.5, 0.5, 'No connectivity patterns\navailable',
                ha='center', va='center', transform=ax2_inter.transAxes, fontsize=14, color='gray')
        ax2_inter.set_title('Interhemispheric Connectivity\n(Same Region, Different Hemispheres)',
                     fontsize=13, fontweight='bold', pad=20)

    # Finalize first figure
    fig_inter.tight_layout(pad=2.8)

    # ===== FIGURE 2: FC Matrix + Ipsilateral Connectivity =====
    fig_ipsi = plt.figure(figsize=(16, 8))

    # 1. Static FC Matrix (left side) - reordered based on ipsilateral pairs
    ax1_ipsi = plt.subplot(1, 2, 1)

    # === REORDER FC MATRIX BASED ON IPSILATERAL CONNECTIVITY ===
    # Extract ipsilateral ordering from connectivity patterns
    ipsi_network_region_order = []
    if connectivity_patterns and 'ipsilateral' in connectivity_patterns:
        ipsi_data = connectivity_patterns['ipsilateral']['pairs']
        seen_ipsi_network_regions = set()

        # Iterate through ipsilateral pairs in insertion order
        for pair_key in ipsi_data.keys():
            parts = pair_key.split('_')

            # Parse pair_key to extract region_network_key (same logic as bar plot)
            if len(parts) >= 7:  # Cortical with network
                region1 = parts[0]
                network1 = parts[2]
                # Find second region
                second_region_idx = None
                hemi = parts[1]
                for i in range(3, len(parts)):
                    if parts[i] == hemi:
                        second_region_idx = i - 1
                        break

                if second_region_idx and second_region_idx + 2 < len(parts):
                    region2 = parts[second_region_idx]
                    network2 = parts[second_region_idx + 2]

                    # Add both region_network keys
                    region_network_key1 = f"{region1}_{network1}"
                    region_network_key2 = f"{region2}_{network2}"

                    if region_network_key1 not in seen_ipsi_network_regions:
                        ipsi_network_region_order.append(region_network_key1)
                        seen_ipsi_network_regions.add(region_network_key1)
                    if region_network_key2 not in seen_ipsi_network_regions:
                        ipsi_network_region_order.append(region_network_key2)
                        seen_ipsi_network_regions.add(region_network_key2)

            elif len(parts) >= 6:  # Subcortical without network
                region1 = parts[0]
                subdivision1 = parts[2]
                region2 = parts[3]
                subdivision2 = parts[5]

                region_network_key1 = f"{region1}_{subdivision1}"
                region_network_key2 = f"{region2}_{subdivision2}"

                if region_network_key1 not in seen_ipsi_network_regions:
                    ipsi_network_region_order.append(region_network_key1)
                    seen_ipsi_network_regions.add(region_network_key1)
                if region_network_key2 not in seen_ipsi_network_regions:
                    ipsi_network_region_order.append(region_network_key2)
                    seen_ipsi_network_regions.add(region_network_key2)

    # If no ipsilateral patterns available, fall back to original order
    if not ipsi_network_region_order:
        # Extract all unique region_network_keys in the order they appear
        seen = set()
        for label in roi_labels:
            region_network_key, _, _, _ = parse_channel_label(label)
            if region_network_key not in seen:
                ipsi_network_region_order.append(region_network_key)
                seen.add(region_network_key)

    # Create list of (label, sort_key, original_index) tuples for ipsilateral ordering
    ipsi_label_sort_data = []
    for idx, label in enumerate(roi_labels):
        region_network_key, region, hemisphere, parcel_id = parse_channel_label(label)

        # Create hierarchical sort key for ipsilateral:
        # 1. Primary: Hemisphere (RH=0 comes before LH=1) - HEMISPHERE FIRST
        # 2. Secondary: Network/region group order (matches ipsilateral bar plot order)
        # 3. Tertiary: Parcel ID for stable sorting within region/network
        try:
            ipsi_network_order_idx = ipsi_network_region_order.index(region_network_key)
        except ValueError:
            # If region_network_key not in ordering list, place at end
            ipsi_network_order_idx = len(ipsi_network_region_order)

        hemi_order = 0 if hemisphere == 'RH' else 1
        ipsi_sort_key = (hemi_order, ipsi_network_order_idx, parcel_id)  # HEMISPHERE FIRST

        ipsi_label_sort_data.append((label, ipsi_sort_key, idx))

    # Sort by hierarchical key
    ipsi_label_sort_data.sort(key=lambda x: x[1])

    # Extract new order indices
    ipsi_new_order = [item[2] for item in ipsi_label_sort_data]
    ipsi_reordered_labels = [item[0] for item in ipsi_label_sort_data]

    # Get the original unreordered matrices (before interhemispheric reordering)
    # We need to work from the original roi_labels order, not the interhemispheric-reordered one
    # So we'll reorder from the original corr_matrix passed to the function
    # But wait - we already modified corr_matrix and p_values above for interhemispheric
    # We need to use the ORIGINAL matrices before any reordering
    # Let's save them at the beginning of the function or reconstruct from roi_labels

    # Actually, we need to back out the interhemispheric reordering first
    # Create inverse mapping from interhemispheric reordering
    inverse_inter_order = [0] * len(new_order)
    for new_idx, orig_idx in enumerate(new_order):
        inverse_inter_order[orig_idx] = new_idx

    # Get back to original matrix order
    corr_matrix_original = corr_matrix[np.ix_(inverse_inter_order, inverse_inter_order)]
    p_values_original = None
    if p_values is not None:
        p_values_original = p_values[np.ix_(inverse_inter_order, inverse_inter_order)]

    # Now apply ipsilateral reordering to original matrices
    corr_matrix_ipsi_reordered = corr_matrix_original[np.ix_(ipsi_new_order, ipsi_new_order)]
    p_values_ipsi_reordered = None
    if p_values_original is not None:
        p_values_ipsi_reordered = p_values_original[np.ix_(ipsi_new_order, ipsi_new_order)]

    # Apply same mask logic to ipsilateral-reordered matrix
    mask_ipsi_unavailable = np.isnan(corr_matrix_ipsi_reordered)
    has_ipsi_unavailable = np.any(mask_ipsi_unavailable)
    mask_ipsi_combined = None

    if mask_diagonal:
        mask_ipsi_combined = np.eye(corr_matrix_ipsi_reordered.shape[0], dtype=bool)

    if mask_nonsignificant and p_values_ipsi_reordered is not None:
        nonsig_ipsi_mask = (p_values_ipsi_reordered >= alpha) & ~mask_ipsi_unavailable
        np.fill_diagonal(nonsig_ipsi_mask, False)

        if mask_ipsi_combined is not None:
            mask_ipsi_combined = mask_ipsi_combined | nonsig_ipsi_mask
        else:
            mask_ipsi_combined = nonsig_ipsi_mask

    if mask_ipsi_combined is not None:
        mask_ipsi_combined = mask_ipsi_combined | mask_ipsi_unavailable
    else:
        mask_ipsi_combined = mask_ipsi_unavailable

    # Use descriptive labels for ipsilateral figure
    ipsi_display_labels = ipsi_reordered_labels
    if channel_label_map:
        ipsi_display_labels = [channel_label_map.get(label, label) for label in ipsi_reordered_labels]

    # Verbose output for ipsilateral reordering
    if verbose:
        print(f"\n{'='*80}")
        print(f"IPSILATERAL MATRIX REORDERING (Subject: {subject_id if subject_id else 'Unknown'})")
        print(f"{'='*80}")
        print(f"Reordered axis labels (matrix rows/columns):")
        for idx, label in enumerate(ipsi_display_labels):
            print(f"  [{idx:2d}] {label}")
        print(f"\nNetwork/Region unique groups (for ipsilateral matrix ordering):")
        for idx, network_region in enumerate(ipsi_network_region_order):
            print(f"  [{idx:2d}] {network_region}")
        print(f"\n(Ipsilateral pairs will be printed in sorted order below)")
        print(f"{'='*80}\n")

    # Plot ipsilateral-reordered FC matrix
    sns.heatmap(corr_matrix_ipsi_reordered,
                annot=False,
                xticklabels=ipsi_display_labels,
                yticklabels=ipsi_display_labels,
                center=0,
                cmap='RdBu_r',
                vmin=-1, vmax=1,
                mask=mask_ipsi_combined,
                ax=ax1_ipsi,
                cbar_kws={'label': 'Pearson Correlation'},
                square=True)

    # Add asterisk to non-significant correlations (only if not masking them)
    if not mask_nonsignificant and p_values_ipsi_reordered is not None:
        n_rois = corr_matrix_ipsi_reordered.shape[0]
        for i in range(n_rois):
            for j in range(n_rois):
                # Check if correlation is non-significant (and not on diagonal)
                if i != j and p_values_ipsi_reordered[i, j] >= alpha:
                    # Add small black asterisk in center of non-significant cells
                    ax1_ipsi.text(j + 0.5, i + 0.5, '*',
                            ha='center', va='center',
                            color='black', fontsize=7, fontweight='bold')

    # Add title for ipsilateral figure
    if band_name and frequency_range:
        title_text_ipsi = f'Static Functional Connectivity - {band_name}'
    else:
        title_text_ipsi = 'Static Functional Connectivity Matrix'

    # Subtitle with availability and significance info
    subtitle_parts_ipsi = []
    if not mask_nonsignificant and p_values_ipsi_reordered is not None:
        subtitle_parts_ipsi.append(f'$*$ non-significant, p ≥ {alpha}')
    # if n_available_channels is not None and has_ipsi_unavailable:
    #     n_total_ipsi = corr_matrix_ipsi_reordered.shape[0]
    #     subtitle_parts_ipsi.append(f'{n_available_channels}/{n_total_ipsi} channels available')
    # if frequency_range:
    #     subtitle_parts_ipsi.append(frequency_range)

    if subtitle_parts_ipsi:
        title_text_ipsi += '\n' + ' | '.join(subtitle_parts_ipsi)

    ax1_ipsi.set_title(title_text_ipsi, fontsize=14, fontweight='bold', pad=20)
    ax1_ipsi.tick_params(axis='x', rotation=90, labelsize=7)
    ax1_ipsi.tick_params(axis='y', rotation=0, labelsize=7)

    # 2. Ipsilateral Connectivity Analysis (right side)
    ax2_ipsi = plt.subplot(1, 2, 2)

    if connectivity_patterns and 'ipsilateral' in connectivity_patterns:
        ipsi_data = connectivity_patterns['ipsilateral']['pairs']

        if ipsi_data:
            # Sort ipsilateral pairs: HEMISPHERE FIRST, then by region/network
            def get_ipsi_sort_key(pair_key):
                parts = pair_key.split('_')
                # Find second region start (after first hemisphere marker)
                second_region_idx = None
                for i in range(3, len(parts)):
                    if parts[i] in ['RH', 'LH']:
                        second_region_idx = i - 1
                        break

                if second_region_idx is not None and second_region_idx + 2 < len(parts):
                    region1 = parts[0]
                    hemi = parts[1]
                    network1 = parts[2]
                    region2 = parts[second_region_idx]
                    network2 = parts[second_region_idx + 2] if second_region_idx + 2 < len(parts) else ''

                    # Use matrix order for sorting within hemisphere
                    region_network_key_1 = f"{region1}_{network1}"
                    try:
                        matrix_order = ipsi_network_region_order.index(region_network_key_1)
                    except (ValueError, NameError):
                        matrix_order = 999

                    # Hemisphere order: RH=0, LH=1
                    hemi_order = 0 if hemi == 'RH' else 1

                    # Sort by: (HEMISPHERE FIRST, then matrix_order, region2, network2)
                    return (hemi_order, matrix_order, region2, network2)
                else:
                    return (999, 999, '', '')

            # Sort pairs
            sorted_ipsi_pairs = sorted(ipsi_data.items(), key=lambda x: get_ipsi_sort_key(x[0]))

            # Extract sorted data
            ipsi_pairs = [pair[0] for pair in sorted_ipsi_pairs]
            ipsi_corrs = [pair[1]['correlation'] for pair in sorted_ipsi_pairs]

            # Rebuild ipsi_data as OrderedDict to maintain sorted order
            from collections import OrderedDict
            ipsi_data = OrderedDict(sorted_ipsi_pairs)

            # Helper function to parse ipsilateral pair information
            def parse_ipsi_region_network(pair_key):
                """
                Parse ipsilateral pair key to extract region and network information.
                Returns (region_network_key, display_label, is_cortical, hemisphere)

                Ipsilateral format (same hemisphere, different regions):
                Cortical: {region1}_{hemi}_{network1}_p{subarea1}_{region2}_{hemi}_{network2}_p{subarea2}
                Subcortical: {region1}_{hemi}_{subdivision1}_{region2}_{hemi}_{subdivision2}
                """
                parts = pair_key.split('_')

                if len(parts) >= 7:  # Cortical with network
                    region1 = parts[0]
                    hemi = parts[1]
                    network1 = parts[2]

                    # Find where second region starts
                    second_region_idx = None
                    for i in range(3, len(parts)):
                        if parts[i] == hemi:  # Same hemisphere marker
                            second_region_idx = i - 1
                            break

                    if second_region_idx and second_region_idx + 2 < len(parts):
                        region2 = parts[second_region_idx]
                        network2 = parts[second_region_idx + 2]
                    else:
                        region2 = parts[4] if len(parts) > 4 else 'Unknown'
                        network2 = parts[6] if len(parts) > 6 else 'Unknown'

                    # Create unique key for this ipsilateral connection (include hemisphere to distinguish RH/LH)
                    region_network_key = f"{region1}_{network1}_to_{region2}_{network2}_{hemi}"
                    display_label = f"{region1} ({network1}) - {region2} ({network2}) [{hemi}]"
                    is_cortical = True

                elif len(parts) >= 6:  # Subcortical without network
                    region1 = parts[0]
                    hemi = parts[1]
                    subdivision1 = parts[2]
                    region2 = parts[3]
                    subdivision2 = parts[5]

                    region_network_key = f"{region1}_{subdivision1}_to_{region2}_{subdivision2}_{hemi}"
                    display_label = f"{region1} ({subdivision1}) - {region2} ({subdivision2}) [{hemi}]"
                    is_cortical = False

                else:
                    region_network_key = parts[0] if parts else 'Unknown'
                    display_label = region_network_key
                    is_cortical = False
                    hemi = 'Unknown'

                return region_network_key, display_label, is_cortical, hemi

            # Assign colors to unique ipsilateral connections
            unique_ipsi_connections = {}
            pair_ipsi_connections = []

            for pair_key in ipsi_pairs:
                region_network_key, display_label, is_cortical, hemi = parse_ipsi_region_network(pair_key)
                pair_ipsi_connections.append((region_network_key, display_label, is_cortical, hemi))

                if region_network_key not in unique_ipsi_connections:
                    unique_ipsi_connections[region_network_key] = {
                        'display_label': display_label,
                        'is_cortical': is_cortical,
                        'hemisphere': hemi,
                        'indices': []
                    }
                unique_ipsi_connections[region_network_key]['indices'].append(len(pair_ipsi_connections) - 1)

            # Distinct, vibrant color palette for ipsilateral connections
            # Carefully selected for high contrast and visual distinction
            # Avoiding very light colors (poor visibility) and very dark colors (black/gray)
            ipsi_colors = [
                '#E74C3C',  # Vivid red
                '#3498DB',  # Bright blue
                '#2ECC71',  # Emerald green
                '#F39C12',  # Vibrant orange
                '#9B59B6',  # Amethyst purple
                '#1ABC9C',  # Turquoise
                '#E91E63',  # Pink/magenta
                '#00BCD4',  # Cyan
                '#FF5722',  # Deep orange
                '#673AB7',  # Deep purple
                '#009688',  # Teal
                '#FF9800',  # Orange
                '#8E44AD',  # Purple
                '#27AE60',  # Green
                '#F44336',  # Red
                '#2196F3',  # Blue
                '#CDDC39',  # Lime
                '#FF6F00',  # Amber
                '#7B1FA2',  # Purple
                '#00897B',  # Teal
                '#D32F2F',  # Dark red
                '#1976D2',  # Dark blue
                '#AFB42B',  # Olive
                '#E64A19',  # Burnt orange
                '#5E35B1',  # Indigo
                '#00796B',  # Dark teal
                '#FFA726',  # Light orange
                '#AB47BC',  # Light purple
                '#26A69A',  # Medium teal
                '#EF5350',  # Light red
            ]

            # Assign colors to each unique ipsilateral connection
            ipsi_color_map = {}
            for idx, (connection_key, info) in enumerate(unique_ipsi_connections.items()):
                ipsi_color_map[connection_key] = ipsi_colors[idx % len(ipsi_colors)]

            # Prepare bar colors and hatching for ipsilateral plot
            ipsi_bar_colors = []
            ipsi_bar_hatches = []
            for i, (corr_val, pair_data) in enumerate(zip(ipsi_corrs, ipsi_data.values())):
                region_network_key, _, _, _ = pair_ipsi_connections[i]
                ipsi_bar_colors.append(ipsi_color_map[region_network_key])

                # Only use hatching for non-significant correlations
                if p_values is not None and not pair_data.get('significant', False):
                    ipsi_bar_hatches.append('//////')
                else:
                    ipsi_bar_hatches.append(None)

            # Create all bars at once
            ipsi_bars = ax2_ipsi.bar(range(len(ipsi_corrs)), ipsi_corrs, color=ipsi_bar_colors,
                          edgecolor='black', alpha=0.8, width=0.8, linewidth=0.8)

            # Apply hatching to non-significant bars
            for bar, hatch in zip(ipsi_bars, ipsi_bar_hatches):
                if hatch:
                    bar.set_hatch(hatch)
                    bar.set_alpha(0.5)

            # Remove x-axis tick labels
            ax2_ipsi.set_xticks([])
            ax2_ipsi.set_xlabel(f'{len(ipsi_corrs)} ipsilateral connections', fontsize=11)
            ax2_ipsi.set_ylabel('Pearson Correlation', fontsize=12)
            ax2_ipsi.set_title('Ipsilateral Connectivity\n(Same Hemisphere, Different Regions)',
                         fontsize=12, fontweight='bold', pad=20)
            ax2_ipsi.set_ylim(-1, 1)
            ax2_ipsi.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2_ipsi.grid(True, alpha=0.3, axis='y')

            # Calculate significance statistics
            total_ipsi_pairs = len(ipsi_data)
            significant_ipsi_pairs = sum(1 for pair_data in ipsi_data.values()
                                   if pair_data.get('significant', False))
            ipsi_significance_pct = (significant_ipsi_pairs / total_ipsi_pairs * 100) if total_ipsi_pairs > 0 else 0

            # Add group label in upper left if provided
            if subject_group:
                ax2_ipsi.text(0.02, 0.98, f'Group: {subject_group}',
                        transform=ax2_ipsi.transAxes, fontsize=10, verticalalignment='top',
                        fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, edgecolor='gray'))

            # Add significance percentage text in bottom left
            ax2_ipsi.text(0.02, 0.02, f'{ipsi_significance_pct:.1f}% significant pairs\n({significant_ipsi_pairs}/{total_ipsi_pairs})',
                    transform=ax2_ipsi.transAxes, fontsize=9, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

            # Create legend with color-coded ipsilateral connections
            from matplotlib.patches import Patch
            ipsi_legend_elements = []

            # Add color patches for each ipsilateral connection type
            for connection_key, info in unique_ipsi_connections.items():
                color = ipsi_color_map[connection_key]
                ipsi_legend_elements.append(
                    Patch(facecolor=color, edgecolor='black', linewidth=0.5,
                         alpha=0.9, label=info['display_label'])
                )

            # Add hatching pattern explanation
            ipsi_legend_elements.append(
                Patch(facecolor='gray', edgecolor='black', hatch='//////',
                     alpha=0.5, label='Non-significant (p ≥ 0.05)')
            )

            # Place legend outside plot area to the right
            ax2_ipsi.legend(handles=ipsi_legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
                      fontsize=7, framealpha=0.95, edgecolor='black',
                      handlelength=1.5, handleheight=1.0)

            # Add colored rectangles to ipsilateral FC matrix highlighting connection pairs
            # For each ipsilateral connection, find the region indices and draw rectangles
            from matplotlib.patches import Rectangle

            if verbose:
                print(f"\n{'='*80}")
                print(f"DRAWING IPSILATERAL RECTANGLES")
                print(f"{'='*80}")
                print(f"Total unique ipsilateral connections: {len(unique_ipsi_connections)}")

            rectangles_drawn = 0
            for connection_key, info in unique_ipsi_connections.items():
                # Parse connection_key to extract region/network info for both ends
                # Format: "region1_network1_to_region2_network2_hemi" or "region1_subdivision1_to_region2_subdivision2_hemi"

                if '_to_' not in connection_key:
                    if verbose:
                        print(f"  [SKIP] {connection_key}: No '_to_' separator")
                    continue

                parts = connection_key.split('_to_')
                if len(parts) != 2:
                    if verbose:
                        print(f"  [SKIP] {connection_key}: Invalid split (got {len(parts)} parts)")
                    continue

                region_network_1 = parts[0]  # e.g., "PFCm_DefaultA" or "AMY_lAMY"
                region_network_2_with_hemi = parts[1]  # e.g., "PFCv_DefaultB_RH" or "AMY_mAMY_LH"

                # Extract hemisphere from the end of region_network_2
                rn2_parts = region_network_2_with_hemi.split('_')
                if len(rn2_parts) < 3:  # Need at least region_network_hemi
                    if verbose:
                        print(f"  [SKIP] {connection_key}: Invalid region/network parts in part 2")
                    continue

                # Hemisphere is the last part
                hemisphere = rn2_parts[-1]
                # Region and network are before the hemisphere
                region2 = rn2_parts[0]
                network2 = rn2_parts[1]

                # Extract region and network/subdivision from first part
                rn1_parts = region_network_1.split('_')
                if len(rn1_parts) < 2:
                    if verbose:
                        print(f"  [SKIP] {connection_key}: Invalid region/network parts in part 1")
                    continue

                region1 = rn1_parts[0]
                network1 = rn1_parts[1]

                # Find all parcels for region1_network1 in this hemisphere
                region1_parcels = []
                region2_parcels = []

                for idx, label in enumerate(ipsi_display_labels):
                    label_parts = label.split('_')
                    if len(label_parts) >= 3:
                        label_region = label_parts[0]
                        label_hemi = label_parts[1]
                        label_network = label_parts[2]

                        # Check if this label matches region1/network1 in the correct hemisphere
                        if label_region == region1 and label_network == network1 and label_hemi == hemisphere:
                            region1_parcels.append(idx)
                        # Check if this label matches region2/network2 in the correct hemisphere
                        elif label_region == region2 and label_network == network2 and label_hemi == hemisphere:
                            region2_parcels.append(idx)

                if verbose:
                    print(f"  Connection: {connection_key}")
                    print(f"    Hemisphere: {hemisphere}")
                    print(f"    Region1 ({region1}_{network1}): {len(region1_parcels)} parcels")
                    print(f"    Region2 ({region2}_{network2}): {len(region2_parcels)} parcels")

                # Draw rectangles if we have parcels for both regions
                if region1_parcels and region2_parcels:
                    color = ipsi_color_map[connection_key]

                    # Only draw rectangle in UPPER TRIANGLE (above diagonal)
                    # Determine which region appears first in the sorted order
                    row_start_1 = min(region1_parcels)
                    col_start_2 = min(region2_parcels)

                    # Only draw if the rectangle would be in the upper triangle
                    # Upper triangle: column index > row index
                    if col_start_2 < row_start_1:
                        # region2 comes before region1 -> draw at (rows=region1, cols=region2)
                        row_start = min(region1_parcels)
                        row_end = max(region1_parcels) + 1
                        col_start = min(region2_parcels)
                        col_end = max(region2_parcels) + 1

                        rect = Rectangle(
                            (col_start, row_start),
                            col_end - col_start,
                            row_end - row_start,
                            linewidth=1.5,
                            edgecolor=color,
                            facecolor='none',
                            clip_on=False,
                            zorder=10
                        )
                        ax1_ipsi.add_patch(rect)
                        rectangles_drawn += 1
                        if verbose:
                            print(f"    [DRAWN] Rectangle at rows[{row_start}:{row_end}], cols[{col_start}:{col_end}]")
                    elif row_start_1 < col_start_2:
                        # region1 comes before region2 -> draw at (rows=region2, cols=region1)
                        row_start = min(region2_parcels)
                        row_end = max(region2_parcels) + 1
                        col_start = min(region1_parcels)
                        col_end = max(region1_parcels) + 1

                        rect = Rectangle(
                            (col_start, row_start),
                            col_end - col_start,
                            row_end - row_start,
                            linewidth=1.5,
                            edgecolor=color,
                            facecolor='none',
                            clip_on=False,
                            zorder=10
                        )
                        ax1_ipsi.add_patch(rect)
                        rectangles_drawn += 1
                        if verbose:
                            print(f"    [DRAWN] Rectangle at rows[{row_start}:{row_end}], cols[{col_start}:{col_end}]")
                    else:
                        if verbose:
                            print(f"    [SKIP] Rectangle on diagonal (row_start={row_start_1}, col_start={col_start_2})")
                else:
                    if verbose:
                        if not region1_parcels:
                            print(f"    [SKIP] No parcels found for region1")
                        if not region2_parcels:
                            print(f"    [SKIP] No parcels found for region2")

            if verbose:
                print(f"\nRectangles drawn: {rectangles_drawn}/{len(unique_ipsi_connections)}")
                print(f"{'='*80}\n")

        else:
            ax2_ipsi.text(0.5, 0.5, 'No ipsilateral\nconnections found',
                    ha='center', va='center', transform=ax2_ipsi.transAxes, fontsize=14, color='gray')
            ax2_ipsi.set_title('Ipsilateral Connectivity\n(Same Hemisphere, Different Regions)',
                         fontsize=13, fontweight='bold', pad=20)
            ax2_ipsi.set_xlim(0, 1)
            ax2_ipsi.set_ylim(0, 1)
    else:
        ax2_ipsi.text(0.5, 0.5, 'No connectivity patterns\navailable',
                ha='center', va='center', transform=ax2_ipsi.transAxes, fontsize=14, color='gray')
        ax2_ipsi.set_title('Ipsilateral Connectivity\n(Same Hemisphere, Different Regions)',
                     fontsize=13, fontweight='bold', pad=20)

    # Finalize second figure
    fig_ipsi.tight_layout(pad=2.8)

    # Return both figures
    return fig_inter, fig_ipsi

def plot_signal_decomposition(original, components, subject_id=None, channel_label_map=None, center_freqs=None, max_figures_per_batch=20):
    """
    Plot signal decomposition for each channel separately.

    Args:
        original: The original signal that has been decomposed (channels x samples)
        components: The decomposed modes of the original signal (modes x channels x samples)
        subject_id: Optional subject identifier to add to figure title
        channel_label_map: Optional dictionary mapping channel indices to labels
        center_freqs: Optional array of center frequencies for each mode
        max_figures_per_batch: Maximum number of figures to create per batch to avoid memory issues

    Returns:
        generator: Generator yielding batches of matplotlib.figure.Figure objects
    """
    mode_count = components.shape[0]
    channel_count = original.shape[0]
    subplot_count = mode_count + 1  # Original signal + all components

    current_batch = []

    # Create a separate figure for each channel, yielding in batches
    for channel_idx in range(channel_count):
        fig, axes = plt.subplots(subplot_count, 1, figsize=(14, 2 * subplot_count), sharex=True)

        # Handle single subplot case
        if subplot_count == 1:
            axes = [axes]

        # Plot original signal for this channel
        axes[0].plot(original[channel_idx], color='black', linewidth=1.2, alpha=0.8)
        axes[0].set_ylabel('Amplitude', fontsize=10)
        axes[0].set_title('Original Signal', fontsize=11, fontweight='bold', pad=8)
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)

        # Plot each decomposed component for this channel
        for mode_idx in range(mode_count):
            axes[mode_idx + 1].plot(components[mode_idx, channel_idx], color='steelblue', linewidth=1.0, alpha=0.8)
            axes[mode_idx + 1].set_ylabel('Amplitude', fontsize=10)

            # Include center frequency in title if available
            if center_freqs is not None:
                title = f'$u_{{{mode_idx+1}}}$ (f = {center_freqs[mode_idx]:.3f} Hz)'
            else:
                title = f'$u_{{{mode_idx+1}}}$'

            axes[mode_idx + 1].set_title(title, fontsize=11, fontweight='bold', pad=8)
            axes[mode_idx + 1].grid(True, alpha=0.3, linestyle='--')
            axes[mode_idx + 1].spines['top'].set_visible(False)
            axes[mode_idx + 1].spines['right'].set_visible(False)

        # Set x-label on bottom plot
        axes[-1].set_xlabel('Time (samples)', fontsize=11)

        # Add figure title with channel information
        title_parts = ['Signal Decomposition']

        # Add channel number and label if available
        if channel_label_map is not None:
            channel_label = channel_label_map.get(channel_idx, f'Channel {channel_idx}')
            title_parts.append(f'Channel {channel_idx}: {channel_label}')
        else:
            title_parts.append(f'Channel {channel_idx}')

        # Add subject ID if available
        if subject_id is not None:
            title_parts.append(f'({subject_id})')

        fig.suptitle(' - '.join(title_parts), fontsize=14, fontweight='bold', y=0.995)

        # Use tight_layout
        plt.tight_layout(pad=1.5, h_pad=1.5, rect=[0, 0, 1, 0.99])

        current_batch.append(fig)

        # Yield batch when it reaches max size
        if len(current_batch) >= max_figures_per_batch:
            yield current_batch
            current_batch = []

    # Yield any remaining figures
    if current_batch:
        yield current_batch

def plot_slow_band_decomposition(original, band_signals, subject_id=None, channel_label_map=None, max_figures_per_batch=20):
    """
    Plot signal decomposition organized by slow frequency bands for each channel separately.
    Only plots the banded signals (Slow-2 through Slow-6), excluding frequencies outside all bands.

    Args:
        original: The original signal that has been decomposed (channels x samples)
        band_signals: Dictionary from combine_slow_band_components() containing band-separated signals
        subject_id: Optional subject identifier to add to figure title
        channel_label_map: Optional dictionary mapping channel indices to labels
        max_figures_per_batch: Maximum number of figures to create per batch to avoid memory issues

    Returns:
        generator: Generator yielding batches of matplotlib.figure.Figure objects
    """
    channel_count = original.shape[0]
    current_batch = []

    # Define band order for plotting: Slow-6 through Slow-2 (exclude 'excluded')
    band_order = ['6', '5', '4', '3', '2']
    band_names = {
        '6': 'Slow-6 (0-0.01 Hz)',
        '5': 'Slow-5 (0.01-0.027 Hz)',
        '4': 'Slow-4 (0.027-0.073 Hz)',
        '3': 'Slow-3 (0.073-0.198 Hz)',
        '2': 'Slow-2 (0.198-0.25 Hz)',
    }

    # Count total subplots needed per channel (original + slow bands only)
    bands_with_data = []
    for band_key in band_order:
        if band_key in band_signals and len(band_signals[band_key]['components']) > 0:
            bands_with_data.append(band_key)

    subplot_count = 1 + len(bands_with_data)  # Original + slow bands only

    # Create a separate figure for each channel
    for channel_idx in range(channel_count):
        fig, axes = plt.subplots(subplot_count, 1, figsize=(14, 2 * subplot_count), sharex=True)

        # Handle single subplot case
        if subplot_count == 1:
            axes = [axes]

        current_subplot = 0

        # Plot original signal for this channel
        axes[current_subplot].plot(original[channel_idx], color='black', linewidth=1.2, alpha=0.8)
        axes[current_subplot].set_ylabel('Amplitude', fontsize=10)
        axes[current_subplot].set_title('Original Signal', fontsize=11, fontweight='bold', pad=8)
        axes[current_subplot].grid(True, alpha=0.3, linestyle='--')
        axes[current_subplot].spines['top'].set_visible(False)
        axes[current_subplot].spines['right'].set_visible(False)
        current_subplot += 1

        # Plot slow band signals (summed components for each band)
        for band_key in ['6', '5', '4', '3', '2']:
            if band_key in band_signals and len(band_signals[band_key]['components']) > 0:
                band_signal = band_signals[band_key]['band_signal']
                center_freqs = band_signals[band_key]['center_freqs']

                axes[current_subplot].plot(band_signal[channel_idx], color='steelblue', linewidth=1.0, alpha=0.8)
                axes[current_subplot].set_ylabel('Amplitude', fontsize=10)

                # Create title with band name and center frequencies
                freq_str = ', '.join([f'{f:.3f}' for f in center_freqs])
                # Extract frequency range from band_names
                freq_range = band_names[band_key].split('(')[1].rstrip(')')
                title = f'Slow-{band_key} ({freq_range}) - f = [{freq_str}] Hz'

                axes[current_subplot].set_title(title, fontsize=11, fontweight='bold', pad=8)
                axes[current_subplot].grid(True, alpha=0.3, linestyle='--')
                axes[current_subplot].spines['top'].set_visible(False)
                axes[current_subplot].spines['right'].set_visible(False)
                current_subplot += 1

        # Set x-label on bottom plot
        axes[-1].set_xlabel('Time (samples)', fontsize=11)

        # Add figure title with channel information
        title_parts = ['Slow Band Decomposition']

        # Add channel number and label if available
        if channel_label_map is not None:
            channel_label = channel_label_map.get(channel_idx, f'Channel {channel_idx}')
            title_parts.append(f'Channel {channel_idx}: {channel_label}')
        else:
            title_parts.append(f'Channel {channel_idx}')

        # Add subject ID if available
        if subject_id is not None:
            title_parts.append(f'({subject_id})')

        fig.suptitle(' - '.join(title_parts), fontsize=14, fontweight='bold', y=0.995)

        # Use tight_layout
        plt.tight_layout(pad=1.5, h_pad=1.5, rect=[0, 0, 1, 0.99])

        current_batch.append(fig)

        # Yield batch when it reaches max size
        if len(current_batch) >= max_figures_per_batch:
            yield current_batch
            current_batch = []

    # Yield any remaining figures
    if current_batch:
        yield current_batch

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

        # Signal properties
        TR = 800e-3 # Repetition Time [seconds]
        sampling_rate = 1 / TR  # 1.25 Hz
        nyquist_frequency = 0.5 * sampling_rate  # 0.625 Hz

        # Filter properties
        filter_order = 2
        cutoff_frequency = 0.2  # Frequency at which signal starts to attenuate
                                # Digital filter critical frequencies must be 0 < Wn < 1
        normalized_cutoff = cutoff_frequency / nyquist_frequency
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

            if verbose:
                pattern_labels = '\n\t- '.join(connectivity_patterns['interhemispheric']['pairs'].keys())
                print(f"\nInterhemispheric connections (same order): \n\t- {pattern_labels}")

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

        # Compute slow-band FC after MVMD decomposition
        slow_band_fc_results = {}
        if mvmd_result['success']:
            # Combine modes into slow-bands
            band_signals = combine_slow_band_components(time_modes, center_freqs, verbose=verbose)

            for band_key in ['6', '5', '4', '3', '2']:
                band_data = band_signals.get(band_key)

                # Skip if no modes in this band
                if band_data is None or band_data.get('band_signal') is None:
                    if verbose:
                        print(f"\nSlow-{band_key}: No modes assigned to this band, skipping FC")
                    continue

                band_signal = band_data['band_signal']  # Shape: (n_channels, n_samples)
                n_channels, n_samples = band_signal.shape

                # Detect available channels
                available_mask = detect_available_channels(band_signal, threshold=1e-10)
                n_available = np.sum(available_mask)

                if verbose:
                    print(f"\nSlow-{band_key} FC computation:")
                    print(f"  Band signal shape: {band_signal.shape}")
                    print(f"  Available channels: {n_available}/{n_channels}")
                    print(f"  Components used: {band_data['indeces']}")
                    print(f"  Center frequencies: {band_data['center_freqs']}")

                # Need at least 2 channels for correlation
                if n_available < 2:
                    if verbose:
                        print(f"  Insufficient available channels, skipping FC computation")
                    continue

                # Create FC timeseries dict (include all channels)
                band_fc_timeseries = {
                    all_channel_labels[ch]: band_signal[ch, :]
                    for ch in range(n_channels)
                }

                # Compute FC matrix for all channels
                band_fc_matrix, band_fc_labels, band_fc_pvalues = compute_fc_matrix(
                    band_fc_timeseries,
                    roi_names=all_channel_labels
                )

                # Set unavailable channel correlations to NaN
                unavailable_indices = np.where(~available_mask)[0]
                for idx in unavailable_indices:
                    band_fc_matrix[idx, :] = np.nan
                    band_fc_matrix[:, idx] = np.nan
                    band_fc_pvalues[idx, :] = np.nan
                    band_fc_pvalues[:, idx] = np.nan

                # Analyze connectivity patterns (will skip NaN pairs automatically)
                band_connectivity_patterns = analyze_connectivity_patterns(
                    band_fc_matrix,
                    band_fc_labels,
                    band_fc_pvalues,
                    alpha=0.05
                )

                # Store results
                slow_band_fc_results[f'slow-{band_key}'] = {
                    'fc_matrix': band_fc_matrix,
                    'fc_labels': band_fc_labels,
                    'fc_pvalues': band_fc_pvalues,
                    'connectivity_patterns': band_connectivity_patterns,
                    'available_channels': available_mask,
                    'n_available_channels': n_available,
                    'n_unavailable_channels': n_channels - n_available,
                    'unavailable_indices': unavailable_indices.tolist(),
                    'components_used': band_data['indeces'].tolist(),
                    'center_freqs': band_data['center_freqs'].tolist(),
                    'band_signal': band_signal,
                    'frequency_range': get_frequency_range(band_key),
                    'channel_label_map': mvmd_channel_label_map,
                }

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
    accessible_anhedonic = []
    accessible_non_anhedonic = []

    # Check anhedonic subjects (hammer task only)
    for subject_id in anhedonic_subjects:
        try:
            # Get only hammer task files
            hammer_files = manager.get_subject_files_by_task(subject_id, 'timeseries', 'hammer')
            if hammer_files:
                # Check if at least one hammer file is actually downloaded (not a git-annex symlink)
                first_file = loader.resolve_file_path(hammer_files[0])
                if is_actual_file(first_file):
                    accessible_anhedonic.append(subject_id)
                else:
                    print(f"    Warning: {subject_id} - hammer file exists but not downloaded (git-annex symlink)")
            else:
                print(f"    Warning: {subject_id} - no hammer task files available")
        except Exception as e:
            print(f"    Error accessing {subject_id}: {e}")

    # Check non-anhedonic subjects (hammer task only)
    for subject_id in non_anhedonic_subjects:
        try:
            # Get only hammer task files
            hammer_files = manager.get_subject_files_by_task(subject_id, 'timeseries', 'hammer')
            if hammer_files:
                first_file = loader.resolve_file_path(hammer_files[0])
                if is_actual_file(first_file):
                    accessible_non_anhedonic.append(subject_id)
                else:
                    print(f"    Warning: {subject_id} - hammer file exists but not downloaded (git-annex symlink)")
            else:
                print(f"    Warning: {subject_id} - no hammer task files available")
        except Exception as e:
            print(f"    Error accessing {subject_id}: {e}")

    # Report final accessible counts
    print(f"\nFinal Processing Summary:")
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
        ('Non-anhedonic', non_anhedonic_subjects),
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
    slow_bands = ['slow-6', 'slow-5', 'slow-4', 'slow-3', 'slow-2']

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

    # ===== INDIVIDUAL SUBJECT PLOTS =====
    individual_plots_created = 0
    figures_saved_count = 0

    # Create output directories for different analysis types
    fc_figures_dir = None
    mvmd_figures_dir = None
    roi_figures_dir = None

    if save_figures and create_plots:
        fc_figures_dir = run_parent_dir / 'figures' / 'fc_analysis'
        mvmd_figures_dir = run_parent_dir / 'figures' / 'mvmd_analysis'
        roi_figures_dir = run_parent_dir / 'figures' / 'roi_extraction'

        fc_figures_dir.mkdir(parents=True, exist_ok=True)
        mvmd_figures_dir.mkdir(parents=True, exist_ok=True)
        roi_figures_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[INFO] Figures will be saved to:")
        print(f"  FC analysis: {fc_figures_dir}")
        print(f"  MVMD analysis: {mvmd_figures_dir}")
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
            'mvmd_slow_bands': []
        }

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
                        'save_path': roi_figures_dir / f'{subject_id}_roi_timeseries_cortical_yeo17.svg' if save_figures and roi_figures_dir else None
                    })
                    plots_for_subject += 1

            # 2. Prepare subcortical ROI timeseries plot
            if roi_results.get('subcortical'):
                subcortical_data = roi_results['subcortical']
                if subcortical_data.get('extraction_successful'):
                    plot_batches['roi_subcortical'].append({
                        'subject_id': subject_id,
                        'data': subcortical_data,
                        'save_path': roi_figures_dir / f'{subject_id}_roi_timeseries_subcortical_tian.svg' if save_figures and roi_figures_dir else None
                    })
                    plots_for_subject += 1

            # 3. Skip averaged signals plot (no longer applicable with individual channels)
            # Individual channel signals are preserved - averaging would destroy temporal information

            # # 4. Plot analytic signal with envelopes (raw and LP-filtered)
            # # COMMENTED OUT: Generates a large number of figures (one per channel) which can cause memory issues
            # activity_data = result.get('activity')
            # if activity_data:
            #     # Extract required data
            #     analytic_signal = activity_data.get('analytic_signal')
            #     analytic_envelope = activity_data.get('analytic_envelope')
            #     smoothed_envelope = activity_data.get('smoothed_envelope')
            #     channel_labels = activity_data.get('channel_labels')
            #
            #     if analytic_signal is not None and analytic_envelope is not None and smoothed_envelope is not None:
            #         # Plot with raw analytic envelope
            #         print(f"  Creating analytic signal with raw envelope plot for {subject_id}...")
            #         raw_envelope_fig = plot_timeseries_with_envelopes(
            #             analytic_signal,
            #             analytic_envelope,
            #             smoothed_envelope,
            #             channel_labels,
            #             subject_id=subject_id,
            #             envelope_type='raw'
            #         )
            #         plots_for_subject += 1
            #
            #         # Save figure if enabled
            #         if save_figures and roi_figures_dir:
            #             fig_path = roi_figures_dir / f'{subject_id}_analytic_signal_raw_envelope.svg'
            #             raw_envelope_fig.savefig(fig_path, format='svg', bbox_inches='tight', dpi=300)
            #             figures_saved_count += 1
            #
            #         # Plot with LP-filtered envelope
            #         print(f"  Creating analytic signal with LP-filtered envelope plot for {subject_id}...")
            #         filtered_envelope_fig = plot_timeseries_with_envelopes(
            #             analytic_signal,
            #             analytic_envelope,
            #             smoothed_envelope,
            #             channel_labels,
            #             subject_id=subject_id,
            #             envelope_type='filtered'
            #         )
            #         plots_for_subject += 1
            #
            #         # Save figure if enabled
            #         if save_figures and roi_figures_dir:
            #             fig_path = roi_figures_dir / f'{subject_id}_analytic_signal_filtered_envelope.svg'
            #             filtered_envelope_fig.savefig(fig_path, format='svg', bbox_inches='tight', dpi=300)
            #             figures_saved_count += 1

            # 5. Prepare static functional connectivity analysis plot
            if result.get('static_functional_connectivity'):
                static_fc_data = result['static_functional_connectivity']
                if static_fc_data.get('static_fc_matrix') is not None:
                    plot_batches['fc_static'].append({
                        'subject_id': subject_id,
                        'subject_group': subject_group,
                        'data': static_fc_data,
                        'mask_diagonal': mask_diagonal,
                        'mask_nonsignificant': mask_nonsignificant,
                        'save_path': fc_figures_dir / f'{subject_id}_static_fc_pearson_correlation.svg' if save_figures and fc_figures_dir else None
                    })
                    plots_for_subject += 1

            # 5b. Prepare slow-band FC plots
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
                            'save_path': fc_figures_dir / f'{subject_id}_{band_key}_fc.svg' if save_figures and fc_figures_dir else None
                        })
                        plots_for_subject += 1

            # 6. Prepare MVMD decomposition plots
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

            # 7. Prepare MVMD slow-band plots
            if result.get('mvmd'):
                mvmd_data = result['mvmd']
                if mvmd_data.get('time_modes') is not None:
                    center_freqs = mvmd_data['center_freqs'][-1, :] if mvmd_data.get('center_freqs') is not None else None
                    band_signals = combine_slow_band_components(mvmd_data['time_modes'], center_freqs, verbose=verbose)
                    channel_label_map = mvmd_data.get('channel_label_map')

                    plot_batches['mvmd_slow_bands'].append({
                        'subject_id': subject_id,
                        'mvmd_data': mvmd_data,
                        'band_signals': band_signals,
                        'channel_label_map': channel_label_map,
                        'save_dir': mvmd_figures_dir if save_figures and mvmd_figures_dir else None
                    })

                    # Estimate channel count for progress tracking
                    channel_count = mvmd_data['original'].shape[0] if 'original' in mvmd_data else 0
                    plots_for_subject += channel_count


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
                    'save_path': fc_figures_dir / f'group_avg_{group_name_clean}_static_fc.svg' if save_figures and fc_figures_dir else None,
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
                        'save_path': fc_figures_dir / f'group_avg_{group_name_clean}_{band_key}_fc.svg' if save_figures and fc_figures_dir else None,
                        'is_slow_band': True,
                        'band_key': band_key
                    })

        print(f"  ✓ Prepared {len(plot_batches['fc_group_avg'])} group-averaged FC plots")

        # Now create and display plots in batches by type
        print(f"\n{'='*80}")
        print(f"CREATING AND DISPLAYING PLOTS BY TYPE")
        print(f"{'='*80}")

        # Batch 1: ROI Cortical Timeseries
        if plot_batches['roi_cortical']:
            print(f"\n[Batch 1/7] Creating {len(plot_batches['roi_cortical'])} cortical ROI timeseries plots...")
            for plot_info in plot_batches['roi_cortical']:
                figures = plot_roi_timeseries_result(plot_info['data'], subject_id=plot_info['subject_id'], atlas_type='Cortical')
                # plot_roi_timeseries_result now returns a list of figures (one per ROI)
                for fig in figures:
                    if plot_info['save_path']:
                        # Generate separate filenames for each ROI
                        # Extract ROI name from figure title
                        fig_title = fig._suptitle.get_text() if fig._suptitle else ''
                        roi_name = 'unknown'
                        if 'PFCm' in fig_title:
                            roi_name = 'PFCm'
                        elif 'PFCv' in fig_title:
                            roi_name = 'PFCv'

                        # Create ROI-specific filename
                        save_path = plot_info['save_path']
                        save_path_with_roi = save_path.parent / f"{save_path.stem}_{roi_name}{save_path.suffix}"
                        fig.savefig(save_path_with_roi, format='svg', bbox_inches='tight', dpi=300)
                        figures_saved_count += 1
                    if not show_plots:
                        plt.close(fig)
            if show_plots:
                print(f"  Displaying {len(plot_batches['roi_cortical'])} cortical ROI plots. Close all figures to continue...")
                plt.show()

        # Batch 2: ROI Subcortical Timeseries
        if plot_batches['roi_subcortical']:
            print(f"\n[Batch 2/7] Creating {len(plot_batches['roi_subcortical'])} subcortical ROI timeseries plots...")
            for plot_info in plot_batches['roi_subcortical']:
                figures = plot_roi_timeseries_result(plot_info['data'], subject_id=plot_info['subject_id'], atlas_type='Subcortical')
                # plot_roi_timeseries_result now returns a list of figures (one per ROI)
                for fig in figures:
                    if plot_info['save_path']:
                        # Generate separate filenames for each ROI
                        # Extract ROI name from figure title
                        fig_title = fig._suptitle.get_text() if fig._suptitle else ''
                        roi_name = 'unknown'
                        if 'AMY' in fig_title:
                            roi_name = 'AMY'

                        # Create ROI-specific filename
                        save_path = plot_info['save_path']
                        save_path_with_roi = save_path.parent / f"{save_path.stem}_{roi_name}{save_path.suffix}"
                        fig.savefig(save_path_with_roi, format='svg', bbox_inches='tight', dpi=300)
                        figures_saved_count += 1
                    if not show_plots:
                        plt.close(fig)
            if show_plots:
                print(f"  Displaying {len(plot_batches['roi_subcortical'])} subcortical ROI plots. Close all figures to continue...")
                plt.show()

        # Batch 3: Static FC Analysis
        if plot_batches['fc_static']:
            print(f"\n[Batch 3/7] Creating {len(plot_batches['fc_static'])} static FC plots...")
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
                print(f"  Displaying {len(plot_batches['fc_static'])} FC plots. Close all figures to continue...")
                plt.show()

        # Batch 4: Slow-Band FC Analysis
        if plot_batches['fc_slow_bands']:
            print(f"\n[Batch 4/7] Creating {len(plot_batches['fc_slow_bands'])} slow-band FC plots...")
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
                print(f"  Displaying {len(plot_batches['fc_slow_bands'])} slow-band FC plots. Close all figures to continue...")
                plt.show()

        # Batch 5: Group-Averaged FC Analysis
        if plot_batches['fc_group_avg']:
            print(f"\n[Batch 5/7] Creating {len(plot_batches['fc_group_avg'])} group-averaged FC plots...")
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

        # Batch 6: MVMD Mode Decomposition
        if plot_batches['mvmd_modes']:
            total_mode_figs = sum(p['mvmd_data']['original'].shape[0] for p in plot_batches['mvmd_modes'])
            print(f"\n[Batch 6/7] Creating {total_mode_figs} MVMD mode decomposition plots ({len(plot_batches['mvmd_modes'])} subjects)...")

            # Process in sub-batches of 20 figures to avoid memory issues
            MAX_FIGS_PER_BATCH = 20
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

        # Batch 7: MVMD Slow-Band Decomposition
        if plot_batches['mvmd_slow_bands']:
            total_band_figs = sum(p['mvmd_data']['original'].shape[0] for p in plot_batches['mvmd_slow_bands'])
            print(f"\n[Batch 7/7] Creating {total_band_figs} MVMD slow-band plots ({len(plot_batches['mvmd_slow_bands'])} subjects)...")
            current_batch_figs = []

            for plot_info in plot_batches['mvmd_slow_bands']:
                slow_band_generator = plot_slow_band_decomposition(
                    plot_info['mvmd_data']['original'],
                    plot_info['band_signals'],
                    subject_id=plot_info['subject_id'],
                    channel_label_map=plot_info['channel_label_map'],
                    max_figures_per_batch=MAX_FIGS_PER_BATCH
                )

                # Process each batch of figures from the generator
                channel_idx_base = 0
                for slow_band_figures in slow_band_generator:
                    if plot_info['save_dir']:
                        # Create subject-specific subdirectory for organization
                        subject_mvmd_dir = plot_info['save_dir'] / plot_info['subject_id']
                        subject_mvmd_dir.mkdir(parents=True, exist_ok=True)

                        for fig_idx, fig in enumerate(slow_band_figures):
                            channel_idx = channel_idx_base + fig_idx
                            if plot_info['channel_label_map'] is not None:
                                channel_label = plot_info['channel_label_map'].get(channel_idx, f'ch{channel_idx}')
                                channel_label_clean = channel_label.replace('/', '_').replace(' ', '_')
                            else:
                                channel_label_clean = f'ch{channel_idx}'

                            fig_path = subject_mvmd_dir / f'mvmd_slow_bands_decomposition_{channel_label_clean}.svg'
                            fig.savefig(fig_path, format='svg', bbox_inches='tight', dpi=300)
                            figures_saved_count += 1

                    if show_plots:
                        current_batch_figs.extend(slow_band_figures)

                        # Display and clear batch when reaching limit
                        if len(current_batch_figs) >= MAX_FIGS_PER_BATCH:
                            print(f"  Displaying {len(current_batch_figs)} MVMD slow-band plots. Close all figures to continue...")
                            plt.show()
                            current_batch_figs = []
                    else:
                        for fig in slow_band_figures:
                            plt.close(fig)

                    channel_idx_base += len(slow_band_figures)

            # Display remaining figures if any
            if show_plots and current_batch_figs:
                print(f"  Displaying {len(current_batch_figs)} MVMD slow-band plots. Close all figures to continue...")
                plt.show()

        print(f"\n{'='*80}")
        print(f"PLOTTING COMPLETE")
        print(f"{'='*80}")

        # Summary of saved figures
        if save_figures and figures_saved_count > 0:
            print(f"✓ Saved {figures_saved_count} figures across multiple directories:")
            print(f"  FC analysis: {fc_figures_dir}")
            print(f"  MVMD analysis: {mvmd_figures_dir}")
            print(f"  ROI extraction: {roi_figures_dir}")
    elif not create_plots:
        print(f"\n[INFO] Plot creation disabled (CREATE_PLOTS=False)")
    else:
        print(f"\n[INFO] No plots created (no successful subjects)")

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
            print(f"    - FC analysis: {fc_figures_dir}")
            print(f"    - MVMD analysis: {mvmd_figures_dir}")
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

    args = parser.parse_args()

    # Display configuration
    VERBOSE_OUTPUT = args.verbose
    CREATE_PLOTS = True  # Whether to create plots (required for both displaying and saving)
    SHOW_PLOTS = args.show_plots  # Whether to display plots interactively (requires CREATE_PLOTS=True)
    SAVE_FIGURES = not args.no_save  # Whether to save figures to disk as SVG files (requires CREATE_PLOTS=True)

    # FC Matrix display mode:
    # - False: Show all correlations, mark non-significant with asterisks
    # - True: Hide non-significant correlations (masked)
    MASK_NONSIGNIFICANT = False
    MASK_DIAGONAL = False

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
