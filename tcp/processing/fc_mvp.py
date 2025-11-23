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
            if ts.size > 0:  # Check if timeseries is not empty
                timeseries_list.append(ts)
                roi_labels.append(roi_name)

    if len(timeseries_list) < 2:
        print(f"[WARNING] Need at least 2 ROI timeseries for FC computation, got {len(timeseries_list)}")
        return None, roi_labels, None

    # Stack timeseries for correlation computation
    stacked_timeseries = np.vstack(timeseries_list)

    # Compute correlation matrix
    corr_matrix = np.corrcoef(stacked_timeseries)

    # Compute p-values for correlations
    n_rois = len(roi_labels)
    p_values = np.ones((n_rois, n_rois))

    for i in range(n_rois):
        for j in range(i+1, n_rois):
            # Compute p-value for correlation
            _, p_val = stats.pearsonr(timeseries_list[i], timeseries_list[j])
            p_values[i, j] = p_val
            p_values[j, i] = p_val  # Symmetric matrix

    return corr_matrix, roi_labels, p_values


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
            significant_count = sum(pair.get('significant', False) for pair in pairs.values())
            significance_pct = significant_count / pair_count
        else:
            significance_pct = 0.0

        # Store statistics separately from pair data
        results[connection_type]['stats'] = {
            'total_pairs': pair_count,
            'significant_pairs': significant_count if pair_count > 0 else 0,
            'significance_percentage': significance_pct
        }

    return results


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

    # Return the first figure (or create empty if none)
    if figures:
        return figures[0]
    else:
        fig = plt.figure(figsize=(16, 6))
        fig.text(0.5, 0.5, 'No matching ROIs found',
                ha='center', va='center', fontsize=14, color='gray')
        return fig


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


def plot_fc_results(corr_matrix, roi_labels, p_values=None, connectivity_patterns=None, channel_label_map=None, alpha=0.05, mask_diagonal=False, mask_nonsignificant=False):
    """
    Create a clean visualization focusing on static FC matrix and interhemispheric connectivity.

    Args:
        corr_matrix: Correlation matrix
        roi_labels: ROI labels (channel keys)
        p_values: Optional p-values matrix
        connectivity_patterns: Optional results from analyze_connectivity_patterns
        channel_label_map: Dictionary mapping channel keys to descriptive labels
        alpha: Significance threshold for marking significant correlations
        mask_nonsignificant: If True, hide non-significant correlations. If False, mark them with asterisks.
    """
    fig = plt.figure(figsize=(16, 8))

    # 1. Static FC Matrix (left side)
    ax1 = plt.subplot(1, 2, 1)

    # Use descriptive labels if channel_label_map is provided
    display_labels = roi_labels
    if channel_label_map:
        display_labels = [channel_label_map.get(label, label) for label in roi_labels]

    # Create mask
    # Start with no masking
    mask_combined = None

    if mask_diagonal:
        # Mask the diagonal
        mask_combined = np.eye(corr_matrix.shape[0], dtype=bool)

    # Optionally mask non-significant correlations
    if mask_nonsignificant and p_values is not None:
        # Create non-significant mask, but exclude the diagonal from this check
        nonsig_mask = p_values >= alpha
        # Ensure diagonal is NOT masked by the nonsignificant filter (only by mask_diagonal flag)
        np.fill_diagonal(nonsig_mask, False)

        if mask_combined is not None:
            # Combine diagonal mask with non-significant mask
            mask_combined = mask_combined | nonsig_mask
        else:
            # Only mask non-significant correlations
            mask_combined = nonsig_mask

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

    # Add note about asterisk marker in the title (only if not masking)
    title_text = 'Static Functional Connectivity Matrix'
    if not mask_nonsignificant and p_values is not None:
        title_text += f'\n($*$ non-significant, p ≥ {alpha})'

    ax1.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
    ax1.tick_params(axis='x', rotation=90, labelsize=7)
    ax1.tick_params(axis='y', rotation=0, labelsize=7)

    # 2. Interhemispheric Connectivity Analysis (right side)
    ax2 = plt.subplot(1, 2, 2)

    if connectivity_patterns and 'interhemispheric' in connectivity_patterns:
        inter_data = connectivity_patterns['interhemispheric']['pairs']

        if inter_data:
            # Extract interhemispheric correlations and labels
            inter_pairs = list(inter_data.keys())
            inter_corrs = [v['correlation'] for v in inter_data.values()]

            # Helper function to extract region/network information from pair labels
            def parse_region_network(pair_key):
                """
                Parse pair key to extract region and network information.
                Returns (region_network_key, display_label, is_cortical)

                Cortical format: {region}_{hemi}_{network}_p{subarea}_{region}_{hemi}_{network}_p{subarea}
                Subcortical format: {region}_{hemi}_{subdivision}_{region}_{hemi}_{subdivision}
                """
                parts = pair_key.split('_')

                # Try to identify if this is cortical (has network info) or subcortical
                # Cortical labels have format: PFCm_RH_DefaultA_p1_PFCm_LH_DefaultA_p2
                # Subcortical labels have format: AMY_RH_lAMY_AMY_LH_lAMY

                if len(parts) >= 8:  # Cortical with network
                    # Extract first ROI components
                    region = parts[0]
                    network = parts[2] if len(parts) > 2 else 'Unknown'

                    # Create unique key: region+network (e.g., "PFCm_DefaultA")
                    region_network_key = f"{region}_{network}"
                    display_label = f"{region} ({network})"
                    is_cortical = True

                elif len(parts) >= 6:  # Subcortical without network
                    # Extract region and subdivision
                    region = parts[0]
                    subdivision = parts[2] if len(parts) > 2 else 'Unknown'

                    # Create unique key: region+subdivision (e.g., "AMY_lAMY")
                    region_network_key = f"{region}_{subdivision}"
                    display_label = f"{region} ({subdivision})"
                    is_cortical = False

                else:
                    # Fallback for unexpected formats
                    region_network_key = parts[0] if parts else 'Unknown'
                    display_label = region_network_key
                    is_cortical = False

                return region_network_key, display_label, is_cortical

            # Assign colors to unique region/network combinations
            unique_region_networks = {}
            pair_region_networks = []

            for pair_key in inter_pairs:
                region_network_key, display_label, is_cortical = parse_region_network(pair_key)
                pair_region_networks.append((region_network_key, display_label, is_cortical))

                if region_network_key not in unique_region_networks:
                    unique_region_networks[region_network_key] = {
                        'display_label': display_label,
                        'is_cortical': is_cortical,
                        'indices': []
                    }
                unique_region_networks[region_network_key]['indices'].append(len(pair_region_networks) - 1)

            # Define distinct color palette (using colorblind-friendly colors)
            # Using a qualitative palette with good distinction
            base_colors = [
                '#1f77b4',  # Blue
                '#ff7f0e',  # Orange
                '#2ca02c',  # Green
                '#d62728',  # Red
                '#9467bd',  # Purple
                '#8c564b',  # Brown
                '#e377c2',  # Pink
                '#7f7f7f',  # Gray
                '#bcbd22',  # Olive
                '#17becf',  # Cyan
                '#aec7e8',  # Light blue
                '#ffbb78',  # Light orange
                '#98df8a',  # Light green
                '#ff9896',  # Light red
                '#c5b0d5',  # Light purple
            ]

            # Assign colors to each unique region/network
            color_map = {}
            for idx, (region_network_key, info) in enumerate(unique_region_networks.items()):
                color_map[region_network_key] = base_colors[idx % len(base_colors)]

            # Prepare bar colors and hatching
            bar_colors = []
            bar_hatches = []
            for i, (corr_val, pair_data) in enumerate(zip(inter_corrs, inter_data.values())):
                region_network_key, _, _ = pair_region_networks[i]
                bar_colors.append(color_map[region_network_key])

                # Set hatch pattern for non-significant correlations
                # Use dense hatching pattern for better visibility
                if p_values is not None and not pair_data.get('significant', False):
                    bar_hatches.append('//////')  # Dense diagonal lines
                else:
                    bar_hatches.append(None)

            # Create all bars at once
            bars = ax2.bar(range(len(inter_corrs)), inter_corrs, color=bar_colors,
                          edgecolor='black', alpha=0.8, width=0.8, linewidth=0.8)

            # Apply hatching to non-significant bars
            for bar, hatch in zip(bars, bar_hatches):
                if hatch:
                    bar.set_hatch(hatch)
                    bar.set_alpha(0.7)

            # Remove x-axis tick labels (too cluttered with many connections)
            ax2.set_xticks([])
            ax2.set_xlabel(f'{len(inter_corrs)} interhemispheric connections', fontsize=11)
            ax2.set_ylabel('Pearson Correlation', fontsize=12)
            ax2.set_title('Interhemispheric Connectivity\n(Same Region, Different Hemispheres)',
                         fontsize=12, fontweight='bold', pad=20)
            ax2.set_ylim(-1, 1)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.grid(True, alpha=0.3, axis='y')

            # Find the maximum correlation value to position indicators above it
            max_corr = max(inter_corrs)
            indicator_offset = 0.08  # Offset above the max correlation
            bar_indicator_height = 0.03  # Height of the indicator bar
            bar_indicator_y = max_corr + indicator_offset

            # Threshold for determining if a region group is "narrow" (needs diagonal text)
            narrow_threshold = 12  # Groups with ≤12 bars are considered narrow
            # Character width estimate for determining if label is too wide (roughly 0.04 units per char at fontsize 7)
            char_width = 0.04

            # Determine the rightmost group index to apply reverse slant
            max_index = max(max(info['indices']) for info in unique_region_networks.values() if info['indices'])
            rightmost_threshold = max_index * 0.75  # Last 25% of bars

            for region_network_key, info in unique_region_networks.items():
                indices = info['indices']
                if not indices:
                    continue

                x_start = min(indices) - 0.4
                x_width = max(indices) - min(indices) + 0.8
                color = color_map[region_network_key]
                group_size = len(indices)

                # Estimate if label text is wider than the bar
                label_text_width = len(info['display_label']) * char_width
                is_label_too_wide = label_text_width > x_width

                # Draw compact colored rectangle spanning the group
                rect = plt.Rectangle(
                    (x_start, bar_indicator_y), x_width, bar_indicator_height,
                    facecolor=color, edgecolor='black', linewidth=0.5,
                    clip_on=False, alpha=0.9
                )
                ax2.add_patch(rect)

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
                     alpha=0.8, label='Non-significant (p ≥ 0.05)')
            )

            ax2.legend(handles=legend_elements, loc='lower right', fontsize=8,
                      framealpha=0.95, edgecolor='black')

        else:
            ax2.text(0.5, 0.5, 'No interhemispheric\nconnections found',
                    ha='center', va='center', transform=ax2.transAxes, fontsize=14, color='gray')
            ax2.set_title('Interhemispheric Connectivity\n(Same Region, Different Hemispheres)',
                         fontsize=13, fontweight='bold', pad=20)
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
    else:
        ax2.text(0.5, 0.5, 'No connectivity patterns\navailable',
                ha='center', va='center', transform=ax2.transAxes, fontsize=14, color='gray')
        ax2.set_title('Interhemispheric Connectivity\n(Same Region, Different Hemispheres)',
                     fontsize=13, fontweight='bold', pad=20)

    # Use sufficient padding for FC plots
    plt.tight_layout(pad=2.8)
    return fig


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
                fc_labels_ordered = '\n\t- '.join(sorted(fc_labels))
                print(f"FC Matrix shape: {fc_matrix.shape}")
                print(f"ROI labels (alphabetical): \n\t- {fc_labels_ordered}")

            connectivity_patterns = analyze_connectivity_patterns(fc_matrix, fc_labels, fc_pvalues)

            if verbose:
                pattern_labels = '\n\t- '.join(connectivity_patterns['interhemispheric']['pairs'].keys())
                print(f"\nInterhemispheric connections (alphabetical): \n\t- {pattern_labels}")

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
        mvmd_result = mvmd.decompose(all_channels, num_modes=5)

        time_modes = mvmd_result['time_modes']
        center_freqs = mvmd_result['center_freqs'][-1, :]

        reconstructed_timeseries = np.sum(time_modes, axis=0)
        reconstruction_error = np.linalg.norm(all_channels - reconstructed_timeseries) / np.linalg.norm(analytic_timeseries)

        mvmd_result = {
            **mvmd_result,
            'ts_reconstruction': reconstructed_timeseries,
            'reconstruction_error': reconstruction_error,
            'channel_label_map': channel_label_map,
        }

        if verbose:
            print(f"\nExtracted centre frequencies: {center_freqs} Hz")
            print(f"Modes shape: {time_modes.shape}")
            print(f"Signal reconstruction error: {reconstruction_error:.4f}")


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
        }

    except Exception as e:
        if verbose:
            print(f"[ERROR] Failed to process subject {subject_id}: {str(e)}")
        return {
            'subject_id': subject_id,
            'error': str(e),
            'success': False
        }


def main(mask_diagonal=False, mask_nonsignificant=False, create_plots=True, show_plots=True, save_figures=False, verbose=True):
    """Main function for FC MVP analysis"""
    print("=== Functional Connectivity MVP ===")

    # ===== CONFIGURATION FOR MULTI-SUBJECT ANALYSIS =====
    LIMIT_SUBJECTS = True  # Set False for full analysis
    MAX_SUBJECTS_PER_GROUP = 2  # Limit when testing (only used if LIMIT_SUBJECTS=True)

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
        anhedonic_subjects_to_process = accessible_anhedonic[:MAX_SUBJECTS_PER_GROUP]
        non_anhedonic_subjects_to_process = accessible_non_anhedonic[:MAX_SUBJECTS_PER_GROUP]
        print(f"LIMITING ENABLED: Processing {len(anhedonic_subjects_to_process)} anhedonic + {len(non_anhedonic_subjects_to_process)} non-anhedonic subjects")
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
    print(f"\n{'='*80}")
    print(f"EXPORTING STATIC FC RESULTS TO CSV")
    print(f"{'='*80}")

    # Create output directory for FC CSV files using cross-platform path system
    # This will automatically use the correct base path for macOS, Windows 11, or CentOS
    fc_output_dir = get_analysis_path('fc_analysis/static_fc')
    fc_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {fc_output_dir}")
    print(f"  (Platform-specific path automatically configured)")

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

    # ===== INDIVIDUAL SUBJECT PLOTS =====
    individual_plots_created = 0
    figures_saved_count = 0

    # Create output directory for saved figures if needed
    figures_output_dir = None
    if save_figures and create_plots:
        figures_output_dir = get_analysis_path('fc_analysis/figures')
        figures_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[INFO] Figures will be saved to: {figures_output_dir}")

    if create_plots and total_success > 0:
        print(f"\n{'='*80}")
        print(f"CREATING INDIVIDUAL SUBJECT PLOTS")
        print(f"{'='*80}")

        if show_plots and total_success > 3:
            print(f"[WARNING] {total_success} subjects processed - plots will be created but not displayed (too many)")
            print(f"          Set SHOW_PLOTS=False to suppress this warning")

        all_results = {**anhedonic_results, **non_anhedonic_results}
        for subject_id, result in all_results.items():
            if not result['success']:
                continue

            plots_for_subject = 0

            # 1. Plot cortical ROI timeseries
            roi_results = result.get('roi_extraction_results', {})
            if roi_results.get('cortical'):
                cortical_data = roi_results['cortical']
                if cortical_data.get('extraction_successful'):
                    print(f"  Creating cortical ROI timeseries plot for {subject_id}...")
                    cortical_roi_fig = plot_roi_timeseries_result(cortical_data, subject_id=subject_id, atlas_type='Cortical')
                    plots_for_subject += 1

                    # Save figure if enabled
                    if save_figures and figures_output_dir:
                        fig_path = figures_output_dir / f'{subject_id}_cortical_timeseries.svg'
                        cortical_roi_fig.savefig(fig_path, format='svg', bbox_inches='tight', dpi=300)
                        figures_saved_count += 1

            # 2. Plot subcortical ROI timeseries
            if roi_results.get('subcortical'):
                subcortical_data = roi_results['subcortical']
                if subcortical_data.get('extraction_successful'):
                    print(f"  Creating subcortical ROI timeseries plot for {subject_id}...")
                    subcortical_roi_fig = plot_roi_timeseries_result(subcortical_data, subject_id=subject_id, atlas_type='Subcortical')
                    plots_for_subject += 1

                    # Save figure if enabled
                    if save_figures and figures_output_dir:
                        fig_path = figures_output_dir / f'{subject_id}_subcortical_timeseries.svg'
                        subcortical_roi_fig.savefig(fig_path, format='svg', bbox_inches='tight', dpi=300)
                        figures_saved_count += 1

            # 3. Skip averaged signals plot (no longer applicable with individual channels)
            # Individual channel signals are preserved - averaging would destroy temporal information

            # 4. Plot analytic signal with envelopes (raw and LP-filtered)
            activity_data = result.get('activity')
            if activity_data:
                # Extract required data
                analytic_signal = activity_data.get('analytic_signal')
                analytic_envelope = activity_data.get('analytic_envelope')
                smoothed_envelope = activity_data.get('smoothed_envelope')
                channel_labels = activity_data.get('channel_labels')

                if analytic_signal is not None and analytic_envelope is not None and smoothed_envelope is not None:
                    # Plot with raw analytic envelope
                    print(f"  Creating analytic signal with raw envelope plot for {subject_id}...")
                    raw_envelope_fig = plot_timeseries_with_envelopes(
                        analytic_signal,
                        analytic_envelope,
                        smoothed_envelope,
                        channel_labels,
                        subject_id=subject_id,
                        envelope_type='raw'
                    )
                    plots_for_subject += 1

                    # Save figure if enabled
                    if save_figures and figures_output_dir:
                        fig_path = figures_output_dir / f'{subject_id}_analytic_signal_raw_envelope.svg'
                        raw_envelope_fig.savefig(fig_path, format='svg', bbox_inches='tight', dpi=300)
                        figures_saved_count += 1

                    # Plot with LP-filtered envelope
                    print(f"  Creating analytic signal with LP-filtered envelope plot for {subject_id}...")
                    filtered_envelope_fig = plot_timeseries_with_envelopes(
                        analytic_signal,
                        analytic_envelope,
                        smoothed_envelope,
                        channel_labels,
                        subject_id=subject_id,
                        envelope_type='filtered'
                    )
                    plots_for_subject += 1

                    # Save figure if enabled
                    if save_figures and figures_output_dir:
                        fig_path = figures_output_dir / f'{subject_id}_analytic_signal_filtered_envelope.svg'
                        filtered_envelope_fig.savefig(fig_path, format='svg', bbox_inches='tight', dpi=300)
                        figures_saved_count += 1

            # 5. Plot static functional connectivity analysis results (clean version)
            if result.get('static_functional_connectivity'):
                print(f"  Creating static FC analysis plot for {subject_id}...")
                static_fc_data = result['static_functional_connectivity']

                if static_fc_data.get('static_fc_matrix') is not None:
                    fc_fig = plot_fc_results(
                        static_fc_data['static_fc_matrix'],
                        static_fc_data['static_fc_labels'],
                        static_fc_data['static_fc_pvalues'],
                        static_fc_data['static_connectivity_patterns'],
                        static_fc_data.get('channel_label_map'),
                        mask_diagonal=mask_diagonal,
                        mask_nonsignificant=mask_nonsignificant
                    )
                    fc_fig.suptitle(f'Functional Connectivity Analysis - {subject_id}', fontsize=16, fontweight='bold')
                    plots_for_subject += 1

                    # Save figure if enabled
                    if save_figures and figures_output_dir:
                        fig_path = figures_output_dir / f'{subject_id}_static_fc_analysis.svg'
                        fc_fig.savefig(fig_path, format='svg', bbox_inches='tight', dpi=300)
                        figures_saved_count += 1

            if plots_for_subject > 0:
                individual_plots_created += 1
                print(f"  ✓ Created {plots_for_subject} plots for {subject_id}")

        print(f"\nCreated plots for {individual_plots_created} subjects")

        # Summary of saved figures
        if save_figures and figures_saved_count > 0:
            print(f"✓ Saved {figures_saved_count} figures to: {figures_output_dir}")
    elif not create_plots:
        print(f"\n[INFO] Plot creation disabled (CREATE_PLOTS=False)")
    else:
        print(f"\n[INFO] No plots created (no successful subjects)")

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
    # Display configuration
    VERBOSE_OUTPUT = True
    CREATE_PLOTS = True  # Whether to create plots (required for both displaying and saving)
    SHOW_PLOTS = True  # Whether to display plots interactively (requires CREATE_PLOTS=True)
    SAVE_FIGURES = False  # Whether to save figures to disk as SVG files (requires CREATE_PLOTS=True)

    # FC Matrix display mode:
    # - False: Show all correlations, mark non-significant with asterisks
    # - True: Hide non-significant correlations (masked)
    MASK_NONSIGNIFICANT = False
    MASK_DIAGONAL = False

    main(mask_diagonal=MASK_DIAGONAL, mask_nonsignificant=MASK_NONSIGNIFICANT, create_plots=CREATE_PLOTS, show_plots=SHOW_PLOTS, save_figures=SAVE_FIGURES, verbose=VERBOSE_OUTPUT)

    if CREATE_PLOTS and SHOW_PLOTS:
        plt.show()
