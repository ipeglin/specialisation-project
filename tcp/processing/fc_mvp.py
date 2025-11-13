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
from tcp.processing.context import ProcessingContext
from tcp.processing.roi import (
    CorticalAtlasLookup,
    ROIExtractionService,
    SubCorticalAtlasLookup,
)
from tcp.processing.services import (
    ActivityAnalysisService,
    DataLoadingService,
    FCAnalysisService,
    HemisphereExtractionService,
)


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

        # Extract all pairwise data
        all_pairwise = connectivity_patterns.get('all_pairwise', {})

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

            # Check in connectivity patterns
            if pair_key in connectivity_patterns.get('interhemispheric', {}):
                connection_type = 'interhemispheric'
                regions = connectivity_patterns['interhemispheric'][pair_key].get('region', '')
            elif pair_key in connectivity_patterns.get('cross_regional', {}):
                connection_type = 'cross_regional'
                regions = connectivity_patterns['cross_regional'][pair_key].get('regions', '')
                if pair_key in connectivity_patterns.get('ipsilateral', {}):
                    connection_type = 'ipsilateral_cross_regional'
                    hemispheres = connectivity_patterns['ipsilateral'][pair_key].get('hemisphere', '')
                elif pair_key in connectivity_patterns.get('contralateral', {}):
                    connection_type = 'contralateral_cross_regional'
                    hemispheres = connectivity_patterns['contralateral'][pair_key].get('hemispheres', '')

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
        dict: Dictionary with different connectivity pattern results
    """
    results = {
        'interhemispheric': {},
        'cross_regional': {},
        'ipsilateral': {},
        'contralateral': {},
        'all_pairwise': {}
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
            results['all_pairwise'][pair_key] = {
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
                    results['interhemispheric'][pair_key] = {
                        'correlation': corr_val,
                        'p_value': p_val,
                        'significant': is_significant,
                        'region': roi1_region
                    }

                # Cross-regional (different regions)
                elif roi1_region != roi2_region:
                    results['cross_regional'][pair_key] = {
                        'correlation': corr_val,
                        'p_value': p_val,
                        'significant': is_significant,
                        'regions': f"{roi1_region}_{roi2_region}"
                    }

                    # Ipsilateral (same hemisphere, different regions)
                    if roi1_hemi == roi2_hemi:
                        results['ipsilateral'][pair_key] = {
                            'correlation': corr_val,
                            'p_value': p_val,
                            'significant': is_significant,
                            'hemisphere': roi1_hemi,
                            'regions': f"{roi1_region}_{roi2_region}"
                        }

                    # Contralateral (different hemisphere, different regions)
                    elif roi1_hemi != roi2_hemi:
                        results['contralateral'][pair_key] = {
                            'correlation': corr_val,
                            'p_value': p_val,
                            'significant': is_significant,
                            'hemispheres': f"{roi1_hemi}_{roi2_hemi}",
                            'regions': f"{roi1_region}_{roi2_region}"
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


def plot_fc_results(corr_matrix, roi_labels, p_values=None, connectivity_patterns=None, channel_label_map=None, alpha=0.05, mask_nonsignificant=False):
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

    # Create mask for diagonal (autocorrelation always = 1, not interesting)
    mask_diagonal = np.eye(corr_matrix.shape[0], dtype=bool)

    # Optionally mask non-significant correlations
    mask_combined = mask_diagonal.copy()
    if mask_nonsignificant and p_values is not None:
        # Mask both diagonal and non-significant correlations
        mask_combined = mask_diagonal | (p_values >= alpha)

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
        inter_data = connectivity_patterns['interhemispheric']

        if inter_data:
            # Extract interhemispheric correlations and labels
            inter_pairs = list(inter_data.keys())
            inter_corrs = [v['correlation'] for v in inter_data.values()]

            # Create bar plot (no x-axis labels to avoid clutter with many connections)
            bars = ax2.bar(range(len(inter_corrs)), inter_corrs, color='lightcoral',
                          edgecolor='darkred', alpha=0.7, width=0.8)

            # Remove x-axis tick labels (too cluttered with many connections)
            ax2.set_xticks([])
            ax2.set_xlabel(f'{len(inter_corrs)} interhemispheric connections', fontsize=11)
            ax2.set_ylabel('Pearson Correlation', fontsize=12)
            # Main title larger, subtitle smaller
            ax2.set_title('Interhemispheric Connectivity\n(Same Region, Different Hemispheres)',
                         fontsize=12, fontweight='bold', pad=23)
            ax2.set_ylim(-1, 1)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.grid(True, alpha=0.3, axis='y')

            # Mark significant correlations with different color (no text labels)
            if p_values is not None:
                for i, (pair_key, pair_data) in enumerate(inter_data.items()):
                    if pair_data.get('significant', False):
                        bars[i].set_color('darkred')
                        bars[i].set_alpha(0.9)

            # Add legend for significance
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='lightcoral', edgecolor='darkred', alpha=0.7, label='Non-significant'),
                Patch(facecolor='darkred', edgecolor='darkred', alpha=0.9, label='Significant (p<0.05)')
            ]
            ax2.legend(handles=legend_elements, loc='upper right', fontsize=10)

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


def compare_fc_between_groups(group1_fc_results, group2_fc_results, group1_name="Group 1", group2_name="Group 2", alpha=0.05):
    """
    Compare functional connectivity patterns between two groups using statistical tests.

    Args:
        group1_fc_results: List of FC result dictionaries for group 1
        group2_fc_results: List of FC result dictionaries for group 2
        group1_name: Name of group 1 (for reporting)
        group2_name: Name of group 2 (for reporting)
        alpha: Significance threshold

    Returns:
        dict: Statistical comparison results
    """
    from scipy import stats as scipy_stats

    # Initialize comparison results
    comparison_results = {
        'group1_name': group1_name,
        'group2_name': group2_name,
        'comparisons': {}
    }

    # Pattern types to compare
    pattern_types = ['all_pairwise', 'interhemispheric', 'cross_regional', 'ipsilateral', 'contralateral']

    for pattern_type in pattern_types:
        # Collect correlations for this pattern from both groups
        group1_corrs = []
        group2_corrs = []

        for fc_result in group1_fc_results:
            connectivity_patterns = fc_result.get('static_connectivity_patterns', {})
            pattern_data = connectivity_patterns.get(pattern_type, {})
            for pair_data in pattern_data.values():
                group1_corrs.append(pair_data['correlation'])

        for fc_result in group2_fc_results:
            connectivity_patterns = fc_result.get('static_connectivity_patterns', {})
            pattern_data = connectivity_patterns.get(pattern_type, {})
            for pair_data in pattern_data.values():
                group2_corrs.append(pair_data['correlation'])

        # Skip if insufficient data
        if len(group1_corrs) < 2 or len(group2_corrs) < 2:
            comparison_results['comparisons'][pattern_type] = {
                'note': f'Insufficient data for {pattern_type} comparison'
            }
            continue

        # Compute statistics
        group1_mean = np.mean(group1_corrs)
        group1_std = np.std(group1_corrs, ddof=1)
        group2_mean = np.mean(group2_corrs)
        group2_std = np.std(group2_corrs, ddof=1)

        # Independent samples t-test
        t_stat, p_val = scipy_stats.ttest_ind(group1_corrs, group2_corrs)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(group1_corrs) - 1) * group1_std**2 +
                             (len(group2_corrs) - 1) * group2_std**2) /
                            (len(group1_corrs) + len(group2_corrs) - 2))
        cohens_d = (group1_mean - group2_mean) / pooled_std if pooled_std > 0 else 0

        comparison_results['comparisons'][pattern_type] = {
            'group1_mean': group1_mean,
            'group1_std': group1_std,
            'group1_n': len(group1_corrs),
            'group2_mean': group2_mean,
            'group2_std': group2_std,
            'group2_n': len(group2_corrs),
            'ttest_statistic': t_stat,
            'ttest_pvalue': p_val,
            'cohens_d': cohens_d,
            'significant': p_val < alpha
        }

    return comparison_results


def process_subject(subject_id: str, context: ProcessingContext):
    """
    Process a single subject for ROI extraction and functional connectivity analysis.

    This function orchestrates the entire processing pipeline for a single subject,
    using dependency injection through the ProcessingContext and service classes.

    Args:
        subject_id: Subject identifier
        context: ProcessingContext containing all dependencies

    Returns:
        dict: Subject analysis results
    """
    verbose = context.verbose

    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing Subject: {subject_id}")
        print(f"{'='*60}")

    try:
        # 1. Get subject file path
        hammer_files = context.manager.get_subject_files_by_task(subject_id, 'timeseries', 'hammer')
        if not hammer_files:
            return {
                'subject_id': subject_id,
                'error': 'No hammer task files found',
                'success': False
            }

        subject_file = context.loader.resolve_file_path(hammer_files[0])

        # 2. Load and segment timeseries data
        data_service = DataLoadingService()
        segmented_data = data_service.load_and_segment_timeseries(subject_file, verbose=verbose)

        # 3. Extract hemisphere-specific ROI timeseries with labels
        hemisphere_service = HemisphereExtractionService()

        # Cortical extraction
        if verbose:
            print(f"\n=== HEMISPHERE-SPECIFIC CORTICAL EXTRACTION ===")

        cortical_R_channels, cortical_L_channels, cortical_R_labels, cortical_L_labels = \
            hemisphere_service.extract_with_labels(
                segmented_data['cortical'],  # Pass full 400-parcel cortical timeseries
                context.cortical_extractor,
                context.cortical_extractor.atlas_lookup,
                context.cortical_rois,
                verbose=verbose
            )

        cortical_channels = np.vstack([cortical_R_channels, cortical_L_channels])
        cortical_labels = cortical_R_labels + cortical_L_labels

        if verbose:
            print(f"Individual cortical ({', '.join(context.cortical_rois)}) channel extraction results:")
            print(f"  RIGHT channels: shape {cortical_R_channels.shape}")
            print(f"  LEFT channels: shape {cortical_L_channels.shape}")
            print(f"  Cortical channel labels: {len(cortical_labels)} channels")

        # Subcortical extraction
        if verbose:
            print(f"\n=== HEMISPHERE-SPECIFIC SUBCORTICAL EXTRACTION ===")

        subcortical_R_channels, subcortical_L_channels, subcortical_R_labels, subcortical_L_labels = \
            hemisphere_service.extract_with_labels(
                segmented_data['subcortical'],  # Pass full 32-parcel subcortical timeseries
                context.subcortical_extractor,
                context.subcortical_extractor.atlas_lookup,
                context.subcortical_rois,
                verbose=verbose
            )

        subcortical_channels = np.vstack([subcortical_R_channels, subcortical_L_channels])
        subcortical_labels = subcortical_R_labels + subcortical_L_labels

        if verbose:
            print(f"Individual subcortical ({', '.join(context.subcortical_rois)}) channel extraction results:")
            print(f"  RIGHT channels: shape {subcortical_R_channels.shape}")
            print(f"  LEFT channels: shape {subcortical_L_channels.shape}")
            print(f"  Subcortical channel labels: {len(subcortical_labels)} channels")
            print(f"  Total channel mapping: {len(cortical_labels) + len(subcortical_labels)} channels")

        # 4. Combine all channels
        all_channels = np.vstack([cortical_channels, subcortical_channels])
        all_labels = cortical_labels + subcortical_labels

        # 5. Activity analysis (Hilbert transform, envelopes)
        activity_results = None
        if verbose:
            print(f"\n=== ACTIVITY ANALYSIS ===")

        activity_service = ActivityAnalysisService()
        activity_results = activity_service.compute_envelope_analysis(
            all_channels,
            all_labels,
            verbose=verbose
        )

        # 6. Static FC analysis
        static_fc_results = None
        if verbose:
            print(f"\n=== FUNCTIONAL CONNECTIVITY ANALYSIS ===")

        fc_service = FCAnalysisService()
        static_fc_results = fc_service.compute_static_fc(
            all_channels,
            all_labels,
            verbose=verbose
        )

        # 7. Package results
        return {
            'subject_id': subject_id,
            'success': True,
            'roi_extraction_results': {
                'cortical': {
                    'atlas_name': context.cortical_extractor.atlas_lookup.atlas_name,
                    'roi_timeseries': segmented_data['cortical'],
                    'requested_rois': context.cortical_rois,
                    'extraction_successful': True,
                    'hemisphere_specific': {
                        'right': cortical_R_channels,
                        'left': cortical_L_channels
                    },
                    'parcel_labels': {
                        'right': cortical_R_labels,
                        'left': cortical_L_labels
                    }
                },
                'subcortical': {
                    'atlas_name': context.subcortical_extractor.atlas_lookup.atlas_name,
                    'roi_timeseries': segmented_data['subcortical'],
                    'requested_rois': context.subcortical_rois,
                    'extraction_successful': True,
                    'hemisphere_specific': {
                        'right': subcortical_R_channels,
                        'left': subcortical_L_channels
                    },
                    'parcel_labels': {
                        'right': subcortical_R_labels,
                        'left': subcortical_L_labels
                    }
                }
            },
            'activity': activity_results,
            'static_functional_connectivity': static_fc_results,
            'channel_signals': {
                'all_channels': all_channels,
                'all_labels': all_labels
            }
        }

    except Exception as e:
        return {
            'subject_id': subject_id,
            'error': str(e),
            'success': False
        }


def setup_analysis_configuration(
    limit_subjects: bool = True,
    max_subjects_per_group: int = 2,
    show_individual_plots: bool = True,
    verbose_subject_output: bool = True,
    show_group_summary: bool = True
) -> dict:
    """
    Setup and display analysis configuration.

    Args:
        limit_subjects: Whether to limit number of subjects processed
        max_subjects_per_group: Maximum subjects per group when limiting
        show_individual_plots: Whether to show individual subject plots
        verbose_subject_output: Whether to show detailed per-subject output
        show_group_summary: Whether to show aggregated group analysis

    Returns:
        dict: Configuration dictionary
    """
    print("=== Functional Connectivity MVP ===")
    print(f"Configuration:")
    print(f"  Subject limiting: {'ENABLED' if limit_subjects else 'DISABLED'}")
    if limit_subjects:
        print(f"  Max subjects per group: {max_subjects_per_group}")
    print(f"  Individual plots: {'ENABLED' if show_individual_plots else 'DISABLED'}")
    print(f"  Verbose output: {'ENABLED' if verbose_subject_output else 'DISABLED'}")
    print()

    return {
        'limit_subjects': limit_subjects,
        'max_subjects_per_group': max_subjects_per_group,
        'show_individual_plots': show_individual_plots,
        'verbose_subject_output': verbose_subject_output,
        'show_group_summary': show_group_summary
    }


def initialize_infrastructure(manager: SubjectManager, loader: DataLoader) -> dict:
    """
    Initialize data infrastructure and display availability summary.

    Args:
        manager: SubjectManager instance
        loader: DataLoader instance

    Returns:
        dict: Infrastructure information including groups and availability
    """
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

    return {
        'groups': groups,
        'availability': availability,
        'use_downloaded_only': use_downloaded_only
    }


def select_and_validate_subjects(
    manager: SubjectManager,
    loader: DataLoader,
    group_name: str,
    use_downloaded_only: bool
) -> dict:
    """
    Select subjects for analysis and validate file access.

    Args:
        manager: SubjectManager instance
        loader: DataLoader instance
        group_name: Analysis group name
        use_downloaded_only: Whether to use only downloaded subjects

    Returns:
        dict: Subject lists and validation results
    """
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

    non_anhedonic_subjects = manager.filter_subjects(
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
            hammer_files = manager.get_subject_files_by_task(subject_id, 'timeseries', 'hammer')
            if hammer_files:
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
            'accessible_anhedonic': [],
            'accessible_non_anhedonic': []
        }

    # Show example hammer task file paths for first few accessible subjects
    if accessible_anhedonic:
        print(f"\nExample accessible hammer task file paths:")
        for subject_id in accessible_anhedonic[:2]:
            print(f"\n  Subject: {subject_id}")
            try:
                hammer_files = manager.get_subject_files_by_task(subject_id, 'timeseries', 'hammer')
                for file_path in hammer_files[:2]:
                    full_path = loader.resolve_file_path(file_path)
                    print(f"    {full_path}")
            except Exception as e:
                print(f"    Error: {e}")

    return {
        'accessible_anhedonic': accessible_anhedonic,
        'accessible_non_anhedonic': accessible_non_anhedonic
    }


def initialize_processing_context(
    manager: SubjectManager,
    loader: DataLoader,
    mask_nonsignificant: bool,
    create_plots: bool,
    show_plots: bool,
    save_figures: bool,
    verbose: bool
) -> ProcessingContext:
    """
    Initialize processing configuration and context with all dependencies.

    Args:
        manager: SubjectManager instance
        loader: DataLoader instance
        mask_nonsignificant: Whether to mask non-significant correlations
        create_plots: Whether to create plots
        show_plots: Whether to show plots interactively
        save_figures: Whether to save figures to disk
        verbose: Whether to show detailed output

    Returns:
        ProcessingContext: Initialized processing context
    """
    print(f"\n{'='*50}")
    print(f"INITIALIZING PROCESSING CONFIGURATION")
    print(f"{'='*50}")

    from tcp.processing.config.analysis_config import (
        ProcessingConfig as AnalysisProcessingConfig,
    )

    script_dir = Path(__file__).parent
    output_base = get_analysis_path('')

    processing_config = AnalysisProcessingConfig.create_default(
        script_dir=script_dir,
        output_base=output_base
    )

    # Override config with function parameters
    processing_config.plotting_config.create_plots = create_plots
    processing_config.plotting_config.show_plots = show_plots
    processing_config.plotting_config.mask_nonsignificant = mask_nonsignificant
    processing_config.output_config.save_figures = save_figures
    processing_config.verbose = verbose

    # Initialize atlases and extractors
    cortical_atlas = CorticalAtlasLookup(processing_config.atlas_config.cortical_lut_path)
    subcortical_atlas = SubCorticalAtlasLookup(processing_config.atlas_config.subcortical_lut_path)
    cortical_roi_extractor = ROIExtractionService(cortical_atlas)
    subcortical_roi_extractor = ROIExtractionService(subcortical_atlas)

    # Create ProcessingContext for dependency injection
    context = ProcessingContext(
        manager=manager,
        loader=loader,
        cortical_extractor=cortical_roi_extractor,
        subcortical_extractor=subcortical_roi_extractor,
        config=processing_config
    )

    print(f"Initialized atlases:")
    print(f"  Cortical: {cortical_atlas.atlas_name} ({cortical_atlas.total_parcels} parcels)")
    print(f"  Subcortical: {subcortical_atlas.atlas_name} ({subcortical_atlas.total_parcels} parcels)")
    print(f"ROIs of interest:")
    print(f"  Cortical: {processing_config.atlas_config.cortical_rois}")
    print(f"  Subcortical: {processing_config.atlas_config.subcortical_rois}")

    return context


def process_subject_group(
    subject_list: list,
    group_name: str,
    context: ProcessingContext
) -> dict:
    """
    Process all subjects in a group.

    Args:
        subject_list: List of subject IDs to process
        group_name: Name of the group (for display)
        context: ProcessingContext with all dependencies

    Returns:
        dict: Dictionary mapping subject IDs to their results
    """
    print(f"\n{'='*50}")
    print(f"PROCESSING {group_name.upper()} SUBJECTS ({len(subject_list)})")
    print(f"{'='*50}")

    results = {}

    for i, subject_id in enumerate(subject_list, 1):
        print(f"\n[{i}/{len(subject_list)}] Processing {group_name} subject: {subject_id}")

        subject_result = process_subject(subject_id, context)
        results[subject_id] = subject_result

        if subject_result['success']:
            print(f"    ✅ Success: {subject_id}")
        else:
            print(f"    ❌ Failed: {subject_id} - {subject_result.get('error', 'Unknown error')}")

    return results


def collect_fc_results(results_dict: dict) -> list:
    """
    Collect functional connectivity results from processed subjects.

    Args:
        results_dict: Dictionary of subject results

    Returns:
        list: List of FC results from successful subjects
    """
    fc_results = []
    for subject_id, result in results_dict.items():
        if result['success'] and result.get('static_functional_connectivity'):
            fc_results.append(result['static_functional_connectivity'])
    return fc_results


def perform_group_comparison(
    anhedonic_fc_results: list,
    non_anhedonic_fc_results: list
) -> dict:
    """
    Perform statistical comparison between two groups.

    Args:
        anhedonic_fc_results: FC results from anhedonic group
        non_anhedonic_fc_results: FC results from non-anhedonic group

    Returns:
        dict: Group comparison results, or None if insufficient data
    """
    if len(anhedonic_fc_results) == 0 or len(non_anhedonic_fc_results) == 0:
        print(f"\n[WARNING] Insufficient FC data for group comparison")
        print(f"  Need at least 1 subject per group with successful FC analysis")
        return None

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
        if not comparison.get('note'):
            print(f"\n{pattern_type.upper()}:")
            print(f"  Anhedonic: M={comparison['group1_mean']:.3f}, SD={comparison['group1_std']:.3f}, N={comparison['group1_n']}")
            print(f"  Non-anhedonic: M={comparison['group2_mean']:.3f}, SD={comparison['group2_std']:.3f}, N={comparison['group2_n']}")
            print(f"  t({comparison['group1_n']+comparison['group2_n']-2})={comparison['ttest_statistic']:.3f}, p={comparison['ttest_pvalue']:.3f}")
            print(f"  Cohen's d={comparison['cohens_d']:.3f}, Significant={'Yes' if comparison['significant'] else 'No'}")

    return group_comparison_results


def export_all_fc_results(
    all_results: dict,
    total_success: int
) -> int:
    """
    Export functional connectivity results to CSV files.

    Args:
        all_results: Combined dictionary of all subject results
        total_success: Total number of successful subjects

    Returns:
        int: Number of subjects with exported CSV files
    """
    print(f"\n{'='*80}")
    print(f"EXPORTING STATIC FC RESULTS TO CSV")
    print(f"{'='*80}")

    fc_output_dir = get_analysis_path('fc_analysis/static_fc')
    fc_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {fc_output_dir}")
    print(f"  (Platform-specific path automatically configured)")

    csv_export_count = 0

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

    return csv_export_count


def create_individual_plots(
    all_results: dict,
    total_success: int,
    show_individual_plots: bool,
    create_plots: bool,
    save_figures: bool,
    mask_nonsignificant: bool
) -> tuple:
    """
    Create individual subject plots.

    Args:
        all_results: Combined dictionary of all subject results
        total_success: Total number of successful subjects
        show_individual_plots: Whether to show individual plots
        create_plots: Whether to create plots
        save_figures: Whether to save figures to disk
        mask_nonsignificant: Whether to mask non-significant correlations

    Returns:
        tuple: (plots_created_count, figures_saved_count)
    """
    individual_plots_created = 0
    figures_saved_count = 0

    figures_output_dir = None
    if save_figures and create_plots:
        figures_output_dir = get_analysis_path('fc_analysis/figures')
        figures_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[INFO] Figures will be saved to: {figures_output_dir}")

    if not create_plots:
        print(f"\n[INFO] No individual plots created (create_plots set to False)")
        return individual_plots_created, figures_saved_count

    if not show_individual_plots or total_success > 3:
        if total_success > 3:
            print(f"\n[INFO] Skipping individual plots (too many subjects: {total_success})")
            print(f"      Individual plots only shown for ≤3 subjects")
        else:
            print(f"\n[INFO] No individual plots created (no successful subjects)")
        return individual_plots_created, figures_saved_count

    print(f"\n{'='*80}")
    print(f"CREATING INDIVIDUAL SUBJECT PLOTS")
    print(f"{'='*80}")

    for subject_id, result in all_results.items():
        if not result['success']:
            continue

        plots_for_subject = 0
        roi_results = result.get('roi_extraction_results', {})

        # Plot cortical ROI timeseries
        if roi_results.get('cortical', {}).get('extraction_successful'):
            print(f"  Creating cortical ROI timeseries plot for {subject_id}...")
            cortical_fig = plot_roi_timeseries_result(
                roi_results['cortical'], subject_id=subject_id, atlas_type='Cortical'
            )
            plots_for_subject += 1

            if save_figures and figures_output_dir:
                fig_path = figures_output_dir / f'{subject_id}_cortical_timeseries.svg'
                cortical_fig.savefig(fig_path, format='svg', bbox_inches='tight', dpi=300)
                figures_saved_count += 1

        # Plot subcortical ROI timeseries
        if roi_results.get('subcortical', {}).get('extraction_successful'):
            print(f"  Creating subcortical ROI timeseries plot for {subject_id}...")
            subcortical_fig = plot_roi_timeseries_result(
                roi_results['subcortical'], subject_id=subject_id, atlas_type='Subcortical'
            )
            plots_for_subject += 1

            if save_figures and figures_output_dir:
                fig_path = figures_output_dir / f'{subject_id}_subcortical_timeseries.svg'
                subcortical_fig.savefig(fig_path, format='svg', bbox_inches='tight', dpi=300)
                figures_saved_count += 1

        # Plot analytic signal with envelopes
        activity_data = result.get('activity')
        if activity_data:
            analytic_signal = activity_data.get('analytic_signal')
            analytic_envelope = activity_data.get('analytic_envelope')
            smoothed_envelope = activity_data.get('smoothed_envelope')
            channel_labels = activity_data.get('channel_labels')

            if all([analytic_signal is not None, analytic_envelope is not None, smoothed_envelope is not None]):
                # Raw envelope
                print(f"  Creating analytic signal with raw envelope plot for {subject_id}...")
                raw_fig = plot_timeseries_with_envelopes(
                    analytic_signal, analytic_envelope, smoothed_envelope,
                    channel_labels, subject_id=subject_id, envelope_type='raw'
                )
                plots_for_subject += 1

                if save_figures and figures_output_dir:
                    fig_path = figures_output_dir / f'{subject_id}_analytic_signal_raw_envelope.svg'
                    raw_fig.savefig(fig_path, format='svg', bbox_inches='tight', dpi=300)
                    figures_saved_count += 1

                # Filtered envelope
                print(f"  Creating analytic signal with LP-filtered envelope plot for {subject_id}...")
                filtered_fig = plot_timeseries_with_envelopes(
                    analytic_signal, analytic_envelope, smoothed_envelope,
                    channel_labels, subject_id=subject_id, envelope_type='filtered'
                )
                plots_for_subject += 1

                if save_figures and figures_output_dir:
                    fig_path = figures_output_dir / f'{subject_id}_analytic_signal_filtered_envelope.svg'
                    filtered_fig.savefig(fig_path, format='svg', bbox_inches='tight', dpi=300)
                    figures_saved_count += 1

        # Plot static FC
        static_fc_data = result.get('static_functional_connectivity')
        if static_fc_data and static_fc_data.get('static_fc_matrix') is not None:
            print(f"  Creating static FC analysis plot for {subject_id}...")
            fc_fig = plot_fc_results(
                static_fc_data['static_fc_matrix'],
                static_fc_data['static_fc_labels'],
                static_fc_data['static_fc_pvalues'],
                static_fc_data['static_connectivity_patterns'],
                static_fc_data.get('channel_label_map'),
                mask_nonsignificant=mask_nonsignificant
            )
            fc_fig.suptitle(f'Functional Connectivity Analysis - {subject_id}', fontsize=16, fontweight='bold')
            plots_for_subject += 1

            if save_figures and figures_output_dir:
                fig_path = figures_output_dir / f'{subject_id}_static_fc_analysis.svg'
                fc_fig.savefig(fig_path, format='svg', bbox_inches='tight', dpi=300)
                figures_saved_count += 1

        if plots_for_subject > 0:
            individual_plots_created += 1
            print(f"  ✓ Created {plots_for_subject} plots for {subject_id}")

    print(f"\nCreated plots for {individual_plots_created} subjects")

    if save_figures and figures_saved_count > 0:
        print(f"✓ Saved {figures_saved_count} figures to: {figures_output_dir}")

    return individual_plots_created, figures_saved_count


def main(mask_nonsignificant=False, create_plots=True, show_plots=True, save_figures=False):
    """
    Main orchestration function for functional connectivity MVP analysis.

    This function coordinates the entire analysis pipeline using focused helper functions:
    1. Configuration setup
    2. Infrastructure initialization
    3. Subject selection and validation
    4. Processing context initialization
    5. Multi-subject processing
    6. Group comparison
    7. CSV export
    8. Plotting

    Args:
        mask_nonsignificant: Whether to mask non-significant correlations in FC plots
        create_plots: Whether to create plots
        show_plots: Whether to display plots interactively
        save_figures: Whether to save figures to disk

    Returns:
        dict: Comprehensive analysis results
    """
    # Analysis configuration constants
    LIMIT_SUBJECTS = True
    MAX_SUBJECTS_PER_GROUP = 2
    SHOW_INDIVIDUAL_PLOTS = True
    VERBOSE_SUBJECT_OUTPUT = True
    SHOW_GROUP_SUMMARY = True

    # 1. Setup configuration
    config = setup_analysis_configuration(
        limit_subjects=LIMIT_SUBJECTS,
        max_subjects_per_group=MAX_SUBJECTS_PER_GROUP,
        show_individual_plots=SHOW_INDIVIDUAL_PLOTS,
        verbose_subject_output=VERBOSE_SUBJECT_OUTPUT,
        show_group_summary=SHOW_GROUP_SUMMARY
    )

    # 2. Initialize infrastructure
    loader = DataLoader()
    manager = SubjectManager(data_loader=loader)
    infrastructure = initialize_infrastructure(manager, loader)

    # 3. Select and validate subjects
    group_name = 'anhedonic_vs_non_anhedonic'
    subject_validation = select_and_validate_subjects(
        manager,
        loader,
        group_name,
        infrastructure['use_downloaded_only']
    )

    # Handle error case (no accessible subjects)
    if 'error' in subject_validation:
        return {
            'error': subject_validation['error'],
            'anhedonic_subjects': [],
            'non_anhedonic_subjects': [],
            'processing_mode': 'downloaded_only' if infrastructure['use_downloaded_only'] else 'all_available',
        }

    accessible_anhedonic = subject_validation['accessible_anhedonic']
    accessible_non_anhedonic = subject_validation['accessible_non_anhedonic']

    # 4. Apply subject limiting if configured
    print(f"\n{'='*80}")
    print(f"STARTING MULTI-SUBJECT ANALYSIS")
    print(f"{'='*80}")

    if config['limit_subjects']:
        anhedonic_subjects_to_process = accessible_anhedonic[:config['max_subjects_per_group']]
        non_anhedonic_subjects_to_process = accessible_non_anhedonic[:config['max_subjects_per_group']]
        print(f"LIMITING ENABLED: Processing {len(anhedonic_subjects_to_process)} anhedonic + {len(non_anhedonic_subjects_to_process)} non-anhedonic subjects")
    else:
        anhedonic_subjects_to_process = accessible_anhedonic
        non_anhedonic_subjects_to_process = accessible_non_anhedonic
        print(f"FULL ANALYSIS: Processing {len(anhedonic_subjects_to_process)} anhedonic + {len(non_anhedonic_subjects_to_process)} non-anhedonic subjects")

    # 5. Initialize processing context
    context = initialize_processing_context(
        manager, loader, mask_nonsignificant, create_plots, show_plots,
        save_figures, config['verbose_subject_output']
    )

    # 6. Process subject groups
    anhedonic_results = process_subject_group(
        anhedonic_subjects_to_process, "anhedonic", context
    )
    non_anhedonic_results = process_subject_group(
        non_anhedonic_subjects_to_process, "non-anhedonic", context
    )

    # 7. Calculate summary statistics
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

    # 8. Collect FC results and perform group comparison
    anhedonic_fc_results = collect_fc_results(anhedonic_results)
    non_anhedonic_fc_results = collect_fc_results(non_anhedonic_results)

    print(f"FC analysis available:")
    print(f"  Anhedonic: {len(anhedonic_fc_results)} subjects")
    print(f"  Non-anhedonic: {len(non_anhedonic_fc_results)} subjects")

    group_comparison_results = perform_group_comparison(
        anhedonic_fc_results, non_anhedonic_fc_results
    )

    # 9. Export FC results to CSV
    all_results = {**anhedonic_results, **non_anhedonic_results}
    csv_export_count = export_all_fc_results(all_results, total_success)

    # 10. Create individual plots
    individual_plots_created, figures_saved_count = create_individual_plots(
        all_results, total_success, config['show_individual_plots'],
        create_plots, save_figures, mask_nonsignificant
    )

    # 11. Return comprehensive results
    return {
        'anhedonic_subjects': anhedonic_subjects_to_process,
        'non_anhedonic_subjects': non_anhedonic_subjects_to_process,
        'processing_mode': 'downloaded_only' if infrastructure['use_downloaded_only'] else 'all_available',
        'configuration': config,
        'summary': {
            'total_processed': total_processed,
            'total_successful': total_success,
            'anhedonic_processed': len(anhedonic_results),
            'anhedonic_successful': anhedonic_success,
            'non_anhedonic_processed': len(non_anhedonic_results),
            'non_anhedonic_successful': non_anhedonic_success,
            'individual_plots_created': individual_plots_created,
            'figures_saved': figures_saved_count,
            'csv_exports': csv_export_count
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
    CREATE_PLOTS = True  # Whether to create plots (required for both displaying and saving)
    SHOW_PLOTS = False  # Whether to display plots interactively (requires CREATE_PLOTS=True)
    SAVE_FIGURES = False  # Whether to save figures to disk as SVG files (requires CREATE_PLOTS=True)

    # FC Matrix display mode:
    # - False: Show all correlations, mark non-significant with asterisks
    # - True: Hide non-significant correlations (masked)
    MASK_NONSIGNIFICANT = True

    main(mask_nonsignificant=MASK_NONSIGNIFICANT, create_plots=CREATE_PLOTS, show_plots=SHOW_PLOTS, save_figures=SAVE_FIGURES)

    if CREATE_PLOTS and SHOW_PLOTS:
        plt.show()
