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
from scipy import stats

from config.paths import get_analysis_path
from tcp.processing import DataLoader, SubjectManager
from tcp.processing.roi import (
    CorticalAtlasLookup,
    ROIExtractionService,
    SubCorticalAtlasLookup,
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


def export_fc_results_to_csv(fc_results, subject_id, output_dir):
    """
    Export STATIC functional connectivity results to CSV files for later analysis.

    Creates three CSV files per subject:
    1. {subject_id}_fc_matrix.csv - Full correlation matrix
    2. {subject_id}_fc_pvalues.csv - P-values matrix
    3. {subject_id}_fc_pairwise.csv - Pairwise connections with metadata

    NOTE: This exports STATIC FC (whole-session correlations).
    Dynamic FC (time-windowed) is not yet implemented.

    Args:
        fc_results: FC results dictionary from process_subject
        subject_id: Subject identifier
        output_dir: Directory to save CSV files (Path object or string)

    Returns:
        dict: Paths to created CSV files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not fc_results or 'fc_matrix' not in fc_results:
        print(f"[WARNING] No FC results to export for {subject_id}")
        return None

    fc_matrix = fc_results['fc_matrix']
    fc_labels = fc_results['fc_labels']
    fc_pvalues = fc_results.get('fc_pvalues')
    connectivity_patterns = fc_results.get('connectivity_patterns', {})
    channel_label_map = fc_results.get('channel_label_map', {})

    created_files = {}

    # 1. Export FC correlation matrix
    matrix_file = output_dir / f'{subject_id}_fc_matrix.csv'
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
        pvalues_file = output_dir / f'{subject_id}_fc_pvalues.csv'
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
    pairwise_file = output_dir / f'{subject_id}_fc_pairwise.csv'
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


def plot_averaged_signals(mean_signals, subject_id=None):
    """
    Create visualization for averaged ROI signals (vmPFC and AMY).

    Args:
        mean_signals: Dictionary with keys 'vmPFC_right', 'vmPFC_left', 'amy_right', 'amy_left'
        subject_id: Optional subject identifier for title

    Returns:
        matplotlib.figure.Figure: Figure with averaged signal plots
    """
    vmPFC_right = mean_signals.get('vmPFC_right')
    vmPFC_left = mean_signals.get('vmPFC_left')
    amy_right = mean_signals.get('amy_right')
    amy_left = mean_signals.get('amy_left')

    # Check if we have any data
    available_signals = [s for s in [vmPFC_right, vmPFC_left, amy_right, amy_left] if s is not None]
    if not available_signals:
        fig = plt.figure(figsize=(15, 8))
        return fig

    # Create 2x2 subplot layout with better spacing
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)

    title_suffix = f' ({subject_id})' if subject_id else ''

    # Plot vmPFC signals
    if vmPFC_right is not None:
        axes[0, 0].plot(vmPFC_right, color='red', linewidth=2, label='vmPFC Right')
        axes[0, 0].set_ylabel('Signal', fontsize=12)
        axes[0, 0].set_title(f'vmPFC Right Hemisphere{title_suffix}', fontsize=13, fontweight='bold')
        axes[0, 0].legend(loc='upper right')
        axes[0, 0].grid(True, alpha=0.3)

    if vmPFC_left is not None:
        axes[0, 1].plot(vmPFC_left, color='blue', linewidth=2, label='vmPFC Left')
        axes[0, 1].set_ylabel('Signal', fontsize=12)
        axes[0, 1].set_title(f'vmPFC Left Hemisphere{title_suffix}', fontsize=13, fontweight='bold')
        axes[0, 1].legend(loc='upper right')
        axes[0, 1].grid(True, alpha=0.3)

    # Plot AMY signals
    if amy_right is not None:
        axes[1, 0].plot(amy_right, color='red', linewidth=2, label='AMY Right')
        axes[1, 0].set_ylabel('Signal', fontsize=12)
        axes[1, 0].set_xlabel('Time (volumes)', fontsize=12)
        axes[1, 0].set_title(f'Amygdala Right Hemisphere{title_suffix}', fontsize=13, fontweight='bold')
        axes[1, 0].legend(loc='upper right')
        axes[1, 0].grid(True, alpha=0.3)

    if amy_left is not None:
        axes[1, 1].plot(amy_left, color='blue', linewidth=2, label='AMY Left')
        axes[1, 1].set_ylabel('Signal', fontsize=12)
        axes[1, 1].set_xlabel('Time (volumes)', fontsize=12)
        axes[1, 1].set_title(f'Amygdala Left Hemisphere{title_suffix}', fontsize=13, fontweight='bold')
        axes[1, 1].legend(loc='upper right')
        axes[1, 1].grid(True, alpha=0.3)

    # Use tight_layout with padding to prevent title overlap
    plt.tight_layout(pad=2.0, h_pad=1.5)
    return fig




def plot_fc_results_clean(corr_matrix, roi_labels, p_values=None, connectivity_patterns=None, channel_label_map=None, alpha=0.05, mask_nonsignificant=False):
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


def plot_fc_results(corr_matrix, roi_labels, p_values=None, connectivity_patterns=None, alpha=0.05):
    """
    Create visualizations for functional connectivity results.

    Args:
        corr_matrix: Correlation matrix
        roi_labels: ROI labels
        p_values: Optional p-values matrix
        connectivity_patterns: Optional results from analyze_connectivity_patterns
        alpha: Significance threshold for marking significant correlations
    """
    fig = plt.figure(figsize=(15, 10))

    # 1. Correlation matrix heatmap
    ax1 = plt.subplot(2, 3, (1, 4))

    # Create significance mask if p-values available
    if p_values is not None:
        mask = p_values >= alpha
    else:
        mask = None

    sns.heatmap(corr_matrix,
                annot=True,
                fmt='.3f',
                xticklabels=roi_labels,
                yticklabels=roi_labels,
                center=0,
                cmap='RdBu_r',
                vmin=-1, vmax=1,
                mask=mask,
                ax=ax1)
    ax1.set_title('Functional Connectivity Matrix\n(Pearson Correlations)')

    if connectivity_patterns is not None:
        # 2. Interhemispheric connectivity
        ax2 = plt.subplot(2, 3, 2)
        inter_corrs = [v['correlation'] for v in connectivity_patterns['interhemispheric'].values()]
        inter_labels = [k.replace('_', '\n') for k in connectivity_patterns['interhemispheric'].keys()]

        if inter_corrs:
            bars = ax2.bar(range(len(inter_corrs)), inter_corrs, color='lightblue', edgecolor='navy')
            ax2.set_xticks(range(len(inter_labels)))
            ax2.set_xticklabels(inter_labels, rotation=45, ha='right')
            ax2.set_ylabel('Correlation')
            ax2.set_title('Interhemispheric\nConnectivity')
            ax2.set_ylim(-1, 1)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

            # Mark significant correlations
            if p_values is not None:
                for i, (k, v) in enumerate(connectivity_patterns['interhemispheric'].items()):
                    if v['significant']:
                        bars[i].set_color('red')
                        bars[i].set_alpha(0.8)
        else:
            ax2.text(0.5, 0.5, 'No interhemispheric\nconnections found',
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Interhemispheric\nConnectivity')

        # 3. Cross-regional connectivity
        ax3 = plt.subplot(2, 3, 3)
        cross_corrs = [v['correlation'] for v in connectivity_patterns['cross_regional'].values()]
        cross_labels = [k.replace('_', '\n') for k in connectivity_patterns['cross_regional'].keys()]

        if cross_corrs:
            bars = ax3.bar(range(len(cross_corrs)), cross_corrs, color='lightgreen', edgecolor='darkgreen')
            ax3.set_xticks(range(len(cross_labels)))
            ax3.set_xticklabels(cross_labels, rotation=45, ha='right')
            ax3.set_ylabel('Correlation')
            ax3.set_title('Cross-Regional\nConnectivity')
            ax3.set_ylim(-1, 1)
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)

            # Mark significant correlations
            if p_values is not None:
                for i, (k, v) in enumerate(connectivity_patterns['cross_regional'].items()):
                    if v['significant']:
                        bars[i].set_color('red')
                        bars[i].set_alpha(0.8)
        else:
            ax3.text(0.5, 0.5, 'No cross-regional\nconnections found',
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Cross-Regional\nConnectivity')

        # 4. Ipsilateral vs Contralateral
        ax4 = plt.subplot(2, 3, 5)
        ipsi_corrs = [v['correlation'] for v in connectivity_patterns['ipsilateral'].values()]
        contra_corrs = [v['correlation'] for v in connectivity_patterns['contralateral'].values()]

        if ipsi_corrs or contra_corrs:
            width = 0.35
            ipsi_mean = np.mean(ipsi_corrs) if ipsi_corrs else 0
            contra_mean = np.mean(contra_corrs) if contra_corrs else 0

            bars = ax4.bar(['Ipsilateral', 'Contralateral'],
                          [ipsi_mean, contra_mean],
                          color=['orange', 'purple'],
                          alpha=0.7)
            ax4.set_ylabel('Mean Correlation')
            ax4.set_title('Ipsilateral vs\nContralateral Connectivity')
            ax4.set_ylim(-1, 1)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)

            # Add error bars if multiple connections
            if len(ipsi_corrs) > 1:
                ax4.errorbar([0], [ipsi_mean], yerr=[np.std(ipsi_corrs)],
                           color='black', capsize=5)
            if len(contra_corrs) > 1:
                ax4.errorbar([1], [contra_mean], yerr=[np.std(contra_corrs)],
                           color='black', capsize=5)
        else:
            ax4.text(0.5, 0.5, 'No ipsilateral/contralateral\nconnections found',
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Ipsilateral vs\nContralateral Connectivity')

        # 5. Summary statistics
        ax5 = plt.subplot(2, 3, 6)
        ax5.axis('off')

        summary_text = "FC Summary Statistics:\n\n"
        summary_text += f"Total connections: {len(connectivity_patterns['all_pairwise'])}\n"
        summary_text += f"Interhemispheric: {len(connectivity_patterns['interhemispheric'])}\n"
        summary_text += f"Cross-regional: {len(connectivity_patterns['cross_regional'])}\n"
        summary_text += f"Ipsilateral: {len(connectivity_patterns['ipsilateral'])}\n"
        summary_text += f"Contralateral: {len(connectivity_patterns['contralateral'])}\n\n"

        if p_values is not None:
            sig_count = sum(1 for v in connectivity_patterns['all_pairwise'].values()
                           if v['significant'])
            summary_text += f"Significant (p<{alpha}): {sig_count}\n"

        # Strongest connections
        all_pairs = connectivity_patterns['all_pairwise']
        if all_pairs:
            strongest = max(all_pairs.items(), key=lambda x: abs(x[1]['correlation']))
            summary_text += f"\nStrongest connection:\n{strongest[0]}: r={strongest[1]['correlation']:.3f}"

        ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes,
                verticalalignment='top', fontsize=10, family='monospace')

    # Use tight_layout with padding to prevent title overlap
    plt.tight_layout(pad=2.0, rect=[0, 0, 1, 0.96])
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
            if fc_result and 'connectivity_patterns' in fc_result:
                patterns = fc_result['connectivity_patterns']
                for pattern_type in group_patterns.keys():
                    if pattern_type in patterns:
                        correlations = [v['correlation'] for v in patterns[pattern_type].values()]
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


def create_research_summary(fc_results, subject_info=None):
    """
    Create a structured summary for research reporting.

    Args:
        fc_results: FC analysis results
        subject_info: Optional subject information

    Returns:
        dict: Structured research summary
    """
    if not fc_results:
        return {'error': 'No FC results available'}

    summary = {
        'sample_info': subject_info if subject_info else {},
        'methodology': {
            'roi_extraction': 'Hemisphere-specific extraction using atlas-based parcellation',
            'fc_computation': 'Pearson correlation between ROI mean timeseries',
            'significance_testing': 'Two-tailed correlation with p<0.05 threshold'
        },
        'results': {}
    }

    if 'connectivity_patterns' in fc_results:
        patterns = fc_results['connectivity_patterns']

        # Key findings
        summary['results']['key_findings'] = {
            'total_connections_tested': len(patterns.get('all_pairwise', {})),
            'significant_connections': sum(1 for v in patterns.get('all_pairwise', {}).values()
                                         if v.get('significant', False)),
            'strongest_connection': None,
            'interhemispheric_connectivity': {},
            'cross_regional_connectivity': {}
        }

        # Find strongest connection
        all_pairs = patterns.get('all_pairwise', {})
        if all_pairs:
            strongest = max(all_pairs.items(), key=lambda x: abs(x[1]['correlation']))
            summary['results']['key_findings']['strongest_connection'] = {
                'pair': strongest[0],
                'correlation': strongest[1]['correlation'],
                'p_value': strongest[1]['p_value'],
                'significant': strongest[1]['significant']
            }

        # Interhemispheric summary
        inter_connections = patterns.get('interhemispheric', {})
        if inter_connections:
            summary['results']['key_findings']['interhemispheric_connectivity'] = {
                'vmPFC_hemispheric_correlation': inter_connections.get('vmPFC_RH_vmPFC_LH', {}).get('correlation'),
                'AMY_hemispheric_correlation': inter_connections.get('AMY_rh_AMY_lh', {}).get('correlation'),
                'significant_interhemispheric': sum(1 for v in inter_connections.values() if v.get('significant', False))
            }

        # Cross-regional summary
        cross_connections = patterns.get('cross_regional', {})
        if cross_connections:
            summary['results']['key_findings']['cross_regional_connectivity'] = {
                'vmPFC_AMY_connections': {pair: stats_dict['correlation']
                                        for pair, stats_dict in cross_connections.items()},
                'significant_cross_regional': sum(1 for v in cross_connections.values() if v.get('significant', False))
            }

        # Statistical summary
        all_correlations = [v['correlation'] for v in all_pairs.values()]
        if all_correlations:
            summary['results']['statistical_summary'] = {
                'mean_correlation   ': np.mean(all_correlations),
                'std_correlation': np.std(all_correlations),
                'range': [np.min(all_correlations), np.max(all_correlations)],
                'median_correlation': np.median(all_correlations)
            }

    return summary


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
            # For subcortical, get actual parcel names from atlas (e.g., lAMY-rh, mAMY-rh)
            for roi_name in subcortical_valid_rois:
                if roi_name not in subcortical_parcel_labels:
                    subcortical_parcel_labels[roi_name] = {'rh': [], 'lh': []}

                # Right hemisphere labels - get actual subdivision names from atlas
                if roi_name in subcortical_right_timeseries:
                    # Get parcel indices for this ROI in right hemisphere
                    rh_indices_dict = subcortical_atlas.get_roi_indices_by_hemisphere([roi_name], hemisphere='rh')
                    rh_indices = rh_indices_dict.get(roi_name, [])
                    rh_parcel_names = []

                    for idx in rh_indices:
                        # Get the actual parcel name from the atlas (e.g., 'lAMY-rh', 'mAMY-rh')
                        parcel_name = subcortical_atlas.get_parcel_name(idx)
                        if parcel_name:
                            rh_parcel_names.append(parcel_name)
                        else:
                            # Fallback if parcel name not available
                            rh_parcel_names.append(f'{roi_name}_rh_parcel{len(rh_parcel_names)+1}')

                    subcortical_parcel_labels[roi_name]['rh'] = rh_parcel_names

                # Left hemisphere labels
                if roi_name in subcortical_left_timeseries:
                    lh_indices_dict = subcortical_atlas.get_roi_indices_by_hemisphere([roi_name], hemisphere='lh')
                    lh_indices = lh_indices_dict.get(roi_name, [])
                    lh_parcel_names = []

                    for idx in lh_indices:
                        parcel_name = subcortical_atlas.get_parcel_name(idx)
                        if parcel_name:
                            lh_parcel_names.append(parcel_name)
                        else:
                            lh_parcel_names.append(f'{roi_name}_lh_parcel{len(lh_parcel_names)+1}')

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

        # Functional connectivity analysis
        fc_results = None
        is_missing_timeseries = any([v is None for v in [cortical_right_timeseries, cortical_left_timeseries,
                                                     subcortical_right_timeseries, subcortical_left_timeseries,
                                                     vmPFC_right_channels, vmPFC_left_channels,
                                                     amy_right_channels, amy_left_channels,
                                                     channel_label_map]])

        if not is_missing_timeseries:
            if verbose:
                print(f"\n=== FUNCTIONAL CONNECTIVITY ANALYSIS ===")

            # Create FC timeseries dictionary using actual parcel labels
            fc_timeseries = {}
            all_channel_labels = cortical_channel_labels + subcortical_channel_labels

            # Combine all channel timeseries in order
            all_channels = np.vstack([
                vmPFC_right_channels,
                vmPFC_left_channels,
                amy_right_channels,
                amy_left_channels
            ])

            # Map each channel timeseries to its actual parcel label
            for i, channel_label in enumerate(all_channel_labels):
                fc_timeseries[channel_label] = all_channels[i]

            fc_matrix, fc_labels, fc_pvalues = compute_fc_matrix(fc_timeseries)

            if fc_matrix is not None:
                if verbose:
                    print(f"FC Matrix shape: {fc_matrix.shape}")
                    print(f"ROI labels (alphabetical): {sorted(fc_labels)}")

                connectivity_patterns = analyze_connectivity_patterns(fc_matrix, fc_labels, fc_pvalues)
                print(f"Interhemispheric connections (ALL): {connectivity_patterns['interhemispheric'].keys()}")

                if verbose:
                    print(f"\nConnectivity Pattern Analysis:")
                    print(f"  Total pairwise connections: {len(connectivity_patterns['all_pairwise'])}")
                    print(f"  Interhemispheric connections: {len(connectivity_patterns['interhemispheric'])}")
                    print(f"  Cross-regional connections: {len(connectivity_patterns['cross_regional'])}")

                fc_results = {
                    'fc_matrix': fc_matrix,
                    'fc_labels': fc_labels,
                    'fc_pvalues': fc_pvalues,
                    'connectivity_patterns': connectivity_patterns,
                    'timeseries_used': fc_timeseries,
                    'channel_label_map': channel_label_map
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
            'functional_connectivity': fc_results,
            'channel_signals': {
                'vmPFC_right_channels': vmPFC_right_channels,
                'vmPFC_left_channels': vmPFC_left_channels,
                'amy_right_channels': amy_right_channels,
                'amy_left_channels': amy_left_channels,
                'channel_label_map': channel_label_map
            }
        }

    except Exception as e:
        if verbose:
            print(f"[ERROR] Failed to process subject {subject_id}: {str(e)}")
        return {
            'subject_id': subject_id,
            'error': str(e),
            'success': False
        }


def main(mask_nonsignificant=False, create_plots=True, show_plots=True, save_figures=False):
    """Main function for FC MVP analysis"""
    print("=== Functional Connectivity MVP ===")

    # ===== CONFIGURATION FOR MULTI-SUBJECT ANALYSIS =====
    LIMIT_SUBJECTS = True  # Set False for full analysis
    MAX_SUBJECTS_PER_GROUP = 2  # Limit when testing (only used if LIMIT_SUBJECTS=True)
    SHOW_INDIVIDUAL_PLOTS = True  # Show individual subject plots
    VERBOSE_SUBJECT_OUTPUT = True  # Detailed per-subject printing
    SHOW_GROUP_SUMMARY = True  # Show aggregated group analysis

    print(f"Configuration:")
    print(f"  Subject limiting: {'ENABLED' if LIMIT_SUBJECTS else 'DISABLED'}")
    if LIMIT_SUBJECTS:
        print(f"  Max subjects per group: {MAX_SUBJECTS_PER_GROUP}")
    print(f"  Individual plots: {'ENABLED' if SHOW_INDIVIDUAL_PLOTS else 'DISABLED'}")
    print(f"  Verbose output: {'ENABLED' if VERBOSE_SUBJECT_OUTPUT else 'DISABLED'}")
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
            verbose=VERBOSE_SUBJECT_OUTPUT
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
            verbose=VERBOSE_SUBJECT_OUTPUT
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
        if result['success'] and result.get('functional_connectivity'):
            anhedonic_fc_results.append(result['functional_connectivity'])

    for subject_id, result in non_anhedonic_results.items():
        if result['success'] and result.get('functional_connectivity'):
            non_anhedonic_fc_results.append(result['functional_connectivity'])

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
    print(f"EXPORTING FC RESULTS TO CSV (STATIC FC)")
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

        fc_data = result.get('functional_connectivity')
        if fc_data:
            print(f"\nExporting FC results for {subject_id}...")
            exported_files = export_fc_results_to_csv(fc_data, subject_id, fc_output_dir)
            if exported_files:
                csv_export_count += 1

    print(f"\n✓ Exported FC results for {csv_export_count}/{total_success} subjects")
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

    if not create_plots:
        print(f"\n[INFO] No individual plots created (create_plots set to False)")
    elif SHOW_INDIVIDUAL_PLOTS and total_success <= 3:  # Only show individual plots for ≤3 subjects
        print(f"\n{'='*80}")
        print(f"CREATING INDIVIDUAL SUBJECT PLOTS")
        print(f"{'='*80}")

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

            # 4. Plot functional connectivity analysis results (clean version)
            if result.get('functional_connectivity'):
                print(f"  Creating clean FC analysis plot for {subject_id}...")
                fc_data = result['functional_connectivity']

                if fc_data.get('fc_matrix') is not None:
                    fc_fig = plot_fc_results_clean(
                        fc_data['fc_matrix'],
                        fc_data['fc_labels'],
                        fc_data['fc_pvalues'],
                        fc_data['connectivity_patterns'],
                        fc_data.get('channel_label_map'),
                        mask_nonsignificant=mask_nonsignificant
                    )
                    fc_fig.suptitle(f'Functional Connectivity Analysis - {subject_id}', fontsize=16, fontweight='bold')
                    plots_for_subject += 1

                    # Save figure if enabled
                    if save_figures and figures_output_dir:
                        fig_path = figures_output_dir / f'{subject_id}_fc_analysis.svg'
                        fc_fig.savefig(fig_path, format='svg', bbox_inches='tight', dpi=300)
                        figures_saved_count += 1

            if plots_for_subject > 0:
                individual_plots_created += 1
                print(f"  ✓ Created {plots_for_subject} plots for {subject_id}")

        print(f"\nCreated plots for {individual_plots_created} subjects")

        # Summary of saved figures
        if save_figures and figures_saved_count > 0:
            print(f"✓ Saved {figures_saved_count} figures to: {figures_output_dir}")
    elif total_success > 3:
        print(f"\n[INFO] Skipping individual plots (too many subjects: {total_success})")
        print(f"      Individual plots only shown for ≤3 subjects")
    else:
        print(f"\n[INFO] No individual plots created (no successful subjects)")

    # ===== RETURN MULTI-SUBJECT RESULTS =====
    return {
        'anhedonic_subjects': anhedonic_subjects_to_process,
        'non_anhedonic_subjects': non_anhedonic_subjects_to_process,
        'processing_mode': 'downloaded_only' if use_downloaded_only else 'all_available',
        'configuration': {
            'limit_subjects': LIMIT_SUBJECTS,
            'max_subjects_per_group': MAX_SUBJECTS_PER_GROUP if LIMIT_SUBJECTS else None,
            'show_individual_plots': SHOW_INDIVIDUAL_PLOTS,
            'verbose_subject_output': VERBOSE_SUBJECT_OUTPUT
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
