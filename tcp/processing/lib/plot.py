import csv
import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams['figure.max_open_warning'] = 0  # Disable matplotlib warning for >20 open figures

# Global flag to avoid repeating the open-figure warning
_open_fig_warning_shown = False
logger = logging.getLogger(__name__)

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
    # Guard against empty or invalid matrices to avoid seaborn y-lim warnings
    matrix_shape = getattr(corr_matrix, 'shape', ())
    invalid_matrix = (
        corr_matrix is None
        or not hasattr(corr_matrix, 'size')
        or corr_matrix.size == 0
        or (len(matrix_shape) == 2 and min(matrix_shape) == 0)
    )

    if invalid_matrix:
        reason = f"empty correlation matrix (shape={matrix_shape})"
        msg = f"[plot_fc_results] Skipping FC plot for subject_id={subject_id or 'unknown'}: {reason}"
        if verbose:
            print(msg)
        logger.warning(msg)
        return None, None

    # Ensure labels are present and aligned with the matrix
    if roi_labels is None:
        roi_labels = []

    if len(roi_labels) == 0:
        roi_labels = [f'ROI{i+1}' for i in range(corr_matrix.shape[0])]
        msg = f"[plot_fc_results] Missing ROI labels; generated {len(roi_labels)} placeholder labels for subject_id={subject_id or 'unknown'}"
        if verbose:
            print(msg)
        logger.warning(msg)
    elif len(roi_labels) != corr_matrix.shape[0]:
        msg = (f"[plot_fc_results] ROI label count ({len(roi_labels)}) does not match matrix dimension "
               f"({corr_matrix.shape[0]}). Adjusting labels for subject_id={subject_id or 'unknown'}.")
        if verbose:
            print(msg)
        logger.warning(msg)
        # Trim or pad labels to match matrix size
        if len(roi_labels) > corr_matrix.shape[0]:
            roi_labels = roi_labels[:corr_matrix.shape[0]]
        else:
            roi_labels = roi_labels + [f'ROI{i+1+len(roi_labels)}' for i in range(corr_matrix.shape[0] - len(roi_labels))]

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
    Only plots the banded signals (Slow-1 through Slow-6), excluding frequencies outside all bands.

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

    # Define band order for plotting: Slow-6 through Slow-1 (exclude 'excluded')
    band_order = ['6', '5', '4', '3', '2', '1']
    band_names = {
        '6': 'Slow-6 (0-0.01 Hz)',
        '5': 'Slow-5 (0.01-0.027 Hz)',
        '4': 'Slow-4 (0.027-0.073 Hz)',
        '3': 'Slow-3 (0.073-0.198 Hz)',
        '2': 'Slow-2 (0.198-0.5 Hz)',
        '1': 'Slow-1 (0.5-0.75 Hz)',
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
        for band_key in ['6', '5', '4', '3', '2', '1']:
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


def plot_multivariate_hilbert_spectrum(hsa_data, subject_id=None, center_freqs=None, channel_labels=None):
    """
    Plot Hilbert Spectrum for each mode with channels kept completely separate.

    CRITICAL: Channels are NEVER aggregated. Each channel's spectrum is plotted
    separately, grouped by region+network.

    Args:
        hsa_data: Dictionary from compute_hilbert_transform_per_mode()
        subject_id: Optional subject identifier
        center_freqs: Optional array of MVMD center frequencies for each mode
        channel_labels: List of channel labels (e.g., ['PFCm_RH_DefaultA_p1', ...])

    Returns:
        List of tuples: Each tuple contains (figure, region_network_key) for one region+network group
    """
    modes_data = hsa_data['modes_data']
    sampling_rate = hsa_data['sampling_rate']
    n_samples = hsa_data['n_samples']
    n_modes = hsa_data['n_modes']

    # Get number of channels from first mode
    n_channels = modes_data[0]['instantaneous_frequency'].shape[0]

    # Define frequency bins (0 to Nyquist)
    nyquist_freq = sampling_rate / 2
    n_freq_bins = 256  # Resolution for frequency axis
    freq_bins = np.linspace(0, nyquist_freq, n_freq_bins + 1)
    freq_bin_centers = (freq_bins[:-1] + freq_bins[1:]) / 2

    # Create volume indices (discrete samples)
    volumes = np.arange(n_samples)

    # Helper function to parse channel labels and group by region+network
    def parse_channel_label(label):
        """Extract region_network_key and display label with hemisphere from channel label."""
        parts = label.split('_')
        if len(parts) >= 4 and parts[3].startswith('p'):
            # Cortical: region_hemi_network_pSubarea (e.g., PFCm_RH_DefaultA_p1)
            region_network_key = f"{parts[0]}_{parts[2]}"  # e.g., "PFCm_DefaultA"
            display_label = f"{parts[1]} {parts[3]}"  # e.g., "RH p1"
            return region_network_key, display_label, label
        elif len(parts) >= 3:
            # Subcortical: region_hemi_subdivision (e.g., AMY_RH_lAMY)
            region_network_key = f"{parts[0]}_{parts[2]}"  # e.g., "AMY_lAMY"
            display_label = f"{parts[1]} {parts[2]}"  # e.g., "RH lAMY"
            return region_network_key, display_label, label
        else:
            return 'Unknown', label, label

    # Group channels by region+network
    if channel_labels is None:
        # Fallback: create generic labels
        channel_labels = [f'Ch{i+1}' for i in range(n_channels)]

    region_network_groups = {}
    for ch_idx, label in enumerate(channel_labels):
        region_network_key, display_label, full_label = parse_channel_label(label)
        if region_network_key not in region_network_groups:
            region_network_groups[region_network_key] = []
        region_network_groups[region_network_key].append((ch_idx, display_label, full_label))

    # Create one figure per region+network group
    figures = []

    for region_network_key, channels in region_network_groups.items():
        n_channels_in_group = len(channels)

        # Create figure: 1 row, columns = channels (all modes superimposed per channel)
        fig, axes = plt.subplots(1, n_channels_in_group, figsize=(4 * n_channels_in_group, 5),
                                 sharex=True, sharey=True)

        # Handle single channel case
        if n_channels_in_group == 1:
            axes = [axes]

        # Process each channel: superimpose ALL modes
        for col_idx, (ch_idx, display_label, full_label) in enumerate(channels):
            ax = axes[col_idx]

            # Initialize Hilbert Spectrum for THIS CHANNEL (summing across ALL modes)
            hilbert_spectrum_superimposed = np.zeros((n_freq_bins, n_samples))

            # Superimpose all K modes for this channel
            for mode_idx, mode_data in enumerate(modes_data):
                inst_freq = mode_data['instantaneous_frequency']  # (channels, samples)
                inst_amp = mode_data['instantaneous_amplitude']    # (channels, samples)

                # Process this channel's contribution from this mode
                for t_idx in range(n_samples):
                    freq_val = inst_freq[ch_idx, t_idx]
                    amp_val = inst_amp[ch_idx, t_idx]

                    # Only bin if frequency is within valid range
                    if 0 <= freq_val < nyquist_freq:
                        freq_bin_idx = np.searchsorted(freq_bins, freq_val) - 1
                        freq_bin_idx = np.clip(freq_bin_idx, 0, n_freq_bins - 1)

                        # Superimpose energy from this mode
                        hilbert_spectrum_superimposed[freq_bin_idx, t_idx] += amp_val ** 2

            # Use log scale for better visualization
            hs_log = np.log10(hilbert_spectrum_superimposed + 1e-10)

            # Plot as heatmap with hot colormap
            extent = [volumes[0], volumes[-1], freq_bin_centers[0], freq_bin_centers[-1]]
            im = ax.imshow(hs_log, aspect='auto', origin='lower', cmap='hot',
                           extent=extent, interpolation='bilinear')

            # Labels
            ax.set_title(display_label, fontsize=10, fontweight='bold')
            ax.set_xlabel('Volumes', fontsize=9)
            if col_idx == 0:
                ax.set_ylabel('Frequency (Hz)', fontsize=9, fontweight='bold')

            ax.tick_params(labelsize=8)

        # Add single colorbar for entire figure positioned to the right of all subplots
        # Create space for colorbar by adjusting figure layout
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        fig.colorbar(im, cax=cbar_ax, label='$\log_{10}$(Energy)')

        # Figure title
        title_parts = [f'Hilbert Spectrum ({n_modes} Modes Superimposed): {region_network_key}']
        if subject_id:
            title_parts.append(f'({subject_id})')

        fig.suptitle(' - '.join(title_parts), fontsize=14, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0, 0.92, 0.96])

        figures.append((fig, region_network_key))

    return figures
def plot_marginal_spectrum_per_mode(hsa_data, subject_id=None, center_freqs=None, channel_labels=None):
    """
    Plot Marginal Hilbert Spectrum with separate cell for each mode and channel.

    The marginal spectrum is computed by integrating the Hilbert Spectrum over time only,
    showing the frequency content for each mode and channel combination in a grid layout.

    Args:
        hsa_data: Dictionary from compute_hilbert_transform_per_mode()
        subject_id: Optional subject identifier
        center_freqs: Optional array of MVMD center frequencies for each mode
        channel_labels: List of channel labels (e.g., ['PFCm_RH_DefaultA_p1', ...])

    Returns:
        List of tuples: Each tuple contains (figure, region_network_key) for one region+network group
    """
    modes_data = hsa_data['modes_data']
    sampling_rate = hsa_data['sampling_rate']
    n_modes = hsa_data['n_modes']
    n_samples = hsa_data['n_samples']

    # Get number of channels
    n_channels = modes_data[0]['instantaneous_frequency'].shape[0]

    # Define frequency bins
    nyquist_freq = sampling_rate / 2
    n_freq_bins = 256
    freq_bins = np.linspace(0, nyquist_freq, n_freq_bins + 1)
    freq_bin_centers = (freq_bins[:-1] + freq_bins[1:]) / 2

    # Helper function to parse channel labels and group by region+network
    def parse_channel_label(label):
        """Extract region_network_key and display label from channel label."""
        parts = label.split('_')
        if len(parts) >= 4 and parts[3].startswith('p'):
            # Cortical: region_hemi_network_pSubarea (e.g., PFCm_RH_DefaultA_p1)
            region_network_key = f"{parts[0]}_{parts[2]}"
            display_label = f"{parts[1]} {parts[3]}"
            return region_network_key, display_label, label
        elif len(parts) >= 3:
            # Subcortical: region_hemi_subdivision (e.g., AMY_RH_lAMY)
            region_network_key = f"{parts[0]}_{parts[2]}"
            display_label = f"{parts[1]} {parts[2]}"
            return region_network_key, display_label, label
        else:
            return 'Unknown', label, label

    # Group channels by region+network
    if channel_labels is None:
        channel_labels = [f'Ch{i+1}' for i in range(n_channels)]

    region_network_groups = {}
    for ch_idx, label in enumerate(channel_labels):
        region_network_key, display_label, full_label = parse_channel_label(label)
        if region_network_key not in region_network_groups:
            region_network_groups[region_network_key] = []
        region_network_groups[region_network_key].append((ch_idx, display_label, full_label))

    # Create one figure per region+network group
    figures = []

    for region_network_key, channels in region_network_groups.items():
        n_channels_in_group = len(channels)

        # Create grid: rows = modes, columns = channels
        fig, axes = plt.subplots(n_modes, n_channels_in_group,
                                figsize=(n_channels_in_group * 3, n_modes * 2.5),
                                squeeze=False)

        # Iterate through each mode and each channel
        for mode_idx, mode_data in enumerate(modes_data):
            inst_freq = mode_data['instantaneous_frequency']  # (channels, samples)
            inst_amp = mode_data['instantaneous_amplitude']    # (channels, samples)
            mode_num = mode_data['mode_idx'] + 1  # Convert 0-based index to 1-based for display

            # Get center frequency for this mode
            center_freq = center_freqs[mode_idx] if center_freqs is not None else None

            for col_idx, (ch_idx, display_label, full_label) in enumerate(channels):
                ax = axes[mode_idx, col_idx]

                # Compute marginal spectrum for this specific mode and channel
                mhs = np.zeros(n_freq_bins)

                # Integrate over time only (NOT over channels or modes)
                for t_idx in range(n_samples):
                    freq_val = inst_freq[ch_idx, t_idx]
                    amp_val = inst_amp[ch_idx, t_idx]

                    if 0 <= freq_val < nyquist_freq:
                        freq_bin_idx = np.searchsorted(freq_bins, freq_val) - 1
                        freq_bin_idx = np.clip(freq_bin_idx, 0, n_freq_bins - 1)

                        # Accumulate energy (integrate over time only)
                        mhs[freq_bin_idx] += amp_val ** 2

                # Normalize by number of samples
                mhs /= n_samples

                # Plot marginal spectrum for this mode-channel combination
                ax.plot(freq_bin_centers, mhs, color='black', linewidth=0.8, alpha=0.9)

                # Add vertical line at center frequency for this mode
                if center_freq is not None:
                    ax.axvline(x=center_freq, color='crimson', linestyle='--',
                             linewidth=0.8, alpha=0.6)
                    y_min, y_max = ax.get_ylim()
                    y_text = y_min + 0.02 * (y_max - y_min) if y_max > y_min else y_min
                    ax.text(center_freq, y_text,
                            f'$\\omega_{{{mode_num}}}={center_freq:.3f}Hz$',
                            fontsize=7, color='crimson', ha='center', va='top')

                # Set labels
                if mode_idx == 0:
                    # Top row: add channel labels as column titles
                    ax.set_title(display_label, fontsize=9, fontweight='bold')

                if col_idx == 0:
                    # Leftmost column: add mode labels as y-axis labels
                    mode_label = f'$u_{{{mode_num}}}$'
                    if center_freq is not None:
                        mode_label += f'\n{center_freq:.3f} Hz'
                    ax.set_ylabel(mode_label, fontsize=9, fontweight='bold', rotation=0,
                                ha='right', va='center')

                if mode_idx == n_modes - 1:
                    # Bottom row: add x-axis label
                    ax.set_xlabel('Freq (Hz)', fontsize=8)
                else:
                    ax.set_xticklabels([])

                # Styling
                ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(labelsize=7)

        # Figure title
        title_parts = [f'Marginal Hilbert Spectrum per Mode and Channel: {region_network_key}']
        if subject_id:
            title_parts.append(f'({subject_id})')

        fig.suptitle(' - '.join(title_parts), fontsize=14, fontweight='bold', y=0.995)

        plt.tight_layout(rect=[0, 0, 1, 0.99])

        figures.append((fig, region_network_key))

    return figures
def plot_band_hilbert_spectrum(hsa_band_data, subject_id=None, channel_labels=None):
    """
    Plot Hilbert Spectrum for each slow-band signal with channels kept completely separate.

    CRITICAL: Channels are NEVER aggregated. Each channel's spectrum is plotted
    separately, grouped by region+network. Bands replace modes in this visualization.

    Args:
        hsa_band_data: Dictionary from compute_hilbert_transform_per_band()
        subject_id: Optional subject identifier
        channel_labels: List of channel labels (e.g., ['PFCm_RH_DefaultA_p1', ...])

    Returns:
        List of tuples: Each tuple contains (figure, region_network_key) for one region+network group
    """
    bands_data = hsa_band_data['bands_data']
    sampling_rate = hsa_band_data['sampling_rate']
    n_samples = hsa_band_data['n_samples']
    n_bands = hsa_band_data['n_bands']

    # Get number of channels from first band
    n_channels = bands_data[0]['instantaneous_frequency'].shape[0]

    # Define frequency bins (0 to Nyquist)
    nyquist_freq = sampling_rate / 2
    n_freq_bins = 256  # Resolution for frequency axis
    freq_bins = np.linspace(0, nyquist_freq, n_freq_bins + 1)
    freq_bin_centers = (freq_bins[:-1] + freq_bins[1:]) / 2

    # Create volume indices (discrete samples)
    volumes = np.arange(n_samples)

    # Helper function to parse channel labels and group by region+network
    def parse_channel_label(label):
        """Extract region_network_key and display label with hemisphere from channel label."""
        parts = label.split('_')
        if len(parts) >= 4 and parts[3].startswith('p'):
            # Cortical: region_hemi_network_pSubarea (e.g., PFCm_RH_DefaultA_p1)
            region_network_key = f"{parts[0]}_{parts[2]}"
            display_label = f"{parts[1]} {parts[3]}"
            return region_network_key, display_label, label
        elif len(parts) >= 3:
            # Subcortical: region_hemi_subdivision (e.g., AMY_RH_lAMY)
            region_network_key = f"{parts[0]}_{parts[2]}"
            display_label = f"{parts[1]} {parts[2]}"
            return region_network_key, display_label, label
        else:
            return 'Unknown', label, label

    # Group channels by region+network
    if channel_labels is None:
        channel_labels = [f'Ch{i+1}' for i in range(n_channels)]

    region_network_groups = {}
    for ch_idx, label in enumerate(channel_labels):
        region_network_key, display_label, full_label = parse_channel_label(label)
        if region_network_key not in region_network_groups:
            region_network_groups[region_network_key] = []
        region_network_groups[region_network_key].append((ch_idx, display_label, full_label))

    # Create one figure per region+network group
    figures = []

    for region_network_key, channels in region_network_groups.items():
        n_channels_in_group = len(channels)

        # Create figure: rows = bands, cols = channels in this group
        n_rows = n_bands
        n_cols = n_channels_in_group
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows),
                                 sharex=True, sharey=True)

        # Handle single band or single channel cases
        if n_bands == 1 and n_channels_in_group == 1:
            axes = np.array([[axes]])
        elif n_bands == 1:
            axes = axes.reshape(1, -1)
        elif n_channels_in_group == 1:
            axes = axes.reshape(-1, 1)

        # Process each band and channel separately (NO AGGREGATION)
        for band_idx, band_data in enumerate(bands_data):
            inst_freq = band_data['instantaneous_frequency']  # (channels, samples)
            inst_amp = band_data['instantaneous_amplitude']    # (channels, samples)
            band_name = band_data['band_name']

            for col_idx, (ch_idx, display_label, full_label) in enumerate(channels):
                ax = axes[band_idx, col_idx]

                # Initialize Hilbert Spectrum for THIS CHANNEL ONLY
                hilbert_spectrum = np.zeros((n_freq_bins, n_samples))

                # Process this single channel
                for t_idx in range(n_samples):
                    freq_val = inst_freq[ch_idx, t_idx]
                    amp_val = inst_amp[ch_idx, t_idx]

                    # Only bin if frequency is within valid range
                    if 0 <= freq_val < nyquist_freq:
                        freq_bin_idx = np.searchsorted(freq_bins, freq_val) - 1
                        freq_bin_idx = np.clip(freq_bin_idx, 0, n_freq_bins - 1)

                        # Store energy for THIS channel only (NO aggregation)
                        hilbert_spectrum[freq_bin_idx, t_idx] = amp_val ** 2

                # Use log scale for better visualization
                hs_log = np.log10(hilbert_spectrum + 1e-10)

                # Plot as heatmap with hot colormap
                extent = [volumes[0], volumes[-1], freq_bin_centers[0], freq_bin_centers[-1]]
                im = ax.imshow(hs_log, aspect='auto', origin='lower', cmap='hot',
                               extent=extent, interpolation='bilinear')

                # Labels
                if col_idx == 0:
                    ax.set_ylabel(f'{band_name}\nFreq (Hz)', fontsize=9, fontweight='bold', rotation=0, ha='right', va='center')
                if band_idx == 0:
                    ax.set_title(display_label, fontsize=9, fontweight='bold')
                if band_idx == n_bands - 1:
                    ax.set_xlabel('Volumes', fontsize=8)

                ax.tick_params(labelsize=7)

        # Add single colorbar for entire figure positioned to the right of all subplots
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        fig.colorbar(im, cax=cbar_ax, label='$\log_{10}$(Energy)')

        # Figure title
        title_parts = [f'Band Hilbert Spectrum: {region_network_key}']
        if subject_id:
            title_parts.append(f'({subject_id})')

        fig.suptitle(' - '.join(title_parts), fontsize=14, fontweight='bold', y=0.995)

        plt.tight_layout(rect=[0, 0, 0.92, 0.99])

        figures.append((fig, region_network_key))

    return figures


def plot_band_marginal_spectrum(hsa_band_data, subject_id=None, channel_labels=None):
    """
    Plot Marginal Hilbert Spectrum for each slow-band signal, integrating over time.

    Creates one subplot per band, integrating over time and all channels in each region+network.
    Bands are kept SEPARATE (not superimposed).

    Args:
        hsa_band_data: Dictionary from compute_hilbert_transform_per_band()
        subject_id: Optional subject identifier
        channel_labels: List of channel labels (e.g., ['PFCm_RH_DefaultA_p1', ...])

    Returns:
        List of tuples: Each tuple contains (figure, region_network_key) for one region+network group
    """
    bands_data = hsa_band_data['bands_data']
    sampling_rate = hsa_band_data['sampling_rate']
    n_bands = hsa_band_data['n_bands']
    n_samples = hsa_band_data['n_samples']

    # Get number of channels
    n_channels = bands_data[0]['instantaneous_frequency'].shape[0]

    # Define frequency bins
    nyquist_freq = sampling_rate / 2
    n_freq_bins = 256
    freq_bins = np.linspace(0, nyquist_freq, n_freq_bins + 1)
    freq_bin_centers = (freq_bins[:-1] + freq_bins[1:]) / 2

    # Helper function to parse channel labels and group by region+network
    def parse_channel_label(label):
        """Extract region_network_key and display label from channel label."""
        parts = label.split('_')
        if len(parts) >= 4 and parts[3].startswith('p'):
            # Cortical: region_hemi_network_pSubarea (e.g., PFCm_RH_DefaultA_p1)
            region_network_key = f"{parts[0]}_{parts[2]}"
            display_label = f"{parts[1]} {parts[3]}"
            return region_network_key, display_label, label
        elif len(parts) >= 3:
            # Subcortical: region_hemi_subdivision (e.g., AMY_RH_lAMY)
            region_network_key = f"{parts[0]}_{parts[2]}"
            display_label = f"{parts[1]} {parts[2]}"
            return region_network_key, display_label, label
        else:
            return 'Unknown', label, label

    # Group channels by region+network
    if channel_labels is None:
        channel_labels = [f'Ch{i+1}' for i in range(n_channels)]

    region_network_groups = {}
    for ch_idx, label in enumerate(channel_labels):
        region_network_key, display_label, full_label = parse_channel_label(label)
        if region_network_key not in region_network_groups:
            region_network_groups[region_network_key] = []
        region_network_groups[region_network_key].append((ch_idx, display_label, full_label))

    # Create one figure per region+network group
    figures = []

    for region_network_key, channels in region_network_groups.items():
        n_channels_in_group = len(channels)

        # Compute integrated MHS for each band (integrating over time and channels)
        mhs_per_band = []

        for band_data in bands_data:
            inst_freq = band_data['instantaneous_frequency']  # (channels, samples)
            inst_amp = band_data['instantaneous_amplitude']    # (channels, samples)
            band_name = band_data['band_name']

            # Initialize MHS for this band
            mhs_band_integrated = np.zeros(n_freq_bins)

            # Integrate over all channels in this group
            for ch_idx, display_label, full_label in channels:
                # Integrate over time for this channel
                for t_idx in range(n_samples):
                    freq_val = inst_freq[ch_idx, t_idx]
                    amp_val = inst_amp[ch_idx, t_idx]

                    if 0 <= freq_val < nyquist_freq:
                        freq_bin_idx = np.searchsorted(freq_bins, freq_val) - 1
                        freq_bin_idx = np.clip(freq_bin_idx, 0, n_freq_bins - 1)

                        # Accumulate energy (integrate over time and channels)
                        mhs_band_integrated[freq_bin_idx] += amp_val ** 2

            # Normalize by time samples
            mhs_band_integrated /= n_samples

            mhs_per_band.append((mhs_band_integrated, band_name))

        # Create subplot layout - one subplot per band
        n_cols = 2
        n_rows = int(np.ceil(n_bands / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), sharex=True)

        # Handle single row case
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for band_idx in range(n_bands):
            row = band_idx // n_cols
            col = band_idx % n_cols
            ax = axes[row, col]

            mhs_band_integrated, band_name = mhs_per_band[band_idx]

            # Plot integrated marginal spectrum for this band
            ax.plot(freq_bin_centers, mhs_band_integrated, color='black', linewidth=0.8, alpha=0.9)

            ax.set_ylabel('Energy', fontsize=10, fontweight='bold')
            ax.set_title(f'{band_name} - Marginal Spectrum\n({n_channels_in_group} Channels Integrated)',
                        fontsize=11, fontweight='bold', pad=8)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Hide unused subplots
        for band_idx in range(n_bands, n_rows * n_cols):
            row = band_idx // n_cols
            col = band_idx % n_cols
            axes[row, col].axis('off')

        # Set x-label on bottom plots
        for col in range(n_cols):
            if (n_rows - 1) * n_cols + col < n_bands:
                axes[-1, col].set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold')

        # Figure title
        title_parts = [f'Band Marginal Hilbert Spectrum: {region_network_key}']
        if subject_id:
            title_parts.append(f'({subject_id})')

        fig.suptitle(' - '.join(title_parts), fontsize=16, fontweight='bold', y=0.995)

        plt.tight_layout(pad=2.0, h_pad=2.5, rect=[0, 0, 1, 0.99])

        figures.append((fig, region_network_key))

    return figures


def plot_interhemispheric_intra_network_violin(stat_data, anova_results):
    """
    Create violin plots for interhemispheric intra-network connectivity results.

    Includes BOTH static FC (whole-signal) and slow-band FC, with Welch ANOVA statistics
    and Games-Howell post-hoc annotations.
    """
    import pandas as pd
    import seaborn as sns

    static_groups = stat_data.get('static_coherence_by_group', {})
    static_anova = anova_results.get('static_results', {})
    slow_band_groups = stat_data.get('slow_band_coherence_by_group', {})
    slow_band_anova = anova_results.get('slow_band_results', {})
    post_hoc_collection = anova_results.get('post_hoc_collection', [])

    if not static_groups and not slow_band_groups:
        return []

    # Build post-hoc lookup for easy access
    post_hoc_lookup = {}
    for ph_data in post_hoc_collection:
        network = ph_data['network']
        band = ph_data['band']
        key = (band, network)
        post_hoc_lookup[key] = ph_data['post_hoc']

    def _make_violin(df, band_name, network_key, anova_lookup):
        network_anova = anova_lookup.get(network_key, {})
        omnibus_p = network_anova.get('omnibus_p', np.nan)
        fdr_p = network_anova.get('fdr_corrected_p', np.nan)
        group_sizes = network_anova.get('group_sizes', {})
        post_hoc_df = post_hoc_lookup.get((band_name, network_key))

        global _open_fig_warning_shown
        if not _open_fig_warning_shown and len(plt.get_fignums()) >= 20:
            print("[WARN] More than 20 figures are open; consider closing figures to conserve memory.")
            _open_fig_warning_shown = True

        fig, ax = plt.subplots(figsize=(10, 6))

        group_order = ['non-anhedonic', 'low-anhedonic', 'high-anhedonic']
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        color_map = dict(zip(group_order, colors))

        sns.violinplot(
            data=df,
            x='group',
            y='fisher_z',
            order=group_order,
            hue='group',
            palette=color_map,
            dodge=False,
            inner=None,
            ax=ax,
            alpha=0.6,
            legend=False,
        )

        sns.boxplot(
            data=df, x='group', y='fisher_z', order=group_order,
            width=0.3, showfliers=False, ax=ax,
            boxprops=dict(alpha=0.7),
            whiskerprops=dict(alpha=0.7),
            capprops=dict(alpha=0.7)
        )

        sns.stripplot(
            data=df, x='group', y='fisher_z', order=group_order,
            color='black', size=4, alpha=0.4, ax=ax, jitter=True
        )

        ax.set_xlabel('Group', fontsize=12, fontweight='bold')
        ax.set_ylabel('Fisher-Z Coherence', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(group_order)))
        ax.set_xticklabels(['Non-anhedonic', 'Low anhedonic', 'High anhedonic'], fontsize=11, rotation=0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        title = f'{network_key} - {band_name}'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        stats_text = []
        n_text = 'N: ' + ', '.join([
            f"{g.split('-')[0].capitalize()}={group_sizes.get(g, 0)}"
            for g in group_order
            if g in group_sizes
        ])
        stats_text.append(n_text)

        if not np.isnan(omnibus_p):
            stats_text.append(f'Welch ANOVA p = {omnibus_p:.4f}')
        if not np.isnan(fdr_p):
            fdr_sig = '***' if fdr_p < 0.001 else '**' if fdr_p < 0.01 else '*' if fdr_p < 0.05 else 'ns'
            stats_text.append(f'FDR-corrected p = {fdr_p:.4f} {fdr_sig}')

        show_posthoc = post_hoc_df is not None and (np.isnan(fdr_p) or fdr_p < 0.05)
        if show_posthoc:
            stats_text.append('Games-Howell post-hoc:')
            for _, row in post_hoc_df.iterrows():
                a = row['A']
                b = row['B']
                p_val = row['pval']
                sig_marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                a_short = a.split('-')[0].capitalize()
                b_short = b.split('-')[0].capitalize()
                stats_text.append(f'  {a_short} vs {b_short}: p = {p_val:.4f} {sig_marker}')
        elif post_hoc_df is not None:
            stats_text.append('Games-Howell post-hoc not shown (omnibus not significant after FDR)')

        if stats_text:
            ax.text(0.02, 0.98, '\n'.join(stats_text),
                    transform=ax.transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    family='monospace')

        fig.tight_layout()
        return fig

    figures = []

    # Static FC violins
    if static_groups:
        all_static_networks = set()
        for group_data in static_groups.values():
            all_static_networks.update(group_data.keys())
        for network_key in sorted(all_static_networks):
            plot_data = []
            for group_name in ['non-anhedonic', 'low-anhedonic', 'high-anhedonic']:
                network_data = static_groups.get(group_name, {}).get(network_key, {})
                for val in network_data.get('observed_values', []):
                    plot_data.append({'group': group_name, 'fisher_z': val})
            if not plot_data:
                continue
            df = pd.DataFrame(plot_data)
            fig = _make_violin(df, 'static', network_key, static_anova)
            figures.append((fig, {
                'band_name': 'static',
                'network_key': network_key,
                'safe_network': network_key.replace('/', '_').replace(' ', '_')
            }))

    # Slow-band violins
    all_bands = set()
    all_networks = set()
    for group_data in slow_band_groups.values():
        for network_key, network_bands in group_data.items():
            all_networks.add(network_key)
            all_bands.update(network_bands.keys())

    sorted_bands = sorted(all_bands, key=lambda x: int(x.split('-')[1]), reverse=True)

    for band_name in sorted_bands:
        band_anova_results = slow_band_anova.get(band_name, {})
        if not band_anova_results:
            continue
        for network_key in sorted(all_networks):
            plot_data = []
            for group_name in ['non-anhedonic', 'low-anhedonic', 'high-anhedonic']:
                band_data = slow_band_groups.get(group_name, {}).get(network_key, {}).get(band_name, {})
                for val in band_data.get('observed_values', []):
                    plot_data.append({'group': group_name, 'fisher_z': val})
            if not plot_data:
                continue
            df = pd.DataFrame(plot_data)
            fig = _make_violin(df, band_name, network_key, band_anova_results)
            figures.append((fig, {
                'band_name': band_name,
                'network_key': network_key,
                'safe_network': network_key.replace('/', '_').replace(' ', '_')
            }))

    return figures


def plot_ipsilateral_intra_network_violin(stat_data, anova_results):
    """
    Create violin plots for ipsilateral intra-network connectivity (within hemisphere).
    Uses ANOVA/post-hoc structures keyed under ipsilateral results.
    """
    import pandas as pd
    import seaborn as sns

    ipsi_static_groups = stat_data.get('ipsi_static_coherence_by_group', {})
    ipsi_slow_groups = stat_data.get('ipsi_slow_band_coherence_by_group', {})
    ipsi_static_anova = anova_results.get('ipsi_static_results', {})
    ipsi_slow_anova = anova_results.get('ipsi_slow_band_results', {})
    post_hoc_collection = anova_results.get('post_hoc_collection', [])

    if not ipsi_static_groups and not ipsi_slow_groups:
        return []

    # Build post-hoc lookup
    post_hoc_lookup = {}
    for ph_data in post_hoc_collection:
        network = ph_data['network']
        band = ph_data['band']
        key = (band, network)
        post_hoc_lookup[key] = ph_data['post_hoc']

    def _make_violin(df, band_name, conn_key, anova_lookup, title_label):
        network_anova = anova_lookup.get(conn_key, {})
        omnibus_p = network_anova.get('omnibus_p', np.nan)
        fdr_p = network_anova.get('fdr_corrected_p', np.nan)
        group_sizes = network_anova.get('group_sizes', {})
        post_hoc_df = post_hoc_lookup.get((band_name, conn_key))

        global _open_fig_warning_shown
        if not _open_fig_warning_shown and len(plt.get_fignums()) >= 20:
            print("[WARN] More than 20 figures are open; consider closing figures to conserve memory.")
            _open_fig_warning_shown = True

        fig, ax = plt.subplots(figsize=(10, 6))

        group_order = ['non-anhedonic', 'low-anhedonic', 'high-anhedonic']
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        color_map = dict(zip(group_order, colors))

        sns.violinplot(
            data=df,
            x='group',
            y='fisher_z',
            order=group_order,
            hue='group',
            palette=color_map,
            dodge=False,
            inner=None,
            ax=ax,
            alpha=0.6,
            legend=False,
        )

        sns.boxplot(
            data=df, x='group', y='fisher_z', order=group_order,
            width=0.3, showfliers=False, ax=ax,
            boxprops=dict(alpha=0.7),
            whiskerprops=dict(alpha=0.7),
            capprops=dict(alpha=0.7)
        )

        sns.stripplot(
            data=df, x='group', y='fisher_z', order=group_order,
            color='black', size=4, alpha=0.4, ax=ax, jitter=True
        )

        ax.set_xlabel('Group', fontsize=12, fontweight='bold')
        ax.set_ylabel('Fisher-Z Connectivity', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(group_order)))
        ax.set_xticklabels(['Non-anhedonic', 'Low anhedonic', 'High anhedonic'], fontsize=11, rotation=0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        ax.set_title(f'{title_label} - {band_name}', fontsize=14, fontweight='bold', pad=20)

        stats_text = []
        n_text = 'N: ' + ', '.join([
            f"{g.split('-')[0].capitalize()}={group_sizes.get(g, 0)}"
            for g in group_order
            if g in group_sizes
        ])
        stats_text.append(n_text)

        if not np.isnan(omnibus_p):
            stats_text.append(f'Welch ANOVA p = {omnibus_p:.4f}')
        if not np.isnan(fdr_p):
            fdr_sig = '***' if fdr_p < 0.001 else '**' if fdr_p < 0.01 else '*' if fdr_p < 0.05 else 'ns'
            stats_text.append(f'FDR-corrected p = {fdr_p:.4f} {fdr_sig}')

        show_posthoc = post_hoc_df is not None and (np.isnan(fdr_p) or fdr_p < 0.05)
        if show_posthoc:
            stats_text.append('Games-Howell post-hoc:')
            for _, row in post_hoc_df.iterrows():
                a = row['A']
                b = row['B']
                p_val = row['pval']
                sig_marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                a_short = a.split('-')[0].capitalize()
                b_short = b.split('-')[0].capitalize()
                stats_text.append(f'  {a_short} vs {b_short}: p = {p_val:.4f} {sig_marker}')
        elif post_hoc_df is not None:
            stats_text.append('Games-Howell post-hoc not shown (omnibus not significant after FDR)')

        if stats_text:
            ax.text(0.02, 0.98, '\n'.join(stats_text),
                    transform=ax.transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    family='monospace')

        fig.tight_layout()
        return fig

    figures = []

    # Static ipsilateral
    if ipsi_static_groups:
        all_conn_keys = set()
        for group_data in ipsi_static_groups.values():
            all_conn_keys.update(group_data.keys())
        for conn_key in sorted(all_conn_keys):
            plot_data = []
            for group_name in ['non-anhedonic', 'low-anhedonic', 'high-anhedonic']:
                conn_data = ipsi_static_groups.get(group_name, {}).get(conn_key, {})
                for val in conn_data.get('observed_values', []):
                    plot_data.append({'group': group_name, 'fisher_z': val})
            if not plot_data:
                continue
            df = pd.DataFrame(plot_data)
            title_label = conn_key
            fig = _make_violin(df, 'static_ipsi', conn_key, ipsi_static_anova, title_label)
            figures.append((fig, {'band_name': 'static_ipsi', 'network_key': conn_key, 'safe_network': conn_key.replace('/', '_')}))

    # Slow-band ipsilateral
    if ipsi_slow_groups:
        all_bands = set()
        all_conn_keys = set()
        for group_data in ipsi_slow_groups.values():
            for conn_key, band_dict in group_data.items():
                all_conn_keys.add(conn_key)
                all_bands.update(band_dict.keys())

        for band_name in sorted(all_bands, key=lambda x: int(x.split('-')[1]) if 'slow-' in x else 0, reverse=True):
            for conn_key in sorted(all_conn_keys):
                plot_data = []
                for group_name in ['non-anhedonic', 'low-anhedonic', 'high-anhedonic']:
                    band_data = ipsi_slow_groups.get(group_name, {}).get(conn_key, {}).get(band_name, {})
                    for val in band_data.get('observed_values', []):
                        plot_data.append({'group': group_name, 'fisher_z': val})
                if not plot_data:
                    continue
                df = pd.DataFrame(plot_data)
                title_label = f"{conn_key}"
                fig = _make_violin(df, f'{band_name}_ipsi', conn_key, ipsi_slow_anova, title_label)
                figures.append((fig, {'band_name': f'{band_name}_ipsi', 'network_key': conn_key, 'safe_network': conn_key.replace('/', '_')}))

    return figures
