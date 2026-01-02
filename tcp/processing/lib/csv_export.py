import csv
from pathlib import Path


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


def export_significance_fractions_to_csv(stats_results, output_dir, verbose=True):
    """
    Export significance fraction summary tables to CSV.

    Creates two CSV files (interhemispheric and ipsilateral) with:
    - Rows: Whole-signal, slow-2, slow-3, slow-4, slow-5, slow-6
    - Columns: Non-anhedonic, Low-anhedonic, High-anhedonic
    - Cells: Mean ± SD of significance fractions

    Args:
        stats_results: Output from prepare_statistics_data() containing:
            - static_significance_by_group
            - slow_band_significance_by_group
        output_dir: Directory to save CSV files
        verbose: Whether to print progress
    """
    import pandas as pd
    import numpy as np
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    static_sig = stats_results.get('static_significance_by_group', {})
    slow_band_sig = stats_results.get('slow_band_significance_by_group', {})

    # Define groups and bands
    groups = ['non-anhedonic', 'low-anhedonic', 'high-anhedonic']
    bands = ['whole-signal', 'slow-2', 'slow-3', 'slow-4', 'slow-5', 'slow-6']
    connectivity_types = ['interhemispheric', 'ipsilateral']

    for conn_type in connectivity_types:
        if verbose:
            print(f"  Exporting {conn_type} significance fractions...")
            print(f"    DEBUG: Sample sizes in source data:")
            # Print sample sizes for debugging
            for group in groups:
                static_vals = static_sig.get(group, {}).get(conn_type, [])
                print(f"      {group} whole-signal: n={len(static_vals)}")
                slow_bands_dict = slow_band_sig.get(group, {}).get(conn_type, {})
                for band_name in sorted(slow_bands_dict.keys()):
                    band_vals = slow_bands_dict[band_name]
                    print(f"      {group} {band_name}: n={len(band_vals)}")

        # Build table data
        table_data = []

        for band in bands:
            row = {'Band': band}

            for group in groups:
                if band == 'whole-signal':
                    # Static (whole-signal) data
                    values = static_sig.get(group, {}).get(conn_type, [])
                else:
                    # Slow-band data
                    band_name = band  # e.g., 'slow-4'
                    values = slow_band_sig.get(group, {}).get(conn_type, {}).get(band_name, [])

                if values and len(values) > 0:
                    mean_val = np.mean(values)
                    std_val = np.std(values, ddof=1)  # Sample SD
                    n_val = len(values)

                    # Format as "mean ± std (n=X)"
                    row[f'{group}_mean'] = mean_val
                    row[f'{group}_std'] = std_val
                    row[f'{group}_n'] = n_val
                    row[f'{group}_formatted'] = f"{mean_val:.4f} ± {std_val:.4f} (n={n_val})"
                else:
                    row[f'{group}_mean'] = np.nan
                    row[f'{group}_std'] = np.nan
                    row[f'{group}_n'] = 0
                    row[f'{group}_formatted'] = 'N/A'

            table_data.append(row)

        # Create DataFrame
        df = pd.DataFrame(table_data)

        # Reorder columns for readability
        column_order = ['Band']
        for group in groups:
            column_order.extend([f'{group}_mean', f'{group}_std', f'{group}_n', f'{group}_formatted'])
        df = df[column_order]

        # Export to CSV
        csv_path = output_dir / f'significance_fractions_{conn_type}.csv'
        df.to_csv(csv_path, index=False)

        if verbose:
            print(f"    Saved to: {csv_path}")

    if verbose:
        print(f"  Significance fraction export complete!")
