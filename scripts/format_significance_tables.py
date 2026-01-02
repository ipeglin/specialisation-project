#!/usr/bin/env python3
"""
LaTeX Table Formatter for Significance Fractions

This script processes exported significance fraction CSV files and formats them
for LaTeX tables. It reads the CSV exports from the analysis pipeline and outputs:
1. LaTeX table format (printed to console)
2. Simplified CSV files with formatted values

Usage:
    # Process CSVs from latest run
    python3 scripts/format_significance_tables.py

    # Specify custom input directory
    python3 scripts/format_significance_tables.py --input-dir path/to/csv_exports

    # Save LaTeX output to file
    python3 scripts/format_significance_tables.py --output-latex tables.tex

    # Only print to console (no CSV export)
    python3 scripts/format_significance_tables.py --no-export

Author: Ian Philip Eglin
Date: 2026-01-02
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import project utilities
from config.paths import get_analysis_path


def find_latest_run_dir():
    """
    Find the most recent analysis run directory

    Returns:
        Path: Path to significance_fractions directory in latest run

    Raises:
        FileNotFoundError: If no analysis runs or significance_fractions directory found
    """
    analysis_runs = get_analysis_path('analysis_runs')

    if not analysis_runs.exists():
        raise FileNotFoundError(f"No analysis runs found at {analysis_runs}")

    # Find all run directories
    run_dirs = sorted(
        [d for d in analysis_runs.iterdir() if d.is_dir() and d.name.startswith('run_')],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    if not run_dirs:
        raise FileNotFoundError("No run directories found")

    latest_run = run_dirs[0]
    sig_frac_dir = latest_run / 'csv_exports' / 'significance_fractions'

    if not sig_frac_dir.exists():
        raise FileNotFoundError(f"No significance_fractions directory in {latest_run}")

    return sig_frac_dir


def read_significance_csv(csv_path):
    """
    Read significance fraction CSV file

    Args:
        csv_path: Path to CSV file

    Returns:
        pd.DataFrame with cleaned column names and processed data

    Raises:
        ValueError: If required columns are missing
    """
    df = pd.read_csv(csv_path)

    # Validate required columns exist
    required_cols = ['Band']
    for group in ['non-anhedonic', 'low-anhedonic', 'high-anhedonic']:
        required_cols.extend([f'{group}_mean', f'{group}_std', f'{group}_n'])

    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {missing_cols}")

    # Clean band names (capitalize for display)
    df['Band'] = df['Band'].str.replace('whole-signal', 'Whole-signal')
    df['Band'] = df['Band'].str.replace('slow-', 'Slow-')

    return df


def format_for_latex(df, table_type, precision=3):
    """
    Format DataFrame for LaTeX table output

    Args:
        df: DataFrame with significance fraction data
        table_type: 'interhemispheric' or 'ipsilateral'
        precision: Number of decimal places

    Returns:
        str: LaTeX table code
    """
    # Group abbreviations
    group_abbrev = {
        'non-anhedonic': 'NA',
        'low-anhedonic': 'LA',
        'high-anhedonic': 'HA'
    }

    # Table caption
    caption = {
        'interhemispheric': 'Interhemispheric Intra-Network Connectivity',
        'ipsilateral': 'Ipsilateral Intra-Network Connectivity'
    }

    # Build LaTeX table
    lines = []
    lines.append("\\begin{tabular}{l l l l}")
    lines.append("    \\toprule")

    # Header row
    header = "    \\textbf{Signal}"
    for group_full, group_abbr in group_abbrev.items():
        header += f" & \\textbf{{{group_abbr}}} $[\\bar{{s}} \\pm \\sigma]$"
    header += " \\\\"
    lines.append(header)
    lines.append("    \\midrule")

    # Data rows
    for _, row in df.iterrows():
        band = row['Band']
        line = f"    {band}"

        for group in ['non-anhedonic', 'low-anhedonic', 'high-anhedonic']:
            mean_val = row[f'{group}_mean']
            std_val = row[f'{group}_std']

            if pd.isna(mean_val) or pd.isna(std_val):
                line += " & N/A"
            else:
                # Format as: $0.423 \pm 0.051$
                line += f" & ${mean_val:.{precision}f} \\pm {std_val:.{precision}f}$"

        line += " \\\\"
        lines.append(line)

    lines.append("    \\bottomrule")
    lines.append("\\end{tabular}")

    # Add caption comment
    output = [f"% {caption[table_type]}", ""]
    output.extend(lines)

    return "\n".join(output)


def export_formatted_csv(df, output_path, precision=3):
    """
    Export simplified CSV with formatted mean ± std values

    Args:
        df: DataFrame with significance fraction data
        output_path: Path to save CSV file
        precision: Number of decimal places
    """
    # Create simplified DataFrame
    export_df = pd.DataFrame()
    export_df['Signal'] = df['Band']

    group_abbrev = {
        'non-anhedonic': 'NA',
        'low-anhedonic': 'LA',
        'high-anhedonic': 'HA'
    }

    for group_full, group_abbr in group_abbrev.items():
        mean_col = f'{group_full}_mean'
        std_col = f'{group_full}_std'

        # Format as "mean ± std"
        formatted_values = []
        for _, row in df.iterrows():
            mean_val = row[mean_col]
            std_val = row[std_col]

            if pd.isna(mean_val) or pd.isna(std_val):
                formatted_values.append('N/A')
            else:
                formatted_values.append(f"{mean_val:.{precision}f} ± {std_val:.{precision}f}")

        export_df[group_abbr] = formatted_values

    # Save to CSV
    export_df.to_csv(output_path, index=False)
    print(f"  Exported: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Format significance fraction CSVs for LaTeX tables"
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        help='Directory containing significance_fractions CSV files'
    )
    parser.add_argument(
        '--output-latex',
        type=Path,
        help='Save LaTeX output to file instead of printing to console'
    )
    parser.add_argument(
        '--output-csv-dir',
        type=Path,
        help='Directory to save formatted CSV files (default: same as input)'
    )
    parser.add_argument(
        '--no-export',
        action='store_true',
        help='Only print LaTeX output, do not export CSV files'
    )
    parser.add_argument(
        '--precision',
        type=int,
        default=3,
        help='Number of decimal places for values (default: 3)'
    )
    args = parser.parse_args()

    # 1. Determine input directory
    if args.input_dir:
        input_dir = args.input_dir
        if not input_dir.exists():
            print(f"Error: Input directory does not exist: {input_dir}")
            sys.exit(1)
    else:
        try:
            input_dir = find_latest_run_dir()
            print(f"Using latest run: {input_dir.parent.parent.name}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("\nPlease specify --input-dir with the path to significance_fractions CSV files")
            sys.exit(1)

    # 2. Find CSV files
    inter_csv = input_dir / 'significance_fractions_interhemispheric.csv'
    ipsi_csv = input_dir / 'significance_fractions_ipsilateral.csv'

    if not inter_csv.exists():
        print(f"Error: Interhemispheric CSV not found: {inter_csv}")
        sys.exit(1)
    if not ipsi_csv.exists():
        print(f"Error: Ipsilateral CSV not found: {ipsi_csv}")
        sys.exit(1)

    print(f"\nReading CSV files from: {input_dir}")

    # 3. Read CSV files
    try:
        df_inter = read_significance_csv(inter_csv)
        df_ipsi = read_significance_csv(ipsi_csv)
    except ValueError as e:
        print(f"Error reading CSV files: {e}")
        print("\nThe CSV file may be corrupted or from an incompatible version.")
        sys.exit(1)

    # 4. Generate LaTeX tables
    latex_inter = format_for_latex(df_inter, 'interhemispheric', args.precision)
    latex_ipsi = format_for_latex(df_ipsi, 'ipsilateral', args.precision)

    # 5. Output LaTeX
    if args.output_latex:
        with open(args.output_latex, 'w') as f:
            f.write("% Interhemispheric Intra-Network Connectivity\n")
            f.write(latex_inter)
            f.write("\n\n")
            f.write("% Ipsilateral Intra-Network Connectivity\n")
            f.write(latex_ipsi)
        print(f"\nLaTeX tables saved to: {args.output_latex}")
    else:
        print("\n" + "="*80)
        print("LATEX TABLE OUTPUT")
        print("="*80)
        print("\n" + latex_inter)
        print("\n" + latex_ipsi)

    # 6. Export formatted CSVs
    if not args.no_export:
        output_csv_dir = args.output_csv_dir or input_dir
        output_csv_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nExporting formatted CSV files to: {output_csv_dir}")
        export_formatted_csv(
            df_inter,
            output_csv_dir / 'significance_fractions_interhemispheric_latex.csv',
            args.precision
        )
        export_formatted_csv(
            df_ipsi,
            output_csv_dir / 'significance_fractions_ipsilateral_latex.csv',
            args.precision
        )

    print("\nDone!")


if __name__ == '__main__':
    main()
