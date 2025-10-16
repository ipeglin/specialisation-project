#!/usr/bin/env python3
"""
TCP Group Summarization Script

Provides human and machine-readable statistics about patient/control group composition
of filtered subjects. This is an OPTIONAL analytical step that does not affect data
organization or downstream processing.

Author: Ian Philip Eglin
Date: 2025-10-16
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import json
from datetime import datetime

# Add project root to path to import config
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.paths import get_tcp_dataset_path, get_script_output_path


class GroupSummarizer:
    """Generates group-based statistics for filtered subjects"""

    def __init__(self,
                 filtered_subjects_dir: Optional[Path] = None,
                 dataset_path: Optional[Path] = None,
                 output_dir: Optional[Path] = None):
        self.filtered_subjects_dir = Path(filtered_subjects_dir) if filtered_subjects_dir else \
            get_script_output_path('tcp_preprocessing', 'filter_subjects')
        self.dataset_path = Path(dataset_path) if dataset_path else get_tcp_dataset_path()
        self.output_dir = Path(output_dir) if output_dir else \
            get_script_output_path('tcp_preprocessing', 'summarize_groups')

        print(f"Filtered subjects directory: {self.filtered_subjects_dir}")
        print(f"Dataset path: {self.dataset_path}")
        print(f"Output directory: {self.output_dir}")

    def load_filtered_subjects(self) -> pd.DataFrame:
        """Load filtered subjects from filter_subjects output"""
        print("\nLoading filtered subjects...")

        subjects_file = self.filtered_subjects_dir / "task_filtered_subjects.csv"

        if not subjects_file.exists():
            raise FileNotFoundError(
                f"Filtered subjects file not found: {subjects_file}\n"
                f"Please run filter_subjects.py first."
            )

        subjects_df = pd.read_csv(subjects_file)
        print(f"  Loaded {len(subjects_df)} filtered subjects")

        return subjects_df

    def load_phenotype_data(self) -> pd.DataFrame:
        """Load phenotype data to get Group classifications"""
        print("Loading phenotype data...")

        demos_file = self.dataset_path / "phenotype" / "demos.tsv"

        if not demos_file.exists():
            raise FileNotFoundError(
                f"Phenotype demographics file not found: {demos_file}\n"
                f"Please run fetch_global_data.py first."
            )

        try:
            # demos.tsv is CSV format with metadata header (skip first row)
            demos_df = pd.read_csv(demos_file, sep=',', encoding='utf-8', skiprows=1)
        except UnicodeDecodeError:
            demos_df = pd.read_csv(demos_file, sep=',', encoding='latin-1', skiprows=1)

        print(f"  Loaded phenotype data: {demos_df.shape[0]} rows")

        return demos_df

    def merge_group_info(self, subjects_df: pd.DataFrame, demos_df: pd.DataFrame) -> pd.DataFrame:
        """Merge group information into subjects dataframe"""
        print("Merging group information...")

        # Ensure participant_id column for merging
        if 'participant_id' not in demos_df.columns:
            raise ValueError("participant_id column not found in phenotype data")

        # Handle subject_id format conversion if needed
        subjects_for_merge = subjects_df.copy()
        if 'subject_id' in subjects_df.columns:
            # Convert sub-NDARINV to NDARINV for matching
            if subjects_df['subject_id'].iloc[0].startswith('sub-'):
                subjects_for_merge['participant_id_for_merge'] = \
                    subjects_df['subject_id'].str.replace('sub-', '')
            else:
                subjects_for_merge['participant_id_for_merge'] = \
                    'sub-' + subjects_df['subject_id']
        else:
            raise ValueError("subject_id column not found in subjects data")

        # Merge with demos data
        merged = subjects_for_merge.merge(
            demos_df[['participant_id', 'Group', 'sex', 'age', 'Primary_Dx', 'Site']],
            left_on='participant_id_for_merge',
            right_on='participant_id',
            how='left'
        )

        # Check for missing Group assignments
        missing_group = merged['Group'].isna().sum()
        if missing_group > 0:
            print(f"  ⚠ Warning: {missing_group} subjects missing Group assignment")

        print(f"  Merged successfully")

        return merged

    def calculate_group_statistics(self, subjects_with_groups: pd.DataFrame) -> Dict:
        """Calculate comprehensive group statistics"""
        print("\nCalculating group statistics...")

        # Group subjects
        groups = {}

        for group_name in ['Patient', 'GenPop']:
            group_subjects = subjects_with_groups[subjects_with_groups['Group'] == group_name]

            if len(group_subjects) == 0:
                print(f"  No subjects found for group: {group_name}")
                continue

            # Calculate demographics
            demographics = {}

            # Age statistics
            if 'age' in group_subjects.columns:
                age_data = group_subjects['age'].dropna()
                if len(age_data) > 0:
                    demographics['age'] = {
                        'mean': float(age_data.mean()),
                        'std': float(age_data.std()),
                        'median': float(age_data.median()),
                        'min': float(age_data.min()),
                        'max': float(age_data.max())
                    }

            # Sex distribution
            if 'sex' in group_subjects.columns:
                sex_counts = group_subjects['sex'].value_counts(dropna=False).to_dict()
                demographics['sex'] = {str(k): int(v) for k, v in sex_counts.items()}

            # Site distribution
            if 'Site' in group_subjects.columns:
                site_counts = group_subjects['Site'].value_counts(dropna=False).to_dict()
                demographics['site'] = {str(k): int(v) for k, v in site_counts.items()}

            # Primary diagnosis distribution (for patients)
            if group_name == 'Patient' and 'Primary_Dx' in group_subjects.columns:
                dx_counts = group_subjects['Primary_Dx'].value_counts(dropna=False).to_dict()
                demographics['primary_diagnosis'] = {str(k): int(v) for k, v in dx_counts.items()}

            # Task availability
            task_availability = {}
            for task in ['hammer', 'stroop']:
                for data_type in ['raw', 'timeseries', 'events']:
                    col_name = f'has_{task}_{data_type}'
                    if col_name in group_subjects.columns:
                        count = group_subjects[col_name].sum()
                        task_availability[f'{task}_{data_type}'] = int(count)

            groups[group_name] = {
                'count': len(group_subjects),
                'subject_ids': sorted(group_subjects['subject_id'].tolist()),
                'demographics': demographics,
                'task_availability': task_availability
            }

            print(f"  {group_name}: {len(group_subjects)} subjects")

        return groups

    def export_results(self, groups: Dict, subjects_with_groups: pd.DataFrame) -> None:
        """Export group summary results"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nExporting results to {self.output_dir}")

        # 1. Export comprehensive JSON summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'dataset_path': str(self.dataset_path),
            'filtered_subjects_source': str(self.filtered_subjects_dir),
            'total_filtered_subjects': len(subjects_with_groups),
            'groups': groups,
            'note': 'This is an analytical summary. Group classification does not affect data organization.'
        }

        with open(self.output_dir / "group_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  ✓ Group summary JSON: group_summary.json")

        # 2. Export human-readable text report
        self._export_text_report(groups, subjects_with_groups)

        # 3. Export flat CSV with group assignments
        group_csv = subjects_with_groups[['subject_id', 'Group', 'sex', 'age', 'Site', 'Primary_Dx']].copy()
        group_csv.to_csv(self.output_dir / "subjects_with_groups.csv", index=False)
        print(f"  ✓ Subjects with groups CSV: subjects_with_groups.csv")

    def _export_text_report(self, groups: Dict, subjects_with_groups: pd.DataFrame) -> None:
        """Export human-readable text report"""
        report_file = self.output_dir / "group_breakdown.txt"

        with open(report_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("TCP FILTERED SUBJECTS - GROUP BREAKDOWN\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total filtered subjects: {len(subjects_with_groups)}\n\n")

            for group_name, group_data in groups.items():
                f.write(f"\n{'-' * 70}\n")
                f.write(f"GROUP: {group_name}\n")
                f.write(f"{'-' * 70}\n\n")
                f.write(f"Count: {group_data['count']} subjects\n\n")

                # Demographics
                if 'demographics' in group_data:
                    f.write("Demographics:\n")

                    if 'age' in group_data['demographics']:
                        age = group_data['demographics']['age']
                        f.write(f"  Age: {age['mean']:.1f} ± {age['std']:.1f} years ")
                        f.write(f"(range: {age['min']:.1f}-{age['max']:.1f})\n")

                    if 'sex' in group_data['demographics']:
                        f.write(f"  Sex distribution: {group_data['demographics']['sex']}\n")

                    if 'site' in group_data['demographics']:
                        f.write(f"  Sites: {len(group_data['demographics']['site'])} unique sites\n")

                    if 'primary_diagnosis' in group_data['demographics']:
                        f.write(f"  Primary diagnoses: ")
                        f.write(f"{len(group_data['demographics']['primary_diagnosis'])} unique\n")

                # Task availability
                if 'task_availability' in group_data and group_data['task_availability']:
                    f.write("\nTask Data Availability:\n")
                    for task_data, count in group_data['task_availability'].items():
                        percentage = (count / group_data['count']) * 100 if group_data['count'] > 0 else 0
                        f.write(f"  {task_data}: {count}/{group_data['count']} ({percentage:.1f}%)\n")

                f.write(f"\n")

        print(f"  ✓ Human-readable report: group_breakdown.txt")

    def print_summary(self, groups: Dict) -> None:
        """Print summary to console"""
        print(f"\n{'=' * 60}")
        print(f"GROUP SUMMARY")
        print(f"{'=' * 60}")

        for group_name, group_data in groups.items():
            print(f"\n{group_name}: {group_data['count']} subjects")

            if 'demographics' in group_data and 'age' in group_data['demographics']:
                age = group_data['demographics']['age']
                print(f"  Age: {age['mean']:.1f} ± {age['std']:.1f} years")

            if 'demographics' in group_data and 'sex' in group_data['demographics']:
                print(f"  Sex: {group_data['demographics']['sex']}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Summarize patient/control groups for filtered TCP subjects'
    )
    parser.add_argument('--filtered-subjects-dir', type=Path,
                        help='Override filtered subjects directory (auto-detected by default)')
    parser.add_argument('--dataset-path', type=Path,
                        help='Override dataset path')
    parser.add_argument('--output-dir', type=Path,
                        help='Override output directory')

    args = parser.parse_args()

    print("TCP Group Summarization")
    print("=" * 50)

    try:
        # Initialize summarizer
        summarizer = GroupSummarizer(
            filtered_subjects_dir=args.filtered_subjects_dir,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir
        )

        # Load data
        subjects_df = summarizer.load_filtered_subjects()
        demos_df = summarizer.load_phenotype_data()

        # Merge group information
        subjects_with_groups = summarizer.merge_group_info(subjects_df, demos_df)

        # Calculate statistics
        groups = summarizer.calculate_group_statistics(subjects_with_groups)

        # Export results
        summarizer.export_results(groups, subjects_with_groups)

        # Print summary
        summarizer.print_summary(groups)

        print(f"\n{'=' * 60}")
        print(f"GROUP SUMMARIZATION COMPLETE")
        print(f"{'=' * 60}")
        print(f"Results saved to: {summarizer.output_dir}")
        print(f"\nNote: This is an analytical summary only.")
        print(f"Data organization remains group-agnostic.")
        print(f"\nNext steps:")
        print(f"  1. Run map_subject_files.py to create file path mapping")
        print(f"  2. Run fetch_filtered_data.py to download MRI data")

        return 0

    except Exception as e:
        print(f"❌ Error during group summarization: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
