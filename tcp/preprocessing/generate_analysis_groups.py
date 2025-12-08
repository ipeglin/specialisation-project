#!/usr/bin/env python3
"""
TCP Analysis Groups Generation Script

Generates the final 4 analysis group datasets by combining anhedonia and diagnosis classifications:

PRIMARY: Anhedonia Analysis (ALL subjects with valid SHAPS)
- Group A: non-anhedonic (SHAPS 0-2)
- Group B: low-anhedonic (SHAPS 3-8)
- Group C: high-anhedonic (SHAPS 9-14)

SECONDARY: MDD Primary + Controls
- Controls (Group = "GenPop") + MDD Primary patients

TERTIARY: MDD Primary/Comorbid + Controls
- Controls + MDD Primary + MDD Comorbid

QUATERNARY: All MDD types + Controls
- Controls + MDD Primary + MDD Comorbid + MDD Past

Author: Ian Philip Eglin
Date: 2025-10-25
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Add project root to path to import config
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.paths import get_script_output_path, get_tcp_dataset_path
from tcp.preprocessing.utils.unicode_compat import CHECK, ERROR


class AnalysisGroupsGenerator:
    """Generator for final analysis group datasets"""

    def __init__(self,
                 anhedonia_dir: Optional[Path] = None,
                 diagnosis_dir: Optional[Path] = None,
                 output_dir: Optional[Path] = None):
        self.anhedonia_dir = Path(anhedonia_dir) if anhedonia_dir else \
            get_script_output_path('tcp_preprocessing', 'classify_anhedonia')
        self.diagnosis_dir = Path(diagnosis_dir) if diagnosis_dir else \
            get_script_output_path('tcp_preprocessing', 'classify_diagnoses')
        self.output_dir = Path(output_dir) if output_dir else \
            get_script_output_path('tcp_preprocessing', 'generate_analysis_groups')

        self.anhedonia_subjects: Optional[pd.DataFrame] = None
        self.diagnosis_subjects: Optional[pd.DataFrame] = None

        print(f"Anhedonia classification input: {self.anhedonia_dir}")
        print(f"Diagnosis classification input: {self.diagnosis_dir}")
        print(f"Output directory: {self.output_dir}")

    def load_classification_data(self) -> None:
        """Load anhedonia and diagnosis classification results"""
        print("Loading classification data...")

        # Load anhedonia-classified subjects
        anhedonia_file = self.anhedonia_dir / "anhedonia_classified_subjects.csv"
        if not anhedonia_file.exists():
            raise FileNotFoundError(
                f"Anhedonia classified subjects file not found: {anhedonia_file}\n"
                f"Please run classify_anhedonia.py first."
            )

        self.anhedonia_subjects = pd.read_csv(anhedonia_file)
        print(f"  Loaded anhedonia classifications: {len(self.anhedonia_subjects)} subjects")

        # Load diagnosis-classified subjects
        diagnosis_file = self.diagnosis_dir / "diagnosis_classified_subjects.csv"
        if not diagnosis_file.exists():
            raise FileNotFoundError(
                f"Diagnosis classified subjects file not found: {diagnosis_file}\n"
                f"Please run classify_diagnoses.py first."
            )

        self.diagnosis_subjects = pd.read_csv(diagnosis_file)
        print(f"  Loaded diagnosis classifications: {len(self.diagnosis_subjects)} subjects")

    def merge_classifications(self) -> pd.DataFrame:
        """Merge anhedonia and diagnosis classifications"""
        print("Merging anhedonia and diagnosis classifications...")

        # Drop patient_control from anhedonia if it exists (we'll use diagnosis version)
        anhedonia_cols_to_use = [col for col in self.anhedonia_subjects.columns if col != 'patient_control']
        anhedonia_subset = self.anhedonia_subjects[anhedonia_cols_to_use]

        # Merge on subject_id
        merged_subjects = pd.merge(
            anhedonia_subset,
            self.diagnosis_subjects[['subject_id', 'mdd_status', 'patient_control']],
            on='subject_id',
            how='inner'
        )

        print(f"  Successfully merged: {len(merged_subjects)} subjects")
        print(f"  Subjects lost in merge: {len(self.anhedonia_subjects) + len(self.diagnosis_subjects) - 2*len(merged_subjects)}")

        # Verify required columns are present
        required_cols = ['subject_id', 'anhedonia_class', 'anhedonic_status', 'mdd_status', 'patient_control']
        missing_cols = [col for col in required_cols if col not in merged_subjects.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns after merge: {missing_cols}")

        return merged_subjects

    def generate_primary_analysis_group(self, merged_subjects: pd.DataFrame) -> pd.DataFrame:
        """Generate PRIMARY analysis group: Anhedonia Analysis"""
        print("Generating PRIMARY analysis group (Anhedonia Analysis)...")

        # PRIMARY includes ALL subjects with valid anhedonia classification
        primary_subjects = merged_subjects.copy()

        # Add analysis group identifier
        primary_subjects['analysis_group'] = 'primary'
        primary_subjects['analysis_subgroup'] = primary_subjects['anhedonia_class']

        # Count subjects by anhedonia class
        anhedonia_counts = primary_subjects['anhedonia_class'].value_counts()
        print(f"  PRIMARY group composition:")
        for anhedonia_class, count in anhedonia_counts.items():
            print(f"    {anhedonia_class}: {count}")

        print(f"  Total PRIMARY subjects: {len(primary_subjects)}")
        return primary_subjects

    def generate_secondary_analysis_group(self, merged_subjects: pd.DataFrame) -> pd.DataFrame:
        """Generate SECONDARY analysis group: MDD Primary + Controls"""
        print("Generating SECONDARY analysis group (MDD Primary + Controls)...")

        # SECONDARY includes Controls and MDD_Primary only
        secondary_subjects = merged_subjects[
            merged_subjects['mdd_status'].isin(['Control', 'MDD_Primary'])
        ].copy()

        # Add analysis group identifier
        secondary_subjects['analysis_group'] = 'secondary'
        secondary_subjects['analysis_subgroup'] = secondary_subjects['mdd_status']

        # Count subjects by MDD status
        mdd_counts = secondary_subjects['mdd_status'].value_counts()
        print(f"  SECONDARY group composition:")
        for mdd_status, count in mdd_counts.items():
            print(f"    {mdd_status}: {count}")

        print(f"  Total SECONDARY subjects: {len(secondary_subjects)}")
        return secondary_subjects

    def generate_tertiary_analysis_group(self, merged_subjects: pd.DataFrame) -> pd.DataFrame:
        """Generate TERTIARY analysis group: MDD Primary/Comorbid + Controls"""
        print("Generating TERTIARY analysis group (MDD Primary/Comorbid + Controls)...")

        # TERTIARY includes Controls, MDD_Primary, and MDD_Comorbid
        tertiary_subjects = merged_subjects[
            merged_subjects['mdd_status'].isin(['Control', 'MDD_Primary', 'MDD_Comorbid'])
        ].copy()

        # Add analysis group identifier
        tertiary_subjects['analysis_group'] = 'tertiary'
        tertiary_subjects['analysis_subgroup'] = tertiary_subjects['mdd_status']

        # Count subjects by MDD status
        mdd_counts = tertiary_subjects['mdd_status'].value_counts()
        print(f"  TERTIARY group composition:")
        for mdd_status, count in mdd_counts.items():
            print(f"    {mdd_status}: {count}")

        print(f"  Total TERTIARY subjects: {len(tertiary_subjects)}")
        return tertiary_subjects

    def generate_quaternary_analysis_group(self, merged_subjects: pd.DataFrame) -> pd.DataFrame:
        """Generate QUATERNARY analysis group: All MDD types + Controls"""
        print("Generating QUATERNARY analysis group (All MDD + Controls)...")

        # QUATERNARY includes Controls and all MDD types
        quaternary_subjects = merged_subjects[
            merged_subjects['mdd_status'].isin(['Control', 'MDD_Primary', 'MDD_Comorbid', 'MDD_Past'])
        ].copy()

        # Add analysis group identifier
        quaternary_subjects['analysis_group'] = 'quaternary'
        quaternary_subjects['analysis_subgroup'] = quaternary_subjects['mdd_status']

        # Count subjects by MDD status
        mdd_counts = quaternary_subjects['mdd_status'].value_counts()
        print(f"  QUATERNARY group composition:")
        for mdd_status, count in mdd_counts.items():
            print(f"    {mdd_status}: {count}")

        print(f"  Total QUATERNARY subjects: {len(quaternary_subjects)}")
        return quaternary_subjects

    def generate_all_analysis_groups(self, merged_subjects: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Generate all 4 analysis groups"""
        print("\nGenerating all analysis groups...")

        analysis_groups = {}

        # Generate each analysis group
        analysis_groups['primary'] = self.generate_primary_analysis_group(merged_subjects)
        analysis_groups['secondary'] = self.generate_secondary_analysis_group(merged_subjects)
        analysis_groups['tertiary'] = self.generate_tertiary_analysis_group(merged_subjects)
        analysis_groups['quaternary'] = self.generate_quaternary_analysis_group(merged_subjects)

        return analysis_groups

    def calculate_cross_analysis_statistics(self, analysis_groups: Dict[str, pd.DataFrame]) -> Dict:
        """Calculate comprehensive statistics across all analysis groups"""
        print("Calculating cross-analysis statistics...")

        statistics = {
            'group_sizes': {},
            'overlap_analysis': {},
            'anhedonia_by_diagnosis': {},
            'diagnosis_by_anhedonia': {}
        }

        # Group sizes
        for group_name, group_data in analysis_groups.items():
            statistics['group_sizes'][group_name] = len(group_data)

        # Overlap analysis (how many subjects appear in multiple groups)
        all_subject_sets = {}
        for group_name, group_data in analysis_groups.items():
            all_subject_sets[group_name] = set(group_data['subject_id'])

        # Calculate pairwise overlaps
        overlap_matrix = {}
        for group1 in all_subject_sets.keys():
            overlap_matrix[group1] = {}
            for group2 in all_subject_sets.keys():
                overlap = len(all_subject_sets[group1] & all_subject_sets[group2])
                overlap_matrix[group1][group2] = overlap

        statistics['overlap_analysis'] = overlap_matrix

        # Cross-tabulations using merged data
        merged_subjects = pd.merge(
            self.anhedonia_subjects,
            self.diagnosis_subjects[['subject_id', 'mdd_status', 'patient_control']],
            on='subject_id',
            how='inner'
        )

        # Anhedonia by diagnosis
        if 'anhedonia_class' in merged_subjects.columns and 'mdd_status' in merged_subjects.columns:
            anhedonia_by_diagnosis = pd.crosstab(
                merged_subjects['anhedonia_class'],
                merged_subjects['mdd_status'],
                margins=True
            )
            statistics['anhedonia_by_diagnosis'] = anhedonia_by_diagnosis.to_dict()

        # Diagnosis by anhedonia (binary)
        if 'anhedonic_status' in merged_subjects.columns and 'patient_control' in merged_subjects.columns:
            diagnosis_by_anhedonia = pd.crosstab(
                merged_subjects['patient_control'],
                merged_subjects['anhedonic_status'],
                margins=True
            )
            statistics['diagnosis_by_anhedonia'] = diagnosis_by_anhedonia.to_dict()

        return statistics

    def export_results(self, analysis_groups: Dict[str, pd.DataFrame], statistics: Dict) -> None:
        """Export analysis group datasets and summary"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nExporting results to {self.output_dir}")

        # 1. Export each analysis group
        for group_name, group_data in analysis_groups.items():
            group_file = self.output_dir / f"{group_name}_analysis_subjects.csv"
            group_data.to_csv(group_file, index=False)
            print(f"  {CHECK} {group_name.upper()} analysis group: {group_file}")

        # 2. Export combined dataset with all classifications
        # Drop patient_control from anhedonia if it exists (we'll use diagnosis version)
        anhedonia_cols_to_use = [col for col in self.anhedonia_subjects.columns if col != 'patient_control']
        merged_subjects = pd.merge(
            self.anhedonia_subjects[anhedonia_cols_to_use],
            self.diagnosis_subjects[['subject_id', 'mdd_status', 'patient_control']],
            on='subject_id',
            how='inner'
        )
        combined_file = self.output_dir / "all_subjects_with_classifications.csv"
        merged_subjects.to_csv(combined_file, index=False)
        print(f"  {CHECK} Combined classifications: {combined_file}")

        # 3. Export analysis groups summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'anhedonia_source': str(self.anhedonia_dir),
            'diagnosis_source': str(self.diagnosis_dir),
            'analysis_groups': {
                'primary': {
                    'description': 'Anhedonia Analysis: non-anhedonic vs low-anhedonic vs high-anhedonic',
                    'inclusion_criteria': 'ALL subjects with valid SHAPS scores',
                    'subgroups': ['non-anhedonic', 'low-anhedonic', 'high-anhedonic']
                },
                'secondary': {
                    'description': 'MDD Primary + Controls',
                    'inclusion_criteria': 'Controls (GenPop) + MDD Primary diagnosis',
                    'subgroups': ['Control', 'MDD_Primary']
                },
                'tertiary': {
                    'description': 'MDD Primary/Comorbid + Controls',
                    'inclusion_criteria': 'Controls + MDD Primary + MDD Comorbid',
                    'subgroups': ['Control', 'MDD_Primary', 'MDD_Comorbid']
                },
                'quaternary': {
                    'description': 'All MDD types + Controls',
                    'inclusion_criteria': 'Controls + all MDD types (Primary, Comorbid, Past)',
                    'subgroups': ['Control', 'MDD_Primary', 'MDD_Comorbid', 'MDD_Past']
                }
            },
            'statistics': statistics,
            'note': 'Analysis-ready datasets for anhedonia-focused research'
        }

        summary_file = self.output_dir / "analysis_groups_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  {CHECK} Summary: {summary_file}")

    def print_summary(self, analysis_groups: Dict[str, pd.DataFrame], statistics: Dict) -> None:
        """Print comprehensive summary to console"""
        print(f"\n{'=' * 60}")
        print(f"ANALYSIS GROUPS GENERATION SUMMARY")
        print(f"{'=' * 60}")

        print(f"\nGenerated analysis groups:")
        for group_name, group_data in analysis_groups.items():
            print(f"  {group_name.upper()}: {len(group_data)} subjects")

        print(f"\nGroup overlap analysis:")
        overlap = statistics.get('overlap_analysis', {})
        for group1, overlaps in overlap.items():
            for group2, count in overlaps.items():
                if group1 != group2:
                    total1 = statistics['group_sizes'].get(group1, 0)
                    percentage = (count / total1) * 100 if total1 > 0 else 0
                    print(f"  {group1.upper()} AND {group2.upper()}: {count} subjects ({percentage:.1f}%)")

        # Print cross-tabulations if available
        if 'anhedonia_by_diagnosis' in statistics:
            print(f"\nAnhedonia by Diagnosis cross-tabulation available in summary file")

        if 'diagnosis_by_anhedonia' in statistics:
            print(f"Patient/Control by Anhedonic status cross-tabulation available in summary file")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Generate final analysis group datasets for TCP anhedonia research'
    )
    parser.add_argument('--anhedonia-dir', type=Path,
                        help='Override anhedonia classification directory')
    parser.add_argument('--diagnosis-dir', type=Path,
                        help='Override diagnosis classification directory')
    parser.add_argument('--output-dir', type=Path,
                        help='Override output directory')
    parser.add_argument('--groups', nargs='+',
                        choices=['primary', 'secondary', 'tertiary', 'quaternary', 'all'],
                        default=['all'],
                        help='Which analysis groups to generate (default: all)')

    args = parser.parse_args()

    print("TCP Analysis Groups Generation")
    print("=" * 50)

    try:
        # Initialize generator
        generator = AnalysisGroupsGenerator(
            anhedonia_dir=args.anhedonia_dir,
            diagnosis_dir=args.diagnosis_dir,
            output_dir=args.output_dir
        )

        # Load classification data
        generator.load_classification_data()

        # Merge classifications
        merged_subjects = generator.merge_classifications()

        # Generate requested analysis groups
        if 'all' in args.groups:
            requested_groups = ['primary', 'secondary', 'tertiary', 'quaternary']
        else:
            requested_groups = args.groups

        print(f"\nGenerating requested groups: {', '.join(requested_groups)}")

        analysis_groups = {}
        if 'primary' in requested_groups:
            analysis_groups['primary'] = generator.generate_primary_analysis_group(merged_subjects)
        if 'secondary' in requested_groups:
            analysis_groups['secondary'] = generator.generate_secondary_analysis_group(merged_subjects)
        if 'tertiary' in requested_groups:
            analysis_groups['tertiary'] = generator.generate_tertiary_analysis_group(merged_subjects)
        if 'quaternary' in requested_groups:
            analysis_groups['quaternary'] = generator.generate_quaternary_analysis_group(merged_subjects)

        # Calculate statistics
        statistics = generator.calculate_cross_analysis_statistics(analysis_groups)

        # Export results
        generator.export_results(analysis_groups, statistics)

        # Print summary
        generator.print_summary(analysis_groups, statistics)

        print(f"\n{'=' * 60}")
        print(f"ANALYSIS GROUPS GENERATION COMPLETE")
        print(f"{'=' * 60}")
        print(f"Results saved to: {generator.output_dir}")
        print(f"\nNext steps:")
        print(f"  1. Run sample_subjects_for_download.py to select subjects for data fetching")
        print(f"  2. Run map_subject_files.py and fetch_analysis_data.py to download data")
        print(f"  3. Begin analysis with your selected groups")

        return 0

    except Exception as e:
        print(f"{ERROR} Error during analysis groups generation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
