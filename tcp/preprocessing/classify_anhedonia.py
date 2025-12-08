#!/usr/bin/env python3
"""
TCP Anhedonia Classification Script

Classifies subjects into anhedonia categories based on SHAPS total scores:
- non-anhedonic: SHAPS 0-2
- low-anhedonic: SHAPS 3-8
- high-anhedonic: SHAPS 9-14

This classification is the foundation for the PRIMARY analysis group comparing
anhedonia levels across all subjects regardless of diagnosis.

Author: Ian Philip Eglin
Date: 2025-10-25
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path to import config
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.paths import get_script_output_path, get_tcp_dataset_path
from tcp.preprocessing.utils.phenotype_filters import AnhedoniaSegmentationFilter
from tcp.preprocessing.utils.unicode_compat import CHECK, ERROR


class AnhedoniaClassificationPipeline:
    """Pipeline for classifying subjects by anhedonia severity"""

    def __init__(self,
                 dataset_path: Optional[Path] = None,
                 base_subjects_dir: Optional[Path] = None,
                 output_dir: Optional[Path] = None):
        self.dataset_path = Path(dataset_path) if dataset_path else get_tcp_dataset_path()
        self.base_subjects_dir = Path(base_subjects_dir) if base_subjects_dir else \
            get_script_output_path('tcp_preprocessing', 'filter_base_subjects')
        self.output_dir = Path(output_dir) if output_dir else \
            get_script_output_path('tcp_preprocessing', 'classify_anhedonia')

        self.subjects_df: Optional[pd.DataFrame] = None
        self.phenotype_data: Dict[str, pd.DataFrame] = {}

        print(f"Dataset path: {self.dataset_path}")
        print(f"Base subjects input: {self.base_subjects_dir}")
        print(f"Output directory: {self.output_dir}")

    def load_base_subjects(self) -> None:
        """Load base filtered subjects"""
        print("Loading base filtered subjects...")

        base_subjects_file = self.base_subjects_dir / "base_filtered_subjects.csv"

        if not base_subjects_file.exists():
            raise FileNotFoundError(
                f"Base filtered subjects file not found: {base_subjects_file}\n"
                f"Please run filter_base_subjects.py first."
            )

        self.subjects_df = pd.read_csv(base_subjects_file)
        print(f"  Loaded {len(self.subjects_df)} base filtered subjects")

        if 'subject_id' not in self.subjects_df.columns:
            raise ValueError("subject_id column not found in base subjects data")

    def load_phenotype_data(self) -> None:
        """Load phenotype data files from dataset"""
        print("Loading phenotype data...")

        phenotype_files = {
            'shaps01': 'phenotype/shaps01.tsv'
        }

        for file_key, file_path in phenotype_files.items():
            full_path = self.dataset_path / file_path

            if full_path.exists():
                try:
                    df = pd.read_csv(full_path, sep='\t', encoding='utf-8')
                    self.phenotype_data[file_key] = df
                    print(f"  Loaded {file_key}: {df.shape[0]} rows, {df.shape[1]} columns")

                    # Show relevant columns for SHAPS
                    shaps_cols = [col for col in df.columns if 'shaps' in col.lower()]
                    if shaps_cols:
                        print(f"    SHAPS columns: {shaps_cols}")

                except Exception as e:
                    print(f"  ERROR: Could not load {file_key} from {full_path}: {e}")
                    raise
            else:
                print(f"  ERROR: Phenotype file not found: {full_path}")
                raise FileNotFoundError(f"Required phenotype file not found: {full_path}")

    def classify_anhedonia_levels(self) -> Tuple[pd.DataFrame, Dict]:
        """Classify subjects into anhedonia categories"""
        print("Classifying subjects by anhedonia levels...")

        # Create anhedonia segmentation filter
        anhedonia_filter = AnhedoniaSegmentationFilter()

        # Apply classification
        filter_result = anhedonia_filter.apply(
            subjects_df=self.subjects_df,
            phenotype_data=self.phenotype_data
        )

        classified_subjects = filter_result.included_subjects.copy()
        excluded_subjects = filter_result.excluded_subjects.copy()

        print(f"  Classification results:")
        print(f"    Successfully classified: {len(classified_subjects)}")
        print(f"    Excluded (invalid SHAPS): {len(excluded_subjects)}")

        # Count subjects in each anhedonia category
        if len(classified_subjects) > 0 and 'anhedonia_class' in classified_subjects.columns:
            anhedonia_counts = classified_subjects['anhedonia_class'].value_counts()
            print(f"    Anhedonia distribution:")
            for category, count in anhedonia_counts.items():
                print(f"      {category}: {count}")
        else:
            print(f"    WARNING: No anhedonia_class column found in classified subjects")

        # Create statistics
        statistics = {
            'total_input_subjects': len(self.subjects_df),
            'successfully_classified': len(classified_subjects),
            'excluded_invalid_shaps': len(excluded_subjects),
            'anhedonia_distribution': {}
        }

        if len(classified_subjects) > 0 and 'anhedonia_class' in classified_subjects.columns:
            statistics['anhedonia_distribution'] = classified_subjects['anhedonia_class'].value_counts().to_dict()

        return classified_subjects, statistics

    def add_binary_anhedonia_status(self, classified_subjects: pd.DataFrame) -> pd.DataFrame:
        """Add binary anhedonic vs non-anhedonic status for easier analysis"""
        print("Adding binary anhedonia status...")

        classified_subjects = classified_subjects.copy()

        # Create binary anhedonic status (anhedonic >= 3, non-anhedonic < 3)
        # IMPORTANT: non-anhedonic group only includes controls (patient_control == 'Control')
        if 'anhedonia_class' in classified_subjects.columns:
            def determine_anhedonic_status(row):
                """Determine anhedonic status, filtering non-anhedonic to controls only"""
                anhedonia_class = row.get('anhedonia_class')
                patient_control = row.get('patient_control', '')

                if anhedonia_class in ['low-anhedonic', 'high-anhedonic']:
                    return 'anhedonic'
                elif anhedonia_class == 'non-anhedonic' and patient_control == 'Control':
                    return 'non-anhedonic'
                else:
                    # Non-anhedonic patients (not controls) are excluded from binary analysis
                    return None

            classified_subjects['anhedonic_status'] = classified_subjects.apply(
                determine_anhedonic_status, axis=1
            )

            binary_counts = classified_subjects['anhedonic_status'].value_counts()
            excluded_count = classified_subjects['anhedonic_status'].isna().sum()

            print(f"  Binary anhedonia distribution:")
            for status, count in binary_counts.items():
                print(f"    {status}: {count}")
            if excluded_count > 0:
                print(f"    Excluded (non-anhedonic patients): {excluded_count}")

        return classified_subjects

    def export_results(self, classified_subjects: pd.DataFrame, statistics: Dict) -> None:
        """Export anhedonia classification results"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nExporting results to {self.output_dir}")

        # 1. Export all classified subjects
        classified_file = self.output_dir / "anhedonia_classified_subjects.csv"
        classified_subjects.to_csv(classified_file, index=False)
        print(f"  {CHECK} Classified subjects: {classified_file}")

        # 2. Export subjects by anhedonia category
        if 'anhedonia_class' in classified_subjects.columns:
            categories = classified_subjects['anhedonia_class'].unique()
            for category in categories:
                category_subjects = classified_subjects[classified_subjects['anhedonia_class'] == category]
                category_file = self.output_dir / f"{category.replace('-', '_')}_subjects.csv"
                category_subjects.to_csv(category_file, index=False)
                print(f"  {CHECK} {category} subjects: {category_file}")

        # 3. Export binary groups (anhedonic vs non-anhedonic)
        if 'anhedonic_status' in classified_subjects.columns:
            for status in ['anhedonic', 'non-anhedonic']:
                status_subjects = classified_subjects[classified_subjects['anhedonic_status'] == status]
                if len(status_subjects) > 0:
                    status_file = self.output_dir / f"{status.replace('-', '_')}_subjects.csv"
                    status_subjects.to_csv(status_file, index=False)
                    print(f"  {CHECK} {status} subjects: {status_file}")

        # 4. Export excluded subjects
        anhedonia_filter = AnhedoniaSegmentationFilter()
        filter_result = anhedonia_filter.apply(
            subjects_df=self.subjects_df,
            phenotype_data=self.phenotype_data
        )

        if len(filter_result.excluded_subjects) > 0:
            excluded_file = self.output_dir / "anhedonia_excluded_subjects.csv"
            filter_result.excluded_subjects.to_csv(excluded_file, index=False)
            print(f"  {CHECK} Excluded subjects: {excluded_file}")

        # 5. Export classification summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'dataset_path': str(self.dataset_path),
            'base_subjects_source': str(self.base_subjects_dir),
            'classification_criteria': {
                'non-anhedonic': 'SHAPS score 0-2',
                'low-anhedonic': 'SHAPS score 3-8',
                'high-anhedonic': 'SHAPS score 9-14'
            },
            'statistics': statistics,
            'note': 'Anhedonia classification for PRIMARY analysis group'
        }

        summary_file = self.output_dir / "anhedonia_classification_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  {CHECK} Summary: {summary_file}")

    def print_summary(self, statistics: Dict) -> None:
        """Print summary to console"""
        print(f"\n{'=' * 60}")
        print(f"ANHEDONIA CLASSIFICATION SUMMARY")
        print(f"{'=' * 60}")

        print(f"\nClassification criteria:")
        print(f"  non-anhedonic: SHAPS score 0-2")
        print(f"  low-anhedonic: SHAPS score 3-8")
        print(f"  high-anhedonic: SHAPS score 9-14")

        print(f"\nSubject classification:")
        print(f"  Input subjects: {statistics['total_input_subjects']}")
        print(f"  Successfully classified: {statistics['successfully_classified']}")
        print(f"  Excluded (invalid SHAPS): {statistics['excluded_invalid_shaps']}")

        if statistics['anhedonia_distribution']:
            print(f"\nAnhedonia distribution:")
            total_classified = statistics['successfully_classified']
            for category, count in statistics['anhedonia_distribution'].items():
                percentage = (count / total_classified) * 100 if total_classified > 0 else 0
                print(f"  {category}: {count} ({percentage:.1f}%)")

        classification_rate = (statistics['successfully_classified'] / statistics['total_input_subjects']) * 100
        print(f"\nClassification rate: {classification_rate:.1f}%")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Classify TCP subjects by anhedonia severity based on SHAPS scores'
    )
    parser.add_argument('--dataset-path', type=Path,
                        help='Override dataset path')
    parser.add_argument('--base-subjects-dir', type=Path,
                        help='Override base subjects directory')
    parser.add_argument('--output-dir', type=Path,
                        help='Override output directory')

    args = parser.parse_args()

    print("TCP Anhedonia Classification")
    print("=" * 50)

    try:
        # Initialize pipeline
        pipeline = AnhedoniaClassificationPipeline(
            dataset_path=args.dataset_path,
            base_subjects_dir=args.base_subjects_dir,
            output_dir=args.output_dir
        )

        # Load input data
        pipeline.load_base_subjects()
        pipeline.load_phenotype_data()

        # Classify anhedonia levels
        classified_subjects, statistics = pipeline.classify_anhedonia_levels()

        # Add binary anhedonia status
        classified_subjects = pipeline.add_binary_anhedonia_status(classified_subjects)

        # Export results
        pipeline.export_results(classified_subjects, statistics)

        # Print summary
        pipeline.print_summary(statistics)

        print(f"\n{'=' * 60}")
        print(f"ANHEDONIA CLASSIFICATION COMPLETE")
        print(f"{'=' * 60}")
        print(f"Results saved to: {pipeline.output_dir}")
        print(f"\nNext steps:")
        print(f"  1. Run classify_diagnoses.py to classify subjects by MDD status")
        print(f"  2. Run generate_analysis_groups.py to create analysis-specific datasets")
        print(f"  3. Primary analysis group ready: anhedonic vs non-anhedonic subjects")

        return 0

    except Exception as e:
        print(f"{ERROR} Error during anhedonia classification: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
