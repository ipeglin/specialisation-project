#!/usr/bin/env python3
"""
TCP SHAPS Questionnaire Score Segmentation Script

Segmenting subjects into classes based on phenotype data on their total SHAPS questionnaire scores. All subjects have valid SHAPS scores, seeing at a prerequisite for this pipeline stage is filter_phenotype.py which uses the `ShapsCompletionFilter` in the default pipeline in `create_default_mdd_pipeline()`.

Classes:
- non-anhedonic: 0 <= score <= 2
- low-anhedonic: 3 <= score <= 8  
- high-anhedonic: 9 <= score <= 14

Author: Ian Philip Eglin
Date: 2025-10-20
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import json
from datetime import datetime
import argparse

# Add project root to path to import config
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.paths import get_script_output_path, get_tcp_dataset_path
from tcp.preprocessing.utils.phenotype_filters import (
    PhenotypeFilter, PhenotypeFilterResult, AnhedoniaSegmentationFilter
)


class AnhedoniaSegmentationPipeline:
    """Pipeline for anhedonia segmentation with dependency injection"""

    def __init__(self,
                 dataset_path: Optional[Path] = None,
                 subjects_input_dir: Optional[Path] = None,
                 output_dir: Optional[Path] = None):
        self.dataset_path = Path(dataset_path) if dataset_path else get_tcp_dataset_path()
        self.subjects_input_dir = Path(subjects_input_dir) if subjects_input_dir else get_script_output_path('tcp_preprocessing', 'filter_subjects')
        self.output_dir = Path(output_dir) if output_dir else get_script_output_path('tcp_preprocessing', 'anhedonia_segmentation')

        self.segmentation_filter: Optional[AnhedoniaSegmentationFilter] = None
        self.subjects_df: Optional[pd.DataFrame] = None
        self.phenotype_data: Dict[str, pd.DataFrame] = {}
        self.segmentation_result: Optional[PhenotypeFilterResult] = None

        print(f"Dataset path: {self.dataset_path}")
        print(f"Subjects input: {self.subjects_input_dir}")
        print(f"Output directory: {self.output_dir}")

    def set_segmentation_filter(self, filter_instance: AnhedoniaSegmentationFilter) -> 'AnhedoniaSegmentationPipeline':
        """Set the segmentation filter (fluent interface)"""
        self.segmentation_filter = filter_instance
        print(f"Set segmentation filter: {filter_instance.filter_name}")
        return self

    def load_subjects_data(self) -> None:
        """Load subject data from filter_subjects.py output"""
        print("Loading task-filtered subject data...")

        filtered_subjects_file = self.subjects_input_dir / "task_filtered_subjects.csv"

        if not filtered_subjects_file.exists():
            raise FileNotFoundError(
                f"Filtered subjects file not found: {filtered_subjects_file}\n"
                f"Please run filter_subjects.py first."
            )

        self.subjects_df = pd.read_csv(filtered_subjects_file)
        print(f"Loaded {len(self.subjects_df)} filtered subjects")

        if 'subject_id' not in self.subjects_df.columns:
            raise ValueError("subject_id column not found in subjects data")

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

                    if len(df.columns) <= 10:
                        print(f"    Columns: {list(df.columns)}")
                    else:
                        print(f"    Columns: {list(df.columns[:5])} ... (and {len(df.columns)-5} more)")

                except Exception as e:
                    print(f"  ✗ Error loading {file_key}: {e}")
            else:
                print(f"  ⚠ File not found: {file_path}")

        if not self.phenotype_data:
            raise FileNotFoundError(
                "No phenotype data files could be loaded. "
                "Please run fetch_global_data.py first to download phenotype files."
            )

    def apply_segmentation(self) -> None:
        """Apply anhedonia segmentation"""
        if self.subjects_df is None:
            raise ValueError("Subjects data must be loaded before applying segmentation")

        if not self.phenotype_data:
            raise ValueError("Phenotype data must be loaded before applying segmentation")

        if self.segmentation_filter is None:
            raise ValueError("Segmentation filter must be set before applying segmentation")

        print(f"\nApplying anhedonia segmentation...")
        print(f"  Description: {self.segmentation_filter.description}")
        print(f"  Criteria: {self.segmentation_filter.get_criteria_description()}")

        try:
            self.segmentation_result = self.segmentation_filter.apply(self.subjects_df, self.phenotype_data)

            print(f"  Results: {len(self.segmentation_result.included_subjects)} subjects with valid classifications")
            print(f"  Excluded: {len(self.segmentation_result.excluded_subjects)} subjects with invalid/missing scores")
            print(f"  Success rate: {self.segmentation_result.statistics.get('inclusion_rate', 0)*100:.1f}%")

            if 'classification_distribution' in self.segmentation_result.statistics:
                class_dist = self.segmentation_result.statistics['classification_distribution']
                print(f"  Classification distribution: {class_dist}")

        except Exception as e:
            print(f"  ✗ Segmentation failed: {e}")
            raise

    def get_segmented_subjects(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get segmented subjects and excluded subjects"""
        if self.segmentation_result is None:
            raise ValueError("Segmentation must be applied before getting results")

        return self.segmentation_result.included_subjects, self.segmentation_result.excluded_subjects

    def create_summary_report(self) -> Dict:
        """Create comprehensive summary report"""
        if self.segmentation_result is None:
            raise ValueError("No segmentation results available")

        segmented_subjects, excluded_subjects = self.get_segmented_subjects()

        summary = {
            'timestamp': datetime.now().isoformat(),
            'dataset_path': str(self.dataset_path),
            'subjects_input_path': str(self.subjects_input_dir),
            'original_subjects': len(self.subjects_df),
            'segmented_subjects': len(segmented_subjects),
            'excluded_subjects': len(excluded_subjects),
            'segmentation_success_rate': len(segmented_subjects) / len(self.subjects_df) if len(self.subjects_df) > 0 else 0,
            'filter_applied': {
                'name': self.segmentation_filter.filter_name,
                'description': self.segmentation_filter.description,
                'criteria': self.segmentation_filter.get_criteria_description()
            },
            'classification_statistics': self.segmentation_result.statistics,
            'class_breakdown': self._calculate_class_breakdown(segmented_subjects)
        }

        return summary

    def _calculate_class_breakdown(self, segmented_subjects: pd.DataFrame) -> Dict:
        """Calculate detailed breakdown by anhedonia class"""
        if len(segmented_subjects) == 0:
            return {"note": "No subjects segmented"}

        breakdown = {}
        
        for anhedonia_class in ['non-anhedonic', 'low-anhedonic', 'high-anhedonic']:
            class_subjects = segmented_subjects[segmented_subjects['anhedonia_class'] == anhedonia_class]
            count = len(class_subjects)
            percentage = (count / len(segmented_subjects)) * 100 if len(segmented_subjects) > 0 else 0
            
            breakdown[anhedonia_class] = {
                'count': count,
                'percentage': percentage,
                'subject_ids': class_subjects['subject_id'].tolist()[:10]  # First 10 for reference
            }

        return breakdown

    def export_results(self) -> Path:
        """Export segmentation results to files"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nExporting results to {self.output_dir}")

        segmented_subjects, excluded_subjects = self.get_segmented_subjects()

        # Export segmented subjects with anhedonia classifications
        segmented_subjects.to_csv(self.output_dir / "anhedonia_segmented_subjects.csv", index=False)
        excluded_subjects.to_csv(self.output_dir / "anhedonia_excluded_subjects.csv", index=False)

        print(f"  ✓ Segmented subjects: {len(segmented_subjects)} subjects")
        print(f"  ✓ Excluded subjects: {len(excluded_subjects)} subjects")

        # Export by class
        for anhedonia_class in ['non-anhedonic', 'low-anhedonic', 'high-anhedonic']:
            class_subjects = segmented_subjects[segmented_subjects['anhedonia_class'] == anhedonia_class]
            if len(class_subjects) > 0:
                class_file = self.output_dir / f"{anhedonia_class.replace('-', '_')}_subjects.csv"
                class_subjects.to_csv(class_file, index=False)
                print(f"  ✓ {anhedonia_class} subjects: {len(class_subjects)} subjects")

        # Export detailed inclusion/exclusion reasons
        reasons_data = {
            'inclusion_reasons': self.segmentation_result.inclusion_reasons,
            'exclusion_reasons': self.segmentation_result.exclusion_reasons
        }

        with open(self.output_dir / "segmentation_reasons.json", 'w') as f:
            json.dump(reasons_data, f, indent=2)

        # Export summary report
        summary = self.create_summary_report()
        with open(self.output_dir / "anhedonia_segmentation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"  ✓ Summary report: anhedonia_segmentation_summary.json")
        print(f"  ✓ Segmentation reasons: segmentation_reasons.json")

        return self.output_dir

    def print_summary(self) -> None:
        """Print segmentation summary to console"""
        if self.segmentation_result is None:
            print("No segmentation results available")
            return

        segmented_subjects, excluded_subjects = self.get_segmented_subjects()
        total_original = len(self.subjects_df)

        print(f"\n{'='*60}")
        print(f"ANHEDONIA SEGMENTATION SUMMARY")
        print(f"{'='*60}")
        print(f"Original subjects: {total_original}")
        print(f"Successfully segmented: {len(segmented_subjects)} ({len(segmented_subjects)/total_original*100:.1f}%)")
        print(f"Excluded (invalid/missing): {len(excluded_subjects)} ({len(excluded_subjects)/total_original*100:.1f}%)")

        # Show classification breakdown
        if 'classification_distribution' in self.segmentation_result.statistics:
            class_dist = self.segmentation_result.statistics['classification_distribution']
            print(f"\nAnhedonia classification breakdown:")
            for class_name, count in class_dist.items():
                percentage = (count / len(segmented_subjects)) * 100 if len(segmented_subjects) > 0 else 0
                print(f"  {class_name}: {count} subjects ({percentage:.1f}%)")

        # Show score statistics
        if 'score_statistics' in self.segmentation_result.statistics:
            score_stats = self.segmentation_result.statistics['score_statistics']
            if score_stats['mean_score'] is not None:
                print(f"\nSHAPS score statistics:")
                print(f"  Mean: {score_stats['mean_score']:.2f} ± {score_stats['std_score']:.2f}")
                print(f"  Range: {score_stats['min_score']:.0f}-{score_stats['max_score']:.0f}")


def create_default_anhedonia_pipeline() -> AnhedoniaSegmentationPipeline:
    """Create a default pipeline for anhedonia segmentation"""
    pipeline = AnhedoniaSegmentationPipeline()

    # Add anhedonia segmentation filter
    anhedonia_filter = AnhedoniaSegmentationFilter(
        score_column='shaps_total',
        phenotype_file='shaps01'
    )
    pipeline.set_segmentation_filter(anhedonia_filter)

    return pipeline


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Segment TCP subjects based on anhedonia scores')
    parser.add_argument('--dataset-path', type=Path,
                       help='Override dataset path')
    parser.add_argument('--subjects-input', type=Path,
                       help='Override subjects input directory (filtered subjects from filter_subjects.py)')
    parser.add_argument('--output-dir', type=Path,
                       help='Override output directory')

    args = parser.parse_args()

    print("TCP Anhedonia Segmentation")
    print("=" * 50)

    try:
        # Create pipeline
        pipeline = create_default_anhedonia_pipeline()

        # Override paths if specified
        if args.dataset_path:
            pipeline.dataset_path = args.dataset_path
        if args.subjects_input:
            pipeline.subjects_input_dir = args.subjects_input
        if args.output_dir:
            pipeline.output_dir = args.output_dir

        # Load data
        pipeline.load_subjects_data()
        pipeline.load_phenotype_data()

        # Apply segmentation
        pipeline.apply_segmentation()

        # Export results
        output_dir = pipeline.export_results()

        # Print summary
        pipeline.print_summary()

        print(f"\n{'='*60}")
        print(f"ANHEDONIA SEGMENTATION COMPLETE")
        print(f"{'='*60}")
        print(f"Results saved to: {output_dir}")
        print(f"\nNext steps:")
        print(f"  1. Use segmented subjects for downstream analysis")
        print(f"  2. Compare anhedonia classes in your research")

        return 0

    except Exception as e:
        print(f"❌ Error during anhedonia segmentation: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())