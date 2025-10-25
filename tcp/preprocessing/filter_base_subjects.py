#!/usr/bin/env python3
"""
TCP Base Subject Filtering Script

Applies universal inclusion criteria that must be met by ALL subjects 
across all analysis groups:
1. Valid BIDS directory structure (from validate_subjects.py)
2. Has at least one task-based scan (hammer OR stroop)
3. Has completed SHAPS questionnaire (shaps_total ≠ 999)

This creates the foundational subject pool for all subsequent analysis groups.

Author: Ian Philip Eglin
Date: 2025-10-25
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

from config.paths import get_script_output_path, get_tcp_dataset_path
from tcp.preprocessing.utils.phenotype_filters import (
    PhenotypeFilter, PhenotypeFilterResult, ShapsCompletionFilter
)


class BaseSubjectFilterPipeline:
    """Pipeline for applying universal subject inclusion criteria"""

    def __init__(self,
                 dataset_path: Optional[Path] = None,
                 validated_subjects_dir: Optional[Path] = None,
                 task_filtered_subjects_dir: Optional[Path] = None,
                 output_dir: Optional[Path] = None):
        self.dataset_path = Path(dataset_path) if dataset_path else get_tcp_dataset_path()
        self.validated_subjects_dir = Path(validated_subjects_dir) if validated_subjects_dir else \
            get_script_output_path('tcp_preprocessing', 'validate_subjects')
        self.task_filtered_subjects_dir = Path(task_filtered_subjects_dir) if task_filtered_subjects_dir else \
            get_script_output_path('tcp_preprocessing', 'filter_subjects')
        self.output_dir = Path(output_dir) if output_dir else \
            get_script_output_path('tcp_preprocessing', 'filter_base_subjects')

        self.subjects_df: Optional[pd.DataFrame] = None
        self.phenotype_data: Dict[str, pd.DataFrame] = {}

        print(f"Dataset path: {self.dataset_path}")
        print(f"Validated subjects input: {self.validated_subjects_dir}")
        print(f"Task filtered subjects input: {self.task_filtered_subjects_dir}")
        print(f"Output directory: {self.output_dir}")

    def load_input_subjects(self) -> None:
        """Load subjects that have passed basic validation and task filtering"""
        print("Loading input subjects...")

        # Try to load from task_filtered_subjects first (if available)
        task_filtered_file = self.task_filtered_subjects_dir / "task_filtered_subjects.csv"
        validated_subjects_file = self.validated_subjects_dir / "valid_subjects.csv"

        if task_filtered_file.exists():
            print(f"  Loading from task-filtered subjects: {task_filtered_file}")
            self.subjects_df = pd.read_csv(task_filtered_file)
            source = "task_filtered"
        elif validated_subjects_file.exists():
            print(f"  Loading from validated subjects: {validated_subjects_file}")
            self.subjects_df = pd.read_csv(validated_subjects_file)
            source = "validated"
        else:
            raise FileNotFoundError(
                f"No input subjects found. Please run:\n"
                f"  1. validate_subjects.py (required)\n"
                f"  2. filter_subjects.py (recommended)\n"
                f"Expected files:\n"
                f"  - {task_filtered_file} (preferred)\n"
                f"  - {validated_subjects_file} (fallback)"
            )

        print(f"  Loaded {len(self.subjects_df)} subjects from {source} data")

        if 'subject_id' not in self.subjects_df.columns:
            raise ValueError("subject_id column not found in input subjects data")

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
                    print(f"  WARNING: Could not load {file_key} from {full_path}: {e}")
                    self.phenotype_data[file_key] = pd.DataFrame()
            else:
                print(f"  WARNING: Phenotype file not found: {full_path}")
                self.phenotype_data[file_key] = pd.DataFrame()

    def apply_shaps_completion_filter(self) -> PhenotypeFilterResult:
        """Apply SHAPS completion filter to exclude subjects with missing SHAPS data"""
        print("Applying SHAPS completion filter...")

        # Create SHAPS completion filter
        shaps_filter = ShapsCompletionFilter(exclude_incomplete=True)

        # Apply filter
        filter_result = shaps_filter.apply(
            subjects_df=self.subjects_df,
            phenotype_data=self.phenotype_data
        )

        # Print summary
        print(f"  SHAPS completion filter results:")
        print(f"    Included subjects: {len(filter_result.included_subjects)}")
        print(f"    Excluded subjects: {len(filter_result.excluded_subjects)}")
        print(f"    Criteria: {filter_result.criteria_description}")

        return filter_result

    def create_base_subject_pool(self) -> Tuple[pd.DataFrame, Dict]:
        """Create the base subject pool meeting all universal criteria"""
        print("\nCreating base subject pool...")

        # Start with input subjects (already validated and task-filtered)
        initial_count = len(self.subjects_df)
        print(f"  Starting with {initial_count} subjects from input data")

        # Apply SHAPS completion filter
        shaps_result = self.apply_shaps_completion_filter()
        
        # Use included subjects as our base pool
        base_subjects = shaps_result.included_subjects.copy()
        excluded_subjects = shaps_result.excluded_subjects.copy()

        # Create comprehensive statistics
        statistics = {
            'input_subjects': initial_count,
            'shaps_excluded': len(excluded_subjects),
            'final_base_subjects': len(base_subjects),
            'exclusion_breakdown': {}
        }
        
        # Add exclusion breakdown if there are excluded subjects
        if len(excluded_subjects) > 0 and 'exclusion_reason' in excluded_subjects.columns:
            statistics['exclusion_breakdown']['missing_shaps'] = len(
                excluded_subjects[excluded_subjects['exclusion_reason'].str.contains('SHAPS', na=False)]
            )
        else:
            statistics['exclusion_breakdown']['missing_shaps'] = 0

        print(f"\nBase subject pool creation summary:")
        print(f"  Input subjects: {statistics['input_subjects']}")
        print(f"  SHAPS exclusions: {statistics['shaps_excluded']}")
        print(f"  Final base pool: {statistics['final_base_subjects']}")

        return base_subjects, statistics

    def export_results(self, base_subjects: pd.DataFrame, statistics: Dict) -> None:
        """Export base subject filtering results"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nExporting results to {self.output_dir}")

        # 1. Export base filtered subjects
        base_subjects_file = self.output_dir / "base_filtered_subjects.csv"
        base_subjects.to_csv(base_subjects_file, index=False)
        print(f"  ✓ Base subjects: {base_subjects_file}")

        # 2. Export excluded subjects (from SHAPS filter)
        shaps_result = self.apply_shaps_completion_filter()
        if len(shaps_result.excluded_subjects) > 0:
            excluded_file = self.output_dir / "base_excluded_subjects.csv"
            shaps_result.excluded_subjects.to_csv(excluded_file, index=False)
            print(f"  ✓ Excluded subjects: {excluded_file}")

        # 3. Export filtering summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'dataset_path': str(self.dataset_path),
            'input_sources': {
                'validated_subjects': str(self.validated_subjects_dir),
                'task_filtered_subjects': str(self.task_filtered_subjects_dir)
            },
            'universal_criteria': [
                'Valid BIDS directory structure',
                'Has at least one task-based scan (hammer OR stroop)',
                'Has completed SHAPS questionnaire (shaps_total ≠ 999)'
            ],
            'statistics': statistics,
            'note': 'Base subject pool for all subsequent analysis groups'
        }

        summary_file = self.output_dir / "base_filtering_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  ✓ Summary: {summary_file}")

    def print_summary(self, statistics: Dict) -> None:
        """Print summary to console"""
        print(f"\n{'=' * 60}")
        print(f"BASE SUBJECT FILTERING SUMMARY")
        print(f"{'=' * 60}")
        
        print(f"\nUniversal inclusion criteria:")
        print(f"  ✓ Valid BIDS directory structure")
        print(f"  ✓ Has at least one task-based scan (hammer OR stroop)")
        print(f"  ✓ Has completed SHAPS questionnaire (shaps_total ≠ 999)")

        print(f"\nSubject flow:")
        print(f"  Input subjects: {statistics['input_subjects']}")
        print(f"  SHAPS exclusions: {statistics['shaps_excluded']}")
        print(f"  Final base pool: {statistics['final_base_subjects']}")

        retention_rate = (statistics['final_base_subjects'] / statistics['input_subjects']) * 100
        print(f"  Retention rate: {retention_rate:.1f}%")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Apply universal inclusion criteria for TCP subject analysis'
    )
    parser.add_argument('--dataset-path', type=Path,
                        help='Override dataset path')
    parser.add_argument('--validated-subjects-dir', type=Path,
                        help='Override validated subjects directory')
    parser.add_argument('--task-filtered-subjects-dir', type=Path,
                        help='Override task-filtered subjects directory')
    parser.add_argument('--output-dir', type=Path,
                        help='Override output directory')

    args = parser.parse_args()

    print("TCP Base Subject Filtering")
    print("=" * 50)

    try:
        # Initialize pipeline
        pipeline = BaseSubjectFilterPipeline(
            dataset_path=args.dataset_path,
            validated_subjects_dir=args.validated_subjects_dir,
            task_filtered_subjects_dir=args.task_filtered_subjects_dir,
            output_dir=args.output_dir
        )

        # Load input data
        pipeline.load_input_subjects()
        pipeline.load_phenotype_data()

        # Create base subject pool
        base_subjects, statistics = pipeline.create_base_subject_pool()

        # Export results
        pipeline.export_results(base_subjects, statistics)

        # Print summary
        pipeline.print_summary(statistics)

        print(f"\n{'=' * 60}")
        print(f"BASE FILTERING COMPLETE")
        print(f"{'=' * 60}")
        print(f"Results saved to: {pipeline.output_dir}")
        print(f"\nNext steps:")
        print(f"  1. Run classify_anhedonia.py to classify subjects by SHAPS scores")
        print(f"  2. Run classify_diagnoses.py to classify subjects by MDD status")
        print(f"  3. Run generate_analysis_groups.py to create analysis-specific datasets")

        return 0

    except Exception as e:
        print(f"❌ Error during base subject filtering: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())