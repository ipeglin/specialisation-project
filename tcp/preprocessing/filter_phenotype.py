#!/usr/bin/env python3
"""
TCP Phenotype Filtering Script

Filters subjects based on phenotype data (demographics, diagnoses, clinical measures).
This is an optional step that reduces subjects before downloading MRI data.

Author: Ian Philip Eglin
Date: 2025-09-23
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Add project root to path to import config
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.paths import get_script_output_path, get_tcp_dataset_path
from tcp.preprocessing.utils.phenotype_filters import (
    AgeRangeFilter, ColumnValueFilter, FilterAction, NonPrimaryDiagnosisFilter,
    PhenotypeFilter, PhenotypeFilterResult, PrimaryDiagnosisFilter,
    ShapsCompletionFilter)


class PhenotypeFilterPipeline:
    """Orchestrates multiple phenotype filters with dependency injection"""

    def __init__(self,
                 dataset_path: Optional[Path] = None,
                 subjects_input_dir: Optional[Path] = None,
                 output_dir: Optional[Path] = None):
        self.dataset_path = Path(dataset_path) if dataset_path else get_tcp_dataset_path()
        self.subjects_input_dir = Path(subjects_input_dir) if subjects_input_dir else get_script_output_path('tcp_preprocessing', 'validate_subjects')
        self.output_dir = Path(output_dir) if output_dir else get_script_output_path('tcp_preprocessing', 'filter_phenotype')

        self.filters: List[PhenotypeFilter] = []
        self.subjects_df: Optional[pd.DataFrame] = None
        self.phenotype_data: Dict[str, pd.DataFrame] = {}
        self.filtering_results: List[PhenotypeFilterResult] = []

        print(f"Dataset path: {self.dataset_path}")
        print(f"Subjects input: {self.subjects_input_dir}")
        print(f"Output directory: {self.output_dir}")

    def add_filter(self, filter_instance: PhenotypeFilter) -> 'PhenotypeFilterPipeline':
        """Add a filter to the pipeline (fluent interface)"""
        self.filters.append(filter_instance)
        print(f"Added filter: {filter_instance.filter_name}")
        return self

    def load_subjects_data(self) -> None:
        """Load subject data from validate_subjects.py output"""
        print("Loading subject validation data...")

        # Load valid subjects from validation step
        valid_subjects_file = self.subjects_input_dir / "valid_subjects.csv"

        if not valid_subjects_file.exists():
            raise FileNotFoundError(
                f"Valid subjects file not found: {valid_subjects_file}\n"
                f"Please run validate_subjects.py first."
            )

        self.subjects_df = pd.read_csv(valid_subjects_file)
        print(f"Loaded {len(self.subjects_df)} valid subjects")

        # Verify required columns
        if 'subject_id' not in self.subjects_df.columns:
            raise ValueError("subject_id column not found in subjects data")

    def load_phenotype_data(self) -> None:
        """Load phenotype data files from dataset"""
        print("Loading phenotype data...")

        # Define phenotype files to load
        phenotype_files = {
            'demos': 'phenotype/demos.tsv',
            'shaps01': 'phenotype/shaps01.tsv',
            'assessment': 'phenotype/assessment.tsv'
        }

        for file_key, file_path in phenotype_files.items():
            full_path = self.dataset_path / file_path

            if full_path.exists():
                try:
                    # Special handling for demos.tsv which is CSV format with metadata header
                    if file_key == 'demos':
                        # Skip first row (metadata) and use comma separator
                        try:
                            df = pd.read_csv(full_path, sep=',', encoding='utf-8', skiprows=1)
                        except UnicodeDecodeError:
                            df = pd.read_csv(full_path, sep=',', encoding='latin-1', skiprows=1)
                    else:
                        # Standard TSV handling for other files
                        try:
                            df = pd.read_csv(full_path, sep='\t', encoding='utf-8')
                        except UnicodeDecodeError:
                            df = pd.read_csv(full_path, sep='\t', encoding='latin-1')

                    self.phenotype_data[file_key] = df
                    print(f"  Loaded {file_key}: {df.shape[0]} rows, {df.shape[1]} columns")

                    # Show column info for debugging
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

    def apply_filters(self) -> None:
        """Apply all filters sequentially"""
        if self.subjects_df is None:
            raise ValueError("Subjects data must be loaded before applying filters")

        if not self.phenotype_data:
            raise ValueError("Phenotype data must be loaded before applying filters")

        print(f"\nApplying {len(self.filters)} phenotype filters...")

        current_subjects = self.subjects_df.copy()

        for i, filter_instance in enumerate(self.filters, 1):
            print(f"\n[{i}/{len(self.filters)}] Applying {filter_instance.filter_name}...")
            print(f"  Description: {filter_instance.description}")
            print(f"  Criteria: {filter_instance.get_criteria_description()}")

            try:
                # Apply filter
                filter_result = filter_instance.apply(current_subjects, self.phenotype_data)
                self.filtering_results.append(filter_result)

                # Update current subjects for next filter
                current_subjects = filter_result.included_subjects

                print(f"  Results: {len(filter_result.included_subjects)} included, "
                      f"{len(filter_result.excluded_subjects)} excluded")
                print(f"  Inclusion rate: {filter_result.statistics.get('inclusion_rate', 0)*100:.1f}%")

            except Exception as e:
                print(f"  ✗ Filter failed: {e}")
                raise

    def get_final_results(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get final included and excluded subjects"""
        if not self.filtering_results:
            raise ValueError("Filters must be applied before getting results")

        # Get results from last filter applied
        last_result = self.filtering_results[-1]
        return last_result.included_subjects, last_result.excluded_subjects

    def create_summary_report(self) -> Dict:
        """Create comprehensive summary report"""
        if not self.filtering_results:
            raise ValueError("No filtering results available")

        final_included, final_excluded = self.get_final_results()

        # Collect all inclusion/exclusion reasons
        all_inclusion_reasons = {}
        all_exclusion_reasons = {}

        for result in self.filtering_results:
            all_inclusion_reasons.update(result.inclusion_reasons)
            all_exclusion_reasons.update(result.exclusion_reasons)

        # Calculate statistics
        total_original = len(self.subjects_df)

        summary = {
            'timestamp': datetime.now().isoformat(),
            'dataset_path': str(self.dataset_path),
            'subjects_input_path': str(self.subjects_input_dir),
            'original_subjects': total_original,
            'final_included': len(final_included),
            'final_excluded': len(final_excluded),
            'overall_inclusion_rate': len(final_included) / total_original if total_original > 0 else 0,
            'filters_applied': [],
            'final_demographics': self._calculate_demographics_summary(final_included),
            'exclusion_reason_summary': self._summarize_exclusion_reasons(all_exclusion_reasons)
        }

        # Add detailed filter results
        for i, (filter_instance, result) in enumerate(zip(self.filters, self.filtering_results)):
            filter_summary = {
                'step': i + 1,
                'filter_name': filter_instance.filter_name,
                'description': filter_instance.description,
                'criteria': filter_instance.get_criteria_description(),
                'included_count': len(result.included_subjects),
                'excluded_count': len(result.excluded_subjects),
                'inclusion_rate': result.statistics.get('inclusion_rate', 0),
                'statistics': result.statistics
            }
            summary['filters_applied'].append(filter_summary)

        return summary

    def _calculate_demographics_summary(self, subjects_df: pd.DataFrame) -> Dict:
        """Calculate demographic summary of final included subjects"""
        if len(subjects_df) == 0:
            return {"note": "No subjects included"}

        # Try to get demographics from phenotype data
        demographics = {
            'total_subjects': len(subjects_df),
            'subject_count': len(subjects_df)
        }

        # If we have demos data, add demographic breakdown
        if 'demos' in self.phenotype_data:
            demos_df = self.phenotype_data['demos']

            # Merge with subjects to get demographics
            if 'participant_id' in demos_df.columns:
                # Handle subject ID conversion if needed
                subjects_for_merge = subjects_df.copy()
                if subjects_df['subject_id'].iloc[0].startswith('sub-'):
                    subjects_for_merge['participant_id_for_merge'] = subjects_df['subject_id'].str.replace('sub-', '')
                else:
                    subjects_for_merge['participant_id_for_merge'] = 'sub-' + subjects_df['subject_id']

                merged = subjects_for_merge.merge(
                    demos_df,
                    left_on='participant_id_for_merge',
                    right_on='participant_id',
                    how='left'
                )

                # Add demographic summaries
                if 'sex' in merged.columns:
                    demographics['sex_distribution'] = merged['sex'].value_counts(dropna=False).to_dict()

                if 'age' in merged.columns:
                    age_data = merged['age'].dropna()
                    if len(age_data) > 0:
                        demographics['age_statistics'] = {
                            'mean': float(age_data.mean()),
                            'median': float(age_data.median()),
                            'min': float(age_data.min()),
                            'max': float(age_data.max()),
                            'std': float(age_data.std())
                        }

                if 'Primary_Dx' in merged.columns:
                    demographics['diagnosis_distribution'] = merged['Primary_Dx'].value_counts(dropna=False).to_dict()

        return demographics

    def _summarize_exclusion_reasons(self, exclusion_reasons: Dict[str, str]) -> Dict:
        """Summarize exclusion reasons by type"""
        reason_counts = {}
        for reason in exclusion_reasons.values():
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        # Sort by frequency
        sorted_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)

        return {
            'total_excluded': len(exclusion_reasons),
            'reason_counts': dict(sorted_reasons),
            'top_reasons': dict(sorted_reasons[:5])  # Top 5 reasons
        }

    def export_results(self) -> Path:
        """Export filtering results to files"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nExporting results to {self.output_dir}")

        # Get final results
        final_included, final_excluded = self.get_final_results()

        # Export subject lists
        final_included.to_csv(self.output_dir / "phenotype_filtered_subjects.csv", index=False)
        final_excluded.to_csv(self.output_dir / "phenotype_excluded_subjects.csv", index=False)

        print(f"  ✓ Included subjects: {len(final_included)} subjects")
        print(f"  ✓ Excluded subjects: {len(final_excluded)} subjects")

        # Export detailed inclusion/exclusion reasons
        all_inclusion_reasons = {}
        all_exclusion_reasons = {}

        for result in self.filtering_results:
            all_inclusion_reasons.update(result.inclusion_reasons)
            all_exclusion_reasons.update(result.exclusion_reasons)

        reasons_data = {
            'inclusion_reasons': all_inclusion_reasons,
            'exclusion_reasons': all_exclusion_reasons
        }

        with open(self.output_dir / "filtering_reasons.json", 'w') as f:
            json.dump(reasons_data, f, indent=2)

        # Export summary report
        summary = self.create_summary_report()
        with open(self.output_dir / "phenotype_filtering_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"  ✓ Summary report: phenotype_filtering_summary.json")
        print(f"  ✓ Filtering reasons: filtering_reasons.json")

        return self.output_dir

    def print_summary(self) -> None:
        """Print filtering summary to console"""
        if not self.filtering_results:
            print("No filtering results available")
            return

        final_included, final_excluded = self.get_final_results()
        total_original = len(self.subjects_df)

        print(f"\n{'='*60}")
        print(f"PHENOTYPE FILTERING SUMMARY")
        print(f"{'='*60}")
        print(f"Original subjects: {total_original}")
        print(f"Final included: {len(final_included)} ({len(final_included)/total_original*100:.1f}%)")
        print(f"Final excluded: {len(final_excluded)} ({len(final_excluded)/total_original*100:.1f}%)")

        print(f"\nFilter-by-filter results:")
        for i, (filter_instance, result) in enumerate(zip(self.filters, self.filtering_results)):
            print(f"  {i+1}. {filter_instance.filter_name}: {len(result.included_subjects)} → {len(result.excluded_subjects)} excluded")
            print(f"     {result.criteria_description}")

        # Show final demographics if available
        demographics = self._calculate_demographics_summary(final_included)
        if 'sex_distribution' in demographics:
            print(f"\nFinal sample demographics:")
            print(f"  Sex distribution: {demographics['sex_distribution']}")

        if 'age_statistics' in demographics:
            age_stats = demographics['age_statistics']
            print(f"  Age: {age_stats['mean']:.1f} ± {age_stats['std']:.1f} years (range: {age_stats['min']:.1f}-{age_stats['max']:.1f})")

        if 'diagnosis_distribution' in demographics:
            print(f"  Diagnosis distribution: {demographics['diagnosis_distribution']}")

def create_default_mdd_pipeline() -> PhenotypeFilterPipeline:
    """Create a default pipeline for MDD vs control analysis"""
    pipeline = PhenotypeFilterPipeline()

    # Add primary diagnosis filter (MDD + controls)
    primary_dx_filter = PrimaryDiagnosisFilter(
        include_mdd=True,
        include_control=True
    )
    pipeline.add_filter(primary_dx_filter)

    # Add non-primary diagnosis filter (MDD) [OPTIONAL]
    # non_primary_dx_filter = NonPrimaryDiagnosisFilter()
    # pipeline.add_filter(non_primary_dx_filter)

    # Add SHAPS questionnaire completion filter
    shaps_completion_filter = ShapsCompletionFilter(
        exclude_incomplete=True
    )
    pipeline.add_filter(shaps_completion_filter)

    """ Testing
    # Add age range filter (adults only, reasonable age range)
    # interview_age is in months, so convert years to months
    age_filter = AgeRangeFilter(
        min_age=18*12,  # 18 years = 216 months
        max_age=65*12,  # 65 years = 780 months
        age_column='interview_age'
    )
    pipeline.add_filter(age_filter)
    """

    return pipeline

def create_custom_pipeline(args) -> PhenotypeFilterPipeline:
    """Create a custom pipeline based on command line arguments"""
    pipeline = PhenotypeFilterPipeline()

    # Primary diagnosis filter
    if args.include_mdd or args.include_controls:
        primary_dx_filter = PrimaryDiagnosisFilter(
            include_mdd=args.include_mdd,
            include_control=args.include_controls
        )
        pipeline.add_filter(primary_dx_filter)

    # Age filter
    if args.min_age is not None or args.max_age is not None:
        # Convert years to months for interview_age column
        min_age_months = args.min_age * 12 if args.min_age is not None else None
        max_age_months = args.max_age * 12 if args.max_age is not None else None
        age_filter = AgeRangeFilter(
            min_age=min_age_months,
            max_age=max_age_months,
            age_column='interview_age'
        )
        pipeline.add_filter(age_filter)

    # Custom column filters
    if args.custom_filters:
        for filter_spec in args.custom_filters:
            # Parse filter specification: file:column:value[:match_type]
            parts = filter_spec.split(':')
            if len(parts) < 3:
                raise ValueError(f"Invalid filter specification: {filter_spec}. "
                               f"Expected format: file:column:value[:match_type]")

            file_name, column_name, value = parts[:3]
            match_type = parts[3] if len(parts) > 3 else 'exact'

            custom_filter = ColumnValueFilter(
                phenotype_file=file_name,
                column_name=column_name,
                filter_values=value,
                match_type=match_type,
                case_sensitive=False,
                action=FilterAction.INCLUDE
            )
            pipeline.add_filter(custom_filter)

    return pipeline

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Filter TCP subjects based on phenotype data')
    parser.add_argument('--include-mdd', action='store_true', default=True,
                       help='Include subjects with MDD diagnosis')
    parser.add_argument('--include-controls', action='store_true', default=True,
                       help='Include control subjects (Primary_Dx=999)')
    parser.add_argument('--min-age', type=float,
                       help='Minimum age (inclusive)')
    parser.add_argument('--max-age', type=float,
                       help='Maximum age (inclusive)')
    parser.add_argument('--custom-filters', nargs='+',
                       help='Custom filters in format file:column:value[:match_type]')
    parser.add_argument('--dataset-path', type=Path,
                       help='Override dataset path')
    parser.add_argument('--subjects-input', type=Path,
                       help='Override subjects input directory')
    parser.add_argument('--output-dir', type=Path,
                       help='Override output directory')

    args = parser.parse_args()

    print("TCP Phenotype Filtering")
    print("=" * 50)

    try:
        # Create pipeline
        if args.custom_filters or args.min_age is not None or args.max_age is not None:
            pipeline = create_custom_pipeline(args)
        else:
            pipeline = create_default_mdd_pipeline()

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

        # Apply filters
        pipeline.apply_filters()

        # Export results
        output_dir = pipeline.export_results()

        # Print summary
        pipeline.print_summary()

        print(f"\n{'='*60}")
        print(f"PHENOTYPE FILTERING COMPLETE")
        print(f"{'='*60}")
        print(f"Results saved to: {output_dir}")
        print(f"\nNext steps:")
        print(f"  1. Run filter_subjects.py for task data filtering")
        print(f"  2. Run fetch_filtered_data.py to download MRI data")

        return 0

    except Exception as e:
        print(f"❌ Error during phenotype filtering: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
