#!/usr/bin/env python3
"""
Robust subject filtering for TCP dataset preprocessing.

Filters subjects missing task data (hammer/stroop) using a flexible,
extensible architecture with dependency injection. Maintains data integrity
by separating included/excluded subjects rather than deleting data.

Author: Ian Philip Eglin
Date: 2025-09-12
"""

import sys

from pathlib import Path

# Add project root to path to import config
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.paths import get_script_output_path
from tcp.preprocessing.utils.filter_pipeline import SubjectFilterPipeline
from tcp.preprocessing.utils.subject_filters import TaskAvailabilityFilter

def main():
    """Main execution function"""
    extracted_data_path = get_script_output_path('tcp_preprocessing', 'extract_subjects')
    output_dir = get_script_output_path('tcp_preprocessing', 'filter_subjects')

    print(f"TCP Subject Filtering Pipeline")
    print(f"==============================")
    print(f"Input data: {extracted_data_path}")
    print(f"Output directory: {output_dir}")

    # Initialize filtering pipeline
    pipeline = SubjectFilterPipeline(str(extracted_data_path), str(output_dir))

    # Configure task availability filter
    # Filter subjects that have AT LEAST ONE of hammer or stroop tasks
    # Check across all data types (raw_nifti, timeseries, events) to ensure no subjects are missed
    task_filter = TaskAvailabilityFilter(
        required_tasks=['hammer', 'stroop'],
        require_all_tasks=False,  # Require at least one task, not both
        data_types=['raw_nifti', 'timeseries', 'events']
    )

    # Add filters to pipeline
    pipeline.add_filter(task_filter)

    # Load and process data
    pipeline.load_extracted_data()
    pipeline.apply_filters()

    # Export results
    output_path = pipeline.export_results()

    # Print summary
    included_patients, excluded_patients, included_controls, excluded_controls = pipeline.get_final_results()

    print(f"\n=== FILTERING COMPLETE ===")
    print(f"Output saved to: {output_path}")
    print(f"\nFinal Results:")
    print(f"  Included: {len(included_patients)} patients, {len(included_controls)} controls")
    print(f"  Excluded: {len(excluded_patients)} patients, {len(excluded_controls)} controls")
    print(f"  Inclusion Rate: {(len(included_patients) + len(included_controls)) / (len(included_patients) + len(excluded_patients) + len(included_controls) + len(excluded_controls)) * 100:.1f}%")

    # Show task data summary for included subjects
    if len(included_patients) > 0:
        hammer_patients = included_patients['has_hammer_raw'].sum() if 'has_hammer_raw' in included_patients.columns else 0
        stroop_patients = included_patients['has_stroop_raw'].sum() if 'has_stroop_raw' in included_patients.columns else 0
        print(f"\nIncluded Patients Task Data:")
        print(f"  - With Hammer task: {hammer_patients}")
        print(f"  - With Stroop task: {stroop_patients}")

    if len(included_controls) > 0:
        hammer_controls = included_controls['has_hammer_raw'].sum() if 'has_hammer_raw' in included_controls.columns else 0
        stroop_controls = included_controls['has_stroop_raw'].sum() if 'has_stroop_raw' in included_controls.columns else 0
        print(f"\nIncluded Controls Task Data:")
        print(f"  - With Hammer task: {hammer_controls}")
        print(f"  - With Stroop task: {stroop_controls}")

    print(f"\nNext steps:")
    print(f"  - Review filtering_report.json for detailed statistics")
    print(f"  - Use included/ directory for downstream pipeline processing")
    print(f"  - Refer to excluded/ directory if you need to analyze filtered-out subjects")

    return pipeline

if __name__ == "__main__":
    results = main()
