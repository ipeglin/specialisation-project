#!/usr/bin/env python3
"""
Robust subject filtering for TCP dataset preprocessing.

Filters subjects missing task data (hammer/stroop) using a flexible,
extensible architecture with dependency injection. Maintains data integrity
by separating included/excluded subjects rather than deleting data.

Updated to work with new pipeline: automatically detects input from either
phenotype filtering or subject validation steps.

Author: Ian Philip Eglin
Date: 2025-09-23
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add project root to path to import config
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from config.paths import get_script_output_path, get_tcp_dataset_path
from tcp.preprocessing.config.data_source_config import DataSourceConfig, DataSourceType, create_datalad_config
from tcp.preprocessing.utils.filter_pipeline import SubjectFilterPipeline
from tcp.preprocessing.utils.subject_filters import TaskAvailabilityFilter
from tcp.preprocessing.utils.unicode_compat import CHECK, CROSS, ERROR


def detect_input_source() -> Path:
    """Automatically detect input source from new pipeline steps"""
    
    # Option 1: Use validated subjects (standard input for new pipeline)
    
    # Use valid subjects from validation step (standard in new pipeline)
    validation_dir = get_script_output_path('tcp_preprocessing', 'validate_subjects')
    validation_subjects_file = validation_dir / 'valid_subjects.csv'

    if validation_subjects_file.exists():
        print(f"{CHECK} Found validated subjects: {validation_dir}")
        return validation_dir
    
    # No valid input found
    raise FileNotFoundError(
        "No valid input data found. Please run validate_subjects.py first to create validated subjects."
    )

class NewSubjectFilterPipeline:
    """Updated pipeline that works with new data structure and supports multiple data sources"""

    def __init__(self, input_dir: Path, output_dir: Path, dataset_path: Path,
                 data_source_config: Optional[DataSourceConfig] = None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.dataset_path = Path(dataset_path)
        self.data_source_config = data_source_config or create_datalad_config(dataset_path)
        self.filters = []

        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Dataset path: {self.dataset_path}")
        print(f"Data source: {self.data_source_config.source_type.value}")
    
    def add_filter(self, filter_instance):
        """Add a filter to the pipeline"""
        self.filters.append(filter_instance)
        print(f"Added filter: {filter_instance.filter_name}")
        return self
    
    def load_and_convert_data(self):
        """Load data from new pipeline format - unified subject list (group-agnostic)"""
        import json
        from pathlib import Path

        import pandas as pd

        # Load datalad subjects from validation/phenotype files
        phenotype_file = self.input_dir / 'phenotype_filtered_subjects.csv'
        validation_file = self.input_dir / 'valid_subjects.csv'

        datalad_subjects_df = None

        if phenotype_file.exists():
            # Load from phenotype filtering
            print("Loading phenotype filtered subjects (datalad)...")
            datalad_subjects_df = pd.read_csv(phenotype_file)
            print(f"  Loaded {len(datalad_subjects_df)} phenotype-filtered subjects")

        elif validation_file.exists():
            # Load from validation step
            print("Loading validated subjects (datalad)...")
            datalad_subjects_df = pd.read_csv(validation_file)
            print(f"  Loaded {len(datalad_subjects_df)} validated subjects")

        else:
            # Try legacy format (backward compatibility)
            patient_file = self.input_dir / 'patient_subjects.csv'
            control_file = self.input_dir / 'control_subjects.csv'

            if patient_file.exists() and control_file.exists():
                print("Loading legacy extract_subjects format...")
                patients_df = pd.read_csv(patient_file)
                controls_df = pd.read_csv(control_file)
                # Combine into unified list
                datalad_subjects_df = pd.concat([patients_df, controls_df], ignore_index=True)
                print(f"  Loaded {len(datalad_subjects_df)} subjects ({len(patients_df)} patients + {len(controls_df)} controls)")
            else:
                raise FileNotFoundError(f"No valid subject files found in {self.input_dir}")

        # If HCP is enabled, discover HCP subjects and merge
        if self.data_source_config.is_hcp_enabled():
            hcp_subjects_df = self._discover_hcp_subjects()

            if self.data_source_config.is_combined_mode():
                # COMBINED mode: merge datalad and HCP subjects
                print("\nCOMBINED mode: Merging datalad and HCP subjects...")
                subjects_df = self._merge_subject_sources(datalad_subjects_df, hcp_subjects_df)
            else:
                # HCP-only mode
                print("\nHCP-only mode: Using only HCP subjects...")
                subjects_df = hcp_subjects_df
                if 'data_source' not in subjects_df.columns:
                    subjects_df['data_source'] = 'hcp'
        else:
            # DATALAD-only mode (default)
            subjects_df = datalad_subjects_df
            if 'data_source' not in subjects_df.columns:
                subjects_df['data_source'] = 'datalad'

        # Generate task file paths by scanning the dataset
        # Note: This currently only scans datalad dataset
        # HCP files will be mapped later in map_subject_files.py
        print("\nGenerating task file paths...")
        file_paths = self._generate_task_file_paths(subjects_df)

        return subjects_df, file_paths
    
    def _generate_task_file_paths(self, subjects_df):
        """Generate task file paths by scanning dataset structure (unified subject list)"""
        import glob

        # Get all subject IDs from unified list
        if 'subject_id' not in subjects_df.columns:
            raise ValueError("subject_id column not found in subjects data")

        all_subjects = subjects_df['subject_id'].tolist()
        
        file_paths = {
            'raw_nifti': {},
            'timeseries': {},
            'events': {}
        }
        
        print(f"  Scanning {len(all_subjects)} subjects for task files...")
        
        # Check for timeseries data path
        timeseries_path = self.dataset_path / "fMRI_timeseries_clean_denoised_GSR_parcellated"
        
        for i, subject_id in enumerate(all_subjects):
            if (i + 1) % 50 == 0 or (i + 1) == len(all_subjects):
                print(f"    Progress: {i+1}/{len(all_subjects)} subjects")
            
            # Initialize file paths for this subject
            file_paths['raw_nifti'][subject_id] = {'hammer': [], 'stroop': []}
            file_paths['timeseries'][subject_id] = {'hammer': [], 'stroop': []}
            file_paths['events'][subject_id] = {'hammer': [], 'stroop': []}
            
            # Check for raw NIFTI files
            subj_func_dir = self.dataset_path / subject_id / "func"
            if subj_func_dir.exists():
                # Hammer task files
                hammer_files = list(subj_func_dir.glob("*task-hammer*_bold.nii.gz"))
                file_paths['raw_nifti'][subject_id]['hammer'] = [str(f) for f in hammer_files]
                
                # Stroop task files
                stroop_files = list(subj_func_dir.glob("*task-stroop*_bold.nii.gz"))
                file_paths['raw_nifti'][subject_id]['stroop'] = [str(f) for f in stroop_files]
                
                # Event files
                hammer_events = list(subj_func_dir.glob("*task-hammer*_events.tsv"))
                stroop_events = list(subj_func_dir.glob("*task-stroop*_events.tsv"))
                file_paths['events'][subject_id]['hammer'] = [str(f) for f in hammer_events]
                file_paths['events'][subject_id]['stroop'] = [str(f) for f in stroop_events]
            
            # Check for timeseries files
            if timeseries_path.exists():
                # Convert BIDS ID to timeseries ID format
                if subject_id.startswith('sub-NDAR'):
                    timeseries_id = subject_id.replace('sub-NDAR', 'NDAR_').replace('INV', 'INV')
                    ts_subj_dir = timeseries_path / timeseries_id
                    
                    if ts_subj_dir.exists():
                        hammer_ts = list(ts_subj_dir.glob("*hammer*.h5"))
                        stroop_ts = list(ts_subj_dir.glob("*stroop*.h5"))
                        file_paths['timeseries'][subject_id]['hammer'] = [str(f) for f in hammer_ts]
                        file_paths['timeseries'][subject_id]['stroop'] = [str(f) for f in stroop_ts]
        
        print(f"  Task file scanning complete")
        return file_paths

    def _discover_hcp_subjects(self) -> pd.DataFrame:
        """
        Scan HCP directory for subjects with task data.

        Returns:
            DataFrame with columns: subject_id, data_source='hcp'
        """
        if not self.data_source_config.is_hcp_enabled():
            return pd.DataFrame(columns=['subject_id', 'data_source'])

        print("Discovering HCP subjects...")
        hcp_subjects = self.data_source_config.discover_hcp_subjects()

        if not hcp_subjects:
            print("  No HCP subjects found")
            return pd.DataFrame(columns=['subject_id', 'data_source'])

        print(f"  Found {len(hcp_subjects)} HCP subjects with {self.data_source_config.default_task} task data")

        return pd.DataFrame([
            {'subject_id': subj, 'data_source': 'hcp'}
            for subj in hcp_subjects
        ])

    def _merge_subject_sources(self, datalad_df: pd.DataFrame,
                              hcp_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge datalad and HCP subject lists with duplicate resolution.

        Args:
            datalad_df: DataFrame of datalad subjects
            hcp_df: DataFrame of HCP subjects

        Returns:
            Combined DataFrame with data_source column
        """
        # Tag sources
        datalad_df = datalad_df.copy()
        hcp_df = hcp_df.copy()
        datalad_df['data_source'] = 'datalad'
        if 'data_source' not in hcp_df.columns:
            hcp_df['data_source'] = 'hcp'

        # Find duplicates
        duplicates = set(datalad_df['subject_id']) & set(hcp_df['subject_id'])

        if duplicates:
            strategy = self.data_source_config.duplicate_resolution
            print(f"WARNING: {len(duplicates)} subjects found in both sources")
            print(f"  Duplicate resolution strategy: {strategy}")
            print(f"  Duplicate subject IDs: {sorted(list(duplicates))[:5]}{'...' if len(duplicates) > 5 else ''}")

            if strategy == "error":
                raise ValueError(
                    f"Duplicate subjects found in both datalad and HCP: {duplicates}\n"
                    f"Set --duplicate-resolution to 'prefer_hcp' or 'prefer_datalad' to resolve."
                )
            elif strategy == "prefer_hcp":
                # Keep HCP fMRI data, remove datalad fMRI data for these subjects
                print(f"  Keeping HCP fMRI data for {len(duplicates)} subjects")
                datalad_df = datalad_df[~datalad_df['subject_id'].isin(duplicates)]
            elif strategy == "prefer_datalad":
                # Keep datalad fMRI data, remove HCP fMRI data for these subjects
                print(f"  Keeping datalad fMRI data for {len(duplicates)} subjects")
                hcp_df = hcp_df[~hcp_df['subject_id'].isin(duplicates)]

        # Merge
        combined_df = pd.concat([datalad_df, hcp_df], ignore_index=True)
        print(f"Combined subject list: {len(datalad_df)} datalad + {len(hcp_df)} HCP = {len(combined_df)} total")

        return combined_df

    def apply_filters(self, subjects_df, file_paths):
        """Apply filters to unified subject list (group-agnostic)"""
        import json
        from datetime import datetime

        print(f"\nApplying {len(self.filters)} filters to {len(subjects_df)} subjects...")

        # Ensure participant_id column exists for filter compatibility
        if 'subject_id' in subjects_df.columns and 'participant_id' not in subjects_df.columns:
            subjects_df['participant_id'] = subjects_df['subject_id']

        # Apply each filter sequentially
        current_subjects = subjects_df.copy()
        all_inclusion_reasons = {}
        all_exclusion_reasons = {}
        filter_statistics = []

        for i, filter_instance in enumerate(self.filters, 1):
            print(f"\n[Filter {i}/{len(self.filters)}] {filter_instance.filter_name}")
            print(f"  Criteria: {filter_instance.get_criteria_description()}")

            # Apply filter
            included_subjects, excluded_subjects, inclusion_reasons, exclusion_reasons = \
                filter_instance.apply(current_subjects, file_paths)

            # Update tracking
            all_inclusion_reasons.update(inclusion_reasons)
            all_exclusion_reasons.update(exclusion_reasons)

            # Track filter statistics
            filter_stat = {
                'filter_name': filter_instance.filter_name,
                'criteria': filter_instance.get_criteria_description(),
                'input_count': len(current_subjects),
                'included_count': len(included_subjects),
                'excluded_count': len(excluded_subjects),
                'inclusion_rate': len(included_subjects) / len(current_subjects) if len(current_subjects) > 0 else 0
            }
            filter_statistics.append(filter_stat)

            print(f"  Results: {len(included_subjects)} included, {len(excluded_subjects)} excluded")
            print(f"  Inclusion rate: {filter_stat['inclusion_rate']*100:.1f}%")

            # Update current subjects for next filter
            current_subjects = included_subjects

        # Final results
        included_subjects = current_subjects
        excluded_subject_ids = set(subjects_df['subject_id']) - set(included_subjects['subject_id'])
        excluded_subjects = subjects_df[subjects_df['subject_id'].isin(excluded_subject_ids)]

        # Export results
        self._export_unified_results(
            included_subjects, excluded_subjects,
            all_inclusion_reasons, all_exclusion_reasons,
            filter_statistics, subjects_df
        )

        return included_subjects, excluded_subjects

    def _export_unified_results(self, included_subjects, excluded_subjects,
                               inclusion_reasons, exclusion_reasons,
                               filter_statistics, original_subjects):
        """Export unified filtering results (phenotype filtering style)"""
        import json
        from datetime import datetime

        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nExporting results to {self.output_dir}")

        # Export subject lists
        included_subjects.to_csv(self.output_dir / "task_filtered_subjects.csv", index=False)
        excluded_subjects.to_csv(self.output_dir / "task_excluded_subjects.csv", index=False)
        print(f"  {CHECK} Included subjects: {len(included_subjects)}")
        print(f"  {CHECK} Excluded subjects: {len(excluded_subjects)}")

        # Export inclusion/exclusion reasons
        reasons_data = {
            'inclusion_reasons': inclusion_reasons,
            'exclusion_reasons': exclusion_reasons
        }
        with open(self.output_dir / "filtering_reasons.json", 'w') as f:
            json.dump(reasons_data, f, indent=2)
        print(f"  {CHECK} Filtering reasons exported")

        # Create summary report
        summary = {
            'timestamp': datetime.now().isoformat(),
            'dataset_path': str(self.dataset_path),
            'input_directory': str(self.input_dir),
            'total_input_subjects': len(original_subjects),
            'total_included_subjects': len(included_subjects),
            'total_excluded_subjects': len(excluded_subjects),
            'overall_inclusion_rate': len(included_subjects) / len(original_subjects) if len(original_subjects) > 0 else 0,
            'filters_applied': filter_statistics,
            'note': 'Filtering is group-agnostic. Use classify_diagnoses.py and integrate_cross_analysis.py for patient/control analysis.'
        }

        with open(self.output_dir / "task_filtering_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  {CHECK} Summary report: task_filtering_summary.json")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Filter TCP subjects based on task data availability')
    parser.add_argument('--input-dir', type=Path,
                       help='Override input directory (auto-detected by default)')
    parser.add_argument('--output-dir', type=Path,
                       help='Override output directory')
    parser.add_argument('--dataset-path', type=Path,
                       help='Override dataset path')
    parser.add_argument('--require-all-tasks', action='store_true',
                       help='Require both hammer AND stroop tasks (default: at least one)')
    parser.add_argument('--tasks', nargs='+', default=['hammer', 'stroop'],
                       help='Tasks to filter for (default: hammer stroop)')
    parser.add_argument('--data-types', nargs='+', default=['raw_nifti', 'timeseries', 'events'],
                       help='Data types to check (default: raw_nifti timeseries events)')

    # Data source configuration
    parser.add_argument('--data-source-type', choices=['datalad', 'hcp', 'combined'],
                       default='datalad',
                       help='Data source type: datalad (default), hcp, or combined')
    parser.add_argument('--hcp-root', type=Path,
                       help='Path to HCP output directory (required for hcp/combined modes)')
    parser.add_argument('--hcp-parcellated-output', type=Path,
                       help='Directory to store parcellated HCP .h5 files (required for hcp/combined modes)')
    parser.add_argument('--duplicate-resolution', choices=['prefer_hcp', 'prefer_datalad', 'error'],
                       default='prefer_hcp',
                       help='How to handle subjects in both datalad and HCP (combined mode only, default: prefer_hcp)')
    parser.add_argument('--default-task', type=str, default='hammer',
                       help='Default task name for HCP data discovery (default: hammer)')

    args = parser.parse_args()
    
    print(f"TCP Subject Task Data Filtering")
    print(f"===============================")
    
    # Determine input source
    if args.input_dir:
        input_dir = args.input_dir
        print(f"Using specified input: {input_dir}")
    else:
        input_dir = detect_input_source()
    
    # Set paths
    output_dir = args.output_dir or get_script_output_path('tcp_preprocessing', 'filter_subjects')
    dataset_path = args.dataset_path or get_tcp_dataset_path()

    print(f"Output directory: {output_dir}")

    # Create data source configuration
    if args.data_source_type == 'datalad':
        data_source_config = create_datalad_config(
            dataset_path=dataset_path,
            default_task=args.default_task
        )
    elif args.data_source_type == 'hcp':
        if not args.hcp_root or not args.hcp_parcellated_output:
            parser.error("--hcp-root and --hcp-parcellated-output are required for HCP mode")
        from tcp.preprocessing.config.data_source_config import create_hcp_config
        data_source_config = create_hcp_config(
            hcp_root=args.hcp_root,
            parcellated_output=args.hcp_parcellated_output,
            default_task=args.default_task
        )
    elif args.data_source_type == 'combined':
        if not args.hcp_root or not args.hcp_parcellated_output:
            parser.error("--hcp-root and --hcp-parcellated-output are required for combined mode")
        from tcp.preprocessing.config.data_source_config import create_combined_config
        data_source_config = create_combined_config(
            dataset_path=dataset_path,
            hcp_root=args.hcp_root,
            hcp_parcellated_output=args.hcp_parcellated_output,
            duplicate_resolution=args.duplicate_resolution,
            default_task=args.default_task
        )

    # Initialize new pipeline
    pipeline = NewSubjectFilterPipeline(
        input_dir, output_dir, dataset_path,
        data_source_config=data_source_config
    )

    # Configure task availability filter
    task_filter = TaskAvailabilityFilter(
        required_tasks=args.tasks,
        require_all_tasks=args.require_all_tasks,
        data_types=args.data_types
    )
    
    # Add filters to pipeline
    pipeline.add_filter(task_filter)
    
    try:
        # Load data (unified subject list)
        subjects_df, file_paths = pipeline.load_and_convert_data()

        # Apply filters
        included_subjects, excluded_subjects = pipeline.apply_filters(subjects_df, file_paths)

        print(f"\n{'='*60}")
        print(f"TASK FILTERING COMPLETE")
        print(f"{'='*60}")
        print(f"\nFinal Results:")
        print(f"  Total input subjects: {len(subjects_df)}")
        print(f"  Included subjects: {len(included_subjects)}")
        print(f"  Excluded subjects: {len(excluded_subjects)}")

        if len(subjects_df) > 0:
            inclusion_rate = (len(included_subjects) / len(subjects_df)) * 100
            print(f"  Overall inclusion rate: {inclusion_rate:.1f}%")

        # Show task availability summary if possible
        if 'has_hammer_raw' in included_subjects.columns or 'has_stroop_raw' in included_subjects.columns:
            print(f"\nTask Data Availability (Included Subjects):")
            if 'has_hammer_raw' in included_subjects.columns:
                hammer_count = included_subjects['has_hammer_raw'].sum()
                print(f"  - Hammer task: {hammer_count}/{len(included_subjects)} ({hammer_count/len(included_subjects)*100:.1f}%)")
            if 'has_stroop_raw' in included_subjects.columns:
                stroop_count = included_subjects['has_stroop_raw'].sum()
                print(f"  - Stroop task: {stroop_count}/{len(included_subjects)} ({stroop_count/len(included_subjects)*100:.1f}%)")

        print(f"\nOutput saved to: {output_dir}")
        print(f"\nNext steps:")
        print(f"  1. (Optional) Run classify_diagnoses.py and integrate_cross_analysis.py for analysis groups")
        print(f"  2. Run map_subject_files.py to create file path mapping")
        print(f"  3. Run fetch_filtered_data.py to download MRI data")
        print(f"  4. Review task_filtering_summary.json for detailed statistics")

        return 0

    except Exception as e:
        print(f"{ERROR} Error during task filtering: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
