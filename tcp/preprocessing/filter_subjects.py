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

import sys
import argparse
from pathlib import Path
from typing import Optional

# Add project root to path to import config
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.paths import get_script_output_path, get_tcp_dataset_path
from tcp.preprocessing.utils.filter_pipeline import SubjectFilterPipeline
from tcp.preprocessing.utils.subject_filters import TaskAvailabilityFilter
import pandas as pd

def detect_input_source() -> Path:
    """Automatically detect input source from new pipeline steps"""
    
    # Option 1: Use phenotype filtered subjects (if available)
    phenotype_dir = get_script_output_path('tcp_preprocessing', 'filter_phenotype')
    phenotype_subjects_file = phenotype_dir / 'phenotype_filtered_subjects.csv'
    
    if phenotype_subjects_file.exists():
        print(f"✓ Found phenotype filtered subjects: {phenotype_dir}")
        return phenotype_dir
    
    # Option 2: Use valid subjects from validation step
    validation_dir = get_script_output_path('tcp_preprocessing', 'validate_subjects')
    validation_subjects_file = validation_dir / 'valid_subjects.csv'
    
    if validation_subjects_file.exists():
        print(f"✓ Found validated subjects: {validation_dir}")
        return validation_dir
    
    # Option 3: Fallback to old extract_subjects output (backward compatibility)
    extract_dir = get_script_output_path('tcp_preprocessing', 'extract_subjects')
    extract_patients_file = extract_dir / 'patient_subjects.csv'
    extract_controls_file = extract_dir / 'control_subjects.csv'
    
    if extract_patients_file.exists() and extract_controls_file.exists():
        print(f"⚠ Using legacy extract_subjects output: {extract_dir}")
        print(f"  Consider running the new pipeline: validate_subjects.py → filter_phenotype.py")
        return extract_dir
    
    # No valid input found
    raise FileNotFoundError(
        "No valid input data found. Please run one of the following first:\n"
        "  1. validate_subjects.py (required)\n"
        "  2. filter_phenotype.py (optional, for diagnosis filtering)\n"
        "  3. extract_subjects.py (legacy - for backward compatibility)"
    )

class NewSubjectFilterPipeline:
    """Updated pipeline that works with new data structure"""
    
    def __init__(self, input_dir: Path, output_dir: Path, dataset_path: Path):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.dataset_path = Path(dataset_path)
        self.filters = []
        
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Dataset path: {self.dataset_path}")
    
    def add_filter(self, filter_instance):
        """Add a filter to the pipeline"""
        self.filters.append(filter_instance)
        print(f"Added filter: {filter_instance.filter_name}")
        return self
    
    def load_and_convert_data(self):
        """Load data from new pipeline format and convert to old format for compatibility"""
        import pandas as pd
        import json
        from pathlib import Path
        
        # Check input format and load accordingly
        phenotype_file = self.input_dir / 'phenotype_filtered_subjects.csv'
        validation_file = self.input_dir / 'valid_subjects.csv'
        
        if phenotype_file.exists():
            # Load from phenotype filtering
            print("Loading phenotype filtered subjects...")
            subjects_df = pd.read_csv(phenotype_file)
            
            # Split into patients and controls based on Group if available
            if 'Group' in subjects_df.columns:
                patients_df = subjects_df[subjects_df['Group'] == 'Patient'].copy()
                controls_df = subjects_df[subjects_df['Group'] == 'GenPop'].copy()
            else:
                # If no Group column, put all subjects as "patients" for processing
                patients_df = subjects_df.copy()
                controls_df = pd.DataFrame()
                
        elif validation_file.exists():
            # Load from validation step - need to create patient/control split
            print("Loading validated subjects...")
            subjects_df = pd.read_csv(validation_file)
            
            # Since validation doesn't split by diagnosis, treat all as potential subjects
            # The task filter will determine which have usable data
            patients_df = subjects_df.copy()
            controls_df = pd.DataFrame()
            
        else:
            # Try legacy format
            patient_file = self.input_dir / 'patient_subjects.csv'
            control_file = self.input_dir / 'control_subjects.csv'
            
            if patient_file.exists() and control_file.exists():
                print("Loading legacy extract_subjects format...")
                patients_df = pd.read_csv(patient_file)
                controls_df = pd.read_csv(control_file)
            else:
                raise FileNotFoundError(f"No valid subject files found in {self.input_dir}")
        
        # Generate task file paths by scanning the dataset
        print("Generating task file paths...")
        file_paths = self._generate_task_file_paths(patients_df, controls_df)
        
        return patients_df, controls_df, file_paths
    
    def _generate_task_file_paths(self, patients_df, controls_df):
        """Generate task file paths by scanning dataset structure"""
        import glob
        
        all_subjects = []
        if len(patients_df) > 0:
            all_subjects.extend(patients_df['subject_id'].tolist())
        if len(controls_df) > 0:
            all_subjects.extend(controls_df['subject_id'].tolist())
        
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
    
    def apply_filters(self, patients_df, controls_df, file_paths):
        """Apply filters using the existing filter pipeline logic"""
        # Use the existing SubjectFilterPipeline but with custom data
        from tcp.preprocessing.utils.filter_pipeline import SubjectFilterPipeline
        
        # Create temporary directory structure for compatibility
        temp_input_dir = self.output_dir / "temp_input"
        temp_input_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure DataFrames have 'participant_id' column for compatibility with filters
        if 'subject_id' in patients_df.columns and 'participant_id' not in patients_df.columns:
            patients_df['participant_id'] = patients_df['subject_id']
        if 'subject_id' in controls_df.columns and 'participant_id' not in controls_df.columns:
            controls_df['participant_id'] = controls_df['subject_id']
        
        # Save data in format expected by SubjectFilterPipeline
        patients_df.to_csv(temp_input_dir / "patient_subjects.csv", index=False)
        
        # Ensure controls_df has proper column structure even if empty
        if controls_df.empty and not patients_df.empty:
            # Create empty DataFrame with same columns as patients_df
            controls_df = pd.DataFrame(columns=patients_df.columns)
        controls_df.to_csv(temp_input_dir / "control_subjects.csv", index=False)
        
        import json
        with open(temp_input_dir / "task_file_paths.json", 'w') as f:
            json.dump(file_paths, f, indent=2)
        
        # Create summary for compatibility
        summary = {
            'dataset_info': {
                'total_patients': len(patients_df),
                'total_controls': len(controls_df)
            }
        }
        with open(temp_input_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Use existing pipeline
        pipeline = SubjectFilterPipeline(str(temp_input_dir), str(self.output_dir))
        
        # Add all filters
        for filter_instance in self.filters:
            pipeline.add_filter(filter_instance)
        
        # Run pipeline
        pipeline.load_extracted_data()
        pipeline.apply_filters()
        
        # Export results
        output_path = pipeline.export_results()
        
        # Clean up temp directory
        import shutil
        shutil.rmtree(temp_input_dir, ignore_errors=True)
        
        return pipeline

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
    
    # Initialize new pipeline
    pipeline = NewSubjectFilterPipeline(input_dir, output_dir, dataset_path)

    # Configure task availability filter
    task_filter = TaskAvailabilityFilter(
        required_tasks=args.tasks,
        require_all_tasks=args.require_all_tasks,
        data_types=args.data_types
    )
    
    # Add filters to pipeline
    pipeline.add_filter(task_filter)
    
    try:
        # Load and convert data to compatible format
        patients_df, controls_df, file_paths = pipeline.load_and_convert_data()
        
        # Apply filters
        filter_pipeline = pipeline.apply_filters(patients_df, controls_df, file_paths)
        
        # Get final results
        included_patients, excluded_patients, included_controls, excluded_controls = filter_pipeline.get_final_results()
        
        print(f"\n=== FILTERING COMPLETE ===")
        print(f"Output saved to: {output_dir}")
        print(f"\nFinal Results:")
        print(f"  Included: {len(included_patients)} patients, {len(included_controls)} controls")
        print(f"  Excluded: {len(excluded_patients)} patients, {len(excluded_controls)} controls")
        
        total_subjects = len(included_patients) + len(excluded_patients) + len(included_controls) + len(excluded_controls)
        if total_subjects > 0:
            inclusion_rate = (len(included_patients) + len(included_controls)) / total_subjects * 100
            print(f"  Inclusion Rate: {inclusion_rate:.1f}%")
        
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
        
        print(f"\n{'='*60}")
        print(f"TASK FILTERING COMPLETE")
        print(f"{'='*60}")
        print(f"Results saved to: {output_dir}")
        print(f"\nNext steps:")
        print(f"  1. Run fetch_filtered_data.py to download MRI data for included subjects")
        print(f"  2. Review filtering_report.json for detailed statistics")
        print(f"  3. Use included/ directory for downstream analysis")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during task filtering: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
