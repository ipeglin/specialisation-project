#!/usr/bin/env python3
"""
Example script demonstrating how to access and analyze 
MDD subjects and task fMRI data from the TCP dataset.

This script shows how to:
1. Load patient and control subjects
2. Filter for task data availability
3. Access both raw NIFTI and preprocessed timeseries data
4. Create analysis-ready datasets

Author: Ian Philip Eglin
Date: 2025-09-11
"""

import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from data_loader_utils import *

def main_mdd_analysis():
    """Main analysis workflow for MDD subjects and controls"""
    
    print("="*60)
    print("TCP DATASET: MDD ANALYSIS WORKFLOW")
    print("="*60)
    
    # Step 1: Load the extracted data
    print("\n1. Loading TCP analysis results...")
    patients_df, controls_df, file_paths, summary = load_tcp_results()
    
    # Step 2: Show dataset overview
    print(f"\nDataset Overview:")
    print(f"  Total subjects: {len(patients_df) + len(controls_df)}")
    print(f"  Patient subjects: {len(patients_df)} (potential MDD cases)")
    print(f"  Control subjects: {len(controls_df)}")
    print(f"  Patients with task data: {summary['dataset_info']['patients_with_task_data']}")
    print(f"  Controls with task data: {summary['dataset_info']['controls_with_task_data']}")
    
    # Step 3: Focus on Stroop task data (cognitive control)
    print(f"\n2. Focusing on Stroop Task Data (Cognitive Control)...")
    
    # Filter subjects with Stroop task data
    patients_stroop = filter_subjects_with_task_data(patients_df, tasks=['stroop'])
    controls_stroop = filter_subjects_with_task_data(controls_df, tasks=['stroop'])
    
    get_demographics_summary(patients_stroop, "Patients with Stroop Data")
    get_demographics_summary(controls_stroop, "Controls with Stroop Data")
    
    # Step 4: Create balanced analysis dataset
    print(f"\n3. Creating Balanced Analysis Dataset...")
    patients_analysis, controls_analysis = create_analysis_dataset(
        patients_df, controls_df, 
        task='stroop', 
        min_age=18, 
        max_age=65, 
        balance_groups=True
    )
    
    # Step 5: Demonstrate data access
    print(f"\n4. Demonstrating Data Access...")
    
    # Pick example subjects
    example_patient = patients_analysis.iloc[0]['participant_id']
    example_control = controls_analysis.iloc[0]['participant_id']
    
    print(f"\nExample Patient: {example_patient}")
    print(f"  Age: {patients_analysis.iloc[0]['age']:.1f}, Sex: {patients_analysis.iloc[0]['sex']}")
    
    # Get file paths
    patient_stroop_raw = get_subject_task_files(example_patient, file_paths, 'stroop', 'raw_nifti')
    patient_stroop_ts = get_subject_task_files(example_patient, file_paths, 'stroop', 'timeseries')
    patient_events = get_subject_task_files(example_patient, file_paths, 'stroop', 'events')
    
    print(f"  Raw NIFTI files: {len(patient_stroop_raw)}")
    if patient_stroop_raw:
        for i, f in enumerate(patient_stroop_raw):
            print(f"    {i+1}. {Path(f).name}")
            
    print(f"  Preprocessed timeseries files: {len(patient_stroop_ts)}")
    if patient_stroop_ts:
        for i, f in enumerate(patient_stroop_ts):
            print(f"    {i+1}. {Path(f).name}")
            
    print(f"  Event files: {len(patient_events)}")
    if patient_events:
        for i, f in enumerate(patient_events):
            print(f"    {i+1}. {Path(f).name}")
    
    print(f"\nExample Control: {example_control}")
    print(f"  Age: {controls_analysis.iloc[0]['age']:.1f}, Sex: {controls_analysis.iloc[0]['sex']}")
    
    # Step 6: Demonstrate loading actual data
    print(f"\n5. Loading Example Data...")
    
    # Load events file
    if patient_events:
        events_df = load_task_events(patient_events[0])
        if events_df is not None:
            print(f"\nStroop task events loaded:")
            print(f"  Shape: {events_df.shape}")
            print(f"  Columns: {list(events_df.columns)}")
            if 'trial_type' in events_df.columns:
                print(f"  Trial types: {events_df['trial_type'].value_counts().to_dict()}")
    
    # Load timeseries data (preprocessed)
    if patient_stroop_ts:
        print(f"\nLoading preprocessed timeseries data...")
        timeseries = load_timeseries_data(patient_stroop_ts[0])
        if timeseries is not None:
            print(f"  Timeseries shape: {timeseries.shape}")
            print(f"  Data type: {timeseries.dtype}")
            print(f"  Time points: {timeseries.shape[0] if len(timeseries.shape) > 1 else 'N/A'}")
            print(f"  Regions: {timeseries.shape[1] if len(timeseries.shape) > 1 else 'N/A'}")
    
    # Step 7: Create analysis summary
    print(f"\n6. Analysis Summary...")
    
    analysis_summary = {
        'total_subjects': len(patients_analysis) + len(controls_analysis),
        'patients': len(patients_analysis),
        'controls': len(controls_analysis),
        'age_range': [
            min(patients_analysis['age'].min(), controls_analysis['age'].min()),
            max(patients_analysis['age'].max(), controls_analysis['age'].max())
        ],
        'sex_distribution': {
            'patients': patients_analysis['sex'].value_counts().to_dict(),
            'controls': controls_analysis['sex'].value_counts().to_dict()
        },
        'data_availability': {
            'stroop_raw_nifti': int(len(patients_analysis) + len(controls_analysis)),
            'stroop_timeseries': int(patients_analysis['has_stroop_timeseries'].sum() + 
                                controls_analysis['has_stroop_timeseries'].sum())
        }
    }
    
    print(f"\nFinal Analysis Dataset:")
    print(f"  Total subjects: {analysis_summary['total_subjects']}")
    print(f"  Patients (potential MDD): {analysis_summary['patients']}")
    print(f"  Controls: {analysis_summary['controls']}")
    print(f"  Age range: {analysis_summary['age_range'][0]:.1f} - {analysis_summary['age_range'][1]:.1f} years")
    print(f"  Stroop task data available for all subjects")
    
    # Step 8: Save analysis-ready datasets
    print(f"\n7. Saving Analysis-Ready Datasets...")
    
    output_dir = Path("/Users/ipeglin/git/ds005237/mdd_analysis_datasets")
    output_dir.mkdir(exist_ok=True)
    
    # Save subject lists
    patients_analysis.to_csv(output_dir / "mdd_patients_stroop.csv", index=False)
    controls_analysis.to_csv(output_dir / "controls_stroop.csv", index=False)
    
    # Save file path mappings for analysis subjects only
    analysis_subject_ids = list(patients_analysis['participant_id']) + list(controls_analysis['participant_id'])
    
    analysis_file_paths = {}
    for data_type in ['raw_nifti', 'timeseries', 'events']:
        analysis_file_paths[data_type] = {
            subj_id: file_paths[data_type][subj_id] 
            for subj_id in analysis_subject_ids 
            if subj_id in file_paths[data_type]
        }
    
    import json
    with open(output_dir / "analysis_file_paths.json", 'w') as f:
        json.dump(analysis_file_paths, f, indent=2)
    
    with open(output_dir / "analysis_summary.json", 'w') as f:
        json.dump(analysis_summary, f, indent=2)
    
    print(f"Analysis datasets saved to: {output_dir}")
    print(f"  - mdd_patients_stroop.csv: {len(patients_analysis)} patient subjects")
    print(f"  - controls_stroop.csv: {len(controls_analysis)} control subjects")
    print(f"  - analysis_file_paths.json: File paths for analysis subjects")
    print(f"  - analysis_summary.json: Dataset summary")
    
    return patients_analysis, controls_analysis, analysis_file_paths

def demonstrate_data_loading():
    """Demonstrate how to load actual fMRI data"""
    
    print("\n" + "="*60)
    print("DEMONSTRATION: LOADING ACTUAL fMRI DATA")
    print("="*60)
    
    # Load analysis results
    patients_analysis, controls_analysis, analysis_file_paths = main_mdd_analysis()
    
    # Pick one subject for detailed demonstration
    example_subject = patients_analysis.iloc[0]['participant_id']
    print(f"\nDetailed data loading for: {example_subject}")
    
    # Get all file types for this subject
    stroop_nifti = analysis_file_paths['raw_nifti'][example_subject]['stroop']
    stroop_timeseries = analysis_file_paths['timeseries'][example_subject]['stroop']
    stroop_events = analysis_file_paths['events'][example_subject]['stroop']
    
    print(f"\nAvailable data for {example_subject}:")
    print(f"  NIFTI files: {len(stroop_nifti)}")
    print(f"  Timeseries files: {len(stroop_timeseries)}")
    print(f"  Event files: {len(stroop_events)}")
    
    # Load and examine one of each
    if stroop_events:
        print(f"\n--- Loading Events ---")
        events = load_task_events(stroop_events[0])
        if events is not None:
            print(events.head())
            
    if stroop_timeseries:
        print(f"\n--- Loading Timeseries ---")
        ts_data = load_timeseries_data(stroop_timeseries[0])
        if ts_data is not None:
            print(f"Timeseries data shape: {ts_data.shape}")
            print(f"Data preview (first 5 timepoints, first 5 regions):")
            print(ts_data[:5, :5] if len(ts_data.shape) > 1 else ts_data[:5])
    
    # Note about NIFTI files - they're large, so just show how to load them
    if stroop_nifti:
        print(f"\n--- NIFTI Data Info ---")
        print(f"To load NIFTI data, use:")
        print(f"  data, affine, header = load_nifti_data('{stroop_nifti[0]}')")
        print(f"Note: NIFTI files are large (~100-200MB each), loading may take time")

if __name__ == "__main__":
    # Run the main analysis
    patients_analysis, controls_analysis, analysis_file_paths = main_mdd_analysis()
    
    # Uncomment to demonstrate actual data loading
    # demonstrate_data_loading()
    
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"You now have:")
    print(f"  - {len(patients_analysis)} patient subjects (potential MDD)")
    print(f"  - {len(controls_analysis)} control subjects")
    print(f"  - File paths to all Stroop task fMRI data")
    print(f"  - Both raw NIFTI and preprocessed timeseries data")
    print(f"  - Task event files for experimental design")
    print(f"\nReady for MDD vs Control analysis!")
