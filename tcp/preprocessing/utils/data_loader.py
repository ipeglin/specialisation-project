#!/usr/bin/env python3
"""
Utility functions for loading and working with TCP dataset
including task fMRI data and subject identification.

Author: Ian Philip Eglin
Date: 2025-09-11
"""

import pandas as pd
import numpy as np
import h5py
import json
from pathlib import Path
import nibabel as nib

def load_tcp_results(results_dir="/Users/ipeglin/git/ds005237/tcp_analysis_results"):
    """Load the extracted TCP analysis results"""
    results_path = Path(results_dir)
    
    # Load subject dataframes
    patients_df = pd.read_csv(results_path / "patient_subjects.csv")
    controls_df = pd.read_csv(results_path / "control_subjects.csv")
    
    # Load file paths
    with open(results_path / "task_file_paths.json", 'r') as f:
        file_paths = json.load(f)
        
    # Load summary
    with open(results_path / "summary.json", 'r') as f:
        summary = json.load(f)
        
    print(f"Loaded TCP results:")
    print(f"  - {len(patients_df)} patient subjects")
    print(f"  - {len(controls_df)} control subjects")
    print(f"  - {summary['dataset_info']['patients_with_task_data']} patients with task data")
    print(f"  - {summary['dataset_info']['controls_with_task_data']} controls with task data")
    
    return patients_df, controls_df, file_paths, summary

def filter_subjects_with_task_data(subjects_df, tasks=['hammer', 'stroop'], data_type='raw'):
    """Filter subjects that have specific task data"""
    if data_type == 'raw':
        task_cols = [f'has_{task}_raw' for task in tasks]
    else:
        task_cols = [f'has_{task}_timeseries' for task in tasks]
    
    # Subjects with any of the specified tasks
    mask = subjects_df[task_cols].any(axis=1)
    filtered_df = subjects_df[mask].copy()
    
    print(f"Filtered subjects with {tasks} {data_type} data: {len(filtered_df)}/{len(subjects_df)}")
    
    return filtered_df

def get_subject_task_files(subject_id, file_paths, task='stroop', data_type='raw_nifti'):
    """Get file paths for a specific subject and task"""
    if subject_id not in file_paths[data_type]:
        return []
    
    task_files = file_paths[data_type][subject_id].get(task, [])
    return task_files

def load_timeseries_data(h5_file_path):
    """Load timeseries data from HDF5 file"""
    try:
        with h5py.File(h5_file_path, 'r') as f:
            # Check what datasets are available
            datasets = list(f.keys())
            print(f"Available datasets in {Path(h5_file_path).name}: {datasets}")
            
            # Common dataset names in this format
            if 'data' in datasets:
                timeseries = f['data'][:]
            elif 'timeseries' in datasets:
                timeseries = f['timeseries'][:]
            else:
                # Take the first dataset
                timeseries = f[datasets[0]][:]
            
            return timeseries
            
    except Exception as e:
        print(f"Error loading {h5_file_path}: {e}")
        return None

def load_task_events(events_file_path):
    """Load task events from TSV file"""
    try:
        events_df = pd.read_csv(events_file_path, sep='\t')
        return events_df
    except Exception as e:
        print(f"Error loading {events_file_path}: {e}")
        return None

def identify_potential_mdd_subjects(patients_df, age_range=(18, 65), sex=None):
    """
    Identify potential MDD subjects based on available criteria.
    Since diagnostic data is not accessible, this provides a starting point
    for MDD subject identification based on demographics.
    
    Note: This is a placeholder - actual MDD diagnosis should be done
    using clinical interviews and diagnostic instruments when available.
    """
    filtered_df = patients_df.copy()
    
    # Filter by age
    if age_range:
        age_mask = (filtered_df['age'] >= age_range[0]) & (filtered_df['age'] <= age_range[1])
        filtered_df = filtered_df[age_mask]
        
    # Filter by sex if specified
    if sex:
        sex_mask = filtered_df['sex'] == sex
        filtered_df = filtered_df[sex_mask]
        
    print(f"Potential MDD subjects (patients aged {age_range}, sex={sex}): {len(filtered_df)}")
    print(f"Age distribution: {filtered_df['age'].describe()}")
    print(f"Sex distribution: {filtered_df['sex'].value_counts()}")
    
    return filtered_df

def create_analysis_dataset(patients_df, controls_df, task='stroop', 
                          min_age=18, max_age=65, balance_groups=True):
    """
    Create a balanced dataset for analysis with patients and controls
    """
    print(f"Creating analysis dataset for {task} task...")
    
    # Filter subjects with task data
    patients_with_task = filter_subjects_with_task_data(
        patients_df, tasks=[task], data_type='raw'
    )
    controls_with_task = filter_subjects_with_task_data(
        controls_df, tasks=[task], data_type='raw'
    )
    
    # Age filtering
    age_mask_p = (patients_with_task['age'] >= min_age) & (patients_with_task['age'] <= max_age)
    age_mask_c = (controls_with_task['age'] >= min_age) & (controls_with_task['age'] <= max_age)
    
    patients_filtered = patients_with_task[age_mask_p]
    controls_filtered = controls_with_task[age_mask_c]
    
    # Balance groups if requested
    if balance_groups:
        min_n = min(len(patients_filtered), len(controls_filtered))
        patients_balanced = patients_filtered.sample(n=min_n, random_state=42)
        controls_balanced = controls_filtered.sample(n=min_n, random_state=42)
        
        print(f"Balanced dataset: {min_n} patients, {min_n} controls")
        
        return patients_balanced, controls_balanced
    else:
        print(f"Unbalanced dataset: {len(patients_filtered)} patients, {len(controls_filtered)} controls")
        return patients_filtered, controls_filtered

def get_demographics_summary(subjects_df, group_name="Subjects"):
    """Get demographics summary for a group of subjects"""
    print(f"\n{group_name} Demographics:")
    print(f"  N = {len(subjects_df)}")
    print(f"  Age: {subjects_df['age'].mean():.1f} ± {subjects_df['age'].std():.1f} years")
    print(f"  Age range: {subjects_df['age'].min():.1f} - {subjects_df['age'].max():.1f} years")
    print(f"  Sex: {subjects_df['sex'].value_counts().to_dict()}")
    print(f"  Site: {subjects_df['Site'].value_counts().to_dict()}")
    
    # Task data availability
    if 'has_hammer_raw' in subjects_df.columns:
        print(f"  Hammer task: {subjects_df['has_hammer_raw'].sum()}/{len(subjects_df)}")
    if 'has_stroop_raw' in subjects_df.columns:
        print(f"  Stroop task: {subjects_df['has_stroop_raw'].sum()}/{len(subjects_df)}")

def load_nifti_data(nifti_file_path):
    """Load NIFTI data using nibabel"""
    try:
        img = nib.load(nifti_file_path)
        data = img.get_fdata()
        affine = img.affine
        header = img.header
        
        print(f"Loaded {Path(nifti_file_path).name}: shape {data.shape}")
        return data, affine, header
        
    except Exception as e:
        print(f"Error loading {nifti_file_path}: {e}")
        return None, None, None

# Example usage and demo functions
def demo_data_loading():
    """Demonstrate how to use the data loading functions"""
    print("=== TCP Data Loading Demo ===\n")
    
    # Load results
    patients_df, controls_df, file_paths, summary = load_tcp_results()
    
    # Get demographics
    get_demographics_summary(patients_df, "Patients")
    get_demographics_summary(controls_df, "Controls")
    
    # Create analysis dataset
    patients_analysis, controls_analysis = create_analysis_dataset(
        patients_df, controls_df, task='stroop', balance_groups=True
    )
    
    # Show example of getting file paths
    if len(patients_analysis) > 0:
        example_subject = patients_analysis.iloc[0]['participant_id']
        stroop_files = get_subject_task_files(example_subject, file_paths, task='stroop')
        hammer_files = get_subject_task_files(example_subject, file_paths, task='hammer')
        
        print(f"\nExample subject {example_subject}:")
        print(f"  Stroop files: {len(stroop_files)}")
        print(f"  Hammer files: {len(hammer_files)}")
        
        if stroop_files:
            print(f"  First stroop file: {Path(stroop_files[0]).name}")
            
        # Example of loading timeseries data
        ts_files = get_subject_task_files(example_subject, file_paths, 
                                        task='stroop', data_type='timeseries')
        if ts_files:
            print(f"  Timeseries files: {len(ts_files)}")
            # Note: Uncomment to actually load data
            # timeseries = load_timeseries_data(ts_files[0])
            # if timeseries is not None:
            #     print(f"  Timeseries shape: {timeseries.shape}")
    
    return patients_analysis, controls_analysis, file_paths

if __name__ == "__main__":
    demo_data_loading()
