#!/usr/bin/env python3
"""
Extract Major Depressive Disorder subjects and task fMRI data
from the Transdiagnostic Connectome Project dataset.

Author: Ian Philip Eglin
Date: 2025-09-11
"""

import glob
import os
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

class TCPDataExtractor:
    """Extract MDD subjects and task data from TCP dataset"""
    
    def __init__(self, dataset_path, output_dir):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.phenotype_path = self.dataset_path / "phenotype"
        self.timeseries_path = self.dataset_path / "fMRI_timeseries_clean_denoised_GSR_parcellated"
        self.participants_file = self.dataset_path / "participants.tsv"
        
        # Subject mappings
        self.bids_to_timeseries_map = {}
        self.timeseries_to_bids_map = {}
        
        print(f"Initializing TCP Data Extractor for: {dataset_path}")
        
    def create_subject_mappings(self):
        """Create mappings between BIDS IDs and timeseries IDs"""
        print("Creating subject ID mappings...")
        
        # Load participants data
        if self.participants_file.exists():
            participants = pd.read_csv(self.participants_file, sep='\t')
            print(f"Found {len(participants)} participants in participants.tsv")
            
            # Extract subject IDs and create mappings
            for _, row in participants.iterrows():
                bids_id = row['participant_id']  # e.g., sub-NDARINVAG023WG3
                # Convert to timeseries format: remove sub- prefix and change INV to _INV
                timeseries_id = bids_id.replace('sub-NDAR', 'NDAR_').replace('INV', 'INV')
                
                self.bids_to_timeseries_map[bids_id] = timeseries_id
                self.timeseries_to_bids_map[timeseries_id] = bids_id
                
            print(f"Created mappings for {len(self.bids_to_timeseries_map)} subjects")
        else:
            print("Warning: participants.tsv not found")
            
    def load_accessible_phenotype_data(self):
        """Load phenotype data that's accessible (not git-annex symlinks)"""
        print("Loading accessible phenotype data...")
        
        # Find files that are actually readable
        accessible_files = []
        phenotype_files = list(self.phenotype_path.glob("*.tsv"))
        
        for file_path in phenotype_files:
            try:
                # Try to read first few lines to see if accessible
                with open(file_path, 'r') as f:
                    f.read(100)  # Try to read first 100 chars
                accessible_files.append(file_path)
            except (OSError, IOError):
                # This is likely a git-annex symlink pointing to missing file
                continue
                
        print(f"Found {len(accessible_files)} accessible phenotype files:")
        for file_path in accessible_files:
            print(f"  - {file_path.name}")
            
        # Load accessible data
        phenotype_data = {}
        for file_path in accessible_files:
            if file_path.name.endswith('_definitions.tsv'):
                continue  # Skip definition files for now
                
            try:
                df = pd.read_csv(file_path, sep='\t')
                phenotype_data[file_path.stem] = df
                print(f"  Loaded {file_path.name}: {df.shape}")
            except Exception as e:
                print(f"  Error loading {file_path.name}: {e}")
                
        return phenotype_data
        
    def load_participants_data(self):
        """Load and analyze participants data"""
        print("Loading participants data...")
        
        if self.participants_file.exists():
            participants = pd.read_csv(self.participants_file, sep='\t')
            
            print(f"Participants shape: {participants.shape}")
            print(f"Columns: {list(participants.columns)}")
            print(f"Groups: {participants['Group'].value_counts()}")
            print(f"Sites: {participants['Site'].value_counts()}")
            print(f"Sex distribution: {participants['sex'].value_counts()}")
            print(f"Age range: {participants['age'].min():.1f} - {participants['age'].max():.1f}")
            
            return participants
        else:
            print("Error: participants.tsv not found")
            return None
            
    def identify_subjects_by_group(self, participants_df):
        """Identify subjects based on Patient/GenPop groups"""
        print("Identifying subjects by group...")
        
        # Get patients (potential MDD subjects)
        patients = participants_df[participants_df['Group'] == 'Patient'].copy()
        print(f"Found {len(patients)} patient subjects")
        
        # Get controls (general population)
        controls = participants_df[participants_df['Group'] == 'GenPop'].copy()
        print(f"Found {len(controls)} control subjects")
        
        return patients, controls
        
    def check_task_data_availability(self, subject_ids):
        """Check what task data is available for given subjects"""
        print("Checking task data availability...")
        
        # Check raw NIFTI data
        raw_data_availability = {}
        for subj_id in subject_ids:
            subj_dir = self.dataset_path / subj_id / "func"
            if subj_dir.exists():
                task_files = list(subj_dir.glob("*task-*.nii.gz"))
                tasks = set()
                for task_file in task_files:
                    # Extract task name
                    filename = task_file.name
                    if "task-hammer" in filename:
                        tasks.add("hammer")
                    elif "task-stroop" in filename:
                        tasks.add("stroop")
                raw_data_availability[subj_id] = list(tasks)
            else:
                raw_data_availability[subj_id] = []
                
        # Check preprocessed timeseries data
        timeseries_availability = {}
        if self.timeseries_path.exists():
            for subj_id in subject_ids:
                # Convert BIDS ID to timeseries ID
                ts_id = self.bids_to_timeseries_map.get(subj_id, None)
                if ts_id:
                    ts_dir = self.timeseries_path / ts_id
                    if ts_dir.exists():
                        h5_files = list(ts_dir.glob("*.h5"))
                        tasks = set()
                        for h5_file in h5_files:
                            filename = h5_file.name
                            if "hammer" in filename:
                                tasks.add("hammer")
                            elif "stroop" in filename:
                                tasks.add("stroop")
                        timeseries_availability[subj_id] = list(tasks)
                    else:
                        timeseries_availability[subj_id] = []
                else:
                    timeseries_availability[subj_id] = []
        
        print("Task data availability summary:")
        print(f"  Raw NIFTI - Subjects with any task data: {sum(1 for v in raw_data_availability.values() if v)}")
        print(f"  Timeseries - Subjects with any task data: {sum(1 for v in timeseries_availability.values() if v)}")
        
        return raw_data_availability, timeseries_availability
        
    def extract_task_file_paths(self, subject_ids, task_types=["hammer", "stroop"]):
        """Extract file paths for task data"""
        print(f"Extracting file paths for tasks: {task_types}")
        
        file_paths = {
            'raw_nifti': {},
            'timeseries': {},
            'events': {}
        }
        
        for subj_id in subject_ids:
            file_paths['raw_nifti'][subj_id] = {}
            file_paths['events'][subj_id] = {}
            file_paths['timeseries'][subj_id] = {}
            
            # Raw NIFTI files
            subj_dir = self.dataset_path / subj_id / "func"
            if subj_dir.exists():
                for task in task_types:
                    task_files = list(subj_dir.glob(f"*task-{task}*.nii.gz"))
                    event_files = list(subj_dir.glob(f"*task-{task}*_events.tsv"))
                    
                    file_paths['raw_nifti'][subj_id][task] = [str(f) for f in task_files]
                    file_paths['events'][subj_id][task] = [str(f) for f in event_files]
                    
            # Timeseries files
            ts_id = self.bids_to_timeseries_map.get(subj_id, None)
            if ts_id and self.timeseries_path.exists():
                ts_dir = self.timeseries_path / ts_id
                if ts_dir.exists():
                    for task in task_types:
                        h5_files = list(ts_dir.glob(f"*task-{task}*.h5"))
                        file_paths['timeseries'][subj_id][task] = [str(f) for f in h5_files]
                        
        return file_paths
        
    def create_subject_dataframes(self, participants_df, phenotype_data, file_paths):
        """Create structured dataframes for MDD and control subjects"""
        print("Creating structured subject dataframes...")
        
        # Separate patients and controls
        patients, controls = self.identify_subjects_by_group(participants_df)
        
        # Add file path information
        def add_task_data_info(df):
            df = df.copy()
            df['has_hammer_raw'] = df['participant_id'].apply(
                lambda x: len(file_paths['raw_nifti'].get(x, {}).get('hammer', [])) > 0
            )
            df['has_stroop_raw'] = df['participant_id'].apply(
                lambda x: len(file_paths['raw_nifti'].get(x, {}).get('stroop', [])) > 0
            )
            df['has_hammer_timeseries'] = df['participant_id'].apply(
                lambda x: len(file_paths['timeseries'].get(x, {}).get('hammer', [])) > 0
            )
            df['has_stroop_timeseries'] = df['participant_id'].apply(
                lambda x: len(file_paths['timeseries'].get(x, {}).get('stroop', [])) > 0
            )
            
            # Add task data counts
            df['n_hammer_files'] = df['participant_id'].apply(
                lambda x: len(file_paths['raw_nifti'].get(x, {}).get('hammer', []))
            )
            df['n_stroop_files'] = df['participant_id'].apply(
                lambda x: len(file_paths['raw_nifti'].get(x, {}).get('stroop', []))
            )
            
            return df
        
        patients_df = add_task_data_info(patients)
        controls_df = add_task_data_info(controls)
        
        print(f"Patient subjects: {len(patients_df)}")
        print(f"  - With hammer task data: {patients_df['has_hammer_raw'].sum()}")
        print(f"  - With stroop task data: {patients_df['has_stroop_raw'].sum()}")
        print(f"  - With any task data: {(patients_df['has_hammer_raw'] | patients_df['has_stroop_raw']).sum()}")
        
        print(f"Control subjects: {len(controls_df)}")
        print(f"  - With hammer task data: {controls_df['has_hammer_raw'].sum()}")
        print(f"  - With stroop task data: {controls_df['has_stroop_raw'].sum()}")
        print(f"  - With any task data: {(controls_df['has_hammer_raw'] | controls_df['has_stroop_raw']).sum()}")
        
        return patients_df, controls_df
        
    def export_results(self, patients_df, controls_df, file_paths):
        """Export results to files"""
        output_path = Path(self.output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"Exporting results to {output_path}")
        
        # Export subject dataframes
        patients_df.to_csv(output_path / "patient_subjects.csv", index=False)
        controls_df.to_csv(output_path / "control_subjects.csv", index=False)
        
        # Export file paths as JSON for easy loading
        import json
        with open(output_path / "task_file_paths.json", 'w') as f:
            json.dump(file_paths, f, indent=2)
            
        # Create summary statistics
        summary = {
            'dataset_info': {
                'total_patients': len(patients_df),
                'total_controls': len(controls_df),
                'patients_with_task_data': int((patients_df['has_hammer_raw'] | patients_df['has_stroop_raw']).sum()),
                'controls_with_task_data': int((controls_df['has_hammer_raw'] | controls_df['has_stroop_raw']).sum())
            },
            'task_data_summary': {
                'hammer_patients': int(patients_df['has_hammer_raw'].sum()),
                'hammer_controls': int(controls_df['has_hammer_raw'].sum()),
                'stroop_patients': int(patients_df['has_stroop_raw'].sum()),
                'stroop_controls': int(controls_df['has_stroop_raw'].sum())
            }
        }
        
        with open(output_path / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
            
        print("Export completed:")
        print(f"  - patient_subjects.csv: {len(patients_df)} subjects")
        print(f"  - control_subjects.csv: {len(controls_df)} subjects")
        print(f"  - task_file_paths.json: File paths for task data")
        print(f"  - summary.json: Dataset summary statistics")
        
        return output_path

def main():
    """Main execution function"""
    dataset_path = '/Users/ipeglin/Git/ds005237'
    output_dir = '/Users/ipeglin/Git/ds005237'
    
    # Initialize extractor
    extractor = TCPDataExtractor(dataset_path, output_dir)
    
    # Create subject mappings
    extractor.create_subject_mappings()
    
    # Load participants data
    participants_df = extractor.load_participants_data()
    if participants_df is None:
        return
    
    # Load accessible phenotype data
    phenotype_data = extractor.load_accessible_phenotype_data()
    
    # Check task data availability
    all_subject_ids = participants_df['participant_id'].tolist()
    raw_availability, ts_availability = extractor.check_task_data_availability(all_subject_ids)
    
    # Extract file paths
    file_paths = extractor.extract_task_file_paths(all_subject_ids)
    
    # Create subject dataframes
    patients_df, controls_df = extractor.create_subject_dataframes(participants_df, phenotype_data, file_paths)
    
    # Export results
    output_dir = extractor.export_results(patients_df, controls_df, file_paths)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    
    return patients_df, controls_df, file_paths, phenotype_data

if __name__ == "__main__":
    results = main()
