# TCP Dataset: MDD Analysis Setup

This directory contains Python scripts and datasets for analyzing Major Depressive Disorder (MDD) subjects and controls from the Transdiagnostic Connectome Project.

## What was accomplished

? **Dataset Analysis**: Analyzed the complete TCP dataset structure  
? **Subject Identification**: Identified 149 patient subjects (potential MDD cases) and 96 control subjects  
? **Task Data Extraction**: Located Stroop and Hammer task fMRI data for both groups  
? **Data Access Tools**: Created Python utilities to load both raw NIFTI and preprocessed timeseries data  
? **Analysis-Ready Datasets**: Generated balanced datasets (87 patients, 87 controls) with task data

## Files

### Core Scripts
- **`extract_mdd_subjects.py`** - Main extraction script that analyzes the dataset and identifies subjects
- **`data_loader_utils.py`** - Utility functions for loading fMRI data and working with subjects  
- **`examples/mdd_analysis.py`** - Complete workflow demonstrating MDD vs control analysis

### Generated Datasets
- **`/**/tcp_analysis_results/`** - Complete dataset extraction results
  - `patient_subjects.csv` - All 149 patient subjects with demographics and task data availability
  - `control_subjects.csv` - All 96 control subjects with demographics and task data availability
  - `task_file_paths.json` - File paths to all fMRI data for all subjects
  - `summary.json` - Dataset overview statistics

- **`/**/mdd_analysis_datasets/`** - Analysis-ready balanced datasets
  - `mdd_patients_stroop.csv` - 87 patient subjects with Stroop task data
  - `controls_stroop.csv` - 87 control subjects with Stroop task data  
  - `analysis_file_paths.json` - File paths for the 174 analysis subjects
  - `analysis_summary.json` - Analysis dataset statistics

### Environment
- **`tcp_analysis_env/`** - Python virtual environment with required packages

## Dataset Overview

### Subjects Available
- **149 Patient subjects** (potential MDD cases)
  - 139 have task fMRI data
  - Age: 34.8 \pm 13.2 years (range: 18.3-64.5)
  - Sex: 87 Female, 59 Male, 3 Other
  
- **96 Control subjects** (general population)
  - 89 have task fMRI data  
  - Age: 33.7 \pm 13.7 years (range: 18.1-68.3)
  - Sex: 52 Female, 43 Male, 1 Other

### Task Data Available
- **Stroop Task**: 138 patients + 88 controls (cognitive control paradigm)
- **Hammer Task**: 130 patients + 89 controls (emotional faces vs shapes)

### Data Types
1. **Raw NIFTI files** - Original fMRI timeseries data (~100-200MB each)
2. **Preprocessed Timeseries** - HCP-processed, denoised, parcellated to 434 regions (HDF5 format)
3. **Task Events** - Experimental design files with trial timings and conditions

## Important Notes About MDD Identification

?? **Diagnostic Limitation**: The clinical diagnostic files (DSM diagnoses, depression scales) are stored as git-annex symlinks and not directly accessible in this dataset copy. 

**Current Approach**: We're using the "Patient" group classification as a proxy for psychiatric conditions, which includes various disorders beyond just MDD.

**For True MDD Analysis**: You would need access to:
- `tmb_dsm01.tsv` - DSM-5 diagnostic information  
- `madrs01.tsv` - Montgomery-Asberg Depression Rating Scale
- `qids01.tsv` - Quick Inventory of Depressive Symptomatology
- Clinical interview data from SCID-5

## How to Use

### Quick Start
```python
# Activate the environment
source tcp_analysis_env/bin/activate

# Run the complete analysis workflow
python examples/mdd_analysis.py
```

### Loading Your Analysis Dataset
```python
from data_loader_utils import load_tcp_results
import pandas as pd

# Load the balanced analysis datasets
patients_df = pd.read_csv("mdd_analysis_datasets/mdd_patients_stroop.csv")
controls_df = pd.read_csv("mdd_analysis_datasets/controls_stroop.csv")

print(f"Analysis subjects: {len(patients_df)} patients, {len(controls_df)} controls")
```

### Accessing fMRI Data
```python
from data_loader_utils import get_subject_task_files, load_timeseries_data
import json

# Load file paths
with open("mdd_analysis_datasets/analysis_file_paths.json", 'r') as f:
    file_paths = json.load(f)

# Get data for a specific subject
subject_id = "sub-NDARINVZY232VM1"  # Example patient
stroop_nifti = file_paths['raw_nifti'][subject_id]['stroop']
stroop_events = file_paths['events'][subject_id]['stroop']

print(f"Subject {subject_id} has {len(stroop_nifti)} Stroop NIFTI files")
```

### Loading Preprocessed Timeseries
```python
# Note: Timeseries files may also be git-annex symlinks
# Check if files are accessible before loading
timeseries_files = file_paths['timeseries'][subject_id]['stroop']
if timeseries_files:
    try:
        timeseries = load_timeseries_data(timeseries_files[0])
        if timeseries is not None:
            print(f"Timeseries shape: {timeseries.shape}")  # (timepoints, 434 regions)
    except:
        print("Timeseries file not accessible (git-annex)")
```
