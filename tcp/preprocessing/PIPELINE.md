# TCP Pre-processing Pipeline for Task-based Analysis

This document describes the optimized TCP preprocessing pipeline that minimizes data downloads and provides early filtering capabilities.

## Overview

The new pipeline performs filtering **before** downloading large MRI files, reducing download requirements from ~1TB to only the data you need. The pipeline consists of the following steps:

1. **Dataset Initialization** - Clone and validate dataset
2. **Subject Validation** - Check subject directory structure  
3. **Global Data Fetching** - Download metadata files for filtering
4. **Phenotype Filtering** (Optional) - Filter by diagnosis (MDD vs controls)
5. **Task Data Filtering** - Filter subjects with hammer/stroop task data
6. **MRI Data Fetching** - Download actual MRI files for selected subjects

## Quick Start

### Option 1: Run Complete Pipeline (Recommended)

```bash
# Run the entire pipeline with default settings
python3 tcp/preprocessing/run_pipeline.py

# Run with custom filtering (adults only, MDD + controls)
python3 tcp/preprocessing/run_pipeline.py --min-age 18 --max-age 65

# Dry run to see what would be executed
python3 tcp/preprocessing/run_pipeline.py --dry-run
```

### Option 2: Run Individual Steps

```bash
# Step 1: Initialize dataset (clone if needed)
python3 tcp/preprocessing/initialize_dataset.py

# Step 2: Validate subject directories
python3 tcp/preprocessing/validate_subjects.py

# Step 3: Fetch global metadata files
python3 tcp/preprocessing/fetch_global_data.py

# Step 4: (Optional) Filter by phenotype/diagnosis
python3 tcp/preprocessing/filter_phenotype.py

# Step 5: Filter subjects with task data
python3 tcp/preprocessing/filter_subjects.py

# Step 6: Download MRI data for filtered subjects
python3 tcp/preprocessing/fetch_filtered_data.py
```

## Detailed Pipeline Steps

### 1. Dataset Initialization (`initialize_dataset.py`)

**Purpose**: Ensures the TCP dataset is properly cloned and initialized.

```bash
python3 tcp/preprocessing/initialize_dataset.py
```

**What it does**:
- Checks if dataset exists at configured path
- Clones dataset using `datalad install` if needed
- Validates dataset structure and git-annex setup
- Creates initialization report

**Output**: Dataset cloned to data path, initialization report

### 2. Subject Validation (`validate_subjects.py`)

**Purpose**: Validates which subjects have proper directory structure.

```bash
python3 tcp/preprocessing/validate_subjects.py
```

**What it does**:
- Scans for all `sub-*` directories in dataset
- Validates BIDS compliance and directory structure
- Checks for presence of `func/` and `anat/` subdirectories
- Separates valid vs invalid subjects with reasons

**Output**: 
- `valid_subjects.csv` - Subjects with proper structure
- `invalid_subjects.csv` - Subjects with issues
- `validation_summary.json` - Summary statistics

### 3. Global Data Fetching (`fetch_global_data.py`)

**Purpose**: Downloads metadata files needed for filtering decisions.

```bash
python3 tcp/preprocessing/fetch_global_data.py
```

**What it does**:
- Downloads `participants.tsv` using datalad get
- Downloads `phenotype/demos.tsv` for diagnosis information
- Downloads other phenotype files if available
- Validates downloaded files are readable

**Output**: Global metadata files downloaded and validated

### 4. Phenotype Filtering (`filter_phenotype.py`) - Optional

**Purpose**: Filter subjects based on demographics and diagnosis.

```bash
# Default: Include MDD patients and controls, ages 18-65
python3 tcp/preprocessing/filter_phenotype.py

# Custom filtering options
python3 tcp/preprocessing/filter_phenotype.py --min-age 21 --max-age 60
python3 tcp/preprocessing/filter_phenotype.py --include-mdd --include-controls
```

**What it does**:
- Filters subjects with Primary_Dx = "999" (controls) or containing "MDD" 
- Applies age range filtering if specified
- Supports custom column-based filtering
- Uses modular filter architecture with dependency injection

**Output**:
- `phenotype_filtered_subjects.csv` - Subjects passing phenotype criteria
- `phenotype_excluded_subjects.csv` - Excluded subjects with reasons
- `phenotype_filtering_summary.json` - Filtering statistics

**Filter Options**:
- `--include-mdd` - Include MDD subjects
- `--include-controls` - Include control subjects  
- `--min-age X` - Minimum age (inclusive)
- `--max-age Y` - Maximum age (inclusive)
- `--custom-filters` - Custom column:value filters

### 5. Task Data Filtering (`filter_subjects.py`)

**Purpose**: Filter subjects who have hammer and/or stroop task fMRI data.

```bash
# Default: Require at least one task (hammer OR stroop)
python3 tcp/preprocessing/filter_subjects.py

# Require both tasks
python3 tcp/preprocessing/filter_subjects.py --require-all-tasks

# Check specific data types
python3 tcp/preprocessing/filter_subjects.py --data-types raw_nifti events
```

**What it does**:
- Automatically detects input from previous pipeline steps
- Scans dataset for task-specific files (hammer/stroop)
- Filters subjects with at least one task (default) or all tasks
- Checks across multiple data types (raw NIFTI, timeseries, events)
- Uses existing filter pipeline architecture

**Input Auto-detection**:
1. Uses phenotype filtered subjects (if available)
2. Falls back to validated subjects
3. Falls back to legacy extract_subjects output

**Output**:
- `included/patient_subjects.csv` - Patients with task data
- `included/control_subjects.csv` - Controls with task data  
- `excluded/` - Excluded subjects with reasons
- `filtering_report.json` - Detailed filtering statistics

### 6. MRI Data Fetching (`fetch_filtered_data.py`)

**Purpose**: Download actual MRI files for subjects passing all filters.

```bash
# Default: Download task data, anatomical, and metadata
python3 tcp/preprocessing/fetch_filtered_data.py

# Dry run to see what would be downloaded
python3 tcp/preprocessing/fetch_filtered_data.py --dry-run

# Download specific data types
python3 tcp/preprocessing/fetch_filtered_data.py --data-types raw_nifti_hammer raw_nifti_stroop events_hammer events_stroop
```

**What it does**:
- Automatically detects filtered subjects from previous steps
- Downloads only files for included subjects
- Supports selective data type downloading
- Provides progress tracking and error handling
- Validates downloaded files

**Data Types Available**:
- `raw_nifti_hammer`, `raw_nifti_stroop` - Task fMRI NIFTI files
- `events_hammer`, `events_stroop` - Task timing files
- `json_metadata_hammer`, `json_metadata_stroop` - BIDS metadata
- `anatomical_t1w`, `anatomical_t2w` - Structural scans
- `phenotype`, `participants` - Metadata files

## Pipeline Benefits

### 🚀 **Reduced Downloads**
- Filter subjects before downloading MRI data
- Potential reduction from ~1TB to ~100-200GB depending on filtering
- Only download data for subjects of interest

### 🎯 **Early Filtering** 
- Filter by diagnosis before downloading large files
- Apply age, sex, and other demographic filters
- Customizable filtering criteria

### 🔧 **Modular Design**
- Each step is independent and reusable
- Can resume pipeline from any step
- Easy to add new filtering criteria

### 📊 **Comprehensive Tracking**
- Detailed inclusion/exclusion reasons at each step
- Progress reporting and error handling
- Pipeline state saving and resumption

### 🔄 **Backward Compatible**
- Works with existing analysis scripts
- Maintains same output format as original pipeline
- Legacy extract_subjects.py still supported

## Advanced Usage

### Resume Pipeline from Specific Step

```bash
# Resume from phenotype filtering
python3 tcp/preprocessing/run_pipeline.py --start-from filter_phenotype

# Stop at subject filtering (don't download MRI data)
python3 tcp/preprocessing/run_pipeline.py --stop-at filter_subjects
```

### Skip Optional Steps

```bash
# Skip phenotype filtering (use all validated subjects)
python3 tcp/preprocessing/run_pipeline.py --skip-optional
```

### Custom Phenotype Filtering

```bash
# Custom age range and diagnosis filtering
python3 tcp/preprocessing/filter_phenotype.py --min-age 25 --max-age 55 --include-mdd

# Custom column-based filtering
python3 tcp/preprocessing/filter_phenotype.py --custom-filters demos:sex:F demos:Site:UCLA
```

### Selective Data Download

```bash
# Download only essential task data (faster)
python3 tcp/preprocessing/fetch_filtered_data.py --data-types raw_nifti_hammer raw_nifti_stroop events_hammer events_stroop

# Download everything including anatomical scans
python3 tcp/preprocessing/fetch_filtered_data.py --data-types raw_nifti_hammer raw_nifti_stroop events_hammer events_stroop anatomical_t1w json_metadata_hammer json_metadata_stroop
```

## Migration from Legacy Pipeline

If you were using the old `extract_subjects.py` pipeline:

1. **New pipeline is backward compatible** - existing analysis scripts will work
2. **Run the new pipeline** for better optimization:
   ```bash
   python3 tcp/preprocessing/run_pipeline.py
   ```
3. **Benefits**: Reduced downloads, better filtering, progress tracking

## Troubleshooting

### Pipeline State Management
The pipeline saves its state in `pipeline_state.json`. To reset:
```bash
rm -rf /path/to/output/tcp_preprocessing/pipeline/
```

### Resume Failed Pipeline
```bash
# Check what failed and resume
python3 tcp/preprocessing/run_pipeline.py
```

### Manual Step Execution
If the orchestrator fails, run individual steps:
```bash
python3 tcp/preprocessing/initialize_dataset.py
python3 tcp/preprocessing/validate_subjects.py
# etc.
```

### Check Pipeline Status
```bash
# Dry run shows current state
python3 tcp/preprocessing/run_pipeline.py --dry-run
```
