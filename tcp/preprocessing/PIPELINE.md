# TCP Pre-processing Pipeline for Task-based Analysis

This document describes the optimized TCP preprocessing pipeline that minimizes data downloads and provides early filtering capabilities.

## Overview

The new pipeline performs filtering **before** downloading large MRI files, reducing download requirements from ~1TB to only the data you need. The pipeline consists of the following steps:

1. **Dataset Initialization** - Clone and validate dataset
2. **Subject Validation** - Check subject directory structure
3. **Global Data Fetching** - Download metadata files for filtering
4. **Phenotype Filtering** (Optional) - Filter by diagnosis (MDD vs controls)
5. **Task Data Filtering** - Filter subjects with hammer/stroop task data (group-agnostic)
6. **Anhedonia Segmentation** (Optional) - Classify subjects by anhedonia severity
7. **Group Summarization** (Optional) - Generate patient/control statistics
8. **File Path Mapping** - Map all data files for filtered subjects
9. **MRI Data Fetching** - Download actual MRI files for selected subjects

**Key Design Principle**: Filtering is group-agnostic. All subjects are processed equally during filtering, regardless of patient/control status. Group classification is only used for analytical summaries and does not affect data organization.

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

# Step 5: Filter subjects with task data (group-agnostic)
python3 tcp/preprocessing/filter_subjects.py

# Step 6: (Optional) Segment subjects by anhedonia severity
python3 tcp/preprocessing/anhedonia_segmentation.py

# Step 7: (Optional) Summarize patient/control groups
python3 tcp/preprocessing/summarize_groups.py

# Step 8: Map file paths for all filtered subjects
python3 tcp/preprocessing/map_subject_files.py

# Step 9: Download MRI data for filtered subjects
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

### 5. Task Data Filtering (`filter_subjects.py`) - **REFACTORED**

**Purpose**: Filter subjects who have hammer and/or stroop task fMRI data. **Group-agnostic filtering** - processes all subjects equally.

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
- **NEW**: Outputs unified subject list (no patient/control split)
- Tracks which filters were applied and detailed inclusion/exclusion reasons

**Input Auto-detection**:
1. Uses phenotype filtered subjects (if available)
2. Falls back to validated subjects
3. Falls back to legacy extract_subjects output (backward compatible)

**Output** (New unified format):
- `task_filtered_subjects.csv` - All subjects passing task filters (unified list)
- `task_excluded_subjects.csv` - All subjects failing task filters
- `task_filtering_summary.json` - Detailed filtering statistics
- `filtering_reasons.json` - Per-subject inclusion/exclusion reasons

**Note**: Group column (Patient/GenPop) is preserved in output CSV for later analysis, but filtering is group-agnostic.

### 6. Anhedonia Segmentation (`anhedonia_segmentation.py`) - **NEW** (Optional)

**Purpose**: Classify subjects into anhedonia severity classes based on SHAPS total scores. **This is purely for data organization** - it helps categorize subjects for anhedonia-focused analysis.

```bash
# Segment subjects by anhedonia severity
python3 tcp/preprocessing/anhedonia_segmentation.py
```

**What it does**:
- Reads task-filtered subjects from `filter_subjects` output
- Loads SHAPS questionnaire data to get total anhedonia scores
- Classifies subjects into three categories:
  - **non-anhedonic**: SHAPS total score 0-2
  - **low-anhedonic**: SHAPS total score 3-8  
  - **high-anhedonic**: SHAPS total score 9-14
- Excludes subjects with missing or invalid SHAPS scores (score = 999 or outside 0-14 range)
- Adds `anhedonia_class` column to subject data for downstream analysis

**Output**:
- `anhedonia_segmented_subjects.csv` - All subjects with valid classifications and anhedonia_class column
- `anhedonia_excluded_subjects.csv` - Subjects with missing/invalid SHAPS scores
- `non_anhedonic_subjects.csv`, `low_anhedonic_subjects.csv`, `high_anhedonic_subjects.csv` - Separate files by anhedonia class
- `anhedonia_segmentation_summary.json` - Detailed statistics and class distribution
- `segmentation_reasons.json` - Per-subject classification reasons

**When to use**: Run this step when you want to organize subjects by anhedonia severity for research focused on anhedonic symptoms. This classification is based on established SHAPS score ranges and provides a systematic way to compare different levels of anhedonic presentation.

### 7. Group Summarization (`summarize_groups.py`) - **NEW** (Optional)

**Purpose**: Generate analytical summary of patient/control group composition. **This is purely for understanding your data** - it does not affect data organization or downstream processing.

```bash
# Generate group statistics
python3 tcp/preprocessing/summarize_groups.py
```

**What it does**:
- Reads filtered subjects from `filter_subjects` output
- Loads phenotype data to get Group classifications (Patient vs GenPop)
- Calculates demographics by group (age, sex, site, diagnosis)
- Computes task availability statistics per group
- Creates both machine-readable (JSON) and human-readable (TXT) reports

**Output**:
- `group_summary.json` - Complete statistics breakdown
- `group_breakdown.txt` - Human-readable report
- `subjects_with_groups.csv` - Flat table with group assignments

**When to use**: Run this step when you want to understand how many patients vs controls passed filtering, and their demographic characteristics. Skip this step if you don't need group-based statistics yet.

### 8. File Path Mapping (`map_subject_files.py`) - **NEW**

**Purpose**: Create comprehensive mapping of all data file paths for filtered subjects. **Group-agnostic** - maps files for all filtered subjects regardless of classification.

```bash
# Map all file paths
python3 tcp/preprocessing/map_subject_files.py
```

**What it does**:
- Scans dataset directory structure for each filtered subject
- Discovers all available data files (raw NIFTI, events, anatomical, timeseries)
- Handles both BIDS format (`sub-NDARINV...`) and timeseries format (`NDAR_INV...`)
- Creates JSON mapping for efficient downstream processing
- Identifies subjects missing expected data

**Output**:
- `subject_file_mapping.json` - **Main output**: Complete file paths for all subjects and global files
- `file_mapping_summary.json` - Statistics on file availability
- `missing_files_report.csv` - Subjects missing expected files

**Why this step**: Pre-computing file paths makes the fetch step much faster and more reliable. The fetch script can simply read this mapping instead of discovering files at runtime.

### 9. MRI Data Fetching (`fetch_filtered_data.py`)

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
- Group-agnostic task availability filtering
- Customizable filtering criteria

### 🔧 **Modular Design**
- Each step is independent and reusable
- Can resume pipeline from any step
- Easy to add new filtering criteria
- Clean separation of concerns: Filtering → Analysis → Mapping → Fetching

### 📊 **Comprehensive Tracking**
- Detailed inclusion/exclusion reasons at each step
- Progress reporting and error handling
- Pipeline state saving and resumption
- Optional group statistics for understanding data composition

### 🔄 **Backward Compatible**
- Works with existing analysis scripts
- Maintains same output format as original pipeline
- Legacy extract_subjects.py still supported

### 🔬 **Group-Agnostic Architecture**
- All subjects treated equally during filtering
- Group classification decoupled from data organization
- Optional analytical summaries for patient/control groups
- No group-based subdirectories - simplified file structure

## Advanced Usage

### Resume Pipeline from Specific Step

```bash
# Resume from phenotype filtering
python3 tcp/preprocessing/run_pipeline.py --start-from filter_phenotype

# Resume from task filtering
python3 tcp/preprocessing/run_pipeline.py --start-from filter_subjects

# Stop at file mapping (don't download MRI data yet)
python3 tcp/preprocessing/run_pipeline.py --stop-at map_subject_files
```

### Skip Optional Steps

```bash
# Skip optional steps (phenotype filtering and group summarization)
python3 tcp/preprocessing/run_pipeline.py --skip-optional
```

### Task Filtering Options

```bash
# Default: Require at least one task (hammer OR stroop)
python3 tcp/preprocessing/filter_subjects.py

# Require both tasks for inclusion
python3 tcp/preprocessing/filter_subjects.py --require-all-tasks

# Check only specific data types
python3 tcp/preprocessing/filter_subjects.py --data-types raw_nifti events
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
3. **Key Changes**:
   - Unified subject lists (no patient/control split during filtering)
   - Optional group summarization step for statistics
   - Pre-computed file path mapping for faster fetching
   - Group-agnostic data organization
4. **Benefits**: Reduced downloads, better filtering, cleaner architecture, progress tracking

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
python3 tcp/preprocessing/fetch_global_data.py
python3 tcp/preprocessing/filter_phenotype.py  # Optional
python3 tcp/preprocessing/filter_subjects.py
python3 tcp/preprocessing/anhedonia_segmentation.py  # Optional
python3 tcp/preprocessing/summarize_groups.py  # Optional
python3 tcp/preprocessing/map_subject_files.py
python3 tcp/preprocessing/fetch_filtered_data.py
```

### Check Pipeline Status
```bash
# Dry run shows current state
python3 tcp/preprocessing/run_pipeline.py --dry-run
```
