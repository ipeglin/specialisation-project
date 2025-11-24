# TCP Preprocessing Pipeline for Anhedonia-Focused Analysis

This document describes the TCP preprocessing pipeline optimized for anhedonia research. The pipeline performs filtering **before** downloading large MRI files, reducing download requirements from ~1TB to only the data you need.

## Overview

The pipeline consists of 12 steps that filter subjects, classify them by anhedonia severity and diagnosis, generate analysis groups, and selectively download data:

1. **Dataset Initialization** - Clone and validate dataset
2. **Subject Validation** - Check subject directory structure
3. **Global Data Fetching** - Download metadata files (participants.tsv, phenotype data)
4. **Task Data Filtering** - Filter subjects with hammer/stroop task data
5. **Base Subject Filtering** - Apply universal inclusion criteria (SHAPS completion)
6. **Anhedonia Classification** - Classify by anhedonia severity (SHAPS scores)
7. **Diagnosis Classification** - Classify by MDD diagnosis status
8. **Analysis Group Generation** - Generate 4 research groups (Primary/Secondary/Tertiary/Quaternary)
9. **Subject Sampling** - Sample subjects for download (development/production/custom modes)
10. **File Path Mapping** - Map all data files for selected subjects
11. **Cross-Analysis Integration** (Optional) - Generate cross-analysis statistics
12. **MRI Data Fetching** - Download actual MRI files for sampled subjects

**Key Design Principle**: The pipeline is anhedonia-focused but group-agnostic during filtering. Subjects are classified and organized for analysis, but data organization remains unified.

## Quick Start

### Option 1: Run Complete Pipeline (Recommended)

```bash
# Run the entire pipeline with default settings (development mode - minimal subjects)
python3 tcp/preprocessing/run_pipeline.py

# Run in production mode (download ALL subjects)
python3 tcp/preprocessing/run_pipeline.py --sample-mode production --analysis-groups primary

# Run in custom mode (N subjects per anhedonia class)
python3 tcp/preprocessing/run_pipeline.py --sample-mode custom --analysis-groups primary

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

# Step 4: Filter subjects with task data
python3 tcp/preprocessing/filter_subjects.py

# Step 5: Filter subjects by SHAPS completion
python3 tcp/preprocessing/filter_base_subjects.py

# Step 6: Classify subjects by anhedonia severity
python3 tcp/preprocessing/classify_anhedonia.py

# Step 7: Classify subjects by MDD diagnosis
python3 tcp/preprocessing/classify_diagnoses.py

# Step 8: Generate analysis groups
python3 tcp/preprocessing/generate_analysis_groups.py

# Step 9: Sample subjects for download
python3 tcp/preprocessing/sample_subjects_for_download.py --sample-mode development --analysis-groups primary

# Step 10: Map file paths for sampled subjects
python3 tcp/preprocessing/map_subject_files.py

# Step 11: (Optional) Generate cross-analysis statistics
python3 tcp/preprocessing/integrate_cross_analysis.py

# Step 12: Download MRI data for sampled subjects
python3 tcp/preprocessing/fetch_filtered_data.py --data-types timeseries --tasks hammer
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

**Estimated time**: 10-15 minutes

### 2. Subject Validation (`validate_subjects.py`)

**Purpose**: Validates which subjects have proper BIDS directory structure.

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

**Estimated time**: 2-5 minutes

### 3. Global Data Fetching (`fetch_global_data.py`)

**Purpose**: Downloads metadata files needed for filtering and classification.

```bash
python3 tcp/preprocessing/fetch_global_data.py
```

**What it does**:
- Downloads `participants.tsv` using datalad get
- Downloads `phenotype/demos.tsv` for demographic/diagnosis information
- Downloads `phenotype/shaps.tsv` for anhedonia scores
- Downloads other phenotype files if available
- Validates downloaded files are readable

**Output**: Global metadata files downloaded and validated

**Estimated time**: 1-3 minutes

### 4. Task Data Filtering (`filter_subjects.py`)

**Purpose**: Filter subjects who have hammer and/or stroop task fMRI data.

```bash
# Default: Require at least one task (hammer OR stroop)
python3 tcp/preprocessing/filter_subjects.py

# Require both tasks
python3 tcp/preprocessing/filter_subjects.py --require-all-tasks

# Check specific data types
python3 tcp/preprocessing/filter_subjects.py --data-types raw_nifti events timeseries
```

**What it does**:
- Scans dataset for task-specific files (hammer/stroop)
- Filters subjects with at least one task (default) or all tasks
- Checks across multiple data types (raw NIFTI, timeseries, events)
- Tracks detailed inclusion/exclusion reasons

**Output**:
- `task_filtered_subjects.csv` - Subjects with task data
- `task_excluded_subjects.csv` - Subjects without required task data
- `task_filtering_summary.json` - Detailed filtering statistics

**Estimated time**: 5-15 minutes

### 5. Base Subject Filtering (`filter_base_subjects.py`)

**Purpose**: Apply universal inclusion criteria - primarily SHAPS questionnaire completion.

```bash
python3 tcp/preprocessing/filter_base_subjects.py
```

**What it does**:
- Loads SHAPS phenotype data
- Filters subjects with valid SHAPS total scores (0-14, not missing/999)
- Applies universal inclusion criteria for the anhedonia pipeline
- Excludes subjects with incomplete or invalid questionnaire data

**Output**:
- `base_filtered_subjects.csv` - Subjects meeting inclusion criteria
- `base_excluded_subjects.csv` - Subjects failing criteria with reasons
- `base_filtering_summary.json` - Statistics

**Estimated time**: 1-2 minutes

### 6. Anhedonia Classification (`classify_anhedonia.py`)

**Purpose**: Classify subjects into anhedonia severity classes based on SHAPS total scores.

```bash
python3 tcp/preprocessing/classify_anhedonia.py
```

**What it does**:
- Reads base-filtered subjects
- Classifies subjects into three categories based on SHAPS scores:
  - **non-anhedonic**: SHAPS total score 0-2
  - **low-anhedonic**: SHAPS total score 3-8
  - **high-anhedonic**: SHAPS total score 9-14
- Adds `anhedonia_class` and `anhedonic_status` columns
- Tracks classification for downstream analysis

**Output**:
- `anhedonia_classified_subjects.csv` - All subjects with anhedonia classifications
- `anhedonia_classification_summary.json` - Class distribution statistics

**Classification Criteria**:
- Based on established SHAPS scoring ranges
- Provides systematic comparison of anhedonic symptom levels
- Used for Primary analysis group formation

**Estimated time**: 1-2 minutes

### 7. Diagnosis Classification (`classify_diagnoses.py`)

**Purpose**: Classify subjects by MDD diagnosis status using phenotype data.

```bash
python3 tcp/preprocessing/classify_diagnoses.py
```

**What it does**:
- Reads anhedonia-classified subjects
- Extracts diagnosis information from `demos.tsv`
- Classifies subjects into diagnostic categories:
  - **MDD primary**: Primary diagnosis is MDD (no comorbid diagnoses)
  - **MDD comorbid**: MDD with comorbid diagnoses
  - **MDD past**: Past MDD diagnosis
  - **Control**: No MDD diagnosis (Primary_Dx = "999")
- Adds `mdd_status` column for analysis group assignment

**Output**:
- `diagnosis_classified_subjects.csv` - Subjects with diagnosis classifications
- `diagnosis_classification_summary.json` - Diagnosis distribution statistics

**Estimated time**: 1-2 minutes

### 8. Analysis Group Generation (`generate_analysis_groups.py`)

**Purpose**: Generate 4 research-focused analysis groups based on anhedonia and diagnosis.

```bash
python3 tcp/preprocessing/generate_analysis_groups.py
```

**What it does**:
- Creates 4 distinct analysis groups:
  - **PRIMARY**: Anhedonia analysis (non-anhedonic vs low-anhedonic vs high-anhedonic)
  - **SECONDARY**: MDD + Anhedonia (MDD primary with anhedonia classes)
  - **TERTIARY**: Diagnosis comparison (MDD primary vs comorbid vs control)
  - **QUATERNARY**: Full spectrum (MDD primary vs comorbid vs past vs control)
- Each group is self-contained and analysis-ready
- Enables multiple research questions from single preprocessing run

**Output**:
- `primary_analysis_subjects.csv` - Primary anhedonia group
- `secondary_analysis_subjects.csv` - MDD + anhedonia group
- `tertiary_analysis_subjects.csv` - Diagnosis comparison group
- `quaternary_analysis_subjects.csv` - Full diagnostic spectrum group
- `analysis_groups_summary.json` - Group composition statistics

**Analysis Group Details**:

| Group | Focus | Subjects | Comparison |
|-------|-------|----------|------------|
| Primary | Anhedonia severity | All with SHAPS | non-anhedonic vs low vs high |
| Secondary | MDD + Anhedonia | MDD primary only | MDD with varying anhedonia |
| Tertiary | Diagnosis types | MDD + Controls | Primary vs comorbid vs control |
| Quaternary | Full spectrum | All diagnostic categories | Primary vs comorbid vs past vs control |

**Estimated time**: 1-2 minutes

### 9. Subject Sampling (`sample_subjects_for_download.py`)

**Purpose**: Sample subjects for data download based on research needs and storage constraints.

```bash
# Development mode: Minimal subjects for testing (1 per category) - ~15GB
python3 tcp/preprocessing/sample_subjects_for_download.py --sample-mode development --analysis-groups primary

# Production mode: ALL subjects from selected groups - ~300GB per group
python3 tcp/preprocessing/sample_subjects_for_download.py --sample-mode production --analysis-groups primary

# Custom mode: N subjects per category for balanced testing
python3 tcp/preprocessing/sample_subjects_for_download.py --sample-mode custom --analysis-groups primary secondary

# Select multiple analysis groups
python3 tcp/preprocessing/sample_subjects_for_download.py --sample-mode production --analysis-groups primary secondary tertiary
```

**What it does**:
- Implements three sampling strategies:
  - **Development**: 1 subject per category (~15GB) - fast testing
  - **Production**: ALL subjects from selected groups (~300GB) - full analysis
  - **Custom**: N subjects per category (configurable via `--subjects-per-group`)
- Samples from one or more analysis groups
- Uses random seed for reproducibility
- Prioritizes subjects by data quality

**Sampling Examples**:
- **Development + Primary**: 3 subjects (1 non-anhedonic, 1 low, 1 high)
- **Custom + Primary (N=10)**: 30 subjects (10 per anhedonia class)
- **Production + Primary**: ALL subjects with anhedonia classifications
- **Production + All groups**: Complete dataset across all 4 analysis groups

**Output**:
- `sampled_subjects_for_download.csv` - Selected subjects for data download
- `sampling_summary.json` - Sampling strategy and statistics
- Separate CSV files per category for reference

**Critical Note**: This step determines which subjects' MRI data will be downloaded. Choose sampling mode based on:
- **Development**: Local testing, method development
- **Custom**: Balanced testing with specific sample sizes
- **Production**: Full analysis, cluster computing, publication-ready results

**Estimated time**: 30 seconds

### 10. File Path Mapping (`map_subject_files.py`)

**Purpose**: Create comprehensive mapping of all data file paths for sampled subjects.

```bash
python3 tcp/preprocessing/map_subject_files.py
```

**What it does**:
- Reads sampled subjects from previous step
- Scans dataset directory structure for each subject
- Discovers all available data files (raw NIFTI, events, anatomical, timeseries, JSON metadata)
- Handles both BIDS format (`sub-NDARINV...`) and timeseries format (`NDAR_INV...`)
- Creates JSON mapping for efficient downstream processing
- Identifies subjects missing expected data

**Output**:
- `subject_file_mapping.json` - **Main output**: Complete file paths for all sampled subjects and global files
- `file_mapping_summary.json` - Statistics on file availability
- `missing_files_report.csv` - Subjects missing expected files

**Why this step**: Pre-computing file paths makes the fetch step much faster and more reliable. The fetch script reads this mapping instead of discovering files at runtime.

**Estimated time**: 1-10 minutes (depends on number of sampled subjects)

### 11. Cross-Analysis Integration (`integrate_cross_analysis.py`) - Optional

**Purpose**: Generate cross-analysis statistics and integrated datasets for comparison across analysis groups.

```bash
python3 tcp/preprocessing/integrate_cross_analysis.py
```

**What it does**:
- Loads all 4 analysis groups
- Computes cross-group statistics and overlaps
- Identifies subjects appearing in multiple groups
- Generates master summary of entire preprocessing pipeline
- Creates visualization-ready data exports

**Output**:
- `cross_analysis_master_summary.json` - Complete statistics across all groups
- `cross_analysis_subject_mapping.csv` - Subject-to-group membership table
- `cross_analysis_overlap_report.json` - Inter-group overlap analysis

**When to use**: Run this step when you want comprehensive statistics across all analysis groups or need to understand subject overlap between groups. Skip if only using one analysis group.

**Estimated time**: 1-2 minutes

### 12. MRI Data Fetching (`fetch_filtered_data.py`)

**Purpose**: Download actual MRI files for sampled subjects.

```bash
# Default: Download ALL data types (raw NIFTI, events, JSON, anatomical, timeseries)
python3 tcp/preprocessing/fetch_filtered_data.py

# Download only timeseries for hammer task (fastest for analysis)
python3 tcp/preprocessing/fetch_filtered_data.py --data-types timeseries --tasks hammer

# Download specific data types
python3 tcp/preprocessing/fetch_filtered_data.py --data-types raw_nifti events json_metadata --tasks hammer stroop

# Dry run to see what would be downloaded
python3 tcp/preprocessing/fetch_filtered_data.py --dry-run

# Include anatomical scans
python3 tcp/preprocessing/fetch_filtered_data.py --data-types raw_nifti events anatomical timeseries --tasks hammer
```

**What it does**:
- Reads file mapping from `map_subject_files.py`
- Downloads only files for sampled subjects
- Uses `datalad get` for efficient fetching
- Supports selective data type downloading
- Provides progress tracking and error handling
- Validates downloaded files
- Skips already-downloaded files

**Data Types Available**:
- `raw_nifti` - Task fMRI NIFTI files (4D volumes)
- `events` - Task timing/condition files
- `json_metadata` - BIDS sidecar JSON files
- `anatomical` - Structural scans (T1w/T2w)
- `timeseries` - Parcellated timeseries (H5 format) - **recommended for FC analysis**

**Tasks Available**:
- `hammer` - Hammer task (emotional faces)
- `stroop` - Stroop task (cognitive control)
- `t1w`, `t2w` - Anatomical scan types

**Download Size Estimates**:
- Timeseries only: ~500MB per subject
- Raw NIFTI + events + JSON: ~2-3GB per subject
- Full data (all types): ~4-5GB per subject

**Estimated time**: 2-10 hours (depends on sampling mode and data types)

## Pipeline Benefits

### 🚀 **Reduced Downloads**
- Sample subjects before downloading MRI data
- Development mode: ~15GB vs Production mode: ~300GB per group
- Custom sampling: balance between testing and full analysis
- Only download data for subjects of interest

### 🎯 **Anhedonia-Focused**
- Systematic classification by SHAPS scores
- Four analysis-ready research groups
- Handles full diagnostic spectrum (MDD, comorbidity, controls)
- Supports multiple research questions from single preprocessing

### 🔧 **Modular Design**
- Each step is independent and reusable
- Can resume pipeline from any step
- Easy to add new filtering criteria
- Clean separation of concerns

### 📊 **Comprehensive Tracking**
- Detailed inclusion/exclusion reasons at each step
- Progress reporting and error handling
- Pipeline state saving and resumption
- Cross-analysis integration for multi-group studies

### 🔄 **Flexible Sampling**
- Development mode for local testing
- Production mode for complete analysis
- Custom mode for balanced sample sizes
- Reproducible sampling with random seeds

### 🔬 **Analysis-Ready Output**
- Pre-classified subjects by anhedonia and diagnosis
- Four distinct analysis groups ready for research
- File paths pre-mapped for efficient data loading
- Compatible with downstream analysis scripts

## Advanced Usage

### Resume Pipeline from Specific Step

```bash
# Resume from subject sampling
python3 tcp/preprocessing/run_pipeline.py --start-from sample_subjects

# Resume from file mapping
python3 tcp/preprocessing/run_pipeline.py --start-from map_subject_files

# Stop at file mapping (don't download MRI data yet)
python3 tcp/preprocessing/run_pipeline.py --stop-at map_subject_files
```

### Skip Optional Steps

```bash
# Skip optional cross-analysis integration step
python3 tcp/preprocessing/run_pipeline.py --skip-optional
```

### Custom Sampling Configurations

```bash
# Development mode with multiple groups
python3 tcp/preprocessing/run_pipeline.py --sample-mode development --analysis-groups primary secondary

# Production mode for Primary group only
python3 tcp/preprocessing/run_pipeline.py --sample-mode production --analysis-groups primary

# Custom mode: 20 subjects per anhedonia class
python3 tcp/preprocessing/sample_subjects_for_download.py --sample-mode custom --subjects-per-group 20 --analysis-groups primary
```

### Selective Data Download

```bash
# Download only timeseries (fastest, recommended for FC analysis)
python3 tcp/preprocessing/fetch_filtered_data.py --data-types timeseries --tasks hammer

# Download task fMRI with metadata
python3 tcp/preprocessing/fetch_filtered_data.py --data-types raw_nifti events json_metadata --tasks hammer stroop

# Download everything including anatomical scans
python3 tcp/preprocessing/fetch_filtered_data.py --data-types raw_nifti events json_metadata anatomical timeseries --tasks hammer stroop
```

### Cluster Computing Setup

For running analysis on a cluster with more subjects:

```bash
# Step 1: Run sampling in custom/production mode
python3 tcp/preprocessing/sample_subjects_for_download.py \
    --sample-mode custom \
    --subjects-per-group 50 \
    --analysis-groups primary

# Step 2: Map files for all sampled subjects
python3 tcp/preprocessing/map_subject_files.py

# Step 3: Download data on cluster (or locally first then transfer)
python3 tcp/preprocessing/fetch_filtered_data.py \
    --data-types timeseries \
    --tasks hammer
```

## Understanding Analysis Groups

### PRIMARY: Anhedonia Severity Analysis
**Research Question**: How does anhedonia severity affect brain function?

**Subjects**: All subjects with valid SHAPS scores

**Groups**:
- Non-anhedonic (SHAPS 0-2)
- Low-anhedonic (SHAPS 3-8)
- High-anhedonic (SHAPS 9-14)

**Use Cases**:
- Anhedonia-specific functional connectivity
- Reward processing comparisons
- Dimensional anhedonia research

### SECONDARY: MDD + Anhedonia Analysis
**Research Question**: How does anhedonia vary within MDD patients?

**Subjects**: MDD primary diagnosis only (no comorbidity)

**Groups**:
- MDD + non-anhedonic
- MDD + low-anhedonic
- MDD + high-anhedonic

**Use Cases**:
- Anhedonia as MDD subtype
- Treatment response prediction
- Cleaner diagnostic cohort (no comorbidity)

### TERTIARY: Diagnosis Comparison
**Research Question**: How do MDD subtypes differ from controls?

**Subjects**: MDD patients and healthy controls

**Groups**:
- MDD primary (no comorbidity)
- MDD comorbid (with other diagnoses)
- Controls (no diagnosis)

**Use Cases**:
- MDD vs control comparisons
- Comorbidity effects
- Diagnostic validity studies

### QUATERNARY: Full Diagnostic Spectrum
**Research Question**: What are the differences across the full diagnostic spectrum?

**Subjects**: All subjects with valid data

**Groups**:
- MDD primary
- MDD comorbid
- MDD past (remitted)
- Controls

**Use Cases**:
- Full spectrum analysis
- Remission studies
- Dimensional approaches to psychopathology

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
If the orchestrator fails, run individual steps in order:
```bash
python3 tcp/preprocessing/initialize_dataset.py
python3 tcp/preprocessing/validate_subjects.py
python3 tcp/preprocessing/fetch_global_data.py
python3 tcp/preprocessing/filter_subjects.py
python3 tcp/preprocessing/filter_base_subjects.py
python3 tcp/preprocessing/classify_anhedonia.py
python3 tcp/preprocessing/classify_diagnoses.py
python3 tcp/preprocessing/generate_analysis_groups.py
python3 tcp/preprocessing/sample_subjects_for_download.py --sample-mode development --analysis-groups primary
python3 tcp/preprocessing/map_subject_files.py
python3 tcp/preprocessing/integrate_cross_analysis.py  # Optional
python3 tcp/preprocessing/fetch_filtered_data.py --data-types timeseries --tasks hammer
```

### Check Pipeline Status
```bash
# Dry run shows current state without executing
python3 tcp/preprocessing/run_pipeline.py --dry-run
```

### Adjust Sampling for More Subjects
```bash
# Increase subjects per group
python3 tcp/preprocessing/sample_subjects_for_download.py --sample-mode custom --subjects-per-group 50 --analysis-groups primary

# Then re-run file mapping and download
python3 tcp/preprocessing/map_subject_files.py
python3 tcp/preprocessing/fetch_filtered_data.py --data-types timeseries --tasks hammer
```

### Validate Downloaded Data
```bash
# Check if files are actually downloaded (not git-annex symlinks)
python3 tcp/preprocessing/fetch_filtered_data.py --dry-run
```

## Common Workflows

### Workflow 1: Quick Local Testing
```bash
# Development mode: 3 subjects from Primary group
python3 tcp/preprocessing/run_pipeline.py --sample-mode development --analysis-groups primary --data-types timeseries --tasks hammer
```

### Workflow 2: Balanced Sample for Methods Development
```bash
# Custom mode: 10 subjects per anhedonia class (30 total)
python3 tcp/preprocessing/run_pipeline.py --sample-mode custom --analysis-groups primary --data-types timeseries --tasks hammer
```

### Workflow 3: Full Analysis on Cluster
```bash
# Production mode: ALL subjects from Primary group
python3 tcp/preprocessing/run_pipeline.py --sample-mode production --analysis-groups primary --data-types timeseries --tasks hammer
```

### Workflow 4: Multi-Group Comparison Study
```bash
# Production mode: Multiple analysis groups
python3 tcp/preprocessing/run_pipeline.py --sample-mode production --analysis-groups primary secondary tertiary --data-types timeseries --tasks hammer
```

## Output Directory Structure

```
/path/to/output/tcp_preprocessing/
├── pipeline/                          # Pipeline orchestrator state
│   └── pipeline_state.json           # Resume information
├── initialize_dataset/                # Step 1 outputs
├── validate_subjects/                 # Step 2 outputs
├── fetch_global_data/                 # Step 3 outputs
├── filter_subjects/                   # Step 4 outputs
│   └── task_filtered_subjects.csv
├── filter_base_subjects/              # Step 5 outputs
│   └── base_filtered_subjects.csv
├── classify_anhedonia/                # Step 6 outputs
│   └── anhedonia_classified_subjects.csv
├── classify_diagnoses/                # Step 7 outputs
│   └── diagnosis_classified_subjects.csv
├── generate_analysis_groups/          # Step 8 outputs
│   ├── primary_analysis_subjects.csv
│   ├── secondary_analysis_subjects.csv
│   ├── tertiary_analysis_subjects.csv
│   └── quaternary_analysis_subjects.csv
├── sample_subjects_for_download/      # Step 9 outputs
│   └── sampled_subjects_for_download.csv
├── map_subject_files/                 # Step 10 outputs
│   └── subject_file_mapping.json
├── integrate_cross_analysis/          # Step 11 outputs (optional)
│   └── cross_analysis_master_summary.json
└── fetch_filtered_data/               # Step 12 outputs
    └── fetch_report_YYYYMMDD_HHMMSS.json
```

## Next Steps After Preprocessing

Once preprocessing is complete, you can:

1. **Load subject data** using the DataLoader class:
   ```python
   from tcp.processing import DataLoader
   loader = DataLoader()
   subjects = loader.get_subjects_by_group('primary')
   ```

2. **Run functional connectivity analysis**:
   ```python
   python3 tcp/processing/fc_mvp.py
   ```

3. **Access preprocessed timeseries** directly from the dataset

4. **Use analysis groups** for group comparisons and statistical analysis
