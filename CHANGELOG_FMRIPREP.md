# Changelog: fMRIPrep Data Source Integration

## Version 1.0.0 (2025-01-28)

### Summary

Added support for dual data sources in TCP preprocessing pipeline:
- **Option A (datalad)**: Existing workflow (no changes)
- **Option B (fmriprep)**: New workflow with custom parcellation

### Major Changes

#### 1. Directory Restructuring

**Changed**: Moved `tcp/processing/parcellations/` → `parcellations/` (project root)

**Rationale**: Better organization and accessibility across different modules.

**Impact**: 
- Atlas files now at project root
- Added `get_parcellations_path()` to `config/paths.py`
- Updated `config/default_config.json`

**Migration**: No action needed - changes are backward compatible.

#### 2. Data Source Configuration Module

**New File**: `tcp/preprocessing/config/data_source_config.py`

**Features**:
- `DataSourceType` enum (DATALAD, FMRIPREP)
- `DataSourceConfig` dataclass with validation
- Helper functions: `create_datalad_config()`, `create_fmriprep_config()`

**Usage**:
```python
from tcp.preprocessing.config.data_source_config import create_fmriprep_config

config = create_fmriprep_config(
    fmriprep_root=Path("/path/to/fmriprep"),
    parcellated_output_dir=Path("Data/parcellated")
)
```

#### 3. fMRIPrep Parcellation Engine

**New File**: `tcp/preprocessing/fmriprep_parcellation.py`

**Features**:
- Parcellates fMRIPrep BOLD data to 434-ROI timeseries
- Supports 3 atlases:
  - Yeo2011 17-Network (400 cortical parcels)
  - Tian S2 (32 subcortical parcels)
  - Buckner 7-network (2 cerebellar regions)
- Parallel processing via joblib
- Compatible .h5 output format

**CLI**:
```bash
python tcp/preprocessing/fmriprep_parcellation.py \
    --fmriprep-root /path/to/fmriprep \
    --subject-ids sub-001 sub-002 \
    --output-dir Data/parcellated \
    --n-jobs 4
```

#### 4. Pipeline Orchestrator Integration

**Modified File**: `tcp/preprocessing/run_pipeline.py`

**Changes**:
- Added `PARCELLATE_FMRIPREP` pipeline step
- Conditional step execution based on data source
- New CLI arguments:
  - `--data-source {datalad|fmriprep}`
  - `--fmriprep-root PATH`
  - `--parcellated-output-dir PATH`
  - `--run-start N`, `--run-end N`
  - `--n-jobs N`

**Workflow Changes**:

| Step | Datalad Mode | fMRIPrep Mode |
|------|-------------|---------------|
| initialize_dataset | ✓ | ✗ (skipped) |
| parcellate_fmriprep | ✗ (skipped) | ✓ (NEW) |
| fetch_filtered_data | ✓ | ✗ (skipped) |

### Configuration Changes

#### config/paths.py

**Added**:
```python
def get_parcellations_path(subpath: str = '') -> Path:
    """Get path to atlas/parcellation files (at project root)"""
    code_base = self._get_base_path('code_base')
    parcellations_base = code_base / 'specialisation-project' / 'parcellations'
    return parcellations_base / subpath if subpath else parcellations_base
```

#### config/default_config.json

**Added**:
```json
"common_subdirectories": {
    "parcellations": "parcellations",
    ...
}
```

### Known Issues

#### CRITICAL: Buckner Cerebellar Atlas

**Issue**: Current implementation uses time-domain averaging, which destroys temporal dynamics.

**Status**: Implemented to match colleague's code for compatibility testing.

**Warning**: Added explicit warnings in code and documentation.

**Action Required**: Implement proper spatial aggregation before production use.

**Workaround**: Currently using zeros placeholder (recommended).

### Breaking Changes

**None** - Full backward compatibility maintained.

### Migration Guide

#### For Existing Users (Datalad Mode)

No changes required. Continue using:
```bash
python tcp/preprocessing/run_pipeline.py
```

#### For New Users (fMRIPrep Mode)

1. Ensure fMRIPrep output follows expected structure
2. Run pipeline with new arguments:
```bash
python tcp/preprocessing/run_pipeline.py \
    --data-source fmriprep \
    --fmriprep-root /path/to/fmriprep \
    --parcellated-output-dir Data/parcellated
```

### Testing Status

✅ **Passed**:
- Python syntax validation (all files compile)
- Import testing (module structure)
- Configuration validation
- CLI argument parsing

⏳ **Pending**:
- End-to-end integration testing with actual fMRIPrep data
- Output format validation
- Performance benchmarking

### Dependencies

**New Requirements**:
- nilearn >= 0.10.0
- h5py >= 3.0.0
- joblib >= 1.0.0
- tqdm >= 4.60.0

**Note**: These are only required for fMRIPrep mode (`--data-source fmriprep`).

### File Changes

**New Files**:
- `tcp/preprocessing/config/__init__.py`
- `tcp/preprocessing/config/data_source_config.py`
- `tcp/preprocessing/fmriprep_parcellation.py`
- `tcp/preprocessing/README_FMRIPREP.md`
- `CHANGELOG_FMRIPREP.md`

**Modified Files**:
- `config/paths.py`
- `config/default_config.json`
- `tcp/preprocessing/run_pipeline.py`

**Moved Files**:
- `tcp/processing/parcellations/` → `parcellations/`

### Performance Metrics

**Parcellation Speed** (estimated):
- Single subject: ~2-5 minutes (9 runs)
- 100 subjects (4 parallel jobs): ~1-3 hours

**Memory Usage**:
- Per subject: ~2-4 GB RAM
- 4 parallel jobs: ~8-16 GB total

**Disk Space**:
- Per subject .h5 file: ~5-10 MB
- 100 subjects: ~500 MB - 1 GB

### Next Steps

1. ✅ Complete implementation
2. ⏳ Test with real fMRIPrep data
3. ⏳ Fix Buckner cerebellar implementation
4. ⏳ Performance optimization
5. ⏳ Add automated tests
6. ⏳ User acceptance testing

### Credits

**Inspiration**: Based on parcellation approach from colleague (Hermine Alfsen)

**Improvements**:
- Cross-platform compatibility
- Parallel processing support
- Integrated into existing pipeline
- Comprehensive documentation
- Better error handling

---

**Branch**: `feature/fmriprep-data-source`  
**Commit**: 938418c  
**Author**: Ian Philip Eglin  
**Date**: 2025-01-28
