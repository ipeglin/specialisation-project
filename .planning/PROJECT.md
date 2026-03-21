# TCP fMRI Pipeline — fmriprep Migration

## What This Is

A neuroimaging research pipeline that preprocesses and analyses fMRI BOLD data from the Transdiagnostic Connectome Project (TCP). The pipeline reads subject-level NIfTI volumes, parcellates them into timeseries using brain atlases, and runs functional connectivity analysis. The goal of this work is to migrate the parcellation step from the legacy HCP preprocessing output to fmriprep 25.1.4 BIDS-compliant outputs.

## Core Value

The parcellation pipeline must correctly locate and load the fmriprep BOLD NIfTI for each subject so that downstream `.h5` timeseries and functional connectivity results are computed from the correct preprocessed data.

## Requirements

### Validated

- ✓ Cross-platform path resolution via `config/paths.py` — existing
- ✓ HCP BOLD parcellation producing `.h5` timeseries via `HCPParcellator` — existing
- ✓ Subject enumeration via `DataSourceConfig.discover_hcp_subjects()` — existing
- ✓ Resumable pipeline orchestration via `TCPPipeline` / `run_pipeline.py` — existing
- ✓ Atlas-backed ROI extraction via `ROIExtractionService` / `AtlasLookupInterface` — existing
- ✓ Manifest-driven processing pipeline (`processing_data_manifest.json`) — existing

### Active

- [ ] fmriprep BOLD path construction replaces HCP path logic in `HCPParcellator` / `DataSourceConfig`
- [ ] Subject discovery enumerates fmriprep output directory (`fmriprep-25.1.4/sub-*/func/`) instead of HCP
- [ ] Config base path `hcp_output` replaced/extended with `fmriprep_output` pointing to `fmriprep-25.1.4/`
- [ ] BOLD filename constructed per-subject using fmriprep BIDS pattern: `{subject_id}_task-hammerAP_run-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz`
- [ ] CLI arguments (`--hcp-root`) updated or aliased to accept fmriprep root
- [ ] All hardcoded references to `hcp_output` path updated in config and scripts (parcellation scope only)
- [x] Optional `--participants-file` flag on both parcellation entry points with hard error on missing subjects — Validated in Phase 01.1: participants-txt-subject-filter

### Out of Scope

- Subject mapping/filtering scripts (`map_subject_files.py`, `filter_subjects.py`) — not in this migration
- Support for running both HCP and fmriprep pipelines simultaneously — replacing HCP only
- fmriprep confound/nuisance regressor loading — only the BOLD NIfTI file is needed
- Brain mask loading from fmriprep sidecar files — not required by current parcellation code

## Context

**Current state:**
- HCP BOLD path pattern: `{hcp_root}/sub-{id}/MNINonLinear/Results/task-hammerAP_run-01_bold/task-hammerAP_run-01_bold.nii.gz`
- fmriprep BOLD path pattern: `{fmriprep_root}/sub-{id}/func/sub-{id}_task-hammerAP_run-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz`

**Key files to change (parcellation scope):**
- `tcp/preprocessing/config/data_source_config.py` — `DataSourceConfig.hcp_root`, `get_hcp_bold_path()`, `discover_hcp_subjects()`, `validate_hcp_structure()`
- `tcp/preprocessing/hcp_parcellation.py` — `HCPParcellator.__init__()`, `discover_bold_files()`
- `tcp/preprocessing/parcellate_hcp_subjects.py` — CLI args and config construction
- `scripts/parcellate_missing_hcp_subjects.py` — CLI args and root path validation
- `config/default_config.json` — `hcp_output` key for each platform
- `scripts/verify_paths.py` — path verification references

**Cluster paths:**
- Old: `/cluster/projects/itea_lille-ie/Transdiagnostic/output/hcp_output`
- New: `/cluster/projects/itea_lille-ie/Transdiagnostic/output/fmriprep-25.1.4`

**Subject ID format:** unchanged — `sub-NDARINV{suffix}` for all subjects

## Constraints

- **Tech stack**: Python 3.11, conda `masters_thesis` environment — no new dependencies
- **Analysis restriction**: No signal/data analysis code may be introduced by AI — only path/plumbing changes
- **Compatibility**: Must continue to run on macOS (dev), Windows 11, and CentOS IDUN cluster
- **Scope**: Parcellation pipeline only — processing pipeline (`tcp/processing/`) is not touched

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Hard-code fmriprep BIDS filename entities (space, res, desc) with per-subject ID substitution | Explicit — fails loudly if fmriprep config changes; glob approach could silently match wrong file | — Pending |
| Replace HCP path logic entirely (not switchable) | Simplicity — no dual-pipeline complexity needed | — Pending |
| Scope migration to parcellation only | Subject mapping/filtering not yet needed for fmriprep data | — Pending |

---
*Last updated: 2026-03-21 — Phase 01.1 complete: participants file filter added to both parcellation entry points*
