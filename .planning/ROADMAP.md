# TCP fMRI Pipeline — fmriprep Migration Roadmap

**Project:** fMRI parcellation pipeline — HCP to fmriprep 25.1.4 BIDS path migration
**Goal:** Migrate the parcellation pipeline so it loads BOLD NIfTI files from fmriprep 25.1.4 BIDS outputs instead of HCP-structured outputs, producing correctly-structured `.h5` timeseries with accurate provenance metadata.

---

## Phases

### Phase 1: Functional Path Migration

**Goal:** Replace all HCP-specific path logic with fmriprep BIDS path logic so the pipeline can locate, load, and parcellate fmriprep BOLD files and write correctly-structured `.h5` outputs with accurate provenance metadata.

**Plans:** [To be planned]

**Status:** unplanned

**Addresses:**
- Config registry: add `fmriprep_output` key per platform in `config/default_config.json`
- `DataSourceConfig`: add `fmriprep_root` field, `get_fmriprep_bold_path()`, `discover_fmriprep_subjects()`, `validate_fmriprep_structure()`, `create_fmriprep_config()` factory
- `HCPParcellator`: replace `discover_bold_files()` path construction; normalise H5 key in `_save_h5()`; update `source` metadata attribute
- Runner scripts (`parcellate_hcp_subjects.py`, `parcellate_missing_hcp_subjects.py`, `integrate_cross_analysis.py`): rename CLI args, call `create_fmriprep_config()`
- Orchestrator (`run_pipeline.py`): update subprocess kwarg names

**Deferred:**
- Renaming `HCPParcellator` class and `hcp_parcellation.py` file
- BIDS entity parser utility
- fmriprep confound/mask path helpers

---

### Phase 01.1: participants-txt subject filter (INSERTED)

**Goal:** Add optional `--participants-file` CLI flag to parcellation entry points that filters discovered fmriprep subjects against a user-specified subject list, with hard error on missing subjects.
**Requirements**: TBD
**Depends on:** Phase 1
**Plans:** 1/1 plans complete

Plans:
- [x] 01.1-01-PLAN.md -- Create participants_filter utility and wire into both parcellation CLI scripts

### Phase 2: Verification and Hardening

**Goal:** Validate the migrated pipeline against real fmriprep data on the IDUN cluster, confirm correct BOLD image shape and atlas alignment, and harden path verification tooling for ongoing use.

**Plans:** [To be planned]

**Status:** unplanned

**Addresses:**
- Glob-based BOLD file discovery (resilience to future fmriprep minor version changes)
- `verify_paths.py` label and path-key update
- MNI template mismatch visual QA (research decision for thesis author — flag, do not resolve)
- FOV shape assertion (`bold_img.shape[:3] == (97, 115, 97)`)
- 10-item "Looks Done But Isn't" checklist from PITFALLS.md

**Depends on:** Phase 1

---

## Build Order (Phase 1 internal)

1. Pre-migration audit: `grep -r "hcp.root\|hcp_root\|hcp-root\|MNINonLinear" tcp/ scripts/ --include="*.py"`
2. `config/default_config.json`: add `fmriprep_output` key per platform
3. `data_source_config.py`: add `fmriprep_root` field and all fmriprep-specific methods
4. `hcp_parcellation.py`: replace `discover_bold_files()` path construction + normalise H5 key + update `source` attribute
5. Runner scripts + orchestrator: rename CLI args, call `create_fmriprep_config()`

---

*Bootstrapped from research: 2026-03-02*
*Research confidence: HIGH*
