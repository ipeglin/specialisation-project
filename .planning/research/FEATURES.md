# Feature Research

**Domain:** fMRI parcellation pipeline — HCP-style paths to fmriprep BIDS outputs
**Researched:** 2026-03-02
**Confidence:** HIGH (based on direct codebase analysis + fmriprep BIDS spec)

---

## Context: What Is Actually Changing

The pipeline currently hard-codes two path patterns:

| Location | HCP pattern |
|----------|-------------|
| `HCPParcellator.discover_bold_files()` | `{hcp_root}/sub-{id}/MNINonLinear/Results/task-{task}AP_run-0{run}_bold/task-{task}AP_run-0{run}_bold.nii.gz` |
| `DataSourceConfig.discover_hcp_subjects()` | Scans `{hcp_root}/sub-*/MNINonLinear/Results/` |
| `DataSourceConfig.get_hcp_bold_path()` | Same HCP path |
| `DataSourceConfig.validate_hcp_structure()` | Checks `MNINonLinear/Results/` exists |
| `verify_paths.py` | Hardcodes label `hcp_output_path` pointing to `get_data_path('hcp_output')` |

Target fmriprep 25.1.4 BIDS pattern:

```
{fmriprep_root}/sub-{id}/func/
    sub-{id}_task-hammerAP_run-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz
```

Key structural differences:
- Intermediate dirs: `MNINonLinear/Results/task-{task}AP_run-0{run}_bold/` gone; replaced by flat `func/`
- Filename convention: changes from `task-{task}AP_run-0{run}_bold.nii.gz` to full BIDS entities (`space-`, `res-`, `desc-`)
- Run zero-padding: HCP uses `run-0{run}` (single digit zero-padded); fmriprep uses `run-01` (two digits)
- Task naming: HCP names directory `task-hammerAP_run-01_bold`; fmriprep uses `task-hammerAP` entity (no `AP` suffix in task name — verify per actual files)
- Space entity: fmriprep files include `space-MNI152NLin2009cAsym_res-2` which must match the atlas space in use

---

## Feature Landscape

### Table Stakes (Must Change for Pipeline to Run)

Features that are broken until changed. The pipeline produces `FileNotFoundError` or silently processes zero subjects without these.

| Feature | Why Required | Complexity | Files Affected | Notes |
|---------|-------------|------------|----------------|-------|
| **Update BOLD path construction in `HCPParcellator`** | `discover_bold_files()` builds the wrong path; no files are found | LOW | `hcp_parcellation.py` | Change `MNINonLinear/Results/task-{task}AP_run-0{run}_bold/task-{task}AP_run-0{run}_bold.nii.gz` to `func/sub-{id}_task-{task}AP_run-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz` |
| **Update subject discovery in `DataSourceConfig`** | `discover_hcp_subjects()` scans for `MNINonLinear/Results/` which does not exist under fmriprep layout | LOW | `data_source_config.py` | Change glob target to `func/` subdirectory and check for `*_desc-preproc_bold.nii.gz` |
| **Update `get_hcp_bold_path()` path construction** | Method returns `None` for every subject; parcellation is skipped entirely | LOW | `data_source_config.py` | Same path formula as above |
| **Update `validate_hcp_structure()` directory check** | Validates that `MNINonLinear/Results/` exists; always returns `False` under fmriprep | LOW | `data_source_config.py` | Change to check that `func/` directory exists |
| **Rename config field `hcp_root` to `fmriprep_root` (or equivalent)** | The field name `hcp_root` is misleading once pointing at fmriprep output; `HCPParcellator.__init__` also stores `self.hcp_root` | MEDIUM | `data_source_config.py`, `hcp_parcellation.py`, `parcellate_hcp_subjects.py`, `parcellate_missing_hcp_subjects.py` | Rename is a ripple change across 4 files; a compatibility alias is safer if other callers exist |
| **Update CLI flag `--hcp-root` label and help text** | CLI still says "HCP output directory"; post-migration it points at fmriprep output | LOW | `parcellate_hcp_subjects.py`, `parcellate_missing_hcp_subjects.py`, `hcp_parcellation.py` | Rename to `--fmriprep-root`; update help text |
| **Update `verify_paths.py` label and path key** | Prints `hcp_output_path` in output; uses `get_data_path('hcp_output')` which still resolves to old subdir name | LOW | `scripts/verify_paths.py` | Update label and optionally rename `hcp_output` subdir key |

### Differentiators (Nice-to-Have Improvements)

Features not required for functional correctness, but that reduce fragility and improve maintainability.

| Feature | Value Proposition | Complexity | Files Affected | Notes |
|---------|-------------------|------------|----------------|-------|
| **Glob-based BOLD file discovery instead of hardcoded entity values** | fmriprep filenames contain variable entities (`space-`, `res-`, `desc-`); a glob pattern is more robust than a hardcoded full filename | LOW | `hcp_parcellation.py`, `data_source_config.py` | Use `func/*_task-{task}*_desc-preproc_bold.nii.gz` as glob; extract run number from matched filename rather than constructing it | 
| **Validate space/resolution of discovered BOLD files** | Atlas space (`MNI152NLin2009cAsym`) and resolution (`res-2`) must match parcellation atlases; silent mismatch produces wrong timeseries | MEDIUM | `hcp_parcellation.py` | Parse BIDS entities from filename, warn if `space-` or `res-` do not match expected values |
| **Add fmriprep-specific confound/mask path helpers to `DataSourceConfig`** | fmriprep co-produces brain masks (`desc-brain_mask.nii.gz`) and confound TSVs alongside BOLD; future denoising steps need these paths | MEDIUM | `data_source_config.py` | Add `get_fmriprep_mask_path()` and `get_fmriprep_confounds_path()` alongside existing BOLD helper; not used by current parcellation but documents available files |
| **Update `.h5` metadata attribute `source` from `'hcp_parcellation'` to `'fmriprep_parcellation'`** | `_save_h5()` writes `source = 'hcp_parcellation'`; downstream code checking this attribute will see wrong provenance | LOW | `hcp_parcellation.py` | Change string literal in `_save_h5()`; consider making it a constructor parameter |
| **Add BIDS entity parser utility** | Multiple methods manually construct/deconstruct filenames; a shared `parse_bids_entities(filename)` function would centralise this logic | MEDIUM | New utility, or extend `utils/` | Candidate home: `tcp/preprocessing/utils/bids_utils.py`; out of scope unless multiple callers |
| **Update `DataSourceType` enum and related terminology** | Enum value `HCP = "hcp"` and `COMBINED = "combined"` labels are fine, but docstrings and comments still say "HCP-preprocessed" | LOW | `data_source_config.py` | Rename docstring references only; enum values are safe to leave as-is to avoid breaking manifest JSON files that store `"data_source": "hcp"` |

### Anti-Features (Do Not Add)

Features that seem like natural additions during this migration, but should be explicitly deferred.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **Automatic space/resolution resampling** | fmriprep outputs may have a different voxel grid than the hardcoded atlases | Resampling is a signal-processing decision that belongs in analysis code, not the path config layer; introducing it silently would change results | Let `NiftiLabelsMasker(resampling_target="data")` handle it (already set); just validate and warn |
| **fmriprep JSON sidecar parsing** | fmriprep produces `*_bold.json` with TR and slice-timing; pipeline could auto-read TR | Adds a dependency on sidecar presence; current pipeline does not use TR; scope creep | Leave as explicit parameter if TR is needed in future |
| **Automatic subject discovery by scanning the fmriprep directory tree** | Could replace manual subject lists | Discovery logic would need to replicate BIDS validation; risk of picking up incomplete subjects | Keep explicit subject lists; update `discover_hcp_subjects()` to use the new `func/` path only |
| **Migrating the class name `HCPParcellator` to `FmriprepParcellator`** | Name is now misleading | Class rename requires updating all import statements across `parcellate_hcp_subjects.py`, `parcellate_missing_hcp_subjects.py`, and any downstream consumers; risk of breaking things for minimal value in this milestone | Add a docstring note; rename in a dedicated refactor milestone |
| **Adding fmriprep version detection or validation** | fmriprep output structure changed between versions | Version sniffing is fragile; path schema for fmriprep 25.x is stable | Pin to fmriprep 25.1.4 convention; document expected version in README |

---

## Feature Dependencies

```
[Update BOLD path in HCPParcellator]
    └──requires──> [Update subject discovery in DataSourceConfig]
                       (discovery must find subjects before parcellator is called)

[Update CLI flags --hcp-root -> --fmriprep-root]
    └──requires──> [Rename config field hcp_root]
                       (CLI feeds DataSourceConfig constructor)

[Glob-based BOLD file discovery]
    └──enhances──> [Update BOLD path in HCPParcellator]
                       (replaces hardcoded path with pattern)

[Validate space/resolution of BOLD files]
    └──requires──> [Glob-based BOLD file discovery]
                       (need the matched filename to parse entities)

[Update .h5 metadata source attribute]
    └──independent──> (self-contained one-line change in _save_h5)

[Update verify_paths.py label]
    └──independent──> (cosmetic; does not block pipeline execution)
```

### Dependency Notes

- **Subject discovery requires path change first:** `DataSourceConfig.discover_hcp_subjects()` is called before `HCPParcellator` is instantiated; if discovery returns an empty list the parcellator is never reached, so this is the first failure point.
- **CLI rename requires field rename:** `--hcp-root` feeds directly into `DataSourceConfig(hcp_root=...)`. Both must change together or the CLI will pass the value to a field that no longer exists.
- **Glob-based discovery enhances but does not block:** Hardcoded path with correct fmriprep pattern will work; glob is an improvement for resilience to future fmriprep minor version changes.
- **`.h5` metadata change is independent:** Can be done in any order without affecting pipeline execution.

---

## MVP Definition

### Launch With (v1) — Minimum to make pipeline functional

These are the table stakes changes only. Together they make the pipeline execute correctly against fmriprep 25.1.4 outputs.

- [ ] **Update BOLD path construction in `HCPParcellator.discover_bold_files()`** — pipeline produces no output without this
- [ ] **Update `DataSourceConfig.discover_hcp_subjects()`** — discovery returns empty list without this; no subjects are processed
- [ ] **Update `DataSourceConfig.get_hcp_bold_path()`** — returns `None` for every subject without this
- [ ] **Update `DataSourceConfig.validate_hcp_structure()`** — always returns `False` without this
- [ ] **Update CLI `--hcp-root` argument label and help text in all three CLI scripts** — low-effort; prevents operator confusion

### Add After Validation (v1.x)

- [ ] **Glob-based BOLD file discovery** — add once v1 is confirmed working; makes path pattern more resilient
- [ ] **Update `.h5` metadata `source` attribute** — add before any downstream code reads provenance
- [ ] **Update `verify_paths.py` label** — cosmetic; useful for cluster operators

### Future Consideration (v2+)

- [ ] **Rename `hcp_root` field and `HCPParcellator` class** — dedicated refactor, worth doing but needs coordinated import updates across all consumers
- [ ] **BIDS entity parser utility** — only justified if a second caller emerges
- [ ] **fmriprep confound/mask path helpers** — only justified when denoising is added to the pipeline

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Update BOLD path in `HCPParcellator` | HIGH (pipeline broken without it) | LOW | P1 |
| Update `discover_hcp_subjects()` | HIGH (zero subjects processed without it) | LOW | P1 |
| Update `get_hcp_bold_path()` | HIGH (returns None always) | LOW | P1 |
| Update `validate_hcp_structure()` | HIGH (always False) | LOW | P1 |
| Update CLI flags and help text | MEDIUM (operator UX) | LOW | P1 |
| Glob-based BOLD discovery | MEDIUM (robustness) | LOW | P2 |
| Update `.h5` source metadata | MEDIUM (provenance correctness) | LOW | P2 |
| Update `verify_paths.py` label | LOW (cosmetic) | LOW | P2 |
| Validate BOLD space/resolution | MEDIUM (catch atlas mismatch) | MEDIUM | P2 |
| Rename `hcp_root` field + class | LOW (readability) | MEDIUM | P3 |
| fmriprep confound/mask helpers | LOW (not used by current pipeline) | MEDIUM | P3 |
| BIDS entity parser utility | LOW (premature abstraction) | MEDIUM | P3 |

**Priority key:**
- P1: Must have for this milestone (pipeline broken without it)
- P2: Should have, add in same milestone if possible
- P3: Defer to future milestone

---

## Sources

- Direct codebase analysis: `tcp/preprocessing/hcp_parcellation.py`, `tcp/preprocessing/config/data_source_config.py`, `tcp/preprocessing/parcellate_hcp_subjects.py`, `scripts/parcellate_missing_hcp_subjects.py`, `scripts/verify_paths.py`, `config/default_config.json`
- fmriprep 25.1.4 BIDS output spec as stated in milestone context (path confirmed: `sub-{id}/func/sub-{id}_task-hammerAP_run-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz`)
- BIDS standard for `func/` layout and entity conventions

---
*Feature research for: HCP-to-fmriprep parcellation pipeline migration*
*Researched: 2026-03-02*
