# Project Research Summary

**Project:** fMRI Parcellation Pipeline — HCP to fmriprep BIDS Migration
**Domain:** Python neuroimaging pipeline path-resolution migration
**Researched:** 2026-03-02
**Confidence:** HIGH

## Executive Summary

This project migrates an existing fMRI parcellation pipeline from HCP-structured data paths to fmriprep 25.1.4 BIDS output paths. The pipeline currently hardcodes path patterns pointing at the HCP `MNINonLinear/Results/` directory tree; after migration, it must resolve BOLD files from the flat fmriprep `func/` layout using fully-qualified BIDS entity filenames (`sub-{id}_task-hammerAP_run-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz`). All required libraries (`pathlib`, `nilearn`, `nibabel`) are already present in the `masters_thesis` conda environment — no new dependencies are needed for this migration.

The recommended approach is a strictly layered, sequential migration targeting only the path-resolution layer. The processing core (`NiftiLabelsMasker`, atlas loading, `.h5` serialisation) is unchanged. All path logic should be consolidated into a single authority — `DataSourceConfig` — with `HCPParcellator.discover_bold_files()` delegating path construction to that class rather than duplicating it. A 5-step build order (config registry → `DataSourceConfig` → `HCPParcellator` → runner scripts → orchestrator) ensures each step is independently testable before the next.

The dominant risks are silent failures: a wrong `.h5` dataset key derived from the long fmriprep filename stem, a stale `data_source='hcp'` metadata tag that triggers incorrect de-meaning in downstream analysis, and a potential MNI template mismatch between the atlases (registered to MNI152Lin) and the fmriprep BOLD output (MNI152NLin2009cAsym). None of these produce an error — they corrupt results silently. A mandatory verification checklist should be run against real fmriprep data on the IDUN cluster before any downstream analysis proceeds.

---

## Key Findings

### Recommended Stack

No new packages are required. The migration is implemented entirely using `pathlib.Path` for path construction (stdlib, zero overhead) and the existing `nilearn.maskers.NiftiLabelsMasker` API. `pybids` is explicitly out of scope for this milestone — subject IDs are known in advance, so the 2–10 second indexing overhead of `BIDSLayout` provides no benefit. The fmriprep 25.x BIDS filename pattern has been stable since fmriprep 21.0; the `desc-preproc_bold.nii.gz` file is only present at `--level full` (the default) — it will be absent if fmriprep was run at `--level minimal`.

See [STACK.md](./STACK.md) for full version compatibility table and installation guidance.

**Core technologies:**
- `pathlib.Path` (stdlib): BOLD path construction — OS-agnostic, zero dependency cost, deterministic given known BIDS entities
- `nilearn.maskers.NiftiLabelsMasker` (>=0.10, already installed): ROI timeseries extraction — no API changes needed for fmriprep vs HCP BOLD NIfTI files
- `nibabel` (>=5.3, already installed): NIfTI loading and header inspection — NumPy 2.0 compatible, required by nilearn

### Expected Features

The migration has two tiers of changes. The P1 (table stakes) tier consists of 4–5 targeted method/field replacements that make the pipeline produce any output at all — without them the pipeline either crashes on `FileNotFoundError` or processes zero subjects silently. The P2 tier adds robustness without being blockers; they should be included in the same milestone if feasible. P3 items are explicitly deferred.

See [FEATURES.md](./FEATURES.md) for full dependency graph and prioritisation matrix.

**Must have (table stakes — pipeline broken without these):**
- Update `HCPParcellator.discover_bold_files()` path construction to fmriprep BIDS pattern
- Update `DataSourceConfig.discover_hcp_subjects()` to glob `func/` with full filename existence check
- Update `DataSourceConfig.get_hcp_bold_path()` to return the correct fmriprep path
- Update `DataSourceConfig.validate_hcp_structure()` to check `func/` not `MNINonLinear/Results/`
- Update CLI `--hcp-root` argument labels in all three runner scripts

**Should have (add in same milestone):**
- Glob-based BOLD file discovery instead of hardcoded full filename — resilience to future fmriprep minor version changes
- Update `.h5` metadata `source` attribute from `'hcp_parcellation'` to `'fmriprep_parcellation'`
- Update `verify_paths.py` labels from "HCP output" to "fmriprep output"

**Defer (v2+):**
- Rename `hcp_root` field and `HCPParcellator` class — cosmetic refactor, coordinate import updates
- BIDS entity parser utility — premature abstraction until a second caller exists
- fmriprep confound/mask path helpers — only justified when denoising is added

### Architecture Approach

The pipeline has three distinct layers. This migration touches only the middle layer (Preprocessing Path Logic) and the config registry that feeds it. The orchestration layer (`run_pipeline.py`, CLI scripts) and the processing/analysis layer (`DataLoader`, `main.py`, signal processing) are unchanged in substance, though CLI argument names must be updated in the orchestration layer. The key architectural principle is single-site path construction: all path logic lives in `DataSourceConfig`, and `HCPParcellator` delegates to it rather than maintaining its own path assembly. The factory function pattern (`create_fmriprep_config()`) keeps runner scripts trivially simple — they call one factory and pass the resulting dataclass to the parcellator.

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the full 5-step build order, data flow diagram, and anti-pattern catalogue.

**Major components:**
1. `config/default_config.json` — Platform path registry: add `fmriprep_output` key per platform alongside or replacing `hcp_output`
2. `DataSourceConfig` (dataclass) — Single authority for path construction and subject discovery; all fmriprep-specific logic lives here
3. `HCPParcellator` — Parcellation engine: replace `discover_bold_files()` path construction only; parcellation maths (`NiftiLabelsMasker`, `.h5` writing) unchanged
4. Runner scripts (`parcellate_hcp_subjects.py`, `parcellate_missing_hcp_subjects.py`, `integrate_cross_analysis.py`) — CLI surface: rename arguments, call `create_fmriprep_config()` factory
5. `run_pipeline.py` — Orchestrator: update subprocess kwarg names passed to runner scripts

### Critical Pitfalls

The full catalogue with prevention strategies and recovery costs is in [PITFALLS.md](./PITFALLS.md). The top five to address in the migration phase:

1. **H5 dataset key derived from long fmriprep filename stem** — normalise the key to `task-{task}_run-{run:02d}` (e.g., `task-hammerAP_run-01`) explicitly in `_save_h5()`, independent of the source filename; a key like `sub-NDARINVXXXXX_task-hammerAP_run-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold` will silently break any future key-by-name lookup
2. **Stale `data_source='hcp'` tag in manifest triggers de-meaning in `main.py`** — grep all occurrences of the `'hcp'` data-source string (not just filename patterns) before finishing; define `DATA_SOURCE_FMRIPREP = "fmriprep"` as a named constant; the `main.py` de-meaning branch is a silent analysis error
3. **`integrate_cross_analysis.py` missed during migration** — it also accepts `--hcp-root` and is called by `run_pipeline.py`; it is not in the PROJECT.md file list; do a full-codebase grep for `hcp.root|hcp_root|hcp-root|MNINonLinear` before writing any code
4. **Subject discovery glob without file existence check** — checking for the `func/` directory alone admits subjects with partial fmriprep outputs (brain mask only, no `_desc-preproc_bold.nii.gz`); always verify the full BIDS filename including `space-`, `res-`, and `desc-` entities
5. **MNI template mismatch between atlases and fmriprep BOLD** — atlases may be registered to MNI152Lin (FSL 91x109x91) while fmriprep outputs MNI152NLin2009cAsym (97x115x97); nilearn does not warn; visual QA required after first subject parcellation (this is a research decision if reslicing is needed — cannot be AI-generated)

---

## Implications for Roadmap

Based on the combined research, this migration naturally splits into two phases with a clear dependency boundary between them.

### Phase 1: Functional Path Migration

**Rationale:** All P1 (table stakes) features share a common dependency — they must be correct before any subject can be parcellated. The build order (config registry → `DataSourceConfig` → `HCPParcellator` → runner scripts → orchestrator) allows each layer to be tested independently. The critical pitfalls for silent data corruption (H5 key normalisation, `data_source` tag, `integrate_cross_analysis.py` audit) must be addressed in this phase, not deferred.

**Delivers:** A pipeline that can load fmriprep 25.1.4 BOLD files, parcellate subjects, and write correctly-structured `.h5` outputs with accurate provenance metadata

**Addresses:**
- All 4 table-stakes path changes (P1 features)
- CLI argument renaming across all 5 affected scripts (including `integrate_cross_analysis.py`)
- H5 key normalisation in `_save_h5()`
- `data_source` metadata tag update
- `config/default_config.json` `fmriprep_output` key per platform

**Avoids:**
- Pitfall 1: H5 key from filename stem (fix in `_save_h5()`)
- Pitfall 4: Stale `data_source='hcp'` tag (grep and replace with constant)
- Pitfall 3: Discovery glob without file existence check (verify full BIDS filename in `discover_subjects()`)
- Pitfall 6: `integrate_cross_analysis.py` missed update (pre-migration audit)
- Anti-pattern 1: Duplicating path construction at call sites (single authority in `DataSourceConfig`)

**Build order within phase:**
1. Pre-migration audit: `grep -r "hcp.root\|hcp_root\|hcp-root\|MNINonLinear" tcp/ scripts/ --include="*.py"`
2. `config/default_config.json`: add `fmriprep_output` key per platform
3. `data_source_config.py`: add `fmriprep_root` field, `get_fmriprep_bold_path()`, `discover_fmriprep_subjects()`, `validate_fmriprep_structure()`, `create_fmriprep_config()` factory; update `data_source` constant
4. `hcp_parcellation.py`: replace `discover_bold_files()` path construction; normalise H5 key in `_save_h5()`; update `source` attribute
5. Runner scripts + orchestrator: rename CLI args, call `create_fmriprep_config()`

### Phase 2: Verification and Hardening

**Rationale:** Several risks (MNI template mismatch, FOV difference, atlas alignment) cannot be validated without real fmriprep data on the IDUN cluster. These are post-pipeline checks that inform whether the migration is scientifically valid, not just technically functional. The P2 robustness features (glob-based discovery, `verify_paths.py` update) are low-effort and should be bundled here if not already done in Phase 1.

**Delivers:** Confirmed-correct parcellation outputs on real cluster data, with documented BOLD image shape, parcel count, and atlas alignment checks; updated `verify_paths.py` for ongoing ops use

**Uses:**
- `nibabel` direct load for shape inspection (`bold_img.shape[:3] == (97, 115, 97)`)
- `h5py` for H5 key format verification
- FSLeyes or `nilearn.plotting.plot_roi` for atlas overlay visual QA

**Addresses:**
- P2 features: glob-based BOLD discovery, `verify_paths.py` label update
- Pitfall 2: MNI template mismatch — visual QA, document finding as research decision if atlas reslicing is needed
- Pitfall 7: FOV difference (97x115x97 vs 91x109x91) — shape assertion in verification
- "Looks Done But Isn't" checklist from PITFALLS.md (10-item checklist)

**Avoids:**
- Silent wrong-space file inclusion (verify `space-MNI152NLin2009cAsym_res-2` entity in discovered files)
- Mixing pre-migration (HCP) and post-migration (fmriprep) `.h5` files in the same analysis run

### Phase Ordering Rationale

- Phase 1 must precede Phase 2 because verification requires correctly-written `.h5` outputs from real data
- The 5-step build order within Phase 1 is dictated by the dependency graph: `DataSourceConfig` is the authority, runner scripts call it, orchestrator calls runner scripts
- `integrate_cross_analysis.py` must be treated as part of Phase 1 despite being absent from PROJECT.md's explicit file list — the pipeline orchestration will fail at that step if it is deferred
- MNI template mismatch (Pitfall 2) is deferred to Phase 2 because it is a research decision about atlas reslicing, not an implementation decision — it cannot be resolved without visual QA on real data and potentially requires guidance from the thesis author

### Research Flags

Phases with standard, well-documented patterns (no deeper research needed):
- **Phase 1, Steps 1–4:** fmriprep BIDS path patterns are fully documented and verified against official fmriprep 25.0 output docs; `pathlib.Path` construction pattern is trivial; nilearn API is unchanged
- **Phase 1, Step 5:** CLI argument renaming is mechanical; factory function pattern is already in use in the codebase

Phases that need domain knowledge validation (not additional research, but human judgment):
- **Phase 2, MNI template check:** Whether the atlases need to be resliced to MNI152NLin2009cAsym is a research decision outside AI scope; the thesis author must confirm the correct template for each atlas before proceeding with analysis

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All claims verified against official fmriprep 25.0 docs, nilearn 0.13.1 changelog, nibabel 5.3 changelog, pybids docs; existing conda env confirmed compatible |
| Features | HIGH | Based on direct codebase read of all 5 affected files; path change requirements are deterministic given the fmriprep BIDS spec |
| Architecture | HIGH | Full codebase read; all component boundaries and integration points verified from source; build order derived from actual dependency graph |
| Pitfalls | HIGH | All critical pitfalls grounded in actual source code line references and official nilearn/fmriprep documentation; no speculation |

**Overall confidence:** HIGH

### Gaps to Address

- **Atlas template identity:** It is not confirmed in research which MNI template variant each of the two atlases (`400Parcels_Yeo2011_17Networks_FSLMNI152_2mm.nii.gz`, `Tian_Subcortex_S2_3T.nii`) is registered to. This must be confirmed by checking atlas documentation before Phase 2 visual QA. If either atlas is registered to MNI152Lin rather than MNI152NLin2009cAsym, atlas reslicing is a research decision that the thesis author must make — handle by flagging clearly in Phase 2 checklist, not by implementing a resolution
- **Exact task name entity in fmriprep files:** FEATURES.md notes that the HCP task directory was named `task-hammerAP_run-01_bold` but the fmriprep task entity may differ (`task-hammerAP` vs `task-hammerAP` with or without `AP` suffix) — verify against an actual fmriprep output file on the cluster before hardcoding the filename template
- **`pybids` compatibility:** If subject IDs ever need to be discovered dynamically in the future, pin `pybids==0.16.4` and avoid `universal-pathlib` — a known incompatibility documented in fmriprep 25.2.3 changelog

---

## Sources

### Primary (HIGH confidence)

- https://fmriprep.org/en/25.0.0/outputs.html — fmriprep 25.0 official output layout; functional derivatives section; confirmed BIDS layout default, `desc-preproc` only at `--level full`
- https://fmriprep.org/en/stable/changes.html — fmriprep changelog; confirmed 25.1.4 is patch-only with no layout changes
- https://nilearn.github.io/stable/modules/generated/nilearn.maskers.NiftiLabelsMasker.html — confirmed `resampling_target="data"` behaviour and no template-identity check
- https://bids-standard.github.io/pybids/ — pybids 0.16.4; confirmed latest stable version and derivatives API
- https://nipy.org/nibabel/changelog.html — nibabel 5.3.0 changelog; NumPy 2.0 compatibility confirmed
- Direct codebase read: `tcp/preprocessing/hcp_parcellation.py`, `tcp/preprocessing/config/data_source_config.py`, `tcp/preprocessing/parcellate_hcp_subjects.py`, `tcp/preprocessing/run_pipeline.py`, `tcp/processing/main.py`, `scripts/verify_paths.py`, `scripts/parcellate_missing_hcp_subjects.py`, `config/default_config.json`, `config/paths.py`

### Secondary (MEDIUM confidence)

- BIDS specification for `func/` flat layout and entity naming conventions — inferred from fmriprep docs and confirmed against project file examples; not independently fetched from bids-specification.readthedocs.io
- `integrate_cross_analysis.py` CLI surface — confirmed by grep; file content not fully reviewed beyond the `--hcp-root` argument presence

---
*Research completed: 2026-03-02*
*Ready for roadmap: yes*
