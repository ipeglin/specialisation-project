# Architecture Research

**Domain:** fMRI preprocessing pipeline — path-resolution migration (HCP -> fmriprep BIDS)
**Researched:** 2026-03-02
**Confidence:** HIGH (full codebase read; no speculation required)

---

## Standard Architecture

### System Overview

The pipeline has three distinct layers. The migration touches only the middle layer (Preprocessing
Path Logic) and the config registry that feeds it. Nothing in the top (Orchestration) or bottom
(Processing / Analysis) layer changes.

```
┌─────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION LAYER                           │
│                                                                  │
│  run_pipeline.py (TCPPipeline)                                   │
│    - reads CLI args: --hcp-root / --hcp-parcellated-output       │
│    - dispatches to per-step scripts via subprocess               │
│                                                                  │
│  scripts/parcellate_missing_hcp_subjects.py                      │
│    - standalone re-run utility; same CLI surface                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │ constructs DataSourceConfig
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PREPROCESSING PATH LOGIC LAYER                  │
│                           [MIGRATION TARGET]                     │
│                                                                  │
│  DataSourceConfig (dataclass)                                    │
│    fields:  hcp_root, hcp_parcellated_output, default_task       │
│    methods: get_hcp_bold_path()                                  │
│             discover_hcp_subjects()                              │
│             validate_hcp_structure()                             │
│                                                                  │
│  HCPParcellator (class)                                          │
│    init:    hcp_root → validates directory exists                │
│    method:  discover_bold_files(subject_id, task)                │
│             parcellate_bold(path) → calls nilearn                │
│             parcellate_subject() → save .h5                      │
│                                                                  │
│  config/default_config.json                                      │
│    key:     hcp_output  (platform-scoped base paths)             │
│             hcp_parcellated_output                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │ produces .h5 timeseries files
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                PROCESSING / ANALYSIS LAYER                       │
│                    [UNCHANGED — out of scope]                    │
│                                                                  │
│  processing_data_manifest.json                                   │
│    - subject → .h5 file mapping                                  │
│    - consumed by DataLoader (tcp/processing/data_loader.py)      │
│                                                                  │
│  ProcessingConfig → DataLoader → SubjectManager                  │
│    - loads .h5 timeseries by subject ID                          │
│    - drives ROIExtractionService, signal processing              │
└─────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Migration action |
|-----------|----------------|------------------|
| `config/default_config.json` | Registry: platform -> base directory for each named data source | Add `fmriprep_output` key alongside (or replacing) `hcp_output` per platform |
| `config/paths.py` (PathConfig) | Singleton: resolves named keys to `Path` objects; applies env overrides | Add `get_fmriprep_path()` convenience function; or reuse `get_data_path()` with new key |
| `DataSourceConfig` (dataclass) | Configuration carrier: root paths + task defaults; subject enumeration; per-subject path construction | Rename/add `fmriprep_root` field; replace `get_hcp_bold_path()` → `get_fmriprep_bold_path()`, `discover_hcp_subjects()` → `discover_fmriprep_subjects()` |
| `HCPParcellator` | Engine: loads BOLD NIfTI, applies nilearn maskers, writes .h5 | Replace `discover_bold_files()` path construction with fmriprep BIDS pattern; rename class to `FmriprepParcellator` or keep name with updated docs |
| `parcellate_hcp_subjects.py` | Runner script: orchestrates parcellation for a set of subjects | Update CLI arg `--hcp-root` → `--fmriprep-root` (or alias); update config factory calls |
| `scripts/parcellate_missing_hcp_subjects.py` | Standalone re-run utility | Same CLI surface change as above |
| `scripts/verify_paths.py` | Path sanity checker | Add fmriprep path verification; remove or skip HCP path check |

---

## Recommended Project Structure

The migration does **not** require new files or new directories. All changes are in-place replacements
within the existing structure:

```
specialisation-project/
├── config/
│   ├── default_config.json          # ADD fmriprep_output key per platform
│   └── paths.py                     # ADD get_fmriprep_path() convenience fn (optional)
│
├── tcp/preprocessing/
│   ├── config/
│   │   └── data_source_config.py    # RENAME fields; replace HCP-specific methods
│   ├── hcp_parcellation.py          # REPLACE discover_bold_files() path construction
│   └── parcellate_hcp_subjects.py   # UPDATE CLI args + factory function call
│
└── scripts/
    ├── parcellate_missing_hcp_subjects.py   # UPDATE CLI args
    └── verify_paths.py                      # UPDATE path check
```

### Structure Rationale

- **config/**: Centralised path registry means platforms paths change in one file.
  The `PathConfig` singleton means callers never hard-code directory strings.
- **data_source_config.py**: Owning all subject-enumeration and path-construction logic
  in one place (the dataclass) keeps `HCPParcellator` / runner scripts free of path logic.
  The factory function pattern (`create_hcp_config`) means callers only change one call site.
- **hcp_parcellation.py**: The `discover_bold_files()` method is the single site where the
  directory structure is encoded. Changing only that method leaves all parcellation maths
  (`parcellate_bold`, atlas loading, .h5 serialisation) untouched.

---

## Architectural Patterns

### Pattern 1: Single-Site Path Construction

**What:** All subject-level path logic lives in one method (`get_hcp_bold_path()` /
`discover_bold_files()`). All other code calls that method and never concatenates path fragments.

**When to use:** Any time a path pattern changes (new preprocessing tool, different BIDS variant).

**Trade-offs:** The method becomes the single point of failure — a typo breaks everything — but
also the single point of fix.

**Current implementation (HCP):**
```python
# DataSourceConfig.get_hcp_bold_path()
task_dir = results_dir / f"task-{task}AP_run-0{run}_bold"
bold_file = task_dir / f"task-{task}AP_run-0{run}_bold.nii.gz"
```

**Target implementation (fmriprep):**
```python
# DataSourceConfig.get_fmriprep_bold_path()
func_dir = self.fmriprep_root / subject_id / "func"
bold_file = (
    func_dir /
    f"{subject_id}_task-{task}AP_run-{run:02d}"
    f"_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
)
return bold_file if bold_file.exists() else None
```

### Pattern 2: Factory Function as CLI Adapter

**What:** `create_hcp_config()` / `create_fmriprep_config()` convert flat CLI args into a
`DataSourceConfig` dataclass. All scripts call the factory, never the dataclass constructor.

**When to use:** Any time CLI args need to be translated into internal config objects.

**Trade-offs:** One extra indirection, but script code becomes trivial to update — only the
factory call site changes, not `__post_init__` validation logic.

```python
# Before: create_hcp_config(hcp_root, parcellated_output, default_task)
# After:  create_fmriprep_config(fmriprep_root, parcellated_output, default_task)
#
# Internal dataclass changes, script changes one line.
```

### Pattern 3: Config Registry (default_config.json) for Platform Paths

**What:** Base paths for named datasets are stored per-platform in JSON and loaded via
`PathConfig._get_base_path(key)`. Scripts never hard-code `/cluster/...` paths.

**When to use:** Any time a new data location appears (new dataset, new cluster path).

**Trade-offs:** Requires JSON edit per new platform, but eliminates per-file path duplication.

```json
// config/default_config.json — add alongside existing keys:
"idun": {
  "fmriprep_output": "/cluster/projects/itea_lille-ie/Transdiagnostic/output/fmriprep-25.1.4"
}
```

---

## Data Flow

### BOLD Parcellation Flow

```
CLI invocation
    --fmriprep-root /path/to/fmriprep-25.1.4
    --fmriprep-parcellated-output /path/to/output
         |
         | create_fmriprep_config()
         v
DataSourceConfig
    .fmriprep_root
    .fmriprep_parcellated_output
    .default_task = "hammerAP"
         |
         | .discover_fmriprep_subjects()
         |   glob: fmriprep_root / "sub-*" / "func" /
         |          "*_task-hammerAP_run-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
         v
List[subject_id]
         |
         | for each subject_id:
         |   .get_fmriprep_bold_path(subject_id)
         v
Path  →  fmriprep_root / sub-ID / func /
         sub-ID_task-hammerAP_run-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz
         |
         | HCPParcellator / FmriprepParcellator
         |   .discover_bold_files(subject_id) → same path construction
         |   .parcellate_bold(path) → nilearn NiftiLabelsMasker (UNCHANGED)
         v
np.ndarray (434, timepoints)
         |
         | ._save_h5(timeseries, subject_id, task, output_dir)
         v
.h5 file: {parcellated_output}/sub-ID_task-hammerAP_parcellated.h5
         |
         | (later) integrate_cross_analysis.py builds manifest
         v
processing_data_manifest.json
         |
         | DataLoader.get_subject_files(subject_id, 'timeseries')
         v
tcp/processing/* (UNCHANGED)
```

### Key Data Flows

1. **Path resolution:** CLI arg → `DataSourceConfig.fmriprep_root` → `get_fmriprep_bold_path()`
   → absolute `Path` object consumed by `HCPParcellator` (or renamed class).
   The `HCPParcellator` itself receives only a resolved path; it has no fmriprep knowledge.

2. **Subject discovery:** `discover_fmriprep_subjects()` replaces `discover_hcp_subjects()`.
   Old: glob `sub-*/MNINonLinear/Results/task-hammerAP_run-*_bold/*.nii.gz`.
   New: glob `sub-*/func/*_task-hammerAP_run-01_*_desc-preproc_bold.nii.gz` (flatter tree).

3. **Downstream manifest:** The `.h5` output format and path template are unchanged.
   `integrate_cross_analysis.py` and `DataLoader` see no difference once `.h5` files exist.

---

## Scaling Considerations

This is a research pipeline, not a web service. Scale = number of subjects.

| Scale | Architecture Adjustments |
|-------|--------------------------|
| 1-50 subjects | Sequential parcellation in `parcellate_subject()` loop — current approach is fine |
| 50-500 subjects | `parcellate_subjects_parallel()` with `joblib` already exists — use `n_jobs=8` on IDUN |
| 500+ subjects | SLURM array job per subject; each job calls `parcellate_hcp_subjects.py` with `--subject-id` |

### Scaling Priorities

1. **First bottleneck:** I/O — NIfTI file read is the rate limiter per subject. Parallel jobs help
   by overlapping reads. Already addressed by `parcellate_subjects_parallel()`.
2. **Second bottleneck:** nilearn masker re-initialisation per subject in parallel mode.
   `HCPParcellator` already creates a fresh instance per parallel worker (see `process_one_subject`
   inner function), so masker state is not shared. No change needed.

---

## Anti-Patterns

### Anti-Pattern 1: Duplicating Path Construction at Call Sites

**What people do:** Inline the fmriprep filename template in multiple scripts
(`parcellate_hcp_subjects.py`, `scripts/parcellate_missing_hcp_subjects.py`, etc.)

**Why it's wrong:** When fmriprep output naming changes (e.g., `res-2` becomes `res-02`), every
call site must be found and updated. One will be missed.

**Do this instead:** Keep the single authoritative template in `DataSourceConfig.get_fmriprep_bold_path()`.
All scripts call that method. `HCPParcellator.discover_bold_files()` also delegates to
`DataSourceConfig` rather than re-implementing path assembly.

### Anti-Pattern 2: Renaming Everything "fmriprep" Immediately

**What people do:** Rename `HCPParcellator` → `FmriprepParcellator`, `hcp_parcellation.py` →
`fmriprep_parcellation.py`, and update every import in one large commit.

**Why it's wrong:** Large rename diffs make review difficult and the pipeline untestable mid-rename.
The class name is cosmetic — the important functional change is in `discover_bold_files()` and
the `fmriprep_root` field.

**Do this instead:** Rename class and file last, as a dedicated cleanup commit after the functional
path changes are verified working on real data. Or keep `HCPParcellator` as an alias if import
compatibility is needed during transition.

### Anti-Pattern 3: Switching on Source Type Inside HCPParcellator

**What people do:** Add `if source_type == 'fmriprep': ... elif source_type == 'hcp': ...` branches
inside `discover_bold_files()` or `__init__()`.

**Why it's wrong:** The PROJECT.md explicitly states this migration replaces HCP entirely — no
dual-pipeline complexity. Switches inside the engine class would contradict that decision and
accumulate technical debt.

**Do this instead:** Replace the HCP logic outright. If backward compatibility is needed during
testing, keep the old code in a git branch, not in an `if` branch.

### Anti-Pattern 4: Hard-coding Cluster Paths in Python Files

**What people do:** Set `fmriprep_root = Path("/cluster/projects/itea_lille-ie/Transdiagnostic/output/fmriprep-25.1.4")` inside a script or config dataclass default.

**Why it's wrong:** The path breaks on macOS and Windows. The `PathConfig` + `default_config.json`
pattern already solves cross-platform resolution.

**Do this instead:** Add `fmriprep_output` as a key in `default_config.json` for each platform,
then expose it via `PathConfig.get_fmriprep_path()`. Scripts read from config; they do not
contain absolute paths.

---

## Integration Points

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `DataSourceConfig` ↔ `HCPParcellator` | `DataSourceConfig` passed as constructor arg to `HCPParcellationRunner`; `HCPParcellator` receives only `hcp_root` Path | After migration: `HCPParcellator.__init__` receives `fmriprep_root`; path method moves into `DataSourceConfig` or stays in parcellator — either is consistent |
| `run_pipeline.py` ↔ `parcellate_hcp_subjects.py` | CLI subprocess with `--hcp-root` / `--hcp-parcellated-output` string args | Both files must update CLI arg names together to stay in sync |
| `parcellate_hcp_subjects.py` ↔ `map_subject_files.py` | JSON manifest (`subject_file_mapping.json`) written by parcellator, read by mapper | Format unchanged; only the `.h5` file paths differ in content (still valid .h5 files) |
| `HCPParcellator._save_h5()` ↔ `DataLoader` | `.h5` file format with `source` attribute = `'hcp_parcellation'` | After migration, update `source` attribute value to `'fmriprep_parcellation'` for provenance; `DataLoader` does not filter on this attribute so it is non-breaking |

### Build Order for Migration

The dependency graph dictates this order. Each step is independently testable before the next.

```
Step 1 — Config registry (no code)
  config/default_config.json: add fmriprep_output per platform
  Testable: python scripts/verify_paths.py

Step 2 — DataSourceConfig field + methods (core path logic)
  data_source_config.py:
    - add fmriprep_root field
    - add get_fmriprep_bold_path()
    - add discover_fmriprep_subjects()
    - add validate_fmriprep_structure()
    - add create_fmriprep_config() factory
  Testable: unit test path construction against known fixture path

Step 3 — HCPParcellator discover_bold_files() (engine path construction)
  hcp_parcellation.py:
    - replace discover_bold_files() to use fmriprep path pattern
    - update __init__ to accept fmriprep_root instead of hcp_root
  Testable: point at a real or mock fmriprep output directory

Step 4 — Runner scripts (CLI surface)
  parcellate_hcp_subjects.py:
    - rename --hcp-root → --fmriprep-root
    - rename --hcp-parcellated-output → --fmriprep-parcellated-output
    - call create_fmriprep_config() instead of create_hcp_config()
  scripts/parcellate_missing_hcp_subjects.py: same changes
  Testable: dry-run --help output; end-to-end single subject

Step 5 — Orchestrator and path verifier (surface cleanup)
  run_pipeline.py:
    - update --hcp-root / --hcp-parcellated-output arg names
    - update kwargs forwarded to PARCELLATE_HCP_SUBJECTS step
  scripts/verify_paths.py:
    - add fmriprep path verification
  Testable: run_pipeline.py --dry-run
```

**Rationale for this order:**
- Steps 1–2 can be validated without touching any running code.
- Step 3 is the highest-risk change (path construction used in actual NIfTI loading); isolating
  it to one method in one class limits blast radius.
- Steps 4–5 are mechanical CLI renames; they can be done only after Step 3 is verified on real
  data on the cluster.

---

## Sources

- Full read of: `config/paths.py`, `config/default_config.json`,
  `tcp/preprocessing/config/data_source_config.py`, `tcp/preprocessing/hcp_parcellation.py`,
  `tcp/preprocessing/parcellate_hcp_subjects.py`, `tcp/preprocessing/run_pipeline.py`,
  `tcp/processing/data_loader.py`, `tcp/processing/config/processing_config.py`,
  `scripts/parcellate_missing_hcp_subjects.py`, `scripts/verify_paths.py`,
  `.planning/PROJECT.md`
- fmriprep 25.1.4 BIDS output naming convention (from PROJECT.md context, HIGH confidence —
  filename entity order is canonical BIDS: `sub-ID_task-X_run-NN_space-Y_res-Z_desc-W_bold.nii.gz`)
- BIDS specification for `func/` subdirectory layout (HIGH confidence — flat `func/` directory
  with fully-described filename is the BIDS standard; no intermediate `MNINonLinear/Results/` tree)

---
*Architecture research for: fMRI BIDS path-resolution migration (HCP -> fmriprep)*
*Researched: 2026-03-02*
