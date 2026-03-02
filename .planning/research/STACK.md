# Stack Research

**Domain:** Python neuroimaging pipeline — fmriprep BIDS output integration
**Researched:** 2026-03-02
**Confidence:** HIGH

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python `pathlib.Path` | stdlib | BOLD file path construction | The fmriprep 25.x BOLD filename is fully deterministic from known entities (subject, task, run, space, res, desc). `pathlib` gives OS-agnostic path composition with zero added dependencies and no indexing overhead. This is the correct choice when the pattern is fixed and all inputs are known at call time. |
| `nilearn.maskers.NiftiLabelsMasker` | >=0.10 (already installed) | Extract ROI time series from BOLD NIfTI | Already in use in the pipeline. Accepts a file path string or nibabel image object directly. No changes to the loading API are needed for fmriprep outputs versus HCP outputs — the BOLD file is standard NIfTI-1/2 in MNI152NLin2009cAsym space. |
| `nibabel` | >=5.3 (already installed) | Low-level NIfTI loading / header inspection | Required by nilearn. Direct use (`nib.load(path)`) is appropriate if header metadata (TR, affine, shape) is needed before masking. nibabel 5.3 supports NumPy 2.0 and Python 3.13 — safe with the existing Python 3.11 conda env. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pybids` (`bids`) | 0.16.4 | Query a BIDS dataset tree by entity (subject, task, space, etc.) | Only needed if: (a) subject IDs are not known ahead of time and must be discovered dynamically, or (b) the pipeline must iterate across many tasks/runs/spaces and you want validated entity-based filtering rather than manual glob. **Not needed for this milestone** — the BOLD path pattern is fully specified and deterministic. |
| `nilearn.interfaces.fmriprep` | >=0.10 | Load fmriprep confound TSV files (`load_confounds`, `load_confounds_strategy`) | Only relevant if confound regression is added later. Out of scope for this milestone (only the BOLD NIfTI is needed). |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `conda` (masters_thesis env) | Python 3.11 environment management | Do not create a new venv; use the existing `masters_thesis` conda env as specified in CLAUDE.md |
| `pytest` | Unit testing path construction and file existence checks | Existing `test_processing_core.py` and `test_simple_processing.py` suggest pytest is already in use |

## Installation

No new packages are required for the path-construction approach. The existing conda
environment already contains `nilearn`, `nibabel`, `numpy`, and `h5py`.

If `pybids` is ever needed (see decision guidance below):

```bash
# Only install if dynamic dataset discovery is required
conda install -n masters_thesis -c conda-forge pybids
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| `pathlib.Path` string construction | `pybids.BIDSLayout` | Use pybids when subject IDs or run entities are not known in advance and must be discovered by scanning the directory tree, or when multiple pipelines/spaces need to be queried interchangeably. For this milestone the pattern is fixed, so pybids would add a 2-10 second indexing overhead and a new dependency for zero benefit. |
| `nilearn.maskers.NiftiLabelsMasker` (existing) | Direct `nibabel.load()` | Use nibabel directly only if you need raw voxel arrays without atlas parcellation, e.g., for whole-brain analysis. The existing pipeline already commits to `NiftiLabelsMasker`. |
| fmriprep 25.1.x BIDS layout (default) | fmriprep legacy layout (`--output-layout legacy`) | The cluster output uses the BIDS layout (default since fmriprep 21.0). Do not use legacy layout patterns (`<root>/fmriprep/sub-.../`) — they are not present in this dataset. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `glob.glob` / `os.walk` for path discovery | Fragile against filename pattern changes; returns unsorted results; no entity validation | `pathlib.Path` construction with known entities, or `pybids.BIDSLayout` if discovery is truly needed |
| Hardcoded string concatenation (e.g., `root + "/sub-" + id + "/func/..."`) | Breaks on Windows path separators; no type safety | `pathlib.Path(root) / f"sub-{subject_id}" / "func" / filename` |
| `nilearn.interfaces.fmriprep.load_confounds` | This milestone explicitly requires only the BOLD NIfTI, not confounds. Adding confound loading now would exceed scope and may introduce signal-analysis logic prematurely. | Pass the BOLD path directly to `NiftiLabelsMasker.fit_transform()` |

## Stack Patterns by Variant

**If subject IDs are provided as a list (current use case):**
- Construct paths with `pathlib.Path` using the known BIDS entity template
- Validate file existence with `Path.exists()` before loading
- Pass string path or `Path` object to `NiftiLabelsMasker`

```python
from pathlib import Path

def resolve_bold_path(fmriprep_root: str | Path, subject_id: str) -> Path:
    root = Path(fmriprep_root)
    filename = (
        f"sub-{subject_id}_task-hammerAP_run-01"
        f"_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
    )
    return root / f"sub-{subject_id}" / "func" / filename
```

**If subject IDs must be discovered from the filesystem:**
- Use `pybids.BIDSLayout` with `derivatives=True` pointed at the fmriprep root
- Query with `.get(suffix='bold', desc='preproc', space='MNI152NLin2009cAsym', res='2', return_type='filename')`
- Accept ~2-10 s one-time indexing cost per session

**If the pipeline is later extended to multiple tasks or runs:**
- The path template should be parameterised on `task` and `run` entities, not hardcoded
- `pathlib.Path` construction scales cleanly; no library change needed

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| `nilearn >= 0.10` | `nibabel >= 5.0`, `numpy >= 2.0` | Already satisfied by the conda env (nilearn not pinned in requirements.txt but present in CLAUDE.md context; nibabel present) |
| `nibabel 5.3.0` | `numpy 2.3.x`, Python 3.11 | Oct 2024 release; supports NumPy 2.0 as required by the env's `numpy=2.3.3` |
| `pybids 0.16.4` | Python 3.8-3.11, nibabel 5.x | Latest stable as of Nov 2023. Note: fmriprep 25.2.3 changelog explicitly documents a dependency incompatibility between pybids and `universal-pathlib` that was reverted — if pybids is added, pin to `pybids==0.16.4` and avoid `universal-pathlib` conflicts. |
| fmriprep 25.1.4 BOLD output | Same file pattern as 25.0, 24.x, 23.x | The `sub-{id}/func/sub-{id}_[specifiers]_space-{space}_desc-preproc_bold.nii.gz` pattern has been stable since fmriprep 21.0 (BIDS layout default). No layout breaking changes in 25.1.4 (patch: B0FieldIdentifier fix only). |

## fmriprep 25.x BIDS Output Layout Reference

Verified from official fmriprep 25.0 documentation (stable as of 2026-03-02):

```
{fmriprep_root}/
  sub-{subject_label}/
    func/
      sub-{subject_label}_[task-{task}]_[run-{run}]_space-{space}_desc-preproc_bold.nii.gz
      sub-{subject_label}_[task-{task}]_[run-{run}]_space-{space}_desc-brain_mask.nii.gz
      sub-{subject_label}_[task-{task}]_[run-{run}]_desc-confounds_timeseries.tsv
```

For this project, the concrete resolved path is:
```
{fmriprep_root}/sub-{id}/func/sub-{id}_task-hammerAP_run-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz
```

**Key facts confirmed from official docs:**
- BIDS layout is the default since fmriprep 21.0; legacy layout (`/fmriprep/sub-.../`) is opt-in via `--output-layout legacy`. The cluster data uses BIDS layout.
- `desc-preproc_bold.nii.gz` is only generated at `--level full` (the default). It is NOT present at `--level minimal`.
- `res-2` entity denotes 2mm resolution resampling and is set by the `--output-resolution` flag at preprocessing time.
- `space-MNI152NLin2009cAsym` is the fmriprep default volumetric output space.

## Sources

- https://fmriprep.org/en/25.0.0/outputs.html — fmriprep 25.0 official output layout documentation (HIGH confidence; verified functional derivatives section directly)
- https://fmriprep.org/en/stable/changes.html — fmriprep changelog 25.1.4 and 25.2.x (HIGH confidence; confirmed 25.1.4 is a patch release with no layout changes)
- https://bids-standard.github.io/pybids/ — pybids 0.16.4 documentation (HIGH confidence; confirmed latest version and derivatives API)
- https://bids-standard.github.io/pybids/examples/pybids_tutorial.html — pybids tutorial showing derivatives loading with `scope='derivatives'` (HIGH confidence)
- https://nipy.org/nibabel/changelog.html — nibabel changelog confirming 5.3.0 as latest stable with NumPy 2.0 support (HIGH confidence)
- https://nilearn.github.io/stable/changes/whats_new.html — nilearn 0.13.1 (Feb 2026) confirmed as latest stable (HIGH confidence)

---
*Stack research for: fmriprep BIDS output integration in Python neuroimaging pipeline*
*Researched: 2026-03-02*
