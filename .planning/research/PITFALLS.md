# Pitfalls Research

**Domain:** fMRI parcellation pipeline — HCP-to-fmriprep BIDS path migration
**Researched:** 2026-03-02
**Confidence:** HIGH (all findings are grounded in reading the actual source files and official fmriprep/nilearn docs)

---

## Critical Pitfalls

### Pitfall 1: run_name key changes silently in the .h5 file, breaking the downstream reader

**What goes wrong:**
The current code derives the .h5 dataset key from the BOLD filename stem:

```python
# hcp_parcellation.py line 291
run_name = bold_path.stem.replace('.nii', '')
# HCP result: "task-hammerAP_run-01_bold"
```

After migration, the fmriprep filename stem will be:
`sub-NDARINVXXXXX_task-hammerAP_run-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold`

This becomes the .h5 dataset key instead. The downstream reader in `tcp/processing/main.py` loads the first key positionally:

```python
a_group_key = list(file.keys())[0]   # line 1404 — index, not name
```

This means it will silently load data from whatever the first key is, regardless of the key name. The **correct** data will still be loaded if there is only one run, but the key name embedded in any existing .h5 files will permanently differ between HCP-sourced and fmriprep-sourced outputs. If any code anywhere looks up the key by name (e.g., future refactor), it will silently get a `KeyError` or wrong data.

**Why it happens:**
The key name is not specified explicitly — it is derived automatically from the filename path stem with no sanitisation step. The longer fmriprep BIDS filename entities pass through unchanged.

**How to avoid:**
When writing the new fmriprep path construction, **normalise the .h5 dataset key to a fixed format** independent of the source filename. A safe convention is `task-{task}_run-{run:02d}` (e.g., `task-hammerAP_run-01`). Set this explicitly in `_save_h5()` instead of deriving it from the path.

**Warning signs:**
- `h5py.File.keys()` output looks like `['sub-NDARINVXXXXX_task-...']` instead of `['task-hammerAP_run-01_bold']`
- Any future code that does `file['task-hammerAP_run-01_bold']` raises `KeyError` silently masked by a try/except

**Phase to address:** Path migration phase (the commit that changes `discover_bold_files()` and `_save_h5()`).

---

### Pitfall 2: MNI template mismatch between atlases and fmriprep BOLD silently shifts parcel boundaries

**What goes wrong:**
The cortical atlas (`400Parcels_Yeo2011_17Networks_FSLMNI152_2mm.nii.gz`) is registered to the FSL MNI152 1mm/2mm template (MNI152Lin). fmriprep 25.1.4's default volumetric output space is `MNI152NLin2009cAsym` — a nonlinear asymmetric template. These are **different templates**. They are not aligned voxel-for-voxel.

`NiftiLabelsMasker` with `resampling_target="data"` resamples the atlas to the BOLD voxel grid using nearest-neighbour interpolation, guided solely by the affine transform. When the BOLD affine corresponds to a different MNI template from the atlas affine, the resampling still completes — no error is raised — but the atlas parcels land in subtly wrong voxel positions. The subcortical atlas (`Tian_Subcortex_S2_3T.nii`) may be especially sensitive because small subcortical structures occupy only a few voxels.

**Why it happens:**
Nilearn's `resampling_target="data"` performs affine-based resampling without any template-identity check. It trusts that both images are in the same coordinate space. If they are registered to different MNI variants, the code does not warn — it just resamples.

**How to avoid:**
1. Confirm with the atlas documentation which MNI template each atlas is registered to.
2. Verify that fmriprep was run with `--output-spaces MNI152NLin2009cAsym:res-2` and not a different template.
3. After migration, overlay the atlas on one subject's fmriprep BOLD in a viewer (e.g., FSLeyes or nilearn `plot_roi`) and visually verify that parcel boundaries land on anatomically plausible grey-matter structures.
4. If the atlas is registered to MNI152Lin, consider whether the atlas needs to be resliced to MNI152NLin2009cAsym — this is a **research decision** outside the scope of this migration and should not be changed by AI.

**Warning signs:**
- Parcel counts (n_labels) change after migration (nilearn may drop empty parcels)
- Parcellation produces fewer than 400 cortical parcels — the existing `ValueError` in `parcellate_bold()` will catch this loudly
- Visual check shows parcels shifted toward edge of brain mask

**Phase to address:** Verification phase, immediately after the first successful fmriprep subject parcellation.

---

### Pitfall 3: Subject discovery glob changes silently admit wrong subjects or miss subjects

**What goes wrong:**
The old discovery in `discover_hcp_subjects()` globbed `{hcp_root}/sub-*/MNINonLinear/Results/` and then verified the presence of a BOLD `.nii.gz` file inside a task subdirectory before accepting the subject. The new discovery must glob `{fmriprep_root}/sub-*/func/` and verify the presence of the fmriprep BOLD file. If the glob pattern is written incorrectly — e.g., matching `sub-*/func/` without verifying the full filename — the pipeline can:

- Include subjects who have a `func/` directory but whose fmriprep run failed (partial outputs with only mask files, no `_desc-preproc_bold.nii.gz`)
- Miss subjects if the `func/` directory is missing entirely (fmriprep crashed before writing it)
- Accept subjects from fmriprep runs with a **different space or resolution** entity (e.g., `_space-T1w_` instead of `_space-MNI152NLin2009cAsym_res-2_`), since fmriprep can output multiple spaces simultaneously

**Why it happens:**
The old code verified the BOLD file existence explicitly inside the task subdirectory. The flat `func/` layout requires verifying the full BIDS filename, not just the directory. Forgetting this check is easy when adapting the glob-and-verify pattern.

**How to avoid:**
In the new `discover_subjects()` implementation, verify the full BIDS filename exists:
```python
bold_file = func_dir / f"{subject_id}_task-hammerAP_run-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
if bold_file.exists() and bold_file.is_file():
    subjects.append(...)
```
Do not rely on the presence of the `func/` directory alone.

**Warning signs:**
- Subject count after migration differs significantly from expected (fmriprep has more or fewer subjects than HCP)
- `FileNotFoundError` appears for a subject during parcellation even though `discover_subjects()` returned that subject ID
- A subject's `func/` directory exists but contains only `_desc-brain_mask.nii.gz` (minimal processing level, no BOLD)

**Phase to address:** Path migration phase (`discover_hcp_subjects()` replacement).

---

### Pitfall 4: The HCP `data_source` metadata tag in the manifest propagates stale de-meaning logic to fmriprep data

**What goes wrong:**
`tcp/processing/main.py` checks `data_source == 'hcp'` to decide whether to de-mean each ROI timeseries before analysis:

```python
# main.py lines 1414-1425
data_source = subject_metadata.get('data_source', 'datalad')
if data_source == 'hcp':
    data = data - roi_means   # de-meaned here
```

The parcellation `_save_h5()` writes `f.attrs['source'] = 'hcp_parcellation'` into the .h5 file, but the manifest `data_source` field is set separately via `parcellate_hcp_subjects.py` subject tracking. If the migrated code continues writing `data_source: 'hcp'` into the manifest (because the class is still called `HCPParcellator`), the de-meaning will still be applied to fmriprep data. Whether this is correct or not is a **research decision** — the key risk is that it happens without anyone noticing.

**Why it happens:**
The `data_source` tag is a string literal set at multiple places — in the manifest JSON and in the .h5 attrs. When renaming the class/config from HCP to fmriprep, it is easy to miss one of these string literals. The processing pipeline reads this tag to alter analysis behaviour, so a stale value changes results silently.

**How to avoid:**
When updating the manifest integration code, search for all occurrences of the string `'hcp'` used as a data source tag (not just in filenames) and update them consistently. The relevant occurrences are:
- `hcp_parcellation.py` line 407: `f.attrs['source'] = 'hcp_parcellation'`
- Any place that writes `data_source: 'hcp'` into the manifest JSON

Define the source tag as a named constant (e.g., `DATA_SOURCE_FMRIPREP = "fmriprep"`) rather than a bare string to prevent divergence.

**Warning signs:**
- `grep -r "'hcp'" tcp/` still returns hits in path-unrelated contexts after migration
- The processing `main.py` prints "Detected HCP data - applying de-meaning" for fmriprep subjects

**Phase to address:** Path migration phase, as part of renaming/updating `DataSourceConfig` and `HCPParcellator`.

---

### Pitfall 5: `verify_paths.py` and CLI help text reference `hcp_output` after migration — silent documentation rot

**What goes wrong:**
`scripts/verify_paths.py` explicitly references `get_data_path('hcp_output')` (lines 69 and 105) and prints `HCP parcellated output: ...` and `HCP output at: ...` in its output. The `.env.example` file and all CLI `--help` strings reference `--hcp-root`. After migration, these will:
- Show a path that no longer exists or is no longer the operative path
- Mislead future operators who run `verify_paths.py` to diagnose path issues
- Cause CI/smoke-test confusion if `verify_paths.py` is run as a health check

**Why it happens:**
Path verification scripts are typically written once and not kept in sync with migrations. The `hcp_output` string is hardcoded as a `get_data_path()` argument, not derived from a config constant.

**How to avoid:**
Update `scripts/verify_paths.py` in the same commit or PR as the path migration. Replace `get_data_path('hcp_output')` with the fmriprep root path lookup. Update the printed labels from "HCP output" to "fmriprep output". Update `--hcp-root` CLI argument labels or add aliases.

**Warning signs:**
- `python scripts/verify_paths.py --check-exists` prints "NOT FOUND" for the old HCP path but you have already confirmed fmriprep data is present at the new path
- `--help` output still says `--hcp-root` after migration

**Phase to address:** Path migration phase — bundle `verify_paths.py` update in the same change set.

---

### Pitfall 6: `integrate_cross_analysis.py` also has `--hcp-root` CLI args — a missed update breaks pipeline orchestration

**What goes wrong:**
`tcp/preprocessing/integrate_cross_analysis.py` (lines 592–608, confirmed by grep) also accepts `--hcp-root` and `--hcp-parcellated-output` CLI arguments and passes them to `DataSourceConfig`. `run_pipeline.py` (lines 375–376 and 694–716) constructs the CLI call for `integrate_cross_analysis.py` by passing `kwargs['hcp_root']`. If only the files listed in PROJECT.md are updated but `integrate_cross_analysis.py` is missed, the pipeline orchestration step will pass the wrong root path or fail with a validation error when `hcp_root` is no longer set.

**Why it happens:**
`integrate_cross_analysis.py` is not on the explicit list in PROJECT.md under "Key files to change." The run_pipeline orchestrator dynamically constructs shell commands that pass `--hcp-root`, so the coupling is indirect and easy to miss during a targeted file-by-file migration.

**How to avoid:**
Before beginning the migration, do a full-codebase search:
```bash
grep -r "hcp.root\|hcp_root\|hcp-root\|MNINonLinear" tcp/ scripts/ --include="*.py"
```
Treat every match as a potential change site. The PROJECT.md list is a starting point, not an exhaustive inventory.

**Warning signs:**
- `run_pipeline.py` raises a subprocess error on the `integrate_cross_analysis` step with a message about an unrecognised argument or missing path
- `integrate_cross_analysis.py` runs without error but produces a manifest that references the old HCP path

**Phase to address:** Path migration phase — audit all call sites before writing any code.

---

### Pitfall 7: `NiftiLabelsMasker` with `resampling_target="data"` at `res-2` vs HCP 2mm — subtle voxel count difference

**What goes wrong:**
HCP BOLD at 2mm isotropic in MNI space and fmriprep BOLD at `res-2` in MNI152NLin2009cAsym may have different FOV (field of view), different number of voxels, or a different affine origin even if both are nominally "2mm." The `NiftiLabelsMasker` uses the BOLD image's shape and affine as the resampling target. If the voxel count changes sufficiently, the parcellated ROI timeseries may have slightly different numerical values because the atlas is resampled to a different grid. This will not raise an error, but it will make pre-/post-migration .h5 outputs numerically non-comparable.

**Why it happens:**
HCP uses the FSL MNI152Lin 91x109x91 2mm template. fmriprep `res-2` in MNI152NLin2009cAsym uses the 97x115x97 2mm template. The FOV is larger in fmriprep output. When the atlas (designed for 91x109x91) is resampled to 97x115x97 grid, the parcel boundaries at the brain edge shift by up to 1 voxel.

**How to avoid:**
After the first successful test parcellation, print the BOLD image shape and compare to the expected template shape: a fmriprep `res-2 MNI152NLin2009cAsym` BOLD should be 97x115x97 at 2mm. Document this shape in a post-migration verification check. Do not mix pre-migration (HCP) and post-migration (fmriprep) .h5 timeseries files in the same analysis run.

**Warning signs:**
- `bold_img.shape` prints `(97, 115, 97, T)` for fmriprep vs `(91, 109, 91, T)` for HCP
- Parcel count or timeseries values change between runs on the same subject with different data sources

**Phase to address:** Verification phase.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Keep class named `HCPParcellator` and just change path logic inside it | No rename churn | Confusing for any future reader; class name contradicts its function | Never — rename it |
| Leave `data_source = 'hcp'` string literal in manifest output | No manifest format change | `main.py` applies HCP-specific de-meaning to fmriprep data silently | Never |
| Derive .h5 key from filename stem without normalisation | Simple to implement | Any future key-by-name lookup breaks; long fmriprep names pollute .h5 | Never for new files |
| Skip updating `verify_paths.py` | Fewer files to change | Verification script gives misleading output; operators make wrong diagnoses | Never |
| Glob `func/` directory without verifying BOLD file exists | Simpler discovery code | Includes partial fmriprep outputs (mask-only) or wrong-space files | Never |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| `NiftiLabelsMasker` + fmriprep BOLD | Assuming atlas and BOLD are in the same template — they may not be | Confirm template identity of each atlas file and verify against fmriprep output space entity |
| fmriprep BIDS flat `func/` layout | Globbing for the directory and assuming a BOLD file is inside | Check for the full BIDS filename with all entities (space, res, desc) |
| .h5 dataset key naming | Using `bold_path.stem` directly as key, which embeds the full fmriprep filename | Normalise to a fixed short key format at write time |
| `run_pipeline.py` subprocess construction | Updating `parcellate_hcp_subjects.py` CLI arg names but not `run_pipeline.py` subprocess kwarg building | Audit `run_pipeline.py` lines 375–376 and 694–716 |
| `integrate_cross_analysis.py` | Not updating it because it is not in the PROJECT.md file list | It accepts `--hcp-root` — must be updated alongside the other files |
| `data_source` metadata tag | Keeping `'hcp'` string in manifest — triggers de-meaning branch in `main.py` | Update the tag to `'fmriprep'` and search all occurrences |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Recreating `NiftiLabelsMasker` per-subject in parallel jobs | Redundant atlas file I/O on every subject; slow on IDUN cluster NFS | Pre-load atlas images once outside the parallel loop; pass loaded `Nifti1Image` objects | Any parallel batch >10 subjects |
| Globbing `func/` with `Path.glob("sub-*")` at fmriprep root | Iterates all subject directories on every discovery call | Cache the subject list or do discovery once at pipeline start | N/A for single runs; matters for resume/rerun |

---

## "Looks Done But Isn't" Checklist

After migration, verify each item before calling the migration complete:

- [ ] **Path construction:** Run `discover_subjects()` on a real fmriprep directory and confirm the returned subject IDs match known subjects with valid BOLD files — not just directory names
- [ ] **File existence:** Confirm `get_fmriprep_bold_path()` returns the correct path for at least one subject and that `Path.exists()` is True on IDUN cluster
- [ ] **Parcel count:** Run `parcellate_bold()` on one fmriprep subject and confirm the output shape is `(434, T)` — the existing `ValueError` guards will catch deviations loudly
- [ ] **Atlas alignment:** Visually overlay the cortical atlas on the fmriprep BOLD reference image in FSLeyes and confirm parcels land on grey matter
- [ ] **BOLD image shape:** Confirm fmriprep BOLD shape is `(97, 115, 97, T)` — not the HCP `(91, 109, 91, T)` shape
- [ ] **H5 key:** Open a newly written .h5 file with `h5py` and print `list(file.keys())` — confirm the key format is the normalised form, not the long filename stem
- [ ] **data_source tag:** Open the updated manifest JSON and confirm `data_source` values for fmriprep subjects are not `'hcp'`
- [ ] **verify_paths.py output:** Run `python scripts/verify_paths.py --check-exists` and confirm it references the fmriprep root, not `hcp_output`
- [ ] **CLI help text:** Run `python tcp/preprocessing/parcellate_hcp_subjects.py --help` and confirm argument names and help strings do not reference HCP-specific language
- [ ] **Cluster paths:** Confirm the IDUN cluster fmriprep root `/cluster/projects/itea_lille-ie/Transdiagnostic/output/fmriprep-25.1.4` resolves correctly before submitting a full batch job

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Wrong .h5 key format in written files | LOW | Delete output .h5 files and re-run parcellation; no manifest or source data changes needed |
| MNI template mismatch confirmed by visual check | HIGH | Requires atlas reslicing to correct MNI template — research decision; cannot be AI-generated |
| Stale `data_source='hcp'` in manifest | LOW | Edit or regenerate the manifest JSON; no re-parcellation needed |
| `integrate_cross_analysis.py` missed update | LOW | Update it and rerun the integration step; no re-parcellation needed |
| Wrong subject list (glob without file existence check) | LOW | Fix discovery logic, rerun subject enumeration |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| .h5 key name from filename stem | Path migration — update `_save_h5()` | `list(h5py.File(path,'r').keys())[0]` equals normalised key |
| MNI template mismatch | Verification phase — visual QA after first subject | Parcel count == 434; visual atlas overlay check |
| Discovery glob without file existence check | Path migration — `discover_subjects()` implementation | Manual count of returned subjects vs known fmriprep outputs |
| Stale `data_source='hcp'` tag | Path migration — grep for string literals | Manifest JSON inspection; no "Detected HCP data" in processing log |
| `verify_paths.py` not updated | Path migration — bundle in same changeset | `verify_paths.py --check-exists` shows fmriprep path as EXISTS |
| `integrate_cross_analysis.py` missed | Pre-migration audit — grep all call sites | Pipeline orchestration runs end-to-end without subprocess errors |
| `data_source_config.hcp_root` field name | Path migration — rename field | No remaining `hcp_root` attribute references in fmriprep-only code paths |
| FOV difference (97x115x97 vs 91x109x91) | Verification phase — shape check on loaded BOLD | `bold_img.shape[:3] == (97, 115, 97)` assertion or log |

---

## Sources

- fmriprep 25.1.4 official outputs documentation: https://fmriprep.org/en/25.1.4/outputs.html (confirmed: functional derivatives live in `func/` flat layout; default space is `MNI152NLin2009cAsym`)
- nilearn `NiftiLabelsMasker` API documentation: https://nilearn.github.io/stable/modules/generated/nilearn.maskers.NiftiLabelsMasker.html (confirmed: `resampling_target="data"` resamples atlas to BOLD grid with no template-identity check)
- Project source code read directly: `tcp/preprocessing/hcp_parcellation.py`, `tcp/preprocessing/config/data_source_config.py`, `tcp/preprocessing/parcellate_hcp_subjects.py`, `tcp/preprocessing/run_pipeline.py`, `tcp/processing/main.py`, `scripts/verify_paths.py`, `scripts/parcellate_missing_hcp_subjects.py`, `config/default_config.json`

---

*Pitfalls research for: HCP-to-fmriprep BIDS parcellation pipeline migration*
*Researched: 2026-03-02*
