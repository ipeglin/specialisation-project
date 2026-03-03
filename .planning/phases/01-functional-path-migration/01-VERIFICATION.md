---
phase: 01-functional-path-migration
verified: 2026-03-03T00:00:00Z
status: passed
score: 7/7 must-haves verified
re_verification: false
gaps: []
---

# Phase 01: Functional Path Migration — Verification Report

**Phase Goal:** Replace all HCP-specific path logic with fmriprep BIDS path logic so the pipeline can locate, load, and parcellate fmriprep BOLD files and write correctly-structured .h5 outputs with accurate provenance metadata.
**Verified:** 2026-03-03
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| #  | Truth                                                                                                  | Status     | Evidence                                                                                                |
|----|--------------------------------------------------------------------------------------------------------|------------|---------------------------------------------------------------------------------------------------------|
| 1  | `config/default_config.json` has `fmriprep_output` key for all 4 platforms                             | VERIFIED   | Lines 12, 26, 40, 54 — all 4 platforms have `fmriprep_output` and `fmriprep_parcellated_output`        |
| 2  | `config/paths.py` exports `get_fmriprep_output_path()` and `get_fmriprep_parcellated_output_path()` routed through `_get_base_path` | VERIFIED   | Lines 403–434: both functions call `_path_config._get_base_path('fmriprep_output')` / `'fmriprep_parcellated_output'` |
| 3  | `data_source_config.py` has `DATA_SOURCE_FMRIPREP`, `fmriprep_root` field, `get_fmriprep_bold_path()`, `discover_fmriprep_subjects()`, `validate_fmriprep_structure()`, `create_fmriprep_config()` | VERIFIED   | Lines 20, 54–55, 188, 221, 248, 370 — all 6 items present and substantive                              |
| 4  | `hcp_parcellation.py`: `__init__` accepts `fmriprep_root`; `discover_bold_files()` searches `func/`; `_save_h5` uses `DATA_SOURCE_FMRIPREP`; H5 key uses `re.search` for `task-{task}_run-{NN}` | VERIFIED   | Lines 46–55 (fmriprep_root param + self.fmriprep_root); 235 (func/ path); 413 (DATA_SOURCE_FMRIPREP); 295–297 (re.search + normalised_key) |
| 5  | All 4 runner scripts accept `--fmriprep-root` and `--fmriprep-parcellated-output`; no `--hcp-root` arg; `create_fmriprep_config()` called for hcp data-source-type | VERIFIED   | Confirmed across all 4 scripts; no `add_argument.*--hcp-root` found; `create_fmriprep_config` called in parcellate_hcp_subjects.py (line 256), integrate_cross_analysis.py (line 595), parcellate_missing_hcp_subjects.py (line 533), run_pipeline.py (passes through kwargs) |
| 6  | `scripts/verify_paths.py` imports `get_fmriprep_output_path()` and displays fmriprep path (not hcp_output) in summary | VERIFIED   | Line 35: `get_fmriprep_output_path` imported; lines 106–108, 131–132, 153: summary section shows `fmriprep output (fmriprep-25.1.4)` |
| 7  | No `MNINonLinear` references remain in path logic across all 5 target files                             | VERIFIED   | Grep across all 5 files returned zero matches                                                           |

**Score:** 7/7 truths verified

---

## Required Artifacts

| Artifact                                                           | Expected                                          | Status     | Details                                                                                          |
|--------------------------------------------------------------------|---------------------------------------------------|------------|--------------------------------------------------------------------------------------------------|
| `config/default_config.json`                                       | `fmriprep_output` key in all 4 platforms          | VERIFIED   | Present in macos (L12), windows (L26), idun (L40), linux (L54)                                  |
| `config/paths.py`                                                  | `get_fmriprep_output_path()` + `get_fmriprep_parcellated_output_path()` | VERIFIED   | Lines 403–434; both use `_get_base_path`                                                         |
| `tcp/preprocessing/config/data_source_config.py`                   | Full fmriprep API surface                         | VERIFIED   | `DATA_SOURCE_FMRIPREP`, `fmriprep_root` field, 3 methods, 1 factory — all substantive           |
| `tcp/preprocessing/hcp_parcellation.py`                            | fmriprep-native path logic throughout             | VERIFIED   | `fmriprep_root` param, `func/` traversal, `DATA_SOURCE_FMRIPREP`, normalised key with `re.search` |
| `tcp/preprocessing/parcellate_hcp_subjects.py`                     | `--fmriprep-root`, `--fmriprep-parcellated-output`; `create_fmriprep_config` | VERIFIED   | Lines 224–226, 256–259                                                                           |
| `tcp/preprocessing/integrate_cross_analysis.py`                    | `--fmriprep-root`, `--fmriprep-parcellated-output`; `create_fmriprep_config` | VERIFIED   | Lines 567–569, 594–595                                                                           |
| `tcp/preprocessing/run_pipeline.py`                                | `--fmriprep-root`, `--fmriprep-parcellated-output`; forwarded to subprocess | VERIFIED   | Lines 676–678, 375–378                                                                           |
| `scripts/parcellate_missing_hcp_subjects.py`                       | `--fmriprep-root`, `--fmriprep-parcellated-output`; `create_fmriprep_config` | VERIFIED   | Lines 499–501, 533–535                                                                           |
| `scripts/verify_paths.py`                                          | Imports and prints fmriprep path in summary       | VERIFIED   | Import line 35; summary lines 106–108, 153                                                       |

---

## Key Link Verification

| From                            | To                                | Via                                       | Status  | Details                                                                                     |
|---------------------------------|-----------------------------------|-------------------------------------------|---------|---------------------------------------------------------------------------------------------|
| `config/default_config.json`    | `config/paths.py`                 | `_get_base_path('fmriprep_output')`       | WIRED   | JSON key read at runtime via `_get_base_path`; no hardcoding                                |
| `config/paths.py`               | `scripts/verify_paths.py`         | `from config.paths import get_fmriprep_output_path` | WIRED   | Explicit import, used in JSON output and summary print                                      |
| `data_source_config.create_fmriprep_config` | `HCPParcellator.__init__` | `fmriprep_root=data_source_config.fmriprep_root` | WIRED   | All 4 runner scripts pass `fmriprep_root` from config to parcellator                       |
| `HCPParcellator.discover_bold_files` | `func/` directory          | `self.fmriprep_root / subject_id / "func"` | WIRED   | No `MNINonLinear/Results/` — pure BIDS `func/` path                                         |
| `HCPParcellator._save_h5`       | `.h5` provenance metadata         | `f.attrs['source'] = DATA_SOURCE_FMRIPREP` | WIRED   | Constant imported from `data_source_config`, stored in H5 attributes                       |
| `parcellate_subject` run loop   | H5 dataset keys                   | `re.search(r'run-(\d+)', stem)` + `f"task-{task}_run-{run_num}"` | WIRED   | Normalised key written as H5 dataset name                                                   |

---

## Anti-Patterns Found

| File                         | Line  | Pattern                                      | Severity | Impact                                                                                  |
|------------------------------|-------|----------------------------------------------|----------|-----------------------------------------------------------------------------------------|
| `hcp_parcellation.py`        | 64–65 | `TODO: Add Buckner cerebellar atlas`          | INFO     | Pre-existing known gap; cerebellar zeros placeholder is intentional and logged at runtime |
| `hcp_parcellation.py`        | 391   | `output_dir = Path.cwd() / "hcp_parcellated"` | WARNING  | CLI fallback default in `_save_h5`; only applies if `output_dir=None` is passed directly — normal paths always supply `output_dir` via runner scripts |

Neither anti-pattern is a blocker. The cerebellar TODO predates this phase and is tracked separately. The fallback `output_dir` is a defensive default that is never reached in normal pipeline invocations.

---

## Human Verification Required

None identified. All path logic, wiring, and argument plumbing is statically verifiable from source.

---

## Summary

All 7 must-haves pass. The fmriprep BIDS path migration is complete and correctly wired end-to-end:

- `default_config.json` provides platform-specific `fmriprep_output` paths for all 4 supported platforms.
- `config/paths.py` exposes two named convenience functions (`get_fmriprep_output_path`, `get_fmriprep_parcellated_output_path`) both routed through the existing `_get_base_path` mechanism.
- `DataSourceConfig` carries `fmriprep_root` and `fmriprep_parcellated_output` fields; all four discovery/validation/path methods and the `create_fmriprep_config` factory are substantive and non-stub.
- `HCPParcellator.__init__` accepts `fmriprep_root` (with a backward-compatible `self.hcp_root` alias); `discover_bold_files` traverses `func/` not `MNINonLinear/Results/`; `_save_h5` writes `DATA_SOURCE_FMRIPREP` as provenance; run keys are normalised via `re.search`.
- All 4 runner scripts expose `--fmriprep-root` and `--fmriprep-parcellated-output` with no residual `--hcp-root` argument; each calls `create_fmriprep_config` when `--data-source-type hcp` is specified.
- `verify_paths.py` imports and surfaces `get_fmriprep_output_path` in both its JSON output and human-readable summary section.
- Zero `MNINonLinear` references remain in the five target files.

The phase goal is fully achieved.

---

_Verified: 2026-03-03_
_Verifier: Claude (gsd-verifier)_
