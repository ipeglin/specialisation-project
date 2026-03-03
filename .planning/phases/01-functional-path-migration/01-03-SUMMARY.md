---
phase: 01-functional-path-migration
plan: "03"
subsystem: api
tags: [fmriprep, bids, path-construction, hcp-parcellator, h5-metadata, re]

# Dependency graph
requires:
  - phase: 01-functional-path-migration
    provides: DATA_SOURCE_FMRIPREP constant, DataSourceConfig fmriprep fields
provides:
  - HCPParcellator.__init__() accepting fmriprep_root (not hcp_root)
  - HCPParcellator.discover_bold_files() using fmriprep func/ structure
  - Normalised H5 dataset key (task-{task}_run-{NN}) in parcellate_subject()
  - _save_h5() writing DATA_SOURCE_FMRIPREP as source attribute
  - self.hcp_root backward-compatible alias for any existing callers
affects:
  - 01-functional-path-migration (Plans 04, 05 can now use updated HCPParcellator)
  - 02-verification-and-hardening (parcellator produces correct H5 output)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Single-site path construction: discover_bold_files() owns fmriprep BIDS pattern exclusively"
    - "Normalised H5 key: task-{task}_run-{NN} derived via re.search from filename, not stem"
    - "Backward-compatible alias: self.hcp_root = self.fmriprep_root for smooth transition"

key-files:
  created: []
  modified:
    - tcp/preprocessing/hcp_parcellation.py

key-decisions:
  - "import re added at top level (stdlib) to support normalised key extraction in parcellate_subject()"
  - "self.hcp_root alias maintained as backward-compatible reference to fmriprep_root"
  - "CLI --hcp-root renamed to --fmriprep-root; both main() instantiation calls updated"

patterns-established:
  - "Pattern: fmriprep BOLD path in discover_bold_files(): {fmriprep_root}/{subject_id}/func/{subject_id}_task-{task}AP_run-{run:02d}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
  - "Pattern: normalised H5 key format is task-{task}_run-{NN} (e.g., task-hammer_run-01)"

# Metrics
duration: 3min
completed: 2026-03-03
---

# Phase 01 Plan 03: HCPParcellator fmriprep Path Migration Summary

**HCPParcellator updated to discover fmriprep BOLD files under func/ with normalised task-{task}_run-{NN} H5 keys and DATA_SOURCE_FMRIPREP provenance metadata — eliminating all MNINonLinear/Results path references**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-03T14:58:46Z
- **Completed:** 2026-03-03T16:01:04Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- Added `import re` (top-level stdlib) and `DATA_SOURCE_FMRIPREP` import to `hcp_parcellation.py`
- Updated `HCPParcellator.__init__()` to accept `fmriprep_root` parameter; set `self.fmriprep_root` and backward-compatible `self.hcp_root` alias
- Updated existence check and verbose output to reference `fmriprep_root`
- Replaced `discover_bold_files()` body entirely: now uses `fmriprep_root / subject_id / "func"` with fmriprep 25.1.4 BIDS filename pattern
- Updated `parcellate_subject()` to derive normalised H5 key (`task-{task}_run-{NN}`) via `re.search` instead of raw filename stem
- Updated `_save_h5()` source attribute from `'hcp_parcellation'` to `DATA_SOURCE_FMRIPREP`
- Updated parallel worker and both `main()` instantiation calls to use `fmriprep_root=`
- Renamed CLI `--hcp-root` to `--fmriprep-root` with updated help text and epilog examples

## Task Commits

Each task was committed atomically:

1. **Task 1: Update HCPParcellator.__init__() to accept fmriprep_root** - `fda0bfc` (feat)
2. **Task 2: Replace discover_bold_files() path construction and fix _save_h5() key + source** - `128abc9` (feat)

**Plan metadata:** `(pending docs commit)` (docs: complete plan)

## Files Created/Modified

- `tcp/preprocessing/hcp_parcellation.py` - Updated `__init__`, `discover_bold_files`, `parcellate_subject`, `_save_h5`, parallel worker, and CLI

## Decisions Made

- **import re at top level:** Added with stdlib imports (near `import sys`) as required by the plan — ensures `re.search` in `parcellate_subject()` can use the module-level import without conditional re-import.
- **self.hcp_root backward-compatible alias:** Set `self.hcp_root = self.fmriprep_root` in `__init__` so any callers that still reference `self.hcp_root` (e.g., older code) continue to work during the transition period.
- **CLI epilog updated:** Example paths in argparse epilog updated from `hcp_output` to `fmriprep-25.1.4` and `hcp_parcellated` to `fmriprep_parcellated` for accuracy.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 04 (runner script CLI update) can now pass `fmriprep_root=` to `HCPParcellator` directly; all internal path logic updated
- Plan 05 (orchestrator/verify_paths cleanup) can verify the complete end-to-end path chain
- All parcellation mathematics methods (`_parcellate_cortical`, `_parcellate_subcortical`, `_parcellate_cerebellar`, `parcellate_bold`) are unchanged
- Existing HCP `self.hcp_root` alias ensures no regression if any currently untracked callers use the old attribute name

---
*Phase: 01-functional-path-migration*
*Completed: 2026-03-03*

## Self-Check: PASSED

- FOUND: `tcp/preprocessing/hcp_parcellation.py`
- FOUND: `.planning/phases/01-functional-path-migration/01-03-SUMMARY.md`
- FOUND commit: `fda0bfc` (feat(01-03): update HCPParcellator.__init__() to accept fmriprep_root)
- FOUND commit: `128abc9` (feat(01-03): replace discover_bold_files() with fmriprep path structure and fix _save_h5())
