---
phase: 01-functional-path-migration
plan: "02"
subsystem: api
tags: [fmriprep, bids, path-construction, dataclass, factory-function]

# Dependency graph
requires: []
provides:
  - DATA_SOURCE_FMRIPREP constant for provenance-safe .h5 metadata
  - DataSourceConfig.fmriprep_root and fmriprep_parcellated_output fields
  - DataSourceConfig.get_fmriprep_bold_path() for per-subject BIDS path construction
  - DataSourceConfig.discover_fmriprep_subjects() for subject enumeration from fmriprep output
  - DataSourceConfig.validate_fmriprep_structure() for func/ directory validation
  - DataSourceConfig.is_fmriprep_enabled() convenience predicate
  - create_fmriprep_config() factory function for fmriprep-only mode
affects:
  - 01-functional-path-migration (Plans 03, 04, 05 depend on these fields and methods)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Factory function as CLI adapter: create_fmriprep_config() converts flat args to DataSourceConfig"
    - "Single-site path construction: all fmriprep BIDS path logic lives in get_fmriprep_bold_path()"
    - "Compatibility shim: hcp_root aliased to fmriprep_root in factory until runner scripts updated in Plan 04"

key-files:
  created: []
  modified:
    - tcp/preprocessing/config/data_source_config.py

key-decisions:
  - "Used DataSourceType.HCP in create_fmriprep_config() to maintain compatibility with existing runner code until Plan 04"
  - "Added hcp_root=fmriprep_root shim to avoid cascade failure before Plan 04 updates runner scripts"
  - "fmriprep_root not validated in __post_init__ to avoid breaking HCP source_type validation"

patterns-established:
  - "Pattern: fmriprep BIDS filename always constructed as {subject_id}_task-{task}AP_run-{run:02d}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
  - "Pattern: discover_fmriprep_subjects() verifies full BIDS filename (not just func/ dir) to exclude partial outputs"

# Metrics
duration: 2min
completed: 2026-03-03
---

# Phase 01 Plan 02: DataSourceConfig fmriprep Fields and Methods Summary

**DataSourceConfig extended with fmriprep_root field, four path methods (BIDS filename construction, subject discovery, structure validation), and create_fmriprep_config() factory — making it the single authority for all fmriprep 25.1.4 path logic**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-03T14:53:58Z
- **Completed:** 2026-03-03T14:55:30Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- Added `DATA_SOURCE_FMRIPREP = "fmriprep_parcellation"` constant — prevents stale string literals in `_save_h5()` and any future callers
- Extended `DataSourceConfig` with `fmriprep_root` and `fmriprep_parcellated_output` Optional[Path] fields with full Path conversion in `__post_init__`
- Added `get_fmriprep_bold_path()` implementing the exact fmriprep 25.1.4 BIDS filename pattern with subject-prefix normalisation
- Added `discover_fmriprep_subjects()` which globs `func/` and verifies the full BIDS filename (not just directory existence) to exclude partial outputs
- Added `validate_fmriprep_structure()` checking that `func/` directory exists for a subject
- Added `is_fmriprep_enabled()` convenience predicate
- Added `create_fmriprep_config()` factory with deliberate `hcp_root` compatibility shim for Plan 04 transition
- All existing HCP fields, methods, and factory functions unchanged (verified)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add DATA_SOURCE_FMRIPREP constant and fmriprep_root field** - `94c2834` (feat)
2. **Task 2: Add fmriprep path methods and create_fmriprep_config factory** - `4bbc874` (feat)

**Plan metadata:** `(pending docs commit)` (docs: complete plan)

## Files Created/Modified

- `tcp/preprocessing/config/data_source_config.py` - Added constant, two new dataclass fields, four methods, and factory function

## Decisions Made

- **DataSourceType.HCP in create_fmriprep_config():** Uses `DataSourceType.HCP` internally to avoid breaking existing runner code that checks `source_type`. Plan 04 will update runner scripts to use `fmriprep_root` directly, at which point the shim can be removed.
- **No validation of fmriprep_root in __post_init__:** The HCP source type already validates `hcp_root` — adding fmriprep_root validation there would conflict. Validation is deferred to `validate_fmriprep_structure()` and `discover_fmriprep_subjects()` where it belongs.
- **Full BIDS filename check in discover_fmriprep_subjects():** Verifies `desc-preproc_bold.nii.gz` exists (not just `func/` dir) to avoid enumerating subjects with incomplete fmriprep outputs.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 03 (HCPParcellator `discover_bold_files` update) can now delegate path construction to `DataSourceConfig.get_fmriprep_bold_path()`
- Plan 04 (runner script CLI update) can call `create_fmriprep_config()` and remove the `hcp_root` shim
- All existing HCP code continues to work; no regressions introduced

---
*Phase: 01-functional-path-migration*
*Completed: 2026-03-03*

## Self-Check: PASSED

- FOUND: `tcp/preprocessing/config/data_source_config.py`
- FOUND: `.planning/phases/01-functional-path-migration/01-02-SUMMARY.md`
- FOUND commit: `94c2834` (feat(01-02): add DATA_SOURCE_FMRIPREP constant and fmriprep fields)
- FOUND commit: `4bbc874` (feat(01-02): add fmriprep path methods and create_fmriprep_config factory)
