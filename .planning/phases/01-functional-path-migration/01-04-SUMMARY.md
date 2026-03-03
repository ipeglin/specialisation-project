---
phase: 01-functional-path-migration
plan: "04"
subsystem: api
tags: [fmriprep, cli, argparse, runner-scripts, pipeline-orchestrator]

# Dependency graph
requires:
  - phase: 01-functional-path-migration
    provides: create_fmriprep_config() factory function (from Plan 02)
provides:
  - All four runner/orchestrator scripts accept --fmriprep-root and --fmriprep-parcellated-output CLI arguments
  - run_pipeline.py forwards fmriprep_root and fmriprep_parcellated_output to subprocess steps
  - All scripts call create_fmriprep_config() for the 'hcp' data-source-type branch
  - --hcp-root removed from all four entry points
affects:
  - 01-functional-path-migration (Plan 05 - verify_paths.py cleanup can now assume fmriprep CLI surface)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "CLI surface: --fmriprep-root replaces --hcp-root across all runner scripts"
    - "subprocess passthrough: run_pipeline.py forwards --fmriprep-root flag to child scripts via kwargs dict"

key-files:
  created: []
  modified:
    - tcp/preprocessing/parcellate_hcp_subjects.py
    - scripts/parcellate_missing_hcp_subjects.py
    - tcp/preprocessing/integrate_cross_analysis.py
    - tcp/preprocessing/run_pipeline.py

key-decisions:
  - "combined mode: --fmriprep-root passed as hcp_root to create_combined_config() — internal field name unchanged, only CLI surface renamed"
  - "create_hcp_config() no longer called for 'hcp' branch; replaced with create_fmriprep_config()"

patterns-established:
  - "Pattern: all scripts that accept root path arguments use --fmriprep-root (not --hcp-root)"
  - "Pattern: run_pipeline.py kwargs dict mirrors CLI arg names (fmriprep_root, not hcp_root)"

# Metrics
duration: 9min
completed: 2026-03-03
---

# Phase 01 Plan 04: Runner Scripts CLI Migration to fmriprep Arguments Summary

**All four entry point scripts (parcellate_hcp_subjects.py, parcellate_missing_hcp_subjects.py, integrate_cross_analysis.py, run_pipeline.py) updated to accept --fmriprep-root and --fmriprep-parcellated-output, with run_pipeline.py forwarding these flags to subprocess steps and the 'hcp' branch calling create_fmriprep_config()**

## Performance

- **Duration:** 9 min
- **Started:** 2026-03-03T14:58:54Z
- **Completed:** 2026-03-03T15:07:50Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Renamed `--hcp-root` to `--fmriprep-root` and `--hcp-parcellated-output` to `--fmriprep-parcellated-output` in all four scripts
- `parcellate_hcp_subjects.py`: `create_hcp_config()` call replaced with `create_fmriprep_config()`; runner prints updated to `fmriprep_root`/`fmriprep_parcellated_output`; `HCPParcellator` instantiated with `fmriprep_root`
- `scripts/parcellate_missing_hcp_subjects.py`: same argparse and config factory changes plus `MissingHCPParcellator.__init__`, `check_existing_h5_files`, and `parcellate_subjects` updated to use `fmriprep_root`/`fmriprep_parcellated_output`
- `integrate_cross_analysis.py`: 'hcp' branch uses `create_fmriprep_config()`; 'combined' branch validation checks `fmriprep_root`/`fmriprep_parcellated_output`
- `run_pipeline.py`: kwargs dict renamed to `fmriprep_root`/`fmriprep_parcellated_output`; `_execute_step` subprocess builder passes `--fmriprep-root`/`--fmriprep-parcellated-output` flags

## Task Commits

Each task was committed atomically:

1. **Task 1: Update parcellate_hcp_subjects.py and scripts/parcellate_missing_hcp_subjects.py CLI arguments** - `d746522` (feat)
2. **Task 2: Update integrate_cross_analysis.py and run_pipeline.py to use fmriprep arguments** - `34c7ba9` (feat)

**Plan metadata:** `(pending docs commit)` (docs: complete plan)

## Files Created/Modified

- `tcp/preprocessing/parcellate_hcp_subjects.py` - Renamed CLI args, updated config factory call, updated runner class to use fmriprep fields
- `scripts/parcellate_missing_hcp_subjects.py` - Renamed CLI args, updated config factory call, updated MissingHCPParcellator to use fmriprep fields
- `tcp/preprocessing/integrate_cross_analysis.py` - Renamed CLI args, updated 'hcp' branch to call create_fmriprep_config()
- `tcp/preprocessing/run_pipeline.py` - Renamed CLI args, updated kwargs dict, updated subprocess argument building

## Decisions Made

- **combined mode passes fmriprep_root as hcp_root to create_combined_config():** The combined config factory still uses `hcp_root` internally (the combined DataSourceConfig field). The CLI surface rename is complete; the internal combined mode wiring is a separate concern.
- **create_hcp_config() preserved in imports:** `create_hcp_config` is still imported in `parcellate_hcp_subjects.py` (used by combined fallback path) — not removed to avoid breaking existing usage.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- `scripts/parcellate_missing_hcp_subjects.py --help` cannot be verified via `conda run` because `nilearn` is not installed in the `masters_thesis` environment at this time. The argparse configuration was verified by text search confirming `--fmriprep-root` is present and `--hcp-root` is absent. The import check for `HCPParcellationRunner` from `parcellate_hcp_subjects.py` passed (which does not import nilearn at module level).
- Plan 03 (`hcp_parcellation.py` update) was already executed but not committed/summarized — discovered during this plan's execution. The uncommitted Plan 03 changes were NOT included in this plan's commits.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 05 (verify_paths.py cleanup) can now assume the fmriprep CLI surface is complete across all entry points
- The compatibility shim (`hcp_root=fmriprep_root` in `create_fmriprep_config()`) can be removed once Plan 03 (HCPParcellator) is committed and verified
- All four runner scripts are aligned: CLI accepts `--fmriprep-root`, config uses `fmriprep_root`, subprocess forwarding passes `--fmriprep-root`

---
*Phase: 01-functional-path-migration*
*Completed: 2026-03-03*

## Self-Check: PASSED

- FOUND: `tcp/preprocessing/parcellate_hcp_subjects.py`
- FOUND: `tcp/preprocessing/integrate_cross_analysis.py`
- FOUND: `tcp/preprocessing/run_pipeline.py`
- FOUND: `scripts/parcellate_missing_hcp_subjects.py`
- FOUND: `.planning/phases/01-functional-path-migration/01-04-SUMMARY.md`
- FOUND commit: `d746522` (feat(01-04): update runner scripts CLI to fmriprep arguments)
- FOUND commit: `34c7ba9` (feat(01-04): update integrate_cross_analysis and run_pipeline to fmriprep arguments)
