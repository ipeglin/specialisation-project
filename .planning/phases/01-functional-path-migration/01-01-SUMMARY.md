---
phase: 01-functional-path-migration
plan: "01"
subsystem: config
tags: [pathlib, json, fmriprep, platform-config]

# Dependency graph
requires: []
provides:
  - "config/default_config.json with fmriprep_output and fmriprep_parcellated_output keys per platform"
  - "get_fmriprep_output_path() and get_fmriprep_parcellated_output_path() in config/paths.py"
  - "Pre-migration audit documenting all HCP reference locations across the codebase"
affects:
  - 01-02
  - 01-03
  - 01-04
  - 01-05

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Config registry pattern: platform-scoped base paths in default_config.json resolved via _get_base_path(key)"
    - "Accessor function pattern: module-level convenience functions delegating to global _path_config singleton"

key-files:
  created: []
  modified:
    - config/default_config.json
    - config/paths.py

key-decisions:
  - "fmriprep_output is added as a top-level platform key (not under data_base) to match the existing hcp_output pattern; accessor uses _get_base_path() directly"
  - "macos/windows/linux get platform-specific placeholder paths; idun gets the real cluster path /cluster/projects/itea_lille-ie/Transdiagnostic/output/fmriprep-25.1.4"
  - "Audit revealed two additional files with HCP references beyond the known six: tcp/preprocessing/map_subject_files.py and tcp/preprocessing/filter_subjects.py — these must be addressed in Plan 04"

patterns-established:
  - "New data source paths are added as top-level platform keys in default_config.json (not derived from data_base)"
  - "Corresponding accessor functions in config/paths.py call _path_config._get_base_path(key) directly"

# Metrics
duration: 2min
completed: 2026-03-03
---

# Phase 01 Plan 01: HCP Audit and fmriprep Config Registry Summary

**fmriprep_output and fmriprep_parcellated_output keys added to all four platform sections in default_config.json, with matching get_fmriprep_output_path() and get_fmriprep_parcellated_output_path() accessor functions in config/paths.py**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-03T14:53:54Z
- **Completed:** 2026-03-03T14:55:25Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Full pre-migration grep audit completed — all HCP path reference locations documented before any migration code is written
- config/default_config.json updated with fmriprep_output and fmriprep_parcellated_output for macos, windows, idun, and linux platforms
- get_fmriprep_output_path() and get_fmriprep_parcellated_output_path() added to config/paths.py and verified importable on macOS

## Audit Findings (Task 1)

Files with HCP path references (grep: `hcp.root|hcp_root|hcp-root|MNINonLinear|hcp_output|hcp_parcellated|create_hcp_config|discover_hcp_subjects|get_hcp_bold_path|validate_hcp_structure`):

**Known files (from PROJECT.md):**
- `tcp/preprocessing/config/data_source_config.py` — hcp_root field, get_hcp_bold_path(), discover_hcp_subjects(), validate_hcp_structure(), create_hcp_config()
- `tcp/preprocessing/hcp_parcellation.py` — HCPParcellator.__init__(), discover_bold_files(), MNINonLinear path construction
- `tcp/preprocessing/parcellate_hcp_subjects.py` — CLI args, create_hcp_config() call
- `scripts/parcellate_missing_hcp_subjects.py` — CLI args, create_hcp_config() call
- `config/default_config.json` — (no hcp_output key existed; fmriprep_output added in Task 2)
- `scripts/verify_paths.py` — get_data_path('hcp_output') references

**Additional files found (not in original PROJECT.md list — require migration in Plan 04):**
- `tcp/preprocessing/integrate_cross_analysis.py` — --hcp-root CLI arg, create_hcp_config() call (expected per research SUMMARY.md)
- `tcp/preprocessing/run_pipeline.py` — --hcp-root CLI arg, hcp_root/hcp_parcellated_output kwargs
- `tcp/preprocessing/map_subject_files.py` — hcp_root field access, MNINonLinear path construction, --hcp-root CLI arg
- `tcp/preprocessing/filter_subjects.py` — discover_hcp_subjects() call, --hcp-root CLI arg

## Task Commits

Each task was committed atomically:

1. **Task 1: Pre-migration audit** — audit-only, no files modified (no commit)
2. **Task 2: Add fmriprep_output keys and accessor functions** — `480a423` (feat)

## Files Created/Modified

- `config/default_config.json` — Added fmriprep_output and fmriprep_parcellated_output to all four platform sections (macos, windows, idun, linux); no existing keys modified
- `config/paths.py` — Added get_fmriprep_output_path() and get_fmriprep_parcellated_output_path() after get_mdd_analysis_path()

## Decisions Made

- fmriprep_output is a top-level platform key (not a subdirectory under data_base). Using `get_data_path('fmriprep_output')` would silently append 'fmriprep_output' as a filesystem path under data_base — incorrect. The accessor calls `_get_base_path('fmriprep_output')` directly, which looks up the platform-scoped value from the JSON registry.
- macOS/Windows/Linux platforms use local placeholder paths since fmriprep data is only available on the IDUN cluster. idun gets the real cluster path `/cluster/projects/itea_lille-ie/Transdiagnostic/output/fmriprep-25.1.4`.
- Audit revealed map_subject_files.py and filter_subjects.py also reference HCP paths. These are outside the current parcellation-scope migration plan but must be tracked for Plan 04.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- config/default_config.json and config/paths.py are complete prerequisites for Plan 02 (DataSourceConfig migration)
- Plan 02 can now call get_fmriprep_output_path() or _get_base_path('fmriprep_output') to resolve the cluster path without hardcoding
- Additional HCP references in map_subject_files.py and filter_subjects.py noted for Plan 04

---
*Phase: 01-functional-path-migration*
*Completed: 2026-03-03*
