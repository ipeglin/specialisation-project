# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-02)

**Core value:** Parcellation pipeline correctly locates and loads fmriprep BOLD NIfTI for each subject
**Current focus:** Phase 01 - Functional Path Migration

## Current Position

Phase: 1 of 2 (Functional Path Migration)
Plan: 5 of 5 in current phase (phase complete)
Status: Complete
Last activity: 2026-03-03 - Completed 01-05: verify_paths.py update and final integration check

Progress: [################..............] 71% (5 of 7 total plans)

## Performance Metrics

**Velocity:**
- Total plans completed: 4
- Average duration: ~5 min
- Total execution time: ~18 min

**By Phase:**

| Phase | Plans | Completed | Avg/Plan |
|-------|-------|-----------|----------|
| 01 - Functional Path Migration | 5 | 5 | ~4 min |
| 02 - Verification and Hardening | 2 | 0 | - |

**Recent Trend:**
- Last 5 plans: ~4 min, ~2 min, ~3 min, ~9 min, ~2 min
- Trend: Stable

*Updated after each plan completion*

## Accumulated Context

### Decisions

- [01-02]: DataSourceType.HCP used in create_fmriprep_config() for runner compatibility - Plan 04 removes shim
- [01-02]: fmriprep_root not validated in __post_init__ to avoid breaking HCP source_type validation; validated in methods
- [01-02]: discover_fmriprep_subjects() verifies full BIDS filename not just func/ dir to exclude partial outputs
- [01-03]: import re added at top level (stdlib) to support normalised key extraction in parcellate_subject()
- [01-03]: self.hcp_root alias maintained as backward-compatible reference to fmriprep_root
- [01-03]: CLI --hcp-root renamed to --fmriprep-root; both main() instantiation calls updated
- [01-04]: combined mode passes fmriprep_root as hcp_root to create_combined_config() — internal field name unchanged
- [01-04]: create_hcp_config() call replaced with create_fmriprep_config() for the 'hcp' data-source-type branch
- [01-05]: MNINonLinear refs in data_source_config.py HCP methods (discover_hcp_subjects, validate_hcp_structure, get_hcp_bold_path) intentionally preserved for combined mode — fmriprep-specific methods are clean
- [01-05]: verify_paths.py uses get_fmriprep_output_path() (not get_data_path('fmriprep_output')) to look up the top-level platform key

### Pending Todos

None yet.

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-03-03 16:12
Stopped at: Completed 01-05-PLAN.md (Phase 01 complete)
Resume file: None
