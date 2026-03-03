# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-02)

**Core value:** Parcellation pipeline correctly locates and loads fmriprep BOLD NIfTI for each subject
**Current focus:** Phase 01 - Functional Path Migration

## Current Position

Phase: 1 of 2 (Functional Path Migration)
Plan: 5 of 5 in current phase (01-05 is next)
Status: In progress
Last activity: 2026-03-03 - Completed 01-04: Runner scripts CLI migration to fmriprep arguments

Progress: [###########...................] 57% (4 of 7 total plans)

## Performance Metrics

**Velocity:**
- Total plans completed: 4
- Average duration: ~5 min
- Total execution time: ~18 min

**By Phase:**

| Phase | Plans | Completed | Avg/Plan |
|-------|-------|-----------|----------|
| 01 - Functional Path Migration | 5 | 4 | ~5 min |
| 02 - Verification and Hardening | 2 | 0 | - |

**Recent Trend:**
- Last 4 plans: ~4 min, ~2 min, ~3 min, ~9 min
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

### Pending Todos

None yet.

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-03-03 16:10
Stopped at: Completed 01-04-PLAN.md
Resume file: None
