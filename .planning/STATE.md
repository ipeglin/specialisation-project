# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-02)

**Core value:** Parcellation pipeline correctly locates and loads fmriprep BOLD NIfTI for each subject
**Current focus:** Phase 01 - Functional Path Migration

## Current Position

Phase: 1 of 2 (Functional Path Migration)
Plan: 4 of 5 in current phase (01-04 is next)
Status: In progress
Last activity: 2026-03-03 - Completed 01-03: HCPParcellator fmriprep path migration

Progress: [#########.....................] 43% (3 of 7 total plans)

## Performance Metrics

**Velocity:**
- Total plans completed: 3
- Average duration: ~3 min
- Total execution time: ~9 min

**By Phase:**

| Phase | Plans | Completed | Avg/Plan |
|-------|-------|-----------|----------|
| 01 - Functional Path Migration | 5 | 3 | ~3 min |
| 02 - Verification and Hardening | 2 | 0 | - |

**Recent Trend:**
- Last 3 plans: ~4 min, ~2 min, ~3 min
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

### Pending Todos

None yet.

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-03-03 16:01
Stopped at: Completed 01-03-PLAN.md
Resume file: None
