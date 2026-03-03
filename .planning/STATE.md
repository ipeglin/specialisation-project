# Project State

**Project:** TCP fMRI Pipeline — fmriprep Migration
**Last session:** 2026-03-03
**Stopped at:** Completed 01-01-PLAN.md

---

## Current Position

**Phase:** 01-functional-path-migration
**Current Plan:** 02
**Progress:** [##--------] 1/5 plans complete (est.)

---

## Decisions

- **[01-01]** fmriprep_output added as top-level platform key in default_config.json, not derived from data_base; accessor uses _get_base_path() directly
- **[01-01]** macOS/Windows/Linux get local placeholder paths; idun gets real cluster path /cluster/projects/itea_lille-ie/Transdiagnostic/output/fmriprep-25.1.4
- **[01-01]** Audit found map_subject_files.py and filter_subjects.py also have HCP references — must be addressed in later plan

---

## Blockers

None

---

## Performance Metrics

| Phase | Plan | Duration | Tasks | Files |
|-------|------|----------|-------|-------|
| 01-functional-path-migration | 01 | 2min | 2 | 2 |

---

## Session Info

**Last session:** 2026-03-03T14:55:25Z
**Stopped at:** Completed 01-01-PLAN.md
**Resume file:** None
