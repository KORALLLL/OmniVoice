# Implementation Notes

## 2026-08-02 - Balalaika memorization and hard-number validation

- Decision: Execute the approved checkpoint-isolated design through the 12-task subagent-driven implementation plan.
- Assumption: Balalaika text is authoritative only from `rover_punctuated_accented`, joined by `source_relative_path`; TAR-side text and quality metadata do not influence selection.
- Tradeoff: Prefer 3–12 second references but permit clips over 12 seconds only as a deterministic fallback, with no hard upper limit.
- Constraint: Initial real GPU work is one path with no automatic retry and one shared one-hour monotonic deadline.
