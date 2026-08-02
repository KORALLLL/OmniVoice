# Implementation Notes

## 2026-08-02 - Balalaika memorization and hard-number validation

- Decision: Execute the approved checkpoint-isolated design through the 12-task subagent-driven implementation plan.
- Assumption: Balalaika text is authoritative only from `rover_punctuated_accented`, joined by `source_relative_path`; TAR-side text and quality metadata do not influence selection.
- Tradeoff: Prefer 3–12 second references but permit clips over 12 seconds only as a deterministic fallback, with no hard upper limit.
- Constraint: Initial real GPU work is one path with no automatic retry and one shared one-hour monotonic deadline.
- Validation: Pre-implementation baseline passed 59 tests with six existing warnings in the worktree virtual environment; the system Python is not a valid test environment because it lacks the required Transformers API.
- Decision: Keep GigaAM ONNX GPU and W&B dependencies in the `validation` optional extra so default OmniVoice installs remain unchanged.
- Validation: Task 1 dependency test passed; `uv.lock` resolves the exact validation packages, and independent review approved the diff with no findings.
- Decision: Stream the 4.075M-row sidecar through bounded deterministic priority pools, with pool sizes 512, 2,048, and 8,192 for CPU-only candidate recovery.
- Tradeoff: Content-addressed converted WAV names preserve an existing published manifest if a later preparation attempt fails before atomic replacement.
- Validation: Task 2 passed five focused tests using real miniature TAR/MP3 fixtures, FFmpeg mono 24 kHz PCM16 conversion, malformed audio, metadata invariance, and >12-second fallback; independent review found no issues. Full proprietary-sidecar execution remains for Task 12.
