# Balalaika Memorization and Hard-Number Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Balalaika-v2 four-sample LoRA memorization run and an eight-GPU, checkpoint-isolated 2,000-prompt hard-number validation workflow with local audio artifacts and online W&B reporting.

**Architecture:** A deterministic Balalaika extractor joins the punctuation/stress sidecar to TAR-resident MP3 audio and publishes four memorization plus twenty validation references. Training gains generic bounded-stop controls while retaining the original total-step scheduler; a controller alternates segmented training checkpoints with separate distributed TTS and GigaAM ASR processes. Pure scoring and artifact modules enforce 2,000/2,000 coverage before true corpus micro-averaged CER/WER can be logged.

**Tech Stack:** Python 3.12, PyTorch 2.8, Transformers, PEFT, Accelerate, WebDataset, Hugging Face Hub, FFmpeg, SoundFile, ONNX ASR/GigaAM-v3 RNN-T, ONNX Runtime GPU, SoXR, W&B, pytest, Ruff, Bash.

## Global Constraints

- Use `/workspace/balalaika_proprietary_v2` as the Balalaika source root.
- Use `/workspace/balalaika_proprietary_v2/combined_sidecars/rover-punctuation-stress-v1/rover-punctuation-stress.jsonl` as the authoritative Balalaika text source.
- Join Balalaika text and audio by `source_relative_path`; model text is `rover_punctuated_accented`.
- Selection seed is exactly `42`; four memorization and twenty validation references must be disjoint.
- Selection must not inspect or filter on `DistillMOS`, `music_prob`, TAR-member text, or other quality metadata.
- Prefer decodable, non-empty audio lasting 3–12 seconds; if fewer than 24 are available, admit clips longer than 12 seconds with no hard maximum.
- Convert every selected clip to mono 24 kHz PCM16 WAV and record source/output SHA-256 hashes.
- The hard-number dataset is `bitmanagerai/hard_number_eval_for_tts` at revision `57b964492ccfcedd6a24d0225ef4b7d3697ffdca` and must contain exactly 2,000 unique rows.
- TTS input is `stressed`, scoring reference is `normalized_gold`, and digit-bearing `text` is used only for number-span discovery.
- Twenty validation references are assigned exactly 100 prompts each; all assignments are immutable across checkpoints.
- Generate with Russian, `num_step=32`, `guidance_scale=2.0`, `t_shift=0.1`, `layer_penalty_factor=5.0`, `position_temperature=0.0`, and `class_temperature=0.0`.
- Full validation uses eight ranks with 250 prompt assignments per rank and runs TTS and ASR as separate processes so both models never occupy GPU memory together.
- Report metrics only after exact 2,000/2,000 unique synthesis and ASR coverage.
- Use true reference-character weighting for corpus CER, correcting the benchmark script's word-weighted CER bug.
- Log online to W&B project `omnivoice-lora-validation`; fail before GPU work when online authentication is unavailable.
- Upload exactly four fixed audio examples per validation and keep every generated WAV locally.
- Validate every `ceil(steps_per_epoch / 8)` optimizer steps using checkpoint isolation.
- Memorization succeeds only after two consecutive dev losses `<= 1e-4`; report `<= 1e-5` as a stretch result.
- Initial GPU work has one solution path, no automatic retry, and a hard one-hour wall-clock cap: at most 20 minutes for memorization and the remaining time for base-model validation.
- The initial 2,000-prompt validation target is `k2-fsa/OmniVoice` at logical step 0.
- Existing non-LoRA training, TensorBoard logging, checkpoint rotation, and accepted CUDA embedding reconstruction behavior must remain unchanged.
- Do not push a model, adapter, dataset, branch, or other artifact to an external repository; W&B metric/audio logging is the only authorized external write.
- Use `rtk` for every shell command and `apply_patch` for every file edit.

---

### Task 5: Pinned Hard-Number Dataset and Balanced Voice Assignments

**Files:**
- Create: `omnivoice/validation/hard_numbers.py`
- Create: `tests/validation/test_hard_numbers.py`

**Interfaces:**
- Consumes: pinned `hard_number_eval.jsonl` and twenty Task 2 validation voice rows.
- Produces: `HardNumberRow`, `ValidationAssignment`, `load_hard_number_rows(...)`, `assign_voices(...)`, `partition_assignments(...)`, and `write_assignment_manifest(...)`.

- [ ] **Step 1: Write dataset and assignment tests**

```python
rows = load_hard_number_rows(local_jsonl, expected_count=2000)
assignments = assign_voices(rows, voices, seed=42)
assert len(assignments) == 2000
assert Counter(item.voice_id for item in assignments) == {
    voice.id: 100 for voice in voices
}
assert [len(partition_assignments(assignments, rank, 8)) for rank in range(8)] == [250] * 8
```

Also assert duplicate IDs, missing required fields, an unpinned revision, and any count other than 2,000 raise descriptive errors. Reorder input JSONL and voices and assert the persisted assignment manifest is byte-identical.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
rtk pytest -q tests/validation/test_hard_numbers.py
```

Expected: FAIL because the hard-number module is absent.

- [ ] **Step 3: Implement pinned Hub download and strict row validation**

```python
HARD_NUMBER_REPO = "bitmanagerai/hard_number_eval_for_tts"
HARD_NUMBER_REVISION = "57b964492ccfcedd6a24d0225ef4b7d3697ffdca"
HARD_NUMBER_FILENAME = "hard_number_eval.jsonl"
REQUIRED_FIELDS = (
    "id", "category", "hard_number", "text", "normalized_gold", "stressed"
)


def download_hard_number_jsonl() -> Path:
    return Path(hf_hub_download(
        repo_id=HARD_NUMBER_REPO,
        repo_type="dataset",
        filename=HARD_NUMBER_FILENAME,
        revision=HARD_NUMBER_REVISION,
    ))
```

Load all rows, reject blank required strings, reject duplicate IDs, require exactly 2,000, and sort by a normalized string ID before assignment.

- [ ] **Step 4: Implement immutable 100-per-voice assignment and eight-way partitioning**

Sort voices by ID, shuffle with `random.Random(42)`, then assign `voice = shuffled[index % 20]`. Define partitioning as `assignments[rank::world_size]`; validate `0 <= rank < world_size`. Serialize all dataset fields plus reference WAV/text/hash and generation configuration so each checkpoint uses the same byte-identical manifest.

- [ ] **Step 5: Run focused tests and commit**

Run:

```bash
rtk pytest -q tests/validation/test_hard_numbers.py
rtk ruff check omnivoice/validation/hard_numbers.py tests/validation/test_hard_numbers.py
```

Expected: PASS.

Commit:

```bash
rtk git add omnivoice/validation/hard_numbers.py tests/validation/test_hard_numbers.py
rtk git commit -m "feat: prepare hard-number validation assignments"
```

---

### Task 6: Correct Utterance and Number-Span Metrics

**Files:**
- Create: `omnivoice/validation/metrics.py`
- Create: `tests/validation/test_metrics.py`

**Interfaces:**
- Consumes: raw digit-bearing text, normalized gold, ASR hypothesis, category.
- Produces: `normalize_ru(...)`, `edit_counts(...)`, `number_span(...)`, `hypothesis_span(...)`, `score_utterance(...)`, and `aggregate_scores(...)` with overall and per-category `utt_cer`, `utt_wer`, `num_cer`, and `num_wer`.

- [ ] **Step 1: Write golden alignment and micro-average tests**

```python
def test_normalize_ru_matches_benchmark_rules():
    assert normalize_ru("Ёж +ёл: 12!") == "еж ел 12"


def test_number_span_maps_gold_into_hypothesis():
    score = score_utterance(
        raw_text="У меня 21 книга",
        gold="у меня двадцать одна книга",
        hypothesis="у меня двадцать две книги",
    )
    assert score.ref_number == "двадцать одна"
    assert score.hyp_number == "двадцать две"
```

Add a CER regression where one row has one reference word containing ten characters and a second row has ten one-character words. Assert corpus CER is `(char errors)/(reference characters)`, not the mean weighted by word counts. Cover substitutions, deletions, insertions, empty hypothesis, missing number span, and category aggregation.

- [ ] **Step 2: Run metric tests and verify RED**

Run:

```bash
rtk pytest -q tests/validation/test_metrics.py
```

Expected: FAIL because the metrics module is absent.

- [ ] **Step 3: Implement deterministic Levenshtein operations and normalization**

```python
def normalize_ru(text: str) -> str:
    value = text.lower().replace("ё", "е").replace("+", "")
    value = re.sub(r"[^а-я0-9 ]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


@dataclass(frozen=True)
class EditCounts:
    substitutions: int
    deletions: int
    insertions: int
    correct: int
    reference_units: int

    @property
    def rate(self) -> float:
        errors = self.substitutions + self.deletions + self.insertions
        return errors / self.reference_units if self.reference_units else float(errors > 0)
```

Port the benchmark's tie-breaking order exactly: diagonal, deletion, insertion. Keep word and character counts separately.

- [ ] **Step 4: Implement number-span mapping and true micro aggregation**

Use raw-to-gold alignment to locate all substituted/inserted gold word indices, then gold-to-hypothesis alignment to select the corresponding hypothesis range. Aggregate each metric as:

```python
rate = sum(item.errors for item in scores) / sum(item.reference_units for item in scores)
```

Never reuse word denominators for character metrics. Return raw S/D/I/C/N counts for utterance words/chars and number words/chars.

- [ ] **Step 5: Run focused tests and commit**

Run:

```bash
rtk pytest -q tests/validation/test_metrics.py
rtk ruff check omnivoice/validation/metrics.py tests/validation/test_metrics.py
```

Expected: PASS.

Commit:

```bash
rtk git add omnivoice/validation/metrics.py tests/validation/test_metrics.py
rtk git commit -m "feat: score hard-number speech recognition"
```

---

### Task 7: Atomic, Resumable Validation Artifacts and Coverage Gates

**Files:**
- Create: `omnivoice/validation/artifacts.py`
- Create: `tests/validation/test_artifacts.py`

**Interfaces:**
- Consumes: rank-local synthesis or ASR records keyed by benchmark ID.
- Produces: `AtomicJsonlLedger`, `valid_completed_ids(...)`, `merge_rank_ledgers(...)`, `require_exact_coverage(...)`, and `ValidationPaths`.

- [ ] **Step 1: Write interruption, duplicate, and incomplete-coverage tests**

```python
ledger = AtomicJsonlLedger(tmp_path / "rank-0.jsonl", key="id")
ledger.upsert({"id": "1", "wav": str(valid_wav), "sha256": wav_hash})
ledger.upsert({"id": "2", "error": "decode failed"})
reloaded = AtomicJsonlLedger(tmp_path / "rank-0.jsonl", key="id")
assert reloaded.successful_ids(required_files=("wav",)) == {"1"}
```

Assert a truncated final line is rejected or recovered from the last valid atomic file, duplicate IDs across ranks raise, missing IDs report a deterministic sorted list, extra IDs raise, and metrics cannot be published at 1,999/2,000.

- [ ] **Step 2: Run artifact tests and verify RED**

Run:

```bash
rtk pytest -q tests/validation/test_artifacts.py
```

Expected: FAIL because the artifact APIs are absent.

- [ ] **Step 3: Implement atomic ledgers and content validation**

Each `upsert` rewrites a rank-local temporary file, flushes, calls `os.fsync`, then uses `os.replace`. Preserve one record per key. A synthesis record is complete only when its WAV exists and its SHA-256 matches; an ASR record is complete only when `hypothesis` is a string and `error` is absent.

```python
def require_exact_coverage(expected_ids: Collection[str], actual_ids: Collection[str]) -> None:
    expected, actual = set(expected_ids), set(actual_ids)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra or len(actual_ids) != len(actual):
        raise CoverageError(missing=missing, extra=extra, duplicate_count=len(actual_ids) - len(actual))
```

- [ ] **Step 4: Encode the artifact layout**

`ValidationPaths(root, run_id, step)` must resolve `selection/voices.jsonl`, `step-N/wavs`, `rank-manifests`, `rank-hypotheses`, merged files, `per_utt.tsv`, `metrics.json`, `report.md`, and `run_metadata.json`. It may create directories but must never delete an existing valid artifact.

```python
@dataclass(frozen=True)
class ValidationPaths:
    root: Path
    run_id: str
    step: int

    @property
    def step_dir(self) -> Path:
        return self.root / self.run_id / f"step-{self.step}"

    def rank_manifest(self, rank: int) -> Path:
        return self.step_dir / "rank-manifests" / f"rank-{rank}.jsonl"

    def rank_hypotheses(self, rank: int) -> Path:
        return self.step_dir / "rank-hypotheses" / f"rank-{rank}.jsonl"
```

- [ ] **Step 5: Run focused tests and commit**

Run:

```bash
rtk pytest -q tests/validation/test_artifacts.py
rtk ruff check omnivoice/validation/artifacts.py tests/validation/test_artifacts.py
```

Expected: PASS.

Commit:

```bash
rtk git add omnivoice/validation/artifacts.py tests/validation/test_artifacts.py
rtk git commit -m "feat: add resumable validation artifacts"
```

---

### Task 8: Eight-Rank OmniVoice Synthesis Stage

**Files:**
- Create: `omnivoice/validation/synthesis.py`
- Create: `omnivoice/cli/validate_hard_numbers.py`
- Create: `tests/validation/test_synthesis.py`
- Create: `tests/cli/test_validate_hard_numbers.py`

**Interfaces:**
- Consumes: immutable assignment manifest, base model or LoRA checkpoint, rank/world size, `ValidationPaths`.
- Produces: `load_validation_tts(...)`, `synthesize_rank(...)`, rank manifest records, rank error records, and `validate_hard_numbers synth` CLI.

- [ ] **Step 1: Write fake-model synthesis and resume tests**

```python
summary = synthesize_rank(
    assignments=assignments,
    model=fake_model,
    output_dir=tmp_path,
    rank=1,
    world_size=8,
)
assert summary.expected == 250
assert summary.completed == 250
assert fake_model.generation_configs == [EXPECTED_GENERATION_CONFIG] * 250
```

Rerun and assert the fake model receives zero calls because hashes and rank records are valid. Corrupt one WAV and assert exactly one regeneration. Assert `--model k2-fsa/OmniVoice` calls `OmniVoice.from_pretrained`; `--adapter-checkpoint checkpoint-625` calls `OmniVoice.from_lora_pretrained`.

- [ ] **Step 2: Run synthesis tests and verify RED**

Run:

```bash
rtk pytest -q tests/validation/test_synthesis.py tests/cli/test_validate_hard_numbers.py
```

Expected: FAIL because the synthesis module and CLI are absent.

- [ ] **Step 3: Implement one-model-per-rank loading and exact generation**

Resolve rank from `RANK`, `LOCAL_RANK`, and `WORLD_SIZE`, require world size 8 for full validation, bind CUDA to the local rank, then load exactly one inference model in FP16. Use:

```python
GENERATION_CONFIG = OmniVoiceGenerationConfig(
    num_step=32,
    guidance_scale=2.0,
    t_shift=0.1,
    layer_penalty_factor=5.0,
    position_temperature=0.0,
    class_temperature=0.0,
)
```

Create and cache one `VoiceClonePrompt` per rank-local voice, then call `model.generate(text=row.stressed, language="Russian", voice_clone_prompt=prompt, generation_config=GENERATION_CONFIG)`. Save mono 24 kHz PCM16 WAV, hash it, and atomically upsert rank progress after every utterance.

- [ ] **Step 4: Add explicit synth CLI arguments and clean interruption handling**

```python
synth.add_argument("--assignments", type=Path, required=True)
synth.add_argument("--output-root", type=Path, required=True)
synth.add_argument("--run-id", required=True)
synth.add_argument("--step", type=int, required=True)
source = synth.add_mutually_exclusive_group(required=True)
source.add_argument("--model")
source.add_argument("--adapter-checkpoint", type=Path)
synth.add_argument("--deadline-monotonic", type=float)
```

Check the deadline between utterances. On deadline or SIGTERM, finish the current atomic update, write a partial summary, synchronize ranks, unload the model, and exit with a distinct incomplete status code; never claim coverage.

- [ ] **Step 5: Run focused tests and commit**

Run:

```bash
rtk pytest -q tests/validation/test_synthesis.py tests/cli/test_validate_hard_numbers.py
rtk ruff check omnivoice/validation/synthesis.py omnivoice/cli/validate_hard_numbers.py tests/validation/test_synthesis.py tests/cli/test_validate_hard_numbers.py
```

Expected: PASS.

Commit:

```bash
rtk git add omnivoice/validation/synthesis.py omnivoice/cli/validate_hard_numbers.py tests/validation/test_synthesis.py tests/cli/test_validate_hard_numbers.py
rtk git commit -m "feat: synthesize hard-number validation audio"
```

---

### Task 1: Reproducibility Ledger and Validation Dependencies

**Files:**
- Create: `/root/ml-intern-runs/omnivoice-bal-hard-number/TASK.md`
- Create: `/root/ml-intern-runs/omnivoice-bal-hard-number/BUDGET.md`
- Create: `/root/ml-intern-runs/omnivoice-bal-hard-number/RESEARCH.md`
- Create: `/root/ml-intern-runs/omnivoice-bal-hard-number/PLAN.md`
- Create: `/root/ml-intern-runs/omnivoice-bal-hard-number/EXPERIMENTS.md`
- Modify: `pyproject.toml:47-66`
- Modify: `uv.lock`
- Test: `tests/validation/test_dependencies.py`

**Interfaces:**
- Consumes: approved design and this implementation plan.
- Produces: a bounded experiment ledger and installable `validation` extra containing `wandb`, `onnx-asr`, and GPU ONNX Runtime.

- [ ] **Step 1: Write the run ledger before implementation**

Create the five run files with these exact budgets and path states:

```markdown
# BUDGET
max_paths            = 1
max_retries_per_path = 0
compute_cap          = 1 wall-clock hour of GPU workloads
memorization_cap     = 20 wall-clock minutes
baseline_cap         = remaining wall-clock time, at most 40 minutes
scale_ceiling        = k2-fsa/OmniVoice plus one broad rank-64 LoRA adapter
token_budget         = 10000 optimizer steps over four tokenized samples
--- spent ---
paths_launched = 0
gpu_min_used   = 0
retries_used   = 0
```

`TASK.md` must restate the two requested real runs, `RESEARCH.md` must link the OmniVoice repository, pinned dataset revision, GigaAM paper/model, ONNX ASR package, and W&B documentation, `PLAN.md` must link this repository plan and record “one checkpoint-isolated solution path,” and `EXPERIMENTS.md` must start with:

```markdown
| path_id | approach (one line) | status | final_loss | verify | failure_cause | retry_of | gpu_min |
|---------|---------------------|--------|------------|--------|---------------|----------|---------|
| isolated-controller | segmented train, distributed TTS then distributed ASR | queued | | | | | 0 |
```

- [ ] **Step 2: Write a failing optional-dependency test**

```python
import tomllib
from pathlib import Path


def test_validation_extra_contains_runtime_dependencies():
    project = tomllib.loads(Path("pyproject.toml").read_text())
    extra = project["project"]["optional-dependencies"]["validation"]
    assert any(item.startswith("wandb") for item in extra)
    assert "onnx-asr==0.12.0" in extra
    assert any(item.startswith("onnxruntime-gpu") for item in extra)
```

- [ ] **Step 3: Run the dependency test and verify RED**

Run:

```bash
rtk pytest -q tests/validation/test_dependencies.py
```

Expected: FAIL because the `validation` extra is absent.

- [ ] **Step 4: Add the isolated validation extra and lock it**

Add:

```toml
validation = [
    "onnx-asr==0.12.0",
    "onnxruntime-gpu>=1.23,<2",
    "wandb>=0.22,<1",
]
```

SoXR is already installed transitively through Librosa. Keep GPU ONNX dependencies out of the default package to avoid changing ordinary training environments. Regenerate the lock:

```bash
rtk uv lock
```

- [ ] **Step 5: Run the dependency test and commit**

Run:

```bash
rtk pytest -q tests/validation/test_dependencies.py
rtk git diff --check
```

Expected: PASS and no whitespace errors.

Commit:

```bash
rtk git add pyproject.toml uv.lock tests/validation/test_dependencies.py
rtk git commit -m "build: add hard-number validation dependencies"
```

---

### Task 2: Deterministic Balalaika Sidecar-to-Audio Preparation

**Files:**
- Create: `omnivoice/validation/__init__.py`
- Create: `omnivoice/validation/balalaika.py`
- Create: `omnivoice/scripts/prepare_balalaika_samples.py`
- Create: `tests/validation/test_balalaika.py`

**Interfaces:**
- Consumes: sidecar JSONL rows `{schema_version, source_relative_path, rover_punctuated_accented}` and `train/shard_NNNNNN.tar` MP3 members.
- Produces: `BalalaikaCandidate`, `SelectedBalalaikaClip`, `rank_candidates(...)`, `select_and_convert(...)`, and a 24-row atomic JSONL manifest whose first four rows have role `memorization` and final twenty have role `validation_voice`.

- [ ] **Step 1: Write TAR-plus-sidecar fixture tests**

Generate short WAV payloads with SoundFile, encode them to MP3 with FFmpeg, place them in two miniature `shard_*.tar` files, and write sidecar rows. Assert:

```python
selected = select_and_convert(
    root=fixture.root,
    sidecar_path=fixture.sidecar,
    output_dir=tmp_path / "selected",
    memorization_count=4,
    validation_voice_count=20,
    seed=42,
)
assert [row.role for row in selected].count("memorization") == 4
assert [row.role for row in selected].count("validation_voice") == 20
assert len({row.source_relative_path for row in selected}) == 24
assert all(row.text.startswith("sidecar text") for row in selected)
assert all(row.sample_rate == 24000 and row.channels == 1 for row in selected)
```

Include rows whose TAR JSON contains contradictory `accent.txt`, extreme `DistillMOS`, and extreme `music_prob`; changing those values must not change selection. Include 23 valid 3–12 second clips and one 13-second clip and assert only the final fallback row has `duration_tier == "fallback_over_12s"`. Include missing, empty, and undecodable MP3 members and assert they are skipped.

- [ ] **Step 2: Run Balalaika tests and verify RED**

Run:

```bash
rtk pytest -q tests/validation/test_balalaika.py
```

Expected: FAIL because `omnivoice.validation.balalaika` is absent.

- [ ] **Step 3: Implement stable candidate ranking without loading the 1.3 GB sidecar into memory**

Define immutable records and stable priority:

```python
@dataclass(frozen=True)
class BalalaikaCandidate:
    source_relative_path: str
    text: str
    schema_version: int
    priority: bytes
    estimated_duration: float


def stable_priority(seed: int, source_relative_path: str) -> bytes:
    value = f"{seed}\0{source_relative_path}".encode("utf-8")
    return hashlib.sha256(value).digest()
```

Stream the sidecar once. Validate only the required strings, parse the shard directory and filename start/end fields for an estimated duration, and retain a bounded heap of the lowest 512 priorities in each duration tier. Never read `DistillMOS`, `music_prob`, or TAR JSON members. If fewer than 24 decoded candidates survive, repeat with pool sizes 2,048 and 8,192; these are deterministic CPU/data retries, not GPU experiment retries.

- [ ] **Step 4: Implement exact TAR extraction, decode validation, conversion, and atomic publication**

Map `000123/name.mp3` to `train/shard_000123.tar` member `name.mp3`. For each priority-sorted candidate, extract to a temporary directory, inspect/decode it, then run:

```python
command = [
    "ffmpeg", "-nostdin", "-v", "error", "-y",
    "-i", str(source_mp3),
    "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le",
    str(staged_wav),
]
subprocess.run(command, check=True)
```

Accept actual durations `>=3.0 and <=12.0` into the preferred tier and actual durations `>12.0` into fallback. Select all possible preferred candidates before fallback, assign disjoint roles, calculate both SHA-256 hashes, and atomically replace `<output-dir>/selected.jsonl` only after all 24 WAVs exist.

- [ ] **Step 5: Add the preparation CLI**

Expose these exact arguments and defaults:

```python
parser.add_argument("--root", type=Path, default=Path("/workspace/balalaika_proprietary_v2"))
parser.add_argument("--sidecar", type=Path, default=Path("/workspace/balalaika_proprietary_v2/combined_sidecars/rover-punctuation-stress-v1/rover-punctuation-stress.jsonl"))
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--memorization-count", type=int, default=4)
parser.add_argument("--validation-voice-count", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
```

The CLI prints one JSON summary containing output manifest, counts, and duration-tier counts.

- [ ] **Step 6: Run focused tests and commit**

Run:

```bash
rtk pytest -q tests/validation/test_balalaika.py
rtk python -m compileall -q omnivoice/validation/balalaika.py omnivoice/scripts/prepare_balalaika_samples.py
rtk ruff check omnivoice/validation/balalaika.py omnivoice/scripts/prepare_balalaika_samples.py tests/validation/test_balalaika.py
```

Expected: all commands pass.

Commit:

```bash
rtk git add omnivoice/validation omnivoice/scripts/prepare_balalaika_samples.py tests/validation/test_balalaika.py
rtk git commit -m "feat: prepare Balalaika validation references"
```

---

### Task 3: Bounded Training, Evaluation History, and Scheduler-Safe Segmentation

**Files:**
- Modify: `omnivoice/training/config.py:48-102`
- Modify: `omnivoice/training/trainer.py:76-365`
- Modify: `omnivoice/cli/train.py:27-67`
- Create: `omnivoice/training/control.py`
- Create: `tests/training/test_training_control.py`
- Modify: `tests/training/test_lora.py`

**Interfaces:**
- Consumes: original total `TrainingConfig.steps`, optional invocation bound `stop_after_step`, evaluation loss, and monotonic elapsed seconds.
- Produces: `EvaluationStopPolicy.observe(step, loss, elapsed_seconds) -> StopDecision`, `append_loss_history(...)`, CLI overrides `--stop-after-step` and `--resume-from-checkpoint`, and `OmniTrainer.train() -> TrainingOutcome`.

- [ ] **Step 1: Write policy and configuration tests**

```python
def test_two_consecutive_losses_are_required():
    policy = EvaluationStopPolicy(threshold=1e-4, patience=2, wall_limit_seconds=1200)
    assert not policy.observe(25, 9e-5, 10).stop
    assert not policy.observe(50, 2e-4, 20).stop
    assert not policy.observe(75, 8e-5, 30).stop
    decision = policy.observe(100, 7e-5, 40)
    assert decision.stop and decision.reason == "eval_loss_target"


def test_scheduler_uses_total_steps_not_invocation_bound(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        trainer_module,
        "get_cosine_schedule_with_warmup",
        lambda optimizer, num_warmup_steps, num_training_steps: (
            captured.update(total=num_training_steps) or FakeScheduler()
        ),
    )
    config = TrainingConfig(steps=5000, stop_after_step=625)
    trainer = object.__new__(OmniTrainer)
    trainer.model = TinyModel()
    trainer.config = config
    trainer.create_optimizer_and_scheduler()
    assert captured["total"] == 5000
```

Also cover positive `steps_per_epoch`, `stop_after_step <= steps`, wall-clock stop only at an evaluation boundary, atomic JSONL history entries `{step, loss, elapsed_seconds}`, and CLI override precedence without mutating the JSON file.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
rtk pytest -q tests/training/test_training_control.py tests/training/test_lora.py
```

Expected: FAIL because bounded-control fields and APIs do not exist.

- [ ] **Step 3: Add generic configuration and result types**

Add:

```python
steps_per_epoch: Optional[int] = None
stop_after_step: Optional[int] = None
max_wall_clock_seconds: Optional[float] = None
early_stop_eval_loss: Optional[float] = None
early_stop_patience: int = 1
eval_history_path: Optional[str] = None
```

Define:

```python
@dataclass(frozen=True)
class StopDecision:
    stop: bool
    reason: str | None
    consecutive_hits: int


@dataclass(frozen=True)
class TrainingOutcome:
    step: int
    stop_reason: str
    last_eval_loss: float | None
    target_reached: bool
```

Reject non-positive configured values and bounds beyond total `steps` before model construction.

- [ ] **Step 4: Integrate evaluation-boundary stopping without changing scheduler construction**

Set:

```python
invocation_stop = config.stop_after_step or config.steps
```

Keep `create_optimizer_and_scheduler()` parameterized by `config.steps`. After each scheduled `evaluate()`, append history on the main rank, broadcast the decision, and stop all ranks together. Stop for `stop_after_step` immediately after the corresponding optimizer step; if no evaluation occurred at that exact step, run one final evaluation before saving. Always save `checkpoint-{global_step}`, return `TrainingOutcome`, and preserve the existing `accelerator.end_training()` path.

- [ ] **Step 5: Add safe CLI overrides**

```python
parser.add_argument("--stop-after-step", type=int)
parser.add_argument("--resume-from-checkpoint")
```

Apply overrides after `TrainingConfig.from_json`, validate the finished config, and print the `TrainingOutcome` as one JSON object from the main process.

- [ ] **Step 6: Run training-control regression tests and commit**

Run:

```bash
rtk pytest -q tests/training/test_training_control.py tests/training/test_lora.py tests/training/test_lora_checkpoint.py
rtk ruff check omnivoice/training/control.py omnivoice/training/config.py omnivoice/training/trainer.py omnivoice/cli/train.py tests/training/test_training_control.py
```

Expected: all tests pass, including existing checkpoint resume tests.

Commit:

```bash
rtk git add omnivoice/training omnivoice/cli/train.py tests/training
rtk git commit -m "feat: add scheduler-safe bounded training"
```

---

### Task 4: Four-Utterance Memorization Orchestrator and Audio Artifacts

**Files:**
- Create: `omnivoice/validation/memorization.py`
- Create: `omnivoice/cli/run_memorization.py`
- Modify: `omnivoice/cli/eval_memorization.py:16-126`
- Modify: `examples/run_lora_memorization.sh`
- Modify: `examples/config/train_config_lora_memorization.json`
- Create: `tests/validation/test_memorization.py`
- Modify: `tests/cli/test_eval_memorization.py`
- Modify: `tests/scripts/test_select_memorization_samples.py`

**Interfaces:**
- Consumes: the four `role=memorization` rows from Task 2, existing audio token extractor, bounded trainer from Task 3, base model or adapter checkpoint.
- Produces: `MemorizationRunResult`, `choose_generation_checkpoint(...)`, saved `original/`, `generated/initial/`, `generated/final/`, `loss_history.jsonl`, and machine-readable `result.json`.

- [ ] **Step 1: Write orchestrator behavior tests with fake commands and model loaders**

```python
result = run_memorization(
    selected_manifest=selected_manifest,
    output_dir=tmp_path / "exp",
    train_config=train_config,
    data_config=data_config,
    max_wall_clock_seconds=1200,
    command_runner=fake_runner,
    model_loader=fake_model_loader,
)
assert result.required_target_reached is True
assert result.stretch_target_reached is False
assert result.qualifying_step == 100
assert len(list((tmp_path / "exp/original").glob("*.wav"))) == 4
assert len(list((tmp_path / "exp/generated/initial").glob("*.wav"))) == 4
assert len(list((tmp_path / "exp/generated/final").glob("*.wav"))) == 4
```

Feed losses `[2e-4, 9e-5, 8e-5]` and assert the selected checkpoint is step 100. Add a timeout case that selects the final saved checkpoint and records an explicit miss. Assert initial generation loads `k2-fsa/OmniVoice`, final generation uses `OmniVoice.from_lora_pretrained`, and all four generation calls use the same source transcript as both target and reference text.

- [ ] **Step 2: Run memorization tests and verify RED**

Run:

```bash
rtk pytest -q tests/validation/test_memorization.py tests/cli/test_eval_memorization.py tests/scripts/test_select_memorization_samples.py
```

Expected: FAIL because the orchestrator and patience result do not exist.

- [ ] **Step 3: Implement reusable audio generation and checkpoint selection**

```python
def generate_four(
    rows: Sequence[SelectedBalalaikaClip],
    output_dir: Path,
    model: OmniVoice,
) -> list[Path]:
    config = OmniVoiceGenerationConfig(
        num_step=32,
        guidance_scale=2.0,
        t_shift=0.1,
        layer_penalty_factor=5.0,
        position_temperature=0.0,
        class_temperature=0.0,
    )
    for row in rows:
        [audio] = model.generate(
            text=row.text,
            language="Russian",
            ref_text=row.text,
            ref_audio=str(row.wav_path),
            generation_config=config,
        )
        sf.write(output_dir / f"{row.id}.wav", audio, model.sampling_rate, subtype="PCM_16")
```

`choose_generation_checkpoint` returns the second consecutive qualifying checkpoint when present, otherwise the highest completed checkpoint.

- [ ] **Step 4: Implement the end-to-end command sequence**

The Python CLI must:

1. copy the four converted WAVs and write a raw four-row JSONL using sidecar text;
2. generate `initial/` before training;
3. invoke the existing token extractor once;
4. invoke single-GPU Accelerate training with `steps=10000`, `eval_steps=25`, `early_stop_eval_loss=1e-4`, `early_stop_patience=2`, and `max_wall_clock_seconds=1200`;
5. evaluate/reload the selected adapter, generate `final/`, and write `result.json` including minimum loss and `<=1e-5` status.

Use argument lists, `subprocess.run(check=True)`, and monotonic time; do not build commands with shell strings.

```python
started = time.monotonic()
deadline = started + args.max_wall_clock_seconds
experiment_deadline = started + args.experiment_wall_clock_seconds
copy_originals(rows, output_dir / "original")
generate_four(rows, output_dir / "generated/initial", load_base_model())
run_tokenizer(rows_jsonl, token_dir)
outcome = run_training(
    stop_after_step=10_000,
    deadline_monotonic=deadline,
    eval_history_path=output_dir / "loss_history.jsonl",
)
checkpoint = choose_generation_checkpoint(output_dir, read_loss_history(output_dir))
generate_four(rows, output_dir / "generated/final", load_adapter(checkpoint))
write_result(
    output_dir / "result.json",
    outcome,
    checkpoint,
    experiment_deadline_monotonic=experiment_deadline,
)
```

- [ ] **Step 5: Make the example launcher a thin, tested wrapper**

Replace duplicated orchestration with:

```bash
CUDA_VISIBLE_DEVICES=0 python -m omnivoice.cli.run_memorization \
    --selected-manifest "${SELECTED_MANIFEST}" \
    --output-dir "${OUTPUT_DIR}" \
    --train-config "${TRAIN_CONFIG}" \
    --data-config "${DATA_CONFIG}" \
    --max-wall-clock-seconds 1200 \
    --experiment-wall-clock-seconds 3600
```

Update the memorization config to 10,000 total steps, evaluation every 25 steps, threshold `1e-4`, patience 2, and rank-64 broad LoRA. Preserve source-path resolution tests.

- [ ] **Step 6: Run focused tests and commit**

Run:

```bash
rtk pytest -q tests/validation/test_memorization.py tests/cli/test_eval_memorization.py tests/scripts/test_select_memorization_samples.py
rtk bash -n examples/run_lora_memorization.sh
rtk ruff check omnivoice/validation/memorization.py omnivoice/cli/run_memorization.py omnivoice/cli/eval_memorization.py tests/validation/test_memorization.py
```

Expected: all commands pass.

Commit:

```bash
rtk git add omnivoice/validation/memorization.py omnivoice/cli/run_memorization.py omnivoice/cli/eval_memorization.py examples/run_lora_memorization.sh examples/config/train_config_lora_memorization.json tests
rtk git commit -m "feat: run bounded Balalaika memorization"
```

---

### Task 9: Separate Eight-Rank GigaAM ASR Stage

**Files:**
- Create: `omnivoice/validation/asr.py`
- Modify: `omnivoice/cli/validate_hard_numbers.py`
- Create: `tests/validation/test_asr.py`
- Modify: `tests/cli/test_validate_hard_numbers.py`

**Interfaces:**
- Consumes: exactly 2,000 synthesis WAV records, rank/world size, optional deadline.
- Produces: `load_gigaam(...)`, `transcribe_rank(...)`, rank hypothesis ledgers, error records, and `validate_hard_numbers asr` CLI.

- [ ] **Step 1: Write fake-ASR partition, resampling, and resume tests**

```python
summary = transcribe_rank(
    synthesis_records=records,
    recognizer=fake_recognizer,
    output_dir=tmp_path,
    rank=3,
    world_size=8,
)
assert summary.expected == 250
assert fake_recognizer.sample_rates == [16000] * 250
assert all(array.dtype == np.float32 for array in fake_recognizer.arrays)
```

Rerun and assert zero recognition calls. Remove one hypothesis and assert exactly one call. Assert stereo/24 kHz input becomes mono/16 kHz float32. Assert model-load or recognition exceptions are persisted with ID/rank/type/message and do not count as complete.

- [ ] **Step 2: Run ASR tests and verify RED**

Run:

```bash
rtk pytest -q tests/validation/test_asr.py tests/cli/test_validate_hard_numbers.py
```

Expected: FAIL because the ASR stage is absent.

- [ ] **Step 3: Implement rank-bound GPU GigaAM loading**

```python
def load_gigaam(local_rank: int):
    providers = [
        ("CUDAExecutionProvider", {"device_id": local_rank}),
        "CPUExecutionProvider",
    ]
    model = onnx_asr.load_model("gigaam-v3-rnnt", providers=providers)
    active = model.providers if hasattr(model, "providers") else []
    if active and "CUDAExecutionProvider" not in active:
        raise RuntimeError("GigaAM did not activate CUDAExecutionProvider")
    return model
```

Read WAV with SoundFile, average channels, resample through `soxr.resample(waveform, sample_rate, 16000)`, cast float32, call `recognize`, normalize a singleton-list result to string, and upsert after every utterance.

- [ ] **Step 4: Add ASR CLI with synthesis coverage precondition**

Before loading GigaAM, merge rank synthesis manifests and require exact 2,000/2,000 valid WAVs. Accept the same run/step/deadline arguments as synthesis. Require world size 8 for a full run, handle deadline/SIGTERM cleanly, and return incomplete without aggregate scoring when any hypothesis is absent or failed.

```python
asr.add_argument("--assignments", type=Path, required=True)
asr.add_argument("--output-root", type=Path, required=True)
asr.add_argument("--run-id", required=True)
asr.add_argument("--step", type=int, required=True)
asr.add_argument("--deadline-monotonic", type=float)
```

- [ ] **Step 5: Run focused tests and commit**

Run:

```bash
rtk pytest -q tests/validation/test_asr.py tests/cli/test_validate_hard_numbers.py
rtk ruff check omnivoice/validation/asr.py omnivoice/cli/validate_hard_numbers.py tests/validation/test_asr.py tests/cli/test_validate_hard_numbers.py
```

Expected: PASS.

Commit:

```bash
rtk git add omnivoice/validation/asr.py omnivoice/cli/validate_hard_numbers.py tests/validation/test_asr.py tests/cli/test_validate_hard_numbers.py
rtk git commit -m "feat: transcribe validation audio with GigaAM"
```

---

### Task 10: Scoring Reports and Stable Online W&B Logging

**Files:**
- Create: `omnivoice/validation/reporting.py`
- Create: `omnivoice/validation/wandb_logging.py`
- Modify: `omnivoice/cli/validate_hard_numbers.py`
- Create: `tests/validation/test_reporting.py`
- Create: `tests/validation/test_wandb_logging.py`
- Modify: `tests/cli/test_validate_hard_numbers.py`

**Interfaces:**
- Consumes: exact-coverage merged synthesis/hypothesis records, immutable assignments, logical step, `steps_per_epoch`, four fixed example IDs.
- Produces: `score_validation_run(...)`, `write_validation_report(...)`, `WandbRunStore`, `log_validation(...)`, and `validate_hard_numbers score` CLI.

- [ ] **Step 1: Write report and fake-W&B tests**

```python
result = score_validation_run(assignments, hypotheses)
assert result.coverage == 2000
assert set(result.overall) >= {"utt_cer", "utt_wer", "num_cer", "num_wer"}

log_validation(fake_wandb_run, result, fixed_audio_records, step=625, steps_per_epoch=5000)
assert fake_wandb_run.last_step == 625
assert fake_wandb_run.last_metrics["validation/fractional_epoch"] == pytest.approx(0.125)
assert len(fake_wandb_run.audio_objects) == 4
```

Assert the four audio IDs are persisted on first use and unchanged at later steps, category table rows are stable, all S/D/I and throughput fields are present, and no W&B call occurs when coverage is incomplete.

- [ ] **Step 2: Run reporting tests and verify RED**

Run:

```bash
rtk pytest -q tests/validation/test_reporting.py tests/validation/test_wandb_logging.py tests/cli/test_validate_hard_numbers.py
```

Expected: FAIL because report/W&B APIs are absent.

- [ ] **Step 3: Implement exact merged outputs and human-readable report**

Write, in stable ID order:

```text
manifest.jsonl
hypotheses.jsonl
per_utt.tsv
metrics.json
report.md
run_metadata.json
```

`metrics.json` must include overall and category metrics, word/character S/D/I/C/N, failures, 2,000 coverage, synthesis/ASR throughput, and wall time. `report.md` must state that CER uses reference-character micro-weighting and therefore intentionally differs from the source evaluator.

- [ ] **Step 4: Implement online authentication preflight and stable run identity**

```python
class WandbRunStore:
    def __init__(self, path: Path, project: str = "omnivoice-lora-validation"):
        self.path = path
        self.project = project

    def preflight(self) -> None:
        api = wandb.Api(timeout=15)
        if not api.api_key:
            raise RuntimeError("W&B online authentication is required; run `wandb login`")

    def init(self, config: Mapping[str, object]):
        run_id = self.load_or_create_id()
        return wandb.init(project=self.project, id=run_id, resume="allow", config=dict(config))
```

Persist the run ID atomically in `wandb_ids.json`. Preflight happens in the controller before any GPU subprocess. Log metrics at optimizer step and only instantiate four `wandb.Audio` values.

- [ ] **Step 5: Add the score CLI**

Require exact rank-ledger coverage, compute scores, write all local artifacts first, then initialize/resume W&B and log. Accept `--dev-loss` for checkpoint validation loss and `--steps-per-epoch` for fractional epoch. A W&B failure leaves complete local metrics intact but returns nonzero so training does not silently resume.

```python
require_exact_coverage(expected_ids, synthesis_ids)
require_exact_coverage(expected_ids, hypothesis_ids)
result = score_validation_run(assignments, hypotheses)
write_validation_report(paths, result)
run = WandbRunStore(paths.wandb_ids).init(config=run_metadata)
log_validation(
    run,
    result,
    fixed_audio_records,
    step=args.step,
    steps_per_epoch=args.steps_per_epoch,
)
```

- [ ] **Step 6: Run focused tests and commit**

Run:

```bash
rtk pytest -q tests/validation/test_reporting.py tests/validation/test_wandb_logging.py tests/cli/test_validate_hard_numbers.py
rtk ruff check omnivoice/validation/reporting.py omnivoice/validation/wandb_logging.py omnivoice/cli/validate_hard_numbers.py tests/validation
```

Expected: PASS.

Commit:

```bash
rtk git add omnivoice/validation/reporting.py omnivoice/validation/wandb_logging.py omnivoice/cli/validate_hard_numbers.py tests/validation tests/cli/test_validate_hard_numbers.py
rtk git commit -m "feat: report validation metrics to W&B"
```

---

### Task 11: Checkpoint-Isolated Train/Validate Controller

**Files:**
- Create: `omnivoice/validation/controller.py`
- Create: `omnivoice/cli/run_lora_validation.py`
- Create: `examples/config/hard_number_validation.json`
- Create: `examples/run_finetune_lora_with_validation.sh`
- Create: `tests/validation/test_controller.py`
- Modify: `tests/test_lora_examples.py`
- Modify: `docs/training.md`

**Interfaces:**
- Consumes: training/data/validation configs, optional resume checkpoint, stable W&B state, GPU deadline.
- Produces: `validation_interval_steps(...)`, `validation_boundaries(...)`, `ValidationController.run(...)`, base step-0 evaluation, and segmented train → synth → ASR → score → resume execution.

- [ ] **Step 1: Write boundary and subprocess-contract tests**

```python
assert validation_interval_steps(5000) == 625
assert validation_interval_steps(1) == 1
assert validation_boundaries(current_step=0, total_steps=1400, steps_per_epoch=5000) == [625, 1250, 1400]
```

With a fake command runner, assert exact ordering:

```text
wandb-preflight
synth base step-0
asr step-0
score step-0
train --stop-after-step 625
synth --adapter-checkpoint checkpoint-625
asr step-625
score step-625
train --resume-from-checkpoint checkpoint-625 --stop-after-step 1250
```

Assert training never starts after incomplete synthesis, ASR, scoring, or W&B logging; rerunning resumes the same stage and W&B ID; and train config `steps` remains unchanged in every segmented invocation.

- [ ] **Step 2: Run controller tests and verify RED**

Run:

```bash
rtk pytest -q tests/validation/test_controller.py tests/test_lora_examples.py
```

Expected: FAIL because controller APIs and example files are absent.

- [ ] **Step 3: Implement boundaries and subprocess plans as pure functions**

```python
def validation_interval_steps(steps_per_epoch: int) -> int:
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive")
    return math.ceil(steps_per_epoch / 8)


def validation_boundaries(current_step: int, total_steps: int, steps_per_epoch: int) -> list[int]:
    interval = validation_interval_steps(steps_per_epoch)
    boundaries = list(range(((current_step // interval) + 1) * interval, total_steps + 1, interval))
    if current_step < total_steps and (not boundaries or boundaries[-1] != total_steps):
        boundaries.append(total_steps)
    return boundaries
```

Represent each command as `list[str]` and log it to controller state before `subprocess.run(check=True)`. Use `accelerate launch --multi_gpu --gpu_ids 0,1,2,3,4,5,6,7 --num_processes 8` for training/synthesis/ASR and a single process for scoring.

- [ ] **Step 4: Implement controller state, base evaluation, and segmented resume**

Persist `controller_state.json` atomically with current checkpoint, stage, step, deadline, resolved base revision, dataset revision, assignments hash, and W&B ID. The CLI accepts `--deadline-state`; it reads `experiment_deadline_monotonic` from the memorization `result.json` so both phases share one clock. On a fresh run, perform W&B and HF preflight, create assignments, and validate the base at step 0. At each boundary, pass `--stop-after-step` while leaving total `steps` untouched, require the complete checkpoint, parse the trainer's `TrainingOutcome.last_eval_loss`, pass that value to scoring, validate the checkpoint, then resume from that same checkpoint.

```python
if state.step == 0 and not state.base_validated:
    run_validation(model=base_model, step=0, dev_loss=None, deadline=deadline)
for boundary in validation_boundaries(
    state.step, train.steps, validation.steps_per_epoch
):
    outcome = run_train_segment(
        resume_checkpoint=state.checkpoint,
        stop_after_step=boundary,
        total_steps=train.steps,
    )
    checkpoint = output_dir / f"checkpoint-{boundary}"
    require_complete_checkpoint(checkpoint)
    run_validation(
        adapter_checkpoint=checkpoint,
        step=boundary,
        dev_loss=outcome.last_eval_loss,
        deadline=deadline,
    )
    state = state.after_validation(checkpoint, boundary)
```

- [ ] **Step 5: Add configuration, launcher, and operator documentation**

The validation JSON contains:

```json
{
  "steps_per_epoch": 5000,
  "dataset_repo": "bitmanagerai/hard_number_eval_for_tts",
  "dataset_revision": "57b964492ccfcedd6a24d0225ef4b7d3697ffdca",
  "base_model": "k2-fsa/OmniVoice",
  "wandb_project": "omnivoice-lora-validation",
  "world_size": 8,
  "seed": 42
}
```

The Bash launcher resolves every config relative to its own directory and calls only `python -m omnivoice.cli.run_lora_validation`. Document W&B login, artifact layout, resume behavior, 1/8-epoch semantics, and that 20 random clips are not guaranteed to be 20 speakers.

- [ ] **Step 6: Run controller/example tests and commit**

Run:

```bash
rtk pytest -q tests/validation/test_controller.py tests/test_lora_examples.py
rtk bash -n examples/run_finetune_lora_with_validation.sh
rtk python -m json.tool examples/config/hard_number_validation.json
rtk ruff check omnivoice/validation/controller.py omnivoice/cli/run_lora_validation.py tests/validation/test_controller.py
```

Expected: all commands pass.

Commit:

```bash
rtk git add omnivoice/validation/controller.py omnivoice/cli/run_lora_validation.py examples/config/hard_number_validation.json examples/run_finetune_lora_with_validation.sh tests/validation/test_controller.py tests/test_lora_examples.py docs/training.md
rtk git commit -m "feat: orchestrate checkpoint validation cycles"
```

---

### Task 12: Full Verification and Budgeted Real Runs

**Files:**
- Create: `/root/ml-intern-runs/omnivoice-bal-hard-number/path-isolated-controller/VERIFY.md`
- Create: `/root/ml-intern-runs/omnivoice-bal-hard-number/path-isolated-controller/train.log`
- Create: `/root/ml-intern-runs/omnivoice-bal-hard-number/RESULTS.md`
- Modify: `/root/ml-intern-runs/omnivoice-bal-hard-number/BUDGET.md`
- Modify: `/root/ml-intern-runs/omnivoice-bal-hard-number/EXPERIMENTS.md`
- Runtime artifacts: `exp/omnivoice_lora_memorization/`
- Runtime artifacts: `exp/omnivoice_validation/<run-id>/`

**Interfaces:**
- Consumes: every preceding implementation task, authenticated HF account, authenticated online W&B account, eight visible GPUs.
- Produces: passing static/unit/integration verification, four original/initial/final memorization audio sets, bounded memorization result, resumable or complete step-0 2,000-prompt validation, W&B run URL, and final experiment ledger.

- [ ] **Step 1: Install the validation extra and run all static checks**

Run:

```bash
rtk uv sync --extra validation
rtk python -m compileall -q omnivoice tests
rtk ruff check omnivoice tests
rtk bash -n examples/run_lora_memorization.sh examples/run_finetune_lora_with_validation.sh
rtk python -m json.tool examples/config/train_config_lora_memorization.json
rtk python -m json.tool examples/config/hard_number_validation.json
rtk git diff --check
```

Expected: every command exits 0.

- [ ] **Step 2: Run the complete test suite**

Run:

```bash
rtk pytest -q
```

Expected: all repository tests pass with no new warning category.

- [ ] **Step 3: Perform external-auth and hardware preflight before starting the clock**

Run:

```bash
rtk hf auth whoami --format json
rtk wandb status
rtk nvidia-smi -L
rtk python -m omnivoice.scripts.prepare_balalaika_samples --output-dir exp/omnivoice_validation/selection
```

Expected: HF identifies the existing account, W&B is online/authenticated, eight GPUs are listed, and selection publishes exactly 4+20 disjoint clips. If W&B is unauthenticated, stop before GPU work and ask the user to run `wandb login`; do not switch to offline mode.

- [ ] **Step 4: Fire ML milestone notifications and start one shared monotonic deadline**

Run the skill notification script with `plan_ready`, then `code_ready`, then `train_started`. Record `experiment_started_monotonic`, `deadline_monotonic = start + 3600`, and initial `nvidia-smi` inventory in the run ledger. Notification scripts may no-op when credentials are absent.

```bash
rtk bash /root/.agents/skills/ml-intern/scripts/notify.sh plan_ready "checkpoint-isolated OmniVoice validation plan ready"
rtk bash /root/.agents/skills/ml-intern/scripts/notify.sh code_ready "OmniVoice validation implementation passed smoke tests"
rtk bash /root/.agents/skills/ml-intern/scripts/notify.sh train_started "bounded memorization and base hard-number validation"
```

- [ ] **Step 5: Run the four-sample memorization phase for at most 20 minutes**

Run:

```bash
rtk env CUDA_VISIBLE_DEVICES=0 python -m omnivoice.cli.run_memorization \
  --selected-manifest exp/omnivoice_validation/selection/selected.jsonl \
  --output-dir exp/omnivoice_lora_memorization \
  --train-config examples/config/train_config_lora_memorization.json \
  --data-config examples/config/data_config_lora_memorization.json \
  --max-wall-clock-seconds 1200 \
  --experiment-wall-clock-seconds 3600
```

Expected: it either reaches two consecutive losses `<=1e-4` and saves qualifying audio, or stops cleanly at an evaluation boundary with explicit failure and resumable checkpoint. Record minimum loss, qualifying step, stretch status, elapsed GPU minutes, and all twelve listening paths in `VERIFY.md`.

- [ ] **Step 6: Run one local rank-0 synthesis/ASR/score smoke**

Use the first assignment with `world_size=1` and a separate `smoke/` directory. Expected: base TTS generates a non-empty 24 kHz WAV, GigaAM returns a string, the score contains all four metrics, and no TTS process remains while ASR loads. This smoke uses the shared one-hour deadline and is not retried automatically.

- [ ] **Step 7: Run the base-model 2,000-prompt validation with the remaining deadline**

Run:

```bash
rtk python -m omnivoice.cli.run_lora_validation \
  --validation-config examples/config/hard_number_validation.json \
  --train-config examples/config/train_config_finetune_lora.json \
  --data-config examples/config/data_config_finetune.json \
  --output-root exp/omnivoice_validation \
  --base-only \
  --deadline-state exp/omnivoice_lora_memorization/result.json
```

Expected: complete runs produce 2,000 WAVs, 2,000 hypotheses, four metrics, per-category tables, four W&B audio examples, and local report files at step 0. Deadline-limited runs stop atomically and are labeled incomplete with exact completed counts; they are not automatically retried and do not log aggregate metrics.

- [ ] **Step 8: Write adapted ML verification and results**

`VERIFY.md` records:

```markdown
## Generation sanity
initial and final paths for four fixed utterances; non-empty 24 kHz PCM16 checks

## Loss sanity
minimum teacher-forced loss, two-consecutive-hit evidence, required/stretch verdicts

## Eval tracks train
last training loss and deterministic four-row dev loss with absolute difference

## Data consumption
planned optimizer steps, actual steps, stop reason, four unique sample IDs

## Stderr scan
Traceback/RuntimeError/Warning findings, each classified as benign or failing

## Adapter count
trainable and total parameter count plus proof only LoRA parameters are trainable

## Hard-number validation
coverage, four metrics if complete, W&B URL, local artifact root, deadline status
```

Update `EXPERIMENTS.md` to `passed` only when code/tests pass and memorization meets `<=1e-4`; otherwise record the exact first failing check. Update spent GPU minutes even for incomplete work. `RESULTS.md` summarizes both phases without claiming missing metrics.

- [ ] **Step 9: Run final repository verification and commit only code/docs**

Run:

```bash
rtk pytest -q
rtk ruff check omnivoice tests
rtk git diff --check
rtk git status --short
```

Expected: tests/lint pass; generated audio, checkpoints, W&B state, and run ledgers remain untracked and are not committed.

Commit implementation documentation changes, if any:

```bash
rtk git add docs examples omnivoice tests pyproject.toml uv.lock
rtk git commit -m "docs: finish hard-number validation workflow"
```

Do not create a commit when the index is empty. Do not push the branch or publish artifacts. Fire `train_done` only when the bounded run has a passing verification result; otherwise fire `error` with the exact failed verdict and preserve resumable artifacts.

---

## Execution Order

Execute tasks strictly in numeric order: 1 through 12. Tasks 5–8 are placed earlier in this document only to keep the pure hard-number data path adjacent to the global metric constraints; their interfaces still consume Tasks 1–4 exactly as stated.
