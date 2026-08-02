# Balalaika Memorization and Hard-Number Validation Design

Date: 2026-08-02

## Summary

Extend the OmniVoice LoRA branch with two reproducible GPU workflows:

1. A real four-utterance Balalaika v2 memorization run that saves original and
   generated audio and stops after validation loss is at most `1e-4` for two
   consecutive evaluations. Reaching `1e-5` is reported as a stretch result.
2. A checkpoint-isolated validation stage that synthesizes all 2,000 prompts
   from `bitmanagerai/hard_number_eval_for_tts`, scores utterance- and
   number-region CER/WER with GigaAM-v3 RNN-T, saves complete local artifacts,
   and logs metrics plus four fixed audio examples to Weights & Biases.

The initial real validation run evaluates `k2-fsa/OmniVoice`. Future LoRA
training invokes the full 2,000-prompt validation every one eighth of a
configured epoch. The initial GPU experiment has a hard one-hour wall-clock
budget and no automatic retry.

## Existing System Constraints

- The training loop is optimizer-step based and consumes an iterable,
  token-batched WebDataset. It cannot derive epoch length reliably.
- Training model construction uses `train=True` and therefore does not load the
  inference audio tokenizer or voice-clone helpers.
- A validation stage must not perturb training RNG, optimizer, scheduler, DDP,
  or model mode.
- The LoRA checkpoint root is the stable handoff boundary: it contains the
  adapter, tokenizer, metadata, and resumable Accelerate state.
- Full generation validation is intentionally expensive. The user explicitly
  selected all 2,000 prompts at every one-eighth-epoch boundary.

## Pinned Inputs

### Balalaika v2

Source root:

```text
/workspace/balalaika_proprietary_v2
```

The source contains 519 WebDataset TAR shards and approximately 4.08 million
Russian audio samples. Audio members are MP3. The authoritative text input for
both memorization and voice references is the combined punctuation/stress
sidecar:

```text
/workspace/balalaika_proprietary_v2/combined_sidecars/rover-punctuation-stress-v1/rover-punctuation-stress.jsonl
```

It has 4,075,032 rows. Join sidecar rows to shard audio by
`source_relative_path`, and use `rover_punctuated_accented` as the transcript.
Do not use TAR-member transcript fields as model text. `speaker_id` is not
reliable, so a selected reference clip is called a “voice” without claiming
speaker identity.

### Hard-number benchmark

Repository:

```text
bitmanagerai/hard_number_eval_for_tts
```

Pinned revision:

```text
57b964492ccfcedd6a24d0225ef4b7d3697ffdca
```

The consolidated `hard_number_eval.jsonl` has exactly 2,000 rows. Required
fields are `id`, `category`, `hard_number`, `text`, `normalized_gold`, and
`stressed`. TTS input is `stressed`; ASR output is scored against
`normalized_gold`. Raw digit-containing `text` is used only to locate the
number-bearing span.

### Models

- Initial TTS baseline: `k2-fsa/OmniVoice`, with the resolved Hub revision
  recorded in every run manifest.
- ASR: GigaAM-v3 RNN-T, matching the benchmark repository’s reference
  evaluation.

## Balalaika Selection and Conversion

Selection is deterministic with seed `42`.

Eligible rows must have a matching sidecar row with a non-empty
`rover_punctuated_accented` value and a matching MP3 that decodes to a non-empty
waveform. Selection uses audio duration only; it does not inspect or filter on
`DistillMOS`, `music_prob`, TAR-member transcripts, or other quality metadata.

Candidate shards and members are visited in seeded random order. Prefer clips
whose decoded duration is between 3 and 12 seconds. If this preferred pool does
not fill all 24 requested clips, continue in the same deterministic order with
valid clips longer than 12 seconds. There is no hard upper-duration cutoff.
Clips shorter than 3 seconds remain excluded because they provide weak voice
references and poor four-sample memorization examples.

Select two disjoint sets:

- four memorization utterances;
- twenty validation voice references.

Every selected MP3 is converted with FFmpeg to mono, 24 kHz, PCM signed 16-bit
WAV. The manifest records seed, source shard, member name,
`source_relative_path`, sidecar schema version, source SHA-256, converted WAV
SHA-256, `rover_punctuated_accented` transcript, sample rate, duration, whether
the preferred or fallback duration tier supplied the clip, and selection role.
Output publication is atomic.

## Four-Utterance Memorization Workflow

The memorization workflow reuses the existing LoRA tokenization, training,
checkpoint, and teacher-forced evaluation paths.

Inputs:

- converted four-row raw JSONL manifest;
- one training and one dev view referencing the same four tokenized samples;
- single GPU (`cuda:0`), deterministic seed `42`.

Behavior:

1. Save the four converted reference WAVs under `original/`.
2. Generate and save an `initial/` reconstruction from the unmodified base.
3. Tokenize the four samples once and train the broad LoRA adapter.
4. Evaluate frequently and append `{step, loss, elapsed_seconds}` to
   `loss_history.jsonl`.
5. Stop when loss is at most `1e-4` for two consecutive evaluations.
6. Record whether loss at most `1e-5` was reached.
7. Generate and save the same four prompts under `final/` using the best
   qualifying checkpoint, or the final checkpoint if the target was missed.

The memorization GPU phase receives at most 20 minutes of the one-hour initial
experiment budget. A wall-clock guard stops cleanly at the next evaluation
boundary, saves a checkpoint, and reports failure to meet the threshold rather
than claiming memorization. Maximum optimizer steps are `10_000`; evaluation
is every 25 optimizer steps. The existing four-sample one-GPU settings remain
otherwise unchanged unless a smoke test exposes a concrete incompatibility.

## Hard-Number Prompt and Voice Assignment

All 2,000 benchmark rows are sorted by `id`. The twenty validation references
are deterministically shuffled with seed `42`. Prompts are assigned in balanced
round-robin order, producing exactly 100 prompts per reference clip and exactly
2,000 generated utterances total. The mapping is immutable across checkpoints
so W&B curves compare the same prompt/voice pairs.

Generation uses:

- `language="Russian"`;
- benchmark field `stressed` as target text;
- Balalaika sidecar `rover_punctuated_accented` as reference transcript;
- the converted 24 kHz WAV as reference audio;
- `num_step=32`, `guidance_scale=2.0`, `t_shift=0.1`;
- `layer_penalty_factor=5.0`;
- `position_temperature=0.0`, `class_temperature=0.0` for deterministic
  checkpoint comparison;
- the repository defaults for chunking, padding, fades, denoising, and prompt
  preprocessing.

The validator records the complete generation configuration in its manifest.

## Checkpoint-Isolated Validation Controller

Add an explicit positive integer `steps_per_epoch`. Define:

```text
validation_interval_steps = ceil(steps_per_epoch / 8)
```

The controller owns one stable W&B run and alternates:

```text
train to next boundary -> save/exit -> validate checkpoint -> resume
```

Training retains its original total `steps`. A separate `stop_after_step`
limits one invocation without changing scheduler construction. Resume restores
the original optimizer and scheduler state, so segmentation cannot reset or
shorten the learning-rate schedule.

At a boundary:

1. Require the complete LoRA checkpoint and tokenizer metadata.
2. Launch eight validation ranks with one TTS model per visible GPU.
3. Partition the 2,000 immutable assignments evenly (250 rows per rank).
4. Synthesize missing rows only; write one atomic per-rank manifest.
5. Terminate synthesis workers and launch distributed GigaAM ASR so TTS and ASR
   do not coexist in GPU memory.
6. Transcribe missing WAVs only; write one atomic per-rank hypothesis file.
7. Main rank verifies complete unique ID coverage, scores, writes artifacts,
   and logs W&B.
8. Resume training only after validation is complete.

If any stage is interrupted, rerunning the same checkpoint resumes from valid
artifacts. A generation or ASR error records prompt ID, voice ID, checkpoint,
rank, and exception. Incomplete coverage stops the controller; it is never
silently excluded from aggregate metrics.

The first real hard-number run evaluates the base model at logical `step=0`.
It receives the remaining initial experiment budget (nominally 40 minutes).
If the one-hour total cap is reached, workers stop cleanly, partial artifacts
remain resumable, and no complete metric claim is made.

## Metrics

Normalization follows the benchmark evaluation:

- lowercase;
- replace `ё` with `е`;
- remove stress marks and punctuation;
- retain Cyrillic letters, digits, and normalized spaces.

Number-span extraction aligns raw digit-containing `text` with
`normalized_gold`, selects the changed reference span, then maps that span into
the ASR hypothesis through gold-to-hypothesis alignment.

For every utterance, record full-utterance and number-span edit counts. Aggregate
with true corpus micro-averaging:

- `utt_wer = total word S+D+I / total reference words`;
- `utt_cer = total character S+D+I / total reference characters`;
- `num_wer = number-span word S+D+I / number-span reference words`;
- `num_cer = number-span character S+D+I / number-span reference characters`.

Also report per-category metrics, S/D/I counts, ASR failures, synthesis
failures, coverage, synthesis throughput, ASR throughput, and wall time.

This intentionally corrects a bug in the benchmark reference script, which
weights CER using word counts. The output report documents the difference.

## Weights & Biases

Add W&B support while retaining TensorBoard. Online project:

```text
omnivoice-lora-validation
```

The controller fails before expensive GPU work unless W&B online
authentication succeeds. One stable W&B run ID is persisted locally and reused
across segmented training and resume.

At each boundary log:

- optimizer step and fractional epoch;
- training dev loss;
- the four aggregate hard-number metrics;
- per-category metric table;
- S/D/I, coverage, failures, throughput, and elapsed time;
- four fixed `wandb.Audio` examples using immutable prompt/voice pairs.

Only four audio samples are uploaded per validation. All generated audio stays
on local storage to protect the one-hour budget.

## Artifact Layout

```text
exp/omnivoice_lora_memorization/
  selected.jsonl
  original/*.wav
  generated/initial/*.wav
  generated/final/*.wav
  loss_history.jsonl
  checkpoint-*/

exp/omnivoice_validation/<run-id>/
  selection/voices.jsonl
  selection/voices/*.wav
  step-<step>/
    wavs/*.wav
    rank-manifests/*.jsonl
    rank-hypotheses/*.jsonl
    manifest.jsonl
    hypotheses.jsonl
    per_utt.tsv
    metrics.json
    report.md
    run_metadata.json
  wandb_ids.json
```

The four original and generated memorization audios are retained for direct
listening. Every validation WAV is retained locally and linked from its
per-utterance result.

## Testing

Use test-driven implementation with the following behavioral coverage:

- deterministic Balalaika sampling from a miniature TAR-plus-sidecar fixture;
- exact `source_relative_path` sidecar join and authoritative transcript field;
- duration-only preferred/fallback selection and disjoint four/twenty sets;
- explicit proof that MOS, music probability, and TAR-member text do not affect
  selection;
- FFmpeg conversion to mono 24 kHz PCM16 and manifest hashes;
- exact balanced prompt-to-voice assignment;
- metric golden cases for utterance and number spans;
- true character-count weighting for corpus CER;
- missing/failed ID coverage rejection;
- deterministic eight-rank partitioning and resumable artifact discovery;
- `ceil(steps_per_epoch / 8)` boundary scheduling;
- segmented stop/resume without scheduler reset;
- memorization loss patience, wall-clock guard, and best-checkpoint choice;
- W&B logging through a fake client, including exactly four fixed audio rows;
- shell/CLI behavioral tests using stubbed GPU-heavy commands;
- a small local base-model synthesis/score smoke before the full run.

Before real GPU work, run the complete repository suite, compile checks, Ruff on
all branch-added files, shell syntax, JSON validation, and checkpoint/inference
smokes.

## Experiment Budget and Completion

The initial run permits one implementation path, no automatic GPU retry, and a
hard one-hour wall-clock cap for GPU workloads:

- up to 20 minutes for memorization;
- remaining time, nominally 40 minutes, for one base-model 2,000-prompt run.

The implementation is complete when code/tests/docs pass, W&B and local
artifact paths are functional, and the real workflows either finish inside the
budget or stop resumably with an explicit incomplete result. Memorization is a
success only if two consecutive losses are at most `1e-4`. Hard-number metrics
are reportable only at 2,000/2,000 coverage.

## Non-Goals

- No 40,000-audio Cartesian product of 20 voices and 2,000 prompts.
- No guarantee that Balalaika source clips represent 20 distinct people.
- No Balalaika selection filter based on MOS, music probability, or other
  metadata-derived quality scores.
- No asynchronous evaluator competing with training GPUs.
- No raw-digit text-normalizer evaluation; TTS input is corrected stressed
  text.
- No upload of all 2,000 WAVs to W&B.
- No change to the accepted non-bit-exact CUDA embedding reconstruction policy.
