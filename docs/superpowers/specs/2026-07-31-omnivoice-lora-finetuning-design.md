# OmniVoice Broad LoRA Fine-Tuning Design

## Goal

Add production-quality, adapter-only LoRA fine-tuning to OmniVoice. The feature
must support the existing Accelerate multi-GPU training path, target both the
Qwen backbone and OmniVoice-specific audio layers, save compact adapters, resume
training, load adapters for inference, and provide a deterministic four-sample
memorization check.

## Scope

The implementation extends the existing full fine-tuning pipeline rather than
introducing a second trainer. LoRA is opt-in; existing configurations and full
fine-tuning behavior remain unchanged.

The default broad adapter targets are:

- Qwen attention projections: `q_proj`, `k_proj`, `v_proj`, and `o_proj`.
- Qwen MLP projections: `gate_proj`, `up_proj`, and `down_proj`.
- Qwen input token embedding: `embed_tokens`.
- OmniVoice audio token embedding: `audio_embeddings`.
- OmniVoice audio output projection: `audio_heads`.

All original weights remain frozen. Only LoRA parameters are optimized and
saved. Layer names, rank, scaling, and dropout are configurable so narrower or
larger adapters do not require code changes.

The initial defaults are rank 32, alpha 64, dropout 0.05, and no trainable bias.
These defaults favor adaptation capacity for a large dataset while remaining
substantially smaller than full fine-tuning.

## Configuration and validation

`TrainingConfig` gains explicit LoRA fields rather than accepting an untyped
nested object. The fields cover enablement, rank, alpha, dropout, an explicit
adapter-only bias policy, and target module suffixes. The bias policy is fixed
to `none` so original model biases cannot become trainable. The example LoRA configuration starts from
`k2-fsa/OmniVoice` and uses the broad target list.

Model construction loads and resizes the complete pretrained OmniVoice model
before inserting adapters. This ordering prevents tokenizer resizing from
replacing an already wrapped embedding layer.

Adapter setup validates the requested target suffixes against the instantiated
model. It raises a clear error when a configured target is absent or when LoRA
would leave no trainable parameters. Startup logging reports total and trainable
parameter counts and the matched module names.

LoRA requires `init_from_checkpoint`; training a LoRA adapter over a randomly
initialized OmniVoice model is rejected. Resume configuration must reference a
checkpoint whose metadata identifies the same base model and compatible LoRA
structure.

## Model integration

PEFT wraps the full `OmniVoice` `PreTrainedModel`, not only `model.llm`. This
allows the same adapter to include standard Qwen linear/embedding modules and
the custom audio embedding and output head. A generic PEFT wrapper preserves
the OmniVoice `forward` contract used by `OmniTrainer`.

The project adds an adapter-aware loading entry point for inference. It first
loads the base OmniVoice checkpoint normally, including text/audio tokenizers,
then attaches the saved adapter in inference mode. Users supply the adapter
directory; metadata supplies and validates the base checkpoint identity.

## Optimizer and distributed training

The optimizer receives only parameters with `requires_grad=True`. This keeps
optimizer state adapter-sized and provides an invariant that tests can inspect.

The existing Accelerate DDP path remains responsible for gradient
synchronization. Eight visible GPUs use eight processes exactly as in full
fine-tuning. DeepSpeed is optional and is not required for the default LoRA
example. Mixed precision, attention backend selection, token-based batching,
evaluation, and data sharding retain their current behavior.

## Adapter checkpoints and resume

Each training checkpoint contains:

- PEFT adapter configuration and `adapter_model.safetensors`.
- `adapter_metadata.json` containing the base OmniVoice checkpoint, target
  modules, adapter hyperparameters, and training step.
- The tokenizer and OmniVoice training configuration.
- Optimizer, scheduler, scaler, and RNG state needed for exact training resume.

The frozen base model weights are not duplicated in LoRA checkpoints.
Accelerate save hooks remove the normal full-model payload after PEFT saves the
adapter. Resume attaches the adapter before Accelerator restores non-model
training state. Loading fails with a focused diagnostic if adapter files or
base-model metadata are missing or incompatible.

Checkpoint rotation continues to use the existing `checkpoint-<step>` naming
and retention policy. An adapter checkpoint is directly usable for inference;
no merge step is required.

## User-facing scripts and documentation

The repository gains:

- A broad-LoRA training JSON example.
- An eight-GPU-capable fine-tuning shell script following the existing example
  layout and data-tokenization stages.
- Documentation for configuration, training, resume, adapter-only inference,
  checkpoint contents, and target customization.
- A four-sample memorization script that consumes a user-provided raw JSONL
  manifest and selects four valid records deterministically.

The repository currently contains no audio utterances or raw JSONL manifest, so
the memorization script cannot be executed against in-repository data. It must
fail early with a clear message when no suitable manifest is supplied.

## Four-sample memorization validation

The model is trained with masked audio-token cross-entropy, not waveform mean
squared error. Waveform MSE is not an appropriate zero-loss assertion because
iterative generation and audio decoding are stochastic and small token changes
can shift the waveform.

The memorization workflow therefore uses the model's native objective:

1. Select exactly four valid, unique records from a supplied JSONL manifest
   using a fixed seed.
2. Encode those records with the existing Higgs audio tokenizer into an
   isolated WebDataset directory.
3. Use the same four samples as train and evaluation data.
4. Disable conditioning dropout and use fixed prompt and mask ratios so every
   evaluation measures the same target.
5. Train a broad LoRA adapter with a deliberately high-capacity memorization
   configuration.
6. Reload the saved adapter over the untouched base checkpoint.
7. Evaluate deterministic teacher-forced masked-token cross-entropy and fail
   the command unless it is below a configurable threshold, defaulting to 0.01.

This is an integration smoke test, not a measure of generalization. Unit tests
separately cover adapter targeting, frozen parameters, optimizer filtering,
adapter-only checkpoint contents, resume, and load round-tripping without
requiring eight GPUs.

## Error handling

Configuration errors are detected before distributed initialization where
possible. All ranks receive the same adapter structure. Missing PEFT support,
invalid ranks or dropout, unmatched target modules, missing base metadata,
incompatible resume checkpoints, insufficient memorization records, and a
failed memorization threshold produce actionable exceptions.

Checkpoint writes remain main-process-only where required and are synchronized
before training continues. Partial adapter directories are never presented as
valid inference artifacts.

## Non-goals

- QLoRA or quantized base-model training.
- Automatic adapter merging into a full OmniVoice checkpoint.
- Full fine-tuning of embeddings or audio heads alongside LoRA.
- Cross-utterance reference/target pairing changes.
- Changes to the audio tokenizer or WebDataset data format.
- Claiming model quality from the four-sample memorization check.

## Acceptance criteria

- Existing non-LoRA configurations retain their current behavior.
- Broad LoRA wraps every configured target and leaves original weights frozen.
- The optimizer contains only trainable adapter parameters.
- The documented eight-GPU launch uses the existing Accelerate/DDP path.
- Checkpoints contain compact adapter weights and resumable training state but
  no frozen base-model weight copy.
- An adapter checkpoint reloads over its recorded base model for evaluation and
  inference.
- Unit and integration tests cover the new behavior.
- Given a valid four-utterance manifest and adequate compute, the memorization
  command exits successfully only when deterministic masked-token
  cross-entropy is at most 0.01, or a caller-specified threshold.
