# Training

## Training Config

All training is controlled by a JSON training config file and a JSON data config file.

See [examples/config/](../examples/config/) for ready-to-use configs.

Training config file on Emilia is: [examples/config/train_config_emilia.json](../examples/config/train_config_emilia.json)

Data config file for Emilia is: [examples/config/data_config_emilia.json](../examples/config/data_config_emilia.json)


Key fields in training config file:

| Field | Description | Default |
|---|---|---|
| `llm_name_or_path` | local LLM path or huggingface id | Qwen/Qwen3-0.6B |
| `steps` | Total training steps | 300,000 |
| `learning_rate` | Peak learning rate | 1e-4 |
| `batch_tokens` | Tokens per batch on each GPU | 8192 |
| `attn_implementation` | Attention backend: `"flex_attention"` or `"sdpa"` | `"flex_attention"` |

`output_dir` and `data_config` are passed via command line (see below).

## Attention Implementation

By default, training uses `flex_attention`, which requires PyTorch ≥ 2.5 and a compatible GPU (e.g. NVIDIA Ampere or newer). If your environment does not support `flex_attention`, set `attn_implementation` to `"sdpa"` in your training config. See [examples/config/train_config_finetune_sdpa.json](../examples/config/train_config_finetune_sdpa.json) for a ready-to-use SDPA config:

```json
{
    "attn_implementation": "sdpa",
    "max_sample_tokens": 2000,
    "min_sample_tokens": 50,
    "max_batch_size": 64
}
```

`"sdpa"` uses PyTorch's built-in scaled dot-product attention and works on a wider range of hardware.

The following fields only apply when `attn_implementation != "flex_attention"`:

| Field | Description | Default |
|---|---|---|
| `max_sample_tokens` | Maximum token length per sample; longer samples are dropped | 2000 |
| `min_sample_tokens` | Minimum token length per sample; shorter samples are dropped | 50 |
| `max_batch_size` | Cap on the number of samples per batch | 64 |

`batch_tokens` remains the primary control for memory usage — it sets the total token budget per batch. `max_batch_size` is a safety guard to prevent a batch of many short samples from creating an unusually large batch dimension.

### Batching strategy

The two backends use **different batching strategies**, which are selected automatically:

| Backend | Batching strategy | Batch shape | Notes |
|---|---|---|---|
| `flex_attention` | Sequence packing | `[1, C, batch_tokens]` | Multiple samples concatenated into one long sequence; document boundaries tracked via `document_ids` |
| `sdpa` | Length-grouped padding | `[B, C, max_len]` | Samples with similar token lengths are grouped into the same batch and padded to the local maximum length |

**Why different strategies?**

- With `flex_attention`, sequence packing is memory-efficient because a compact `BlockMask` (not a dense matrix) describes which tokens can attend to each other across document boundaries.
- With `sdpa`, length-grouped padding is used instead: samples of similar token lengths are batched together and padded to the local maximum, so a lightweight `[B, 1, max_len, max_len]` boolean attention mask suffices with low overhead and minimal wasted padding.

## Launching Training

```bash
accelerate launch \
    --gpu_ids "0,1,2,3,4,5,6,7" \
    --num_processes 8 \
    -m omnivoice.cli.train \
    --train_config config/train_config_emilia.json \
    --data_config config/data_config_emilia.json \
    --output_dir exp/omnivoice_emilia
```

## Resuming Training

Set `resume_from_checkpoint` in your training config to resume from an existing checkpoint:

```json
{
    "resume_from_checkpoint": "exp/omnivoice/checkpoint-100000"
}
```

## LoRA Fine-tuning

For an eight-GPU adapter-only workflow, use
[examples/run_finetune_lora.sh](../examples/run_finetune_lora.sh) with
[examples/config/train_config_finetune_lora.json](../examples/config/train_config_finetune_lora.json).
It defaults to GPUs `0` through `7`, eight processes, and output directory
`exp/omnivoice_finetune_lora`. The example keeps the `flex_attention` and
bf16 defaults and starts large-dataset LoRA training at a learning rate of
`0.0001`.

The effective global token batch is:

```
batch_tokens × num_processes × gradient_accumulation_steps
```

With the example defaults this is `8192 × 8 × 1 = 65536` tokens. Reduce one
of these factors if memory or optimization behavior requires a smaller global
batch.

The config enables rank-32 LoRA with alpha 64, dropout 0.05, and no bias. Its
broad default targets include attention and MLP projections plus
`embed_tokens`, `audio_embeddings`, and `audio_heads`. To adapt a different
subset, edit `lora_target_modules`; each requested module suffix must match
the selected base model.

### Resume an adapter run

Each checkpoint is restartable: it contains Accelerator state (optimizer,
scheduler, and RNG state), tokenizer and training config files, plus an
`adapter/` directory with the PEFT adapter weights/config and an
`adapter_metadata.json` file. It does not contain a duplicate copy of the
frozen base-model weights. To resume, set the original LoRA config's
`resume_from_checkpoint` to a checkpoint, preserve its base and LoRA settings,
then run stage 1:

```json
{
    "resume_from_checkpoint": "exp/omnivoice_finetune_lora/checkpoint-5000"
}
```

```bash
# In examples/run_finetune_lora.sh, set stage=1 and stop_stage=1 first.
bash examples/run_finetune_lora.sh
```

Keep `init_from_checkpoint` accessible locally or from Hugging Face when
resuming or loading an adapter. The adapter metadata records that base model,
and the adapter alone cannot generate audio without it.

### Load a LoRA adapter for inference

`OmniVoice.from_lora_pretrained` reads the recorded base model, loads it, and
then attaches the adapter checkpoint:

```python
from omnivoice import OmniVoice
import torch

model = OmniVoice.from_lora_pretrained(
    "exp/omnivoice_finetune_lora/checkpoint-5000",
    device_map="cuda:0",
    dtype=torch.float16,
)
```

## Initializing from a Pretrained Model

To start training from a pretrained OmniVoice checkpoint (for fine-tuning):

```json
{
    "init_from_checkpoint": "exp/omnivoice/checkpoint-100000"
}
```

## Monitoring

Training logs to TensorBoard:
```bash
tensorboard --logdir exp/omnivoice_emilia/tensorboard
```
