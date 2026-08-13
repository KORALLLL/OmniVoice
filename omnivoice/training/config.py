#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training configuration dataclass.

Defines ``TrainingConfig``, a dataclass that holds all hyperparameters and paths
for training. Loaded from a JSON config file via ``TrainingConfig.from_json()``
in ``omnivoice.cli.train``.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple


@dataclass
class TrainingConfig:
    # Key Paths
    output_dir: Optional[str] = None
    data_config: Optional[str] = None

    # Model Specific
    llm_name_or_path: str = "Qwen/Qwen3-0.6B"
    audio_vocab_size: int = 1025  # valid vocab size + 1 (mask token)
    audio_mask_id: int = 1024  # 1024 is the 1025-th token
    num_audio_codebook: int = 8

    # Model Training Specific
    audio_codebook_weights: List[float | int] = field(
        default_factory=lambda: [8, 8, 6, 6, 4, 4, 2, 2]
    )
    drop_cond_ratio: float = 0.1
    prompt_ratio_range: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.3))
    mask_ratio_range: Tuple[float, float] = field(default_factory=lambda: (0.0, 1.0))
    language_ratio: float = 0.8
    use_pinyin_ratio: float = 0.3
    instruct_ratio: float = 1.0
    only_instruct_ratio: float = 0.5

    # Init settings
    resume_from_checkpoint: Optional[str] = None
    resume_weights_only: bool = False
    resume_step: int = 0
    init_from_checkpoint: Optional[str] = None

    lora_rank: int = 0
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])
    lora_train_audio_modules: bool = False
    # Keep parameter-efficient adapters as the default, but allow a small
    # trainable suffix of the transformer when the task needs more acoustic
    # adaptation capacity than low-rank updates alone can provide.
    lora_train_last_n_layers: int = 0

    # A full generative validation intentionally keeps non-main DDP ranks at
    # a synchronization barrier while rank zero generates and transcribes the
    # benchmark.  Its duration can exceed PyTorch's one-hour default, so make
    # that watchdog bound explicit and configurable.
    distributed_timeout_minutes: int = 480

    # Training Hyperparams
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    steps: int = 300000
    seed: int = 42
    lr_scheduler_type: str = "cosine"
    warmup_type: str = "ratio"
    warmup_ratio: float = 0.03
    warmup_steps: int = 2000
    # When set, stop after this many fully consumed data epochs. ``steps`` still
    # defines the scheduler horizon and remains the fallback stop condition.
    max_epochs: Optional[int] = None
    # Optional epoch-relative validation schedule. A positive value runs this
    # many full validation passes per epoch, positioned by globally consumed
    # samples (therefore independent of variable packed-batch sizes).
    evals_per_epoch: int = 0
    estimated_steps_per_epoch: int = 0

    # Data
    batch_tokens: int = 8192
    gradient_accumulation_steps: int = 1
    num_workers: int = 8
    pin_memory: bool = True

    # System
    mixed_precision: str = "bf16"
    allow_tf32: bool = True
    use_deepspeed: bool = False
    deepspeed_config: Optional[str] = None
    attn_implementation: str = "flex_attention"

    # Length-grouped batching (only used when attn_implementation != "flex_attention")
    max_sample_tokens: int = 2000
    min_sample_tokens: int = 50
    max_batch_size: int = 64
    # Optional targeted-supervision filter.  It matches the tokenization's
    # selected label["text"] field and never changes audio/text pairings.
    train_label_text_regex: Optional[str] = None
    # Probability of keeping a sample that does NOT match train_label_text_regex.
    # 0.0 = historic hard filter (only matching samples train). That puts all
    # gradient on number-bearing text and trades whole-utterance quality for
    # digit-span accuracy. Use a value in (0, 1] to blend general speech back in
    # so both metrics can improve together; 1.0 disables filtering.
    train_label_text_keep_ratio: float = 0.0
    # Number of source samples in one effective training epoch.  This is used
    # for epoch-relative validation when a stream filter changes the manifest
    # count without materializing a second dataset.
    train_epoch_sample_count: Optional[int] = None

    # Logging
    logging_steps: int = 100
    eval_steps: int = 1000
    save_steps: int = 10000
    save_on_evaluation: bool = False
    deterministic_eval: bool = True
    save_before_evaluation: bool = False
    save_final_checkpoint: bool = True
    keep_last_n_checkpoints: int = -1

    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"

    hard_number_eval_enabled: bool = False
    hard_number_eval_dataset_path: Optional[str] = None
    hard_number_eval_seed: int = 42
    hard_number_eval_model_name: str = "v3_e2e_rnnt"
    hard_number_eval_voice_count: int = 40
    hard_number_eval_voice_manifest_path: Optional[str] = None
    # JSON list of speaker keys that must never be used as hard-eval references.
    # This makes a shard-based dev split genuinely speaker held out.
    hard_number_eval_excluded_speaker_keys_path: Optional[str] = None
    hard_number_eval_batch_size: int = 2
    # Cap on scored (text, voice) pairs for *periodic* in-training validation.
    # 0 keeps the full sweep.  The full sweep is ~2,000 pairs and costs 40-80
    # minutes, during which rank 0 owns the GPUs and every other rank spins in
    # an NCCL barrier; running that four times per epoch dominates wall clock
    # and is the window in which the collective deadlock was observed.  A few
    # hundred pairs tracks the trend well enough to steer a run; score the full
    # set once at the end.
    hard_number_eval_max_pairs: int = 0
    # Periodic evaluation samples this many texts per voice.  The historic
    # hard-coded value was 50 (50 x 40 voices = the 2,000-pair full sweep).
    hard_number_eval_samples_per_voice: int = 50

    @classmethod
    def from_json(cls, json_path: str):
        with open(json_path, "r") as f:
            cfg_dict = json.load(f)
        valid_keys = cls.__annotations__.keys()
        filtered_dict = {k: v for k, v in cfg_dict.items() if k in valid_keys}
        instance = cls(**filtered_dict)
        return instance

    def save_to_json(self, json_path: str):
        data = asdict(self)
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)
