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
import math
import re
from dataclasses import asdict, dataclass, field

from omnivoice.training.lora import DEFAULT_LORA_TARGET_MODULES

_HUB_COMMIT = re.compile(r"^[0-9a-f]{40,64}$")


@dataclass
class TrainingConfig:
    # Key Paths
    output_dir: str | None = None
    data_config: str | None = None

    # Model Specific
    llm_name_or_path: str = "Qwen/Qwen3-0.6B"
    audio_vocab_size: int = 1025  # valid vocab size + 1 (mask token)
    audio_mask_id: int = 1024  # 1024 is the 1025-th token
    num_audio_codebook: int = 8

    # Model Training Specific
    audio_codebook_weights: list[float | int] = field(
        default_factory=lambda: [8, 8, 6, 6, 4, 4, 2, 2]
    )
    drop_cond_ratio: float = 0.1
    prompt_ratio_range: tuple[float, float] = field(
        default_factory=lambda: (0.0, 0.3)
    )
    mask_ratio_range: tuple[float, float] = field(default_factory=lambda: (0.0, 1.0))
    language_ratio: float = 0.8
    use_pinyin_ratio: float = 0.3
    instruct_ratio: float = 1.0
    only_instruct_ratio: float = 0.5

    # Init settings
    resume_from_checkpoint: str | None = None
    init_from_checkpoint: str | None = None
    base_model_revision: str | None = None

    # LoRA fine-tuning
    lora_enabled: bool = False
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_target_modules: list[str] = field(
        default_factory=lambda: list(DEFAULT_LORA_TARGET_MODULES)
    )

    # Training Hyperparams
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    steps: int = 300000
    steps_per_epoch: int | None = None
    stop_after_step: int | None = None
    max_wall_clock_seconds: float | None = None
    early_stop_eval_loss: float | None = None
    early_stop_patience: int = 1
    eval_history_path: str | None = None
    seed: int = 42
    lr_scheduler_type: str = "cosine"
    warmup_type: str = "ratio"
    warmup_ratio: float = 0.03
    warmup_steps: int = 2000

    # Data
    batch_tokens: int = 8192
    gradient_accumulation_steps: int = 1
    num_workers: int = 8

    # System
    mixed_precision: str = "bf16"
    allow_tf32: bool = True
    use_deepspeed: bool = False
    deepspeed_config: str | None = None
    attn_implementation: str = "flex_attention"

    # Length-grouped batching (only used when attn_implementation != "flex_attention")
    max_sample_tokens: int = 2000
    min_sample_tokens: int = 50
    max_batch_size: int = 64

    # Logging
    logging_steps: int = 100
    eval_steps: int = 1000
    save_steps: int = 10000
    keep_last_n_checkpoints: int = -1

    def validate(self):
        """Validate bounded-training values before constructing the model."""
        integer_controls = (
            ("steps", self.steps),
            ("steps_per_epoch", self.steps_per_epoch),
            ("stop_after_step", self.stop_after_step),
            ("early_stop_patience", self.early_stop_patience),
        )
        for name, value in integer_controls:
            if value is None:
                continue
            if type(value) is not int:
                raise ValueError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        finite_controls = (
            ("max_wall_clock_seconds", self.max_wall_clock_seconds),
            ("early_stop_eval_loss", self.early_stop_eval_loss),
        )
        for name, value in finite_controls:
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        if self.stop_after_step is not None and self.stop_after_step > self.steps:
            raise ValueError("stop_after_step cannot exceed steps")
        if self.eval_history_path is not None and (
            not isinstance(self.eval_history_path, str)
            or not self.eval_history_path.strip()
        ):
            raise ValueError("eval_history_path must be a non-empty string")
        if self.base_model_revision is not None and (
            not isinstance(self.base_model_revision, str)
            or not _HUB_COMMIT.fullmatch(self.base_model_revision)
        ):
            raise ValueError("base_model_revision must be an immutable Hub commit")
        return self

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
