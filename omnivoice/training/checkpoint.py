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

"""Checkpoint saving, resuming, and training logging.

Provides utilities for saving/loading training checkpoints and logging metrics
to console and trackers (TensorBoard/WandB). Used by ``OmniTrainer``.

Key components:
- ``TrainLogger``: Logs training metrics to console and Accelerate trackers.
- ``save_checkpoint()``: Saves model, optimizer, and scheduler state.
- ``load_checkpoint()``: Restores training state from a checkpoint directory.
"""

import json
import logging
import os
import shutil
import time
from typing import Any, Dict, Optional

import torch
from accelerate import Accelerator
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

LORA_AUDIO_MODULES_FILENAME = "omnivoice_audio_modules.safetensors"
LORA_TRAINABLE_MODULES_FILENAME = "omnivoice_trainable_non_lora.safetensors"


def save_lora_audio_modules(model, adapter_dir, *, base_omnivoice_checkpoint):
    os.makedirs(adapter_dir, exist_ok=True)
    state = {
        f"{name}.{key}": value.detach().cpu().contiguous()
        for name in ("audio_embeddings", "audio_heads")
        for key, value in getattr(model, name).state_dict().items()
    }
    save_safetensors(state, os.path.join(adapter_dir, LORA_AUDIO_MODULES_FILENAME))
    with open(os.path.join(adapter_dir, "omnivoice_lora.json"), "w") as handle:
        json.dump({"format_version": 1, "base_omnivoice_checkpoint": base_omnivoice_checkpoint,
                   "audio_module_state": LORA_AUDIO_MODULES_FILENAME,
                   "audio_modules": ["audio_embeddings", "audio_heads"]}, handle)


def load_lora_audio_modules(model, adapter_dir):
    path = os.path.join(adapter_dir, LORA_AUDIO_MODULES_FILENAME)
    if not os.path.isfile(path):
        return False
    state = load_safetensors(path, device="cpu")
    for name in ("audio_embeddings", "audio_heads"):
        prefix = name + "."
        getattr(model, name).load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)})
    return True


def save_lora_trainable_modules(model, adapter_dir):
    """Persist non-adapter trainables in a lightweight LoRA checkpoint.

    Audio embeddings/heads have their own stable sidecar.  This captures an
    optional unfrozen transformer suffix so a checkpoint is an exact inference
    artifact rather than silently reverting those layers to the base model.
    """
    state = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and "lora_" not in name
        and not name.startswith("audio_embeddings.")
        and not name.startswith("audio_heads.")
    }
    if not state:
        return False
    path = os.path.join(adapter_dir, LORA_TRAINABLE_MODULES_FILENAME)
    save_safetensors(state, path)
    metadata_path = os.path.join(adapter_dir, "omnivoice_lora.json")
    with open(metadata_path) as handle:
        metadata = json.load(handle)
    metadata["trainable_non_lora_state"] = LORA_TRAINABLE_MODULES_FILENAME
    metadata["trainable_non_lora_parameters"] = int(sum(value.numel() for value in state.values()))
    with open(metadata_path, "w") as handle:
        json.dump(metadata, handle)
    return True


def load_lora_trainable_modules(model, adapter_dir):
    path = os.path.join(adapter_dir, LORA_TRAINABLE_MODULES_FILENAME)
    if not os.path.isfile(path):
        return False
    state = load_safetensors(path, device="cpu")
    named_parameters = dict(model.named_parameters())
    missing = sorted(set(state) - set(named_parameters))
    if missing:
        raise RuntimeError(
            "Lightweight LoRA checkpoint has trainable tensors absent from the current model: "
            + ", ".join(missing[:3])
        )
    with torch.no_grad():
        for name, value in state.items():
            named_parameters[name].copy_(value.to(dtype=named_parameters[name].dtype))
    return True


class TrainLogger:
    """
    Handles logging to console and trackers (TensorBoard/WandB)
    """

    def __init__(self, accelerator: Accelerator, total_steps: int, logging_steps: int):
        self.accelerator = accelerator
        self.total_steps = total_steps
        self.logging_steps = logging_steps
        self.start_time = None
        self.progress_bar = None
        self.epoch_progress_bar = None

    def start(self, start_step: int = 0):
        self.start_time = time.time()

        if self.accelerator.is_main_process:
            self.progress_bar = tqdm(
                total=self.total_steps,
                initial=start_step,
                desc="Training",
                dynamic_ncols=True,
                disable=not self.accelerator.is_local_main_process,
            )

    def update(
        self, step: int, loss: Optional[float] = None, lr: Optional[float] = None
    ):
        """
        Called every step to update the progress bar UI.
        """
        if self.progress_bar is not None:
            self.progress_bar.update(1)

            # Update real-time metrics on the progress bar itself
            postfix = {}
            if loss is not None:
                postfix["loss"] = f"{loss:.4f}"
            if lr is not None:
                postfix["lr"] = f"{lr:.2e}"

            if postfix:
                self.progress_bar.set_postfix(postfix)

        if self.epoch_progress_bar is not None:
            self.epoch_progress_bar.update(1)

    def start_epoch(self, epoch: int, total_steps: Optional[int] = None):
        """Start the progress bar for the current pass through the dataloader."""
        if self.epoch_progress_bar is not None:
            self.epoch_progress_bar.close()

        if self.accelerator.is_main_process:
            self.epoch_progress_bar = tqdm(
                total=total_steps,
                desc=f"Epoch {epoch + 1}",
                dynamic_ncols=True,
                disable=not self.accelerator.is_local_main_process,
                leave=False,
            )

    def close_epoch(self):
        """Close the current epoch progress bar, if one is active."""
        if self.epoch_progress_bar is not None:
            self.epoch_progress_bar.close()
            self.epoch_progress_bar = None

    def log_metrics(self, step: int, metrics: Dict[str, Any]):
        """
        Called periodically to log to TensorBoard/WandB and console.
        """
        # Log to trackers (TensorBoard, etc.)
        self.accelerator.log(metrics, step=step)

        if self.accelerator.is_main_process:
            # Format for console log (separate from tqdm)
            # Remove keys that are redundant or too verbose for one line
            formatted_metrics = []
            for k, v in metrics.items():
                if isinstance(v, float):
                    val_str = f"{v:.4f}"
                    if val_str == "0.0000" and v != 0:
                        formatted_metrics.append(f"{k}: {v:.2e}")
                    else:
                        formatted_metrics.append(f"{k}: {val_str}")
                else:
                    formatted_metrics.append(f"{k}: {v}")

            # Use external logger to write to file, tqdm.write to avoid breaking bar
            msg = f"Step {step} | " + " | ".join(formatted_metrics)
            if self.progress_bar is not None:
                self.progress_bar.write(msg)
            else:
                logger.info(msg)

    def close(self):
        self.close_epoch()
        if self.progress_bar is not None:
            self.progress_bar.close()


def save_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    tokenizer: Any,
    output_dir: str,
    step: int,
    keep_last_n: int = 3,
):
    """
    Saves model, tokenizer, and accelerator states (optimizer/scheduler).
    Manages rotation of checkpoints.
    """
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")

    # A LoRA adapter plus the audio-module sidecar is a complete inference
    # checkpoint when paired with the immutable base model.  Saving
    # Accelerate's model state as well writes another multi-gigabyte copy of
    # that base model at every validation checkpoint and exhausts local disk
    # during a normal 4-epoch run.  Preserve exact engine state for full-model
    # fine-tuning; keep LoRA checkpoints lightweight and portable.
    unwrapped_model = accelerator.unwrap_model(model)
    is_lora_checkpoint = hasattr(getattr(unwrapped_model, "llm", None), "peft_config")
    if is_lora_checkpoint:
        os.makedirs(checkpoint_dir, exist_ok=True)
    else:
        accelerator.save_state(checkpoint_dir)
    accelerator.wait_for_everyone()

    # The adapter sidecar is exported by OmniTrainer after this engine-level
    # checkpoint setup.  Save tokenizer metadata here for both checkpoint
    # formats.
    if accelerator.is_main_process:
        tokenizer.save_pretrained(checkpoint_dir)
    accelerator.wait_for_everyone()

    logger.info(f"Saved checkpoint to {checkpoint_dir}")

    # 4. Rotate checkpoints (Keep last N)
    if accelerator.is_main_process and keep_last_n > 0:
        checkpoints = [
            d
            for d in os.listdir(output_dir)
            if d.startswith("checkpoint-")
            and os.path.isdir(os.path.join(output_dir, d))
        ]
        # Sort by step number
        checkpoints.sort(key=lambda x: int(x.split("-")[-1]))

        if len(checkpoints) > keep_last_n:
            to_remove = checkpoints[:-keep_last_n]
            for d in to_remove:
                shutil.rmtree(os.path.join(output_dir, d))
                logger.info(f"Removed old checkpoint {d}")


def load_checkpoint(accelerator: Accelerator, checkpoint_path: str):
    """
    Resumes training state.
    """
    logger.info(f"Resuming from {checkpoint_path}")
    accelerator.load_state(checkpoint_path)

    # Try to infer step
    try:
        clean_path = os.path.normpath(checkpoint_path)
        step = int(os.path.basename(clean_path).split("-")[-1])
        return step
    except ValueError:
        return 0
