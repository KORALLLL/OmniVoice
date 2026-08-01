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

import logging
import os
import shutil
import time
from typing import Any, Dict, Optional

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType, gather_object
from tqdm.auto import tqdm

from omnivoice.training.lora import (
    _run_main_process_io,
    is_lora_model,
    save_lora_adapter,
)

logger = logging.getLogger(__name__)


def _run_all_process_io(accelerator, operation):
    """Run per-rank state I/O and raise one gathered error on every rank."""
    if getattr(accelerator, "distributed_type", None) == DistributedType.XLA:
        return operation()
    local_error = None
    result = None
    try:
        result = operation()
    except BaseException as exc:
        process_index = getattr(accelerator, "process_index", 0)
        local_error = f"process {process_index} {type(exc).__name__}: {exc}"
    gathered_errors = gather_object([local_error])
    first_error = next(
        (error for error in gathered_errors if error is not None), None
    )
    if first_error is not None:
        raise RuntimeError(f"Accelerate state I/O failed: {first_error}")
    return result


def _remove_path(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def _publish_checkpoint(staging_dir, checkpoint_dir):
    """Publish a complete checkpoint, restoring the prior one on failure."""
    checkpoint_parent = os.path.dirname(checkpoint_dir)
    checkpoint_name = os.path.basename(checkpoint_dir)
    backup_dir = os.path.join(checkpoint_parent, f".{checkpoint_name}.old")
    if os.path.exists(backup_dir):
        if os.path.exists(checkpoint_dir):
            _remove_path(backup_dir)
        else:
            os.replace(backup_dir, checkpoint_dir)
    if os.path.exists(checkpoint_dir):
        os.replace(checkpoint_dir, backup_dir)
    try:
        os.replace(staging_dir, checkpoint_dir)
    except BaseException:
        if os.path.exists(backup_dir):
            os.replace(backup_dir, checkpoint_dir)
        raise
    if os.path.exists(backup_dir):
        _remove_path(backup_dir)


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
        if self.progress_bar:
            self.progress_bar.update(1)

            # Update real-time metrics on the progress bar itself
            postfix = {}
            if loss is not None:
                postfix["loss"] = f"{loss:.4f}"
            if lr is not None:
                postfix["lr"] = f"{lr:.2e}"

            if postfix:
                self.progress_bar.set_postfix(postfix)

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
            if self.progress_bar:
                self.progress_bar.write(msg)
            else:
                logger.info(msg)

    def close(self):
        if self.progress_bar:
            self.progress_bar.close()


def save_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    tokenizer: Any,
    config: Any,
    output_dir: str,
    step: int,
    keep_last_n: int = 3,
):
    """
    Saves model, tokenizer, and accelerator states (optimizer/scheduler).
    Manages rotation of checkpoints.
    """
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    staging_dir = os.path.join(output_dir, f".checkpoint-{step}.tmp")

    def prepare_staging_dir():
        os.makedirs(output_dir, exist_ok=True)
        _remove_path(staging_dir)

    _run_main_process_io(accelerator, prepare_staging_dir)

    try:
        # 1. Save Accelerator State (Optimizer, Scheduler, RNG, Scaler)
        _run_all_process_io(
            accelerator, lambda: accelerator.save_state(staging_dir)
        )

        # 2. Save either the compact adapter or full model in HF format.
        unwrap_model = accelerator.unwrap_model(model)
        if is_lora_model(unwrap_model):
            save_lora_adapter(
                unwrap_model,
                staging_dir,
                config=config,
                step=step,
                accelerator=accelerator,
            )
        else:
            sharded_backends = {
                DistributedType.DEEPSPEED,
                DistributedType.FSDP,
                DistributedType.MEGATRON_LM,
            }
            if accelerator.distributed_type in sharded_backends:
                unwrap_model.save_pretrained(
                    staging_dir,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                )
            else:
                def save_full_model():
                    unwrap_model.save_pretrained(
                        staging_dir,
                        is_main_process=True,
                        save_function=accelerator.save,
                    )

                _run_main_process_io(accelerator, save_full_model)

        # 3. Save tokenizer/config, then publish the complete checkpoint.
        def save_metadata_and_publish():
            tokenizer.save_pretrained(staging_dir)
            if hasattr(config, "save_to_json"):
                config.save_to_json(os.path.join(staging_dir, "train_config.json"))
            _publish_checkpoint(staging_dir, checkpoint_dir)

        _run_main_process_io(accelerator, save_metadata_and_publish)
    except BaseException:
        if accelerator.is_main_process:
            _remove_path(staging_dir)
        raise

    logger.info(f"Saved checkpoint to {checkpoint_dir}")

    # 4. Rotate checkpoints (Keep last N)
    if keep_last_n > 0:
        def rotate_checkpoints():
            checkpoints = [
                d
                for d in os.listdir(output_dir)
                if d.startswith("checkpoint-")
                and os.path.isdir(os.path.join(output_dir, d))
            ]
            checkpoints.sort(key=lambda x: int(x.split("-")[-1]))

            if len(checkpoints) > keep_last_n:
                to_remove = checkpoints[:-keep_last_n]
                for checkpoint in to_remove:
                    shutil.rmtree(os.path.join(output_dir, checkpoint))
                    logger.info(f"Removed old checkpoint {checkpoint}")

        _run_main_process_io(accelerator, rotate_checkpoints)


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
