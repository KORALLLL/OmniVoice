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

"""Training loop for OmniVoice.

Wraps the HuggingFace Accelerate training loop with checkpoint saving/resuming,
evaluation, gradient accumulation, and learning rate scheduling.
Launched via ``omnivoice.cli.train``.
"""

import logging
import math
import os
import random
import sys
import time
from datetime import timedelta
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import (
    DeepSpeedPlugin,
    InitProcessGroupKwargs,
    broadcast_object_list,
    gather_object,
    set_seed,
)
from torch.utils.data import DataLoader
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from omnivoice.training.checkpoint import TrainLogger, load_checkpoint
from omnivoice.training.checkpoint import save_checkpoint as engine_save_checkpoint
from omnivoice.training.control import (
    EvaluationStopPolicy,
    StopDecision,
    TrainingOutcome,
    append_loss_history,
)
from omnivoice.training.lora import register_lora_state_hooks

logger = logging.getLogger(__name__)


def _to_device(batch, device):
    """Move all tensors in a batch dict to the target device."""
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def _raise_if_rank_failed(accelerator, stage, local_error):
    """Raise the same rank-local stage error on every distributed process."""
    errors = [local_error]
    if accelerator.num_processes > 1:
        errors = gather_object(errors)
    first_error = next((error for error in errors if error is not None), None)
    if first_error is not None:
        raise RuntimeError(f"{stage} failed: {first_error}")


def _capture_host_rng_state():
    """Capture host RNG state while reconstructing a resumed data cursor."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_host_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


class OmniTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        config: Any,  # TrainingConfig
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader | None = None,
        tokenizer: Any | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        lr_scheduler: Any | None = None,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # 1. Initialize Accelerator
        self.accelerator = self._init_accelerator()
        if self.config.lora_enabled:
            register_lora_state_hooks(self.accelerator)

        # 2. Setup Optimizer & Scheduler if not provided
        if optimizer is None:
            self.optimizer, self.lr_scheduler = self.create_optimizer_and_scheduler()
        else:
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler

        # 3. DeepSpeed Hack (Batch Size fix)
        if self.accelerator.distributed_type == "DEEPSPEED":
            self.accelerator.state.deepspeed_plugin.deepspeed_config[
                "train_micro_batch_size_per_gpu"
            ] = 1

        # 4. Prepare with Accelerator
        (
            self.model,
            self.optimizer,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.lr_scheduler,
        )

        self.global_step = 0
        self.epoch = 0

    def _init_accelerator(self) -> Accelerator:
        """Initialize Accelerator, DeepSpeed, and Logging."""
        # TF32 setup
        if getattr(self.config, "allow_tf32", False):
            torch.set_float32_matmul_precision("high")

        # Init handlers
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=60))

        # DeepSpeed setup
        deepspeed_plugin = None
        if self.config.use_deepspeed and self.config.deepspeed_config:
            if not os.path.exists(self.config.deepspeed_config):
                raise FileNotFoundError(
                    f"DeepSpeed config not found: {self.config.deepspeed_config}"
                )
            deepspeed_plugin = DeepSpeedPlugin(
                hf_ds_config=self.config.deepspeed_config,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                gradient_clipping=self.config.max_grad_norm,
            )

        accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            mixed_precision=self.config.mixed_precision,
            log_with="tensorboard",
            project_dir=self.config.output_dir,
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs, init_kwargs],
            deepspeed_plugin=deepspeed_plugin,
            split_batches=False,
        )

        # Logging setup
        if accelerator.is_main_process:
            os.makedirs(self.config.output_dir, exist_ok=True)
            # Try to save config if it has the method
            if hasattr(self.config, "save_to_json"):
                self.config.save_to_json(
                    os.path.join(self.config.output_dir, "initial_config.json")
                )

            logging.basicConfig(
                format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                datefmt="%m/%d/%Y %H:%M:%S",
                level=logging.INFO,
                handlers=[
                    logging.StreamHandler(sys.stdout),
                    logging.FileHandler(
                        os.path.join(self.config.output_dir, "train.log")
                    ),
                ],
            )
        else:
            logging.basicConfig(level=logging.ERROR)

        logger.info(f"Loaded Config: {self.config}")
        set_seed(self.config.seed)
        accelerator.init_trackers("tensorboard")
        return accelerator

    def create_optimizer_and_scheduler(self):
        """Default AdamW + configurable LR Scheduler."""
        trainable_parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise RuntimeError("Model has no trainable parameters")
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        if self.config.warmup_type == "ratio":
            final_warmup_steps = math.ceil(self.config.steps * self.config.warmup_ratio)
        else:
            final_warmup_steps = self.config.warmup_steps

        if self.config.lr_scheduler_type == "constant":
            lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=final_warmup_steps,
            )
        else:
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=final_warmup_steps,
                num_training_steps=self.config.steps,
            )
        return optimizer, lr_scheduler

    def save_checkpoint(self, step):
        """Wrapper for engine save_checkpoint."""
        engine_save_checkpoint(
            self.accelerator,
            self.model,
            self.tokenizer,
            self.config,
            self.config.output_dir,
            step,
            self.config.keep_last_n_checkpoints,
        )

    def load_checkpoint(self, checkpoint_path):
        """Wrapper for loading."""
        step = load_checkpoint(self.accelerator, checkpoint_path)
        self.global_step = step
        logger.info(f"Resumed from step {self.global_step}")
        return step

    def evaluate(self):
        """Evaluation loop."""
        if self.eval_dataloader is None:
            return {}

        logger.info(f"Running evaluation at step {self.global_step}...")
        local_loss_sum = torch.tensor(0.0, device=self.accelerator.device)
        eval_count = 0
        try:
            local_error = None
            try:
                self.model.eval()
                with torch.no_grad():
                    for eval_batch in self.eval_dataloader:
                        eval_batch = _to_device(eval_batch, self.accelerator.device)
                        outputs = self.model(**eval_batch)
                        local_loss_sum += outputs.loss.detach()
                        eval_count += 1
            except BaseException as exc:  # noqa: BLE001
                process_index = getattr(self.accelerator, "process_index", 0)
                local_error = (
                    f"process {process_index} {type(exc).__name__}: {exc}"
                )
            _raise_if_rank_failed(
                self.accelerator, "Evaluation forward pass", local_error
            )

            local_stats = torch.stack(
                (local_loss_sum, local_loss_sum.new_tensor(eval_count))
            )
            all_stats = self.accelerator.gather(local_stats).reshape(-1, 2)
            global_loss_sum = all_stats[:, 0].sum()
            global_eval_count = all_stats[:, 1].sum()
            if global_eval_count.item() == 0:
                raise ValueError("Evaluation dataloader produced no batches")
            final_eval_loss = (global_loss_sum / global_eval_count).item()

            eval_metrics = {"eval/loss": final_eval_loss}
            local_error = None
            try:
                self.accelerator.log(eval_metrics, step=self.global_step)
                logger.info(f"Eval Loss: {final_eval_loss:.4f}")
            except BaseException as exc:  # noqa: BLE001
                process_index = getattr(self.accelerator, "process_index", 0)
                local_error = (
                    f"process {process_index} {type(exc).__name__}: {exc}"
                )
            _raise_if_rank_failed(
                self.accelerator, "Evaluation logging", local_error
            )
            self.accelerator.wait_for_everyone()
            return eval_metrics
        finally:
            self.model.train()

    def train(self):
        """Run one bounded training invocation and always tear down logging."""
        logger.info("Starting Training Loop...")
        train_logger = TrainLogger(
            self.accelerator, self.config.steps, self.config.logging_steps
        )
        try:
            return self._train(train_logger)
        finally:
            try:
                train_logger.close()
            finally:
                self.accelerator.end_training()

    def _train(self, train_logger):
        invocation_start_time = time.monotonic()
        invocation_stop = self.config.stop_after_step or self.config.steps
        stop_policy = EvaluationStopPolicy(
            threshold=self.config.early_stop_eval_loss,
            patience=self.config.early_stop_patience,
            wall_limit_seconds=self.config.max_wall_clock_seconds,
        )
        last_eval_loss = None
        last_evaluated_step = None
        stop_reason = None

        # Resume if configured
        if self.config.resume_from_checkpoint:
            self.load_checkpoint(self.config.resume_from_checkpoint)
        if self.global_step > invocation_stop:
            raise ValueError(
                f"Resumed step {self.global_step} exceeds stop_after_step "
                f"{invocation_stop}"
            )
        if self.config.steps_per_epoch is not None:
            self.epoch = self.global_step // self.config.steps_per_epoch

        # Handle IterableDataset Epochs
        if hasattr(self.train_dataloader.dataset, "set_epoch"):
            self.train_dataloader.dataset.set_epoch(self.epoch)

        train_logger.start(self.global_step)

        self.model.train()
        optimizer_steps_into_epoch = (
            self.global_step % self.config.steps_per_epoch
            if self.config.resume_from_checkpoint
            and self.config.steps_per_epoch is not None
            else 0
        )
        batches_to_skip = (
            optimizer_steps_into_epoch
            * self.config.gradient_accumulation_steps
        )
        if batches_to_skip > 0:
            restored_rng_state = _capture_host_rng_state()
            try:
                train_iterator = iter(self.train_dataloader)
                for _ in range(batches_to_skip):
                    next(train_iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    "steps_per_epoch exceeds the available resumed epoch data"
                ) from exc
            finally:
                _restore_host_rng_state(restored_rng_state)
        else:
            train_iterator = iter(self.train_dataloader)

        logging_start_time = time.time()
        logging_start_step = self.global_step
        tr_loss = torch.tensor(0.0).to(self.accelerator.device)
        logging_loss_scalar = 0.0

        def evaluate_and_decide():
            nonlocal last_eval_loss, last_evaluated_step
            metrics = self.evaluate()
            last_evaluated_step = self.global_step
            last_eval_loss = float(metrics["eval/loss"])
            elapsed_seconds = time.monotonic() - invocation_start_time

            payload = [None]
            if self.accelerator.is_main_process:
                try:
                    if self.config.eval_history_path is not None:
                        append_loss_history(
                            self.config.eval_history_path,
                            step=self.global_step,
                            loss=last_eval_loss,
                            elapsed_seconds=elapsed_seconds,
                        )
                    decision = stop_policy.observe(
                        self.global_step, last_eval_loss, elapsed_seconds
                    )
                    payload[0] = (decision, None)
                except BaseException as exc:  # noqa: BLE001
                    payload[0] = (
                        StopDecision(False, None, stop_policy.consecutive_hits),
                        f"{type(exc).__name__}: {exc}",
                    )

            if self.accelerator.num_processes > 1:
                payload = broadcast_object_list(payload)
            decision, history_error = payload[0]
            if history_error is not None:
                raise RuntimeError(
                    f"Failed to record evaluation history: {history_error}"
                )
            return decision

        while self.global_step < invocation_stop:
            try:
                batch = next(train_iterator)
            except StopIteration:
                self.epoch += 1
                logger.info(f"Epoch {self.epoch} starting. Resetting dataloader...")
                if hasattr(self.train_dataloader.dataset, "set_epoch"):
                    self.train_dataloader.dataset.set_epoch(self.epoch)

                train_iterator = iter(self.train_dataloader)
                batch = next(train_iterator)

            batch = _to_device(batch, self.accelerator.device)

            with self.accelerator.accumulate(self.model):
                outputs = self.model(**batch)
                loss = outputs.loss
                tr_loss += loss.detach()
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    # Clipping
                    grad_norm = 0.0
                    if self.config.max_grad_norm > 0:
                        grad_norm = self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )
                        grad_norm = grad_norm.item() if grad_norm is not None else 0.0

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    # Logging
                    current_lr = self.lr_scheduler.get_last_lr()[0]
                    train_logger.update(
                        step=self.global_step, loss=loss.item(), lr=current_lr
                    )

                    if self.global_step % self.config.logging_steps == 0:
                        elapsed = time.time() - logging_start_time
                        steps_per_sec = (
                            (self.global_step - logging_start_step) / elapsed
                            if elapsed > 0
                            else 0
                        )

                        tr_loss_scalar = self.accelerator.gather(tr_loss).mean().item()
                        current_interval_loss = tr_loss_scalar - logging_loss_scalar
                        avg_loss = current_interval_loss / (
                            self.config.logging_steps
                            * self.config.gradient_accumulation_steps
                        )
                        logging_loss_scalar = tr_loss_scalar

                        logs = {
                            "train/loss": avg_loss,
                            "train/learning_rate": current_lr,
                            "train/grad_norm": grad_norm,
                            "train/epoch": self.epoch,
                            "train/steps_per_sec": steps_per_sec,
                        }
                        train_logger.log_metrics(step=self.global_step, metrics=logs)

                        logging_start_time = time.time()
                        logging_start_step = self.global_step

                    # Evaluate
                    if (
                        self.eval_dataloader is not None
                        and self.global_step % self.config.eval_steps == 0
                    ):
                        decision = evaluate_and_decide()
                        if decision.stop:
                            stop_reason = decision.reason
                            break

                    if self.global_step >= invocation_stop:
                        if (
                            self.config.stop_after_step is not None
                            and self.eval_dataloader is not None
                            and last_evaluated_step != self.global_step
                        ):
                            decision = evaluate_and_decide()
                            if decision.stop:
                                stop_reason = decision.reason
                        if stop_reason is None:
                            stop_reason = (
                                "stop_after_step"
                                if self.config.stop_after_step is not None
                                else "completed"
                            )
                        break

                    # Save
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint(self.global_step)

        if stop_reason is None:
            if (
                self.config.stop_after_step is not None
                and self.eval_dataloader is not None
                and last_evaluated_step != self.global_step
            ):
                decision = evaluate_and_decide()
                if decision.stop:
                    stop_reason = decision.reason
            if stop_reason is None:
                stop_reason = (
                    "stop_after_step"
                    if self.config.stop_after_step is not None
                    else "completed"
                )

        # Final Save
        self.save_checkpoint(self.global_step)
        return TrainingOutcome(
            step=self.global_step,
            stop_reason=stop_reason,
            last_eval_loss=last_eval_loss,
            target_reached=stop_reason == "eval_loss_target",
        )
