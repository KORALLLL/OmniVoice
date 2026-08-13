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

"""Training loop for OmniVoice.

Wraps the HuggingFace Accelerate training loop with checkpoint saving/resuming,
evaluation, gradient accumulation, and learning rate scheduling.
Launched via ``omnivoice.cli.train``.
"""

import logging
import math
import os
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import timedelta
from typing import Any, Optional

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import DeepSpeedPlugin, InitProcessGroupKwargs, set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)

from omnivoice.training.checkpoint import (
    TrainLogger,
    load_checkpoint,
    save_lora_audio_modules,
    save_lora_trainable_modules,
)
from omnivoice.training.checkpoint import save_checkpoint as engine_save_checkpoint

logger = logging.getLogger(__name__)


def _to_device(batch, device):
    """Move all tensors in a batch dict to the target device."""
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


class OmniTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        config: Any,  # TrainingConfig
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        tokenizer: Optional[Any] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # 1. Initialize Accelerator
        self.accelerator = self._init_accelerator()

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

    def _epoch_sample_total(self) -> Optional[int]:
        """Return the global source-sample count when the dataset exposes it."""
        dataset = getattr(self.train_dataloader, "dataset", None)
        raw_dataset = getattr(dataset, "dataset", None)
        sample_count = getattr(raw_dataset, "num_items", None)
        return sample_count if isinstance(sample_count, int) and sample_count > 0 else None

    def _batch_sample_count(self, batch: dict[str, Any]) -> int:
        """Count original samples represented by one collated local batch."""
        document_ids = batch.get("document_ids")
        if isinstance(document_ids, torch.Tensor):
            maximum = int(document_ids.max().item())
            return maximum + 1 if maximum >= 0 else 0
        input_ids = batch.get("input_ids")
        # The non-packed path has one sample per batch row.
        return int(input_ids.shape[0]) if isinstance(input_ids, torch.Tensor) else 0

    def _next_synchronized_batch(self, iterator):
        """Fetch one batch, ending the epoch on every rank at the same time.

        Length-grouped streaming batches do not necessarily yield an identical
        number of batches per DDP rank.  Letting an exhausted rank enter epoch
        handling while another rank still calls backward deadlocks DDP and can
        also make the ranks take different validation schedules.  We therefore
        collectively stop at the first exhausted rank; a batch fetched by a
        longer rank for that final probe is intentionally discarded.
        """
        try:
            batch = next(iterator)
            has_batch = 1
        except StopIteration:
            batch = None
            has_batch = 0
        common_has_batch = self.accelerator.reduce(
            torch.tensor(has_batch, device=self.accelerator.device), reduction="min"
        )
        return batch if int(common_has_batch.item()) else None

    def _assert_rank_bool_consensus(self, value: bool, name: str) -> None:
        """Fail deterministically if a control-flow branch differs by rank.

        A branch containing checkpointing or generation validation must be
        entered by every DDP rank.  Without this check, one rank can start a
        barrier while another rank starts a different collective, producing an
        opaque NCCL timeout many hours later.
        """
        local = torch.tensor(int(value), device=self.accelerator.device)
        minimum = self.accelerator.reduce(local, reduction="min")
        maximum = self.accelerator.reduce(local, reduction="max")
        if int(minimum.item()) != int(maximum.item()):
            raise RuntimeError(
                f"DDP rank disagreement for {name}: validation/checkpoint "
                "control flow must be identical on every rank"
            )

    def _seek_resumed_iterator(self, iterator, microbatches: int) -> None:
        """Advance a restored iterable stream without one collective per batch.

        ``_epoch_microbatches`` is written only after a batch has passed the
        normal cross-rank availability check.  Every rank must therefore have
        exactly that many deterministic batches on restore.  Calling
        ``_next_synchronized_batch`` for every skipped item needlessly issues
        thousands of NCCL reductions before the first optimizer update, which
        can make a restart slower than the training itself.  Seek locally, then
        perform one collective integrity check before allowing gradients.
        """
        local_complete = 1
        for _ in range(microbatches):
            try:
                next(iterator)
            except StopIteration:
                local_complete = 0
                break
        complete_on_all_ranks = self.accelerator.reduce(
            torch.tensor(local_complete, device=self.accelerator.device),
            reduction="min",
        )
        if not int(complete_on_all_ranks.item()):
            raise RuntimeError(
                "Saved iterable-dataset position exceeds the restored epoch "
                "on at least one DDP rank"
            )

    def _init_accelerator(self) -> Accelerator:
        """Initialize Accelerator, DeepSpeed, and Logging."""
        # TF32 setup
        if getattr(self.config, "allow_tf32", False):
            torch.set_float32_matmul_precision("high")

        # Init handlers
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        init_kwargs = InitProcessGroupKwargs(
            timeout=timedelta(
                minutes=max(1, int(getattr(self.config, "distributed_timeout_minutes", 480)))
            )
        )

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

        log_with = ["tensorboard"]
        if getattr(self.config, "wandb_project", None):
            log_with.append("wandb")

        accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            mixed_precision=self.config.mixed_precision,
            log_with=log_with,
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
        tracker_config = (
            asdict(self.config) if is_dataclass(self.config) else vars(self.config)
        )
        if getattr(self.config, "wandb_project", None):
            wandb_kwargs = {"mode": self.config.wandb_mode}
            if self.config.wandb_entity:
                wandb_kwargs["entity"] = self.config.wandb_entity
            if self.config.wandb_run_name:
                wandb_kwargs["name"] = self.config.wandb_run_name
            accelerator.init_trackers(
                self.config.wandb_project,
                config=tracker_config,
                init_kwargs={"wandb": wandb_kwargs},
            )
        else:
            accelerator.init_trackers("tensorboard", config=tracker_config)
        return accelerator

    def create_optimizer_and_scheduler(self):
        """Default AdamW + configurable LR Scheduler."""
        optimizer = torch.optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
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
            self.config.output_dir,
            step,
            self.config.keep_last_n_checkpoints,
        )
        # Save config copy for convenience
        if self.accelerator.is_main_process and hasattr(self.config, "save_to_json"):
            checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
            self.config.save_to_json(os.path.join(checkpoint_dir, "train_config.json"))
            # Accelerate restores model/optimizer/RNG state, but it deliberately
            # does not know where an IterableDataset iterator was.  Persist that
            # position here so a resume neither repeats the beginning of an epoch
            # nor silently turns a requested N-epoch run into more epochs.
            with open(os.path.join(checkpoint_dir, "trainer_state.json"), "w") as handle:
                json.dump(
                    {
                        "format_version": 1,
                        "global_step": int(self.global_step),
                        "epoch": int(self.epoch),
                        "epoch_microbatches": int(getattr(self, "_epoch_microbatches", 0)),
                        "epoch_steps": int(getattr(self, "_epoch_step", 0)),
                        "epoch_samples": int(getattr(self, "_epoch_samples", 0)),
                        "evals_this_epoch": int(getattr(self, "_evals_this_epoch", 0)),
                    },
                    handle,
                )
            model = self.accelerator.unwrap_model(self.model)
            if hasattr(getattr(model, "llm", None), "peft_config"):
                adapter_dir = os.path.join(checkpoint_dir, "lora_adapter")
                model.llm.save_pretrained(adapter_dir, safe_serialization=True)
                save_lora_audio_modules(model, adapter_dir, base_omnivoice_checkpoint=self.config.init_from_checkpoint)
                save_lora_trainable_modules(model, adapter_dir)
        # A lightweight LoRA artifact is written only by rank 0.  Do not allow
        # peers to start the next collective (or external validation) until the
        # complete artifact is visible to every rank.
        self.accelerator.wait_for_everyone()

    def run_hard_number_generation_eval(self, step):
        if not getattr(self.config, "hard_number_eval_enabled", False):
            return
        checkpoint = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        output = os.path.join(checkpoint, "hard_number_eval")

        # The full hard-number scorer deliberately reloads the checkpoint in a
        # separate process so generation and GigaAM are isolated from the
        # training graph.  Keeping a DDP training replica resident on GPU 0 at
        # the same time, however, leaves too little memory for that second
        # model.  Offload every rank (including optimizer moments and stale
        # gradients) while the external process owns the GPUs, then restore the
        # identical training state afterwards.  This keeps the required full
        # 2,000-utterance validation rather than silently shrinking it.
        logger.info("Hard-number evaluation: offloading training state on all ranks")
        self._offload_training_state_for_external_eval()
        self.accelerator.wait_for_everyone()
        failure_path = os.path.join(output, "FAILED")
        try:
            if self.accelerator.is_main_process:
                os.makedirs(output, exist_ok=True)
                if os.path.exists(failure_path):
                    os.unlink(failure_path)
                samples_per_voice = getattr(self.config, "hard_number_eval_samples_per_voice", 50)
                command = [sys.executable, "-m", "omnivoice.scripts.hard_number_voice_generation_eval", "--checkpoint", checkpoint, "--dataset-path", self.config.hard_number_eval_dataset_path, "--voice-manifest-path", self.config.hard_number_eval_voice_manifest_path, "--output-dir", output, "--selection-path", os.path.join(self.config.output_dir, "hard_number_voice_selection_omnivoice8.json"), "--num-voices", str(self.config.hard_number_eval_voice_count), "--selection-seed", str(self.config.hard_number_eval_seed), "--batch-size", str(self.config.hard_number_eval_batch_size), "--samples-per-voice", str(samples_per_voice), "--gigaam-model", self.config.hard_number_eval_model_name]
                # Bound the periodic sweep.  The full ~2,000-pair validation
                # costs 40-80 minutes, and every non-zero rank sits in an NCCL
                # barrier for its duration; four of those per epoch dominate
                # the run and widen the window for a collective timeout.
                max_pairs = getattr(self.config, "hard_number_eval_max_pairs", 0)
                if max_pairs:
                    command += ["--max-pairs", str(max_pairs)]
                excluded = getattr(self.config, "hard_number_eval_excluded_speaker_keys_path", None)
                if excluded:
                    command += ["--excluded-speaker-keys-path", excluded]
                done = subprocess.run(command, text=True, capture_output=True)
                with open(os.path.join(output, "runner.log"), "w") as handle:
                    handle.write(done.stdout + done.stderr)
                if done.returncode:
                    with open(failure_path, "w") as handle:
                        handle.write(f"exit code {done.returncode}; see runner.log\n")
                else:
                    metrics = json.load(open(os.path.join(output, "metrics.json")))
                    self.accelerator.log({f"hard_number/{k}": float(v) for k,v in metrics.items() if isinstance(v, (int,float))}, step=step)
                    # Metrics are written *inside* the checkpoint directory, so
                    # keep_last_n_checkpoints deletes the eval history along
                    # with the weights and the run loses its own trend line.
                    # Mirror them out to a directory the pruner never touches.
                    history = os.path.join(self.config.output_dir, "validation_history", f"step-{step}")
                    os.makedirs(history, exist_ok=True)
                    for name in ("metrics.json", "per_utt.jsonl"):
                        source = os.path.join(output, name)
                        if os.path.isfile(source):
                            shutil.copy2(source, os.path.join(history, name))
        except Exception as error:
            if self.accelerator.is_main_process:
                os.makedirs(output, exist_ok=True)
                with open(failure_path, "w") as handle:
                    handle.write(f"{type(error).__name__}: {error}\n")
            logger.exception("Hard-number evaluation failed before completion")
        finally:
            # Keep *every* DDP replica on CPU while rank 0 owns GPU 0 for
            # generation/GigaAM.  The rendezvous happens before restoration,
            # so no rank can resume collectives or allocate model memory early.
            self.accelerator.wait_for_everyone()
            logger.info("Hard-number evaluation: restoring training state on all ranks")
            self._restore_training_state_after_external_eval()
        self.accelerator.wait_for_everyone()
        if os.path.isfile(failure_path):
            raise RuntimeError(f"Hard-number validation failed; see {output}/runner.log")

    def _move_optimizer_state(self, device):
        """Move Adam state tensors without replacing the optimizer's state map."""
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device, non_blocking=device.type == "cuda")

    def _offload_training_state_for_external_eval(self):
        """Free GPU memory on every DDP rank for an external full generator."""
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.grad = None
        self.model.to("cpu")
        self._move_optimizer_state(torch.device("cpu"))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _restore_training_state_after_external_eval(self):
        """Return a temporary CPU-offloaded DDP replica to its original GPU."""
        device = self.accelerator.device
        self.model.to(device)
        self._move_optimizer_state(device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model.train()

    def load_checkpoint(self, checkpoint_path):
        """Wrapper for loading."""
        step = load_checkpoint(self.accelerator, checkpoint_path)
        self.global_step = step
        state_path = os.path.join(checkpoint_path, "trainer_state.json")
        if os.path.isfile(state_path):
            with open(state_path) as handle:
                state = json.load(handle)
            self.epoch = int(state.get("epoch", 0))
            self._epoch_microbatches = int(state.get("epoch_microbatches", 0))
            self._epoch_step = int(state.get("epoch_steps", 0))
            self._epoch_samples = int(state.get("epoch_samples", 0))
            self._evals_this_epoch = int(state.get("evals_this_epoch", 0))
            logger.info(
                "Restored epoch progress: epoch=%d microbatches=%d updates=%d samples=%d evals=%d",
                self.epoch,
                self._epoch_microbatches,
                self._epoch_step,
                self._epoch_samples,
                self._evals_this_epoch,
            )
        else:
            logger.warning(
                "Checkpoint has no trainer_state.json; exact iterable-dataset resume is unavailable. "
                "Start a fresh epoch-boundary run instead of continuing this checkpoint."
            )
        logger.info(f"Resumed from step {self.global_step}")
        return step

    def evaluate(self):
        """Evaluation loop."""
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        logger.info(f"Running evaluation at step {self.global_step}...")

        local_loss_sum = torch.tensor(0.0, device=self.accelerator.device)
        eval_count = 0

        try:
            eval_total = len(self.eval_dataloader)
        except TypeError:
            eval_total = None

        eval_iterator = tqdm(
            self.eval_dataloader,
            total=eval_total,
            desc="Validation",
            dynamic_ncols=True,
            disable=not self.accelerator.is_local_main_process,
            leave=False,
        )
        with torch.no_grad():
            for eval_batch in eval_iterator:
                eval_batch = _to_device(eval_batch, self.accelerator.device)
                outputs = self.model(**eval_batch)
                local_loss_sum += outputs.loss.detach()
                eval_count += 1

        eval_iterator.close()

        if eval_count > 0:
            local_mean = local_loss_sum / eval_count
        else:
            local_mean = torch.tensor(0.0, device=self.accelerator.device)

        all_means = self.accelerator.gather(local_mean)
        final_eval_loss = all_means.mean().item()

        eval_metrics = {"eval/loss": final_eval_loss}
        self.accelerator.log(eval_metrics, step=self.global_step)
        logger.info(f"Eval Loss: {final_eval_loss:.4f}")

        self.accelerator.wait_for_everyone()
        self.model.train()
        return eval_metrics

    def train(self):
        """Main training loop."""
        logger.info("Starting Training Loop...")

        # Resume if configured
        checkpoint_model_state = (
            os.path.join(self.config.resume_from_checkpoint, "model.safetensors")
            if self.config.resume_from_checkpoint
            else None
        )
        has_full_engine_state = bool(
            checkpoint_model_state
            and os.path.isfile(checkpoint_model_state)
            and os.path.getsize(checkpoint_model_state) > 0
        )
        if (
            self.config.resume_from_checkpoint
            and not getattr(self.config, "resume_weights_only", False)
            and has_full_engine_state
        ):
            self.load_checkpoint(self.config.resume_from_checkpoint)
        elif self.config.resume_from_checkpoint:
            # The builder has already restored the adapter and optional audio
            # sidecar.  Do not load Accelerate state here: its optimizer and
            # scheduler parameter groups can legitimately differ from this
            # targeted continuation.  A lightweight checkpoint deliberately
            # has no optimizer state, but it can still continue at the saved
            # global step and deterministic iterable position (especially
            # important for constant-LR LoRA recovery after an external eval).
            state_path = os.path.join(
                self.config.resume_from_checkpoint, "trainer_state.json"
            )
            if (
                not getattr(self.config, "resume_weights_only", False)
                and os.path.isfile(state_path)
            ):
                with open(state_path) as handle:
                    state = json.load(handle)
                self.global_step = int(state.get("global_step", 0))
                self.epoch = int(state.get("epoch", 0))
                self._epoch_microbatches = int(state.get("epoch_microbatches", 0))
                self._epoch_step = int(state.get("epoch_steps", 0))
                self._epoch_samples = int(state.get("epoch_samples", 0))
                self._evals_this_epoch = int(state.get("evals_this_epoch", 0))
                logger.info(
                    "Restored lightweight LoRA position from %s at step %d with a fresh optimizer/scheduler",
                    self.config.resume_from_checkpoint,
                    self.global_step,
                )
            else:
                logger.info(
                    "Initialized adapter weights from %s with fresh optimizer, scheduler, and data iterator",
                    self.config.resume_from_checkpoint,
                )

        # Handle IterableDataset Epochs
        if hasattr(self.train_dataloader.dataset, "set_epoch"):
            self.train_dataloader.dataset.set_epoch(self.epoch)

        # Logger
        train_logger = TrainLogger(
            self.accelerator, self.config.steps, self.config.logging_steps
        )
        train_logger.start(self.global_step)
        try:
            epoch_total = len(self.train_dataloader)
        except TypeError:
            epoch_total = None
        train_logger.start_epoch(self.epoch, epoch_total)

        self.model.train()
        train_iterator = iter(self.train_dataloader)

        logging_start_time = time.time()
        logging_start_step = self.global_step
        tr_loss = torch.tensor(0.0).to(self.accelerator.device)
        logging_loss_scalar = 0.0

        # These counters are checkpointed because the training dataset is an
        # IterableDataset.  Recreating its iterator without skipping the
        # consumed microbatches repeats data after every restart.
        epoch_step = int(getattr(self, "_epoch_step", 0))
        epoch_samples = int(getattr(self, "_epoch_samples", 0))
        evals_this_epoch = int(getattr(self, "_evals_this_epoch", 0))
        epoch_microbatches = int(getattr(self, "_epoch_microbatches", 0))
        fourth_epoch_tail = False
        epoch_sample_total = self._epoch_sample_total()
        sample_eval_targets = []
        if getattr(self.config, "evals_per_epoch", 0) > 0 and epoch_sample_total:
            sample_eval_targets = [
                math.ceil(epoch_sample_total * index / self.config.evals_per_epoch)
                for index in range(1, self.config.evals_per_epoch + 1)
            ]
        epoch_eval_interval = 0
        if (
            getattr(self.config, "evals_per_epoch", 0) > 0
            and not sample_eval_targets
            and getattr(self.config, "estimated_steps_per_epoch", 0) > 0
        ):
            epoch_eval_interval = max(
                1,
                math.ceil(
                    self.config.estimated_steps_per_epoch
                    / self.config.evals_per_epoch
                ),
            )

        # Seek each rank through its deterministic stream before its first
        # resumed batch.  This is intentionally done before the training loop
        # so it cannot contribute gradients or epoch sample accounting.
        if epoch_microbatches:
            logger.info(
                "Seeking resumed IterableDataset to microbatch %d of epoch %d",
                epoch_microbatches,
                self.epoch,
            )
            self._seek_resumed_iterator(train_iterator, epoch_microbatches)

        while self.global_step < self.config.steps:
            batch = self._next_synchronized_batch(train_iterator)
            if batch is None:
                # Make sure the end of every completed epoch receives a full
                # validation pass when using an epoch-relative schedule.
                if (
                    getattr(self.config, "evals_per_epoch", 0) > 0
                    and evals_this_epoch < self.config.evals_per_epoch
                ):
                    if getattr(self.config, "save_before_evaluation", False):
                        self.save_checkpoint(self.global_step)
                    if self.eval_dataloader is not None:
                        self.evaluate()
                    self.run_hard_number_generation_eval(self.global_step)
                train_logger.close_epoch()
                self.epoch += 1
                if (
                    getattr(self.config, "max_epochs", None) is not None
                    and self.epoch >= self.config.max_epochs
                ):
                    # The requested scheduler horizon can be a small number of
                    # updates longer than four full iterable-dataset passes.
                    # Keep the fourth epoch deterministic and open only that
                    # short tail; do not begin a fifth epoch or schedule an
                    # extra validation.
                    if self.global_step >= self.config.steps:
                        logger.info("Reached configured max_epochs=%s", self.config.max_epochs)
                        break
                    self.epoch = self.config.max_epochs - 1
                    fourth_epoch_tail = True
                    logger.info(
                        "Completed four source epochs at step %d; continuing deterministic fourth-epoch tail to step %d",
                        self.global_step,
                        self.config.steps,
                    )
                logger.info(f"Epoch {self.epoch} starting. Resetting dataloader...")
                if hasattr(self.train_dataloader.dataset, "set_epoch"):
                    self.train_dataloader.dataset.set_epoch(self.epoch)

                train_iterator = iter(self.train_dataloader)
                train_logger.start_epoch(self.epoch, epoch_total)
                epoch_step = 0
                epoch_samples = 0
                evals_this_epoch = (
                    getattr(self.config, "evals_per_epoch", 0)
                    if fourth_epoch_tail
                    else 0
                )
                epoch_microbatches = 0
                self._epoch_step = 0
                self._epoch_samples = 0
                self._evals_this_epoch = evals_this_epoch
                self._epoch_microbatches = 0
                # All ranks start the next epoch together.  A completely empty
                # epoch is an invalid dataset/configuration rather than a DDP
                # synchronization condition to spin through.
                batch = self._next_synchronized_batch(train_iterator)
                if batch is None:
                    raise RuntimeError("No synchronized training batches were produced for the epoch")

            batch = _to_device(batch, self.accelerator.device)
            epoch_microbatches += 1
            self._epoch_microbatches = epoch_microbatches
            # Count every accumulation microbatch for epoch-relative validation.
            local_samples = self._batch_sample_count(batch)
            global_samples = self.accelerator.reduce(
                torch.tensor(local_samples, device=self.accelerator.device), reduction="sum"
            )
            epoch_samples += int(global_samples.item())
            self._epoch_samples = epoch_samples

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
                    epoch_step += 1
                    self._epoch_step = epoch_step

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
                    scheduled_global_eval = (
                        self.config.eval_steps > 0
                        and self.global_step % self.config.eval_steps == 0
                    )
                    scheduled_epoch_eval = (
                        evals_this_epoch < getattr(self.config, "evals_per_epoch", 0)
                        and (
                            (
                                bool(sample_eval_targets)
                                and epoch_samples
                                >= sample_eval_targets[evals_this_epoch]
                            )
                            or (
                                epoch_eval_interval > 0
                                and epoch_step
                                >= (evals_this_epoch + 1) * epoch_eval_interval
                            )
                        )
                    )
                    did_evaluate = scheduled_global_eval or scheduled_epoch_eval
                    self._assert_rank_bool_consensus(
                        did_evaluate,
                        f"evaluation scheduling at global step {self.global_step}",
                    )
                    if did_evaluate:
                        if getattr(self.config, "save_before_evaluation", False):
                            self.save_checkpoint(self.global_step)
                        if self.eval_dataloader is not None:
                            self.evaluate()
                        self.run_hard_number_generation_eval(self.global_step)
                        if scheduled_epoch_eval:
                            evals_this_epoch += 1
                            self._evals_this_epoch = evals_this_epoch

                    # Save
                    if did_evaluate and getattr(
                        self.config, "save_on_evaluation", False
                    ):
                        self.save_checkpoint(self.global_step)
                    elif (
                        self.config.save_steps > 0
                        and self.global_step % self.config.save_steps == 0
                    ):
                        self.save_checkpoint(self.global_step)

        # Final Save
        if getattr(self.config, "save_final_checkpoint", True):
            self.save_checkpoint(self.global_step)
        train_logger.close()
        self.accelerator.end_training()
