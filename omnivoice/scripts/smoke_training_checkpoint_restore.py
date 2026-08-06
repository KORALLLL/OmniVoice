#!/usr/bin/env python3
"""Exercise training, full validation, checkpoint saving, and restoration."""

import argparse
import json
from pathlib import Path

import torch

from omnivoice.training.builder import build_dataloaders, build_model_and_tokenizer
from omnivoice.training.config import TrainingConfig
from omnivoice.training.trainer import OmniTrainer


def fingerprint(trainer: OmniTrainer) -> list[float]:
    """A compact, deterministic fingerprint of the unwrapped model weights."""
    with torch.no_grad():
        sums = torch.zeros(4, device=trainer.accelerator.device, dtype=torch.float64)
        for parameter in trainer.accelerator.unwrap_model(trainer.model).parameters():
            value = parameter.detach().to(dtype=torch.float64)
            sums[0] += value.sum()
            sums[1] += value.square().sum()
            sums[2] += value.abs().sum()
            sums[3] += value.reshape(-1)[0]
    return [float(x) for x in sums.cpu()]


def build_trainer(config: TrainingConfig) -> OmniTrainer:
    model, tokenizer = build_model_and_tokenizer(config)
    train_loader, eval_loader = build_dataloaders(config, tokenizer)
    return OmniTrainer(model, config, train_loader, eval_loader, tokenizer)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = TrainingConfig.from_json(args.train_config)
    config.output_dir = args.output_dir
    config.data_config = args.data_config
    trainer = build_trainer(config)
    trainer.train()  # Includes the configured full validation and checkpoint.

    checkpoint = Path(args.output_dir) / f"checkpoint-{config.steps}"
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Expected checkpoint was not saved: {checkpoint}")
    before = fingerprint(trainer)

    restored_config = TrainingConfig.from_json(args.train_config)
    restored_config.output_dir = str(Path(args.output_dir) / "restored")
    restored_config.data_config = args.data_config
    restored = build_trainer(restored_config)
    restored_step = restored.load_checkpoint(str(checkpoint))
    after = fingerprint(restored)
    max_abs_difference = max(abs(a - b) for a, b in zip(before, after))
    if restored_step != config.steps or max_abs_difference != 0.0:
        raise RuntimeError(
            f"Checkpoint restore failed: step={restored_step}, "
            f"max_fingerprint_difference={max_abs_difference}"
        )
    restored_eval = restored.evaluate()  # Full held-out validation after restore.
    if not torch.isfinite(torch.tensor(restored_eval["eval/loss"])):
        raise RuntimeError(f"Non-finite restored validation result: {restored_eval}")

    if restored.accelerator.is_main_process:
        marker = {
            "checkpoint": str(checkpoint),
            "restored_step": restored_step,
            "max_fingerprint_difference": max_abs_difference,
            "restored_eval": restored_eval,
        }
        Path(args.output_dir, "smoke_verification.json").write_text(
            json.dumps(marker, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(marker))
    restored.accelerator.end_training()


if __name__ == "__main__":
    main()
