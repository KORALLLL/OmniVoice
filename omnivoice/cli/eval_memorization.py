#!/usr/bin/env python3
"""Evaluate a LoRA checkpoint against the deterministic memorization dev set."""

import argparse
import json
import random
from collections.abc import Iterable
from pathlib import Path

import torch

from omnivoice.training.builder import build_dataloaders, build_model_and_tokenizer
from omnivoice.training.config import TrainingConfig


def check_memorization(loss: float, threshold: float) -> int:
    """Return a process exit code for the configured memorization threshold."""
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    return 0 if loss <= threshold else 1


def mean_eval_loss(
    model: torch.nn.Module,
    batches: Iterable[dict],
    device: str | torch.device,
    dtype: torch.dtype,
) -> tuple[float, int]:
    """Evaluate and return the unweighted mean loss and batch count."""
    target_device = torch.device(device)
    use_autocast = dtype in (torch.float16, torch.bfloat16)
    loss_sum = 0.0
    num_batches = 0

    model.eval()
    with torch.inference_mode():
        for batch in batches:
            batch = {
                key: value.to(target_device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }
            with torch.autocast(
                device_type=target_device.type,
                dtype=dtype,
                enabled=use_autocast,
            ):
                outputs = model(**batch)
            loss_sum += outputs.loss.detach().float().item()
            num_batches += 1

    if num_batches == 0:
        raise ValueError("dev data is empty")
    return loss_sum / num_batches, num_batches


def _autocast_dtype(mixed_precision: str) -> torch.dtype:
    dtypes = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "no": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return dtypes[mixed_precision]
    except KeyError as error:
        raise ValueError(
            f"unsupported mixed_precision value: {mixed_precision!r}"
        ) from error


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a four-sample LoRA memorization checkpoint."
    )
    parser.add_argument("--adapter-checkpoint", required=True, type=Path)
    parser.add_argument("--train-config", required=True, type=Path)
    parser.add_argument("--data-config", required=True, type=Path)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.threshold <= 0:
        raise ValueError("threshold must be positive")

    config = TrainingConfig.from_json(args.train_config)
    config.resume_from_checkpoint = str(args.adapter_checkpoint)
    config.data_config = str(args.data_config)

    random.seed(config.seed)
    torch.manual_seed(config.seed)

    model, tokenizer = build_model_and_tokenizer(config)
    _, dev_loader = build_dataloaders(config, tokenizer)
    if dev_loader is None:
        raise ValueError("dev data is empty")

    device = torch.device(args.device)
    model = model.to(device)
    loss, num_batches = mean_eval_loss(
        model,
        dev_loader,
        device=device,
        dtype=_autocast_dtype(config.mixed_precision),
    )
    exit_code = check_memorization(loss, args.threshold)
    print(
        json.dumps(
            {
                "loss": loss,
                "threshold": args.threshold,
                "passed": exit_code == 0,
                "num_batches": num_batches,
            }
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
