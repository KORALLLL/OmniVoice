"""Stable online Weights & Biases state and validation logging."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from omnivoice.validation.hard_numbers import HARD_NUMBER_COUNT
from omnivoice.validation.reporting import ValidationResult

try:
    import wandb as _installed_wandb
except ModuleNotFoundError:
    _installed_wandb = None

wandb: Any = _installed_wandb


def _client() -> Any:
    if wandb is None:
        raise RuntimeError(
            "W&B support is not installed; install the project validation extra"
        )
    return wandb


class WandbRunStore:
    """Atomically persist one resumable W&B run and four fixed example IDs."""

    def __init__(
        self, path: Path, project: str = "omnivoice-lora-validation"
    ) -> None:
        self.path = Path(path)
        self.project = project

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError("W&B ID store must contain a JSON object")
        return value

    def _write(self, state: Mapping[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
                json.dump(
                    state,
                    destination,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                destination.write("\n")
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary, self.path)
            directory_descriptor = os.open(
                self.path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            temporary.unlink(missing_ok=True)

    def preflight(self) -> None:
        """Require an online API key before the controller launches GPU work."""
        api = _client().Api(timeout=15)
        if not api.api_key:
            raise RuntimeError("W&B online authentication is required; run `wandb login`")

    def load_or_create_id(self) -> str:
        state = self._load()
        run_id = state.get("run_id")
        if run_id is None:
            run_id = uuid.uuid4().hex
            state["run_id"] = run_id
            self._write(state)
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("persisted W&B run_id must be a non-blank string")
        return run_id

    def load_or_create_audio_ids(self, candidates: Sequence[str]) -> list[str]:
        state = self._load()
        audio_ids = state.get("audio_ids")
        if audio_ids is None:
            values = list(candidates)
            if (
                len(values) != 4
                or len(set(values)) != 4
                or not all(isinstance(value, str) and value.strip() for value in values)
            ):
                raise ValueError("exactly four unique non-blank audio IDs are required")
            audio_ids = values
            state["audio_ids"] = audio_ids
            self._write(state)
        if (
            not isinstance(audio_ids, list)
            or len(audio_ids) != 4
            or len(set(audio_ids)) != 4
            or not all(isinstance(value, str) and value.strip() for value in audio_ids)
        ):
            raise ValueError("persisted audio_ids must contain four unique strings")
        return list(audio_ids)

    def init(self, config: Mapping[str, object]) -> Any:
        """Initialize or resume the stable online run."""
        run_id = self.load_or_create_id()
        return _client().init(
            project=self.project,
            id=run_id,
            resume="allow",
            config=dict(config),
        )


def _category_table(result: ValidationResult) -> Any:
    columns = [
        "category", "utterances", "utt_wer", "utt_cer", "num_wer", "num_cer",
    ]
    data = [
        [
            category,
            block["utterances"],
            block["utt_wer"],
            block["utt_cer"],
            block["num_wer"],
            block["num_cer"],
        ]
        for category, block in result.categories.items()
    ]
    return _client().Table(columns=columns, data=data)


def log_validation(
    run: Any,
    result: ValidationResult,
    fixed_audio_records: Sequence[Mapping[str, Any]],
    *,
    step: int,
    steps_per_epoch: int,
    dev_loss: float | None = None,
) -> None:
    """Log one complete validation at its optimizer step with four audio files."""
    if result.coverage != HARD_NUMBER_COUNT:
        raise ValueError(f"W&B logging requires exactly {HARD_NUMBER_COUNT:,}/2,000 coverage")
    if type(step) is not int or step < 0:
        raise ValueError("step must be a non-negative optimizer step")
    if type(steps_per_epoch) is not int or steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be a positive integer")
    audio_records = [dict(record) for record in fixed_audio_records]
    identifiers = [record.get("id") for record in audio_records]
    if (
        len(audio_records) != 4
        or len(set(identifiers)) != 4
        or not all(isinstance(identifier, str) and identifier.strip() for identifier in identifiers)
        or not all(isinstance(record.get("wav"), str) and record["wav"] for record in audio_records)
    ):
        raise ValueError("exactly four unique audio records with WAV paths are required")

    metrics: dict[str, object] = {
        "validation/asr_failures": result.failures["asr"],
        "validation/asr_utterances_per_second": result.throughput["asr_utterances_per_second"],
        "validation/category_metrics": _category_table(result),
        "validation/coverage": result.coverage,
        "validation/fractional_epoch": step / steps_per_epoch,
        "validation/num_cer": result.overall["num_cer"],
        "validation/num_wer": result.overall["num_wer"],
        "validation/optimizer_step": step,
        "validation/synthesis_failures": result.failures["synthesis"],
        "validation/synthesis_utterances_per_second": result.throughput["synthesis_utterances_per_second"],
        "validation/utt_cer": result.overall["utt_cer"],
        "validation/utt_wer": result.overall["utt_wer"],
        "validation/wall_time_seconds": result.wall_time_seconds,
    }
    if dev_loss is not None:
        metrics["validation/dev_loss"] = dev_loss
    for scope, prefix in (("utterance", "utterance"), ("number", "number")):
        for unit in ("words", "chars"):
            for field, value in result.overall[f"{scope}_{unit}"].items():
                metrics[f"validation/{prefix}_{unit}_{field}"] = value
    for record in audio_records:
        identifier = record["id"]
        caption_parts = [identifier]
        if isinstance(record.get("voice_id"), str):
            caption_parts.append(record["voice_id"])
        if isinstance(record.get("text"), str):
            caption_parts.append(record["text"])
        metrics[f"validation/audio/{identifier}"] = _client().Audio(
            record["wav"], caption=" | ".join(caption_parts)
        )
    run.log(metrics, step=step)
