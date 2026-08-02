"""Evaluation-boundary stopping controls and training results."""

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StopDecision:
    """A rank-consistent decision made after an evaluation."""

    stop: bool
    reason: str | None
    consecutive_hits: int


@dataclass(frozen=True)
class TrainingOutcome:
    """Summary returned by one bounded trainer invocation."""

    step: int
    stop_reason: str
    last_eval_loss: float | None
    target_reached: bool


class EvaluationStopPolicy:
    """Track evaluation-loss patience and an evaluation-boundary wall limit."""

    def __init__(
        self,
        threshold: float | None,
        patience: int,
        wall_limit_seconds: float | None,
    ) -> None:
        if threshold is not None and threshold <= 0:
            raise ValueError("threshold must be positive")
        if patience <= 0:
            raise ValueError("patience must be positive")
        if wall_limit_seconds is not None and wall_limit_seconds <= 0:
            raise ValueError("wall_limit_seconds must be positive")
        self.threshold = threshold
        self.patience = patience
        self.wall_limit_seconds = wall_limit_seconds
        self.consecutive_hits = 0

    def observe(
        self, step: int, loss: float, elapsed_seconds: float
    ) -> StopDecision:
        """Observe one completed evaluation and return its stop decision."""
        del step
        if self.threshold is not None and loss <= self.threshold:
            self.consecutive_hits += 1
        else:
            self.consecutive_hits = 0

        if self.consecutive_hits >= self.patience:
            return StopDecision(True, "eval_loss_target", self.consecutive_hits)
        if (
            self.wall_limit_seconds is not None
            and elapsed_seconds >= self.wall_limit_seconds
        ):
            return StopDecision(True, "wall_clock_limit", self.consecutive_hits)
        return StopDecision(False, None, self.consecutive_hits)


def append_loss_history(
    path: str | os.PathLike[str],
    *,
    step: int,
    loss: float,
    elapsed_seconds: float,
) -> None:
    """Atomically append one evaluation record to a JSONL history file."""
    history_path = Path(path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    existing = history_path.read_bytes() if history_path.exists() else b""
    entry = json.dumps(
        {
            "step": step,
            "loss": loss,
            "elapsed_seconds": elapsed_seconds,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    prefix = existing
    if prefix and not prefix.endswith(b"\n"):
        prefix += b"\n"

    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=history_path.parent,
        prefix=f".{history_path.name}.",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as temporary_file:
            temporary_file.write(prefix)
            temporary_file.write(entry)
            temporary_file.write(b"\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, history_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
