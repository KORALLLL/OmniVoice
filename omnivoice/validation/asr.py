"""Distributed, resumable GigaAM transcription for hard-number validation."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import soxr

from omnivoice.validation.artifacts import AtomicJsonlLedger, ValidationPaths
from omnivoice.validation.hard_numbers import HARD_NUMBER_COUNT
from omnivoice.validation.synthesis import FULL_VALIDATION_WORLD_SIZE

ASR_SAMPLE_RATE = 16_000
RANK_TRANSCRIPTION_COUNT = HARD_NUMBER_COUNT // FULL_VALIDATION_WORLD_SIZE


@dataclass(frozen=True)
class AsrSummary:
    """Durable progress for one rank's GigaAM hypothesis ledger."""

    rank: int
    expected: int
    completed: int
    transcribed: int
    skipped: int
    failed: int
    complete: bool
    stop_reason: str | None


def load_gigaam(local_rank: int) -> Any:
    """Load GigaAM on one rank-bound CUDA execution provider."""
    if type(local_rank) is not int or local_rank < 0:
        raise ValueError("local_rank must be a non-negative integer")
    import onnx_asr

    providers = [
        ("CUDAExecutionProvider", {"device_id": local_rank}),
        "CPUExecutionProvider",
    ]
    model = onnx_asr.load_model("gigaam-v3-rnnt", providers=providers)
    active = model.providers if hasattr(model, "providers") else []
    if active and "CUDAExecutionProvider" not in active:
        raise RuntimeError("GigaAM did not activate CUDAExecutionProvider")
    return model


def _step_directory(output_dir: str | Path | ValidationPaths) -> Path:
    return output_dir.step_dir if isinstance(output_dir, ValidationPaths) else Path(output_dir)


def _rank_hypotheses_path(output_dir: str | Path | ValidationPaths, rank: int) -> Path:
    if isinstance(output_dir, ValidationPaths):
        return output_dir.rank_hypotheses(rank)
    return _step_directory(output_dir) / "rank-hypotheses" / f"rank-{rank}.jsonl"


def _validate_synthesis_records(
    synthesis_records: Sequence[Mapping[str, Any]], *, world_size: int
) -> list[Mapping[str, Any]]:
    if len(synthesis_records) != HARD_NUMBER_COUNT:
        raise ValueError(
            f"ASR requires exactly {HARD_NUMBER_COUNT} synthesis records; "
            f"got {len(synthesis_records)}"
        )
    ids: set[str] = set()
    rank_counts = [0] * world_size
    validated: list[Mapping[str, Any]] = []
    for index, record in enumerate(synthesis_records, start=1):
        if not isinstance(record, Mapping):
            raise TypeError(f"synthesis record {index} must be a mapping")
        identifier = record.get("id")
        rank = record.get("rank")
        wav = record.get("wav")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"synthesis record {index} id must be a non-blank string")
        if identifier in ids:
            raise ValueError(f"synthesis records contain duplicate id {identifier!r}")
        if type(rank) is not int or not 0 <= rank < world_size:
            raise ValueError(f"synthesis record {identifier!r} has an invalid rank")
        if not isinstance(wav, str) or not wav:
            raise ValueError(f"synthesis record {identifier!r} has no WAV path")
        if "error" in record:
            raise ValueError(f"synthesis record {identifier!r} contains an error")
        ids.add(identifier)
        rank_counts[rank] += 1
        validated.append(record)
    if rank_counts != [RANK_TRANSCRIPTION_COUNT] * world_size:
        raise ValueError("synthesis records must contain exactly 250 rows per rank")
    return validated


def _asr_audio(wav_path: str | Path) -> np.ndarray:
    waveform, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
    array = np.asarray(waveform)
    if array.ndim == 2:
        array = array.mean(axis=1)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("synthesis WAV must contain a nonempty mono or stereo waveform")
    if not np.isfinite(array).all():
        raise ValueError("synthesis WAV must contain only finite samples")
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("synthesis WAV sample rate must be a positive integer")
    if sample_rate != ASR_SAMPLE_RATE:
        array = soxr.resample(array, sample_rate, ASR_SAMPLE_RATE)
    return np.asarray(array, dtype=np.float32)


def _hypothesis(result: Any) -> str:
    if isinstance(result, str):
        return result
    if (
        isinstance(result, Sequence)
        and not isinstance(result, (bytes, bytearray))
        and len(result) == 1
        and isinstance(result[0], str)
    ):
        return result[0]
    raise ValueError("GigaAM recognize must return a string or singleton list of strings")


def _error_record(identifier: str, rank: int, error: BaseException) -> dict[str, Any]:
    return {
        "error": {"message": str(error), "type": type(error).__name__},
        "id": identifier,
        "rank": rank,
    }


def persist_rank_failure(
    *,
    synthesis_records: Sequence[Mapping[str, Any]],
    output_dir: str | Path | ValidationPaths,
    rank: int,
    world_size: int,
    error: BaseException,
) -> AsrSummary:
    """Persist a lifecycle failure for each missing rank-local hypothesis."""
    if world_size != FULL_VALIDATION_WORLD_SIZE:
        raise ValueError("hard-number ASR requires world_size=8")
    if type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError("rank must satisfy 0 <= rank < world_size")
    records = _validate_synthesis_records(synthesis_records, world_size=world_size)
    local_records = [record for record in records if record["rank"] == rank]
    ledger = AtomicJsonlLedger(_rank_hypotheses_path(output_dir, rank))
    existing = {record["id"]: record for record in ledger.records}
    expected_ids = {record["id"] for record in local_records}
    extras = sorted(set(existing) - expected_ids)
    if extras:
        raise ValueError(f"rank hypothesis ledger contains non-local IDs: {extras}")
    skipped = sum(
        isinstance(existing.get(record["id"], {}).get("hypothesis"), str)
        and "error" not in existing[record["id"]]
        for record in local_records
    )
    for record in local_records:
        identifier = record["id"]
        if identifier not in existing or not isinstance(
            existing[identifier].get("hypothesis"), str
        ) or "error" in existing[identifier]:
            ledger.upsert(_error_record(identifier, rank, error))
    return AsrSummary(
        rank=rank,
        expected=len(local_records),
        completed=skipped,
        transcribed=0,
        skipped=skipped,
        failed=len(local_records) - skipped,
        complete=False,
        stop_reason="error",
    )


def transcribe_rank(
    *,
    synthesis_records: Sequence[Mapping[str, Any]],
    recognizer: Any,
    output_dir: str | Path | ValidationPaths,
    rank: int,
    world_size: int,
    deadline_monotonic: float | None = None,
    stop_requested: Callable[[], bool] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> AsrSummary:
    """Transcribe one 250-row rank stride and atomically resume valid outputs."""
    if world_size != FULL_VALIDATION_WORLD_SIZE:
        raise ValueError("hard-number ASR requires world_size=8")
    if type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError("rank must satisfy 0 <= rank < world_size")
    if deadline_monotonic is not None and (
        isinstance(deadline_monotonic, bool)
        or not isinstance(deadline_monotonic, (int, float))
        or not math.isfinite(deadline_monotonic)
    ):
        raise ValueError("deadline_monotonic must be finite")

    records = _validate_synthesis_records(synthesis_records, world_size=world_size)
    local_records = [record for record in records if record["rank"] == rank]
    ledger = AtomicJsonlLedger(_rank_hypotheses_path(output_dir, rank))
    existing = {record["id"]: record for record in ledger.records}
    expected_ids = {record["id"] for record in local_records}
    extras = sorted(set(existing) - expected_ids)
    if extras:
        raise ValueError(f"rank hypothesis ledger contains non-local IDs: {extras}")

    valid_ids = {
        identifier
        for identifier, record in existing.items()
        if isinstance(record.get("hypothesis"), str) and "error" not in record
    }
    skipped = len(valid_ids)
    transcribed = 0
    stop_reason: str | None = None
    should_stop = stop_requested if stop_requested is not None else lambda: False

    for record in local_records:
        identifier = record["id"]
        if identifier in valid_ids:
            continue
        if should_stop():
            stop_reason = "signal"
            break
        if deadline_monotonic is not None and monotonic() >= deadline_monotonic:
            stop_reason = "deadline"
            break
        try:
            ledger.upsert(
                {
                    "hypothesis": _hypothesis(
                        recognizer.recognize(_asr_audio(record["wav"]), ASR_SAMPLE_RATE)
                    ),
                    "id": identifier,
                    "rank": rank,
                }
            )
            transcribed += 1
        except Exception as error:  # noqa: BLE001 - durable per-row failure contract
            ledger.upsert(_error_record(identifier, rank, error))

    final_records = {record["id"]: record for record in ledger.records}
    completed = sum(
        identifier in final_records
        and isinstance(final_records[identifier].get("hypothesis"), str)
        and "error" not in final_records[identifier]
        for identifier in expected_ids
    )
    failed = sum(
        identifier in final_records and "error" in final_records[identifier]
        for identifier in expected_ids
    )
    return AsrSummary(
        rank=rank,
        expected=len(local_records),
        completed=completed,
        transcribed=transcribed,
        skipped=skipped,
        failed=failed,
        complete=completed == len(local_records) and failed == 0 and stop_reason is None,
        stop_reason=stop_reason,
    )
