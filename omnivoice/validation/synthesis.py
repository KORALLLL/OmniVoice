"""Distributed, resumable OmniVoice synthesis for hard-number validation."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.validation.artifacts import AtomicJsonlLedger, ValidationPaths
from omnivoice.validation.hard_numbers import (
    HARD_NUMBER_COUNT,
    ValidationAssignment,
    ValidationGenerationConfig,
    _validate_assignment_manifest_contract,
    partition_assignments,
)

FULL_VALIDATION_WORLD_SIZE = 8
RANK_ASSIGNMENT_COUNT = HARD_NUMBER_COUNT // FULL_VALIDATION_WORLD_SIZE
SAMPLE_RATE = 24_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

GENERATION_CONFIG = OmniVoiceGenerationConfig(
    num_step=32,
    guidance_scale=2.0,
    t_shift=0.1,
    layer_penalty_factor=5.0,
    position_temperature=0.0,
    class_temperature=0.0,
)


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int


@dataclass(frozen=True)
class SynthesisSummary:
    rank: int
    expected: int
    completed: int
    generated: int
    skipped: int
    failed: int
    complete: bool
    stop_reason: str | None


class _StrictJsonError(ValueError):
    pass


def _reject_constant(value: str) -> None:
    raise _StrictJsonError(f"non-finite JSON constant {value!r}")


def _reject_duplicate_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _StrictJsonError(f"duplicate object member {key!r}")
        result[key] = value
    return result


def _decode_assignment(raw: object, line_number: int) -> ValidationAssignment:
    if not isinstance(raw, dict):
        raise TypeError(f"assignment manifest line {line_number} must be an object")
    assignment_fields = {field.name for field in fields(ValidationAssignment)}
    if set(raw) != assignment_fields:
        missing = sorted(assignment_fields - set(raw))
        extra = sorted(set(raw) - assignment_fields)
        raise ValueError(
            f"assignment manifest line {line_number} fields are not exact: "
            f"missing={missing}, extra={extra}"
        )
    generation = raw["generation_config"]
    if not isinstance(generation, dict):
        raise TypeError(
            f"assignment manifest line {line_number} generation_config must be an object"
        )
    generation_fields = {field.name for field in fields(ValidationGenerationConfig)}
    if set(generation) != generation_fields:
        missing = sorted(generation_fields - set(generation))
        extra = sorted(set(generation) - generation_fields)
        raise ValueError(
            f"assignment manifest line {line_number} generation_config fields "
            f"are not exact: missing={missing}, extra={extra}"
        )
    values = dict(raw)
    values["generation_config"] = ValidationGenerationConfig(**generation)
    try:
        return ValidationAssignment(**values)
    except TypeError as error:
        raise ValueError(
            f"assignment manifest line {line_number} is invalid: {error}"
        ) from error


def load_assignment_manifest(path: str | Path) -> list[ValidationAssignment]:
    """Load a Task 5 manifest and reapply its complete immutable contract."""
    source_path = Path(path)
    assignments: list[ValidationAssignment] = []
    with source_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                raw = json.loads(
                    line,
                    parse_constant=_reject_constant,
                    object_pairs_hook=_reject_duplicate_members,
                )
            except (json.JSONDecodeError, _StrictJsonError) as error:
                detail = error.msg if isinstance(error, json.JSONDecodeError) else str(error)
                raise ValueError(
                    f"invalid assignment JSON on line {line_number}: {detail}"
                ) from error
            assignments.append(_decode_assignment(raw, line_number))
    return _validate_assignment_manifest_contract(assignments)


def _environment_integer(environment: Mapping[str, str], name: str) -> int:
    try:
        raw = environment[name]
    except KeyError as error:
        raise ValueError("RANK, LOCAL_RANK, and WORLD_SIZE must all be set") from error
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer; got {raw!r}") from error


def resolve_distributed_context(
    environment: Mapping[str, str] | None = None,
) -> DistributedContext:
    """Resolve and strictly validate the torchrun rank environment."""
    source = os.environ if environment is None else environment
    rank = _environment_integer(source, "RANK")
    local_rank = _environment_integer(source, "LOCAL_RANK")
    world_size = _environment_integer(source, "WORLD_SIZE")
    if world_size != FULL_VALIDATION_WORLD_SIZE:
        raise ValueError(
            f"WORLD_SIZE must be exactly {FULL_VALIDATION_WORLD_SIZE}; got {world_size}"
        )
    if not 0 <= rank < world_size:
        raise ValueError("RANK must satisfy 0 <= RANK < WORLD_SIZE")
    if not 0 <= local_rank < world_size:
        raise ValueError("LOCAL_RANK must satisfy 0 <= LOCAL_RANK < WORLD_SIZE")
    return DistributedContext(rank=rank, local_rank=local_rank, world_size=world_size)


def load_validation_tts(
    *,
    context: DistributedContext,
    model_name: str | None = None,
    adapter_checkpoint: str | Path | None = None,
    model_class: Any = OmniVoice,
    torch_module: Any = torch,
) -> Any:
    """Bind one CUDA device and load exactly one base or LoRA model in FP16."""
    if context.world_size != FULL_VALIDATION_WORLD_SIZE:
        raise ValueError("validation TTS requires exactly eight ranks")
    if (model_name is None) == (adapter_checkpoint is None):
        raise ValueError("specify exactly one of model_name or adapter_checkpoint")
    if model_name is not None and (
        not isinstance(model_name, str) or not model_name.strip()
    ):
        raise ValueError("model_name must be a non-blank string")
    checkpoint = None if adapter_checkpoint is None else Path(adapter_checkpoint)
    if checkpoint is not None and not checkpoint.is_dir():
        raise FileNotFoundError(f"adapter checkpoint directory does not exist: {checkpoint}")
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA is required for eight-rank validation synthesis")
    if context.local_rank >= torch_module.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK {context.local_rank} has no visible CUDA device"
        )

    torch_module.cuda.set_device(context.local_rank)
    loader_kwargs = {
        "device_map": f"cuda:{context.local_rank}",
        "dtype": torch_module.float16,
    }
    if model_name is not None:
        return model_class.from_pretrained(model_name, **loader_kwargs)
    return model_class.from_lora_pretrained(checkpoint, **loader_kwargs)


def initialize_distributed(
    context: DistributedContext,
    *,
    dist_module: Any = torch.distributed,
    timeout_seconds: float = 30.0,
) -> None:
    """Initialize a bounded NCCL process group when torchrun has not done so."""
    if not dist_module.is_available():
        raise RuntimeError("torch.distributed is unavailable")
    if dist_module.is_initialized():
        return
    dist_module.init_process_group(
        backend="nccl",
        rank=context.rank,
        world_size=context.world_size,
        timeout=timedelta(seconds=timeout_seconds),
    )


def synchronize_distributed(
    *,
    dist_module: Any = torch.distributed,
    timeout_seconds: float = 30.0,
) -> bool:
    """Attempt a bounded barrier and always tear down the process group."""
    if not dist_module.is_available() or not dist_module.is_initialized():
        return True
    synchronized = False
    try:
        work = dist_module.barrier(async_op=True)
        work.wait(timeout=timedelta(seconds=timeout_seconds))
        synchronized = True
    except (RuntimeError, TimeoutError):
        synchronized = False
    finally:
        try:
            dist_module.destroy_process_group()
        except RuntimeError:
            synchronized = False
    return synchronized


def release_validation_tts(*, torch_module: Any = torch) -> None:
    """Collect released model objects and clear this rank's CUDA allocator cache."""
    gc.collect()
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()


def _step_directory(output_dir: str | Path | ValidationPaths) -> Path:
    if isinstance(output_dir, ValidationPaths):
        return output_dir.step_dir
    return Path(output_dir)


def _wav_path(wav_dir: Path, assignment_id: str) -> Path:
    filename = hashlib.sha256(assignment_id.encode("utf-8")).hexdigest() + ".wav"
    return (wav_dir / filename).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_wav(path: Path, expected_hash: object) -> bool:
    if not isinstance(expected_hash, str) or not _SHA256.fullmatch(expected_hash):
        return False
    if not path.is_file() or _sha256_file(path) != expected_hash:
        return False
    try:
        info = sf.info(path)
    except (OSError, RuntimeError):
        return False
    return (
        info.format == "WAV"
        and info.subtype == "PCM_16"
        and info.samplerate == SAMPLE_RATE
        and info.channels == 1
        and info.frames > 0
    )


def _record_is_resumable(
    record: Mapping[str, Any],
    *,
    row: ValidationAssignment,
    rank: int,
    checkpoint: str,
    wav_path: Path,
) -> bool:
    expected = {
        "id": row.id,
        "voice_id": row.voice_id,
        "rank": rank,
        "checkpoint": checkpoint,
        "stressed": row.stressed,
        "reference_wav_sha256": row.reference_wav_sha256,
        "generation_config": asdict(row.generation_config),
        "wav": str(wav_path),
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
    }
    if "error" in record or any(record.get(key) != value for key, value in expected.items()):
        return False
    return _valid_wav(wav_path, record.get("sha256"))


def _audio_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    audio = np.asarray(value)
    if audio.ndim == 2 and 1 in audio.shape:
        audio = audio.reshape(-1)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("OmniVoice must return one nonempty mono waveform")
    if not np.issubdtype(audio.dtype, np.number) or not np.isfinite(audio).all():
        raise ValueError("OmniVoice waveform must contain only finite numbers")
    return audio.astype(np.float32, copy=False)


def _atomic_write_wav(path: Path, audio: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".wav", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        sf.write(temporary, audio, SAMPLE_RATE, subtype="PCM_16", format="WAV")
        if not _valid_wav(temporary, _sha256_file(temporary)):
            raise ValueError("generated WAV does not satisfy mono 24 kHz PCM16")
        with temporary.open("rb") as source:
            os.fsync(source.fileno())
        os.replace(temporary, path)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256_file(path)


def _atomic_write_summary(path: Path, summary: SynthesisSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            json.dump(
                asdict(summary),
                destination,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _success_record(
    row: ValidationAssignment,
    *,
    rank: int,
    checkpoint: str,
    wav_path: Path,
    wav_hash: str,
) -> dict[str, Any]:
    return {
        "category": row.category,
        "channels": 1,
        "checkpoint": checkpoint,
        "generation_config": asdict(row.generation_config),
        "hard_number": row.hard_number,
        "id": row.id,
        "normalized_gold": row.normalized_gold,
        "rank": rank,
        "reference_audio_path": row.reference_audio_path,
        "reference_wav_sha256": row.reference_wav_sha256,
        "sample_rate": SAMPLE_RATE,
        "sha256": wav_hash,
        "stressed": row.stressed,
        "text": row.text,
        "voice_id": row.voice_id,
        "wav": str(wav_path),
    }


def synthesize_rank(
    *,
    assignments: Sequence[ValidationAssignment],
    model: Any,
    output_dir: str | Path | ValidationPaths,
    rank: int,
    world_size: int,
    checkpoint: str = "<injected-model>",
    deadline_monotonic: float | None = None,
    stop_requested: Callable[[], bool] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> SynthesisSummary:
    """Synthesize one exact 250-row stride with hash-validated resume."""
    if world_size != FULL_VALIDATION_WORLD_SIZE:
        raise ValueError("hard-number synthesis requires world_size=8")
    if not 0 <= rank < world_size:
        raise ValueError("rank must satisfy 0 <= rank < world_size")
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError("checkpoint must be a non-blank string")
    if deadline_monotonic is not None and (
        isinstance(deadline_monotonic, bool)
        or not isinstance(deadline_monotonic, (int, float))
        or not math.isfinite(deadline_monotonic)
    ):
        raise ValueError("deadline_monotonic must be finite")
    if type(model.sampling_rate) is not int or model.sampling_rate != SAMPLE_RATE:
        raise ValueError("OmniVoice sampling rate must be exactly 24000 Hz")

    ordered = _validate_assignment_manifest_contract(assignments)
    local_rows = partition_assignments(ordered, rank, world_size)
    if len(local_rows) != RANK_ASSIGNMENT_COUNT:
        raise ValueError(
            f"rank {rank} must receive exactly {RANK_ASSIGNMENT_COUNT} assignments"
        )

    step_dir = _step_directory(output_dir)
    wav_dir = step_dir / "wavs"
    ledger_path = step_dir / "rank-manifests" / f"rank-{rank}.jsonl"
    summary_path = step_dir / "rank-manifests" / f"rank-{rank}.summary.json"
    wav_dir.mkdir(parents=True, exist_ok=True)
    ledger = AtomicJsonlLedger(ledger_path)
    records = {record["id"]: record for record in ledger.records}
    expected_ids = {row.id for row in local_rows}
    extras = sorted(set(records) - expected_ids)
    if extras:
        raise ValueError(f"rank ledger contains non-local assignment IDs: {extras}")

    valid_ids = {
        row.id
        for row in local_rows
        if row.id in records
        and _record_is_resumable(
            records[row.id],
            row=row,
            rank=rank,
            checkpoint=checkpoint,
            wav_path=_wav_path(wav_dir, row.id),
        )
    }
    skipped = len(valid_ids)
    generated = 0
    prompt_cache: dict[str, Any] = {}
    stop_reason: str | None = None
    should_stop = stop_requested if stop_requested is not None else lambda: False

    for row in local_rows:
        if row.id in valid_ids:
            continue
        if should_stop():
            stop_reason = "signal"
            break
        if deadline_monotonic is not None and monotonic() >= deadline_monotonic:
            stop_reason = "deadline"
            break

        wav_path = _wav_path(wav_dir, row.id)
        try:
            prompt = prompt_cache.get(row.voice_id)
            if prompt is None:
                prompt = model.create_voice_clone_prompt(
                    ref_audio=row.reference_audio_path,
                    ref_text=row.reference_text,
                    preprocess_prompt=True,
                )
                prompt_cache[row.voice_id] = prompt
            audios = model.generate(
                text=row.stressed,
                language="Russian",
                voice_clone_prompt=prompt,
                generation_config=GENERATION_CONFIG,
            )
            if not isinstance(audios, Sequence) or len(audios) != 1:
                raise ValueError("OmniVoice must return exactly one waveform")
            wav_hash = _atomic_write_wav(wav_path, _audio_array(audios[0]))
            ledger.upsert(
                _success_record(
                    row,
                    rank=rank,
                    checkpoint=checkpoint,
                    wav_path=wav_path,
                    wav_hash=wav_hash,
                )
            )
            generated += 1
        except Exception as error:  # noqa: BLE001 - durable per-row failure contract
            ledger.upsert(
                {
                    "checkpoint": checkpoint,
                    "error": f"{type(error).__name__}: {error}",
                    "id": row.id,
                    "rank": rank,
                    "voice_id": row.voice_id,
                }
            )

    final_records = {record["id"]: record for record in ledger.records}
    completed = sum(
        row.id in final_records
        and _record_is_resumable(
            final_records[row.id],
            row=row,
            rank=rank,
            checkpoint=checkpoint,
            wav_path=_wav_path(wav_dir, row.id),
        )
        for row in local_rows
    )
    failed = sum(
        row.id in final_records and "error" in final_records[row.id]
        for row in local_rows
    )
    complete = completed == len(local_rows) and failed == 0 and stop_reason is None
    summary = SynthesisSummary(
        rank=rank,
        expected=len(local_rows),
        completed=completed,
        generated=generated,
        skipped=skipped,
        failed=failed,
        complete=complete,
        stop_reason=stop_reason,
    )
    _atomic_write_summary(summary_path, summary)
    return summary
