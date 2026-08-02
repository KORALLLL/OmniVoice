"""Distributed, resumable OmniVoice synthesis for hard-number validation."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import stat
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download

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
_HUB_COMMIT = re.compile(r"^[0-9a-f]{40,64}$")

GENERATION_CONFIG = OmniVoiceGenerationConfig(
    num_step=32,
    guidance_scale=2.0,
    t_shift=0.1,
    layer_penalty_factor=5.0,
    position_temperature=0.0,
    class_temperature=0.0,
)


def _adapter_composite_sha256(content_sha256: str, base_source: Any) -> str:
    canonical = json.dumps(
        {"adapter_sha256": content_sha256, "base_source": asdict(base_source)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int


@dataclass(frozen=True)
class ModelSourceIdentity:
    """A validated immutable model load target and its stable identity."""

    kind: str
    requested: str
    load_path: str
    immutable_id: str
    content_sha256: str | None = None
    base_source: ModelSourceIdentity | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"base", "adapter"}:
            raise ValueError("model source kind must be 'base' or 'adapter'")
        if not isinstance(self.requested, str) or not self.requested.strip():
            raise ValueError("requested model source must be a non-blank string")
        load_path = Path(self.load_path)
        if not load_path.is_dir() or str(load_path.resolve()) != self.load_path:
            raise ValueError("model source load_path must be a resolved directory")
        expected_prefix = "hf:" if self.kind == "base" else "sha256:"
        digest = self.immutable_id.removeprefix(expected_prefix)
        pattern = _HUB_COMMIT if self.kind == "base" else _SHA256
        if not self.immutable_id.startswith(expected_prefix) or not pattern.fullmatch(
            digest
        ):
            raise ValueError("model source immutable_id is malformed")
        if self.kind == "base":
            if self.content_sha256 is not None or self.base_source is not None:
                raise ValueError("base source may not contain adapter identity fields")
        elif (
            not isinstance(self.content_sha256, str)
            or not _SHA256.fullmatch(self.content_sha256)
            or not isinstance(self.base_source, ModelSourceIdentity)
            or self.base_source.kind != "base"
        ):
            raise ValueError(
                "adapter source requires a content SHA-256 and immutable base source"
            )
        elif self.immutable_id != (
            "sha256:" + _adapter_composite_sha256(self.content_sha256, self.base_source)
        ):
            raise ValueError("adapter composite identity does not match its sources")


@dataclass(frozen=True)
class LifecycleError:
    """Structured reason a rank summary was corrected after synthesis."""

    stage: str
    type: str
    message: str


@dataclass(frozen=True)
class SynthesisSummary:
    rank: int
    expected: int | None
    completed: int | None
    generated: int | None
    skipped: int | None
    failed: int | None
    complete: bool
    stop_reason: str | None
    source_identity: ModelSourceIdentity
    error: LifecycleError | None = None
    run_id: str | None = None
    step: int | None = None
    primary_error: LifecycleError | None = None
    cleanup_error: LifecycleError | None = None


def run_bounded(
    function: Callable[[], Any],
    *,
    deadline_monotonic: float,
    description: str,
    monotonic: Callable[[], float] = time.monotonic,
) -> Any:
    """Run lifecycle work in a daemon thread so a hung call cannot hold exit."""
    if (
        isinstance(deadline_monotonic, bool)
        or not isinstance(deadline_monotonic, (int, float))
        or not math.isfinite(deadline_monotonic)
    ):
        raise ValueError("deadline_monotonic must be finite")
    remaining = deadline_monotonic - monotonic()
    if remaining <= 0:
        raise TimeoutError(f"{description} exceeded the cleanup deadline")
    done = threading.Event()
    outcome: list[tuple[bool, Any]] = []

    def invoke() -> None:
        try:
            outcome.append((True, function()))
        except BaseException as error:  # noqa: BLE001 - propagate caller failure
            outcome.append((False, error))
        finally:
            done.set()

    worker = threading.Thread(
        target=invoke,
        name=f"omnivoice-{description.replace(' ', '-')}",
        daemon=True,
    )
    worker.start()
    if not done.wait(remaining):
        raise TimeoutError(f"{description} exceeded the cleanup deadline")
    succeeded, value = outcome[0]
    if not succeeded:
        raise value
    return value


def _adapter_checkpoint_root(checkpoint_path: str | Path) -> Path:
    from omnivoice.training.lora import (
        _validate_adapter_config,
        read_lora_metadata,
        resolve_adapter_dir,
    )

    checkpoint_root, adapter_dir = resolve_adapter_dir(checkpoint_path)
    checkpoint_root = checkpoint_root.resolve()
    adapter_dir = adapter_dir.resolve()
    metadata = read_lora_metadata(checkpoint_root)
    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError("LoRA adapter_config.json is missing")
    if not any(
        (adapter_dir / filename).is_file()
        for filename in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise FileNotFoundError("LoRA adapter weights are missing")
    _validate_adapter_config(adapter_dir, metadata)
    return checkpoint_root


def fingerprint_adapter_checkpoint(checkpoint_path: str | Path) -> str:
    """Hash a canonical recursive manifest of every checkpoint regular file."""
    checkpoint_root = _adapter_checkpoint_root(checkpoint_path)
    manifest: list[dict[str, Any]] = []
    for path in sorted(checkpoint_root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(checkpoint_root).as_posix()
        metadata = path.stat(follow_symlinks=False)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"LoRA checkpoint may not contain symlinks: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(
                f"LoRA checkpoint may contain only regular files: {relative}"
            )
        manifest.append(
            {
                "path": relative,
                "sha256": _sha256_file(path),
                "size": metadata.st_size,
            }
        )
    if not manifest:
        raise ValueError("LoRA checkpoint contains no regular files")
    canonical = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def resolve_model_source(
    *,
    model_name: str | None = None,
    adapter_checkpoint: str | Path | None = None,
    snapshot_resolver: Callable[..., str | Path] = snapshot_download,
    adapter_fingerprinter: Callable[[str | Path], str] = fingerprint_adapter_checkpoint,
) -> ModelSourceIdentity:
    """Resolve a requested base or adapter into an immutable load identity."""
    if (model_name is None) == (adapter_checkpoint is None):
        raise ValueError("specify exactly one of model_name or adapter_checkpoint")
    if model_name is not None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-blank string")
        snapshot = Path(snapshot_resolver(repo_id=model_name)).resolve()
        if not snapshot.is_dir() or not _HUB_COMMIT.fullmatch(snapshot.name):
            raise ValueError(
                "base model did not resolve to an immutable Hub snapshot directory"
            )
        return ModelSourceIdentity(
            kind="base",
            requested=model_name,
            load_path=str(snapshot),
            immutable_id=f"hf:{snapshot.name}",
        )

    requested = str(adapter_checkpoint)
    checkpoint_root = _adapter_checkpoint_root(adapter_checkpoint)
    from omnivoice.training.lora import read_lora_metadata

    metadata = read_lora_metadata(checkpoint_root)
    base_model_name = metadata.get("base_model_name_or_path")
    if not isinstance(base_model_name, str) or not base_model_name.strip():
        raise ValueError(
            "LoRA metadata base_model_name_or_path must be a non-blank string"
        )
    revision = metadata.get("base_model_revision")
    if revision is not None and (not isinstance(revision, str) or not revision.strip()):
        raise ValueError("LoRA metadata base_model_revision must be a non-blank string")
    resolver_kwargs = {"repo_id": base_model_name}
    if revision is not None:
        resolver_kwargs["revision"] = revision
    base_snapshot = Path(snapshot_resolver(**resolver_kwargs)).resolve()
    if not base_snapshot.is_dir() or not _HUB_COMMIT.fullmatch(base_snapshot.name):
        raise ValueError(
            "adapter base model did not resolve to an immutable Hub snapshot directory"
        )
    base_source = ModelSourceIdentity(
        kind="base",
        requested=base_model_name,
        load_path=str(base_snapshot),
        immutable_id=f"hf:{base_snapshot.name}",
    )
    digest = adapter_fingerprinter(checkpoint_root)
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise ValueError("adapter fingerprinter must return a lowercase SHA-256")
    composite = _adapter_composite_sha256(digest, base_source)
    return ModelSourceIdentity(
        kind="adapter",
        requested=requested,
        load_path=str(checkpoint_root),
        immutable_id=f"sha256:{composite}",
        content_sha256=digest,
        base_source=base_source,
    )


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
                detail = (
                    error.msg if isinstance(error, json.JSONDecodeError) else str(error)
                )
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
    source_identity: ModelSourceIdentity,
    model_class: Any = OmniVoice,
    torch_module: Any = torch,
    adapter_fingerprinter: Callable[[str | Path], str] = fingerprint_adapter_checkpoint,
) -> Any:
    """Bind one CUDA device and load exactly one base or LoRA model in FP16."""
    if context.world_size != FULL_VALIDATION_WORLD_SIZE:
        raise ValueError("validation TTS requires exactly eight ranks")
    if not isinstance(source_identity, ModelSourceIdentity):
        raise TypeError("source_identity must be a ModelSourceIdentity")
    load_path = Path(source_identity.load_path)
    if source_identity.kind == "base":
        if source_identity.immutable_id != f"hf:{load_path.name}":
            raise ValueError("base snapshot identity no longer matches its load path")
    else:
        current_digest = adapter_fingerprinter(load_path)
        if source_identity.content_sha256 != current_digest:
            raise ValueError("adapter checkpoint changed after identity resolution")
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
    if source_identity.kind == "base":
        return model_class.from_pretrained(str(load_path), **loader_kwargs)
    model = model_class.from_lora_pretrained(
        load_path,
        base_model_override=source_identity.base_source.load_path,
        **loader_kwargs,
    )
    if adapter_fingerprinter(load_path) != source_identity.content_sha256:
        raise ValueError("adapter checkpoint changed during model loading")
    return model


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
    """Attempt a barrier and tear down under the caller's process deadline."""
    if not dist_module.is_available() or not dist_module.is_initialized():
        return True
    synchronized = False
    try:
        work = dist_module.barrier(async_op=True)
        synchronized = work.wait(timeout=timedelta(seconds=timeout_seconds)) is True
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


def _assignment_provenance(row: ValidationAssignment) -> tuple[dict[str, Any], str]:
    snapshot = asdict(row)
    canonical = json.dumps(
        snapshot,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return snapshot, hashlib.sha256(canonical).hexdigest()


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
    source_identity: ModelSourceIdentity,
    wav_path: Path,
) -> bool:
    assignment, assignment_sha256 = _assignment_provenance(row)
    expected = {
        "assignment": assignment,
        "assignment_sha256": assignment_sha256,
        "category": row.category,
        "hard_number": row.hard_number,
        "id": row.id,
        "normalized_gold": row.normalized_gold,
        "text": row.text,
        "voice_id": row.voice_id,
        "rank": rank,
        "checkpoint": source_identity.requested,
        "stressed": row.stressed,
        "reference_wav_sha256": row.reference_wav_sha256,
        "generation_config": asdict(row.generation_config),
        "wav": str(wav_path),
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "source_identity": asdict(source_identity),
    }
    if "error" in record or any(
        record.get(key) != value for key, value in expected.items()
    ):
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


def write_synthesis_summary(
    output_dir: str | Path | ValidationPaths, summary: SynthesisSummary
) -> Path:
    """Atomically publish the durable summary for one synthesis rank."""
    step_dir = _step_directory(output_dir)
    path = step_dir / "rank-manifests" / f"rank-{summary.rank}.summary.json"
    _atomic_write_summary(path, summary)
    return path


def read_synthesis_summary(
    output_dir: str | Path | ValidationPaths, rank: int
) -> SynthesisSummary | None:
    """Read a previously published rank summary for lifecycle correction."""
    path = _step_directory(output_dir) / "rank-manifests" / f"rank-{rank}.summary.json"
    if not path.exists():
        return None
    raw = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
        object_pairs_hook=_reject_duplicate_members,
    )
    if not isinstance(raw, dict):
        raise TypeError("synthesis summary must be a JSON object")
    values = dict(raw)
    source = values.get("source_identity")
    if not isinstance(source, dict):
        raise TypeError("synthesis summary source_identity must be an object")
    nested_base = source.get("base_source")
    if nested_base is not None:
        if not isinstance(nested_base, dict):
            raise ValueError("adapter summary base_source must be an object")
        source = dict(source)
        source["base_source"] = ModelSourceIdentity(**nested_base)
    values["source_identity"] = ModelSourceIdentity(**source)
    for field_name in ("error", "primary_error", "cleanup_error"):
        error = values.get(field_name)
        if error is not None:
            if not isinstance(error, dict):
                raise ValueError(
                    f"synthesis summary {field_name} must be an object or null"
                )
            values[field_name] = LifecycleError(**error)
    return SynthesisSummary(**values)


def _success_record(
    row: ValidationAssignment,
    *,
    rank: int,
    source_identity: ModelSourceIdentity,
    wav_path: Path,
    wav_hash: str,
) -> dict[str, Any]:
    assignment, assignment_sha256 = _assignment_provenance(row)
    return {
        "assignment": assignment,
        "assignment_sha256": assignment_sha256,
        "category": row.category,
        "channels": 1,
        "checkpoint": source_identity.requested,
        "generation_config": asdict(row.generation_config),
        "hard_number": row.hard_number,
        "id": row.id,
        "normalized_gold": row.normalized_gold,
        "rank": rank,
        "reference_audio_path": row.reference_audio_path,
        "reference_wav_sha256": row.reference_wav_sha256,
        "sample_rate": SAMPLE_RATE,
        "sha256": wav_hash,
        "source_identity": asdict(source_identity),
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
    source_identity: ModelSourceIdentity,
    deadline_monotonic: float | None = None,
    stop_requested: Callable[[], bool] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> SynthesisSummary:
    """Synthesize one exact 250-row stride with hash-validated resume."""
    if world_size != FULL_VALIDATION_WORLD_SIZE:
        raise ValueError("hard-number synthesis requires world_size=8")
    if not 0 <= rank < world_size:
        raise ValueError("rank must satisfy 0 <= rank < world_size")
    if not isinstance(source_identity, ModelSourceIdentity):
        raise TypeError("source_identity must be a ModelSourceIdentity")
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

    run_id = output_dir.run_id if isinstance(output_dir, ValidationPaths) else None
    step = output_dir.step if isinstance(output_dir, ValidationPaths) else None
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
            source_identity=source_identity,
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
                    source_identity=source_identity,
                    wav_path=wav_path,
                    wav_hash=wav_hash,
                )
            )
            generated += 1
        except Exception as error:  # noqa: BLE001 - durable per-row failure contract
            ledger.upsert(
                {
                    "checkpoint": source_identity.requested,
                    "error": f"{type(error).__name__}: {error}",
                    "id": row.id,
                    "rank": rank,
                    "source_identity": asdict(source_identity),
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
            source_identity=source_identity,
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
        source_identity=source_identity,
        run_id=run_id,
        step=step,
    )
    _atomic_write_summary(summary_path, summary)
    return summary
