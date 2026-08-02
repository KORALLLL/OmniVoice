"""Prepare reproducible hard-number prompts and Balalaika voice assignments."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import soundfile as sf
from huggingface_hub import hf_hub_download

HARD_NUMBER_REPO = "bitmanagerai/hard_number_eval_for_tts"
HARD_NUMBER_REVISION = "57b964492ccfcedd6a24d0225ef4b7d3697ffdca"
HARD_NUMBER_FILENAME = "hard_number_eval.jsonl"
HARD_NUMBER_COUNT = 2_000
VALIDATION_VOICE_COUNT = 20
ASSIGNMENTS_PER_VOICE = 100
REQUIRED_FIELDS = (
    "id",
    "category",
    "hard_number",
    "text",
    "normalized_gold",
    "stressed",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_RELATIVE_PATH = re.compile(r"^(?P<shard>\d{6})/(?P<member>[^/]+\.mp3)$")
_PREFERRED_TIER = "preferred_3_to_12s"
_FALLBACK_TIER = "fallback_over_12s"


@dataclass(frozen=True)
class HardNumberRow:
    id: str
    category: str
    hard_number: str
    text: str
    normalized_gold: str
    stressed: str


@dataclass(frozen=True)
class ValidationGenerationConfig:
    language: str = "Russian"
    num_step: int = 32
    guidance_scale: float = 2.0
    t_shift: float = 0.1
    layer_penalty_factor: float = 5.0
    position_temperature: float = 0.0
    class_temperature: float = 0.0
    denoise: bool = True
    preprocess_prompt: bool = True
    postprocess_output: bool = True
    audio_chunk_duration: float = 15.0
    audio_chunk_threshold: float = 30.0
    pad_duration: float = 0.1
    fade_duration: float = 0.1


@dataclass(frozen=True)
class ValidationAssignment:
    dataset_repo: str
    dataset_revision: str
    dataset_filename: str
    id: str
    category: str
    hard_number: str
    text: str
    normalized_gold: str
    stressed: str
    voice_id: str
    reference_role: str
    reference_source_relative_path: str
    reference_text: str
    reference_schema_version: int
    reference_seed: int
    reference_source_shard: str
    reference_member_name: str
    reference_audio_path: str
    reference_source_sha256: str
    reference_wav_sha256: str
    reference_sample_rate: int
    reference_channels: int
    reference_duration: float
    reference_duration_tier: str
    generation_config: ValidationGenerationConfig


@dataclass(frozen=True)
class _ReferenceMetadata:
    role: str
    source_relative_path: str
    text: str
    schema_version: int
    seed: int
    source_shard: str
    member_name: str
    audio_path: str
    source_sha256: str
    wav_sha256: str
    sample_rate: int
    channels: int
    duration: float
    duration_tier: str


def download_hard_number_jsonl() -> Path:
    """Download the benchmark file from its immutable Hub revision."""
    return Path(
        hf_hub_download(
            repo_id=HARD_NUMBER_REPO,
            repo_type="dataset",
            filename=HARD_NUMBER_FILENAME,
            revision=HARD_NUMBER_REVISION,
        )
    )


def load_hard_number_rows(
    path: str | Path,
    *,
    expected_count: int = HARD_NUMBER_COUNT,
    revision: str = HARD_NUMBER_REVISION,
) -> list[HardNumberRow]:
    """Load, validate, and deterministically order the pinned benchmark rows."""
    if revision != HARD_NUMBER_REVISION:
        raise ValueError(
            "dataset revision must be pinned to " f"{HARD_NUMBER_REVISION}; got {revision!r}"
        )
    if expected_count != HARD_NUMBER_COUNT:
        raise ValueError(
            f"expected_count must be exactly {HARD_NUMBER_COUNT}; got {expected_count}"
        )

    rows: list[HardNumberRow] = []
    seen_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number}: {error.msg}") from error
            if not isinstance(raw, dict):
                raise TypeError(f"line {line_number} must contain a JSON object")

            raw_id = raw.get("id")
            if isinstance(raw_id, bool) or not isinstance(raw_id, (int, str)):
                raise ValueError(  # noqa: TRY004 - malformed dataset value
                    f"line {line_number} field 'id' must be a non-blank string or integer"
                )
            normalized_id = str(raw_id).strip()
            if not normalized_id:
                raise ValueError(
                    f"line {line_number} field 'id' must be a non-blank string or integer"
                )

            values: dict[str, str] = {"id": normalized_id}
            for field in REQUIRED_FIELDS[1:]:
                value = raw.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"line {line_number} field {field!r} must be a non-blank string"
                    )
                values[field] = value

            if normalized_id in seen_ids:
                raise ValueError(f"duplicate hard-number id {normalized_id!r}")
            seen_ids.add(normalized_id)
            rows.append(HardNumberRow(**values))

    if len(rows) != expected_count:
        raise ValueError(
            f"hard-number dataset must contain exactly {expected_count} rows; "
            f"found {len(rows)}"
        )
    return sorted(rows, key=lambda row: row.id)


def _required_voice_value(voice: object, field: str) -> Any:
    if not hasattr(voice, field):
        raise ValueError(f"validation voice is missing required field {field!r}")
    return getattr(voice, field)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_nonblank(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-blank string")
    return value


def _reference_from_voice(voice: object) -> _ReferenceMetadata:
    return _ReferenceMetadata(
        role=_required_voice_value(voice, "role"),
        source_relative_path=_required_voice_value(voice, "source_relative_path"),
        text=_required_voice_value(voice, "text"),
        schema_version=_required_voice_value(voice, "schema_version"),
        seed=_required_voice_value(voice, "seed"),
        source_shard=_required_voice_value(voice, "source_shard"),
        member_name=_required_voice_value(voice, "member_name"),
        audio_path=_required_voice_value(voice, "audio_path"),
        source_sha256=_required_voice_value(voice, "source_sha256"),
        wav_sha256=_required_voice_value(voice, "wav_sha256"),
        sample_rate=_required_voice_value(voice, "sample_rate"),
        channels=_required_voice_value(voice, "channels"),
        duration=_required_voice_value(voice, "duration"),
        duration_tier=_required_voice_value(voice, "duration_tier"),
    )


def _reference_from_assignment(assignment: ValidationAssignment) -> _ReferenceMetadata:
    return _ReferenceMetadata(
        role=assignment.reference_role,
        source_relative_path=assignment.reference_source_relative_path,
        text=assignment.reference_text,
        schema_version=assignment.reference_schema_version,
        seed=assignment.reference_seed,
        source_shard=assignment.reference_source_shard,
        member_name=assignment.reference_member_name,
        audio_path=assignment.reference_audio_path,
        source_sha256=assignment.reference_source_sha256,
        wav_sha256=assignment.reference_wav_sha256,
        sample_rate=assignment.reference_sample_rate,
        channels=assignment.reference_channels,
        duration=assignment.reference_duration,
        duration_tier=assignment.reference_duration_tier,
    )


def _validate_reference(reference: _ReferenceMetadata, voice_id: str) -> None:
    if reference.role != "validation_voice":
        raise ValueError(f"voice {voice_id!r} role must be 'validation_voice'")
    source_relative_path = _require_nonblank(
        reference.source_relative_path, "source_relative_path"
    )
    _require_nonblank(reference.text, "reference text")
    source_shard = _require_nonblank(reference.source_shard, "source_shard")
    member_name = _require_nonblank(reference.member_name, "member_name")
    audio_path_value = _require_nonblank(reference.audio_path, "audio_path")
    if type(reference.schema_version) is not int or reference.schema_version <= 0:
        raise ValueError("schema_version must be a positive integer")
    if type(reference.seed) is not int or reference.seed != 42:
        raise ValueError("reference seed must be exactly 42")
    if not isinstance(reference.source_sha256, str) or not _SHA256.fullmatch(
        reference.source_sha256
    ):
        raise ValueError("source_sha256 must be a lowercase SHA-256")
    if not isinstance(reference.wav_sha256, str) or not _SHA256.fullmatch(
        reference.wav_sha256
    ):
        raise ValueError("wav_sha256 must be a lowercase SHA-256")

    source_match = _SOURCE_RELATIVE_PATH.fullmatch(source_relative_path)
    if source_match is None:
        raise ValueError("source_relative_path has an invalid format")
    expected_shard = f"train/shard_{source_match.group('shard')}.tar"
    if source_shard != expected_shard:
        raise ValueError("source_shard does not match source_relative_path")
    if member_name != source_match.group("member"):
        raise ValueError("member_name does not match source_relative_path")

    if type(reference.sample_rate) is not int or reference.sample_rate != 24_000:
        raise ValueError("reference sample rate must be exactly 24000")
    if type(reference.channels) is not int or reference.channels != 1:
        raise ValueError("reference channels must be exactly 1")
    if (
        isinstance(reference.duration, bool)
        or not isinstance(reference.duration, (int, float))
        or not math.isfinite(reference.duration)
        or reference.duration <= 0
    ):
        raise ValueError("reference duration must be finite and positive")
    if reference.duration_tier not in {_PREFERRED_TIER, _FALLBACK_TIER}:
        raise ValueError("reference duration tier is invalid")

    audio_path = Path(audio_path_value)
    if not audio_path.is_file():
        raise FileNotFoundError(f"reference WAV does not exist: {audio_path}")
    try:
        info = sf.info(audio_path)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"reference WAV is unreadable: {audio_path}") from error
    if (
        info.format != "WAV"
        or info.subtype != "PCM_16"
        or info.samplerate != 24_000
        or info.channels != 1
        or info.frames <= 0
    ):
        raise ValueError(
            f"reference WAV must be nonempty mono 24000 Hz PCM16: {audio_path}"
        )
    if reference.sample_rate != info.samplerate or reference.channels != info.channels:
        raise ValueError("reference WAV metadata does not match the audio")
    actual_duration = info.frames / info.samplerate
    if abs(float(reference.duration) - actual_duration) > 1 / info.samplerate:
        raise ValueError("reference duration does not match WAV sample count")
    if reference.duration_tier == _PREFERRED_TIER and not (
        3.0 <= actual_duration <= 12.0
    ):
        raise ValueError("reference duration tier does not match WAV duration")
    if reference.duration_tier == _FALLBACK_TIER and actual_duration <= 12.0:
        raise ValueError("reference duration tier does not match WAV duration")
    if _sha256_file(audio_path) != reference.wav_sha256:
        raise ValueError("reference WAV hash does not match wav_sha256")


def _voice_id(voice: object) -> str:
    explicit_id = getattr(voice, "id", None)
    value = explicit_id if explicit_id is not None else _required_voice_value(
        voice, "source_relative_path"
    )
    if not isinstance(value, str) or not value.strip():
        raise ValueError("validation voice id must be a non-blank string")
    return value.strip()


def _validate_voices(voices: Sequence[object]) -> list[tuple[str, object]]:
    if len(voices) != VALIDATION_VOICE_COUNT:
        raise ValueError(
            f"exactly {VALIDATION_VOICE_COUNT} validation voices are required; "
            f"found {len(voices)}"
        )

    identified: list[tuple[str, object]] = []
    references: list[_ReferenceMetadata] = []
    seen_ids: set[str] = set()
    for voice in voices:
        voice_id = _voice_id(voice)
        if voice_id in seen_ids:
            raise ValueError(f"duplicate voice id {voice_id!r}")
        seen_ids.add(voice_id)
        reference = _reference_from_voice(voice)
        _validate_reference(reference, voice_id)
        references.append(reference)
        identified.append((voice_id, voice))
    resolved_audio_paths = [str(Path(item.audio_path).resolve()) for item in references]
    if (
        len({item.source_relative_path for item in references}) != len(references)
        or len(set(resolved_audio_paths)) != len(references)
        or len({item.source_sha256 for item in references}) != len(references)
        or len({item.wav_sha256 for item in references}) != len(references)
    ):
        raise ValueError("validation voices must have unique identities, paths, and hashes")
    return identified


def assign_voices(
    rows: Sequence[HardNumberRow],
    voices: Sequence[object],
    *,
    seed: int = 42,
) -> list[ValidationAssignment]:
    """Assign exactly 100 prompts to each of twenty voices deterministically."""
    if type(seed) is not int or seed != 42:
        raise ValueError("assignment seed must be exactly 42")
    if len(rows) != HARD_NUMBER_COUNT or len({row.id for row in rows}) != len(rows):
        raise ValueError(
            f"hard-number assignments require exactly {HARD_NUMBER_COUNT} unique rows"
        )

    ordered_rows = sorted(rows, key=lambda row: row.id.strip())
    identified_voices = sorted(_validate_voices(voices), key=lambda item: item[0])
    random.Random(seed).shuffle(identified_voices)
    generation_config = ValidationGenerationConfig()

    assignments: list[ValidationAssignment] = []
    for index, row in enumerate(ordered_rows):
        voice_id, voice = identified_voices[index % VALIDATION_VOICE_COUNT]
        reference = _reference_from_voice(voice)
        assignments.append(
            ValidationAssignment(
                dataset_repo=HARD_NUMBER_REPO,
                dataset_revision=HARD_NUMBER_REVISION,
                dataset_filename=HARD_NUMBER_FILENAME,
                id=row.id,
                category=row.category,
                hard_number=row.hard_number,
                text=row.text,
                normalized_gold=row.normalized_gold,
                stressed=row.stressed,
                voice_id=voice_id,
                reference_role=reference.role,
                reference_source_relative_path=reference.source_relative_path,
                reference_text=reference.text,
                reference_schema_version=reference.schema_version,
                reference_seed=reference.seed,
                reference_source_shard=reference.source_shard,
                reference_member_name=reference.member_name,
                reference_audio_path=str(Path(reference.audio_path).resolve()),
                reference_source_sha256=reference.source_sha256,
                reference_wav_sha256=reference.wav_sha256,
                reference_sample_rate=reference.sample_rate,
                reference_channels=reference.channels,
                reference_duration=float(reference.duration),
                reference_duration_tier=reference.duration_tier,
                generation_config=generation_config,
            )
        )
    return assignments


def partition_assignments(
    assignments: Sequence[ValidationAssignment], rank: int, world_size: int
) -> list[ValidationAssignment]:
    """Return the deterministic stride partition for one distributed rank."""
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not 0 <= rank < world_size:
        raise ValueError("rank must satisfy 0 <= rank < world_size")
    return list(assignments[rank::world_size])


def _validate_assignment_manifest_contract(
    assignments: Sequence[ValidationAssignment],
) -> list[ValidationAssignment]:
    if len(assignments) != HARD_NUMBER_COUNT:
        raise ValueError(
            f"assignment manifest must contain exactly {HARD_NUMBER_COUNT} rows; "
            f"found {len(assignments)}"
        )
    ordered = sorted(assignments, key=lambda item: item.id)
    if len({assignment.id for assignment in ordered}) != len(ordered):
        raise ValueError("assignment manifest contains duplicate hard-number IDs")

    expected_provenance = (
        HARD_NUMBER_REPO,
        HARD_NUMBER_REVISION,
        HARD_NUMBER_FILENAME,
    )
    expected_generation_config = ValidationGenerationConfig()
    for assignment in ordered:
        if (
            assignment.dataset_repo,
            assignment.dataset_revision,
            assignment.dataset_filename,
        ) != expected_provenance:
            raise ValueError("assignment dataset provenance is not exactly pinned")
        if assignment.generation_config != expected_generation_config:
            raise ValueError("assignment generation config is not exactly fixed")
        for field in REQUIRED_FIELDS:
            _require_nonblank(getattr(assignment, field), f"dataset field {field}")

    voice_counts = Counter(assignment.voice_id for assignment in ordered)
    if len(voice_counts) != VALIDATION_VOICE_COUNT or set(voice_counts.values()) != {
        ASSIGNMENTS_PER_VOICE
    }:
        raise ValueError("assignment manifest must contain 20 voices with 100 rows each")
    for voice_id in voice_counts:
        _require_nonblank(voice_id, "voice_id")

    references: dict[str, _ReferenceMetadata] = {}
    for assignment in ordered:
        reference = _reference_from_assignment(assignment)
        previous = references.setdefault(assignment.voice_id, reference)
        if previous != reference:
            raise ValueError("reference fields must be stable for each voice_id")

    for voice_id, reference in references.items():
        _validate_reference(reference, voice_id)
    reference_values = list(references.values())
    resolved_audio_paths = [
        str(Path(reference.audio_path).resolve()) for reference in reference_values
    ]
    if (
        len({item.source_relative_path for item in reference_values})
        != VALIDATION_VOICE_COUNT
        or len(set(resolved_audio_paths)) != VALIDATION_VOICE_COUNT
        or len({item.source_sha256 for item in reference_values})
        != VALIDATION_VOICE_COUNT
        or len({item.wav_sha256 for item in reference_values})
        != VALIDATION_VOICE_COUNT
    ):
        raise ValueError("validation voices must have unique identities, paths, and hashes")

    shuffled_voice_ids = sorted(references)
    random.Random(42).shuffle(shuffled_voice_ids)
    for index, assignment in enumerate(ordered):
        expected_voice_id = shuffled_voice_ids[index % VALIDATION_VOICE_COUNT]
        if assignment.voice_id != expected_voice_id:
            raise ValueError("assignment manifest does not use canonical voice round-robin")
    return ordered


def write_assignment_manifest(
    assignments: Sequence[ValidationAssignment], path: str | Path
) -> Path:
    """Atomically write assignments as canonical, ID-ordered JSONL."""
    ordered_assignments = _validate_assignment_manifest_contract(assignments)

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            for assignment in ordered_assignments:
                output.write(
                    json.dumps(
                        asdict(assignment),
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination
