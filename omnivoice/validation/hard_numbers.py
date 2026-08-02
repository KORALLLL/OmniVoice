"""Prepare reproducible hard-number prompts and Balalaika voice assignments."""

from __future__ import annotations

import json
import os
import random
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

HARD_NUMBER_REPO = "bitmanagerai/hard_number_eval_for_tts"
HARD_NUMBER_REVISION = "57b964492ccfcedd6a24d0225ef4b7d3697ffdca"
HARD_NUMBER_FILENAME = "hard_number_eval.jsonl"
HARD_NUMBER_COUNT = 2_000
VALIDATION_VOICE_COUNT = 20
REQUIRED_FIELDS = (
    "id",
    "category",
    "hard_number",
    "text",
    "normalized_gold",
    "stressed",
)


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
    seen_ids: set[str] = set()
    for voice in voices:
        voice_id = _voice_id(voice)
        if voice_id in seen_ids:
            raise ValueError(f"duplicate voice id {voice_id!r}")
        seen_ids.add(voice_id)
        if _required_voice_value(voice, "role") != "validation_voice":
            raise ValueError(f"voice {voice_id!r} role must be 'validation_voice'")
        identified.append((voice_id, voice))
    return identified


def assign_voices(
    rows: Sequence[HardNumberRow],
    voices: Sequence[object],
    *,
    seed: int = 42,
) -> list[ValidationAssignment]:
    """Assign exactly 100 prompts to each of twenty voices deterministically."""
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
        assignments.append(
            ValidationAssignment(
                dataset_repo=HARD_NUMBER_REPO,
                dataset_revision=HARD_NUMBER_REVISION,
                id=row.id,
                category=row.category,
                hard_number=row.hard_number,
                text=row.text,
                normalized_gold=row.normalized_gold,
                stressed=row.stressed,
                voice_id=voice_id,
                reference_role=_required_voice_value(voice, "role"),
                reference_source_relative_path=_required_voice_value(
                    voice, "source_relative_path"
                ),
                reference_text=_required_voice_value(voice, "text"),
                reference_schema_version=_required_voice_value(voice, "schema_version"),
                reference_seed=_required_voice_value(voice, "seed"),
                reference_source_shard=_required_voice_value(voice, "source_shard"),
                reference_member_name=_required_voice_value(voice, "member_name"),
                reference_audio_path=_required_voice_value(voice, "audio_path"),
                reference_source_sha256=_required_voice_value(voice, "source_sha256"),
                reference_wav_sha256=_required_voice_value(voice, "wav_sha256"),
                reference_sample_rate=_required_voice_value(voice, "sample_rate"),
                reference_channels=_required_voice_value(voice, "channels"),
                reference_duration=_required_voice_value(voice, "duration"),
                reference_duration_tier=_required_voice_value(voice, "duration_tier"),
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


def write_assignment_manifest(
    assignments: Sequence[ValidationAssignment], path: str | Path
) -> Path:
    """Atomically write assignments as canonical, ID-ordered JSONL."""
    if len(assignments) != HARD_NUMBER_COUNT:
        raise ValueError(
            f"assignment manifest must contain exactly {HARD_NUMBER_COUNT} rows; "
            f"found {len(assignments)}"
        )
    if len({assignment.id for assignment in assignments}) != len(assignments):
        raise ValueError("assignment manifest contains duplicate hard-number IDs")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            for assignment in sorted(assignments, key=lambda item: item.id):
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
