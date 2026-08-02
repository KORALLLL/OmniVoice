from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf

from omnivoice.validation.balalaika import SelectedBalalaikaClip
from omnivoice.validation.hard_numbers import (
    HARD_NUMBER_FILENAME,
    HARD_NUMBER_REPO,
    HARD_NUMBER_REVISION,
    assign_voices,
    download_hard_number_jsonl,
    load_hard_number_rows,
    partition_assignments,
    write_assignment_manifest,
)


def _hard_number_row(index: int) -> dict[str, str]:
    return {
        "id": f"prompt-{index:04d}",
        "category": "integer",
        "hard_number": str(index),
        "text": f"У меня {index} примеров.",
        "normalized_gold": f"у меня число {index} примеров",
        "stressed": f"У меня число {index} примеров.",
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _voices(tmp_path: Path) -> list[SelectedBalalaikaClip]:
    voices = []
    for index in range(20):
        audio_path = tmp_path / f"voice-{index:02d}.wav"
        sf.write(
            audio_path,
            np.full(72_000, index / 100.0, dtype=np.float32),
            24_000,
            subtype="PCM_16",
        )
        wav_sha256 = hashlib.sha256(audio_path.read_bytes()).hexdigest()
        voices.append(
            SelectedBalalaikaClip(
                role="validation_voice",
                source_relative_path=f"{index:06d}/voice.mp3",
                text=f"Референс {index}",
                schema_version=1 + index % 2,
                seed=42,
                source_shard=f"train/shard_{index:06d}.tar",
                member_name="voice.mp3",
                audio_path=str(audio_path),
                source_sha256=f"{index + 1:064x}",
                wav_sha256=wav_sha256,
                sample_rate=24_000,
                channels=1,
                duration=3.0,
                duration_tier="preferred_3_to_12s",
            )
        )
    return voices


def _assignments(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    dataset_path = tmp_path / "hard_number_eval.jsonl"
    _write_jsonl(dataset_path, [_hard_number_row(i) for i in range(2000)])
    voices = _voices(tmp_path)
    return assign_voices(load_hard_number_rows(dataset_path), voices), voices


def test_load_assign_partition_balances_all_rows(tmp_path: Path) -> None:
    dataset_path = tmp_path / "hard_number_eval.jsonl"
    _write_jsonl(dataset_path, list(reversed([_hard_number_row(i) for i in range(2000)])))
    voices = _voices(tmp_path)

    rows = load_hard_number_rows(dataset_path, expected_count=2000)
    assignments = assign_voices(rows, voices, seed=42)

    assert [row.id for row in rows[:3]] == [
        "prompt-0000",
        "prompt-0001",
        "prompt-0002",
    ]
    assert len(assignments) == 2000
    assert Counter(item.voice_id for item in assignments) == {
        voice.source_relative_path: 100 for voice in voices
    }
    assert [
        len(partition_assignments(assignments, rank, 8)) for rank in range(8)
    ] == [250] * 8
    assert [item.id for item in partition_assignments(assignments, 3, 8)[:2]] == [
        "prompt-0003",
        "prompt-0011",
    ]


def test_load_normalizes_numeric_dataset_ids_to_sorted_strings(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = [_hard_number_row(i) for i in range(2000)]
    for index, row in enumerate(rows, start=1):
        row["id"] = index
    path = tmp_path / "numeric-ids.jsonl"
    _write_jsonl(path, rows)

    loaded = load_hard_number_rows(path)

    assert [row.id for row in loaded[:3]] == ["1", "10", "100"]


def test_download_uses_exact_pinned_hub_coordinates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloaded = tmp_path / HARD_NUMBER_FILENAME
    hub_download = Mock(return_value=str(downloaded))
    monkeypatch.setattr(
        "omnivoice.validation.hard_numbers.hf_hub_download", hub_download
    )

    result = download_hard_number_jsonl()

    assert result == downloaded
    hub_download.assert_called_once_with(
        repo_id=HARD_NUMBER_REPO,
        repo_type="dataset",
        filename=HARD_NUMBER_FILENAME,
        revision=HARD_NUMBER_REVISION,
    )


def test_manifest_is_canonical_and_serializes_complete_inputs(tmp_path: Path) -> None:
    rows_in_order = [_hard_number_row(i) for i in range(2000)]
    first_jsonl = tmp_path / "first.jsonl"
    second_jsonl = tmp_path / "second.jsonl"
    _write_jsonl(first_jsonl, rows_in_order)
    _write_jsonl(second_jsonl, list(reversed(rows_in_order)))
    voices = _voices(tmp_path)

    first_assignments = assign_voices(load_hard_number_rows(first_jsonl), voices)
    second_assignments = assign_voices(
        load_hard_number_rows(second_jsonl), list(reversed(voices))
    )
    first_manifest = tmp_path / "assignments-a.jsonl"
    second_manifest = tmp_path / "assignments-b.jsonl"
    write_assignment_manifest(first_assignments, first_manifest)
    write_assignment_manifest(list(reversed(second_assignments)), second_manifest)

    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    record = json.loads(first_manifest.read_text(encoding="utf-8").splitlines()[0])
    assert record == {
        "category": "integer",
        "dataset_filename": HARD_NUMBER_FILENAME,
        "dataset_repo": "bitmanagerai/hard_number_eval_for_tts",
        "dataset_revision": HARD_NUMBER_REVISION,
        "generation_config": {
            "audio_chunk_duration": 15.0,
            "audio_chunk_threshold": 30.0,
            "class_temperature": 0.0,
            "denoise": True,
            "fade_duration": 0.1,
            "guidance_scale": 2.0,
            "language": "Russian",
            "layer_penalty_factor": 5.0,
            "num_step": 32,
            "pad_duration": 0.1,
            "position_temperature": 0.0,
            "postprocess_output": True,
            "preprocess_prompt": True,
            "t_shift": 0.1,
        },
        "hard_number": "0",
        "id": "prompt-0000",
        "normalized_gold": "у меня число 0 примеров",
        "reference_audio_path": str(tmp_path / "voice-19.wav"),
        "reference_channels": 1,
        "reference_duration": 3.0,
        "reference_duration_tier": "preferred_3_to_12s",
        "reference_member_name": "voice.mp3",
        "reference_role": "validation_voice",
        "reference_sample_rate": 24_000,
        "reference_schema_version": 2,
        "reference_seed": 42,
        "reference_source_relative_path": "000019/voice.mp3",
        "reference_source_sha256": f"{20:064x}",
        "reference_source_shard": "train/shard_000019.tar",
        "reference_text": "Референс 19",
        "reference_wav_sha256": voices[19].wav_sha256,
        "stressed": "У меня число 0 примеров.",
        "text": "У меня 0 примеров.",
        "voice_id": "000019/voice.mp3",
    }


def test_load_rejects_duplicate_ids(tmp_path: Path) -> None:
    rows = [_hard_number_row(i) for i in range(2000)]
    rows[-1]["id"] = rows[0]["id"]
    path = tmp_path / "duplicates.jsonl"
    _write_jsonl(path, rows)

    with pytest.raises(ValueError, match="duplicate hard-number id 'prompt-0000'"):
        load_hard_number_rows(path)


@pytest.mark.parametrize("field", ["id", "category", "hard_number", "text", "normalized_gold", "stressed"])
@pytest.mark.parametrize("value", [None, "   "])
def test_load_rejects_missing_or_blank_required_fields(
    tmp_path: Path, field: str, value: object
) -> None:
    rows = [_hard_number_row(i) for i in range(2000)]
    if value is None:
        rows[4].pop(field)
    else:
        rows[4][field] = value  # type: ignore[assignment]
    path = tmp_path / f"invalid-{field}.jsonl"
    _write_jsonl(path, rows)

    with pytest.raises(ValueError, match=rf"line 5.*{field}.*non-blank string"):
        load_hard_number_rows(path)


def test_load_rejects_unpinned_revision_and_non_2000_contract(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    _write_jsonl(path, [_hard_number_row(i) for i in range(2000)])

    with pytest.raises(ValueError, match="dataset revision must be pinned"):
        load_hard_number_rows(path, revision="main")
    with pytest.raises(ValueError, match="expected_count must be exactly 2000"):
        load_hard_number_rows(path, expected_count=1999)


def test_load_rejects_malformed_json_and_wrong_actual_count(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON on line 1"):
        load_hard_number_rows(malformed)

    short = tmp_path / "short.jsonl"
    _write_jsonl(short, [_hard_number_row(i) for i in range(1999)])
    with pytest.raises(ValueError, match="must contain exactly 2000 rows; found 1999"):
        load_hard_number_rows(short)


def test_assign_rejects_invalid_voice_sets(tmp_path: Path) -> None:
    rows_path = tmp_path / "rows.jsonl"
    _write_jsonl(rows_path, [_hard_number_row(i) for i in range(2000)])
    rows = load_hard_number_rows(rows_path)
    voices = _voices(tmp_path)

    with pytest.raises(ValueError, match="exactly 20 validation voices"):
        assign_voices(rows, voices[:19])
    with pytest.raises(ValueError, match="duplicate voice id"):
        assign_voices(rows, [*voices[:19], replace(voices[18], audio_path=voices[19].audio_path)])
    with pytest.raises(ValueError, match="role must be 'validation_voice'"):
        assign_voices(rows, [replace(voices[0], role="memorization"), *voices[1:]])


def test_assign_requires_fixed_assignment_and_reference_seed(tmp_path: Path) -> None:
    rows_path = tmp_path / "rows.jsonl"
    _write_jsonl(rows_path, [_hard_number_row(i) for i in range(2000)])
    rows = load_hard_number_rows(rows_path)
    voices = _voices(tmp_path)

    with pytest.raises(ValueError, match="assignment seed must be exactly 42"):
        assign_voices(rows, voices, seed=41)
    with pytest.raises(ValueError, match="reference seed must be exactly 42"):
        assign_voices(rows, [replace(voices[0], seed=41), *voices[1:]])


@pytest.mark.parametrize("audio_case", ["missing", "corrupt"])
def test_assign_rejects_missing_or_corrupt_reference_wav(
    tmp_path: Path, audio_case: str
) -> None:
    rows_path = tmp_path / "rows.jsonl"
    _write_jsonl(rows_path, [_hard_number_row(i) for i in range(2000)])
    rows = load_hard_number_rows(rows_path)
    voices = _voices(tmp_path)
    path = Path(voices[0].audio_path)
    if audio_case == "missing":
        path.unlink()
    else:
        path.write_bytes(b"not a WAV")

    with pytest.raises((FileNotFoundError, ValueError), match="WAV|voice-00"):
        assign_voices(rows, voices)


@pytest.mark.parametrize("metadata_case", ["hash", "rate", "duration", "tier"])
def test_assign_rejects_reference_audio_metadata_mismatch(
    tmp_path: Path, metadata_case: str
) -> None:
    rows_path = tmp_path / "rows.jsonl"
    _write_jsonl(rows_path, [_hard_number_row(i) for i in range(2000)])
    rows = load_hard_number_rows(rows_path)
    voices = _voices(tmp_path)
    if metadata_case == "hash":
        voices[0] = replace(voices[0], wav_sha256="0" * 64)
    elif metadata_case == "rate":
        path = Path(voices[0].audio_path)
        sf.write(path, np.zeros(48_000), 16_000, subtype="PCM_16")
        voices[0] = replace(
            voices[0], wav_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
    elif metadata_case == "duration":
        voices[0] = replace(voices[0], duration=4.0)
    else:
        voices[0] = replace(voices[0], duration_tier="fallback_over_12s")

    with pytest.raises(ValueError, match="hash|24000|duration|tier"):
        assign_voices(rows, voices)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("text", " ", "text"),
        ("schema_version", 0, "schema_version"),
        ("source_sha256", "bad", "source_sha256"),
        ("source_relative_path", "bad/path", "source_relative_path"),
        ("source_shard", "train/shard_999999.tar", "source_shard"),
        ("member_name", "different.mp3", "member_name"),
    ],
)
def test_assign_rejects_invalid_reference_provenance(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    rows_path = tmp_path / "rows.jsonl"
    _write_jsonl(rows_path, [_hard_number_row(i) for i in range(2000)])
    rows = load_hard_number_rows(rows_path)
    voices = _voices(tmp_path)
    voices[0] = replace(voices[0], **{field: value})

    with pytest.raises(ValueError, match=message):
        assign_voices(rows, voices)


@pytest.mark.parametrize("duplicate", ["audio_path", "source_sha256", "wav_sha256"])
def test_assign_rejects_duplicate_reference_paths_and_hashes(
    tmp_path: Path, duplicate: str
) -> None:
    rows_path = tmp_path / "rows.jsonl"
    _write_jsonl(rows_path, [_hard_number_row(i) for i in range(2000)])
    rows = load_hard_number_rows(rows_path)
    voices = _voices(tmp_path)
    if duplicate == "audio_path":
        voices[1] = replace(
            voices[1],
            audio_path=voices[0].audio_path,
            wav_sha256=voices[0].wav_sha256,
        )
    elif duplicate == "source_sha256":
        voices[1] = replace(voices[1], source_sha256=voices[0].source_sha256)
    else:
        first_bytes = Path(voices[0].audio_path).read_bytes()
        Path(voices[1].audio_path).write_bytes(first_bytes)
        voices[1] = replace(voices[1], wav_sha256=voices[0].wav_sha256)

    with pytest.raises(ValueError, match="unique.*paths.*hashes"):
        assign_voices(rows, voices)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_repo", "someone/else"),
        ("dataset_revision", "main"),
        ("dataset_filename", "other.jsonl"),
    ],
)
def test_writer_rejects_noncanonical_dataset_provenance(
    tmp_path: Path, field: str, value: str
) -> None:
    assignments, _ = _assignments(tmp_path)
    object.__setattr__(assignments[0], field, value)

    with pytest.raises(ValueError, match="dataset provenance"):
        write_assignment_manifest(assignments, tmp_path / "assignments.jsonl")


def test_writer_rejects_altered_generation_config(tmp_path: Path) -> None:
    assignments, _ = _assignments(tmp_path)
    assignments[0] = replace(
        assignments[0],
        generation_config=replace(
            assignments[0].generation_config, guidance_scale=3.0
        ),
    )

    with pytest.raises(ValueError, match="generation config"):
        write_assignment_manifest(assignments, tmp_path / "assignments.jsonl")


@pytest.mark.parametrize(
    ("assignment_index", "mutated_id", "message"),
    [
        (1, " prompt-0001", "dataset ID must be a canonical normalized string"),
        (1, "prompt-0001 ", "dataset ID must be a canonical normalized string"),
        (1, "prompt-0000 ", "duplicate normalized hard-number IDs"),
        (1, 1, "dataset ID must be a canonical normalized string"),
    ],
)
def test_writer_rejects_noncanonical_or_normalized_collision_ids(
    tmp_path: Path, assignment_index: int, mutated_id: object, message: str
) -> None:
    assignments, _ = _assignments(tmp_path)
    assignments[assignment_index] = replace(
        assignments[assignment_index], id=mutated_id
    )

    with pytest.raises(ValueError, match=message):
        write_assignment_manifest(assignments, tmp_path / "assignments.jsonl")


@pytest.mark.parametrize("distribution", ["one_voice", "imbalanced"])
def test_writer_rejects_non_balanced_voice_distribution(
    tmp_path: Path, distribution: str
) -> None:
    assignments, _ = _assignments(tmp_path)
    reference_fields = [
        field
        for field in assignments[0].__dataclass_fields__
        if field.startswith("reference_")
    ]
    if distribution == "one_voice":
        source = assignments[0]
        assignments = [
            replace(
                assignment,
                voice_id=source.voice_id,
                **{field: getattr(source, field) for field in reference_fields},
            )
            for assignment in assignments
        ]
    else:
        source = assignments[1]
        assignments[0] = replace(
            assignments[0],
            voice_id=source.voice_id,
            **{field: getattr(source, field) for field in reference_fields},
        )

    with pytest.raises(ValueError, match="20 voices.*100"):
        write_assignment_manifest(assignments, tmp_path / "assignments.jsonl")


def test_writer_rejects_unstable_reference_and_blank_dataset_fields(
    tmp_path: Path,
) -> None:
    assignments, _ = _assignments(tmp_path)
    same_voice = next(
        index
        for index, assignment in enumerate(assignments[1:], start=1)
        if assignment.voice_id == assignments[0].voice_id
    )
    assignments[same_voice] = replace(
        assignments[same_voice], reference_text="different reference"
    )
    with pytest.raises(ValueError, match="reference fields must be stable"):
        write_assignment_manifest(assignments, tmp_path / "unstable.jsonl")

    assignments, _ = _assignments(tmp_path / "blank")
    assignments[0] = replace(assignments[0], stressed=" ")
    with pytest.raises(ValueError, match="dataset field.*stressed"):
        write_assignment_manifest(assignments, tmp_path / "blank.jsonl")


def test_writer_revalidates_reference_audio_after_assignment(tmp_path: Path) -> None:
    assignments, voices = _assignments(tmp_path)
    Path(voices[0].audio_path).write_bytes(b"corrupted after assignment")

    with pytest.raises(ValueError, match="WAV|hash"):
        write_assignment_manifest(assignments, tmp_path / "assignments.jsonl")


@pytest.mark.parametrize(
    ("rank", "world_size", "message"),
    [(-1, 8, "rank must satisfy"), (8, 8, "rank must satisfy"), (0, 0, "world_size must be positive")],
)
def test_partition_rejects_invalid_rank(
    rank: int, world_size: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        partition_assignments([], rank, world_size)
