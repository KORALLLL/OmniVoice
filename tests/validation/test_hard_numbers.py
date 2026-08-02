from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from omnivoice.validation.balalaika import SelectedBalalaikaClip
from omnivoice.validation.hard_numbers import (
    HARD_NUMBER_REVISION,
    assign_voices,
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
        audio_path.write_bytes(f"voice-{index}".encode())
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
                wav_sha256=f"{index + 101:064x}",
                sample_rate=24_000,
                channels=1,
                duration=3.5 + index,
                duration_tier="preferred_3_to_12s",
            )
        )
    return voices


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
        "reference_duration": 22.5,
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
        "reference_wav_sha256": f"{120:064x}",
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


@pytest.mark.parametrize(
    ("rank", "world_size", "message"),
    [(-1, 8, "rank must satisfy"), (8, 8, "rank must satisfy"), (0, 0, "world_size must be positive")],
)
def test_partition_rejects_invalid_rank(
    rank: int, world_size: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        partition_assignments([], rank, world_size)
