import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

from omnivoice.scripts.select_memorization_samples import select_records


ROOT = Path(__file__).resolve().parents[2]


def _write_manifest(tmp_path: Path, rows: list[object]) -> Path:
    manifest = tmp_path / "source.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return manifest


def _valid_row(tmp_path: Path, record_id: str) -> dict[str, str]:
    audio_path = tmp_path / f"{record_id}.wav"
    audio_path.touch()
    return {
        "id": record_id,
        "audio_path": str(audio_path),
        "text": f"text for {record_id}",
    }


def test_select_records_returns_four_unique_valid_records(tmp_path):
    valid_rows = [_valid_row(tmp_path, f"valid-{index}") for index in range(5)]
    rows = [
        valid_rows[0],
        {**valid_rows[0], "text": "duplicate"},
        {
            "id": "missing-audio",
            "audio_path": str(tmp_path / "missing.wav"),
            "text": "x",
        },
        {"id": "empty-text", "audio_path": valid_rows[1]["audio_path"], "text": "  "},
        ["not", "an", "object"],
        *valid_rows[1:],
    ]
    manifest = _write_manifest(tmp_path, rows)

    first = select_records(manifest, count=4, seed=42)
    second = select_records(manifest, count=4, seed=42)

    assert first == second
    assert len(first) == 4
    assert len({row["id"] for row in first}) == 4
    assert all(Path(row["audio_path"]).is_file() for row in first)


def test_select_records_reports_available_count(tmp_path):
    manifest = _write_manifest(
        tmp_path, [_valid_row(tmp_path, f"valid-{index}") for index in range(3)]
    )

    with pytest.raises(ValueError, match="found 3 valid unique records; need 4"):
        select_records(manifest, count=4, seed=42)


def test_memorization_launcher_preserves_relative_source_path_across_repo_chdir(
    tmp_path,
):
    source = tmp_path / "source.jsonl"
    source.touch()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    command_log = tmp_path / "commands.log"

    for stub_name in ("python", "accelerate"):
        stub = bin_dir / stub_name
        stub.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$COMMAND_LOG"\n')
        stub.chmod(0o755)

    result = subprocess.run(
        ["bash", str(ROOT / "examples/run_lora_memorization.sh")],
        cwd=tmp_path,
        env=os.environ
        | {
            "COMMAND_LOG": str(command_log),
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "SOURCE_JSONL": source.name,
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    selector_command = shlex.split(command_log.read_text().splitlines()[0])
    source_index = selector_command.index("--input-jsonl") + 1
    assert selector_command[source_index] == str(source)
