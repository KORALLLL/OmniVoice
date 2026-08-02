from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from omnivoice.scripts.prepare_balalaika_samples import main
from omnivoice.validation.balalaika import rank_candidates, select_and_convert


@dataclass(frozen=True)
class BalalaikaFixture:
    root: Path
    sidecar: Path
    invalid_paths: tuple[str, ...]


def _encoded_mp3(tmp_path: Path, duration: float) -> bytes:
    wav_path = tmp_path / f"source-{duration}.wav"
    mp3_path = tmp_path / f"source-{duration}.mp3"
    frames = int(24_000 * duration)
    samples = 0.05 * np.sin(2 * np.pi * 220 * np.arange(frames) / 24_000)
    sf.write(wav_path, samples.astype(np.float32), 24_000)
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(wav_path),
            "-c:a",
            "libmp3lame",
            str(mp3_path),
        ],
        check=True,
    )
    return mp3_path.read_bytes()


def _add_tar_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


@pytest.fixture
def balalaika_fixture(tmp_path: Path) -> BalalaikaFixture:
    root = tmp_path / "balalaika"
    train = root / "train"
    train.mkdir(parents=True)
    preferred_mp3 = _encoded_mp3(tmp_path, 4.0)
    fallback_mp3 = _encoded_mp3(tmp_path, 13.0)

    rows: list[dict[str, object]] = []
    members: dict[str, list[tuple[str, bytes]]] = {"000000": [], "000001": []}
    for index in range(23):
        shard = f"{index % 2:06d}"
        member = f"0_00_4_00_valid_{index:03d}.mp3"
        source_path = f"{shard}/{member}"
        rows.append(
            {
                "schema_version": 1,
                "source_relative_path": source_path,
                "rover_punctuated_accented": f"sidecar text {index}",
                "DistillMOS": -1_000_000 + index,
                "music_prob": 1_000_000 - index,
            }
        )
        members[shard].append((member, preferred_mp3))
        metadata = json.dumps(
            {
                "accent.txt": f"contradictory TAR text {index}",
                "DistillMOS": (-1) ** index * 1_000_000,
                "music_prob": (-1) ** (index + 1) * 1_000_000,
            }
        ).encode()
        members[shard].append((member.removesuffix(".mp3") + ".json", metadata))

    fallback_path = "000001/0_00_13_00_fallback.mp3"
    rows.append(
        {
            "schema_version": 1,
            "source_relative_path": fallback_path,
            "rover_punctuated_accented": "sidecar text fallback",
        }
    )
    members["000001"].append((Path(fallback_path).name, fallback_mp3))

    invalid = (
        "000000/0_00_4_00_missing.mp3",
        "000000/0_00_4_00_empty.mp3",
        "000001/0_00_4_00_undecodable.mp3",
    )
    rows.extend(
        {
            "schema_version": 1,
            "source_relative_path": source_path,
            "rover_punctuated_accented": f"sidecar text invalid {index}",
        }
        for index, source_path in enumerate(invalid)
    )
    members["000000"].append((Path(invalid[1]).name, b""))
    members["000001"].append((Path(invalid[2]).name, b"not an mp3"))

    for shard, shard_members in members.items():
        with tarfile.open(train / f"shard_{shard}.tar", "w") as archive:
            for name, payload in shard_members:
                _add_tar_bytes(archive, name, payload)

    sidecar = root / "sidecar.jsonl"
    with sidecar.open("w", encoding="utf-8") as output:
        output.write("not-json\n")
        output.write(json.dumps({"source_relative_path": "missing text"}) + "\n")
        for row in reversed(rows):
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    return BalalaikaFixture(root=root, sidecar=sidecar, invalid_paths=invalid)


def test_rank_candidates_keeps_lowest_priorities_in_each_duration_tier(
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "sidecar.jsonl"
    rows = []
    for index in range(10):
        rows.extend(
            [
                {
                    "schema_version": 1,
                    "source_relative_path": f"000000/0_00_4_00_p_{index}.mp3",
                    "rover_punctuated_accented": f"preferred {index}",
                },
                {
                    "schema_version": 1,
                    "source_relative_path": f"000000/0_00_13_00_f_{index}.mp3",
                    "rover_punctuated_accented": f"fallback {index}",
                },
            ]
        )
    sidecar.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    ranked = rank_candidates(sidecar, seed=42, pool_size=3)

    preferred = [row for row in ranked if row.estimated_duration <= 12]
    fallback = [row for row in ranked if row.estimated_duration > 12]
    expected_preferred = sorted(
        (row for row in rows if "_p_" in str(row["source_relative_path"])),
        key=lambda row: hashlib.sha256(
            f"42\0{row['source_relative_path']}".encode()
        ).digest(),
    )[:3]
    expected_fallback = sorted(
        (row for row in rows if "_f_" in str(row["source_relative_path"])),
        key=lambda row: hashlib.sha256(
            f"42\0{row['source_relative_path']}".encode()
        ).digest(),
    )[:3]
    assert [row.source_relative_path for row in preferred] == [
        str(row["source_relative_path"]) for row in expected_preferred
    ]
    assert [row.source_relative_path for row in fallback] == [
        str(row["source_relative_path"]) for row in expected_fallback
    ]


def test_rank_candidates_deduplicates_repeated_source_paths(tmp_path: Path) -> None:
    sidecar = tmp_path / "duplicates.jsonl"
    source_path = "000000/0_00_4_00_duplicate.mp3"
    sidecar.write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": 1,
                    "source_relative_path": source_path,
                    "rover_punctuated_accented": text,
                }
            )
            + "\n"
            for text in ("first authoritative text", "duplicate text")
        ),
        encoding="utf-8",
    )

    ranked = rank_candidates(sidecar, seed=42, pool_size=3)

    assert [(row.source_relative_path, row.text) for row in ranked] == [
        (source_path, "first authoritative text")
    ]


def test_select_and_convert_uses_sidecar_text_and_actual_duration_tiers(
    balalaika_fixture: BalalaikaFixture, tmp_path: Path
) -> None:
    selected = select_and_convert(
        root=balalaika_fixture.root,
        sidecar_path=balalaika_fixture.sidecar,
        output_dir=tmp_path / "selected",
        memorization_count=4,
        validation_voice_count=20,
        seed=42,
    )

    assert [row.role for row in selected].count("memorization") == 4
    assert [row.role for row in selected].count("validation_voice") == 20
    assert len({row.source_relative_path for row in selected}) == 24
    assert all(row.text.startswith("sidecar text") for row in selected)
    assert all(row.sample_rate == 24_000 and row.channels == 1 for row in selected)
    assert [row.duration_tier for row in selected[:-1]] == ["preferred_3_to_12s"] * 23
    assert selected[-1].duration_tier == "fallback_over_12s"
    assert not set(balalaika_fixture.invalid_paths) & {
        row.source_relative_path for row in selected
    }

    manifest_path = tmp_path / "selected" / "selected.jsonl"
    manifest = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    assert [row["source_relative_path"] for row in manifest] == [
        row.source_relative_path for row in selected
    ]
    assert all(Path(row["audio_path"]).is_file() for row in manifest)
    assert all(len(row["source_sha256"]) == 64 for row in manifest)
    assert all(len(row["wav_sha256"]) == 64 for row in manifest)

    before = [row.source_relative_path for row in selected]
    for tar_path in sorted((balalaika_fixture.root / "train").glob("*.tar")):
        rewritten = tar_path.with_suffix(".rewritten")
        with tarfile.open(tar_path) as source, tarfile.open(rewritten, "w") as target:
            for member in source:
                payload = source.extractfile(member).read() if member.size else b""
                if member.name.endswith(".json"):
                    payload = json.dumps(
                        {
                            "accent.txt": "different contradictory text",
                            "DistillMOS": 999_999_999,
                            "music_prob": -999_999_999,
                        }
                    ).encode()
                _add_tar_bytes(target, member.name, payload)
        rewritten.replace(tar_path)

    after = select_and_convert(
        root=balalaika_fixture.root,
        sidecar_path=balalaika_fixture.sidecar,
        output_dir=tmp_path / "selected-after-metadata-change",
        memorization_count=4,
        validation_voice_count=20,
        seed=42,
    )
    assert [row.source_relative_path for row in after] == before


def test_failed_selection_does_not_replace_existing_manifest(
    balalaika_fixture: BalalaikaFixture, tmp_path: Path
) -> None:
    sidecar = tmp_path / "missing-only.jsonl"
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_relative_path": balalaika_fixture.invalid_paths[0],
                "rover_punctuated_accented": "sidecar text missing",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "atomic"
    output_dir.mkdir()
    manifest = output_dir / "selected.jsonl"
    manifest.write_text("existing manifest\n", encoding="utf-8")

    with pytest.raises(ValueError, match="found 0 decodable clips; need 1"):
        select_and_convert(
            root=balalaika_fixture.root,
            sidecar_path=sidecar,
            output_dir=output_dir,
            memorization_count=1,
            validation_voice_count=0,
            seed=42,
        )

    assert manifest.read_text(encoding="utf-8") == "existing manifest\n"


def test_cli_prints_one_json_summary(
    balalaika_fixture: BalalaikaFixture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "cli"
    main(
        [
            "--root",
            str(balalaika_fixture.root),
            "--sidecar",
            str(balalaika_fixture.sidecar),
            "--output-dir",
            str(output_dir),
            "--memorization-count",
            "1",
            "--validation-voice-count",
            "0",
            "--seed",
            "42",
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "manifest": str(output_dir / "selected.jsonl"),
        "counts": {"memorization": 1, "validation_voice": 0},
        "duration_tiers": {"preferred_3_to_12s": 1},
    }
