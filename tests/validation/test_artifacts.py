from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from omnivoice.validation.artifacts import (
    AtomicJsonlLedger,
    CoverageError,
    ValidationPaths,
    merge_rank_ledgers,
    require_exact_coverage,
    valid_completed_ids,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def test_ledger_reloads_one_latest_record_per_key_and_validates_wav_hash(
    tmp_path: Path,
) -> None:
    valid_wav = tmp_path / "valid.wav"
    valid_wav.write_bytes(b"RIFF-valid-audio")
    corrupt_wav = tmp_path / "corrupt.wav"
    corrupt_wav.write_bytes(b"RIFF-corrupt-audio")
    ledger_path = tmp_path / "rank-0.jsonl"
    ledger = AtomicJsonlLedger(ledger_path, key="id")

    ledger.upsert({"id": "2", "error": "decode failed"})
    ledger.upsert({"id": "1", "wav": str(valid_wav), "sha256": _sha256(valid_wav)})
    ledger.upsert({"id": "3", "wav": str(corrupt_wav), "sha256": "0" * 64})
    ledger.upsert({"id": "2", "error": "retry failed"})

    reloaded = AtomicJsonlLedger(ledger_path, key="id")
    assert reloaded.records == [
        {"id": "1", "sha256": _sha256(valid_wav), "wav": str(valid_wav)},
        {"error": "retry failed", "id": "2"},
        {"id": "3", "sha256": "0" * 64, "wav": str(corrupt_wav)},
    ]
    assert reloaded.successful_ids(required_files=("wav",)) == {"1"}
    assert ledger_path.read_text(encoding="utf-8").count('"id":') == 3


def test_asr_completion_requires_string_hypothesis_and_absent_error() -> None:
    records = [
        {"id": "empty-is-valid", "hypothesis": ""},
        {"id": "normal", "hypothesis": "двадцать один"},
        {"id": "failed", "hypothesis": "partial", "error": "timeout"},
        {"id": "not-string", "hypothesis": None},
        {"id": "missing"},
    ]

    assert valid_completed_ids(records) == {"empty-is-valid", "normal"}


def test_file_completion_rejects_missing_digest_missing_file_and_error(
    tmp_path: Path,
) -> None:
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"audio")
    records = [
        {"id": "valid", "wav": str(wav), "sha256": _sha256(wav)},
        {"id": "no-hash", "wav": str(wav)},
        {"id": "missing", "wav": str(tmp_path / "missing.wav"), "sha256": "0" * 64},
        {"id": "error", "wav": str(wav), "sha256": _sha256(wav), "error": None},
    ]

    assert valid_completed_ids(records, required_files=("wav",)) == {"valid"}


def test_truncated_or_malformed_existing_ledger_is_rejected_without_rewrite(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "rank-0.jsonl"
    original = b'{"id":"1","hypothesis":"ok"}\n{"id":"2"'
    ledger_path.write_bytes(original)

    with pytest.raises(ValueError, match=r"invalid JSON.*line 2"):
        AtomicJsonlLedger(ledger_path, key="id")

    assert ledger_path.read_bytes() == original


def test_existing_ledger_rejects_duplicate_keys_before_collapsing(tmp_path: Path) -> None:
    ledger_path = tmp_path / "rank-0.jsonl"
    _write_jsonl(
        ledger_path,
        [{"id": "same", "hypothesis": "first"}, {"id": "same", "hypothesis": "second"}],
    )

    with pytest.raises(ValueError, match="duplicate ledger key 'same'.*line 2"):
        AtomicJsonlLedger(ledger_path, key="id")


def test_upsert_flushes_file_and_directory_and_atomically_replaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger_path = tmp_path / "rank-0.jsonl"
    calls: list[tuple[str, object]] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def recording_fsync(fd: int) -> None:
        calls.append(("fsync", fd))
        real_fsync(fd)

    def recording_replace(source: str | bytes | os.PathLike[str] | os.PathLike[bytes], destination: str | bytes | os.PathLike[str] | os.PathLike[bytes]) -> None:
        calls.append(("replace", Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    monkeypatch.setattr(os, "replace", recording_replace)

    AtomicJsonlLedger(ledger_path).upsert({"id": "1", "hypothesis": "ok"})

    replace_index = next(index for index, call in enumerate(calls) if call[0] == "replace")
    assert calls[replace_index] == ("replace", ledger_path)
    assert any(call[0] == "fsync" for call in calls[:replace_index])
    assert any(call[0] == "fsync" for call in calls[replace_index + 1 :])


def test_failed_rewrite_preserves_valid_existing_artifact(tmp_path: Path) -> None:
    ledger_path = tmp_path / "rank-0.jsonl"
    ledger = AtomicJsonlLedger(ledger_path)
    ledger.upsert({"id": "1", "hypothesis": "ok"})
    original = ledger_path.read_bytes()

    with pytest.raises(TypeError):
        ledger.upsert({"id": "2", "hypothesis": {"not", "json"}})

    assert ledger_path.read_bytes() == original
    assert AtomicJsonlLedger(ledger_path).records == [
        {"hypothesis": "ok", "id": "1"}
    ]
    assert list(tmp_path.glob(".rank-0.jsonl.*.tmp")) == []


@pytest.mark.parametrize("failure_operation", ["fsync", "open", "close"])
def test_post_replace_durability_failure_keeps_memory_aligned_with_disk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_operation: str,
) -> None:
    ledger_path = tmp_path / "rank-0.jsonl"
    ledger = AtomicJsonlLedger(ledger_path)
    ledger.upsert({"id": "1", "hypothesis": "one"})
    replaced = False
    real_fsync = os.fsync
    real_open = os.open
    real_close = os.close
    real_replace = os.replace

    def track_replace(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    ) -> None:
        nonlocal replaced
        real_replace(source, destination)
        replaced = True

    def fail_fsync_after_replace(fd: int) -> None:
        if replaced and failure_operation == "fsync":
            raise OSError("injected directory fsync failure")
        real_fsync(fd)

    def fail_open_after_replace(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
    ) -> int:
        if replaced and failure_operation == "open":
            raise OSError("injected directory open failure")
        return real_open(path, flags, mode)

    def fail_close_after_replace(fd: int) -> None:
        real_close(fd)
        if replaced and failure_operation == "close":
            raise OSError("injected directory close failure")

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", track_replace)
        patch.setattr(os, "fsync", fail_fsync_after_replace)
        patch.setattr(os, "open", fail_open_after_replace)
        patch.setattr(os, "close", fail_close_after_replace)
        with pytest.raises(OSError, match=f"injected directory {failure_operation}"):
            ledger.upsert({"id": "2", "hypothesis": "two"})

    assert [record["id"] for record in ledger.records] == ["1", "2"]
    assert [record["id"] for record in AtomicJsonlLedger(ledger_path).records] == [
        "1",
        "2",
    ]

    ledger.upsert({"id": "3", "hypothesis": "three"})

    assert [record["id"] for record in AtomicJsonlLedger(ledger_path).records] == [
        "1",
        "2",
        "3",
    ]


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_existing_ledger_rejects_non_finite_json_constants(
    tmp_path: Path, constant: str
) -> None:
    ledger_path = tmp_path / "rank-0.jsonl"
    original = f'{{"id":"bad","value":{constant}}}\n'.encode()
    ledger_path.write_bytes(original)

    with pytest.raises(ValueError, match=r"invalid JSON.*non-finite"):
        AtomicJsonlLedger(ledger_path)

    assert ledger_path.read_bytes() == original


@pytest.mark.parametrize(
    ("raw", "duplicate_member"),
    [
        ('{"id":"first","id":"second","hypothesis":"ok"}\n', "id"),
        ('{"id":"one","metadata":{"value":1,"value":2}}\n', "value"),
    ],
)
def test_existing_ledger_rejects_duplicate_json_object_members(
    tmp_path: Path, raw: str, duplicate_member: str
) -> None:
    ledger_path = tmp_path / "rank-0.jsonl"
    original = raw.encode()
    ledger_path.write_bytes(original)

    with pytest.raises(
        ValueError, match=rf"invalid JSON.*duplicate object member {duplicate_member!r}"
    ):
        AtomicJsonlLedger(ledger_path)

    assert ledger_path.read_bytes() == original


def test_upsert_copies_nested_caller_values_before_storing(tmp_path: Path) -> None:
    ledger = AtomicJsonlLedger(tmp_path / "rank-0.jsonl")
    record = {"id": "1", "metadata": {"tokens": ["original"]}}

    ledger.upsert(record)
    record["metadata"]["tokens"].append("caller mutation")  # type: ignore[index]

    assert ledger.records == [
        {"id": "1", "metadata": {"tokens": ["original"]}}
    ]


def test_records_returns_nested_values_isolated_from_ledger_state(tmp_path: Path) -> None:
    ledger = AtomicJsonlLedger(tmp_path / "rank-0.jsonl")
    ledger.upsert({"id": "1", "metadata": {"tokens": ["original"]}})

    returned = ledger.records
    returned[0]["metadata"]["tokens"].append("consumer mutation")

    assert ledger.records == [
        {"id": "1", "metadata": {"tokens": ["original"]}}
    ]


def test_merge_detects_cross_rank_duplicates_before_dict_collapse(tmp_path: Path) -> None:
    rank_zero = tmp_path / "rank-0.jsonl"
    rank_one = tmp_path / "rank-1.jsonl"
    _write_jsonl(rank_zero, [{"id": "shared", "rank": 0}])
    _write_jsonl(rank_one, [{"id": "shared", "rank": 1}])

    with pytest.raises(ValueError, match="duplicate key 'shared'.*rank-1.jsonl"):
        merge_rank_ledgers([rank_zero, rank_one])


def test_merge_writes_canonical_sorted_jsonl_and_preserves_record_schema(
    tmp_path: Path,
) -> None:
    rank_zero = tmp_path / "rank-0.jsonl"
    rank_one = tmp_path / "rank-1.jsonl"
    merged_path = tmp_path / "merged" / "manifest.jsonl"
    _write_jsonl(rank_zero, [{"wav": "b.wav", "id": "b", "sha256": "b" * 64}])
    _write_jsonl(rank_one, [{"sha256": "a" * 64, "id": "a", "wav": "a.wav"}])

    merged = merge_rank_ledgers([rank_zero, rank_one], output_path=merged_path)

    assert merged == [
        {"id": "a", "sha256": "a" * 64, "wav": "a.wav"},
        {"id": "b", "sha256": "b" * 64, "wav": "b.wav"},
    ]
    assert merged_path.read_text(encoding="utf-8") == (
        '{"id":"a","sha256":"' + "a" * 64 + '","wav":"a.wav"}\n'
        '{"id":"b","sha256":"' + "b" * 64 + '","wav":"b.wav"}\n'
    )


def test_exact_coverage_reports_sorted_missing_extra_and_duplicate_ids() -> None:
    with pytest.raises(CoverageError) as captured:
        require_exact_coverage(
            ["id-3", "id-1", "id-2"],
            ["id-4", "id-2", "id-2"],
        )

    assert captured.value.missing == ["id-1", "id-3"]
    assert captured.value.extra == ["id-4"]
    assert captured.value.duplicate_count == 1
    assert str(captured.value) == (
        "exact coverage required: missing=['id-1', 'id-3']; "
        "extra=['id-4']; duplicate_count=1"
    )


def test_exact_coverage_rejects_1999_of_2000_and_accepts_exact_set() -> None:
    expected = [f"prompt-{index:04d}" for index in range(2_000)]

    with pytest.raises(CoverageError) as captured:
        require_exact_coverage(expected, expected[:-1])

    assert captured.value.missing == ["prompt-1999"]
    require_exact_coverage(expected, list(reversed(expected)))


@pytest.mark.parametrize(
    ("run_id", "step", "message"),
    [
        ("../escape", 1, "run_id"),
        ("nested/run", 1, "run_id"),
        (".", 1, "run_id"),
        ("valid", -1, "step"),
        ("valid", True, "step"),
    ],
)
def test_validation_paths_reject_path_traversal_and_invalid_steps(
    tmp_path: Path, run_id: str, step: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ValidationPaths(tmp_path, run_id, step)


def test_validation_paths_resolve_full_layout_and_preserve_existing_files(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "run-1" / "step-625" / "metrics.json"
    existing.parent.mkdir(parents=True)
    existing.write_text('{"coverage":2000}\n', encoding="utf-8")

    paths = ValidationPaths(tmp_path, "run-1", 625)

    assert paths.selection_manifest == tmp_path / "run-1" / "selection" / "voices.jsonl"
    assert paths.selection_voices == tmp_path / "run-1" / "selection" / "voices"
    assert paths.wavs == tmp_path / "run-1" / "step-625" / "wavs"
    assert paths.rank_manifest(7) == tmp_path / "run-1" / "step-625" / "rank-manifests" / "rank-7.jsonl"
    assert paths.rank_hypotheses(7) == tmp_path / "run-1" / "step-625" / "rank-hypotheses" / "rank-7.jsonl"
    assert paths.manifest == tmp_path / "run-1" / "step-625" / "manifest.jsonl"
    assert paths.hypotheses == tmp_path / "run-1" / "step-625" / "hypotheses.jsonl"
    assert paths.per_utt == tmp_path / "run-1" / "step-625" / "per_utt.tsv"
    assert paths.metrics == existing
    assert paths.report == tmp_path / "run-1" / "step-625" / "report.md"
    assert paths.run_metadata == tmp_path / "run-1" / "step-625" / "run_metadata.json"
    assert paths.wandb_ids == tmp_path / "run-1" / "wandb_ids.json"
    assert existing.read_text(encoding="utf-8") == '{"coverage":2000}\n'
    assert paths.rank_manifest(0).parent.is_dir()
    assert paths.rank_hypotheses(0).parent.is_dir()
    assert paths.wavs.is_dir()


@pytest.mark.parametrize("rank", [-1, True, "0", "../0"])
def test_validation_paths_reject_invalid_rank(tmp_path: Path, rank: object) -> None:
    paths = ValidationPaths(tmp_path, "run", 0)

    with pytest.raises(ValueError, match="rank"):
        paths.rank_manifest(rank)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rank"):
        paths.rank_hypotheses(rank)  # type: ignore[arg-type]
