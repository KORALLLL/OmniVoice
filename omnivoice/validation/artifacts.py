"""Durable, resumable artifacts for distributed validation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


def _require_key(record: Mapping[str, Any], key: str, context: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} field {key!r} must be a non-blank string")
    return value


def _canonical_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {field: record[field] for field in sorted(record)}


def _atomic_write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            for record in records:
                json.dump(
                    record,
                    destination,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_path, path)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _load_jsonl(path: Path, key: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON in {path} on line {line_number}: {error.msg}"
                ) from error
            if not isinstance(raw, dict):
                raise TypeError(
                    f"ledger {path} line {line_number} must contain a JSON object"
                )
            identifier = _require_key(raw, key, f"ledger {path} line {line_number}")
            if identifier in seen:
                raise ValueError(
                    f"duplicate ledger key {identifier!r} in {path} on line {line_number}"
                )
            seen.add(identifier)
            records.append(_canonical_record(raw))
    return records


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _record_file_is_valid(
    record: Mapping[str, Any], field: str, required_file_count: int
) -> bool:
    raw_path = record.get(field)
    if not isinstance(raw_path, str) or not raw_path:
        return False
    path = Path(raw_path)
    if not path.is_file():
        return False

    field_digest = record.get(f"{field}_sha256")
    expected_digest = (
        field_digest
        if field_digest is not None
        else record.get("sha256") if required_file_count == 1 else None
    )
    if not isinstance(expected_digest, str) or not _SHA256.fullmatch(expected_digest):
        return False
    return _sha256_file(path) == expected_digest.lower()


def valid_completed_ids(
    records: Iterable[Mapping[str, Any]],
    *,
    key: str = "id",
    required_files: Sequence[str] = (),
) -> set[str]:
    """Return IDs with complete synthesis files or complete ASR output."""
    completed: set[str] = set()
    required = tuple(required_files)
    for record_index, record in enumerate(records, start=1):
        identifier = _require_key(record, key, f"record {record_index}")
        if "error" in record:
            continue
        if required:
            if all(
                _record_file_is_valid(record, field, len(required))
                for field in required
            ):
                completed.add(identifier)
        elif isinstance(record.get("hypothesis"), str):
            completed.add(identifier)
    return completed


class AtomicJsonlLedger:
    """A JSONL ledger rewritten atomically after each keyed update."""

    def __init__(self, path: str | Path, key: str = "id") -> None:
        self.path = Path(path)
        if not isinstance(key, str) or not key.strip():
            raise ValueError("key must be a non-blank string")
        self.key = key
        loaded = _load_jsonl(self.path, self.key)
        self._records = {record[self.key]: record for record in loaded}

    @property
    def records(self) -> list[dict[str, Any]]:
        """Return canonical records ordered by the ledger key."""
        return [dict(self._records[value]) for value in sorted(self._records)]

    def upsert(self, record: Mapping[str, Any]) -> None:
        """Atomically insert or replace one record without risking the old file."""
        if not isinstance(record, Mapping):
            raise TypeError("record must be a mapping")
        identifier = _require_key(record, self.key, "record")
        replacement = dict(self._records)
        replacement[identifier] = _canonical_record(record)
        ordered = [replacement[value] for value in sorted(replacement)]
        _atomic_write_jsonl(self.path, ordered)
        self._records = replacement

    def successful_ids(self, required_files: Sequence[str] = ()) -> set[str]:
        """Return complete IDs after checking content, not only file existence."""
        return valid_completed_ids(
            self.records,
            key=self.key,
            required_files=required_files,
        )


def merge_rank_ledgers(
    rank_paths: Sequence[str | Path],
    *,
    output_path: str | Path | None = None,
    key: str = "id",
) -> list[dict[str, Any]]:
    """Merge rank ledgers in stable key order, rejecting every duplicate."""
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank_path_value in rank_paths:
        rank_path = Path(rank_path_value)
        for record in _load_jsonl(rank_path, key):
            identifier = record[key]
            if identifier in seen:
                raise ValueError(
                    f"duplicate key {identifier!r} while merging {rank_path}"
                )
            seen.add(identifier)
            merged.append(record)
    merged.sort(key=lambda record: record[key])
    if output_path is not None:
        _atomic_write_jsonl(Path(output_path), merged)
    return merged


@dataclass(eq=False)
class CoverageError(ValueError):
    """Exact-coverage failure details suitable for deterministic reporting."""

    missing: list[str]
    extra: list[str]
    duplicate_count: int

    def __str__(self) -> str:
        return (
            f"exact coverage required: missing={self.missing}; extra={self.extra}; "
            f"duplicate_count={self.duplicate_count}"
        )


def require_exact_coverage(
    expected_ids: Collection[str], actual_ids: Collection[str]
) -> None:
    """Reject missing, extra, or duplicated actual benchmark IDs."""
    expected = set(expected_ids)
    actual = set(actual_ids)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    duplicate_count = len(actual_ids) - len(actual)
    if missing or extra or duplicate_count:
        raise CoverageError(
            missing=missing,
            extra=extra,
            duplicate_count=duplicate_count,
        )


@dataclass(frozen=True)
class ValidationPaths:
    """Validated filesystem layout for one logical validation step."""

    root: Path
    run_id: str
    step: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        if not isinstance(self.run_id, str) or not _RUN_ID.fullmatch(self.run_id):
            raise ValueError(
                "run_id must be one safe path component containing only "
                "letters, digits, '.', '_', or '-'"
            )
        if type(self.step) is not int or self.step < 0:
            raise ValueError("step must be a non-negative integer")
        for directory in (
            self.selection_voices,
            self.wavs,
            self.rank_manifests,
            self.rank_hypothesis_ledgers,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def run_dir(self) -> Path:
        return self.root / self.run_id

    @property
    def selection_dir(self) -> Path:
        return self.run_dir / "selection"

    @property
    def selection_manifest(self) -> Path:
        return self.selection_dir / "voices.jsonl"

    @property
    def selection_voices(self) -> Path:
        return self.selection_dir / "voices"

    @property
    def step_dir(self) -> Path:
        return self.run_dir / f"step-{self.step}"

    @property
    def wavs(self) -> Path:
        return self.step_dir / "wavs"

    @property
    def rank_manifests(self) -> Path:
        return self.step_dir / "rank-manifests"

    @property
    def rank_hypothesis_ledgers(self) -> Path:
        return self.step_dir / "rank-hypotheses"

    @staticmethod
    def _validate_rank(rank: int) -> None:
        if type(rank) is not int or rank < 0:
            raise ValueError("rank must be a non-negative integer")

    def rank_manifest(self, rank: int) -> Path:
        self._validate_rank(rank)
        return self.rank_manifests / f"rank-{rank}.jsonl"

    def rank_hypotheses(self, rank: int) -> Path:
        self._validate_rank(rank)
        return self.rank_hypothesis_ledgers / f"rank-{rank}.jsonl"

    @property
    def manifest(self) -> Path:
        return self.step_dir / "manifest.jsonl"

    @property
    def hypotheses(self) -> Path:
        return self.step_dir / "hypotheses.jsonl"

    @property
    def per_utt(self) -> Path:
        return self.step_dir / "per_utt.tsv"

    @property
    def metrics(self) -> Path:
        return self.step_dir / "metrics.json"

    @property
    def report(self) -> Path:
        return self.step_dir / "report.md"

    @property
    def run_metadata(self) -> Path:
        return self.step_dir / "run_metadata.json"

    @property
    def wandb_ids(self) -> Path:
        return self.run_dir / "wandb_ids.json"
