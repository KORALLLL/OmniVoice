"""Deterministic scoring results and local hard-number validation reports."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, TextIO

from omnivoice.validation.artifacts import ValidationPaths, require_exact_coverage
from omnivoice.validation.hard_numbers import HARD_NUMBER_COUNT
from omnivoice.validation.metrics import (
    AggregateBlock,
    EditCounts,
    aggregate_scores,
    score_utterance,
)


@dataclass(frozen=True)
class ValidationRow:
    """One stable-ID utterance result used by TSV and online reporting."""

    id: str
    category: str
    raw_text: str
    gold: str
    hypothesis: str
    ref_number: str
    hyp_number: str
    utt_wer: float
    utt_cer: float
    num_wer: float | None
    num_cer: float | None


@dataclass(frozen=True)
class ValidationResult:
    """Complete local score state for exactly one 2,000-row validation."""

    coverage: int
    overall: dict[str, Any]
    categories: dict[str, dict[str, Any]]
    rows: tuple[ValidationRow, ...]
    synthesis_records: tuple[dict[str, Any], ...]
    hypothesis_records: tuple[dict[str, Any], ...]
    failures: dict[str, int]
    throughput: dict[str, float]
    wall_time_seconds: float


def _finite_nonnegative(value: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return result


def _failure_count(value: int, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _record(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    try:
        return dict(vars(value))
    except TypeError as error:
        raise TypeError("validation rows must be mappings, dataclasses, or objects") from error


def _identifier(record: Mapping[str, Any], context: str) -> str:
    identifier = record.get("id")
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError(f"{context} id must be a non-blank string")
    return identifier


def _counts(counts: EditCounts) -> dict[str, int]:
    return {
        "C": counts.correct,
        "D": counts.deletions,
        "I": counts.insertions,
        "N": counts.reference_units,
        "S": counts.substitutions,
    }


def _aggregate(block: AggregateBlock) -> dict[str, Any]:
    return {
        "num_cer": block.num_cer,
        "num_wer": block.num_wer,
        "number_chars": _counts(block.number_chars),
        "number_spans": block.number_spans,
        "number_words": _counts(block.number_words),
        "utt_cer": block.utt_cer,
        "utt_wer": block.utt_wer,
        "utterance_chars": _counts(block.utterance_chars),
        "utterance_words": _counts(block.utterance_words),
        "utterances": block.utterances,
    }


def score_validation_run(
    assignments: Sequence[object],
    hypotheses: Sequence[Mapping[str, Any]],
    *,
    synthesis_records: Sequence[Mapping[str, Any]] | None = None,
    synthesis_seconds: float = 0.0,
    asr_seconds: float = 0.0,
    wall_time_seconds: float = 0.0,
    synthesis_failures: int = 0,
    asr_failures: int = 0,
) -> ValidationResult:
    """Score exact-coverage merged records using the benchmark metric primitives."""
    assignment_records = [_record(assignment) for assignment in assignments]
    assignment_ids = [_identifier(record, "assignment") for record in assignment_records]
    require_exact_coverage(assignment_ids, assignment_ids)
    if len(assignment_ids) != HARD_NUMBER_COUNT:
        raise ValueError(
            f"validation scoring requires exactly {HARD_NUMBER_COUNT:,} assignments; "
            f"got {len(assignment_ids):,}"
        )

    hypothesis_records = [dict(record) for record in hypotheses]
    hypothesis_ids = [_identifier(record, "hypothesis") for record in hypothesis_records]
    require_exact_coverage(assignment_ids, hypothesis_ids)
    valid_hypothesis_ids = [
        record["id"]
        for record in hypothesis_records
        if "error" not in record and isinstance(record.get("hypothesis"), str)
    ]
    require_exact_coverage(assignment_ids, valid_hypothesis_ids)

    ordered_assignments = sorted(assignment_records, key=lambda record: record["id"])
    hypothesis_by_id = {record["id"]: record for record in hypothesis_records}
    scores = []
    rows: list[ValidationRow] = []
    for assignment in ordered_assignments:
        identifier = assignment["id"]
        category = assignment.get("category")
        raw_text = assignment.get("text")
        gold = assignment.get("normalized_gold")
        hypothesis = hypothesis_by_id[identifier]["hypothesis"]
        if not all(isinstance(value, str) for value in (category, raw_text, gold)):
            raise TypeError(
                f"assignment {identifier!r} requires string category, text, and normalized_gold"
            )
        score = score_utterance(raw_text, gold, hypothesis, category)
        scores.append(score)
        rows.append(
            ValidationRow(
                id=identifier,
                category=category,
                raw_text=score.raw_text,
                gold=score.gold,
                hypothesis=score.hypothesis,
                ref_number=score.ref_number,
                hyp_number=score.hyp_number,
                utt_wer=score.utt_wer,
                utt_cer=score.utt_cer,
                num_wer=score.num_wer,
                num_cer=score.num_cer,
            )
        )

    synthesis = (
        [_record(record) for record in synthesis_records]
        if synthesis_records is not None
        else ordered_assignments
    )
    synthesis_ids = [_identifier(record, "synthesis record") for record in synthesis]
    require_exact_coverage(assignment_ids, synthesis_ids)
    aggregates = aggregate_scores(scores)
    synth_seconds = _finite_nonnegative(synthesis_seconds, "synthesis_seconds")
    transcription_seconds = _finite_nonnegative(asr_seconds, "asr_seconds")
    return ValidationResult(
        coverage=len(rows),
        overall=_aggregate(aggregates.overall),
        categories={
            category: _aggregate(block)
            for category, block in aggregates.per_category.items()
        },
        rows=tuple(rows),
        synthesis_records=tuple(sorted(synthesis, key=lambda record: record["id"])),
        hypothesis_records=tuple(
            sorted(hypothesis_records, key=lambda record: record["id"])
        ),
        failures={
            "asr": _failure_count(asr_failures, "asr_failures"),
            "synthesis": _failure_count(synthesis_failures, "synthesis_failures"),
        },
        throughput={
            "asr_utterances_per_second": (
                len(rows) / transcription_seconds if transcription_seconds else 0.0
            ),
            "synthesis_utterances_per_second": (
                len(rows) / synth_seconds if synth_seconds else 0.0
            ),
        },
        wall_time_seconds=_finite_nonnegative(wall_time_seconds, "wall_time_seconds"),
    )


def _atomic_text(path: Path, writer: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as destination:
            writer(destination)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: object) -> None:
    def write(destination: TextIO) -> None:
        json.dump(
            value,
            destination,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        destination.write("\n")

    _atomic_text(path, write)


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    def write(destination: TextIO) -> None:
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

    _atomic_text(path, write)


def write_validation_report(
    paths: ValidationPaths,
    result: ValidationResult,
    *,
    run_metadata: Mapping[str, object] | None = None,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    """Atomically write all complete local artifacts before online logging."""
    if result.coverage != HARD_NUMBER_COUNT:
        raise ValueError(f"reports require exactly {HARD_NUMBER_COUNT:,}/2,000 coverage")

    _write_jsonl(paths.manifest, result.synthesis_records)
    _write_jsonl(paths.hypotheses, result.hypothesis_records)

    def write_tsv(destination: TextIO) -> None:
        writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
        fields = [
            "id", "category", "raw_text", "gold", "hypothesis", "ref_number",
            "hyp_number", "utt_wer", "utt_cer", "num_wer", "num_cer",
        ]
        writer.writerow(fields)
        for row in result.rows:
            writer.writerow([getattr(row, field) for field in fields])

    _atomic_text(paths.per_utt, write_tsv)
    metrics = {
        "categories": result.categories,
        "coverage": result.coverage,
        "failures": result.failures,
        "overall": result.overall,
        "throughput": result.throughput,
        "wall_time_seconds": result.wall_time_seconds,
    }
    _write_json(paths.metrics, metrics)

    def write_markdown(destination: TextIO) -> None:
        destination.write("# Hard-number validation report\n\n")
        destination.write(f"Coverage: {result.coverage:,}/{HARD_NUMBER_COUNT:,}.\n\n")
        destination.write(
            "CER uses reference-character micro-weighting and therefore "
            "intentionally differs from the source evaluator, which weights its "
            "CER aggregation by reference words.\n\n"
        )
        destination.write("| category | utterances | utt WER | utt CER | num WER | num CER |\n")
        destination.write("|---|---:|---:|---:|---:|---:|\n")
        destination.writelines(
                f"| {category} | {block['utterances']} | {block['utt_wer']:.6f} | "
                f"{block['utt_cer']:.6f} | {block['num_wer']:.6f} | "
                f"{block['num_cer']:.6f} |\n"
            for category, block in result.categories.items()
        )

    _atomic_text(paths.report, write_markdown)
    _write_json(paths.run_metadata, dict(run_metadata or {}))
    return (
        paths.manifest,
        paths.hypotheses,
        paths.per_utt,
        paths.metrics,
        paths.report,
        paths.run_metadata,
    )
