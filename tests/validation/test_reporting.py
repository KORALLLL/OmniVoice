from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import soundfile as sf

from omnivoice.validation.artifacts import CoverageError, ValidationPaths
from omnivoice.validation.reporting import (
    score_validation_run,
    write_validation_report,
)


def _complete_inputs() -> tuple[list[SimpleNamespace], list[dict[str, object]]]:
    assignments: list[SimpleNamespace] = []
    hypotheses: list[dict[str, object]] = []
    for index in range(2_000):
        identifier = f"utt-{index:04d}"
        if index == 0:
            category, raw_text, gold, hypothesis = "b", "кот", "кот", "кит"
        elif index == 1:
            category, raw_text, gold, hypothesis = "a", "2", "два", ""
        else:
            category, raw_text, gold, hypothesis = "z", "1", "один", "один"
        assignments.append(
            SimpleNamespace(
                id=identifier,
                category=category,
                text=raw_text,
                normalized_gold=gold,
            )
        )
        hypotheses.append({"id": identifier, "hypothesis": hypothesis, "rank": index % 8})
    return assignments, hypotheses


def test_score_validation_run_uses_true_reference_unit_micro_averages() -> None:
    """Catches averaging utterance rates instead of summing reference units."""
    assignments, hypotheses = _complete_inputs()

    result = score_validation_run(
        assignments,
        list(reversed(hypotheses)),
        synthesis_seconds=100.0,
        asr_seconds=50.0,
        wall_time_seconds=160.0,
    )

    assert result.coverage == 2_000
    assert result.overall["utt_wer"] == pytest.approx(2 / 2_000)
    assert result.overall["utt_cer"] == pytest.approx(4 / 7_998)
    assert result.overall["num_wer"] == pytest.approx(1 / 1_999)
    assert result.overall["num_cer"] == pytest.approx(3 / 7_995)
    assert list(result.categories) == ["a", "b", "z"]
    assert result.throughput == {"asr_utterances_per_second": 40.0, "synthesis_utterances_per_second": 20.0}
    assert result.rows[0].id == "utt-0000"
    assert result.rows[-1].id == "utt-1999"


def test_score_validation_run_rejects_incomplete_hypothesis_coverage() -> None:
    """Catches aggregate metrics being reported from fewer than 2,000 rows."""
    assignments, hypotheses = _complete_inputs()

    with pytest.raises(CoverageError, match="utt-1999"):
        score_validation_run(assignments, hypotheses[:-1])


def test_score_validation_run_rejects_error_bearing_synthesis_records(
    tmp_path: Path,
) -> None:
    """Catches the public scoring API certifying failed synthesis rows by ID."""
    assignments, hypotheses = _complete_inputs()
    wav_path = tmp_path / "valid.wav"
    sf.write(wav_path, [0.25], 24_000, subtype="PCM_16")
    digest = hashlib.sha256(wav_path.read_bytes()).hexdigest()
    synthesis = [
        {"id": item.id, "sha256": digest, "wav": str(wav_path)}
        for item in assignments
    ]
    synthesis[-1]["error"] = "synthesis failed"

    with pytest.raises(CoverageError, match="utt-1999"):
        score_validation_run(
            assignments,
            hypotheses,
            synthesis_records=synthesis,
        )


def test_score_validation_run_rejects_hash_valid_non_wav_synthesis(
    tmp_path: Path,
) -> None:
    """Catches the public scoring API accepting a hash-valid non-WAV by ID."""
    assignments, hypotheses = _complete_inputs()
    wav_path = tmp_path / "valid.wav"
    sf.write(wav_path, [0.25], 24_000, subtype="PCM_16")
    valid_digest = hashlib.sha256(wav_path.read_bytes()).hexdigest()
    synthesis = [
        {"id": item.id, "sha256": valid_digest, "wav": str(wav_path)}
        for item in assignments
    ]
    invalid_wav = tmp_path / "hash-valid-but-not-wav.wav"
    invalid_wav.write_bytes(b"not a WAV")
    synthesis[-1]["wav"] = str(invalid_wav)
    synthesis[-1]["sha256"] = hashlib.sha256(invalid_wav.read_bytes()).hexdigest()

    with pytest.raises(CoverageError, match="utt-1999"):
        score_validation_run(
            assignments,
            hypotheses,
            synthesis_records=synthesis,
        )


def test_write_validation_report_persists_complete_stable_artifacts(tmp_path: Path) -> None:
    """Catches reports that omit raw counts, merged rows, or timing metadata."""
    assignments, hypotheses = _complete_inputs()
    wav_path = tmp_path / "valid.wav"
    sf.write(wav_path, [0.25], 24_000, subtype="PCM_16")
    digest = hashlib.sha256(wav_path.read_bytes()).hexdigest()
    synthesis = [
        {
            "id": item.id,
            "rank": index % 8,
            "sha256": digest,
            "wav": str(wav_path),
        }
        for index, item in enumerate(reversed(assignments))
    ]
    result = score_validation_run(
        assignments,
        hypotheses,
        synthesis_records=synthesis,
        synthesis_seconds=100.0,
        asr_seconds=50.0,
        wall_time_seconds=160.0,
        synthesis_failures=0,
        asr_failures=0,
    )
    paths = ValidationPaths(tmp_path, "run-1", 625)

    written = write_validation_report(
        paths,
        result,
        run_metadata={"dev_loss": 0.25, "run_id": "run-1", "step": 625},
    )

    assert written == (
        paths.manifest,
        paths.hypotheses,
        paths.per_utt,
        paths.metrics,
        paths.report,
        paths.run_metadata,
    )
    assert [json.loads(line)["id"] for line in paths.manifest.read_text().splitlines()[:2]] == ["utt-0000", "utt-0001"]
    assert [json.loads(line)["id"] for line in paths.hypotheses.read_text().splitlines()[:2]] == ["utt-0000", "utt-0001"]
    assert paths.per_utt.read_text(encoding="utf-8").splitlines()[0].split("\t") == [
        "id", "category", "raw_text", "gold", "hypothesis", "ref_number", "hyp_number",
        "utt_wer", "utt_cer", "num_wer", "num_cer",
    ]
    metrics = json.loads(paths.metrics.read_text(encoding="utf-8"))
    assert metrics["coverage"] == 2_000
    assert metrics["failures"] == {"asr": 0, "synthesis": 0}
    assert metrics["throughput"] == {"asr_utterances_per_second": 40.0, "synthesis_utterances_per_second": 20.0}
    assert metrics["wall_time_seconds"] == 160.0
    for scope in (metrics["overall"], metrics["categories"]["a"]):
        for unit in ("utterance_words", "utterance_chars", "number_words", "number_chars"):
            assert set(scope[unit]) == {"C", "D", "I", "N", "S"}
    report = paths.report.read_text(encoding="utf-8")
    assert "reference-character micro-weighting" in report
    assert "intentionally differs from the source evaluator" in report
    assert json.loads(paths.run_metadata.read_text()) == {"dev_loss": 0.25, "run_id": "run-1", "step": 625}


def test_report_writer_rejects_mutated_incomplete_result_before_writing(tmp_path: Path) -> None:
    """Catches local complete-looking artifacts from an incomplete score result."""
    assignments, hypotheses = _complete_inputs()
    result = replace(score_validation_run(assignments, hypotheses), coverage=1_999)
    paths = ValidationPaths(tmp_path, "run-1", 0)

    with pytest.raises(ValueError, match="2,000"):
        write_validation_report(paths, result)

    assert not paths.metrics.exists()
