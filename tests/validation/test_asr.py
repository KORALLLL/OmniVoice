from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from omnivoice.validation.artifacts import AtomicJsonlLedger, ValidationPaths
from omnivoice.validation.asr import transcribe_rank


class _FakeRecognizer:
    """External GigaAM boundary replacement retaining received audio."""

    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.arrays: list[np.ndarray] = []
        self.sample_rates: list[int] = []
        self.fail_on_call = fail_on_call
        self.calls = 0

    def recognize(self, waveform: np.ndarray, sample_rate: int) -> list[str]:
        self.calls += 1
        self.arrays.append(waveform.copy())
        self.sample_rates.append(sample_rate)
        if self.fail_on_call == self.calls:
            raise RuntimeError("recognition exploded")
        return [f"hypothesis-{self.calls}"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthesis_records(tmp_path: Path) -> list[dict[str, Any]]:
    """Create independently valid exact-coverage 24 kHz synthesis records."""
    wav_path = tmp_path / "stereo-24khz.wav"
    sf.write(
        wav_path,
        np.array([[0.25, -0.25], [0.5, -0.5], [0.75, -0.75]], dtype=np.float64),
        24_000,
        subtype="PCM_16",
    )
    digest = _sha256(wav_path)
    return [
        {
            "id": f"utt-{index:04d}",
            "rank": index % 8,
            "sha256": digest,
            "wav": str(wav_path),
        }
        for index in range(2_000)
    ]


def test_transcribe_rank_resamples_rank_stride_and_resumes_only_missing_hypothesis(
    tmp_path: Path,
) -> None:
    """Catches ASR changes that skip mono/16 kHz conversion or redo valid work."""
    records = _synthesis_records(tmp_path)
    recognizer = _FakeRecognizer()
    paths = ValidationPaths(tmp_path, "run-1", 0)

    summary = transcribe_rank(
        synthesis_records=records,
        recognizer=recognizer,
        output_dir=paths,
        rank=3,
        world_size=8,
    )

    assert summary.expected == 250
    assert summary.completed == 250
    assert summary.transcribed == 250
    assert summary.skipped == 0
    assert summary.failed == 0
    assert summary.complete is True
    assert recognizer.sample_rates == [16_000] * 250
    assert all(array.dtype == np.float32 for array in recognizer.arrays)
    assert all(array.ndim == 1 for array in recognizer.arrays)
    assert all(np.allclose(array, 0.0, atol=1e-5) for array in recognizer.arrays)

    resumed = transcribe_rank(
        synthesis_records=records,
        recognizer=recognizer,
        output_dir=paths,
        rank=3,
        world_size=8,
    )
    assert resumed.skipped == 250
    assert recognizer.calls == 250

    ledger = AtomicJsonlLedger(paths.rank_hypotheses(3))
    ledger.upsert({"id": "utt-0003", "rank": 3})
    repaired = transcribe_rank(
        synthesis_records=records,
        recognizer=recognizer,
        output_dir=paths,
        rank=3,
        world_size=8,
    )
    assert repaired.completed == 250
    assert repaired.transcribed == 1
    assert repaired.skipped == 249
    assert recognizer.calls == 251


def test_transcribe_rank_persists_recognition_failure_without_marking_it_complete(
    tmp_path: Path,
) -> None:
    """Catches ASR exceptions that vanish or are incorrectly counted as complete."""
    records = _synthesis_records(tmp_path)
    recognizer = _FakeRecognizer(fail_on_call=1)
    paths = ValidationPaths(tmp_path, "run-1", 0)

    summary = transcribe_rank(
        synthesis_records=records,
        recognizer=recognizer,
        output_dir=paths,
        rank=3,
        world_size=8,
    )

    assert summary.completed == 249
    assert summary.failed == 1
    assert summary.complete is False
    record = next(
        row
        for row in AtomicJsonlLedger(paths.rank_hypotheses(3)).records
        if row["id"] == "utt-0003"
    )
    assert record == {
        "error": {"message": "recognition exploded", "type": "RuntimeError"},
        "id": "utt-0003",
        "rank": 3,
    }
