from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from omnivoice.validation import wandb_logging
from omnivoice.validation.reporting import score_validation_run
from omnivoice.validation.wandb_logging import WandbRunStore, log_validation


class FakeRun:
    def __init__(self) -> None:
        self.last_metrics: dict[str, object] = {}
        self.last_step: int | None = None
        self.log_calls = 0

    def log(self, metrics: dict[str, object], *, step: int) -> None:
        self.log_calls += 1
        self.last_metrics = metrics
        self.last_step = step


class FakeWandb:
    def __init__(self, *, api_key: str | None = "key") -> None:
        self.api_key = api_key
        self.api_timeouts: list[int] = []
        self.init_kwargs: dict[str, object] | None = None
        self.audio_objects: list[SimpleNamespace] = []
        self.table_objects: list[SimpleNamespace] = []

    def Api(self, *, timeout: int) -> SimpleNamespace:
        self.api_timeouts.append(timeout)
        return SimpleNamespace(api_key=self.api_key)

    def init(self, **kwargs: object) -> FakeRun:
        self.init_kwargs = kwargs
        return FakeRun()

    def Audio(self, path: str, *, caption: str) -> SimpleNamespace:
        value = SimpleNamespace(path=path, caption=caption)
        self.audio_objects.append(value)
        return value

    def Table(
        self, *, columns: list[str], data: list[list[object]]
    ) -> SimpleNamespace:
        value = SimpleNamespace(columns=columns, data=data)
        self.table_objects.append(value)
        return value


def _result():
    assignments = [
        SimpleNamespace(id=f"utt-{index:04d}", category="b" if index % 2 else "a", text="1", normalized_gold="один")
        for index in range(2_000)
    ]
    hypotheses = [{"id": item.id, "hypothesis": "один"} for item in assignments]
    return score_validation_run(
        assignments,
        hypotheses,
        synthesis_seconds=100.0,
        asr_seconds=50.0,
        wall_time_seconds=160.0,
    )


def test_wandb_run_store_preflights_online_authentication(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Catches preflight silently accepting an unauthenticated online client."""
    fake = FakeWandb(api_key=None)
    monkeypatch.setattr(wandb_logging, "wandb", fake)

    with pytest.raises(RuntimeError, match=r"wandb login"):
        WandbRunStore(tmp_path / "wandb_ids.json").preflight()

    assert fake.api_timeouts == [15]


def test_wandb_run_store_persists_run_and_audio_identity_atomically(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Catches resumed validations changing the W&B run or four audio rows."""
    fake = FakeWandb()
    monkeypatch.setattr(wandb_logging, "wandb", fake)
    monkeypatch.setattr(wandb_logging.uuid, "uuid4", lambda: SimpleNamespace(hex="stable-run-id"))
    path = tmp_path / "state" / "wandb_ids.json"
    store = WandbRunStore(path)

    first_audio_ids = store.load_or_create_audio_ids(["utt-3", "utt-1", "utt-2", "utt-4"])
    second_audio_ids = store.load_or_create_audio_ids(["new-1", "new-2", "new-3", "new-4"])
    run = store.init({"dataset_revision": "abc"})

    assert first_audio_ids == ["utt-3", "utt-1", "utt-2", "utt-4"]
    assert second_audio_ids == first_audio_ids
    assert isinstance(run, FakeRun)
    assert json.loads(path.read_text()) == {"audio_ids": first_audio_ids, "run_id": "stable-run-id"}
    assert fake.init_kwargs == {
        "config": {"dataset_revision": "abc"},
        "id": "stable-run-id",
        "project": "omnivoice-lora-validation",
        "resume": "allow",
    }
    assert list(path.parent.glob(".wandb_ids.json.*.tmp")) == []


def test_log_validation_uses_optimizer_step_stable_categories_and_four_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches W&B payload drift, batch-step logging, or excess audio uploads."""
    fake = FakeWandb()
    monkeypatch.setattr(wandb_logging, "wandb", fake)
    run = FakeRun()
    audio = [
        {"id": f"utt-{index}", "wav": f"/audio/{index}.wav", "text": f"prompt {index}", "voice_id": f"voice-{index}"}
        for index in range(4)
    ]

    log_validation(run, _result(), audio, step=625, steps_per_epoch=5_000, dev_loss=0.25)

    assert run.last_step == 625
    assert run.last_metrics["validation/optimizer_step"] == 625
    assert run.last_metrics["validation/fractional_epoch"] == pytest.approx(0.125)
    assert run.last_metrics["validation/dev_loss"] == 0.25
    assert len(fake.audio_objects) == 4
    assert [item.caption for item in fake.audio_objects] == [
        "utt-0 | voice-0 | prompt 0", "utt-1 | voice-1 | prompt 1", "utt-2 | voice-2 | prompt 2", "utt-3 | voice-3 | prompt 3",
    ]
    assert len(fake.table_objects) == 1
    assert [row[0] for row in fake.table_objects[0].data] == ["a", "b"]
    required = {
        "validation/coverage", "validation/synthesis_failures", "validation/asr_failures",
        "validation/synthesis_utterances_per_second", "validation/asr_utterances_per_second",
        "validation/wall_time_seconds", "validation/utt_wer", "validation/utt_cer",
        "validation/num_wer", "validation/num_cer",
    }
    required.update(
        f"validation/{scope}_{unit}_{field}"
        for scope in ("utterance", "number")
        for unit in ("words", "chars")
        for field in ("S", "D", "I", "C", "N")
    )
    assert required <= set(run.last_metrics)


def test_log_validation_makes_no_wandb_objects_for_incomplete_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches partial coverage reaching any W&B boundary."""
    fake = FakeWandb()
    monkeypatch.setattr(wandb_logging, "wandb", fake)
    run = FakeRun()

    with pytest.raises(ValueError, match="2,000"):
        log_validation(
            run,
            replace(_result(), coverage=1_999),
            [{"id": str(index), "wav": f"{index}.wav"} for index in range(4)],
            step=625,
            steps_per_epoch=5_000,
        )

    assert fake.audio_objects == []
    assert fake.table_objects == []
    assert run.log_calls == 0
