import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import weakref
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from omnivoice.training.control import TrainingOutcome, append_loss_history
from omnivoice.training.lora import DEFAULT_LORA_TARGET_MODULES
from omnivoice.validation.balalaika import SelectedBalalaikaClip
from omnivoice.validation.memorization import (
    _generate_with_model_loader,
    _parse_training_outcome,
    _run_generation_bounded,
    _run_process_group,
    _validate_run_consistency,
    choose_generation_checkpoint,
    generate_four,
    read_loss_history,
    run_memorization,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selected_manifest(tmp_path: Path) -> tuple[Path, list[SelectedBalalaikaClip]]:
    rows = []
    for index in range(4):
        audio_path = tmp_path / f"source-{index}.wav"
        sf.write(
            audio_path,
            np.full(72_000, index / 100.0, dtype=np.float32),
            24_000,
            subtype="PCM_16",
        )
        rows.append(
            SelectedBalalaikaClip(
                role="memorization",
                source_relative_path=f"000001/{index}.mp3",
                text=f"Русский текст {index}",
                schema_version=1,
                seed=42,
                source_shard="train/shard_000001.tar",
                member_name=f"{index}.mp3",
                audio_path=str(audio_path),
                source_sha256=f"{index + 1:064x}",
                wav_sha256=_sha256(audio_path),
                sample_rate=24_000,
                channels=1,
                duration=3.0,
                duration_tier="preferred_3_to_12s",
            )
        )
    manifest = tmp_path / "selected.jsonl"
    manifest.write_text(
        "".join(json.dumps(asdict(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return manifest, rows


class FakeModel:
    sampling_rate = 24_000

    def __init__(self, generation_calls):
        self.generation_calls = generation_calls

    def generate(self, **kwargs):
        self.generation_calls.append(kwargs)
        return [np.zeros(240, dtype=np.float32)]


def _direct_generation_runner(**kwargs):
    kwargs.pop("deadline_monotonic")
    return _generate_with_model_loader(**kwargs)


class HangingModel(FakeModel):
    def generate(self, **kwargs):
        del kwargs
        time.sleep(10)


def _hanging_model_loader(*, base_model=None, adapter_checkpoint=None):
    del base_model, adapter_checkpoint
    return HangingModel([])


def test_generate_four_uses_deterministic_config_and_sidecar_text(tmp_path):
    _, rows = _selected_manifest(tmp_path)
    calls = []

    paths = generate_four(rows, tmp_path / "generated", FakeModel(calls))

    assert len(paths) == 4
    assert all(path.is_file() for path in paths)
    assert [call["text"] for call in calls] == [row.text for row in rows]
    assert [call["ref_text"] for call in calls] == [row.text for row in rows]
    assert [call["ref_audio"] for call in calls] == [row.audio_path for row in rows]
    assert {call["language"] for call in calls} == {"Russian"}
    config = calls[0]["generation_config"]
    assert (
        config.num_step,
        config.guidance_scale,
        config.t_shift,
        config.layer_penalty_factor,
        config.position_temperature,
        config.class_temperature,
    ) == (32, 2.0, 0.1, 5.0, 0.0, 0.0)


def test_choose_generation_checkpoint_requires_two_consecutive_losses(tmp_path):
    for step in (50, 75, 100):
        (tmp_path / f"checkpoint-{step}").mkdir()
    history = [
        {"step": 50, "loss": 2e-4, "elapsed_seconds": 1.0},
        {"step": 75, "loss": 9e-5, "elapsed_seconds": 2.0},
        {"step": 100, "loss": 8e-5, "elapsed_seconds": 3.0},
    ]

    checkpoint = choose_generation_checkpoint(tmp_path, history)

    assert checkpoint == tmp_path / "checkpoint-100"


def test_choose_generation_checkpoint_falls_back_to_highest_completed(tmp_path):
    for step in (25, 50, 75):
        (tmp_path / f"checkpoint-{step}").mkdir()
    history = [
        {"step": 25, "loss": 2e-4, "elapsed_seconds": 1.0},
        {"step": 50, "loss": 9e-5, "elapsed_seconds": 2.0},
        {"step": 75, "loss": 2e-4, "elapsed_seconds": 3.0},
    ]

    checkpoint = choose_generation_checkpoint(tmp_path, history)

    assert checkpoint == tmp_path / "checkpoint-75"


def test_read_loss_history_accepts_experiment_directory(tmp_path):
    append_loss_history(
        tmp_path / "loss_history.jsonl",
        step=25,
        loss=2e-4,
        elapsed_seconds=3.0,
    )

    assert read_loss_history(tmp_path) == [
        {"step": 25, "loss": 2e-4, "elapsed_seconds": 3.0}
    ]


def _write_configs(tmp_path: Path) -> tuple[Path, Path]:
    train_config = tmp_path / "train.json"
    train_config.write_text(
        json.dumps(
            {
                "steps": 10_000,
                "lora_enabled": True,
                "lora_rank": 64,
                "lora_target_modules": list(DEFAULT_LORA_TARGET_MODULES),
                "init_from_checkpoint": "k2-fsa/OmniVoice",
                "resume_from_checkpoint": None,
            }
        )
    )
    data_config = tmp_path / "data.json"
    data_config.write_text(
        json.dumps(
            {
                "train": [{"manifest_path": ["old-train.lst"]}],
                "dev": [{"manifest_path": ["old-dev.lst"]}],
            }
        )
    )
    return train_config, data_config


def _fake_dependencies(output_dir: Path, losses: list[float], stop_reason: str):
    commands = []
    model_loads = []
    generation_calls = []

    def command_runner(command, **kwargs):
        commands.append((command, kwargs))
        if "omnivoice.cli.train" in command:
            history = [(25, 3e-4), *zip((50, 75, 100), losses)]
            for step, loss in history:
                append_loss_history(
                    output_dir / "loss_history.jsonl",
                    step=step,
                    loss=loss,
                    elapsed_seconds=float(step),
                )
                (output_dir / f"checkpoint-{step}").mkdir(exist_ok=True)
            outcome = TrainingOutcome(
                step=100,
                stop_reason=stop_reason,
                last_eval_loss=losses[-1],
                target_reached=stop_reason == "eval_loss_target",
            )
            return SimpleNamespace(stdout=json.dumps(asdict(outcome)) + "\n")
        return SimpleNamespace(stdout="")

    def model_loader(*, base_model=None, adapter_checkpoint=None):
        model_loads.append(
            {"base_model": base_model, "adapter_checkpoint": adapter_checkpoint}
        )
        return FakeModel(generation_calls)

    return command_runner, model_loader, commands, model_loads, generation_calls


def test_run_memorization_builds_complete_success_artifacts(tmp_path):
    selected_manifest, rows = _selected_manifest(tmp_path)
    train_config, data_config = _write_configs(tmp_path)
    output_dir = tmp_path / "exp"
    runner, loader, commands, loads, generation_calls = _fake_dependencies(
        output_dir, [2e-4, 9e-5, 8e-5], "eval_loss_target"
    )

    result = run_memorization(
        selected_manifest=selected_manifest,
        output_dir=output_dir,
        train_config=train_config,
        data_config=data_config,
        max_wall_clock_seconds=1200,
        experiment_wall_clock_seconds=3600,
        command_runner=runner,
        model_loader=loader,
        generation_runner=_direct_generation_runner,
    )

    assert result.required_target_reached is True
    assert result.stretch_target_reached is False
    assert result.qualifying_step == 100
    assert result.minimum_loss == 8e-5
    assert result.miss_reason is None
    assert len(list((output_dir / "original").glob("*.wav"))) == 4
    assert len(list((output_dir / "generated/initial").glob("*.wav"))) == 4
    assert len(list((output_dir / "generated/final").glob("*.wav"))) == 4
    assert loads == [
        {"base_model": "k2-fsa/OmniVoice", "adapter_checkpoint": None},
        {
            "base_model": None,
            "adapter_checkpoint": output_dir / "checkpoint-100",
        },
    ]
    assert len(generation_calls) == 8
    assert [call["text"] for call in generation_calls[:4]] == [
        row.text for row in rows
    ]
    assert [call["ref_text"] for call in generation_calls] == [
        row.text for row in rows
    ] * 2
    assert sum("omnivoice.scripts.extract_audio_tokens" in cmd for cmd, _ in commands) == 1
    [training_command] = [cmd for cmd, _ in commands if "omnivoice.cli.train" in cmd]
    assert training_command[:7] == [
        "accelerate",
        "launch",
        "--gpu_ids",
        "0",
        "--num_processes",
        "1",
        "-m",
    ]
    runtime_config = json.loads((output_dir / "train_config.json").read_text())
    assert runtime_config["steps"] == 10_000
    assert runtime_config["stop_after_step"] == 10_000
    assert runtime_config["eval_steps"] == 25
    assert runtime_config["early_stop_eval_loss"] == 1e-4
    assert runtime_config["early_stop_patience"] == 2
    assert runtime_config["max_wall_clock_seconds"] <= 1200
    assert json.loads((output_dir / "result.json").read_text())[
        "experiment_deadline_monotonic"
    ] == result.experiment_deadline_monotonic


def test_run_memorization_timeout_records_explicit_miss_and_final_checkpoint(tmp_path):
    selected_manifest, _ = _selected_manifest(tmp_path)
    train_config, data_config = _write_configs(tmp_path)
    output_dir = tmp_path / "exp"
    runner, loader, _, loads, _ = _fake_dependencies(
        output_dir, [2e-4, 9e-5, 2e-4], "wall_clock_limit"
    )

    result = run_memorization(
        selected_manifest=selected_manifest,
        output_dir=output_dir,
        train_config=train_config,
        data_config=data_config,
        max_wall_clock_seconds=1200,
        command_runner=runner,
        model_loader=loader,
        generation_runner=_direct_generation_runner,
    )

    assert result.required_target_reached is False
    assert result.qualifying_step is None
    assert result.selected_checkpoint == str(output_dir / "checkpoint-100")
    assert result.miss_reason == "wall_clock_limit_before_required_target"
    assert loads[-1]["adapter_checkpoint"] == output_dir / "checkpoint-100"


def test_run_memorization_releases_base_model_before_training(tmp_path):
    selected_manifest, _ = _selected_manifest(tmp_path)
    train_config, data_config = _write_configs(tmp_path)
    output_dir = tmp_path / "exp"
    runner, _, _, _, generation_calls = _fake_dependencies(
        output_dir, [2e-4, 9e-5, 8e-5], "eval_loss_target"
    )
    model_references = []

    def model_loader(*, base_model=None, adapter_checkpoint=None):
        del base_model, adapter_checkpoint
        model = FakeModel(generation_calls)
        model_references.append(weakref.ref(model))
        return model

    def memory_checking_runner(command, **kwargs):
        if "omnivoice.cli.train" in command:
            assert model_references[0]() is None
        return runner(command, **kwargs)

    run_memorization(
        selected_manifest=selected_manifest,
        output_dir=output_dir,
        train_config=train_config,
        data_config=data_config,
        command_runner=memory_checking_runner,
        model_loader=model_loader,
        generation_runner=_direct_generation_runner,
    )
    assert all(reference() is None for reference in model_references)


def _published_snapshot(output_dir: Path) -> dict[str, bytes]:
    paths = [
        output_dir / "four.jsonl",
        output_dir / "result.json",
        *sorted((output_dir / "original").glob("*.wav")),
        *sorted((output_dir / "generated/initial").glob("*.wav")),
        *sorted((output_dir / "generated/final").glob("*.wav")),
    ]
    return {str(path.relative_to(output_dir)): path.read_bytes() for path in paths}


def _seed_published_artifacts(output_dir: Path) -> dict[str, bytes]:
    for directory in (
        output_dir / "original",
        output_dir / "generated/initial",
        output_dir / "generated/final",
    ):
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(4):
            (directory / f"old-{index}.wav").write_bytes(f"old-{index}".encode())
    (output_dir / "four.jsonl").write_text("old manifest\n")
    (output_dir / "result.json").write_text('{"old": true}\n')
    return _published_snapshot(output_dir)


def _rewrite_manifest(path: Path, rows: list[SelectedBalalaikaClip]) -> None:
    path.write_text(
        "".join(json.dumps(asdict(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_generation_timeout_is_bounded_and_does_not_start_tokenization(tmp_path):
    selected_manifest, _ = _selected_manifest(tmp_path)
    train_config, data_config = _write_configs(tmp_path)
    commands = []
    started = time.monotonic()

    with pytest.raises(TimeoutError, match="generation exceeded phase deadline"):
        run_memorization(
            selected_manifest=selected_manifest,
            output_dir=tmp_path / "exp",
            train_config=train_config,
            data_config=data_config,
            max_wall_clock_seconds=0.2,
            command_runner=lambda command, **kwargs: commands.append(command),
            model_loader=_hanging_model_loader,
        )

    assert time.monotonic() - started < 2.0
    assert commands == []
    assert not (tmp_path / "exp/result.json").exists()
    assert not (tmp_path / "exp/generated/initial").exists()


def test_tokenizer_timeout_is_bounded_and_does_not_start_training(tmp_path):
    selected_manifest, _ = _selected_manifest(tmp_path)
    train_config, data_config = _write_configs(tmp_path)
    commands = []

    def timing_out_runner(command, **kwargs):
        commands.append(command)
        assert 0 < kwargs["timeout"] <= 1.0
        subprocess.run(
            ["sleep", "10"], check=True, timeout=kwargs["timeout"]
        )

    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        run_memorization(
            selected_manifest=selected_manifest,
            output_dir=tmp_path / "exp",
            train_config=train_config,
            data_config=data_config,
            max_wall_clock_seconds=1.0,
            command_runner=timing_out_runner,
            model_loader=lambda **kwargs: FakeModel([]),
            generation_runner=_direct_generation_runner,
        )

    assert time.monotonic() - started < 2.0
    assert len(commands) == 1
    assert not (tmp_path / "exp/result.json").exists()


def test_run_rejects_reused_output_with_checkpoint_state(tmp_path):
    selected_manifest, _ = _selected_manifest(tmp_path)
    train_config, data_config = _write_configs(tmp_path)
    output_dir = tmp_path / "exp"
    (output_dir / "checkpoint-9999").mkdir(parents=True)

    with pytest.raises(ValueError, match="fresh output directory"):
        run_memorization(
            selected_manifest=selected_manifest,
            output_dir=output_dir,
            train_config=train_config,
            data_config=data_config,
            command_runner=lambda *args, **kwargs: None,
            model_loader=lambda **kwargs: FakeModel([]),
            generation_runner=_direct_generation_runner,
        )


@pytest.mark.parametrize("target_reached", [True, False])
def test_run_requires_exact_current_checkpoint(tmp_path, target_reached):
    selected_manifest, _ = _selected_manifest(tmp_path)
    train_config, data_config = _write_configs(tmp_path)
    output_dir = tmp_path / "exp"
    losses = [2e-4, 9e-5, 8e-5] if target_reached else [2e-4, 9e-5, 2e-4]

    def runner(command, **kwargs):
        del kwargs
        if "omnivoice.cli.train" not in command:
            return SimpleNamespace(stdout="")
        history = [(25, 3e-4), *zip((50, 75, 100), losses)]
        for step, loss in history:
            append_loss_history(
                output_dir / "loss_history.jsonl",
                step=step,
                loss=loss,
                elapsed_seconds=float(step),
            )
        (output_dir / "checkpoint-75").mkdir()
        outcome = TrainingOutcome(
            100,
            "eval_loss_target" if target_reached else "wall_clock_limit",
            losses[-1],
            target_reached,
        )
        return SimpleNamespace(stdout=json.dumps(asdict(outcome)))

    with pytest.raises(FileNotFoundError, match="checkpoint-100"):
        run_memorization(
            selected_manifest=selected_manifest,
            output_dir=output_dir,
            train_config=train_config,
            data_config=data_config,
            command_runner=runner,
            model_loader=lambda **kwargs: FakeModel([]),
            generation_runner=_direct_generation_runner,
        )


def test_run_rejects_checkpoint_newer_than_training_outcome(tmp_path):
    selected_manifest, _ = _selected_manifest(tmp_path)
    train_config, data_config = _write_configs(tmp_path)
    output_dir = tmp_path / "exp"
    runner, loader, _, _, _ = _fake_dependencies(
        output_dir, [2e-4, 9e-5, 8e-5], "eval_loss_target"
    )

    def inconsistent_runner(command, **kwargs):
        completed = runner(command, **kwargs)
        if "omnivoice.cli.train" in command:
            (output_dir / "checkpoint-125").mkdir()
        return completed

    with pytest.raises(ValueError, match="newer than TrainingOutcome"):
        run_memorization(
            selected_manifest=selected_manifest,
            output_dir=output_dir,
            train_config=train_config,
            data_config=data_config,
            command_runner=inconsistent_runner,
            model_loader=loader,
            generation_runner=_direct_generation_runner,
        )


def test_run_rejects_outcome_loss_inconsistent_with_history(tmp_path):
    selected_manifest, _ = _selected_manifest(tmp_path)
    train_config, data_config = _write_configs(tmp_path)
    output_dir = tmp_path / "exp"
    runner, loader, _, _, _ = _fake_dependencies(
        output_dir, [2e-4, 9e-5, 8e-5], "eval_loss_target"
    )

    def inconsistent_runner(command, **kwargs):
        completed = runner(command, **kwargs)
        if "omnivoice.cli.train" in command:
            payload = json.loads(completed.stdout)
            payload["last_eval_loss"] = 7e-5
            completed.stdout = json.dumps(payload)
        return completed

    with pytest.raises(ValueError, match="inconsistent with loss history"):
        run_memorization(
            selected_manifest=selected_manifest,
            output_dir=output_dir,
            train_config=train_config,
            data_config=data_config,
            command_runner=inconsistent_runner,
            model_loader=loader,
            generation_runner=_direct_generation_runner,
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"lora_enabled": False}, "lora_enabled"),
        ({"lora_rank": 32}, "lora_rank"),
        ({"lora_target_modules": ["q_proj"]}, "lora_target_modules"),
        ({"init_from_checkpoint": "wrong"}, "init_from_checkpoint"),
        ({"resume_from_checkpoint": "checkpoint-25"}, "resume_from_checkpoint"),
    ],
)
def test_run_rejects_incompatible_training_contract(tmp_path, override, message):
    selected_manifest, _ = _selected_manifest(tmp_path)
    train_config, data_config = _write_configs(tmp_path)
    payload = json.loads(train_config.read_text()) | override
    train_config.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=message):
        run_memorization(
            selected_manifest=selected_manifest,
            output_dir=tmp_path / "exp",
            train_config=train_config,
            data_config=data_config,
            command_runner=lambda *args, **kwargs: None,
            model_loader=lambda **kwargs: FakeModel([]),
            generation_runner=_direct_generation_runner,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: [rows[0], rows[0], *rows[2:]],
        lambda rows: [{**asdict(row), "text": " "} if index == 0 else asdict(row) for index, row in enumerate(rows)],
        lambda rows: [{**asdict(row), "wav_sha256": "0" * 64} if index == 0 else asdict(row) for index, row in enumerate(rows)],
    ],
)
def test_selected_manifest_rejects_duplicate_empty_or_bad_hash(tmp_path, mutation):
    manifest, rows = _selected_manifest(tmp_path)
    mutated = mutation(rows)
    payloads = [asdict(row) if isinstance(row, SelectedBalalaikaClip) else row for row in mutated]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in payloads))
    train_config, data_config = _write_configs(tmp_path)

    with pytest.raises(ValueError):
        run_memorization(
            selected_manifest=manifest,
            output_dir=tmp_path / "exp",
            train_config=train_config,
            data_config=data_config,
            command_runner=lambda *args, **kwargs: None,
            model_loader=lambda **kwargs: FakeModel([]),
            generation_runner=_direct_generation_runner,
        )


@pytest.mark.parametrize("audio_case", ["stereo", "wrong_rate", "float", "empty"])
def test_selected_manifest_rejects_invalid_audio(tmp_path, audio_case):
    manifest, rows = _selected_manifest(tmp_path)
    path = Path(rows[0].audio_path)
    data = np.zeros((240, 2) if audio_case == "stereo" else 240)
    rate = 16_000 if audio_case == "wrong_rate" else 24_000
    subtype = "FLOAT" if audio_case == "float" else "PCM_16"
    if audio_case == "empty":
        data = np.zeros(0)
    sf.write(path, data, rate, subtype=subtype)
    rows[0] = SelectedBalalaikaClip(**(asdict(rows[0]) | {"wav_sha256": _sha256(path)}))
    _rewrite_manifest(manifest, rows)
    train_config, data_config = _write_configs(tmp_path)

    with pytest.raises(ValueError, match="WAV"):
        run_memorization(
            selected_manifest=manifest,
            output_dir=tmp_path / "exp",
            train_config=train_config,
            data_config=data_config,
            command_runner=lambda *args, **kwargs: None,
            model_loader=lambda **kwargs: FakeModel([]),
            generation_runner=_direct_generation_runner,
        )


def test_generate_four_rejects_incompatible_sampling_rate(tmp_path):
    _, rows = _selected_manifest(tmp_path)
    model = FakeModel([])
    model.sampling_rate = 16_000

    with pytest.raises(ValueError, match="24000"):
        generate_four(rows, tmp_path / "generated", model)


def test_mid_copy_failure_preserves_previous_artifact_set(tmp_path, monkeypatch):
    manifest, _ = _selected_manifest(tmp_path)
    train_config, data_config = _write_configs(tmp_path)
    output_dir = tmp_path / "exp"
    before = _seed_published_artifacts(output_dir)
    real_copy = __import__("shutil").copy2
    calls = 0

    def failing_copy(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("copy failed")
        return real_copy(source, destination)

    monkeypatch.setattr("omnivoice.validation.memorization.shutil.copy2", failing_copy)
    with pytest.raises(OSError, match="copy failed"):
        run_memorization(
            selected_manifest=manifest,
            output_dir=output_dir,
            train_config=train_config,
            data_config=data_config,
            command_runner=lambda *args, **kwargs: None,
            model_loader=lambda **kwargs: FakeModel([]),
            generation_runner=_direct_generation_runner,
        )

    assert _published_snapshot(output_dir) == before


def test_mid_generation_failure_preserves_previous_artifact_set(tmp_path):
    manifest, _ = _selected_manifest(tmp_path)
    train_config, data_config = _write_configs(tmp_path)
    output_dir = tmp_path / "exp"
    before = _seed_published_artifacts(output_dir)

    def failing_generation_runner(*, rows, output_dir, **kwargs):
        del kwargs
        output_dir.mkdir(parents=True)
        sf.write(output_dir / f"{rows[0].wav_sha256}.wav", np.zeros(240), 24_000)
        raise RuntimeError("generation failed")

    with pytest.raises(RuntimeError, match="generation failed"):
        run_memorization(
            selected_manifest=manifest,
            output_dir=output_dir,
            train_config=train_config,
            data_config=data_config,
            command_runner=lambda *args, **kwargs: None,
            model_loader=lambda **kwargs: FakeModel([]),
            generation_runner=failing_generation_runner,
        )

    assert _published_snapshot(output_dir) == before


@pytest.mark.parametrize(
    "rows",
    [
        [{"step": 25, "loss": float("nan"), "elapsed_seconds": 1.0}],
        [{"step": 25, "loss": 0.1, "elapsed_seconds": float("inf")}],
        [{"step": 0, "loss": 0.1, "elapsed_seconds": 1.0}],
        [{"step": 26, "loss": 0.1, "elapsed_seconds": 1.0}],
        [
            {"step": 50, "loss": 0.1, "elapsed_seconds": 2.0},
            {"step": 25, "loss": 0.1, "elapsed_seconds": 3.0},
        ],
        [
            {"step": 25, "loss": 0.1, "elapsed_seconds": 2.0},
            {"step": 50, "loss": 0.1, "elapsed_seconds": 1.0},
        ],
    ],
)
def test_read_loss_history_rejects_nonfinite_or_inconsistent_rows(tmp_path, rows):
    path = tmp_path / "loss_history.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(ValueError, match="loss history"):
        read_loss_history(path)


@pytest.mark.parametrize(
    "payload",
    [
        {"step": True, "stop_reason": "completed", "last_eval_loss": 0.1, "target_reached": False},
        {"step": 100, "stop_reason": "unknown", "last_eval_loss": 0.1, "target_reached": False},
        {"step": 100, "stop_reason": "completed", "last_eval_loss": float("nan"), "target_reached": False},
        {"step": 100, "stop_reason": "completed", "last_eval_loss": 0.1, "target_reached": 0},
    ],
)
def test_parse_training_outcome_rejects_malformed_schema(payload):
    with pytest.raises(ValueError, match="TrainingOutcome"):
        _parse_training_outcome(json.dumps(payload))


def test_process_group_timeout_kills_pipe_holding_grandchild(tmp_path):
    marker = tmp_path / "late-marker"
    pgid_file = tmp_path / "pgid"
    child_code = (
        "import signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(0.8); Path(__import__('sys').argv[1]).write_text('late')"
    )
    launcher_code = (
        "import os,subprocess,sys,time; "
        "open(sys.argv[1], 'w').write(str(os.getpgrp())); "
        "subprocess.Popen([sys.executable, '-c', sys.argv[2], sys.argv[3]]); "
        "time.sleep(10)"
    )
    started = time.monotonic()

    with pytest.raises(subprocess.TimeoutExpired):
        _run_process_group(
            [sys.executable, "-c", launcher_code, str(pgid_file), child_code, str(marker)],
            deadline_monotonic=time.monotonic() + 0.2,
            text=True,
        )

    assert time.monotonic() - started < 1.5
    time.sleep(1.0)
    assert not marker.exists()
    process_group = int(pgid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.killpg(process_group, 0)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_relative_path", " ", "source_relative_path"),
        ("schema_version", True, "schema_version"),
        ("duration_tier", "wrong", "duration_tier"),
        ("sample_rate", True, "sample_rate"),
        ("channels", 2, "channels"),
    ],
)
def test_selected_manifest_rejects_invalid_consumed_metadata(
    tmp_path, field, value, message
):
    manifest, rows = _selected_manifest(tmp_path)
    rows[0] = SelectedBalalaikaClip(**(asdict(rows[0]) | {field: value}))
    _rewrite_manifest(manifest, rows)
    train_config, data_config = _write_configs(tmp_path)

    with pytest.raises(ValueError, match=message):
        run_memorization(
            selected_manifest=manifest,
            output_dir=tmp_path / "exp",
            train_config=train_config,
            data_config=data_config,
            command_runner=lambda *args, **kwargs: None,
            model_loader=lambda **kwargs: FakeModel([]),
            generation_runner=_direct_generation_runner,
        )


def test_selected_manifest_accepts_shard_qualified_member_and_schema_versions(
    tmp_path,
):
    manifest, rows = _selected_manifest(tmp_path)
    rows[1] = SelectedBalalaikaClip(
        **(
            asdict(rows[1])
            | {
                "source_relative_path": "000002/0.mp3",
                "source_shard": "train/shard_000002.tar",
                "member_name": "0.mp3",
                "schema_version": 2,
            }
        )
    )
    _rewrite_manifest(manifest, rows)
    train_config, data_config = _write_configs(tmp_path)
    output_dir = tmp_path / "exp"
    runner, loader, _, _, _ = _fake_dependencies(
        output_dir, [2e-4, 9e-5, 8e-5], "eval_loss_target"
    )

    result = run_memorization(
        selected_manifest=manifest,
        output_dir=output_dir,
        train_config=train_config,
        data_config=data_config,
        command_runner=runner,
        model_loader=loader,
        generation_runner=_direct_generation_runner,
    )

    assert result.required_target_reached is True


@pytest.mark.parametrize("hash_field", ["source_sha256", "wav_sha256"])
def test_selected_manifest_rejects_duplicate_recorded_hashes(tmp_path, hash_field):
    manifest, rows = _selected_manifest(tmp_path)
    updates = {hash_field: getattr(rows[0], hash_field)}
    if hash_field == "wav_sha256":
        first_bytes = Path(rows[0].audio_path).read_bytes()
        Path(rows[1].audio_path).write_bytes(first_bytes)
        updates["duration"] = rows[0].duration
    rows[1] = SelectedBalalaikaClip(**(asdict(rows[1]) | updates))
    _rewrite_manifest(manifest, rows)
    train_config, data_config = _write_configs(tmp_path)

    with pytest.raises(ValueError, match="unique.*hash"):
        run_memorization(
            selected_manifest=manifest,
            output_dir=tmp_path / "exp",
            train_config=train_config,
            data_config=data_config,
            command_runner=lambda *args, **kwargs: None,
            model_loader=lambda **kwargs: FakeModel([]),
            generation_runner=_direct_generation_runner,
        )


def test_copy_corruption_is_detected_before_publication(tmp_path, monkeypatch):
    manifest, _ = _selected_manifest(tmp_path)
    train_config, data_config = _write_configs(tmp_path)
    output_dir = tmp_path / "exp"
    before = _seed_published_artifacts(output_dir)
    real_copy = __import__("shutil").copy2

    def corrupting_copy(source, destination):
        result = real_copy(source, destination)
        with Path(destination).open("ab") as output:
            output.write(b"corruption")
        return result

    monkeypatch.setattr(
        "omnivoice.validation.memorization.shutil.copy2", corrupting_copy
    )
    with pytest.raises(ValueError, match="copied original hash"):
        run_memorization(
            selected_manifest=manifest,
            output_dir=output_dir,
            train_config=train_config,
            data_config=data_config,
            command_runner=lambda *args, **kwargs: None,
            model_loader=lambda **kwargs: FakeModel([]),
            generation_runner=_direct_generation_runner,
        )

    assert _published_snapshot(output_dir) == before


@pytest.mark.parametrize(
    ("steps", "outcome", "message"),
    [
        (
            [50, 75, 100],
            TrainingOutcome(100, "wall_clock_limit", 2e-4, False),
            "exactly every 25",
        ),
        (
            [25, 75, 100],
            TrainingOutcome(100, "wall_clock_limit", 2e-4, False),
            "exactly every 25",
        ),
        (
            [25, 50, 75],
            TrainingOutcome(75, "completed", 2e-4, False),
            "step 10000",
        ),
        (
            [25, 50, 75],
            TrainingOutcome(75, "stop_after_step", 2e-4, False),
            "step 10000",
        ),
    ],
)
def test_run_consistency_rejects_missing_evals_or_wrong_terminal(
    steps, outcome, message
):
    history = [
        {"step": step, "loss": 2e-4, "elapsed_seconds": float(step)}
        for step in steps
    ]

    with pytest.raises(ValueError, match=message):
        _validate_run_consistency(outcome, history)


@pytest.mark.filterwarnings("ignore:This process.*use of fork")
def test_bounded_generation_accepts_local_injected_loader(tmp_path):
    _, rows = _selected_manifest(tmp_path)

    def local_loader(*, base_model=None, adapter_checkpoint=None):
        del base_model, adapter_checkpoint
        return FakeModel([])

    generated = _run_generation_bounded(
        rows=rows,
        output_dir=tmp_path / "generated",
        model_loader=local_loader,
        base_model="k2-fsa/OmniVoice",
        adapter_checkpoint=None,
        deadline_monotonic=time.monotonic() + 5,
    )

    assert len(generated) == 4


def test_generation_start_failure_closes_pipe_endpoints(tmp_path, monkeypatch):
    _, rows = _selected_manifest(tmp_path)

    class FakeConnection:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class FakeProcess:
        pid = None

        def start(self):
            raise OSError("start failed")

    parent_connection = FakeConnection()
    child_connection = FakeConnection()

    class FakeContext:
        def Pipe(self, *, duplex):
            assert duplex is False
            return parent_connection, child_connection

        def Process(self, **kwargs):
            del kwargs
            return FakeProcess()

    monkeypatch.setattr(
        "omnivoice.validation.memorization.mp.get_context",
        lambda method: FakeContext(),
    )

    with pytest.raises(OSError, match="start failed"):
        _run_generation_bounded(
            rows=rows,
            output_dir=tmp_path / "generated",
            model_loader=lambda **kwargs: FakeModel([]),
            base_model="k2-fsa/OmniVoice",
            adapter_checkpoint=None,
            deadline_monotonic=time.monotonic() + 5,
        )

    assert parent_connection.closed is True
    assert child_connection.closed is True
