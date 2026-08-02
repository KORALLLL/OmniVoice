import json
import weakref
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from omnivoice.training.control import TrainingOutcome, append_loss_history
from omnivoice.validation.balalaika import SelectedBalalaikaClip
from omnivoice.validation.memorization import (
    choose_generation_checkpoint,
    generate_four,
    read_loss_history,
    run_memorization,
)


def _selected_manifest(tmp_path: Path) -> tuple[Path, list[SelectedBalalaikaClip]]:
    rows = []
    for index in range(4):
        audio_path = tmp_path / f"source-{index}.wav"
        sf.write(audio_path, np.zeros(240, dtype=np.float32), 24_000, subtype="PCM_16")
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
                wav_sha256=f"{index + 11:064x}",
                sample_rate=24_000,
                channels=1,
                duration=0.01,
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
    train_config.write_text(json.dumps({"steps": 20, "lora_enabled": True}))
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
            for step, loss in zip((50, 75, 100), losses):
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
    )
