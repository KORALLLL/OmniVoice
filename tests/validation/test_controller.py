import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from omnivoice.validation.controller import (
    ValidationController,
    validation_boundaries,
    validation_interval_steps,
)


def _write_configs(tmp_path: Path, *, steps: int = 1400) -> tuple[Path, Path, Path]:
    train_config = tmp_path / "train.json"
    train_config.write_text(
        json.dumps(
            {
                "steps": steps,
                "steps_per_epoch": 5000,
                "lora_enabled": True,
                "init_from_checkpoint": "k2-fsa/OmniVoice",
            }
        )
    )
    data_config = tmp_path / "data.json"
    data_config.write_text('{"train": [], "dev": []}\n')
    validation_config = tmp_path / "validation.json"
    validation_config.write_text(
        json.dumps(
            {
                "steps_per_epoch": 5000,
                "dataset_repo": "bitmanagerai/hard_number_eval_for_tts",
                "dataset_revision": "57b964492ccfcedd6a24d0225ef4b7d3697ffdca",
                "base_model": "k2-fsa/OmniVoice",
                "wandb_project": "omnivoice-lora-validation",
                "world_size": 8,
                "seed": 42,
            }
        )
    )
    return train_config, data_config, validation_config


def _publish_checkpoint(output_dir: Path, step: int, *, complete: bool = True) -> None:
    checkpoint = output_dir / f"checkpoint-{step}"
    adapter = checkpoint / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}\n")
    (checkpoint / "adapter_metadata.json").write_text(
        json.dumps({"format_version": 1, "step": step})
    )
    if complete:
        (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
        (checkpoint / "train_config.json").write_text("{}\n")
        (checkpoint / "tokenizer_config.json").write_text("{}\n")
        (checkpoint / "optimizer.bin").write_bytes(b"optimizer")
        (checkpoint / "scheduler.bin").write_bytes(b"scheduler")


class FakeStore:
    def __init__(self, path: Path, project: str, calls: list[str]) -> None:
        self.path = path
        self.project = project
        self.calls = calls

    def preflight(self) -> None:
        self.calls.append("wandb-preflight")

    def load_or_create_id(self) -> str:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text('{"run_id":"stable-wandb-id"}\n')
        return "stable-wandb-id"


class AdvancingClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        self.value += 1.25
        return self.value


def _stage(command: list[str]) -> str:
    module = command[command.index("-m") + 1]
    if module == "omnivoice.cli.train":
        return "train"
    return command[command.index(module) + 1]


class Harness:
    def __init__(
        self,
        tmp_path: Path,
        *,
        steps: int = 1400,
        fail_stage: str | None = None,
        incomplete_checkpoint: bool = False,
    ) -> None:
        self.steps = steps
        self.train_config, self.data_config, self.validation_config = _write_configs(
            tmp_path, steps=steps
        )
        self.output_dir = tmp_path / "train-output"
        self.validation_root = tmp_path / "validation-output"
        self.selected_manifest = tmp_path / "selected.jsonl"
        self.selected_manifest.write_text('{"role":"validation_voice"}\n')
        self.calls: list[str] = []
        self.commands: list[list[str]] = []
        self.fail_stage = fail_stage
        self.failed = False
        self.incomplete_checkpoint = incomplete_checkpoint
        self.clock = AdvancingClock()

    def store_factory(self, path: Path, project: str) -> FakeStore:
        return FakeStore(path, project, self.calls)

    def base_resolver(self, model: str, revision: str | None = None):
        self.calls.append("hf-preflight")
        assert model == "k2-fsa/OmniVoice"
        assert revision in (None, "base-commit")
        return Path("/hub/snapshots/base-commit"), "base-commit"

    def assignment_preparer(
        self, selected_manifest: Path, assignments_path: Path, config
    ) -> Path:
        self.calls.append("prepare-assignments")
        assert selected_manifest == self.selected_manifest
        assert config.seed == 42
        assignments_path.parent.mkdir(parents=True, exist_ok=True)
        assignments_path.write_text("immutable assignments\n")
        return assignments_path

    def runner(self, command, **kwargs):
        command = list(command)
        stage = _stage(command)
        self.commands.append(command)
        state = json.loads((self.validation_root / "controller_state.json").read_text())
        assert state["stage"] == stage
        assert state["command_status"] == "running"
        assert state["last_command"] == command
        assert kwargs["check"] is True
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["timeout"] > 0

        if self.fail_stage == stage and not self.failed:
            self.failed = True
            raise subprocess.CalledProcessError(2, command, output='{"complete":false}')

        stdout = '{"complete":true}\n'
        if stage == "train":
            assert json.loads(self.train_config.read_text())["steps"] == self.steps
            assert "--steps" not in command
            boundary = int(command[command.index("--stop-after-step") + 1])
            _publish_checkpoint(
                self.output_dir,
                boundary,
                complete=not self.incomplete_checkpoint,
            )
            stdout = json.dumps(
                {
                    "step": boundary,
                    "stop_reason": "stop_after_step",
                    "last_eval_loss": boundary / 100_000,
                    "target_reached": False,
                }
            )
        return SimpleNamespace(stdout=stdout, returncode=0)

    def controller(self) -> ValidationController:
        return ValidationController(
            train_config=self.train_config,
            data_config=self.data_config,
            validation_config=self.validation_config,
            selected_manifest=self.selected_manifest,
            output_dir=self.output_dir,
            validation_output_root=self.validation_root,
            deadline_monotonic=10_000.0,
            command_runner=self.runner,
            wandb_store_factory=self.store_factory,
            assignment_preparer=self.assignment_preparer,
            base_resolver=self.base_resolver,
            monotonic=self.clock,
        )


def test_validation_boundaries_are_one_eighth_epoch_and_include_final_step():
    assert validation_interval_steps(5000) == 625
    assert validation_interval_steps(1) == 1
    assert validation_boundaries(
        current_step=0, total_steps=1400, steps_per_epoch=5000
    ) == [625, 1250, 1400]
    assert validation_boundaries(
        current_step=625, total_steps=1400, steps_per_epoch=5000
    ) == [1250, 1400]


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_validation_interval_rejects_non_positive_integers(value):
    with pytest.raises((TypeError, ValueError), match="positive integer"):
        validation_interval_steps(value)


def test_controller_runs_base_then_checkpoint_isolated_cycles(tmp_path: Path):
    harness = Harness(tmp_path)

    state = harness.controller().run()

    observed = [_stage(command) for command in harness.commands]
    assert harness.calls[:3] == [
        "wandb-preflight",
        "hf-preflight",
        "prepare-assignments",
    ]
    assert observed[:9] == [
        "synth",
        "asr",
        "score",
        "train",
        "synth",
        "asr",
        "score",
        "train",
        "synth",
    ]

    base_synth = harness.commands[0]
    assert base_synth[base_synth.index("--step") + 1] == "0"
    assert base_synth[base_synth.index("--model") + 1] == "/hub/snapshots/base-commit"
    first_train = harness.commands[3]
    assert first_train[first_train.index("--stop-after-step") + 1] == "625"
    assert "--resume-from-checkpoint" not in first_train
    first_adapter_synth = harness.commands[4]
    assert first_adapter_synth[
        first_adapter_synth.index("--adapter-checkpoint") + 1
    ] == str(harness.output_dir / "checkpoint-625")
    second_train = harness.commands[7]
    assert second_train[second_train.index("--resume-from-checkpoint") + 1] == str(
        harness.output_dir / "checkpoint-625"
    )
    assert second_train[second_train.index("--stop-after-step") + 1] == "1250"

    distributed = [
        "accelerate",
        "launch",
        "--multi_gpu",
        "--gpu_ids",
        "0,1,2,3,4,5,6,7",
        "--num_processes",
        "8",
    ]
    assert all(
        command[:7] == distributed
        for command in harness.commands
        if _stage(command) != "score"
    )
    assert all(
        command[0:3] == ["python", "-m", "omnivoice.cli.validate_hard_numbers"]
        for command in harness.commands
        if _stage(command) == "score"
    )
    assert state.stage == "complete"
    assert state.step == 1400
    assert state.base_validated is True
    assert state.wandb_id == "stable-wandb-id"
    assert (
        state.assignments_sha256
        == hashlib.sha256(b"immutable assignments\n").hexdigest()
    )


def test_deadline_and_measured_stage_durations_are_forwarded(tmp_path: Path):
    harness = Harness(tmp_path, steps=625)

    harness.controller().run()

    for command in harness.commands:
        if _stage(command) in {"synth", "asr"}:
            assert command[command.index("--deadline-monotonic") + 1] == "10000.0"
    scores = [command for command in harness.commands if _stage(command) == "score"]
    for command in scores:
        assert float(command[command.index("--synthesis-seconds") + 1]) == 1.25
        assert float(command[command.index("--asr-seconds") + 1]) == 1.25
        assert float(command[command.index("--wall-time-seconds") + 1]) > 0
    checkpoint_score = scores[-1]
    assert checkpoint_score[checkpoint_score.index("--dev-loss") + 1] == "0.00625"


@pytest.mark.parametrize("failure_stage", ["synth", "asr", "score", "train"])
def test_incomplete_stage_halts_and_rerun_resumes_exact_stage(
    tmp_path: Path, failure_stage: str
):
    harness = Harness(tmp_path, steps=625, fail_stage=failure_stage)
    controller = harness.controller()

    with pytest.raises(subprocess.CalledProcessError):
        controller.run()

    failed_command = harness.commands[-1]
    failed_state = json.loads(
        (harness.validation_root / "controller_state.json").read_text()
    )
    assert failed_state["stage"] == failure_stage
    assert failed_state["command_status"] == "failed"
    assert failed_state["wandb_id"] == "stable-wandb-id"
    assert (
        failed_state["assignments_sha256"]
        == hashlib.sha256(b"immutable assignments\n").hexdigest()
    )

    controller.run()

    assert (
        harness.commands[harness.commands.index(failed_command) + 1] == failed_command
    )
    recovered = json.loads(
        (harness.validation_root / "controller_state.json").read_text()
    )
    assert recovered["wandb_id"] == failed_state["wandb_id"]
    assert recovered["assignments_sha256"] == failed_state["assignments_sha256"]


def test_incomplete_checkpoint_halts_before_synthesis_and_retries_train(
    tmp_path: Path,
):
    harness = Harness(tmp_path, steps=625, incomplete_checkpoint=True)
    controller = harness.controller()

    with pytest.raises(FileNotFoundError, match="complete checkpoint"):
        controller.run()

    assert [_stage(command) for command in harness.commands][-1] == "train"
    state = json.loads((harness.validation_root / "controller_state.json").read_text())
    assert state["stage"] == "train"
    assert state["command_status"] == "failed"


def test_process_death_replays_command_persisted_as_running(tmp_path: Path):
    harness = Harness(tmp_path, steps=625, fail_stage="score")
    controller = harness.controller()

    with pytest.raises(subprocess.CalledProcessError):
        controller.run()

    interrupted_command = harness.commands[-1]
    state_path = harness.validation_root / "controller_state.json"
    state = json.loads(state_path.read_text())
    state["command_status"] = "running"
    state["last_error"] = None
    state_path.write_text(json.dumps(state))

    controller.run()

    assert harness.commands[3] == interrupted_command


def test_preflight_failure_starts_no_subprocess(tmp_path: Path):
    harness = Harness(tmp_path)

    class FailingStore(FakeStore):
        def preflight(self) -> None:
            self.calls.append("wandb-preflight")
            raise RuntimeError("login required")

    harness.store_factory = lambda path, project: FailingStore(
        path, project, harness.calls
    )

    with pytest.raises(RuntimeError, match="login required"):
        harness.controller().run()

    assert harness.commands == []
    assert harness.calls == ["wandb-preflight"]
