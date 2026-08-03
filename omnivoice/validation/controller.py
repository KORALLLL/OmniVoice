"""Checkpoint-isolated LoRA training and hard-number validation controller."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from omnivoice.training.config import TrainingConfig
from omnivoice.training.control import TrainingOutcome
from omnivoice.training.lora import read_lora_metadata, resolve_adapter_dir
from omnivoice.validation.artifacts import ValidationPaths
from omnivoice.validation.balalaika import SelectedBalalaikaClip
from omnivoice.validation.hard_numbers import (
    HARD_NUMBER_REPO,
    HARD_NUMBER_REVISION,
    assign_voices,
    download_hard_number_jsonl,
    load_hard_number_rows,
    write_assignment_manifest,
)
from omnivoice.validation.wandb_logging import WandbRunStore

_DISTRIBUTED_PREFIX = [
    "accelerate",
    "launch",
    "--multi_gpu",
    "--gpu_ids",
    "0,1,2,3,4,5,6,7",
    "--num_processes",
    "8",
]
_VALIDATION_MODULE = "omnivoice.cli.validate_hard_numbers"
_TRAIN_MODULE = "omnivoice.cli.train"
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "spiece.model",
    "vocab.json",
    "vocab.txt",
)

CommandRunner = Callable[..., Any]
StoreFactory = Callable[[Path, str], Any]
AssignmentPreparer = Callable[[Path, Path, "ValidationConfig"], Path]
BaseResolver = Callable[[str, str | None], tuple[Path, str]]


def validation_interval_steps(steps_per_epoch: int) -> int:
    """Return the integer optimizer-step interval for one eighth of an epoch."""
    if type(steps_per_epoch) is not int or steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be a positive integer")
    return math.ceil(steps_per_epoch / 8)


def validation_boundaries(
    current_step: int, total_steps: int, steps_per_epoch: int
) -> list[int]:
    """Return future eighth-epoch boundaries, always including the final step."""
    if type(current_step) is not int or current_step < 0:
        raise ValueError("current_step must be a non-negative integer")
    if type(total_steps) is not int or total_steps <= 0:
        raise ValueError("total_steps must be a positive integer")
    if current_step > total_steps:
        raise ValueError("current_step cannot exceed total_steps")
    interval = validation_interval_steps(steps_per_epoch)
    first = ((current_step // interval) + 1) * interval
    boundaries = list(range(first, total_steps + 1, interval))
    if current_step < total_steps and (not boundaries or boundaries[-1] != total_steps):
        boundaries.append(total_steps)
    return boundaries


@dataclass(frozen=True)
class ValidationConfig:
    """Pinned inputs and distributed shape for hard-number validation."""

    steps_per_epoch: int
    dataset_repo: str
    dataset_revision: str
    base_model: str
    wandb_project: str
    world_size: int
    seed: int

    @classmethod
    def from_json(cls, path: str | Path) -> ValidationConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) != {
            field.name for field in fields(cls)
        }:
            raise ValueError("validation config has an invalid schema")
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        validation_interval_steps(self.steps_per_epoch)
        if self.dataset_repo != HARD_NUMBER_REPO:
            raise ValueError(f"dataset_repo must be pinned to {HARD_NUMBER_REPO}")
        if self.dataset_revision != HARD_NUMBER_REVISION:
            raise ValueError(
                f"dataset_revision must be pinned to {HARD_NUMBER_REVISION}"
            )
        if not isinstance(self.base_model, str) or not self.base_model.strip():
            raise ValueError("base_model must be a non-blank string")
        if not isinstance(self.wandb_project, str) or not self.wandb_project.strip():
            raise ValueError("wandb_project must be a non-blank string")
        if type(self.world_size) is not int or self.world_size != 8:
            raise ValueError("world_size must be exactly 8")
        if type(self.seed) is not int or self.seed != 42:
            raise ValueError("seed must be exactly 42")


@dataclass
class ControllerState:
    """Durable controller position and immutable experiment identity."""

    run_id: str
    wandb_id: str | None
    stage: str
    step: int
    checkpoint: str | None
    pending_step: int | None
    pending_checkpoint: str | None
    pending_dev_loss: float | None
    base_validated: bool
    deadline_monotonic: float
    resolved_base_path: str | None
    resolved_base_revision: str | None
    dataset_revision: str
    assignments_path: str
    assignments_sha256: str | None
    train_config_sha256: str
    data_config_sha256: str
    validation_config_sha256: str
    started_monotonic: float
    stage_durations: dict[str, float]
    last_command: list[str] | None = None
    command_status: str | None = None
    last_error: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ControllerState:
        expected = {field.name for field in fields(cls)}
        if set(payload) != expected:
            raise ValueError("controller state has an invalid schema")
        return cls(**dict(payload))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            json.dump(
                payload,
                destination,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _load_validation_voices(path: Path) -> list[SelectedBalalaikaClip]:
    names = {field.name for field in fields(SelectedBalalaikaClip)}
    voices: list[SelectedBalalaikaClip] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict) or not names <= set(payload):
                raise ValueError(f"invalid selected manifest row {line_number}")
            row = SelectedBalalaikaClip(**{name: payload[name] for name in names})
            if row.role == "validation_voice":
                voices.append(row)
    if len(voices) != 20:
        raise ValueError(f"expected exactly 20 validation voices, got {len(voices)}")
    return voices


def prepare_validation_assignments(
    selected_manifest: Path,
    assignments_path: Path,
    config: ValidationConfig,
) -> Path:
    """Download pinned rows and atomically create the immutable assignment file."""
    dataset_path = download_hard_number_jsonl()
    rows = load_hard_number_rows(dataset_path, revision=config.dataset_revision)
    voices = _load_validation_voices(selected_manifest)
    assignments = assign_voices(rows, voices, seed=config.seed)
    return write_assignment_manifest(assignments, assignments_path)


def resolve_base_model(model: str, revision: str | None = None) -> tuple[Path, str]:
    """Resolve the base model to an immutable local Hub snapshot."""
    snapshot = Path(snapshot_download(repo_id=model, revision=revision)).resolve()
    resolved_revision = snapshot.name
    if not resolved_revision:
        raise ValueError("Hugging Face snapshot has no resolved revision")
    return snapshot, resolved_revision


def require_complete_checkpoint(checkpoint: str | Path, expected_step: int) -> Path:
    """Require an exact, restartable LoRA checkpoint at ``expected_step``."""
    path = Path(checkpoint)
    if path.name != f"checkpoint-{expected_step}":
        raise ValueError(
            f"checkpoint path must be exactly checkpoint-{expected_step}: {path}"
        )
    try:
        root, adapter = resolve_adapter_dir(path)
        metadata = read_lora_metadata(root)
    except (FileNotFoundError, OSError, ValueError) as error:
        raise FileNotFoundError(f"complete checkpoint is missing: {path}") from error
    required_files = [
        adapter / "adapter_config.json",
        root / "adapter_metadata.json",
        root / "train_config.json",
    ]
    adapter_weights = [
        adapter / "adapter_model.safetensors",
        adapter / "adapter_model.bin",
    ]
    tokenizer_files = [root / name for name in _TOKENIZER_FILES]
    if (
        metadata.get("step") != expected_step
        or any(
            not item.is_file() or item.stat().st_size <= 0 for item in required_files
        )
        or not any(
            item.is_file() and item.stat().st_size > 0 for item in adapter_weights
        )
        or not any(
            item.is_file() and item.stat().st_size > 0 for item in tokenizer_files
        )
        or not any(root.glob("optimizer*"))
        or not any(root.glob("scheduler*"))
    ):
        raise FileNotFoundError(f"complete checkpoint is missing: {path}")
    return root


def _parse_training_outcome(stdout: str, expected_step: int) -> TrainingOutcome:
    for line in reversed(stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or set(payload) != {
            "step",
            "stop_reason",
            "last_eval_loss",
            "target_reached",
        }:
            continue
        loss = payload["last_eval_loss"]
        if (
            payload["step"] != expected_step
            or payload["stop_reason"] != "stop_after_step"
            or type(payload["target_reached"]) is not bool
            or payload["target_reached"]
            or isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not math.isfinite(loss)
            or loss < 0
        ):
            raise ValueError("training command emitted an invalid TrainingOutcome")
        return TrainingOutcome(
            step=expected_step,
            stop_reason="stop_after_step",
            last_eval_loss=float(loss),
            target_reached=False,
        )
    raise ValueError("training command did not emit a TrainingOutcome")


class ValidationController:
    """Alternate bounded training subprocesses with isolated validation stages."""

    def __init__(
        self,
        *,
        train_config: str | Path,
        data_config: str | Path,
        validation_config: str | Path,
        selected_manifest: str | Path,
        output_dir: str | Path,
        validation_output_root: str | Path,
        deadline_monotonic: float,
        resume_from_checkpoint: str | Path | None = None,
        command_runner: CommandRunner = subprocess.run,
        wandb_store_factory: StoreFactory = WandbRunStore,
        assignment_preparer: AssignmentPreparer = prepare_validation_assignments,
        base_resolver: BaseResolver = resolve_base_model,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.train_config_path = Path(train_config).resolve()
        self.data_config_path = Path(data_config).resolve()
        self.validation_config_path = Path(validation_config).resolve()
        self.selected_manifest = Path(selected_manifest).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.validation_output_root = Path(validation_output_root).resolve()
        self.state_path = self.validation_output_root / "controller_state.json"
        self.deadline_monotonic = float(deadline_monotonic)
        if not math.isfinite(self.deadline_monotonic):
            raise ValueError("deadline_monotonic must be finite")
        self.resume_from_checkpoint = (
            Path(resume_from_checkpoint).resolve()
            if resume_from_checkpoint is not None
            else None
        )
        self.command_runner = command_runner
        self.wandb_store_factory = wandb_store_factory
        self.assignment_preparer = assignment_preparer
        self.base_resolver = base_resolver
        self.monotonic = monotonic
        self.train = TrainingConfig.from_json(str(self.train_config_path))
        self.train.validate()
        if not self.train.lora_enabled:
            raise ValueError("validation controller requires LoRA training")
        if self.train.stop_after_step is not None:
            raise ValueError("train config stop_after_step must remain unset")
        if self.train.resume_from_checkpoint is not None:
            raise ValueError("use the controller --resume-from-checkpoint option")
        self.validation = ValidationConfig.from_json(self.validation_config_path)

    def _write_state(self, state: ControllerState) -> None:
        _atomic_json(self.state_path, asdict(state))

    def _new_state(self) -> ControllerState:
        run_id = uuid.uuid4().hex
        assignments = self.validation_output_root / run_id / "assignments.jsonl"
        step = 0
        checkpoint = None
        stage = "preflight"
        base_validated = False
        pending_step = None
        pending_checkpoint = None
        if self.resume_from_checkpoint is not None:
            try:
                step = int(self.resume_from_checkpoint.name.removeprefix("checkpoint-"))
            except ValueError as error:
                raise ValueError(
                    "resume checkpoint must end in checkpoint-<step>"
                ) from error
            require_complete_checkpoint(self.resume_from_checkpoint, step)
            checkpoint = str(self.resume_from_checkpoint)
            pending_step = step
            pending_checkpoint = checkpoint
            base_validated = True
            stage = "synth"
        state = ControllerState(
            run_id=run_id,
            wandb_id=None,
            stage=stage,
            step=step if stage == "preflight" else 0,
            checkpoint=None if stage != "preflight" else checkpoint,
            pending_step=pending_step,
            pending_checkpoint=pending_checkpoint,
            pending_dev_loss=None,
            base_validated=base_validated,
            deadline_monotonic=self.deadline_monotonic,
            resolved_base_path=None,
            resolved_base_revision=None,
            dataset_revision=self.validation.dataset_revision,
            assignments_path=str(assignments),
            assignments_sha256=None,
            train_config_sha256=_sha256_file(self.train_config_path),
            data_config_sha256=_sha256_file(self.data_config_path),
            validation_config_sha256=_sha256_file(self.validation_config_path),
            started_monotonic=self.monotonic(),
            stage_durations={},
        )
        self._write_state(state)
        return state

    def _load_state(self) -> ControllerState:
        if not self.state_path.exists():
            return self._new_state()
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("controller state must contain a JSON object")
        state = ControllerState.from_dict(payload)
        expected_hashes = {
            "train_config_sha256": _sha256_file(self.train_config_path),
            "data_config_sha256": _sha256_file(self.data_config_path),
            "validation_config_sha256": _sha256_file(self.validation_config_path),
        }
        for name, expected in expected_hashes.items():
            if getattr(state, name) != expected:
                raise ValueError(
                    f"{name.removesuffix('_sha256')} changed during resume"
                )
        if state.deadline_monotonic != self.deadline_monotonic:
            raise ValueError("experiment deadline changed during resume")
        if state.dataset_revision != self.validation.dataset_revision:
            raise ValueError("dataset revision changed during resume")
        if self.resume_from_checkpoint is not None and state.checkpoint != str(
            self.resume_from_checkpoint
        ):
            raise ValueError("resume checkpoint conflicts with controller state")
        return state

    def _preflight(self, state: ControllerState) -> None:
        paths = ValidationPaths(
            self.validation_output_root, state.run_id, state.pending_step or state.step
        )
        store = self.wandb_store_factory(paths.wandb_ids, self.validation.wandb_project)
        store.preflight()
        wandb_id = store.load_or_create_id()
        if state.wandb_id is not None and state.wandb_id != wandb_id:
            raise ValueError("persisted W&B identity changed during resume")
        state.wandb_id = wandb_id

        base_path, revision = self.base_resolver(
            self.validation.base_model, state.resolved_base_revision
        )
        if (
            state.resolved_base_revision is not None
            and state.resolved_base_revision != revision
        ):
            raise ValueError("resolved base revision changed during resume")
        state.resolved_base_path = str(base_path)
        state.resolved_base_revision = revision

        assignments_path = Path(state.assignments_path)
        if state.assignments_sha256 is None:
            prepared = self.assignment_preparer(
                self.selected_manifest, assignments_path, self.validation
            )
            if Path(prepared).resolve() != assignments_path.resolve():
                raise ValueError("assignment preparer returned an unexpected path")
            state.assignments_sha256 = _sha256_file(assignments_path)
        elif (
            not assignments_path.is_file()
            or _sha256_file(assignments_path) != state.assignments_sha256
        ):
            raise ValueError("immutable assignments changed during resume")

        if state.stage == "preflight":
            state.stage = "synth"
            state.pending_step = 0
        self._write_state(state)

    def _execute(self, state: ControllerState, command: Sequence[str]) -> Any:
        materialized = list(command)
        state.last_command = materialized
        state.command_status = "running"
        state.last_error = None
        self._write_state(state)
        started = self.monotonic()
        remaining = self.deadline_monotonic - started
        if remaining <= 0:
            state.command_status = "failed"
            state.last_error = "experiment deadline reached"
            self._write_state(state)
            raise TimeoutError("experiment deadline reached before subprocess")
        try:
            result = self.command_runner(
                materialized,
                check=True,
                capture_output=True,
                text=True,
                timeout=remaining,
            )
        except BaseException as error:
            state.command_status = "failed"
            state.last_error = f"{type(error).__name__}: {error}"
            self._write_state(state)
            raise
        duration = self.monotonic() - started
        key = f"{state.pending_step}:{state.stage}"
        state.stage_durations[key] = duration
        state.command_status = "completed"
        self._write_state(state)
        return result

    def _validation_command(self, state: ControllerState) -> list[str]:
        step = state.pending_step
        if step is None:
            raise RuntimeError("validation stage has no pending step")
        common = [
            "--assignments",
            state.assignments_path,
            "--output-root",
            str(self.validation_output_root),
            "--run-id",
            state.run_id,
            "--step",
            str(step),
        ]
        if state.stage == "synth":
            source = (
                ["--model", state.resolved_base_path]
                if step == 0
                else ["--adapter-checkpoint", state.pending_checkpoint]
            )
            return [
                *_DISTRIBUTED_PREFIX,
                "-m",
                _VALIDATION_MODULE,
                "synth",
                *common,
                *source,
                "--deadline-monotonic",
                str(self.deadline_monotonic),
            ]
        if state.stage == "asr":
            return [
                *_DISTRIBUTED_PREFIX,
                "-m",
                _VALIDATION_MODULE,
                "asr",
                *common,
                "--deadline-monotonic",
                str(self.deadline_monotonic),
            ]
        if state.stage == "score":
            command = [
                "python",
                "-m",
                _VALIDATION_MODULE,
                "score",
                *common,
                "--steps-per-epoch",
                str(self.validation.steps_per_epoch),
                "--synthesis-seconds",
                str(state.stage_durations[f"{step}:synth"]),
                "--asr-seconds",
                str(state.stage_durations[f"{step}:asr"]),
                "--wall-time-seconds",
                str(self.monotonic() - state.started_monotonic),
            ]
            if state.pending_dev_loss is not None:
                command.extend(["--dev-loss", str(state.pending_dev_loss)])
            return command
        raise AssertionError(f"unsupported validation stage {state.stage!r}")

    def _train_command(self, state: ControllerState) -> list[str]:
        if state.pending_step is None:
            boundaries = validation_boundaries(
                state.step, self.train.steps, self.validation.steps_per_epoch
            )
            if not boundaries:
                raise RuntimeError("no remaining training boundary")
            state.pending_step = boundaries[0]
            state.pending_checkpoint = str(
                self.output_dir / f"checkpoint-{state.pending_step}"
            )
            self._write_state(state)
        command = [
            *_DISTRIBUTED_PREFIX,
            "-m",
            _TRAIN_MODULE,
            "--train_config",
            str(self.train_config_path),
            "--data_config",
            str(self.data_config_path),
            "--output_dir",
            str(self.output_dir),
            "--stop-after-step",
            str(state.pending_step),
        ]
        if state.checkpoint is not None:
            command.extend(["--resume-from-checkpoint", state.checkpoint])
        return command

    def run(self) -> ControllerState:
        """Run or resume until every validation boundary is complete."""
        state = self._load_state()
        self._preflight(state)
        while state.stage != "complete":
            if state.stage in {"synth", "asr", "score"}:
                command = (
                    state.last_command
                    if state.command_status in {"failed", "running"}
                    and state.last_command
                    else self._validation_command(state)
                )
                self._execute(state, command)
                if state.stage == "synth":
                    state.stage = "asr"
                elif state.stage == "asr":
                    state.stage = "score"
                elif state.pending_step == 0:
                    state.base_validated = True
                    state.pending_step = None
                    state.stage = "train"
                else:
                    state.step = state.pending_step
                    state.checkpoint = state.pending_checkpoint
                    state.pending_step = None
                    state.pending_checkpoint = None
                    state.pending_dev_loss = None
                    state.stage = (
                        "complete" if state.step == self.train.steps else "train"
                    )
                self._write_state(state)
                continue

            if state.stage == "train":
                command = (
                    state.last_command
                    if state.command_status in {"failed", "running"}
                    and state.last_command
                    else self._train_command(state)
                )
                result = self._execute(state, command)
                try:
                    outcome = _parse_training_outcome(
                        result.stdout, expected_step=state.pending_step
                    )
                    require_complete_checkpoint(
                        state.pending_checkpoint, expected_step=state.pending_step
                    )
                except BaseException as error:
                    state.command_status = "failed"
                    state.last_error = f"{type(error).__name__}: {error}"
                    self._write_state(state)
                    raise
                state.pending_dev_loss = outcome.last_eval_loss
                state.stage = "synth"
                self._write_state(state)
                continue

            raise ValueError(f"unsupported controller stage {state.stage!r}")
        return state
