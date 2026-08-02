"""Bounded four-utterance memorization orchestration and listening artifacts."""

from __future__ import annotations

import gc
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import soundfile as sf
import torch

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.cli.eval_memorization import check_memorization_patience
from omnivoice.training.config import TrainingConfig
from omnivoice.training.control import TrainingOutcome
from omnivoice.validation.balalaika import SelectedBalalaikaClip

BASE_MODEL_ID = "k2-fsa/OmniVoice"
REQUIRED_LOSS = 1e-4
STRETCH_LOSS = 1e-5
REQUIRED_PATIENCE = 2
MAX_OPTIMIZER_STEPS = 10_000
EVAL_STEPS = 25
_CHECKPOINT_NAME = re.compile(r"^checkpoint-(\d+)$")

CommandRunner = Callable[..., Any]
ModelLoader = Callable[..., Any]


@dataclass(frozen=True)
class MemorizationRunResult:
    """Machine-readable outcome of one four-utterance memorization run."""

    required_target_reached: bool
    stretch_target_reached: bool
    qualifying_step: int | None
    minimum_loss: float | None
    selected_checkpoint: str
    training_outcome: TrainingOutcome
    experiment_deadline_monotonic: float
    miss_reason: str | None


def _clip_id(row: SelectedBalalaikaClip, index: int) -> str:
    return f"memorization-{index:02d}-{row.wav_sha256[:16]}"


def _prepare_wav_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for wav_path in path.glob("*.wav"):
        wav_path.unlink()


def generate_four(
    rows: Sequence[SelectedBalalaikaClip], output_dir: Path, model: Any
) -> list[Path]:
    """Generate one deterministic voice-clone WAV for each memorization row."""
    if len(rows) != 4:
        raise ValueError(f"expected exactly four memorization rows, got {len(rows)}")
    output_path = Path(output_dir)
    _prepare_wav_directory(output_path)
    config = OmniVoiceGenerationConfig(
        num_step=32,
        guidance_scale=2.0,
        t_shift=0.1,
        layer_penalty_factor=5.0,
        position_temperature=0.0,
        class_temperature=0.0,
    )
    generated = []
    for index, row in enumerate(rows):
        audios = model.generate(
            text=row.text,
            language="Russian",
            ref_text=row.text,
            ref_audio=str(row.audio_path),
            generation_config=config,
        )
        if len(audios) != 1:
            raise ValueError("OmniVoice must return exactly one audio per utterance")
        output_wav = output_path / f"{_clip_id(row, index)}.wav"
        sf.write(
            output_wav,
            audios[0],
            model.sampling_rate,
            subtype="PCM_16",
        )
        generated.append(output_wav)
    return generated


def read_loss_history(path: str | Path) -> list[dict[str, int | float]]:
    """Read and validate the bounded trainer's JSONL evaluation history."""
    history_path = Path(path)
    if history_path.is_dir():
        history_path = history_path / "loss_history.jsonl"
    history = []
    with history_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                not isinstance(row, dict)
                or type(row.get("step")) is not int
                or not isinstance(row.get("loss"), (int, float))
                or isinstance(row.get("loss"), bool)
            ):
                raise ValueError(f"invalid loss history row {line_number}")
            history.append(
                {
                    "step": row["step"],
                    "loss": float(row["loss"]),
                    "elapsed_seconds": float(row.get("elapsed_seconds", 0.0)),
                }
            )
    if not history:
        raise ValueError("loss history is empty")
    return history


def _completed_checkpoints(output_dir: Path) -> dict[int, Path]:
    checkpoints = {}
    if not output_dir.is_dir():
        return checkpoints
    for candidate in output_dir.iterdir():
        match = _CHECKPOINT_NAME.fullmatch(candidate.name)
        if candidate.is_dir() and match is not None:
            checkpoints[int(match.group(1))] = candidate
    return checkpoints


def choose_generation_checkpoint(
    output_dir: str | Path,
    loss_history: Sequence[Mapping[str, int | float]],
    *,
    threshold: float = REQUIRED_LOSS,
    patience: int = REQUIRED_PATIENCE,
) -> Path:
    """Choose the qualifying checkpoint, or the highest completed fallback."""
    output_path = Path(output_dir)
    completed = _completed_checkpoints(output_path)
    if not completed:
        raise FileNotFoundError(f"no completed checkpoints in {output_path}")
    patience_result = check_memorization_patience(
        ((int(row["step"]), float(row["loss"])) for row in loss_history),
        threshold=threshold,
        patience=patience,
    )
    if (
        patience_result.qualifying_step is not None
        and patience_result.qualifying_step in completed
    ):
        return completed[patience_result.qualifying_step]
    return completed[max(completed)]


def _read_selected_rows(path: Path) -> list[SelectedBalalaikaClip]:
    field_names = {field.name for field in fields(SelectedBalalaikaClip)}
    rows = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(
                    f"selected manifest row {line_number} is not an object"
                )
            try:
                row = SelectedBalalaikaClip(
                    **{name: payload[name] for name in field_names}
                )
            except (KeyError, TypeError) as error:
                raise ValueError(
                    f"invalid selected manifest row {line_number}"
                ) from error
            if row.role == "memorization":
                rows.append(row)
    if len(rows) != 4:
        raise ValueError(f"expected exactly four memorization rows, got {len(rows)}")
    for row in rows:
        if not Path(row.audio_path).is_file():
            raise FileNotFoundError(row.audio_path)
    return rows


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _copy_originals(
    rows: Sequence[SelectedBalalaikaClip], output_dir: Path
) -> list[Path]:
    _prepare_wav_directory(output_dir)
    originals = []
    for index, row in enumerate(rows):
        destination = output_dir / f"{_clip_id(row, index)}.wav"
        shutil.copy2(row.audio_path, destination)
        originals.append(destination)
    return originals


def _write_training_manifest(
    rows: Sequence[SelectedBalalaikaClip], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for index, row in enumerate(rows):
            output.write(
                json.dumps(
                    {
                        "id": _clip_id(row, index),
                        "audio_path": str(Path(row.audio_path).resolve()),
                        "text": row.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _write_runtime_data_config(source: Path, destination: Path, tokens: Path) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("data config must be an object")
    manifest_path = str((tokens / "data.lst").resolve())
    for split in ("train", "dev"):
        entries = payload.get(split)
        if not isinstance(entries, list) or not entries:
            entries = [{}]
            payload[split] = entries
        for entry in entries:
            if not isinstance(entry, dict):
                raise TypeError(f"data config {split} entries must be objects")
            entry["manifest_path"] = [manifest_path]
    _write_json_atomic(destination, payload)


def _write_runtime_train_config(
    source: Path,
    destination: Path,
    *,
    output_dir: Path,
    data_config: Path,
    history_path: Path,
    remaining_wall_clock_seconds: float,
) -> None:
    config = TrainingConfig.from_json(str(source))
    config.output_dir = str(output_dir)
    config.data_config = str(data_config)
    config.steps = MAX_OPTIMIZER_STEPS
    config.stop_after_step = MAX_OPTIMIZER_STEPS
    config.eval_steps = EVAL_STEPS
    config.save_steps = EVAL_STEPS
    config.early_stop_eval_loss = REQUIRED_LOSS
    config.early_stop_patience = REQUIRED_PATIENCE
    config.max_wall_clock_seconds = remaining_wall_clock_seconds
    config.eval_history_path = str(history_path)
    config.validate()
    _write_json_atomic(destination, asdict(config))


def _tokenizer_command(manifest: Path, token_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "omnivoice.scripts.extract_audio_tokens",
        "--input_jsonl",
        str(manifest),
        "--tar_output_pattern",
        str(token_dir / "audios/shard-%06d.tar"),
        "--jsonl_output_pattern",
        str(token_dir / "txts/shard-%06d.jsonl"),
        "--tokenizer_path",
        "eustlb/higgs-audio-v2-tokenizer",
        "--samples_per_shard",
        "1",
        "--min_num_shards",
        "4",
        "--nj_per_gpu",
        "1",
        "--loader_workers",
        "1",
        "--shuffle",
        "False",
        "--shuffle-seed",
        "42",
    ]


def _training_command(
    train_config: Path, data_config: Path, output_dir: Path
) -> list[str]:
    return [
        "accelerate",
        "launch",
        "--gpu_ids",
        "0",
        "--num_processes",
        "1",
        "-m",
        "omnivoice.cli.train",
        "--train_config",
        str(train_config),
        "--data_config",
        str(data_config),
        "--output_dir",
        str(output_dir),
        "--stop-after-step",
        str(MAX_OPTIMIZER_STEPS),
    ]


def _parse_training_outcome(stdout: str) -> TrainingOutcome:
    for line in reversed(stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and {
            "step",
            "stop_reason",
            "last_eval_loss",
            "target_reached",
        } <= payload.keys():
            return TrainingOutcome(
                step=payload["step"],
                stop_reason=payload["stop_reason"],
                last_eval_loss=payload["last_eval_loss"],
                target_reached=payload["target_reached"],
            )
    raise ValueError("training command did not emit a TrainingOutcome JSON object")


def _default_model_loader(*, base_model=None, adapter_checkpoint=None):
    if (base_model is None) == (adapter_checkpoint is None):
        raise ValueError("specify exactly one model source")
    if base_model is not None:
        return OmniVoice.from_pretrained(base_model, device_map="cuda:0")
    return OmniVoice.from_lora_pretrained(adapter_checkpoint, device_map="cuda:0")


def run_memorization(
    *,
    selected_manifest: str | Path,
    output_dir: str | Path,
    train_config: str | Path,
    data_config: str | Path,
    max_wall_clock_seconds: float = 1200,
    experiment_wall_clock_seconds: float = 3600,
    command_runner: CommandRunner = subprocess.run,
    model_loader: ModelLoader = _default_model_loader,
) -> MemorizationRunResult:
    """Run bounded training and save original, initial, and final WAVs."""
    if max_wall_clock_seconds <= 0:
        raise ValueError("max_wall_clock_seconds must be positive")
    if experiment_wall_clock_seconds <= 0:
        raise ValueError("experiment_wall_clock_seconds must be positive")

    started = time.monotonic()
    phase_deadline = started + max_wall_clock_seconds
    experiment_deadline = started + experiment_wall_clock_seconds
    selected_path = Path(selected_manifest).resolve()
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    rows = _read_selected_rows(selected_path)

    _copy_originals(rows, output_path / "original")
    training_manifest = output_path / "four.jsonl"
    _write_training_manifest(rows, training_manifest)
    base_model = model_loader(base_model=BASE_MODEL_ID, adapter_checkpoint=None)
    generate_four(rows, output_path / "generated/initial", base_model)
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    token_dir = output_path / "tokens"
    command_runner(_tokenizer_command(training_manifest, token_dir), check=True)

    remaining = phase_deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("memorization phase expired before training")
    runtime_data_config = output_path / "data_config.json"
    runtime_train_config = output_path / "train_config.json"
    history_path = output_path / "loss_history.jsonl"
    history_path.unlink(missing_ok=True)
    _write_runtime_data_config(Path(data_config), runtime_data_config, token_dir)
    _write_runtime_train_config(
        Path(train_config),
        runtime_train_config,
        output_dir=output_path,
        data_config=runtime_data_config,
        history_path=history_path,
        remaining_wall_clock_seconds=remaining,
    )
    completed = command_runner(
        _training_command(runtime_train_config, runtime_data_config, output_path),
        check=True,
        capture_output=True,
        text=True,
    )
    outcome = _parse_training_outcome(completed.stdout)

    history = read_loss_history(history_path)
    patience_result = check_memorization_patience(
        ((int(row["step"]), float(row["loss"])) for row in history),
        threshold=REQUIRED_LOSS,
        patience=REQUIRED_PATIENCE,
    )
    checkpoint = choose_generation_checkpoint(output_path, history)
    adapter_model = model_loader(base_model=None, adapter_checkpoint=checkpoint)
    generate_four(rows, output_path / "generated/final", adapter_model)

    required_target_reached = patience_result.passed
    minimum_loss = patience_result.minimum_loss
    stretch_target_reached = (
        minimum_loss is not None and minimum_loss <= STRETCH_LOSS
    )
    miss_reason = None
    if not required_target_reached:
        miss_reason = f"{outcome.stop_reason}_before_required_target"
    result = MemorizationRunResult(
        required_target_reached=required_target_reached,
        stretch_target_reached=stretch_target_reached,
        qualifying_step=patience_result.qualifying_step,
        minimum_loss=minimum_loss,
        selected_checkpoint=str(checkpoint),
        training_outcome=outcome,
        experiment_deadline_monotonic=experiment_deadline,
        miss_reason=miss_reason,
    )
    _write_json_atomic(output_path / "result.json", asdict(result))
    return result
