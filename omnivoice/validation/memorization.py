"""Bounded four-utterance memorization orchestration and listening artifacts."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
import re
import shutil
import signal
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
from omnivoice.training.lora import DEFAULT_LORA_TARGET_MODULES
from omnivoice.validation.balalaika import SelectedBalalaikaClip

BASE_MODEL_ID = "k2-fsa/OmniVoice"
REQUIRED_LOSS = 1e-4
STRETCH_LOSS = 1e-5
REQUIRED_PATIENCE = 2
MAX_OPTIMIZER_STEPS = 10_000
EVAL_STEPS = 25
_CHECKPOINT_NAME = re.compile(r"^checkpoint-(\d+)$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_RELATIVE_PATH = re.compile(r"^(?P<shard>\d{6})/(?P<member>[^/]+\.mp3)$")
_KNOWN_STOP_REASONS = {
    "completed",
    "eval_loss_target",
    "stop_after_step",
    "wall_clock_limit",
}
_PUBLISHED_TARGETS = ("original", "generated", "four.jsonl", "result.json")
_TERMINATE_JOIN_SECONDS = 0.5

CommandRunner = Callable[..., Any]
ModelLoader = Callable[..., Any]
GenerationRunner = Callable[..., list[Path]]


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


def _run_process_group(
    command: Sequence[str],
    *,
    deadline_monotonic: float,
    text: bool = False,
) -> subprocess.CompletedProcess:
    """Run a command in a new process group and kill all descendants on timeout."""
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
        start_new_session=os.name == "posix",
    )
    try:
        try:
            stdout, stderr = process.communicate(
                timeout=_remaining_seconds(deadline_monotonic, "subprocess")
            )
        except subprocess.TimeoutExpired as error:
            _terminate_process_group(process)
            raise subprocess.TimeoutExpired(
                command,
                error.timeout,
                output=error.output,
                stderr=error.stderr,
            ) from None
        completed = subprocess.CompletedProcess(
            list(command), process.returncode, stdout, stderr
        )
        if process.returncode:
            raise subprocess.CalledProcessError(
                process.returncode,
                list(command),
                output=stdout,
                stderr=stderr,
            )
        return completed
    finally:
        for stream in (process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                stream.close()


def _signal_process_group(process: subprocess.Popen, signal_number: int) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal_number)
        elif signal_number == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()
    except ProcessLookupError:
        pass


def _process_group_exists(process: subprocess.Popen) -> bool:
    if os.name != "posix":
        return process.poll() is None
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return False
    return True


def _terminate_process_group(process: subprocess.Popen) -> None:
    _signal_process_group(process, signal.SIGTERM)
    try:
        process.communicate(timeout=_TERMINATE_JOIN_SECONDS)
    except subprocess.TimeoutExpired:
        pass
    if _process_group_exists(process):
        _signal_process_group(process, signal.SIGKILL)
    try:
        process.communicate(timeout=_TERMINATE_JOIN_SECONDS)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("subprocess group did not terminate after SIGKILL") from error


def _invoke_command(
    command: Sequence[str],
    *,
    deadline_monotonic: float,
    command_runner: CommandRunner | None,
    capture_output: bool = False,
    text: bool = False,
) -> Any:
    if command_runner is None or command_runner is subprocess.run:
        return _run_process_group(
            command,
            deadline_monotonic=deadline_monotonic,
            text=text,
        )
    return command_runner(
        list(command),
        check=True,
        capture_output=capture_output,
        text=text,
        timeout=_remaining_seconds(deadline_monotonic, "subprocess"),
    )


def _clip_id(row: SelectedBalalaikaClip, index: int) -> str:
    return f"memorization-{index:02d}-{row.wav_sha256[:16]}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_wav(path: Path, *, label: str) -> sf.SoundFile:
    try:
        info = sf.info(path)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"{label} WAV is unreadable: {path}") from error
    if (
        info.format != "WAV"
        or info.subtype != "PCM_16"
        or info.samplerate != 24_000
        or info.channels != 1
        or info.frames <= 0
    ):
        raise ValueError(
            f"{label} WAV must be nonempty mono 24000 Hz PCM16: {path}"
        )
    return info


def _validate_wav_set(
    rows: Sequence[SelectedBalalaikaClip], output_dir: Path, *, label: str
) -> list[Path]:
    expected = {
        output_dir / f"{_clip_id(row, index)}.wav"
        for index, row in enumerate(rows)
    }
    actual = set(output_dir.glob("*.wav")) if output_dir.is_dir() else set()
    if actual != expected:
        raise ValueError(f"{label} artifact set must contain exactly four WAVs")
    for path in sorted(actual):
        _validate_wav(path, label=label)
    return sorted(actual)


def generate_four(
    rows: Sequence[SelectedBalalaikaClip], output_dir: Path, model: Any
) -> list[Path]:
    """Generate and validate one deterministic WAV for each memorization row."""
    if len(rows) != 4:
        raise ValueError(f"expected exactly four memorization rows, got {len(rows)}")
    if type(model.sampling_rate) is not int or model.sampling_rate != 24_000:
        raise ValueError("OmniVoice sampling rate must be exactly 24000 Hz")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=False)
    config = OmniVoiceGenerationConfig(
        num_step=32,
        guidance_scale=2.0,
        t_shift=0.1,
        layer_penalty_factor=5.0,
        position_temperature=0.0,
        class_temperature=0.0,
    )
    for index, row in enumerate(rows):
        audios = model.generate(
            text=row.text,
            language="Russian",
            ref_text=row.text,
            ref_audio=str(row.audio_path),
            generation_config=config,
        )
        if not isinstance(audios, Sequence) or len(audios) != 1:
            raise ValueError("OmniVoice must return exactly one audio per utterance")
        output_wav = output_path / f"{_clip_id(row, index)}.wav"
        sf.write(output_wav, audios[0], 24_000, subtype="PCM_16")
        _validate_wav(output_wav, label="generated")
    return _validate_wav_set(rows, output_path, label="generated")


def _release_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _generate_with_model_loader(
    *,
    rows: Sequence[SelectedBalalaikaClip],
    output_dir: Path,
    model_loader: ModelLoader,
    base_model: str | None,
    adapter_checkpoint: Path | None,
) -> list[Path]:
    """Load, generate, and always release the base model or adapter."""
    model = model_loader(
        base_model=base_model,
        adapter_checkpoint=adapter_checkpoint,
    )
    try:
        return generate_four(rows, output_dir, model)
    finally:
        del model
        _release_cuda_cache()


def _generation_child(connection, kwargs: dict[str, Any]) -> None:
    try:
        _generate_with_model_loader(**kwargs)
        connection.send((True, None))
    except BaseException as error:  # noqa: BLE001
        connection.send((False, f"{type(error).__name__}: {error}"))
    finally:
        connection.close()


def _remaining_seconds(deadline: float, operation: str) -> float:
    remaining = deadline - time.monotonic()
    if not math.isfinite(remaining) or remaining <= 0:
        raise TimeoutError(f"phase deadline exceeded before {operation}")
    return remaining


def _stop_child(process: mp.Process) -> None:
    process.terminate()
    process.join(_TERMINATE_JOIN_SECONDS)
    if process.is_alive():
        process.kill()
        process.join(_TERMINATE_JOIN_SECONDS)
    if process.is_alive():
        raise RuntimeError("generation child could not be terminated")


def _run_generation_bounded(
    *,
    rows: Sequence[SelectedBalalaikaClip],
    output_dir: Path,
    model_loader: ModelLoader,
    base_model: str | None,
    adapter_checkpoint: Path | None,
    deadline_monotonic: float,
) -> list[Path]:
    """Generate in a terminable child process under the phase deadline."""
    _remaining_seconds(deadline_monotonic, "generation")
    try:
        pickle.dumps(model_loader)
    except (AttributeError, pickle.PicklingError, TypeError):
        method = "fork" if os.name == "posix" else "spawn"
    else:
        method = "spawn"
    context = mp.get_context(method)
    parent_connection, child_connection = context.Pipe(duplex=False)
    kwargs = {
        "rows": rows,
        "output_dir": output_dir,
        "model_loader": model_loader,
        "base_model": base_model,
        "adapter_checkpoint": adapter_checkpoint,
    }
    process = None
    started = False
    try:
        process = context.Process(
            target=_generation_child,
            args=(child_connection, kwargs),
            daemon=False,
        )
        process.start()
        started = True
        child_connection.close()
        process.join(_remaining_seconds(deadline_monotonic, "generation"))
        if process.is_alive():
            _stop_child(process)
            raise TimeoutError("generation exceeded phase deadline")
        if not parent_connection.poll():
            raise RuntimeError(
                f"generation child exited without a result (exit {process.exitcode})"
            )
        succeeded, message = parent_connection.recv()
        if not succeeded:
            raise RuntimeError(f"generation child failed: {message}")
    finally:
        child_connection.close()
        if started and process is not None and process.is_alive():
            _stop_child(process)
        parent_connection.close()
    _remaining_seconds(deadline_monotonic, "generation completion")
    return _validate_wav_set(rows, output_dir, label="generated")


def read_loss_history(path: str | Path) -> list[dict[str, int | float]]:
    """Read a strict, finite, cadence-consistent evaluation history."""
    history_path = Path(path)
    if history_path.is_dir():
        history_path = history_path / "loss_history.jsonl"
    history = []
    previous_step = 0
    previous_elapsed = -1.0
    with history_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid loss history row {line_number}") from error
            if not isinstance(row, dict) or set(row) != {
                "step",
                "loss",
                "elapsed_seconds",
            }:
                raise ValueError(f"invalid loss history row {line_number}")
            step = row["step"]
            loss = row["loss"]
            elapsed = row["elapsed_seconds"]
            if (
                type(step) is not int
                or step <= previous_step
                or step > MAX_OPTIMIZER_STEPS
                or step % EVAL_STEPS != 0
                or isinstance(loss, bool)
                or not isinstance(loss, (int, float))
                or not math.isfinite(loss)
                or loss < 0
                or isinstance(elapsed, bool)
                or not isinstance(elapsed, (int, float))
                or not math.isfinite(elapsed)
                or elapsed < 0
                or elapsed <= previous_elapsed
            ):
                raise ValueError(f"invalid loss history row {line_number}")
            history.append(
                {"step": step, "loss": float(loss), "elapsed_seconds": float(elapsed)}
            )
            previous_step = step
            previous_elapsed = float(elapsed)
    if not history:
        raise ValueError("loss history is empty")
    return history


def _completed_checkpoints(output_dir: Path) -> dict[int, Path]:
    checkpoints = {}
    if output_dir.is_dir():
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
    outcome: TrainingOutcome | None = None,
) -> Path:
    """Choose an exact current checkpoint, or legacy highest fallback."""
    output_path = Path(output_dir)
    completed = _completed_checkpoints(output_path)
    if not completed:
        raise FileNotFoundError(f"no completed checkpoints in {output_path}")
    patience_result = check_memorization_patience(
        ((int(row["step"]), float(row["loss"])) for row in loss_history),
        threshold=threshold,
        patience=patience,
    )
    if outcome is None:
        desired_step = patience_result.qualifying_step
        if desired_step is None:
            desired_step = max(completed)
    else:
        newer_steps = [step for step in completed if step > outcome.step]
        if newer_steps:
            raise ValueError(
                "checkpoint state is newer than TrainingOutcome: "
                f"{max(newer_steps)} > {outcome.step}"
            )
        desired_step = (
            patience_result.qualifying_step
            if patience_result.passed
            else outcome.step
        )
    checkpoint = output_path / f"checkpoint-{desired_step}"
    if desired_step not in completed:
        raise FileNotFoundError(f"required checkpoint is missing: {checkpoint}")
    return checkpoint


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
        if (
            not isinstance(row.source_relative_path, str)
            or not row.source_relative_path.strip()
        ):
            raise ValueError("source_relative_path must be nonempty")
        if type(row.schema_version) is not int or row.schema_version <= 0:
            raise ValueError("schema_version must be a positive integer")
        if row.role != "memorization":
            raise ValueError("role must be memorization")
        if row.duration_tier not in {
            "preferred_3_to_12s",
            "fallback_over_12s",
        }:
            raise ValueError("duration_tier is invalid")
        if type(row.sample_rate) is not int or row.sample_rate != 24_000:
            raise ValueError("sample_rate must be exactly 24000")
        if type(row.channels) is not int or row.channels != 1:
            raise ValueError("channels must be exactly 1")
        if type(row.seed) is not int:
            raise ValueError("seed must be an integer")
        if not isinstance(row.source_shard, str) or not row.source_shard.strip():
            raise ValueError("source_shard must be nonempty")
        if not isinstance(row.member_name, str) or not row.member_name.strip():
            raise ValueError("member_name must be nonempty")
        source_match = _SOURCE_RELATIVE_PATH.fullmatch(row.source_relative_path)
        if source_match is None:
            raise ValueError("source_relative_path has an invalid format")
        expected_shard = f"train/shard_{source_match.group('shard')}.tar"
        if row.source_shard != expected_shard:
            raise ValueError("source_shard does not match source_relative_path")
        if row.member_name != source_match.group("member"):
            raise ValueError("member_name does not match source_relative_path")
    source_paths = [row.source_relative_path for row in rows]
    audio_paths = [str(Path(row.audio_path).resolve()) for row in rows]
    artifact_ids = [_clip_id(row, index) for index, row in enumerate(rows)]
    source_hashes = [row.source_sha256 for row in rows]
    wav_hashes = [row.wav_sha256 for row in rows]
    if (
        len(set(source_paths)) != 4
        or len(set(audio_paths)) != 4
        or len(set(artifact_ids)) != 4
        or len(set(source_hashes)) != 4
        or len(set(wav_hashes)) != 4
    ):
        raise ValueError("memorization rows must have unique IDs, paths, and hashes")
    for row in rows:
        if not isinstance(row.text, str) or not row.text.strip():
            raise ValueError("memorization text must be nonempty")
        if not _SHA256.fullmatch(row.source_sha256):
            raise ValueError("source_sha256 must be a lowercase SHA-256")
        if not _SHA256.fullmatch(row.wav_sha256):
            raise ValueError("wav_sha256 must be a lowercase SHA-256")
        audio_path = Path(row.audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(row.audio_path)
        info = _validate_wav(audio_path, label="selected input")
        if row.sample_rate != info.samplerate or row.channels != info.channels:
            raise ValueError("selected WAV metadata does not match the audio")
        actual_duration = info.frames / info.samplerate
        if (
            isinstance(row.duration, bool)
            or not isinstance(row.duration, (int, float))
            or not math.isfinite(row.duration)
            or row.duration <= 0
            or abs(float(row.duration) - actual_duration) > 1 / info.samplerate
        ):
            raise ValueError("selected WAV duration metadata is invalid")
        if row.duration_tier == "preferred_3_to_12s" and not (
            3.0 <= actual_duration <= 12.0
        ):
            raise ValueError("duration_tier does not match WAV duration")
        if row.duration_tier == "fallback_over_12s" and actual_duration <= 12.0:
            raise ValueError("duration_tier does not match WAV duration")
        if _sha256_file(audio_path) != row.wav_sha256:
            raise ValueError("selected WAV hash does not match wav_sha256")
    return rows


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _copy_originals(
    rows: Sequence[SelectedBalalaikaClip], output_dir: Path, deadline: float
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=False)
    originals = []
    for index, row in enumerate(rows):
        _remaining_seconds(deadline, "original audio copy")
        destination = output_dir / f"{_clip_id(row, index)}.wav"
        shutil.copy2(row.audio_path, destination)
        if _sha256_file(destination) != row.wav_sha256:
            raise ValueError(f"copied original hash mismatch: {destination}")
        originals.append(destination)
    return _validate_wav_set(rows, output_dir, label="original")


def _validate_original_hashes(
    rows: Sequence[SelectedBalalaikaClip], output_dir: Path
) -> None:
    for index, row in enumerate(rows):
        path = output_dir / f"{_clip_id(row, index)}.wav"
        if _sha256_file(path) != row.wav_sha256:
            raise ValueError(f"copied original hash mismatch: {path}")


def _write_training_manifest(
    rows: Sequence[SelectedBalalaikaClip], path: Path
) -> None:
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
                    allow_nan=False,
                )
                + "\n"
            )
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(lines) != 4 or any(not row["text"].strip() for row in lines):
        raise ValueError("training manifest must contain four nonempty rows")


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


def _load_valid_training_config(source: Path) -> TrainingConfig:
    config = TrainingConfig.from_json(str(source))
    required = {
        "lora_enabled": (config.lora_enabled, True),
        "lora_rank": (config.lora_rank, 64),
        "lora_target_modules": (
            list(config.lora_target_modules),
            list(DEFAULT_LORA_TARGET_MODULES),
        ),
        "init_from_checkpoint": (config.init_from_checkpoint, BASE_MODEL_ID),
        "resume_from_checkpoint": (config.resume_from_checkpoint, None),
    }
    for name, (actual, expected) in required.items():
        if actual != expected:
            raise ValueError(f"{name} must be {expected!r}, got {actual!r}")
    return config


def _write_runtime_train_config(
    config: TrainingConfig,
    destination: Path,
    *,
    output_dir: Path,
    data_config: Path,
    history_path: Path,
    remaining_wall_clock_seconds: float,
) -> None:
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
    config.lora_enabled = True
    config.lora_rank = 64
    config.lora_target_modules = list(DEFAULT_LORA_TARGET_MODULES)
    config.init_from_checkpoint = BASE_MODEL_ID
    config.resume_from_checkpoint = None
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


def _validate_outcome_payload(payload: object) -> TrainingOutcome:
    if not isinstance(payload, dict) or set(payload) != {
        "step",
        "stop_reason",
        "last_eval_loss",
        "target_reached",
    }:
        raise ValueError("invalid TrainingOutcome schema")
    step = payload["step"]
    stop_reason = payload["stop_reason"]
    loss = payload["last_eval_loss"]
    target_reached = payload["target_reached"]
    if (
        type(step) is not int
        or step <= 0
        or step > MAX_OPTIMIZER_STEPS
        or step % EVAL_STEPS != 0
        or stop_reason not in _KNOWN_STOP_REASONS
        or isinstance(loss, bool)
        or not isinstance(loss, (int, float))
        or not math.isfinite(loss)
        or loss < 0
        or type(target_reached) is not bool
        or target_reached != (stop_reason == "eval_loss_target")
    ):
        raise ValueError("invalid TrainingOutcome values")
    return TrainingOutcome(step, stop_reason, float(loss), target_reached)


def _parse_training_outcome(stdout: str) -> TrainingOutcome:
    for line in reversed(stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and set(payload) & {
            "step",
            "stop_reason",
            "last_eval_loss",
            "target_reached",
        }:
            return _validate_outcome_payload(payload)
    raise ValueError("training command did not emit a valid TrainingOutcome")


def _validate_run_consistency(
    outcome: TrainingOutcome,
    history: Sequence[Mapping[str, int | float]],
):
    actual_steps = [int(row["step"]) for row in history]
    expected_steps = list(range(EVAL_STEPS, outcome.step + 1, EVAL_STEPS))
    if actual_steps != expected_steps:
        raise ValueError(
            "loss history must evaluate exactly every 25 steps from step 25 "
            f"through {outcome.step}"
        )
    if (
        outcome.stop_reason in {"completed", "stop_after_step"}
        and outcome.step != MAX_OPTIMIZER_STEPS
    ):
        raise ValueError(
            f"{outcome.stop_reason} is valid only at step {MAX_OPTIMIZER_STEPS}"
        )
    final = history[-1]
    if outcome.step != final["step"] or not math.isclose(
        float(outcome.last_eval_loss),
        float(final["loss"]),
        rel_tol=0,
        abs_tol=1e-15,
    ):
        raise ValueError("TrainingOutcome is inconsistent with loss history")
    patience_result = check_memorization_patience(
        ((int(row["step"]), float(row["loss"])) for row in history),
        threshold=REQUIRED_LOSS,
        patience=REQUIRED_PATIENCE,
    )
    if patience_result.passed != outcome.target_reached:
        raise ValueError("TrainingOutcome target status is inconsistent with history")
    if patience_result.passed and patience_result.qualifying_step != outcome.step:
        raise ValueError("qualifying step is inconsistent with TrainingOutcome")
    return patience_result


def _default_model_loader(*, base_model=None, adapter_checkpoint=None):
    if (base_model is None) == (adapter_checkpoint is None):
        raise ValueError("specify exactly one model source")
    if base_model is not None:
        return OmniVoice.from_pretrained(base_model, device_map="cuda:0")
    return OmniVoice.from_lora_pretrained(adapter_checkpoint, device_map="cuda:0")


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _publish_artifact_bundle(stage: Path, output_dir: Path) -> None:
    with tempfile.TemporaryDirectory(
        prefix=".artifacts-backup-", dir=output_dir
    ) as backup_name:
        backup = Path(backup_name)
        backed_up = []
        published = []
        try:
            for name in _PUBLISHED_TARGETS:
                target = output_dir / name
                if target.exists():
                    os.replace(target, backup / name)
                    backed_up.append(name)
            for name in _PUBLISHED_TARGETS:
                os.replace(stage / name, output_dir / name)
                published.append(name)
        except BaseException:
            for name in reversed(published):
                target = output_dir / name
                if target.exists():
                    os.replace(target, stage / name)
            for name in reversed(backed_up):
                target = output_dir / name
                if target.exists():
                    _remove_path(target)
                os.replace(backup / name, target)
            raise


def _validate_artifact_bundle(
    stage: Path, rows: Sequence[SelectedBalalaikaClip]
) -> None:
    _validate_wav_set(rows, stage / "original", label="original")
    _validate_original_hashes(rows, stage / "original")
    _validate_wav_set(rows, stage / "generated/initial", label="generated initial")
    _validate_wav_set(rows, stage / "generated/final", label="generated final")
    if len((stage / "four.jsonl").read_text(encoding="utf-8").splitlines()) != 4:
        raise ValueError("training manifest must contain exactly four rows")
    json.loads((stage / "result.json").read_text(encoding="utf-8"))


def _validate_seconds(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def run_memorization(
    *,
    selected_manifest: str | Path,
    output_dir: str | Path,
    train_config: str | Path,
    data_config: str | Path,
    max_wall_clock_seconds: float = 1200,
    experiment_wall_clock_seconds: float = 3600,
    command_runner: CommandRunner | None = None,
    model_loader: ModelLoader = _default_model_loader,
    generation_runner: GenerationRunner = _run_generation_bounded,
) -> MemorizationRunResult:
    """Run bounded training and transactionally publish listening artifacts."""
    phase_seconds = _validate_seconds(
        max_wall_clock_seconds, "max_wall_clock_seconds"
    )
    experiment_seconds = _validate_seconds(
        experiment_wall_clock_seconds, "experiment_wall_clock_seconds"
    )
    started = time.monotonic()
    phase_deadline = started + phase_seconds
    experiment_deadline = started + experiment_seconds
    selected_path = Path(selected_manifest).resolve()
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    if _completed_checkpoints(output_path):
        raise ValueError("memorization requires a fresh output directory")
    rows = _read_selected_rows(selected_path)
    training_settings = _load_valid_training_config(Path(train_config))

    with tempfile.TemporaryDirectory(
        prefix=".artifacts-stage-", dir=output_path
    ) as stage_name:
        stage = Path(stage_name)
        _copy_originals(rows, stage / "original", phase_deadline)
        training_manifest = stage / "four.jsonl"
        _write_training_manifest(rows, training_manifest)
        generation_runner(
            rows=rows,
            output_dir=stage / "generated/initial",
            model_loader=model_loader,
            base_model=BASE_MODEL_ID,
            adapter_checkpoint=None,
            deadline_monotonic=phase_deadline,
        )
        _remaining_seconds(phase_deadline, "tokenization")
        token_dir = output_path / "tokens"
        _invoke_command(
            _tokenizer_command(training_manifest, token_dir),
            deadline_monotonic=phase_deadline,
            command_runner=command_runner,
        )

        runtime_data_config = output_path / "data_config.json"
        runtime_train_config = output_path / "train_config.json"
        history_path = output_path / "loss_history.jsonl"
        history_path.unlink(missing_ok=True)
        _write_runtime_data_config(Path(data_config), runtime_data_config, token_dir)
        remaining = _remaining_seconds(phase_deadline, "training")
        _write_runtime_train_config(
            training_settings,
            runtime_train_config,
            output_dir=output_path,
            data_config=runtime_data_config,
            history_path=history_path,
            remaining_wall_clock_seconds=remaining,
        )
        completed = _invoke_command(
            _training_command(runtime_train_config, runtime_data_config, output_path),
            deadline_monotonic=phase_deadline,
            command_runner=command_runner,
            capture_output=True,
            text=True,
        )
        outcome = _parse_training_outcome(completed.stdout)
        history = read_loss_history(history_path)
        patience_result = _validate_run_consistency(outcome, history)
        checkpoint = choose_generation_checkpoint(
            output_path, history, outcome=outcome
        )
        generation_runner(
            rows=rows,
            output_dir=stage / "generated/final",
            model_loader=model_loader,
            base_model=None,
            adapter_checkpoint=checkpoint,
            deadline_monotonic=phase_deadline,
        )
        _remaining_seconds(phase_deadline, "artifact publication")

        minimum_loss = patience_result.minimum_loss
        required_target_reached = patience_result.passed
        result = MemorizationRunResult(
            required_target_reached=required_target_reached,
            stretch_target_reached=(
                minimum_loss is not None and minimum_loss <= STRETCH_LOSS
            ),
            qualifying_step=patience_result.qualifying_step,
            minimum_loss=minimum_loss,
            selected_checkpoint=str(checkpoint),
            training_outcome=outcome,
            experiment_deadline_monotonic=experiment_deadline,
            miss_reason=(
                None
                if required_target_reached
                else f"{outcome.stop_reason}_before_required_target"
            ),
        )
        _write_json(stage / "result.json", asdict(result))
        _validate_artifact_bundle(stage, rows)
        _publish_artifact_bundle(stage, output_path)
        return result
