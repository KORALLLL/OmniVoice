"""Run checkpoint-isolated hard-number validation stages."""

from __future__ import annotations

import argparse
import builtins
import gc
import json
import math
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from types import TracebackType
from typing import Any

import soundfile as sf

from omnivoice.validation.artifacts import (
    ValidationPaths,
    merge_rank_ledgers,
    require_exact_coverage,
    valid_completed_ids,
)
from omnivoice.validation.asr import (
    load_gigaam,
    persist_rank_failure,
    transcribe_rank,
)
from omnivoice.validation.hard_numbers import HARD_NUMBER_COUNT
from omnivoice.validation.synthesis import (
    LifecycleError,
    SynthesisSummary,
    initialize_distributed,
    load_assignment_manifest,
    load_validation_tts,
    read_synthesis_summary,
    release_validation_tts,
    resolve_distributed_context,
    resolve_model_source,
    run_bounded,
    synchronize_distributed,
    synthesize_rank,
    write_synthesis_summary,
)

INCOMPLETE_EXIT_CODE = 2
_BASE_EXCEPTION_GROUP = builtins.__dict__.get("BaseExceptionGroup", ())


def _output_json(payload: str) -> None:
    print(payload, flush=True)


def _clear_completed_exception_graph_frames(error: BaseException) -> None:
    """Clear completed traceback frames across a cycle-safe exception graph."""
    active_frames: set[int] = set()
    for frame in sys._current_frames().values():
        while frame is not None:
            active_frames.add(id(frame))
            frame = frame.f_back

    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        traceback = current.__traceback__
        while traceback is not None:
            frame = traceback.tb_frame
            if id(frame) not in active_frames:
                frame.clear()
            traceback = traceback.tb_next
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        if isinstance(current, _BASE_EXCEPTION_GROUP):
            pending.extend(current.exceptions)


def _valid_lifecycle_detail(detail: LifecycleError | None) -> bool:
    return detail is None or (
        isinstance(detail, LifecycleError)
        and isinstance(detail.stage, str)
        and bool(detail.stage.strip())
        and isinstance(detail.type, str)
        and bool(detail.type.strip())
        and isinstance(detail.message, str)
    )


def _summary_matches_current_context(
    summary: Any,
    *,
    run_id: str,
    step: int,
    rank: int,
    source_identity: Any,
) -> bool:
    if (
        not isinstance(summary, SynthesisSummary)
        or summary.run_id != run_id
        or summary.step != step
        or summary.rank != rank
        or summary.source_identity != source_identity
        or type(summary.complete) is not bool
        or not (
            summary.stop_reason is None
            or (
                isinstance(summary.stop_reason, str)
                and bool(summary.stop_reason.strip())
            )
        )
        or not all(
            _valid_lifecycle_detail(detail)
            for detail in (
                summary.error,
                summary.primary_error,
                summary.cleanup_error,
            )
        )
    ):
        return False
    expected = summary.expected
    progress_counts = (
        summary.completed,
        summary.generated,
        summary.skipped,
        summary.failed,
    )
    if expected is None:
        if any(value is not None for value in progress_counts):
            return False
    elif type(expected) is not int or expected < 0:
        return False
    elif not all(value is None for value in progress_counts):
        if any(
            type(value) is not int or not 0 <= value <= expected
            for value in progress_counts
        ):
            return False
        completed, generated, skipped, failed = progress_counts
        if generated + skipped != completed or completed + failed > expected:
            return False
    return not summary.complete or (
        expected is not None
        and summary.completed == expected
        and summary.failed == 0
        and summary.error is None
        and summary.primary_error is None
        and summary.cleanup_error is None
    )


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("must be finite")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    synth = stages.add_parser("synth", help="synthesize one eight-rank TTS stage")
    synth.add_argument("--assignments", type=Path, required=True)
    synth.add_argument("--output-root", type=Path, required=True)
    synth.add_argument("--run-id", required=True)
    synth.add_argument("--step", type=int, required=True)
    source = synth.add_mutually_exclusive_group(required=True)
    source.add_argument("--model")
    source.add_argument("--adapter-checkpoint", type=Path)
    synth.add_argument("--deadline-monotonic", type=_finite_float)
    asr = stages.add_parser("asr", help="transcribe one eight-rank GigaAM stage")
    asr.add_argument("--assignments", type=Path, required=True)
    asr.add_argument("--output-root", type=Path, required=True)
    asr.add_argument("--run-id", required=True)
    asr.add_argument("--step", type=int, required=True)
    asr.add_argument("--deadline-monotonic", type=_finite_float)
    return parser


class _SigtermFlag:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.previous: Any = None
        self.installed = False

    def __enter__(self) -> Callable[[], bool]:
        try:
            self.previous = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, self._handle)
            self.installed = True
        except ValueError:
            self.installed = False
        return self.event.is_set

    def _handle(self, signum: int, frame: Any) -> None:
        del signum, frame
        self.event.set()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        if self.installed:
            signal.signal(signal.SIGTERM, self.previous)


class _FinalSigtermBoundary:
    """Defer SIGTERM while one authoritative result is committed."""

    def __init__(self) -> None:
        self.previous_mask: set[signal.Signals] | None = None

    def __enter__(self) -> Callable[[], bool]:
        self.previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
        return lambda: signal.SIGTERM in signal.sigpending()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        if self.previous_mask is not None:
            signal.pthread_sigmask(signal.SIG_SETMASK, self.previous_mask)


def _run_asr(
    args: argparse.Namespace,
    *,
    assignment_loader: Callable[[Path], Any] = load_assignment_manifest,
    context_resolver: Callable[[], Any] = resolve_distributed_context,
    synthesis_merger: Callable[[Any], list[dict[str, Any]]] = merge_rank_ledgers,
    model_loader: Callable[[int], Any] = load_gigaam,
    transcriber: Callable[..., Any] = transcribe_rank,
    failure_persister: Callable[..., Any] = persist_rank_failure,
    synchronizer: Callable[[], bool] = synchronize_distributed,
    output: Callable[[str], None] = _output_json,
    serializer: Callable[..., str] = json.dumps,
) -> int:
    """Run one rank of ASR after exact hash-validated TTS coverage exists."""
    context = context_resolver()
    paths = ValidationPaths(args.output_root, args.run_id, args.step)
    with _SigtermFlag() as stop_requested:
        try:
            assignments = assignment_loader(args.assignments)
            expected_ids = [assignment.id for assignment in assignments]
            if len(expected_ids) != HARD_NUMBER_COUNT:
                raise ValueError(
                    f"ASR requires exactly {HARD_NUMBER_COUNT} assignments; "
                    f"got {len(expected_ids)}"
                )
            synthesis_records = synthesis_merger(
                [paths.rank_manifest(rank) for rank in range(context.world_size)]
            )
            require_exact_coverage(
                expected_ids,
                [record.get("id") for record in synthesis_records],
            )
            hash_valid_ids = valid_completed_ids(
                synthesis_records, required_files=("wav",)
            )
            require_exact_coverage(
                expected_ids,
                [
                    record["id"]
                    for record in synthesis_records
                    if record.get("id") in hash_valid_ids
                    and _valid_synthesis_wav(record)
                ],
            )
        except (OSError, TypeError, ValueError) as error:
            output(
                serializer(
                    {"complete": False, "stage": "synthesis_coverage"},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            del error
            return INCOMPLETE_EXIT_CODE

        model = None
        try:
            if stop_requested():
                output(
                    serializer(
                        {"complete": False, "stage": "signal"},
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
                return INCOMPLETE_EXIT_CODE
            model = model_loader(context.local_rank)
            summary = transcriber(
                synthesis_records=synthesis_records,
                recognizer=model,
                output_dir=paths,
                rank=context.rank,
                world_size=context.world_size,
                deadline_monotonic=args.deadline_monotonic,
                stop_requested=stop_requested,
            )
        except Exception as error:  # noqa: BLE001 - durable lifecycle failure contract
            summary = failure_persister(
                synthesis_records=synthesis_records,
                output_dir=paths,
                rank=context.rank,
                world_size=context.world_size,
                error=error,
            )
        finally:
            model = None
            gc.collect()

        synchronized = synchronizer()
        hypotheses_complete = False
        if synchronized:
            try:
                hypothesis_records = merge_rank_ledgers(
                    [paths.rank_hypotheses(rank) for rank in range(context.world_size)]
                )
                require_exact_coverage(
                    expected_ids,
                    [record.get("id") for record in hypothesis_records],
                )
                require_exact_coverage(
                    expected_ids,
                    valid_completed_ids(hypothesis_records),
                )
            except (OSError, TypeError, ValueError):
                hypotheses_complete = False
            else:
                hypotheses_complete = True
        payload = asdict(summary)
        payload["synchronized"] = synchronized
        payload["complete"] = bool(
            summary.complete and synchronized and hypotheses_complete
        )
        output(serializer(payload, ensure_ascii=False, sort_keys=True))
        return 0 if payload["complete"] else INCOMPLETE_EXIT_CODE


def _valid_synthesis_wav(record: dict[str, Any]) -> bool:
    """Reject hashed artifacts that are not TTS's mono 24 kHz PCM WAVs."""
    raw_path = record.get("wav")
    if not isinstance(raw_path, str) or not raw_path:
        return False
    try:
        info = sf.info(raw_path)
    except (OSError, RuntimeError):
        return False
    return (
        info.format == "WAV"
        and info.subtype == "PCM_16"
        and info.samplerate == 24_000
        and info.channels == 1
        and info.frames > 0
    )


def _run_synth(
    args: argparse.Namespace,
    *,
    assignment_loader: Callable[[Path], Any] = load_assignment_manifest,
    context_resolver: Callable[[], Any] = resolve_distributed_context,
    source_resolver: Callable[..., Any] = resolve_model_source,
    distributed_initializer: Callable[[Any], None] = initialize_distributed,
    model_loader: Callable[..., Any] = load_validation_tts,
    synthesizer: Callable[..., Any] = synthesize_rank,
    synchronizer: Callable[[], bool] = synchronize_distributed,
    cuda_releaser: Callable[[], None] = release_validation_tts,
    summary_writer: Callable[..., Path] = write_synthesis_summary,
    summary_reader: Callable[..., Any] = read_synthesis_summary,
    serializer: Callable[..., str] = json.dumps,
    output: Callable[[str], None] = _output_json,
    cleanup_timeout_seconds: float = 30.0,
    monotonic: Callable[[], float] = time.monotonic,
    hard_exit: Callable[[int], Any] = os._exit,
) -> int:
    context = context_resolver()
    paths = ValidationPaths(args.output_root, args.run_id, args.step)
    model_holder: list[Any] = [None]
    synchronized = False
    summary = None
    assignments = None
    source_identity = None
    with _SigtermFlag() as stop_requested:
        primary_error: BaseException | None = None
        primary_traceback: TracebackType | None = None
        primary_stage = "assignments"
        cleanup_errors: list[tuple[str, BaseException]] = []
        try:
            assignments = assignment_loader(args.assignments)
            primary_stage = "source"
            source_identity = source_resolver(
                model_name=args.model,
                adapter_checkpoint=args.adapter_checkpoint,
            )
            primary_stage = "model_load"
            model_holder[0] = model_loader(
                source_identity=source_identity,
                context=context,
            )
            primary_stage = "distributed_initialization"
            distributed_initializer(context)
            primary_stage = "synthesis"
            summary = synthesizer(
                assignments=assignments,
                model=model_holder[0],
                output_dir=paths,
                rank=context.rank,
                world_size=context.world_size,
                source_identity=source_identity,
                deadline_monotonic=args.deadline_monotonic,
                stop_requested=stop_requested,
            )
            summary = replace(summary, run_id=args.run_id, step=args.step)
        except BaseException as error:  # noqa: BLE001 - cleanup must still run
            primary_error = error
            primary_traceback = error.__traceback__

        def read_existing_summary() -> None:
            nonlocal summary
            if summary is None:
                try:
                    candidate = summary_reader(paths.step_dir, context.rank)
                except (json.JSONDecodeError, TypeError, ValueError):
                    return
                if _summary_matches_current_context(
                    candidate,
                    run_id=args.run_id,
                    step=args.step,
                    rank=context.rank,
                    source_identity=source_identity,
                ):
                    summary = candidate

        def lifecycle_detail(
            stage: str, lifecycle_error: BaseException
        ) -> LifecycleError:
            return LifecycleError(
                stage=stage,
                type=type(lifecycle_error).__name__,
                message=str(lifecycle_error),
            )

        if primary_error is not None and source_identity is not None:
            primary_detail = lifecycle_detail(primary_stage, primary_error)
            try:
                read_existing_summary()
            except BaseException as error:  # noqa: BLE001 - preserve primary failure
                cleanup_errors.append(("summary", error))
            if summary is None:
                expected = None
                try:
                    assignment_count = len(assignments)
                except (TypeError, AttributeError):
                    pass
                else:
                    expected = len(
                        range(context.rank, assignment_count, context.world_size)
                    )
                summary = SynthesisSummary(
                    rank=context.rank,
                    expected=expected,
                    completed=None,
                    generated=None,
                    skipped=None,
                    failed=None,
                    complete=False,
                    stop_reason="error",
                    source_identity=source_identity,
                    error=primary_detail,
                    run_id=args.run_id,
                    step=args.step,
                    primary_error=primary_detail,
                )
            else:
                summary = replace(
                    summary,
                    complete=False,
                    stop_reason="error",
                    error=primary_detail,
                    run_id=args.run_id,
                    step=args.step,
                    primary_error=primary_detail,
                )
            try:
                summary_writer(paths.step_dir, summary)
            except BaseException as error:  # noqa: BLE001 - preserve primary failure
                cleanup_errors.append(("summary", error))

        cleanup_deadline = monotonic() + cleanup_timeout_seconds

        def correct_summary(stage: str, lifecycle_error: BaseException) -> None:
            nonlocal summary
            read_existing_summary()
            if summary is None:
                return
            detail = lifecycle_detail(stage, lifecycle_error)
            summary = replace(
                summary,
                complete=False,
                stop_reason=stage,
                error=detail,
                run_id=args.run_id,
                step=args.step,
                cleanup_error=detail,
            )
            summary_writer(paths.step_dir, summary)

        def terminate_abandoned_worker(stage: str, error: TimeoutError) -> None:
            try:
                correct_summary(stage, error)
                if summary is not None:
                    payload = asdict(summary)
                    payload["synchronized"] = synchronized
                    payload["complete"] = False
                    output(serializer(payload, ensure_ascii=False, sort_keys=True))
            finally:
                hard_exit(INCOMPLETE_EXIT_CODE)
            raise AssertionError("hard_exit returned unexpectedly")

        try:
            synchronized = run_bounded(
                synchronizer,
                deadline_monotonic=cleanup_deadline,
                description="synchronization",
                monotonic=monotonic,
            )
        except TimeoutError as error:
            terminate_abandoned_worker("synchronization", error)
        except BaseException as error:  # noqa: BLE001 - preserve primary failure
            cleanup_errors.append(("synchronization", error))

        def release_owned_model() -> None:
            if primary_error is not None:
                _clear_completed_exception_graph_frames(primary_error)
            model_holder.clear()
            cuda_releaser()

        try:
            run_bounded(
                release_owned_model,
                deadline_monotonic=cleanup_deadline,
                description="cleanup",
                monotonic=monotonic,
            )
        except TimeoutError as error:
            terminate_abandoned_worker("cleanup", error)
        except BaseException as error:  # noqa: BLE001 - preserve primary failure
            cleanup_errors.append(("cleanup", error))

        if summary is None and (primary_error is not None or cleanup_errors):
            try:
                read_existing_summary()
            except BaseException as error:  # noqa: BLE001 - preserve primary failure
                cleanup_errors.append(("summary", error))

        lifecycle_failure = (
            cleanup_errors[0]
            if cleanup_errors
            else (("synthesis", primary_error) if primary_error is not None else None)
        )
        if summary is not None and lifecycle_failure is not None:
            stage, lifecycle_error = lifecycle_failure
            try:
                if cleanup_errors:
                    correct_summary(stage, lifecycle_error)
            except BaseException as error:  # noqa: BLE001 - preserve primary failure
                cleanup_errors.append(("summary", error))

        if primary_error is not None:
            for stage, cleanup_error in cleanup_errors:
                add_note = getattr(primary_error, "add_note", None)
                if add_note is not None:
                    add_note(
                        f"{stage} failure: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
            raise primary_error.with_traceback(primary_traceback)
        if cleanup_errors:
            _, cleanup_error = cleanup_errors[0]
            for stage, secondary in cleanup_errors[1:]:
                add_note = getattr(cleanup_error, "add_note", None)
                if add_note is not None:
                    add_note(
                        f"additional {stage} failure: "
                        f"{type(secondary).__name__}: {secondary}"
                    )
            raise cleanup_error
        if summary is None:
            raise RuntimeError("synthesis returned no summary")

        with _FinalSigtermBoundary() as pending_sigterm:
            interrupted = stop_requested() or pending_sigterm()
            if interrupted or not synchronized:
                summary = replace(
                    summary,
                    complete=False,
                    stop_reason=(
                        "signal"
                        if interrupted
                        else summary.stop_reason or "synchronization"
                    ),
                    run_id=args.run_id,
                    step=args.step,
                )
                summary_writer(paths.step_dir, summary)

            payload = asdict(summary)
            payload["synchronized"] = synchronized
            payload["complete"] = bool(summary.complete and synchronized)
            serialized = serializer(payload, ensure_ascii=False, sort_keys=True)
            output(serialized)
            code = 0 if payload["complete"] else INCOMPLETE_EXIT_CODE
        return code


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "synth":
        return _run_synth(args)
    if args.stage == "asr":
        return _run_asr(args)
    raise AssertionError(f"unsupported validation stage {args.stage!r}")


if __name__ == "__main__":
    raise SystemExit(main())
