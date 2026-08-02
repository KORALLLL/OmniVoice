"""Run checkpoint-isolated hard-number validation stages."""

from __future__ import annotations

import argparse
import json
import math
import signal
import threading
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from types import TracebackType
from typing import Any

from omnivoice.validation.artifacts import ValidationPaths
from omnivoice.validation.synthesis import (
    LifecycleError,
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
    output: Callable[[str], None] = print,
    cleanup_timeout_seconds: float = 30.0,
) -> int:
    context = context_resolver()
    paths = ValidationPaths(args.output_root, args.run_id, args.step)
    model = None
    synchronized = False
    summary = None
    with _SigtermFlag() as stop_requested:
        primary_error: BaseException | None = None
        primary_traceback: TracebackType | None = None
        cleanup_errors: list[tuple[str, BaseException]] = []
        try:
            assignments = assignment_loader(args.assignments)
            source_identity = source_resolver(
                model_name=args.model,
                adapter_checkpoint=args.adapter_checkpoint,
            )
            model = model_loader(
                source_identity=source_identity,
                context=context,
            )
            distributed_initializer(context)
            summary = synthesizer(
                assignments=assignments,
                model=model,
                output_dir=paths.step_dir,
                rank=context.rank,
                world_size=context.world_size,
                source_identity=source_identity,
                deadline_monotonic=args.deadline_monotonic,
                stop_requested=stop_requested,
            )
        except BaseException as error:  # noqa: BLE001 - cleanup must still run
            primary_error = error
            primary_traceback = error.__traceback__

        try:
            synchronized = run_bounded(
                synchronizer,
                timeout_seconds=cleanup_timeout_seconds,
                description="synchronization",
            )
        except BaseException as error:  # noqa: BLE001 - preserve primary failure
            cleanup_errors.append(("synchronization", error))
        try:
            model = None
            run_bounded(
                cuda_releaser,
                timeout_seconds=cleanup_timeout_seconds,
                description="cleanup",
            )
        except BaseException as error:  # noqa: BLE001 - preserve primary failure
            cleanup_errors.append(("cleanup", error))

        if summary is None and (primary_error is not None or cleanup_errors):
            try:
                summary = summary_reader(paths.step_dir, context.rank)
            except BaseException as error:  # noqa: BLE001 - preserve primary failure
                cleanup_errors.append(("summary", error))

        lifecycle_failure = (
            cleanup_errors[0]
            if cleanup_errors
            else (("synthesis", primary_error) if primary_error is not None else None)
        )
        if summary is not None and lifecycle_failure is not None:
            stage, lifecycle_error = lifecycle_failure
            summary = replace(
                summary,
                complete=False,
                stop_reason=stage,
                error=LifecycleError(
                    stage=stage,
                    type=type(lifecycle_error).__name__,
                    message=str(lifecycle_error),
                ),
            )
            try:
                summary_writer(paths.step_dir, summary)
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

        interrupted = stop_requested()
        if interrupted or not synchronized:
            summary = replace(
                summary,
                complete=False,
                stop_reason=(
                    "signal"
                    if interrupted
                    else summary.stop_reason or "synchronization"
                ),
            )
            summary_writer(paths.step_dir, summary)

        def incomplete_after_signal() -> None:
            nonlocal summary
            if stop_requested() and summary.complete:
                summary = replace(summary, complete=False, stop_reason="signal")
                summary_writer(paths.step_dir, summary)

        def payload_text() -> tuple[dict[str, Any], str]:
            payload = asdict(summary)
            payload["synchronized"] = synchronized
            payload["complete"] = bool(summary.complete and synchronized)
            return payload, serializer(payload, ensure_ascii=False, sort_keys=True)

        incomplete_after_signal()
        payload, serialized = payload_text()
        incomplete_after_signal()
        if payload["complete"] != bool(summary.complete and synchronized):
            payload, serialized = payload_text()
        output(serialized)
        incomplete_after_signal()
        if payload["complete"] != bool(summary.complete and synchronized):
            payload, serialized = payload_text()
            output(serialized)
        incomplete_after_signal()
        if payload["complete"] != bool(summary.complete and synchronized):
            payload, serialized = payload_text()
            output(serialized)
        return 0 if payload["complete"] else INCOMPLETE_EXIT_CODE


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "synth":
        return _run_synth(args)
    raise AssertionError(f"unsupported validation stage {args.stage!r}")


if __name__ == "__main__":
    raise SystemExit(main())
