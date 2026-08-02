"""Run checkpoint-isolated hard-number validation stages."""

from __future__ import annotations

import argparse
import json
import math
import signal
import threading
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from types import TracebackType
from typing import Any

from omnivoice.validation.artifacts import ValidationPaths
from omnivoice.validation.synthesis import (
    initialize_distributed,
    load_assignment_manifest,
    load_validation_tts,
    release_validation_tts,
    resolve_distributed_context,
    synchronize_distributed,
    synthesize_rank,
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
    distributed_initializer: Callable[[Any], None] = initialize_distributed,
    model_loader: Callable[..., Any] = load_validation_tts,
    synthesizer: Callable[..., Any] = synthesize_rank,
    synchronizer: Callable[[], bool] = synchronize_distributed,
    cuda_releaser: Callable[[], None] = release_validation_tts,
) -> int:
    context = context_resolver()
    paths = ValidationPaths(args.output_root, args.run_id, args.step)
    model = None
    synchronized = False
    summary = None
    try:
        with _SigtermFlag() as stop_requested:
            assignments = assignment_loader(args.assignments)
            model = model_loader(
                model_name=args.model,
                adapter_checkpoint=args.adapter_checkpoint,
                context=context,
            )
            distributed_initializer(context)
            checkpoint = (
                args.model if args.model is not None else str(args.adapter_checkpoint)
            )
            summary = synthesizer(
                assignments=assignments,
                model=model,
                output_dir=paths.step_dir,
                rank=context.rank,
                world_size=context.world_size,
                checkpoint=checkpoint,
                deadline_monotonic=args.deadline_monotonic,
                stop_requested=stop_requested,
            )
    finally:
        try:
            synchronized = synchronizer()
        finally:
            model = None
            cuda_releaser()

    payload = asdict(summary)
    payload["synchronized"] = synchronized
    payload["complete"] = bool(summary.complete and synchronized)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["complete"] else INCOMPLETE_EXIT_CODE


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "synth":
        return _run_synth(args)
    raise AssertionError(f"unsupported validation stage {args.stage!r}")


if __name__ == "__main__":
    raise SystemExit(main())
