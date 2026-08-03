"""Run checkpoint-isolated LoRA training with hard-number validation."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

from omnivoice.validation.controller import ValidationController


def _deadline_from_state(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("deadline state must contain a JSON object")
    deadline = payload.get("experiment_deadline_monotonic")
    if (
        isinstance(deadline, bool)
        or not isinstance(deadline, (int, float))
        or not math.isfinite(deadline)
    ):
        raise ValueError(
            "deadline state must contain a finite experiment_deadline_monotonic"
        )
    return float(deadline)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--validation-config", type=Path, required=True)
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--deadline-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-output-root", type=Path, required=True)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    controller = ValidationController(
        train_config=args.train_config,
        data_config=args.data_config,
        validation_config=args.validation_config,
        selected_manifest=args.selected_manifest,
        output_dir=args.output_dir,
        validation_output_root=args.validation_output_root,
        deadline_monotonic=_deadline_from_state(args.deadline_state),
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    state = controller.run()
    print(json.dumps(asdict(state), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
