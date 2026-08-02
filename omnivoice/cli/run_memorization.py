"""Run bounded four-utterance Balalaika memorization."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from omnivoice.validation.memorization import run_memorization


def _positive_seconds(value: str) -> float:
    seconds = float(value)
    if seconds <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return seconds


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-config", required=True, type=Path)
    parser.add_argument("--data-config", required=True, type=Path)
    parser.add_argument(
        "--max-wall-clock-seconds", type=_positive_seconds, default=1200
    )
    parser.add_argument(
        "--experiment-wall-clock-seconds",
        type=_positive_seconds,
        default=3600,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_memorization(
        selected_manifest=args.selected_manifest,
        output_dir=args.output_dir,
        train_config=args.train_config,
        data_config=args.data_config,
        max_wall_clock_seconds=args.max_wall_clock_seconds,
        experiment_wall_clock_seconds=args.experiment_wall_clock_seconds,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
