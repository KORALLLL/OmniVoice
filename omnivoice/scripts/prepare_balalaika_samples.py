#!/usr/bin/env python3
"""Prepare deterministic Balalaika memorization and validation references."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from omnivoice.validation.balalaika import select_and_convert


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("/workspace/balalaika_proprietary_v2")
    )
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=Path(
            "/workspace/balalaika_proprietary_v2/combined_sidecars/"
            "rover-punctuation-stress-v1/rover-punctuation-stress.jsonl"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--memorization-count", type=int, default=4)
    parser.add_argument("--validation-voice-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    selected = select_and_convert(
        root=args.root,
        sidecar_path=args.sidecar,
        output_dir=args.output_dir,
        memorization_count=args.memorization_count,
        validation_voice_count=args.validation_voice_count,
        seed=args.seed,
    )
    role_counts = Counter(row.role for row in selected)
    duration_counts = Counter(row.duration_tier for row in selected)
    print(
        json.dumps(
            {
                "manifest": str(args.output_dir / "selected.jsonl"),
                "counts": {
                    "memorization": role_counts["memorization"],
                    "validation_voice": role_counts["validation_voice"],
                },
                "duration_tiers": dict(duration_counts),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
