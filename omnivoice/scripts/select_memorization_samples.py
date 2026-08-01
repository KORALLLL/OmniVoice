#!/usr/bin/env python3
"""Select a deterministic, valid subset for the LoRA memorization harness."""

import argparse
import json
import random
from pathlib import Path
from typing import Any


def select_records(
    input_path: str | Path, count: int, seed: int
) -> list[dict[str, Any]]:
    """Return ``count`` shuffled valid records with unique IDs."""
    if count <= 0:
        raise ValueError("count must be positive")

    source = Path(input_path)
    valid_records: list[dict[str, Any]] = []
    seen_ids: set[object] = set()

    with source.open("r", encoding="utf-8") as manifest:
        for line in manifest:
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            if not isinstance(row, dict):
                continue

            record_id = row.get("id")
            if not isinstance(record_id, (str, int, float)) or record_id in seen_ids:
                continue
            if isinstance(record_id, str) and not record_id.strip():
                continue

            text = row.get("text")
            audio_path = row.get("audio_path")
            if not isinstance(text, str) or not text.strip():
                continue
            if not isinstance(audio_path, str) or not audio_path.strip():
                continue

            resolved_audio = Path(audio_path).expanduser()
            if not resolved_audio.is_absolute():
                resolved_audio = source.parent / resolved_audio
            resolved_audio = resolved_audio.resolve()
            if not resolved_audio.is_file():
                continue

            selected_row = dict(row)
            selected_row["audio_path"] = str(resolved_audio)
            valid_records.append(selected_row)
            seen_ids.add(record_id)

    if len(valid_records) < count:
        raise ValueError(
            f"found {len(valid_records)} valid unique records; need {count}"
        )

    random.Random(seed).shuffle(valid_records)
    return valid_records[:count]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select deterministic samples for LoRA memorization."
    )
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    records = select_records(args.input_jsonl, count=args.count, seed=args.seed)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
