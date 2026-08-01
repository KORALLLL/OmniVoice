#!/usr/bin/env python3
"""Select a deterministic, valid subset for the LoRA memorization harness."""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


_ENCODED_KEY_PREFIX = "ovkey_"
_SAFE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _namespaced_key(type_tag: str, value: str) -> str:
    return f"{_ENCODED_KEY_PREFIX}{type_tag}_{value.encode('utf-8').hex()}"


def _encode_record_id(value: object) -> str | None:
    """Encode an ID with no periods or path separators for WebDataset.

    Ordinary strings using only ASCII letters, digits, underscores, and hyphens
    remain unchanged. The ``ovkey_`` namespace is reserved; unsafe strings and
    scalar JSON values use a type tag plus the hex-encoded UTF-8 scalar spelling.
    """
    if isinstance(value, str):
        if not value.strip():
            return None
        if _SAFE_KEY.fullmatch(value) and not value.startswith(_ENCODED_KEY_PREFIX):
            return value
        return _namespaced_key("s", value)

    if isinstance(value, bool):
        return _namespaced_key("b", json.dumps(value))
    if isinstance(value, int):
        return _namespaced_key("i", json.dumps(value))
    if isinstance(value, float):
        try:
            scalar = json.dumps(value, allow_nan=False)
        except ValueError:
            return None
        return _namespaced_key("f", scalar)
    return None


def select_records(
    input_path: str | Path, count: int, seed: int
) -> list[dict[str, Any]]:
    """Return ``count`` shuffled valid records with unique IDs."""
    if count <= 0:
        raise ValueError("count must be positive")

    source = Path(input_path)
    valid_records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    with source.open("r", encoding="utf-8") as manifest:
        for line in manifest:
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            if not isinstance(row, dict):
                continue

            record_id = _encode_record_id(row.get("id"))
            if record_id is None or record_id in seen_ids:
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
            selected_row["id"] = record_id
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
