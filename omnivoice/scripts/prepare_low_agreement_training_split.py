#!/usr/bin/env python3
"""Create a deterministic train/dev split from low-agreement token shards.

The tokenization manifest remains immutable. This utility only writes two
manifest files that reference its existing TAR/JSONL shards, selecting evenly
spaced complete shards for validation.
"""

import argparse
from pathlib import Path


def parse_line(line: str) -> tuple[str, str, int, float]:
    tar_path, jsonl_path, count, duration = line.split()
    return tar_path, jsonl_path, int(count), float(duration)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dev-shards", type=int, default=4)
    args = parser.parse_args()

    source = Path(args.manifest)
    output_dir = Path(args.output_dir)
    lines = [line for line in source.read_text(encoding="utf-8").splitlines() if line]
    if args.dev_shards <= 0 or args.dev_shards >= len(lines):
        raise ValueError("--dev-shards must be positive and smaller than shard count")

    # Use complete shards at evenly spaced offsets. This avoids a contiguous
    # source segment becoming the entire held-out set and prevents partial TAR
    # rewrites.
    selected = {
        ((i + 1) * len(lines)) // (args.dev_shards + 1)
        for i in range(args.dev_shards)
    }
    dev_lines = [line for i, line in enumerate(lines) if i in selected]
    train_lines = [line for i, line in enumerate(lines) if i not in selected]
    if any(parse_line(line)[2] != 5000 for line in dev_lines):
        raise RuntimeError("Validation selection unexpectedly included a partial shard")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train.lst").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (output_dir / "dev.lst").write_text("\n".join(dev_lines) + "\n", encoding="utf-8")
    print(
        "train_samples=", sum(parse_line(line)[2] for line in train_lines),
        "dev_samples=", sum(parse_line(line)[2] for line in dev_lines),
        "dev_shards=", len(dev_lines),
    )


if __name__ == "__main__":
    main()
