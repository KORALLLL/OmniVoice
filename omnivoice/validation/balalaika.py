"""Deterministically prepare Balalaika audio from its authoritative sidecar."""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import re
import subprocess
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

import soundfile as sf

_SOURCE_PATH = re.compile(r"^(?P<shard>\d{6})/(?P<member>[^/]+\.mp3)$")
_MEMBER_TIMES = re.compile(
    r"^(?P<start_seconds>\d+)_(?P<start_fraction>\d+)_"
    r"(?P<end_seconds>\d+)_(?P<end_fraction>\d+)(?:_|\.)"
)
_PREFERRED_TIER = "preferred_3_to_12s"
_FALLBACK_TIER = "fallback_over_12s"
_POOL_SIZES = (512, 2_048, 8_192)


@dataclass(frozen=True)
class BalalaikaCandidate:
    source_relative_path: str
    text: str
    schema_version: int
    priority: bytes
    estimated_duration: float


@dataclass(frozen=True)
class SelectedBalalaikaClip:
    role: str
    source_relative_path: str
    text: str
    schema_version: int
    seed: int
    source_shard: str
    member_name: str
    audio_path: str
    source_sha256: str
    wav_sha256: str
    sample_rate: int
    channels: int
    duration: float
    duration_tier: str


@dataclass(frozen=True)
class _PreparedClip:
    candidate: BalalaikaCandidate
    staged_wav: Path
    source_shard: str
    member_name: str
    source_sha256: str
    wav_sha256: str
    sample_rate: int
    channels: int
    duration: float
    duration_tier: str


def stable_priority(seed: int, source_relative_path: str) -> bytes:
    value = f"{seed}\0{source_relative_path}".encode("utf-8")  # noqa: UP012
    return hashlib.sha256(value).digest()


def _parse_estimated_duration(source_relative_path: str) -> float | None:
    source_match = _SOURCE_PATH.fullmatch(source_relative_path)
    if source_match is None:
        return None
    name = source_match.group("member")
    time_match = _MEMBER_TIMES.match(name)
    if time_match is None:
        return None
    start = float(
        f"{time_match.group('start_seconds')}.{time_match.group('start_fraction')}"
    )
    end = float(
        f"{time_match.group('end_seconds')}.{time_match.group('end_fraction')}"
    )
    duration = end - start
    return duration if duration >= 3.0 else None


def _retain_lowest(
    heap: list[tuple[int, str]],
    retained: dict[str, BalalaikaCandidate],
    candidate: BalalaikaCandidate,
    pool_size: int,
) -> None:
    if candidate.source_relative_path in retained:
        return
    priority_value = int.from_bytes(candidate.priority, "big")
    item = (-priority_value, candidate.source_relative_path)
    if len(heap) < pool_size:
        heapq.heappush(heap, item)
        retained[candidate.source_relative_path] = candidate
    elif priority_value < -heap[0][0]:
        removed = heapq.heapreplace(heap, item)
        retained.pop(removed[1])
        retained[candidate.source_relative_path] = candidate


def rank_candidates(
    sidecar_path: str | Path, *, seed: int, pool_size: int = 512
) -> list[BalalaikaCandidate]:
    """Stream a sidecar and retain bounded preferred and fallback pools."""
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")

    preferred_heap: list[tuple[int, str]] = []
    fallback_heap: list[tuple[int, str]] = []
    preferred: dict[str, BalalaikaCandidate] = {}
    fallback: dict[str, BalalaikaCandidate] = {}
    with Path(sidecar_path).open("r", encoding="utf-8") as sidecar:
        for line in sidecar:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue

            source_relative_path = row.get("source_relative_path")
            text = row.get("rover_punctuated_accented")
            schema_version = row.get("schema_version")
            if (
                not isinstance(source_relative_path, str)
                or not source_relative_path.strip()
                or not isinstance(text, str)
                or not text.strip()
                or not isinstance(schema_version, int)
                or isinstance(schema_version, bool)
            ):
                continue
            estimated_duration = _parse_estimated_duration(source_relative_path)
            if estimated_duration is None:
                continue

            candidate = BalalaikaCandidate(
                source_relative_path=source_relative_path,
                text=text,
                schema_version=schema_version,
                priority=stable_priority(seed, source_relative_path),
                estimated_duration=estimated_duration,
            )
            if estimated_duration <= 12.0:
                _retain_lowest(preferred_heap, preferred, candidate, pool_size)
            else:
                _retain_lowest(fallback_heap, fallback, candidate, pool_size)

    return sorted(preferred.values(), key=lambda candidate: candidate.priority) + sorted(
        fallback.values(), key=lambda candidate: candidate.priority
    )


def _copy_and_hash(source: BinaryIO, destination: Path) -> str:
    digest = hashlib.sha256()
    with destination.open("wb") as output:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            output.write(chunk)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_source(
    root: Path, candidate: BalalaikaCandidate, destination: Path
) -> tuple[str, str, str] | None:
    match = _SOURCE_PATH.fullmatch(candidate.source_relative_path)
    if match is None:
        return None
    shard = match.group("shard")
    member_name = match.group("member")
    tar_path = root / "train" / f"shard_{shard}.tar"
    try:
        with tarfile.open(tar_path, "r:*") as archive:
            member = archive.getmember(member_name)
            if not member.isfile() or member.size <= 0:
                return None
            extracted = archive.extractfile(member)
            if extracted is None:
                return None
            with extracted:
                source_sha256 = _copy_and_hash(extracted, destination)
    except (KeyError, OSError, tarfile.TarError):
        return None
    return f"train/shard_{shard}.tar", member_name, source_sha256


def _prepare_candidate(
    root: Path,
    candidate: BalalaikaCandidate,
    staging_dir: Path,
) -> _PreparedClip | None:
    stem = candidate.priority.hex()
    source_mp3 = staging_dir / f"{stem}.mp3"
    staged_wav = staging_dir / f"{stem}.wav"
    source_details = _extract_source(root, candidate, source_mp3)
    if source_details is None:
        return None

    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        str(source_mp3),
        "-ac",
        "1",
        "-ar",
        "24000",
        "-c:a",
        "pcm_s16le",
        str(staged_wav),
    ]
    try:
        subprocess.run(command, check=True)
        info = sf.info(staged_wav)
    except (OSError, RuntimeError, subprocess.CalledProcessError):
        source_mp3.unlink(missing_ok=True)
        staged_wav.unlink(missing_ok=True)
        return None
    source_mp3.unlink(missing_ok=True)

    if info.frames <= 0 or info.samplerate <= 0:
        staged_wav.unlink(missing_ok=True)
        return None
    duration = info.frames / info.samplerate
    if duration < 3.0:
        staged_wav.unlink(missing_ok=True)
        return None
    duration_tier = _PREFERRED_TIER if duration <= 12.0 else _FALLBACK_TIER
    source_shard, member_name, source_sha256 = source_details
    return _PreparedClip(
        candidate=candidate,
        staged_wav=staged_wav,
        source_shard=source_shard,
        member_name=member_name,
        source_sha256=source_sha256,
        wav_sha256=_sha256_file(staged_wav),
        sample_rate=info.samplerate,
        channels=info.channels,
        duration=duration,
        duration_tier=duration_tier,
    )


def _prepare_pool(
    *,
    root: Path,
    candidates: list[BalalaikaCandidate],
    staging_dir: Path,
    count: int,
) -> list[_PreparedClip]:
    preferred: list[_PreparedClip] = []
    fallback: list[_PreparedClip] = []
    attempted_paths: set[str] = set()
    for candidate in candidates:
        if candidate.source_relative_path in attempted_paths:
            continue
        attempted_paths.add(candidate.source_relative_path)
        prepared = _prepare_candidate(root, candidate, staging_dir)
        if prepared is None:
            continue
        if prepared.duration_tier == _PREFERRED_TIER:
            preferred.append(prepared)
            if len(preferred) >= count:
                break
        elif len(fallback) < count:
            fallback.append(prepared)
        else:
            prepared.staged_wav.unlink()

    preferred.sort(key=lambda clip: clip.candidate.priority)
    fallback.sort(key=lambda clip: clip.candidate.priority)
    return preferred[:count] + fallback[: max(0, count - len(preferred))]


def _publish(
    clips: list[_PreparedClip],
    *,
    output_dir: Path,
    memorization_count: int,
    seed: int,
) -> list[SelectedBalalaikaClip]:
    selected: list[SelectedBalalaikaClip] = []
    for index, clip in enumerate(clips):
        role = "memorization" if index < memorization_count else "validation_voice"
        filename = f"{clip.candidate.priority.hex()[:16]}-{clip.wav_sha256[:16]}.wav"
        audio_path = (output_dir / filename).resolve()
        os.replace(clip.staged_wav, audio_path)
        selected.append(
            SelectedBalalaikaClip(
                role=role,
                source_relative_path=clip.candidate.source_relative_path,
                text=clip.candidate.text,
                schema_version=clip.candidate.schema_version,
                seed=seed,
                source_shard=clip.source_shard,
                member_name=clip.member_name,
                audio_path=str(audio_path),
                source_sha256=clip.source_sha256,
                wav_sha256=clip.wav_sha256,
                sample_rate=clip.sample_rate,
                channels=clip.channels,
                duration=clip.duration,
                duration_tier=clip.duration_tier,
            )
        )

    if not all(Path(row.audio_path).is_file() for row in selected):
        raise RuntimeError("not all selected WAV files were published")

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".selected.", suffix=".jsonl", dir=output_dir
    )
    temporary_manifest = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as output:
            for row in selected:
                output.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_manifest, output_dir / "selected.jsonl")
    finally:
        temporary_manifest.unlink(missing_ok=True)
    return selected


def select_and_convert(
    *,
    root: str | Path,
    sidecar_path: str | Path,
    output_dir: str | Path,
    memorization_count: int = 4,
    validation_voice_count: int = 20,
    seed: int = 42,
) -> list[SelectedBalalaikaClip]:
    """Select disjoint clips, convert them, and atomically publish a manifest."""
    if memorization_count < 0 or validation_voice_count < 0:
        raise ValueError("selection counts must be non-negative")
    count = memorization_count + validation_voice_count
    if count <= 0:
        raise ValueError("at least one clip must be requested")

    root_path = Path(root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    best_count = 0
    for pool_size in _POOL_SIZES:
        candidates = rank_candidates(sidecar_path, seed=seed, pool_size=pool_size)
        with tempfile.TemporaryDirectory(
            prefix=".balalaika-staging-", dir=output_path.parent
        ) as temporary_directory:
            clips = _prepare_pool(
                root=root_path,
                candidates=candidates,
                staging_dir=Path(temporary_directory),
                count=count,
            )
            best_count = max(best_count, len(clips))
            if len(clips) == count:
                return _publish(
                    clips,
                    output_dir=output_path,
                    memorization_count=memorization_count,
                    seed=seed,
                )

    raise ValueError(f"found {best_count} decodable clips; need {count}")
