from __future__ import annotations

import hashlib
import json
import threading
import time
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
import soundfile as sf
import torch

from omnivoice.cli.validate_hard_numbers import INCOMPLETE_EXIT_CODE, _run_synth
from omnivoice.validation.artifacts import AtomicJsonlLedger, ValidationPaths
from omnivoice.validation.balalaika import SelectedBalalaikaClip
from omnivoice.validation.hard_numbers import (
    HardNumberRow,
    assign_voices,
    write_assignment_manifest,
)
from omnivoice.validation.synthesis import (
    GENERATION_CONFIG,
    DistributedContext,
    ModelSourceIdentity,
    _record_is_resumable,
    fingerprint_adapter_checkpoint,
    load_assignment_manifest,
    load_validation_tts,
    read_synthesis_summary,
    resolve_distributed_context,
    resolve_model_source,
    run_bounded,
    synchronize_distributed,
    synthesize_rank,
)


class FakeModel:
    sampling_rate = 24_000

    def __init__(self, *, fail_text: str | None = None) -> None:
        self.fail_text = fail_text
        self.prompt_calls: list[dict[str, object]] = []
        self.generate_calls: list[dict[str, object]] = []

    def create_voice_clone_prompt(self, **kwargs):
        self.prompt_calls.append(kwargs)
        return (kwargs["ref_audio"], kwargs["ref_text"])

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        if kwargs["text"] == self.fail_text:
            raise RuntimeError("injected generation failure")
        return [np.linspace(-0.1, 0.1, 240, dtype=np.float32)]


@pytest.fixture(scope="module")
def assignments(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("synthesis-assignments")
    voices = []
    for index in range(20):
        wav = root / f"voice-{index:02d}.wav"
        sf.write(
            wav,
            np.full(72_000, index / 100.0, dtype=np.float32),
            24_000,
            subtype="PCM_16",
        )
        voices.append(
            SelectedBalalaikaClip(
                role="validation_voice",
                source_relative_path=f"{index:06d}/voice.mp3",
                text=f"Референс {index}",
                schema_version=1 + index % 2,
                seed=42,
                source_shard=f"train/shard_{index:06d}.tar",
                member_name="voice.mp3",
                audio_path=str(wav),
                source_sha256=f"{index + 1:064x}",
                wav_sha256=hashlib.sha256(wav.read_bytes()).hexdigest(),
                sample_rate=24_000,
                channels=1,
                duration=3.0,
                duration_tier="preferred_3_to_12s",
            )
        )
    rows = [
        HardNumberRow(
            id=f"prompt-{index:04d}",
            category="integer",
            hard_number=str(index),
            text=f"У меня {index} примеров.",
            normalized_gold=f"у меня число {index} примеров",
            stressed=f"У меня число {index} примеров.",
        )
        for index in range(2_000)
    ]
    return assign_voices(rows, voices)


def _source_identity(
    assignments,
    *,
    requested: str = "k2-fsa/OmniVoice",
    commit: str = "c" * 40,
) -> ModelSourceIdentity:
    load_path = Path(assignments[0].reference_audio_path).parent.resolve()
    return ModelSourceIdentity(
        kind="base",
        requested=requested,
        load_path=str(load_path),
        immutable_id=f"hf:{commit}",
    )


def test_synthesize_exact_stride_caches_prompts_and_resumes_valid_hashes(
    assignments, tmp_path: Path
) -> None:
    model = FakeModel()

    first = synthesize_rank(
        assignments=assignments,
        model=model,
        output_dir=tmp_path,
        rank=1,
        world_size=8,
        source_identity=_source_identity(assignments),
    )

    assert (first.expected, first.completed, first.generated, first.skipped) == (
        250,
        250,
        250,
        0,
    )
    assert first.complete is True
    assert first.source_identity == _source_identity(assignments)
    assert [call["text"] for call in model.generate_calls[:2]] == [
        "У меня число 1 примеров.",
        "У меня число 9 примеров.",
    ]
    assert {call["language"] for call in model.generate_calls} == {"Russian"}
    assert [call["generation_config"] for call in model.generate_calls] == [
        GENERATION_CONFIG
    ] * 250
    assert len(model.prompt_calls) == 5
    assert {call["preprocess_prompt"] for call in model.prompt_calls} == {True}

    ledger = AtomicJsonlLedger(tmp_path / "rank-manifests" / "rank-1.jsonl")
    assert len(ledger.records) == 250
    first_record = ledger.records[0]
    first_wav = Path(first_record["wav"])
    info = sf.info(first_wav)
    assert (info.samplerate, info.channels, info.subtype) == (24_000, 1, "PCM_16")
    assert hashlib.sha256(first_wav.read_bytes()).hexdigest() == first_record["sha256"]
    assert first_record["generation_config"] == {
        "audio_chunk_duration": 15.0,
        "audio_chunk_threshold": 30.0,
        "class_temperature": 0.0,
        "denoise": True,
        "fade_duration": 0.1,
        "guidance_scale": 2.0,
        "language": "Russian",
        "layer_penalty_factor": 5.0,
        "num_step": 32,
        "pad_duration": 0.1,
        "position_temperature": 0.0,
        "postprocess_output": True,
        "preprocess_prompt": True,
        "t_shift": 0.1,
    }
    assert first_record["assignment"] == asdict(assignments[1])
    canonical_assignment = json.dumps(
        asdict(assignments[1]),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert (
        first_record["assignment_sha256"]
        == hashlib.sha256(canonical_assignment).hexdigest()
    )
    assert first_record["source_identity"] == asdict(_source_identity(assignments))

    resumed_model = FakeModel()
    resumed = synthesize_rank(
        assignments=assignments,
        model=resumed_model,
        output_dir=tmp_path,
        rank=1,
        world_size=8,
        source_identity=_source_identity(assignments),
    )
    assert (resumed.generated, resumed.skipped, resumed.completed) == (0, 250, 250)
    assert resumed_model.generate_calls == []
    assert resumed_model.prompt_calls == []

    first_wav.write_bytes(b"corrupt")
    repair_model = FakeModel()
    repaired = synthesize_rank(
        assignments=assignments,
        model=repair_model,
        output_dir=tmp_path,
        rank=1,
        world_size=8,
        source_identity=_source_identity(assignments),
    )
    assert (repaired.generated, repaired.skipped, repaired.completed) == (1, 249, 250)
    assert len(repair_model.generate_calls) == 1


def test_cli_owned_normal_summary_records_context_and_reuses_valid_counts(
    assignments, tmp_path: Path
) -> None:
    args = SimpleNamespace(
        assignments=tmp_path / "assignments.jsonl",
        output_root=tmp_path,
        run_id="normal-cli-run",
        step=37,
        model="base",
        adapter_checkpoint=None,
        deadline_monotonic=0.0,
    )
    source = _source_identity(assignments)
    common = {
        "assignment_loader": lambda path: assignments,
        "context_resolver": lambda: DistributedContext(0, 0, 8),
        "source_resolver": lambda **kwargs: source,
        "distributed_initializer": lambda context: None,
        "synchronizer": lambda: True,
        "cuda_releaser": lambda: None,
        "output": lambda payload: None,
    }

    code = _run_synth(
        args,
        model_loader=lambda **kwargs: FakeModel(),
        **common,
    )

    assert code == INCOMPLETE_EXIT_CODE
    paths = ValidationPaths(tmp_path, args.run_id, args.step)
    durable = read_synthesis_summary(paths, rank=0)
    assert (durable.run_id, durable.step) == (args.run_id, args.step)
    assert (
        durable.expected,
        durable.completed,
        durable.generated,
        durable.skipped,
        durable.failed,
    ) == (250, 0, 0, 0, 0)

    with pytest.raises(ValueError, match="same context load exploded"):
        _run_synth(
            args,
            model_loader=lambda **kwargs: (_ for _ in ()).throw(
                ValueError("same context load exploded")
            ),
            **common,
        )

    reused = read_synthesis_summary(paths, rank=0)
    assert (
        reused.expected,
        reused.completed,
        reused.generated,
        reused.skipped,
        reused.failed,
    ) == (250, 0, 0, 0, 0)
    assert reused.primary_error.message == "same context load exploded"


@pytest.mark.parametrize(
    ("field", "nested_field"),
    [
        ("dataset_repo", None),
        ("category", None),
        ("normalized_gold", None),
        ("stressed", None),
        ("reference_text", None),
        ("reference_duration", None),
        ("reference_seed", None),
        ("reference_source_sha256", None),
        ("generation_config", "language"),
        ("generation_config", "num_step"),
    ],
)
def test_resume_rejects_one_field_assignment_provenance_corruption(
    assignments,
    tmp_path: Path,
    field: str,
    nested_field: str | None,
) -> None:
    row = assignments[0]
    wav = tmp_path / "valid.wav"
    sf.write(wav, np.zeros(240, dtype=np.float32), 24_000, subtype="PCM_16")
    assignment = asdict(row)
    canonical = json.dumps(
        assignment,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    record = {
        "assignment": assignment,
        "assignment_sha256": hashlib.sha256(canonical).hexdigest(),
        "category": row.category,
        "channels": 1,
        "checkpoint": "base",
        "generation_config": asdict(row.generation_config),
        "hard_number": row.hard_number,
        "id": row.id,
        "normalized_gold": row.normalized_gold,
        "rank": 0,
        "reference_audio_path": row.reference_audio_path,
        "reference_wav_sha256": row.reference_wav_sha256,
        "sample_rate": 24_000,
        "sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
        "stressed": row.stressed,
        "text": row.text,
        "voice_id": row.voice_id,
        "wav": str(wav),
        "source_identity": asdict(_source_identity(assignments, requested="base")),
    }
    corrupted = deepcopy(record)
    if nested_field is None:
        corrupted["assignment"][field] = "tampered"
    else:
        corrupted["assignment"][field][nested_field] = "tampered"

    assert not _record_is_resumable(
        corrupted,
        row=row,
        rank=0,
        source_identity=_source_identity(assignments, requested="base"),
        wav_path=wav,
    )


def test_one_assignment_provenance_corruption_regenerates_exactly_one(
    assignments, tmp_path: Path
) -> None:
    synthesize_rank(
        assignments=assignments,
        model=FakeModel(),
        output_dir=tmp_path,
        rank=5,
        world_size=8,
        source_identity=_source_identity(assignments, requested="base"),
    )
    ledger = AtomicJsonlLedger(tmp_path / "rank-manifests" / "rank-5.jsonl")
    record = ledger.records[0]
    record["assignment"]["reference_text"] = "подменено"
    ledger.upsert(record)

    repair = FakeModel()
    summary = synthesize_rank(
        assignments=assignments,
        model=repair,
        output_dir=tmp_path,
        rank=5,
        world_size=8,
        source_identity=_source_identity(assignments, requested="base"),
    )

    assert (summary.generated, summary.skipped, summary.completed) == (1, 249, 250)
    assert len(repair.generate_calls) == 1


def test_changed_immutable_source_identity_regenerates(
    assignments, tmp_path: Path
) -> None:
    synthesize_rank(
        assignments=assignments,
        model=FakeModel(),
        output_dir=tmp_path,
        rank=0,
        world_size=8,
        source_identity=_source_identity(
            assignments, requested="same-source", commit="d" * 40
        ),
    )

    model = FakeModel()
    summary = synthesize_rank(
        assignments=assignments,
        model=model,
        output_dir=tmp_path,
        rank=0,
        world_size=8,
        source_identity=_source_identity(
            assignments, requested="same-source", commit="e" * 40
        ),
    )

    assert (summary.generated, summary.skipped) == (250, 0)
    assert len(model.generate_calls) == 250


def test_mutated_adapter_at_same_path_regenerates_and_unchanged_content_resumes(
    assignments, tmp_path: Path
) -> None:
    checkpoint = _adapter_checkpoint(tmp_path / "source")
    resolver = _snapshot_resolver(tmp_path / "base")
    first_source = resolve_model_source(
        adapter_checkpoint=checkpoint, snapshot_resolver=resolver
    )
    output = tmp_path / "output"
    synthesize_rank(
        assignments=assignments,
        model=FakeModel(),
        output_dir=output,
        rank=6,
        world_size=8,
        source_identity=first_source,
    )

    unchanged_model = FakeModel()
    unchanged = synthesize_rank(
        assignments=assignments,
        model=unchanged_model,
        output_dir=output,
        rank=6,
        world_size=8,
        source_identity=resolve_model_source(
            adapter_checkpoint=checkpoint, snapshot_resolver=resolver
        ),
    )
    assert (unchanged.generated, unchanged.skipped) == (0, 250)

    (checkpoint / "tokenizer.json").write_text(
        '{"vocab":{"mutated":1}}\n', encoding="utf-8"
    )
    changed_source = resolve_model_source(
        adapter_checkpoint=checkpoint, snapshot_resolver=resolver
    )
    changed_model = FakeModel()
    changed = synthesize_rank(
        assignments=assignments,
        model=changed_model,
        output_dir=output,
        rank=6,
        world_size=8,
        source_identity=changed_source,
    )
    assert first_source.load_path == changed_source.load_path
    assert first_source.immutable_id != changed_source.immutable_id
    assert (changed.generated, changed.skipped) == (250, 0)


def test_generation_error_is_atomic_and_retried_without_skipping(
    assignments, tmp_path: Path
) -> None:
    failed_text = assignments[2].stressed
    summary = synthesize_rank(
        assignments=assignments,
        model=FakeModel(fail_text=failed_text),
        output_dir=tmp_path,
        rank=2,
        world_size=8,
        source_identity=_source_identity(assignments, requested="checkpoint-625"),
    )

    assert (summary.expected, summary.completed, summary.failed, summary.complete) == (
        250,
        249,
        1,
        False,
    )
    records = AtomicJsonlLedger(tmp_path / "rank-manifests" / "rank-2.jsonl").records
    error = next(record for record in records if "error" in record)
    assert error == {
        "checkpoint": "checkpoint-625",
        "error": "RuntimeError: injected generation failure",
        "id": assignments[2].id,
        "rank": 2,
        "source_identity": asdict(
            _source_identity(assignments, requested="checkpoint-625")
        ),
        "voice_id": assignments[2].voice_id,
    }

    repair = FakeModel()
    resumed = synthesize_rank(
        assignments=assignments,
        model=repair,
        output_dir=tmp_path,
        rank=2,
        world_size=8,
        source_identity=_source_identity(assignments, requested="checkpoint-625"),
    )
    assert (resumed.generated, resumed.skipped, resumed.completed) == (1, 249, 250)
    assert len(repair.generate_calls) == 1


def test_deadline_and_stop_request_leave_durable_partial_summary(
    assignments, tmp_path: Path
) -> None:
    deadline_model = FakeModel()
    deadline = synthesize_rank(
        assignments=assignments,
        model=deadline_model,
        output_dir=tmp_path / "deadline",
        rank=3,
        world_size=8,
        source_identity=_source_identity(assignments, requested="base"),
        deadline_monotonic=10.0,
        monotonic=lambda: 10.0,
    )
    assert deadline.complete is False
    assert deadline.stop_reason == "deadline"
    assert deadline.completed == 0
    assert deadline_model.generate_calls == []

    stop_model = FakeModel()
    stop_requested = lambda: len(stop_model.generate_calls) == 1
    stopped = synthesize_rank(
        assignments=assignments,
        model=stop_model,
        output_dir=tmp_path / "signal",
        rank=3,
        world_size=8,
        source_identity=_source_identity(assignments, requested="base"),
        stop_requested=stop_requested,
    )
    assert stopped.stop_reason == "signal"
    assert stopped.completed == 1
    assert len(stop_model.generate_calls) == 1
    summary_path = tmp_path / "signal" / "rank-manifests" / "rank-3.summary.json"
    assert json.loads(summary_path.read_text(encoding="utf-8"))["complete"] is False


def test_load_assignment_manifest_preserves_task5_validation(
    assignments, tmp_path: Path
) -> None:
    manifest = write_assignment_manifest(assignments, tmp_path / "assignments.jsonl")
    assert load_assignment_manifest(manifest) == assignments

    rows = manifest.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["dataset_revision"] = "main"
    rows[0] = json.dumps(first, ensure_ascii=False)
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="provenance is not exactly pinned"):
        load_assignment_manifest(manifest)


class _FakeCuda:
    def __init__(self) -> None:
        self.devices: list[int] = []

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def device_count() -> int:
        return 8

    def set_device(self, local_rank: int) -> None:
        self.devices.append(local_rank)


class _FakeOmniVoice:
    calls: ClassVar[list[tuple[str, object, dict[str, object]]]] = []

    @classmethod
    def from_pretrained(cls, source, **kwargs):
        cls.calls.append(("base", source, kwargs))
        return object()

    @classmethod
    def from_lora_pretrained(cls, source, **kwargs):
        cls.calls.append(("adapter", source, kwargs))
        return object()


def test_base_source_resolves_to_immutable_snapshot_before_gpu_load(
    tmp_path: Path,
) -> None:
    commit = "a" * 40
    snapshot = tmp_path / "models--k2-fsa--OmniVoice" / "snapshots" / commit
    snapshot.mkdir(parents=True)
    resolver_calls: list[dict[str, object]] = []

    def snapshot_resolver(**kwargs):
        resolver_calls.append(kwargs)
        return str(snapshot)

    source = resolve_model_source(
        model_name="k2-fsa/OmniVoice", snapshot_resolver=snapshot_resolver
    )

    assert source == ModelSourceIdentity(
        kind="base",
        requested="k2-fsa/OmniVoice",
        load_path=str(snapshot.resolve()),
        immutable_id=f"hf:{commit}",
    )
    assert resolver_calls == [{"repo_id": "k2-fsa/OmniVoice"}]

    _FakeOmniVoice.calls.clear()
    fake_torch = SimpleNamespace(cuda=_FakeCuda(), float16=torch.float16)
    load_validation_tts(
        source_identity=source,
        context=DistributedContext(rank=6, local_rank=3, world_size=8),
        model_class=_FakeOmniVoice,
        torch_module=fake_torch,
    )
    assert _FakeOmniVoice.calls == [
        (
            "base",
            str(snapshot.resolve()),
            {"device_map": "cuda:3", "dtype": torch.float16},
        )
    ]


def _adapter_checkpoint(root: Path) -> Path:
    checkpoint = root / "checkpoint-625"
    adapter = checkpoint / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights-v1")
    (checkpoint / "adapter_metadata.json").write_text(
        '{"base_model_name_or_path":"k2-fsa/OmniVoice","format_version":1}\n',
        encoding="utf-8",
    )
    (checkpoint / "tokenizer.json").write_text('{"vocab":{}}\n', encoding="utf-8")
    return checkpoint


def _snapshot_resolver(root: Path, commit: str = "a" * 40):
    snapshot = root / "snapshots" / commit
    snapshot.mkdir(parents=True, exist_ok=True)

    def resolve(**kwargs):
        assert kwargs == {"repo_id": "k2-fsa/OmniVoice"}
        return snapshot

    return resolve


@pytest.mark.parametrize(
    ("relative_path", "replacement"),
    [
        ("adapter/adapter_model.safetensors", b"weights-v2"),
        ("adapter/adapter_config.json", b'{"inference_mode":true}\n'),
        ("adapter_metadata.json", b'{"format_version":1,"step":626}\n'),
        ("tokenizer.json", b'{"vocab":{"new":1}}\n'),
    ],
)
def test_adapter_identity_fingerprints_every_recursive_checkpoint_file(
    tmp_path: Path, relative_path: str, replacement: bytes
) -> None:
    checkpoint = _adapter_checkpoint(tmp_path)
    before = fingerprint_adapter_checkpoint(checkpoint)
    unchanged = fingerprint_adapter_checkpoint(checkpoint)
    (checkpoint / relative_path).write_bytes(replacement)
    after = fingerprint_adapter_checkpoint(checkpoint)

    assert unchanged == before
    assert after != before
    assert len(before) == 64


def test_adapter_source_uses_canonical_root_and_content_identity(
    tmp_path: Path,
) -> None:
    checkpoint = _adapter_checkpoint(tmp_path)
    digest = fingerprint_adapter_checkpoint(checkpoint)
    resolver = _snapshot_resolver(tmp_path / "base")

    source = resolve_model_source(
        adapter_checkpoint=checkpoint / "adapter", snapshot_resolver=resolver
    )

    assert source.kind == "adapter"
    assert source.requested == str(checkpoint / "adapter")
    assert source.load_path == str(checkpoint.resolve())
    assert source.content_sha256 == digest
    assert source.immutable_id.startswith("sha256:")
    assert source.base_source == ModelSourceIdentity(
        kind="base",
        requested="k2-fsa/OmniVoice",
        load_path=str((tmp_path / "base" / "snapshots" / ("a" * 40)).resolve()),
        immutable_id=f"hf:{'a' * 40}",
    )


def test_adapter_identity_changes_when_recorded_base_snapshot_changes(
    tmp_path: Path,
) -> None:
    checkpoint = _adapter_checkpoint(tmp_path / "adapter")
    first = resolve_model_source(
        adapter_checkpoint=checkpoint,
        snapshot_resolver=_snapshot_resolver(tmp_path / "base-a", "a" * 40),
    )
    second = resolve_model_source(
        adapter_checkpoint=checkpoint,
        snapshot_resolver=_snapshot_resolver(tmp_path / "base-b", "b" * 40),
    )

    assert first.content_sha256 == second.content_sha256
    assert first.base_source.immutable_id == f"hf:{'a' * 40}"
    assert second.base_source.immutable_id == f"hf:{'b' * 40}"
    assert first.immutable_id != second.immutable_id

    with pytest.raises(ValueError, match="composite identity"):
        replace(first, immutable_id=f"sha256:{'f' * 64}")


def test_unresolved_base_and_malformed_adapter_fail_before_cuda(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="immutable Hub snapshot"):
        resolve_model_source(
            model_name="k2-fsa/OmniVoice",
            snapshot_resolver=lambda **kwargs: tmp_path / "not-a-snapshot",
        )
    malformed = tmp_path / "checkpoint"
    malformed.mkdir()
    with pytest.raises(FileNotFoundError, match="LoRA adapter"):
        resolve_model_source(adapter_checkpoint=malformed)

    inconsistent = _adapter_checkpoint(tmp_path / "inconsistent")
    (inconsistent / "adapter_metadata.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "lora_rank": 64,
                "lora_alpha": 128,
                "lora_dropout": 0.05,
                "lora_bias": "none",
                "lora_target_modules": ["q_proj"],
            }
        ),
        encoding="utf-8",
    )
    (inconsistent / "adapter" / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 32,
                "lora_alpha": 128,
                "lora_dropout": 0.05,
                "bias": "none",
                "target_modules": ["q_proj"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="adapter_config.r"):
        resolve_model_source(adapter_checkpoint=inconsistent)


def test_loader_binds_one_fp16_model_and_selects_exact_source(tmp_path: Path) -> None:
    _FakeOmniVoice.calls.clear()
    fake_torch = SimpleNamespace(cuda=_FakeCuda(), float16=torch.float16)
    context = DistributedContext(rank=6, local_rank=3, world_size=8)
    base_snapshot = tmp_path / "snapshots" / ("b" * 40)
    base_snapshot.mkdir(parents=True)
    base_source = ModelSourceIdentity(
        kind="base",
        requested="k2-fsa/OmniVoice",
        load_path=str(base_snapshot.resolve()),
        immutable_id=f"hf:{'b' * 40}",
    )

    load_validation_tts(
        source_identity=base_source,
        context=context,
        model_class=_FakeOmniVoice,
        torch_module=fake_torch,
    )
    checkpoint = _adapter_checkpoint(tmp_path)
    adapter_source = resolve_model_source(
        adapter_checkpoint=checkpoint,
        snapshot_resolver=_snapshot_resolver(tmp_path / "adapter-base"),
    )
    load_validation_tts(
        source_identity=adapter_source,
        context=context,
        model_class=_FakeOmniVoice,
        torch_module=fake_torch,
    )

    assert fake_torch.cuda.devices == [3, 3]
    assert _FakeOmniVoice.calls == [
        (
            "base",
            str(base_snapshot.resolve()),
            {"device_map": "cuda:3", "dtype": torch.float16},
        ),
        (
            "adapter",
            checkpoint.resolve(),
            {
                "base_model_override": str(
                    (tmp_path / "adapter-base" / "snapshots" / ("a" * 40)).resolve()
                ),
                "device_map": "cuda:3",
                "dtype": torch.float16,
            },
        ),
    ]


def test_loader_rechecks_adapter_fingerprint_before_touching_cuda(
    tmp_path: Path,
) -> None:
    checkpoint = _adapter_checkpoint(tmp_path)
    source = resolve_model_source(
        adapter_checkpoint=checkpoint,
        snapshot_resolver=_snapshot_resolver(tmp_path / "base"),
    )
    (checkpoint / "adapter" / "adapter_model.safetensors").write_bytes(
        b"mutated-after-resolution"
    )
    fake_cuda = _FakeCuda()
    fake_torch = SimpleNamespace(cuda=fake_cuda, float16=torch.float16)

    with pytest.raises(ValueError, match="changed after identity resolution"):
        load_validation_tts(
            source_identity=source,
            context=DistributedContext(rank=0, local_rank=0, world_size=8),
            model_class=_FakeOmniVoice,
            torch_module=fake_torch,
        )

    assert fake_cuda.devices == []


def test_loader_rechecks_adapter_after_model_load(tmp_path: Path) -> None:
    checkpoint = _adapter_checkpoint(tmp_path)
    source = resolve_model_source(
        adapter_checkpoint=checkpoint,
        snapshot_resolver=_snapshot_resolver(tmp_path / "base"),
    )

    class MutatingOmniVoice:
        @classmethod
        def from_lora_pretrained(cls, source_path, **kwargs):
            del kwargs
            (Path(source_path) / "tokenizer.json").write_text(
                '{"vocab":{"mutated":1}}\n', encoding="utf-8"
            )
            return object()

    fake_torch = SimpleNamespace(cuda=_FakeCuda(), float16=torch.float16)
    with pytest.raises(ValueError, match="changed during model loading"):
        load_validation_tts(
            source_identity=source,
            context=DistributedContext(rank=0, local_rank=0, world_size=8),
            model_class=MutatingOmniVoice,
            torch_module=fake_torch,
        )


def test_rank_environment_is_strict_and_requires_eight_world_ranks() -> None:
    assert resolve_distributed_context(
        {"RANK": "6", "LOCAL_RANK": "2", "WORLD_SIZE": "8"}
    ) == DistributedContext(rank=6, local_rank=2, world_size=8)
    with pytest.raises(ValueError, match="WORLD_SIZE must be exactly 8"):
        resolve_distributed_context({"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"})
    with pytest.raises(ValueError, match="RANK, LOCAL_RANK, and WORLD_SIZE"):
        resolve_distributed_context({"RANK": "0", "WORLD_SIZE": "8"})


def test_synchronization_timeout_destroys_group_instead_of_deadlocking() -> None:
    class Work:
        def wait(self, timeout):
            raise RuntimeError("barrier timeout")

    class Dist:
        def __init__(self) -> None:
            self.destroyed = False

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def barrier(async_op):
            assert async_op is True
            return Work()

        def destroy_process_group(self):
            self.destroyed = True

    dist = Dist()
    assert synchronize_distributed(dist_module=dist, timeout_seconds=0.01) is False
    assert dist.destroyed is True


def test_synchronization_requires_work_wait_to_return_exactly_true() -> None:
    class Work:
        @staticmethod
        def wait(timeout):
            del timeout
            return False

    class Dist:
        def __init__(self) -> None:
            self.destroyed = False

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def barrier(async_op):
            assert async_op is True
            return Work()

        def destroy_process_group(self):
            self.destroyed = True

    dist = Dist()
    assert synchronize_distributed(dist_module=dist) is False
    assert dist.destroyed is True


def test_outer_lifecycle_deadline_owns_blocking_process_group_destroy() -> None:
    release = threading.Event()
    timer = threading.Timer(0.5, release.set)

    class Work:
        @staticmethod
        def wait(timeout):
            del timeout
            return True

    class Dist:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def barrier(async_op):
            assert async_op is True
            return Work()

        @staticmethod
        def destroy_process_group():
            release.wait()

    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="synchronization"):
            run_bounded(
                lambda: synchronize_distributed(
                    dist_module=Dist(), timeout_seconds=0.01
                ),
                deadline_monotonic=time.monotonic() + 0.02,
                description="synchronization",
            )
        assert time.monotonic() - started < 0.2
    finally:
        release.set()
        timer.cancel()
