from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
import soundfile as sf
import torch

from omnivoice.validation.artifacts import AtomicJsonlLedger
from omnivoice.validation.balalaika import SelectedBalalaikaClip
from omnivoice.validation.hard_numbers import (
    HardNumberRow,
    assign_voices,
    write_assignment_manifest,
)
from omnivoice.validation.synthesis import (
    GENERATION_CONFIG,
    DistributedContext,
    load_assignment_manifest,
    load_validation_tts,
    resolve_distributed_context,
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
        checkpoint="k2-fsa/OmniVoice",
    )

    assert (first.expected, first.completed, first.generated, first.skipped) == (
        250,
        250,
        250,
        0,
    )
    assert first.complete is True
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

    resumed_model = FakeModel()
    resumed = synthesize_rank(
        assignments=assignments,
        model=resumed_model,
        output_dir=tmp_path,
        rank=1,
        world_size=8,
        checkpoint="k2-fsa/OmniVoice",
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
        checkpoint="k2-fsa/OmniVoice",
    )
    assert (repaired.generated, repaired.skipped, repaired.completed) == (1, 249, 250)
    assert len(repair_model.generate_calls) == 1


def test_stale_checkpoint_record_is_regenerated(assignments, tmp_path: Path) -> None:
    synthesize_rank(
        assignments=assignments,
        model=FakeModel(),
        output_dir=tmp_path,
        rank=0,
        world_size=8,
        checkpoint="old-checkpoint",
    )

    model = FakeModel()
    summary = synthesize_rank(
        assignments=assignments,
        model=model,
        output_dir=tmp_path,
        rank=0,
        world_size=8,
        checkpoint="new-checkpoint",
    )

    assert (summary.generated, summary.skipped) == (250, 0)
    assert len(model.generate_calls) == 250


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
        checkpoint="checkpoint-625",
    )

    assert (summary.expected, summary.completed, summary.failed, summary.complete) == (
        250,
        249,
        1,
        False,
    )
    records = AtomicJsonlLedger(
        tmp_path / "rank-manifests" / "rank-2.jsonl"
    ).records
    error = next(record for record in records if "error" in record)
    assert error == {
        "checkpoint": "checkpoint-625",
        "error": "RuntimeError: injected generation failure",
        "id": assignments[2].id,
        "rank": 2,
        "voice_id": assignments[2].voice_id,
    }

    repair = FakeModel()
    resumed = synthesize_rank(
        assignments=assignments,
        model=repair,
        output_dir=tmp_path,
        rank=2,
        world_size=8,
        checkpoint="checkpoint-625",
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
        checkpoint="base",
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
        checkpoint="base",
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


def test_loader_binds_one_fp16_model_and_selects_exact_source(tmp_path: Path) -> None:
    _FakeOmniVoice.calls.clear()
    fake_torch = SimpleNamespace(cuda=_FakeCuda(), float16=torch.float16)
    context = DistributedContext(rank=6, local_rank=3, world_size=8)

    load_validation_tts(
        model_name="k2-fsa/OmniVoice",
        context=context,
        model_class=_FakeOmniVoice,
        torch_module=fake_torch,
    )
    checkpoint = tmp_path / "checkpoint-625"
    checkpoint.mkdir()
    load_validation_tts(
        adapter_checkpoint=checkpoint,
        context=context,
        model_class=_FakeOmniVoice,
        torch_module=fake_torch,
    )

    assert fake_torch.cuda.devices == [3, 3]
    assert _FakeOmniVoice.calls == [
        (
            "base",
            "k2-fsa/OmniVoice",
            {"device_map": "cuda:3", "dtype": torch.float16},
        ),
        (
            "adapter",
            checkpoint,
            {"device_map": "cuda:3", "dtype": torch.float16},
        ),
    ]


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
