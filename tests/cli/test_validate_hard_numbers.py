from __future__ import annotations

import builtins
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

import omnivoice.cli.validate_hard_numbers as validation_cli
from omnivoice.cli.validate_hard_numbers import (
    INCOMPLETE_EXIT_CODE,
    _run_asr,
    _run_synth,
    build_parser,
)
from omnivoice.validation.artifacts import AtomicJsonlLedger, ValidationPaths
from omnivoice.validation.asr import AsrSummary
from omnivoice.validation.synthesis import (
    DistributedContext,
    ModelSourceIdentity,
    SynthesisSummary,
    read_synthesis_summary,
    write_synthesis_summary,
)

_EXCEPTION_GROUP = builtins.ExceptionGroup


def _valid_asr_stage_inputs(
    tmp_path: Path, *, deadline_monotonic: float | None = None
) -> tuple[SimpleNamespace, list[SimpleNamespace], list[dict[str, object]]]:
    wav_path = tmp_path / "synthesis.wav"
    sf.write(wav_path, np.array([0.25], dtype=np.float32), 24_000, subtype="PCM_16")
    digest = hashlib.sha256(wav_path.read_bytes()).hexdigest()
    assignments = [SimpleNamespace(id=f"utt-{index:04d}") for index in range(2_000)]
    records = [
        {
            "id": assignment.id,
            "rank": index % 8,
            "sha256": digest,
            "wav": str(wav_path),
        }
        for index, assignment in enumerate(assignments)
    ]
    return (
        SimpleNamespace(
            assignments=tmp_path / "assignments.jsonl",
            output_root=tmp_path,
            run_id="run-1",
            step=0,
            deadline_monotonic=deadline_monotonic,
        ),
        assignments,
        records,
    )


def _source_identity(
    tmp_path: Path, *, requested: str = "base", commit: str = "a" * 40
) -> ModelSourceIdentity:
    source = tmp_path / "immutable-source"
    source.mkdir(parents=True, exist_ok=True)
    return ModelSourceIdentity(
        kind="base",
        requested=requested,
        load_path=str(source.resolve()),
        immutable_id=f"hf:{commit}",
    )


def test_synth_parser_requires_exactly_one_model_source(tmp_path: Path) -> None:
    common = [
        "synth",
        "--assignments",
        str(tmp_path / "assignments.jsonl"),
        "--output-root",
        str(tmp_path),
        "--run-id",
        "run-1",
        "--step",
        "0",
    ]
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(common)
    with pytest.raises(SystemExit):
        parser.parse_args(
            common
            + [
                "--model",
                "k2-fsa/OmniVoice",
                "--adapter-checkpoint",
                str(tmp_path / "checkpoint-625"),
            ]
        )

    parsed = parser.parse_args(common + ["--model", "k2-fsa/OmniVoice"])
    assert parsed.assignments == tmp_path / "assignments.jsonl"
    assert parsed.deadline_monotonic is None


def test_run_asr_rejects_invalid_synthesis_coverage_before_loading_gigaam(
    tmp_path: Path,
) -> None:
    """Catches a CLI change that loads GigaAM before all 2,000 valid WAVs exist."""
    args = SimpleNamespace(
        assignments=tmp_path / "assignments.jsonl",
        output_root=tmp_path,
        run_id="run-1",
        step=0,
        deadline_monotonic=None,
    )
    assignments = [SimpleNamespace(id=f"utt-{index:04d}") for index in range(2_000)]
    synthesis_records = [
        {"id": assignment.id, "rank": index % 8}
        for index, assignment in enumerate(assignments)
    ]
    loaded: list[bool] = []
    output: list[str] = []

    code = _run_asr(
        args,
        assignment_loader=lambda path: assignments,
        context_resolver=lambda: DistributedContext(0, 0, 8),
        synthesis_merger=lambda paths: synthesis_records,
        model_loader=lambda local_rank: loaded.append(True),
        output=output.append,
    )

    assert code == INCOMPLETE_EXIT_CODE
    assert loaded == []
    assert json.loads(output[0]) == {
        "complete": False,
        "stage": "synthesis_coverage",
    }


def test_run_asr_rejects_non_wav_synthesis_artifacts_before_loading_gigaam(
    tmp_path: Path,
) -> None:
    """Catches a coverage check that trusts a hash without checking WAV validity."""
    args = SimpleNamespace(
        assignments=tmp_path / "assignments.jsonl",
        output_root=tmp_path,
        run_id="run-1",
        step=0,
        deadline_monotonic=None,
    )
    invalid_wav = tmp_path / "not-a-wav.wav"
    invalid_wav.write_bytes(b"hash-valid but not a WAV")
    digest = hashlib.sha256(invalid_wav.read_bytes()).hexdigest()
    assignments = [SimpleNamespace(id=f"utt-{index:04d}") for index in range(2_000)]
    synthesis_records = [
        {
            "id": assignment.id,
            "rank": index % 8,
            "sha256": digest,
            "wav": str(invalid_wav),
        }
        for index, assignment in enumerate(assignments)
    ]
    loaded: list[bool] = []
    output: list[str] = []

    code = _run_asr(
        args,
        assignment_loader=lambda path: assignments,
        context_resolver=lambda: DistributedContext(0, 0, 8),
        synthesis_merger=lambda paths: synthesis_records,
        model_loader=lambda local_rank: loaded.append(True) or object(),
        transcriber=lambda **kwargs: AsrSummary(
            rank=0,
            expected=250,
            completed=250,
            transcribed=0,
            skipped=250,
            failed=0,
            complete=True,
            stop_reason=None,
        ),
        synchronizer=lambda: True,
        output=output.append,
    )

    assert code == INCOMPLETE_EXIT_CODE
    assert loaded == []
    assert json.loads(output[0]) == {
        "complete": False,
        "stage": "synthesis_coverage",
    }


def test_run_asr_requires_global_hypothesis_coverage_after_synchronization(
    tmp_path: Path,
) -> None:
    """Catches a rank-local complete summary that bypasses the 2,000-row ASR gate."""
    args, assignments, synthesis_records = _valid_asr_stage_inputs(tmp_path)
    output: list[str] = []
    synchronized: list[bool] = []

    code = _run_asr(
        args,
        assignment_loader=lambda path: assignments,
        context_resolver=lambda: DistributedContext(0, 0, 8),
        synthesis_merger=lambda paths: synthesis_records,
        model_loader=lambda local_rank: object(),
        transcriber=lambda **kwargs: AsrSummary(
            rank=0,
            expected=250,
            completed=250,
            transcribed=250,
            skipped=0,
            failed=0,
            complete=True,
            stop_reason=None,
        ),
        synchronizer=lambda: synchronized.append(True) or True,
        output=output.append,
    )

    assert synchronized == [True]
    assert code == INCOMPLETE_EXIT_CODE
    assert json.loads(output[0])["complete"] is False


def test_run_asr_accepts_complete_global_hypothesis_coverage(
    tmp_path: Path,
) -> None:
    """Catches a global ASR gate that rejects all valid 2,000-row ledger sets."""
    args, assignments, synthesis_records = _valid_asr_stage_inputs(tmp_path)
    paths = ValidationPaths(tmp_path, "run-1", 0)
    for record in synthesis_records:
        AtomicJsonlLedger(paths.rank_hypotheses(record["rank"])).upsert(
            {"hypothesis": "ok", "id": record["id"], "rank": record["rank"]}
        )
    output: list[str] = []

    code = _run_asr(
        args,
        assignment_loader=lambda path: assignments,
        context_resolver=lambda: DistributedContext(0, 0, 8),
        synthesis_merger=lambda paths: synthesis_records,
        model_loader=lambda local_rank: object(),
        transcriber=lambda **kwargs: AsrSummary(
            rank=0,
            expected=250,
            completed=250,
            transcribed=0,
            skipped=250,
            failed=0,
            complete=True,
            stop_reason=None,
        ),
        synchronizer=lambda: True,
        output=output.append,
    )

    assert code == 0
    assert json.loads(output[0])["complete"] is True


def test_run_asr_expired_deadline_skips_gigaam_load_and_synchronizes(
    tmp_path: Path,
) -> None:
    """Catches an expired ASR deadline that still loads GigaAM before cleanup."""
    args, assignments, synthesis_records = _valid_asr_stage_inputs(
        tmp_path, deadline_monotonic=0.0
    )
    loaded: list[bool] = []
    synchronized: list[bool] = []

    code = _run_asr(
        args,
        assignment_loader=lambda path: assignments,
        context_resolver=lambda: DistributedContext(0, 0, 8),
        synthesis_merger=lambda paths: synthesis_records,
        model_loader=lambda local_rank: loaded.append(True) or object(),
        synchronizer=lambda: synchronized.append(True) or True,
        output=lambda payload: None,
    )

    assert code == INCOMPLETE_EXIT_CODE
    assert loaded == []
    assert synchronized == [True]


def test_run_asr_preload_signal_skips_gigaam_load_and_synchronizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches an early SIGTERM return that bypasses distributed teardown."""
    args, assignments, synthesis_records = _valid_asr_stage_inputs(tmp_path)
    loaded: list[bool] = []
    synchronized: list[bool] = []

    class _SignalAlreadyRequested:
        def __enter__(self):
            return lambda: True

        def __exit__(self, exc_type, exc, traceback):
            del exc_type, exc, traceback

    monkeypatch.setattr(validation_cli, "_SigtermFlag", _SignalAlreadyRequested)
    code = _run_asr(
        args,
        assignment_loader=lambda path: assignments,
        context_resolver=lambda: DistributedContext(0, 0, 8),
        synthesis_merger=lambda paths: synthesis_records,
        model_loader=lambda local_rank: loaded.append(True) or object(),
        synchronizer=lambda: synchronized.append(True) or True,
        output=lambda payload: None,
    )

    assert code == INCOMPLETE_EXIT_CODE
    assert loaded == []
    assert synchronized == [True]


def test_run_asr_signal_during_load_skips_transcription_and_synchronizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a SIGTERM arriving during GigaAM loading that starts transcription."""
    args, assignments, synthesis_records = _valid_asr_stage_inputs(tmp_path)
    stopped = [False]
    transcribed: list[bool] = []
    synchronized: list[bool] = []

    class _SignalDuringLoad:
        def __enter__(self):
            return lambda: stopped[0]

        def __exit__(self, exc_type, exc, traceback):
            del exc_type, exc, traceback

    def load(local_rank: int) -> object:
        del local_rank
        stopped[0] = True
        return object()

    monkeypatch.setattr(validation_cli, "_SigtermFlag", _SignalDuringLoad)
    code = _run_asr(
        args,
        assignment_loader=lambda path: assignments,
        context_resolver=lambda: DistributedContext(0, 0, 8),
        synthesis_merger=lambda paths: synthesis_records,
        model_loader=load,
        transcriber=lambda **kwargs: transcribed.append(True),
        synchronizer=lambda: synchronized.append(True) or True,
        output=lambda payload: None,
    )

    assert code == INCOMPLETE_EXIT_CODE
    assert transcribed == []
    assert synchronized == [True]


@pytest.mark.parametrize(
    ("source_args", "expected_source"),
    [
        ({"model": "k2-fsa/OmniVoice", "adapter_checkpoint": None}, "k2-fsa/OmniVoice"),
        (
            {"model": None, "adapter_checkpoint": Path("checkpoint-625")},
            "checkpoint-625",
        ),
    ],
)
def test_run_synth_forwards_source_and_returns_distinct_incomplete_code(
    tmp_path: Path, source_args: dict[str, object], expected_source: str
) -> None:
    assignments = [object()]
    context = DistributedContext(rank=1, local_rank=1, world_size=8)
    calls: dict[str, object] = {}
    model = object()
    source_identity = _source_identity(tmp_path, requested=expected_source)

    def model_loader(**kwargs):
        calls["loader"] = kwargs
        return model

    def synthesizer(**kwargs):
        calls["synthesizer"] = kwargs
        return SynthesisSummary(
            rank=1,
            expected=250,
            completed=249,
            generated=1,
            skipped=248,
            failed=1,
            complete=False,
            stop_reason=None,
            source_identity=source_identity,
        )

    args = SimpleNamespace(
        assignments=tmp_path / "assignments.jsonl",
        output_root=tmp_path,
        run_id="run-1",
        step=625,
        deadline_monotonic=123.0,
        **source_args,
    )
    synchronized: list[bool] = []
    released: list[bool] = []
    code = _run_synth(
        args,
        assignment_loader=lambda path: assignments,
        context_resolver=lambda: context,
        distributed_initializer=lambda context: None,
        source_resolver=lambda **kwargs: (
            calls.setdefault("source", kwargs) and source_identity
        ),
        model_loader=model_loader,
        synthesizer=synthesizer,
        synchronizer=lambda: synchronized.append(True) or True,
        cuda_releaser=lambda: released.append(True),
    )

    assert code == INCOMPLETE_EXIT_CODE
    assert calls["source"] == {
        "model_name": source_args["model"],
        "adapter_checkpoint": source_args["adapter_checkpoint"],
    }
    assert calls["loader"]["source_identity"] == source_identity
    assert calls["loader"]["context"] == context
    assert calls["synthesizer"]["source_identity"] == source_identity
    assert calls["synthesizer"]["output_dir"] == ValidationPaths(tmp_path, "run-1", 625)
    assert synchronized == [True]
    assert released == [True]


def test_run_synth_complete_requires_successful_rank_synchronization(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = SimpleNamespace(
        assignments=tmp_path / "assignments.jsonl",
        output_root=tmp_path,
        run_id="run",
        step=0,
        model="base",
        adapter_checkpoint=None,
        deadline_monotonic=None,
    )
    source_identity = _source_identity(tmp_path)
    summary = SynthesisSummary(
        rank=0,
        expected=250,
        completed=250,
        generated=250,
        skipped=0,
        failed=0,
        complete=True,
        stop_reason=None,
        source_identity=source_identity,
    )

    code = _run_synth(
        args,
        assignment_loader=lambda path: [],
        context_resolver=lambda: DistributedContext(0, 0, 8),
        distributed_initializer=lambda context: None,
        source_resolver=lambda **kwargs: source_identity,
        model_loader=lambda **kwargs: object(),
        synthesizer=lambda **kwargs: summary,
        synchronizer=lambda: False,
        cuda_releaser=lambda: None,
    )

    assert code == INCOMPLETE_EXIT_CODE
    output = json.loads(capsys.readouterr().out)
    assert output["complete"] is False
    assert output["synchronized"] is False


def test_run_synth_releases_model_and_synchronizes_after_fatal_error(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        assignments=tmp_path / "assignments.jsonl",
        output_root=tmp_path,
        run_id="run",
        step=0,
        model="base",
        adapter_checkpoint=None,
        deadline_monotonic=None,
    )
    events: list[str] = []
    source_identity = _source_identity(tmp_path)

    with pytest.raises(RuntimeError, match="fatal synthesis error"):
        _run_synth(
            args,
            assignment_loader=lambda path: [],
            context_resolver=lambda: DistributedContext(0, 0, 8),
            distributed_initializer=lambda context: None,
            source_resolver=lambda **kwargs: source_identity,
            model_loader=lambda **kwargs: object(),
            synthesizer=lambda **kwargs: (_ for _ in ()).throw(
                RuntimeError("fatal synthesis error")
            ),
            synchronizer=lambda: events.append("synchronize") or True,
            cuda_releaser=lambda: events.append("release"),
        )

    assert events == ["synchronize", "release"]


def test_cleanup_failure_does_not_suppress_primary_synthesis_error(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        assignments=tmp_path / "assignments.jsonl",
        output_root=tmp_path,
        run_id="run",
        step=0,
        model="base",
        adapter_checkpoint=None,
        deadline_monotonic=None,
    )
    source_identity = _source_identity(tmp_path)
    released: list[bool] = []

    with pytest.raises(ValueError, match="primary synthesis failure") as exc_info:
        _run_synth(
            args,
            assignment_loader=lambda path: [],
            context_resolver=lambda: DistributedContext(0, 0, 8),
            source_resolver=lambda **kwargs: source_identity,
            distributed_initializer=lambda context: None,
            model_loader=lambda **kwargs: object(),
            synthesizer=lambda **kwargs: (_ for _ in ()).throw(
                ValueError("primary synthesis failure")
            ),
            synchronizer=lambda: (_ for _ in ()).throw(
                RuntimeError("secondary barrier failure")
            ),
            cuda_releaser=lambda: released.append(True),
        )

    assert released == [True]
    assert any(
        "secondary barrier failure" in note
        for note in getattr(exc_info.value, "__notes__", [])
    )


def test_sigterm_during_load_generation_sync_and_release_stays_incomplete(
    tmp_path: Path,
) -> None:
    child = r"""
import os
import signal
from pathlib import Path
from types import SimpleNamespace

from omnivoice.cli.validate_hard_numbers import _run_synth
from omnivoice.validation.synthesis import (
    DistributedContext,
    ModelSourceIdentity,
    SynthesisSummary,
)

root = Path(os.environ["SYNTH_SIGNAL_ROOT"])
for phase in ("load", "generation", "sync", "release"):
    phase_root = root / phase
    args = SimpleNamespace(
        assignments=phase_root / "assignments.jsonl",
        output_root=phase_root,
        run_id="run",
        step=0,
        model="base",
        adapter_checkpoint=None,
        deadline_monotonic=None,
    )

    def send_if(selected):
        if phase == selected:
            os.kill(os.getpid(), signal.SIGTERM)

    source_path = phase_root / "immutable-source"
    source_path.mkdir(parents=True, exist_ok=True)
    source_identity = ModelSourceIdentity(
        kind="base",
        requested="base",
        load_path=str(source_path.resolve()),
        immutable_id="hf:" + "a" * 40,
    )

    def load(**kwargs):
        del kwargs
        send_if("load")
        return object()

    def synthesize(**kwargs):
        send_if("generation")
        return SynthesisSummary(
            rank=0,
            expected=250,
            completed=250,
            generated=250,
            skipped=0,
            failed=0,
            complete=True,
            stop_reason=None,
            source_identity=source_identity,
        )

    def synchronize():
        send_if("sync")
        return True

    def release():
        send_if("release")
        (phase_root / "released").write_text("yes", encoding="utf-8")

    code = _run_synth(
        args,
        assignment_loader=lambda path: [],
        context_resolver=lambda: DistributedContext(0, 0, 8),
        distributed_initializer=lambda context: None,
        source_resolver=lambda **kwargs: source_identity,
        model_loader=load,
        synthesizer=synthesize,
        synchronizer=synchronize,
        cuda_releaser=release,
    )
    print(f"EXIT:{phase}:{code}")
"""
    environment = {**__import__("os").environ, "SYNTH_SIGNAL_ROOT": str(tmp_path)}

    result = subprocess.run(
        [sys.executable, "-c", child],
        cwd=Path.cwd(),
        env=environment,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert [
        line for line in result.stdout.splitlines() if line.startswith("EXIT:")
    ] == [
        "EXIT:load:2",
        "EXIT:generation:2",
        "EXIT:sync:2",
        "EXIT:release:2",
    ]
    json_lines = [
        json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")
    ]
    assert len(json_lines) == 4
    assert all(row["complete"] is False for row in json_lines)
    assert all(row["stop_reason"] == "signal" for row in json_lines)
    for phase in ("load", "generation", "sync", "release"):
        assert (tmp_path / phase / "released").read_text(encoding="utf-8") == "yes"
        durable = (
            tmp_path
            / phase
            / "run"
            / "step-0"
            / "rank-manifests"
            / "rank-0.summary.json"
        )
        assert json.loads(durable.read_text(encoding="utf-8"))["complete"] is False


@pytest.mark.parametrize(
    ("signal_phase", "expected_complete", "expected_code"),
    [
        ("before_serialization", False, INCOMPLETE_EXIT_CODE),
        ("output", True, 0),
    ],
)
def test_final_payload_is_one_atomic_sigterm_boundary(
    tmp_path: Path,
    signal_phase: str,
    expected_complete: bool,
    expected_code: int,
) -> None:
    args = SimpleNamespace(
        assignments=tmp_path / "assignments.jsonl",
        output_root=tmp_path,
        run_id="run",
        step=0,
        model="base",
        adapter_checkpoint=None,
        deadline_monotonic=None,
    )
    source_identity = _source_identity(tmp_path)
    summary = SynthesisSummary(
        rank=0,
        expected=250,
        completed=250,
        generated=250,
        skipped=0,
        failed=0,
        complete=True,
        stop_reason=None,
        source_identity=source_identity,
    )
    emitted: list[str] = []
    serialized = 0
    previous_handler = signal.getsignal(signal.SIGTERM)
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())

    def serializer(payload, **kwargs):
        nonlocal serialized
        serialized += 1
        return json.dumps(payload, **kwargs)

    def output(payload: str) -> None:
        emitted.append(payload)
        if signal_phase == "output" and len(emitted) == 1:
            os.kill(os.getpid(), signal.SIGTERM)

    code = _run_synth(
        args,
        assignment_loader=lambda path: [],
        context_resolver=lambda: DistributedContext(0, 0, 8),
        source_resolver=lambda **kwargs: source_identity,
        distributed_initializer=lambda context: None,
        model_loader=lambda **kwargs: object(),
        synthesizer=lambda **kwargs: summary,
        synchronizer=lambda: (
            os.kill(os.getpid(), signal.SIGTERM) or True
            if signal_phase == "before_serialization"
            else True
        ),
        cuda_releaser=lambda: None,
        serializer=serializer,
        output=output,
    )

    assert code == expected_code
    assert serialized == 1
    assert len(emitted) == 1
    assert json.loads(emitted[0])["complete"] is expected_complete
    assert signal.getsignal(signal.SIGTERM) is previous_handler
    assert signal.pthread_sigmask(signal.SIG_BLOCK, set()) == previous_mask
    if signal_phase == "before_serialization":
        assert json.loads(emitted[0])["stop_reason"] == "signal"
        durable = tmp_path / "run" / "step-0" / "rank-manifests" / "rank-0.summary.json"
        assert json.loads(durable.read_text(encoding="utf-8"))["complete"] is False


def test_unresolved_source_fails_before_model_loader_or_distributed_gpu_work(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        assignments=tmp_path / "assignments.jsonl",
        output_root=tmp_path,
        run_id="run",
        step=0,
        model="base",
        adapter_checkpoint=None,
        deadline_monotonic=None,
    )
    calls: list[str] = []

    with pytest.raises(ValueError, match="unresolved source"):
        _run_synth(
            args,
            assignment_loader=lambda path: [],
            context_resolver=lambda: DistributedContext(0, 0, 8),
            source_resolver=lambda **kwargs: (_ for _ in ()).throw(
                ValueError("unresolved source")
            ),
            distributed_initializer=lambda context: calls.append("distributed"),
            model_loader=lambda **kwargs: calls.append("model") or object(),
            synthesizer=lambda **kwargs: None,
            synchronizer=lambda: True,
            cuda_releaser=lambda: calls.append("release"),
        )

    assert calls == ["release"]


@pytest.mark.parametrize(
    ("stage", "expected_reason"),
    [
        ("synchronization", "synchronization"),
        ("destructor", "cleanup"),
        ("cleanup", "cleanup"),
    ],
)
def test_blocking_lifecycle_cleanup_is_bounded_and_corrects_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    stage: str,
    expected_reason: str,
) -> None:
    args = SimpleNamespace(
        assignments=tmp_path / "assignments.jsonl",
        output_root=tmp_path,
        run_id="run",
        step=0,
        model="base",
        adapter_checkpoint=None,
        deadline_monotonic=None,
    )
    source_identity = _source_identity(tmp_path)
    summary = SynthesisSummary(
        rank=0,
        expected=250,
        completed=250,
        generated=250,
        skipped=0,
        failed=0,
        complete=True,
        stop_reason=None,
        source_identity=source_identity,
    )
    release = threading.Event()
    timer = threading.Timer(0.5, release.set)

    marker: list[str] = []

    def block() -> bool:
        release.wait()
        marker.append("abandoned-worker-continued")
        return True

    class BlockingDestructor:
        def __del__(self):
            block()

    class HardExit(BaseException):
        def __init__(self, code: int) -> None:
            self.code = code

    def hard_exit(code: int) -> None:
        raise HardExit(code)

    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(HardExit) as exc_info:
            _run_synth(
                args,
                assignment_loader=lambda path: [],
                context_resolver=lambda: DistributedContext(0, 0, 8),
                source_resolver=lambda **kwargs: source_identity,
                distributed_initializer=lambda context: None,
                model_loader=lambda **kwargs: (
                    BlockingDestructor() if stage == "destructor" else object()
                ),
                synthesizer=lambda **kwargs: summary,
                synchronizer=block if stage == "synchronization" else lambda: True,
                cuda_releaser=(
                    block
                    if stage == "cleanup"
                    else lambda: marker.append("release-called")
                ),
                cleanup_timeout_seconds=0.02,
                hard_exit=hard_exit,
            )
        assert exc_info.value.code == INCOMPLETE_EXIT_CODE
        assert time.monotonic() - started < 0.2
        assert marker == []
    finally:
        release.set()
        timer.cancel()

    durable = json.loads(
        (
            tmp_path / "run" / "step-0" / "rank-manifests" / "rank-0.summary.json"
        ).read_text(encoding="utf-8")
    )
    assert durable["complete"] is False
    assert durable["stop_reason"] == expected_reason
    assert durable["error"]["stage"] == expected_reason
    assert durable["error"]["type"] == "TimeoutError"
    payloads = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("{")
    ]
    assert len(payloads) == 1
    assert payloads[0]["complete"] is False


@pytest.mark.parametrize(
    "stage",
    [
        "synchronization",
        "destructor",
        "cleanup",
        "error_destructor",
        "error_sync",
        "error_cause_destructor",
        "partial_loader",
    ],
)
def test_real_rank_process_hard_exits_with_no_abandoned_worker_progress(
    tmp_path: Path, stage: str
) -> None:
    child = r"""
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from omnivoice.cli.validate_hard_numbers import _run_synth
from omnivoice.validation.synthesis import (
    DistributedContext,
    ModelSourceIdentity,
    SynthesisSummary,
)

root = Path(os.environ["SYNTH_HARD_EXIT_ROOT"])
stage = os.environ["SYNTH_HARD_EXIT_STAGE"]
source_path = root / "immutable-source"
source_path.mkdir(parents=True, exist_ok=True)
source_identity = ModelSourceIdentity(
    kind="base",
    requested="base",
    load_path=str(source_path.resolve()),
    immutable_id="hf:" + "a" * 40,
)
summary = SynthesisSummary(
    rank=0,
    expected=250,
    completed=250,
    generated=250,
    skipped=0,
    failed=0,
    complete=True,
    stop_reason=None,
    source_identity=source_identity,
)

def block():
    time.sleep(0.5)
    (root / "abandoned-worker-marker").write_text(stage, encoding="utf-8")
    return True

class BlockingDestructor:
    def __del__(self):
        block()

def synthesize(**kwargs):
    if stage == "error_cause_destructor":
        def raise_nested(model):
            try:
                raise KeyError("nested cause")
            except KeyError as cause:
                assert model is not None
                raise ValueError("primary caused synthesis exploded") from cause

        raise_nested(kwargs["model"])
    if stage.startswith("error_"):
        assert kwargs["model"] is not None
        raise ValueError("primary synthesis exploded")
    return summary

def load_model(**kwargs):
    del kwargs
    if stage == "partial_loader":
        partial_model = BlockingDestructor()
        assert partial_model is not None
        raise RuntimeError("partial loader exploded")
    if stage in {"destructor", "error_destructor", "error_cause_destructor"}:
        return BlockingDestructor()
    return object()

args = SimpleNamespace(
    assignments=root / "assignments.jsonl",
    output_root=root,
    run_id="run",
    step=0,
    model="base",
    adapter_checkpoint=None,
    deadline_monotonic=None,
)
(root / "ready").write_text("yes", encoding="utf-8")
_run_synth(
    args,
    assignment_loader=lambda path: [],
    context_resolver=lambda: DistributedContext(0, 0, 8),
    source_resolver=lambda **kwargs: source_identity,
    distributed_initializer=lambda context: None,
    model_loader=load_model,
    synthesizer=synthesize,
    synchronizer=(
        block if stage in {"synchronization", "error_sync"} else lambda: True
    ),
    cuda_releaser=block if stage == "cleanup" else lambda: None,
    cleanup_timeout_seconds=0.05,
)
(root / "after-hard-exit").write_text("unsafe", encoding="utf-8")
"""
    environment = {
        **os.environ,
        "SYNTH_HARD_EXIT_ROOT": str(tmp_path),
        "SYNTH_HARD_EXIT_STAGE": stage,
    }
    process = subprocess.Popen(
        [sys.executable, "-c", child],
        cwd=Path.cwd(),
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    ready_deadline = time.monotonic() + 120.0
    while (
        not (tmp_path / "ready").exists()
        and process.poll() is None
        and time.monotonic() < ready_deadline
    ):
        time.sleep(0.01)
    assert (tmp_path / "ready").is_file()

    started = time.monotonic()
    stdout, stderr = process.communicate(timeout=2.0)
    assert time.monotonic() - started < 0.5
    assert process.returncode == INCOMPLETE_EXIT_CODE, stderr
    payloads = [
        json.loads(line) for line in stdout.splitlines() if line.startswith("{")
    ]
    assert len(payloads) == 1
    assert payloads[0]["complete"] is False
    assert payloads[0]["stop_reason"] == (
        "synchronization" if stage in {"synchronization", "error_sync"} else "cleanup"
    )
    if stage.startswith("error_") or stage == "partial_loader":
        expected_primary = (
            {
                "stage": "model_load",
                "type": "RuntimeError",
                "message": "partial loader exploded",
            }
            if stage == "partial_loader"
            else {
                "stage": "synthesis",
                "type": "ValueError",
                "message": (
                    "primary caused synthesis exploded"
                    if stage == "error_cause_destructor"
                    else "primary synthesis exploded"
                ),
            }
        )
        assert payloads[0]["primary_error"] == expected_primary
        assert payloads[0]["cleanup_error"]["type"] == "TimeoutError"
        assert payloads[0]["completed"] is None
    durable = json.loads(
        (
            tmp_path / "run" / "step-0" / "rank-manifests" / "rank-0.summary.json"
        ).read_text(encoding="utf-8")
    )
    assert durable["complete"] is False
    time.sleep(0.55)
    assert not (tmp_path / "abandoned-worker-marker").exists()
    assert not (tmp_path / "after-hard-exit").exists()


@pytest.mark.parametrize(
    ("stage", "error"),
    [
        ("synchronization", ValueError("sync exploded")),
        ("cleanup", OSError("release exploded")),
    ],
)
def test_lifecycle_error_corrects_durable_summary_before_propagation(
    tmp_path: Path, stage: str, error: BaseException
) -> None:
    args = SimpleNamespace(
        assignments=tmp_path / "assignments.jsonl",
        output_root=tmp_path,
        run_id="run",
        step=0,
        model="base",
        adapter_checkpoint=None,
        deadline_monotonic=None,
    )
    source_identity = _source_identity(tmp_path)
    summary = SynthesisSummary(
        rank=0,
        expected=250,
        completed=250,
        generated=250,
        skipped=0,
        failed=0,
        complete=True,
        stop_reason=None,
        source_identity=source_identity,
    )

    with pytest.raises(type(error), match=str(error)):
        _run_synth(
            args,
            assignment_loader=lambda path: [],
            context_resolver=lambda: DistributedContext(0, 0, 8),
            source_resolver=lambda **kwargs: source_identity,
            distributed_initializer=lambda context: None,
            model_loader=lambda **kwargs: object(),
            synthesizer=lambda **kwargs: summary,
            synchronizer=(
                (lambda: (_ for _ in ()).throw(error))
                if stage == "synchronization"
                else lambda: True
            ),
            cuda_releaser=(
                (lambda: (_ for _ in ()).throw(error))
                if stage == "cleanup"
                else lambda: None
            ),
        )

    durable = json.loads(
        (
            tmp_path / "run" / "step-0" / "rank-manifests" / "rank-0.summary.json"
        ).read_text(encoding="utf-8")
    )
    assert durable["complete"] is False
    assert durable["stop_reason"] == stage
    assert durable["error"] == {
        "stage": stage,
        "type": type(error).__name__,
        "message": str(error),
    }


def test_primary_error_still_corrects_summary_written_before_raise(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        assignments=tmp_path / "assignments.jsonl",
        output_root=tmp_path,
        run_id="run",
        step=0,
        model="base",
        adapter_checkpoint=None,
        deadline_monotonic=None,
    )
    source_identity = _source_identity(tmp_path)
    summary = SynthesisSummary(
        rank=0,
        expected=250,
        completed=250,
        generated=250,
        skipped=0,
        failed=0,
        complete=True,
        stop_reason=None,
        source_identity=source_identity,
    )

    def synthesize(**kwargs):
        write_synthesis_summary(kwargs["output_dir"], summary)
        raise RuntimeError("primary exploded after durable summary")

    with pytest.raises(RuntimeError, match="primary exploded") as exc_info:
        _run_synth(
            args,
            assignment_loader=lambda path: [],
            context_resolver=lambda: DistributedContext(0, 0, 8),
            source_resolver=lambda **kwargs: source_identity,
            distributed_initializer=lambda context: None,
            model_loader=lambda **kwargs: object(),
            synthesizer=synthesize,
            synchronizer=lambda: (_ for _ in ()).throw(
                ValueError("sync also exploded")
            ),
            cuda_releaser=lambda: None,
        )

    durable = json.loads(
        (
            tmp_path / "run" / "step-0" / "rank-manifests" / "rank-0.summary.json"
        ).read_text(encoding="utf-8")
    )
    assert durable["complete"] is False
    assert durable["stop_reason"] == "synchronization"
    assert durable["error"]["type"] == "ValueError"
    assert any(
        "sync also exploded" in note
        for note in getattr(exc_info.value, "__notes__", [])
    )


def test_primary_traceback_releases_model_inside_bounded_cleanup(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        assignments=tmp_path / "assignments.jsonl",
        output_root=tmp_path,
        run_id="run",
        step=7,
        model="base",
        adapter_checkpoint=None,
        deadline_monotonic=None,
    )
    source_identity = _source_identity(tmp_path)
    destroyed = threading.Event()
    destroyed_when_released: list[bool] = []

    class TrackedModel:
        def __del__(self):
            destroyed.set()

    def synthesize_with_kwargs(**kwargs):
        assert isinstance(kwargs["model"], TrackedModel)
        raise ValueError("primary retains kwargs")

    def release() -> None:
        destroyed_when_released.append(destroyed.is_set())

    with pytest.raises(ValueError, match="primary retains kwargs") as exc_info:
        _run_synth(
            args,
            assignment_loader=lambda path: [],
            context_resolver=lambda: DistributedContext(0, 0, 8),
            source_resolver=lambda **kwargs: source_identity,
            distributed_initializer=lambda context: None,
            model_loader=lambda **kwargs: TrackedModel(),
            synthesizer=synthesize_with_kwargs,
            synchronizer=lambda: (_ for _ in ()).throw(
                RuntimeError("secondary synchronization failure")
            ),
            cuda_releaser=release,
        )

    assert destroyed_when_released == [True]
    traceback_names: list[str] = []
    traceback = exc_info.value.__traceback__
    while traceback is not None:
        traceback_names.append(traceback.tb_frame.f_code.co_name)
        traceback = traceback.tb_next
    assert "synthesize_with_kwargs" in traceback_names
    assert any(
        "secondary synchronization failure" in note
        for note in getattr(exc_info.value, "__notes__", [])
    )


@pytest.mark.parametrize("graph_kind", ["cause", "context", "group", "cycle"])
def test_nested_primary_exception_graph_releases_model_in_cleanup_worker(
    tmp_path: Path, graph_kind: str
) -> None:
    args = SimpleNamespace(
        assignments=tmp_path / "assignments.jsonl",
        output_root=tmp_path,
        run_id="run",
        step=8,
        model="base",
        adapter_checkpoint=None,
        deadline_monotonic=None,
    )
    source_identity = _source_identity(tmp_path)
    destructor_threads: list[int] = []
    release_threads: list[int] = []

    class TrackedModel:
        def __del__(self):
            destructor_threads.append(threading.get_ident())

    def nested_helper(model) -> None:
        assert isinstance(model, TrackedModel)
        raise KeyError("nested retains model")

    def synthesize_nested(**kwargs):
        try:
            nested_helper(kwargs["model"])
        except KeyError as nested:
            if graph_kind == "cause":
                raise ValueError("outer cause") from nested
            if graph_kind == "context":
                raise ValueError("outer context")
            if graph_kind == "group":
                raise _EXCEPTION_GROUP("outer group", [nested])
            outer = ValueError("outer cycle")
            outer.__cause__ = nested
            nested.__cause__ = outer
            raise outer

    def release() -> None:
        release_threads.append(threading.get_ident())

    expected_type = _EXCEPTION_GROUP if graph_kind == "group" else ValueError
    with pytest.raises(expected_type) as exc_info:
        _run_synth(
            args,
            assignment_loader=lambda path: [],
            context_resolver=lambda: DistributedContext(0, 0, 8),
            source_resolver=lambda **kwargs: source_identity,
            distributed_initializer=lambda context: None,
            model_loader=lambda **kwargs: TrackedModel(),
            synthesizer=synthesize_nested,
            synchronizer=lambda: True,
            cuda_releaser=release,
        )

    assert len(destructor_threads) == 1
    assert destructor_threads == release_threads
    assert destructor_threads[0] != threading.get_ident()
    traceback_names: list[str] = []
    traceback = exc_info.value.__traceback__
    while traceback is not None:
        traceback_names.append(traceback.tb_frame.f_code.co_name)
        traceback = traceback.tb_next
    assert "synthesize_nested" in traceback_names
    if graph_kind == "group":
        nested_error = exc_info.value.exceptions[0]
    elif graph_kind == "context":
        nested_error = exc_info.value.__context__
    else:
        nested_error = exc_info.value.__cause__
    assert isinstance(nested_error, KeyError)
    if graph_kind == "cycle":
        assert nested_error.__cause__ is exc_info.value
    nested_traceback_names: list[str] = []
    nested_traceback = nested_error.__traceback__
    while nested_traceback is not None:
        nested_traceback_names.append(nested_traceback.tb_frame.f_code.co_name)
        nested_traceback = nested_traceback.tb_next
    assert "nested_helper" in nested_traceback_names


def test_partial_model_loader_failure_destroys_model_in_cleanup_worker(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        assignments=tmp_path / "assignments.jsonl",
        output_root=tmp_path,
        run_id="run",
        step=9,
        model="base",
        adapter_checkpoint=None,
        deadline_monotonic=None,
    )
    source_identity = _source_identity(tmp_path)
    destructor_threads: list[int] = []
    release_threads: list[int] = []

    class PartialModel:
        def __del__(self):
            destructor_threads.append(threading.get_ident())

    def failing_loader(**kwargs):
        del kwargs
        partial_model = PartialModel()
        assert partial_model is not None
        raise RuntimeError("partial loader exploded")

    def release() -> None:
        release_threads.append(threading.get_ident())

    with pytest.raises(RuntimeError, match="partial loader exploded") as exc_info:
        _run_synth(
            args,
            assignment_loader=lambda path: [],
            context_resolver=lambda: DistributedContext(0, 0, 8),
            source_resolver=lambda **kwargs: source_identity,
            distributed_initializer=lambda context: None,
            model_loader=failing_loader,
            synthesizer=lambda **kwargs: None,
            synchronizer=lambda: True,
            cuda_releaser=release,
        )

    assert len(destructor_threads) == 1
    assert destructor_threads == release_threads
    assert destructor_threads[0] != threading.get_ident()
    traceback_names: list[str] = []
    traceback = exc_info.value.__traceback__
    while traceback is not None:
        traceback_names.append(traceback.tb_frame.f_code.co_name)
        traceback = traceback.tb_next
    assert "failing_loader" in traceback_names


def test_primary_without_summary_is_published_before_sync_timeout(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        assignments=tmp_path / "assignments.jsonl",
        output_root=tmp_path,
        run_id="run-failure",
        step=13,
        model="base",
        adapter_checkpoint=None,
        deadline_monotonic=None,
    )
    source_identity = _source_identity(tmp_path)
    release = threading.Event()
    emitted: list[str] = []

    class HardExit(BaseException):
        def __init__(self, code: int) -> None:
            self.code = code

    def hard_exit(code: int) -> None:
        raise HardExit(code)

    def block() -> bool:
        release.wait()
        return True

    try:
        with pytest.raises(HardExit) as exc_info:
            _run_synth(
                args,
                assignment_loader=lambda path: [],
                context_resolver=lambda: DistributedContext(0, 0, 8),
                source_resolver=lambda **kwargs: source_identity,
                distributed_initializer=lambda context: None,
                model_loader=lambda **kwargs: object(),
                synthesizer=lambda **kwargs: (_ for _ in ()).throw(
                    ValueError("primary synthesis exploded")
                ),
                synchronizer=block,
                cuda_releaser=lambda: (_ for _ in ()).throw(
                    AssertionError("release must not start after sync timeout")
                ),
                cleanup_timeout_seconds=0.02,
                hard_exit=hard_exit,
                output=emitted.append,
            )
        assert exc_info.value.code == INCOMPLETE_EXIT_CODE
    finally:
        release.set()

    assert len(emitted) == 1
    payload = json.loads(emitted[0])
    assert payload["complete"] is False
    assert payload["run_id"] == "run-failure"
    assert payload["rank"] == 0
    assert payload["step"] == 13
    assert payload["source_identity"] == asdict(source_identity)
    assert payload["expected"] == 0
    assert payload["completed"] is None
    assert payload["stop_reason"] == "synchronization"
    assert payload["primary_error"] == {
        "stage": "synthesis",
        "type": "ValueError",
        "message": "primary synthesis exploded",
    }
    assert payload["cleanup_error"]["stage"] == "synchronization"
    assert payload["cleanup_error"]["type"] == "TimeoutError"
    durable = json.loads(
        (
            tmp_path
            / "run-failure"
            / "step-13"
            / "rank-manifests"
            / "rank-0.summary.json"
        ).read_text(encoding="utf-8")
    )
    assert durable == {
        key: value for key, value in payload.items() if key not in {"synchronized"}
    }
    restored = read_synthesis_summary(tmp_path / "run-failure" / "step-13", rank=0)
    assert asdict(restored) == durable


def test_stale_source_summary_cannot_claim_coverage_after_current_timeout(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        assignments=tmp_path / "assignments.jsonl",
        output_root=tmp_path,
        run_id="run-source-b",
        step=21,
        model="base-b",
        adapter_checkpoint=None,
        deadline_monotonic=None,
    )
    source_a = _source_identity(
        tmp_path / "source-a", requested="base-a", commit="a" * 40
    )
    source_b = _source_identity(
        tmp_path / "source-b", requested="base-b", commit="b" * 40
    )
    prior = SynthesisSummary(
        rank=0,
        expected=250,
        completed=250,
        generated=250,
        skipped=0,
        failed=0,
        complete=True,
        stop_reason=None,
        source_identity=source_a,
        run_id=args.run_id,
        step=args.step,
    )
    write_synthesis_summary(tmp_path / args.run_id / "step-21", prior)
    release = threading.Event()
    emitted: list[str] = []

    class HardExit(BaseException):
        pass

    try:
        with pytest.raises(HardExit):
            _run_synth(
                args,
                assignment_loader=lambda path: [object()] * 2_000,
                context_resolver=lambda: DistributedContext(0, 0, 8),
                source_resolver=lambda **kwargs: source_b,
                distributed_initializer=lambda context: None,
                model_loader=lambda **kwargs: (_ for _ in ()).throw(
                    ValueError("base-b load exploded")
                ),
                synthesizer=lambda **kwargs: None,
                synchronizer=lambda: release.wait() or True,
                cuda_releaser=lambda: (_ for _ in ()).throw(
                    AssertionError("release must not run after sync timeout")
                ),
                cleanup_timeout_seconds=0.02,
                hard_exit=lambda code: (_ for _ in ()).throw(HardExit(code)),
                output=emitted.append,
            )
    finally:
        release.set()

    assert len(emitted) == 1
    payload = json.loads(emitted[0])
    assert payload["source_identity"] == asdict(source_b)
    assert payload["expected"] == 250
    assert payload["completed"] is None
    assert payload["generated"] is None
    assert payload["skipped"] is None
    assert payload["failed"] is None
    assert payload["complete"] is False
    assert payload["primary_error"] == {
        "stage": "model_load",
        "type": "ValueError",
        "message": "base-b load exploded",
    }
    assert payload["cleanup_error"]["stage"] == "synchronization"
    durable = read_synthesis_summary(tmp_path / "run-source-b" / "step-21", rank=0)
    assert durable.source_identity == source_b
    assert durable.completed is None
    assert durable.primary_error.message == "base-b load exploded"
    assert durable.cleanup_error.stage == "synchronization"


def test_exact_same_source_prior_summary_reuses_known_counts(tmp_path: Path) -> None:
    args = SimpleNamespace(
        assignments=tmp_path / "assignments.jsonl",
        output_root=tmp_path,
        run_id="same-source",
        step=22,
        model="base",
        adapter_checkpoint=None,
        deadline_monotonic=None,
    )
    source = _source_identity(tmp_path)
    prior = SynthesisSummary(
        rank=0,
        expected=250,
        completed=123,
        generated=123,
        skipped=0,
        failed=0,
        complete=False,
        stop_reason="deadline",
        source_identity=source,
        run_id=args.run_id,
        step=args.step,
    )
    write_synthesis_summary(tmp_path / args.run_id / "step-22", prior)

    with pytest.raises(ValueError, match="same source load exploded"):
        _run_synth(
            args,
            assignment_loader=lambda path: [object()] * 2_000,
            context_resolver=lambda: DistributedContext(0, 0, 8),
            source_resolver=lambda **kwargs: source,
            distributed_initializer=lambda context: None,
            model_loader=lambda **kwargs: (_ for _ in ()).throw(
                ValueError("same source load exploded")
            ),
            synthesizer=lambda **kwargs: None,
            synchronizer=lambda: True,
            cuda_releaser=lambda: None,
        )

    durable = read_synthesis_summary(tmp_path / "same-source" / "step-22", rank=0)
    assert (durable.expected, durable.completed, durable.generated) == (250, 123, 123)
    assert durable.source_identity == source
    assert durable.primary_error.message == "same source load exploded"


@pytest.mark.parametrize(
    ("expected", "completed", "generated", "skipped", "failed"),
    [
        pytest.param(250, None, 999, 999, 999, id="unknown-completed-with-counts"),
        pytest.param(None, 999, None, None, None, id="unknown-expected-with-count"),
        pytest.param(250, 249, 249, 0, 250, id="completed-plus-failed"),
        pytest.param(250, 100, None, 100, 0, id="mixed-null-and-known"),
        pytest.param(-1, None, None, None, None, id="negative-expected"),
        pytest.param(250, 0, 0, 0, -1, id="negative-failed"),
        pytest.param(250, True, True, 0, 0, id="boolean-count"),
        pytest.param(True, None, None, None, None, id="boolean-expected"),
        pytest.param(250, 250, 250, 0, 251, id="over-expected-failed"),
    ],
)
def test_impossible_summary_counts_are_not_reused_for_current_fallback(
    tmp_path: Path,
    expected: int | None,
    completed: int | None,
    generated: int | None,
    skipped: int | None,
    failed: int | None,
) -> None:
    args = SimpleNamespace(
        assignments=tmp_path / "assignments.jsonl",
        output_root=tmp_path,
        run_id="current-counts",
        step=24,
        model="base",
        adapter_checkpoint=None,
        deadline_monotonic=None,
    )
    source = _source_identity(tmp_path)
    prior = SynthesisSummary(
        rank=0,
        expected=expected,
        completed=completed,
        generated=generated,
        skipped=skipped,
        failed=failed,
        complete=False,
        stop_reason="deadline",
        source_identity=source,
        run_id=args.run_id,
        step=args.step,
    )
    write_synthesis_summary(tmp_path / args.run_id / "step-24", prior)

    with pytest.raises(ValueError, match="current load exploded"):
        _run_synth(
            args,
            assignment_loader=lambda path: [object()] * 2_000,
            context_resolver=lambda: DistributedContext(0, 0, 8),
            source_resolver=lambda **kwargs: source,
            distributed_initializer=lambda context: None,
            model_loader=lambda **kwargs: (_ for _ in ()).throw(
                ValueError("current load exploded")
            ),
            synthesizer=lambda **kwargs: None,
            synchronizer=lambda: True,
            cuda_releaser=lambda: None,
        )

    durable = read_synthesis_summary(tmp_path / "current-counts" / "step-24", rank=0)
    assert (
        durable.expected,
        durable.completed,
        durable.generated,
        durable.skipped,
        durable.failed,
    ) == (250, None, None, None, None)
    assert durable.primary_error.message == "current load exploded"


@pytest.mark.parametrize("stale_field", ["run_id", "step", "rank", "malformed"])
def test_stale_or_malformed_summary_is_ignored_for_current_fallback(
    tmp_path: Path, stale_field: str
) -> None:
    args = SimpleNamespace(
        assignments=tmp_path / "assignments.jsonl",
        output_root=tmp_path,
        run_id="current-run",
        step=23,
        model="base",
        adapter_checkpoint=None,
        deadline_monotonic=None,
    )
    source = _source_identity(tmp_path)
    summary_path = (
        tmp_path / "current-run" / "step-23" / "rank-manifests" / "rank-0.summary.json"
    )
    summary_path.parent.mkdir(parents=True)
    if stale_field == "malformed":
        summary_path.write_text("{not-json\n", encoding="utf-8")
    else:
        values = asdict(
            SynthesisSummary(
                rank=0,
                expected=250,
                completed=250,
                generated=250,
                skipped=0,
                failed=0,
                complete=True,
                stop_reason=None,
                source_identity=source,
                run_id=args.run_id,
                step=args.step,
            )
        )
        values[stale_field] = {
            "run_id": "other-run",
            "step": 999,
            "rank": 1,
        }[stale_field]
        summary_path.write_text(json.dumps(values) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="current load exploded"):
        _run_synth(
            args,
            assignment_loader=lambda path: [object()] * 2_000,
            context_resolver=lambda: DistributedContext(0, 0, 8),
            source_resolver=lambda **kwargs: source,
            distributed_initializer=lambda context: None,
            model_loader=lambda **kwargs: (_ for _ in ()).throw(
                ValueError("current load exploded")
            ),
            synthesizer=lambda **kwargs: None,
            synchronizer=lambda: True,
            cuda_releaser=lambda: None,
        )

    durable = read_synthesis_summary(tmp_path / "current-run" / "step-23", rank=0)
    assert durable.run_id == args.run_id
    assert durable.step == args.step
    assert durable.rank == 0
    assert durable.source_identity == source
    assert durable.expected == 250
    assert durable.completed is None
    assert durable.cleanup_error is None
    assert durable.primary_error.message == "current load exploded"
