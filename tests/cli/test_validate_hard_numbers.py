from __future__ import annotations

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

import pytest

from omnivoice.cli.validate_hard_numbers import (
    INCOMPLETE_EXIT_CODE,
    _run_synth,
    build_parser,
)
from omnivoice.validation.synthesis import (
    DistributedContext,
    ModelSourceIdentity,
    SynthesisSummary,
    read_synthesis_summary,
    write_synthesis_summary,
)


def _source_identity(tmp_path: Path, *, requested: str = "base") -> ModelSourceIdentity:
    source = tmp_path / "immutable-source"
    source.mkdir(parents=True, exist_ok=True)
    return ModelSourceIdentity(
        kind="base",
        requested=requested,
        load_path=str(source.resolve()),
        immutable_id=f"hf:{'a' * 40}",
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
    assert calls["synthesizer"]["output_dir"] == tmp_path / "run-1" / "step-625"
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
    if stage.startswith("error_"):
        assert kwargs["model"] is not None
        raise ValueError("primary synthesis exploded")
    return summary

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
    model_loader=lambda **kwargs: (
        BlockingDestructor()
        if stage in {"destructor", "error_destructor"}
        else object()
    ),
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
    if stage.startswith("error_"):
        assert payloads[0]["primary_error"] == {
            "stage": "synthesis",
            "type": "ValueError",
            "message": "primary synthesis exploded",
        }
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
