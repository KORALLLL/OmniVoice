from __future__ import annotations

import json
import subprocess
import sys
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
        ({"model": None, "adapter_checkpoint": Path("checkpoint-625")}, "checkpoint-625"),
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
        source_resolver=lambda **kwargs: calls.setdefault("source", kwargs)
        and source_identity,
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
    child = r'''
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
'''
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
    assert [line for line in result.stdout.splitlines() if line.startswith("EXIT:")] == [
        "EXIT:load:2",
        "EXIT:generation:2",
        "EXIT:sync:2",
        "EXIT:release:2",
    ]
    json_lines = [
        json.loads(line)
        for line in result.stdout.splitlines()
        if line.startswith("{")
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
