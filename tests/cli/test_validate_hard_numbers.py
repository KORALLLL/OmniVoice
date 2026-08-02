from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from omnivoice.cli.validate_hard_numbers import (
    INCOMPLETE_EXIT_CODE,
    _run_synth,
    build_parser,
)
from omnivoice.validation.synthesis import DistributedContext, SynthesisSummary


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
        model_loader=model_loader,
        synthesizer=synthesizer,
        synchronizer=lambda: synchronized.append(True) or True,
        cuda_releaser=lambda: released.append(True),
    )

    assert code == INCOMPLETE_EXIT_CODE
    assert calls["loader"]["model_name"] == source_args["model"]
    assert calls["loader"]["adapter_checkpoint"] == source_args["adapter_checkpoint"]
    assert calls["loader"]["context"] == context
    assert calls["synthesizer"]["checkpoint"] == expected_source
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
    summary = SynthesisSummary(
        rank=0,
        expected=250,
        completed=250,
        generated=250,
        skipped=0,
        failed=0,
        complete=True,
        stop_reason=None,
    )

    code = _run_synth(
        args,
        assignment_loader=lambda path: [],
        context_resolver=lambda: DistributedContext(0, 0, 8),
        distributed_initializer=lambda context: None,
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

    with pytest.raises(RuntimeError, match="fatal synthesis error"):
        _run_synth(
            args,
            assignment_loader=lambda path: [],
            context_resolver=lambda: DistributedContext(0, 0, 8),
            distributed_initializer=lambda context: None,
            model_loader=lambda **kwargs: object(),
            synthesizer=lambda **kwargs: (_ for _ in ()).throw(
                RuntimeError("fatal synthesis error")
            ),
            synchronizer=lambda: events.append("synchronize") or True,
            cuda_releaser=lambda: events.append("release"),
        )

    assert events == ["synchronize", "release"]
