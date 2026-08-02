import json
import sys
from contextlib import nullcontext
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from omnivoice.cli import train as train_cli
from omnivoice.training import control as control_module
from omnivoice.training import trainer as trainer_module
from omnivoice.training.config import TrainingConfig
from omnivoice.training.control import (
    EvaluationStopPolicy,
    StopDecision,
    TrainingOutcome,
    append_loss_history,
)
from omnivoice.training.trainer import OmniTrainer


def test_two_consecutive_losses_are_required():
    policy = EvaluationStopPolicy(
        threshold=1e-4, patience=2, wall_limit_seconds=1200
    )

    assert not policy.observe(25, 9e-5, 10).stop
    assert not policy.observe(50, 2e-4, 20).stop
    assert not policy.observe(75, 8e-5, 30).stop
    decision = policy.observe(100, 7e-5, 40)

    assert decision.stop
    assert decision.reason == "eval_loss_target"
    assert decision.consecutive_hits == 2


def test_wall_clock_limit_is_observed_at_an_evaluation_boundary():
    policy = EvaluationStopPolicy(
        threshold=None, patience=1, wall_limit_seconds=1200
    )

    assert not policy.observe(25, 0.5, 1199.9).stop
    decision = policy.observe(50, 0.5, 1200)

    assert decision == StopDecision(
        stop=True, reason="wall_clock_limit", consecutive_hits=0
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"steps": 0}, "steps must be positive"),
        ({"steps_per_epoch": 0}, "steps_per_epoch must be positive"),
        ({"stop_after_step": 0}, "stop_after_step must be positive"),
        ({"steps": 10, "stop_after_step": 11}, "cannot exceed steps"),
        ({"max_wall_clock_seconds": 0}, "max_wall_clock_seconds must be positive"),
        ({"early_stop_eval_loss": 0}, "early_stop_eval_loss must be positive"),
        ({"early_stop_patience": 0}, "early_stop_patience must be positive"),
    ],
)
def test_training_control_config_rejects_invalid_bounds(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TrainingConfig(**kwargs).validate()


def test_append_loss_history_atomically_appends_jsonl(tmp_path, monkeypatch):
    history_path = tmp_path / "eval-history.jsonl"
    replacements = []
    real_replace = control_module.os.replace

    def record_replace(source, destination):
        replacements.append(
            (
                source.read_text(),
                destination,
                history_path.read_text() if history_path.exists() else None,
            )
        )
        real_replace(source, destination)

    monkeypatch.setattr(control_module.os, "replace", record_replace)

    append_loss_history(history_path, step=25, loss=0.125, elapsed_seconds=10.5)
    append_loss_history(history_path, step=50, loss=0.0625, elapsed_seconds=20.25)

    assert [json.loads(line) for line in history_path.read_text().splitlines()] == [
        {"step": 25, "loss": 0.125, "elapsed_seconds": 10.5},
        {"step": 50, "loss": 0.0625, "elapsed_seconds": 20.25},
    ]
    assert len(replacements) == 2
    assert replacements[0][2] is None
    assert json.loads(replacements[1][2]) == {
        "step": 25,
        "loss": 0.125,
        "elapsed_seconds": 10.5,
    }
    assert not list(tmp_path.glob(".eval-history.jsonl.*"))


class FakeAccelerator:
    def __init__(self, *, is_main_process=True, num_processes=1):
        self.device = torch.device("cpu")
        self.sync_gradients = True
        self.is_main_process = is_main_process
        self.is_local_main_process = is_main_process
        self.num_processes = num_processes
        self.ended = False

    def accumulate(self, model):
        return nullcontext()

    def backward(self, loss):
        pass

    def clip_grad_norm_(self, parameters, max_grad_norm):
        return torch.tensor(0.0)

    def gather(self, value):
        return value

    def log(self, metrics, step):
        pass

    def wait_for_everyone(self):
        pass

    def end_training(self):
        self.ended = True


class FakeModel:
    def __call__(self, **batch):
        return SimpleNamespace(loss=torch.tensor(0.5))

    def parameters(self):
        return []

    def train(self):
        pass


class FakeOptimizer:
    def step(self):
        pass

    def zero_grad(self):
        pass


class FakeScheduler:
    def step(self):
        pass

    def get_last_lr(self):
        return [1e-4]


class FakeTrainLogger:
    def __init__(self, accelerator, total_steps, logging_steps):
        pass

    def start(self, start_step=0):
        pass

    def update(self, step, loss=None, lr=None):
        pass

    def log_metrics(self, step, metrics):
        pass

    def close(self):
        pass


def make_bare_trainer(config, *, is_main_process=True, num_processes=1):
    class FakeDataLoader(list):
        dataset = SimpleNamespace()

    trainer = object.__new__(OmniTrainer)
    trainer.config = config
    trainer.model = FakeModel()
    trainer.tokenizer = None
    trainer.train_dataloader = FakeDataLoader(
        [{"input_ids": torch.tensor([[1]])}]
    )
    trainer.eval_dataloader = [{}]
    trainer.accelerator = FakeAccelerator(
        is_main_process=is_main_process, num_processes=num_processes
    )
    trainer.optimizer = FakeOptimizer()
    trainer.lr_scheduler = FakeScheduler()
    trainer.global_step = 0
    trainer.epoch = 0
    return trainer


def test_stop_after_step_runs_final_evaluation_and_returns_outcome(
    tmp_path, monkeypatch
):
    config = TrainingConfig(
        output_dir=str(tmp_path),
        steps=10,
        stop_after_step=3,
        eval_steps=2,
        save_steps=20,
        logging_steps=20,
    )
    trainer = make_bare_trainer(config)
    evaluated_steps = []
    saved_steps = []
    monkeypatch.setattr(trainer_module, "TrainLogger", FakeTrainLogger)
    monkeypatch.setattr(
        trainer,
        "evaluate",
        lambda: evaluated_steps.append(trainer.global_step)
        or {"eval/loss": 0.25},
    )
    monkeypatch.setattr(trainer, "save_checkpoint", saved_steps.append)

    outcome = trainer.train()

    assert outcome == TrainingOutcome(
        step=3,
        stop_reason="stop_after_step",
        last_eval_loss=0.25,
        target_reached=False,
    )
    assert evaluated_steps == [2, 3]
    assert saved_steps == [3]
    assert trainer.accelerator.ended


def test_eval_loss_target_stops_training_and_records_every_evaluation(
    tmp_path, monkeypatch
):
    history_path = tmp_path / "history.jsonl"
    config = TrainingConfig(
        output_dir=str(tmp_path),
        steps=10,
        eval_steps=1,
        save_steps=20,
        logging_steps=20,
        early_stop_eval_loss=0.2,
        early_stop_patience=2,
        eval_history_path=str(history_path),
    )
    trainer = make_bare_trainer(config)
    losses = iter([0.1, 0.3, 0.1, 0.05])
    saved_steps = []
    monkeypatch.setattr(trainer_module, "TrainLogger", FakeTrainLogger)
    monkeypatch.setattr(
        trainer, "evaluate", lambda: {"eval/loss": next(losses)}
    )
    monkeypatch.setattr(trainer, "save_checkpoint", saved_steps.append)

    outcome = trainer.train()

    assert outcome == TrainingOutcome(
        step=4,
        stop_reason="eval_loss_target",
        last_eval_loss=0.05,
        target_reached=True,
    )
    assert [
        (entry["step"], entry["loss"])
        for entry in map(json.loads, history_path.read_text().splitlines())
    ] == [(1, 0.1), (2, 0.3), (3, 0.1), (4, 0.05)]
    assert saved_steps == [4]


def test_wall_clock_limit_does_not_stop_between_evaluations(tmp_path, monkeypatch):
    config = TrainingConfig(
        output_dir=str(tmp_path),
        steps=10,
        eval_steps=3,
        save_steps=20,
        logging_steps=20,
        max_wall_clock_seconds=5,
    )
    trainer = make_bare_trainer(config)
    monotonic_times = iter([100.0, 106.0])
    monkeypatch.setattr(trainer_module, "TrainLogger", FakeTrainLogger)
    monkeypatch.setattr(trainer_module.time, "monotonic", lambda: next(monotonic_times))
    monkeypatch.setattr(trainer, "evaluate", lambda: {"eval/loss": 0.5})
    monkeypatch.setattr(trainer, "save_checkpoint", lambda step: None)

    outcome = trainer.train()

    assert outcome.step == 3
    assert outcome.stop_reason == "wall_clock_limit"


def test_non_main_rank_never_writes_evaluation_history(tmp_path, monkeypatch):
    history_path = tmp_path / "history.jsonl"
    config = TrainingConfig(
        output_dir=str(tmp_path),
        steps=10,
        stop_after_step=1,
        eval_steps=10,
        save_steps=20,
        logging_steps=20,
        eval_history_path=str(history_path),
    )
    trainer = make_bare_trainer(
        config, is_main_process=False, num_processes=2
    )
    monkeypatch.setattr(trainer_module, "TrainLogger", FakeTrainLogger)
    monkeypatch.setattr(trainer, "evaluate", lambda: {"eval/loss": 0.5})
    monkeypatch.setattr(trainer, "save_checkpoint", lambda step: None)
    monkeypatch.setattr(
        trainer_module,
        "broadcast_object_list",
        lambda payload: [(StopDecision(False, None, 0), None)],
    )

    outcome = trainer.train()

    assert outcome.stop_reason == "stop_after_step"
    assert not history_path.exists()


def test_single_process_evaluation_does_not_initialize_distributed_state(
    tmp_path, monkeypatch
):
    config = TrainingConfig(
        output_dir=str(tmp_path),
        steps=1,
        eval_steps=1,
        save_steps=20,
        logging_steps=20,
    )
    trainer = make_bare_trainer(config)
    monkeypatch.setattr(trainer_module, "TrainLogger", FakeTrainLogger)
    monkeypatch.setattr(trainer, "evaluate", lambda: {"eval/loss": 0.5})
    monkeypatch.setattr(trainer, "save_checkpoint", lambda step: None)
    monkeypatch.setattr(
        trainer_module,
        "broadcast_object_list",
        lambda payload: pytest.fail("single-process evaluation must not broadcast"),
    )

    outcome = trainer.train()

    assert outcome.step == 1


def test_cli_overrides_take_precedence_without_mutating_json(
    tmp_path, monkeypatch, capsys
):
    config_path = tmp_path / "train.json"
    config_path.write_text(json.dumps({"steps": 20, "stop_after_step": 5}))
    original_bytes = config_path.read_bytes()
    captured = {}

    monkeypatch.setattr(
        train_cli,
        "build_model_and_tokenizer",
        lambda config: (captured.setdefault("config", config) or object(), object()),
    )
    monkeypatch.setattr(
        train_cli, "build_dataloaders", lambda config, tokenizer: ([], [])
    )

    class FakeCliTrainer:
        def __init__(self, **kwargs):
            self.accelerator = SimpleNamespace(is_main_process=True)

        def train(self):
            return TrainingOutcome(8, "stop_after_step", 0.1, False)

    monkeypatch.setattr(train_cli, "OmniTrainer", FakeCliTrainer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "omnivoice-train",
            "--train_config",
            str(config_path),
            "--data_config",
            "data.json",
            "--output_dir",
            str(tmp_path / "output"),
            "--stop-after-step",
            "8",
            "--resume-from-checkpoint",
            "checkpoint-4",
        ],
    )

    train_cli.main()

    assert captured["config"].stop_after_step == 8
    assert captured["config"].resume_from_checkpoint == "checkpoint-4"
    assert config_path.read_bytes() == original_bytes
    assert json.loads(capsys.readouterr().out) == asdict(
        TrainingOutcome(8, "stop_after_step", 0.1, False)
    )
