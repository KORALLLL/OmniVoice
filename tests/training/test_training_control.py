import json
import multiprocessing
import sys
import warnings
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

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
        ({"steps": True}, "steps must be an integer"),
        ({"steps": 1.5}, "steps must be an integer"),
        ({"steps_per_epoch": 0}, "steps_per_epoch must be positive"),
        ({"steps_per_epoch": False}, "steps_per_epoch must be an integer"),
        ({"steps_per_epoch": 1.5}, "steps_per_epoch must be an integer"),
        ({"stop_after_step": 0}, "stop_after_step must be positive"),
        ({"stop_after_step": True}, "stop_after_step must be an integer"),
        ({"stop_after_step": 1.5}, "stop_after_step must be an integer"),
        ({"steps": 10, "stop_after_step": 11}, "cannot exceed steps"),
        ({"max_wall_clock_seconds": 0}, "max_wall_clock_seconds must be positive"),
        ({"max_wall_clock_seconds": float("inf")}, "must be finite"),
        ({"max_wall_clock_seconds": float("nan")}, "must be finite"),
        ({"early_stop_eval_loss": 0}, "early_stop_eval_loss must be positive"),
        ({"early_stop_eval_loss": float("inf")}, "must be finite"),
        ({"early_stop_eval_loss": float("nan")}, "must be finite"),
        ({"early_stop_patience": 0}, "early_stop_patience must be positive"),
        ({"early_stop_patience": True}, "early_stop_patience must be an integer"),
        ({"early_stop_patience": 1.5}, "early_stop_patience must be an integer"),
        ({"eval_history_path": ""}, "eval_history_path must be a non-empty string"),
        ({"eval_history_path": "  "}, "eval_history_path must be a non-empty string"),
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

    def eval(self):
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


class SequenceDataset:
    def __init__(self, steps_per_epoch):
        self.steps_per_epoch = steps_per_epoch
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        for offset in range(self.steps_per_epoch):
            yield {"sample_id": self.epoch * 10 + offset}


class SequenceLoader:
    def __init__(self, steps_per_epoch, *, consume_rng_on_iter=False):
        self.dataset = SequenceDataset(steps_per_epoch)
        self.consume_rng_on_iter = consume_rng_on_iter

    def __iter__(self):
        if self.consume_rng_on_iter:
            torch.rand(1)
        return iter(self.dataset)


class RecordingModel(FakeModel):
    def __init__(self):
        self.sample_ids = []

    def __call__(self, **batch):
        self.sample_ids.append(batch["sample_id"])
        return SimpleNamespace(loss=torch.tensor(0.5))


class EvalLossModel(FakeModel):
    def __call__(self, **batch):
        return SimpleNamespace(loss=torch.tensor(batch["loss"], dtype=torch.float32))


class GatherStatsAccelerator(FakeAccelerator):
    def __init__(self, gathered_stats):
        super().__init__()
        self.gathered_stats = torch.tensor(gathered_stats, dtype=torch.float32)

    def gather(self, value):
        return self.gathered_stats


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


def _run_recording_trainer(config, monkeypatch, *, resume_step=None, loader=None):
    trainer = make_bare_trainer(config)
    trainer.model = RecordingModel()
    trainer.train_dataloader = loader or SequenceLoader(config.steps_per_epoch)
    trainer.eval_dataloader = None
    monkeypatch.setattr(trainer_module, "TrainLogger", FakeTrainLogger)
    monkeypatch.setattr(trainer, "save_checkpoint", lambda step: None)
    if resume_step is not None:
        monkeypatch.setattr(
            trainer,
            "load_checkpoint",
            lambda checkpoint_path: setattr(trainer, "global_step", resume_step)
            or resume_step,
        )
    outcome = trainer.train()
    return trainer.model.sample_ids, outcome


def test_segmented_resume_consumes_same_data_sequence_as_uninterrupted(
    tmp_path, monkeypatch
):
    uninterrupted, _ = _run_recording_trainer(
        TrainingConfig(
            output_dir=str(tmp_path / "full"),
            steps=6,
            steps_per_epoch=5,
            logging_steps=20,
            save_steps=20,
        ),
        monkeypatch,
    )
    first_segment, _ = _run_recording_trainer(
        TrainingConfig(
            output_dir=str(tmp_path / "first"),
            steps=6,
            steps_per_epoch=5,
            stop_after_step=3,
            logging_steps=20,
            save_steps=20,
        ),
        monkeypatch,
    )
    resumed_segment, outcome = _run_recording_trainer(
        TrainingConfig(
            output_dir=str(tmp_path / "second"),
            steps=6,
            steps_per_epoch=5,
            stop_after_step=6,
            resume_from_checkpoint="checkpoint-3",
            logging_steps=20,
            save_steps=20,
        ),
        monkeypatch,
        resume_step=3,
    )

    assert first_segment + resumed_segment == uninterrupted
    assert outcome.step == 6


def test_resume_cursor_reconstruction_preserves_restored_rng(tmp_path, monkeypatch):
    checkpoint_rng = torch.Generator().manual_seed(9876).get_state()
    expected_generator = torch.Generator()
    expected_generator.set_state(checkpoint_rng)
    expected_next_random = torch.rand(1, generator=expected_generator)
    config = TrainingConfig(
        output_dir=str(tmp_path),
        steps=5,
        steps_per_epoch=5,
        stop_after_step=3,
        resume_from_checkpoint="checkpoint-2",
        logging_steps=20,
        save_steps=20,
    )
    loader = SequenceLoader(5, consume_rng_on_iter=True)
    trainer = make_bare_trainer(config)
    trainer.model = RecordingModel()
    trainer.train_dataloader = loader
    trainer.eval_dataloader = None
    monkeypatch.setattr(trainer_module, "TrainLogger", FakeTrainLogger)
    monkeypatch.setattr(trainer, "save_checkpoint", lambda step: None)

    def restore_checkpoint(checkpoint_path):
        trainer.global_step = 2
        torch.set_rng_state(checkpoint_rng)
        return 2

    monkeypatch.setattr(trainer, "load_checkpoint", restore_checkpoint)

    trainer.train()

    torch.testing.assert_close(torch.rand(1), expected_next_random)


def test_evaluate_uses_global_loss_sum_and_count():
    trainer = object.__new__(OmniTrainer)
    trainer.model = EvalLossModel()
    trainer.eval_dataloader = [{"loss": 1.0}, {"loss": 3.0}]
    trainer.accelerator = GatherStatsAccelerator([4.0, 2.0, 10.0, 1.0])
    trainer.global_step = 7

    metrics = trainer.evaluate()

    assert metrics["eval/loss"] == pytest.approx(14 / 3)


def test_evaluate_rejects_global_zero_batch_count():
    trainer = object.__new__(OmniTrainer)
    trainer.model = EvalLossModel()
    trainer.eval_dataloader = []
    trainer.accelerator = GatherStatsAccelerator([0.0, 0.0])
    trainer.global_step = 7

    with pytest.raises(ValueError, match="no batches"):
        trainer.evaluate()


def test_default_completion_does_not_force_an_unscheduled_final_evaluation(
    tmp_path, monkeypatch
):
    config = TrainingConfig(
        output_dir=str(tmp_path),
        steps=3,
        eval_steps=2,
        logging_steps=20,
        save_steps=20,
    )
    trainer = make_bare_trainer(config)
    evaluated_steps = []
    monkeypatch.setattr(trainer_module, "TrainLogger", FakeTrainLogger)
    monkeypatch.setattr(
        trainer,
        "evaluate",
        lambda: evaluated_steps.append(trainer.global_step)
        or {"eval/loss": 0.25},
    )
    monkeypatch.setattr(trainer, "save_checkpoint", lambda step: None)

    outcome = trainer.train()

    assert evaluated_steps == [2]
    assert outcome.last_eval_loss == 0.25


def test_already_complete_resume_does_not_force_evaluation(tmp_path, monkeypatch):
    config = TrainingConfig(
        output_dir=str(tmp_path),
        steps=3,
        resume_from_checkpoint="checkpoint-3",
        eval_steps=2,
        logging_steps=20,
        save_steps=20,
    )
    trainer = make_bare_trainer(config)
    monkeypatch.setattr(trainer_module, "TrainLogger", FakeTrainLogger)
    monkeypatch.setattr(
        trainer,
        "load_checkpoint",
        lambda checkpoint_path: setattr(trainer, "global_step", 3) or 3,
    )
    monkeypatch.setattr(
        trainer,
        "evaluate",
        lambda: pytest.fail("already-complete default resume must not evaluate"),
    )
    monkeypatch.setattr(trainer, "save_checkpoint", lambda step: None)

    outcome = trainer.train()

    assert outcome.last_eval_loss is None


def test_train_tears_down_when_evaluation_raises(tmp_path, monkeypatch):
    config = TrainingConfig(
        output_dir=str(tmp_path),
        steps=1,
        eval_steps=1,
        logging_steps=20,
        save_steps=20,
    )
    trainer = make_bare_trainer(config)
    train_logger = FakeTrainLogger(None, 1, 1)
    train_logger.closed = False
    monkeypatch.setattr(
        train_logger, "close", lambda: setattr(train_logger, "closed", True)
    )
    monkeypatch.setattr(
        trainer_module, "TrainLogger", lambda *args: train_logger
    )
    monkeypatch.setattr(
        trainer, "evaluate", lambda: (_ for _ in ()).throw(ValueError("eval boom"))
    )
    monkeypatch.setattr(trainer, "save_checkpoint", lambda step: None)

    with pytest.raises(ValueError, match="eval boom"):
        trainer.train()

    assert train_logger.closed
    assert trainer.accelerator.ended


class DistributedEvalAccelerator:
    def __init__(self, rank, world_size):
        self.device = torch.device("cpu")
        self.process_index = rank
        self.num_processes = world_size

    def gather(self, value):
        gathered = [torch.empty_like(value) for _ in range(self.num_processes)]
        dist.all_gather(gathered, value)
        return torch.cat(gathered)

    def log(self, metrics, step):
        pass

    def wait_for_everyone(self):
        dist.barrier()


class RankFailingEvalModel(FakeModel):
    def __init__(self, rank):
        self.rank = rank

    def __call__(self, **batch):
        if self.rank == 1:
            raise ValueError("rank-one boom")
        return SimpleNamespace(loss=torch.tensor(0.5))


def _distributed_evaluation_error_worker(rank, init_path, result_dir):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=5),
    )

    def gather_errors(local_errors):
        gathered = [None, None]
        dist.all_gather_object(gathered, local_errors)
        return [error for rank_errors in gathered for error in rank_errors]

    trainer_module.gather_object = gather_errors
    trainer = object.__new__(OmniTrainer)
    trainer.model = RankFailingEvalModel(rank)
    trainer.eval_dataloader = [{}]
    trainer.accelerator = DistributedEvalAccelerator(rank, 2)
    trainer.global_step = 1
    try:
        trainer.evaluate()
    except BaseException as exc:  # noqa: BLE001
        Path(result_dir, f"rank-{rank}.txt").write_text(
            f"{type(exc).__name__}: {exc}"
        )
    else:
        Path(result_dir, f"rank-{rank}.txt").write_text("NO_ERROR")
    finally:
        dist.destroy_process_group()


def test_rank_local_evaluation_error_reaches_every_rank_without_hang(tmp_path):
    init_path = tmp_path / "gloo-init"
    context = multiprocessing.get_context("fork")
    processes = [
        context.Process(
            target=_distributed_evaluation_error_worker,
            args=(rank, str(init_path), str(tmp_path)),
        )
        for rank in range(2)
    ]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"This process .* is multi-threaded, use of fork\(\)",
            category=DeprecationWarning,
        )
        for process in processes:
            process.start()
    for process in processes:
        process.join(timeout=15)
    hung_processes = [process for process in processes if process.is_alive()]
    for process in hung_processes:
        process.terminate()
        process.join(timeout=5)

    assert not hung_processes
    assert [process.exitcode for process in processes] == [0, 0]
    messages = [
        (tmp_path / f"rank-{rank}.txt").read_text() for rank in range(2)
    ]
    assert all("process 1 ValueError: rank-one boom" in message for message in messages)


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
