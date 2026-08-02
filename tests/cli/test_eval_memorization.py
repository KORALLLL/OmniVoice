import argparse
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from omnivoice.cli.eval_memorization import (
    check_memorization,
    check_memorization_patience,
    mean_eval_loss,
)
from omnivoice.cli.run_memorization import _positive_seconds


class FakeModel(nn.Module):
    def forward(self, loss, **kwargs):
        return SimpleNamespace(loss=loss)


def test_check_memorization_accepts_loss_at_threshold():
    assert check_memorization(loss=0.01, threshold=0.01) == 0


def test_check_memorization_rejects_loss_above_threshold():
    assert check_memorization(loss=0.0101, threshold=0.01) == 1


def test_check_memorization_rejects_non_positive_threshold():
    with pytest.raises(ValueError, match="threshold must be positive"):
        check_memorization(loss=0.0, threshold=0.0)


def test_check_memorization_patience_reports_second_consecutive_hit():
    result = check_memorization_patience(
        [(50, 2e-4), (75, 9e-5), (100, 8e-5)],
        threshold=1e-4,
        patience=2,
    )

    assert result.passed is True
    assert result.qualifying_step == 100
    assert result.minimum_loss == 8e-5


def test_check_memorization_patience_resets_after_miss():
    result = check_memorization_patience(
        [(25, 9e-5), (50, 2e-4), (75, 8e-5)],
        threshold=1e-4,
        patience=2,
    )

    assert result.passed is False
    assert result.qualifying_step is None
    assert result.minimum_loss == 8e-5


def test_mean_eval_loss_is_weighted_by_batch_count():
    batches = [
        {"loss": torch.tensor(0.2)},
        {"loss": torch.tensor(0.3)},
    ]

    loss, num_batches = mean_eval_loss(
        FakeModel(), batches, device="cpu", dtype=torch.float32
    )

    assert loss == pytest.approx(0.25)
    assert num_batches == 2


def test_mean_eval_loss_rejects_empty_dev_data():
    with pytest.raises(ValueError, match="dev data is empty"):
        mean_eval_loss(FakeModel(), [], device="cpu", dtype=torch.float32)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0", "-1"])
def test_memorization_cli_rejects_nonfinite_or_nonpositive_durations(value):
    with pytest.raises(argparse.ArgumentTypeError, match="finite and positive"):
        _positive_seconds(value)
