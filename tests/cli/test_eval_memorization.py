from types import SimpleNamespace

import pytest
import torch
from torch import nn

from omnivoice.cli.eval_memorization import check_memorization, mean_eval_loss


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
