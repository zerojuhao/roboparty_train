"""Tests for AMP velocity-estimation metrics."""

import pytest
import torch
from tensordict import TensorDict
from torch import nn

from rsl_rl.models.amp_model import AMPModel, resolve_velocity_target_indices


class _FixedVelocityEstimator(nn.Module):
    def __init__(self, prediction: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("prediction", prediction)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.prediction[: obs.shape[0]]


def _make_observations() -> TensorDict:
    critic = torch.zeros(2, 9)
    critic[:, 6:8] = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
    return TensorDict({"policy": torch.zeros(2, 4), "critic": critic}, batch_size=[2])


def test_velocity_estimation_metrics_report_xy_errors() -> None:
    observations = _make_observations()
    model = AMPModel(
        observations,
        {"actor": ["policy"]},
        "actor",
        2,
        hidden_dims=[8],
        enable_velocity_estimation=True,
        velocity_target_indices=(6, 7),
    )
    model.velocity_estimator = _FixedVelocityEstimator(torch.tensor([[4.0, 6.0], [-1.0, 0.5]]))

    model.get_latent(observations)
    metrics = model.get_aux_metrics()

    assert torch.allclose(model.get_aux_loss(), torch.tensor(6.25))
    assert torch.allclose(metrics["Estimation/velocity/rmse_xy"], torch.sqrt(torch.tensor(12.5)))
    assert torch.allclose(metrics["Estimation/velocity/mae_x"], torch.tensor(1.5))
    assert torch.allclose(metrics["Estimation/velocity/mae_y"], torch.tensor(2.0))


def test_velocity_estimation_metrics_are_absent_when_disabled() -> None:
    observations = _make_observations()
    model = AMPModel(
        observations,
        {"actor": ["policy"]},
        "actor",
        2,
        hidden_dims=[8],
        enable_velocity_estimation=False,
    )

    model.get_latent(observations)

    assert model.get_aux_loss() is None
    assert model.get_aux_metrics() == {}


@pytest.mark.parametrize("history", [0, 1, 3, 5])
def test_velocity_target_follows_runtime_history_and_order(history):
    velocity_width = 3 * max(1, history)
    term_names = ["joint_pos", "base_ang_vel", "base_lin_vel", "actions"]
    term_shapes = [(24,), (6,), (velocity_width,), (24,)]
    indices = resolve_velocity_target_indices(term_names, term_shapes)
    assert indices == (30 + velocity_width - 3, 30 + velocity_width - 2)
    critic = torch.full((2, 54 + velocity_width), 100.0)
    target = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
    critic[:, list(indices)] = target
    observations = TensorDict({"policy": torch.zeros(2, 4), "critic": critic}, batch_size=[2])
    model = AMPModel(
        observations, {"actor": ["policy"]}, "actor", 2,
        hidden_dims=[8], velocity_target_indices=indices,
    )
    model.velocity_estimator = _FixedVelocityEstimator(target)
    model.get_latent(observations)
    assert model.get_aux_loss().item() == 0


def test_velocity_target_rejects_missing_or_nonflat_term():
    with pytest.raises(ValueError, match="base_lin_vel"):
        resolve_velocity_target_indices(["joint_pos"], [(24,)])
    with pytest.raises(ValueError):
        resolve_velocity_target_indices(["base_lin_vel"], [(5, 3)])


def test_velocity_estimator_cannot_silently_assume_three_frames():
    with pytest.raises(ValueError, match="velocity_target_indices"):
        AMPModel(_make_observations(), {"actor": ["policy"]}, "actor", 2, hidden_dims=[8])
