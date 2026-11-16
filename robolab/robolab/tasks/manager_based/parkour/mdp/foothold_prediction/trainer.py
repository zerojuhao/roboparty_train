"""Foothold predictor optimization and RMSE reward gating."""

from __future__ import annotations

import copy
import warnings

import torch
import torch.nn.functional as F
import torch.optim as optim

from isaaclab.utils.math import wrap_to_pi
from rsl_rl.utils import reduce_gradients_in_buckets

from .config import FootholdPredictorCfg
from .grid import FootholdGaussianGeometry
from .model import FootholdPredictor
from .normalization import RunningMeanStd
from .replay_buffer import FOOTHOLD_REPLAY_FAMILY_IDS, FootholdReplayBuffer

# Supported-touchdown body-frame XY+yaw labels in the full-root-quaternion frame.
LABEL_FRAME = "supported_touchdown_body_quat_xy_v2"


def empty_predictor_logs() -> dict[str, float]:
    """Zeroed foothold logs used before the trainer exists or replay is ready."""
    return {
        "Foothold/Accuracy/xy_rmse_m": 0.0,
        "Foothold/Accuracy/yaw_rmse_rad": 0.0,
        "Foothold/Distribution/sigma_mean_m": 0.0,
        "Foothold/Distribution/yaw_sigma_mean_rad": 0.0,
    }


class FootholdPredictorTrainer:
    def __init__(
        self,
        input_dim: int,
        cfg: FootholdPredictorCfg,
        device: torch.device | str,
    ) -> None:
        self.input_dim = input_dim
        self.cfg = cfg
        self.device = device
        self.grid = FootholdGaussianGeometry(cfg.grid, device)
        self.input_normalizer = RunningMeanStd((input_dim,), device)
        if not cfg.hidden_dims:
            raise ValueError("FootholdPredictorCfg.hidden_dims is required.")
        self.model = FootholdPredictor(input_dim, list(cfg.hidden_dims)).to(device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for parameter in self.model.parameters():
                torch.distributed.broadcast(parameter.data, src=0)
        self.ema_model = copy.deepcopy(self.model).to(device)
        self.ema_model.eval()
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.train_buffer = FootholdReplayBuffer(
            cfg.train_buffer_capacity, input_dim, device
        )
        self.reward_enabled = False

    def add_samples(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        foot_ids: torch.Tensor,
        steps_to_contact: torch.Tensor,
        family_ids: torch.Tensor,
    ) -> None:
        self.train_buffer.append(
            inputs, targets, foot_ids, steps_to_contact, family_ids
        )

    def clear_train_buffer(self) -> None:
        self.train_buffer.clear()

    @torch.no_grad()
    def predict(
        self, predictor_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return means and standard deviations for XY and yaw."""
        self.ema_model.eval()
        mean_xy, mean_yaw, sigma_xy, sigma_yaw = self.grid.decode_distribution(
            self.ema_model(self.input_normalizer(predictor_input))
        )
        return mean_xy, mean_yaw, sigma_xy, sigma_yaw

    def _horizon_nll_weight(self, steps_to_contact: torch.Tensor) -> torch.Tensor:
        """Per-sample NLL weight. ``nll_horizon_tau <= 0`` is uniform (far = near)."""
        tau = float(self.cfg.nll_horizon_tau)
        if tau <= 0.0:
            return torch.ones(steps_to_contact.shape[0], device=self.device, dtype=torch.float32)
        floor = max(float(self.cfg.nll_horizon_weight_min), 0.0)
        return torch.exp(-steps_to_contact.to(dtype=torch.float32) / tau).clamp(min=floor)

    def _losses(
        self,
        model: FootholdPredictor,
        predictor_input: torch.Tensor,
        target: torch.Tensor,
        foot_ids: torch.Tensor,
        steps_to_contact: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pred_xy, pred_yaw, pred_sigma_xy, pred_sigma_yaw = self.grid.decode_distribution(
            model(self.input_normalizer(predictor_input))
        )
        batch_ids = torch.arange(predictor_input.shape[0], device=self.device)
        xy_foot = pred_xy[batch_ids, foot_ids]
        yaw_foot = pred_yaw[batch_ids, foot_ids]
        sigma_xy = pred_sigma_xy[batch_ids, foot_ids]
        sigma_yaw = pred_sigma_yaw[batch_ids, foot_ids]

        xy_squared_error = (xy_foot - target[..., :2]).square().sum(dim=-1)
        yaw_squared_error = wrap_to_pi(yaw_foot - target[..., 2]).square()
        per_sample = (
            xy_squared_error / (2.0 * sigma_xy.square())
            + 2.0 * torch.log(sigma_xy)
            + yaw_squared_error / (2.0 * sigma_yaw.square())
            + torch.log(sigma_yaw)
        )
        finite = torch.isfinite(target).all(dim=-1)
        sample_weight = self._horizon_nll_weight(steps_to_contact)
        if finite.any():
            weighted = sample_weight[finite]
            nll = (per_sample[finite] * weighted).sum() / weighted.sum().clamp_min(1.0e-8)
            xy_mean_loss = F.smooth_l1_loss(
                xy_foot[finite], target[finite, :2], beta=float(self.cfg.mean_loss_beta)
            )
            yaw_mean_loss = F.smooth_l1_loss(
                wrap_to_pi(yaw_foot[finite] - target[finite, 2]),
                torch.zeros_like(yaw_foot[finite]),
                beta=float(self.cfg.mean_loss_beta),
            )
        else:
            nll = xy_foot.sum() * 0.0
            xy_mean_loss = xy_foot.sum() * 0.0
            yaw_mean_loss = yaw_foot.sum() * 0.0

        valid_xy = xy_squared_error[finite]
        valid_yaw = yaw_squared_error[finite]
        valid_sigma_xy = sigma_xy[finite]
        valid_sigma_yaw = sigma_yaw[finite]
        xy_err = xy_foot[finite] - target[finite, :2]
        yaw_err = wrap_to_pi(yaw_foot[finite] - target[finite, 2])
        near = finite & (steps_to_contact < 8)
        zero = xy_foot.new_zeros(())
        metrics = {
            "nll": nll.detach(),
            "xy_mean_loss": xy_mean_loss.detach(),
            "yaw_mean_loss": yaw_mean_loss.detach(),
            "xy_rmse": (valid_xy.mean().sqrt() if valid_xy.numel() else zero).detach(),
            "xy_rmse_near": (
                xy_squared_error[near].mean().sqrt() if near.any() else zero
            ).detach(),
            "yaw_rmse": (valid_yaw.mean().sqrt() if valid_yaw.numel() else zero).detach(),
            "sigma_mean": (
                valid_sigma_xy.mean() if valid_sigma_xy.numel() else zero
            ).detach(),
            "yaw_sigma_mean": (
                valid_sigma_yaw.mean() if valid_sigma_yaw.numel() else zero
            ).detach(),
            "nll_horizon_weight": (
                sample_weight[finite].mean() if finite.any() else zero
            ).detach(),
        }
        loss = (
            float(self.cfg.nll_loss_coef) * nll
            + float(self.cfg.xy_mean_loss_coef) * xy_mean_loss
            + float(self.cfg.yaw_mean_loss_coef) * yaw_mean_loss
        )
        return loss, metrics

    @torch.no_grad()
    def _update_ema_model(self) -> None:
        for ema_parameter, parameter in zip(
            self.ema_model.parameters(), self.model.parameters()
        ):
            ema_parameter.mul_(self.cfg.ema_decay).add_(
                parameter, alpha=1.0 - self.cfg.ema_decay
            )

    def idle_metrics(self) -> dict[str, float]:
        return empty_predictor_logs()

    def update(self) -> dict[str, float]:
        locally_ready = self.train_buffer.size >= self.cfg.min_train_samples
        globally_ready = locally_ready
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            ready = torch.tensor(int(locally_ready), device=self.device)
            torch.distributed.all_reduce(ready, op=torch.distributed.ReduceOp.MIN)
            globally_ready = bool(ready.item())

        if not globally_ready:
            return self.idle_metrics()

        self.model.train()
        for _ in range(self.cfg.updates_per_iteration):
            inputs, targets, foot_ids, steps_to_contact = self.train_buffer.sample(
                self.cfg.batch_size,
                balanced=self.cfg.balanced_sampling,
                candidate_multiplier=self.cfg.balanced_candidate_multiplier,
            )
            self.input_normalizer.update(inputs)
            loss, _ = self._losses(
                self.model, inputs, targets, foot_ids, steps_to_contact
            )
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                reduce_gradients_in_buckets(
                    self.model.parameters(),
                    world_size,
                    self.cfg.grad_reduce_bucket_mb,
                )
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self._update_ema_model()

        with torch.no_grad():
            inputs, targets, foot_ids, steps_to_contact = self.train_buffer.sample(
                self.cfg.batch_size
            )
            _, metrics = self._losses(
                self.ema_model, inputs, targets, foot_ids, steps_to_contact
            )
            metric_values = {key: float(value.item()) for key, value in metrics.items()}

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            packed_keys = ("xy_rmse", "yaw_rmse", "sigma_mean", "yaw_sigma_mean")
            packed = torch.tensor(
                tuple(metric_values[key] for key in packed_keys),
                dtype=torch.float64,
                device=self.device,
            )
            torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
            packed /= torch.distributed.get_world_size()
            metric_values = dict(zip(packed_keys, packed.tolist()))

        if (
            not self.reward_enabled
            and metric_values["xy_rmse"] < self.cfg.enable_xy_rmse_threshold
            and metric_values["yaw_rmse"] < self.cfg.enable_yaw_rmse_threshold
        ):
            self.reward_enabled = True

        return {
            "Foothold/Accuracy/xy_rmse_m": metric_values["xy_rmse"],
            "Foothold/Accuracy/yaw_rmse_rad": metric_values["yaw_rmse"],
            "Foothold/Distribution/sigma_mean_m": metric_values["sigma_mean"],
            "Foothold/Distribution/yaw_sigma_mean_rad": metric_values["yaw_sigma_mean"],
        }

    def state_dict(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "grid_signature": self.grid.signature,
            "label_frame": LABEL_FRAME,
            "foothold_family_ids": FOOTHOLD_REPLAY_FAMILY_IDS,
            "reward_enabled": self.reward_enabled,
            "input_normalizer": self.input_normalizer.state_dict(),
            "model": self.model.state_dict(),
            "ema_model": self.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict, load_optimizer: bool = True) -> None:
        if state.get("grid_signature") != self.grid.signature:
            raise RuntimeError(
                "Foothold predictor checkpoint uses a different output distribution; "
                "train the Gaussian predictor from scratch."
            )
        if int(state["input_dim"]) != self.input_dim:
            raise RuntimeError(
                f"Foothold predictor input changed from {state['input_dim']} to {self.input_dim}."
            )
        if state["label_frame"] != LABEL_FRAME:
            raise RuntimeError("Foothold predictor checkpoint uses a different label frame.")
        self.model.load_state_dict(state["model"])
        self.ema_model.load_state_dict(state["ema_model"])
        self.input_normalizer.load_state_dict(state["input_normalizer"])
        if load_optimizer:
            self.optimizer.load_state_dict(state["optimizer"])
        saved_family_ids = tuple(state["foothold_family_ids"])
        family_coverage_matches = saved_family_ids == FOOTHOLD_REPLAY_FAMILY_IDS
        if not family_coverage_matches:
            warnings.warn(
                "Foothold predictor terrain-family coverage changed; retaining model "
                "weights but disabling guidance until the refreshed replay reaches "
                "the configured accuracy thresholds.",
                stacklevel=2,
            )
        self.reward_enabled = (
            bool(state["reward_enabled"]) and family_coverage_matches
        )
