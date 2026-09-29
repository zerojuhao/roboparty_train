"""Distribution decoding and XY quadrature for Gaussian foothold prediction."""

from __future__ import annotations

import torch

from isaaclab.utils.math import wrap_to_pi

wrap_angle = wrap_to_pi


class FootholdGaussianGeometry:
    """Decode foothold distributions and sample XY/yaw expectations."""

    def __init__(self, cfg, device: torch.device | str) -> None:
        self.cfg = cfg
        self.device = device
        self.sigma_min = float(cfg.sigma_min)
        self.sigma_max = float(cfg.sigma_max)
        self.yaw_sigma_min = float(cfg.yaw_sigma_min)
        self.yaw_sigma_max = float(cfg.yaw_sigma_max)
        if not 0.0 < self.sigma_min < self.sigma_max:
            raise ValueError(
                f"Expected 0 < sigma_min < sigma_max, got {self.sigma_min} and {self.sigma_max}."
            )
        if not 0.0 < self.yaw_sigma_min < self.yaw_sigma_max:
            raise ValueError(
                "Expected 0 < yaw_sigma_min < yaw_sigma_max, got "
                f"{self.yaw_sigma_min} and {self.yaw_sigma_max}."
            )

        grid_size = int(cfg.expectation_grid_size)
        if grid_size < 1 or grid_size % 2 == 0:
            raise ValueError("expectation_grid_size must be a positive odd integer.")
        std_range = float(cfg.expectation_std_range)
        if std_range <= 0.0:
            raise ValueError("expectation_std_range must be positive.")
        axis = torch.linspace(-std_range, std_range, grid_size, device=device)
        xx, yy = torch.meshgrid(axis, axis, indexing="ij")
        self.standard_offsets = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
        log_weights = -0.5 * self.standard_offsets.square().sum(dim=-1)
        self.standard_weights = torch.softmax(log_weights, dim=0)
        self.standard_yaw_offsets = torch.tensor([-1.0, 0.0, 1.0], device=device)
        self.standard_yaw_weights = torch.softmax(
            -0.5 * self.standard_yaw_offsets.square(), dim=0
        )
        weighting = str(cfg.expectation_weighting).lower()
        if weighting not in ("max", "uniform", "gaussian"):
            raise ValueError(
                "expectation_weighting must be 'max', 'uniform', or 'gaussian', "
                f"got {weighting!r}."
            )
        self.expectation_weighting = weighting

    def decode_distribution(
        self, raw_distribution: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode raw ``[..., 5]`` into mean XY, mean yaw, XY sigma, and yaw sigma."""
        if raw_distribution.shape[-1] != 5:
            raise ValueError(
                "Expected raw foothold distribution last dim 5, "
                f"got {raw_distribution.shape[-1]}."
            )
        mean_xy = raw_distribution[..., :2]
        mean_yaw = wrap_to_pi(raw_distribution[..., 2])
        sigma_xy = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(
            raw_distribution[..., 3]
        )
        sigma_yaw = self.yaw_sigma_min + (
            self.yaw_sigma_max - self.yaw_sigma_min
        ) * torch.sigmoid(raw_distribution[..., 4])
        return mean_xy, mean_yaw, sigma_xy, sigma_yaw

    def expectation_points(
        self, mean_xy: torch.Tensor, sigma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return XY quadrature points ``[..., G, 2]`` and normalized weights ``[..., G]``."""
        points = mean_xy.unsqueeze(-2) + sigma[..., None, None] * self.standard_offsets
        weight_shape = (*mean_xy.shape[:-1], self.standard_weights.shape[0])
        if self.expectation_weighting == "gaussian":
            weights = self.standard_weights.expand(weight_shape)
        else:
            n_points = self.standard_weights.shape[0]
            weights = torch.full(
                weight_shape,
                1.0 / n_points,
                device=self.device,
                dtype=self.standard_weights.dtype,
            )
        return points, weights

    def reward_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        """Widen XY sigma for support quadrature without changing the trained σ."""
        floor = float(self.cfg.reward_sigma_min)
        if floor <= 0.0:
            return sigma
        return sigma.clamp(min=floor)

    def expectation_poses(
        self,
        mean_xy: torch.Tensor,
        sigma_xy: torch.Tensor,
        mean_yaw: torch.Tensor,
        sigma_yaw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return Cartesian-product XY/yaw quadrature and normalized weights."""
        xy, xy_weights = self.expectation_points(mean_xy, sigma_xy)
        yaw = wrap_to_pi(
            mean_yaw.unsqueeze(-1)
            + sigma_yaw.unsqueeze(-1) * self.standard_yaw_offsets
        )
        yaw_weights = self.standard_yaw_weights.expand_as(yaw)
        num_xy = xy.shape[-2]
        num_yaw = yaw.shape[-1]
        poses_xy = xy.unsqueeze(-2).expand(*xy.shape[:-2], num_xy, num_yaw, 2)
        poses_yaw = yaw.unsqueeze(-2).expand(*yaw.shape[:-1], num_xy, num_yaw)
        weights = xy_weights.unsqueeze(-1) * yaw_weights.unsqueeze(-2)
        return (
            poses_xy.reshape(*xy.shape[:-2], num_xy * num_yaw, 2),
            poses_yaw.reshape(*yaw.shape[:-1], num_xy * num_yaw),
            weights.reshape(*weights.shape[:-2], num_xy * num_yaw),
        )

    @property
    def signature(self) -> tuple:
        return (
            "gaussian_xy_yaw_v1",
            self.sigma_min,
            self.sigma_max,
            self.yaw_sigma_min,
            self.yaw_sigma_max,
            int(self.cfg.expectation_grid_size),
            float(self.cfg.expectation_std_range),
        )
