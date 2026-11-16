"""Agent-owned configuration schema for foothold prediction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class FootholdGridCfg:
    """Gaussian foothold distribution bounds and XY/yaw quadrature."""

    sigma_min: float = 0.01
    """Lower bound of isotropic XY sigma, in meters."""
    sigma_max: float = 0.25
    """Upper bound of isotropic XY sigma, in meters."""
    yaw_sigma_min: float = 0.05
    """Lower bound of landing-yaw sigma, in radians."""
    yaw_sigma_max: float = 0.5
    """Upper bound of landing-yaw sigma, in radians."""
    expectation_grid_size: int = 5
    """Odd number of XY quadrature samples per axis."""
    expectation_std_range: float = 2.0
    """XY quadrature half-width in units of sigma."""
    expectation_eval_chunk_size: int = 64
    """Environment-foot pairs per support-evaluation chunk."""
    expectation_weighting: str = "max"
    """How to reduce support over the XY quadrature grid.

    ``max`` takes the worst hanging sample in the neighborhood so an edge in
    the tail is not washed out. ``uniform`` averages equally. ``gaussian`` is
    the learned-density expectation.
    """
    reward_sigma_min: float = 0.04
    """Floor on XY sigma used only for swing-guidance quadrature, in meters.

    NLL still uses the learned sigma. ``<=0`` disables the floor. With
    ``expectation_std_range=2``, 0.04 m covers ±8 cm around μ. Larger floors
    (e.g. 0.08 → ±16 cm) let max() fire on centered landings: treads are
    0.30 m and the sole is 0.26 m, so the on-tread ankle window is ~4 cm.
    """


@configclass
class FootholdPredictorCfg:
    """Privileged foothold teacher / guidance training options."""

    enabled: bool = False
    hidden_dims: list[int] = MISSING
    learning_rate: float = MISSING
    weight_decay: float = MISSING
    ema_decay: float = MISSING
    grid: FootholdGridCfg = MISSING
    nll_loss_coef: float = 1.0
    """Weight on heteroscedastic XY/yaw Gaussian NLL."""
    xy_mean_loss_coef: float = 2.0
    """Weight on direct XY Smooth L1 regression."""
    yaw_mean_loss_coef: float = 0.5
    """Weight on wrapped-yaw Smooth L1 regression."""
    mean_loss_beta: float = 0.02
    """Smooth L1 transition width for XY meters and yaw radians."""
    nll_horizon_tau: float = 0.0
    """NLL horizon scale in remaining steps-to-contact. ``<=0`` is uniform NLL."""
    nll_horizon_weight_min: float = 0.25
    """Floor on NLL sample weights when ``nll_horizon_tau > 0``."""
    guidance_horizon_tau: float = 8.0
    """Guidance weight scale in elapsed swing steps. ``<=0`` is uniform guidance.

    Early swing (action can still change the landing) is up-weighted:
    ``w = max(min, exp(-elapsed / tau))``. Independent of NLL weighting.
    """
    guidance_horizon_weight_min: float = 0.25
    """Floor on late-swing guidance weights when ``guidance_horizon_tau > 0``."""
    max_pending_steps: int = MISSING
    pending_sample_stride: int = MISSING
    train_pending_tail_steps: int = MISSING
    """Keep the last K swing steps when labeling a touchdown. ``<=0`` keeps all."""
    train_buffer_capacity: int = MISSING
    batch_size: int = MISSING
    updates_per_iteration: int = MISSING
    balanced_sampling: bool = True
    """Balance replay batches over terrain family, foot, and contact horizon."""
    balanced_candidate_multiplier: int = 8
    """Uniform candidate-pool multiplier used by approximate balanced sampling."""
    grad_reduce_bucket_mb: float = 25.0
    """Maximum packed gradient bucket size used by distributed predictor training."""
    min_train_samples: int = MISSING
    curriculum_level_threshold: float = MISSING
    """Enable terrain-ready after mean inverted-stairs (``stairs``+``inv``) curriculum exceeds this."""
    enable_xy_rmse_threshold: float = MISSING
    """Enable swing guidance after teacher XY RMSE falls below this value, in meters."""
    enable_yaw_rmse_threshold: float = MISSING
    """Enable swing guidance after teacher yaw RMSE falls below this value, in radians."""
    clear_train_buffer_on_resume: bool = True


def normalize_foothold_grid_cfg(cfg: FootholdGridCfg | Mapping) -> FootholdGridCfg:
    if isinstance(cfg, FootholdGridCfg):
        return cfg
    if isinstance(cfg, Mapping):
        return FootholdGridCfg(**cfg)
    raise TypeError(f"Unsupported foothold grid config type: {type(cfg)!r}")


def normalize_foothold_predictor_cfg(
    cfg: FootholdPredictorCfg | Mapping,
) -> FootholdPredictorCfg:
    if isinstance(cfg, FootholdPredictorCfg):
        return cfg
    if isinstance(cfg, Mapping):
        values = dict(cfg)
        values["grid"] = normalize_foothold_grid_cfg(values["grid"])
        return FootholdPredictorCfg(**values)
    raise TypeError(f"Unsupported foothold predictor config type: {type(cfg)!r}")
