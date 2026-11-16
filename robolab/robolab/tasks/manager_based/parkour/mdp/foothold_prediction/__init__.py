"""Foothold prediction components (privileged teacher / guidance)."""

from .config import (
    FootholdGridCfg,
    FootholdPredictorCfg,
    normalize_foothold_grid_cfg,
    normalize_foothold_predictor_cfg,
)
from .grid import FootholdGaussianGeometry, wrap_angle
from .model import FootholdPredictor
from .support import (
    FootholdSupportCfg,
    FootholdSupportEvaluator,
    normalize_foothold_support_cfg,
)
from .trainer import FootholdPredictorTrainer, empty_predictor_logs

__all__ = [
    "FootholdGridCfg",
    "FootholdPredictor",
    "FootholdPredictorCfg",
    "FootholdPredictorTrainer",
    "FootholdGaussianGeometry",
    "FootholdSupportCfg",
    "FootholdSupportEvaluator",
    "empty_predictor_logs",
    "normalize_foothold_predictor_cfg",
    "normalize_foothold_support_cfg",
    "normalize_foothold_grid_cfg",
    "wrap_angle",
]
