"""Shared configuration types for UFO and PUFO agents."""

from typing import Literal

from isaaclab.utils import configclass


@configclass
class UFOWorldModelCfg:
    """Real dynamics with quality-gated FB value-expansion gradients."""

    enabled: bool = True
    hidden_dims: list[int] = [512, 512, 512]
    ensemble_size: int = 2
    window_length: int = 8
    objective: Literal["fb_sequence", "fb_value", "context"] = "fb_sequence"
    # Full 0.64 s policy BPTT; dynamics fitting still uses 8 real replay steps.
    horizon: int = 32
    batch_size: int = 2048
    fit_batches: int = 10
    bootstrap_members: bool = True
    imagination_batch_size: int = 4096
    imagination_microbatch_size: int = 64
    scoring_chunk_size: int = 2048
    imagination_interval: int = 1
    update_interval: int = 4
    learning_rate: float = 3e-4
    actor_weight: float = 0.05
    # Counts fitting phases: 100 phases x 10 batches = 1000 optimizer steps.
    warmup_updates: int = 100
    value_warmup_updates: int = 100
    value_lambda: float = 0.95
    ramp_updates: int = 1000
    grad_clip: float = 1.0
    heading_weight: float = 1.0
    velocity_weight: float = 1.0
    value_hidden_dims: list[int] = [256, 256]
    value_tau: float = 0.01
    sequence_context_length: int = 10
    validation_interval: int = 10
    validation_batch_size: int = 64
    validation_horizons: tuple[int, ...] = (1, 4, 8, 16, 32)
    # Errors use the dynamics increment scales. All thresholds are detached gates.
    validation_max_p95: float = 0.5
    imagination_max_disagreement: float = 0.25
    imagination_action_radius: float = 0.5
    fb_covariance_batch_size: int = 4096
    fb_covariance_chunk_size: int = 1024
    fb_covariance_ridge: float = 1e-3
    fb_max_projection_residual: float = 0.1


@configclass
class FbCprResidualModelCfg:
    """Two embedding towers followed by a constant-width residual trunk."""

    hidden_dim: int = 1024
    hidden_layers: int = 6
    embedding_layers: int = 2
    activation: str = "mish"
    layer_norm: bool = True


@configclass
class FbCprMlpModelCfg:
    """MLP with an optional normalized first hidden layer."""

    hidden_dims: list[int] = [256]
    activation: str = "relu"
    first_activation: str = "tanh"
    layer_norm: bool = True


@configclass
class FbCprDepthEncoderCfg:
    """Input shape and network for temporal depth feature extraction."""

    input_shape: tuple[int, int, int] = (8, 18, 36)
    channels: list[int] = [16, 32, 64]
    kernel_sizes: list[int] = [3, 3, 3]
    strides: list[int] = [1, 2, 2]
    paddings: list[int] = [1, 1, 1]
    pool_shape: tuple[int, int] = (2, 4)
    hidden_dims: list[int] = [256]
    activation: str = "relu"
    output_size: int = 256
    gru_layers: int = 1
    # Deployment input contract.  The environment observation is normalized
    # to [0, 1] and quantized once at the PUFO observation boundary; the
    # network then consumes the same uint8 tensor used by the exported ONNX.
    input_dtype: str = "uint8"
    input_range: tuple[float, float] = (0.0, 255.0)


@configclass
class FbCprDiscriminatorCfg(FbCprMlpModelCfg):
    """Discriminator architecture, classification loss and critic reward."""

    hidden_dims: list[int] = [1024, 1024, 1024]
    loss_type: Literal["bce", "gan", "lsgan", "wgan"] = "bce"
    """BCE matches H200; GAN uses AMP's half-scaled BCE objective."""

    reward_type: Literal["log_odds", "amp", "softplus", "raw"] = "log_odds"
    """H200 defaults to log_odds. AMP pairs LSGAN/GAN/WGAN with amp/softplus/raw."""
