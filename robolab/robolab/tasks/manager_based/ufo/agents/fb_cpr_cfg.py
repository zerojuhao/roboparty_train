"""Shared configuration types for UFO and PUFO agents."""

from typing import Literal

from isaaclab.utils import configclass


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
