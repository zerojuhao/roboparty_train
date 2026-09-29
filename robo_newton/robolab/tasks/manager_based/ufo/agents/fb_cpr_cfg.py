"""Shared configuration types for UFO and PUFO agents."""

from typing import Literal

from isaaclab.utils import configclass


@configclass
class FbCprDiscriminatorCfg:
    """Discriminator architecture, classification loss and critic reward."""

    hidden_dims: list[int] = [1024, 1024, 1024]
    loss_type: Literal["bce", "gan", "lsgan", "wgan"] = "bce"
    """BCE matches H200; GAN uses AMP's half-scaled BCE objective."""

    reward_type: Literal["log_odds", "amp", "softplus", "raw"] = "log_odds"
    """H200 defaults to log_odds. AMP pairs LSGAN/GAN/WGAN with amp/softplus/raw."""
