"""Configuration terms for different managers."""

from __future__ import annotations

import torch
from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

from isaaclab.utils import configclass


@configclass
class MultiRewardCfg:
    """Configuration for a reward group. Please inherit it if you want to define
    your own reward group so that the manager can recognize it.
    """

    pass


@configclass
class DummyRewardCfg:
    """A placeholder for reward cfg."""

    pass
