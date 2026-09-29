"""UFO motion command: thin wrapper around the NPZ-backed motion library.

Keeps the original UFO sampling semantics: one (motion_id, motion_time) pair
per environment, motion time advances with the control dt, uniform resampling
happens on episode reset.
"""

from __future__ import annotations

import torch
from dataclasses import MISSING
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils.configclass import configclass
from omegaconf import DictConfig

from robolab.tasks.manager_based.ufo.motion.motion_lib_npz import MotionLibNPZ


class UFOMotionCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        motion_cfg = DictConfig(
            {
                "motion_file": cfg.manifest_path,
                "step_dt": cfg.step_dt,
                "clip_seconds": cfg.clip_seconds,
                "min_clip_seconds": cfg.min_clip_seconds,
            }
        )
        self._motion_lib = MotionLibNPZ(motion_cfg, num_envs=self.num_envs, device=self.device)
        self._motion_lib.load_motions_for_training()
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_times = torch.zeros(self.num_envs, device=self.device)
        self.truncate_time = cfg.truncate_time_s

    @property
    def motion_lib(self) -> MotionLibNPZ:
        return self._motion_lib

    def _resample_command(self, env_ids):
        if len(env_ids) == 0:
            return
        self.motion_ids[env_ids] = self._motion_lib.sample_motions(len(env_ids))
        times = self._motion_lib.sample_time(self.motion_ids[env_ids], truncate_time=self.truncate_time)
        self.motion_times[env_ids] = times.to(self.device)

    def _update_command(self):
        self.motion_times += self._env.step_dt

    def get_motion_state(self, motion_ids=None, motion_times=None):
        ids = self.motion_ids if motion_ids is None else motion_ids
        times = self.motion_times if motion_times is None else motion_times
        return self._motion_lib.get_motion_state(ids, times)

    @property
    def command(self):
        return torch.stack([self.motion_ids.float(), self.motion_times], dim=-1)

    def _update_metrics(self):
        self.metrics["motion_time"] = self.motion_times.clone()


@configclass
class UFOMotionCommandCfg(CommandTermCfg):
    class_type: type = UFOMotionCommand

    asset_name: str = MISSING
    # Path to the BeyondMimic-format motion manifest (its ``npz`` clips are
    # loaded by MotionLibNPZ).
    manifest_path: str = MISSING
    step_dt: float = 0.02
    truncate_time_s: float | None = None
    # Original-UFO clip windowing: motions are split into fixed-duration
    # non-overlapping windows at load time (humanoidverse clip_seconds=10.0 /
    # min_clip_seconds=1.0). 0 disables windowing (full-motion granularity).
    clip_seconds: float = 10.0
    min_clip_seconds: float = 1.0
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    debug_vis: bool = False
