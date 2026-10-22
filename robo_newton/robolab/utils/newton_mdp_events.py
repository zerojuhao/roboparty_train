"""Newton-only event terms shared by the training tasks.

The IsaacLab generic CoM randomizer reads ``body_com_pose_b`` before choosing
the backend. Newton stores only the CoM position and synthesizes an identity
quaternion for that property, so use the native position-only API here.
"""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils import math as math_utils
from robolab.utils.tensors import resolve_env_ids


class _RandomizeRigidBodyComNewton(ManagerTermBase):
    """Randomize body CoM positions without materializing Newton quaternions."""

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]
        self.default_com_pos = None

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        com_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg,
    ):
        if self.default_com_pos is None:
            self.default_com_pos = self.asset.data.body_com_pos_b.torch.clone()

        env_ids = resolve_env_ids(env_ids, env.scene.num_envs, self.asset.device)

        if self.asset_cfg.body_ids == slice(None):
            body_ids = torch.arange(self.asset.num_bodies, dtype=torch.int, device=self.asset.device)
        else:
            body_ids = torch.as_tensor(self.asset_cfg.body_ids, dtype=torch.int, device=self.asset.device)

        range_list = [com_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z")]
        ranges = torch.tensor(range_list, device=self.asset.device)
        rand_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=self.asset.device
        ).unsqueeze(1)

        com_pos = self.default_com_pos.clone()
        com_pos[env_ids[:, None], body_ids] += rand_samples
        self.asset.set_coms_index(coms=com_pos[env_ids[:, None], body_ids], body_ids=body_ids, env_ids=env_ids)


randomize_rigid_body_com = _RandomizeRigidBodyComNewton

__all__ = ["randomize_rigid_body_com"]
