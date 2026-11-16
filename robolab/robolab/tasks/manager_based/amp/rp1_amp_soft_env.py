"""Walking and interaction populations in one simulation batch and shared actor."""

import torch

from .amp_env import AmpEnv


class RP1AmpSoftEnv(AmpEnv):
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        command = self.command_manager.get_term('base_velocity')
        if env_ids is None or isinstance(env_ids, slice):
            ids = torch.arange(self.num_envs, device=self.device)[slice(None) if env_ids is None else env_ids]
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        ids = ids[command.compliance_envs[ids]]
        if not len(ids):
            return
        # Compliance starts from standing; motion RSI remains on the speed group.
        robot = self.scene['robot']
        pose = robot.data.default_root_state.torch[ids, :7].clone()
        pose[:, :3] += self.scene.env_origins[ids]
        robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(ids), 6, device=self.device), env_ids=ids)
        position = robot.data.default_joint_pos.torch[ids].clone()
        arm = self.reward_manager.get_term_cfg('arm_compliance').func
        if arm.rest_positions is not None:
            for side, arm_ids in enumerate(arm.joint_ids):
                position[:, arm_ids] = arm.rest_positions[side]
        robot.write_joint_position_to_sim_index(position=position, env_ids=ids)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros(len(ids), robot.num_joints, device=self.device), env_ids=ids)
