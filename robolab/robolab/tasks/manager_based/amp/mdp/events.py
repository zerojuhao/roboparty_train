from __future__ import annotations

from typing import TYPE_CHECKING
import math

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg, ManagerTermBase
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from rsl_rl.modules.force_estimator import (
        ARM_JOINT_NAMES, COMPACT_FORCE_DIM, MULTI_CONTACT_BODIES, REGION_LOAD_FORCE_N,
        compact_estimator_target, compact_wrench_target, regional_forces)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class sustained_standing_push(ManagerTermBase):
    """Smooth, finite external loads on the arms and torso of interaction environments.

    Every load event independently draws arm point pulls, a horizontal torso pull, a torso yaw
    torque and, at zero command, one arm joint torque. Called every control step by a
    zero-interval event. Owns the robot's permanent wrench buffer; do not combine with another
    permanent-wrench event.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        p = cfg.params
        for name, value in p.items():
            if name.endswith('_probability') and not 0 <= value <= 1:
                raise ValueError(f'{name} must be in [0, 1]')
            if name.endswith(('_range_s', 'count_range')) and not (
                    0 < value[0] <= value[1] < math.inf):
                raise ValueError(f'{name} must contain finite positive ordered bounds')
            if name.endswith(('_range_n', '_range_nm')) and not (0 <= value[0] <= value[1] < math.inf):
                raise ValueError(f'{name} must contain finite nonnegative ordered bounds')
        for name in ('arm_joint_limit_margin_rad', 'max_total_force_ratio'):
            if not 0 < p[name] < math.inf:
                raise ValueError(f'{name} must be finite and positive')
        if p['ramp_time_range_s'][1] > p['duration_range_s'][0] / 2:
            raise ValueError('The longest ramp must fit within half of the shortest load')
        self.robot = env.scene[p['asset_cfg'].name]
        self.body_ids = p['asset_cfg'].body_ids
        if isinstance(self.body_ids, slice) or tuple(
                self.robot.body_names[i] for i in self.body_ids) != MULTI_CONTACT_BODIES:
            raise ValueError('sustained_standing_push requires the ordered multi-contact bodies')
        if p['contact_count_range'][1] > 10:
            raise ValueError('contact_count_range cannot exceed the ten arm links')

        n, device = env.num_envs, env.device
        self.ids = torch.arange(n, device=device)
        self.active = torch.zeros(n, dtype=torch.bool, device=device)
        self.abrupt_release = torch.zeros_like(self.active)
        self.elapsed = torch.zeros(n, device=device)
        self.duration = torch.zeros(n, device=device)
        self.ramp_duration = torch.ones(n, device=device)
        self.wait = torch.zeros(n, device=device)
        # Sampled loads: world-frame link forces at link-frame offsets, torque about the torso
        # link z axis, and generalized arm joint torques in ARM_JOINT_NAMES order.
        self.force = torch.zeros(n, 11, 3, device=device)
        self.point_offset = torch.zeros_like(self.force)
        self.torso_yaw_torque = torch.zeros(n, device=device)
        self.arm_joint_torque = torch.zeros(n, 10, device=device)
        # Loads currently applied, after ramping, the force budget and the joint-limit fade.
        self.applied_force = torch.zeros_like(self.force)
        self.applied_torque = torch.zeros_like(self.force)
        self.applied_joint_torque = torch.zeros_like(self.arm_joint_torque)
        self.explicit_torso_torque_w = torch.zeros(n, 3, device=device)
        self.applied_positions = torch.zeros_like(self.force)
        self.compact_labels = torch.zeros(n, COMPACT_FORCE_DIM, device=device)

        # Body slots of each arm joint's child link, parent link and same-side wrist link.
        joints = self.robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)[0]
        if len(joints) != 10:
            raise ValueError('sustained_standing_push requires the ten RP1 arm joints')
        child = [MULTI_CONTACT_BODIES.index(name.replace('_joint', '_link')) for name in ARM_JOINT_NAMES]
        self.arm_joints = torch.tensor(joints, device=device)
        self.joint_child = torch.tensor(child, device=device)
        self.joint_parent = torch.tensor([10 if i % 5 == 0 else child[i - 1] for i in range(10)], device=device)
        self.joint_wrist = torch.tensor([child[i - i % 5 + 4] for i in range(10)], device=device)
        # Jacobian rows skip the fixed base; columns of a floating base start after its six DOFs.
        fixed = self.robot.is_fixed_base
        self.bodies = torch.tensor(self.body_ids, device=device)
        self.jacobian_bodies = self.bodies - (1 if fixed else 0)
        self.jacobian_joints = self.arm_joints + (0 if fixed else 6)
        self.reset()

    def reset(self, env_ids=None):
        ids = self.ids if env_ids is None else self.ids[env_ids]
        for buffer in (self.active, self.abrupt_release, self.elapsed, self.duration, self.force,
                       self.point_offset, self.torso_yaw_torque, self.arm_joint_torque, self.applied_force,
                       self.applied_torque, self.applied_joint_torque, self.explicit_torso_torque_w,
                       self.compact_labels):
            buffer[ids] = 0
        self.wait[ids] = torch.empty(len(ids), device=self._env.device).uniform_(
            *self.cfg.params['initial_rest_range_s'])
        self.robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=self.applied_force[ids], torques=self.applied_torque[ids],
            body_ids=self.body_ids, env_ids=ids, is_global=True)

    def __call__(self, env, env_ids, asset_cfg, command_name,
                 point_force_probability, point_force_range_n, contact_count_range, point_offset_range,
                 coherent_pull_probability, arm_couple_probability, small_horizontal_pull_probability,
                 torso_horizontal_pull_probability, torso_horizontal_force_range_n,
                 torso_yaw_probability, zero_command_torso_yaw_probability, torso_yaw_torque_range_nm,
                 arm_joint_torque_probability, arm_joint_torque_range_nm, arm_joint_limit_margin_rad,
                 max_total_force_ratio, duration_range_s, long_duration_probability, long_duration_range_s,
                 rest_range_s, initial_rest_range_s, ramp_time_range_s, abrupt_release_probability,
                 release_reserve_s):
        device, data = env.device, self.robot.data

        def uniform(bounds, *shape):
            return torch.empty(shape, device=device).uniform_(*bounds)

        self.elapsed += env.step_dt * self.active
        finished = self.active & (self.elapsed >= self.duration)
        self.active &= ~finished
        self.wait -= env.step_dt * ~self.active
        self.wait[finished] = uniform(rest_range_s, int(finished.sum()))
        # Leave an unloaded tail before timeout, including after long loads.
        available = env.max_episode_length_s - env.episode_length_buf * env.step_dt - release_reserve_s
        start = (env.command_manager.get_term(command_name).interaction_envs & ~self.active
                 & (self.wait <= 0) & (available >= duration_range_s[0]))
        ids = self.ids[start]
        if len(ids):
            k = len(ids)
            rows = torch.arange(k, device=device)
            zero_command = (env.command_manager.get_command(command_name)[ids] == 0).all(-1)

            def chance(probability):
                return torch.rand(k, device=device) < probability

            def sign():
                return 1. - 2. * torch.randint(2, (k,), device=device)

            def horizontal(bounds):
                heading = uniform((-math.pi, math.pi), k)
                return uniform(bounds, k)[:, None] * torch.stack((heading.cos(), heading.sin()), dim=-1)

            self.point_offset[ids] = uniform(point_offset_range, k, 11, 3)

            # Arm point pulls on distinct links, in uniformly random directions.
            low, high = contact_count_range
            count = torch.randint(low, high + 1, (k,), device=device)
            pulls = torch.randn(k, high, 3, device=device)
            pulls *= (uniform(point_force_range_n, k, high) / pulls.norm(dim=-1).clamp_min(1e-6))[..., None]
            pulls *= (torch.arange(high, device=device) < count[:, None])[..., None]
            force = torch.zeros(k, 11, 3, device=device)
            force[rows[:, None], torch.rand(k, 10, device=device).topk(high).indices] = pulls

            # Coherent pulls: every loaded link pulls within 30 degrees of one random axis.
            coherent = chance(coherent_pull_probability)
            axis = torch.nn.functional.normalize(torch.randn(k, 3, device=device), dim=-1)
            helper = torch.where(axis[:, 2:].abs() > .9,
                                 axis.new_tensor([1., 0., 0.]), axis.new_tensor([0., 0., 1.]))
            tangent = torch.nn.functional.normalize(torch.cross(axis, helper, dim=-1), dim=-1)
            bitangent = torch.cross(axis, tangent, dim=-1)
            cosine = uniform((math.cos(math.pi / 6), 1.), k, 11, 1)
            azimuth = uniform((-math.pi, math.pi), k, 11, 1)
            direction = cosine * axis[:, None] + (1 - cosine.square()).sqrt() * (
                azimuth.cos() * tangent[:, None] + azimuth.sin() * bitangent[:, None])
            force = torch.where(coherent[:, None, None], force.norm(dim=-1, keepdim=True) * direction, force)

            # Arm couple: opposite pulls on the wrist and on one arm joint's parent link load that
            # joint; the remaining contacts keep their strongest pulls on other arm links.
            joint = torch.randint(10, (k,), device=device)
            pair = torch.stack((self.joint_wrist[joint], self.joint_parent[joint]), dim=1)
            offset = quat_apply(data.body_quat_w.torch[ids[:, None], self.bodies[pair]],
                                self.point_offset[ids[:, None], pair])
            jacobian = data.body_com_jacobian_w.torch[
                ids[:, None], self.jacobian_bodies[pair], :, self.jacobian_joints[joint][:, None]]
            # Motion of the two pull points per unit joint velocity; the couple acts along their difference.
            point_motion = jacobian[..., :3] + torch.cross(jacobian[..., 3:], offset, dim=-1)
            lever = point_motion[:, 0] - point_motion[:, 1]
            couple = (count >= 2) & ~coherent & chance(arm_couple_probability) & (lever.norm(dim=-1) > 1e-3)
            pull = torch.nn.functional.normalize(lever, dim=-1) * (pulls[:, 0].norm(dim=-1) * sign())[:, None]
            strongest = pulls.norm(dim=-1).argsort(dim=-1, descending=True)[..., None].expand(-1, -1, 3)
            others = torch.rand(k, 11, device=device).scatter_(1, pair, -1.)
            others[:, 10] = -1.
            coupled = torch.zeros_like(force)
            coupled[rows[:, None], others.topk(high).indices] = pulls.gather(1, strongest) * (
                torch.arange(high, device=device) < count[:, None] - 2)[..., None]
            coupled[rows, pair[:, 0]] = pull
            coupled[rows, pair[:, 1]] = -pull
            force = torch.where(couple[:, None, None], coupled, force)

            # Small pull: one horizontal pull on a single arm link replaces the point pulls.
            single = torch.zeros_like(force)
            single[rows, torch.randint(10, (k,), device=device), :2] = horizontal(point_force_range_n)
            force = torch.where(chance(small_horizontal_pull_probability)[:, None, None], single, force)
            force *= chance(point_force_probability)[:, None, None]

            # Torso pull and arm joint torque start only at zero command; torso yaw is likelier there.
            force[:, 10, :2] += horizontal(torso_horizontal_force_range_n) * (
                zero_command & chance(torso_horizontal_pull_probability))[:, None]
            self.force[ids] = force
            self.torso_yaw_torque[ids] = sign() * uniform(torso_yaw_torque_range_nm, k) * chance(
                torch.where(zero_command, zero_command_torso_yaw_probability, torso_yaw_probability))
            torque = torch.zeros(k, 10, device=device)
            torque[rows, torch.randint(10, (k,), device=device)] = sign() * uniform(arm_joint_torque_range_nm, k) * (
                zero_command & chance(arm_joint_torque_probability))
            self.arm_joint_torque[ids] = torque

            long = chance(long_duration_probability) & (available[ids] >= long_duration_range_s[0])
            duration = torch.where(long, uniform(long_duration_range_s, k), uniform(duration_range_s, k))
            self.duration[ids] = torch.minimum(duration, available[ids])
            self.ramp_duration[ids] = uniform(ramp_time_range_s, k)
            self.abrupt_release[ids] = chance(abrupt_release_probability)
            self.elapsed[ids] = 0
            self.active[ids] = True

        # Smooth ramp in and, unless released abruptly, out; the summed point force stays within budget.
        ramp = (torch.where(self.abrupt_release, self.elapsed, torch.minimum(self.elapsed, self.duration - self.elapsed))
                / self.ramp_duration).clamp(0, 1)
        scale = ramp.square() * (3 - 2 * ramp) * self.active
        body_weight = data.body_mass.torch.sum(dim=1) * abs(env.cfg.sim.gravity[2])
        budget = (body_weight * max_total_force_ratio / self.force.norm(dim=-1).sum(-1).clamp_min(1e-6)).clamp_max(1)
        self.applied_force = self.force * (scale * budget)[:, None, None]

        # Like a hand easing off: a joint torque fades to zero as it pushes into the soft limit.
        position = data.joint_pos.torch[:, self.arm_joints]
        limits = data.soft_joint_pos_limits.torch[:, self.arm_joints]
        margin = (.5 * (limits[..., 1] - limits[..., 0])).clamp(1e-6, arm_joint_limit_margin_rad)
        free = (torch.where(self.arm_joint_torque > 0, limits[..., 1] - position, position - limits[..., 0])
                / margin).clamp(0, 1)
        self.applied_joint_torque = self.arm_joint_torque * scale[:, None] * free.square() * (3 - 2 * free)

        # Torso yaw torque, plus equal and opposite child/parent moments realizing each joint torque.
        quat = data.body_quat_w.torch[:, self.body_ids]
        yaw = torch.zeros_like(self.explicit_torso_torque_w)
        yaw[:, 2] = self.torso_yaw_torque * scale
        self.explicit_torso_torque_w = quat_apply(quat[:, 10], yaw)
        self.applied_torque = torch.zeros_like(self.force)
        self.applied_torque[:, 10] = self.explicit_torso_torque_w
        moments = data.body_com_jacobian_w.torch[
            self.ids[:, None], self.jacobian_bodies[self.joint_child], 3:, self.jacobian_joints
        ] * self.applied_joint_torque[..., None]
        self.applied_torque.index_add_(1, self.joint_child, moments)
        self.applied_torque.index_add_(1, self.joint_parent, -moments)

        self.applied_positions = data.body_com_pos_w.torch[:, self.body_ids] + quat_apply(quat, self.point_offset)
        self.robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=self.applied_force, torques=self.applied_torque, positions=self.applied_positions,
            body_ids=self.body_ids, env_ids=self.ids, is_global=True)

    def current_compact_wrench(self):
        """World-frame net wrench and generalized arm torques at the current geometry."""
        data = self.robot.data
        com = data.body_com_pos_w.torch[:, self.body_ids]
        moment = self.applied_torque + torch.cross(self.applied_positions - com, self.applied_force, dim=-1)
        jacobian = data.body_com_jacobian_w.torch[:, self.jacobian_bodies][..., self.jacobian_joints]
        return compact_wrench_target(self.applied_force, moment, com, data.root_pos_w.torch, jacobian)

    def capture_compact_labels(self):
        """Snapshot root-frame net wrench, regional forces, and load bits."""
        root = self.robot.data.root_quat_w.torch
        target = self.current_compact_wrench()
        force_b = quat_apply_inverse(root, target[:, :3])
        moment_b = quat_apply_inverse(root, target[:, 3:6])
        region_b = quat_apply_inverse(root[:, None], regional_forces(self.applied_force))
        load = (region_b.norm(dim=-1) >= REGION_LOAD_FORCE_N).to(force_b.dtype)
        self.compact_labels.copy_(
            compact_estimator_target(force_b, moment_b, target[:, 6:16], region_b, load))


def reset_from_ref(
    env: ManagerBasedEnv,
    env_ids,
    animation: str = "animation",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Write clip joints, orientation, and velocity. Keep reset_base xy; z from robot init_state.

    Motion clip root height is not used: clips can float or penetrate. Ground contact
    comes from ``ArticulationCfg.init_state.pos[2]``.
    """
    if env_ids is None or env_ids == slice(None):
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    elif torch.is_tensor(env_ids):
        env_ids = env_ids.to(device=env.device, dtype=torch.long)
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    animation_term = env.animation_manager.get_term(animation)
    robot: Articulation = env.scene[asset_cfg.name]
    joint_pos = animation_term.get_dof_pos(env_ids)[:, 0].clone()
    joint_vel = animation_term.get_dof_vel(env_ids)[:, 0]
    soft_limits = robot.data.soft_joint_pos_limits.torch[env_ids]
    joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
    robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

    root_states = robot.data.default_root_state.torch[env_ids].clone()
    root_xy = robot.data.root_link_state_w.torch[env_ids, :2]
    root_states[:, 0:2] = root_xy
    root_states[:, 2] = env.scene.env_origins[env_ids, 2] + float(robot.cfg.init_state.pos[2])
    root_states[:, 3:7] = animation_term.get_root_quat(env_ids)[:, 0]
    root_states[:, 7:10] = animation_term.get_root_vel_w(env_ids)[:, 0]
    root_states[:, 10:13] = animation_term.get_root_ang_vel_w(env_ids)[:, 0]
    robot.write_root_pose_to_sim_index(root_pose=root_states[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim_index(root_velocity=root_states[:, 7:13], env_ids=env_ids)
