"""PUFO motion and recovery starts, with reset-only contact clearance."""

import math

import torch
import warp as wp

from .events import TERRAIN_RESET_CLEARANCE
from ..motion.torch_utils import my_quat_rotate, quat_from_angle_axis, quat_mul
from robolab.utils.tensors import as_torch_tensor


def sample_reset_states(
    motion, env_ids, origins, default_joints, *,
    lie_down_probability=0.3,
    standing_startup_probability=0.0,
    joint_noise_range=(0.0, 0.0),
    arm_joint_noise_range=(0.0, 0.0),
    arm_joint_indices=None,
):
    """Pure pose construction; never change motion tensors or joint reference zeros.

    standing_startup_probability: start from default standing with zero root velocity
    while the motion command/z pairing still points at the sampled clip frame. This
    targets deploy-style default→hold-frame0 gaps without changing the expert z path.
    """
    count, device = len(env_ids), origins.device
    root = torch.zeros(count, 13, device=device)
    root[:, :2] = origins[env_ids, :2]
    root[:, 2] = motion["rg_pos_t"][env_ids, 0, 2]
    axis = torch.zeros(count, 3, device=device)
    axis[:, 2] = 1
    yaw = quat_from_angle_axis(torch.empty(count, device=device).uniform_(-torch.pi, torch.pi), axis, w_last=True)
    root[:, 3:7] = quat_mul(yaw, motion["rg_rot_t"][env_ids, 0], w_last=True)
    root[:, 7:10] = my_quat_rotate(yaw, motion["body_vel_t"][env_ids, 0])
    root[:, 10:13] = my_quat_rotate(yaw, motion["body_ang_vel_t"][env_ids, 0])
    joints = motion["dof_pos"][env_ids].clone()
    velocities = motion["dof_vel"][env_ids].clone()
    lying = torch.rand(count, device=device) < lie_down_probability
    standing = (~lying) & (torch.rand(count, device=device) < standing_startup_probability)
    # Standing: default joints, upright yaw-only root, zero spatial velocity.
    if standing.any():
        joints[standing] = default_joints[env_ids][standing]
        velocities[standing] = 0
        root[standing, 3:7] = yaw[standing]
        # Keep a nominal standing height from default pose (~motion standing z when available).
        root[standing, 2] = motion["rg_pos_t"][env_ids, 0, 2][standing].clamp(min=0.7, max=0.9)
        root[standing, 7:] = 0
    poses = torch.randint(4, (count,), device=device)
    axes = torch.zeros_like(axis)
    axes[poses < 2, 0] = 1
    axes[poses >= 2, 1] = 1
    angles = torch.where(poses % 2 == 0, -torch.pi / 2, torch.pi / 2)
    rotations = quat_mul(yaw, quat_from_angle_axis(angles, axes, w_last=True), w_last=True)
    root[lying, 3:7] = rotations[lying]
    root[lying, 2] = root.new_tensor((0.30, 0.30, 0.21, 0.26))[poses[lying]]
    root[lying, 7:] = 0
    joints[lying] = default_joints[env_ids][lying]
    velocities[lying] = 0
    # Joint noise after pose selection (skip lying which already uses defaults cleanly).
    active = ~lying
    if active.any() and joint_noise_range != (0.0, 0.0):
        lo, hi = joint_noise_range
        joints[active] += torch.empty_like(joints[active]).uniform_(lo, hi)
    if active.any() and arm_joint_noise_range != (0.0, 0.0) and arm_joint_indices:
        lo, hi = arm_joint_noise_range
        idx = torch.as_tensor(arm_joint_indices, device=device, dtype=torch.long)
        joints[active][:, idx] += torch.empty(int(active.sum()), len(idx), device=device).uniform_(lo, hi)
    root[:, 2] += TERRAIN_RESET_CLEARANCE
    return root, joints, velocities, lying, standing


def correct_terrain_penetration(env, env_ids, root, clearance=0.05, penetration_scale=1.25):
    """Use one fresh contact pass and a conservative lift, without advancing time."""
    import mujoco_warp as mjwarp
    from isaaclab_newton.physics.newton_manager import NewtonManager

    if not math.isfinite(clearance) or clearance < 0:
        raise ValueError("Reset contact clearance must be finite and non-negative")
    if not math.isfinite(penetration_scale) or penetration_scale < 1:
        raise ValueError("Reset penetration scale must be finite and at least one")
    if len(env_ids) == 0:
        return
    solver = NewtonManager._solver
    if solver is None or not hasattr(solver, "mjw_data"):
        raise RuntimeError("PUFO contact-aware reset requires the Newton MuJoCo Warp backend")
    terrain = env.scene.terrain
    from robolab.terrains.native_importer import validate_terrain_colliders
    validate_terrain_colliders(env)
    robot = env.scene["robot"]
    dynamic_geom = torch.as_tensor(solver.mj_model.geom_bodyid > 0, device=env.device)
    env._pufo_reset_lift[env_ids] = 0
    env.scene.write_data_to_sim()
    env.sim.forward()
    # Newton forward() only runs FK. Recompute contacts at the sampled pose,
    # not contacts left over from the previous episode.
    solver._update_mjc_data(solver.mjw_data, solver.model, NewtonManager.backend.state_0)
    mjwarp.fwd_position(solver.mjw_model, solver.mjw_data)
    from robolab.utils.physics_memory import check_solver_overflow

    check_solver_overflow(solver.mjw_data)
    data = solver.mjw_data
    count = int(wp.to_torch(data.nacon)[0].item())
    if count == 0:
        return
    pairs = wp.to_torch(data.contact.geom)[:count].long()
    worlds = wp.to_torch(data.contact.worldid)[:count].long()
    distances = wp.to_torch(data.contact.dist)[:count]
    valid = ((pairs >= 0) & (pairs < len(dynamic_geom))).all(-1)
    valid &= (worlds >= 0) & (worlds < env.num_envs) & (distances < 0)
    safe = pairs.clamp(0, len(dynamic_geom) - 1)
    valid &= dynamic_geom[safe[:, 0]] ^ dynamic_geom[safe[:, 1]]
    penetration = torch.zeros(env.num_envs, device=env.device)
    penetration.scatter_reduce_(0, worlds[valid], -distances[valid], reduce="amax")
    penetration = penetration[env_ids]
    correction = torch.where(penetration > 0, penetration * penetration_scale + clearance, 0)
    if torch.any(correction > 0):
        root[:, 2] += correction
        robot.write_root_pose_to_sim_index(root_pose=root[:, :7], env_ids=env_ids)
    env._pufo_reset_lift[env_ids] = correction


def reset_pufo_from_motion(
    env, env_ids, command_name, lie_down_probability=0.3,
    contact_clearance=0.05, penetration_scale=1.25,
    standing_startup_probability=0.0,
    joint_noise_range=(0.0, 0.0),
    arm_joint_noise_range=(0.0, 0.0),
    arm_joint_indices=None,
    hold_motion_on_standing=True,
):
    if len(env_ids) == 0:
        return
    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    robot = env.scene["robot"]
    motion = env.command_manager.get_term(command_name).get_motion_state()
    root, joints, velocities, lying, standing = sample_reset_states(
        motion, env_ids, env.scene.env_origins, as_torch_tensor(robot.data.default_joint_pos),
        lie_down_probability=lie_down_probability,
        standing_startup_probability=standing_startup_probability,
        joint_noise_range=joint_noise_range,
        arm_joint_noise_range=arm_joint_noise_range,
        arm_joint_indices=arm_joint_indices,
    )
    desired_clearance = root[:, 2].clone()
    root[:, 2] += env.scene.env_origins[env_ids, 2]
    robot.write_joint_position_to_sim_index(position=joints, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=velocities, env_ids=env_ids)
    robot.write_root_pose_to_sim_index(root_pose=root[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim_index(root_velocity=root[:, 7:], env_ids=env_ids)
    if env.scene.terrain.cfg.terrain_type == "generator":
        from isaaclab.utils.warp import raycast_mesh
        from isaaclab.utils.math import quat_apply_yaw

        sensor = env.scene.sensors["terrain_priv"]
        pattern = sensor.cfg.pattern_cfg
        starts, _ = pattern.func(pattern, env.device)
        xy = starts[:, :2] + starts.new_tensor(sensor.cfg.offset.pos[:2])
        radii = torch.linalg.vector_norm(xy, dim=-1)
        offsets = torch.cat((xy, torch.zeros_like(xy[:, :1])), dim=-1)
        rays = offsets.unsqueeze(0).expand(len(env_ids), -1, -1).clone()
        rays = quat_apply_yaw(root[:, None, 3:7].expand(-1, len(offsets), -1).contiguous(), rays)
        rays[:, :, :2] += root[:, None, :2]
        # Reset probes use a tile-relative origin independent of the sampled jump
        # height and never enter the policy history.
        rays[:, :, 2] = env.scene.env_origins[env_ids, None, 2] + 10.0
        directions = torch.zeros_like(rays)
        directions[:, :, 2] = -1
        mesh = sensor.meshes[(sensor.cfg.mesh_prim_paths[0], sensor._device)]
        heights = raycast_mesh(rays, directions, mesh, max_dist=20.)[0][..., 2]
        ground = heights[:, radii.argmin()]
        if not torch.isfinite(ground).all():
            raise RuntimeError("PUFO reset ground query missed the terrain")
        footprint = torch.where(torch.isfinite(heights[:, radii <= 0.65]),
                                heights[:, radii <= 0.65], -torch.inf).amax(-1)
        delta = torch.relu(footprint - ground)
        lift = torch.where(delta > 1e-4, delta + 0.05, 0)
        root[:, 2] = ground + desired_clearance + lift
        robot.write_root_pose_to_sim_index(root_pose=root[:, :7], env_ids=env_ids)
    if not hasattr(env, "_pufo_reset_lift"):
        env._pufo_reset_lift = torch.zeros(env.num_envs, device=env.device)
        env._pufo_reset_lying = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._pufo_reset_lying[env_ids] = lying
    if not hasattr(env, "_pufo_reset_standing"):
        env._pufo_reset_standing = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._pufo_reset_standing[env_ids] = standing
    if hold_motion_on_standing and standing.any():
        command = env.command_manager.get_term(command_name)
        held = env_ids[standing]
        # Freeze at the just-sampled clip time so z pairing stays on that frame.
        if hasattr(command, "hold_motion_mask"):
            command.hold_motion_mask[held] = True
        else:
            # Fallback: pin times; command term may still advance without the mask.
            command.motion_times[held] = command.motion_times[held]
    correct_terrain_penetration(
        env, env_ids, root, clearance=contact_clearance, penetration_scale=penetration_scale
    )
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(dt=0.0)
    refresh_reset_camera(env, env_ids)


def refresh_reset_camera(env, env_ids):
    """Populate history only after the final reset pose is written."""
    camera = env.scene.sensors["camera"]
    camera.reset(env_ids)
    camera.update(dt=0.0, force_recompute=True)
