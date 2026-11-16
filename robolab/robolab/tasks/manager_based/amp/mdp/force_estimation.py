"""Force-estimator targets, force-vel-target supervision, and compact critic state."""

import torch

from rsl_rl.modules.force_estimator import COMPACT_FORCE_DIM, mirror_compact_force

COMPACT_CRITIC_DIM = 48


def force_vel_target(env):
    """Force-follow vx, vy, yaw_rate. User commands stay on the actor command channel."""
    command = env.command_manager.get_command('base_velocity')
    manager = getattr(env, 'reward_manager', None)
    if manager is None:
        return torch.zeros_like(command)
    linear = manager.get_term_cfg('track_lin_vel_xy_exp').func
    yaw = manager.get_term_cfg('track_ang_vel_z_exp').func
    return torch.cat((linear.force_target_velocity_b, yaw.force_target_yaw_rate[:, None]), -1).detach().clone()


def mirror_force_vel_target(target):
    if target.shape[-1] != 3:
        raise ValueError('Force-vel target requires vx, vy, yaw_rate')
    return target * target.new_tensor([1., -1., -1.])


def compact_interaction_critic_state(env):
    """Privileged interaction state, 48-D:

    0:3 root Fxyz/50, 3 root Mz/10, 4:14 tau10/10,
    14:17 filtered heading net force/50, 17 force-magnitude/50, 18 filtered yaw/10,
    19:25 heading per-arm force/50, 25:35 filtered tau10/10,
    35:37 per-arm return gate, 37 mean return gate, 38:48 per-joint compliance gate.
    """
    state = torch.zeros(env.num_envs, COMPACT_CRITIC_DIM, device=env.device)
    manager = getattr(env, 'reward_manager', None)
    events = getattr(env, 'event_manager', None)
    if manager is None or events is None:
        return state
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat
    push = events.get_term_cfg('sustained_standing_push').func
    body = manager.get_term_cfg('track_lin_vel_xy_exp').func
    yaw = manager.get_term_cfg('track_ang_vel_z_exp').func
    arm = manager.get_term_cfg('arm_compliance').func
    recovery = manager.get_term_cfg('full_body_return').func
    heading = yaw_quat(env.scene['robot'].data.root_quat_w.torch)
    labels = push.compact_labels
    if labels.shape[-1] != COMPACT_FORCE_DIM:
        raise ValueError(f'Compact labels must be {COMPACT_FORCE_DIM} (fxyz, mxyz, tau10, region9, load3)')
    state[:, :3] = labels[:, :3] / 50.
    state[:, 3] = labels[:, 5] / 10.
    state[:, 4:14] = labels[:, 6:16] / 10.
    state[:, 14:17] = quat_apply_inverse(heading, body.filtered_force_w) / 50.
    state[:, 17] = body.filtered_force_magnitude / 50.
    state[:, 18] = yaw.filtered_torque_z / 10.
    state[:, 19:25] = (quat_apply_inverse(heading[:, None], arm.filtered_force_w) / 50.).flatten(1)
    state[:, 25:35] = arm.filtered_joint_torque.flatten(1) / 10.
    state[:, 35:37] = recovery.arm_gate
    state[:, 37] = recovery.gate
    state[:, 38:48] = arm.passive_observed_gate.flatten(1)
    return state.detach()


def mirror_compact_critic_state(state):
    """Reflect polar forces, axial moments, signed arm torques and scalar gates."""
    if state.shape[-1] != COMPACT_CRITIC_DIM:
        raise ValueError(f'Compact interaction critic requires {COMPACT_CRITIC_DIM} channels')
    result = state.clone()
    result[..., :14] = mirror_compact_force(state[..., :14])
    result[..., 15] *= -1
    result[..., 18] *= -1
    result[..., 19:25] = state[..., [22, 23, 24, 19, 20, 21]]
    result[..., [20, 23]] *= -1
    result[..., 25:35] = -state[..., [30, 31, 32, 33, 34, 25, 26, 27, 28, 29]]
    result[..., 35:37] = state[..., [36, 35]]
    result[..., 38:48] = state[..., [43, 44, 45, 46, 47, 38, 39, 40, 41, 42]]
    return result


def compact_interaction_target(env):
    return env.event_manager.get_term_cfg('sustained_standing_push').func.compact_labels.clone()


def configure_force_estimation(env_cfg, agent_cfg):
    """One actor switch configures the estimator's history and target observations."""
    actor = getattr(agent_cfg, 'actor', None)
    if not getattr(actor, 'enable_force_estimation', False):
        return
    from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
    from isaaclab.utils import configclass
    if getattr(env_cfg.events, 'sustained_standing_push', None) is None:
        raise ValueError('Force estimation requires sustained_standing_push supervision')
    if not (actor.force_estimation_compact and actor.force_estimation_multi_contact
            and actor.force_estimation_with_torque):
        raise ValueError('sustained_standing_push only supervises the compact multi-contact wrench estimator')
    if not agent_cfg.algorithm.enable_aux_loss or agent_cfg.algorithm.aux_loss_coef <= 0:
        raise ValueError('Force estimation requires enabled auxiliary losses')
    if agent_cfg.actor.force_estimation_history_length < 1:
        raise ValueError('force_estimation_history_length must be positive')

    @configclass
    class TargetCfg(ObservationGroupCfg):
        target = ObservationTermCfg(func=compact_interaction_target)
        enable_corruption = False
        concatenate_terms = True

    shared_history = env_cfg.observations.policy.history_length == agent_cfg.actor.force_estimation_history_length
    if not shared_history:
        history = env_cfg.observations.policy.copy()
        history.history_length = agent_cfg.actor.force_estimation_history_length
        env_cfg.observations.force_history = history
    env_cfg.observations.force_target = TargetCfg()
    env_cfg.force_estimation_compact = True
    agent_cfg.obs_groups['force_history'] = ['policy'] if shared_history else ['force_history']
    agent_cfg.obs_groups['force_target'] = ['force_target']
    if getattr(actor, 'enable_force_vel_target_estimation', False):

        @configclass
        class ForceVelTargetCfg(ObservationGroupCfg):
            target = ObservationTermCfg(func=force_vel_target)
            enable_corruption = False
            concatenate_terms = True

        env_cfg.observations.force_vel_target = ForceVelTargetCfg()
        agent_cfg.obs_groups['force_vel_target'] = ['force_vel_target']
