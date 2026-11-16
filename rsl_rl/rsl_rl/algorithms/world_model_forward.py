"""Optional F-only successor-measure targets, anchored by a real first transition."""

import torch
from torch.nn import functional as F

from .world_model import quat_apply, yaw_quat
from .world_model_imagination import quality_metrics


@torch.no_grad()
def forward_model_target(model, agent, batch, next_obs, z, discount, next_action, heading_valid):
    cfg = model.cfg
    ready, _ = quality_metrics(model)
    if cfg.forward_weight == 0 or int(model.updates) <= cfg.warmup_updates or not ready:
        return None
    if "world_model" not in batch:
        raise ValueError("F model targets require real physical replay; collect fresh WM replay")
    physical = batch["world_model"]
    horizon, gamma = cfg.forward_horizon, agent.cfg.discount
    futures, terminals, masks = [], [], []
    for start in range(0, len(z), cfg.imagination_microbatch_size):
        end = start + cfg.imagination_microbatch_size
        latent = z[start:end].detach()
        state = physical["next_state"][start:end].to(model.device)
        xi = physical["xi"][start:end].to(model.device)
        obs = {key: value[start:end].detach() for key, value in next_obs.items()}
        valid = torch.cat((heading_valid[start:end].reshape(-1, 1).float(),
                           obs["velocity_command"][:, -1:]), -1)
        frame = yaw_quat(state[:, 1:5])
        heading_local = torch.cat((1 - obs["heading"][:, :1], obs["heading"][:, 1:2]), -1)
        heading = quat_apply(frame, F.pad(heading_local, (0, 1)))[:, :2]
        local_velocity = obs["velocity_command"][:, :-1]
        dim = local_velocity.shape[-1]
        velocity = quat_apply(frame, F.pad(local_velocity, (0, 3 - dim)))[:, :dim]
        trusted = (discount[start:end] > 0) & ~physical["disturbance_mask"][start:end].to(model.device).bool().flatten(1).any(-1)
        anchor_action = next_action[start:end].detach()
        action = anchor_action
        future = torch.zeros_like(latent)
        member = int(model.updates) % cfg.ensemble_size
        for step in range(1, horizon):
            prediction = model.predict(state, action, xi, member)
            # Real next action anchors support; no recorded future action belongs to this fixed-z rollout.
            distance = ((model.codec.env_action(action) - model.codec.env_action(anchor_action))
                        / model.codec.action_rescale).abs().amax(-1)
            trusted &= torch.isfinite(prediction).all(-1)
            trusted &= distance <= cfg.imagination_action_radius
            history = model.codec.advance_history(obs["history_actor"], obs)
            state = prediction
            obs = model.codec.observation(state, model.codec.env_action(action), history,
                                          heading, velocity, valid, noisy=True)
            obs["privileged_state"] = model.kinematics(state)
            normalized = agent._obs_normalizer.normalize_dict(obs)
            with agent._autocast_context():
                embedding = agent._backward_map(agent._privileged_input(normalized)).float()
                action = agent._actor_forward(normalized, latent, imagination=True).sample(clip=agent.cfg.stddev_clip).float()
            future += gamma ** step * embedding
        with agent._autocast_context():
            terminal = agent._target_forward_map(agent._full_input(normalized), latent, action).float()
        trusted &= torch.isfinite(future).all(-1) & torch.isfinite(terminal).all(dim=(0, 2))
        futures.append(torch.where(trusted[:, None], future, 0.))
        terminals.append(torch.where(trusted[None, :, None], terminal, 0.))
        masks.append(trusted)
    weight = cfg.forward_weight * min(1.0, (int(model.updates) - cfg.warmup_updates) / cfg.ramp_updates)
    return {"future": torch.cat(futures), "terminal": torch.cat(terminals, 1),
            "trusted": torch.cat(masks), "discount": discount.detach() * gamma ** (horizon - 1),
            "weight": weight}


def forward_model_loss(agent, fs, b, target_b, target, *, check_finite=True):
    """Fit real s1 + imagined s2..sH occupancy; only the online F receives gradients."""
    with torch.no_grad():
        target_ms = target["terminal"].detach() @ target_b.detach().T
        _, _, bootstrap = agent._targets_uncertainty(target_ms, agent.cfg.fb_pessimism_penalty)
        bootstrap = target["discount"][:, None] * bootstrap
    ms = fs @ b.detach().T
    if target.get("mbfb", False):
        offdiag = .5 * ((ms - bootstrap) * agent.off_diag).square().sum() / agent.off_diag_sum
        linear = (torch.diagonal(ms, dim1=1, dim2=2)
                  + (fs * target["future"].detach()[None]).sum(-1) * target["alive"][None])
        loss = target["weight"] * (offdiag - linear.mean() * fs.shape[0])
        if check_finite and not torch.isfinite(loss):
            raise FloatingPointError("Non-finite MBFB F loss")
        return loss, {"wm/forward_loss": loss.detach(), "wm/forward_weight": loss.new_full((), target["weight"])}
    trusted = target["trusted"].detach()
    squared = ((ms - bootstrap) * agent.off_diag).square()
    offdiag = .5 * torch.where(trusted[None, :, None], squared, 0.).sum() / agent.off_diag_sum
    linear = torch.diagonal(ms, dim1=1, dim2=2) + (fs * target["future"].detach()[None]).sum(-1)
    diagonal = -torch.where(trusted[None], linear, 0.).mean() * fs.shape[0]
    loss = target["weight"] * (offdiag + diagonal)
    if check_finite and not torch.isfinite(loss):
        raise FloatingPointError("Non-finite model-generated F loss")
    return loss, {"wm/forward_loss": loss.detach(), "wm/forward_weight": loss.new_full((), target["weight"]),
                  "wm/forward_trusted_fraction": trusted.float().mean()}
