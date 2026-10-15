
"""PUFO visual FB-CPR-Aux agent.

The visual branch is deliberately kept outside the terrain task package.  It
uses the FB objective, plus heading and velocity Q heads for the deployable
context. Their shared trunk remains separate from the reward critic trunk.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .fb_cpr_aux import FbCprAuxAgent, tree_to_device
from .fb_networks import ResidualForwardMap, SharedCriticMap, model_kwargs, soft_update_params


HEADING_SOURCE_INVALID = 0
HEADING_SOURCE_EXACT_TRACKING = 1


@dataclass
class PufoRolloutContext:
    z: torch.Tensor
    heading_target_world: torch.Tensor
    velocity_target_world: torch.Tensor
    heading_valid: torch.Tensor
    velocity_valid: torch.Tensor
    context_id: torch.Tensor
    source_type: torch.Tensor
    expert_env_ids: torch.Tensor
    tracking_z: torch.Tensor
    tracking_heading_world: torch.Tensor
    tracking_velocity_world: torch.Tensor
    tracking_velocity_valid: torch.Tensor


class PufoAgent(FbCprAuxAgent):
    def __init__(self, obs_dims, action_dim, cfg, device="cuda", *, inference_only=False):
        super().__init__(obs_dims, action_dim, cfg, device=device, inference_only=inference_only)
        if inference_only:
            return
        self.heading_critic_enabled = cfg.heading_critic_enabled
        self.velocity_critic_enabled = cfg.velocity_critic_enabled
        if not (self.heading_critic_enabled or self.velocity_critic_enabled):
            raise ValueError("PUFO requires at least one context critic")
        self.shared_context_value_net = (
            bool(getattr(cfg, "shared_context_value_net", False))
            and self.heading_critic_enabled and self.velocity_critic_enabled
        )
        if self.shared_context_value_net:
            self._context_critic_shared = SharedCriticMap(
                self.full_obs_dim, self.z_dim, self.action_dim,
                num_parallel=self.critic_members, head_names=("heading", "velocity"),
                **model_kwargs(cfg.heading_critic),
            ).to(device)
            self._target_context_critic_shared = copy.deepcopy(self._context_critic_shared)
            self.context_critic_optimizer = torch.optim.Adam(
                self._context_critic_shared.parameters(), lr=float(cfg.lr_heading_critic)
            )
        elif self.heading_critic_enabled:
            self._heading_critic = ResidualForwardMap(
                self.full_obs_dim,
                self.z_dim,
                self.action_dim,
                num_parallel=self.critic_members,
                output_dim=1,
                **model_kwargs(cfg.heading_critic),
            ).to(device)
            self._target_heading_critic = copy.deepcopy(self._heading_critic).to(device)
            self.heading_critic_optimizer = torch.optim.Adam(
                self._heading_critic.parameters(), lr=float(cfg.lr_heading_critic)
            )
        if self.velocity_critic_enabled and not self.shared_context_value_net:
            self._velocity_critic = ResidualForwardMap(
                self.full_obs_dim,
                self.z_dim,
                self.action_dim,
                num_parallel=self.critic_members,
                output_dim=1,
                **model_kwargs(cfg.velocity_critic),
            ).to(device)
            self._target_velocity_critic = copy.deepcopy(self._velocity_critic).to(device)
            self.velocity_critic_optimizer = torch.optim.Adam(
                self._velocity_critic.parameters(), lr=float(cfg.lr_velocity_critic)
            )

    @torch.no_grad()
    def _sample_tracking_context(self, replay_buffers, batch_dim: int, length: int):
        batch = replay_buffers["expert_slicer"].sample(batch_dim * length, seq_length=length)
        next_obs = tree_to_device(batch["next"]["observation"], self.device)
        z = self._backward_map(self._b_in(next_obs)).view(batch_dim, length, -1)
        for step in range(length):
            z[:, step] = z[:, step : min(step + self.cfg.seq_length, length)].mean(dim=1)
        heading = batch["heading_forward_xy"].to(self.device).view(batch_dim, length, 2)
        velocity = batch["reference_velocity_world_xy"].to(self.device).view(batch_dim, length, 2)
        velocity_valid = batch["reference_velocity_valid"].to(self.device).view(batch_dim, length, 1)
        return self.project_z(z), heading, velocity, velocity_valid.bool()

    @torch.no_grad()
    def advance_rollout_context(
        self,
        state: PufoRolloutContext | None,
        step_count: torch.Tensor,
        current_heading_xy: torch.Tensor,
        replay_buffers,
    ) -> PufoRolloutContext:
        """Advance the z-paired deployable context for one vectorized collector."""
        from robolab.tasks.manager_based.ufo.mdp.pufo_observations import align_world_vector

        counts = step_count.to(device=self.device, dtype=torch.long).reshape(-1)
        current_heading_xy = current_heading_xy.to(self.device)
        num_envs = counts.shape[0]
        length = int(self.cfg.rollout_expert_trajectories_length)

        if state is None:
            num_expert = (
                min(num_envs, int(self.cfg.rollout_expert_trajectories_percentage * num_envs))
                if self.cfg.rollout_expert_trajectories
                else 0
            )
            expert_env_ids = torch.randperm(num_envs, device=self.device)[:num_expert]
            state = PufoRolloutContext(
                z=self.sample_z(num_envs),
                heading_target_world=current_heading_xy.clone(),
                velocity_target_world=torch.zeros(num_envs, 2, device=self.device),
                heading_valid=torch.zeros(num_envs, 1, dtype=torch.bool, device=self.device),
                velocity_valid=torch.zeros(num_envs, 1, dtype=torch.bool, device=self.device),
                context_id=torch.zeros(num_envs, 1, dtype=torch.long, device=self.device),
                source_type=torch.zeros(num_envs, 1, dtype=torch.long, device=self.device),
                expert_env_ids=expert_env_ids,
                tracking_z=torch.empty(num_expert, length, self.z_dim, device=self.device),
                tracking_heading_world=torch.empty(num_expert, length, 2, device=self.device),
                tracking_velocity_world=torch.empty(num_expert, length, 2, device=self.device),
                tracking_velocity_valid=torch.empty(num_expert, length, 1, dtype=torch.bool, device=self.device),
            )
            refresh_rows = torch.arange(num_expert, device=self.device)
        else:
            random_mask = counts.remainder(int(self.cfg.update_z_every_step)) == 0
            random_mask[state.expert_env_ids] = False
            if random_mask.any():
                count = int(random_mask.sum())
                new_z = (
                    self.z_buffer.sample(count)
                    if self.cfg.use_mix_rollout and not self.z_buffer.empty()
                    else self.sample_z(count)
                )
                state.z[random_mask] = new_z
                state.context_id[random_mask] += 1
            expert_counts = counts[state.expert_env_ids]
            refresh_rows = torch.nonzero(expert_counts.remainder(length) == 0, as_tuple=False).flatten()

        if refresh_rows.numel() > 0:
            refreshed_env_ids = state.expert_env_ids[refresh_rows]
            z, heading, velocity, velocity_valid = self._sample_tracking_context(
                replay_buffers, int(refresh_rows.numel()), length
            )
            reference_anchor = heading[:, :1].expand_as(heading)
            rollout_anchor = current_heading_xy[refreshed_env_ids, None, :].expand_as(heading)
            state.tracking_z[refresh_rows] = z
            state.tracking_heading_world[refresh_rows] = align_world_vector(
                heading, reference_anchor, rollout_anchor
            )
            state.tracking_velocity_world[refresh_rows] = align_world_vector(
                velocity, reference_anchor, rollout_anchor
            )
            state.tracking_velocity_valid[refresh_rows] = velocity_valid
            state.context_id[refreshed_env_ids] += 1

        ids = state.expert_env_ids
        rows = torch.arange(ids.numel(), device=self.device)
        indices = counts[ids].remainder(length)
        state.z[ids] = state.tracking_z[rows, indices]
        state.heading_target_world[ids] = state.tracking_heading_world[rows, indices]
        state.velocity_target_world[ids] = state.tracking_velocity_world[rows, indices]
        state.heading_valid.zero_()
        state.velocity_valid.zero_()
        state.source_type.fill_(HEADING_SOURCE_INVALID)
        state.heading_valid[ids] = True
        state.velocity_valid[ids] = state.tracking_velocity_valid[rows, indices]
        state.source_type[ids] = HEADING_SOURCE_EXACT_TRACKING
        return state

    @staticmethod
    def context_observations(state: PufoRolloutContext, current_heading_xy: torch.Tensor):
        from robolab.tasks.manager_based.ufo.mdp.pufo_observations import (
            heading_observation,
            velocity_observation,
        )

        return {
            "heading": heading_observation(
                current_heading_xy, state.heading_target_world, state.heading_valid
            ),
            "velocity_command": velocity_observation(
                state.velocity_target_world, current_heading_xy, state.velocity_valid
            ),
        }

    @torch.no_grad()
    def context_transition(
        self,
        state: PufoRolloutContext,
        step_count: torch.Tensor,
        current_heading_xy: torch.Tensor,
        next_heading_xy: torch.Tensor,
        actual_velocity_world_xy: torch.Tensor,
        done: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build action-outcome-aligned replay metadata before any context refresh."""
        from robolab.tasks.manager_based.ufo.mdp.pufo_observations import (
            heading_observation,
            velocity_observation,
        )

        done = done.to(device=self.device, dtype=torch.bool).reshape(-1, 1)
        counts = step_count.to(device=self.device, dtype=torch.long).reshape(-1)
        length = int(self.cfg.rollout_expert_trajectories_length)
        next_z = state.z.clone()
        next_heading_target = state.heading_target_world.clone()
        next_velocity_target = state.velocity_target_world.clone()
        next_velocity_valid = state.velocity_valid.clone()
        ids = state.expert_env_ids
        rows = torch.arange(ids.numel(), device=self.device)
        indices = counts[ids].remainder(length)
        next_indices = (indices + 1).clamp(max=length - 1)
        next_z[ids] = state.tracking_z[rows, next_indices]
        next_heading_target[ids] = state.tracking_heading_world[rows, next_indices]
        next_velocity_target[ids] = state.tracking_velocity_world[rows, next_indices]
        next_velocity_valid[ids] = state.tracking_velocity_valid[rows, next_indices]
        next_velocity_valid &= ~done

        heading_valid = state.heading_valid & ~done
        velocity_valid = state.velocity_valid & ~done
        continues = heading_valid.clone()
        continues[ids] &= (indices + 1 < length).unsqueeze(-1)
        next_heading = heading_observation(next_heading_xy, next_heading_target, heading_valid)
        next_velocity = velocity_observation(
            next_velocity_target, next_heading_xy, next_velocity_valid
        )
        velocity_error = torch.linalg.vector_norm(
            (actual_velocity_world_xy - state.velocity_target_world)
            / float(self.cfg.velocity_error_scale_mps),
            dim=-1,
            keepdim=True,
        )
        velocity_reward = -F.huber_loss(
            velocity_error, torch.zeros_like(velocity_error), reduction="none", delta=1.0
        )
        return {
            "heading_next": next_heading,
            "velocity_next": next_velocity,
            "heading_z_next": next_z,
            "heading_valid": heading_valid,
            "velocity_valid": velocity_valid,
            "heading_context_continues": continues,
            "heading_context_id": state.context_id,
            "heading_source_type": state.source_type,
            "root_heading_xy": current_heading_xy,
            "next_root_heading_xy": next_heading_xy,
            "heading_reward": -next_heading[:, 0],
            "velocity_reward": torch.where(
                velocity_valid, velocity_reward, torch.zeros_like(velocity_reward)
            ).reshape(-1),
        }

    def _context_critic_update(
        self,
        critic,
        target,
        optimizer,
        obs,
        action,
        discount,
        reward,
        valid,
        next_obs,
        z,
        next_z,
        next_action,
        penalty,
    ):
        with torch.no_grad():
            next_qs = target(next_obs, next_z, next_action).float()
            _, _, next_value = self._targets_uncertainty(next_qs, penalty)
            target_q = reward + discount.unsqueeze(-1) * next_value
        qs = critic(obs, z, action).float()
        loss, value, target_value = self._context_critic_loss(qs, target_q, valid)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._avg_grad(critic.parameters())
        optimizer.step()
        return loss.detach(), value, target_value

    @staticmethod
    def _context_critic_loss(qs, target_q, valid):
        expanded = target_q.expand_as(qs)
        valid = valid.to(device=qs.device, dtype=qs.dtype).reshape(1, -1, 1)
        denom = valid.sum().clamp_min(1.0)
        loss = 0.5 * ((qs - expanded).square() * valid).sum() / denom
        valid_flat = valid.reshape(-1)
        return (
            loss,
            (qs.mean(dim=0).reshape(-1) * valid_flat).sum().detach() / denom,
            (target_q.reshape(-1) * valid_flat).sum().detach() / denom,
        )

    def _prepare_train_observations(self, train_batch, obs, next_obs):
        metadata = train_batch["aux_rewards"]
        required = {
            "heading_next",
            "velocity_next",
            "heading_z_next",
            "heading_valid",
            "velocity_valid",
            "heading_context_continues",
            "heading_context_id",
            "heading_source_type",
            "root_heading_xy",
            "next_root_heading_xy",
        }
        missing = sorted(required.difference(metadata))
        if missing:
            raise RuntimeError(
                "PUFO replay context metadata is incomplete; collect a new replay buffer. "
                f"Missing: {', '.join(missing)}"
            )
        next_obs = dict(next_obs)
        next_obs["heading"] = metadata["heading_next"].to(self.device)
        next_obs["velocity_command"] = metadata["velocity_next"].to(self.device)
        return obs, next_obs

    def _relabel_train_observations(
        self,
        train_batch,
        expert_batch,
        obs,
        next_obs,
        sampled_z,
        mix_source,
        expert_indices,
        relabel_mask,
    ):
        from robolab.tasks.manager_based.ufo.mdp.pufo_observations import (
            align_world_vector,
            heading_observation,
            velocity_observation,
        )

        metadata = train_batch["aux_rewards"]
        current_heading = metadata["root_heading_xy"].to(self.device)
        next_heading = metadata["next_root_heading_xy"].to(self.device)
        expert_heading = expert_batch["heading_forward_xy"].to(self.device)[expert_indices]
        expert_next_heading = expert_batch["next"]["heading_forward_xy"].to(self.device)[expert_indices]
        selected = mix_source == 1
        target_next = align_world_vector(expert_next_heading, expert_heading, current_heading)
        sampled_heading = heading_observation(current_heading, current_heading, selected)
        sampled_next_heading = heading_observation(next_heading, target_next, selected)

        expert_velocity = expert_batch["reference_velocity_world_xy"].to(self.device)[expert_indices]
        expert_next_velocity = expert_batch["next"]["reference_velocity_world_xy"].to(self.device)[expert_indices]
        velocity_valid = selected & expert_batch["reference_velocity_valid"].to(self.device)[expert_indices].bool()
        next_velocity_valid = (
            selected
            & expert_batch["next"]["reference_velocity_valid"].to(self.device)[expert_indices].bool()
        )
        sampled_velocity = velocity_observation(
            align_world_vector(expert_velocity, expert_heading, current_heading),
            current_heading,
            velocity_valid,
        )
        sampled_next_velocity = velocity_observation(
            align_world_vector(expert_next_velocity, expert_heading, current_heading),
            next_heading,
            next_velocity_valid,
        )

        obs = dict(obs)
        next_obs = dict(next_obs)
        obs["heading"] = torch.where(relabel_mask, sampled_heading, obs["heading"])
        next_obs["heading"] = torch.where(relabel_mask, sampled_next_heading, next_obs["heading"])
        obs["velocity_command"] = torch.where(
            relabel_mask, sampled_velocity, obs["velocity_command"]
        )
        next_obs["velocity_command"] = torch.where(
            relabel_mask, sampled_next_velocity, next_obs["velocity_command"]
        )
        return obs, next_obs

    def _norm_relabelled_parts(self, obs, normalized):
        changed = {key: obs[key] for key in ("heading", "velocity_command")}
        return {**normalized, **self._norm_parts(changed)}

    def _context_batch(self, train_batch, obs, next_obs, z, discount):
        metadata = train_batch["aux_rewards"]
        continues = metadata["heading_context_continues"].to(self.device).reshape(-1).bool()
        return {
            "actor_obs": obs,
            "next_actor_obs": next_obs,
            "obs": self._full_in_from_parts(obs),
            "next_obs": self._full_in_from_parts(next_obs),
            "z": z,
            "next_z": metadata["heading_z_next"].to(self.device),
            "discount": float(self.cfg.discount) * continues,
            "heading_valid": metadata["heading_valid"].to(self.device).reshape(-1).bool(),
            "velocity_valid": metadata["velocity_valid"].to(self.device).reshape(-1).bool(),
        }

    def _tracking_batch(self, train_batch):
        rewards = train_batch["aux_rewards"]
        missing = [name for name in ("heading_reward", "velocity_reward") if name not in rewards]
        if missing:
            raise RuntimeError(
                "PUFO replay is missing context rewards: "
                + ", ".join(missing)
                + ". Collect a new replay buffer with PUFO tracking enabled."
            )
        heading = rewards["heading_reward"].to(self.device).reshape(-1)
        velocity = rewards["velocity_reward"].to(self.device).reshape(-1)
        heading_valid = rewards.get("heading_valid")
        velocity_valid = rewards.get("velocity_valid")
        if heading_valid is None:
            heading_valid = torch.isfinite(heading)
        else:
            heading_valid = heading_valid.to(self.device).reshape(-1).bool()
        if velocity_valid is None:
            velocity_valid = torch.isfinite(velocity)
        else:
            velocity_valid = velocity_valid.to(self.device).reshape(-1).bool()
        return heading.unsqueeze(-1), velocity.unsqueeze(-1), heading_valid, velocity_valid

    def _update_context_critics(
        self,
        *,
        train_batch,
        actor_obs,
        next_actor_obs,
        obs,
        next_obs,
        action,
        z,
        next_z,
        discount,
        heading_valid,
        velocity_valid,
    ):
        metrics = {}
        heading_reward, velocity_reward, _, _ = self._tracking_batch(train_batch)
        target_actions = self._sample_target_actions(next_actor_obs, next_z, count=2)
        if self.shared_context_value_net:
            velocity_discount = discount * (train_batch["aux_rewards"]["velocity_next"].to(self.device)[:, 2] > 0)
            with torch.no_grad():
                next_heading = self._target_context_critic_shared(next_obs, next_z, target_actions[0])[0].float()
                next_velocity = self._target_context_critic_shared(next_obs, next_z, target_actions[1])[1].float()
                heading_v = self._targets_uncertainty(next_heading, float(self.cfg.heading_critic_pessimism_penalty))[2]
                velocity_v = self._targets_uncertainty(next_velocity, float(self.cfg.velocity_critic_pessimism_penalty))[2]
                heading_target = heading_reward + discount.unsqueeze(-1) * heading_v
                velocity_target = velocity_reward + velocity_discount.unsqueeze(-1) * velocity_v
            heading_qs, velocity_qs = self._context_critic_shared(obs, z, action)
            heading_loss, heading_value, heading_target_value = self._context_critic_loss(
                heading_qs.float(), heading_target, heading_valid
            )
            velocity_loss, velocity_value, velocity_target_value = self._context_critic_loss(
                velocity_qs.float(), velocity_target, velocity_valid
            )
            self.context_critic_optimizer.zero_grad(set_to_none=True)
            (heading_loss + velocity_loss).backward()
            self._avg_grad(self._context_critic_shared.parameters())
            self.context_critic_optimizer.step()
            soft_update_params(
                self._context_critic_shared, self._target_context_critic_shared, float(self.cfg.critic_target_tau)
            )
            metrics.update(
                heading_critic_loss=heading_loss.detach(), heading_Q=heading_value,
                heading_target_Q=heading_target_value, velocity_critic_loss=velocity_loss.detach(),
                velocity_Q=velocity_value, velocity_target_Q=velocity_target_value,
            )
        elif self.heading_critic_enabled:
            loss, value, target = self._context_critic_update(
                self._heading_critic,
                self._target_heading_critic,
                self.heading_critic_optimizer,
                obs,
                action,
                discount,
                heading_reward,
                heading_valid,
                next_obs,
                z,
                next_z,
                target_actions[0],
                float(self.cfg.heading_critic_pessimism_penalty),
            )
            soft_update_params(self._heading_critic, self._target_heading_critic, float(self.cfg.critic_target_tau))
            metrics.update({"heading_critic_loss": loss, "heading_Q": value, "heading_target_Q": target})
        if self.velocity_critic_enabled and not self.shared_context_value_net:
            velocity_discount = discount * (train_batch["aux_rewards"]["velocity_next"].to(self.device)[:, 2] > 0)
            loss, value, target = self._context_critic_update(
                self._velocity_critic,
                self._target_velocity_critic,
                self.velocity_critic_optimizer,
                obs,
                action,
                velocity_discount,
                velocity_reward,
                velocity_valid,
                next_obs,
                z,
                next_z,
                target_actions[1],
                float(self.cfg.velocity_critic_pessimism_penalty),
            )
            soft_update_params(self._velocity_critic, self._target_velocity_critic, float(self.cfg.critic_target_tau))
            metrics.update({"velocity_critic_loss": loss, "velocity_Q": value, "velocity_target_Q": target})
        metrics["heading_valid_fraction"] = heading_valid.float().mean()
        metrics["velocity_valid_fraction"] = velocity_valid.float().mean()
        return metrics

    def state_dict(self):
        state = super().state_dict()
        if self.inference_only:
            return state
        if self.shared_context_value_net:
            state["context_critic_shared"] = self._context_critic_shared.state_dict()
            state["target_context_critic_shared"] = self._target_context_critic_shared.state_dict()
            return state
        if self.heading_critic_enabled:
            state["heading_critic"] = self._heading_critic.state_dict()
            state["target_heading_critic"] = self._target_heading_critic.state_dict()
        if self.velocity_critic_enabled:
            state["velocity_critic"] = self._velocity_critic.state_dict()
            state["target_velocity_critic"] = self._target_velocity_critic.state_dict()
        return state

    def load_state_dict(self, state):
        if self.inference_only:
            return super().load_state_dict(state)
        if self.shared_context_value_net != ("context_critic_shared" in state):
            raise ValueError("Checkpoint context critic layout does not match shared_context_value_net")
        super().load_state_dict(state)
        if self.shared_context_value_net:
            self._context_critic_shared.load_state_dict(state["context_critic_shared"])
            self._target_context_critic_shared.load_state_dict(
                state.get("target_context_critic_shared", state["context_critic_shared"])
            )
            return
        if self.heading_critic_enabled:
            self._heading_critic.load_state_dict(state["heading_critic"])
            self._target_heading_critic.load_state_dict(state.get("target_heading_critic", state["heading_critic"]))
        if self.velocity_critic_enabled:
            self._velocity_critic.load_state_dict(state["velocity_critic"])
            self._target_velocity_critic.load_state_dict(state.get("target_velocity_critic", state["velocity_critic"]))

    def optimizer_state_dict(self):
        state = super().optimizer_state_dict()
        if self.shared_context_value_net:
            state["context_critic_shared"] = self.context_critic_optimizer.state_dict()
            return state
        if self.heading_critic_enabled:
            state["heading_critic"] = self.heading_critic_optimizer.state_dict()
        if self.velocity_critic_enabled:
            state["velocity_critic"] = self.velocity_critic_optimizer.state_dict()
        return state

    def load_optimizer_state_dict(self, state):
        if self.shared_context_value_net != ("context_critic_shared" in state):
            raise ValueError("Checkpoint context optimizer layout does not match shared_context_value_net")
        super().load_optimizer_state_dict(state)
        if self.shared_context_value_net:
            self.context_critic_optimizer.load_state_dict(state["context_critic_shared"])
            return
        if self.heading_critic_enabled and "heading_critic" in state:
            self.heading_critic_optimizer.load_state_dict(state["heading_critic"])
        if self.velocity_critic_enabled and "velocity_critic" in state:
            self.velocity_critic_optimizer.load_state_dict(state["velocity_critic"])
