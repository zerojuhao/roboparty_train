
"""PUFO visual FB-CPR-Aux agent.

The visual branch is deliberately kept outside the terrain task package.  It
uses the shared FB objective, plus dedicated heading and velocity critics for
the deployable context supplied by the PUFO environment.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from .fb_cpr_aux import FbCprAuxAgent
from .fb_networks import ResidualForwardMap, soft_update_params


class PufoAgent(FbCprAuxAgent):
    def __init__(self, obs_dims, action_dim, cfg, device="cuda"):
        super().__init__(obs_dims, action_dim, cfg, device=device)
        self.heading_critic_enabled = bool(getattr(cfg, "heading_critic_enabled", True))
        self.velocity_critic_enabled = bool(getattr(cfg, "velocity_critic_enabled", True))
        self.context_critic_hidden_dim = int(getattr(cfg, "context_critic_hidden_dim", 1024))
        self.context_critic_layers = int(getattr(cfg, "context_critic_layers", 3))
        if not (self.heading_critic_enabled or self.velocity_critic_enabled):
            raise ValueError("PUFO requires at least one context critic")
        if self.heading_critic_enabled:
            self._heading_critic = ResidualForwardMap(
                self.full_obs_dim,
                self.z_dim,
                self.action_dim,
                hidden_dim=self.context_critic_hidden_dim,
                hidden_layers=self.context_critic_layers,
                num_parallel=self.critic_members,
                output_dim=1,
            ).to(device)
            self._target_heading_critic = copy.deepcopy(self._heading_critic).to(device)
            self.heading_critic_optimizer = torch.optim.Adam(
                self._heading_critic.parameters(), lr=float(cfg.lr_heading_critic)
            )
        if self.velocity_critic_enabled:
            self._velocity_critic = ResidualForwardMap(
                self.full_obs_dim,
                self.z_dim,
                self.action_dim,
                hidden_dim=self.context_critic_hidden_dim,
                hidden_layers=self.context_critic_layers,
                num_parallel=self.critic_members,
                output_dim=1,
            ).to(device)
            self._target_velocity_critic = copy.deepcopy(self._velocity_critic).to(device)
            self.velocity_critic_optimizer = torch.optim.Adam(
                self._velocity_critic.parameters(), lr=float(cfg.lr_velocity_critic)
            )

    def _context_critic_update(self, critic, target, optimizer, obs, action, discount, reward, next_obs, z, next_action, penalty):
        with torch.no_grad():
            next_qs = target(next_obs, z, next_action).float()
            _, _, next_value = self._targets_uncertainty(next_qs, penalty)
            target_q = reward + discount.unsqueeze(-1) * next_value
            expanded = target_q.expand(self.critic_members, -1, -1)
        qs = critic(obs, z, action).float()
        loss = 0.5 * self.critic_members * F.mse_loss(qs, expanded)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._avg_grad(critic.parameters())
        optimizer.step()
        return loss.detach(), qs.mean().detach(), target_q.mean().detach()

    def update(self, replay_buffers):
        metrics = super().update(replay_buffers)
        train_batch = replay_buffers["train"].sample(self.cfg.batch_size)
        train_obs = self._norm_parts({k: v.to(self.device) for k, v in train_batch["observation"].items()})
        next_obs = self._norm_parts({k: v.to(self.device) for k, v in train_batch["next"]["observation"].items()})
        action = train_batch["action"].to(self.device)
        z = train_batch["z"].to(self.device)
        discount = self.cfg.discount * ~train_batch["next"]["terminated"].to(self.device).squeeze(-1)
        obs_input = self._full_input(train_obs)
        next_input = self._full_input(next_obs)
        if self.heading_critic_enabled:
            raw = train_batch["aux_rewards"].get("heading_tracking")
            if raw is None:
                raw = torch.zeros((action.shape[0],), device=self.device)
            reward = -raw.to(self.device).reshape(-1, 1)
            loss, value, target = self._context_critic_update(
                self._heading_critic,
                self._target_heading_critic,
                self.heading_critic_optimizer,
                obs_input,
                action,
                discount,
                reward,
                next_input,
                z,
                self._sample_target_actions(next_obs, z, 1)[0],
                float(self.cfg.heading_critic_pessimism_penalty),
            )
            soft_update_params(self._heading_critic, self._target_heading_critic, float(self.cfg.critic_target_tau))
            metrics.update({"heading_critic_loss": loss, "heading_Q": value, "heading_target_Q": target})
        if self.velocity_critic_enabled:
            raw = train_batch["aux_rewards"].get("velocity_tracking")
            if raw is None:
                raw = torch.zeros((action.shape[0],), device=self.device)
            reward = -raw.to(self.device).reshape(-1, 1)
            loss, value, target = self._context_critic_update(
                self._velocity_critic,
                self._target_velocity_critic,
                self.velocity_critic_optimizer,
                obs_input,
                action,
                discount,
                reward,
                next_input,
                z,
                self._sample_target_actions(next_obs, z, 1)[0],
                float(self.cfg.velocity_critic_pessimism_penalty),
            )
            soft_update_params(self._velocity_critic, self._target_velocity_critic, float(self.cfg.critic_target_tau))
            metrics.update({"velocity_critic_loss": loss, "velocity_Q": value, "velocity_target_Q": target})
        return metrics

    def state_dict(self):
        state = super().state_dict()
        if self.heading_critic_enabled:
            state["heading_critic"] = self._heading_critic.state_dict()
            state["target_heading_critic"] = self._target_heading_critic.state_dict()
        if self.velocity_critic_enabled:
            state["velocity_critic"] = self._velocity_critic.state_dict()
            state["target_velocity_critic"] = self._target_velocity_critic.state_dict()
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        if self.heading_critic_enabled:
            self._heading_critic.load_state_dict(state["heading_critic"])
            self._target_heading_critic.load_state_dict(state.get("target_heading_critic", state["heading_critic"]))
        if self.velocity_critic_enabled:
            self._velocity_critic.load_state_dict(state["velocity_critic"])
            self._target_velocity_critic.load_state_dict(state.get("target_velocity_critic", state["velocity_critic"]))

    def optimizer_state_dict(self):
        state = super().optimizer_state_dict()
        if self.heading_critic_enabled:
            state["heading_critic"] = self.heading_critic_optimizer.state_dict()
        if self.velocity_critic_enabled:
            state["velocity_critic"] = self.velocity_critic_optimizer.state_dict()
        return state

    def load_optimizer_state_dict(self, state):
        super().load_optimizer_state_dict(state)
        if self.heading_critic_enabled and "heading_critic" in state:
            self.heading_critic_optimizer.load_state_dict(state["heading_critic"])
        if self.velocity_critic_enabled and "velocity_critic" in state:
            self.velocity_critic_optimizer.load_state_dict(state["velocity_critic"])
