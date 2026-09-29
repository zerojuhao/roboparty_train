"""AMP policy with supervised body-frame XY velocity estimation."""

import copy

import torch
from torch import nn

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP


class AMPModel(MLPModel):
    def __init__(
        self, *args, enable_velocity_estimation=True,
        velocity_estimation_hidden_dims=(128, 64), critic_history_length=3, **kwargs,
    ):
        self.enable_velocity_estimation = enable_velocity_estimation
        self.critic_history_length = critic_history_length
        super().__init__(*args, **kwargs)
        self.velocity_estimator = (
            MLP(self.obs_dim, 2, velocity_estimation_hidden_dims, kwargs.get("activation", "elu"))
            if enable_velocity_estimation else None
        )
        self._velocity_loss = None

    def _get_latent_dim(self):
        return self.obs_dim + (2 if self.enable_velocity_estimation else 0)

    def get_latent(self, obs, masks=None, hidden_state=None):
        proprio = super().get_latent(obs, masks, hidden_state)
        self._velocity_loss = None
        if self.velocity_estimator is None:
            return proprio
        velocity = self.velocity_estimator(proprio)
        if "critic" in obs:
            # AMP concatenates each term's history; base_lin_vel is the first term.
            end = 3 * self.critic_history_length
            target = obs["critic"][:, end - 3:end - 1].detach()
            self._velocity_loss = (velocity - target).square().mean()
        return torch.cat((proprio, velocity.detach()), dim=-1)

    def get_aux_loss(self):
        return self._velocity_loss

    def as_jit(self):
        return _ExportAMPModel(self)

    def as_onnx(self, verbose=False):
        return _ExportAMPModel(self, verbose)


class _ExportAMPModel(nn.Module):
    is_recurrent = False

    def __init__(self, model, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.input_size = model.obs_dim
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.velocity_estimator = copy.deepcopy(model.velocity_estimator)
        self.mlp = copy.deepcopy(model.mlp)
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module()
            if model.distribution is not None else nn.Identity()
        )

    def forward(self, obs):
        latent = self.obs_normalizer(obs)
        if self.velocity_estimator is not None:
            latent = torch.cat((latent, self.velocity_estimator(latent)), dim=-1)
        return self.deterministic_output(self.mlp(latent))

    @torch.jit.export
    def reset(self):
        pass

    def get_dummy_inputs(self):
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self):
        return ["obs"]

    @property
    def output_names(self):
        return ["actions"]
