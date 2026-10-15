"""AMP policy with supervised body-frame XY velocity estimation."""

import copy

import torch
from torch import nn

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP


def resolve_velocity_target_indices(term_names, term_shapes) -> tuple[int, int]:
    """Locate XY velocity in the newest frame of the resolved critic layout."""
    velocity_term = term_names.index("base_lin_vel")
    widths = [width for (width,) in term_shapes]
    velocity_end = sum(widths[:velocity_term + 1])
    return velocity_end - 3, velocity_end - 2


class AMPModel(MLPModel):
    def __init__(
        self, *args, enable_velocity_estimation=True,
        velocity_estimation_hidden_dims=(128, 64), velocity_target_indices=None,
        **kwargs,
    ):
        self.enable_velocity_estimation = enable_velocity_estimation
        if enable_velocity_estimation and velocity_target_indices is None:
            raise ValueError("Resolve velocity_target_indices from the observation manager before building AMPModel")
        self.velocity_target_indices = velocity_target_indices
        super().__init__(*args, **kwargs)
        self.velocity_estimator = (
            MLP(self.obs_dim, 2, velocity_estimation_hidden_dims, kwargs.get("activation", "elu"))
            if enable_velocity_estimation else None
        )
        self._velocity_loss = None
        self._velocity_error = None

    def _get_latent_dim(self):
        return self.obs_dim + (2 if self.enable_velocity_estimation else 0)

    def get_latent(self, obs, masks=None, hidden_state=None):
        proprio = super().get_latent(obs, masks, hidden_state)
        self._velocity_loss = None
        self._velocity_error = None
        if self.velocity_estimator is None:
            return proprio
        velocity = self.velocity_estimator(proprio)
        if "critic" in obs:
            target = obs["critic"][:, list(self.velocity_target_indices)].detach()
            self._velocity_error = velocity - target
            self._velocity_loss = self._velocity_error.square().mean()
        return torch.cat((proprio, velocity.detach()), dim=-1)

    def get_aux_loss(self):
        return self._velocity_loss

    def get_aux_metrics(self) -> dict[str, torch.Tensor]:
        if self._velocity_error is None:
            return {}
        error = self._velocity_error.detach()
        return {
            "Estimation/velocity/rmse_xy": torch.sqrt(error.square().sum(dim=-1).mean()),
            "Estimation/velocity/mae_x": error[:, 0].abs().mean(),
            "Estimation/velocity/mae_y": error[:, 1].abs().mean(),
        }

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
