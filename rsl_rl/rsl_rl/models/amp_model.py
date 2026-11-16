"""AMP policy with supervised body-frame XYZ velocity estimation."""

import copy

import torch
from torch import nn

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP, EmpiricalNormalization
from rsl_rl.modules.force_estimator import (
    ForceEstimator, ForceVelEstimator, regional_forces, COMPACT_FORCE_DIM, balanced_contact_mean,
)


def resolve_velocity_target_indices(term_names, term_shapes) -> tuple[int, int, int]:
    """Locate XYZ velocity in the newest frame of the resolved critic layout."""
    velocity_term = term_names.index("base_lin_vel")
    widths = [width for (width,) in term_shapes]
    velocity_end = sum(widths[:velocity_term + 1])
    return velocity_end - 3, velocity_end - 2, velocity_end - 1


def force_vel_target_loss(prediction, target, idle_threshold=0.05):
    """Balance idle/active MSE and extra-penalize same-sign underestimation."""
    error = prediction - target
    square = error.square()
    mse = square.sum(-1)
    active = target.norm(dim=-1) > idle_threshold
    balanced = balanced_contact_mean(mse, active.float())
    under = ((prediction * target) > 0) & (prediction.abs() < target.abs()) & active.unsqueeze(-1)
    extra = (square * under.float()).sum() / active.sum().clamp_min(1)
    return balanced + extra, error, active, under


class AMPModel(MLPModel):
    def __init__(
        self, *args, enable_velocity_estimation=True,
        velocity_estimation_hidden_dims=(512, 256), velocity_target_indices=None,
        enable_force_estimation=False, force_estimation_hidden_dims=(512, 256),
        force_estimation_history_length=8, force_estimation_with_torque=False,
        force_estimation_multi_contact=False,
        force_estimation_compact=False,
        enable_force_vel_target_estimation=False,
        **kwargs,
    ):
        self.enable_velocity_estimation = enable_velocity_estimation
        if enable_velocity_estimation and velocity_target_indices is None:
            raise ValueError("Resolve velocity_target_indices from the observation manager before building AMPModel")
        self.velocity_target_indices = velocity_target_indices
        # Saved XY models remain exportable; new training resolves all three axes.
        self.velocity_dim = len(velocity_target_indices) if enable_velocity_estimation else 0
        if enable_velocity_estimation and self.velocity_dim not in (2, 3):
            raise ValueError("Velocity estimation requires XY or XYZ target indices")
        self.enable_force_estimation = enable_force_estimation
        self.force_estimation_with_torque = force_estimation_with_torque
        self.force_estimation_multi_contact = force_estimation_multi_contact
        self.force_estimation_compact = force_estimation_compact
        self.enable_force_vel_target_estimation = enable_force_vel_target_estimation
        if enable_force_vel_target_estimation and not force_estimation_compact:
            raise ValueError('Force-vel-target estimation requires compact force estimation')
        if force_estimation_compact and not (enable_force_estimation and force_estimation_multi_contact and force_estimation_with_torque):
            raise ValueError('Compact estimation requires the multi-contact wrench contract')
        if force_estimation_multi_contact and not force_estimation_with_torque:
            raise ValueError('Multi-contact estimation requires the torso torque contract')
        if force_estimation_with_torque and not enable_force_estimation:
            raise ValueError('Torque estimation requires force estimation')
        self.force_input_dim = 81 * force_estimation_history_length
        super().__init__(*args, **kwargs)
        self.shared_force_history = self.obs_dim == self.force_input_dim
        self.velocity_estimator = (
            MLP(self.obs_dim, self.velocity_dim, velocity_estimation_hidden_dims, kwargs.get("activation", "elu"))
            if enable_velocity_estimation else None
        )
        self._velocity_loss = None
        self._velocity_error = None
        self.force_estimator = ForceEstimator(self.force_input_dim, force_estimation_hidden_dims,
                                             force_estimation_with_torque, force_estimation_multi_contact,
                                             force_estimation_compact) if enable_force_estimation else None
        self.force_vel_estimator = (
            ForceVelEstimator(self.force_input_dim, force_estimation_hidden_dims)
            if enable_force_vel_target_estimation else None
        )
        self.force_input_normalizer = (
            None if (not enable_force_estimation or self.shared_force_history)
            else EmpiricalNormalization(self.force_input_dim)
        )
        self._force_loss = None
        self._force_metrics = {}
        self._force_vel_target_loss = None
        self._force_vel_target_metrics = {}

    def _get_latent_dim(self):
        return self.obs_dim + self.velocity_dim + (
            (COMPACT_FORCE_DIM if self.force_estimation_compact else 81 if self.force_estimation_multi_contact else 15 if self.force_estimation_with_torque else 13)
            if self.enable_force_estimation else 0) + (3 if self.enable_force_vel_target_estimation else 0)

    def get_latent(self, obs, masks=None, hidden_state=None):
        proprio = super().get_latent(obs, masks, hidden_state)
        self._velocity_loss = None
        self._velocity_error = None
        features = [proprio]
        if self.velocity_estimator is not None:
            velocity = self.velocity_estimator(proprio)
            if "critic" in obs:
                target = obs["critic"][:, list(self.velocity_target_indices)].detach()
                self._velocity_error = velocity - target
                self._velocity_loss = self._velocity_error.square().mean()
            features.append(velocity.detach())
        self._force_loss = None
        self._force_metrics = {}
        self._force_vel_target_loss = None
        self._force_vel_target_metrics = {}
        if self.force_estimator is not None:
            history = proprio if self.shared_force_history else self.force_input_normalizer(self._force_history(obs))
            raw = self.force_estimator(history)
            features.append(raw.detach() if self.force_estimation_compact else ForceEstimator.features(raw).detach())
            if self.force_vel_estimator is not None:
                prediction = self.force_vel_estimator(history)
                features.append(prediction.detach())
                if 'force_vel_target' in obs:
                    target = obs['force_vel_target'].detach()
                    if target.shape != prediction.shape:
                        raise ValueError('Force-vel target must contain vx, vy, yaw_rate')
                    loss, error, active, under = force_vel_target_loss(prediction, target)
                    self._force_vel_target_loss = loss
                    with torch.no_grad():
                        square_xy = error[..., :2].square().sum(-1)
                        self._force_vel_target_metrics.update({
                            'Estimation/force_vel_target/loss': loss.detach(),
                            'Estimation/force_vel_target/rmse_xy_mps': square_xy.mean().sqrt(),
                            'Estimation/force_vel_target/rmse_yaw_rps': error[..., 2].square().mean().sqrt(),
                            'Estimation/force_vel_target/active_rmse_xy_mps': (
                                (square_xy * active).sum() / active.sum().clamp_min(1)).sqrt(),
                            'Estimation/force_vel_target/active_fraction': active.float().mean(),
                            'Estimation/force_vel_target/idle_prediction_rms': (
                                (prediction.square().sum(-1) * ~active).sum() / (~active).sum().clamp_min(1)).sqrt(),
                            'Estimation/force_vel_target/underestimate_fraction': under.float().mean(),
                        })
            if 'force_target' in obs:
                target = obs['force_target'].detach()
                if self.force_estimation_compact:
                    if raw.shape != target.shape:
                        raise ValueError('Compact force estimator and supervision layouts must match')
                    terms = ForceEstimator.compact_loss_terms(raw, target)
                    self._force_loss = sum(terms.values())
                    self._force_metrics = self._compact_force_metrics(raw, target)
                    self._force_metrics.update({f'Estimation/force_loss/{name}': value.detach()
                                                for name, value in terms.items()})
                    return torch.cat(features, dim=-1)
                if self.force_estimation_multi_contact:
                    if raw.shape != target.shape:
                        raise ValueError('Multi-contact estimator and supervision layouts must match')
                    loss_terms = ForceEstimator.multi_loss_terms(raw, target)
                    self._force_loss = sum(loss_terms.values())
                    self._force_metrics = self._multi_force_metrics(raw, target)
                    self._force_metrics.update({f'Estimation/force_loss/{name}': value.detach()
                                                for name, value in loss_terms.items()})
                    return torch.cat(features, dim=-1)
                self._force_loss = ForceEstimator.loss(raw, target)
                end = 14 if self.force_estimation_with_torque else 13
                contact = target[..., 6:end].argmax(-1)
                mask = contact != 0
                error = raw[..., :3]*50.-target[..., :3]
                self._force_metrics = {
                    'Estimation/force/rmse_n': error.square().sum(-1).mean().sqrt(),
                    'Estimation/force/region_accuracy': (raw[..., 6:end].argmax(-1)==contact).float().mean(),
                    'Estimation/force/point_error_m': ((raw[..., 3:6]-target[..., 3:6]).norm(dim=-1)*mask).sum()/mask.sum().clamp_min(1),
                    'Estimation/force/contact_rmse_n': ((error.square().sum(-1)*mask).sum()/mask.sum().clamp_min(1)).sqrt(),
                    'Estimation/force/false_positive_rate': (((raw[..., 6:end].argmax(-1)!=0)&~mask).float().sum()/(~mask).sum().clamp_min(1)),
                    'Estimation/force/contact_fraction': mask.float().mean(),
                }
                if self.force_estimation_with_torque:
                    self._force_metrics['Estimation/force/yaw_torque_rmse_nm'] = (
                        raw[..., 14]*10.-target[..., 14]).square().mean().sqrt()
        return torch.cat(features, dim=-1)

    @staticmethod
    def _compact_force_metrics(raw, target):
        force_error = raw[..., :3].detach() * 50. - target[..., :3]
        force_square = force_error.square()
        loaded = target[..., :2].norm(dim=-1) >= 10.
        error = (raw[..., 4:14]*10.-target[..., 4:14]).square().unflatten(-1, (2, 5))
        active = (target[..., 4:14].abs() > 1.).unflatten(-1, (2, 5))
        return {
            'Estimation/force/rmse_xy_n': force_square[..., :2].sum(-1).mean().sqrt(),
            'Estimation/force/rmse_z_n': force_square[..., 2].mean().sqrt(),
            'Estimation/force/loaded_rmse_xy_n': (
                (force_square[..., :2].sum(-1) * loaded).sum() / loaded.sum().clamp_min(1)).sqrt(),
            'Estimation/force/loaded_fraction': loaded.float().mean(),
            'Estimation/force/fx_rmse_n': force_square[..., 0].mean().sqrt(),
            'Estimation/force/fy_rmse_n': force_square[..., 1].mean().sqrt(),
            'Estimation/force/fx_bias_n': force_error[..., 0].mean(),
            'Estimation/force/fy_bias_n': force_error[..., 1].mean(),
            'Estimation/force/fz_bias_n': force_error[..., 2].mean(),
            'Estimation/force/yaw_moment_rmse_nm': (raw[..., 3]*10.-target[..., 3]).square().mean().sqrt(),
            'Estimation/force/arm_torque_rmse_nm': error.mean().sqrt(),
            'Estimation/force/loaded_arm_torque_rmse_nm': ((error*active).sum()/active.sum().clamp_min(1)).sqrt(),
        }

    @staticmethod
    def _multi_force_metrics(raw, target):
        pred = raw[..., 3:80].unflatten(-1, (11, 7))
        truth = target[..., 3:80].unflatten(-1, (11, 7))
        contact = truth[..., 6] > .5
        detected = pred[..., 6] > 0
        count = contact.sum().clamp_min(1)
        error = (pred[..., :3] * 50. - truth[..., :3]).square().sum(-1)
        region_error = (regional_forces(pred[..., :3]) * 50. - regional_forces(truth[..., :3])).square().sum(-1)
        region_contact = regional_forces(contact[..., None].float()).squeeze(-1) > 0
        return {
            'Estimation/force/rmse_n': (raw[..., :3] * 50. - target[..., :3]).square().sum(-1).mean().sqrt(),
            'Estimation/force/link_contact_recall': (detected & contact).sum() / count,
            'Estimation/force/link_contact_precision': (detected & contact).sum() / detected.sum().clamp_min(1),
            'Estimation/force/false_positive_rate': (detected & ~contact).sum() / (~contact).sum().clamp_min(1),
            'Estimation/force/contact_rmse_n': ((error * contact).sum() / count).sqrt(),
            'Estimation/force/unloaded_link_rmse_n': ((error * ~contact).sum() / (~contact).sum().clamp_min(1)).sqrt(),
            'Estimation/force/regional_contact_rmse_n': ((region_error * region_contact).sum() / region_contact.sum().clamp_min(1)).sqrt(),
            'Estimation/force/point_error_m': ((pred[..., 3:6] - truth[..., 3:6]).norm(dim=-1) * contact).sum() / count,
            'Estimation/force/multi_contact_fraction': (contact.sum(-1) > 1).float().mean(),
            'Estimation/force/yaw_torque_rmse_nm': (raw[..., 80] * 10. - target[..., 80]).square().mean().sqrt(),
        }

    def get_aux_loss(self):
        losses = [loss for loss in (self._velocity_loss, self._force_loss, self._force_vel_target_loss) if loss is not None]
        return sum(losses) if losses else None

    def get_aux_metrics(self) -> dict[str, torch.Tensor]:
        metrics = dict(self._force_metrics)
        metrics.update(self._force_vel_target_metrics)
        if self._velocity_error is None:
            return metrics
        error = self._velocity_error.detach()
        metrics.update({
            "Estimation/velocity/rmse_xy": torch.sqrt(error[:, :2].square().sum(dim=-1).mean()),
            "Estimation/velocity/mae_x": error[:, 0].abs().mean(),
            "Estimation/velocity/mae_y": error[:, 1].abs().mean(),
        })
        if self.velocity_dim == 3:
            metrics.update({
                "Estimation/velocity/rmse_xyz": error.square().sum(dim=-1).mean().sqrt(),
                "Estimation/velocity/rmse_z": error[:, 2].square().mean().sqrt(),
                "Estimation/velocity/mae_z": error[:, 2].abs().mean(),
            })
        return metrics

    def _force_history(self, obs):
        if self.shared_force_history:
            return torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        return obs['force_history']

    def update_normalization(self, obs):
        super().update_normalization(obs)
        if self.force_input_normalizer is not None:
            self.force_input_normalizer.update(self._force_history(obs))

    def as_jit(self):
        if self.force_estimator is not None:
            return _ExportForceAMPModel(self)
        return _ExportAMPModel(self)

    def as_onnx(self, verbose=False):
        if self.force_estimator is not None:
            return _ExportForceAMPModel(self, verbose)
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


class _ExportForceAMPModel(_ExportAMPModel):
    def __init__(self, model, verbose=False):
        super().__init__(model, verbose)
        self.force_estimator = copy.deepcopy(model.force_estimator)
        self.use_compact_force = model.force_estimation_compact
        self.use_force_vel_estimator = model.force_vel_estimator is not None
        self.force_vel_estimator = (
            copy.deepcopy(model.force_vel_estimator) if model.force_vel_estimator is not None else nn.Identity()
        )
        self.force_input_normalizer = (
            copy.deepcopy(model.force_input_normalizer) if model.force_input_normalizer is not None else nn.Identity()
        )
        self.force_input_dim = model.force_input_dim
        self.shared_force_history = model.shared_force_history

    def forward(self, obs, force_history: torch.Tensor | None = None):
        if force_history is None:
            if not self.shared_force_history:
                raise ValueError('This policy requires a separate force history')
            force_history = obs
        normalized = self.obs_normalizer(obs)
        latent = normalized
        if self.velocity_estimator is not None:
            latent = torch.cat((latent, self.velocity_estimator(normalized)), -1)
        history = normalized if self.shared_force_history else self.force_input_normalizer(force_history)
        raw = self.force_estimator(history)
        force_feat = raw if self.use_compact_force else ForceEstimator.features(raw)
        if self.use_force_vel_estimator:
            latent = torch.cat((latent, force_feat, self.force_vel_estimator(history)), -1)
        else:
            latent = torch.cat((latent, force_feat), -1)
        return self.deterministic_output(self.mlp(latent))

    def get_dummy_inputs(self):
        if self.shared_force_history:
            return (torch.zeros(1, self.input_size),)
        return torch.zeros(1, self.input_size), torch.zeros(1, self.force_input_dim)

    @property
    def input_names(self):
        if self.shared_force_history:
            return ['obs']
        return ['obs', 'force_history']
