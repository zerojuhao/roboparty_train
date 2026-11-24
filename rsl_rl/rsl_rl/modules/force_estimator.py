"""Proprioceptive force estimation; labels never enter the predictor."""
import torch
from torch import nn

CONTACT_BODIES = ('left_wrist_link', 'right_wrist_link', 'left_shoulder_yaw_link',
                  'right_shoulder_yaw_link', 'left_shoulder_pitch_link', 'right_shoulder_pitch_link')
CONTACT_REGIONS = ('none', 'left_hand', 'right_hand', 'left_upper_arm',
                   'right_upper_arm', 'left_shoulder', 'right_shoulder')
MULTI_CONTACT_BODIES = ('left_wrist_link', 'right_wrist_link',
    'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link',
    'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link',
    'waist_yaw_link')
# Total force XYZ, then each link's force XYZ / point XYZ / contact, then explicit torso yaw torque.
MULTI_FORCE_DIM = 81
# World-frame net F3/M3 + generalized arm tau10, used by the compliance reward.
COMPACT_WRENCH_WORLD_DIM = 16
# Root-frame net Fxyz, net Mxyz, tau10, L/R/torso Fxyz, L/R/torso load.
COMPACT_FORCE_DIM = 28
REGION_LOAD_FORCE_N = 1.0
ARM_JOINT_NAMES = tuple(f'{side}_{part}_joint' for side in ('left', 'right')
                        for part in ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist'))


def compact_wrench_target(force_w, moment_com_w, com_w, root_w, arm_jacobian_w):
    """Known perturbations only; Jacobian at COM, columns in ARM_JOINT_NAMES order.

    Returns world-frame net force, net moment about the root, and generalized arm torques.
    """
    force = force_w.sum(-2)
    moment = (moment_com_w + torch.cross(com_w-root_w[:, None], force_w, dim=-1)).sum(-2)
    wrench = torch.cat((force_w, moment_com_w), dim=-1)
    joint = torch.einsum('nbij,nbi->nj', arm_jacobian_w, wrench)
    return torch.cat((force, moment, joint), -1)


def compact_estimator_target(force_b, moment_b, joint, region_b, load):
    """Root-frame net Fxyz, net Mxyz, tau10, regional Fxyz (L/R/torso), and load bits."""
    return torch.cat((force_b, moment_b, joint, region_b.flatten(1), load), -1)


def mirror_compact_force(target):
    result = target.clone()
    if target.shape[-1] == COMPACT_FORCE_DIM:
        result[..., 1] *= -1
        result[..., [3, 5]] *= -1
        result[..., 6:16] = -target[..., [11, 12, 13, 14, 15, 6, 7, 8, 9, 10]]
        regions = target[..., 16:25].unflatten(-1, (3, 3))
        mirrored = regions[..., [1, 0, 2], :].clone()
        mirrored[..., 1] *= -1
        result[..., 16:25] = mirrored.flatten(-2)
        result[..., 25:28] = target[..., [26, 25, 27]]
        return result
    if target.shape[-1] == 14:
        result[..., 1] *= -1
        result[..., 3] *= -1
        result[..., 4:14] = -target[..., [9, 10, 11, 12, 13, 4, 5, 6, 7, 8]]
        return result
    if target.shape[-1] == 13:
        result[..., 1] *= -1
        result[..., 2] *= -1
        result[..., 3:13] = -target[..., [8, 9, 10, 11, 12, 3, 4, 5, 6, 7]]
        return result
    result[..., 1] *= -1
    result[..., [3, 5]] *= -1
    result[..., 6:16] = -target[..., [11, 12, 13, 14, 15, 6, 7, 8, 9, 10]]
    if target.shape[-1] == 28:
        regions = target[..., 16:25].unflatten(-1, (3, 3))
        mirrored = regions[..., [1, 0, 2], :].clone()
        mirrored[..., 1] *= -1
        result[..., 16:25] = mirrored.flatten(-2)
        result[..., 25:28] = target[..., [26, 25, 27]]
    elif target.shape[-1] == 18:
        result[..., 16:18] = target[..., [17, 16]]
    return result


def regional_forces(link_forces):
    """Left arm, right arm and torso, in the fixed multi-contact link order."""
    return torch.stack((link_forces[..., [0, 2, 3, 4, 5], :].sum(-2),
                        link_forces[..., [1, 6, 7, 8, 9], :].sum(-2),
                        link_forces[..., 10, :]), dim=-2)


def balanced_contact_mean(error, contact):
    """Give loaded and unloaded samples equal weight, ignoring empty groups."""
    positive_count = contact.sum()
    negative_count = (1 - contact).sum()
    positive = (error * contact).sum() / positive_count.clamp_min(1)
    negative = (error * (1 - contact)).sum() / negative_count.clamp_min(1)
    groups = (positive_count > 0).to(error.dtype) + (negative_count > 0).to(error.dtype)
    return (positive + negative) / groups.clamp_min(1)


def weighted_contact_mean(error, contact, negative_weight=1.0):
    """Group means with the unloaded group weighted by ``negative_weight``; 1.0 equals balanced_contact_mean.

    Equal group weights shift detection logits up by ln(N_unloaded / N_loaded); weighting the unloaded
    group by k removes ln(k) of that bias, trading a little recall for fewer false load detections.
    """
    positive_count = contact.sum()
    negative_count = (1 - contact).sum()
    positive = (error * contact).sum() / positive_count.clamp_min(1)
    negative = (error * (1 - contact)).sum() / negative_count.clamp_min(1)
    wp = (positive_count > 0).to(error.dtype)
    wn = (negative_count > 0).to(error.dtype) * negative_weight
    return (wp * positive + wn * negative) / (wp + wn).clamp_min(1e-6)


def balanced_joint_torque_mean(error, torque_nm):
    """Balance joint-level errors without dropping weak or unloaded joints."""
    fraction = ((torque_nm.detach().abs() - .2) / .8).clamp(0, 1)
    active = fraction.square() * (3 - 2 * fraction)
    inactive = 1 - active
    active_count, inactive_count = active.sum(), inactive.sum()
    positive = (error * active).sum() / active_count.clamp_min(1)
    negative = (error * inactive).sum() / inactive_count.clamp_min(1)
    # Fractional group availability avoids a discontinuity at the first weak load.
    groups = active_count.clamp_max(1) + inactive_count.clamp_max(1)
    return (positive + negative) / groups.clamp_min(1)


def mirror_multi_force(target):
    result = target.clone()
    links = target[..., 3:80].reshape(*target.shape[:-1], 11, 7)
    mirrored = links[..., [1, 0, 6, 7, 8, 9, 2, 3, 4, 5, 10], :].clone()
    mirrored[..., [1, 4]] *= -1
    result[..., 3:80] = mirrored.flatten(-2)
    result[..., 1] *= -1
    result[..., 80] *= -1
    return result


def contact_region(body_name):
    if body_name == 'waist_yaw_link':
        return 7
    for side, hand, arm, shoulder in (('left_', 1, 3, 5), ('right_', 2, 4, 6)):
        if body_name.startswith(side):
            if 'wrist' in body_name:
                return hand
            if 'elbow' in body_name or 'shoulder_yaw' in body_name:
                return arm
            if 'shoulder_pitch' in body_name or 'shoulder_roll' in body_name:
                return shoulder
    raise ValueError(f'Unsupported external-force body: {body_name}')


class ForceVelEstimator(nn.Module):
    def __init__(self, input_dim, hidden_dims=(512, 256)):
        super().__init__()
        from rsl_rl.modules import MLP
        self.network = MLP(input_dim, 3, hidden_dims, 'elu')

    def forward(self, history):
        return self.network(history)


class ForceEstimator(nn.Module):
    def __init__(self, input_dim, hidden_dims=(512, 256), with_torque=False, multi_contact=False, compact=False):
        super().__init__()
        from rsl_rl.modules import MLP
        self.with_torque = with_torque
        self.compact = compact
        output_dim = COMPACT_FORCE_DIM if compact else MULTI_FORCE_DIM if multi_contact else 15 if with_torque else 13
        self.network = MLP(input_dim, output_dim, hidden_dims, 'elu')

    def forward(self, history):
        return self.network(history)

    @staticmethod
    def features(raw):
        if raw.shape[-1] == COMPACT_FORCE_DIM:
            return torch.cat((raw[..., :25], raw[..., 25:28].sigmoid()), -1)
        if raw.shape[-1] == 14:
            return raw
        if raw.shape[-1] == 31:
            return torch.cat((raw[..., :25], raw[..., 25:28].sigmoid(), raw[..., 28:31]), -1)
        if raw.shape[-1] == 28:
            return torch.cat((raw[..., :25], raw[..., 25:28].sigmoid()), -1)
        if raw.shape[-1] == 18:
            return torch.cat((raw[..., :16], raw[..., 16:18].sigmoid()), -1)
        if raw.shape[-1] == 81:
            links = raw[..., 3:80].unflatten(-1, (11, 7))
            contact = links[..., 6:7].sigmoid()
            features = torch.cat((links[..., :3], links[..., 3:6] * contact, contact), -1)
            return torch.cat((raw[..., :3], features.flatten(-2), raw[..., 80:81]), -1)
        # Gate location when there is no predicted contact. Scores are uncalibrated.
        end = 14 if raw.shape[-1] == 15 else 13
        probability = raw[..., 6:end].softmax(-1)
        features = torch.cat((raw[..., :3], raw[..., 3:6]*(1-probability[..., :1]), probability), -1)
        return torch.cat((features, raw[..., 14:15]), -1) if end == 14 else features

    @staticmethod
    def multi_loss_terms(raw, target):
        pred = raw[..., 3:80].unflatten(-1, (11, 7))
        truth = target[..., 3:80].unflatten(-1, (11, 7))
        contact = truth[..., 6]
        link_error = (pred[..., :3] - truth[..., :3] / 50.).square().mean(-1)
        point_error = (pred[..., 3:6] - truth[..., 3:6]).square().mean(-1)
        detection = nn.functional.binary_cross_entropy_with_logits(pred[..., 6], contact, reduction='none')
        return {
            'total_force': (raw[..., :3] - target[..., :3] / 50.).square().mean(),
            'link_force': balanced_contact_mean(link_error, contact),
            'regional_force': 0.5 * (regional_forces(pred[..., :3]) - regional_forces(truth[..., :3]) / 50.).square().mean(),
            'point': (point_error * contact).sum() / contact.sum().clamp_min(1),
            'contact': balanced_contact_mean(detection, contact),
            'torque': (raw[..., 80] - target[..., 80] / 10.).square().mean(),
        }

    @staticmethod
    def compact_loss_terms(raw, target, load_negative_weight=1.0):
        if raw.shape[-1] != COMPACT_FORCE_DIM:
            raise ValueError(
                f'Compact force layout must be {COMPACT_FORCE_DIM} (fxyz, mxyz, tau10, region9, load3)')
        arm_error = (raw[..., 6:16] - target[..., 6:16]/10.).square()
        force_error = (raw[..., :3]-target[..., :3]/50.).square().mean(-1)
        loaded = ((target[..., :2].detach().norm(dim=-1) - 5.) / 5.).clamp(0., 1.)
        loaded = loaded.square() * (3. - 2. * loaded)
        region_error = (raw[..., 16:25] - target[..., 16:25]/50.).square().unflatten(-1, (3, 3)).mean(-1)
        load = target[..., 25:28]
        detection = nn.functional.binary_cross_entropy_with_logits(raw[..., 25:28], load, reduction='none')
        return {
            'force': balanced_contact_mean(force_error, loaded),
            'net_moment': (raw[..., 3:6]-target[..., 3:6]/10.).square().mean(),
            'arm_torque': balanced_joint_torque_mean(arm_error, target[..., 6:16]),
            'regional_force': 0.5 * balanced_contact_mean(region_error.reshape(-1), load.reshape(-1)),
            'load': 0.1 * weighted_contact_mean(detection.reshape(-1), load.reshape(-1), load_negative_weight),
        }

    @staticmethod
    def loss(raw, target):
        if raw.shape != target.shape or raw.shape[-1] not in (13, 15, 18, 28, 81):
            raise ValueError('Force estimator and supervision layouts must match (13, 15, 18, 28 or 81 features)')
        if raw.shape[-1] in (18, 28):
            raise ValueError('Removed compact layouts 18/28; use the 13-dim fx/fy/mz/tau10 contract')
        if raw.shape[-1] == 81:
            return sum(ForceEstimator.multi_loss_terms(raw, target).values())
        end = 14 if raw.shape[-1] == 15 else 13
        force_loss = (raw[..., :3]-target[..., :3]/50.).square().mean()
        contact = target[..., 6:end].argmax(-1)
        mask = (contact != 0).to(raw.dtype)
        point_error = (raw[..., 3:6]-target[..., 3:6]).square().mean(-1)
        point_loss = (point_error*mask).sum()/mask.sum().clamp_min(1)
        region_loss = nn.functional.cross_entropy(raw[..., 6:end], contact)
        torque_loss = (raw[..., 14]-target[..., 14]/10.).square().mean() if end == 14 else 0.
        return force_loss+point_loss+region_loss+torque_loss
