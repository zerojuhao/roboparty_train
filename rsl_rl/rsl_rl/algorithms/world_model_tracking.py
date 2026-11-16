"""Continuous expert relabeling and physical pose costs for UFO imagination."""

import torch
from torch.nn import functional as F

from .world_model import quat_apply, yaw_quat


POSE_DIM = 223  # RP1: height, 24 relative body positions, 25 tangent/normal rotations.


def tracking_pose_loss(prediction, reference):
    """Raw heading-local body poses, with 0.1 m and 0.5 rad error scales."""
    if prediction.shape != reference.shape or prediction.shape[-1] != POSE_DIM:
        raise ValueError("Tracking requires matching raw RP1 pose observations")
    reference = reference.detach()
    positions = (prediction[..., 1:73] - reference[..., 1:73]).reshape(*prediction.shape[:-1], 24, 3)
    position_loss = F.smooth_l1_loss(positions / .1, torch.zeros_like(positions), reduction="none").sum(-1).mean(-1)
    rotation, target = (value[..., 73:].reshape(*prediction.shape[:-1], 25, 6)
                        for value in (prediction, reference))
    x, z = rotation[..., :3], rotation[..., 3:]
    tx, tz = target[..., :3], target[..., 3:]
    y, ty = torch.linalg.cross(z, x), torch.linalg.cross(tz, tx)
    rotation_error = .5 * ((x - tx).square().sum(-1) + (y - ty).square().sum(-1)
                          + (z - tz).square().sum(-1)).mean(-1)
    height = prediction[..., 0] - reference[..., 0]
    height_loss = F.smooth_l1_loss(height / .1, torch.zeros_like(height), reduction="none")
    errors = torch.stack((positions.square().sum(-1).mean(-1), rotation_error, height.square()), -1)
    return position_loss + rotation_error / .5 ** 2 + height_loss, errors


@torch.no_grad()
def prepare_tracking_batch(model, agent, batch, expert):
    """Re-encode one contiguous expert sequence per tracking row before microbatching."""
    if expert is None or "tracking_source" not in batch:
        raise ValueError("Tracking imagination needs expert replay and fresh tracking-source metadata")
    count, length = batch["z"].shape[:2]
    window = int(agent.cfg.seq_length)
    required = length + window - 1
    selected = batch["tracking_source"][:, 0, 0].bool()
    batch["tracking_mask"] = torch.zeros(count, length, 1, device=model.device, dtype=torch.bool)
    batch["reference_pose"] = torch.zeros(count, length, POSE_DIM, device=model.device)
    rows = selected.nonzero().flatten()
    if not rows.numel():
        return batch
    sample = expert.sample_tracking_variable(len(rows), required)
    complete = sample["valid"].to(model.device).reshape(len(rows), required).all(-1)
    usable = complete.nonzero().flatten()
    rows = rows[usable]
    if not rows.numel():
        return batch
    # Padding is discarded before encoding; a goal window never crosses a clip end.
    observations = {key: value.to(model.device).reshape(len(complete), required, -1)[usable].flatten(0, 1)
                    for key, value in sample["next"]["observation"].items()}
    embeddings = []
    for start in range(0, len(rows) * required, model.cfg.scoring_chunk_size):
        parts = {key: value[start:start + model.cfg.scoring_chunk_size] for key, value in observations.items()}
        with agent._autocast_context():
            embeddings.append(agent._backward_map(agent._b_in(parts)).float())
    embedded = torch.cat(embeddings).reshape(len(rows), required, -1)
    z = agent.project_z(embedded.unfold(1, window, 1).mean(-1))
    batch["z"][rows] = z
    reference = observations["privileged_state"].reshape(len(rows), required, -1)
    batch["reference_pose"][rows] = reference[:, :length, :POSE_DIM]
    batch["tracking_mask"][rows] = True

    def field(name):
        return sample["next"][name].to(model.device).reshape(len(complete), required, -1)[usable, :length]

    velocity_key = "reference_velocity_world" if batch["velocity"].shape[-1] == 3 else "reference_velocity_world_xy"
    heading, velocity = field("heading_forward_xy"), field(velocity_key)
    forward = torch.zeros(len(rows), 3, device=model.device)
    forward[:, 0] = 1
    anchor = quat_apply(yaw_quat(batch["state"][rows, 0, 1:5]), forward)[:, None, :2]
    reference_anchor = F.normalize(heading[:, :1], dim=-1)
    cosine = (reference_anchor * anchor).sum(-1)
    sine = reference_anchor[..., 0] * anchor[..., 1] - reference_anchor[..., 1] * anchor[..., 0]
    def align(vector):
        x, y = vector[..., :2].unbind(-1)
        xy = torch.stack((cosine * x - sine * y, sine * x + cosine * y), -1)
        return torch.cat((xy, vector[..., 2:]), -1)

    batch["heading"][rows] = align(heading)
    batch["velocity"][rows] = align(velocity)
    batch["valid"][rows, :, :1] = 1
    batch["valid"][rows, :, 1:] = field("reference_velocity_valid").to(batch["valid"].dtype)
    return batch
