"""Command-relative path feedback shared by training and MuJoCo playback."""

import torch


def command_path_features(
    displacement: torch.Tensor,
    heading_delta: torch.Tensor,
    command: torch.Tensor,
) -> torch.Tensor:
    """Return body-heading-frame offset XY and straight-line heading error.

    Displacement is expressed in the heading frame at command start. Straight
    motion constrains only the lateral axis; in-place turns constrain both axes
    but must remain free to rotate. Heading features are sin(delta) and
    1-cos(delta), avoiding the discontinuity of wrapped angles.
    """
    moving = torch.linalg.vector_norm(command[..., :2], dim=-1) >= 0.1
    turning = command[..., 2].abs() > 0.1
    straight = moving & ~turning
    in_place = ~moving & turning
    offset_x = displacement[..., 0] * in_place
    offset_y = displacement[..., 1] * (straight | in_place)
    sine, cosine = torch.sin(heading_delta), torch.cos(heading_delta)
    return torch.stack(
        (
            cosine * offset_x + sine * offset_y,
            -sine * offset_x + cosine * offset_y,
            sine * straight,
            (1.0 - cosine) * straight,
        ),
        dim=-1,
    )
