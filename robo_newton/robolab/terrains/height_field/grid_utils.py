from __future__ import annotations

import numpy as np


def centered_step_levels(
    length_pixels: int,
    platform_pixels: int,
    step_pixels: int,
) -> np.ndarray:
    """Create a centered 1-D stair profile with an exact platform pixel count."""
    if length_pixels < 1 or platform_pixels < 1 or step_pixels < 1:
        raise ValueError("Terrain, platform, and step pixel counts must be positive.")
    if platform_pixels > length_pixels:
        raise ValueError("platform_pixels cannot exceed length_pixels.")

    platform_start = (length_pixels - platform_pixels) // 2
    platform_stop = platform_start + platform_pixels
    indices = np.arange(length_pixels)
    distance = np.where(
        indices < platform_start,
        platform_start - indices,
        np.where(indices >= platform_stop, indices - platform_stop + 1, 0),
    )
    num_steps = int(np.ceil(distance.max() / step_pixels))
    return num_steps - np.ceil(distance / step_pixels).astype(np.int32)
