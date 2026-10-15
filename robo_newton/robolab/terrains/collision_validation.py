"""Bounded startup checks of observation mesh versus solver collision heights."""

import copy

import numpy as np

from .native_collision import HEIGHT_DATA


def terrain_probe_points(generator):
    """Cover every tile plus representative component edges and rough samples."""
    points = []
    size = np.asarray(generator.cfg.size)
    for origin in generator.terrain_origins.reshape(-1, 3):
        for x, y in ((0., 0.), (-.31, 0.), (.31, 0.), (0., -.31), (0., .31)):
            points.append(origin[:2] + size * (x, y))
    components = generator.collision_components
    # Stratify by representation so thousands of boxes cannot crowd out fields.
    for fields in (False, True):
        candidates = [mesh for mesh, _ in components if (HEIGHT_DATA in mesh.metadata) == fields]
        if not candidates:
            continue
        for i in np.unique(np.linspace(0, len(candidates) - 1, min(64, len(candidates)), dtype=int)):
            mesh = candidates[i]
            bounds, center = mesh.bounds[:, :2], mesh.bounds[:, :2].mean(axis=0)
            points.append(center)
            if fields:
                heights = mesh.metadata[HEIGHT_DATA]
                spacing = (bounds[1] - bounds[0]) / (np.array(heights.shape) - 1)
                for index in (np.argmin(heights), np.argmax(heights)):
                    # Probe slightly inside outer boundaries, avoiding edge ambiguity.
                    xy = bounds[0] + np.array(np.unravel_index(index, heights.shape)) * spacing
                    points.append(np.clip(xy, bounds[0] + 1e-3, bounds[1] - 1e-3))
            else:
                for axis in (0, 1):
                    for edge in bounds[:, axis]:
                        for delta in (-.01, .01):
                            xy = center.copy()
                            xy[axis] = edge + delta
                            points.append(xy)
    # Box center/extent storage is float32: exactly shared boundaries can have
    # sub-micron seams. Probe their two sides, not a numerically ambiguous seam.
    return np.unique(np.asarray(points), axis=0) + np.array([.00037, .00023])


def validate_surface_heights(model, mesh, points, device="cpu", tolerance=1e-3, probe_labels=None):
    """Ray-check static colliders without changing the live solver or its model."""
    import mujoco
    import torch
    from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh

    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        raise ValueError("Terrain collision validation requires probe points")
    if probe_labels is not None and len(probe_labels) != len(points):
        raise ValueError("Terrain probe labels must match the number of points")
    starts = np.c_[points, np.full(len(points), mesh.bounds[1, 2] + 2.)]
    # Both ray implementations must receive identical float32-origin coordinates.
    starts = starts.astype(np.float32).astype(np.float64)
    directions = np.tile((0., 0., -1.), (len(points), 1))
    oracle = convert_to_warp_mesh(mesh.vertices, mesh.faces, device)
    reference = raycast_mesh(torch.tensor(starts, dtype=torch.float32, device=device),
                             torch.tensor(directions, dtype=torch.float32, device=device), oracle)[0][:, 2].cpu().numpy()
    # Exclude robot geoms using a private model copy, leaving group filtering in
    # the live model untouched. This also avoids robot occlusion near tile origins.
    probe_model = copy.copy(model)
    probe_model.geom_group[:] = (probe_model.geom_bodyid == 0).astype(np.int32)
    groups = np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8)
    data = mujoco.MjData(probe_model)
    mujoco.mj_kinematics(probe_model, data)
    mujoco.mj_comPos(probe_model, data)
    actual = np.full(len(points), np.inf)
    geom = np.empty(1, dtype=np.int32)
    for i, (start, direction) in enumerate(zip(starts, directions)):
        distance = mujoco.mj_ray(probe_model, data, start, direction, groups, True, -1, geom)
        if distance >= 0:
            actual[i] = start[2] - distance
    hit = np.isfinite(reference)
    mismatch = hit != np.isfinite(actual)
    errors = np.zeros(len(points))
    both = hit & np.isfinite(actual)
    errors[both] = np.abs(reference[both] - actual[both])
    bad = mismatch | (errors > tolerance)
    if not hit.any() or bad.any():
        i = int(np.flatnonzero(bad)[0]) if bad.any() else 0
        label = f" {probe_labels[i]}" if probe_labels is not None else ""
        raise RuntimeError(f"Terrain collision/observation mismatch{label} at xy={points[i].tolist()}: "
                           f"mesh_z={reference[i]}, solver_z={actual[i]}, "
                           f"bad={bad.sum()}/{len(points)}, tolerance={tolerance} m")
    return {"probes": len(points), "hits": int(hit.sum()), "max_error": float(errors.max())}
