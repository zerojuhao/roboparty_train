"""Startup-only checks on both sides of every authored stair-component edge."""

from time import perf_counter

import numpy as np

from robolab.terrains.collision_validation import validate_surface_heights
from robolab.terrains.native_collision import HEIGHT_DATA
from robolab.terrains.native_importer import box_bounds


def stair_edge_probe_points(components):
    """Probe each rectangle edge at three positions, 2/10 mm on either side.

    Boxes and their rough caps often have identical XY bounds; those edges are
    deduplicated. Narrow rectangles reduce the offsets to stay near the feature.
    No global sampling cap may crowd out a stair level or a small rough cap.
    """
    edges = {}
    for mesh, _ in components:
        if HEIGHT_DATA not in mesh.metadata and box_bounds(mesh) is None:
            raise ValueError("Stair-edge checks require authored heightfields or axis-aligned boxes")
        bounds = mesh.bounds[:, :2]
        if not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
            raise ValueError("Stair-edge checks require finite, nonempty component bounds")
        for axis in (0, 1):
            tangent = 1 - axis
            for edge in bounds[:, axis]:
                key = (axis, float(edge), float(bounds[0, tangent]), float(bounds[1, tangent]))
                width = float(bounds[1, axis] - bounds[0, axis])
                edges[key] = min(edges.get(key, width), width)
    if not edges:
        raise ValueError("Stair tile has no authored collision edges")
    points = []
    for (axis, edge, start, end), width in sorted(edges.items()):
        # Avoid corners and exact midpoints where other tread partitions meet.
        for fraction in (.2113, .5031, .7887):
            tangent_pos = start + fraction * (end - start)
            for distance in sorted({min(.002, width / 4), min(.01, width / 4)}):
                for sign in (-1, 1):
                    point = np.empty(2)
                    point[axis] = edge + sign * distance
                    point[1 - axis] = tangent_pos
                    points.append(point)
    return np.unique(np.asarray(points), axis=0)


def validate_stair_edges(model, generator, device="cpu"):
    """Compare full observation-mesh heights, including rough caps, to physics."""
    start = perf_counter()
    points, labels, families = [], [], {}
    for tile in generator.stair_collision_tiles:
        label = f"{tile['family']}[row={tile['row']}, col={tile['col']}]"
        try:
            tile_points = stair_edge_probe_points(tile["components"])
        except ValueError as exc:
            raise RuntimeError(f"Parkour stair-edge check {label}: {exc}") from exc
        points.append(tile_points)
        labels.extend([label] * len(tile_points))
        summary = families.setdefault(tile["family"], {"tiles": 0, "probes": 0})
        summary["tiles"] += 1
        summary["probes"] += len(tile_points)
    if not points:
        return {"tiles": 0, "probes": 0, "families": {}}
    report = validate_surface_heights(
        model, generator.terrain_mesh, np.concatenate(points), device,
        probe_labels=labels,
    )
    return {"tiles": len(points), **report, "families": families, "elapsed_s": round(perf_counter() - start, 3)}
