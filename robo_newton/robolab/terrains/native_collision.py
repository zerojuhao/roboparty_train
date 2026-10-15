"""Authored terrain samples shared by observation meshes and Newton colliders."""

import numpy as np
import trimesh


HEIGHT_DATA = "robolab_heightfield"


def heightfield_mesh(heights, spacing, offset=(0., 0., 0.)):
    """Build an unshifted grid; metadata survives Trimesh.copy, not concatenation."""
    from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh

    heights = np.asarray(heights, dtype=np.float32)
    if heights.ndim != 2 or min(heights.shape) < 2 or not np.isfinite(heights).all():
        raise ValueError("Native terrain requires a finite height array of at least 2x2.")
    sx, sy = np.broadcast_to(spacing, (2,))
    if min(sx, sy) <= 0:
        raise ValueError("Native terrain grid spacing must be positive.")
    vertices, faces = convert_height_field_to_mesh(heights, 1., 1., None)
    vertices[:, :2] *= (sx, sy)
    # MuJoCo also collides with the vertical perimeter down to min_z - 1e-4.
    # Include it in the sensor mesh, especially along noisy stair risers.
    nx, ny = heights.shape
    perimeter = np.r_[np.arange(ny), np.arange(1, nx) * ny + ny - 1,
                      (nx - 1) * ny + np.arange(ny - 2, -1, -1),
                      np.arange(nx - 2, 0, -1) * ny]
    lower = vertices[perimeter].copy()
    lower[:, 2] = heights.min() - 1e-4
    bottom = np.arange(len(lower)) + len(vertices)
    next_top, next_bottom = np.roll(perimeter, -1), np.roll(bottom, -1)
    sides = np.concatenate((np.stack((perimeter, next_top, bottom), axis=1),
                            np.stack((next_top, next_bottom, bottom), axis=1)))
    vertices, faces = np.concatenate((vertices, lower)), np.concatenate((faces, sides))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.apply_translation(offset)
    mesh.metadata[HEIGHT_DATA] = heights.copy()
    return mesh


def flat_heightfield(mesh):
    """Represent an axis-aligned rectangular zero-thickness floor exactly."""
    if mesh.extents[2] > 1e-8 or np.any(mesh.extents[:2] <= 0):
        return None
    xy = np.unique(mesh.vertices[:, :2], axis=0)
    if len(xy) != 4 or not (np.isclose(xy, mesh.bounds[0, :2]) | np.isclose(xy, mesh.bounds[1, :2])).all():
        return None
    return heightfield_mesh(np.full((2, 2), mesh.bounds[0, 2]), mesh.extents[:2], (*mesh.bounds[0, :2], 0.))


def terrace_meshes(heights, levels, spacing):
    """Partition authored stair levels into boxes with optional rough heightfield caps.

    Boundaries lie halfway between original samples. Noise is sampled only
    within its own tread, never interpolated across a riser. No noise is removed.
    """
    heights = np.asarray(heights, dtype=np.float64)
    levels = np.asarray(levels)
    if heights.shape != levels.shape:
        raise ValueError("Stair level and height arrays must have the same shape.")
    nx, ny = heights.shape
    rectangles, active = [], {}
    for x in range(nx):
        cuts = np.r_[0, np.flatnonzero(np.diff(levels[x])) + 1, ny]
        following = {}
        for y0, y1 in zip(cuts[:-1], cuts[1:]):
            key = (int(y0), int(y1), float(levels[x, y0]))
            rect = active.pop(key, [x, x, int(y0), int(y1)])
            rect[1] = x + 1
            following[key] = rect
        rectangles.extend(active.values())
        active = following
    rectangles.extend(active.values())
    bottom = float(heights.min()) - 0.1
    meshes = []
    for x0, x1, y0, y1 in rectangles:
        lo = np.maximum((x0 - .5, y0 - .5), 0) * spacing
        hi = np.minimum((x1 - .5, y1 - .5), (nx - 1, ny - 1)) * spacing
        patch = heights[x0:x1, y0:y1]
        top = float(patch.min())
        box = trimesh.creation.box(
            extents=(* (hi - lo), top - bottom),
            transform=trimesh.transformations.translation_matrix((*((lo + hi) / 2), (top + bottom) / 2)),
        )
        meshes.append(box)
        if np.ptp(patch) > 0:
            # Include both edges without extrapolating across the neighboring step.
            # Half-cell edges and every original sample must both land on the
            # native uniform grid; a half-spacing cap retains those samples.
            count = np.maximum(np.rint((hi - lo) / (spacing / 2)).astype(int) + 1, 2)
            xs, ys = (np.linspace(lo[i], hi[i], count[i]) / spacing for i in range(2))
            along_y = np.stack([np.interp(ys, np.arange(y0, y1), row) for row in patch])
            cap = np.stack([np.interp(xs, np.arange(x0, x1), col) for col in along_y.T], axis=1)
            meshes.append(heightfield_mesh(cap, (hi - lo) / (count - 1), (*lo, 0.)))
    return meshes


def author_heightfield(stage, path, mesh, material):
    """Store samples on the collider prim, avoiding process-global registries."""
    from pxr import Gf, Sdf, Vt

    heights = np.asarray(mesh.metadata[HEIGHT_DATA], dtype=np.float32)
    prim = stage.GetPrimAtPath(path)
    parent = prim.GetParent()
    if parent.GetName() == "native_heightfields":
        parent.CreateAttribute("robolab:heightfield:container", Sdf.ValueTypeNames.Bool).Set(True)
    for name, kind, value in (
        ("heights", Sdf.ValueTypeNames.FloatArray, Vt.FloatArray(heights.ravel().tolist())),
        ("shape", Sdf.ValueTypeNames.Int2, Gf.Vec2i(*heights.shape)),
        ("bounds", Sdf.ValueTypeNames.Double4, tuple(mesh.bounds[:, :2].ravel())),
        ("zOffset", Sdf.ValueTypeNames.Double, float(mesh.bounds[1, 2] - heights.max())),
        ("friction", Sdf.ValueTypeNames.Double, float(material.dynamic_friction)),
        ("restitution", Sdf.ValueTypeNames.Double, float(material.restitution)),
    ):
        prim.CreateAttribute(f"robolab:heightfield:{name}", kind).Set(value)


def inject_heightfields(stage, builder, root_paths):
    """Import raw X/Y samples as MuJoCo's Y/X grid, with explicit material."""
    import newton
    import warp as wp
    from pxr import Usd, UsdGeom

    ignored = []
    containers = set()
    cache = UsdGeom.XformCache()
    for root in root_paths:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            attr = prim.GetAttribute("robolab:heightfield:heights")
            if not attr or not attr.HasAuthoredValue():
                continue
            def get(name):
                return prim.GetAttribute(f"robolab:heightfield:{name}").Get()
            # Vertices and metadata are already in world coordinates. Reject
            # transformed parents instead of silently placing collision elsewhere.
            if not np.allclose(np.asarray(cache.GetLocalToWorldTransform(prim)), np.eye(4)):
                raise ValueError(f"Native terrain requires identity USD transforms: {prim.GetPath()}")
            heights = np.asarray(attr.Get(), dtype=np.float32).reshape(tuple(get("shape")))
            if not np.isfinite(heights).all():
                raise ValueError(f"Non-finite native heightfield: {prim.GetPath()}")
            lo, hi = np.asarray(get("bounds")).reshape(2, 2)
            field = newton.Heightfield(heights.T, heights.shape[1], heights.shape[0],
                                      hx=(hi[0] - lo[0]) / 2, hy=(hi[1] - lo[1]) / 2)
            z = field.min_z + get("zOffset")
            field.max_z -= field.min_z
            field.min_z = 0.
            cfg = builder.default_shape_cfg.copy()
            cfg.mu, cfg.restitution = get("friction"), get("restitution")
            builder.add_shape_heightfield(heightfield=field, cfg=cfg,
                                          xform=wp.transform((*((lo + hi) / 2), z), wp.quat_identity()),
                                          label=prim.GetPath().pathString)
            ignored.append(prim.GetPath().pathString)
            parent = prim.GetParent()
            if parent.GetAttribute("robolab:heightfield:container").Get():
                containers.add(parent.GetPath().pathString)
    if ignored:
        print(f"[Native terrain] imported {len(ignored)} authored height arrays (no ray rasterization)", flush=True)
    # Thousands of individual regexes overflow Python's regex cache in add_usd.
    # Only collapse explicitly owned containers whose children are all fields.
    for path in containers:
        children = stage.GetPrimAtPath(path).GetChildren()
        if not all(child.GetPath().pathString in ignored for child in children):
            raise ValueError(f"Native heightfield container contains an unrelated prim: {path}")
    return sorted(containers) + [path for path in ignored if path.rsplit("/", 1)[0] not in containers]
