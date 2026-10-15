"""Process-local affine-coordinate experiments for a captured PUFO collision."""

import warp as wp
from mujoco_warp._src import collision_gjk


original_affine = collision_gjk._tri_affine_coord
original_attach = collision_gjk._attach_face


@wp.func
def attach_without_slivers(pt: collision_gjk.Polytope, idx: int, v1: int, v2: int, v3: int) -> float:
    distance = original_attach(pt, idx, v1, v2, v3)
    if distance <= 0.0:
        return distance
    a = pt.vert[2 * v1] - pt.vert[2 * v1 + 1]
    b = pt.vert[2 * v2] - pt.vert[2 * v2 + 1]
    c = pt.vert[2 * v3] - pt.vert[2 * v3 + 1]
    e1, e2, e3 = b - a, c - a, c - b
    normal = wp.cross(e1, e2)
    scale = wp.max(wp.dot(e1, e1), wp.max(wp.dot(e2, e2), wp.dot(e3, e3)))
    tolerance = 64.0 * 1.1920928955078125e-7
    if wp.dot(normal, normal) <= tolerance * tolerance * wp.max(scale * scale, 1.0e-30):
        pt.face[idx] = collision_gjk._invalidate_face(pt.face[idx])
    return distance


@wp.func
def _closest_triangle_weights(v1: wp.vec3, v2: wp.vec3, v3: wp.vec3, p: wp.vec3) -> wp.vec3:
    """Return weights for the closest point on a triangle boundary/interior."""
    best = wp.vec3(1.0, 0.0, 0.0)
    best_dist = wp.dot(v1 - p, v1 - p)
    edge = v2 - v1
    denom = wp.dot(edge, edge)
    t = wp.where(denom > 1.0e-30, wp.clamp(wp.dot(p - v1, edge) / denom, 0.0, 1.0), 0.0)
    point = v1 + t * edge
    distance = wp.dot(point - p, point - p)
    if distance < best_dist:
        best = wp.vec3(1.0 - t, t, 0.0)
        best_dist = distance

    edge = v3 - v1
    denom = wp.dot(edge, edge)
    t = wp.where(denom > 1.0e-30, wp.clamp(wp.dot(p - v1, edge) / denom, 0.0, 1.0), 0.0)
    point = v1 + t * edge
    distance = wp.dot(point - p, point - p)
    if distance < best_dist:
        best = wp.vec3(1.0 - t, 0.0, t)
        best_dist = distance

    edge = v3 - v2
    denom = wp.dot(edge, edge)
    t = wp.where(denom > 1.0e-30, wp.clamp(wp.dot(p - v2, edge) / denom, 0.0, 1.0), 0.0)
    point = v2 + t * edge
    distance = wp.dot(point - p, point - p)
    if distance < best_dist:
        best = wp.vec3(0.0, 1.0 - t, t)
    return best


@wp.func
def affine_clamped(v1: wp.vec3, v2: wp.vec3, v3: wp.vec3, p: wp.vec3) -> wp.vec3:
    # Keep the stock coordinates for normal triangles.  Only replace the
    # undefined near-collinear case, so outside-triangle tests retain their
    # original signed barycentric coordinates.
    edge1 = v2 - v1
    edge2 = v3 - v1
    normal = wp.cross(edge1, edge2)
    edge3 = v3 - v2
    edge_scale = wp.max(wp.dot(edge1, edge1), wp.max(wp.dot(edge2, edge2), wp.dot(edge3, edge3)))
    if wp.dot(normal, normal) <= 1.0e-12 * wp.max(edge_scale * edge_scale, 1.0e-30):
        return _closest_triangle_weights(v1, v2, v3, p)
    return original_affine(v1, v2, v3, p)


@wp.func
def trace_affine(v1: wp.vec3, v2: wp.vec3, v3: wp.vec3, p: wp.vec3) -> wp.vec3:
    result = original_affine(v1, v2, v3, p)
    total = result[0] + result[1] + result[2]
    if wp.abs(total - 1.0) > 0.01:
        wp.printf("AFFINE_BAD sum=%g weights=(%g,%g,%g) v1=(%.9g,%.9g,%.9g) v2=(%.9g,%.9g,%.9g) v3=(%.9g,%.9g,%.9g) p=(%.9g,%.9g,%.9g)\n",
                  total, result[0], result[1], result[2],
                  v1[0], v1[1], v1[2], v2[0], v2[1], v2[2], v3[0], v3[1], v3[2], p[0], p[1], p[2])
    return result


@wp.func
def affine_float64(v1: wp.vec3, v2: wp.vec3, v3: wp.vec3, p: wp.vec3) -> wp.vec3:
    a = wp.vec3d(wp.float64(v1[0]), wp.float64(v1[1]), wp.float64(v1[2]))
    b = wp.vec3d(wp.float64(v2[0]), wp.float64(v2[1]), wp.float64(v2[2]))
    c = wp.vec3d(wp.float64(v3[0]), wp.float64(v3[1]), wp.float64(v3[2]))
    q = wp.vec3d(wp.float64(p[0]), wp.float64(p[1]), wp.float64(p[2]))
    e1 = b - a
    e2 = c - a
    normal = wp.cross(e1, e2)
    axis = int(0)
    if wp.abs(normal[1]) > wp.abs(normal[axis]):
        axis = 1
    if wp.abs(normal[2]) > wp.abs(normal[axis]):
        axis = 2
    denominator = normal[axis]
    if denominator == wp.float64(0.0):
        # Do not invent behavior for exact degeneracy in this diagnostic experiment.
        return original_affine(v1, v2, v3, p)
    l2 = wp.cross(q - a, e2)[axis] / denominator
    l3 = wp.cross(e1, q - a)[axis] / denominator
    l1 = wp.float64(1.0) - l2 - l3
    return wp.vec3(wp.float32(l1), wp.float32(l2), wp.float32(l3))


def install(mode):
    if mode == "faces":
        from robolab.utils import mjwarp_contact_guard

        mjwarp_contact_guard.install()
        collision_gjk._attach_face = attach_without_slivers
    elif mode == "trace":
        collision_gjk._tri_affine_coord = trace_affine
    elif mode == "float64":
        collision_gjk._tri_affine_coord = affine_float64
    elif mode == "clamp":
        collision_gjk._tri_affine_coord = affine_clamped
