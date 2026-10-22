"""Numerical guard for degenerate MuJoCo Warp EPA witness triangles.

MuJoCo Warp 3.12 can select an almost-collinear EPA face for a convex
contact.  Its affine-coordinate calculation then produces huge cancelling
weights and a contact witness far outside both geometries. The guard excludes
ill-conditioned EPA faces from closest-face selection (but retains them for
horizon construction). Valid faces retain the upstream calculation; affine
degeneracies also have a bounded closest-edge last resort.

This is installed by the shared PUFO/parkour Newton manager. It is a
process-local patch for the installed MuJoCo Warp module; no site-package is
modified.
"""

import hashlib
import inspect
from importlib.metadata import version
from pathlib import Path

import warp as wp
from mujoco_warp._src import collision_gjk


_original_affine = None
_original_attach = None

GUARD_REVISION = "epa-face-v2"
VALIDATED_VERSIONS = {
    "mujoco": "3.12.0", "mujoco-warp": "3.12.0",
    "warp-lang": "1.17.0", "newton": "1.6.1",
}
_SOURCE_HASHES = {
    "collision_gjk": "fe2741a492f4e108a776aeb64c67dacf518845b16087209a6b9edda6367affdb",
    "collision_convex": "9a030d1756b0c0cc6bc9df818011b0c44859bc45420cd37254df7087773bc85c",
}


def validate_compatibility():
    """Reject unvalidated dependencies before mutating their collision code."""
    from mujoco_warp._src import collision_convex

    actual = {name: version(name) for name in VALIDATED_VERSIONS}
    if actual != VALIDATED_VERSIONS:
        raise RuntimeError(
            f"Contact guard {GUARD_REVISION}: unvalidated physics dependencies {actual}; "
            f"expected {VALIDATED_VERSIONS}. Use requirements-physics.txt or validate the upgrade first."
        )
    for module in (collision_gjk, collision_convex):
        name = module.__name__.rsplit(".", 1)[-1]
        digest = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        if digest != _SOURCE_HASHES[name]:
            raise RuntimeError(f"Contact guard: unvalidated {name} source; rerun collision regressions before upgrading.")
    for name, expected in {
        "_tri_affine_coord": ("v1", "v2", "v3", "p"),
        "_attach_face": ("pt", "idx", "v1", "v2", "v3"),
        "_invalidate_face": ("face",),
    }.items():
        function = getattr(getattr(collision_gjk, name, None), "func", None)
        if function is None or tuple(inspect.signature(function).parameters) != expected:
            raise RuntimeError(f"Contact guard: incompatible or already patched interface {name}")
    return actual


@wp.func
def _ill_conditioned_triangle(v1: wp.vec3, v2: wp.vec3, v3: wp.vec3) -> bool:
    edge1, edge2, edge3 = v2 - v1, v3 - v1, v3 - v2
    normal = wp.cross(edge1, edge2)
    scale = wp.max(wp.dot(edge1, edge1), wp.max(wp.dot(edge2, edge2), wp.dot(edge3, edge3)))
    # The old 1e-12 squared threshold missed captured faces just above it.
    relative_error = 64.0 * 1.1920928955078125e-7
    return wp.dot(normal, normal) <= relative_error * relative_error * wp.max(scale * scale, 1.0e-30)


@wp.func
def _guarded_attach(pt: collision_gjk.Polytope, idx: int, v1: int, v2: int, v3: int) -> float:
    distance = _original_attach(pt, idx, v1, v2, v3)
    # A zero return can mean exhausted capacity; no face was written then.
    if distance > 0.0:
        a = pt.vert[2 * v1] - pt.vert[2 * v1 + 1]
        b = pt.vert[2 * v2] - pt.vert[2 * v2 + 1]
        c = pt.vert[2 * v3] - pt.vert[2 * v3 + 1]
        if _ill_conditioned_triangle(a, b, c):
            # INVALID is not DELETED: keep the face's plane for EPA visibility
            # and horizon construction, but never select its unstable witness.
            pt.face[idx] = collision_gjk._invalidate_face(pt.face[idx])
    return distance


@wp.func
def _closest_triangle_weights(v1: wp.vec3, v2: wp.vec3, v3: wp.vec3, p: wp.vec3) -> wp.vec3:
    """Return barycentric weights for the closest point on a triangle edge."""
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
def _guarded_affine(v1: wp.vec3, v2: wp.vec3, v3: wp.vec3, p: wp.vec3) -> wp.vec3:
    """Preserve stock affine coordinates except for degenerate triangles."""
    if _ill_conditioned_triangle(v1, v2, v3):
        return _closest_triangle_weights(v1, v2, v3, p)
    result = _original_affine(v1, v2, v3, p)
    # A non-finite stock result is never a valid barycentric coordinate. This
    # branch is independent of the signed outside-triangle values used by GJK.
    if (not wp.isfinite(result[0]) or not wp.isfinite(result[1]) or not wp.isfinite(result[2])):
        return _closest_triangle_weights(v1, v2, v3, p)
    # Affine coordinates must sum to one, including signed outside-triangle
    # coordinates. Lost partition of unity translates a world-space witness
    # by an arbitrary multiple of the geometry's distance from the origin.
    if wp.abs(result[0] + result[1] + result[2] - 1.0) > 1.0e-3:
        return _closest_triangle_weights(v1, v2, v3, p)
    return result


def install():
    """Install the guard and return the original affine function."""
    global _original_affine, _original_attach

    if _original_affine is None:
        actual = validate_compatibility()
        _original_affine = collision_gjk._tri_affine_coord
        _original_attach = collision_gjk._attach_face
        collision_gjk._tri_affine_coord = _guarded_affine
        collision_gjk._attach_face = _guarded_attach
        print(f"[CONTACT_GUARD] enabled {GUARD_REVISION}, dependencies={actual}", flush=True)
    elif (collision_gjk._tri_affine_coord is not _guarded_affine
          or collision_gjk._attach_face is not _guarded_attach):
        raise RuntimeError("Contact guard was replaced while active; refusing unprotected physics")
    return _original_affine


def restore():
    """Restore the original function after the solver has been destroyed."""
    global _original_affine, _original_attach
    if _original_affine is None:
        return
    if collision_gjk._tri_affine_coord is _guarded_affine:
        collision_gjk._tri_affine_coord = _original_affine
    if collision_gjk._attach_face is _guarded_attach:
        collision_gjk._attach_face = _original_attach
    _original_affine = None
    _original_attach = None

