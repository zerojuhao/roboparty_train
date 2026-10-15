"""Shared native terrain import without whole-map convexification."""

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter as BaseTerrainImporter
from robolab.terrains.terrain_importer import TerrainImporter
from robolab.terrains.native_collision import HEIGHT_DATA, author_heightfield, flat_heightfield


def box_bounds(mesh):
    """Recognize exact axis-aligned boxes without convexifying other surfaces."""
    bounds = mesh.bounds
    vertices = np.unique(np.round(mesh.vertices, 7), axis=0)
    if len(vertices) != 8 or not mesh.is_watertight:
        return None
    corners = np.isclose(vertices, bounds[0], atol=1e-6, rtol=0) | np.isclose(
        vertices, bounds[1], atol=1e-6, rtol=0
    )
    return bounds if corners.all() and np.all(bounds[1] > bounds[0]) else None


def box_collider_pose(mesh):
    """Return exact box extents/translation and IsaacLab 3's XYZW quaternion."""
    if len(mesh.vertices) != 8:
        return None
    bounds = box_bounds(mesh)
    if bounds is not None:
        transform = trimesh.transformations.translation_matrix(bounds.mean(axis=0))
        extents = bounds[1] - bounds[0]
    else:
        box = mesh.bounding_box_oriented.primitive
        transform, extents = box.transform, box.extents
    local = (mesh.vertices - transform[:3, 3]) @ transform[:3, :3]
    if not np.allclose(np.abs(local), extents / 2, atol=1e-6):
        return None
    rotation = Rotation.from_matrix(transform[:3, :3]).as_quat()
    return tuple(extents), tuple(transform[:3, 3]), tuple(rotation)


def merge_box_strips(components):
    """Coalesce contiguous Y strips with identical X/Z bounds, preserving their union."""
    result, groups = [], {}
    for mesh, is_hf in components:
        bounds = mesh.bounds
        if not is_hf and len(mesh.vertices) == 8 and mesh.is_watertight and np.all(mesh.extents > 0):
            corners = np.isclose(mesh.vertices, bounds[0], atol=1e-8, rtol=0) | np.isclose(
                mesh.vertices, bounds[1], atol=1e-8, rtol=0
            )
            if corners.all():
                key = tuple(np.round(bounds[:, [0, 2]].flatten(), 8))
                groups.setdefault(key, []).append((bounds.copy(), mesh))
                continue
        result.append((mesh, is_hf))
    for strips in groups.values():
        strips.sort(key=lambda item: item[0][0, 1])
        bounds, original = strips[0]
        count = 1

        def emit():
            if count == 1:
                return original
            return trimesh.creation.box(
                extents=bounds[1] - bounds[0],
                transform=trimesh.transformations.translation_matrix(bounds.mean(axis=0)),
            )

        for following, mesh in strips[1:]:
            if following[0, 1] <= bounds[1, 1] + 1e-8:
                bounds[1, 1] = max(bounds[1, 1], following[1, 1])
                count += 1
            else:
                result.append((emit(), False))
                bounds, original, count = following, mesh, 1
        result.append((emit(), False))
    return result


def collision_kind(mesh, authored_heightfield=False):
    """Never let MuJoCo silently convexify a concave terrain component."""
    if HEIGHT_DATA in mesh.metadata:
        return "heightfield"
    # The mesh-box generator represents a flat floor as a zero-thickness box.
    # MuJoCo cannot compile that as a solid convex mesh.
    if mesh.extents[2] < 1e-8 and np.all(mesh.extents[:2] > 0):
        xy = np.unique(mesh.vertices[:, :2], axis=0)
        if len(xy) == 4 and (np.isclose(xy, mesh.bounds[0, :2]) | np.isclose(xy, mesh.bounds[1, :2])).all():
            return "heightfield"
    if not authored_heightfield and mesh.is_watertight:
        if mesh.is_convex:
            return "convex"
        raise ValueError("Parkour Newton collision requires separate convex solids, not a concave solid.")
    # Slope correction can shift boundary vertices by one grid cell. Permit
    # that small outline discrepancy, but reject holes and downward surfaces.
    triangles = mesh.triangles
    projected = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])[:, 2] * 0.5
    area = np.prod(mesh.extents[:2])
    if (area <= 0 or (not authored_heightfield and (mesh.euler_number != 1 or np.any(projected < -1e-8)))
            or not np.isclose(projected.sum(), area, rtol=0.01)):
        raise ValueError(
            "Unsupported parkour surface: expected a rectangular height field without overhangs; "
            f"bounds={mesh.bounds.tolist()}, euler={mesh.euler_number}, "
            f"coverage={projected.sum() / area if area else 0}, min_projected={projected.min()}"
        )
    return "heightfield"


class NativeTerrainImporter(TerrainImporter):
    def __init__(self, cfg):
        from isaaclab_newton.physics import NewtonCfg

        cfg = cfg.copy()
        self._native_collision = (
            isinstance(sim_utils.SimulationContext.instance().cfg.physics, NewtonCfg)
            and cfg.terrain_type == "generator"
        )
        if self._native_collision and cfg.terrain_generator is not None:
            cfg.terrain_generator.native_collision = True
        super().__init__(cfg)

    def _import_observation_mesh(self, name, mesh):
        TerrainImporter.import_mesh(self, name, mesh)

    def import_mesh(self, name, mesh):
        # PhysX already supports the authored triangle mesh directly.
        if not self._native_collision:
            return self._import_observation_mesh(name, mesh)

        # Splitting a welded height field breaks its degenerate cliff triangles.
        # The generator retains the actual source pieces, including rotated boxes.
        components = self.terrain_generator.collision_components
        kinds = [collision_kind(component, is_hf) for component, is_hf in components]
        self._import_observation_mesh(name, mesh)
        stage = sim_utils.get_current_stage()
        for prim in Usd.PrimRange(stage.GetPrimAtPath(f"{self.cfg.prim_path}/{name}")):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)

        self.collision_counts = {"convex": 0, "heightfield": 0, "box": 0}
        authored_boxes = []
        for index, ((component, _), kind) in enumerate(zip(components, kinds)):
            box = box_collider_pose(component) if kind == "convex" else None
            if box is not None:
                extents, translation, rotation = box
                cfg = sim_utils.CuboidCfg(
                    size=extents, visible=False,
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                    physics_material=self.cfg.physics_material,
                )
                cfg.func(f"{self.cfg.prim_path}/collision_box_{index:05d}", cfg,
                         translation=translation, orientation=rotation)
                authored_boxes.append(component.bounds)
                self.collision_counts["box"] += 1
                continue
            # Fixed-width names avoid prefix matches in Newton's ignore paths.
            name = f"collision_{kind}_{index:05d}"
            if kind == "heightfield":
                name = f"native_heightfields/{name}"
            BaseTerrainImporter.import_mesh(self, name, component)
            if kind == "heightfield":
                field = component if HEIGHT_DATA in component.metadata else flat_heightfield(component)
                if field is None:
                    raise ValueError("Native heightfield is missing authored samples.")
                author_heightfield(stage, self.terrain_prim_paths[-1], field, self.cfg.physics_material)
            self.collision_counts[kind] += 1
        self.authored_boxes = np.asarray(authored_boxes).reshape(-1, 2, 3)
        print(f"[Native terrain] collision components: {self.collision_counts}", flush=True)

    def validate_solver_geometry(self, model):
        """Check native colliders survived USD import; never accept a map hull."""
        import mujoco

        static = model.geom_bodyid == 0
        actual = {
            "convex": int(np.count_nonzero(static & (model.geom_type == mujoco.mjtGeom.mjGEOM_MESH))),
            "heightfield": int(np.count_nonzero(static & (model.geom_type == mujoco.mjtGeom.mjGEOM_HFIELD))),
            "box": int(np.count_nonzero(static & (model.geom_type == mujoco.mjtGeom.mjGEOM_BOX))),
        }
        if actual != self.collision_counts:
            raise RuntimeError(f"Native terrain collider import mismatch: {actual}, expected={self.collision_counts}")
        from .collision_validation import terrain_probe_points, validate_surface_heights

        self.collision_validation = validate_surface_heights(
            model, self.terrain_generator.terrain_mesh, terrain_probe_points(self.terrain_generator), self.device,
        )
        print(f"[Native terrain] collision/observation ray check: {self.collision_validation}", flush=True)
        return actual


def validate_terrain_colliders(env, env_ids=None):
    """One-time solver validation, after scene construction and before rollout."""
    terrain = env.scene.terrain
    if not getattr(terrain, "_native_collision", False) or getattr(terrain, "_collision_validated", False):
        return
    from isaaclab_newton.physics import NewtonManager

    solver = NewtonManager._solver
    if solver is None or not hasattr(solver, "mj_model"):
        raise RuntimeError("Native terrain validation requires an initialized MuJoCo Warp solver")
    terrain.validate_solver_geometry(solver.mj_model)
    terrain._collision_validated = True
