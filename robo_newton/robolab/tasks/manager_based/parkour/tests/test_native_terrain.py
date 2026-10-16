"""Native terrain import must preserve axes, signed heights and task distribution."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import newton
import numpy as np
from pxr import Usd, UsdGeom

from robolab.terrains.native_collision import (
    HEIGHT_DATA, author_heightfield, heightfield_mesh, inject_heightfields, terrace_meshes,
)
from robolab.terrains.terrain_generator import GuardedTerrainGenerator
from robolab.tasks.manager_based.ufo.pufo_terrain_cfg import PUFO_TERRAIN_CFG
from robolab.tasks.manager_based.parkour.terrain_generator_cfg import ROUGH_TERRAINS_CFG
from robolab.tasks.manager_based.parkour.terrain_importer import ParkourTerrainGenerator, collision_kind


class TestNativeTerrain(unittest.TestCase):
    def make_field(self, heights, offset=(0., 0., 0.)):
        mesh = heightfield_mesh(heights, (.2, .3), offset)
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, "/World/terrain")
        author_heightfield(stage, "/World/terrain", mesh,
                           SimpleNamespace(dynamic_friction=.63, restitution=.12))
        builder = newton.ModelBuilder()
        with patch.object(newton.Heightfield, "create_from_mesh", side_effect=AssertionError("rasterization")):
            paths = inject_heightfields(stage, builder, ["/World"])
        self.assertEqual(paths, ["/World/terrain"])
        self.assertAlmostEqual(builder.shape_material_mu[0], .63)
        self.assertAlmostEqual(builder.shape_material_restitution[0], .12)
        field, transform = builder.shape_source[0], builder.shape_transform[0]
        spec = mujoco.MjSpec()
        spec.add_hfield(name="terrain", nrow=field.nrow, ncol=field.ncol,
                        size=(field.hx, field.hy, max(field.max_z, 1e-4), 1e-4),
                        userdata=field.data.ravel())
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="terrain",
                               pos=transform.p, quat=(1., 0., 0., 0.))
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return mesh, model, data

    def test_signed_asymmetric_heights_match_mujoco_rays(self):
        heights = np.array([[-.3, .1, .2, -.1], [.2, -.2, .4, .3], [.5, .1, -.4, .2]])
        mesh, model, data = self.make_field(heights, (12., -7., -.8))
        rng = np.random.default_rng(42)
        starts = np.c_[rng.uniform(mesh.bounds[0, 0] + .001, mesh.bounds[1, 0] - .001, 50),
                       rng.uniform(mesh.bounds[0, 1] + .001, mesh.bounds[1, 1] - .001, 50), np.full(50, 5.)]
        directions = np.tile((0., 0., -1.), (50, 1))
        hits, rays, _ = mesh.ray.intersects_location(starts, directions, multiple_hits=False)
        self.assertEqual(len(rays), 50)
        for hit, ray in zip(hits, rays):
            distance = mujoco.mj_ray(model, data, starts[ray], directions[ray], None, True, -1,
                                     np.empty(1, dtype=np.int32))
            self.assertAlmostEqual(5. - distance, hit[2], places=5)

    def test_shared_validator_checks_fields_and_rejects_height_drift(self):
        from robolab.terrains.collision_validation import validate_surface_heights
        mesh, model, _ = self.make_field(np.array([[-.3, .1, .2], [.2, -.2, .4], [.5, .1, -.4]]))
        points = np.array([[.1, .1], [.31, .43]])
        groups = model.geom_group.copy()
        report = validate_surface_heights(model, mesh, points)
        self.assertEqual(report["hits"], 2)
        self.assertLess(report["max_error"], 1e-5)
        np.testing.assert_array_equal(model.geom_group, groups)
        model.geom_pos[:, 2] += .1
        with self.assertRaisesRegex(RuntimeError, "collision/observation mismatch"):
            validate_surface_heights(model, mesh, points)

    def test_shared_validator_preserves_gaps_and_excludes_robot(self):
        import trimesh
        from robolab.terrains.collision_validation import validate_surface_heights
        boxes = []
        for x in (-1., 1.):
            box = trimesh.creation.box((1., 1., .2))
            box.apply_translation((x, 0., .1))
            boxes.append(box)
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
            <geom type="box" pos="-1 0 .1" size=".5 .5 .1"/>
            <geom type="box" pos="1 0 .1" size=".5 .5 .1"/>
            <body pos="1 0 1"><freejoint/><geom type="sphere" size=".1"/></body>
            </worldbody></mujoco>''')
        report = validate_surface_heights(model, trimesh.util.concatenate(boxes),
                                          np.array([[-1., 0.], [1., 0.], [0., 0.]]))
        self.assertEqual(report["hits"], 2)
        # A collider shifted off its visible box creates a hit/miss mismatch.
        model.geom_pos[0, 0] -= 2.
        with self.assertRaisesRegex(RuntimeError, "collision/observation mismatch"):
            validate_surface_heights(model, trimesh.util.concatenate(boxes), np.array([[-1., 0.]]))

    def test_shared_importer_and_probe_coverage(self):
        from robolab.terrains.native_importer import NativeTerrainImporter
        from robolab.terrains.collision_validation import terrain_probe_points
        from robolab.tasks.manager_based.parkour.terrain_importer import ParkourTerrainImporter
        from robolab.tasks.manager_based.ufo.pufo_terrain_importer import PUFOTerrainImporter
        self.assertTrue(issubclass(ParkourTerrainImporter, NativeTerrainImporter))
        self.assertTrue(issubclass(PUFOTerrainImporter, NativeTerrainImporter))
        origins = np.array([[[0., 0., 0.], [10., 0., 0.]]])
        generator = SimpleNamespace(cfg=SimpleNamespace(size=(10., 10.)), terrain_origins=origins,
                                    collision_components=[])
        points = terrain_probe_points(generator)
        self.assertEqual(len(points), 10)
        for origin in origins.reshape(-1, 3):
            self.assertLess(np.linalg.norm(points - origin[:2], axis=1).min(), 1e-3)

    def test_native_cfg_copies_and_disables_legacy_cache(self):
        from robolab.terrains.native_importer import NativeTerrainImporter
        from robolab.terrains.terrain_importer_cfg import TerrainImporterCfg

        source_generator = ROUGH_TERRAINS_CFG.copy()
        source_generator.use_cache = True
        source_generator.native_collision = False
        cfg = TerrainImporterCfg(terrain_type="generator", terrain_generator=source_generator)
        prepared = NativeTerrainImporter._prepare_native_cfg(cfg, enabled=True)

        self.assertIsNot(prepared, cfg)
        self.assertIsNot(prepared.terrain_generator, source_generator)
        self.assertFalse(source_generator.native_collision)
        self.assertTrue(source_generator.use_cache)
        self.assertTrue(prepared.terrain_generator.native_collision)
        self.assertFalse(prepared.terrain_generator.use_cache)

    def test_rotated_box_pose_matches_actual_usd_xyzw_convention(self):
        import itertools
        import trimesh
        import isaaclab.sim as sim_utils
        from robolab.terrains.native_importer import box_collider_pose
        mesh = trimesh.creation.box((.3, .7, .2))
        mesh.apply_transform(trimesh.transformations.rotation_matrix(.73, (0, 0, 1)))
        mesh.apply_translation((-36.9, 46.5, .1))
        extents, translation, rotation = box_collider_pose(mesh)
        stage = Usd.Stage.CreateInMemory()
        sim_utils.create_prim("/World/box", prim_type="Xform", translation=translation,
                              orientation=rotation, stage=stage)
        transform = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath("/World/box"))).T
        corners = np.array(list(itertools.product((-1., 1.), repeat=3))) * np.asarray(extents) / 2
        world = corners @ transform[:3, :3].T + transform[:3, 3]
        distances = np.linalg.norm(world[:, None] - mesh.vertices[None], axis=-1)
        # USD translations are float32; coordinates near 50 m have micron ULPs.
        self.assertLess(distances.min(axis=1).max(), 1e-5)

    def test_box_float32_seam_is_avoided_without_hiding_height_errors(self):
        import trimesh
        from robolab.terrains.collision_validation import validate_surface_heights
        spec = mujoco.MjSpec()
        boxes = []
        for x in (-5.9, -5.1):
            box = trimesh.creation.box((.8, .2, .2))
            box.apply_translation((x, 14.4, 0.))
            boxes.append(box)
            spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                                    pos=np.array([x, 14.4, 0.], dtype=np.float32),
                                    size=np.array([.4, .1, .1], dtype=np.float32))
        model = spec.compile()
        mesh = trimesh.util.concatenate(boxes)
        with self.assertRaisesRegex(RuntimeError, "collision/observation mismatch"):
            validate_surface_heights(model, mesh, [[-5.5, 14.31]])
        report = validate_surface_heights(model, mesh,
                                          [[-5.5 - .00037, 14.31], [-5.5 + .00037, 14.31]])
        self.assertEqual(report["hits"], 2)

    def test_flat_negative_height_does_not_get_normalized_to_zero(self):
        _, model, data = self.make_field(np.full((3, 4), -1.25), (3., 4., -.5))
        distance = mujoco.mj_ray(model, data, np.array([3.2, 4.3, 2.]), np.array([0., 0., -1.]),
                                 None, True, -1, np.empty(1, dtype=np.int32))
        self.assertAlmostEqual(2. - distance, -1.75, places=6)

    def test_terraces_keep_step_edges_and_noise(self):
        levels = np.zeros((9, 7))
        levels[3:6] = .2
        noise = np.linspace(-.01, .01, 7)[None, :]
        meshes = terrace_meshes(levels + noise, levels, .05)
        boxes = [m for m in meshes if HEIGHT_DATA not in m.metadata]
        caps = [m for m in meshes if HEIGHT_DATA in m.metadata]
        self.assertEqual(len(boxes), 3)
        self.assertEqual(len(caps), 3)
        self.assertAlmostEqual(boxes[1].bounds[0, 0], .125)
        self.assertAlmostEqual(boxes[1].bounds[1, 0], .275)
        self.assertTrue(all(np.ptp(m.metadata[HEIGHT_DATA]) > .019 for m in caps))

    def test_all_parkour_families_have_authored_samples(self):
        cfg = ROUGH_TERRAINS_CFG.copy()
        cfg.num_rows = 1
        cfg.difficulty_range = (.95, 1.)
        for sub in cfg.sub_terrains.values():
            sub.flat_patch_sampling = None
        generated = ParkourTerrainGenerator(cfg, device="cpu")
        from robolab.terrains.native_collision import flat_heightfield
        for mesh, is_hf in generated.collision_components:
            if collision_kind(mesh, is_hf) == "heightfield":
                self.assertTrue(HEIGHT_DATA in mesh.metadata or flat_heightfield(mesh) is not None)

    def test_pufo_guard_origins_and_distribution_are_preserved(self):
        cfg = PUFO_TERRAIN_CFG.copy()
        cfg.num_rows = 1
        cfg.use_cache = False
        np.random.seed(7)
        baseline = GuardedTerrainGenerator(cfg.copy(), "cpu")
        cfg.native_collision = True
        np.random.seed(7)
        generated = GuardedTerrainGenerator(cfg, "cpu")
        np.testing.assert_array_equal(generated.terrain_origins, baseline.terrain_origins)
        np.testing.assert_array_equal(generated.subterrain_index_grid, baseline.subterrain_index_grid)
        self.assertEqual(len(generated.subterrain_specific_cfgs), 7)
        fields = [m for m, _ in generated.collision_components if HEIGHT_DATA in m.metadata]
        self.assertGreater(len(fields), 20)  # includes the surrounding guard ring
        self.assertTrue(any(m.bounds[0, 0] < -5 for m in fields))
        self.assertTrue(any(m.bounds[1, 0] > 5 for m in fields))

    def test_invalid_height_arrays_fail_closed(self):
        for heights in (np.zeros((1, 3)), np.full((3, 3), np.nan)):
            with self.assertRaises(ValueError):
                heightfield_mesh(heights, .1)

    def test_owned_container_is_one_ignore_path_and_rejects_unrelated_children(self):
        stage = Usd.Stage.CreateInMemory()
        material = SimpleNamespace(dynamic_friction=.8, restitution=0.)
        for i in range(3):
            path = f"/World/native_heightfields/hf_{i:05d}"
            UsdGeom.Xform.Define(stage, path)
            author_heightfield(stage, path, heightfield_mesh(np.zeros((3, 4)), .1), material)
        self.assertEqual(inject_heightfields(stage, newton.ModelBuilder(), ["/World"]),
                         ["/World/native_heightfields"])
        UsdGeom.Cube.Define(stage, "/World/native_heightfields/unrelated_box")
        with self.assertRaisesRegex(ValueError, "unrelated prim"):
            inject_heightfields(stage, newton.ModelBuilder(), ["/World"])

    def test_parent_transform_cannot_silently_move_only_visual_geometry(self):
        stage = Usd.Stage.CreateInMemory()
        parent = UsdGeom.Xform.Define(stage, "/World")
        parent.AddTranslateOp().Set((1., 0., 0.))
        UsdGeom.Xform.Define(stage, "/World/field")
        author_heightfield(stage, "/World/field", heightfield_mesh(np.zeros((3, 4)), .1),
                           SimpleNamespace(dynamic_friction=1., restitution=0.))
        with self.assertRaisesRegex(ValueError, "identity USD transforms"):
            inject_heightfields(stage, newton.ModelBuilder(), ["/World"])


if __name__ == "__main__":
    unittest.main()
