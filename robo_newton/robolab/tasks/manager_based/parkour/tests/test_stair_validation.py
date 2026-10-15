"""Stair-specific startup probes must catch missing/misaligned edges and caps."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import newton
import numpy as np
import trimesh
from pxr import Usd, UsdGeom

from robolab.terrains.native_collision import HEIGHT_DATA, author_heightfield, inject_heightfields, terrace_meshes
from robolab.terrains.native_importer import NativeTerrainImporter, box_bounds, validate_terrain_colliders
from robolab.tasks.manager_based.parkour.stair_validation import stair_edge_probe_points, validate_stair_edges
from robolab.tasks.manager_based.parkour.terrain_importer import ParkourTerrainGenerator, ParkourTerrainImporter
from robolab.tasks.manager_based.parkour.terrain_generator_cfg import ROUGH_TERRAINS_CFG


class TestStairValidation(unittest.TestCase):
    def make_stairs(self, inverted=False, rough=False, omit_caps=False):
        levels = np.repeat([0., .2, .4], 8)[:, None] * np.ones((1, 13))
        if inverted:
            levels = -levels
        heights = levels + (np.linspace(0., .03, 13)[None, :] if rough else 0.)
        meshes = terrace_meshes(heights, levels, .05)
        for mesh in meshes:
            mesh.apply_translation((12., -7., -.3))
        components = [(mesh, HEIGHT_DATA in mesh.metadata) for mesh in meshes]
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, "/World/native_heightfields")
        material = SimpleNamespace(dynamic_friction=.8, restitution=0.)
        spec = mujoco.MjSpec()
        for i, (mesh, is_hf) in enumerate(components):
            if is_hf:
                if not omit_caps:
                    UsdGeom.Xform.Define(stage, f"/World/native_heightfields/cap_{i}")
                    author_heightfield(stage, f"/World/native_heightfields/cap_{i}", mesh, material)
            else:
                bounds = box_bounds(mesh)
                spec.worldbody.add_geom(name=f"box_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                        pos=bounds.mean(axis=0), size=(bounds[1] - bounds[0]) / 2)
        if rough and not omit_caps:
            builder = newton.ModelBuilder()
            inject_heightfields(stage, builder, ["/World"])
            for i, (field, transform) in enumerate(zip(builder.shape_source, builder.shape_transform)):
                name = f"cap_{i}"
                spec.add_hfield(name=name, nrow=field.nrow, ncol=field.ncol,
                                size=(field.hx, field.hy, max(field.max_z, 1e-4), 1e-4),
                                userdata=field.data.ravel())
                spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=name,
                                        pos=transform.p, quat=(1., 0., 0., 0.))
        generator = SimpleNamespace(
            terrain_mesh=trimesh.util.concatenate(meshes),
            stair_collision_tiles=[{"family": "test_stairs", "row": 3, "col": 2, "components": components}],
        )
        return spec.compile(), generator

    def test_each_edge_gets_both_near_and_far_probes(self):
        box = trimesh.creation.box((1., 2., .2))
        points = stair_edge_probe_points([(box, False)])
        self.assertEqual(len(points), 48)  # four edges, three positions, two distances, two sides
        for axis in (0, 1):
            for edge in box.bounds[:, axis]:
                for offset in (-.01, -.002, .002, .01):
                    self.assertEqual(np.isclose(points[:, axis], edge + offset).sum(), 3)
        np.testing.assert_array_equal(points, stair_edge_probe_points([(box, False), (box.copy(), False)]))

    def test_ascending_descending_and_rough_stairs_match(self):
        for inverted in (False, True):
            for rough in (False, True):
                with self.subTest(inverted=inverted, rough=rough):
                    model, generator = self.make_stairs(inverted, rough)
                    report = validate_stair_edges(model, generator)
                    self.assertEqual(report["tiles"], 1)
                    self.assertGreater(report["probes"], 100)
                    self.assertLess(report["max_error"], 1e-5)
                    self.assertEqual(report["families"]["test_stairs"]["probes"], report["probes"])

    def test_three_mm_edge_shift_is_detected_and_located(self):
        model, generator = self.make_stairs()
        model.geom_pos[1, 0] += .003
        with self.assertRaisesRegex(RuntimeError, r"test_stairs\[row=3, col=2\].*xy="):
            validate_stair_edges(model, generator)

    def test_rough_cap_cannot_be_replaced_by_box_height(self):
        model, generator = self.make_stairs(rough=True, omit_caps=True)
        with self.assertRaisesRegex(RuntimeError, "collision/observation mismatch"):
            validate_stair_edges(model, generator)

    def test_missing_collision_and_filled_valley_are_detected(self):
        for inverted in (False, True):
            model, generator = self.make_stairs(inverted)
            model.geom_pos[1, 2] += .15 if inverted else -2.
            with self.assertRaisesRegex(RuntimeError, "collision/observation mismatch"):
                validate_stair_edges(model, generator)

    def test_no_stairs_is_valid_but_empty_stair_metadata_is_not(self):
        generator = SimpleNamespace(stair_collision_tiles=[])
        self.assertEqual(validate_stair_edges(None, generator)["probes"], 0)
        generator.stair_collision_tiles = [{"family": "broken", "row": 0, "col": 1, "components": []}]
        with self.assertRaisesRegex(RuntimeError, "broken.*no authored collision edges"):
            validate_stair_edges(None, generator)

    def test_generated_metadata_covers_every_stair_family_and_row(self):
        cfg = ROUGH_TERRAINS_CFG.copy()
        cfg.num_rows = 2
        names = ("pyramid_stairs", "pyramid_stairs_inv", "trapezoid_stairs", "trapezoid_stairs_inv")
        cfg.sub_terrains = {name: cfg.sub_terrains[name] for name in (*names, "perlin_rough")}
        cfg.difficulty_range = (.8, 1.)
        for sub in cfg.sub_terrains.values():
            sub.flat_patch_sampling = None
        generator = ParkourTerrainGenerator(cfg, "cpu")
        tiles = generator.stair_collision_tiles
        self.assertEqual(len(tiles), 8)
        self.assertEqual({(t["family"], t["row"]) for t in tiles}, {(n, r) for n in names for r in (0, 1)})
        all_components = {id(mesh) for mesh, _ in generator.collision_components}
        for tile in tiles:
            self.assertTrue(all(id(mesh) in all_components for mesh, _ in tile["components"]))
            points = stair_edge_probe_points(tile["components"])
            center = generator.terrain_origins[tile["row"], tile["col"], :2]
            wall_margin = cfg.sub_terrains[tile["family"]].wall_thickness
            self.assertTrue((np.abs(points - center) <= np.array(cfg.size) / 2 + wall_margin + .011).all())
            self.assertGreater(len(points), 48)

    def test_parkour_override_runs_once_via_startup_guard(self):
        from isaaclab_newton.physics import NewtonManager
        terrain = object.__new__(ParkourTerrainImporter)
        terrain._native_collision = True
        terrain.terrain_generator = SimpleNamespace(stair_collision_tiles=[])
        terrain.device = "cpu"
        env = SimpleNamespace(scene=SimpleNamespace(terrain=terrain))
        with patch.object(NewtonManager, "_solver", SimpleNamespace(mj_model=object())), \
             patch.object(NativeTerrainImporter, "validate_solver_geometry", return_value={}) as shared, \
             patch("robolab.tasks.manager_based.parkour.stair_validation.validate_stair_edges", return_value={}) as stairs:
            validate_terrain_colliders(env)
            validate_terrain_colliders(env)
        shared.assert_called_once()
        stairs.assert_called_once()


if __name__ == "__main__":
    unittest.main()
