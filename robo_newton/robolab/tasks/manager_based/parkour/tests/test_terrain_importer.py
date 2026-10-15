"""Do not replace terrain valleys/obstacle gaps with a global convex hull."""

import unittest
import numpy as np
import trimesh

from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh
from robolab.tasks.manager_based.parkour.terrain_importer import (
    collision_kind, merge_box_strips, ParkourTerrainImporter, ParkourTerrainGenerator,
)
from robolab.tasks.manager_based.parkour.terrain_generator_cfg import ROUGH_TERRAINS_CFG
from robolab.terrains.terrain_generator import FiledTerrainGenerator
from robolab.tasks.manager_based.parkour.rp1_parkour_env_cfg import RP1ParkourEnvCfg
from robolab.tasks.manager_based.parkour.rpo_parkour_env_cfg import RPOParkourEnvCfg


class TestParkourTerrain(unittest.TestCase):
    def test_both_robots_use_new_importer(self):
        for cfg in (RP1ParkourEnvCfg(), RPOParkourEnvCfg()):
            self.assertIs(cfg.scene.terrain.class_type, ParkourTerrainImporter)
            self.assertTrue(set(cfg.commands.base_velocity.velocity_ranges).issubset(
                cfg.scene.terrain.terrain_generator.sub_terrains))
            reward = cfg.rewards.locomotion.pelvis_orientation_l2
            self.assertTrue(set(reward.params["terrain_forward_tolerances"]).issubset(
                cfg.scene.terrain.terrain_generator.sub_terrains))

    def test_rotated_solid_remains_exact_convex_mesh(self):
        box = trimesh.creation.box((1., 2., .4))
        box.apply_transform(trimesh.transformations.rotation_matrix(.37, (0, 0, 1)))
        self.assertEqual(collision_kind(box), "convex")

    def test_zero_thickness_floor_is_not_a_convex_solid(self):
        self.assertEqual(collision_kind(trimesh.creation.box((8., 8., 0.))), "heightfield")

    def test_merge_only_contiguous_equal_height_strips(self):
        parts = []
        for y, height in ((0., 1.), (1., 1.), (3., 1.), (4., 2.)):
            box = trimesh.creation.box((2., 1., height))
            box.apply_translation((0., y, height / 2))
            parts.append((box, False))
        merged = merge_box_strips(parts)
        self.assertEqual(len(merged), 3)
        self.assertAlmostEqual(sum(m.volume for m, _ in merged), sum(m.volume for m, _ in parts))
        self.assertTrue(any(np.allclose(m.bounds, [[-1., -.5, 0.], [1., 1.5, 1.]]) for m, _ in merged))

    def test_rotated_boxes_and_heightfields_are_not_merged(self):
        box = trimesh.creation.box((1., 2., .4))
        box.apply_transform(trimesh.transformations.rotation_matrix(.37, (0, 0, 1)))
        parts = [(box, False), (box.copy(), False), (box.copy(), True)]
        self.assertEqual(len(merge_box_strips(parts)), 3)

    def test_inverted_stair_infill_union_is_preserved(self):
        cfg = ROUGH_TERRAINS_CFG.sub_terrains["trapezoid_stairs_inv"].copy()
        cfg.size = ROUGH_TERRAINS_CFG.size
        cfg.horizontal_scale = ROUGH_TERRAINS_CFG.horizontal_scale
        cfg.vertical_scale = ROUGH_TERRAINS_CFG.vertical_scale
        cfg.slope_threshold = ROUGH_TERRAINS_CFG.slope_threshold
        meshes, _ = cfg.function(.5, cfg)
        merged = merge_box_strips([(m, i == 0) for i, m in enumerate(meshes)])
        self.assertGreater(len(meshes), 1000)
        self.assertLess(len(merged), 100)
        self.assertIs(merged[0][0], meshes[0])
        self.assertAlmostEqual(sum(m.volume for m in meshes[1:]),
                               sum(m.volume for m, is_hf in merged if not is_hf), places=6)

    def test_adjacent_solids_stay_separate_before_welding(self):
        boxes = [trimesh.creation.box((1., 1., 1.)), trimesh.creation.box((1., 1., 1.))]
        boxes[1].apply_translation((1., 0., 0.))
        components = trimesh.util.concatenate(boxes).split(only_watertight=False, repair=False)
        self.assertEqual(len(components), 2)
        self.assertEqual([collision_kind(m) for m in components], ["convex", "convex"])

    def test_heightfield_retains_deep_gaps_and_vertical_faces(self):
        heights = np.zeros((21, 21), dtype=np.int16)
        heights[5:10, :] = -200
        vertices, faces = convert_height_field_to_mesh(heights, .05, .005, 1.)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        self.assertEqual(collision_kind(mesh), "heightfield")
        self.assertAlmostEqual(mesh.bounds[0, 2], -1.)

    def test_surface_with_hole_is_rejected(self):
        vertices, faces = convert_height_field_to_mesh(np.zeros((4, 4)), .1, .01)
        mesh = trimesh.Trimesh(vertices=vertices, faces=np.delete(faces, 8, axis=0), process=False)
        with self.assertRaisesRegex(ValueError, "Unsupported parkour surface"):
            collision_kind(mesh)

    def test_source_components_preserve_generated_map_and_origins(self):
        cfg = ROUGH_TERRAINS_CFG.copy()
        cfg.num_rows = 1
        cfg.border_width = 1
        cfg.sub_terrains = {key: cfg.sub_terrains[key] for key in ("square_gaps", "mesh_boxes")}
        for sub in cfg.sub_terrains.values():
            sub.flat_patch_sampling = None
        np.random.seed(42)
        baseline = FiledTerrainGenerator(cfg.copy(), device="cpu")
        native_cfg = cfg.copy()
        native_cfg.native_collision = True
        np.random.seed(42)
        native = FiledTerrainGenerator(native_cfg, device="cpu")
        np.random.seed(42)
        generated = ParkourTerrainGenerator(cfg.copy(), device="cpu")
        np.testing.assert_array_equal(generated.terrain_mesh.vertices, native.terrain_mesh.vertices)
        np.testing.assert_array_equal(generated.terrain_mesh.faces, native.terrain_mesh.faces)
        np.testing.assert_array_equal(generated.terrain_origins, baseline.terrain_origins)
        combined = trimesh.util.concatenate([mesh for mesh, _ in generated.collision_components])
        np.testing.assert_allclose(np.unique(combined.triangles.reshape(-1, 3), axis=0),
                                   np.unique(generated.terrain_mesh.triangles.reshape(-1, 3), axis=0), atol=1e-6)
        kinds = [collision_kind(mesh, is_hf) for mesh, is_hf in generated.collision_components]
        self.assertEqual(kinds.count("heightfield"), 6)  # two tiles plus four border strips
        self.assertGreater(kinds.count("convex"), 0)

    def test_legacy_cache_cannot_silently_lose_components(self):
        cfg = ROUGH_TERRAINS_CFG.copy()
        cfg.use_cache = True
        with self.assertRaisesRegex(ValueError, "legacy OBJ caches"):
            ParkourTerrainGenerator(cfg, device="cpu")


if __name__ == "__main__":
    unittest.main()
