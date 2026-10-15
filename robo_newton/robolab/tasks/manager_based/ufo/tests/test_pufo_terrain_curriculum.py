"""Regression tests for the PUFO terrain schedule and H200 terrain contract."""

import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch
import warp as wp
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains.trimesh import (
    MeshInvertedPyramidStairsTerrainCfg,
    MeshPlaneTerrainCfg,
    MeshPyramidStairsTerrainCfg,
)

from robolab.tasks.manager_based.ufo.mdp.curriculums import terrain_levels_by_global_step
from robolab.tasks.manager_based.ufo.mdp.events import randomize_camera_installation
from robolab.tasks.manager_based.ufo.pufo_env_cfg import RP1PUFOEnvCfg, RP1PUFOEvalEnvCfg
from robolab.terrains import BorderedMeshRandomGridTerrainCfg
from robolab.terrains.terrain_generator_cfg import GuardedTerrainGeneratorCfg
from isaaclab.terrains import TerrainImporter
from isaaclab_newton.sensors.ray_caster.legacy_ray_caster import LegacyRayCaster
from robolab.tasks.manager_based.ufo.pufo_terrain_cfg import (
    PUFO_FLAT_SAFETY_WIDTH,
    PUFO_NONFLAT_GUARD_WIDTH,
    PUFO_TERRAIN_CFG,
    PUFO_TILE_FLAT_BORDER_WIDTH,
    PUFO_TILE_SIZE,
)


class TestPUFOTerrainCurriculum(unittest.TestCase):
    def test_generated_terrain_uses_component_colliders(self):
        from robolab.tasks.manager_based.ufo.pufo_terrain_importer import PUFOTerrainImporter
        self.assertIs(RP1PUFOEnvCfg().scene.terrain.class_type, PUFOTerrainImporter)
        generator_cfg = PUFO_TERRAIN_CFG.copy()
        for name in ("low_stairs_up", "low_stairs_down", "low_platforms"):
            self.assertFalse(generator_cfg.sub_terrains[name].convert_to_heightfield)
        importer = object.__new__(TerrainImporter)
        self.assertFalse(importer._is_heightfield_collider_requested(generator_cfg))

    def test_terrain_rays_exclude_robot_and_control_period_is_preserved(self):
        cfg = RP1PUFOEnvCfg()
        self.assertIs(cfg.scene.terrain_priv.class_type, LegacyRayCaster)
        self.assertEqual(cfg.scene.terrain_priv.mesh_prim_paths, ["/World/ground/terrain"])
        self.assertAlmostEqual(cfg.sim.dt * cfg.decimation, 0.02)
        self.assertAlmostEqual(cfg.sim.dt / cfg.sim.physics.num_substeps, 0.0025)

    def make_env(self, global_step):
        origins = torch.arange(10 * 7 * 3, dtype=torch.float32).reshape(10, 7, 3)
        terrain = SimpleNamespace(
            cfg=SimpleNamespace(terrain_type="generator"),
            terrain_origins=origins,
            terrain_levels=torch.zeros(16, dtype=torch.long),
            terrain_types=torch.arange(16) % 7,
            env_origins=torch.zeros(16, 3),
        )
        return SimpleNamespace(
            device="cpu",
            _ufo_global_step=global_step,
            scene=SimpleNamespace(terrain=terrain),
        )

    def test_global_step_schedule_updates_reset_origins(self):
        env = self.make_env(38_400_000)
        env_ids = torch.arange(16)
        state = terrain_levels_by_global_step(
            env,
            env_ids,
            start_step=0,
            interval=19_200_000,
            initial_max_level=2,
            level_increment=2,
            easy_fraction=0.0,
        )
        self.assertEqual(state["maximum_level"], 6)
        self.assertTrue((env.scene.terrain.terrain_levels <= 6).all())
        expected = env.scene.terrain.terrain_origins[
            env.scene.terrain.terrain_levels, env.scene.terrain.terrain_types
        ]
        torch.testing.assert_close(env.scene.terrain.env_origins, expected)

    def test_terrain_uses_ten_meter_tiles_with_h200_primitives(self):
        self.assertEqual(PUFO_TILE_SIZE, (10.0, 10.0))
        self.assertEqual(PUFO_TERRAIN_CFG.size, PUFO_TILE_SIZE)
        self.assertEqual(RP1PUFOEnvCfg().scene.env_spacing, 10.0)
        self.assertEqual(PUFO_TILE_FLAT_BORDER_WIDTH, 0.5)
        self.assertEqual(PUFO_NONFLAT_GUARD_WIDTH, 10.0)
        self.assertEqual(PUFO_FLAT_SAFETY_WIDTH, 2.0)
        self.assertIsInstance(PUFO_TERRAIN_CFG, GuardedTerrainGeneratorCfg)
        self.assertEqual(PUFO_TERRAIN_CFG.guard_width, 10.0)
        self.assertEqual(PUFO_TERRAIN_CFG.border_width - PUFO_TERRAIN_CFG.guard_width, 2.0)
        terrains = PUFO_TERRAIN_CFG.sub_terrains
        self.assertEqual(
            tuple(terrains),
            (
                "flat",
                "perlin_rough",
                "low_stairs_up",
                "low_stairs_down",
                "low_platforms",
                "hf_pyramid_slope_inv",
                "boxes",
            ),
        )
        self.assertEqual(
            [terrain.proportion for terrain in terrains.values()],
            [0.20, 0.10, 0.20, 0.20, 0.10, 0.10, 0.10],
        )
        self.assertIsInstance(terrains["flat"], MeshPlaneTerrainCfg)
        self.assertIsInstance(terrains["low_stairs_up"], MeshPyramidStairsTerrainCfg)
        self.assertIsInstance(terrains["low_stairs_down"], MeshInvertedPyramidStairsTerrainCfg)
        self.assertIsInstance(terrains["low_platforms"], BorderedMeshRandomGridTerrainCfg)
        self.assertEqual(terrains["low_platforms"].border_width, PUFO_TILE_FLAT_BORDER_WIDTH)
        self.assertTrue(terrains["low_platforms"].merge_similar_heights)
        self.assertIsNone(terrains["hf_pyramid_slope_inv"].perlin_cfg)
        self.assertIsNone(terrains["boxes"].perlin_cfg)
        for name, terrain in terrains.items():
            if name != "flat":
                self.assertEqual(terrain.border_width, PUFO_TILE_FLAT_BORDER_WIDTH)
            if name in {"perlin_rough", "hf_pyramid_slope_inv", "boxes"}:
                self.assertTrue(terrain.exact_flat_border)
        for guard in PUFO_TERRAIN_CFG.guard_terrains.values():
            self.assertEqual(guard.interior_height_offset, 0.05)
            self.assertTrue(guard.exact_flat_border)

    def test_platform_grid_has_exact_half_meter_flat_border(self):
        cfg = PUFO_TERRAIN_CFG.sub_terrains["low_platforms"].copy()
        cfg.size = PUFO_TILE_SIZE
        meshes, _ = cfg.function(1.0, cfg)
        cells = meshes[4:-1]
        self.assertAlmostEqual(min(mesh.bounds[0, 0] for mesh in cells), PUFO_TILE_FLAT_BORDER_WIDTH)
        self.assertAlmostEqual(min(mesh.bounds[0, 1] for mesh in cells), PUFO_TILE_FLAT_BORDER_WIDTH)
        self.assertAlmostEqual(
            max(mesh.bounds[1, 0] for mesh in cells), PUFO_TILE_SIZE[0] - PUFO_TILE_FLAT_BORDER_WIDTH
        )
        self.assertAlmostEqual(
            max(mesh.bounds[1, 1] for mesh in cells), PUFO_TILE_SIZE[1] - PUFO_TILE_FLAT_BORDER_WIDTH
        )

    def test_height_fields_have_exact_half_meter_flat_border(self):
        for name in ("perlin_rough", "hf_pyramid_slope_inv", "boxes"):
            cfg = PUFO_TERRAIN_CFG.sub_terrains[name].copy()
            cfg.size = PUFO_TILE_SIZE
            cfg.horizontal_scale = 0.2
            mesh = cfg.function(0.8, cfg)[0][0]
            x, y, z = mesh.vertices.T
            self.assertEqual(len(np.unique(x)), 61)
            self.assertEqual(len(np.unique(y)), 61)
            self.assertTrue(np.isclose(x, 0.5).any())
            self.assertTrue(np.isclose(x, 9.5).any())
            self.assertTrue(np.isclose(y, 0.5).any())
            self.assertTrue(np.isclose(y, 9.5).any())
            border = (x <= 0.5) | (x >= 9.5) | (y <= 0.5) | (y >= 9.5)
            np.testing.assert_allclose(z[border], 0.0, atol=1.0e-8)

    def test_guarded_terrain_layout_is_centered(self):
        cfg = PUFO_TERRAIN_CFG.copy()
        generator = cfg.class_type(cfg, device="cpu")

        np.testing.assert_allclose(generator.terrain_mesh.bounds[:, 0], (-62.05, 62.05))
        np.testing.assert_allclose(generator.terrain_mesh.bounds[:, 1], (-47.05, 47.05))
        self.assertEqual(generator.terrain_origins.shape, (10, 7, 3))
        self.assertEqual(len(generator.subterrain_specific_cfgs), 70)
        np.testing.assert_allclose(np.diff(generator.terrain_origins[:, :, 0], axis=0), 10.0)
        np.testing.assert_allclose(np.diff(generator.terrain_origins[:, :, 1], axis=1), 10.0)

    def test_manager_terms_are_configured(self):
        cfg = RP1PUFOEnvCfg()
        curriculum = cfg.curriculum.terrain_levels.params
        self.assertEqual(curriculum["interval"], 19_200_000)
        self.assertEqual(curriculum["initial_max_level"], 2)
        self.assertEqual(curriculum["easy_fraction"], 0.25)
        camera = cfg.events.randomize_camera_installation.params
        self.assertEqual(camera["position_error"], 0.02)
        self.assertEqual(
            camera["angle_error_rpy"],
            (math.radians(3.0), math.radians(10.0), math.radians(3.0)),
        )
        self.assertEqual(cfg.events.randomize_camera_installation.mode, "startup")
        for name in (
            "randomize_rigid_body_material",
            "randomize_rigid_body_com",
            "scale_link_mass",
        ):
            self.assertEqual(getattr(cfg.events, name).mode, "startup")
        self.assertEqual(cfg.events.scale_actuator_gains.mode, "reset")
        self.assertNotIn("refresh_camera", cfg.events.to_dict())
        self.assertEqual(cfg.events.push_robot.mode, "interval")
        eval_camera = RP1PUFOEvalEnvCfg().events.randomize_camera_installation.params
        self.assertEqual(eval_camera["position_error"], camera["position_error"])
        self.assertEqual(eval_camera["angle_error_rpy"], camera["angle_error_rpy"])

    def test_camera_installation_randomization_updates_selected_offsets(self):
        wp.init()
        position_buffer = torch.zeros(4, 3)
        orientation_buffer = torch.zeros(4, 4)
        orientation_buffer[:, 0] = 1.0
        sensor = SimpleNamespace(
            cfg=SimpleNamespace(
                offset=SimpleNamespace(pos=(0.2, 0.0, 0.1), rot=(1.0, 0.0, 0.0, 0.0), convention="world")
            ),
            _offset_pos_wp=wp.from_torch(position_buffer, dtype=wp.vec3),
            _offset_quat_wp=wp.from_torch(orientation_buffer, dtype=wp.quat),
        )
        env = SimpleNamespace(scene={"camera": sensor}, num_envs=4, device="cpu")
        env_ids = torch.tensor([1, 3])

        randomize_camera_installation(
            env,
            env_ids,
            SceneEntityCfg("camera"),
            position_error=0.03,
            angle_error_rpy=(math.radians(3.0), math.radians(10.0), math.radians(3.0)),
        )

        nominal_position = torch.tensor([0.2, 0.0, 0.1])
        position_error = (position_buffer[env_ids] - nominal_position).abs()
        self.assertTrue((position_error <= 0.03).all())
        self.assertTrue((position_buffer[env_ids] != 0.0).any())
        torch.testing.assert_close(
            torch.linalg.vector_norm(orientation_buffer[env_ids], dim=-1), torch.ones(2)
        )
        w, x, y, z = orientation_buffer[env_ids].unbind(dim=-1)
        roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x.square() + y.square()))
        pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
        self.assertTrue((roll.abs() <= math.radians(3.0)).all())
        self.assertTrue((pitch.abs() <= math.radians(10.0)).all())
        self.assertTrue((yaw.abs() <= math.radians(3.0)).all())
        torch.testing.assert_close(position_buffer[[0, 2]], torch.zeros(2, 3))

        # Startup events receive env_ids=None and must initialize every environment.
        position_buffer.zero_()
        randomize_camera_installation(
            env,
            None,
            SceneEntityCfg("camera"),
            position_error=0.03,
            angle_error_rpy=(math.radians(3.0), math.radians(10.0), math.radians(3.0)),
        )
        position_error = (position_buffer - nominal_position).abs()
        self.assertTrue((position_error <= 0.03).all())
        self.assertTrue((position_buffer != 0.0).any(dim=-1).all())
        torch.testing.assert_close(
            torch.linalg.vector_norm(orientation_buffer, dim=-1), torch.ones(4)
        )


if __name__ == "__main__":
    unittest.main()
