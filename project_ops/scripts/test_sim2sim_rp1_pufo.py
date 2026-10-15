"""Run with python -m unittest discover -s project_ops/scripts."""

import json
import copy
import sys
from pathlib import Path
_MUJOCO = Path(__file__).resolve().parents[2] / "robo_newton" / "scripts" / "mujoco"
if str(_MUJOCO) not in sys.path:
    sys.path.insert(0, str(_MUJOCO))

import tempfile
import unittest

import numpy as np
import torch

import sim2sim_rp1_pufo as sim


class TestPufoSim2Sim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.checkpoint = sim.latest_checkpoint()
        cls.config = sim.read_config(cls.checkpoint.parent / "params/env.yaml")
        manifest = json.loads(
            (sim.ROOT / "robo_newton/data/motions/rp1_lafan/manifest.json").read_text()
        )
        cls.names = manifest["npz"]["joint_order"]
        cls.body_names = manifest["npz"]["body_order"]
        with np.load(
            sim.ROOT / "robo_newton/data/motions/rp1_lafan/npz/dance1_subject2.npz"
        ) as data:
            cls.motion = {key: data[key][:2].copy() for key in data.files}

    def make_sim(self, stairs=False):
        return sim.Simulation(sim.make_model(stairs), self.names, self.config)

    def test_checkpoint_selection_is_numeric_and_ignores_partial_files(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("model_9.pt", "model_100.pt", ".model_200.pt.tmp"):
                Path(directory, name).touch()
            self.assertEqual(sim.latest_checkpoint(directory).name, "model_100.pt")

    def test_yaml_rejects_python_execution(self):
        with self.assertRaises(sim.yaml.constructor.ConstructorError):
            sim.yaml.load("!!python/object/apply:os.system ['echo forbidden']", Loader=sim.ConfigLoader)

    def test_reference_mapping_and_state_match_dataset(self):
        environment = self.make_sim()
        environment.set_reference(self.motion, 0)
        ids = [environment.model.body(name).id for name in self.body_names]
        np.testing.assert_allclose(environment.data.xpos[ids], self.motion["body_pos_w"][0], atol=1e-5)
        obs, values = environment.observe(torch.tensor([1., 0.]), torch.zeros(2))
        np.testing.assert_allclose(obs["state"][0, :24], self.motion["joint_pos"][0] - environment.default, atol=1e-6)
        np.testing.assert_allclose(obs["state"][0, 24:48], self.motion["joint_vel"][0], atol=1e-6)
        self.assertEqual(obs["depth_image"].shape, (1, 8, 18, 36))
        self.assertEqual(obs["depth_image"].dtype, torch.uint8)
        self.assertEqual(obs["history_actor"].count_nonzero(), 0)
        environment.history.add(values)
        self.assertGreater(environment.history.flat().count_nonzero(), 0)

    def test_action_history_uses_environment_scaled_actions(self):
        environment = self.make_sim()
        environment.set_reference(self.motion, 0)
        action = np.full(24, .02)
        environment.advance(action, 8.)
        obs, _ = environment.observe(torch.tensor([1., 0.]), torch.zeros(2))
        torch.testing.assert_close(obs["last_action"], torch.full((1, 24), .16))
        self.assertAlmostEqual(environment.data.time, .02)
        self.assertTrue(np.isfinite(environment.data.qpos).all())

    def test_angular_velocity_uses_link_axes_not_inertial_axes(self):
        environment = self.make_sim()
        environment.set_reference(self.motion, 0)
        for quat in ([0., 0., 0., 1.], [.2, -.3, .1, .9]):
            rotation = sim.Rotation.from_quat(quat)
            environment.data.qpos[3:7] = rotation.as_quat()[[3, 0, 1, 2]]
            for angular in np.eye(3):
                environment.data.qvel[3:6] = angular
                sim.mujoco.mj_forward(environment.model, environment.data)
                obs, values = environment.observe(torch.tensor([1., 0.]), torch.zeros(2))
                np.testing.assert_allclose(values['base_ang_vel'][0], angular, atol=1e-6)
                np.testing.assert_allclose(obs['state'][0, -3:], angular, atol=1e-6)
                environment.history.reset([0])
                environment.history.add(values)
                np.testing.assert_allclose(environment.history.history['base_ang_vel'][0, -1], angular, atol=1e-6)

    def test_depth_history_is_oldest_first_and_delayed_one_frame(self):
        environment = self.make_sim()
        environment.set_reference(self.motion, 0)
        environment.depth_history.extend([np.full((18, 36), index, dtype=np.uint8) for index in range(37)])
        depth = environment.depth()
        self.assertEqual(depth[0, :, 0, 0].tolist(), [1, 6, 11, 16, 21, 26, 31, 36])

    def test_aligned_depth_profile_uses_saved_crop_and_history(self):
        for side_crop, width in ((14, 36), (16, 32)):
            with self.subTest(width=width):
                config = copy.deepcopy(self.config)
                camera = config["scene"]["camera"]
                camera["noise_pipeline"]["sanitize_depth"]["invalid_depth"] = 2.5
                camera["noise_pipeline"]["crop_and_resize"]["crop_region"] = [18, 0, side_crop, side_crop]
                camera["data_histories"]["distance_to_image_plane_noised"] = 37
                environment = sim.Simulation(sim.make_model(False), self.names, config)
                environment.set_reference(self.motion, 0)
                depth = environment.depth()
                self.assertEqual(depth.shape, (1, 8, 18, width))
                self.assertEqual(len(environment.depth_history), 37)
                self.assertTrue(environment.aligned_depth)

    def test_full_h200_terrain_replaces_ground_plane(self):
        environment = self.make_sim(True)
        self.assertGreaterEqual(environment.model.ngeom, sim.TERRAIN_META["ngeom"])
        self.assertEqual(environment.model.nhfield, sim.TERRAIN_META["nhfield"])
        # No infinite ground plane remains once the training map is attached.
        names = [environment.model.geom(i).name for i in range(environment.model.ngeom)]
        self.assertTrue(any(name.startswith("terrain_") for name in names))
        self.assertFalse(any(name in {"floor", "ground"} for name in names))

    def test_stairs_reset_lands_on_training_map_origin(self):
        environment = self.make_sim(True)
        environment.set_reference(self.motion, 0)
        original_xy = environment.data.qpos[:2].copy()
        height = sim.place_on_stairs(environment)
        origin = np.asarray(sim.TERRAIN_META["stairs_reset"]["origin"], dtype=np.float64)
        np.testing.assert_allclose(environment.data.qpos[:2], origin[:2], atol=1e-9)
        self.assertNotEqual(tuple(original_xy), tuple(environment.data.qpos[:2]))
        self.assertAlmostEqual(height, origin[2], places=5)
        self.assertGreater(environment.data.qpos[2], height)

    def test_h200_stair_surface_at_reset_origin_has_center_pit(self):
        environment = self.make_sim(True)
        origin = np.asarray(sim.TERRAIN_META["stairs_reset"]["origin"], dtype=np.float64)
        # Park the robot far away so rays only validate terrain collision.
        environment.data.qpos[:3] = [80., 80., 2.]
        sim.mujoco.mj_forward(environment.model, environment.data)
        point = np.array([origin[0], origin[1], 5.])
        hit = np.empty(1, dtype=np.int32)
        ray = sim.mujoco.mj_ray(environment.model, environment.data, point,
            np.array([0., 0., -1.]), np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8),
            True, -1, hit)
        self.assertGreaterEqual(ray, 0)
        self.assertTrue(environment.model.geom(int(hit[0])).name.startswith("terrain_"))
        self.assertAlmostEqual(5. - ray, origin[2], places=5)


if __name__ == "__main__":
    unittest.main()
