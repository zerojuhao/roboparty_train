"""Check camera parity and delayed history without starting a simulator."""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from robolab.sensors.noisy_camera.noisy_camera import NoisyCameraMixin
from robolab.tasks.manager_based.parkour.rp1_parkour_env_cfg import RP1ParkourEnvCfg
from robolab.tasks.manager_based.parkour.mdp.observations.exteroception import delayed_visualizable_image
from robolab.tasks.manager_based.ufo.pufo_env_cfg import RP1PUFOEnvCfg, RP1PUFOEvalEnvCfg
from robolab.tasks.manager_based.ufo.agents.rp1_pufo_agent_cfg import RP1PUFORunnerCfg
from robolab.tasks.manager_based.ufo.ufo_env_cfg import UFOObservationsCfg
from robolab.tasks.manager_based.ufo.mdp.pufo_observations import depth_image


class TestPUFOCamera(unittest.TestCase):
    def test_camera_uses_full_rotation_in_training_and_evaluation(self):
        for cfg_type in (RP1PUFOEnvCfg, RP1PUFOEvalEnvCfg):
            with self.subTest(cfg=cfg_type.__name__):
                cfg = cfg_type()
                self.assertEqual(cfg.scene.camera.ray_alignment, "base")
                self.assertEqual(cfg.scene.to_dict()["camera"]["ray_alignment"], "base")
                self.assertEqual(cfg.scene.terrain_priv.ray_alignment, "yaw")

    def test_camera_preserves_shared_parkour_settings(self):
        reference = RP1ParkourEnvCfg()
        pufo = RP1PUFOEnvCfg()
        for policy in (UFOObservationsCfg().policy, pufo.observations.policy):
            self.assertTrue(policy.enable_corruption)
            for name in ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel"):
                self.assertEqual(
                    getattr(policy, name).noise.to_dict(),
                    getattr(reference.observations.policy, name).noise.to_dict(),
                )
        self.assertEqual(pufo.scene.camera.noise_pipeline["crop_and_resize"].crop_region, (18, 0, 14, 14))
        self.assertEqual(pufo.scene.camera.data_histories["distance_to_image_plane_noised"], 37)
        self.assertEqual(pufo.observations.visual.depth_image.params["delayed_frame_ranges"], (0, 1))

        parkour_camera = reference.events.randomize_camera_offset
        self.assertEqual(parkour_camera.mode, "startup")
        self.assertEqual(parkour_camera.params["distribution"], "uniform")
        ranges = parkour_camera.params["offset_pose_ranges"]
        self.assertEqual(ranges["x"], (-0.02, 0.02))
        self.assertEqual(ranges["y"], (-0.02, 0.02))
        self.assertEqual(ranges["z"], (-0.02, 0.02))
        self.assertEqual(ranges["roll"], (-math.radians(3), math.radians(3)))
        self.assertEqual(ranges["pitch"], (-math.radians(10), math.radians(10)))
        self.assertEqual(ranges["yaw"], (-math.radians(3), math.radians(3)))
        self.assertNotIn("randomize_camera_pitch_angle", reference.events.to_dict())

    def test_invalid_depth_is_max_filled_and_capped_before_blur(self):
        camera = RP1PUFOEnvCfg().scene.camera
        self.assertEqual(camera.depth_clipping_behavior, "max")
        self.assertEqual(next(iter(camera.noise_pipeline)), "sanitize_depth")
        cfg = camera.noise_pipeline["sanitize_depth"]
        image = torch.tensor([float("nan"), float("inf"), -float("inf"), -1.0, 0.0, 0.05, 1.0, 2.5, 10.0, 1e6])
        image = image.reshape(1, 1, -1, 1)
        actual = cfg.func(image, cfg, torch.tensor([0]))
        expected = torch.tensor([2.5, 2.5, 2.5, 2.5, 0.0, 0.05, 1.0, 2.5, 2.5, 2.5]).reshape_as(image)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.isnan(image[0, 0, 0, 0]))

    def test_full_pipeline_handles_invalid_depth(self):
        camera = RP1PUFOEnvCfg().scene.camera
        processor = NoisyCameraMixin()
        processor.noise_pipeline = list(camera.noise_pipeline.values())
        image = torch.ones(2, 36, 64, 1)
        image[:, 22, 22:26, 0] = torch.tensor([float("nan"), float("inf"), -float("inf"), 1e6])
        for gate in (0.0, 1.0):
            with self.subTest(gate=gate), patch("robolab.utils.noise.noise_model.random.random", return_value=gate):
                actual = processor.apply_noise_pipeline(image, torch.arange(2))
                self.assertEqual(actual.shape, (2, 18, 36, 1))
                self.assertTrue(torch.isfinite(actual).all())
                self.assertTrue(((actual >= 0) & (actual <= 1)).all())

    def test_parkour_augmentations_match_and_use_half_probability(self):
        parkour = RP1ParkourEnvCfg().scene.camera.noise_pipeline
        for cfg_type in (RP1PUFOEnvCfg, RP1PUFOEvalEnvCfg):
            pipeline = cfg_type().scene.camera.noise_pipeline
            for name in ("scale_randomization", "random_conv", "pixel_failures"):
                with self.subTest(cfg=cfg_type.__name__, noise=name):
                    cfg = pipeline[name]
                    self.assertEqual(cfg.apply_probability, 0.5)
                    self.assertEqual(cfg.to_dict(), parkour[name].to_dict())
                    image = torch.ones(2, 18, 36, 1)
                    with patch("robolab.utils.noise.noise_model.random.random", return_value=1.0):
                        actual = cfg.func(image, cfg, torch.arange(2))
                    torch.testing.assert_close(actual, image)

    def test_crop_matches_network_and_gaussian_blur_is_active(self):
        camera = RP1PUFOEnvCfg().scene.camera
        crop = camera.noise_pipeline["crop_and_resize"]
        blur = camera.noise_pipeline["gaussian_blur"]
        image = torch.zeros(2, 36, 64, 1)
        image[:, 26, 32] = 1.0
        cropped = crop.func(image, crop, torch.arange(2))
        self.assertEqual(cropped.shape, (2, 18, 36, 1))
        coordinates = torch.arange(36 * 64).reshape(1, 36, 64, 1).float()
        torch.testing.assert_close(
            crop.func(coordinates, crop, torch.tensor([0])), coordinates[:, 18:, 14:50]
        )
        self.assertEqual(RP1PUFORunnerCfg().depth_encoder.input_shape, (8, *cropped.shape[1:3]))
        blurred = blur.func(cropped, blur, torch.arange(2))
        self.assertEqual(blurred.shape, cropped.shape)
        self.assertTrue(0 < blurred.max().item() < 1)
        self.assertGreater(torch.count_nonzero(blurred).item(), torch.count_nonzero(cropped).item())

    def test_gaussian_noise_has_metric_scale_and_respects_valid_range(self):
        pipeline = RP1PUFOEnvCfg().scene.camera.noise_pipeline
        noise = pipeline["gaussian_noise"]
        self.assertEqual(noise.min_value, 0.1)
        self.assertEqual(noise.max_value, 2.5)
        self.assertEqual(noise.noise_std, 0.01)
        self.assertLess(list(pipeline).index("crop_and_resize"), list(pipeline).index("gaussian_noise"))
        self.assertLess(list(pipeline).index("gaussian_noise"), list(pipeline).index("gaussian_blur"))
        self.assertLess(list(pipeline).index("gaussian_blur"), list(pipeline).index("depth_normalization"))
        image = torch.ones(64, 18, 36, 1)
        image[:, :, 0] = 0.05
        image[:, :, -1] = 3.0
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            noised = noise.func(image, noise, torch.arange(64))
            next_frame = noise.func(image, noise, torch.arange(64))
        error = (noised - image)[:, :, 1:-1]
        self.assertAlmostEqual(error.mean().item(), 0.0, delta=0.0002)
        self.assertAlmostEqual(error.std().item(), 0.01, delta=0.0003)
        torch.testing.assert_close(noised[:, :, 0], image[:, :, 0])
        torch.testing.assert_close(noised[:, :, -1], image[:, :, -1])
        self.assertFalse(torch.equal(noised, next_frame))

    def make_terms(self, history_length=37):
        cfg = RP1PUFOEnvCfg().observations.visual.depth_image
        # Serialized pre-alignment jobs retain the continuous-delay contract.
        cfg.params.pop("integer_delay")
        cfg.params["delayed_frame_ranges"] = (0, 1)
        history = torch.arange(history_length, dtype=torch.float32).reshape(1, -1, 1, 1, 1)
        history = history.expand(2, -1, 18, 36, 1).clone() / 36
        sensor = SimpleNamespace(data=SimpleNamespace(output={cfg.params["data_type"]: history}))
        env = SimpleNamespace(num_envs=2, device="cpu", scene=SimpleNamespace(sensors={"camera": sensor}))
        return cfg, env, depth_image(cfg, env), delayed_visualizable_image(cfg, env)

    def test_delayed_history_matches_reference_without_renormalizing(self):
        cfg, env, term, reference = self.make_terms()
        for delays in ([0.0, 0.0], [0.2, 0.9], [1.0, 1.0]):
            term.delayed_frames[:] = torch.tensor(delays)
            # Reset refreshes the reference sampler's cached frame indices.
            with patch(
                "robolab.tasks.manager_based.parkour.mdp.observations.exteroception._randomize_prop_by_op",
                return_value=torch.tensor(delays).unsqueeze(-1),
            ):
                reference.reset()
            actual = term(env, **cfg.params)
            torch.testing.assert_close(actual, reference(env, **cfg.params))
            self.assertEqual(actual.shape, (2, 8, 18, 36))
        term.delayed_frames.zero_()
        actual = term(env, **cfg.params)
        torch.testing.assert_close(actual[0, :, 0, 0], torch.arange(1, 37, 5) / 36)
        self.assertEqual(actual[0, -1, 0, 0].item(), 1.0)

    def test_reset_only_changes_requested_environments(self):
        _, _, term, _ = self.make_terms()
        term.delayed_frames.fill_(-1)
        term.reset(torch.tensor([1]))
        self.assertEqual(term.delayed_frames[0].item(), -1)
        self.assertTrue(0 <= term.delayed_frames[1].item() <= 1)
        term.reset()
        self.assertTrue(((term.delayed_frames >= 0) & (term.delayed_frames <= 1)).all())

    def test_insufficient_history_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "history is too short"):
            self.make_terms(history_length=36)

    def test_integer_delay_samples_all_three_values(self):
        cfg, env, _, _ = self.make_terms(history_length=38)
        cfg.params.update(integer_delay=True, delayed_frame_ranges=(0, 2))
        term = depth_image(cfg, env)
        seen = set()
        for _ in range(100):
            term.reset()
            seen.update(term.delayed_frames.tolist())
            actual = term(env, **cfg.params)
            expected = (37 - term.delayed_frames) / 36
            torch.testing.assert_close(actual[:, -1, 0, 0], expected)
        self.assertEqual(seen, {0., 1., 2.})

    def test_training_and_evaluation_use_gaussian_noise_pipeline(self):
        for cfg_type in (RP1PUFOEnvCfg, RP1PUFOEvalEnvCfg):
            with self.subTest(cfg=cfg_type.__name__):
                self.assertEqual(list(cfg_type().scene.camera.noise_pipeline),
                                 ["sanitize_depth", "crop_and_resize", "gaussian_noise",
                                  "scale_randomization", "random_conv", "pixel_failures",
                                  "gaussian_blur", "depth_normalization"])


if __name__ == "__main__":
    unittest.main()
