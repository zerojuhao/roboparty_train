"""Ground-relative observation contracts, including real CPU mesh raycasts."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh
from robolab.tasks.manager_based.ufo.mdp.pufo_observations import local_self_obs, terrain_priv
from robolab.tasks.manager_based.ufo.pufo_env_cfg import RP1PUFOEnvCfg


def make_env():
    position = torch.tensor([[[0., 0., .8], [0., 0., .1]],
                             [[0., 0., 1.8], [0., 0., 1.1]]])
    robot = SimpleNamespace(data=SimpleNamespace(
        body_pos_w=position,
        body_quat_w=torch.tensor([0., 0., 0., 1.]).expand(2, 2, 4),
        body_link_lin_vel_w=torch.zeros(2, 2, 3),
        body_link_ang_vel_w=torch.zeros(2, 2, 3),
    ))
    hits = torch.zeros(2, 273, 3)
    hits[1, :, 2] = 1.
    return SimpleNamespace(scene={
        "robot": robot, "terrain_priv": SimpleNamespace(data=SimpleNamespace(ray_hits_w=hits)),
    })


class GroundObservationTests(unittest.TestCase):
    def test_same_pose_is_invariant_to_terrain_elevation(self):
        env = make_env()
        obs = local_self_obs(env)
        torch.testing.assert_close(obs[:, 0], torch.tensor([.8, .8]))
        torch.testing.assert_close(obs[0], obs[1])
        torch.testing.assert_close(env.scene["robot"].data.body_pos_w[:, 0, 2], torch.tensor([.8, 1.8]))

    def test_raised_terrain_is_positive_and_misses_are_not_obstacles(self):
        env = make_env()
        hits = env.scene["terrain_priv"].data.ray_hits_w
        hits[:, 0, 2] += .2
        hits[:, 1, 2] -= .3
        hits[:, 2, 2] = torch.inf
        hits[:, 3, 2] = torch.nan
        expected = torch.tensor([.2, -.3, -.5, -.5]).expand(2, -1)
        torch.testing.assert_close(terrain_priv(env)[:, :4], expected)

    def test_missing_root_ground_fails_explicitly(self):
        env = make_env()
        env.scene["terrain_priv"].data.ray_hits_w[:, 6 * 21 + 4, 2] = torch.inf
        for observation in (local_self_obs, terrain_priv):
            with self.assertRaisesRegex(RuntimeError, "root-ground ray"):
                observation(env)

    def test_missing_root_diagnostic_captures_raw_warp_sensor_buffers(self):
        import warp as wp

        env = make_env()
        sensor = env.scene["terrain_priv"]
        sensor.data.ray_hits_w[1, 6 * 21 + 4, 2] = torch.inf
        sensor._ray_starts_w = wp.zeros((2, 273), dtype=wp.vec3, device="cpu")
        sensor._ray_directions_w = wp.zeros((2, 273), dtype=wp.vec3, device="cpu")
        with self.assertRaisesRegex(RuntimeError, "finite_roots=True"):
            terrain_priv(env)
        self.assertEqual(env._pufo_ground_failure["env_ids"].tolist(), [1])
        self.assertEqual(env._pufo_ground_failure["_ray_starts_w"].shape, (1, 3))
        env.scene["robot"].data.body_pos_w[1, 0, 0] = torch.nan
        with self.assertRaisesRegex(RuntimeError, "finite_roots=False"):
            terrain_priv(env)

    def test_constructor_shape_probe_allows_staging_grid_misses_only(self):
        class ConstructingEnv(SimpleNamespace):
            pass

        env = ConstructingEnv(scene=make_env().scene)
        hits = env.scene["terrain_priv"].data.ray_hits_w
        hits[0, :, 2] = torch.inf
        hits[1, :, 2] = torch.nan
        with patch(
            "robolab.tasks.manager_based.ufo.mdp.pufo_observations.ManagerBasedEnv", ConstructingEnv
        ):
            for observation in (local_self_obs, terrain_priv):
                self.assertTrue(torch.isfinite(observation(env)).all())
            self.assertTrue(torch.isinf(hits[0, :, 2]).all())
            self.assertTrue(torch.isnan(hits[1, :, 2]).all())
            env.observation_manager = object()
            for observation in (local_self_obs, terrain_priv):
                with self.assertRaisesRegex(RuntimeError, "root-ground ray"):
                    observation(env)

    def test_scan_origin_and_range_cover_jump_and_descent(self):
        cfg = RP1PUFOEnvCfg().scene.terrain_priv
        self.assertEqual(cfg.prim_path, "{ENV_REGEX_NS}/Robot/Geometry/base_link")
        starts, directions = cfg.pattern_cfg.func(cfg.pattern_cfg, "cpu")
        starts += torch.tensor(cfg.offset.pos)
        at_root = torch.linalg.vector_norm(starts[:, :2], dim=-1).argmin()
        torch.testing.assert_close(starts[at_root, :2], torch.zeros(2), atol=1e-6, rtol=0)
        vertices = np.array([[-5, -5, -.3], [5, -5, -.3], [5, 5, -.3], [-5, 5, -.3]], dtype=np.float32)
        mesh = convert_to_warp_mesh(vertices, np.array([[0, 1, 2], [0, 2, 3]]), device="cpu")
        for root_z in (.8, 3.):
            rays = starts.clone()
            rays[:, 2] += root_z
            hits = raycast_mesh(rays[None], directions[None], mesh, max_dist=cfg.max_distance)[0]
            self.assertTrue(torch.isfinite(hits).all())
            torch.testing.assert_close(hits[..., 2], torch.full((1, 273), -.3), atol=1e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
