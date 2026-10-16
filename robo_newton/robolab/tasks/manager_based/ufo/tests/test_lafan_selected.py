"""Pose distribution and collision contracts for the H200-aligned PUFO profile."""

import unittest
import numpy as np
import torch
import trimesh

from robolab.tasks.manager_based.ufo.mdp.pufo_reset import sample_reset_states
from robolab.tasks.manager_based.ufo.pufo_terrain_importer import box_bounds
from robolab.tasks.manager_based.ufo.pufo_env_cfg import RP1PUFOEnvCfg, RP1PUFOEvalEnvCfg
from robolab.tasks.manager_based.ufo.agents.rp1_pufo_agent_cfg import RP1PUFORunnerCfg
from robolab.tasks.manager_based.ufo.lafan_selected import MOTION_NAMES


class TestMotionAlignment(unittest.TestCase):
    def test_motion_reset_is_fresh_and_preserves_height_and_velocities(self):
        n = 512
        ids = torch.arange(n)
        origins = torch.randn(n, 3)
        defaults = torch.randn(n, 24)
        original_defaults = defaults.clone()
        motion = dict(rg_pos_t=torch.randn(n, 1, 3),
                      rg_rot_t=torch.tensor([0., 0., 0., 1.]).expand(n, 1, 4),
                      body_vel_t=torch.randn(n, 1, 3), body_ang_vel_t=torch.randn(n, 1, 3),
                      dof_pos=torch.randn(n, 24), dof_vel=torch.randn(n, 24))
        for _ in range(100):
            motion["dof_pos"] += 0.001
            root, joints, velocity, lying, standing = sample_reset_states(
                motion, ids, origins, defaults, lie_down_probability=0.)
            self.assertFalse(lying.any())
            self.assertFalse(standing.any())
            torch.testing.assert_close(joints, motion["dof_pos"])
            torch.testing.assert_close(velocity, motion["dof_vel"])
            torch.testing.assert_close(root[:, 2], motion["rg_pos_t"][:, 0, 2] + .05)
            torch.testing.assert_close(root[:, :2], origins[:, :2])
            torch.testing.assert_close(root[:, 7:10].norm(dim=-1), motion["body_vel_t"][:, 0].norm(dim=-1))
        torch.testing.assert_close(defaults, original_defaults)
        root, joints, velocity, lying, standing = sample_reset_states(
            motion, ids, origins, defaults, lie_down_probability=1.)
        self.assertTrue(lying.all())
        self.assertFalse(standing.any())
        torch.testing.assert_close(joints, defaults)
        self.assertFalse(velocity.any())
        self.assertFalse(root[:, 7:].any())
        self.assertEqual(set(root[:, 2].mul(100).round().tolist()), {26., 31., 35.})
        torch.testing.assert_close(root[:, 3:7].norm(dim=-1), torch.ones(n))

    def test_profile_preserves_control_and_selects_h200_distribution(self):
        cfg, agent = RP1PUFOEnvCfg(), RP1PUFORunnerCfg()
        self.assertEqual(len(MOTION_NAMES), 40)
        self.assertEqual(cfg.commands.motion.motion_names, MOTION_NAMES)
        self.assertEqual(RP1PUFOEnvCfg().commands.motion.motion_names, MOTION_NAMES)
        self.assertEqual(cfg.events.reset_root_state.params["lie_down_probability"], 0.)
        self.assertEqual(RP1PUFOEvalEnvCfg().events.reset_root_state.params["lie_down_probability"], 0.)
        self.assertIsNone(getattr(cfg.events, "reset_joint_state", None))
        self.assertEqual(cfg.actions.joint_pos.scale, .4)
        self.assertTrue(agent.prioritization)
        self.assertEqual((agent.prioritization_min_val, agent.prioritization_max_val, agent.prioritization_scale), (.5, 2., 2.))

    def test_boxes_are_not_merged_into_a_convex_stair_ramp(self):
        a = trimesh.creation.box(extents=(3., 3., .2))
        b = trimesh.creation.box(extents=(2.4, 2.4, .2))
        b.apply_translation((0, 0, .2))
        combined = trimesh.util.concatenate([a, b])
        self.assertIsNone(box_bounds(combined))
        components = combined.split(only_watertight=False)
        self.assertEqual(len(components), 2)
        for mesh in components:
            np.testing.assert_allclose(box_bounds(mesh), mesh.bounds)
        self.assertIsNone(box_bounds(trimesh.creation.icosphere(subdivisions=1)))


if __name__ == "__main__":
    unittest.main()
