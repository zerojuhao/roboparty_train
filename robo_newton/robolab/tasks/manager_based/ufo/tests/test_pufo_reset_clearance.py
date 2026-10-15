"""Single-pass PUFO reset correction without a running simulator."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from robolab.tasks.manager_based.ufo.mdp.pufo_reset import correct_terrain_penetration
from robolab.tasks.manager_based.ufo.pufo_env_cfg import RP1PUFOEnvCfg


class TestPufoResetClearance(unittest.TestCase):
    def run_correction(self, *, no_contacts=False, clearance=0.05, penetration_scale=1.25):
        contact = SimpleNamespace(
            geom=torch.tensor([[0, 1], [1, 0], [1, 2], [0, 3], [-1, 1],
                               [99, 1], [0, 1], [0, 1], [0, 1]]),
            worldid=torch.tensor([1, 1, 1, 1, 1, 1, 0, 3, 2]),
            dist=torch.zeros(9),
        )
        data = SimpleNamespace(nacon=torch.tensor([0]), contact=contact, overflow=torch.zeros(4, dtype=torch.int32))
        solver = SimpleNamespace(
            mj_model=SimpleNamespace(geom_bodyid=np.array([0, 1, 2, 0])),
            mjw_model=object(), mjw_data=data, model=object(), _update_mjc_data=Mock(),
        )
        manager = SimpleNamespace(_solver=solver, backend=SimpleNamespace(state_0=object()))
        robot = SimpleNamespace(write_root_pose_to_sim_index=Mock())

        class Scene(dict):
            terrain = SimpleNamespace()
            write_data_to_sim = Mock()

        env = SimpleNamespace(
            device="cpu", num_envs=4, scene=Scene(robot=robot),
            sim=SimpleNamespace(forward=Mock()), _pufo_reset_lift=torch.full((4,), 7.0),
        )
        root = torch.arange(26, dtype=torch.float32).reshape(2, 13)
        before = root.clone()
        ids = torch.tensor([1, 2])

        def fresh_contacts(*_):
            if not no_contacts:
                data.nacon[0] = 9
                contact.dist[:] = torch.tensor([-.04, -.08, -1., -1., -1., -1., -.5, -.2, .1])

        with (
            patch("isaaclab_newton.physics.newton_manager.NewtonManager", manager),
            patch("mujoco_warp.fwd_position", side_effect=fresh_contacts) as detect,
            patch("robolab.tasks.manager_based.ufo.mdp.pufo_reset.wp.to_torch", side_effect=lambda x: x),
        ):
            correct_terrain_penetration(env, ids, root, clearance, penetration_scale)
        detect.assert_called_once_with(solver.mjw_model, data)
        solver._update_mjc_data.assert_called_once_with(data, solver.model, manager.backend.state_0)
        env.scene.write_data_to_sim.assert_called_once()
        env.sim.forward.assert_called_once()
        return env, robot, before, root

    def test_single_fresh_pass_lifts_only_reset_environment_terrain_contacts(self):
        env, robot, before, root = self.run_correction()
        expected = before.clone()
        expected[0, 2] += .08 * 1.25 + .05
        torch.testing.assert_close(root, expected)
        torch.testing.assert_close(env._pufo_reset_lift, torch.tensor([7., .15, 0., 7.]))
        robot.write_root_pose_to_sim_index.assert_called_once()
        written = robot.write_root_pose_to_sim_index.call_args.kwargs
        torch.testing.assert_close(written["env_ids"], torch.tensor([1, 2]))
        torch.testing.assert_close(written["root_pose"], expected[:, :7])

    def test_no_contacts_do_not_add_height_or_reuse_previous_lift(self):
        env, robot, before, root = self.run_correction(no_contacts=True)
        torch.testing.assert_close(root, before)
        torch.testing.assert_close(env._pufo_reset_lift, torch.tensor([7., 0., 0., 7.]))
        robot.write_root_pose_to_sim_index.assert_not_called()

    def test_clearance_and_scale_are_configurable(self):
        params = RP1PUFOEnvCfg().events.reset_root_state.params
        self.assertEqual(params["contact_clearance"], .05)
        self.assertEqual(params["penetration_scale"], 1.25)
        env, _, _, _ = self.run_correction(clearance=.1, penetration_scale=2.)
        self.assertAlmostEqual(env._pufo_reset_lift[1].item(), .26)

    def test_invalid_parameters_fail_before_touching_simulation(self):
        for clearance, scale in ((-.01, 1.25), (float("nan"), 1.25), (.05, .5), (.05, float("inf"))):
            with self.subTest(clearance=clearance, scale=scale), self.assertRaises(ValueError):
                correct_terrain_penetration(None, torch.tensor([0]), None, clearance, scale)


if __name__ == "__main__":
    unittest.main()
