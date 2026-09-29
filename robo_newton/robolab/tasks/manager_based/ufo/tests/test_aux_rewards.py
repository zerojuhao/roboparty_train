"""CPU regressions for the auxiliary reward path, independent of simulator startup."""

from __future__ import annotations

import ast
import copy
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch


def matrix_from_quat(q):
    x, y, z, w = q.unbind(-1)
    s = 2.0 / q.square().sum(-1)
    return torch.stack(
        (1 - s * (y*y + z*z), s * (x*y - z*w), s * (x*z + y*w),
         s * (x*y + z*w), 1 - s * (x*x + z*z), s * (y*z - x*w),
         s * (x*z - y*w), s * (y*z + x*w), 1 - s * (x*x + y*y)), -1
    ).reshape(*q.shape[:-1], 3, 3)


def load_rewards(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and node.module.startswith("isaaclab")
    )]
    namespace = {"ManagerBasedEnv": object, "matrix_from_quat": matrix_from_quat}
    exec(compile(tree, str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


REWARDS = load_rewards(Path(__file__).parents[1] / "mdp/rewards.py")


class Actuators(dict):
    pass


def make_env(num_bodies=5):
    torch.manual_seed(412)
    n, j, b = 4, 3, num_bodies
    names = ["left_ankle_roll_link", "right_ankle_roll_link"] + [f"body_{i}" for i in range(b - 2)]
    inertia = torch.eye(3).reshape(1, 1, 9).repeat(n, b, 1)
    data = SimpleNamespace(
        body_names=names, body_mass=torch.rand(n, b) + 1.0, body_inertia=inertia,
        body_com_pos_w=torch.randn(n, b, 3), body_com_lin_vel_w=torch.randn(n, b, 3),
        body_com_ang_vel_w=torch.randn(n, b, 3),
        body_com_quat_w=torch.nn.functional.normalize(torch.randn(n, b, 4), dim=-1),
        body_link_pos_w=torch.rand(n, b, 3), body_link_lin_vel_w=torch.randn(n, b, 3),
        joint_vel=torch.randn(n, j), joint_pos=torch.randn(n, j), joint_acc=torch.randn(n, j),
        soft_joint_pos_limits=torch.tensor([-1.0, 1.0]).repeat(n, j, 1),
    )
    actuator = SimpleNamespace(
        applied_effort=torch.randn(n, j), joint_indices=slice(None),
        rated_velocity=torch.full((n, j), 0.5), rated_effort=torch.full((n, j), 0.7),
    )
    actuators = Actuators(all_joints=actuator)
    actuators.applied_effort = actuator.applied_effort
    sensor = SimpleNamespace(data=SimpleNamespace(
        net_normal_forces_w=torch.randn(n, b, 3),
        net_normal_forces_w_history=torch.randn(n, 3, b, 3),
    ))
    return SimpleNamespace(
        scene={"robot": SimpleNamespace(data=data, actuators=actuators), "contact_forces": sensor},
        num_envs=n, device="cpu", step_dt=0.02, common_step_counter=0,
        action_manager=SimpleNamespace(action=torch.randn(n, j), prev_action=torch.randn(n, j)),
        reset_terminated=torch.zeros(n, dtype=torch.bool), reset_time_outs=torch.zeros(n, dtype=torch.bool),
    )


class AuxRewardTests(unittest.TestCase):
    def test_contact_window_threshold_height_and_count(self):
        history = torch.zeros(2, 5, 3, 3)
        history[0, 0, 0, 2] = 1.0
        history[0, 1, 1, 2] = 2.0
        history[0, 3, 2, 2] = 2.0
        history[1, 2, :, 2] = 2.0
        heights = torch.tensor([[0.6, 0.3, 0.6], [0.2, 0.3, 0.4]])
        result = REWARDS._height_gated_undesired_contacts(history, heights, torch.arange(3))
        expected = torch.stack((torch.tensor(0.5), torch.sigmoid((heights[1] - 0.3) / 0.05).sum()))
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        self.assertGreater(result[1].item(), 1.0)

    def test_empty_undesired_selection(self):
        env = make_env(num_bodies=2)
        REWARDS.initialize_aux_rewards(env)
        result = REWARDS.compute_aux_rewards(env)["height_gated_undesired_contacts"]
        torch.testing.assert_close(result, torch.zeros(env.num_envs), rtol=0, atol=0)

    def test_initialize_validates_fixed_configuration(self):
        for dt in (0.0, -1.0, math.nan, math.inf):
            env = make_env()
            env.step_dt = dt
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                REWARDS.initialize_aux_rewards(env)
        env = make_env()
        env.scene["contact_forces"].data.net_normal_forces_w_history = torch.zeros(4, 2, 5, 3)
        with self.assertRaises(ValueError):
            REWARDS.initialize_aux_rewards(env)

    def test_live_mass_and_inertia_and_step_cache(self):
        env = make_env()
        REWARDS.initialize_aux_rewards(env)
        first = REWARDS.compute_aux_rewards(env)
        self.assertIs(REWARDS.compute_aux_rewards(env), first)
        momentum = REWARDS._com_angular_momentum(env)
        data = env.scene["robot"].data
        data.body_mass *= 2.0
        data.body_inertia *= 2.0
        torch.testing.assert_close(REWARDS._com_angular_momentum(env), 2.0 * momentum)
        self.assertEqual(env._ufo_feet_ids.tolist(), [0, 1])
        self.assertEqual(env._ufo_undesired_body_ids.tolist(), [2, 3, 4])

    def test_slippage_requires_loaded_current_foot_contact(self):
        env = make_env()
        REWARDS.initialize_aux_rewards(env)
        robot = env.scene["robot"].data
        robot.body_link_lin_vel_w.zero_()
        robot.body_link_lin_vel_w[:, :2, 0] = 1.0
        env.scene["contact_forces"].data.net_normal_forces_w.zero_()
        env.scene["contact_forces"].data.net_normal_forces_w_history.fill_(100.0)
        no_contact = REWARDS.compute_aux_rewards(env)["penalty_slippage"]
        torch.testing.assert_close(no_contact, torch.zeros_like(no_contact))

        env.common_step_counter += 1
        env.scene["contact_forces"].data.net_normal_forces_w[:, :2, 2] = 2.0
        loaded_contact = REWARDS.compute_aux_rewards(env)["penalty_slippage"]
        self.assertTrue(torch.all(loaded_contact > 0))

    def test_rate_keeps_finite_history_and_reset_protection(self):
        env = make_env()
        REWARDS.initialize_aux_rewards(env)
        values = torch.zeros(4, 3)
        REWARDS._angular_momentum_rate(env, values)
        values[0, 0] = math.nan
        values[1, 0] = math.inf
        values[2:, 0] = 1.0
        env.reset_terminated[2] = True
        with mock.patch.object(torch, "isfinite", wraps=torch.isfinite) as check:
            result = REWARDS._angular_momentum_rate(env, values)
            self.assertEqual(check.call_count, 1)
        torch.testing.assert_close(result, torch.tensor([0.0, 0.0, 0.0, 2500.0]))
        self.assertTrue(torch.isfinite(env._ufo_l_com_prev).all())
        env.reset_terminated.zero_()
        result = REWARDS._angular_momentum_rate(env, torch.zeros(4, 3))
        torch.testing.assert_close(result, torch.tensor([0.0, 0.0, 0.0, 2500.0]))

    def test_tensor_and_proxy_views(self):
        value = torch.randn(2, 3)
        self.assertIs(REWARDS._t(value), value)
        self.assertIs(REWARDS._t(SimpleNamespace(torch=value)), value)

    @unittest.skipUnless(os.environ.get("UFO_REWARD_BASELINE"), "Optional pre-cleanup snapshot comparison")
    def test_all_rewards_match_pre_cleanup_over_resets_and_nonfinite_states(self):
        baseline = load_rewards(os.environ["UFO_REWARD_BASELINE"])
        original = make_env()
        updated = copy.deepcopy(original)
        REWARDS.initialize_aux_rewards(updated)
        torch.manual_seed(812)
        for step in range(12):
            action = torch.randn(4, 3)
            angular_velocity = torch.randn(4, 5, 3)
            if step == 5:
                angular_velocity[0, 0, 0] = math.nan
                angular_velocity[1, 0, 0] = math.inf
            for env in (original, updated):
                env.common_step_counter = step
                env.action_manager.prev_action = env.action_manager.action.clone()
                env.action_manager.action = action.clone()
                env.scene["robot"].data.body_com_ang_vel_w = angular_velocity.clone()
                env.reset_terminated[:] = torch.tensor([step == 3, step == 4, False, False])
                env.reset_time_outs[:] = torch.tensor([False, False, step == 7, False])
                if step == 6:
                    env.scene["robot"].data.body_mass *= 1.2
                    env.scene["robot"].data.body_inertia *= 1.3
            expected = baseline.compute_aux_rewards(original)
            actual = REWARDS.compute_aux_rewards(updated)
            self.assertEqual(actual.keys(), expected.keys())
            for name in actual:
                with self.subTest(step=step, term=name):
                    torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
