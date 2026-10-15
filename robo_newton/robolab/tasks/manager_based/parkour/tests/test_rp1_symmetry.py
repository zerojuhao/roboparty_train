"""Regression coverage for the observation path used by PPO augmentation."""

from types import SimpleNamespace as NS
import unittest

import torch
from tensordict import TensorDict
from robolab.tasks.manager_based.parkour.mdp.symmetry.rp1 import compute_symmetric_states


class TestRP1Symmetry(unittest.TestCase):
    def setUp(self):
        pattern = NS(ordering="xy", resolution=0.1, size=(0.2, 0.4))
        scanner = NS(pattern_cfg=pattern)
        terms = NS(**{k: NS(history_length=2) for k in
                      ("height_scan", "foot_pos_b", "left_foot_height_map")})
        self.env = NS(cfg=NS(scene=NS(height_scanner=scanner, left_height_scanner=scanner),
                             observations=NS(critic=terms)))
        policy = {k: torch.randn(3, n) for k, n in {
            "base_ang_vel": 6, "projected_gravity": 6, "velocity_commands": 6,
            "joint_pos": 48, "joint_vel": 48, "actions": 48,
        }.items()}
        policy["depth_image"] = torch.randint(0, 256, (3, 2, 18, 36), dtype=torch.uint8)
        critic = {k: v.clone() for k, v in policy.items() if k != "depth_image"}
        critic.update({k: torch.randn(3, n) for k, n in {
            "base_lin_vel": 6, "foot_pos_b": 12, "height_scan": 30,
            "left_foot_height_map": 30, "right_foot_height_map": 30,
        }.items()})
        self.obs = TensorDict({"policy": policy, "critic": critic, "disc": torch.randn(3, 20)}, [3])
        self.actions = torch.randn(3, 24)

    def assert_td_equal(self, actual, expected):
        for key in expected.keys(include_nested=True, leaves_only=True):
            torch.testing.assert_close(actual[key], expected[key])

    def test_nested_augmentation_does_not_mutate_input(self):
        original = self.obs.clone()
        actions = self.actions.clone()
        augmented, augmented_actions = compute_symmetric_states(self.env, self.obs, self.actions)
        self.assertEqual(augmented.batch_size, torch.Size([6]))
        self.assert_td_equal(augmented[:3], original)
        self.assert_td_equal(self.obs, original)
        torch.testing.assert_close(self.actions, actions)
        torch.testing.assert_close(augmented_actions[:3], actions)
        torch.testing.assert_close(augmented["disc"][3:], original["disc"])

    def test_mirror_twice_restores_all_inputs(self):
        augmented, actions = compute_symmetric_states(self.env, self.obs, self.actions)
        twice, twice_actions = compute_symmetric_states(self.env, augmented[3:], actions[3:])
        self.assert_td_equal(twice[3:], self.obs)
        torch.testing.assert_close(twice_actions[3:], self.actions)

    def test_depth_and_foot_maps_mirror_correct_axis(self):
        augmented, _ = compute_symmetric_states(self.env, self.obs)
        torch.testing.assert_close(augmented["policy", "depth_image"][3:],
                                   self.obs["policy", "depth_image"].flip(-1))
        expected = self.obs["critic", "right_foot_height_map"].reshape(3, 2, 5, 3).flip(2).reshape(3, 30)
        torch.testing.assert_close(augmented["critic", "left_foot_height_map"][3:], expected)

    def test_optional_inputs(self):
        augmented, actions = compute_symmetric_states(self.env, actions=self.actions)
        self.assertIsNone(augmented)
        self.assertEqual(actions.shape, (6, 24))
        augmented, actions = compute_symmetric_states(self.env, obs=self.obs)
        self.assertIsNone(actions)
        self.assertEqual(augmented.batch_size, torch.Size([6]))
        self.assertEqual(compute_symmetric_states(self.env), (None, None))


if __name__ == "__main__":
    unittest.main()
