"""Saved environment configuration defines sim2sim observation history."""


from pathlib import Path
import sys

_MUJOCO_SCRIPTS = Path(__file__).resolve().parents[2] / "robo_newton" / "scripts" / "mujoco"
if str(_MUJOCO_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_MUJOCO_SCRIPTS))

import unittest

import numpy as np

from sim2sim_rp1_amp import abrupt_speed_command, policy_histories_from_config


class TestPolicyHistory(unittest.TestCase):
    def test_step_protocol_holds_four_for_exactly_five_seconds(self):
        commands = [abrupt_speed_command(i / 50, 4.0, 0.5, 5.5) for i in range(525)]
        np.testing.assert_array_equal(commands[:25], np.zeros(25))
        np.testing.assert_array_equal(commands[25:275], np.full(250, 4.0))
        np.testing.assert_array_equal(commands[275:], np.zeros(250))

    def test_group_history_overrides_each_term(self):
        for length in (0, 1, 3, 5):
            config = {"observations": {"policy": {
                "concatenate_terms": True,
                "history_length": length,
                "velocity_commands": {"func": "commands", "history_length": 7},
                "joint_pos": {"func": "positions", "history_length": 2},
            }}}
            histories = policy_histories_from_config(config)
            self.assertEqual(list(histories), ["velocity_commands", "joint_pos"])
            for history in histories.values():
                self.assertEqual(history.length, max(1, length))

    def test_per_term_histories_and_order_are_preserved(self):
        config = {"observations": {"policy": {
            "concatenate_terms": True,
            "history_length": None,
            "joint_pos": {"func": "positions", "history_length": 5},
            "base_ang_vel": {"func": "angular_velocity", "history_length": 0},
            "velocity_commands": {"func": "commands", "history_length": 2},
        }}}
        histories = policy_histories_from_config(config)
        self.assertEqual(list(histories), ["joint_pos", "base_ang_vel", "velocity_commands"])
        self.assertEqual([history.length for history in histories.values()], [5, 1, 2])
        for history in histories.values():
            history.append(np.ones(history.width))
            history.append(np.full(history.width, 2.0))
            expected = np.concatenate([
                np.ones(history.width * (history.length - 1)), np.full(history.width, 2.0)
            ])
            np.testing.assert_array_equal(history.flat(), expected)

    def test_unknown_term_is_not_silently_dropped(self):
        config = {"observations": {"policy": {
            "concatenate_terms": True, "new_sensor": {"func": "sensor"},
        }}}
        with self.assertRaisesRegex(KeyError, "new_sensor"):
            policy_histories_from_config(config)


if __name__ == "__main__":
    unittest.main()
