"""CPU checks for UFO action configuration without simulator startup."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from rsl_rl.runners.fbcpr_runner import FbCprRunner


def load_action_config():
    path = Path(__file__).parents[1] / "ufo_env_cfg.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    action_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "UFOActionsCfg")
    namespace = {
        "configclass": lambda cls: cls,
        "mdp": SimpleNamespace(JointPositionActionCfg=SimpleNamespace),
    }
    exec(compile(ast.Module(body=[action_class], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["UFOActionsCfg"]().joint_pos


class ActionConfigTests(unittest.TestCase):
    def test_uniform_scale_needs_no_actuator_parameters(self):
        cfg = load_action_config()
        self.assertEqual(cfg.asset_name, "robot")
        self.assertEqual(cfg.joint_names, [".*"])
        self.assertTrue(cfg.use_default_offset)
        self.assertIsInstance(cfg.scale, float)
        self.assertEqual(cfg.scale, 0.4)

    def test_rescaled_actions_broadcast_to_all_24_default_relative_targets(self):
        cfg = load_action_config()
        runner = FbCprRunner.__new__(FbCprRunner)
        runner.action_rescale = 8.0
        action = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]).unsqueeze(1).expand(-1, 24)
        default = torch.linspace(-1.2, 1.2, 24).unsqueeze(0)
        target = default + cfg.scale * runner._env_action(action)
        expected_offsets = torch.tensor([-3.2, -3.2, -1.6, 0.0, 1.6, 3.2, 3.2]).unsqueeze(1)
        torch.testing.assert_close(target, default + expected_offsets)
        self.assertEqual(target.shape, (7, 24))


if __name__ == "__main__":
    unittest.main()
