"""CPU regressions for the auxiliary reward path, independent of simulator startup."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

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
    class SceneEntityCfg:
        def __init__(self, name, **kwargs):
            self.name = name
            self.joint_ids = kwargs.get("joint_ids", slice(None))
            self.body_ids = kwargs.get("body_ids", slice(None))

    namespace = {
        "ManagerBasedEnv": object,
        "JointPositionAction": object,
        "SceneEntityCfg": SceneEntityCfg,
        "matrix_from_quat": matrix_from_quat,
    }
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
    def test_pufo_regularizer_weights_for_motion_softness(self):
        task_root = Path(__file__).parents[2]
        def weights(path):
            tree = ast.parse(path.read_text())
            return {
                node.targets[0].id: ast.literal_eval(next(kw.value for kw in node.value.keywords if kw.arg == "weight"))
                for node in ast.walk(tree)
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "RewTerm"
            }
        pufo = weights(task_root / "ufo/pufo_env_cfg.py")
        parkour = weights(task_root / "parkour/parkour_env_cfg.py")
        self.assertEqual(pufo["penalty_action_rate"], -0.2)
        self.assertEqual(pufo["sound_suppression_acc_per_foot"], -5e-3)
        self.assertEqual(pufo["sound_suppression_acc_per_foot"], parkour["sound_suppression"])
        self.assertEqual(pufo["joint_acc_l2"], -1e-6)
        self.assertEqual(pufo["joint_acc_l2"], 4 * parkour["dof_acc_l2"])
        self.assertEqual(pufo["joint_vel_l2"], -1e-3)
        self.assertEqual(pufo["joint_vel_l2"], 10 * parkour["dof_vel_l2"])

    def test_joint_regularizers_use_selected_joints(self):
        env = make_env()
        cfg = SimpleNamespace(name="robot", joint_ids=[0, 2])
        for name, field in (("joint_vel_l2", "joint_vel"), ("joint_acc_l2", "joint_acc")):
            values = getattr(env.scene["robot"].data, field)
            expected = values[:, [0, 2]].square().sum(-1)
            torch.testing.assert_close(getattr(REWARDS, name)(env, cfg), expected)
            setattr(env.scene["robot"].data, field, SimpleNamespace(torch=values))
            torch.testing.assert_close(getattr(REWARDS, name)(env, cfg), expected)

    def test_sound_suppression_matches_parkour_contact_gate(self):
        env = make_env()
        data = env.scene["robot"].data
        data.body_vel_w = torch.full((4, 5, 6), 1000.0)
        data.body_acc_w = torch.full((4, 5, 6), 1000.0)
        data.body_vel_w[:, :2, 2] = torch.tensor([2.0, 3.0])
        data.body_acc_w[:, :2, 2] = torch.tensor([4.0, 5.0])
        forces = torch.zeros(4, 5, 3)
        forces[:, :2, 2] = torch.tensor([[0., 0.], [1., 0.], [0., -1.], [1., 1.]])
        forces[:, :, 0] = 1000.0
        env.scene["contact_forces"].data.net_forces_w = SimpleNamespace(torch=forces)
        cfg = SimpleNamespace(name="contact_forces", body_ids=[0, 1])
        expected = torch.tensor([0., 20., 34., 54.])
        torch.testing.assert_close(REWARDS.sound_suppression_acc_per_foot(env, cfg), expected)
        data.body_vel_w = SimpleNamespace(torch=data.body_vel_w)
        data.body_acc_w = SimpleNamespace(torch=data.body_acc_w)
        torch.testing.assert_close(REWARDS.sound_suppression_acc_per_foot(env, cfg), expected)

    def test_slippage_requires_loaded_current_foot_contact(self):
        env = make_env()
        asset_cfg = SimpleNamespace(name="robot", body_ids=[0, 1])
        sensor_cfg = SimpleNamespace(name="contact_forces", body_ids=[0, 1])
        robot = env.scene["robot"].data
        robot.body_link_lin_vel_w.zero_()
        robot.body_link_lin_vel_w[:, :2, 0] = 1.0
        env.scene["contact_forces"].data.net_normal_forces_w.zero_()
        env.scene["contact_forces"].data.net_normal_forces_w_history.fill_(100.0)
        no_contact = REWARDS.penalty_slippage(env, asset_cfg, sensor_cfg)
        torch.testing.assert_close(no_contact, torch.zeros_like(no_contact))

        env.common_step_counter += 1
        env.scene["contact_forces"].data.net_normal_forces_w[:, :2, 2] = 2.0
        loaded_contact = REWARDS.penalty_slippage(env, asset_cfg, sensor_cfg)
        self.assertTrue(torch.all(loaded_contact > 0))

    def test_tensor_and_proxy_views(self):
        value = torch.randn(2, 3)
        self.assertIs(REWARDS.as_torch_tensor(value), value)
        self.assertIs(REWARDS.as_torch_tensor(SimpleNamespace(torch=value)), value)



if __name__ == "__main__":
    unittest.main()
