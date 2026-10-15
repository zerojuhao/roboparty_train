"""Check AMP reflection against RP1 kinematics without launching Isaac Sim."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np
import torch
from tensordict import TensorDict
import yaml

ROOT = Path(__file__).resolve().parents[2] / "robo_newton"
SPEC = importlib.util.spec_from_file_location(
    "rp1_symmetry", ROOT / "robolab/tasks/manager_based/amp/mdp/symmetry/rp1.py"
)
symmetry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(symmetry)
NAMES = yaml.safe_load((ROOT / "scripts/tools/retarget/config/rp1_24dof.yaml").read_text())


def observation_manager(policy_history=3, critic_history=3):
    policy_terms = ["base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions"]
    groups = {"policy": policy_terms, "critic": ["base_lin_vel", *policy_terms]}
    dimensions = {}
    for group, history in (("policy", policy_history), ("critic", critic_history)):
        dimensions[group] = [
            ((24 if name in ("joint_pos", "joint_vel", "actions") else 3) * max(1, history),)
            for name in groups[group]
        ]
    return SimpleNamespace(active_terms=groups, group_obs_term_dim=dimensions)


class TestRP1AMPSymmetry(unittest.TestCase):
    def test_joint_identity_for_both_orders(self):
        for order in ("gmr_dof_names", "lab_dof_names"):
            names = tuple(NAMES[order])
            values = torch.arange(1, 25, dtype=torch.float64)
            mirrored = symmetry._switch_joints_left_right(values, names)
            left_hip = names.index("left_hip_pitch_joint")
            right_hip = names.index("right_hip_pitch_joint")
            left_ankle = names.index("left_ankle_pitch_joint")
            right_ankle = names.index("right_ankle_pitch_joint")
            self.assertEqual(mirrored[left_hip], -values[right_hip])
            self.assertEqual(mirrored[left_ankle], values[right_ankle])
            self.assertEqual(mirrored[right_hip], -values[left_hip])
            self.assertEqual(mirrored[right_ankle], values[left_ankle])
            for waist_name in ("waist_roll_joint", "waist_yaw_joint"):
                waist_index = names.index(waist_name)
                self.assertEqual(mirrored[waist_index], -values[waist_index])
            torch.testing.assert_close(values, torch.arange(1, 25, dtype=torch.float64))
            torch.testing.assert_close(symmetry._switch_joints_left_right(mirrored, names), values)

    def test_vector_reflection_rules_for_each_history_frame(self):
        names = tuple(NAMES["gmr_dof_names"])
        expected_signs = {
            "base_lin_vel": [1, -1, 1],
            "base_ang_vel": [-1, 1, -1],
            "projected_gravity": [1, -1, 1],
            "velocity_commands": [1, -1, -1],
        }
        for history in (1, 3, 5):
            manager = observation_manager(history, history)
            for group in ("policy", "critic"):
                widths = [shape[0] for shape in manager.group_obs_term_dim[group]]
                values = torch.arange(1, 2 * sum(widths) + 1, dtype=torch.float64).reshape(2, -1)
                original = values.clone()
                mirrored = symmetry._mirror_observation_group(values, manager, group, names)
                original_terms = dict(zip(manager.active_terms[group], values.split(widths, dim=-1)))
                mirrored_terms = dict(zip(manager.active_terms[group], mirrored.split(widths, dim=-1)))
                for term_name in manager.active_terms[group]:
                    if term_name not in expected_signs:
                        continue
                    term = original_terms[term_name]
                    expected = term.reshape(2, history, 3) * term.new_tensor(expected_signs[term_name])
                    torch.testing.assert_close(mirrored_terms[term_name], expected.reshape_as(term))
                torch.testing.assert_close(values, original)

    def test_history_and_actions_use_runtime_order(self):
        names = tuple(NAMES["gmr_dof_names"])
        env = SimpleNamespace(
            scene={"robot": SimpleNamespace(joint_names=names)},
            action_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace(_joint_names=names)),
            observation_manager=observation_manager(),
        )
        obs = TensorDict(
            {"policy": torch.randn(2, 243), "critic": torch.randn(2, 252)}, batch_size=[2]
        )
        actions = torch.randn(2, 24, dtype=torch.float64)
        wrapped_env = SimpleNamespace(unwrapped=env)
        augmented, mirrored = symmetry.compute_symmetric_states(wrapped_env, obs, actions)
        torch.testing.assert_close(augmented[:2], obs)
        torch.testing.assert_close(mirrored[:2], actions)
        for key, offset in (("policy", 27), ("critic", 36)):
            joint_history = obs[key][:, offset:].reshape(2, 9, 24)
            expected = symmetry._switch_joints_left_right(joint_history, names)
            torch.testing.assert_close(augmented[key][2:, offset:], expected.reshape(2, -1))
        restored, restored_actions = symmetry.compute_symmetric_states(wrapped_env, augmented[2:], mirrored[2:])
        torch.testing.assert_close(restored[2:], obs)
        torch.testing.assert_close(restored_actions[2:], actions)

    def test_independent_histories_and_reordered_terms(self):
        names = tuple(NAMES["gmr_dof_names"])
        for policy_history, critic_history in ((0, 1), (1, 5), (3, 1), (5, 3)):
            manager = observation_manager(policy_history, critic_history)
            for group in ("policy", "critic"):
                # Reordering must not change the physical meaning of any term.
                manager.active_terms[group].reverse()
                manager.group_obs_term_dim[group].reverse()
                widths = [shape[0] for shape in manager.group_obs_term_dim[group]]
                original = torch.randn(2, sum(widths))
                mirrored = symmetry._mirror_observation_group(original, manager, group, names)
                restored = symmetry._mirror_observation_group(mirrored, manager, group, names)
                torch.testing.assert_close(restored, original)
                original_terms = torch.split(original, widths, dim=-1)
                mirrored_terms = torch.split(mirrored, widths, dim=-1)
                for term_name, before, after in zip(manager.active_terms[group], original_terms, mirrored_terms):
                    if term_name == "velocity_commands":
                        expected = before.reshape(2, -1, 3) * torch.tensor([1, -1, -1])
                        torch.testing.assert_close(after, expected.reshape_as(before))
                    elif term_name == "joint_pos":
                        left_hip = names.index("left_hip_pitch_joint")
                        right_hip = names.index("right_hip_pitch_joint")
                        torch.testing.assert_close(
                            after.reshape(2, -1, 24)[..., left_hip],
                            -before.reshape(2, -1, 24)[..., right_hip],
                        )

    def test_mixed_per_term_history(self):
        manager = SimpleNamespace(
            active_terms={"policy": [
                "base_ang_vel", "joint_pos", "projected_gravity", "joint_vel", "actions", "velocity_commands",
            ]},
            group_obs_term_dim={"policy": [(6,), (120,), (9,), (24,), (48,), (3,)]},
        )
        original = torch.arange(210, dtype=torch.float32).reshape(1, -1)
        names = tuple(NAMES["gmr_dof_names"])
        mirrored = symmetry._mirror_observation_group(original, manager, "policy", names)
        torch.testing.assert_close(mirrored[:, :6], original[:, :6] * torch.tensor([-1, 1, -1] * 2))
        torch.testing.assert_close(mirrored[:, -3:], original[:, -3:] * torch.tensor([1, -1, -1]))
        restored = symmetry._mirror_observation_group(mirrored, manager, "policy", names)
        torch.testing.assert_close(restored, original)

    def test_missing_required_observation_term_fails(self):
        manager = observation_manager()
        manager.active_terms["policy"].pop(0)
        manager.group_obs_term_dim["policy"].pop(0)
        width = sum(shape[0] for shape in manager.group_obs_term_dim["policy"])
        with self.assertRaisesRegex(KeyError, "base_ang_vel"):
            symmetry._mirror_observation_group(
                torch.zeros(1, width), manager, "policy", tuple(NAMES["gmr_dof_names"])
            )

    def test_actions_only_mirror_loss_call(self):
        names = tuple(NAMES["gmr_dof_names"])
        env = SimpleNamespace(
            scene={"robot": SimpleNamespace(joint_names=names)},
            action_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace(_joint_names=names)),
        )
        actions = torch.randn(2, 24)
        obs_aug, actions_aug = symmetry.compute_symmetric_states(
            SimpleNamespace(unwrapped=env), obs=None, actions=actions
        )
        self.assertIsNone(obs_aug)
        torch.testing.assert_close(actions_aug[:2], actions)
        torch.testing.assert_close(actions_aug[2:], symmetry._switch_joints_left_right(actions, names))

    def test_mismatched_action_order_fails(self):
        action_term = SimpleNamespace(_joint_names=NAMES["lab_dof_names"])
        env = SimpleNamespace(
            scene={"robot": SimpleNamespace(joint_names=NAMES["gmr_dof_names"])},
            action_manager=SimpleNamespace(get_term=lambda _: action_term),
        )
        with self.assertRaisesRegex(ValueError, "articulation order"):
            symmetry.compute_symmetric_states(SimpleNamespace(unwrapped=env), actions=torch.zeros(1, 24))

    def test_reflection_matches_mujoco_key_body_positions(self):
        model_path = ROOT / "data/robots/roboparty/rp1.3/mjcf/rp1_flat.xml"
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        names = tuple(NAMES["gmr_dof_names"])
        addresses = [model.jnt_qposadr[model.joint(name).id] for name in names]
        body_names = NAMES["lab_key_body_names"]
        body_ids = [model.body(name).id for name in body_names]
        # The dataset orders ankle, knee, and wrist pairs left then right.
        partner_ids = [body_ids[index] for index in (1, 0, 3, 2, 5, 4)]
        # The imported CAD root is about 3 mm off the sagittal plane.
        leg_plane_y = 0.5 * (
            model.body("left_hip_pitch_link").pos[1] + model.body("right_hip_pitch_link").pos[1]
        )
        arm_plane_y = model.body("waist_roll_link").pos[1]
        reflection_offset_y = 2 * np.array([leg_plane_y] * 4 + [arm_plane_y] * 2)
        rng = np.random.default_rng(12)
        for _ in range(20):
            joints = rng.uniform(-0.5, 0.5, 24)
            data.qpos[:3] = [0, 0, 0.8]
            data.qpos[3:7] = [1, 0, 0, 0]
            data.qpos[addresses] = joints
            mujoco.mj_forward(model, data)
            expected = data.xpos[partner_ids].copy() * [1, -1, 1]
            expected[:, 1] += reflection_offset_y
            reflected_joints = symmetry._switch_joints_left_right(torch.tensor(joints), names)
            data.qpos[addresses] = reflected_joints.numpy()
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(data.xpos[body_ids], expected, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
