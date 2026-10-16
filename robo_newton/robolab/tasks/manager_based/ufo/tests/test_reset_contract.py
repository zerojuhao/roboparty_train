"""Reset pose, joint randomization and XYZW contracts without launching the simulator."""

import math
import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from isaaclab.utils.math import quat_apply

from robolab.tasks.manager_based.ufo.mdp.events import reset_joint_state_from_motion, reset_root_state_from_motion
from robolab.tasks.manager_based.ufo.pufo_env_cfg import PUFOEventCfg
from robolab.tasks.manager_based.ufo.ufo_env_cfg import UFOEventCfg
from robolab.tasks.manager_based.ufo.mdp.commands import UFOMotionCommand, UFOMotionCommandCfg
from robolab.tasks.manager_based.ufo.mdp.pufo_reset import reset_pufo_from_motion


class TestResetContract(unittest.TestCase):
    def make_command(self):
        library = Mock()
        library.sample_motions.side_effect = lambda n: torch.full((n,), 3, dtype=torch.long)
        library.sample_time.side_effect = lambda ids, **kwargs: torch.full(ids.shape, 1.25)
        cfg = UFOMotionCommandCfg(asset_name="robot", manifest_path="unused")
        robot = SimpleNamespace(joint_names=["joint_a", "joint_b"], body_names=["base_link", "body_a"])
        env = SimpleNamespace(num_envs=4, device="cpu", step_dt=0.02, scene={"robot": robot})
        with patch("robolab.tasks.manager_based.ufo.mdp.commands.MotionLibNPZ", return_value=library) as constructor:
            command = UFOMotionCommand(cfg, env)
            self.assertEqual(list(constructor.call_args.args[0].joint_names), robot.joint_names)
            self.assertEqual(list(constructor.call_args.args[0].body_names), robot.body_names)
            return command, library

    def test_pose_event_prepares_new_frame_and_manager_reset_does_not_resample(self):
        command, library = self.make_command()
        command.motion_times.fill_(15.0)
        ids = torch.tensor([1, 3])
        command.prepare_reset(ids)
        torch.testing.assert_close(command.motion_times[ids], torch.full((2,), 1.25))
        torch.testing.assert_close(command.motion_ids[ids], torch.full((2,), 3, dtype=torch.long))
        command.reset(ids)
        self.assertEqual(library.sample_motions.call_count, 1)
        self.assertFalse(command._reset_prepared.any())
        self.assertEqual(command.motion_times[0], 15.0)
        torch.testing.assert_close(command.reset_motion_times[ids], command.motion_times[ids])
        command.prepare_reset(ids)
        command.reset(ids)
        self.assertEqual(library.sample_motions.call_count, 2)

    def test_command_reset_without_pose_event_and_empty_reset(self):
        command, library = self.make_command()
        command.prepare_reset([])
        library.sample_motions.assert_not_called()
        command.reset()
        self.assertEqual(library.sample_motions.call_count, 1)
        torch.testing.assert_close(command.motion_times, torch.full((4,), 1.25))

    def test_sampling_event_precedes_both_pose_events(self):
        for cfg in (UFOEventCfg(), PUFOEventCfg()):
            order = list(cfg.to_dict())
            self.assertLess(order.index("resample_motion"), order.index("reset_root_state"))
            if isinstance(cfg, PUFOEventCfg):
                self.assertIs(cfg.reset_root_state.func, reset_pufo_from_motion)
                self.assertIsNone(getattr(cfg, "reset_joint_state", None))
            else:
                self.assertLess(order.index("resample_motion"), order.index("reset_joint_state"))

    def test_running_job_legacy_event_can_still_be_unpickled(self):
        legacy_event = pickle.loads(
            b"crobolab.tasks.manager_based.ufo.mdp.events\nrandomize_default_joint_pos\n."
        )
        self.assertTrue(callable(legacy_event))

    def make_joint_reset_env(self):
        motion = {
            "dof_pos": torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10,
            "dof_vel": torch.arange(12, dtype=torch.float32).reshape(4, 3) / 20,
        }
        robot = SimpleNamespace(
            data=SimpleNamespace(
                default_joint_pos=torch.full((4, 3), -2.0),
                default_joint_vel=torch.zeros(4, 3),
            ),
            write_joint_position_to_sim_index=Mock(),
            write_joint_velocity_to_sim_index=Mock(),
        )

        class Scene(dict):
            terrain = None

        env = SimpleNamespace(
            scene=Scene(robot=robot),
            command_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace(get_motion_state=lambda: motion)),
        )
        return env, robot, motion

    def test_joint_noise_and_reference_zero_randomization_are_disabled(self):
        for cfg in (UFOEventCfg(), PUFOEventCfg()):
            with self.subTest(cfg=type(cfg).__name__):
                self.assertNotIn("randomize_default_joint_pos", cfg.to_dict())
                if isinstance(cfg, PUFOEventCfg):
                    from robolab.tasks.manager_based.ufo.mdp.pufo_reset import reset_pufo_from_motion
                    self.assertIsNone(getattr(cfg, "reset_joint_state", None))
                    self.assertIs(cfg.reset_root_state.func, reset_pufo_from_motion)
                    continue
                self.assertIs(cfg.reset_joint_state.func, reset_joint_state_from_motion)
                self.assertEqual(cfg.reset_joint_state.mode, "reset")
                self.assertEqual(cfg.reset_joint_state.params["joint_pos_range"], (0.0, 0.0))

    def test_pufo_pose_reset_refreshes_camera_without_a_separate_event(self):
        env_ids = torch.tensor([0, 1])
        root = torch.zeros(2, 13)
        joints = torch.zeros(2, 3)
        velocities = torch.zeros(2, 3)
        lying = torch.zeros(2, dtype=torch.bool)
        standing = torch.zeros(2, dtype=torch.bool)
        robot = SimpleNamespace(
            data=SimpleNamespace(default_joint_pos=torch.zeros(2, 3)),
            write_joint_position_to_sim_index=Mock(),
            write_joint_velocity_to_sim_index=Mock(),
            write_root_pose_to_sim_index=Mock(),
            write_root_velocity_to_sim_index=Mock(),
        )

        class Scene(dict):
            terrain = SimpleNamespace(cfg=SimpleNamespace(terrain_type="plane"))
            env_origins = torch.zeros(2, 3)
            write_data_to_sim = Mock()
            update = Mock()

        env = SimpleNamespace(
            scene=Scene(robot=robot),
            command_manager=SimpleNamespace(
                get_term=lambda _: SimpleNamespace(get_motion_state=lambda: {})
            ),
            sim=SimpleNamespace(forward=Mock()),
            num_envs=2,
            device="cpu",
        )
        with (
            patch(
                "robolab.tasks.manager_based.ufo.mdp.pufo_reset.sample_reset_states",
                return_value=(root, joints, velocities, lying, standing),
            ),
            patch("robolab.tasks.manager_based.ufo.mdp.pufo_reset.correct_terrain_penetration"),
            patch("robolab.tasks.manager_based.ufo.mdp.pufo_reset.refresh_reset_camera") as refresh,
        ):
            reset_pufo_from_motion(env, env_ids, "motion")

        refresh.assert_called_once_with(env, env_ids)
        torch.testing.assert_close(env._pufo_reset_standing, standing)
        self.assertNotIn("refresh_camera", PUFOEventCfg().to_dict())

    def test_configured_resets_follow_changing_frames_exactly_without_noise_or_drift(self):
        for cfg in (UFOEventCfg(),):
            with self.subTest(cfg=type(cfg).__name__):
                env, robot, motion = self.make_joint_reset_env()
                env_ids = torch.tensor([1, 3])
                default_pos = robot.data.default_joint_pos.clone()
                default_vel = robot.data.default_joint_vel.clone()
                for _ in range(100):
                    motion["dof_pos"] += 0.01
                    motion["dof_vel"] += 0.02
                    reference_pos = motion["dof_pos"].clone()
                    reference_vel = motion["dof_vel"].clone()
                    cfg.reset_joint_state.func(env, env_ids, **cfg.reset_joint_state.params)
                    position = robot.write_joint_position_to_sim_index.call_args.kwargs
                    velocity = robot.write_joint_velocity_to_sim_index.call_args.kwargs
                    self.assertTrue(torch.equal(position["position"], reference_pos[env_ids]))
                    self.assertTrue(torch.equal(velocity["velocity"], reference_vel[env_ids]))
                    self.assertTrue(torch.equal(motion["dof_pos"], reference_pos))
                    self.assertTrue(torch.equal(motion["dof_vel"], reference_vel))
                    self.assertTrue(torch.equal(robot.data.default_joint_pos, default_pos))
                    self.assertTrue(torch.equal(robot.data.default_joint_vel, default_vel))

    def test_joint_reset_without_noise_preserves_motion_frame(self):
        env, robot, motion = self.make_joint_reset_env()
        env_ids = torch.tensor([1, 3])
        reset_joint_state_from_motion(env, env_ids, "motion")
        position = robot.write_joint_position_to_sim_index.call_args.kwargs
        velocity = robot.write_joint_velocity_to_sim_index.call_args.kwargs
        torch.testing.assert_close(position["position"], motion["dof_pos"][env_ids])
        torch.testing.assert_close(velocity["velocity"], motion["dof_vel"][env_ids])
        torch.testing.assert_close(position["env_ids"], env_ids)
        torch.testing.assert_close(velocity["env_ids"], env_ids)

    def test_repeated_joint_resets_sample_around_motion_without_zero_drift(self):
        env, robot, motion = self.make_joint_reset_env()
        env_ids = torch.tensor([1, 3])
        default_pos = robot.data.default_joint_pos.clone()
        default_vel = robot.data.default_joint_vel.clone()
        errors = []
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(52)
            for _ in range(100):
                motion["dof_pos"] += 0.01
                reference_pos = motion["dof_pos"].clone()
                reference_vel = motion["dof_vel"].clone()
                reset_joint_state_from_motion(env, env_ids, "motion", joint_pos_range=(-0.15, 0.15))
                position = robot.write_joint_position_to_sim_index.call_args.kwargs
                velocity = robot.write_joint_velocity_to_sim_index.call_args.kwargs
                error = position["position"] - reference_pos[env_ids]
                self.assertTrue((error.abs() <= 0.150001).all())
                errors.append(error)
                torch.testing.assert_close(position["env_ids"], env_ids)
                torch.testing.assert_close(velocity["velocity"], reference_vel[env_ids])
                torch.testing.assert_close(robot.data.default_joint_pos, default_pos)
                torch.testing.assert_close(robot.data.default_joint_vel, default_vel)
                torch.testing.assert_close(motion["dof_pos"], reference_pos)
                torch.testing.assert_close(motion["dof_vel"], reference_vel)
        errors = torch.stack(errors)
        self.assertTrue((errors < 0).any())
        self.assertTrue((errors > 0).any())
        self.assertFalse(torch.equal(errors[0], errors[1]))

    def test_joint_reset_with_empty_selection_does_not_write(self):
        env, robot, _ = self.make_joint_reset_env()
        reset_joint_state_from_motion(env, torch.tensor([], dtype=torch.long), "motion", (-0.15, 0.15))
        robot.write_joint_position_to_sim_index.assert_not_called()
        robot.write_joint_velocity_to_sim_index.assert_not_called()

    def test_heightfield_offset_survives_runtime_geometry_updates(self):
        import numpy as np
        from pxr import Sdf, Usd, UsdGeom
        from isaaclab_newton.physics.newton_manager import NewtonManager, PhysicsManager

        stage = Usd.Stage.CreateInMemory()
        mesh = UsdGeom.Mesh.Define(stage, "/ground")
        mesh.GetPointsAttr().Set([(-1, -1, -2), (1, -1, -1), (1, 1, -1), (-1, 1, -2)])
        mesh.GetFaceVertexCountsAttr().Set([3, 3])
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 0, 2, 3])
        mesh.GetPrim().CreateAttribute("newton:heightfield:resolution", Sdf.ValueTypeNames.Float).Set(0.5)
        builder = Mock()
        with patch.object(PhysicsManager, "_device", "cpu"):
            ignored = NewtonManager._inject_terrain_heightfields(stage, builder, root_paths=("/",))
        self.assertEqual(ignored, ["/ground"])
        kwargs = builder.add_shape_heightfield.call_args.kwargs
        heightfield, transform = kwargs["heightfield"], kwargs["xform"]
        self.assertEqual(heightfield.min_z, 0.0)
        self.assertAlmostEqual(transform.p[2], -2.0)
        world_heights = heightfield.data * heightfield.max_z + transform.p[2]
        np.testing.assert_allclose(world_heights[2], [-2, -1.75, -1.5, -1.25, -1], atol=1e-6)

    def test_motion_xyzw_is_not_reordered_and_motion_z_is_not_used(self):
        quaternion = torch.tensor([[math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]])
        motion = {
            "rg_pos_t": torch.tensor([[[0.1, 0.2, -10.0]]]),
            "rg_rot_t": quaternion[:, None],
            "body_vel_t": torch.zeros(1, 1, 3),
            "body_ang_vel_t": torch.zeros(1, 1, 3),
        }
        robot = SimpleNamespace(
            cfg=SimpleNamespace(init_state=SimpleNamespace(pos=(0.0, 0.0, 0.85))),
            data=SimpleNamespace(default_root_state=torch.zeros(1, 13)),
            write_root_pose_to_sim_index=Mock(), write_root_velocity_to_sim_index=Mock(),
        )

        class Scene(dict):
            terrain = None
            env_origins = torch.tensor([[1.0, 2.0, 3.0]])

        env = SimpleNamespace(
            device="cpu", scene=Scene(robot=robot),
            command_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace(get_motion_state=lambda: motion)),
        )
        reset_root_state_from_motion(env, torch.tensor([0]), "motion")
        pose = robot.write_root_pose_to_sim_index.call_args.kwargs["root_pose"]
        torch.testing.assert_close(pose[:, :3], torch.tensor([[1.1, 2.2, 3.85]]))
        torch.testing.assert_close(pose[:, 3:], quaternion)
        torch.testing.assert_close(
            quat_apply(pose[:, 3:], torch.tensor([[0.0, 0.0, 1.0]])),
            torch.tensor([[0.0, -1.0, 0.0]]), atol=1e-6, rtol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
