"""Shared contact, load-gate, and global-boundary regressions."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch
import warp as wp

from robolab.sensors.terrain_contact import (
    reduce_contact_excess, append_excess_history, reset_excess_history, TerrainContactSensorCfg,
)
from robolab.utils.contact_rewards import feet_stumble
from robolab.tasks.manager_based.ufo.pufo_env_cfg import RP1PUFOEnvCfg
from robolab.tasks.manager_based.ufo.mdp.rewards import pufo_feet_stumble, pufo_penalty_slippage
from robolab.tasks.manager_based.ufo.mdp.terminations import terrain_grid_boundary_margin, terrain_out_of_bounds


class ContactAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.env = SimpleNamespace(
            cfg=SimpleNamespace(decimation=4, sim=SimpleNamespace(gravity=(0., 0., -10.))),
            device="cpu", num_envs=2,
            scene={
                "robot": SimpleNamespace(data=SimpleNamespace(
                    body_mass=torch.tensor([[20., 20.], [40., 40.]]),
                    body_link_lin_vel_w=torch.tensor([[[1., 0., 0.], [0., 0., 0.]]] * 2),
                )),
                "contact_forces": SimpleNamespace(stumble_horizontal_ratio=1., data=SimpleNamespace(
                    net_forces_w=torch.zeros(2, 2, 3),
                    contact_excess_history=torch.zeros(2, 4, 3),
                )),
            },
        )
        self.sensor_cfg = SimpleNamespace(name="contact_forces", body_ids=[0, 1])
        self.asset_cfg = SimpleNamespace(name="robot", body_ids=[0, 1])

    def test_stumble_history_and_body_selection(self):
        history = self.env.scene["contact_forces"].data.contact_excess_history
        history[:, 3, 0] = 100.
        history[:, :, 2] = 10000.  # Non-foot body must not affect stumble.
        torch.testing.assert_close(pufo_feet_stumble(self.env, self.sensor_cfg), torch.full((2,), .8))
        from robolab.tasks.manager_based.parkour.mdp.rewards import feet_stumble as parkour_stumble
        torch.testing.assert_close(parkour_stumble(self.env, self.sensor_cfg), torch.full((2,), .8))
        history.zero_()
        self.assertEqual(pufo_feet_stumble(self.env, self.sensor_cfg).sum(), 0)

    def test_stumble_threshold_clamp_and_contract(self):
        history = self.env.scene["contact_forces"].data.contact_excess_history
        for force, expected in ((15., 0.), (20., 0.), (30., .1), (200., 1.)):
            history.fill_(force)
            torch.testing.assert_close(feet_stumble(self.env, self.sensor_cfg), torch.full((2,), expected))
        with self.assertRaisesRegex(ValueError, "must match"):
            feet_stumble(self.env, self.sensor_cfg, horizontal_ratio=2.)
        for kwargs in (dict(force_scale=0.), dict(max_penalty=-1.), dict(force_threshold=-1.)):
            with self.assertRaises(ValueError):
                feet_stumble(self.env, self.sensor_cfg, **kwargs)
        history.fill_(float("inf"))
        self.assertTrue(torch.isinf(feet_stumble(self.env, self.sensor_cfg)).all())
        self.env.cfg.decimation = 3
        with self.assertRaisesRegex(ValueError, "complete policy step"):
            feet_stumble(self.env, self.sensor_cfg)

    def test_terrain_tasks_share_sensor_but_keep_timer_and_weights(self):
        from robolab.sensors.terrain_contact import _base_contact_cfg
        from robolab.tasks.manager_based.parkour.rp1_parkour_env_cfg import RP1ParkourEnvCfg
        from robolab.tasks.manager_based.parkour.rpo_parkour_env_cfg import RPOParkourEnvCfg
        pufo = RP1PUFOEnvCfg()
        self.assertIsInstance(pufo.scene.contact_forces, TerrainContactSensorCfg)
        self.assertFalse(pufo.scene.contact_forces.track_air_time)
        self.assertFalse(hasattr(_base_contact_cfg(pufo.scene.contact_forces), "stumble_horizontal_ratio"))
        self.assertEqual(pufo.rewards.feet_stumble.weight, -2.)
        self.assertEqual(pufo.rewards.feet_stumble.params["sensor_cfg"].body_names, ".*_ankle_roll_link")
        for cfg in (RP1ParkourEnvCfg(), RPOParkourEnvCfg()):
            self.assertIsInstance(cfg.scene.contact_forces, TerrainContactSensorCfg)
            self.assertTrue(cfg.scene.contact_forces.track_air_time)
            self.assertEqual(cfg.scene.contact_forces.history_length, cfg.decimation)
            self.assertEqual(cfg.rewards.locomotion.feet_stumble.weight, -1.)

    def test_slippage_uses_randomized_body_weight(self):
        force = self.env.scene["contact_forces"].data.net_forces_w
        force[:, 0, 2] = 30.
        actual = pufo_penalty_slippage(self.env, self.asset_cfg, self.sensor_cfg)
        torch.testing.assert_close(actual, torch.tensor([1., 0.]))
        force[:, 0, 2] = torch.tensor([20., 40.])
        torch.testing.assert_close(pufo_penalty_slippage(self.env, self.asset_cfg, self.sensor_cfg), torch.zeros(2))

    def test_slippage_projects_along_total_force(self):
        self.env.scene["contact_forces"].data.net_forces_w[:, 0] = torch.tensor([30., 0., 40.])
        torch.testing.assert_close(pufo_penalty_slippage(self.env, self.asset_cfg, self.sensor_cfg),
                                   torch.tensor([.8, .8]))

    def test_config_only_changes_pufo(self):
        from robolab.tasks.manager_based.ufo.ufo_env_cfg import UFOEnvCfg
        cfg = RP1PUFOEnvCfg()
        self.assertEqual(cfg.scene.contact_forces.history_length, cfg.decimation)
        self.assertTrue(cfg.scene.contact_forces.track_friction_forces)
        self.assertEqual(cfg.rewards.penalty_slippage.weight, -4.)
        self.assertIs(cfg.rewards.feet_stumble.func, pufo_feet_stumble)
        self.assertTrue(cfg.terminations.terrain_boundary.time_out)
        self.assertFalse(hasattr(UFOEnvCfg().terminations, "terrain_boundary"))

    def test_body_impact_config_selects_only_waist_yaw_shoulders_and_hips(self):
        from isaaclab.utils.string import resolve_matching_names
        from robolab.assets.robots.roboparty import PR1_LINKS
        from robolab.tasks.manager_based.ufo.agents.rp1_pufo_agent_cfg import RP1PUFORunnerCfg
        from robolab.tasks.manager_based.ufo.ufo_env_cfg import UFOEnvCfg
        from robolab.tasks.manager_based.ufo.mdp.rewards import pufo_penalty_body_impact
        from robolab.tasks.manager_based.ufo.mdp.rewards import JointLimitedBodyImpactPositionAction

        cfg = RP1PUFOEnvCfg()
        term = cfg.rewards.penalty_body_impact
        self.assertIs(term.func, pufo_penalty_body_impact)
        self.assertEqual(term.weight, -1.)
        self.assertIs(cfg.actions.joint_pos.class_type, JointLimitedBodyImpactPositionAction)
        self.assertEqual(term.params["force_threshold_ratio"], .05)
        self.assertEqual(term.params["smooth_l1_beta"], .25)
        self.assertEqual(term.params["impact_velocity_scale"], 1.)
        self.assertEqual(term.params["impact_velocity_cap"], 2.)
        self.assertEqual(term.params["severity_cap"], 4.)
        sensor_cfg = term.params["sensor_cfg"]
        self.assertEqual(sensor_cfg.name, "contact_forces")
        self.assertEqual(term.params["asset_cfg"].body_names, sensor_cfg.body_names)
        self.assertTrue(term.params["asset_cfg"].preserve_order)
        self.assertTrue(sensor_cfg.preserve_order)
        _, names = resolve_matching_names(sensor_cfg.body_names, PR1_LINKS)
        expected = {"waist_yaw_link"} | {
            f"{side}_{part}_{axis}_link"
            for side in ("left", "right")
            for part in ("shoulder", "hip")
            for axis in ("pitch", "roll", "yaw")
        }
        self.assertEqual(set(names), expected)
        ufo_cfg = UFOEnvCfg()
        self.assertFalse(hasattr(ufo_cfg.rewards, "penalty_body_contact"))
        self.assertIs(ufo_cfg.rewards.penalty_body_impact.func, pufo_penalty_body_impact)
        self.assertIsInstance(ufo_cfg.scene.contact_forces, TerrainContactSensorCfg)
        self.assertFalse(hasattr(cfg.rewards, "penalty_body_contact"))
        self.assertEqual(RP1PUFORunnerCfg().training_contract, "pufo_h200_alignment_v7_target_clip")

    def test_body_impact_matches_p3_reference_with_newton_force_sign(self):
        import ast
        import torch.nn.functional as F
        from pathlib import Path
        from robolab.tasks.manager_based.ufo.mdp.rewards import pufo_penalty_body_impact

        # Independent P3 implementation with [env, body, history, xyz] layout.
        source = Path("/tmp/pufo-zip.M1kbtU/pufo-p3-depth-qv-crop18/humanoidverse/agents/envs/humanoidverse_mjlab.py")
        if not source.is_file():
            self.skipTest("P3 reference code package is not available")
        tree = ast.parse(source.read_text())
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("peak_contact_force", "body_contact_severity")]
        reference = {"torch": torch, "F": F}
        exec(compile(tree, str(source), "exec"), reference)
        torch.manual_seed(1701)
        forces = torch.randn(8, 4, 6, 3) * 80
        velocities = torch.randn(8, 6, 3)
        masses = torch.rand(8, 6) * 10 + 1
        ids = [4, 0, 2]
        data = SimpleNamespace(net_forces_w_history=SimpleNamespace(torch=forces))
        env = SimpleNamespace(
            cfg=SimpleNamespace(decimation=4, sim=SimpleNamespace(gravity=(0., 0., -9.81))),
            scene={"robot": SimpleNamespace(data=SimpleNamespace(body_mass=masses)),
                   "contact_forces": SimpleNamespace(data=data)},
            action_manager=SimpleNamespace(get_term=lambda name: SimpleNamespace(pre_body_lin_vel_w=velocities)),
        )
        sensor_cfg = SimpleNamespace(name="contact_forces", body_ids=ids)
        asset_cfg = SimpleNamespace(name="robot", body_ids=ids)
        peak_force = reference["peak_contact_force"](-forces[:, :, ids].transpose(1, 2))
        weight = masses.sum(dim=-1) * 9.81
        # Remove only P3's static baseline before applying its aggregate cap.
        expected = (reference["body_contact_severity"](
            peak_force, velocities[:, ids], weight, severity_cap=float("inf"),
        ) - reference["body_contact_severity"](
            peak_force, torch.zeros_like(velocities[:, ids]), weight, severity_cap=float("inf"),
        )).clamp(max=4.)
        torch.testing.assert_close(pufo_penalty_body_impact(env, asset_cfg, sensor_cfg), expected,
                                   atol=1.e-6, rtol=1.e-6)

    def test_body_impact_peak_direction_threshold_mass_and_clamp(self):
        from robolab.tasks.manager_based.ufo.mdp.rewards import pufo_penalty_body_impact

        forces = torch.zeros(8, 4, 4, 3)
        forces[:, 3, 2, 2] = 100.  # Force on body points up; earlier tick is strongest.
        forces[:, 0, 2, 2] = -50.  # Latest opposite force must not determine direction.
        forces[0, 3, 2, 2] = 50000.  # Static pressing is unpenalized even under a huge load.
        forces[:, :, 0] = 10000.  # Unselected body.
        velocity = torch.zeros(8, 4, 3)
        velocity[1, 1, 2] = -1.5  # Sensor/body order deliberately differs.
        velocity[2, 1, 2] = 1.5
        velocity[3, 1, 0] = 10.  # Tangential speed does not amplify severity.
        forces[4, :, 2] = 0.
        forces[4, 0, 2, 2] = 25.  # Exactly 5% of 500 N weight.
        forces[5, :, 2] = 0.
        forces[5, 1, 2, 2] = 750.
        velocity[5, 1, 2] = -10.  # Speed gain and total severity must be capped.
        velocity[6, 1, 2] = -1.5
        velocity[7, 1, 2] = -10.  # Speed cap without hitting the total cap.
        masses = torch.full((8, 4), 12.5)
        masses[6] *= 2
        data = SimpleNamespace(net_forces_w_history=forces)
        action = SimpleNamespace(pre_body_lin_vel_w=velocity)
        env = SimpleNamespace(
            cfg=SimpleNamespace(decimation=4, sim=SimpleNamespace(gravity=(0., 0., -10.))),
            scene={"robot": SimpleNamespace(data=SimpleNamespace(body_mass=masses)),
                   "contact_forces": SimpleNamespace(data=data)},
            action_manager=SimpleNamespace(get_term=lambda name: action),
        )
        asset_cfg = SimpleNamespace(name="robot", body_ids=[1])
        sensor_cfg = SimpleNamespace(name="contact_forces", body_ids=[2])
        actual = pufo_penalty_body_impact(env, asset_cfg, sensor_cfg)
        torch.testing.assert_close(actual, torch.tensor([0., .10125, 0., 0., 0., 4., .01125, .18]))
        forces.zero_()
        torch.testing.assert_close(pufo_penalty_body_impact(env, asset_cfg, sensor_cfg), torch.zeros(8))
        forces[0, 0, 2, 0] = float("nan")
        self.assertTrue(torch.isnan(pufo_penalty_body_impact(env, asset_cfg, sensor_cfg)[0]))
        action.pre_body_lin_vel_w = None
        with self.assertRaisesRegex(RuntimeError, "pre-action"):
            pufo_penalty_body_impact(env, asset_cfg, sensor_cfg)

    def test_body_impact_action_snapshots_velocity_before_physics_without_aliasing(self):
        from unittest.mock import patch
        from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
        from robolab.tasks.manager_based.ufo.mdp.rewards import BodyImpactJointPositionAction

        action = object.__new__(BodyImpactJointPositionAction)
        velocity = torch.tensor([[[0., 0., -1.5]]])
        action._asset = SimpleNamespace(data=SimpleNamespace(
            body_link_lin_vel_w=SimpleNamespace(torch=velocity),
            joint_pos_limits=torch.tensor([[[-1., 1.], [-2., 2.]]]),
        ))
        action._joint_ids = slice(None)
        action._processed_actions = torch.zeros(1, 2)
        commands = torch.zeros(1, 2)
        with patch.object(JointPositionAction, "process_actions") as parent:
            action.process_actions(commands)
            parent.assert_called_once_with(commands)
        velocity.zero_()  # The collision changes the live velocity, not the snapshot.
        torch.testing.assert_close(action.pre_body_lin_vel_w, torch.tensor([[[0., 0., -1.5]]]))
        with patch.object(JointPositionAction, "process_actions"):
            action.process_actions(commands)
        torch.testing.assert_close(action.pre_body_lin_vel_w, torch.zeros_like(velocity))

    def test_action_clips_final_targets_with_selected_physical_limits(self):
        from unittest.mock import patch
        from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
        from robolab.tasks.manager_based.ufo.mdp.rewards import JointLimitedBodyImpactPositionAction

        action = object.__new__(JointLimitedBodyImpactPositionAction)
        limits = torch.tensor([[[-2., 1.], [-.1, .2], [-3., .06]],
                               [[-1., .5], [-.2, .3], [-2.8, .05]]])
        action._asset = SimpleNamespace(data=SimpleNamespace(
            body_link_lin_vel_w=torch.zeros(2, 1, 3),
            joint_pos_limits=SimpleNamespace(torch=limits),
        ))
        action._joint_ids = [2, 0]
        commands = torch.tensor([[-8., 8.], [.01, -.5]])
        original = commands.clone()
        # Parent's scale/offset stage has already produced absolute targets.
        action._processed_actions = commands.clone()
        with patch.object(JointPositionAction, "process_actions"):
            action.process_actions(commands)
        torch.testing.assert_close(action.processed_actions, torch.tensor([[-3., 1.], [.01, -.5]]))
        torch.testing.assert_close(commands, original)
        torch.testing.assert_close(action._asset.data.joint_pos_limits.torch, limits)

    def test_all_contacts_history_mask_and_reset(self):
        import os
        device = os.environ.get("ROBO_CONTACT_TEST_DEVICE", "cpu")
        # Three shapes on the same foot, another body, env 1, and unsensed terrain.
        shape_rows = wp.array([0, 0, 1, 2, -1], dtype=int, device=device)
        forces = wp.array(np.array([
            [0., 0., 200., 0., 0., 0.], [0., 0., 150., 0., 0., 0.],
            [30., 0., 0., 0., 0., 0.], [-80., 0., 10., 0., 0., 0.],
            [0., -50., -10., 0., 0., 0.], [900., 0., 0., 0., 0., 0.],
        ], dtype=np.float32), dtype=wp.spatial_vector, device=device)
        count = wp.array([5], dtype=int, device=device)
        s0 = wp.array([0, 0, 0, 1, 4, 0], dtype=int, device=device)
        s1 = wp.array([4, 4, -1, 4, 3, 4], dtype=int, device=device)
        mask = wp.array([True, True], dtype=bool, device=device)
        excess = wp.zeros(4, device=device)
        history = wp.zeros((2, 4, 2), device=device)
        args = [count, s0, s1, forces, shape_rows, mask, 2, 1., excess]
        # With only the first three contacts, a weak side collision must survive
        # two larger supporting forces on the very same geometry.
        count.fill_(3)
        wp.launch(reduce_contact_excess, 6, inputs=args, device=device)
        np.testing.assert_allclose(excess.numpy(), [30., 0., 0., 0.])
        count.fill_(5)
        excess.zero_()
        wp.launch(reduce_contact_excess, 6, inputs=args, device=device)
        np.testing.assert_allclose(excess.numpy(), [70., 0., 40., 0.])
        wp.launch(append_excess_history, (2, 2), inputs=[excess, mask, 2, 4, history], device=device)
        # An impact at tick 1 remains through tick 4, then expires at tick 5.
        excess.zero_()
        for _ in range(3):
            wp.launch(append_excess_history, (2, 2), inputs=[excess, mask, 2, 4, history], device=device)
        np.testing.assert_allclose(history.numpy()[:, 3, 0], [70., 40.])
        before = history.numpy().copy()
        masked = wp.array([False, True], dtype=bool, device=device)
        wp.launch(reset_excess_history, history.shape, inputs=[masked, history], device=device)
        np.testing.assert_array_equal(history.numpy()[0], before[0])
        self.assertEqual(np.count_nonzero(history.numpy()[1]), 0)
        wp.launch(append_excess_history, (2, 2), inputs=[excess, masked, 2, 4, history], device=device)
        np.testing.assert_array_equal(history.numpy()[0], before[0])
        wp.launch(append_excess_history, (2, 2), inputs=[excess, mask, 2, 4, history], device=device)
        self.assertEqual(np.count_nonzero(history.numpy()), 0)
        # Masked-out environments cannot acquire another environment's impact.
        excess.zero_()
        args[5] = masked
        wp.launch(reduce_contact_excess, 6, inputs=args, device=device)
        np.testing.assert_allclose(excess.numpy(), [0., 0., 40., 0.])
        invalid = forces.numpy()
        invalid[4, 0] = np.nan
        forces.assign(invalid)
        excess.zero_()
        wp.launch(reduce_contact_excess, 6, inputs=args, device=device)
        self.assertTrue(np.isinf(excess.numpy()[2]))


class BoundaryTests(unittest.TestCase):
    def test_global_map_not_individual_tiles(self):
        cfg = RP1PUFOEnvCfg().scene.terrain
        terrain = SimpleNamespace(cfg=cfg, terrain_origins=torch.zeros(10, 7, 3))
        root = torch.tensor([[5.1, 0., 1.], [59., 0., 1.], [60., 0., 1.],
                             [-60., 0., 1.], [0., 45., 1.], [0., -45., 1.], [100., 0., 1.]])
        class Scene(dict):
            pass
        scene = Scene(robot=SimpleNamespace(data=SimpleNamespace(root_pos_w=root)))
        scene.terrain = terrain
        env = SimpleNamespace(scene=scene, num_envs=len(root), device="cpu")
        # Core half extents (50,35) + 12 m guard/border; required clearance ~2.209 m.
        actual = terrain_out_of_bounds(env)
        torch.testing.assert_close(actual, torch.tensor([False, False, True, True, True, True, True]))
        terrain.cfg = SimpleNamespace(terrain_type="plane")
        self.assertFalse(terrain_out_of_bounds(env).any())

    def test_attachment_margin_formula(self):
        points = torch.tensor([[0., 0.], [62., 0.], [0., -48.]])
        actual = terrain_grid_boundary_margin(points, (10., 10.), num_rows=10, num_cols=7, border_width=12.)
        torch.testing.assert_close(actual, torch.tensor([47., 0., -1.]))


if __name__ == "__main__":
    unittest.main()
