"""Exercise all RP1 sim2sim motor integrations without policies or a viewer."""

import importlib.util
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import yaml

from robolab.utils.mujoco_actuators import RoboPartyMujocoActuator

ROOT = Path(__file__).resolve().parents[3]
MJCF = ROOT / "data/robots/roboparty/rp1.3/mjcf/rp1_flat.xml"


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/mujoco" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(MJCF.is_file(), "RP1 local robot assets are not installed")
class RP1IntegrationTests(unittest.TestCase):
    def test_amp_uses_training_actuator(self):
        from robolab.tasks.manager_based.amp.rp1_amp_env_cfg import RP1AmpEnvCfg
        sim = script("sim2sim_rp1_amp").Simulation(MJCF, RP1AmpEnvCfg().to_dict(), 3.)
        self.addCleanup(sim.close)
        self.assertIsInstance(sim.motor, RoboPartyMujocoActuator)
        with patch.object(sim.motor.actuator, "compute", wraps=sim.motor.actuator.compute) as compute:
            sim.set_action(np.zeros(24))
            sim.advance()
            self.assertEqual(compute.call_count, 1)
        self.assertTrue(np.isfinite(sim.data.qpos).all())

    def test_pufo_uses_training_actuator(self):
        from robolab.assets.robots.roboparty import RP1_JOINT_NAMES
        from robolab.tasks.manager_based.ufo.pufo_env_cfg import RP1PUFOEnvCfg
        module = script("sim2sim_rp1_pufo")
        sim = module.Simulation(module.make_model(False), RP1_JOINT_NAMES, RP1PUFOEnvCfg().to_dict())
        self.addCleanup(sim.close)
        self.assertIsInstance(sim.motor, RoboPartyMujocoActuator)
        with patch.object(sim.motor.actuator, "compute", wraps=sim.motor.actuator.compute) as compute:
            sim.advance(np.zeros(24), 1.)
            self.assertEqual(compute.call_count, 4)
        self.assertTrue(np.isfinite(sim.data.qpos).all())

    def test_pufo_reads_unified_npz_without_slicing_metadata(self):
        module = script("sim2sim_rp1_pufo")
        manifest = module.default_motion_manifest()
        if not manifest.is_file():
            self.skipTest("Local motion data is not installed")
        metadata = json.loads(manifest.read_text())
        path = manifest.parent / metadata["clips"][0]["npz"]
        motion = module.read_motion(path, metadata["npz"]["joint_order"], metadata["npz"]["body_order"])
        self.assertEqual(set(motion), {"fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                                       "body_lin_vel_w", "body_ang_vel_w"})
        for name, values in motion.items():
            if name != "fps":
                self.assertEqual(len(values[:2]), 2)
                self.assertTrue(np.issubdtype(values.dtype, np.number))

    def test_pufo_terrain_asset_location_and_compilation(self):
        module = script("sim2sim_rp1_pufo")
        self.assertEqual(module.TERRAIN_DIR, MJCF.parent / "rp1_simple_terrain")
        self.assertEqual(module.TERRAIN_XML.name, "rp1_simple_terrain.xml")
        metadata = module.terrain_metadata()
        terrain = mujoco.MjModel.from_xml_path(str(module.TERRAIN_XML))
        self.assertEqual(terrain.ngeom, metadata["ngeom"])
        self.assertEqual(terrain.nhfield, metadata["nhfield"])
        self.assertNotIn("H200", metadata["source"])
        combined = module.make_model(True)
        self.assertGreater(combined.ngeom, terrain.ngeom)

    def test_missing_terrain_reports_asset_installation(self):
        module = script("sim2sim_rp1_pufo")
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(module, "TERRAIN_XML", Path(directory) / "rp1_simple_terrain.xml"):
                with self.assertRaisesRegex(FileNotFoundError, "install rp1_simple_terrain.xml and meta.json"):
                    module.make_model(True)

    def test_parkour_reads_robot_configuration_and_binds_motor_by_name(self):
        module = script("sim2sim_rp1_parkour")
        robot = module.load_robot_config()
        model = mujoco.MjModel.from_xml_path(str(MJCF))
        motor, ids = module.make_motor(model, robot)
        self.assertIsInstance(motor, RoboPartyMujocoActuator)
        self.assertEqual(len(ids), 24)
        with patch.object(motor.actuator, "compute", wraps=motor.actuator.compute) as compute:
            torque = motor.compute(np.zeros(24), np.zeros(24), np.zeros(24))
            self.assertEqual(compute.call_count, 1)
        np.testing.assert_array_equal(torque, np.zeros(24))
        self.assertFalse(model.actuator_ctrllimited[ids].any())

    def test_parkour_saved_config_and_full_headless_control_loop(self):
        from robolab.assets.robots.roboparty import RP1_JOINT_NAMES
        module = script("sim2sim_rp1_parkour")
        robot = module.load_robot_config()
        robot["actuators"]["all_joints"]["actuator_effort_limit"] = 100.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "env.yaml"
            path.write_text(yaml.safe_dump({"scene": {"robot": robot}}))
            loaded = module.load_robot_config(path)
        defaults = np.array([next((v for p, v in loaded["init_state"]["joint_pos"].items()
                                   if re.fullmatch(p, name)), 0.) for name in RP1_JOINT_NAMES])
        cfg = SimpleNamespace(
            sim_config=SimpleNamespace(mujoco_model_path=str(MJCF), dt=.005, decimation=4,
                                       sim_duration=.02, depth_camera_body="waist_yaw_link"),
            robot_config=SimpleNamespace(robot=loaded, num_actions=24, default_pos=defaults,
                                         frame_stack=8, depth_encoder_frames=8, action_scale=.25,
                                         usd2urdf=list(range(24))),
        )
        encoder = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name="depth")],
                                  run=lambda *args: [np.zeros((1, 32), dtype=np.float32)])
        actor = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name="obs")],
                                run=lambda *args: [np.zeros((1, 24), dtype=np.float32)])
        motors = []
        make_motor = module.make_motor

        def capture_motor(model, config):
            result = make_motor(model, config)
            motors.append(result[0])
            return result

        with patch.object(module, "make_motor", side_effect=capture_motor):
            module.run_mujoco_onnx(encoder, actor, cfg, headless=True, record_video=False,
                                   save_plots=False, realtime_sync=False, quiet=True)
        self.assertEqual(len(motors), 1)
        np.testing.assert_array_equal(motors[0].effort_limit, np.full(24, 100.))


if __name__ == "__main__":
    unittest.main()
