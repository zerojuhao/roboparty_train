"""CPU regressions for policy-independent MuJoCo physics assets."""

import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from robolab.utils.mujoco_physics import (
    CONTACT_FIELDS, NewtonPhysicsAsset, export_physics_asset,
)
from robolab.utils.mujoco_actuators import RoboPartyMujocoActuator

URDF = """<robot name="test"><link name="base"/><link name="arm"/>
<joint name="hinge" type="revolute"><parent link="base"/><child link="arm"/>
<limit lower="-1" upper="1" effort="20" velocity="10"/></joint></robot>"""
XML = """<mujoco><compiler angle="radian"/><default>
<default class="collision"><geom type="sphere" size="0.1"/></default>
<default class="visual"><geom type="sphere" size="0.1" contype="0" conaffinity="0" group="2"/></default>
</default><worldbody><geom type="plane" size="5 5 0.1"/>
<body name="base" pos="0 0 1"><freejoint/><geom class="collision"/>
<body name="arm" pos="0.2 0 0"><joint name="hinge" type="hinge" range="-1 1"/>
<geom class="collision"/><geom class="visual"/></body></body></worldbody></mujoco>"""


class PhysicsAssetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.urdf = self.root / "robot.urdf"
        self.urdf.write_text(URDF)
        self.path = self.root / "physics.json"
        self.source = mujoco.MjModel.from_xml_string(XML)
        self.profile = export_physics_asset(self.source, self.urdf, self.path,
                                             data_root=self.root, timestep=.0025)
        self.asset = NewtonPhysicsAsset(self.path, data_root=self.root)

    def test_export_has_no_policy_or_motor_snapshot(self):
        self.assertNotIn("checkpoint", self.path.read_text())
        self.assertNotIn("actuators", self.profile)
        self.assertEqual(self.profile["urdf"], "robot.urdf")
        self.assertEqual(self.profile["joint_names"], ["hinge"])

    def test_export_is_readable_json(self):
        self.assertEqual(self.path.read_text(), json.dumps(self.profile, indent=2) + "\n")
        self.assertEqual(json.loads(self.path.read_text()), self.profile)

    def test_roundtrip_collision_and_solver_parameters(self):
        robot = ET.fromstring(XML)
        self.asset.align_robot_xml(robot)
        model = mujoco.MjModel.from_xml_string(ET.tostring(robot, encoding="unicode"))
        self.asset.align_compiled_model(model)
        self.assertTrue(self.asset.is_native_model(model))
        self.assertFalse(self.asset.is_native_model(self.source))
        self.assertEqual(model.opt.timestep, .0025)
        self.assertEqual(model.nbody, self.source.nbody)
        self.assertEqual((model.geom_group == 2).sum(), 1)
        for index, geom in enumerate(self.profile["geometries"]):
            if geom["body"] == "world":
                continue
            i = model.geom(self.asset.prefix + str(index)).id
            for key in CONTACT_FIELDS:
                np.testing.assert_array_equal(getattr(model, "geom_" + key)[i], geom[key])

    def test_stale_source_is_rejected(self):
        self.urdf.write_text(URDF + "\n")
        with self.assertRaisesRegex(ValueError, "stale"):
            _ = self.asset.profile

    def test_profiles_are_not_a_global_singleton(self):
        path = self.root / "other.json"
        other = dict(self.profile, options=dict(self.profile["options"], timestep=.001))
        path.write_text(json.dumps(other))
        self.assertEqual(self.asset.profile["options"]["timestep"], .0025)
        self.assertEqual(NewtonPhysicsAsset(path, data_root=self.root).profile["options"]["timestep"], .001)

    def test_disabled_contacts_are_rejected(self):
        self.profile["options"]["disableflags"] |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        self.path.write_text(json.dumps(self.profile))
        with self.assertRaisesRegex(ValueError, "retain contacts"):
            _ = self.asset.profile

    def test_missing_body_is_rejected(self):
        robot = ET.fromstring(XML)
        robot.find(".//body[@name='arm']").set("name", "unknown")
        with self.assertRaisesRegex(ValueError, "missing.*bodies"):
            self.asset.align_robot_xml(robot)

    def test_joint_range_mismatch_is_rejected(self):
        robot = ET.fromstring(XML)
        robot.find(".//joint[@name='hinge']").set("range", "-2 2")
        with self.assertRaisesRegex(ValueError, "ranges differ"):
            self.asset.align_robot_xml(robot)

    def test_export_requires_nominal_plane_and_valid_timestep(self):
        model = mujoco.MjModel.from_xml_string(XML.replace('type="plane"', 'type="box"'))
        with self.assertRaisesRegex(ValueError, "ground plane"):
            export_physics_asset(model, self.urdf, self.path, data_root=self.root, timestep=.0025)
        with self.assertRaisesRegex(ValueError, "timestep"):
            export_physics_asset(self.source, self.urdf, self.path, data_root=self.root, timestep=0)

    def motor_config(self, **kwargs):
        return dict(class_type="RoboPartyActuator", keep_velocity=5., rated_effort=1.,
                    rated_velocity=2., stiffness=100., damping=5., joint_names_expr=[".*"], **kwargs)

    def test_legacy_solver_limit_does_not_replace_internal_limit(self):
        motor = RoboPartyMujocoActuator(
            self.urdf, ["hinge"], self.motor_config(effort_limit_sim=15., velocity_limit_sim=12.))
        np.testing.assert_array_equal(motor.effort_limit, [20.])
        np.testing.assert_array_equal(motor.velocity_limit, [12.])
        np.testing.assert_array_equal(motor.keep_velocity, [5.])

    def test_explicit_motor_limits_and_config_changes_are_supported(self):
        for peak in (15., 30.):
            motor = RoboPartyMujocoActuator(
                self.urdf, ["hinge"], self.motor_config(actuator_effort_limit=peak,
                                                       actuator_velocity_limit={"hinge": 11.}))
            np.testing.assert_array_equal(motor.effort_limit, [peak])
            np.testing.assert_array_equal(motor.velocity_limit, [11.])

    def test_conflicting_aliases_and_invalid_motor_limits_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "conflicting"):
            RoboPartyMujocoActuator(self.urdf, ["hinge"],
                                   self.motor_config(effort_limit=10., actuator_effort_limit=15.))
        with self.assertRaisesRegex(ValueError, "keep_velocity"):
            RoboPartyMujocoActuator(self.urdf, ["hinge"], self.motor_config(actuator_velocity_limit=4.))

    def test_unsupported_actuator_is_not_silently_reinterpreted(self):
        with self.assertRaisesRegex(ValueError, "RoboPartyActuator"):
            RoboPartyMujocoActuator(self.urdf, ["hinge"], dict(class_type="DCMotor", keep_velocity=5.))

    def test_motor_uses_training_class_and_does_not_mutate_configuration(self):
        from copy import deepcopy
        from robolab.assets.robots.roboparty_actuators import RoboPartyActuator
        cfg = self.motor_config(effort_limit_sim=15., min_delay=1, max_delay=2, rfi_lim=.1)
        before = deepcopy(cfg)
        motor = RoboPartyMujocoActuator(self.urdf, ["hinge"], cfg)
        self.assertIsInstance(motor.actuator, RoboPartyActuator)
        self.assertEqual(cfg, before)
        self.assertEqual(motor.actuator.cfg.max_delay, 0)
        self.assertEqual(motor.actuator.cfg.rfi_lim, 0.)

    def test_real_motor_pd_rolloff_braking_and_reset(self):
        motor = RoboPartyMujocoActuator(self.urdf, ["hinge"], self.motor_config(actuator_effort_limit=15.))
        np.testing.assert_array_equal(motor.compute([.1], [0.], [0.]), [10.])
        np.testing.assert_array_equal(motor.compute([10.], [0.], [7.5]), [7.5])
        np.testing.assert_array_equal(motor.compute([-10.], [0.], [11.]), [-15.])
        np.testing.assert_array_equal(motor.compute([-10.], [0.], [-7.5]), [-7.5])
        motor.reset()
        np.testing.assert_array_equal(motor.compute([.1], [0.], [0.]), [10.])

    def test_nonfinite_motor_inputs_fail_before_simulation(self):
        motor = RoboPartyMujocoActuator(self.urdf, ["hinge"], self.motor_config())
        with self.assertRaisesRegex(ValueError, "finite motor input"):
            motor.compute([float("nan")], [0.], [0.])


if __name__ == "__main__":
    unittest.main()
