"""A fine terrain under a foot must not truncate heightfield candidates."""

import os
import unittest

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from robolab.utils import mjwarp_contact_guard as guard
from robolab.utils.guarded_mjwarp import NewtonGuardedMJWarpManager as Manager


@unittest.skipUnless(os.environ.get("ROBO_TEST_HFIELD_CUDA") == "1", "explicit CUDA collision regression")
class TestHeightfieldCapacity(unittest.TestCase):
    def test_fine_grid_overflow_is_fixed_without_changing_contact_reduction(self):
        xml = """<mujoco><asset><hfield name="ground" nrow="17" ncol="17"
                    size=".2 .2 .01 .1"/></asset><worldbody>
                    <geom type="hfield" hfield="ground"/>
                    <body pos="0 0 .045"><freejoint/>
                    <geom type="box" size=".10 .06 .05" mass="1"/>
                    </body></worldbody></mujoco>"""
        model = mujoco.MjModel.from_xml_string(xml)
        cpu_data = mujoco.MjData(model)
        mujoco.mj_forward(model, cpu_data)
        guard.install()
        try:
            with wp.ScopedDevice("cuda:0"):
                gpu_model = mjw.put_model(model)
                stock = mjw.put_data(model, cpu_data, nworld=1, nconmax=256, njmax=1024)
                mjw.collision(gpu_model, stock)
                self.assertTrue(int(stock.overflow.numpy()[0]) & 32)
                Manager._configure_heightfield_capacity(512)
                fixed = mjw.put_data(model, cpu_data, nworld=1, nconmax=256, njmax=1024)
                mjw.collision(gpu_model, fixed)
                self.assertEqual(int(fixed.overflow.numpy()[0]), 0)
                self.assertEqual(int(fixed.nacon.numpy()[0]), 4)
                self.assertTrue(np.isfinite(fixed.contact.pos.numpy()[:4]).all())
        finally:
            Manager._restore_heightfield_capacity()
            guard.restore()


if __name__ == "__main__":
    unittest.main()
