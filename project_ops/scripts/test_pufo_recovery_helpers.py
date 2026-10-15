import unittest
import evaluate_pufo_push_recovery as recovery


class RecoveryTests(unittest.TestCase):
    def test_down_requires_nonfoot_ground_contact(self):
        self.assertFalse(recovery.is_down(.3,.2,0.))
        self.assertTrue(recovery.is_down(.3,.2,100.))
        self.assertFalse(recovery.is_down(.8,1.,100.))

    def test_standing_requires_low_speed_two_feet_and_no_body_support(self):
        self.assertTrue(recovery.is_standing(.75,.95,.1,[100,100],0))
        self.assertFalse(recovery.is_standing(.75,.95,.8,[100,100],0))
        self.assertFalse(recovery.is_standing(.75,.95,.1,[100,0],0))
        self.assertFalse(recovery.is_standing(.75,.95,.1,[100,100],100))

    def test_stability_must_persist(self):
        self.assertEqual(recovery.first_stable_stand([True]*49+[False]+[True]*50,0),(1.,1.98))
        self.assertEqual(recovery.first_stable_stand([True]*49,0),(None,None))

    def test_force_is_bounded_and_removed(self):
        for t,value in [(2.,0.),(3.,0.),(3.05,150.),(3.2,300.),(4.2,0.),(8.,0.)]:
            self.assertAlmostEqual(recovery.pulse(t,300.),value)


if __name__=='__main__':
    unittest.main()
