import unittest
from tools.train_next_v2.execution_guard.guard import should_hold


class GuardPolicyTests(unittest.TestCase):
    def test_exact_live_target_and_ac_can_hold(self):
        self.assertTrue(should_hold(True,True,dict(ac=1,battery_saver=0)))

    def test_battery_unknown_power_or_saver_releases(self):
        for ac,saver in ((0,0),(255,0),(1,1),(255,1)):
            self.assertFalse(should_hold(True,True,dict(ac=ac,battery_saver=saver)))

    def test_dead_unknown_or_unleased_target_releases(self):
        for alive,lease in ((False,True),(None,True),(True,False),(False,False)):
            self.assertFalse(should_hold(alive,lease,dict(ac=1,battery_saver=0)))


if __name__=='__main__': unittest.main()
