"""Behavioral CPU reset tests; physical scene proof is a separate actual gate."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from .native_reset import FreshResetCalvinEnv,compare_observations


class ResetTests(unittest.TestCase):
    def test_real_controller_updated_before_native_reset_and_refreshed(self):
        env=FreshResetCalvinEnv.__new__(FreshResetCalvinEnv)
        target=np.arange(15,dtype=float);target[-1]=1
        seen=[]
        def native_reset(this,robot,scene):
            seen.append(this.env.robot.gripper_action)
            this.obs={'robot_obs':robot.copy()}
        env.env=SimpleNamespace(robot=SimpleNamespace(gripper_action=-1),get_obs=lambda:dict(robot_obs=target.copy()))
        with patch('preparations.idea1_v3_2_e4_supplement.native_reset.ExactNativeCalvinEnv.reset_state',native_reset):
            result=env.reset_state(target,np.zeros(24))
        self.assertEqual(seen,[1]);self.assertEqual(result['robot_obs'][-1],1)

    def test_invalid_target_rejected(self):
        env=FreshResetCalvinEnv.__new__(FreshResetCalvinEnv)
        with self.assertRaises(ValueError):env.reset_state(np.zeros(15),np.zeros(24))

    def test_comparison_has_explicit_tolerances_not_hashes(self):
        obs={key:np.zeros(shape) for key,shape in dict(robot_obs=15,scene_obs=24,base_rgb=(2,2,3),wrist_rgb=(2,2,3),noise=(5,32)).items()}
        other={k:v.copy() for k,v in obs.items()};other['robot_obs'][0]=1e-13
        result=compare_observations(obs,other)
        self.assertTrue(result['passed']);self.assertFalse(result['comparisons']['robot_obs']['bitwise_equal'])
        other['robot_obs'][-1]=2
        self.assertFalse(compare_observations(obs,other)['passed'])


if __name__=='__main__':unittest.main()
