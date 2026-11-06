import unittest
import numpy as np
from .native_envs import formal_calvin_contract,chain_metrics,calvin_native_action,pusht_native_action,simpler_native_contract,calvin_policy_state


class NativeContractTests(unittest.TestCase):
    def test_official_formal_contract(self):
        contract=formal_calvin_contract()
        self.assertEqual(contract['sequence_count'],1000)
        self.assertEqual(contract['subtasks_per_sequence'],5)
        self.assertEqual(contract['max_native_steps_per_subtask'],360)
        self.assertTrue(contract['test_set_shared_between_training_splits'])

    def test_chain_metrics_keep_long_horizon_denominator(self):
        values=chain_metrics([0,1,3,5])
        self.assertEqual(values['mean_completed_chain_length'],2.25)
        self.assertEqual(values['success_at_least'],{'1':.75,'2':.5,'3':.5,'4':.25,'5':.25})

    def test_native_calvin_relative_is_euler_not_libero_rotvec(self):
        action,audit=calvin_native_action([2,-2,.5,.5,0,-.2,-.7])
        np.testing.assert_array_equal(action,[1,-1,.5,.5,0,-.2,-1])
        self.assertEqual(audit['native_position_scale_m'],.02)
        self.assertEqual(audit['native_euler_scale_rad'],.05)
        absolute,_=calvin_native_action([.1,.2,.6,3.1,0,1.5,1],'absolute')
        self.assertEqual(len(absolute),3)
        self.assertEqual(absolute[1],[3.1,0,1.5])

    def test_native_pusht_is_only_two_dimensional(self):
        np.testing.assert_array_equal(pusht_native_action([-10,600]),[0,512])
        with self.assertRaises(ValueError): pusht_native_action([0]*7)

    def test_training_gripper_semantics_are_explicit(self):
        state=np.r_[np.arange(6),.08,np.zeros(7),-1.]
        self.assertEqual(calvin_policy_state(state,'action_sign')[-1],-1)
        self.assertAlmostEqual(float(calvin_policy_state(state,'opening_width_m')[-1]),.08,places=6)
        with self.assertRaises(ValueError): calvin_policy_state(state,'guess')

    def test_google_widowx_contracts_not_interchangeable(self):
        google=simpler_native_contract('google_robot_static');widow=simpler_native_contract('widowx')
        self.assertEqual(google['native_hz'],3);self.assertEqual(widow['native_hz'],5)
        self.assertFalse(google['use_target_arm']);self.assertTrue(widow['use_target_arm'])
        self.assertEqual(google['frame'],'ee_align');self.assertEqual(widow['frame'],'ee_align2')


if __name__=='__main__': unittest.main()
