"""Fixed cross-module algorithms and paired two-stream noise contracts."""
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
from preparations.idea1_v2_2.common import seed as original_seed
from . import common,rollout
from .noise import model_seed


def manifest(stage,repeat=0):
    tasks=[0] if stage=='calibration' else list(range(10))
    m=dict(phase='cross_module_'+stage+'_v4',noise_repeat=repeat,conditions=common.conditions(),planned_conditions=list(common.CONDITIONS),suites=list(common.SUITES),tasks=tasks,
           case_table=[dict(suite=s,task=t) for s in common.SUITES for t in tasks],science=dict(settling_actions=10,policy_action_limits=dict(common.LIMITS)))
    if stage=='calibration':m.update(calibration_inits=[45],cases_per_condition=4,total_calibration_episodes=12)
    else:m.update(confirmation_inits=[47,48,49],cases_per_condition=120,total_confirmation_episodes=360,preregistered_hypotheses=common.DIRECTIONS)
    return m


class Contracts(unittest.TestCase):
    def test_exact_different_module_budgets_are_declared(self):
        configs={c['id']:c for c in common.conditions()}
        self.assertEqual(configs['vlm_c']['skip'],[4,7,10,13,15,17]);self.assertEqual(configs['vlm_c']['D'],2);self.assertEqual(configs['vlm_c']['ae_skip'],[])
        self.assertEqual(configs['ae14_d3']['skip'],[]);self.assertEqual(configs['ae14_d3']['D'],3);self.assertEqual(configs['ae14_d3']['ae_skip'],[2,6,7,8])

    def test_stream0_preserves_old_seeds_stream1_is_distinct_and_deterministic(self):
        for s in common.SUITES:
            for task in range(10):
                for initial in [45,47,48,49]:
                    for request in [0,1,10]:
                        a=model_seed(0,s,task,initial,request);b=model_seed(1,s,task,initial,request)
                        self.assertEqual(a,original_seed(s,task,initial,request));self.assertNotEqual(a,b)
                        self.assertEqual(b,model_seed(1,s,task,initial,request));self.assertTrue(0<=b<2**32)

    def test_unregistered_noise_stream_is_rejected(self):
        for repeat in [True,-1,2,None]:
            with self.assertRaises(ValueError):model_seed(repeat,'libero_spatial',0,47,0)

    def test_stage_budgets_and_fixed_task_shards(self):
        for stage,repeat,n in [('calibration',0,4),('confirm',0,120),('confirm',1,120)]:
            m=manifest(stage,repeat);self.assertEqual(len(common.cases(m)),n)
            shards=[common.cases(m,stage,i,2) for i in range(2)];self.assertEqual(sum(map(len,shards)),n)
            self.assertFalse({(c['suite'],c['task']) for c in shards[0]}&{(c['suite'],c['task']) for c in shards[1]})

    def test_wrong_physical_states_or_directions_rejected(self):
        m=manifest('confirm');m['confirmation_inits']=[45,46,47]
        with self.assertRaises(ValueError):common.validate_manifest(m)
        m=manifest('confirm');m['preregistered_hypotheses']=list(reversed(common.DIRECTIONS))
        with self.assertRaises(ValueError):common.validate_manifest(m)

    def test_calibration_cannot_use_second_noise_stream(self):
        with self.assertRaises(ValueError):common.validate_manifest(manifest('calibration',1))

    def test_source_algorithms_resolve_without_relabeling_original_records(self):
        for alias,source in [('full_d10','full_d10'),('vlm_c','spread_c'),('ae14_d3','ae14_d3')]:
            r,m,cm,w,cid=common.reference(alias,verify_sources=True);self.assertEqual(cid,source)

    def test_real_rollout_delivers_the_selected_stream_to_initial_gate(self):
        class StopBeforePredict(Exception):pass
        obs=dict(robot0_eef_pos=np.zeros(3),robot0_eef_quat=np.array([0.,0.,0.,1.]))
        class Env:
            def __init__(self):
                ctl=NS(input_min=-np.ones(7),input_max=np.ones(7),output_min=-np.array([.05,.05,.05,.5,.5,.5,1.]),output_max=np.array([.05,.05,.05,.5,.5,.5]))
                self.env=NS(control_freq=20.,robots=[NS(controller=ctl)],sim=NS(data=NS(time=0.)))
            def seed(self,x):pass
            def reset(self):return obs
            def set_init_state(self,v):return obs
            def step(self,a):return obs,0,False,{}
        with tempfile.TemporaryDirectory() as folder:
            bddl=Path(folder)/'task.bddl';bddl.write_text('fixture')
            class Pool:
                def acquire(self,*args):return Env(),NS(language='fixture'),bddl,None,True
                def release(self,*args):pass
            calls=[]
            def rpc(port,request):
                calls.append(request['op'])
                if request['op']=='health':return dict(protocol={})
                if request['op']=='configure':return {}
                raise AssertionError('The gate must stop before prediction')
            for repeat in [0,1]:
                captured=[]
                def gate(identity,request):
                    captured.append((identity['noise_repeat'],request['seed']));raise StopBeforePredict()
                with patch.object(rollout,'official_init_states',return_value=[np.zeros(3,dtype=np.float32)]*50):
                    with self.assertRaises(StopBeforePredict):
                        rollout.episode(Path(folder)/str(repeat),1,dict(suite='libero_spatial',task=0,init=47),common.conditions()[1],Pool(),'science',
                            max_policy_actions=220,noise_repeat=repeat,rpc_call=rpc,initial_input_gate=gate)
                self.assertEqual(captured,[(repeat,model_seed(repeat,'libero_spatial',0,47,0))])
            self.assertNotIn('predict',calls)


if __name__=='__main__':unittest.main()
