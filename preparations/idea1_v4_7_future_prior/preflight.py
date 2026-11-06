from pathlib import Path
import numpy as np
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3.history import DecoderCoordinates
from preparations.idea1_v3_4_v12_compensation.server import selection
from preparations.kv_reuse_probe.probe import load_observation
from preparations.idea1_v4_1_warmstart.schedule import mix
from preparations.idea1_v4_1_warmstart.sampling import sample_gaussian, sample_initial
from .common import MASK, conditions
from .policy import Policy, shift_prediction


def run(backend,output):
    output=Path(output);arrays={};checks=[];calls=0
    old_sample,old_verify=backend.sample,backend.verify
    def forbidden(*args,**kwargs):raise AssertionError('Full/legacy AE or verifier forbidden')
    backend.sample=backend.verify=forbidden
    try:
        for item in selection()[::10]:
            obs,prompt=load_observation(item['current'])
            suite,seed=item['case']['suite'],int(item['current']['seed'])
            firsts=[]
            for cfg in conditions():
                policy=Policy(backend);policy.configure(cfg);previous=None;ack_lengths=(2,5,1,5)
                for i in range(4):
                    result=policy.predict(obs,prompt,suite,seed+i);calls+=1
                    route,timing=result['route'],result['timing']
                    assert route['D']==timing['nodes']['actual_NFE']==1
                    assert not route['full_ae_executed'] and not route['full_action_executed'] and not route['verifier']
                    assert route['residual_refresh']==(i%3==0)
                    expected_start=1. if i==0 else .1
                    assert timing['nodes']['nodes']==[expected_start,0.]
                    assert route['actual_noise_std']==route['actual_gripper_noise_std']==expected_start
                    history=policy.history
                    ctx,_=backend.prepare_residual(obs,prompt,MASK,history=history[-1]['residuals'],older=history[-2]['residuals'] if len(history)==2 else None,ratio=route['extrapolation_ratio'],reference=True)
                    if i==0:
                        a,r,_=sample_gaussian(backend,ctx,seed+i,1)
                        firsts.append(result['raw'])
                    else:
                        shifted=shift_prediction(previous['actions'],ack_lengths[i-1])
                        prior=result['arrays']['prior_commands']
                        assert route['prior_source_round']==i-1 and route['shift_executed_length']==ack_lengths[i-1]
                        assert route['shift_source_step']==previous['step']
                        if policy.kalman is not None:
                            assert np.array_equal(prior[:,:6],policy.kalman.last_forecast[:,:6])
                            assert np.array_equal(prior[:,6],shifted[:,6])
                            assert policy.kalman.filter.count==policy.step
                        else:assert np.array_equal(prior,shifted)
                        coordinates=DecoderCoordinates(backend.adapter.backend.output_transform,ctx['transformed_state'])
                        encoded=coordinates.encode_commands(prior,np.zeros((50,32),np.float32))
                        expected=mix(encoded,seed+i,.1,1.)
                        assert np.array_equal(encoded[:,7:],np.zeros((50,25),np.float32))
                        assert np.array_equal(result['arrays']['prior_encoded_prefix'],encoded[:5])
                        assert np.array_equal(result['arrays']['prior_initial_prefix'],expected[:5])
                        a,r,_=sample_initial(backend,ctx,expected,1,.1)
                        checks.append(dict(name=f'{suite}_{cfg["id"]}_causal_shift_reencode_{i}',passed=True))
                    raw=r.float().cpu().numpy()
                    checks.append(dict(name=f'{suite}_{cfg["id"]}_reference_{i}',passed=bool(np.allclose(raw,result['raw'],atol=1e-5,rtol=1e-5) and np.allclose(a,result['actions'],atol=1e-5,rtol=1e-5)),max_abs=float(np.abs(raw-result['raw']).max())))
                    arrays[f'{suite}_{cfg["id"]}_round{i}']=result['raw']
                    previous=result;policy.acknowledge(result['actions'][:ack_lengths[i]])
                assert policy.counts['full']==policy.counts['verify']==0
                policy.configure(cfg);assert policy.future is None and policy.history==[] and policy.step==0
            checks.append(dict(name=suite+'_identical_gaussian_bootstrap',passed=all(np.allclose(firsts[0],a,atol=1e-5,rtol=1e-5) for a in firsts[1:])))
    finally:backend.sample,backend.verify=old_sample,old_verify
    np.savez(output/'gate_actions.npz',**arrays)
    outcome=dict(passed=all(c['passed'] for c in checks),checks=checks,policy_calls=calls,actual_NFE=1,full_ae_calls=0,verifier_calls=0,native_encoder_graph=False,native_ae_graph=False)
    atomic(output/'preflight.json',outcome)
    if not outcome['passed']:raise RuntimeError('Future-prior mechanism or numerical check failed')
    return outcome
