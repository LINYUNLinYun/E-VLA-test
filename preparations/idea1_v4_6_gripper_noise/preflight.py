from pathlib import Path
import numpy as np
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3.history import DecoderCoordinates
from preparations.idea1_v3_4_v12_compensation.server import selection
from preparations.kv_reuse_probe.probe import load_observation
from preparations.idea1_v4_1_warmstart.schedule import mix
from preparations.idea1_v4_1_warmstart.sampling import sample_gaussian, sample_initial
from .common import MASK, OLD, conditions, hold_conditions
from .policy import Policy, make_initial


def run(backend,output):
    output=Path(output);arrays={};checks=[];historic=[];calls=0
    old_sample,old_verify=backend.sample,backend.verify
    def forbidden(*args,**kwargs):raise AssertionError('Full/legacy AE or verifier forbidden')
    backend.sample=backend.verify=forbidden
    saved=np.load(OLD/'cross_service_reference.npz')
    try:
        for item in selection()[::10]:
            obs,prompt=load_observation(item['current'])
            suite,seed=item['case']['suite'],int(item['current']['seed'])
            firsts=[]
            for cfg in hold_conditions()+conditions():
                new=cfg.get('gripper_initial')=='standard_gaussian'
                policy=Policy(backend);policy.configure(cfg)
                for i in range(4 if new else 2):
                    result=policy.predict(obs,prompt,suite,seed+i);calls+=1
                    route,timing=result['route'],result['timing']
                    assert route['D']==timing['nodes']['actual_NFE']==1
                    assert not route['full_ae_executed'] and not route['full_action_executed'] and not route['verifier']
                    assert route['residual_refresh']==(i%3==0)
                    expected_start=1. if i==0 else cfg['start_time']
                    assert timing['nodes']['nodes']==[expected_start,0.]
                    history=policy.history
                    ctx,_=backend.prepare_residual(obs,prompt,MASK,history=history[-1]['residuals'],older=history[-2]['residuals'] if len(history)==2 else None,ratio=route['extrapolation_ratio'],reference=True)
                    if i==0:
                        a,r,_=sample_gaussian(backend,ctx,seed+i,1)
                        firsts.append(result['raw'])
                    else:
                        coordinates=DecoderCoordinates(backend.adapter.backend.output_transform,ctx['transformed_state'])
                        cross=max(float(np.abs(coordinates.matrix[6,:6]).max()),float(np.abs(coordinates.matrix[:6,6]).max()),float(np.abs(coordinates.inverse[6,:6]).max()),float(np.abs(coordinates.inverse[:6,6]).max()))
                        assert cross<=1e-7, 'Gripper and motion coordinates not independent'
                        prior=policy.kalman.last_forecast
                        encoded=coordinates.encode_commands(prior,np.zeros((50,32),np.float32))
                        epsilon=np.random.default_rng(seed+i).standard_normal((50,32),dtype=np.float32)
                        expected=mix(encoded,seed+i,cfg['start_time'],1.)
                        if new:
                            original=expected.copy();expected[:,6]=epsilon[:,6]
                            assert np.array_equal(expected[:,:6],original[:,:6]) and np.array_equal(expected[:,7:],original[:,7:])
                            assert np.array_equal(result['arrays']['kalman_initial_prefix'],expected[:5])
                            changed=prior.copy();changed[:,6]=-prior[:,6]+.3
                            alternate=coordinates.encode_commands(changed,np.zeros((50,32),np.float32))
                            assert np.array_equal(make_initial(alternate,seed+i,cfg['start_time'],1.,'standard_gaussian'),expected)
                            assert route['actual_gripper_noise_std']==1. and route['gripper_prior_weight']==0.
                            assert route['gripper_denoising_span']==cfg['start_time']
                            checks.append(dict(name=f'{suite}_{cfg["id"]}_independent_channel_{i}',passed=True,cross_max_abs=cross))
                        a,r,_=sample_initial(backend,ctx,expected,1,cfg['start_time'])
                    raw=r.float().cpu().numpy()
                    checks.append(dict(name=f'{suite}_{cfg["id"]}_reference_{i}',passed=bool(np.allclose(raw,result['raw'],atol=1e-5,rtol=1e-5) and np.allclose(a,result['actions'],atol=1e-5,rtol=1e-5)),max_abs=float(np.abs(raw-result['raw']).max())))
                    arrays[f'{suite}_{cfg["id"]}_round{i}']=result['raw']
                    if not new:
                        old=saved[f'{suite}_{cfg["id"]}_round{i}']
                        historic.append(dict(name=f'{suite}_{cfg["id"]}_round{i}',passed=bool(np.allclose(old,result['raw'],atol=1e-5,rtol=1e-5)),max_abs=float(np.abs(old-result['raw']).max())))
                    policy.acknowledge(result['actions'][:5])
                assert policy.counts['full']==policy.counts['verify']==0
            checks.append(dict(name=suite+'_identical_gaussian_bootstrap',passed=all(np.allclose(firsts[0],a,atol=1e-5,rtol=1e-5) for a in firsts[1:])))
    finally:
        backend.sample,backend.verify=old_sample,old_verify;saved.close()
    np.savez(output/'gate_actions.npz',**arrays)
    outcome=dict(passed=all(c['passed'] for c in checks),checks=checks,policy_calls=calls,historical_gate_passed=all(c['passed'] for c in historic),historical_checks=historic,actual_NFE=1,full_ae_calls=0,verifier_calls=0,native_encoder_graph=False,native_ae_graph=False)
    atomic(output/'preflight.json',outcome)
    if not outcome['passed']:raise RuntimeError('Independent gripper noise mechanism check failed')
    return outcome
