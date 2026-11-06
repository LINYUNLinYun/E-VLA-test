"""Real-GPU native arithmetic, VLM residual and observed AE-dispatch gates."""
import numpy as np
from .common import SUITES,atomic_json,validate_condition


def run(backend,manifest,output):
    from preparations.idea1_v3_1.benchmark import load_banks
    from .policy import Policy
    from .worker import validate_ae
    records,identities=load_banks([manifest['bank']['path']])
    if identities[0]['sha256']!=manifest['bank']['sha256']:raise ValueError('Module gate input bank changed')
    checks=[];arrays={};torch=backend.torch
    def require(name,value,**details):
        checks.append(dict(name=name,passed=bool(value),**details))
        if not value:
            atomic_json(output/'gate.json',dict(passed=False,checks=checks));raise RuntimeError('Module numerical gate failed: '+name)
    def same(name,a,r,aa,rr):
        require(name,np.allclose(a,aa,atol=1e-5,rtol=1e-5) and torch.allclose(r.float(),rr.float(),atol=1e-5,rtol=1e-5),
            decoded_max_abs=float(np.abs(a-aa).max()),raw_max_abs=float((r.float()-rr.float()).abs().max().cpu()))
    for suite in SUITES:
        row=next(r for r in records if r['suite']==suite)
        obs,prompt,seed=row['obs'],row['instruction'],int(row['seed'])
        native,_=backend.prepare(obs,prompt)
        prepared,_=backend.prepare_residual(obs,prompt)
        for steps in (10,2):
            a,r,_=backend.sample_native_reference(native,seed,steps)
            aa,rr,cost=backend.sample(prepared,seed,steps,ae_skip=())
            same(f'{suite}/native_no_bypass_D{steps}',a,r,aa,rr)
            require(f'{suite}/native_AE18_NFE{steps}',cost['actual_NFE']==steps
                and cost['ae_execution_audit']['actual_layer_calls']==18*steps
                and cost['ae_execution_audit']['executed_layers_per_step']==list(range(1,19)))
            arrays[f'{suite}_native_d{steps}_raw']=rr.float().cpu().numpy()
        for cfg in manifest['conditions']:
            cid=cfg['id'];vlm,ae=validate_condition(cfg);tag=suite+'/'+cid
            if cid!='full_d10':
                full,_=backend.prepare_residual(obs,prompt,capture_layers=vlm,traced=vlm)
                audit=full['execution_audit']
                require(tag+'/full_capture',audit['qkv_layers']==list(range(1,19)) and audit['hidden_update_layers']==list(range(1,18))
                    and audit['captured_layers']==list(vlm) and not audit['grad_enabled'] and audit['selector_calls']==0)
                expected={l:full['traces'][l]['output'].float()-full['traces'][l]['input'].float() for l in vlm}
                require(tag+'/residual_identity',set(full['residuals'])==set(vlm)
                    and all(torch.equal(full['residuals'][l],expected[l]) for l in vlm))
                rebuilt,_=backend.prepare_residual(obs,prompt,skip=vlm,history=full['residuals'])
                a,r,_=backend.sample(full,seed,2,ae_skip=ae)
                aa,rr,_=backend.sample(rebuilt,seed,2,ae_skip=ae)
                same(tag+'/same_observation_residual_reconstruction',a,r,aa,rr)
                require(tag+'/prefix_KV_reconstruction',all(torch.equal(full[k],rebuilt[k]) for k in ('K','V')))
                actual=rebuilt['execution_audit']
                require(tag+'/actual_VLM_dispatch',actual['kv_only_layers']==list(vlm)
                    and actual['qkv_layers']==[l for l in range(1,19) if l not in vlm]
                    and actual['hidden_update_layers']==[l for l in range(1,18) if l not in vlm]
                    and actual['patched_layers']==list(vlm) and actual['captured_layers']==[])
                pure,_=backend.prepare(obs,prompt,vlm)
                residual,_=backend.prepare_residual(obs,prompt,skip=vlm)
                a,r,_=backend.sample(pure,seed,2,ae_skip=ae)
                aa,rr,_=backend.sample(residual,seed,2,ae_skip=ae)
                same(tag+'/pure_VLM_prefix',a,r,aa,rr)
                earlier=next(r for r in records if r['instruction']==prompt and r is not row)
                old,_=backend.prepare_residual(earlier['obs'],prompt,capture_layers=vlm)
                ref,_=backend.prepare_residual(obs,prompt,skip=vlm,history=full['residuals'],older=old['residuals'],ratio=2/3,reference=True)
                current,_=backend.prepare_residual(obs,prompt,skip=vlm,history=full['residuals'],older=old['residuals'],ratio=2/3)
                a,r,_=backend.sample(ref,seed,2,ae_skip=ae)
                aa,rr,_=backend.sample(current,seed,2,ae_skip=ae)
                same(tag+'/nonzero_residual_reference',a,r,aa,rr)
            policy=Policy(backend);policy.configure(cfg);routes=[]
            for index in range(6):
                reply=policy.predict(obs,prompt,suite,seed+index);routes.append(reply['route'])
                full=cid=='full_d10' or index%3==0
                require(tag+f'/policy_round{index}',reply['round']==index and reply['step']==5*index and reply['route']['refresh']==full)
                validate_ae(reply,[] if full else cfg['ae_skip'],10 if full else 2)
                require(tag+f'/observed_AE_round{index}',True,actual_AE=reply['route']['ae_execution_audit'])
                policy.acknowledge(reply['actions'][:5])
            require(tag+'/schedule',[r['refresh'] for r in routes]==([True]*6 if cid=='full_d10' else [True,False,False,True,False,False]))
            if cid!='full_d10':
                require(tag+'/own_path_history',routes[4]['history_steps']==[0,15]
                    and routes[4]['extrapolation_ratio']==1/3 and routes[5]['extrapolation_ratio']==2/3)
            arrays[suite+'_'+cid+'_raw']=reply['raw']
    np.savez(output/'gate_actions.npz',**arrays)
    value=dict(passed=True,checks=checks,bank=identities,manifest_hash=manifest['manifest_hash'],atol=1e-5,rtol=1e-5,
        condition_ids=[c['id'] for c in manifest['conditions']],native_no_AE_bypass_parity_passed=True,
        actual_AE_visits_and_NFE_passed=True,
        scope='Four real old-bank observations, native no-AE-bypass D10/D2 parity, VLM residual arithmetic and observed AE dispatch; no closed-loop quality claim')
    atomic_json(output/'gate.json',value);return value
