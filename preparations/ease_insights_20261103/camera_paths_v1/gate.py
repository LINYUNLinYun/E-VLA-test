"""Native D5/D10 parity, real batch-one vision and a nontrivial mixed-view oracle."""
import numpy as np
from .common import SUITES,atomic_json


def run(backend,manifest,output):
    from preparations.idea1_v3_1.benchmark import load_banks
    from .policy import Policy
    from .worker import validate_ae,validate_camera
    records,identities=load_banks([manifest['bank']['path']])
    if identities[0]['sha256']!=manifest['bank']['sha256']:raise ValueError('Camera gate input bank changed')
    checks=[];arrays={};torch=backend.torch
    def require(name,value,**details):
        checks.append(dict(name=name,passed=bool(value),**details))
        if not value:
            atomic_json(output/'gate.json',dict(passed=False,checks=checks));raise RuntimeError('Camera numerical gate failed: '+name)
    def close(name,a,b):
        require(name,torch.allclose(a.float(),b.float(),atol=1e-5,rtol=1e-5),
            max_abs=float((a.float()-b.float()).abs().max().cpu()))
    def same(name,a,r,aa,rr):
        require(name,np.allclose(a,aa,atol=1e-5,rtol=1e-5) and torch.allclose(r.float(),rr.float(),atol=1e-5,rtol=1e-5),
            decoded_max_abs=float(np.abs(a-aa).max()),raw_max_abs=float((r.float()-rr.float()).abs().max().cpu()))
    def prepare(obs,prompt,request):
        backend.request_vision(request);return backend.prepare_residual(obs,prompt)[0]
    for suite in SUITES:
        row=next(r for r in records if r['suite']==suite)
        old=next(r for r in records if r['instruction']==row['instruction'] and r is not row)
        obs,prompt,seed=row['obs'],row['instruction'],int(row['seed'])
        native,_=backend.prepare(obs,prompt)
        dense=prepare(obs,prompt,dict(kind='refresh'));fresh=backend.last_full_vision.clone()
        for key in ('K','V'):close(suite+'/full_native_prefix_'+key,native[key],dense[key])
        for steps in (10,5):
            a,r,_=backend.sample_native_reference(native,seed,steps)
            aa,rr,cost=backend.sample(dense,seed,steps,ae_skip=())
            same(f'{suite}/native_no_bypass_D{steps}',a,r,aa,rr)
            fake=dict(route=dict(ae_execution_audit=cost['ae_execution_audit'],ae_bypassed_layers=[]),timing=cost)
            validate_ae(fake,[],steps);require(f'{suite}/actual_AE18_NFE{steps}',True,audit=cost['ae_execution_audit'])
            arrays[f'{suite}_native_d{steps}_raw']=rr.float().cpu().numpy()
        for camera in (0,1):
            tag=f'{suite}/camera{camera}'
            single=prepare(obs,prompt,dict(kind='single',camera=camera,cache=fresh,source_step=0,age_steps=5))
            close(tag+'/same_obs_raw_features',single['runtime'].buffers['vision_x'],fresh)
            for key in ('K','V'):close(tag+'/same_obs_prefix_'+key,single[key],dense[key])
            a,r,_=backend.sample(dense,seed,5,ae_skip=())
            aa,rr,cost=backend.sample(single,seed,5,ae_skip=())
            same(tag+'/same_obs_actions',a,r,aa,rr)
            require(tag+'/real_batch_one',cost['camera_execution_audit']['actual_kernel_input_batch']==1
                and cost['camera_execution_audit']['updated_views']==[camera]
                and cost['camera_execution_audit']['original_vision_helper_restored'] is True)
            prepare(old['obs'],prompt,dict(kind='refresh'));old_raw=backend.last_full_vision.clone()
            mixed=fresh.clone();mixed[1-camera].copy_(old_raw[1-camera])
            require(tag+'/nontrivial_held_features',not torch.equal(mixed,fresh),changed_values=int((mixed!=fresh).sum().cpu()))
            oracle=prepare(obs,prompt,dict(kind='reference',raw=mixed))
            require(tag+'/oracle_explicit',oracle['camera_execution_audit']['numerical_reference_only'] is True
                and oracle['camera_execution_audit']['native_vision_invocations']==0)
            current=prepare(obs,prompt,dict(kind='single',camera=camera,cache=old_raw,source_step=0,age_steps=20))
            close(tag+'/nontrivial_mixed_raw_features',current['runtime'].buffers['vision_x'],mixed)
            for key in ('K','V'):close(tag+'/nontrivial_mixed_prefix_'+key,current[key],oracle[key])
            a,r,_=backend.sample(oracle,seed,5,ae_skip=())
            aa,rr,_=backend.sample(current,seed,5,ae_skip=())
            same(tag+'/nontrivial_oracle_actions',a,r,aa,rr)
            arrays[f'{suite}_camera{camera}_mixed_raw']=rr.float().cpu().numpy()
        for cfg in manifest['conditions']:
            policy=Policy(backend);policy.configure(cfg);routes=[]
            for index in range(10):
                reply=policy.predict(obs,prompt,suite,seed+index);routes.append(reply['route'])
                full=cfg['id']=='full_d10' or index%5==0;source=5*index if cfg['id']=='full_d10' else 25*(index//5)
                require(cfg['id']+f'/{suite}/round{index}',reply['round']==index and reply['step']==5*index and reply['route']['refresh']==full)
                validate_ae(reply,[],10 if full else 5);validate_camera(reply,cfg,full,source,5*index-source)
                require(cfg['id']+f'/{suite}/observed_dispatch{index}',True,
                    AE=reply['route']['ae_execution_audit'],camera=reply['route']['camera_execution_audit'])
                policy.acknowledge(reply['actions'][:5])
            require(cfg['id']+'/'+suite+'/schedule',[r['refresh'] for r in routes]==([True]*10 if cfg['id']=='full_d10' else [True,False,False,False,False]*2))
            require(cfg['id']+'/'+suite+'/reset_and_own_history',routes[0]['history_steps']==[0]
                and routes[-1]['history_steps']==([40,45] if cfg['id']=='full_d10' else [0,25]))
            arrays[suite+'_'+cfg['id']+'_raw']=reply['raw']
    backend.reset_vision();np.savez(output/'gate_actions.npz',**arrays)
    value=dict(passed=True,checks=checks,bank=identities,manifest_hash=manifest['manifest_hash'],atol=1e-5,rtol=1e-5,
        condition_ids=[c['id'] for c in manifest['conditions']],native_no_AE_bypass_parity_passed=True,
        actual_AE_visits_and_NFE_passed=True,actual_single_view_batch_passed=True,nontrivial_mixed_view_oracle_passed=True,
        scope='Four old-bank prompts: native D5/D10 no-bypass parity, true tensor batch1, same-input and nontrivial mixed-view oracle parity, own-Full cache schedule; no quality evidence')
    atomic_json(output/'gate.json',value);return value
