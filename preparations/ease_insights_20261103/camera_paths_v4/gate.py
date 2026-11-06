"""Real old-bank GPU checks: B2 identity,B1 allocation/oracle,native NFE and cadence."""
import numpy as np
from .common import SUITES,FULL,NEW_CONDITIONS,atomic_json
from .nodes import scheduled_nfe


def run(backend,manifest,output):
    from preparations.idea1_v3_1.benchmark import load_banks
    from .policy import Policy
    from .worker import validate_ae,validate_camera
    records,identities=load_banks([manifest['bank']['path']])
    if identities[0]['sha256']!=manifest['bank']['sha256']:raise ValueError('Fresh-vision gate bank changed')
    torch=backend.torch;checks=[];arrays={}
    def require(name,value,**details):
        checks.append(dict(name=name,passed=bool(value),**details))
        if not value:
            atomic_json(output/'gate.json',dict(passed=False,checks=checks));raise RuntimeError('Fresh vision numerical gate failed: '+name)
    def close(name,a,b):
        require(name,torch.allclose(a.float(),b.float(),atol=1e-5,rtol=1e-5),max_abs=float((a.float()-b.float()).abs().max().cpu()))
    def same(name,a,r,aa,rr):
        require(name,np.allclose(a,aa,atol=1e-5,rtol=1e-5) and torch.allclose(r.float(),rr.float(),atol=1e-5,rtol=1e-5),
            decoded_max_abs=float(np.abs(a-aa).max()),raw_max_abs=float((r.float()-rr.float()).abs().max().cpu()))
    def prepare(obs,prompt,request):
        backend.request_vision(request);c,_=backend.prepare_residual(obs,prompt)
        return c,c['runtime'].buffers['vision_x'].clone()
    for suite in SUITES:
        row=next(r for r in records if r['suite']==suite);other=next(r for r in records if r['instruction']==row['instruction'] and r is not row)
        obs,prompt,seed=row['obs'],row['instruction'],int(row['seed']);native,_=backend.prepare(obs,prompt)
        dense,full_raw=prepare(obs,prompt,dict(kind='native'))
        for k in ('K','V'):close(suite+'/native_full_prefix_'+k,native[k],dense[k])
        identity,identity_raw=prepare(obs,prompt,dict(kind='identity_b1'))
        close(suite+'/B1_identity_vision',full_raw,identity_raw)
        for k in ('K','V'):close(suite+'/B1_identity_prefix_'+k,dense[k],identity[k])
        for steps in (10,2):
            a,r,_=backend.sample_native_reference(native,seed,steps)
            aa,rr,cost=backend.sample(dense,seed,steps,ae_skip=());same(f'{suite}/native_no_bypass_D{steps}',a,r,aa,rr)
            aaa,rrr,_=backend.sample(identity,seed,steps,ae_skip=());same(f'{suite}/B1_identity_actions_D{steps}',a,r,aaa,rrr)
            require(f'{suite}/exact_nodes_D{steps}',scheduled_nfe(cost['nodes'])==steps and cost['actual_NFE']==steps)
            fake=dict(route=dict(ae_execution_audit=cost['ae_execution_audit'],ae_bypassed_layers=[]),timing=cost);validate_ae(fake,[],steps)
            arrays[f'{suite}_native_d{steps}_raw']=rr.float().cpu().numpy()
        for cfg in manifest['conditions']:
            cid=cfg['id']
            if cid in NEW_CONDITIONS:
                current,raw=prepare(obs,prompt,dict(kind='allocated',layers=cfg['vision_kept']))
                oracle,oracle_raw=prepare(obs,prompt,dict(kind='oracle',layers=cfg['vision_kept']))
                close(suite+'/'+cid+'/partial_vision_B2_oracle',raw,oracle_raw)
                for k in ('K','V'):close(suite+'/'+cid+'/partial_prefix_B2_oracle_'+k,current[k],oracle[k])
                a,r,_=backend.sample(current,seed,2,ae_skip=());aa,rr,_=backend.sample(oracle,seed,2,ae_skip=())
                same(suite+'/'+cid+'/partial_action_B2_oracle',a,r,aa,rr)
                audit=current['camera_execution_audit'];require(suite+'/'+cid+'/real_B1_layers',audit['actual_kernel_input_batches']==[1,1]
                    and audit['actual_vision_layers_by_camera']==cfg['vision_kept'] and audit['image_layer_evaluations']==45 and audit['fresh_images_consumed'])
                _,changed=prepare(other['obs'],prompt,dict(kind='allocated',layers=cfg['vision_kept']))
                require(suite+'/'+cid+'/both_fresh_nontrivial',all(not torch.equal(raw[v],changed[v]) for v in (0,1)))
                require(suite+'/'+cid+'/oracle_not_production',oracle['camera_execution_audit']['numerical_reference_only'] is True
                    and oracle['camera_execution_audit']['image_layer_evaluations']==54)
            policy=Policy(backend);policy.configure(cfg);routes=[]
            for index in range(6):
                reply=policy.predict(obs,prompt,suite,seed+index);routes.append(reply['route']);full=cid=='full_d10' or index%3==0
                require(f'{suite}/{cid}/round{index}',reply['round']==index and reply['step']==5*index and reply['route']['refresh']==full)
                validate_ae(reply,[],10 if full else 2);validate_camera(reply,cfg,full)
                require(f'{suite}/{cid}/actual_dispatch{index}',scheduled_nfe(reply['timing']['nodes'])==(10 if full else 2),
                    camera=reply['route']['camera_execution_audit'],AE=reply['route']['ae_execution_audit'])
                policy.acknowledge(reply['actions'][:5])
            require(f'{suite}/{cid}/cadence',[r['refresh'] for r in routes]==([True]*6 if cid=='full_d10' else [True,False,False,True,False,False]))
            policy.configure(cfg);reset=policy.predict(obs,prompt,suite,seed);require(f'{suite}/{cid}/reset',reset['round']==0 and reset['step']==0 and reset['route']['refresh']);policy.acknowledge(reset['actions'][:5])
            arrays[f'{suite}_{cid}_raw']=reply['raw']
    backend.reset_vision();np.savez(output/'gate_actions.npz',**arrays)
    value=dict(passed=True,checks=checks,bank=identities,manifest_hash=manifest['manifest_hash'],atol=1e-5,rtol=1e-5,
        condition_ids=[c['id'] for c in manifest['conditions']],native_no_AE_bypass_parity_passed=True,actual_AE_visits_and_NFE_passed=True,
        GPU_original_vision_parity_passed=True,partial_vision_oracle_passed=True,actual_vision_layers_and_B1_passed=True,
        scope='Four real old-bank observations: nativeD2/D10 and both-B1 identity,partial-view independentB2restoreoracle,actual freshvision layers and no-cache cadence;not new quality evidence')
    atomic_json(output/'gate.json',value);return value
