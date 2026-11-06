"""Original-adapter Euler arithmetic and actual AE-depth/time dispatch gates."""
import numpy as np
from .common import SUITES,atomic_json
from .nodes import scheduled_nfe


def run(backend,manifest,output):
    from preparations.idea1_v3_1.benchmark import load_banks
    from .policy import Policy
    from .worker import validate_ae
    records,identities=load_banks([manifest['bank']['path']])
    if identities[0]['sha256']!=manifest['bank']['sha256']:raise ValueError('Depth/time gate input bank changed')
    checks=[];arrays={};torch=backend.torch
    def require(name,value,**details):
        checks.append(dict(name=name,passed=bool(value),**details))
        if not value:
            atomic_json(output/'gate.json',dict(passed=False,checks=checks));raise RuntimeError('Depth/time gate failed: '+name)
    def same(name,a,r,aa,rr):
        require(name,np.allclose(a,aa,atol=1e-5,rtol=1e-5) and torch.allclose(r.float(),rr.float(),atol=1e-5,rtol=1e-5),
            decoded_max_abs=float(np.abs(a-aa).max()),raw_max_abs=float((r.float()-rr.float()).abs().max().cpu()))
    for suite in SUITES:
        row=next(r for r in records if r['suite']==suite);obs,prompt,seed=row['obs'],row['instruction'],int(row['seed'])
        native,_=backend.prepare(obs,prompt);prepared,_=backend.prepare_residual(obs,prompt)
        require(suite+'/full_prefix_KV',all(torch.equal(native[k],prepared[k]) for k in ('K','V')))
        for steps in (10,2,3,4):
            a,r,_=backend.sample_native_reference(native,seed,steps)
            aa,rr,cost=backend.sample(prepared,seed,steps,ae_skip=())
            same(f'{suite}/original_adapter_AE18_D{steps}',a,r,aa,rr)
            fake=dict(timing=cost,route=dict(ae_execution_audit=cost['ae_execution_audit'],ae_bypassed_layers=[]))
            validate_ae(fake,[],steps)
            require(f'{suite}/actual_AE18_and_exact_nodes_D{steps}',scheduled_nfe(cost['nodes'])==steps,audit=cost['ae_execution_audit'])
            arrays[f'{suite}_adapter_d{steps}_raw']=rr.float().cpu().numpy()
        for cfg in manifest['conditions']:
            policy=Policy(backend);policy.configure(cfg);routes=[]
            for index in range(6):
                reply=policy.predict(obs,prompt,suite,seed+index);routes.append(reply['route']);full=cfg['id']=='full_d10' or index%3==0
                nfe=10 if full else cfg['D'];skip=[] if full else cfg['ae_skip'];audit=reply['route']['execution_audit']
                validate_ae(reply,skip,nfe)
                require(cfg['id']+f'/{suite}/round{index}',reply['round']==index and reply['step']==5*index and reply['route']['refresh']==full
                    and scheduled_nfe(reply['timing']['nodes'])==nfe and audit['qkv_layers']==list(range(1,19))
                    and audit['hidden_update_layers']==list(range(1,18)) and audit['kv_only_layers']==[] and audit['patched_layers']==[]
                    and audit['captured_layers']==[] and not audit['grad_enabled'] and audit['selector_calls']==0,
                    actual_AE=reply['route']['ae_execution_audit'])
                policy.acknowledge(reply['actions'][:5])
            require(cfg['id']+'/'+suite+'/schedule',[r['refresh'] for r in routes]==([True]*6 if cfg['id']=='full_d10' else [True,False,False,True,False,False]))
            if cfg['id']!='full_d10':require(cfg['id']+'/'+suite+'/own_Full_timestamps',routes[4]['history_steps']==[0,15]
                and routes[4]['extrapolation_ratio']==1/3 and routes[5]['extrapolation_ratio']==2/3)
            policy.configure(cfg);reset=policy.predict(obs,prompt,suite,seed)
            require(cfg['id']+'/'+suite+'/episode_reset',reset['round']==0 and reset['step']==0 and reset['route']['refresh'] is True)
            policy.acknowledge(reset['actions'][:5]);arrays[suite+'_'+cfg['id']+'_raw']=reply['raw']
    np.savez(output/'gate_actions.npz',**arrays)
    value=dict(passed=True,checks=checks,bank=identities,manifest_hash=manifest['manifest_hash'],atol=1e-5,rtol=1e-5,
        condition_ids=[c['id'] for c in manifest['conditions']],original_adapter_no_AE_bypass_parity_passed=True,
        native_no_AE_bypass_parity_passed=True,actual_AE_visits_and_NFE_passed=True,exact_D2_D3_D4_D10_node_receipts_passed=True,
        scope='Four old-bank observations; original Pi0Adapter uniform Euler schedules D2/D3/D4 and released fusedD10 with AE18 reference; wrapper arithmetic, actual AE visits/NFE, full VLM prefix, period/reset checks. D4 is not claimed as an unmodified released Pi0Inference kernel or independent quality evidence.')
    atomic_json(output/'gate.json',value);return value
