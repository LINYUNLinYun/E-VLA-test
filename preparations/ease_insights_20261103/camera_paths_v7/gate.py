"""Finite real-bank native identity and full-compute/restore component oracle."""
import numpy as np
from .common import SUITES,NEW_CONDITIONS,FIXED_LAYERS,atomic_json
from .nodes import scheduled_nfe


def run(backend,manifest,output):
    from preparations.idea1_v3_1.benchmark import load_banks
    from .policy import Policy
    from .worker import validate_ae,validate_camera,validate_components
    records,identities=load_banks([manifest['bank']['path']])
    if identities[0]['sha256']!=manifest['bank']['sha256']:raise ValueError('Component gate bank changed')
    torch=backend.torch;checks=[];arrays={}
    def require(name,value,**details):
        checks.append(dict(name=name,passed=bool(value),**details))
        if not value:
            atomic_json(output/'gate.json',dict(passed=False,checks=checks))
            raise RuntimeError('Component numerical gate failed: '+name)
    def compare(name,a,r,aa,rr):
        require(name,np.allclose(a,aa,atol=1e-5,rtol=1e-5) and torch.allclose(r.float(),rr.float(),atol=1e-5,rtol=1e-5),
            decoded_max_abs=float(np.abs(a-aa).max()),raw_max_abs=float((r.float()-rr.float()).abs().max().cpu()),atol=1e-5,rtol=1e-5)
    def prefix(name,a,b):
        for key in ('K','V'):
            require(name+'/'+key,torch.allclose(a[key].float(),b[key].float(),atol=1e-5,rtol=1e-5))
    for suite in SUITES:
        row=next(r for r in records if r['suite']==suite);obs,prompt,seed=row['obs'],row['instruction'],int(row['seed'])
        native,_=backend.prepare(obs,prompt)
        backend.request_vision(dict(kind='native'));dense,_=backend.prepare_components(obs,prompt)
        prefix(suite+'/native_prefix',native,dense)
        for nfe in (10,2):
            a,r,_=backend.sample_native_reference(native,seed,nfe);aa,rr,cost=backend.sample(dense,seed,nfe)
            compare(f'{suite}/native_D{nfe}',a,r,aa,rr)
            require(f'{suite}/native_nodes_D{nfe}',scheduled_nfe(cost['nodes'])==nfe)
            arrays[f'{suite}_native_d{nfe}_raw']=rr.float().cpu().numpy()
        for cfg in manifest['conditions']:
            cid=cfg['id'];kind=cfg['component']
            if cid in NEW_CONDITIONS:
                backend.request_vision(dict(kind='native'));captured,_=backend.prepare_components(obs,prompt,capture=kind)
                prefix(suite+'/'+cid+'/capture_identity',dense,captured)
                a,r,_=backend.sample(dense,seed,10);aa,rr,_=backend.sample(captured,seed,10)
                compare(suite+'/'+cid+'/capture_D10_identity',a,r,aa,rr)
                latest=dict(identity=captured['module_identity'],residuals={kind:{l:captured['residuals'][kind][l] for l in FIXED_LAYERS}})
                # Algebra-only stress fixture, not a second physical observation
                # or a quality result. Nonzero history delta tests extrapolation.
                older=dict(identity=captured['module_identity'],residuals={kind:{l:(r*.98).contiguous() for l,r in latest['residuals'][kind].items()}})
                for label,previous,ratio in [('carry',None,0.),('extrap_first',older,1/3),('extrap_second',older,2/3)]:
                    kwargs=dict(**{'skip_'+kind:FIXED_LAYERS},history=latest,older=previous,ratio=ratio)
                    backend.request_vision(dict(kind='native'));prod,_=backend.prepare_components(obs,prompt,**kwargs)
                    a,r,cost=backend.sample(prod,seed,2)
                    backend.request_vision(dict(kind='native'));ref,_=backend.prepare_components(obs,prompt,oracle=True,**kwargs)
                    aa,rr,reference_cost=backend.sample(ref,seed,2)
                    name=suite+'/'+cid+'/'+label
                    prefix(name+'/full_compute_restore_prefix',prod,ref)
                    compare(name+'/full_compute_restore_actions',a,r,aa,rr)
                    require(name+'/gripper_exact',np.array_equal(a[:,6],aa[:,6]))
                    oracle=reference_cost['component_execution_audit']
                    require(name+'/oracle_full_compute_only',reference_cost['component_reference_only'] and oracle['reference_only']
                        and oracle['full_compute_calls']=={'qkv':list(range(1,19)),'attn':list(range(1,18)),'ffn':list(range(1,18))}
                        and oracle['restored_and_patched']=={k:list(FIXED_LAYERS) if k==kind else [] for k in ('attn','ffn')})
                    arrays[name.replace('/','_')+'_raw']=r.float().cpu().numpy()
            policy=Policy(backend);policy.configure(cfg);routes=[]
            for i in range(6):
                reply=policy.predict(obs,prompt,suite,seed+i);full=cid=='full_d10' or i%3==0
                routes.append(reply['route']['refresh'])
                require(f'{suite}/{cid}/round{i}',reply['round']==i and reply['step']==5*i and reply['route']['refresh']==full)
                validate_ae(reply,[],10 if full else 2);validate_camera(reply,cfg,full);validate_components(reply,cfg,full)
                policy.acknowledge(reply['actions'][:5])
            require(suite+'/'+cid+'/cadence',routes==([True]*6 if cid=='full_d10' else [True,False,False]*2))
            arrays[f'{suite}_{cid}_raw']=reply['raw'];policy.configure(cfg)
            reset=policy.predict(obs,prompt,suite,seed)
            require(suite+'/'+cid+'/reset',reset['round']==0 and reset['step']==0 and reset['route']['refresh'])
            policy.acknowledge(reset['actions'][:5])
    np.savez(output/'gate_actions.npz',**arrays)
    value=dict(passed=True,checks=checks,bank=identities,manifest_hash=manifest['manifest_hash'],atol=1e-5,rtol=1e-5,
        condition_ids=[c['id'] for c in manifest['conditions']],original_vision_parity_passed=True,
        native_no_AE_bypass_parity_passed=True,actual_AE_visits_and_NFE_passed=True,
        component_native_identity_passed=True,independent_full_compute_restore_reference_passed=True,
        actual_component_routes_and_history_passed=True,
        scope='Four fixed real bank observations. Native/capture identity and physically bypassed production versus all-compute/restore Torch-patch oracle at1e-5;all grippers exact. Synthetic older bank=.98*latest exercises nonzero extrapolation only. No independent quality evidence;8known control trajectories require bitwise bridge.')
    atomic_json(output/'gate.json',value);return value
