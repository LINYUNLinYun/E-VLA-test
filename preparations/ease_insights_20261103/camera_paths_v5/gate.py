"""Real GPU gates for native controls,identity packing and independent masked reference."""
import numpy as np
from .common import SUITES,NEW_CONDITIONS,atomic_json
from .nodes import scheduled_nfe
PARTIAL_ATOL=.0009765625
PARTIAL_RTOL=.0078125


def run(backend,manifest,output):
    from preparations.idea1_v3_1.benchmark import load_banks
    from reports.ease_insights_20261103.prefix_kv_preparation_v1.layout_contract import plan
    from .policy import Policy
    from .worker import validate_ae,validate_camera
    records,identities=load_banks([manifest['bank']['path']])
    if identities[0]['sha256']!=manifest['bank']['sha256']:raise ValueError('Prefix gate bank changed')
    torch=backend.torch;checks=[];arrays={}
    def require(name,value,**details):
        checks.append(dict(name=name,passed=bool(value),**details))
        if not value:
            atomic_json(output/'gate.json',dict(passed=False,checks=checks));raise RuntimeError('Prefix numerical gate failed: '+name)
    def compare(name,a,r,aa,rr,atol=1e-5,rtol=1e-5):
        require(name,np.allclose(a,aa,atol=atol,rtol=rtol) and torch.allclose(r.float(),rr.float(),atol=atol,rtol=rtol),
            decoded_max_abs=float(np.abs(a-aa).max()),raw_max_abs=float((r.float()-rr.float()).abs().max().cpu()),atol=atol,rtol=rtol)
    for suite in SUITES:
        row=next(r for r in records if r['suite']==suite);obs,prompt,seed=row['obs'],row['instruction'],int(row['seed'])
        native,_=backend.prepare(obs,prompt);backend.request_vision(dict(kind='native'));dense,_=backend.prepare_residual(obs,prompt)
        for name in ('K','V'):
            require(suite+'/original_prefix_'+name,torch.allclose(native[name].float(),dense[name].float(),atol=1e-5,rtol=1e-5))
        for nfe in (10,2):
            a,r,_=backend.sample_native_reference(native,seed,nfe);aa,rr,cost=backend.sample(dense,seed,nfe,prefix_pattern='all')
            compare(f'{suite}/native_D{nfe}',a,r,aa,rr)
            cfg=next(c for c in manifest['conditions'] if c['id']==('full_d10' if nfe==10 else 'dense_d2'))
            fake=dict(timing=cost,route=dict(ae_execution_audit=cost['ae_execution_audit'],ae_bypassed_layers=[],camera_execution_audit=cost['camera_execution_audit'],prefix_pattern='all',prefix_execution_audit=cost['prefix_execution_audit']))
            validate_ae(fake,[],nfe);validate_camera(fake,cfg,nfe==10)
            require(f'{suite}/native_nodes_D{nfe}',scheduled_nfe(cost['nodes'])==nfe)
            aaa,rrr,identity=backend.sample(dense,seed,nfe,prefix_pattern='identity');compare(f'{suite}/identity_packing_D{nfe}',a,r,aaa,rrr)
            audit=identity['prefix_execution_audit'];require(f'{suite}/identity_actual_storage_D{nfe}',audit['actual_prefix_length']==dense['seq'] and audit['actual_helper_calls']==18*nfe and audit['original_buffers_restored'] and audit['original_helper_restored'])
            arrays[f'{suite}_native_d{nfe}_raw']=rr.float().cpu().numpy()
        for cfg in manifest['conditions']:
            cid=cfg['id'];pattern=cfg['prefix_pattern']
            if cid in NEW_CONDITIONS:
                layout=plan(pattern,dense['seq']);rope=dense['runtime'].buffers['decoder_rope_weights']
                with backend.prefix_packer.packed(dense,pattern) as (packed,audit):
                    index=torch.tensor(layout.keep_indices,device=dense['K'].device)
                    require(suite+'/'+cid+'/exact_KV_gather',torch.equal(packed['K'],dense['K'].index_select(1,index)) and torch.equal(packed['V'],dense['V'].index_select(1,index)))
                    require(suite+'/'+cid+'/original_absolute_RoPE',packed['runtime'].buffers['decoder_rope_weights'] is rope)
                a,r,cost=backend.sample(dense,seed,2,prefix_pattern=pattern)
                aa,rr,_,oracle=backend.sample_masked_reference(dense,seed,pattern)
                compare(suite+'/'+cid+'/full_storage_masked_reference',a,r,aa,rr,PARTIAL_ATOL,PARTIAL_RTOL)
                require(suite+'/'+cid+'/gripper_reference_identical',np.array_equal(a[:,6],aa[:,6]))
                require(suite+'/'+cid+'/reference_only_full_work',oracle['reference_only'] and oracle['original_helper_restored'] and len(oracle['calls'])==36
                    and all(v['actual_QK_key_tokens']==dense['seq']+51 and v['masked_original_prefix_tokens']==256 for v in oracle['calls']))
                audit=cost['prefix_execution_audit'];require(suite+'/'+cid+'/real_compact_keys',audit['actual_prefix_length']==dense['seq']-256 and audit['actual_total_key_tokens']==dense['seq']-205 and audit['visual_tokens_per_camera']==[128,128])
            policy=Policy(backend);policy.configure(cfg);routes=[]
            for i in range(6):
                reply=policy.predict(obs,prompt,suite,seed+i);full=cid=='full_d10' or i%3==0;routes.append(reply['route']['refresh'])
                require(f'{suite}/{cid}/round{i}',reply['round']==i and reply['step']==5*i and reply['route']['refresh']==full)
                validate_ae(reply,[],10 if full else 2);validate_camera(reply,cfg,full);policy.acknowledge(reply['actions'][:5])
            require(suite+'/'+cid+'/cadence',routes==([True]*6 if cid=='full_d10' else [True,False,False]*2))
            arrays[f'{suite}_{cid}_raw']=reply['raw'];policy.configure(cfg);reset=policy.predict(obs,prompt,suite,seed)
            require(suite+'/'+cid+'/reset',reset['round']==0 and reset['step']==0 and reset['route']['refresh']);policy.acknowledge(reset['actions'][:5])
    np.savez(output/'gate_actions.npz',**arrays)
    value=dict(passed=True,checks=checks,bank=identities,manifest_hash=manifest['manifest_hash'],atol=1e-5,rtol=1e-5,
        partial_reference_atol=PARTIAL_ATOL,partial_reference_rtol=PARTIAL_RTOL,partial_gripper_exact=True,
        condition_ids=[c['id'] for c in manifest['conditions']],original_vision_parity_passed=True,native_no_AE_bypass_parity_passed=True,
        actual_AE_visits_and_NFE_passed=True,packed_prefix_identity_passed=True,independent_masked_prefix_reference_passed=True,
        actual_compact_prefix_and_original_RoPE_passed=True,
        scope='Four fixed real bank inputs.Native and identity packing1e-5;approximate BF16 compact versus independently full-storage masked reference at predeclared1/1024absolute and1/128relative,all grippers exact.Original controls require bitwise closed-loop bridge.No new quality evidence.')
    atomic_json(output/'gate.json',value);return value
