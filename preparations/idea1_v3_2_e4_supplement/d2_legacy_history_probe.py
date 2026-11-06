"""One fixed GPU0 experiment: historical legacy initialization before case0."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
from .native_envs import digest,sha,write_json
from .evaluation import protocol_identity
from .policy_server import native_libero_raw
from .policy_plus_mirror import audit_cpu_backend
from .policy_plus_v14_d2 import array
from . import d2_cross_device_probe as v1
from preparations.idea1_v3_2.transfer_policy import TransferPolicy,RawTritonBackend,json_safe

PARENT='00bc9f0cad3b4e8b3a82a674859493add471fe1dbcd5f69ae69f5acb0d17d65f'
DEADLINE=1792742812
BANK_SHA='2c2e90a3e1fcac3f02226a433c6db9e19a0f3c169deea77707a9a1fbe8996d55'
ORDER=('full_d10','v14_d10','draft_goal','draft_object','draft_spatial','draft_10')


def register(parent_path,output,bank):
    parent=protocol_identity(parent_path)
    if parent['manifest_hash']!=PARENT or parent['deadline_unix']!=DEADLINE:raise ValueError('Exact frozen v1 diagnostic parent/deadline required')
    if time.time()>=DEADLINE:raise TimeoutError('Original overall diagnostic deadline expired')
    if sha(bank)!=BANK_SHA:raise ValueError('Exact historical same-bank bytes required')
    references=[]
    for ordinal,ref in enumerate(parent['references'][:2]):
        if ref['group']!='perturbed350' or ref['source_index']!=0:raise ValueError('Original case0 startup reference required')
        stable=Path(parent_path).parent/'gpu0'/('reference%02d'%ordinal)/'repeat0.npz'
        references.append(dict(ref,current_stable=v1.bound(stable)))
    sources=dict(parent['source_files']);sources[str(Path(__file__).resolve())]=sha(__file__)
    m=dict(schema=1,kind='D2_case0_exact_legacy_initialization_branch_probe',parent_manifest=v1.bound(parent_path),parent_manifest_hash=PARENT,
        deadline_unix=DEADLINE,created_unix=time.time(),required_gpu_index=0,required_gpu_uuid=v1.CARDS[0],
        legacy_bank=v1.bound(bank),legacy_mode_order=list(ORDER),references=references,source_files=sources,
        source_checkpoint_metadata=parent['checkpoint_metadata'],
        model_call_order='Unchanged TransferPolicy.legacy_preflight: each of6 modes reference then candidate; then uncaptured Full,Full,V14,V14; finally captured Full,V14. No D2 or other request inserted.',
        hypothesis='Historical legacy initialization branch changes case0 outputs relative to v1 direct production route.',
        comparisons='Complete raw50x32/decoded50x7 bytewise comparisons to both historical case0 and fixed v1 current outputs; capture bytewise comparison to first uncaptured corresponding mode.',
        quality_requests=0,no_quality_marker=True,no_video=True,no_device_or_configuration_search=True,
        limitations=['V14 does not recreate the19 historical Full requests with evolving RGB: those complete images were not saved.',
            'The legacy gate is run unchanged and its own original numeric criteria are recorded verbatim. Historical/currents reference comparisons remain bitwise; no relaxation or quality release.'])
    m['manifest_hash']=digest(m);output=Path(output);output.mkdir(parents=True,exist_ok=False);write_json(output/'manifest.json',m);return m


def verify(path):
    m=protocol_identity(path)
    if m['kind']!='D2_case0_exact_legacy_initialization_branch_probe' or m['required_gpu_uuid']!=v1.CARDS[0] or m['deadline_unix']!=DEADLINE:raise ValueError('Fixed single-GPU0 branch manifest required')
    for name,expected in m['source_files'].items():
        source=Path(name);source=source if source.is_absolute() else v1.ROOT/source
        if sha(source)!=expected:raise ValueError('Frozen source changed: '+str(source))
    entries=[m['parent_manifest'],m['legacy_bank']]+[r[k] for r in m['references'] for k in ('observation','result','infer','current_stable')]
    for item in entries:
        if sha(item['path'])!=item['sha256']:raise ValueError('Bound evidence changed: '+item['path'])
    return m


def save_output(path,response):
    np.savez_compressed(path,raw=array(response['raw_actions']),decoded=array(response['actions']))


def execute(policy,m,output):
    """One immutable call order; CPU tests substitute only the model object."""
    output=Path(output);prepared={}
    for ref in m['references']:
        with np.load(ref['observation']['path'],allow_pickle=False) as source:
            obs={k:source[k] for k in source.files if k!='instruction'};instruction=str(source['instruction'])
        with Path(ref['infer']['path']).open() as stream:old=json.loads(stream.readline())
        with np.load(ref['current_stable']['path'],allow_pickle=False) as stable:current={key:stable[key].copy() for key in ('raw','decoded')}
        raw=native_libero_raw(obs);seed=old['seed']
        if v1.input_digest(raw,instruction)!=old['response']['input_hash']:raise ValueError('Historical production input mismatch')
        prepared[ref['mode']]=dict(ref=ref,raw=raw,instruction=instruction,seed=seed,old=old,current=current)
    # No model call before this exact unchanged historical initialization path.
    v1.deadline_check(m['deadline_unix']);gate=policy.legacy_preflight(m['legacy_bank']['path'])
    write_json(output/'legacy_gate_verbatim.json',json_safe(gate))
    if not gate['passed'] or [c['mode'] for c in gate['checks']]!=list(ORDER):raise RuntimeError('Unchanged legacy gate failed or order differs')
    checks=[];uncaptured={}
    for mode in ('full_d10','v14_d10'):
        item=prepared[mode]
        for repeat in range(2):
            v1.deadline_check(m['deadline_unix'])
            response=policy.predict(mode,item['raw'],item['instruction'],item['seed'],category='preflight')
            save_output(output/(mode+'_repeat%d.npz'%repeat),response)
            pair={label:dict(raw=v1.comparison(response['raw_actions'],raw),decoded=v1.comparison(response['actions'],decoded))
                for label,raw,decoded in [('historical',item['old']['response']['raw_actions'],item['old']['actions']),('v1_stable',item['current']['raw'],item['current']['decoded'])]}
            check=dict(mode=mode,repeat=repeat,versus=pair,input_hash=response['input_hash'],noise_hash=response['noise_hash'],
                observed_calls=response['observed_calls'],timing=response['timing'],startup_events=response.get('startup_events',[]))
            if repeat==0:uncaptured[mode]=response
            else:check['repeat_vs_first']=dict(raw=v1.comparison(response['raw_actions'],uncaptured[mode]['raw_actions']),decoded=v1.comparison(response['actions'],uncaptured[mode]['actions']))
            checks.append(check);write_json(output/'uncaptured_checks.json',checks)
    captures=[]
    for mode in ('full_d10','v14_d10'):
        item=prepared[mode];v1.deadline_check(m['deadline_unix'])
        with v1.capture_buffers() as (buffers,euler):response=policy.predict(mode,item['raw'],item['instruction'],item['seed'],category='preflight')
        if len(euler)!=10:raise RuntimeError('Expected10 actual captured Euler updates')
        arrays={key:array(value) for key,value in buffers.items()}
        arrays.update({'Euler_latent_%02d'%(index+1):array(value) for index,value in enumerate(euler)})
        arrays.update(raw=array(response['raw_actions']),decoded=array(response['actions']))
        np.savez_compressed(output/(mode+'_captured.npz'),**arrays)
        captures.append(dict(mode=mode,versus_first_uncaptured=dict(raw=v1.comparison(response['raw_actions'],uncaptured[mode]['raw_actions']),decoded=v1.comparison(response['actions'],uncaptured[mode]['actions'])),
                             observed_calls=response['observed_calls']))
    write_json(output/'capture_checks.json',captures)
    return dict(legacy_gate=gate,uncaptured=checks,captures=captures,actual_category_counters=json_safe(policy.counters),quality_requests=0)


def run(a,m):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=v1.CARDS[0]:raise ValueError('Only preregistered GPU0 UUID allowed')
    v1.deadline_check(m['deadline_unix']);output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    receipt=dict(status='CPU_assembly',manifest_hash=m['manifest_hash'],required_gpu_uuid=v1.CARDS[0],pid=os.getpid(),quality_requests=0,no_video=True,no_quality_marker=True,
                 deadline_unix=m['deadline_unix'],source_sha256=sha(__file__))
    write_json(output/'receipt.json',receipt)
    try:
        from preparations.idea1_v3_2.residual import require_idle_gpu
        initial=require_idle_gpu();receipt['initial_idle_guard']=dict(raw=initial,validated=v1.validate_idle_guard(initial,0))
        backend=RawTritonBackend();cpu=audit_cpu_backend(backend)
        if cpu['cuda_initialized_before_guard']:raise RuntimeError('Unexpected CUDA allocation before final guard')
        v1.deadline_check(m['deadline_unix']);final=require_idle_gpu()
        receipt['final_preCUDA_guard']=dict(cpu=cpu,raw=final,validated=v1.validate_idle_guard(final,0));write_json(output/'receipt.json',receipt)
        policy=TransferPolicy(m,m['manifest_hash'],backend=backend)
        result=execute(policy,m,output);receipt.update(status='complete',**result)
        runtime=v1.metadata(backend);receipt['runtime_metadata']={key:value for key,value in runtime.items() if key!='backend_protocol'}
        receipt['loaded_checkpoint_identity']={key:backend.protocol.get(key) for key in ('checkpoint','checkpoint_sha256')}
        receipt['interpretation']='Fixed one-card legacy initialization diagnostic only. Inspect both historical and v1 stable comparisons; no quality release or historical replacement.'
    except Exception:receipt.update(status='technical_failure',error=traceback.format_exc());raise
    finally:receipt['completed_unix']=time.time();write_json(output/'receipt.json',receipt)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('register','run'),required=True);p.add_argument('--output',required=True)
    p.add_argument('--parent-manifest');p.add_argument('--bank',default='results/idea1_v3_1/input_bank.pkl');p.add_argument('--manifest');p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.stage=='register':
        if not a.parent_manifest:p.error('--parent-manifest required')
        m=register(a.parent_manifest,a.output,a.bank);print(json.dumps(dict(manifest_hash=m['manifest_hash'],deadline_unix=DEADLINE,gpu_uuid=v1.CARDS[0],quality_requests=0)));return
    if not a.manifest:p.error('--manifest required')
    m=verify(a.manifest)
    if not a.launch:print(json.dumps(dict(dry_run=True,manifest_hash=m['manifest_hash'],deadline_unix=DEADLINE,quality_requests=0,model_call_order=m['model_call_order'])));return
    run(a,m)


if __name__=='__main__':main()
