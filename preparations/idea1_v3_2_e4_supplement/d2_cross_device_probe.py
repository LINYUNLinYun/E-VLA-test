"""Fixed two-device, historical production-path bitwise diagnostic. No quality."""
import argparse
from contextlib import contextmanager
import importlib.metadata
import json
import os
import sys
from pathlib import Path
import time
import traceback
import numpy as np
from .native_envs import ROOT,digest,sha,write_json
from .evaluation import protocol_identity
from .policy_server import native_libero_raw
from .policy_plus_mirror import audit_cpu_backend
from .policy_plus_v14_d2 import V14D2Policy,array
from preparations.idea1_v3_2.transfer_policy import TransferPolicy,RawTritonBackend,route_observations,observed_counts
from preparations.idea1_v2_1.runtime_client import input_digest,value_digest

CARDS={0:'GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1',1:'GPU-599477fb-0507-40dc-14df-d491f7a7b202'}
ENV_ALLOWLIST=('CUDA_VISIBLE_DEVICES','JAX_PLATFORMS','NVIDIA_TF32_OVERRIDE','CUBLAS_WORKSPACE_CONFIG',
    'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE','PYTORCH_CUDA_ALLOC_CONF','CUDA_MODULE_LOADING',
    'SPEC_TRITON_DRAFT_GRAPH','SPEC_TRITON_VERIFY_EXACT_INPUT','SPEC_TRITON_INPUT_PREPARE_FAST','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR')


def bound(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))


def validate_idle_guard(guard,card):
    fields=[field.strip() for field in guard['device'].split(',')]
    if len(fields)!=5:raise ValueError('Unrecognized assigned-device identity')
    memory=float(fields[3].removesuffix('MiB').strip());utilization=float(fields[4].removesuffix('%').strip())
    if int(fields[0])!=card or fields[1]!=CARDS[card] or guard['preexisting_compute_processes'] or memory>16 or utilization!=0:
        raise RuntimeError('Assigned physical device is not exactly idle: '+str(guard))
    return dict(physical_index=int(fields[0]),uuid=fields[1],memory_mib=memory,utilization_percent=utilization,passed=True)


def deadline_check(deadline):
    if time.time()>=deadline:raise TimeoutError('Probe deadline reached; no additional calls')


def select_jobs(jobs):
    chosen=[]
    for group in ('perturbed350','original40'):
        for suite in sorted({j['case']['mother_suite'] for j in jobs if j['group']==group}):
            chosen.append(min((j for j in jobs if j['group']==group and j['case']['mother_suite']==suite),key=lambda j:j['source_index']))
    return sorted(chosen,key=lambda j:(j['group']!='perturbed350',j['source_index']))


def comparison(actual,expected):
    actual,expected=array(actual),array(expected)
    if actual.shape!=expected.shape or actual.ndim!=2 or not np.isfinite(actual).all() or not np.isfinite(expected).all():raise ValueError('Invalid diagnostic output')
    difference=np.abs(actual-expected);index=np.unravel_index(np.argmax(difference),difference.shape)
    # All model outputs here originate in BF16 or float32. Preserve signed-zero
    # bits too: numerical array_equal alone is not a bitwise assertion.
    bitwise=actual.dtype==expected.dtype and actual.tobytes()==expected.tobytes()
    return dict(bitwise_equal=bitwise,max_abs=float(difference.max()),argmax=list(map(int,index)),
        actual_at_argmax=float(actual[index]),expected_at_argmax=float(expected[index]),
        per_channel_max_abs=difference.max(axis=0).tolist(),active_first7_max_abs=float(difference[:,:7].max()),
        padding_after7_max_abs=float(difference[:,7:].max()) if difference.shape[1]>7 else None)


def register(parent_path,output,deadline):
    parent=protocol_identity(parent_path)
    if parent['manifest_hash']!='242190d617125f75f960a70a12cbeb3b53b7fd4227e6a93dbe1c6ffbdeca7479':raise ValueError('Exact original D2 registration required')
    if not time.time()<deadline<=time.time()+3600:raise ValueError('Fresh diagnostic deadline at most60min required')
    refs=[]
    for job in select_jobs(parent['jobs']):
        for mode in ('full_d10','v14_d10'):
            old=next(r for r in job['paired_results'] if r['mode']==mode);base=Path(old['path']).parent
            if sha(old['path'])!=old['sha256']:raise ValueError('Historical result changed')
            refs.append(dict(group=job['group'],source_index=job['source_index'],mother_suite=job['case']['mother_suite'],case_hash=job['case_hash'],mode=mode,
                result=bound(old['path']),observation=bound(base/'initial_native_observation.npz'),infer=bound(base/'infer.jsonl')))
    if len(refs)!=16:raise ValueError('Exactly8cases/16references required')
    startup=[r for r in refs if r['group']=='perturbed350' and r['source_index']==0]
    if len(startup)!=2:raise ValueError('Two original startup references must be included')
    for r in refs:r['capture']=r in startup
    sources=dict(parent['source_files']);sources[str(Path(__file__).resolve())]=sha(__file__)
    cpu_evidence=ROOT/'results/idea1_v3_2_e4_supplement/draft_diagnostics_20261023/d2_cpu_input_precision_audit.json'
    m=dict(schema=1,kind='D2_historical_production_cross_device_bitwise_probe',parent_manifest=bound(parent_path),
        original_half_manifest=parent['half_manifest'],original_ledger=parent['audit_ledger'],
        checkpoint_metadata=parent['checkpoint_metadata'],references=refs,repeat_count=2,cards=CARDS,primary_card=0,replication_card=1,
        deadline_unix=deadline,created_unix=time.time(),source_files=sources,cpu_input_evidence=bound(cpu_evidence),
        selection='Outcome-blind minimum frozen source_index within each mother suite separately for perturbed and original; FullD10 and V14D10 only',
        gate='Exact bitwise equality of all raw50x32 and decoded50x7 values to historical and repeat outputs; no padding/future-step exemption, no tolerance',
        production_path='TransferPolicy.predict(category=preflight) -> RawTritonBackend.prepare(native_libero_raw(original obs):float32 state) -> sample(D10)',
        capture='Separate extra pass for two original startup references only; true normalized input buffers, final vision_x, transformer hidden_trace[0] projected prefix, K/V, Euler diffusion_noise after every update. Capture outputs must equal uninstrumented outputs bitwise.',
        quality_requests=0,no_quality_marker=True,no_video=True,no_automatic_card_or_configuration_search=True)
    m['manifest_hash']=digest(m);output=Path(output);output.mkdir(parents=True,exist_ok=False);write_json(output/'manifest.json',m);return m


def verify(path):
    m=protocol_identity(path)
    if m['kind']!='D2_historical_production_cross_device_bitwise_probe':raise ValueError('Diagnostic-only manifest required')
    for name,expected in m['source_files'].items():
        path=Path(name);path=path if path.is_absolute() else ROOT/path
        if sha(path)!=expected:raise ValueError('Frozen source changed: '+str(path))
    evidence=[m['parent_manifest'],m['original_half_manifest'],m['original_ledger'],m['cpu_input_evidence']]
    evidence += [ref[key] for ref in m['references'] for key in ('result','observation','infer')]
    for item in evidence:
        if sha(item['path'])!=item['sha256']:raise ValueError('Diagnostic evidence changed: '+item['path'])
    return m


@contextmanager
def capture_buffers():
    """Request-local observation only; restored on every exit, no kernel edits."""
    from scripts.spec.triton import pi0_infer as kernels
    saved={name:getattr(kernels,name) for name in ('vision_encoder','transformer_encoder','rms_matmul_k_1024_32_bias_res')}
    captured={};euler=[]
    def vision(weights,buffers,*args,**kwargs):
        for key in ('observation_images_normalized','observation_state_normalized'):captured[key]=buffers[key].detach().clone()
        result=saved['vision_encoder'](weights,buffers,*args,**kwargs)
        captured['vision_final_block_output']=buffers['vision_x'].detach().clone();return result
    def vlm(weights,buffers,*args,**kwargs):
        trace=[];kwargs['hidden_trace']=trace
        result=saved['transformer_encoder'](weights,buffers,*args,**kwargs)
        captured['projected_prefix_H0']=trace[0][:512].detach().clone()
        captured['vlm_encoder_K']=buffers['encoder_K'][:,:args[0]].detach().clone()
        captured['vlm_encoder_V']=buffers['encoder_V'][:,:args[0]].detach().clone()
        return result
    def update(x,weight,bias,out,x_norm_factor):
        if not euler:captured['initial_noise_BF16']=out.detach().clone()
        result=saved['rms_matmul_k_1024_32_bias_res'](x,weight,bias,out,x_norm_factor)
        euler.append(out.detach().clone());return result
    kernels.vision_encoder=vision;kernels.transformer_encoder=vlm;kernels.rms_matmul_k_1024_32_bias_res=update
    try:yield captured,euler
    finally:
        for name,original in saved.items():setattr(kernels,name,original)


def metadata(backend):
    torch=backend.torch
    versions={name:importlib.metadata.version(name) for name in ('torch','triton','numpy','Pillow')}
    return dict(versions=versions,sys_executable=sys.executable,cuda_version=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),
        device_name=torch.cuda.get_device_name(0),compute_capability=list(torch.cuda.get_device_capability(0)),
        allowlisted_environment={k:os.environ.get(k) for k in ENV_ALLOWLIST},
        precision=dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled()),backend_protocol=backend.protocol)


def run(a,m):
    if a.card not in CARDS or os.environ.get('CUDA_VISIBLE_DEVICES')!=CARDS[a.card]:raise ValueError('Explicit preregistered GPU0/1 UUID required; no fallback')
    if not time.time()<a.deadline_unix<=m['deadline_unix']:raise ValueError('Fresh frozen60min deadline required')
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    receipt=dict(status='CPU_assembly',manifest_hash=m['manifest_hash'],card=a.card,gpu_uuid=CARDS[a.card],pid=os.getpid(),
        deadline_unix=a.deadline_unix,quality_requests=0,no_video=True,no_quality_marker=True,checks=[],source_sha256=sha(__file__))
    write_json(output/'receipt.json',receipt)
    try:
        from preparations.idea1_v3_2.residual import require_idle_gpu
        receipt['initial_idle_guard']=require_idle_gpu();receipt['initial_idle_validation']=validate_idle_guard(receipt['initial_idle_guard'],a.card)
        backend=RawTritonBackend();cpu=audit_cpu_backend(backend)
        if cpu['cuda_initialized_before_guard'] or time.time()>=a.deadline_unix:raise RuntimeError('Unexpected CUDA initialization or elapsed deadline before allocation')
        final=require_idle_gpu()
        final_validation=validate_idle_guard(final,a.card)
        receipt['final_preCUDA_guard']=dict(cpu=cpu,gpu=final,idle_validation=final_validation)
        policy=TransferPolicy(m,m['manifest_hash'],backend=backend)
        for ordinal,ref in enumerate(m['references']):
            if time.time()>=a.deadline_unix:raise TimeoutError('Probe deadline reached; no additional calls')
            directory=output/('reference%02d'%ordinal);directory.mkdir()
            with np.load(ref['observation']['path'],allow_pickle=False) as saved:
                obs={k:saved[k] for k in saved.files if k!='instruction'};instruction=str(saved['instruction'])
            with Path(ref['infer']['path']).open() as stream:old=json.loads(stream.readline())
            raw=native_libero_raw(obs);seed=old['seed'];noise=np.random.default_rng(seed).standard_normal((50,32),dtype=np.float32)
            if old['response']['input_hash']!=input_digest(raw,instruction) or old['response']['noise_hash']!=value_digest(noise):raise ValueError('Historical production input/noise hash mismatch')
            outputs=[];comparisons=[]
            for repeat in range(2):
                if time.time()>=a.deadline_unix:raise TimeoutError('Deadline before repeat')
                response=policy.predict(ref['mode'],raw,instruction,seed,category='preflight');outputs.append(response)
                np.savez_compressed(directory/('repeat%d.npz'%repeat),raw=array(response['raw_actions']),decoded=array(response['actions']),
                    historical_raw=array(old['response']['raw_actions']),historical_decoded=array(old['actions']),noise=noise)
                comparisons.append(dict(raw=comparison(response['raw_actions'],old['response']['raw_actions']),decoded=comparison(response['actions'],old['actions']),
                    observed_calls=response['observed_calls'],timing=response['timing']))
            check=dict(reference=ref,repeats_vs_history=comparisons,repeat_vs_repeat=dict(raw=comparison(outputs[0]['raw_actions'],outputs[1]['raw_actions']),decoded=comparison(outputs[0]['actions'],outputs[1]['actions'])),
                production_input_hash=input_digest(raw,instruction),noise_hash=value_digest(noise),raw_state_dtype=str(raw['state8'].dtype))
            if ref['capture']:
                deadline_check(a.deadline_unix)
                with capture_buffers() as (captures,euler):captured=policy.predict(ref['mode'],raw,instruction,seed,category='preflight')
                if len(euler)!=10:raise RuntimeError('Expected ten real Euler updates in capture')
                capture_arrays={key:array(value) for key,value in captures.items()}
                capture_arrays.update({'Euler_latent_%02d'%(index+1):array(value) for index,value in enumerate(euler)})
                capture_arrays.update(raw=array(captured['raw_actions']),decoded=array(captured['actions']))
                np.savez_compressed(directory/'captured.npz',**capture_arrays)
                check['capture_vs_uncaptured']=dict(raw=comparison(captured['raw_actions'],outputs[0]['raw_actions']),decoded=comparison(captured['actions'],outputs[0]['actions']))
                # Same raw32 inputs as production, bypass only the outer D2
                # wrapper for this matched native reference. No float64 route.
                deadline_check(a.deadline_unix)
                with route_observations() as observed:
                    context,_=RawTritonBackend.prepare(backend,raw,instruction,(3,4,9,11),encode=True)
                    native,native_raw,costs=RawTritonBackend.sample(backend,context,seed,2)
                deadline_check(a.deadline_unix)
                d2=V14D2Policy(m,backend=backend).predict('v14_d2',raw,instruction,seed,category='preflight')
                np.savez_compressed(directory/'D2_matched_raw32.npz',native_raw=array(native_raw),native_decoded=array(native),wrapped_raw=array(d2['raw_actions']),wrapped_decoded=array(d2['actions']))
                calls=observed_counts(observed)
                check['D2_matched_raw32_native_parity']=dict(raw=comparison(native_raw,d2['raw_actions']),decoded=comparison(native,d2['actions']),native_calls=calls,wrapper_calls=d2['observed_calls'],native_costs=costs,wrapper_costs=d2['timing'])
                if calls['ae_layer_calls']!=36 or costs['actual_NFE']!=2:raise RuntimeError('Native D2 AE36/NFE2 violated')
            receipt['checks'].append(check);receipt['runtime_metadata']=metadata(backend);write_json(directory/'comparison.json',check);write_json(output/'receipt.json',receipt)
        booleans=[]
        for check in receipt['checks']:
            pairs=check['repeats_vs_history']+[check['repeat_vs_repeat']]
            pairs += [check[key] for key in ('capture_vs_uncaptured','D2_matched_raw32_native_parity') if key in check]
            booleans.extend(pair[name]['bitwise_equal'] for pair in pairs for name in ('raw','decoded'))
        receipt.update(status='complete',all_bitwise_equal=all(booleans),interpretation='Diagnostic evidence only; no quality authorized or marker created. Differences remain recorded, without selecting a better card/configuration.')
    except Exception:receipt.update(status='technical_failure',error=traceback.format_exc());raise
    finally:receipt['completed_unix']=time.time();write_json(output/'receipt.json',receipt)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('register','run'),required=True)
    p.add_argument('--output',required=True);p.add_argument('--parent-manifest');p.add_argument('--manifest');p.add_argument('--deadline-unix',type=float,required=True)
    p.add_argument('--card',type=int,choices=(0,1));p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.stage=='register':
        if not a.parent_manifest:p.error('--parent-manifest required')
        m=register(a.parent_manifest,a.output,a.deadline_unix);print(json.dumps(dict(manifest_hash=m['manifest_hash'],references=16,cards=CARDS,quality_requests=0)));return
    if not a.manifest or a.card is None:p.error('--manifest and --card required')
    m=verify(a.manifest)
    if not a.launch:print(json.dumps(dict(dry_run=True,manifest_hash=m['manifest_hash'],references=16,repeats=2,capture_references=2,card=a.card,uuid=CARDS[a.card],quality_requests=0)));return
    run(a,m)


if __name__=='__main__':main()
