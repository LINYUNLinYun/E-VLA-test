"""One idle-only Plus mirror, unchanged a416 science, strict cross-device gate."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time
import traceback
import numpy as np
from preparations.idea1_v3_2.transfer_protocol import load_manifest,sha
from preparations.idea1_v3_1.common import atomic_json
from .policy_native_watch_v3 import sample,rpc,stop_owned,ROOT


def audit_cpu_backend(backend):
    if backend.adapter.backend.runtime is not None:
        raise RuntimeError('Unexpected runtime/CUDA construction before final guard')
    torch=backend.torch;count=0
    def walk(value):
        nonlocal count
        if torch.is_tensor(value):
            count+=1
            if value.device.type!='cpu':raise RuntimeError('Constructor tensor not CPU')
        elif isinstance(value,dict):
            for child in value.values():walk(child)
        elif isinstance(value,(list,tuple)):
            for child in value:walk(child)
    walk(backend.adapter.backend.checkpoint);walk(backend.adapter.backend.embeddings)
    return dict(runtime_none=True,constructor_tensors_checked=count,all_constructor_tensors_cpu=True,
        cuda_initialized_before_guard=bool(torch.cuda.is_initialized()))


def serve():
    from . import policy_server
    from .policy_plus import NativeLiberoPlusPolicy
    class GuardedPlus(NativeLiberoPlusPolicy):
        def legacy_preflight(self,bank):
            cpu=audit_cpu_backend(self.inner.backend)
            if cpu['cuda_initialized_before_guard']:raise RuntimeError('Unexpected CUDA constructor side effect')
            from preparations.idea1_v3_2.residual import require_idle_gpu
            final=require_idle_gpu()
            self.inner.protocol.update(mirror_final_preCUDA_guard=dict(checked=True,gpu_preflight=final,cpu_assembly_audit=cpu),
                mirror_wrapper_sha256=sha(__file__))
            return self.inner.legacy_preflight(bank)
    policy_server.CorrectedTransferPolicy=GuardedPlus
    sys.argv.remove('--serve');policy_server.main()


def cross_gate(source_port,mirror_port,bank):
    from preparations.idea1_v2_1.runtime_models import native_observation
    from preparations.idea1_v3_2.transfer_policy import MODES
    record=pickle.loads(Path(bank).read_bytes())[0];images,state=native_observation(record['obs'])
    raw=dict(base_rgb=np.asarray(images[0]),wrist_rgb=np.asarray(images[1]),state8=np.asarray(state,np.float32))
    checks=[]
    for mode in MODES:
        suite={'draft_goal':'libero_goal','draft_object':'libero_object','draft_spatial':'libero_spatial','draft_10':'libero_10'}.get(mode,'libero_goal')
        payload=dict(op='predict',family='libero_transfer',environment='libero',mother_suite=suite,mode=mode,observation=record['obs'],
            instruction=record['instruction'],seed=int(record['seed']),category='preflight')
        first=rpc(source_port,payload,timeout=120);second=rpc(mirror_port,payload,timeout=120)
        comparisons={}
        for key,shape in (('raw_actions',(50,32)),('actions',(50,7))):
            a,b=np.asarray(first[key],np.float32),np.asarray(second[key],np.float32)
            comparisons[key]=dict(shape=list(a.shape),bitwise_equal=bool(np.array_equal(a,b)),
                max_abs=float(np.max(np.abs(a-b))),passed=bool(a.shape==b.shape==shape and np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a,b,atol=1e-5,rtol=0)))
        if first['input_hash']!=second['input_hash'] or first['noise_hash']!=second['noise_hash']:
            raise RuntimeError('Cross-GPU actual input/noise mismatch')
        checks.append(dict(mode=mode,comparisons=comparisons,input_hash=first['input_hash'],noise_hash=first['noise_hash'],
            source_observed_calls=first['observed_calls'],mirror_observed_calls=second['observed_calls']))
    return dict(passed=all(v['passed'] for row in checks for v in row['comparisons'].values()),checks=checks,
        real_source_preflight_requests=12,source_port=source_port,mirror_port=mirror_port,bank_sha256=sha(bank),
        category='preflight',atol=1e-5,rtol=0,scope='Same actual native LIBERO input+noise cross-device raw/decoded parity; not quality or throughput')


def launch(args):
    manifest=load_manifest(args.protocol);output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);receipt=dict(status='waiting_idle_gpu',controller_pid=os.getpid(),scientific_manifest_hash=manifest['manifest_hash'],
        deadline_unix=args.deadline_unix,source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__)},
        source_service_untouched=True,candidates=[1,0,2],service_port=args.port,started_unix=time.time())
    service=None;selected=None
    def terminate(signum,frame):raise InterruptedError('Own mirror controller termination/deadline')
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    try:
        while time.time()<args.deadline_unix:
            for index in (1,0,2):
                current=sample(index)
                with (output/'gpu_samples.jsonl').open('a') as stream:stream.write(json.dumps(current)+'\n')
                if current['idle']:
                    final=sample(index)
                    if final['idle'] and final['uuid']==current['uuid']:selected=final;break
            if selected:break
            atomic_json(output/'controller.json',receipt);time.sleep(min(30,max(0,args.deadline_unix-time.time())))
        if selected is None:receipt.update(status='deadline_no_idle_gpu');return
        command=[str(ROOT/'policies/realtime-vla-flash/.venv/bin/python'),'-m',
            'preparations.idea1_v3_2_e4_supplement.policy_plus_mirror','--serve','--family','libero_transfer',
            '--protocol',args.protocol,'--port',str(args.port),'--output',str(output/'service.json'),
            '--deadline-unix',str(args.deadline_unix),'--launch']
        log=(output/'service.log').open('x')
        service=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=selected['uuid'],JAX_PLATFORMS='cpu'),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        receipt.update(status='loading_mirror',selected_gpu=selected,service_pid=service.pid,service_command=command)
        atomic_json(output/'controller.json',receipt)
        while time.time()<args.deadline_unix:
            if service.poll() is not None:raise RuntimeError('Mirror service exited before ready: '+str(service.returncode))
            try:
                health=rpc(args.port,dict(op='health'),timeout=2)['protocol']
                if health['server_pid']!=service.pid or health['gpu_uuid']!=selected['uuid'] or health['manifest_hash']!=manifest['manifest_hash']:
                    raise RuntimeError('Mirror endpoint identity mismatch')
                if not health.get('legacy_gate_passed') or not health.get('mirror_final_preCUDA_guard',{}).get('checked'):
                    raise RuntimeError('Mirror startup/finalCUDA gate absent')
                receipt['health']=health;break
            except (ConnectionError,OSError,TimeoutError):time.sleep(2)
        else:raise TimeoutError('Mirror startup deadline')
        receipt['cross_gpu_gate']=cross_gate(args.source_port,args.port,args.bank)
        if not receipt['cross_gpu_gate']['passed']:raise RuntimeError('Cross-GPU raw+decoded strict parity failed; no quality allowed')
        receipt.update(status='ready_for_half_case_shards',ready_unix=time.time(),quality_requests_dispatched_by_controller=0)
        atomic_json(output/'controller.json',receipt)
        while time.time()<args.deadline_unix:
            if service.poll() is not None:raise RuntimeError('Mirror exited during assigned case shards')
            time.sleep(min(10,max(0,args.deadline_unix-time.time())))
        receipt.update(status='execution_deadline')
    except BaseException:
        receipt.update(status='stopped_or_technical_failure',error=traceback.format_exc());raise
    finally:
        stop_owned(service,args.port);receipt.update(completed_unix=time.time(),service_exit_code=service.poll() if service else None)
        atomic_json(output/'controller.json',receipt)


def main():
    if '--serve' in sys.argv:serve();return
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',required=True);p.add_argument('--output',required=True)
    p.add_argument('--deadline-unix',type=float,required=True);p.add_argument('--port',type=int,default=19146)
    p.add_argument('--source-port',type=int,default=19143);p.add_argument('--bank',default='results/idea1_v3_1/input_bank.pkl');p.add_argument('--launch',action='store_true')
    a=p.parse_args()
    if not a.launch:print(json.dumps(dict(dry_run=True,**vars(a))));return
    launch(a)


if __name__=='__main__':main()
