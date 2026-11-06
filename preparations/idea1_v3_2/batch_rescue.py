"""Explicitly approximate B4 and measured B1 after-control on same live service."""
import argparse
from collections import Counter
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from preparations.idea1_v3_1.common import ROOT, PYTHONS, atomic_json, environment
from preparations.idea1_v3_1.server import rpc
from preparations.idea1_v3_1.benchmark import load_banks
from .batch_policy import compare


def run(args):
    output=Path(args.output)
    diagnostics_path=Path(args.diagnostics) if getattr(args,'diagnostics',None) else output/'batch_diagnostics.json'
    diagnostics=json.loads(diagnostics_path.read_text())
    if not (diagnostics['finite'] and diagnostics['no_noise_mixup'] and diagnostics['repeat_exact'] and diagnostics['permutation_exact']):
        raise RuntimeError('Do not run approximate closed-loop before noise/order/stability diagnostics pass')
    records,_=load_banks([args.bank])
    protocol=rpc(args.port,dict(op='health'))['protocol']
    receipt=dict(status='running',controller_pid=os.getpid(),service_pid=protocol['server_pid'],
        approximate=True,numerical_parity_passed=False,
        purpose='Explicit exploratory batching despite failed numerical bank gate; measured B1 after-control provides throughput comparator.',
        prior_diagnostics_path=str(diagnostics_path), prior_diagnostics_service_pid=diagnostics['service_pid'],
        original_controller_state=getattr(args,'original_controller_state','SIGSTOP after first B1 arm; cleanup SIGINT then SIGCONT only after rescue completes'),
        initial_baseline_wall_unavailable=('Not collected in this fresh cohort; measured B1 after-control is the comparator.'
            if getattr(args,'arm_spec',None) else 'Controller was paused before recording elapsed; initial completed eight scenes retained for outcome/input comparison only.'),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'preparations/idea1_v3_2').glob('batch*.py')},
        arms=[],controls=[])
    path=output/'rescue.json'
    atomic_json(path,receipt)
    ref=rpc(args.port,dict(op='control',records=records[:4]))
    warm=[]
    for size in (1,2,3,4):
        for i in range(3):
            warm.append(rpc(args.port,dict(op='control',records=records[:size]))['timing'])
    receipt['all_actual_sizes_warm_before_timing']=warm
    live=[]
    streams=[]
    deadline=datetime.datetime(2026,10,22,5,7,tzinfo=datetime.timezone.utc).timestamp()
    try:
        specifications=getattr(args,'arm_spec',None) or ['approximate_b4:4:40','rescue_b1_after:1:0']
        for specification in specifications:
            name,maximum,wait_ms=specification.split(':')
            maximum,wait_ms=int(maximum),float(wait_ms)
            rpc(args.port,dict(op='configure',maximum=maximum,wait_ms=wait_ms))
            before=rpc(args.port,dict(op='control',records=records[:4]))
            receipt['controls'].append(dict(arm=name,when='before',raw=compare(ref['raw'],before['raw']),decoded=compare(ref['actions'],before['actions'])))
            if not receipt['controls'][-1]['raw']['strict_1e5_passed']:
                raise RuntimeError('Same-B4 fixed input exact repeat failed')
            row=dict(arm=name,maximum=maximum,wait_ms=wait_ms,workers=4,pids=[],commands=[])
            receipt['arms'].append(row)
            tick=time.perf_counter()
            for i in range(4):
                command=[str(PYTHONS['environment']),'-m','preparations.idea1_v3_2.batch_e1','--worker',str(i),
                    '--output',args.output,'--port',str(args.port),'--manifest',args.manifest,'--arm',name,'--launch']
                stream=(output/f'{name}_worker{i}.log').open('x')
                streams.append(stream)
                child=subprocess.Popen(command,cwd=ROOT,env=environment(args.gpu),stdout=stream,stderr=subprocess.STDOUT)
                live.append(child)
                row['pids'].append(child.pid)
                row['commands'].append(command)
            atomic_json(path,receipt)
            while any(c.poll() is None for c in live):
                if time.time()>deadline:
                    raise TimeoutError('Original one-hour E1 hard deadline')
                if any(c.poll() not in (None,0) for c in live):
                    raise RuntimeError('Owned rescue rollout worker failed')
                time.sleep(.1)
            if any(c.returncode for c in live):
                raise RuntimeError('Owned rescue rollout worker failed')
            row['wall_s']=time.perf_counter()-tick
            live.clear()
            counts=rpc(args.port,dict(op='configure',maximum=1))['actual_batches']
            row['actual_batch_histogram']=dict(Counter(counts))
            results=[json.loads(p.read_text()) for p in (output/name).glob('*/init*/result.json')]
            row.update(completed_episodes=len(results),successes=sum(r['success'] for r in results),
                completed_actions=sum(r['actions'] for r in results),predictions=sum(r['predictions'] for r in results),
                episodes_per_hour=len(results)*3600/row['wall_s'])
            after=rpc(args.port,dict(op='control',records=records[:4]))
            receipt['controls'].append(dict(arm=name,when='after',raw=compare(ref['raw'],after['raw']),decoded=compare(ref['actions'],after['actions'])))
            atomic_json(path,receipt)
        receipt['status']='completed'
        baseline=next(a for a in receipt['arms'] if a['maximum']==1)
        receipt['episode_throughput_speedup_vs_measured_b1_after']={a['arm']:baseline['wall_s']/a['wall_s'] for a in receipt['arms']}
    except BaseException as exc:
        receipt.update(status='failed',error=repr(exc))
        raise
    finally:
        for child in live:
            if child.poll() is None:
                child.terminate()
        for child in live:
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for stream in streams: stream.close()
        atomic_json(path,receipt)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--bank',default='results/idea1_v3_1/input_bank.pkl')
    parser.add_argument('--manifest',default='results/idea1_v3_1/manifest_v2.json')
    parser.add_argument('--port',type=int,default=19332)
    parser.add_argument('--gpu',type=int,default=3)
    parser.add_argument('--diagnostics')
    parser.add_argument('--arm-spec',action='append',help='name:maximum_batch:aggregation_wait_ms; include one B1 arm')
    args=parser.parse_args()
    run(args)
