"""True native D2/D10 same-card replay totals, no hard-coded D10 profiler."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from preparations.idea1_v3.bench import load_cases
from preparations.idea1_v3.cli import ROOT,verify_gpu
from preparations.idea1_v3.analysis import write_csv
from preparations.idea1_v3.io import atomic,write_rows
from preparations.idea1_pi05_d2_search.policy import Pi05Policy,REVISION

BENCH_REVISION='pi05-d2-search-native-total-replay-v1'


def run(args,config):
    gpu=verify_gpu(args.gpu); sequences,files=load_cases(args.inputs_root)
    args.output.mkdir(parents=True,exist_ok=False)
    sources={p.name:dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),text=p.read_text()) for p in (Path(__file__),Path(__file__).with_name('policy.py'))}
    atomic(args.output/'benchmark_sources.json',sources)
    tick=time.perf_counter(); policy=Pi05Policy(config); load=time.perf_counter()-tick; records=[]
    for si,sequence in enumerate(sequences):
        for repeat in range(6):
            policy.reset()
            for ci,case in enumerate(sequence):
                tick=time.perf_counter(); result=policy.predict(**case); ms=(time.perf_counter()-tick)*1000
                actions=result['actions']; length=result['execute_length']
                if result['route']['denoise_steps']!=config['denoise_steps']: raise RuntimeError('Actual route D differs')
                records.append(dict(kind='setup_warmup' if repeat==0 else 'measurement',sequence=si,case=ci,repeat=repeat,suite=case['suite'],seed=case['seed'],request_wall_ms=ms,policy_timing=result['timing'],route=result['route'],execute_length=length,virtual_ack=True,contains_runtime_setup='JAX trace not independently observed',actions_sha256=hashlib.sha256(np.asarray(actions).tobytes()).hexdigest()))
                policy.acknowledge(actions[:length])
    write_rows(args.output/'latency_raw.jsonl',records)
    measured=[r for r in records if r['kind']=='measurement']; times=[r['request_wall_ms'] for r in measured]; actions=sum(r['execute_length'] for r in measured)
    grouped=collections.defaultdict(list)
    for r in records: grouped[r['kind']].append(r['request_wall_ms'])
    write_csv(args.output/'latency_by_route_setup.csv',[dict(kind=k,route='pi05_native',full_reason='',contains_runtime_setup='unknown JAX traces',n=len(v),mean_ms=float(np.mean(v)),p50_ms=float(np.percentile(v,50)),p95_ms=float(np.percentile(v,95))) for k,v in grouped.items()])
    summary=dict(config=config,revision=BENCH_REVISION,model_backend_revision=REVISION,model_protocol=policy.protocol,
        physical_gpu=args.gpu,gpu=gpu,thread_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','XLA_FLAGS')},
        model_load_s=load,warmup_requests=20,measured_requests=len(measured),replay_executed_actions=actions,total_request_ms=sum(times),mean_request_ms=float(np.mean(times)),p50_request_ms=float(np.percentile(times,50)),p95_request_ms=float(np.percentile(times,95)),replay_ms_per_executed_action=sum(times)/actions,measured_requests_with_runtime_setup=None,
        input_files=[str(p.resolve()) for p in files],input_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in files],
        order='prompt outermost; five exact-shape warm inputs then five measured repeats without model reconstruction',
        timing_boundary='native fused policy.infer total with actual configured num_steps; preprocessing+AE+decode; no split-stage profiler used',
        setup_visibility='model load/warm requests separate; JAX trace counts unavailable, not a blanket pure-hot claim',
        interpretation='same-input replay total only; virtual ack, not new physical execution; shared CPU/card background remains')
    atomic(args.output/'benchmark_summary.json',summary)
    print(json.dumps(dict(output=str(args.output),measured=len(measured),D=config['denoise_steps'],mean_ms=summary['mean_request_ms'])))


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--attempt',type=Path); p.add_argument('--condition'); p.add_argument('--reference',choices=('FullD10','FullD2'))
    p.add_argument('--inputs-root',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--gpu',type=int,required=True); p.add_argument('--launch',action='store_true'); args=p.parse_args()
    if args.reference:
        d=10 if args.reference=='FullD10' else 2
        config=dict(experiment='E2',model='pi05',reference=args.reference,mask=args.reference,skip=[],denoise_steps=d,revision=REVISION,timing_reference=True)
    else:
        if not args.attempt or not args.condition: p.error('explicit --reference or --attempt/--condition required')
        m=json.loads((args.attempt/'manifest.json').read_text())
        if m.get('revision')!=REVISION: raise ValueError('Only new D2 search conditions')
        config=next(c['config'] for c in m['conditions'] if c['id']==args.condition)
    load_cases(args.inputs_root)
    if not args.launch: print(json.dumps(dict(dry_run=True,config=config,warmup=20,measurement=100,revision=BENCH_REVISION,native_total_only=True))); return
    python=ROOT/'policies/openpi/.venv/bin/python'
    if Path(sys.executable).resolve()!=python.resolve() or os.environ.get('PI05_D2_SEARCH_BENCH_CHILD')!='1':
        env=dict(os.environ,PI05_D2_SEARCH_BENCH_CHILD='1',CUDA_VISIBLE_DEVICES=str(args.gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false',PYTHONPATH=str(ROOT)+':'+os.environ.get('PYTHONPATH',''))
        subprocess.run([str(python),'-m','preparations.idea1_pi05_d2_search.bench',*sys.argv[1:]],cwd=ROOT,env=env,check=True)
    else: run(args,config)


if __name__=='__main__': main()
