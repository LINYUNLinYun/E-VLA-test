"""Same-card same-input prompt-resident replay; NOT closed-loop speedup.

Each prompt's five real observations warm once, then five measured repeats.
Router resets per five observations; backend remains resident across resets.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import numpy as np

from preparations.idea1_v3.bench import load_cases
from preparations.idea1_v3.cli import ROOT,MODEL_PYTHON,verify_gpu
from preparations.idea1_v3.io import atomic,write_rows
from preparations.idea1_v3.analysis import write_csv

REVISION='supplement-prompt-resident-replay-v1'


def load_inputs(args,config):
    if args.captured_input:
        files=[Path(p).resolve() for p in args.captured_input]
        sequences=[pickle.loads(p.read_bytes()) for p in files]
        if len(sequences)!=4 or any(not 1<=len(s)<=25 for s in sequences): raise ValueError('Four actual sequences, each 1..25 requests, required')
        if len({s[0]['suite'] for s in sequences})!=4: raise ValueError('Four distinct suites required')
        for path,sequence in zip(files,sequences):
            receipt=json.loads(path.with_suffix('.json').read_text())
            if receipt['sha256']!=hashlib.sha256(path.read_bytes()).hexdigest() or receipt['requests']!=len(sequence): raise ValueError('Capture receipt mismatch')
            if receipt['task']!=0 or receipt['init']!=7: raise ValueError('Only frozen init7/task0 captures allowed')
        return sequences,files
    if config['experiment']=='E1': raise ValueError('E1 needs actual continuous captures; old five-step resets cannot test P8/none')
    if not args.inputs_root: raise ValueError('E2 needs --inputs-root or captures')
    return load_cases(args.inputs_root)


class Reference:
    def __init__(self,name):
        from preparations.idea1_v3.policy import TritonBackend
        self.backend=TritonBackend(); self.name=name
    def reset(self): return None
    def acknowledge(self,executed): return None
    def predict(self,obs,instruction,suite,seed):
        from preparations.idea1_v3.policy import V14
        tick=time.perf_counter(); start=len(self.backend.startup_events)
        context,t=self.backend.prepare(obs,instruction,V14 if self.name=='V14D2' else ())
        actions,raw,costs=self.backend.sample(context,seed,10 if self.name=='FullD10' else 2)
        return dict(actions=actions,execute_length=5,raw=raw.float().cpu().numpy(),
            route=dict(path=self.name),timing=dict(t,**costs,policy_wall_ms=(time.perf_counter()-tick)*1000),startup_events=self.backend.startup_events[start:])


def factory(config):
    if config.get('reference') in ('FullD10','FullD2','V14D2'): return Reference(config['reference'])
    if config['experiment']=='E1':
        from preparations.idea1_v3_supplement.policy import Policy
        return Policy(config)
    if config.get('reference')=='pi05Full':
        from preparations.idea1_v3.pi05 import Pi05Policy
        return Pi05Policy(dict(experiment='E2',mask='Full'))
    from preparations.idea1_v3_supplement.pi05 import Pi05Policy
    return Pi05Policy(config)


def aggregate(records):
    measured=[r for r in records if r['kind']=='measurement']
    times=[r['request_wall_ms'] for r in measured]; actions=sum(r['execute_length'] for r in measured)
    ages=[r['full_cache_age'] for r in measured if r.get('full_cache_age') is not None]
    age_ranges={name:dict(min=min(a[name] for a in ages),max=max(a[name] for a in ages)) if ages else None for name in ('rounds','env_steps')}
    return dict(measured_requests=len(measured),replay_executed_actions=actions,total_request_ms=sum(times),
        measured_periodic_full_calls=sum(r.get('route',{}).get('full_reason')=='periodic' for r in measured),
        measured_full_cache_age_ranges=age_ranges,measured_requests_with_full_cache_age=len(ages),
        mean_request_ms=float(np.mean(times)),p50_request_ms=float(np.percentile(times,50)),p95_request_ms=float(np.percentile(times,95)),
        replay_ms_per_executed_action=sum(times)/actions,measured_requests_with_runtime_setup=sum(r['contains_runtime_setup'] for r in measured))


def run(args,config):
    identity=verify_gpu(args.gpu); sequences,files=load_inputs(args,config)
    args.output.mkdir(parents=True,exist_ok=False)
    safety=None
    if config['experiment']=='E1':
        from preparations.idea1_v3.safety_runtime import require_recovery_pass,install_in_model_process
        safety=require_recovery_pass(args.safety_gate); install_in_model_process(args.output/'qk_reference_once.json')
    sources={p.name:dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),text=p.read_text()) for p in (Path(__file__),Path(__file__).with_name('policy.py'),Path(__file__).with_name('pi05.py'))}
    atomic(args.output/'benchmark_sources.json',sources)
    tick=time.perf_counter(); policy=factory(config); load_s=time.perf_counter()-tick
    records=[]; checks=[]
    for si,sequence in enumerate(sequences):
        # Long E1: warm <=5 real points then reset and replay the complete
        # actual consecutive sequence once. Never duplicate five frames.
        repetitions=range(2) if args.captured_input else range(6)
        for repeat in repetitions:
            policy.reset()
            current=sequence[:5] if repeat==0 else sequence
            for ci,case in enumerate(current):
                tick=time.perf_counter(); result=policy.predict(**case); elapsed=(time.perf_counter()-tick)*1000
                actions=np.asarray(result['actions']); length=result['execute_length']
                records.append(dict(kind='setup_warmup' if repeat==0 else 'measurement',sequence=si,case=ci,repeat=repeat,suite=case['suite'],seed=case['seed'],
                    request_wall_ms=elapsed,timing=result['timing'],route=result['route'],execute_length=length,virtual_ack=True,
                    full_cache_age=result.get('full_cache_age'),
                    contains_runtime_setup=bool(result.get('startup_events')),actions_sha256=hashlib.sha256(actions.tobytes()).hexdigest()))
                policy.acknowledge(actions[:length])
                if config['experiment']=='E2' and repeat==0 and ci==0 and args.pi05_check:
                    from preparations.idea1_v3.pi05_bench import StageProfiler
                    if si==0: profiler=StageProfiler(policy,revision='v2-native-scanned-suffix')
                    staged,timing=profiler.run(case)
                    timing.update(native_reference_ae_structure='native nn.scan (supplement revision)',attribution='bounded split-stage parity against new supplement native fused total; separate compilation/sync, not summed native latency')
                    passed=bool(np.allclose(staged,actions,atol=.02,rtol=.02))
                    checks.append(dict(sequence=si,max_abs=float(np.max(np.abs(staged-actions))),atol=.02,rtol=.02,passed=passed,stages=timing))
                    if not passed: raise RuntimeError('Bounded native scanned-stage parity failed')
    write_rows(args.output/'latency_raw.jsonl',records)
    grouped=collections.defaultdict(list)
    for r in records: grouped[(r['kind'],r['route']['path'],r['route'].get('full_reason',''),r['contains_runtime_setup'])].append(r)
    write_csv(args.output/'latency_by_route_setup.csv',[dict(kind=k[0],route=k[1],full_reason=k[2],contains_runtime_setup=k[3],n=len(v),
        mean_ms=float(np.mean([r['request_wall_ms'] for r in v])),p50_ms=float(np.percentile([r['request_wall_ms'] for r in v],50)),p95_ms=float(np.percentile([r['request_wall_ms'] for r in v],95)),
        executed_actions=sum(r['execute_length'] for r in v),policy_ms=sum(r['request_wall_ms'] for r in v)) for k,v in sorted(grouped.items())])
    summary=dict(config=config,revision=REVISION,physical_gpu=args.gpu,gpu=identity,model_load_s=load_s,warmup_requests=sum(r['kind']=='setup_warmup' for r in records),**aggregate(records),execution_safety=safety,
        thread_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','XLA_FLAGS')},model_protocol=getattr(policy,'protocol',dict(reference=config.get('reference'))),
        input_files=[str(p.resolve()) for p in files],input_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in files],state_boundary='virtual ack; router reset before each actual sequence (or each five-point E2 repeat); not physically executed history',
        order='prompt outermost; captured mode <=5 warm then full actual <=25-point sequence once, otherwise five warm then five measured repeats; persistent model/backend',
        timing_boundary='full input/predict/decode including rejected draft+verify+immediate Full; startup separately; runtime setup flagged',
        interpretation='same-input replay only, not closed-loop route distribution or causal deployment speedup',native_stage_checks=checks)
    atomic(args.output/'benchmark_summary.json',summary)
    print(json.dumps(dict(output=str(args.output),**aggregate(records))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attempt',type=Path); p.add_argument('--condition'); p.add_argument('--reference',choices=('FullD10','FullD2','V14D2','pi05Full'))
    p.add_argument('--inputs-root',type=Path); p.add_argument('--captured-input',type=Path,action='append',default=[])
    p.add_argument('--output',type=Path,required=True); p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--safety-gate',type=Path); p.add_argument('--pi05-check',action='store_true'); p.add_argument('--launch',action='store_true'); args=p.parse_args()
    if args.reference:
        config=dict(experiment='E2' if args.reference=='pi05Full' else 'E1',reference=args.reference)
    else:
        if not args.attempt or not args.condition: p.error('explicit --reference or --attempt/--condition required')
        manifest=json.loads((args.attempt/'manifest.json').read_text()); config=next(c['config'] for c in manifest['conditions'] if c['id']==args.condition)
    sequences,_=load_inputs(args,config)
    if not args.launch: print(json.dumps(dict(dry_run=True,config=config,revision=REVISION,warmup=sum(min(5,len(s)) for s in sequences),measure=sum(map(len,sequences)) if args.captured_input else 100,order='prompt outermost; actual continuous sequences for E1',virtual_ack=True))); return
    if config['experiment']=='E1':
        if not args.safety_gate: p.error('completed --safety-gate required for Triton')
        from preparations.idea1_v3.safety_runtime import require_recovery_pass
        require_recovery_pass(args.safety_gate)
    python=ROOT/'policies/openpi/.venv/bin/python' if config['experiment']=='E2' else MODEL_PYTHON
    if Path(sys.executable).resolve()!=python.resolve() or os.environ.get('V3_SUPPLEMENT_BENCH_CHILD')!='1':
        env=dict(os.environ,V3_SUPPLEMENT_BENCH_CHILD='1',CUDA_VISIBLE_DEVICES=str(args.gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false')
        env['PYTHONPATH']=str(ROOT)+':'+env.get('PYTHONPATH','')
        subprocess.run([str(python),'-m','preparations.idea1_v3_supplement.bench',*sys.argv[1:]],cwd=ROOT,env=env,check=True)
    else: run(args,config)


if __name__=='__main__': main()
