"""Selected-condition single-request replay; no environment/baseline episodes.

Twenty saved real observations are replayed in four fixed sequences, with a
fresh router/history state per sequence. E1/E3 acknowledgements are explicitly
virtual predictions, not executed physical history. Replay route frequencies
are never substituted for measured closed-loop frequencies.
"""
import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np

from preparations.idea1_v3.cli import ROOT,MODEL_PYTHON,verify_gpu
from preparations.idea1_v3.io import atomic,write_rows


def load_cases(root):
    files=sorted((Path(root)/'episodes').glob('E1/*/A/*/task00/init00/profile_inputs.pkl'))
    sequences=[pickle.loads(path.read_bytes()) for path in files]
    if len(sequences)!=4 or any(len(sequence)!=5 for sequence in sequences):
        raise ValueError('Need exactly four saved five-request real-observation sequences; no fake input substitution')
    return sequences,files


def run(args,config):
    if config['experiment']=='E2':
        from preparations.idea1_v3.pi05 import Pi05Policy
        factory=lambda:Pi05Policy(config)
    elif config['experiment']=='E3':
        from preparations.idea1_v3.history import HistoryPolicy
        factory=lambda:HistoryPolicy(config)
    else:
        from preparations.idea1_v3.policy import Policy
        factory=lambda:Policy(config)
    identity=verify_gpu(args.gpu)
    sequences,files=load_cases(args.inputs_root)
    args.output.mkdir(parents=True,exist_ok=False)
    safety=None
    if config['experiment']!='E2':
        from preparations.idea1_v3.safety_runtime import require_recovery_pass,install_in_model_process
        safety=require_recovery_pass(args.safety_gate)
        # This executes in the respawned model interpreter, before any policy
        # construction/inference; launcher-only installation is insufficient.
        install_in_model_process(args.output/'qk_reference_once.json')
    source_files=('bench.py','pi05_bench.py','pi05_bench_v2.py','pi05.py','policy.py','history.py','safe_qk.py','safety_runtime.py')
    sources={}
    for name in source_files:
        path=Path(__file__).with_name(name); raw=path.read_bytes()
        sources[name]=dict(sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode())
    atomic(args.output/'benchmark_sources.json',sources)
    tick=time.perf_counter(); policy=factory(); load_s=time.perf_counter()-tick
    records=[]; references={}; stage_records=[]
    profiler=None
    if config['experiment']=='E2' and args.pi05_stages:
        from preparations.idea1_v3.pi05_bench import StageProfiler
        profiler=StageProfiler(policy,revision=args.pi05_stage_revision)
    for repeat in range(6):
        for sequence_index,sequence in enumerate(sequences):
            policy.reset()
            for case_index,case in enumerate(sequence):
                backend=getattr(policy,'backend',None)
                startup_offset=len(getattr(backend,'startup_events',[]))
                tick=time.perf_counter(); result=policy.predict(**case)
                elapsed=(time.perf_counter()-tick)*1000
                actions=np.asarray(result['actions']); normalized=np.asarray(result['raw'])
                identity_key=(sequence_index,case_index)
                if repeat==1: references[identity_key]=dict(actions=actions.copy(),route=result['route'],length=result['execute_length'])
                reference=references.get(identity_key)
                records.append(dict(kind='setup_warmup' if repeat==0 else 'measurement',repeat=repeat,
                    suite=case['suite'],sequence=sequence_index,case=case_index,seed=case['seed'],
                    request_wall_ms=elapsed,timing=result['timing'],route=result['route'],counts=result['counts'],
                    execute_length=result['execute_length'],virtual_ack=True,
                    contains_runtime_setup=bool(result.get('startup_events') or len(getattr(backend,'startup_events',[]))>startup_offset),
                    actions_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
                    normalized_sha256=hashlib.sha256(normalized.tobytes()).hexdigest(),
                    max_abs_action_vs_first_measured=float(np.max(np.abs(actions-reference['actions']))) if reference else None,
                    route_equal_first_measured=result['route']==reference['route'] if reference else None,
                    length_equal_first_measured=result['execute_length']==reference['length'] if reference else None))
                # Explicit virtual replay state: this is not a new physical
                # rollout and E3 history here must never be labeled real history.
                policy.acknowledge(actions[:result['execute_length']])
                if profiler is not None:
                    staged,timing=profiler.run(case)
                    difference=float(np.max(np.abs(staged-actions)))
                    stage_records.append(dict(kind='setup_warmup' if repeat==0 else 'measurement',repeat=repeat,
                        sequence=sequence_index,case=case_index,suite=case['suite'],**timing,
                        max_abs_decoded_vs_native=difference,predefined_abs_tolerance=.02,predefined_rel_tolerance=.02,
                        within_tolerance=bool(np.allclose(staged,actions,atol=.02,rtol=.02))))
    write_rows(args.output/'latency_raw.jsonl',records)
    if stage_records: write_rows(args.output/'pi05_stage_raw.jsonl',stage_records)
    grouped=collections.defaultdict(list)
    for row in records: grouped[(row['kind'],row['route']['path'],row['route'].get('full_reason',''),row['contains_runtime_setup'])].append(row['request_wall_ms'])
    with (args.output/'latency_by_route_setup.csv').open('w') as stream:
        writer=csv.writer(stream); writer.writerow(['kind','route','full_reason','contains_runtime_setup','n','mean_ms','p50_ms','p95_ms'])
        for key,values in sorted(grouped.items()): writer.writerow([*key,len(values),float(np.mean(values)),float(np.percentile(values,50)),float(np.percentile(values,95))])
    measured=[row for row in records if row['kind']=='measurement']
    atomic(args.output/'benchmark_summary.json',dict(config=config,physical_gpu=args.gpu,gpu=identity,
        model_load_s=load_s,warmup_requests=20,measured_requests=len(measured),input_files=[str(file.resolve()) for file in files],
        execution_safety=safety,
        own_qk_reference_receipt=str(args.output/'qk_reference_once.json') if safety else None,
        own_qk_reference_status=('passed' if (args.output/'qk_reference_once.json').exists() else 'not exercised: pure learned E4 performs no Full AE QK') if safety else 'N/A native JAX',
        repeated_measured_max_abs_action=max(row['max_abs_action_vs_first_measured'] or 0 for row in measured),
        repeated_measured_route_exact=all(row['route_equal_first_measured'] for row in measured),
        repeated_measured_length_exact=all(row['length_equal_first_measured'] for row in measured),
        input_status='observations and seeds are saved real data; E1/E3 router history uses virtual acknowledgements',
        interpretation='fixed-sequence microbenchmark only; replay route counts are not closed-loop frequencies',
        first_use='repeat0 is excluded from measured requests; measurements may still include prompt-length/runtime reconstruction and shape setup',
        setup_visibility='Triton runtime construction events flagged; JAX recompilations not independently counted',
        pi05_stage_revision=args.pi05_stage_revision if profiler else None,
        sharing='shared CPU/NFS loads not controlled; no end-to-end throughput claim',
        pi05_stages='diagnostic split JAX equations and sync boundaries; sums do not replace native total' if profiler else None))
    if stage_records:
        hot=[row for row in stage_records if row['kind']=='measurement']
        summary={name:dict(mean=float(np.mean([r[name] for r in hot])),p50=float(np.percentile([r[name] for r in hot],50)),
            p95=float(np.percentile([r[name] for r in hot],95))) for name in ('input_ms','image_embedding_ms','vlm_prefill_ms','ae_d10_ms','decode_ms')}
        atomic(args.output/'pi05_stage_summary.json',dict(stages=summary,
            stage_revision=args.pi05_stage_revision,
            max_abs_decoded_vs_native=max(r['max_abs_decoded_vs_native'] for r in hot),all_within_predefined_tolerance=all(r['within_tolerance'] for r in hot),
            adopted_for_scientific_execution=False))
    # Real mixed route frequencies/costs come only from valid closed-loop logs.
    from preparations.idea1_v3.analysis import collect
    real,_=collect([args.attempt]); route_counts=collections.Counter(); real_cost=real_actions=0
    for episode in real:
        if episode['condition']!=args.condition or episode['analysis_stage'] in ('calibration','P0'): continue
        real_actions+=episode['actions']
        for line in (Path(episode['directory'])/'infer.jsonl').read_text().splitlines():
            inference=json.loads(line); route=inference['route']; route_counts[(route['path'],route.get('full_reason',''))]+=1
            real_cost+=inference['timing']['policy_wall_ms']
    with (args.output/'real_closed_loop_routes.csv').open('w') as stream:
        writer=csv.writer(stream); writer.writerow(['route','full_reason','actual_closed_loop_requests'])
        for key,count in sorted(route_counts.items()): writer.writerow([*key,count])
    atomic(args.output/'real_closed_loop_cost.json',dict(actual_actions=real_actions,policy_ms=real_cost,
        policy_ms_per_action=real_cost/real_actions if real_actions else None,
        includes_rejected_draft_and_immediate_full_cost=True,source='valid closed-loop inference journals, not replay route weights'))
    print(json.dumps(dict(measured=len(measured),output=str(args.output))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt',type=Path,required=True); parser.add_argument('--condition',required=True)
    parser.add_argument('--inputs-root',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,required=True); parser.add_argument('--pi05-stages',action='store_true')
    parser.add_argument('--pi05-stage-revision',choices=('v1-unrolled-ae','v2-native-scanned-suffix'),default='v1-unrolled-ae')
    parser.add_argument('--safety-revision',choices=('qk-row-bounds-v1',))
    parser.add_argument('--safety-gate',type=Path)
    parser.add_argument('--launch',action='store_true'); args=parser.parse_args()
    manifest=json.loads((args.attempt/'manifest.json').read_text())
    if manifest['stage'] in ('P0','calibration'): raise ValueError('Benchmark only frozen representative scientific conditions, not all calibration candidates')
    config=next(condition['config'] for condition in manifest['conditions'] if condition['id']==args.condition)
    if not args.launch:
        sequences,_=load_cases(args.inputs_root)
        print(json.dumps(dict(dry_run=True,config=config,warmup_requests=20,measured_requests=100,
            safety_revision=args.safety_revision,safety_gate=str(args.safety_gate) if args.safety_gate else None,
            safety_required_for_triton=config['experiment']!='E2',
            pi05_stage_revision=args.pi05_stage_revision if config['experiment']=='E2' and args.pi05_stages else None,
            state='reset per five-request sequence; virtual ack explicitly labeled',real_input_points=sum(map(len,sequences))),indent=2)); return
    if config['experiment']!='E2':
        if args.safety_revision!='qk-row-bounds-v1' or args.safety_gate is None:
            raise ValueError('Remaining Triton benchmarks require explicit --safety-revision qk-row-bounds-v1 and passed --safety-gate')
        from preparations.idea1_v3.safety_runtime import require_recovery_pass
        require_recovery_pass(args.safety_gate)
    elif args.safety_revision:
        raise ValueError('Triton safety revision does not apply to native JAX E2')
    python=ROOT/'policies/openpi/.venv/bin/python' if config['experiment']=='E2' else MODEL_PYTHON
    if Path(sys.executable).resolve()!=python.resolve() or os.environ.get('V3_BENCH_CHILD')!='1':
        env=dict(os.environ,V3_BENCH_CHILD='1',CUDA_VISIBLE_DEVICES=str(args.gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false')
        env['PYTHONPATH']=str(ROOT)+':'+env.get('PYTHONPATH','')
        subprocess.run([str(python),'-m','preparations.idea1_v3.bench',*sys.argv[1:]],env=env,cwd=ROOT,check=True)
    else: run(args,config)

if __name__=='__main__': main()
