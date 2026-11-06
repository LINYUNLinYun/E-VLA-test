"""Freeze and run the bounded 24-episode pilot with the existing LIBERO runner."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from preparations.idea1_v3_1.common import ROOT, LIMITS, atomic_json, digest, lock, read_manifest
from preparations.idea1_v3_1.server import rpc


def freeze(args):
    observations=json.loads(Path(args.observations).read_text())
    candidate_path=Path(args.candidate).resolve()
    candidate=json.loads(candidate_path.read_text())
    if candidate['state']!='exploratory_relative_prior_needs_quality_test':
        raise ValueError('This pilot tests the separately frozen relative-impact prior')
    scenes=[dict(suite=t['suite'],task=t['task'],init=0,instruction=t['instruction'])
            for t in observations['tasks'] if t['split']=='heldout']
    if len(scenes)!=8:raise ValueError('Expected exactly eight frozen heldout tasks')
    cfg=json.loads(Path(args.collection_config).read_text())
    checkpoint=Path(cfg['checkpoint']).resolve()
    weight=checkpoint/'model.safetensors'
    weight_stat=weight.stat()
    sources=['preparations/idea1_v3_1/rollout.py','preparations/idea1_v3_1/common.py',
             'preparations/idea1_v2_1/runtime_client.py','preparations/idea2/experiments.py',
             'preparations/idea2/openpi_backend.py']
    sources += [str(p.relative_to(ROOT)) for p in Path(__file__).resolve().parent.glob('*.py')]
    manifest=dict(scope='Exploratory 8-task x 1-init x 3-method pilot; no test SR used for rule selection',
        checkpoint=str(checkpoint),assets=str(Path(args.assets).resolve()),candidate=str(candidate_path),
        checkpoints={'pi05':{'identity_artifacts':{
            str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint.rglob('*.json')}}},
        checkpoint_weight_metadata=dict(path=str(weight),bytes=weight_stat.st_size,mtime_ns=weight_stat.st_mtime_ns),
        candidate_sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        conditions=[dict(id=name,mode=mode,model='pi05',horizon=10,execute_length=5)
                    for name,mode in [('original_full','full'),('fixed_mask','fixed'),('sae_semantic','sae_route')]],
        episodes=scenes,H=10,R=5,num_steps=10,limits=LIMITS,environment_seed=7,settling_steps=10,
        mask=candidate['static_mask'],delta_semantic=candidate['delta_semantic'],
        baseline='Same pair for fixed and semantic methods isolates gating; fixed period calibrated on offline data only',
        baseline_budget='Calibration replay counts matched; actual closed-loop budgets must be reported separately',
        seed_rule='Existing preparations.idea1_v3_1.common.seed(suite,task,init,request); independent of method',
        latency_scope='Synchronized complete policy wall time; differing closed-loop inputs are not a fixed-input speed benchmark',
        order='Task-major; rotate three-method order by task index',
        source_files={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sorted(set(sources))})
    manifest['manifest_hash']=digest(manifest)
    target=Path(args.output)
    if target.exists():
        if json.loads(target.read_text())!=manifest:raise ValueError('Do not overwrite a different frozen manifest')
    else:atomic_json(target,manifest)
    print(json.dumps(dict(manifest=str(target),episodes=len(scenes)*3,manifest_hash=manifest['manifest_hash'])))


def summarize(output,manifest):
    rows=[]
    for condition in manifest['conditions']:
        for scene in manifest['episodes']:
            directory=output/condition['id']/scene['suite']/f"task{scene['task']:02d}"/f"init{scene['init']:02d}"
            if not (directory/'complete.json').exists():continue
            result=json.loads((directory/'result.json').read_text())
            infers=[json.loads(line) for line in (directory/'infer.jsonl').read_text().splitlines()]
            masks=[tuple(r['routing']['mask']) for r in infers]
            rows.append(dict(condition=condition['id'],suite=scene['suite'],task=scene['task'],init=scene['init'],
                success=result['success'],steps=result['actions'],predictions=len(infers),
                skipped_calls=sum(bool(m) for m in masks),mean_skipped_ffn=float(np.mean([len(m) for m in masks])),
                full_fraction=float(np.mean([not m for m in masks])),
                policy_ms_p50=float(np.median([r['policy_ms'] for r in infers])),
                policy_ms_p95=float(np.quantile([r['policy_ms'] for r in infers],.95))))
    totals={c['id']:dict(episodes=sum(r['condition']==c['id'] for r in rows),
                        successes=sum(r['success'] for r in rows if r['condition']==c['id']))
            for c in manifest['conditions']}
    report=dict(manifest_hash=manifest['manifest_hash'],completed=len(rows),expected=24,by_condition=totals,
                episodes=rows,interpretation=manifest['latency_scope'],
                state='pilot_complete' if len(rows)==24 else 'pilot_incomplete')
    atomic_json(output/'summary.json',report)
    return report


def run(args):
    from libero.libero import benchmark
    from preparations.idea1_v3_1.rollout import Environments,run_episode
    manifest=read_manifest(args.manifest,verify_sources=True)
    if hashlib.sha256(Path(manifest['candidate']).read_bytes()).hexdigest()!=manifest['candidate_sha256']:
        raise ValueError('Frozen candidate changed')
    for scene in manifest['episodes']:
        actual=benchmark.get_benchmark_dict()[scene['suite']]().get_task(scene['task']).language
        if actual!=scene['instruction']:raise ValueError('Native task instruction/order differs from frozen cases')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    pool=Environments(reuse=False)
    try:
        with lock(output/'pilot.lock',blocking=False):
            for index,scene in enumerate(manifest['episodes']):
                conditions=manifest['conditions'];offset=index%len(conditions)
                for condition in conditions[offset:]+conditions[:offset]:
                    directory=output/condition['id']/scene['suite']/f"task{scene['task']:02d}"/f"init{scene['init']:02d}"
                    if (directory/'complete.json').exists():
                        if json.loads((directory/'complete.json').read_text())['manifest_hash']!=manifest['manifest_hash']:
                            raise ValueError('Completed episode belongs to another protocol')
                        continue
                    if directory.exists():raise ValueError(f'Inspect incomplete technical attempt before resuming: {directory}')
                    rpc(args.port,dict(op='configure',condition=condition['id'],scene=scene))
                    atomic_json(output/'status.json',dict(state='running',condition=condition['id'],scene=scene,updated_unix=time.time()))
                    try:
                        result=run_episode(directory,args.port,condition,scene['suite'],scene['task'],scene['init'],
                            pool,manifest['manifest_hash'],video=False,save_observations=False)
                    except Exception as exc:
                        atomic_json(directory/'technical_error.json',dict(error=repr(exc),valid=False))
                        raise
                    print(json.dumps(result),flush=True)
                    summarize(output,manifest)
            summary=summarize(output,manifest)
            atomic_json(output/'status.json',dict(state=summary['state'],updated_unix=time.time()))
    finally:pool.close()


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('freeze')
    for name in ('observations','candidate','collection-config','assets','output'):p.add_argument('--'+name,required=True)
    p=sub.add_parser('run')
    p.add_argument('--manifest',required=True);p.add_argument('--port',required=True,type=int);p.add_argument('--output',required=True)
    args=parser.parse_args()
    freeze(args) if args.command=='freeze' else run(args)


if __name__=='__main__':main()
