"""Exclusive fixed-observation benchmark, 32 × 13 × (10 warm + 50 measured).

Run only after the scheduler releases this server/GPU. Any intervening model
call, source epoch change, mismatched noise or unresolved attempt stops the run.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

try:
    from .runtime_client import rpc, sha, write_json
except ImportError:
    from runtime_client import rpc, sha, write_json


def choose_observations(probe_file, count=32):
    data=json.loads(Path(probe_file).read_text())
    groups={}
    for point in data['points']:
        source=point['source']
        episode=Path(source['episode'])
        suite=episode.parents[2].name
        group=(suite,source.get('phase','unknown'))
        groups.setdefault(group,[]).append(source)
    selected=[]
    # Deterministic balanced round robin, with no invented or duplicated inputs.
    while len(selected)<count and any(groups.values()):
        for group in sorted(groups):
            if groups[group] and len(selected)<count:
                selected.append(groups[group].pop(0))
    if len(selected)!=count:
        raise ValueError(f'Need {count} real calibration observations; found {len(selected)}')
    if len({x['input_path'] for x in selected})!=count:
        raise ValueError('Calibration observations must be distinct')
    return selected


def gpu_snapshot(gpu):
    result=subprocess.run(['nvidia-smi',f'--id={gpu}',
        '--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw,driver_version',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    return dict(timestamp=time.time(),nvidia_smi=result.stdout.strip())


def append_json(path,row):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(row)+'\n')


def summarize(directory):
    rows=[json.loads(x) for x in (directory/'samples.jsonl').read_text().splitlines()]
    groups={}
    for row in rows:
        for component,value in row['timing'].items():
            groups.setdefault((row['condition'],row['phase'],component),[]).append(value)
    with (directory/'latency_breakdown.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['condition','phase','component','calls','mean_ms','p50_ms','p95_ms'])
        writer.writeheader()
        for (condition,phase,component),values in sorted(groups.items()):
            writer.writerow(dict(condition=condition,phase=phase,component=component,calls=len(values),
                mean_ms=float(np.mean(values)),p50_ms=float(np.percentile(values,50)),p95_ms=float(np.percentile(values,95))))
    return rows


def run(args):
    output=args.output_dir
    output.mkdir(parents=True,exist_ok=True)
    lease=output/'exclusive_lease.json'
    if not args.resume:
        write_json(lease,dict(pid=os.getpid(),gpu=args.gpu,port=args.port,started=time.time(),
                   scheduler_release_note=args.exclusive_note))
    elif not lease.exists():
        raise ValueError('Cannot resume without original exclusive lease')
    masks_data=json.loads(args.masks.read_text())
    candidates=masks_data if isinstance(masks_data,list) else masks_data['candidates']
    conditions=[dict(id='full',mask=[])]+candidates
    if len(conditions)!=13 or len({tuple(x['mask']) for x in conditions})!=13:
        raise ValueError('Exactly Full + 12 distinct frozen masks required')
    observations=choose_observations(args.probes)
    rpc(args.port,{'op':'reset'})
    health=rpc(args.port,{'op':'health'})
    manifest=dict(model=args.model,gpu=args.gpu,port=args.port,protocol=health['protocol'],
                  loaded_server_sha256=health.get('loaded_server_sha256'),
                  observations=observations,conditions=conditions,
                  probes_sha256=sha(args.probes),masks_sha256=sha(args.masks),
                  benchmark_source_sha256=sha(__file__),warmup_per_observation_condition=10,
                  measured_per_observation_condition=50,logical_predictions=24960,
                  cache_policy='reuse loaded weights; native image/text preprocessing on every call',
                  rotation='condition order cyclically shifted by observation index + round index',
                  exclusive_note=args.exclusive_note,hardware=gpu_snapshot(args.gpu))
    if (output/'manifest.json').exists():
        old=json.loads((output/'manifest.json').read_text())
        for key in ('model','gpu','port','observations','conditions','probes_sha256','masks_sha256','benchmark_source_sha256'):
            if old[key]!=manifest[key]: raise ValueError(f'Resume manifest changed: {key}')
    else:
        write_json(output/'manifest.json',manifest)
    completed={}
    if (output/'samples.jsonl').exists():
        for line in (output/'samples.jsonl').read_text().splitlines():
            row=json.loads(line); completed[row['key']]=row
    started=[]
    if (output/'attempts.jsonl').exists():
        started=[json.loads(x) for x in (output/'attempts.jsonl').read_text().splitlines()]
    unresolved=[x for x in started if x['key'] not in completed]
    if unresolved:
        raise ValueError('Unresolved/failed benchmark attempt; diagnose without silently spending extra calls')
    expected_call=health['calls']
    expected_epoch=None
    for observation_index,source in enumerate(observations):
        with np.load(source['input_path'],allow_pickle=False) as data:
            obs={k:data[k] for k in data.files if k not in ('instruction','seed')}
            instruction=str(data['instruction'].item()); seed=int(data['seed'])
        expected_noise=None
        for round_index in range(60):
            shift=(observation_index+round_index)%13
            order=conditions[shift:]+conditions[:shift]
            for condition in order:
                key=f"obs{observation_index:02d}/round{round_index:02d}/{condition['id']}"
                if key in completed:
                    previous=completed[key]
                    if expected_noise is None: expected_noise=previous['noise_hash']
                    continue
                row=dict(key=key,observation=observation_index,condition=condition['id'],
                         skip=condition['mask'],round=round_index,phase='warmup' if round_index<10 else 'measured',
                         seed=seed,input_path=source['input_path'],input_file_sha256=sha(source['input_path']))
                append_json(output/'attempts.jsonl',dict(row,status='started',timestamp=time.time()))
                tick=time.perf_counter()
                # Persist one actual-noise example per observation; all 780 calls
                # must have its hash. Repeated NPZ writes are excluded explicitly.
                cache_path=(str(output/f'noise_examples/obs{observation_index:02d}.npz')
                            if expected_noise is None else None)
                response=rpc(args.port,dict(op='predict',obs=obs,instruction=instruction,seed=seed,
                    skip=condition['mask'],category='benchmark',cache_path=cache_path))
                client_ms=(time.perf_counter()-tick)*1000
                if response['call']!=expected_call+1:
                    raise ValueError('Another client used the supposedly exclusive server')
                expected_call=response['call']
                epoch=response.get('runtime_epoch')
                if expected_epoch is None: expected_epoch=epoch
                if epoch!=expected_epoch: raise ValueError('Runtime code epoch changed during benchmark')
                if expected_noise is None: expected_noise=response['noise_hash']
                if response['noise_hash']!=expected_noise:
                    raise ValueError('Conditions received different actual native sampling noise')
                if np.asarray(response['actions']).shape!=(health['protocol']['H'],7):
                    raise ValueError('Native horizon changed')
                timing=dict(response['timing'],client_roundtrip_ms=client_ms)
                row.update(timing=timing,noise_hash=response['noise_hash'],input_hash=response['input_hash'],
                    loaded_code_hashes=response.get('loaded_code_hashes'),runtime_epoch=epoch,
                    call=response['call'],attempt=response.get('attempt'),
                    raw_noise_example=cache_path,operator_calls=response.get('operator_calls'))
                io_tick=time.perf_counter()
                append_json(output/'samples.jsonl',row)
                append_json(output/'recording_costs.jsonl',dict(key=key,client_jsonl_write_ms=(time.perf_counter()-io_tick)*1000))
                completed[key]=row
        append_json(output/'memory_snapshots.jsonl',dict(observation=observation_index,**gpu_snapshot(args.gpu)))
        print(json.dumps(dict(observations_completed=observation_index+1,calls=len(completed),planned=24960)),flush=True)
    rows=summarize(output)
    if len(rows)!=24960: raise ValueError('Benchmark coverage is incomplete')
    write_json(output/'completion_audit.json',dict(complete=True,logical_predictions=len(rows),
        measured_predictions=sum(x['phase']=='measured' for x in rows),
        warmup_predictions=sum(x['phase']=='warmup' for x in rows),
        source_manifest_sha256=sha(output/'manifest.json'),
        samples_sha256=sha(output/'samples.jsonl'),exclusive=True,
        note='GPU event components and client/server wall costs are separate; diagnostic hooks remain active.'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True,choices=['xiaomi-robotic0','db-cogact'])
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--probes',type=Path,required=True)
    parser.add_argument('--masks',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--exclusive-note',required=True,help='Scheduler lease/release identifier for this server/GPU')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    run(args)


if __name__=='__main__': main()
