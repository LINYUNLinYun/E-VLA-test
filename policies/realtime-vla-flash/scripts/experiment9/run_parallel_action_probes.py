"""Run one fixed-noise probe shard per idle GPU, then verify and merge all three seeds."""
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_standard_vlm_pruning as runner
from probe_action_expert_layers import action_metrics,summarize_candidates,write_csv,CRITERIA,FLOW_STEPS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations',type=Path,required=True)
    parser.add_argument('--converted-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    shards=args.output_dir.parent/'probe_shards';shards.mkdir(exist_ok=True)
    seeds=[12345,23456,34567]
    reused=0
    for path in (args.output_dir/'samples').glob('*.npz'):
        try:
            with np.load(path,allow_pickle=False) as data:
                info=json.loads(str(data['metadata_json'].item()))
                seed=info['noise_seed']
                valid=seed in seeds and info['full_baseline_bitwise_parity'] and info['repeat_baseline_bitwise_parity'] and data['skip_actions'].shape[0]==18
        except Exception:
            valid=False
        if not valid:
            interrupted=args.output_dir/'interrupted';interrupted.mkdir(exist_ok=True)
            path.rename(interrupted/f'{int(time.time())}_{path.name}')
            continue
        target=shards/f'seed{seed}/samples'/path.name
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():os.link(path,target)
        reused+=1
    print(f'Reusing {reused} completed sample/seed probes in seed shards',flush=True)
    pending=list(seeds);active={};next_poll=0
    def job(seed,gpu):
        path=shards/f'seed{seed}';path.mkdir(exist_ok=True)
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
        runner.log_event({'event':'action_expert_seed_shard_start','seed':seed,'gpu':gpu,'output_dir':str(path)})
        command=[runner.SERVER_PYTHON,HERE/'probe_action_expert_layers.py','--observations',args.observations,'--converted-root',args.converted_root,'--output-dir',path,'--noise-seeds',seed,'--worker']
        with (path/'worker.log').open('a') as stream:
            subprocess.run([str(p) for p in command],cwd=runner.ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        runner.log_event({'event':'action_expert_seed_shard_complete','seed':seed,'gpu':gpu})
    with ThreadPoolExecutor(max_workers=3) as pool:
        while pending or active:
            if pending and time.monotonic()>=next_poll:
                busy={gpu for _,gpu in active.values()}
                free=[g for g in runner.free_gpus(runner.load_config()) if g not in busy]
                for gpu in free:
                    if not pending:break
                    seed=pending.pop(0);active[pool.submit(job,seed,gpu)]=(seed,gpu)
                next_poll=time.monotonic()+300
            if active:
                done,_=wait(active,timeout=60,return_when=FIRST_COMPLETED)
                for future in done:
                    _,gpu=active.pop(future);future.result()
                    if pending:
                        seed=pending.pop(0);active[pool.submit(job,seed,gpu)]=(seed,gpu)
            elif pending:time.sleep(60)
    source=json.loads(args.observations.read_text());expected=set()
    source_hashes={}
    for filename in source['observations']:
        path=Path(filename)
        with np.load(path,allow_pickle=False) as data:
            meta=json.loads(str(data['metadata_json'].item()))
        source_hashes[meta['sample_id']]=hashlib.sha256(path.read_bytes()).hexdigest()
        expected.update((meta['sample_id'],seed) for seed in seeds)
    summaries=[json.loads((shards/f'seed{seed}/summary.json').read_text()) for seed in seeds]
    weight_hash=summaries[0]['checkpoint_sha256']
    for seed,summary in zip(seeds,summaries):
        if not summary.get('complete') or not summary.get('checkpoint_unchanged') or summary['checkpoint_sha256']!=weight_hash or summary['noise_seeds']!=[seed] or summary['task_count']!=40 or summary['sample_count']!=len(source_hashes):
            raise RuntimeError(f'Seed shard coverage invalid: {seed}')
    samples=args.output_dir/'samples';samples.mkdir(exist_ok=True)
    observed=set();sensitivity=[];hidden=[]
    for seed in seeds:
        for path in sorted((shards/f'seed{seed}/samples').glob('*.npz')):
            with np.load(path,allow_pickle=False) as data:
                info=json.loads(str(data['metadata_json'].item()));key=(info['sample_id'],info['noise_seed'])
                if key not in expected or key in observed or not info['full_baseline_bitwise_parity'] or not info['repeat_baseline_bitwise_parity'] or info['observation_sha256']!=source_hashes[info['sample_id']] or info['checkpoint_sha256']!=weight_hash:
                    raise RuntimeError(f'Seed/sample identity invalid: {path}')
                observed.add(key)
                for layer in range(1,19):
                    sensitivity.append({**{k:info[k] for k in ('sample_id','suite','task_id','episode_idx','semantic_phase','noise_seed')},'layer':layer,**action_metrics(data['skip_actions'][layer-1],data['baseline_actions'])})
                    for j,step in enumerate(FLOW_STEPS):
                        hidden.append({**{k:info[k] for k in ('sample_id','suite','task_id','semantic_phase','noise_seed')},'layer':layer,'flow_step':step,'flow_t':1-step/10,'adjacent_cosine':float(data['layer_cosine'][j,layer,layer-1]),'cosine_vs_layer18':float(data['layer_cosine'][j,layer,18]),'relative_residual':float(data['adjacent_relative_residual'][j,layer-1])})
            target=samples/path.name
            if target.exists():
                if hashlib.sha256(target.read_bytes()).digest()!=hashlib.sha256(path.read_bytes()).digest():raise RuntimeError(f'Existing merged sample differs: {target}')
            else:os.link(path,target)
    if observed!=expected:raise RuntimeError('Merged probe sample/seed coverage incomplete')
    write_csv(args.output_dir/'layer_action_sensitivity.csv',sensitivity)
    write_csv(args.output_dir/'layer_hidden_similarity.csv',hidden)
    ranking=summarize_candidates(sensitivity)
    candidates=[r['layer'] for r in ranking if r['eligible']][:2]
    summary={'complete':True,'sample_count':len(source_hashes),'noise_seeds':seeds,'task_count':40,'fixed_observation_inference_count':len(source_hashes)*3*21,'checkpoint_sha256':weight_hash,'checkpoint_unchanged':True,'criteria':CRITERIA,'ranking':ranking,'closed_loop_candidates':candidates,'source_manifest':str(args.observations.resolve()),'decision':'screen_selected_layers' if candidates else 'No credible single-layer pruning candidates under the prespecified criteria','seed_shards':[str(shards/f'seed{seed}') for seed in seeds],'sample_seed_coverage_verified':True}
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (args.output_dir/'prespecified_criteria.json').write_text(json.dumps(CRITERIA,indent=2)+'\n')
    print(json.dumps({'merged_observations':len(source_hashes),'sample_seeds':len(observed),'closed_loop_candidates':candidates}),flush=True)


if __name__=='__main__':
    main()
