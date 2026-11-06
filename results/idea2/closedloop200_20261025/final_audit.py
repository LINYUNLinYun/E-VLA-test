"""Terminal-only independent CPU artifact audit; never runs model or simulator.

Refuses incomplete dispatcher before scanning episode artifacts. Reuses the
CPU protocol validator from prove_reuse32, and independently checks every
selected raw episode once. No videos or complete checkpoint bytes are read.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import traceback

os.environ['CUDA_VISIBLE_DEVICES']=''
R=Path(__file__).resolve().parent
ROOT=R.parents[2]
P=ROOT/'reports/idea2_closedloop200_20261025'
HASH_CACHE={}
CHECKS=0


def read(path):return json.loads(Path(path).read_text())

def sha(path):
    path=Path(path).resolve()
    if path not in HASH_CACHE:
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
        HASH_CACHE[path]=h.hexdigest()
    return HASH_CACHE[path]

def require(value,reason):
    global CHECKS
    CHECKS+=1
    if not value:raise ValueError(reason)

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def canonical(row):return (row.get('configuration',row.get('label',row.get('arm'))),row['suite'],row['task'],row['init'])

def guard(record,uuid):
    before,after=record['gpu_before'],record['gpu_after'];observed=record['render_gpu_verified']['snapshot']
    require(before['uuid']==after['uuid']==observed['uuid']==uuid,'Render device mismatch')
    require(not before['graphics_processes'] and not after['graphics_processes'] and len(observed['graphics_processes'])==1,'Render lifetime mismatch')
    require({p['pid'] for p in before['processes']}=={p['pid'] for p in observed['processes']}=={p['pid'] for p in after['processes']},'Compute ownership changed')


def audit():
    dispatch=read(R/'dispatch_execution.json')
    require(dispatch['status']=='passed','Dispatcher incomplete: refuse raw episode scan')
    import numpy as np
    proof=load('closedloop200_protocol_proof',R/'prove_reuse32.py');proof.require=require
    client=load('closedloop200_original_client',ROOT/'preparations/idea1_v2_1/runtime_client.py')
    normalized=read(R/'normalized_cases.json');rows=normalized['cases'] if isinstance(normalized,dict) else normalized
    expected={(arm,suite,task,init) for arm in proof.ARMS for suite in proof.SUITES for task in range(10) for init in range(5)}
    require(len(rows)==800 and {canonical(r) for r in rows}==expected,'Not exactly800 unique cells')
    require(all(r['status']=='valid' and type(r['success']) is bool for r in rows),'Invalid/technical selected outcome')
    require(Counter(r['configuration'] for r in rows)==dict.fromkeys(proof.ARMS,200),'Arm coverage')
    reuse=read(R/'reuse32.json');require(reuse['status']=='passed' and reuse['selected_episodes']==32,'Reuse proof not passed32')
    require(sha(R/'case_universe.json')==reuse['new_case_universe_sha256'],'Universe drift')
    pilot={canonical(r):r for r in reuse['selected_cases']};require(len(pilot)==32,'Duplicate reuse proof')
    original=read(Path(reuse['source_root'])/'execution.json')
    original_cases={canonical(c):c for c in original['cases'] if canonical(c) in pilot}
    require(original['status']=='passed' and original['source_unchanged'] and original['server_exit_code']==0,'Pilot source terminal failure')
    sources={};origins={};chunks=[];technical=[];excluded=[]
    for chunk in dispatch['chunks']:
        require(chunk['status']=='passed','Incomplete chunk')
        exit_code=chunk.get('parent_exit_code')
        if exit_code is None and chunk['id']==0:exit_code=read(R/'initial_parent_exit.json')['exit_code']
        require(exit_code==0,'Chunk parent exit not0')
        directory=Path(chunk['output']);execution=read(directory/'execution.json');plan=read(directory/'plan.json')
        require(execution['status']=='passed' and execution['source_unchanged'] and execution['server_exit_code']==0,'Chunk terminal/source failure')
        require(plan['reuse_manifest']['sha256']==sha(R/'reuse32.json'),'Chunk reuse proof drift')
        require([plan[k] for k in ('D','H','R','env_seed','settling_steps')]==[10,10,5,7,10] and not plan['environment_reuse'] and plan['suite_step_limits']==proof.LIMITS,'Chunk model/env protocol mismatch')
        require(plan['checkpoint']==reuse['checkpoint_identity']['path'],'Checkpoint path mismatch')
        for name,meta in plan['checkpoint_files'].items():
            stat=(Path(plan['checkpoint'])/name).stat();require(meta==dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns),'Checkpoint metadata mismatch')
        require(len(plan['source_sha256'])==77,'Expected77 frozen sources')
        for name,value in plan['source_sha256'].items():require(sha(ROOT/name)==value,'Frozen source changed:'+name);sources[name]=value
        for c in execution['cases']:
            require(c['status']=='valid','Chunk contains invalid selected case')
            if c['source_kind']!='new':continue
            key=canonical(c);require(key not in origins,'New episode rerun/duplicate across chunks');origins[key]=(c,directory,plan)
            attempts=c['attempts'];require(1<=len(attempts)<=2,'Technical retries unbounded')
            require(attempts[-1]['status']=='valid' and attempts[-1]['client_exit_code']==0,'Final episode attempt invalid')
            require(all(a['status']!='valid' for a in attempts[:-1]),'A valid task outcome was rerun')
            technical.extend(dict(case_key=c['case_key'],attempt=a) for a in attempts[:-1])
        for previous in chunk.get('previous_attempts',[]):
            old=Path(previous['output']);require(not(old/'execution.json').exists(),'Excluded port preflight had execution state')
            require(not list(old.glob('episodes/**/result.json')),'Port preflight contained completed episodes')
            excluded.append(dict(kind='zero_episode_port_preflight',path=str(old),exit_code=previous.get('parent_exit_code')))
        chunks.append((directory,execution,plan))
    require(len(origins)==768 and set(origins).isdisjoint(pilot) and set(origins)|set(pilot)==expected,'Expected32reuse+768new')
    for name,value in reuse['current_source_hashes'].items():require(sources.get(name)==value,'Pilot frozen protocol differs:'+name)
    initial={};cases=[];requests=0
    for normalized_row in rows:
        key=canonical(normalized_row);arm,suite,task,init=key;d=Path(normalized_row['directory'])
        if key in pilot:
            require(d.resolve()==Path(pilot[key]['source_directory']).resolve() and pilot[key]['compatible'],'Pilot source identity incompatible')
            c=original_cases[key];guard(c,c['gpu_before']['uuid']);require(c['client_exit_code']==0 and c['parent_drain']['ok'],'Pilot process/drain')
            uuid=c['gpu_before']['uuid'];origin='reused_original8'
        else:
            c,chunk,plan=origins[key];require(d.resolve()==Path(c['directory']).resolve(),'Final selected path differs from valid origin')
            uuid=plan['gpu_uuid'];origin='new'
        identity,result,complete,smoke,env=[read(d/name) for name in ('identity.json','result.json','complete.json','smoke_receipt.json','environment_protocol.json')]
        infers=[json.loads(line) for line in (d/'infer.jsonl').read_text().splitlines()]
        require(result['valid'] and complete['status']=='valid' and complete['valid'] and smoke['status']=='passed' and smoke['drain']['ok'],'Invalid episode receipt:'+str(key))
        require(result['success']==complete['success']==normalized_row['success']==c['result']['success'],'Outcome disagreement')
        require(result['actions']==complete['steps']==normalized_row['steps']<=proof.LIMITS[suite],'Steps/limits differ')
        require(result['termination']==('success' if result['success'] else 'action_limit'),'Unexpected terminal outcome')
        require((identity['suite'],identity['task'],identity['init'],identity['env_seed'])==(suite,task,init,7) and identity['model']=='pi05' and not identity['skip'],'Environment/model identity mismatch')
        init_file=Path(identity['init_file']);bddl=Path(str(init_file).replace('/init_files/','/bddl_files/').replace('.pruned_init','.bddl'))
        require(sha(init_file)==identity['init_file_sha256'] and sha(bddl)==identity['bddl_sha256'],'Environment asset mismatch')
        require(env['loaded_client_sha256']==sources['preparations/idea1_v2_1/runtime_client.py'] and env['control_frequency_hz']==result['control_hz']==20,'Original client/control frequency changed')
        proof.protocol(read(d/'native_protocol.json'),arm,False)
        with (d/'telemetry.jsonl').open() as stream:first_telemetry=json.loads(next(stream));count=1+sum(1 for _ in stream)
        require(count==result['actions']+1 and first_telemetry['when']=='initial' and abs(first_telemetry['sim_time_s']-.5)<1e-9,'Settle/reset telemetry mismatch')
        require(len(infers)==result['predictions']==complete['predictions']==normalized_row['requests'],'Request count mismatch')
        first=infers[0];epoch=first['snapshot']['epoch'];episode=first['episode'];sm,lead,tail=proof.ARMS[arm]
        for i,a in enumerate(infers):
            proof.protocol(a['protocol'],arm,True)
            require(a['ok'] and a['bootstrap']==(i==0) and a['request']==a['replan']==i,'Request/bootstrap sequencing')
            require(a['episode']==episode and a['snapshot']['epoch']==epoch,'Mid-episode epoch changed')
            require(a['seed']==client.model_seed(suite,task,init,i),'Original model seed mismatch')
            require(a['current_observation_id']==a['new_prefix_observation_id']==a['suffix_state_id']==a['reference_state_id']==i and a['old_prefix_observation_id']==max(0,i-1),'KV observation identity mismatch')
            f=a['frontiers'];require(len(f)==10 and f==sorted(f) and all(lead<=v<=18 for v in f),'Invalid frontier')
            require(a['per_iteration_new_layer_counts']==f and a['per_iteration_old_layer_counts']==[18-v for v in f],'KV layer mapping mismatch')
            if i==0 or arm=='fresh18':require(f==[18]*10,'Stale bootstrap/fresh18')
            if tail:require(f[-tail:]==[18]*tail,'Stale mandatory tail')
            require(a['graph_kv_buffers']==2 and a['pool']['live_snapshots']<=2 and (i==0 or a['graph_setup_ms']==0) and not a['next_encoder'] and not a['lookahead'],'KV lifecycle/recapture/lookahead mismatch')
            require(a['generated_length']==10 and a['submitted_length']==5 and 0<a['executed_length']<=5,'Action count mismatch')
            input_file=d/f'inputs/replan{i:04d}.npz';action_file=d/f'chunk{i:04d}.npz'
            require(sha(input_file)==a['input_file_sha256'] and sha(action_file)==a['prediction_sha256'],'Saved input/action hash mismatch')
            with np.load(action_file,allow_pickle=False) as arrays:actions=arrays['actions']
            require(actions.shape==(10,7) and np.isfinite(actions).all() and np.array_equal(actions,np.asarray(a['chunk_actions'],dtype=np.float32)),'Saved action content mismatch')
            norm=np.asarray(a['normalized_actions']);require(norm.shape==(10,32) and np.isfinite(norm).all(),'Invalid normalized output')
        with np.load(d/'inputs/replan0000.npz',allow_pickle=False) as arrays:
            raw={k:arrays[k] for k in arrays.files if k not in ('instruction','seed')}
            require(int(arrays['seed'])==first['seed'] and client.input_digest(raw,str(arrays['instruction']))==first['input_hash'],'Initial raw observation/seed digest differs')
            obs_hash=client.value_digest(raw)
        require(first['input_hash']==complete['first_input_hash'],'Initial receipt digest differs')
        initial[key]=dict(row=first,observation_hash=obs_hash,init_vector_hash=identity['init_vector_hash'],gpu_uuid=uuid)
        if origin=='new':require(c['bootstrap_epoch']==epoch and c['bootstrap_episode']==episode,'Worker bootstrap receipt differs')
        requests+=len(infers);cases.append(dict(configuration=arm,suite=suite,task=task,init=init,success=result['success'],steps=result['actions'],requests=len(infers),source_kind=origin,directory=str(d),gpu_uuid=uuid,first_input_sha256=first['input_hash'],infer_sha256=sha(d/'infer.jsonl'),identity_sha256=sha(d/'identity.json')))
    batch_count=0
    for directory,execution,plan in chunks:
        by_name={c['case_key']:c for c in execution['cases']};prior_episode=None
        for batch in execution['batches']:
            require(batch['client_exit_code']==0 and batch['drain']['ok'],'Batch client/drain failed');guard(batch,plan['gpu_uuid'])
            receipt=read(directory/'requests'/f"{batch['tag']}.receipt.json")
            require([c['case_key'] for c in receipt['cases']]==batch['case_keys'],'Batch case order mismatch')
            if batch['status']!='passed':
                require(batch['status']=='technical_failure','Unknown batch failure');technical.append(dict(batch=batch['tag'],receipt=receipt));continue
            require(receipt['status']=='passed','Batch receipt not passed')
            observed=[initial[canonical(by_name[name])]['row'] for name in batch['case_keys']]
            episodes=[v['episode'] for v in observed];epochs=[v['snapshot']['epoch'] for v in observed]
            require(all(a<b for a,b in zip(episodes,episodes[1:])) and all(a<b for a,b in zip(epochs,epochs[1:])),'Batch reset epoch/episode did not advance')
            require(prior_episode is None or episodes[0]>prior_episode,'Server episode did not advance across batches');prior_episode=episodes[-1];batch_count+=1
    bootstrap=[]
    for suite in proof.SUITES:
        for task in range(10):
            for init in range(5):
                ref=initial[('fresh18',suite,task,init)];case_gpus={initial[(arm,suite,task,init)]['gpu_uuid'] for arm in proof.ARMS}
                require(len(case_gpus)==1,'Paired arms ran on different physical GPUs')
                for arm in proof.ARMS:
                    if arm=='fresh18':continue
                    item=initial[(arm,suite,task,init)]
                    require(item['observation_hash']==ref['observation_hash'] and item['init_vector_hash']==ref['init_vector_hash'] and item['row']['input_hash']==ref['row']['input_hash'] and item['row']['seed']==ref['row']['seed'],'Cross-arm initial state mismatch')
                    metrics={}
                    for field in ('normalized_actions','chunk_actions'):
                        delta=np.abs(np.asarray(item['row'][field],dtype=np.float64)-np.asarray(ref['row'][field],dtype=np.float64));metrics[field]=dict(max_abs=float(delta.max()),mae=float(delta.mean()))
                        require(metrics[field]['max_abs']<=1e-5 and metrics[field]['mae']<=1e-6,'Bootstrap numeric tolerance exceeded')
                    bootstrap.append(dict(configuration=arm,suite=suite,task=task,init=init,metrics=metrics))
    require(len(bootstrap)==600,'Expected600 bootstrap comparisons')
    stat=read(P/'statistics.json');require(stat['status']=='complete' and stat['input_sha256']==sha(R/'normalized_cases.json') and stat['statistics_source_sha256']==sha(R/'statistics.py'),'Statistics source/input/complete mismatch')
    require(stat['new192_sensitivity']['status']=='complete','new192 sensitivity missing')
    for row in stat['configurations']:
        subset=[c for c in cases if c['configuration']==row['configuration'] and (row['scope']=='overall' or c['suite']==row['scope'])]
        require(row['valid_cases']==len(subset) and row['successes']==sum(c['success'] for c in subset),'Statistics counts disagree with raw cases')
    stats_module=load('closedloop200_statistics_check',R/'statistics.py')
    require(stat['method']['seed']==20261025 and stat['method']['bootstrap_draws']==10000,'Prospective RNG/draw settings changed')
    for table,new_only in ((stat,False),(stat['new192_sensitivity'],True)):
        selected=[c for c in cases if not(new_only and c['task'] in (0,5) and c['init']==0)]
        index={(c['configuration'],c['suite'],c['task'],c['init']):c for c in selected}
        require(len(selected)==(768 if new_only else 800),'Statistical sample size differs')
        for row in table['configurations']:
            subset=[c for c in selected if c['configuration']==row['configuration'] and (row['scope']=='overall' or c['suite']==row['scope'])]
            require(row['valid_cases']==len(subset) and row['successes']==sum(c['success'] for c in subset),'Full/new192 success counts differ')
            require(row['wilson95']==stats_module.wilson(row['successes'],len(subset)),'Wilson arithmetic differs')
        require(len(table['comparisons'])==30,'Incomplete paired comparison table')
        for row in table['comparisons']:
            ref=[c for c in selected if c['configuration']==row['reference'] and (row['scope']=='overall' or c['suite']==row['scope'])]
            paired=[(c,index[(row['candidate'],c['suite'],c['task'],c['init'])]) for c in ref]
            b=sum(v['success'] and not u['success'] for u,v in paired);c=sum(u['success'] and not v['success'] for u,v in paired)
            require(row['n']==len(paired) and row['candidate_only_success_b']==b and row['reference_only_success_c']==c,'Paired wins/losses differ')
            require(row['paired_success_rate_difference']==(b-c)/len(paired) and row['exact_mcnemar_two_sided_p']==stats_module.exact_mcnemar(b,c),'Paired delta/exact p differs')
            for field in ('task_cluster_percentile95','episode_percentile95_sensitivity'):
                bounds=row[field];require(len(bounds)==2 and -1<=bounds[0]<=bounds[1]<=1,'Invalid bootstrap interval')
        require(all(v==8 for v in stat['new192_sensitivity']['excluded_pilot_successes'].values()),'Pilot outcomes do not match declared successful8/arm')
    return dict(status='passed',pending=[],cpu_only=True,created_utc=datetime.now(timezone.utc).isoformat(),checks=CHECKS,
        counts=dict(cases=800,new=768,reused=32,requests=requests,chunks=len(chunks),passed_batches=batch_count,bootstrap_comparisons=600,frozen_sources=len(sources)),
        source_hashes=sources,input_normalized_sha256=sha(R/'normalized_cases.json'),reuse_proof_sha256=sha(R/'reuse32.json'),
        statistics_sha256=sha(P/'statistics.json'),cases=cases,bootstrap_comparisons=bootstrap,
        valid_task_failures=[c for c in cases if not c['success']],technical_attempts=technical,excluded_attempts=excluded,
        evidence_limits=['No GPU/model/environment execution by auditor; no videos scanned.','Checkpoint identity uses recorded path,size,mtime; full weight bytes not hashed.','All saved input/action file hashes checked; initial observation content digest recomputed; no raw GPU noise-byte hash.','Original model seed and frozen generator/protocol checked; free trajectories need not match.','Statistics include200 main and192 pilot-excluded sensitivity; pilot reuse is not an independent blinded confirmation.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=R/'final_audit.json');args=parser.parse_args()
    # No failed final receipt is written just because a caller launched early.
    if read(R/'dispatch_execution.json')['status']!='passed':raise SystemExit('Not terminal: final raw audit not started')
    try:result=audit()
    except Exception:result=dict(status='failed',pending=['See traceback'],cpu_only=True,checks=CHECKS,traceback=traceback.format_exc())
    result['audit_source_sha256']=sha(Path(__file__))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result.get(key) for key in ('status','pending','checks','counts','traceback')}))
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
