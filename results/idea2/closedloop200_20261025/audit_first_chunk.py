"""One-time CPU audit of the first 80-case chunk; no runtime/model execution."""
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = ''
R = Path(__file__).resolve().parent
ROOT = R.parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


proof = load('first_chunk_reuse_helpers', R/'prove_reuse32.py')
client = load('first_chunk_original_client', ROOT/'preparations/idea1_v2_1/runtime_client.py')
read, sha, require = proof.read, proof.sha, proof.require


def main():
    chunk = R/'gpu2_chunk0'; execution = read(chunk/'execution.json'); plan = read(chunk/'plan.json')
    parent = read(R/'initial_parent_exit.json'); reuse = read(R/'reuse32.json')
    require(execution['status']=='passed' and execution['server_exit_code']==0 and parent['exit_code']==0, 'Parent/server did not finish cleanly')
    require(execution['source_unchanged'], 'Execution source drift')
    current = {name:sha(ROOT/name) for name in plan['source_sha256']}
    require(current==plan['source_sha256'], 'Source differs from completed execution')
    require(all(current[name]==value for name,value in reuse['current_source_hashes'].items()), 'Original model/client/adapter source differs from pilot')
    require(plan['reuse_manifest']['sha256']==sha(R/'reuse32.json') and reuse['status']=='passed', 'Reuse proof drift')
    require(plan['selected_task_ids']==[0,10,20,30] and len(execution['cases'])==80, 'Wrong audit scope')
    require(not plan['environment_reuse'] and [plan[k] for k in ('D','H','R','env_seed','settling_steps')]==[10,10,5,7,10], 'Environment protocol differs')
    require(plan['suite_step_limits']==proof.LIMITS, 'Step limits differ')
    require(plan['checkpoint']==reuse['checkpoint_identity']['path'], 'Checkpoint path differs')
    for name,meta in plan['checkpoint_files'].items():
        stat=(Path(plan['checkpoint'])/name).stat()
        require(meta==dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns), 'Checkpoint metadata differs')
    cases={c['case_key']:c for c in execution['cases']}; firsts={}; summaries=[]; seed_checks=0
    require(len(cases)==80 and sum(c['source_kind']=='new' for c in cases.values())==64, 'Case duplication/reuse count differs')
    for key,c in cases.items():
        directory=Path(c['directory']); identity=read(directory/'identity.json'); result=read(directory/'result.json')
        complete=read(directory/'complete.json'); smoke=read(directory/'smoke_receipt.json'); env=read(directory/'environment_protocol.json')
        rows=[json.loads(line) for line in (directory/'infer.jsonl').read_text().splitlines()]
        require(c['status']=='valid' and complete['status']=='valid' and result['valid'], 'Invalid/incomplete case')
        require(smoke['status']=='passed' and smoke['drain']['ok'], 'Episode did not drain')
        require(result['success']==complete['success']==c['result']['success'], 'Success record mismatch')
        require(result['actions']==complete['steps']==c['result']['steps']<=proof.LIMITS[c['suite']], 'Action limit mismatch')
        require(result['termination']==('success' if result['success'] else 'action_limit'), 'Unexpected outcome termination')
        require((identity['suite'],identity['task'],identity['init'],identity['env_seed'])==(c['suite'],c['task'],c['init'],7), 'Episode environment identity mismatch')
        require(identity['model']=='pi05' and not identity['skip'], 'Model skip path mismatch')
        init_file=Path(identity['init_file']);bddl=Path(str(init_file).replace('/init_files/','/bddl_files/').replace('.pruned_init','.bddl'))
        require(sha(init_file)==identity['init_file_sha256'] and sha(bddl)==identity['bddl_sha256'], 'Environment asset changed')
        require(env['loaded_client_sha256']==current['preparations/idea1_v2_1/runtime_client.py'] and env['control_frequency_hz']==20, 'Client/control protocol differs')
        proof.protocol(read(directory/'native_protocol.json'),c['arm'],False)
        with (directory/'telemetry.jsonl').open() as handle:
            telemetry_first=json.loads(next(handle)); telemetry_count=1+sum(1 for _ in handle)
        require(telemetry_count==result['actions']+1 and telemetry_first['when']=='initial' and abs(telemetry_first['sim_time_s']-.5)<1e-9, 'Environment settle/reset evidence differs')
        require(len(rows)==result['predictions']==complete['predictions'], 'Prediction count mismatch')
        first=rows[0];epoch=first['snapshot']['epoch'];episode=first['episode']
        for i,row in enumerate(rows):
            proof.protocol(row['protocol'],c['arm'],True)
            require(row['bootstrap']==(i==0) and row['request']==row['replan']==i, 'Request did not restart at zero')
            require(row['episode']==episode and row['snapshot']['epoch']==epoch, 'Episode/epoch changed within trajectory')
            require(row['seed']==client.model_seed(c['suite'],c['task'],c['init'],i), 'Model seed differs')
            require(row['current_observation_id']==row['new_prefix_observation_id']==row['suffix_state_id']==row['reference_state_id']==i, 'Current source mismatch')
            require(row['old_prefix_observation_id']==max(0,i-1), 'Old source crossed episode boundary')
            require(row['graph_kv_buffers']==2 and row['pool']['live_snapshots']<=2 and (not i or row['graph_setup_ms']==0), 'Pool or recapture violation')
            require(not row['next_encoder'] and not row['lookahead'], 'Lookahead unexpectedly enabled')
            f=row['frontiers']; require(len(f)==10 and f==sorted(f) and all(c['lead']<=v<=18 for v in f), 'Frontier invalid')
            require(row['per_iteration_new_layer_counts']==f and row['per_iteration_old_layer_counts']==[18-v for v in f], 'Layer version mapping mismatch')
            if i==0 or c['arm']=='fresh18':require(f==[18]*10, 'Bootstrap/fresh control stale')
            if c['fresh_tail_steps']:require(f[-c['fresh_tail_steps']:]==[18]*c['fresh_tail_steps'], 'Mandatory tail stale')
            seed_checks+=1
        require(first['frontiers']==[18]*10 and first['current_observation_id']==first['old_prefix_observation_id']==first['new_prefix_observation_id']==0, 'Initial stale state')
        initial_input=directory/'inputs/replan0000.npz';initial_action=directory/'chunk0000.npz'
        require(sha(initial_input)==first['input_file_sha256'] and sha(initial_action)==first['prediction_sha256'], 'Initial saved artifact hash changed')
        with np.load(initial_input,allow_pickle=False) as arrays:
            raw={k:arrays[k] for k in arrays.files if k not in ('instruction','seed')}
            require(int(arrays['seed'])==first['seed'] and client.input_digest(raw,str(arrays['instruction']))==first['input_hash'], 'Initial input/seed differs')
            observation_hash=client.value_digest(raw)
        with np.load(initial_action,allow_pickle=False) as arrays:actions=arrays['actions']
        require(np.array_equal(actions,np.asarray(first['chunk_actions'],dtype=np.float32)), 'Decoded action file differs')
        firsts.setdefault(c['case_id'],{})[c['arm']]=dict(row=first,observation_hash=observation_hash,init_vector_hash=identity['init_vector_hash'])
        if c['source_kind']=='new':
            require(len(c['attempts'])==1 and c['attempts'][0]['attempt']==0 and c['attempts'][0]['client_exit_code']==0, 'First chunk unexpectedly retried')
            require(c['bootstrap_episode']==episode and c['bootstrap_epoch']==epoch, 'Batch receipt bootstrap differs')
        summaries.append(dict(case_key=key,case_id=c['case_id'],arm=c['arm'],init=c['init'],source_kind=c['source_kind'],source_directory=str(directory),success=result['success'],steps=result['actions'],requests=len(rows),bootstrap_episode=episode,bootstrap_epoch=epoch,initial_input_sha256=first['input_hash'],initial_seed=first['seed'],first_infer_sha256=sha(directory/'infer.jsonl'),identity_sha256=sha(directory/'identity.json')))
    batch_evidence=[];prior_episode=None
    for batch in execution['batches']:
        require(batch['status']=='passed' and batch['client_exit_code']==0 and batch['attempt']==0 and batch['drain']['ok'], 'Batch failure/retry')
        receipt=read(chunk/'requests'/f'{batch["tag"]}.receipt.json');require(receipt['status']=='passed', 'Worker batch receipt failed')
        require([c['case_key'] for c in receipt['cases']]==batch['case_keys'], 'Batch order mismatch')
        observed=[firsts[cases[k]['case_id']][cases[k]['arm']]['row'] for k in batch['case_keys']]
        episodes=[r['episode'] for r in observed];epochs=[r['snapshot']['epoch'] for r in observed]
        require(all(a<b for a,b in zip(episodes,episodes[1:])) and all(a<b for a,b in zip(epochs,epochs[1:])), 'Episode/epoch did not advance within batch')
        require(prior_episode is None or episodes[0]>prior_episode, 'Server episode did not advance across batches')
        prior_episode=episodes[-1]
        require(not batch['gpu_before']['graphics_processes'] and not batch['gpu_after']['graphics_processes'] and len(batch['render_gpu_verified']['snapshot']['graphics_processes'])==1, 'Render lifetime guard failed')
        batch_evidence.append(dict(tag=batch['tag'],client_pid=batch['client_pid'],episodes=episodes,epochs=epochs,client_exit_code=0,passed=True))
    comparisons=[]
    for case_id,arms in sorted(firsts.items()):
        require(set(arms)==set(proof.ARMS), 'Missing arm for initial comparison')
        reference=arms['fresh18'];reference_row=reference['row']
        for arm,item in arms.items():
            row=item['row']; require(row['seed']==reference_row['seed'] and row['input_hash']==reference_row['input_hash'] and item['observation_hash']==reference['observation_hash'] and item['init_vector_hash']==reference['init_vector_hash'], 'Cross-arm initial input identity mismatch')
            metrics={}
            for name in ('normalized_actions','chunk_actions'):
                delta=np.abs(np.asarray(row[name],dtype=np.float64)-np.asarray(reference_row[name],dtype=np.float64))
                metrics[name]=dict(max_abs=float(delta.max()),mae=float(delta.mean()))
                require(metrics[name]['max_abs']<=1e-5 and metrics[name]['mae']<=1e-6, 'Cross-arm initial numeric tolerance exceeded')
            comparisons.append(dict(case_id=case_id,arm=arm,input_sha256=row['input_hash'],seed=row['seed'],metrics=metrics,passed=True))
    failures=[r for r in summaries if not r['success']]
    report=dict(status='passed',created_utc=datetime.now(timezone.utc).isoformat(),gpu_launched=False,scope='One targeted audit of gpu2_chunk0 only; 64 new episodes plus 16 compatible pilot episodes. No later chunks scanned.',
        source_chunk=str(chunk),source_execution_sha256=sha(chunk/'execution.json'),source_plan_sha256=sha(chunk/'plan.json'),parent_exit=parent,current_source_hashes=current,
        counts=dict(episodes=80,new=64,reused=16,batches=len(batch_evidence),initial_state_groups=len(firsts),all_request_seed_and_source_checks=seed_checks),
        tolerance=dict(max_abs=1e-5,mae=1e-6),maximum_initial_max_abs=max(v['max_abs'] for c in comparisons for v in c['metrics'].values()),maximum_initial_mae=max(v['mae'] for c in comparisons for v in c['metrics'].values()),
        batch_epoch_evidence=batch_evidence,initial_comparisons=comparisons,cases=summaries,valid_task_failures_retained=failures,
        source_path_review=['runner.run_worker_batch calls q.run_worker once per episode; scoped patch restores client.rpc and run_episode after each call.','Original runtime_client.run_episode creates OffScreenRenderEnv, seeds7, resets, applies official init, settles10, resets policy, and closes the environment in finally.','Server episode advances globally across batches. Pool epoch advances within each batch; configuration changes may reconstruct graphs/Pool and restart epoch, so global epoch monotonicity is not required.'],
        conclusions=['No evidence of cross-episode state pollution in this first chunk: every episode starts fresh at observation/request0, all later source IDs are episode-local, and all four arms have identical initial input/seed and zero decoded/normalized action difference.','All valid action_limit outcomes remain quality failures and were not retried.','Zero-episode port preflight failure in another chunk is outside this completed chunk and does not invalidate it.','This is a bounded first-chunk gate, not a claim that later trajectories must remain equal or that all later chunks were audited.'])
    (R/'first_chunk_gate.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# First chunk client-reuse gate','', 'Status: **passed**. CPU-only audit; no model/environment execution.','',
        f'Checked 80 episodes (64 new, 16 reused), {len(batch_evidence)} client batches, and {seed_checks} request seed/source records. All 20 initial states match across four arms. Decoded and normalized initial actions have max error 0 and MAE 0 (limits 1e-5 / 1e-6).','',
        'All episodes bootstrap with fresh KV, request/source IDs restart at zero, and episode/epoch advance within each batch. Server episode numbers also advance across batches; Pool epochs can restart when configuration reconstructs the Pool. Frozen source/checkpoint metadata, original seeds, settle10, H10/R5/D10, limits, clean exits and render guards pass.','',
        'Valid failures are retained, each from its single attempt:','']
    lines += [f'- `{r["case_key"]}`: action limit at {r["steps"]} steps.' for r in failures]
    lines += ['', 'No subsequent chunks were scanned. This gate supports safe Python/import reuse in the measured first chunk; it does not force later free trajectories to coincide.']
    (R/'first_chunk_gate.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='passed',counts=report['counts'],max_error=report['maximum_initial_max_abs'],valid_failures=len(failures))))


if __name__=='__main__':main()
