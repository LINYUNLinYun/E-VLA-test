"""CPU-only provenance proof for 32 existing closed-loop episodes; no reruns."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'results/idea2/fresh_tail_20261025/closedloop_gpu2_eglmap'
ARMS = {'fresh18': (0, 18, 0), 'sm16_tail0': (16, 3, 0),
        'sm32_tail0': (32, 3, 0), 'sm32_tail2': (32, 3, 2)}
SUITES = ['libero_object', 'libero_spatial', 'libero_goal', 'libero_10']
LIMITS = dict(zip(SUITES, [280, 220, 300, 520]))


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def protocol(p, arm, require_actual):
    sm, lead, tail = ARMS[arm]
    expected = dict(num_steps=10, H=10, R=5, cache_slots=2, interval=1,
        execution='progressive_cuda_graph', next_encoder=False, lookahead=False,
        host_workers=0, preparation_core='PreparedModulationCore', lead=lead,
        fresh_tail_steps=tail, approximation=arm != 'fresh18',
        mode='fresh18' if arm == 'fresh18' else 'lead3')
    for key, value in expected.items():
        require(p.get(key) == value, f'Protocol mismatch {key}: {p.get(key)!r}')
    require('pi05=True' in p['config'] and "dtype='bfloat16'" in p['config'], 'Model precision/config changed')
    resource = p['resource_adapter']; requested = resource.get('requested', resource)
    require(requested['ae_sm_min'] == sm and requested['encoder_after_step'] == 1, 'Requested SM/encoder schedule differs')
    layouts = resource['actual_resources']
    require(not sm or not require_actual or bool(layouts), 'Missing actual SM allocation')
    require(all(r['sm_counts'] == [sm, 128-sm] for r in layouts) if sm else not layouts, 'Actual SM allocation differs')


def main():
    plan = read(SOURCE / 'plan.json'); execution = read(SOURCE / 'execution.json')
    launch = read(SOURCE / 'launch_receipt.json'); egl = read(SOURCE / 'egl_mapping_receipt.json')
    universe = read(OUT / 'case_universe.json')
    spec = importlib.util.spec_from_file_location('reuse_original_client', ROOT / 'preparations/idea1_v2_1/runtime_client.py')
    client = importlib.util.module_from_spec(spec); spec.loader.exec_module(client)
    current = {name: sha(ROOT/name) for name in plan['source_sha256']}
    require(current == plan['source_sha256'], 'Frozen source mismatch')
    require(execution['status'] == launch['status'] == 'passed', 'Incomplete parent execution')
    require(launch['exit_code'] == execution['server_exit_code'] == launch['server_exit_code'] == 0, 'Unclean process exit')
    require(execution['source_unchanged'] and launch['source_unchanged'], 'Source changed in old execution')
    require(egl['selected_egl_device'] == plan['egl_device'] == launch['egl_device'] == 1, 'Incorrect old EGL mapping')
    require(egl['mapping']['1'] == egl['requested_physical_gpu'] == 2, 'EGL physical GPU mismatch')
    require(sha(ROOT/'preparations/idea1_v3_3/egl_mapping.py') == egl['source_sha256'], 'EGL mapper source changed')
    require([plan[k] for k in ('D','H','R','env_seed','settling_steps')] == [10,10,5,7,10], 'Old global protocol differs')
    require(plan['suite_step_limits'] == LIMITS == universe['task_limits'], 'Task limits differ')
    require(universe['conditions'] == list(ARMS) and universe['env_seed'] == 7 and universe['settling_steps'] == 10, 'New universe differs')
    require(len(universe['cases']) == 200 and universe['case_count_per_condition'] == 200, 'New universe size differs')
    checkpoint = Path(plan['checkpoint']); files = {}
    for name, old in plan['checkpoint_files'].items():
        f = checkpoint/name; stat = f.stat(); actual = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        require(old == actual, f'Checkpoint metadata differs: {name}')
        files[name] = dict(recorded=old, current=actual, matches=True)
    require(checkpoint.resolve() == (ROOT/'shared/checkpoints/openpi/pi05_libero_pytorch_idea2').resolve(), 'Checkpoint path differs')
    selected = []; initial_by_case = {}
    for arm in ARMS:
        for suite_index, suite in enumerate(SUITES):
            for task in (0, 5):
                case_id = suite_index * 50 + task * 5
                candidate = [c for c in execution['cases'] if (c['label'], c['suite'], c['task'], c['init']) == (arm, suite, task, 0)]
                require(len(candidate) == 1, 'Ambiguous source case')
                c = candidate[0]; directory = SOURCE/c['relative_directory']
                identity = read(directory/'identity.json'); complete = read(directory/'complete.json')
                result = read(directory/'result.json'); smoke = read(directory/'smoke_receipt.json')
                native = read(directory/'native_protocol.json'); env = read(directory/'environment_protocol.json')
                infers = [json.loads(line) for line in (directory/'infer.jsonl').read_text().splitlines()]
                telemetry = [json.loads(line) for line in (directory/'telemetry.jsonl').read_text().splitlines()]
                require(c['status'] == complete['status'] == 'valid' and result['valid'] and complete['valid'], 'Invalid episode')
                require(smoke['status'] == 'passed' and c['client_exit_code'] == 0 and smoke['drain']['ok'] and c['parent_drain']['ok'], 'Exit/drain failure')
                require(c['result'] == complete, 'Parent and episode completion disagree')
                require(result['success'] == complete['success'] and complete['steps'] == result['actions'] <= LIMITS[suite], 'Outcome/limit mismatch')
                require(result['termination'] in ('success', 'timeout') and result['H'] == 10 and result['R'] == 5, 'Unexpected termination/horizon')
                require((identity['suite'], identity['task'], identity['init'], identity['env_seed']) == (suite,task,0,7), 'Environment identity differs')
                require(not identity['skip'] and identity['model'] == 'pi05', 'Model/skip path differs')
                init_file = Path(identity['init_file']); bddl = Path(str(init_file).replace('/init_files/','/bddl_files/').replace('.pruned_init','.bddl'))
                require(sha(init_file) == identity['init_file_sha256'] and sha(bddl) == identity['bddl_sha256'], 'Environment asset bytes changed')
                require(env['loaded_client_sha256'] == current['preparations/idea1_v2_1/runtime_client.py'], 'Loaded client differs')
                require(env['control_frequency_hz'] == 20 and result['control_hz'] == 20, 'Control frequency differs')
                require(len(infers) == result['predictions'] == complete['predictions'] and len(telemetry) == result['actions']+1, 'Incomplete telemetry')
                require(telemetry[0]['when'] == 'initial' and abs(telemetry[0]['sim_time_s']-.5) < 1e-9, 'Initial settle time differs')
                require(not c['gpu_before']['graphics_processes'] and not c['gpu_after']['graphics_processes'], 'Renderer existed outside client')
                render = c['render_gpu_verified']['snapshot']
                require(render['uuid'] == c['gpu_before']['uuid'] == c['gpu_after']['uuid'] == plan['gpu_uuid'], 'Render GPU mismatch')
                require(len(render['graphics_processes']) == 1, 'Renderer attribution ambiguous')
                require({p['pid'] for p in c['gpu_before']['processes']} == {p['pid'] for p in render['processes']} == {p['pid'] for p in c['gpu_after']['processes']}, 'Compute PID guard changed')
                for p in (native, env['raw_health']['protocol'], c['configure']['protocol']): protocol(p,arm,False)
                request_proofs = []
                for index, row in enumerate(infers):
                    protocol(row['protocol'], arm, True)
                    require(row['ok'] and row['request'] == row['replan'] == index and row['bootstrap'] == (index==0), 'Request/reset sequence differs')
                    require(row['seed'] == client.model_seed(suite,task,0,index), 'Model seed differs')
                    require(row['current_observation_id'] == row['new_prefix_observation_id'] == row['suffix_state_id'] == row['reference_state_id'] == index, 'Current source mapping differs')
                    require(row['old_prefix_observation_id'] == max(0,index-1), 'Old source mapping differs')
                    f = row['frontiers']; require(len(f)==10 and f==sorted(f) and all(ARMS[arm][1]<=v<=18 for v in f), 'Invalid frontier')
                    require(row['per_iteration_new_layer_counts'] == f and row['per_iteration_old_layer_counts'] == [18-v for v in f], 'Layer source mapping differs')
                    if index==0 or arm=='fresh18': require(f==[18]*10, 'Fresh/bootstrap is stale')
                    if ARMS[arm][2]: require(f[-ARMS[arm][2]:]==[18]*ARMS[arm][2], 'Fresh tail incomplete')
                    require(not row['next_encoder'] and not row['lookahead'] and row['graph_kv_buffers']==2 and row['pool']['live_snapshots']<=2, 'Lookahead/two-slot mismatch')
                    require(not index or row['graph_setup_ms']==0, 'Steady graph recapture')
                    require(row['generated_length']==10 and row['submitted_length']==5 and 0<row['executed_length']<=5, 'Action lengths differ')
                    input_path=directory/f'inputs/replan{index:04d}.npz'; action_path=directory/f'chunk{index:04d}.npz'
                    require(sha(input_path)==row['input_file_sha256'] and sha(action_path)==row['prediction_sha256'], 'Saved request bytes changed')
                    with np.load(input_path,allow_pickle=False) as arrays:
                        raw={k:arrays[k] for k in arrays.files if k not in ('instruction','seed')}
                        instruction=str(arrays['instruction']);seed=int(arrays['seed'])
                        require(client.input_digest(raw,instruction)==row['input_hash'] and seed==row['seed'], 'Input digest differs')
                        raw_digest=client.value_digest(raw)
                    with np.load(action_path,allow_pickle=False) as arrays: actions=arrays['actions']
                    require(actions.shape==(10,7) and np.isfinite(actions).all() and np.array_equal(actions,np.asarray(row['chunk_actions'],dtype=np.float32)), 'Action record differs')
                    normalized=np.asarray(row['normalized_actions']);require(normalized.shape==(10,32) and np.isfinite(normalized).all(), 'Invalid normalized action')
                    request_proofs.append(dict(replan=index,seed=seed,input_sha256=row['input_hash'],input_file_sha256=row['input_file_sha256'],prediction_file_sha256=row['prediction_sha256'],action_array_sha256=client.value_digest(actions),observation_array_sha256=raw_digest,current_source=index,old_source=max(0,index-1),frontiers=f))
                first=request_proofs[0]
                require(first['input_sha256']==complete['first_input_hash'], 'Initial completion digest differs')
                initial=dict(init_vector_sha256=identity['init_vector_hash'],observation_sha256=first['observation_array_sha256'],input_sha256=first['input_sha256'],action_array_sha256=first['action_array_sha256'],seed=first['seed'])
                if case_id in initial_by_case:require(initial_by_case[case_id]==initial, 'Cross-arm initial observations/actions differ')
                else:initial_by_case[case_id]=initial
                source_request=SOURCE/'requests'/f'case{c["case"]:02d}_{arm}.json'
                selected.append(dict(arm=arm,label=arm,case_id=case_id,suite=suite,task=task,init=0,
                    source_case_id=c['case'],source_directory=str(directory.resolve()),source_request=str(source_request.resolve()),source_execution=str((SOURCE/'execution.json').resolve()),
                    source_gpu_uuid=plan['gpu_uuid'],compatible=True,success=result['success'],steps=result['actions'],requests=len(infers),
                    source_artifact_sha256={name:sha(directory/name) for name in ['identity.json','complete.json','result.json','smoke_receipt.json','native_protocol.json','environment_protocol.json','infer.jsonl','telemetry.jsonl']},
                    proof=dict(protocol=True,seed_every_request=True,source_mapping_every_request=True,two_slots=True,within_episode_recapture_zero=True,all_input_action_file_hashes=True,all_input_digests=True,completion_exit_render_guards=True,initial=initial,request_evidence=request_proofs)))
    require(len(selected)==32 and len({(r['arm'],r['case_id']) for r in selected})==32, 'Wrong reuse count')
    document=dict(schema_version=1,status='passed',created_utc=datetime.now(timezone.utc).isoformat(),gpu_launched=False,
        source_root=str(SOURCE.resolve()),source_files_sha256={name:sha(SOURCE/name) for name in ['plan.json','execution.json','launch_receipt.json','egl_mapping_receipt.json']},
        new_case_universe_sha256=sha(OUT/'case_universe.json'),case_id_rule='suite_index * 50 + task * 5 + init',suite_order=SUITES,
        selected_case_ids=sorted(initial_by_case),selected_episodes=32,selected_cases=selected,
        required_proof=['same checkpoint absolute path and recorded bytes/mtime_ns','all recorded production/model/client/adapter source SHA256 match current','D10 H10 R5, seed namespace 7, env seed7, settle10 and suite limits','per-request lead/SM/fresh-tail/approximation and old/current source mappings','first observation/input/action equality across arms and all saved request hashes','valid complete/result plus clean client/server exit and EGL render guards'],
        current_source_hashes=current,checkpoint_identity=dict(path=str(checkpoint),files=files,config_sha256=sha(checkpoint/'config.json'),scope='Same path and file size/mtime metadata as old run; no newly computed full weights hash is claimed.'),
        expected_new_protocol=dict(D=10,H=10,R=5,env_seed=7,model_seed_namespace='idea1-v21/7/{suite}/{task}/{init}/{replan}',settling_steps=10,limits=LIMITS,next_encoder=False,arms=ARMS),
        integration_rules=['Only these 32 arm/case pairs may be skipped; do not expand reuse by task or suite alone.','Retain source_directory rather than rewriting old artifacts.','New runner must preserve expected_new_protocol and source hashes; changes invalidate this reuse proof.','Reused outcomes count toward case coverage/success; do not pool cross-run GPU latency or use old elapsed time as new-run throughput.','Source trajectories are actual measured outcomes; progressive schedules may vary on rerun, so compatibility is not a promise of bitwise repeated trajectories.'],
        limitations=['Full weight bytes were not rehashed; existing checkpoint file metadata is the identity strength of the original receipt.','GPU noise tensor bytes are not exposed; every request seed and frozen generator source are verified.'])
    (OUT/'reuse32.json').write_text(json.dumps(document,indent=2)+'\n')
    lines=['# Reuse proof: 32 existing episodes','', 'Status: **passed**. CPU-only verification; no environment or GPU was started.','',
        'Reuse 8 cases per arm: fresh18, sm16_tail0, sm32_tail0, sm32_tail2. Case IDs: 0, 25, 50, 75, 100, 125, 150, 175. The four suites follow object, spatial, goal, 10 order; each uses task 0/5, init 0.','',
        'All recorded source hashes and checkpoint file size/mtime metadata match. D10/H10/R5, environment/model seed rules, settle steps, limits, SM/lead/tail semantics, old/new source identities, all input/action hashes, completions and clean exit/render guards pass. Initial observation, input, and decoded action hashes agree across all four arms for each of the eight cases.','',
        'The JSON contains absolute source directories and per-request evidence. Keep these original artifacts unchanged. Reuse success/coverage only; do not treat their old elapsed time as throughput of the new run. No full weight hash or GPU noise-byte hash is claimed.','',
        '| Arm | Case | Suite | Task | Init | Success | Steps | Requests |','|---|---:|---|---:|---:|---|---:|---:|']
    lines += [f'| {r["arm"]} | {r["case_id"]} | {r["suite"]} | {r["task"]} | 0 | {r["success"]} | {r["steps"]} | {r["requests"]} |' for r in selected]
    (OUT/'reuse32.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='passed',episodes=len(selected),requests=sum(r['requests'] for r in selected),source_hashes=len(current),manifest=str(OUT/'reuse32.json'))))


if __name__ == '__main__':
    main()
