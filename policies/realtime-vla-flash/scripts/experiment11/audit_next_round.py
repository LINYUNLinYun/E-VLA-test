"""Independent raw-record completion checks; old completion flags are not enough."""
import argparse
from collections import Counter
import csv
import json
import hashlib
from pathlib import Path
import re
import sqlite3

import numpy as np

from scripts.experiment11.protocol import RESULTS, ROOT, SUITES
from scripts.experiment10.common import dump_json, sha256, json_lines
from scripts.experiment8.render_xyz_video import validate_video
from scripts.experiment11.shadow_worker import _input_digest


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_hashes(path):
    record = json.loads(path.read_text())
    require(record.get('complete'),f'Incomplete component: {path}')
    for field in ('source_hashes','output_hashes','outputs'):
        for name,digest in record.get(field,{}).items():
            target = Path(name)
            if not target.is_absolute():
                target = path.parent/target
            require(target.is_file() and sha256(target)==digest,f'Hash mismatch: {target}')
    return record


def ledger(path,limit):
    require(path.is_file(),f'Missing ledger: {path}')
    connection = sqlite3.connect(f'file:{path}?mode=ro',uri=True)
    try:
        require(connection.execute('select budget from config where id=1').fetchone()[0]==limit,'Budget limit changed')
        rows = connection.execute('select identity,status,details from calls').fetchall()
    finally:
        connection.close()
    statuses = Counter(status for _,status,_ in rows)
    require(len(rows)<=limit and not statuses['reserved'],'Ledger over cap or unfinished invocation')
    require(all(s in ('complete','failed') for s in statuses),'Unknown ledger status')
    return [(json.loads(identity),status,json.loads(details)) for identity,status,details in rows],dict(statuses)


def last_json_record(path):
    """Read the actual terminal telemetry without loading full trajectories."""
    with path.open('rb') as stream:
        stream.seek(0,2);position=stream.tell();tail=b''
        while position>0:
            size=min(position,8192);position-=size;stream.seek(position)
            tail=stream.read(size)+tail
            lines=tail.splitlines()
            nonempty=[line for line in lines if line.strip()]
            if len(nonempty)>=2 or position==0:
                return json.loads(nonempty[-1])
    raise RuntimeError(f'Empty terminal telemetry: {path}')


def audit_reused_success_matrix(path):
    rows=list(csv.DictReader(path.open()))
    require(len(rows)==6200,'Success matrix episode count changed')
    bygroup={}
    for row in rows:
        key=row['condition'],row['suite']
        bygroup.setdefault(key,[]).append(row)
    require(len(bygroup)==124,'Condition/suite coverage changed')
    hashes={};checked=0
    for (condition,suite),group in bygroup.items():
        require(len(group)==50 and {(int(r['task_id']),int(r['episode_idx'])) for r in group}==
                {(t,i) for t in range(10) for i in range(5)},'Reused suite init coverage changed')
        run=RESULTS/'closed_loop'/condition/suite/'run';episode_path=run/'episode_log.json'
        episodes=json.loads(episode_path.read_text());hashes[str(episode_path)]=sha256(episode_path)
        lookup={(int(r['task_id']),int(r['episode_idx'])):r for r in episodes}
        require(len(lookup)==len(episodes)==50,'Reused raw episode log malformed')
        for row in group:
            episode=lookup[int(row['task_id']),int(row['episode_idx'])]
            terminal=last_json_record(run/episode['telemetry_path'])
            require(bool(episode['success'])==(row['success']=='1')==bool(terminal['success']),
                    'CSV,episode log and terminal environment success disagree')
            require(episode['seed']==7 and episode['num_steps_wait']==10,'Reused environment protocol changed')
            require(int(terminal['step'])==episode['executed_action_count'],'Terminal step/action count mismatch')
            checked+=1
    return {'raw_episode_successes_checked':checked,'episode_log_hashes':hashes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt-id',default='round2_20261011')
    parser.add_argument('--retry-id',default='warmup_retry1')
    parser.add_argument('--report',type=Path)
    args = parser.parse_args()
    base = RESULTS/'analysis/next_round'/args.attempt_id
    tests=verify_hashes(base/'cpu_tests.json')
    require(tests['tests_run']>=24 and tests['failures']==0 and tests['errors']==0 and tests['skipped']==0,
            'Required CPU test evidence missing')
    work=verify_hashes(base/'internal_work_accounting.json')
    components = {}
    for name in ('task_type_results','failure_events','shadow_risk','task_type_probe','shadow_phase_offsets'):
        path = base/name/'completion_audit.json'
        components[name] = verify_hashes(path)
    require(components['task_type_results']['source_episodes']==6200,'Task results incomplete')
    require(components['failure_events']['episodes']==800,'Failure cohort coverage incomplete')
    require(components['shadow_risk']['prediction_records']==27916,'Shadow record coverage incomplete')
    require(components['shadow_risk']['offsets']==list(range(1,51)),'Missing predicted offsets')
    require(components['shadow_risk']['prefixes']==[1,2,3,4,5,8,12],'Missing short prefixes')
    require(components['shadow_phase_offsets']['offsets']==list(range(1,51)),'Missing phase-offset coverage')
    reused_matrix=audit_reused_success_matrix(RESULTS/'analysis/closed_loop_d3/episode_results.csv')
    for component in components.values():
        for key in ('new_predictions','new_model_predictions','environment_steps','new_environment_steps'):
            if key in component:
                require(component[key]==0,f'Unexpected analysis-side computation: {key}')
    # Independently traverse original and retry outputs; choose only valid files.
    directory = RESULTS/'task_type_layer_probe'/args.attempt_id
    manifest = json.loads((directory/'manifest.json').read_text())
    samples = {s['sample_id']:s for s in manifest['samples']}
    require(len(samples)==723 and len(manifest['missing_phase_cells'])==77,'Frozen sample coverage changed')
    required = {(sid,layer) for sid in samples for layer in range(1,18)}
    found, numeric,parity,kvonly = set(),set(),0.,set()
    job_paths = []
    for suite in SUITES:
        for task in range(10):
            original = directory/suite/f'task{task:02d}'
            retry = directory/'retries'/args.retry_id/suite/f'task{task:02d}'
            paths = [p for p in (original,retry) if (p/'validated.json').is_file()]
            require(len(paths)==1,f'Missing or ambiguous validated job: {suite}/{task}')
            path = paths[0];job_paths.append(str(path))
            marker = json.loads((path/'validated.json').read_text())
            require(marker.get('complete') and marker['environment_steps']==0,'Invalid probe marker')
            require(marker['suite']==suite and marker['task_id']==task and marker['layers']==list(range(1,18)),
                    'Probe job identity/layers changed')
            require(marker['manifest_sha256']==sha256(directory/'manifest.json'),'Probe manifest changed')
            require(marker['predictions_sha256']==sha256(path/'predictions.jsonl'),'Prediction hash mismatch')
            require(marker['activation_sha256']==sha256(path/'activation_statistics.json'),'Activation hash mismatch')
            require(not (path/'predictions.jsonl.partial').exists(),'Unresolved partial in completed job')
            parity = max(parity,marker['full_parity_max_abs_error'])
            count = 0
            for row in json_lines(path/'predictions.jsonl'):
                key = row['sample_id'],row['layer']
                require(key in required and key not in found,'Duplicate or unexpected prediction')
                found.add(key);count+=1;sample=samples[key[0]]
                require(row['suite']==suite and row['task_id']==task,'Prediction stored under wrong job')
                for field in ('suite','task_id','episode_idx','phase','input_sha256','noise_sha256','checkpoint_sha256'):
                    require(row[field]==sample[field],f'Identity mismatch: {field}')
                full,variant = np.asarray(row['full_actions']),np.asarray(row['variant_actions'])
                require(full.shape==(50,7) and variant.shape==(50,7),'Action shape mismatch')
                require(np.isfinite(full).all() and np.isfinite(variant).all(),'Nonfinite actions')
                require(np.max(np.abs(full-np.asarray(sample['full_actions'])))<=1e-5,'Full saved-output parity failed')
                require({m['horizon'] for m in row['metrics']}=={1,2,3,4,5,8,12},'Missing functional horizon')
            require(count==marker['prediction_records'],'Marker/raw record count mismatch')
            for row in json.loads((path/'activation_statistics.json').read_text())['rows']:
                if row['layer']==18:
                    require(row['tensor']=='kv_only','L18 falsely scored as hidden-update layer')
                    require(row['sample_id'] not in kvonly,'Duplicate KV-only marker')
                    kvonly.add(row['sample_id'])
                    continue
                require(row['sample_id'] in samples,'Unknown activation sample')
                sample=samples[row['sample_id']]
                for field in ('suite','task_id','episode_idx','phase'):
                    require(row[field]==sample[field],'Activation identity mismatch')
                key = row['sample_id'],row['layer'],row['modality'],row['tensor']
                require(key not in numeric,'Duplicate activation statistic');numeric.add(key)
    require(found==required and parity<=1e-5,'Missing probe cell or Full parity failure')
    expected_numeric = {(sid,l,m,t) for sid in samples for l in range(1,18) for m in ('agent','wrist','language')
                        for t in ('H_in','H_attn','H_out','delta_attn','delta_ffn','delta_block')}
    require(numeric==expected_numeric and kvonly==set(samples),'Missing numerical/KV-only coverage')
    rows,statuses = ledger(directory/'call_budget.sqlite',14400)
    require(work['ledger_sha256']==sha256(directory/'call_budget.sqlite') and work['charged_logical_calls']==len(rows),
            'Internal work accounting is not the final ledger snapshot')
    charged_single = {(r.get('sample_id'),r.get('layer')) for r,s,_ in rows if s=='complete' and r.get('kind')=='single_layer'}
    charged_full = {r.get('sample_id') for r,s,_ in rows if s=='complete' and r.get('kind')=='full_activity'}
    require(required<=charged_single and set(samples)<=charged_full,'Formal output lacks charged invocation')
    # V14 raw paired identities must match Full trajectories, not just counts.
    v14root = RESULTS/'shadow_supplement_v14'/args.attempt_id/'V14_A03'
    counts = [1466,1128,1218,3167];paired=set();expected_pairs=set();input_artifacts_checked=0;reference={}
    for suite,count in zip(SUITES,counts):
        marker=json.loads((v14root/suite/'validated.json').read_text())
        prediction=v14root/suite/'predictions.jsonl'
        require(marker['predictions_sha256']==sha256(prediction),'V14 prediction hash mismatch')
        require(marker['environment_steps_executed_by_variant']==0 and marker['skip_layers']==[3,4,9,11],'V14 protocol mismatch')
        require(marker['denoise_steps']==3 and marker['action_horizon']==50 and marker['replan_execute_prefix']==5,'V14 action settings changed')
        full_run=RESULTS/'closed_loop/Full18_D3_R5_FORMAL'/suite/'run'
        episodes=json.loads((full_run/'episode_log.json').read_text())
        require(len(episodes)==50 and {(int(e['task_id']),int(e['episode_idx'])) for e in episodes}==
                {(t,i) for t in range(10) for i in range(5)},'Full suite task/init incomplete')
        full_rows={}
        for episode in episodes:
            for row in json_lines(full_run/episode['infer_path']):
                key=(suite,int(episode['task_id']),int(episode['episode_idx']),int(row['infer_id']))
                require(key not in full_rows,'Duplicate Full source point')
                full_rows[key]=row;expected_pairs.add(key)
        records=json_lines(prediction);require(len(records)==count,'V14 suite coverage mismatch')
        for row in records:
            key=(suite,int(row['task_id']),int(row['episode_idx']),int(row['replan_id']))
            require(key not in paired,'Duplicate V14 point');paired.add(key)
            require(row['variant_id']=='V14_A03' and
                    row['environment_action_source']=='Full18 only; this worker never calls env.step','V14 execution detected')
            require(key in full_rows,'V14 point not on Full source trajectory')
            full=full_rows[key];info=full['experiment11']
            # This row field locates the Full source observation; it is not the
            # number of environment actions performed by this offline worker.
            require(row['environment_action_step']==int(full['frame_idx']),'Shadow Full trajectory step mismatch')
            require(np.array_equal(np.asarray(row['full_actions'],np.float32),np.asarray(full['chunk_actions'],np.float32)),
                    'Shadow Full reference differs from real source')
            require(row['full_input_sha256']==info['full_input_sha256'] and row['noise_sha256']==info['noise_sha256'],
                    'Shadow input/noise differs from Full source')
            reference[key]=(row['full_input_sha256'],row['noise_sha256'],
                            hashlib.sha256(np.asarray(row['full_actions'],np.float32).tobytes()).hexdigest(),
                            int(row['environment_action_step']))
            with np.load(row['full_input_path'],allow_pickle=False) as data:
                require(_input_digest(data,str(data['prompt'].item()),suite)==row['full_input_sha256'],'Cached input bytes changed')
                require(hashlib.sha256(np.ascontiguousarray(data['noise']).tobytes()).hexdigest()==row['noise_sha256'],
                        'Noise bytes changed')
            input_artifacts_checked+=1
    require(paired==expected_pairs,'V14 full-trajectory coverage incomplete')
    for variant,skip in [('V16_A02',[4,10]),('V15_A08',[2,5,9]),('V13_Custom',[3,4,7,12,15])]:
        keys=set()
        for suite,count in zip(SUITES,counts):
            path=RESULTS/'shadow'/variant/suite
            marker=json.loads((path/'validated.json').read_text())
            require(marker['predictions_sha256']==sha256(path/'predictions.jsonl'),'Historical shadow changed')
            require(marker['environment_steps_executed_by_variant']==0,'Historical shadow executed variant')
            records=json_lines(path/'predictions.jsonl');require(len(records)==count,'Historical shadow suite missing records')
            for row in records:
                key=(suite,int(row['task_id']),int(row['episode_idx']),int(row['replan_id']))
                signature=(row['full_input_sha256'],row['noise_sha256'],
                           hashlib.sha256(np.asarray(row['full_actions'],np.float32).tobytes()).hexdigest(),
                           int(row['environment_action_step']))
                require(key in reference and signature==reference[key] and key not in keys,'Historical shadow pair mismatch')
                require(row['variant_id']==variant and row['skip_layers']==skip and
                        row['environment_action_source']=='Full18 only; this worker never calls env.step',
                        'Historical shadow variant/protocol mismatch')
                actions=np.asarray(row['variant_actions'])
                require(actions.shape==(50,7) and np.isfinite(actions).all(),'Invalid historical shadow actions')
                keys.add(key)
        require(keys==expected_pairs,'Historical shadow trajectory incomplete')
    v14calls,v14statuses=ledger(v14root.parent/'call_budget.sqlite',7000)
    require(len(v14calls)==6983 and v14statuses=={'complete':6983},'V14 actual budget/accounting mismatch')
    # Decode all new case videos independently of renderer's completion flag.
    gallery=base/'failure_events/event_clips'
    verify_hashes(gallery/'gallery_audit.json')
    cases=list(csv.DictReader((base/'failure_events/diagnostic_video_examples.csv').open()))
    markers=sorted(gallery.glob('[0-9][0-9]_*.json'))
    require(len(cases)==len(markers)==23,'Missing diagnostic case')
    for marker in markers:
        record=json.loads(marker.read_text());video=Path(record['video_path'])
        require(sha256(video)==record['video_sha256'],'Changed video')
        require(sum(r['event_frame'] for r in record['frame_index'])==1,'Event frame not unique')
        validate_video(video,len(record['frame_index']))
    verify_hashes(base/'shadow_risk/supplemental_figures/completion_audit.json')
    report_checked=False
    if args.report:
        require(args.report.is_file(),'Requested report missing')
        text=args.report.read_text()
        require(len(text)>1000,'Report is not substantive')
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
            if target.startswith(('http:','https:','#')):
                continue
            target=target.split('#',1)[0].strip('<>')
            target=re.sub(r':\d+$','',target)
            path=Path(target) if target.startswith('/') else args.report.parent/target
            require(path.exists(),f'Broken report link: {path}')
        report_checked=True
    record={'experiments_complete':True,'goal_complete':False,'report_checked':report_checked,
            'requested_report_model':'gpt-6.1-sol','requested_reasoning':'xhigh',
            'probe_samples':len(samples),'probe_records':len(found),'numeric_records':len(numeric),
            'probe_full_parity_max_abs':parity,'probe_budget':{'charged':len(rows),'limit':14400,'statuses':statuses},
            'v14_budget':{'charged':len(v14calls),'limit':7000,'statuses':v14statuses},
            'event_videos_decoded':len(markers),'validated_probe_jobs':job_paths,
            'cpu_tests_run':tests['tests_run'],'cpu_tests_sha256':sha256(base/'cpu_tests.json'),
            'raw_shadow_input_artifacts_checked':input_artifacts_checked,
            'reused_success_matrix':reused_matrix,
            'component_audits':{name:sha256(base/name/'completion_audit.json') for name in components},
            'report_sha256':sha256(args.report) if report_checked else None,
            'scope_warning':'Model provenance and substantive report claims require parent review; no completion from flags alone'}
    target=base/('final_handoff_audit.json' if report_checked else 'experiment_completion_audit.json')
    dump_json(target,record);print(json.dumps(record),flush=True)


if __name__=='__main__':
    main()
