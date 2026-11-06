"""v2.2 adapter around the independently validated LIBERO telemetry client."""
import importlib.util, json, time, subprocess,os
from pathlib import Path
import numpy as np
from common import ROOT, SUITES, seed, sha, atomic_json
spec=importlib.util.spec_from_file_location('v21_client_readonly',ROOT/'preparations/idea1_v2_1/runtime_client.py')
legacy=importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
rpc=legacy.rpc
write_json=legacy.write_json
legacy.model_seed=seed

def phase_evidence(row):
    """Keep missing truth explicit instead of interpreting it as no progress."""
    row=dict(row); phase=row.get('semantic_phase','unknown')
    grasp=[v for v in row.get('robot_grasps',{}).values() if v is not None]
    contact=[v for v in row.get('robot_contacts',{}).values() if v is not None]
    distance=[v for v in row.get('eef_object_distances',{}).values() if v is not None]
    displacement=[v for v in row.get('object_displacements',{}).values() if v is not None]
    row['grasp_truth_available']=bool(grasp); row['contact_truth_available']=bool(contact)
    reason=None
    if phase=='initial' and not (grasp or contact or distance or displacement): reason='No object/contact/grasp progress evidence'
    if phase=='release_or_final_action' and row.get('min_object_target_distance') is None: reason='Target alignment unavailable at release'
    if phase=='grasp_or_manipulate' and not grasp and not any(contact): reason='Grasp truth and positive contact evidence unavailable'
    if reason and not row.get('success'): row.update(semantic_phase='unknown',semantic_phase_unknown_reason=reason)
    row['phase_evidence_source']='real object/contact/grasp telemetry; unknown truth remains unknown'
    return row

def finalize_phases(directory):
    d=Path(directory); path=d/'telemetry.jsonl'; states=[phase_evidence(json.loads(x)) for x in path.read_text().splitlines()]
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('x') as f:
        for row in states: f.write(json.dumps(row)+'\n')
        f.flush(); os.fsync(f.fileno())
    os.replace(temp,path)
    result=json.loads((d/'result.json').read_text()); order=['initial','approach','contact','grasp_or_manipulate','transport','goal_alignment','release_or_final_action','success']; known=[x['semantic_phase'] for x in states if x['semantic_phase'] in order]
    result.update(furthest_phase=max(known,key=order.index) if known else 'unknown',final_phase=states[-1]['semantic_phase'],unknown_phase_steps=sum(x['semantic_phase']=='unknown' for x in states))
    atomic_json(d/'result.json',result)

def check_prediction(record,condition,protocol,expected_seed=None):
    audit=record.get('audit',{})
    if audit.get('NFE')!=condition['schedule']['num_steps'] or audit.get('schedule')!=condition['schedule'] or audit.get('skip')!=condition['skip']: raise ValueError('Actual NFE/schedule/skip does not match frozen condition')
    if expected_seed is not None and record.get('seed')!=expected_seed: raise ValueError('Generation seed differs from frozen rule')
    actions=np.asarray(record.get('actions',record.get('chunk_actions')))
    if actions.shape!=(protocol['H'],7) or not np.isfinite(actions).all(): raise ValueError('Full H action shape/value invalid')
    if audit.get('instrumentation_mode')=='minimal':
        if record.get('category') not in ('measurement','warmup','benchmark') or record.get('timing',{}).get('measurement_mode')!='minimal_action_generation': raise ValueError('Minimal audit used outside independent timing')
    else:
        actual=audit.get('actual_timesteps') or audit.get('timesteps') or []
        if len(actual)!=condition['schedule']['num_steps'] or audit.get('actual_NFE')!=condition['schedule']['num_steps']: raise ValueError('Actual timestep/NFE count missing')
    from schedules import flow_schedule
    if condition['model']!='db-cogact':
        expected=flow_schedule(condition['model'],condition['schedule']['m'],condition['schedule']['num_steps'])
        if not np.allclose(audit.get('nodes',[]),expected['nodes'],atol=1e-7) or not np.allclose(audit.get('dt',[]),expected['dt'],atol=1e-7): raise ValueError('Flow update nodes/dt differ from frozen formula')
        if audit.get('instrumentation_mode')!='minimal':
            values=np.asarray(expected['timesteps'],dtype=np.float32)
            if condition['model']=='xiaomi-robotic0':
                bits=values.view(np.uint32); values=((bits+0x7fff+((bits>>16)&1))&0xffff0000).view(np.float32)
            if not np.allclose(actual,values,rtol=0,atol=1e-7): raise ValueError('Actual model timestep coordinate differs')
    else:
        mapping=list(reversed(audit.get('timestep_map',[])))
        if audit.get('timesteps')!=mapping or audit.get('nodes')!=mapping+['clean'] or audit.get('alpha_clean')!=1.: raise ValueError('DDIM mapping/clean endpoint differs')
        if condition['schedule']['kind']=='ddim_custom' and mapping[1]!=round(condition['schedule']['m']*mapping[0]): raise ValueError('DDIM intermediate training index differs')
    evidence=protocol.get('v22_audited_schedules',{}).get(json.dumps(condition['schedule'],sort_keys=True))
    if evidence:
        for field in ('nodes','dt','timestep_map','alpha_cumprod'):
            if field in evidence and audit.get(field)!=evidence[field]: raise ValueError('Schedule differs from parity-audited sampler')
    elif audit.get('instrumentation_mode')=='minimal': raise ValueError('Minimal measurement lacks separately audited schedule evidence')
    for key,hkey in [('noise_path','noise_file_sha256'),('server_prediction_path','raw_prediction_file_sha256')]:
        path=record.get(key)
        if not path or not record.get(hkey) or sha(path)!=record[hkey]: raise ValueError('Raw noise/prediction artifact missing or changed')
    with np.load(record['noise_path'],allow_pickle=False) as z:
        noise=z['initial_noise']
        if legacy.value_digest(noise)!=record['noise_hash']: raise ValueError('Actual initial noise differs from noise hash')
    with np.load(record['server_prediction_path'],allow_pickle=False) as z:
        if legacy.value_digest(z['raw_prediction'])!=record.get('raw_prediction_hash') or not np.array_equal(z['actions'],actions): raise ValueError('Raw/decoded artifact differs from logged actions')

def validate_episode(directory,row=None):
    d=Path(directory); result=json.loads((d/'result.json').read_text())
    states=[json.loads(x) for x in (d/'telemetry.jsonl').read_text().splitlines()]
    infers=[json.loads(x) for x in (d/'infer.jsonl').read_text().splitlines()]
    if not result.get('valid') or len(states)!=result['actions']+1 or len(infers)!=result['predictions']: raise ValueError('Episode record counts mismatch')
    if not all(len(x['eef_pos'])==3 and np.isfinite(x['eef_pos']).all() for x in states): raise ValueError('Missing true EEF XYZ')
    for x in infers:
        if not x.get('noise_hash') or not x.get('input_hash') or not x.get('audit'): raise ValueError('Missing inference provenance/audit')
        if sha(x['input_path'])!=x['input_file_sha256'] or sha(x['prediction_path'])!=x['prediction_sha256']: raise ValueError('Corrupt prediction/input')
        if x['executed_length']<=0: raise ValueError('Empty executed prefix')
    cmd=['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,duration','-of','json',str(d/'video.mp4')]
    probe=json.loads(subprocess.check_output(cmd)); stream=probe['streams'][0]
    if int(stream['nb_read_frames'])!=len(states): raise ValueError('Video frame count mismatch')
    subprocess.run(['ffmpeg','-v','error','-i',str(d/'video.mp4'),'-f','null','-'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    result['artifacts']={str(p.relative_to(d)):sha(p) for p in d.rglob('*') if p.is_file() and p.name!='validated.json'}
    result['video_verified']=True
    if row: result['episode_key']=row['key']
    atomic_json(d/'validated.json',result); return result

def run_episode(directory,port,condition,row,diagnostic_max_predictions=None):
    original=legacy.rpc
    from runtime_data import audited_protocol
    attempt=next(p for p in Path(directory).parents if (p/'condition_manifest.json').exists())
    frozen_runtime=audited_protocol(attempt,condition['model'],original(port,dict(op='health'))['protocol'])
    old_limit=legacy.LIMITS[row['suite']]
    if diagnostic_max_predictions is not None:
        r=frozen_runtime.get('R_by_suite',frozen_runtime['R']); r=r[row['suite']] if isinstance(r,dict) else r
        legacy.LIMITS[row['suite']]=r*diagnostic_max_predictions
    def request(p,req,**kwargs):
        req=dict(req)
        if req.get('op')=='predict':
            req.pop('native',None); req.pop('suite',None); req['schedule']=condition['schedule']
        response=original(p,req,**kwargs)
        if req.get('op')=='predict':
            audit=response.get('audit',{})
            n=audit.get('nfe',audit.get('NFE',audit.get('action_head_calls')))
            if n!=condition['schedule']['num_steps']: raise ValueError('Actual NFE missing or differs from schedule')
            response['noise_file_sha256']=sha(response['noise_path'])
            response['raw_prediction_file_sha256']=sha(response['prediction_path'])
            checked=dict(response,server_prediction_path=response['prediction_path'])
            check_prediction(checked,condition,frozen_runtime,req['seed'])
        return response
    legacy.rpc=request
    try:
        legacy.run_episode(directory,port,condition['model'],row['suite'],row['task'],row['init'],condition['skip'],kind='diagnostic_smoke' if diagnostic_max_predictions is not None else 'formal')
        finalize_phases(directory)
        if diagnostic_max_predictions is not None:
            result=json.loads((Path(directory)/'result.json').read_text()); result.update(diagnostic_prediction_cap=diagnostic_max_predictions,diagnostic_action_cap=legacy.LIMITS[row['suite']],native_action_limit=old_limit,not_formal_success_estimate=True,termination='success_during_diagnostic' if result['success'] else 'diagnostic_cap_reached'); atomic_json(Path(directory)/'result.json',result)
        return validate_episode(directory,row)
    finally: legacy.rpc=original; legacy.LIMITS[row['suite']]=old_limit
