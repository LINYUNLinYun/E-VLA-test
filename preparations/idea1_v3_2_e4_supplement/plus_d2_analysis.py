"""CPU-only strict four-condition success summary; pending/technical are never policy failures."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_RUN=ROOT/'results/idea1_v3_2_e4_supplement/libero_plus_v14_d2_gpu3_novideo_20261023T0543Z_recovery'
EXPECTED_MANIFEST='f35a09c66577443fcc19d8c7fea0127003ce405267afcd74898dd3c0ad3273c4'
MODES=('full_d10','v14_d10','matching_draft','v14_d2')
SUITES=('libero_spatial','libero_object','libero_goal','libero_10')
HEADS=dict(zip(SUITES,('draft_spatial','draft_object','draft_goal','draft_10')))
KEYS=('agentview_image','robot0_eye_in_hand_image','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos')


def read(path):return json.loads(Path(path).read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def resolve(path):
    path=Path(path);return path if path.is_absolute() else ROOT/path
def require(condition,message):
    if not condition:raise ValueError(message)
def bound(entry):
    path=resolve(entry['path']);require(sha(path)==entry['sha256'],'Bound SHA mismatch: '+str(path));return path
def identity(path):
    value=read(path);require(value.get('manifest_hash')==digest({k:v for k,v in value.items() if k!='manifest_hash'}),'Manifest digest mismatch')
    return value


def validate_structure(m):
    require(m.get('kind')=='native_LIBERO_Plus_half_V14_D2_GPU3_strict_novideo_v2','GPU3-only no-video D2 manifest required')
    for key,value in dict(H=50,R=5,D=2,nodes=[1.,.5,0.],condition_count=195,perturbed_case_count=175,original_case_count=20,reused_condition_count=585,required_gpu_index=3,
        required_gpu_uuid='GPU-555cde36-5e41-bc44-0c69-ea47358fd703',skip_vlm_1based=[3,4,9,11]).items():
        require(m.get(key)==value,'Wrong manifest '+key)
    require(m.get('modes')==['v14_d2'] and m.get('video_source_indices')==[],'Wrong mode/video manifest')
    jobs=m['jobs'];require(len(jobs)==195,'Expected195 jobs')
    keys={(j['group'],j['source_index']) for j in jobs};require(len(keys)==195,'Duplicate manifest case key')
    require(Counter(j['group'] for j in jobs)=={'perturbed350':175,'original40':20},'Wrong group denominators')
    require(Counter(j['case']['mother_suite'] for j in jobs if j['group']=='perturbed350')==dict(zip(SUITES,(44,44,44,43))),'Wrong175 suite allocation')
    require(Counter(j['case']['mother_suite'] for j in jobs if j['group']=='original40')==dict.fromkeys(SUITES,5),'Wrong20 suite allocation')
    for job in jobs:
        require(job['case_hash']==digest(job['case']),'Manifest case hash mismatch')
        require(job['case']['matching_draft']==HEADS[job['case']['mother_suite']],'Wrong matching draft head')
        require(job['modes']==['v14_d2'] and job['video'] is False,'Wrong case mode/video')
    return jobs


def load_evidence(run):
    m=identity(run/'manifest.json');jobs=validate_structure(m)
    require(m['manifest_hash']==EXPECTED_MANIFEST,'Expected fixed f35a09 recovery manifest')
    half=identity(bound(m['half_manifest']));parent=identity(bound(m['parent_manifest']));ledger=read(bound(m['audit_ledger']))
    require(m['half_manifest_hash']==half['manifest_hash']==ledger['half_manifest_hash'],'Half provenance mismatch')
    require(m['parent_manifest_hash']==parent['manifest_hash']==ledger['parent_manifest_hash'],'Parent provenance mismatch')
    require(ledger.get('completed')==585 and len(ledger['selected_completed'])==585,'Expected fixed585 ledger')
    for name,expected in m['source_files'].items():require(sha(resolve(name))==expected,'Frozen source changed: '+name)
    half_jobs={(j['group'],j['source_index']):j for j in half['jobs']}
    require(set(half_jobs)=={(j['group'],j['source_index']) for j in jobs},'Half/D2 selection mismatch')
    old={}
    for row in ledger['selected_completed']:
        key=(row['group'],row['source_index'],row['mode']);require(key not in old,'Duplicate585 condition')
        path=bound(row);value=read(path)
        require(value.get('status')=='complete' and type(value.get('success')) is bool,'Invalid585 completion')
        require(value.get('mode')==row['mode'] and value.get('manifest_hash')==parent['manifest_hash'],'Wrong585 identity')
        require(value['success']==row['success'],'585 ledger success mismatch')
        old[key]=(row,value)
    baseline={}
    for job in jobs:
        key=(job['group'],job['source_index']);selected=half_jobs[key]
        require(selected['case_hash']==job['case_hash'] and selected['case']==job['case'],'Wrong D2 case selection')
        modes={'full_d10','v14_d10',job['case']['matching_draft']}
        require(len(job['paired_results'])==3 and {r['mode'] for r in job['paired_results']}==modes,'Expected exact3 baseline conditions')
        paired={}
        for row in job['paired_results']:
            reference,value=old[key+(row['mode'],)]
            require(row==reference,'Manifest/ledger baseline row mismatch')
            require(value['case_hash']==job['case_hash']==digest(value['case']) and value['case']==job['case'],'Wrong585 case')
            label='matching_draft' if row['mode'].startswith('draft_') else row['mode'];paired[label]=value
        baseline[key]=paired
    return m,baseline


def validate_result(result,job,m):
    for key,expected in dict(mode='v14_d2',manifest_hash=m['manifest_hash'],case_hash=job['case_hash'],case=job['case'],group=job['group'],source_index=job['source_index'],H=50,R=5,D=2,gpu_uuid=m['required_gpu_uuid']).items():
        require(result.get(key)==expected,'D2 result identity/config mismatch: '+key)
    require(result.get('status')=='complete' and type(result.get('success')) is bool,'Invalid D2 success/completion')
    require(result.get('paired_results')==job['paired_results'],'D2 baseline provenance mismatch')
    initial=result.get('initial_comparison',{})
    require(initial.get('passed') is True and initial.get('instruction_equal') is True and initial.get('initial_vector_sha256_equal') is True,'D2 initial gate failed/absent')
    require(set(initial.get('fields',{}))==set(KEYS),'Missing initial comparison fields')
    total=0
    for key in KEYS:
        field=initial['fields'][key];require(field.get('passed') is True,'Initial field gate failed')
        if key.endswith('image'):
            require(field['max_abs']<=1 and field['changed_channels']>=0,'Initial image budget violated');total+=field['changed_channels']
        else:require(field.get('exact_equal') is True and field['max_abs']==0 and field['changed_channels']==0,'Initial physical fields not exact')
    require(total<=4 and total==initial.get('image_changed_channels_total'),'Initial total image budget violated')


def validate_stream(path,m,result):
    count=0
    with path.open() as stream:
        for line in stream:
            row=json.loads(line);response=row['response'];count+=1
            require(response.get('mode')=='v14_d2' and response.get('manifest_hash')==m['manifest_hash'] and response.get('execute_length')==5,'D2 request identity/R mismatch')
            timing=response['timing'];require(timing.get('actual_NFE')==2 and timing.get('nodes')==[1.,.5,0.],'D2 actual nodes/NFE mismatch')
            expected=dict(vision_encoder_calls=1,vlm_prefix_calls=1,ae_layer_calls=36,learned_head_calls=0,verify_calls=0,constructor_full_record_calls=0,ae_euler_updates=2)
            require(all(response['observed_calls'].get(k)==v for k,v in expected.items()),'D2 actual route mismatch')
    require(count==result.get('requests') and count>0,'D2 request count mismatch')
    return count


def collect(run,m,baseline):
    jobs={(j['group'],j['source_index']):j for j in m['jobs']};complete={};technical=[];other=[]
    for path in sorted(run.glob('*/case*/v14_d2/attempt*/result.json')):
        result=read(path);key=(path.parents[3].name,int(path.parents[2].name[4:]))
        require(key in jobs,'Attempt outside frozen case selection: '+str(path))
        if result.get('status')!='complete':
            target=technical if result.get('status')=='technical_failure' else other
            target.append(dict(group=key[0],source_index=key[1],path=str(path),status=result.get('status'),error=result.get('error')));continue
        require(key not in complete,'Duplicate valid complete D2 case: '+str(key))
        job=jobs[key];validate_result(result,job,m)
        require(read(path.with_name('initial_comparison.json'))==result['initial_comparison'],'Initial gate receipt mismatch')
        full=baseline[key]['full_d10'];info=result['native_info']
        require(info['initial_vector_sha256']==full['native_info']['initial_vector_sha256'] and info['instruction']==full['native_info']['instruction'],'Initial native provenance mismatch')
        bound(job['initial_observation']);bound(job['first_input'])
        validate_stream(path.with_name('infer.jsonl'),m,result)
        complete[key]=dict(result,path=str(path),result_sha256=sha(path))
    return complete,technical,other


def tables(m,baseline,complete):
    reference=[];paired=[];changes=[]
    for group in ('perturbed350','original40'):
        group_jobs=[j for j in m['jobs'] if j['group']==group]
        for dimension in ('overall','mother_suite','category','difficulty_level'):
            values=['all'] if dimension=='overall' else sorted({str(j['case'].get(dimension)) for j in group_jobs})
            for value in values:
                jobs=[j for j in group_jobs if dimension=='overall' or str(j['case'].get(dimension))==value]
                keys=[(j['group'],j['source_index']) for j in jobs];common=[k for k in keys if k in complete]
                common_n=len(common)
                for mode in MODES:
                    if mode!='v14_d2':
                        successes=sum(baseline[k][mode]['success'] for k in keys)
                        reference.append(dict(group=group,dimension=dimension,value=value,condition=mode,n=len(keys),successes=successes,success_rate=successes/len(keys)))
                    successes=sum((complete[k] if mode=='v14_d2' else baseline[k][mode])['success'] for k in common)
                    paired.append(dict(group=group,dimension=dimension,value=value,condition=mode,target_n=len(keys),n=common_n,
                        pending_n=len(keys)-common_n,successes=successes,success_rate=successes/common_n if common_n else None,status='complete' if common_n==len(keys) else 'partial' if common_n else 'pending'))
                for mode in MODES[:-1]:
                    counts=Counter((bool(baseline[k][mode]['success']),bool(complete[k]['success'])) for k in common)
                    changes.append(dict(group=group,dimension=dimension,value=value,baseline=mode,n=common_n,
                        both_success=counts[(True,True)],baseline_only=counts[(True,False)],D2_only=counts[(False,True)],both_fail=counts[(False,False)]))
    return reference,paired,changes


def analyze(run,output):
    run=Path(run).resolve();m,baseline=load_evidence(run);complete,technical,other=collect(run,m,baseline)
    reference,paired,changes=tables(m,baseline,complete)
    result=dict(status='complete' if len(complete)==195 else 'partial' if complete else 'pending',manifest_hash=m['manifest_hash'],
        manifest_sha256=sha(run/'manifest.json'),analysis_source_sha256=sha(__file__),target=195,target_perturbed=175,target_original=20,
        completed=len(complete),pending=195-len(complete),historical_conditions=585,technical_attempts_excluded=technical,other_attempts=other,
        reference_full585=reference,common_case_four_conditions=paired,paired_changes=changes,
        completed_results=[dict(group=k[0],source_index=k[1],path=r['path'],sha256=r['result_sha256']) for k,r in sorted(complete.items())],
        limitations=['Full585 reference scores and D2 completed-case intersection scores have separate tables; never compare different denominators as paired.',
            'Missing/running/technical attempts are excluded from policy success denominators. Zero completed cases have null rates, not0%.',
            'Matching draft means four different mother-suite heads; suite rows keep their identities. No latency/speedup/video analysis.'])
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    for name,rows in (('reference_full585.csv',reference),('paired_four_conditions.csv',paired),('paired_changes.csv',changes)):
        with (out/name).open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (out/'summary.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=DEFAULT_RUN);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=analyze(a.run,a.output)
    print(json.dumps({k:result[k] for k in ('status','target','completed','pending','historical_conditions')}))
