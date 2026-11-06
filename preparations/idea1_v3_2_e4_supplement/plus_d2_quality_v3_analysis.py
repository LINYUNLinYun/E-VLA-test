"""Read-only CPU summary for fixed run2 GPU0/1 quality shards and old585 controls."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from . import plus_d2_analysis as common

ROOT=common.ROOT
DEFAULT_RUN=ROOT/'results/idea1_v3_2_e4_supplement/libero_plus_v14_d2_quality_gpu01_20261023_run2'
EXPECTED='c823d3ccea915a4cca78a6282b479f290f1cfff7a0c58052d14081a7efad3290'
KIND='native_LIBERO_Plus_half_V14_D2_quality_v3_legacy_bitwise16'
ASSIGNED={'0':'GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1','1':'GPU-599477fb-0507-40dc-14df-d491f7a7b202'}
read,require,sha,digest,resolve=common.read,common.require,common.sha,common.digest,common.resolve


def validate_structure(m):
    require(m.get('kind')==KIND and m.get('manifest_hash')==EXPECTED,'Exact c823 quality-v3 run2 manifest required')
    require(m.get('assigned_gpus')==ASSIGNED,'Fixed GPU0/1 assignments changed')
    for key,value in dict(H=50,R=5,D=2,nodes=[1.,.5,0.],ae_depth=18,skip_vlm_1based=[3,4,9,11],condition_count=195,
        perturbed_case_count=175,original_case_count=20,reused_condition_count=585,repeat_count=2,modes=['v14_d2'],video_source_indices=[]).items():
        require(m.get(key)==value,'Wrong quality-v3 configuration: '+key)
    jobs=m['jobs'];keys={(j['group'],j['source_index']) for j in jobs}
    require(len(jobs)==len(keys)==195,'Expected195 unique jobs')
    require(Counter(j['group'] for j in jobs)=={'perturbed350':175,'original40':20},'Wrong175/20 target')
    require(Counter(j['case']['mother_suite'] for j in jobs if j['group']=='perturbed350')==dict(zip(common.SUITES,(44,44,44,43))),'Wrong175 suite counts')
    require(Counter(j['case']['mother_suite'] for j in jobs if j['group']=='original40')==dict.fromkeys(common.SUITES,5),'Wrong20 suite counts')
    for job in jobs:
        require(job['case_hash']==digest(job['case']) and job['case']['matching_draft']==common.HEADS[job['case']['mother_suite']],'Wrong case/head identity')
        require(job['modes']==['v14_d2'] and job['video'] is False,'Wrong per-case mode/video')
    require(len(m['references'])==16,'Fixed16 reference gate required')
    return jobs


def load_evidence(run):
    m=common.identity(run/'manifest.json');jobs=validate_structure(m);cache={}
    def bound(entry):
        path=resolve(entry['path']);key=str(path)
        if key not in cache:cache[key]=sha(path)
        require(cache[key]==entry['sha256'],'Bound SHA mismatch: '+key);return path
    half=common.identity(bound(m['half_manifest']));parent=common.identity(bound(m['parent_manifest']));ledger=read(bound(m['audit_ledger']))
    require(m['half_manifest_hash']==half['manifest_hash']==ledger['half_manifest_hash'],'Half provenance mismatch')
    require(m['parent_manifest_hash']==parent['manifest_hash']==ledger['parent_manifest_hash'],'Parent provenance mismatch')
    prior=common.identity(bound(m['previous_D2_registration']))
    probe=common.identity(bound(m['cross_device_probe_manifest']));history=common.identity(bound(m['legacy_history_manifest']))
    history_receipt=read(bound(m['legacy_history_receipt']));bound(m['legacy_bank'])
    require(prior['half_manifest_hash']==m['half_manifest_hash'] and prior['parent_manifest_hash']==m['parent_manifest_hash'],'Previous registration provenance mismatch')
    require(history_receipt['status']=='complete' and history_receipt['manifest_hash']==history['manifest_hash'],'History receipt identity mismatch')
    require(history['legacy_bank']==m['legacy_bank'] and probe['references']==m['references'],'Probe/bank references differ')
    require(len(history['references'])==2,'Exact two history diagnostic references required')
    for historical,reference in zip(history['references'],m['references'][:2]):
        require({k:v for k,v in historical.items() if k!='current_stable'}==reference,'History reference identity differs')
        bound(historical['current_stable'])
    for check in history_receipt['uncaptured']:
        require(all(check['versus']['historical'][key]['bitwise_equal'] is True for key in ('raw','decoded')),'History historical bytewise gate failed')
    for manifest in (m,probe,history):
        for path,expected in manifest['source_files'].items():bound(dict(path=path,sha256=expected))
    require(ledger.get('completed')==585 and len(ledger['selected_completed'])==585,'Exact585 old ledger required')
    old={}
    for row in ledger['selected_completed']:
        key=(row['group'],row['source_index'],row['mode']);require(key not in old,'Duplicate585 condition')
        value=read(bound(row));require(value.get('status')=='complete' and type(value.get('success')) is bool,'Invalid585 completion')
        require(value['manifest_hash']==parent['manifest_hash'] and value['mode']==row['mode'] and value['success']==row['success'],'585 result/ledger mismatch')
        old[key]=(row,value)
    half_jobs={(j['group'],j['source_index']):j for j in half['jobs']}
    require(set(half_jobs)=={(j['group'],j['source_index']) for j in jobs},'D2/half selection changed')
    baseline={}
    for job in jobs:
        key=(job['group'],job['source_index']);require(job['case_hash']==half_jobs[key]['case_hash'] and job['case']==half_jobs[key]['case'],'Wrong selected case')
        require(len(job['paired_results'])==3 and {r['mode'] for r in job['paired_results']}=={'full_d10','v14_d10',job['case']['matching_draft']},'Exact3 old conditions required')
        values={}
        for row in job['paired_results']:
            reference,value=old[key+(row['mode'],)];require(row==reference,'Old585 row changed')
            require(value['case_hash']==job['case_hash']==digest(value['case']) and value['case']==job['case'],'Old585 case changed')
            values['matching_draft' if row['mode'].startswith('draft_') else row['mode']]=value
        baseline[key]=values
    for ref in m['references']:
        row,value=old[(ref['group'],ref['source_index'],ref['mode'])]
        require(ref['case_hash']==value['case_hash'] and ref['result']['sha256']==row['sha256'],'Fixed16 historical reference mismatch')
        for field in ('result','observation','infer'):bound(ref[field])
    return m,baseline,history['legacy_mode_order'],dict(verified_bound_files=len(cache),manifest_sha256=sha(run/'manifest.json'))


def bytewise(pair):
    return all(pair[key].get('bitwise_equal') is True and pair[key].get('max_abs')==0 for key in ('raw','decoded'))


def validate_gate(gate,m,card,legacy_order):
    require(gate.get('passed') is True and gate.get('bytewise_only') is True,'Strict runtime gate not passed')
    require(gate.get('assigned_gpu_index')==card and gate.get('assigned_gpu_uuid')==ASSIGNED[str(card)],'Gate GPU assignment mismatch')
    legacy=gate['legacy_verbatim'];require(legacy['passed'] is True and legacy['bank_sha256']==m['legacy_bank']['sha256'],'Legacy initialization/bank gate failed')
    require([c['mode'] for c in legacy['checks']]==legacy_order and len(legacy['checks'])==6,'Legacy6 order mismatch')
    require(all(c['comparison'][key].get('passed') is True for c in legacy['checks'] for key in ('raw','decoded')),'Legacy6 individual comparison failed')
    require(len(gate['D10'])==16 and len(gate['D2'])==2,'Expected16historical+2D2 gates')
    for check,ref in zip(gate['D10'],m['references']):
        require(check['reference']==ref and check['passed'] is True and len(check['repeats_vs_history'])==2,'Wrong fixed16 reference/repetition')
        require(all(bytewise(pair) for pair in check['repeats_vs_history']+[check['repeat_vs_repeat']]),'Historical raw/decoded not bytewise')
    for check,ref in zip(gate['D2'],m['references'][:2]):
        require(check['mode']==ref['mode'] and check['passed'] is True and bytewise(check['comparison']),'D2 wrapper/native not bytewise')
        require(check['costs']['actual_NFE']==2 and check['costs']['nodes']==[1.,.5,0.],'D2 gate nodes mismatch')
        for calls in (check['native_calls'],check['wrapper_calls']):
            require(all(calls.get(k)==v for k,v in dict(vision_encoder_calls=1,vlm_prefix_calls=1,ae_layer_calls=36,learned_head_calls=0,verify_calls=0,constructor_full_record_calls=0).items()),'D2 gate actual calls differ')


def validate_result(result,job,m,ordinal):
    card=ordinal%2
    common.validate_result(result,job,dict(m,required_gpu_uuid=ASSIGNED[str(card)]))
    require(not result.get('video'),'New quality video forbidden')
    binding=result.get('native_info',{}).get('binding',{})
    require(binding.get('gpu')==card and binding.get('gpu_uuid')==ASSIGNED[str(card)],'Native renderer wrong ordinal shard')


def collect(run,m,baseline,legacy_order):
    jobs={(j['group'],j['source_index']):(i,j) for i,j in enumerate(m['jobs'])};complete={};technical=[];other=[];gates={}
    for card in (0,1):
        path=run/('runtime_gpu%d/strict_gate/gate.json'%card)
        if path.exists():
            gate=read(path);validate_gate(gate,m,card,legacy_order)
            for entry in gate['arrays']:common.bound(entry)
            gates[card]=dict(path=str(path),sha256=sha(path),passed=True)
    for path in sorted(run.glob('*/case*/v14_d2/attempt*/result.json')):
        result=read(path);key=(path.parents[3].name,int(path.parents[2].name[4:]));require(key in jobs,'Attempt outside fixed195')
        if result.get('status')!='complete':
            (technical if result.get('status')=='technical_failure' else other).append(dict(group=key[0],source_index=key[1],path=str(path),status=result.get('status'),error=result.get('error')));continue
        require(key not in complete,'Duplicate valid complete D2 case')
        ordinal,job=jobs[key];validate_result(result,job,m,ordinal);require(ordinal%2 in gates,'Completed case lacks strict shard startup proof')
        require(not path.with_name('video.mp4').exists(),'Unexpected new video file')
        require(read(path.with_name('initial_comparison.json'))==result['initial_comparison'],'Initial receipt differs')
        full=baseline[key]['full_d10'];info=result['native_info']
        require(info['initial_vector_sha256']==full['native_info']['initial_vector_sha256'] and info['instruction']==full['native_info']['instruction'],'Native initial provenance differs')
        common.bound(job['initial_observation']);common.bound(job['first_input']);common.validate_stream(path.with_name('infer.jsonl'),m,result)
        complete[key]=dict(result,path=str(path),result_sha256=sha(path),assigned_gpu_index=ordinal%2,global_ordinal=ordinal)
    return complete,technical,other,gates


def analyze(run,output):
    run=Path(run).resolve();m,baseline,order,provenance=load_evidence(run);complete,technical,other,gates=collect(run,m,baseline,order)
    reference,paired,changes=common.tables(m,baseline,complete)
    result=dict(status='complete' if len(complete)==195 else 'partial' if complete else 'pending',manifest_hash=m['manifest_hash'],
        provenance=provenance,analysis_source_sha256=sha(__file__),tables_source_sha256=sha(common.__file__),assigned_gpus=ASSIGNED,shard_target_counts={'0':98,'1':97},
        target=195,target_perturbed=175,target_original=20,completed=len(complete),pending=195-len(complete),historical_conditions=585,
        technical_attempts_excluded=technical,other_attempts=other,strict_runtime_gates=gates,
        reference_full585=reference,common_case_four_conditions=paired,paired_changes=changes,
        completed_results=[dict(group=k[0],source_index=k[1],ordinal=r['global_ordinal'],gpu=r['assigned_gpu_index'],path=r['path'],sha256=r['result_sha256']) for k,r in sorted(complete.items())],
        limitations=['Old585 full-sample references and completed-D2 intersection comparisons use separate tables.',
            'Technical, running and missing attempts are excluded, never scored as failure; zero completed denominators have null rates.',
            'Each fixed GPU0/1 shard requires legacy6 initialization, historical16 repeated bytewise proof and two native/wrapped D2 bytewise gates. Shard uses global ordinal, not source_index modulo2.',
            'No model/simulator/GPU calls, video, latency or speedup analysis. Root separately audits full execution streams.'])
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    for name,rows in (('reference_full585.csv',reference),('paired_four_conditions.csv',paired),('paired_changes.csv',changes)):
        with (out/name).open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (out/'summary.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=DEFAULT_RUN);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=analyze(a.run,a.output);print(json.dumps({k:report[k] for k in ('status','target','completed','pending','historical_conditions')}))
