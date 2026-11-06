"""CPU-only unique195 matrix: parent even98, frozen odd39, explicit GPU0 odd58 retry."""
import argparse
import csv
import json
from pathlib import Path
from . import plus_d2_analysis as common
from . import plus_d2_quality_v3_analysis as quality

ROOT=common.ROOT
DEFAULT_RUN=ROOT/'results/idea1_v3_2_e4_supplement/libero_plus_v14_d2_gpu0_continuation_20261023_run1'
EXPECTED='fbaf319821ae7ad12b2b74d2f6b1004c35b536b1216d7168893844cdbd14671c'
KIND='D2_GPU0_only_interrupted_odd58_continuation'
read,require,sha=common.read,common.require,common.sha


def key(row):return row['group'],row['source_index']


def validate_partition(m,parent):
    require(m.get('manifest_hash')==EXPECTED and m.get('kind')==KIND,'Exact fbaf GPU0 continuation required')
    require(m['parent_quality_manifest_hash']==parent['manifest_hash']==quality.EXPECTED,'Parent c823 identity differs')
    require(m['original_global_jobs']==parent['jobs'],'Original global195 jobs/order changed')
    require(m['assigned_gpus']=={'0':quality.ASSIGNED['0']} and m['assigned_gpu_index']==0 and m['assigned_gpu_uuid']==quality.ASSIGNED['0'],'Continuation device must be GPU0')
    for name,expected in dict(H=50,R=5,D=2,nodes=[1.,.5,0.],condition_count=58,perturbed_case_count=48,original_case_count=10,modes=['v14_d2'],video_source_indices=[]).items():
        require(m.get(name)==expected,'Wrong continuation config: '+name)
    for name in ('half_manifest','parent_manifest','audit_ledger','checkpoint_metadata','references','legacy_bank','legacy_history_manifest','legacy_history_receipt','cross_device_probe_manifest'):
        require(m[name]==parent[name],'Parent/continuation provenance differs: '+name)
    ordinals={key(job):i for i,job in enumerate(parent['jobs'])}
    frozen=m['parent_gpu1_reused_complete'];frozen_keys={key(row) for row in frozen}
    require(len(frozen)==len(frozen_keys)==39,'Exactly39 unique frozen odd results required')
    require(all(k in ordinals and ordinals[k]%2==1 for k in frozen_keys),'Frozen GPU1 key is not original global odd')
    expected=[j for i,j in enumerate(parent['jobs']) if i%2==1 and key(j) not in frozen_keys]
    require(m['jobs']==expected and len(expected)==58,'New58 must be exact remaining odd cases in original order')
    require(len([j for i,j in enumerate(parent['jobs']) if i%2==0])==98,'Expected parent even98')
    return ordinals,frozen_keys


def load_evidence(run):
    m=common.identity(run/'manifest.json');parent_path=common.bound(m['parent_quality_manifest'])
    parent_run=common.resolve(m['parent_results_root']);require(parent_path.resolve()==(parent_run/'manifest.json').resolve(),'Parent root/manifest path differs')
    parent,baseline,order,provenance=quality.load_evidence(parent_run)
    ordinals,frozen_keys=validate_partition(m,parent)
    for path,expected in m['source_files'].items():common.bound(dict(path=path,sha256=expected))
    frozen={}
    for row in m['parent_gpu1_reused_complete']:
        path=common.bound(row);value=read(path)
        require(value.get('status')=='complete' and value.get('success')==row['success'] and value.get('steps')==row['steps'],'Frozen odd result changed')
        frozen[key(row)]=dict(path=str(path),sha256=row['sha256'])
    preserved=[]
    for entry in m['parent_gpu1_preserved_technical_attempts']:
        path=common.bound(entry);value=read(path)
        require(value.get('status')=='technical_failure' and value.get('success') is None,'Preserved interruption must remain technical, not policy failure')
        preserved.append(dict(path=str(path),sha256=entry['sha256'],source_index=value.get('source_index'),status='technical_failure',success=None))
    return m,parent_run,parent,baseline,order,ordinals,frozen,provenance,preserved


def merge_parent(parent_complete,frozen,ordinals):
    merged={};counts=dict(parent_gpu0_even=0,frozen_parent_gpu1_odd=0)
    for k,result in parent_complete.items():
        odd=ordinals[k]%2==1
        if odd:
            require(k in frozen,'New parent GPU1 complete outside frozen39 is forbidden')
            require(result['result_sha256']==frozen[k]['sha256'] and Path(result['path']).resolve()==Path(frozen[k]['path']).resolve(),'Frozen39 SHA/path changed')
        namespace='frozen_parent_gpu1_odd' if odd else 'parent_gpu0_even'
        merged[k]=dict(result,namespace=namespace);counts[namespace]+=1
    require(counts['frozen_parent_gpu1_odd']==39,'All frozen39 must remain complete')
    return merged,counts


def validate_new_result(result,job,m):
    common.validate_result(result,job,dict(m,required_gpu_uuid=quality.ASSIGNED['0']))
    binding=result.get('native_info',{}).get('binding',{})
    require(binding.get('gpu')==0 and binding.get('gpu_uuid')==quality.ASSIGNED['0'],'Explicit continuation renderer must be GPU0, even for original odd key')
    require(not result.get('video'),'No continuation video allowed')


def merge_unique(merged,additional):
    require(not set(merged).intersection(additional),'Duplicate valid complete scientific case across namespaces')
    return {**merged,**additional}


def collect_new(run,m,parent,baseline,order,ordinals):
    jobs={key(j):j for j in m['jobs']};complete={};technical=[];other=[];gate_evidence=None
    gatepath=run/'runtime_gpu0/strict_gate/gate.json'
    if gatepath.exists():
        gate=read(gatepath);quality.validate_gate(gate,m,0,order)
        for entry in gate['arrays']:
            path=common.bound(entry);require(path.resolve().is_relative_to(run.resolve()),'Continuation startup array outside new namespace')
        gate_evidence=dict(path=str(gatepath),sha256=sha(gatepath),passed=True)
    for path in sorted(run.glob('*/case*/v14_d2/attempt*/result.json')):
        result=read(path);k=(path.parents[3].name,int(path.parents[2].name[4:]));require(k in jobs,'New attempt outside frozen58')
        if result.get('status')!='complete':
            (technical if result.get('status')=='technical_failure' else other).append(dict(namespace='continuation_gpu0_odd',path=str(path),group=k[0],source_index=k[1],status=result.get('status'),success=result.get('success'),error=result.get('error')));continue
        require(k not in complete,'Duplicate valid complete continuation case');job=jobs[k];validate_new_result(result,job,m)
        require(gate_evidence is not None,'Complete continuation lacks new startup gate')
        require(not path.with_name('video.mp4').exists(),'Unexpected continuation MP4')
        require(read(path.with_name('initial_comparison.json'))==result['initial_comparison'],'Continuation initial receipt differs')
        full=baseline[k]['full_d10'];info=result['native_info']
        require(info['initial_vector_sha256']==full['native_info']['initial_vector_sha256'] and info['instruction']==full['native_info']['instruction'],'Continuation initial physical/language provenance differs')
        common.bound(job['initial_observation']);common.bound(job['first_input']);common.validate_stream(path.with_name('infer.jsonl'),m,result)
        complete[k]=dict(result,path=str(path),result_sha256=sha(path),namespace='continuation_gpu0_odd',assigned_gpu_index=0,global_ordinal=ordinals[k])
    return complete,technical,other,gate_evidence


def analyze(run,output):
    run=Path(run).resolve();m,parent_run,parent,baseline,order,ordinals,frozen,provenance,preserved=load_evidence(run)
    parent_complete,parent_technical,parent_other,parent_gates=quality.collect(parent_run,parent,baseline,order)
    merged,counts=merge_parent(parent_complete,frozen,ordinals)
    new,new_technical,new_other,new_gate=collect_new(run,m,parent,baseline,order,ordinals)
    merged=merge_unique(merged,new);counts['continuation_gpu0_odd']=len(new)
    reference,paired,changes=common.tables(parent,baseline,merged)
    result=dict(status='complete' if len(merged)==195 else 'partial',target=195,completed=len(merged),pending=195-len(merged),
        target_perturbed=175,target_original=20,historical_conditions=585,manifest_hash=m['manifest_hash'],manifest_sha256=sha(run/'manifest.json'),
        parent_manifest_hash=parent['manifest_hash'],analysis_source_sha256=sha(__file__),source_counts=counts,
        source_targets=dict(parent_gpu0_even=98,frozen_parent_gpu1_odd=39,continuation_gpu0_odd=58),parent_provenance=provenance,
        parent_runtime_gates=parent_gates,continuation_runtime_gate=new_gate,frozen_odd39=[dict(group=k[0],source_index=k[1],**row) for k,row in sorted(frozen.items())],
        preserved_technical_interruptions=preserved,technical_attempts_excluded=parent_technical+new_technical,other_attempts=parent_other+new_other,
        reference_full585=reference,common_case_four_conditions=paired,paired_changes=changes,
        completed_results=[dict(group=k[0],source_index=k[1],namespace=r['namespace'],global_ordinal=ordinals[k],actual_gpu=r['assigned_gpu_index'],
            manifest_hash=r['manifest_hash'],path=r['path'],sha256=r['result_sha256']) for k,r in sorted(merged.items())],
        limitations=['Same195 scientific keys merge three explicit namespaces: parent GPU0 even98, immutable historical GPU1 odd39, new GPU0 odd58.',
            'GPU1 is not queried or used. No new parent odd completions are admitted beyond frozen39. A technical interruption, including case154, is excluded and may be retried in the continuation namespace.',
            'Continuation odd keys use explicit GPU0 rather than original modulo2 GPU1 assignment. Pending/technical attempts are never policy failures.',
            'Full585 references and completed-case paired comparisons have separate denominators. No model, simulator, GPU, video or latency operations.'])
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    for name,rows in (('reference_full585.csv',reference),('paired_four_conditions.csv',paired),('paired_changes.csv',changes)):
        with (out/name).open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (out/'summary.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=DEFAULT_RUN);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=analyze(a.run,a.output);print(json.dumps({k:result[k] for k in ('status','target','completed','pending','source_counts')}))
