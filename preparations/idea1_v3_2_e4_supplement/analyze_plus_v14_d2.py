"""CPU-only four-way common-case summary; no model/simulator imports or writes."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from .native_envs import digest,sha,write_json
from .evaluation import protocol_identity
from .plus_v14_d2 import completed,VIDEOS

LABELS=('full_d10','v14_d10','matching_draft','v14_d2')


def summarize(jobs,matched):
    rows=[];paired=[]
    for group in ('perturbed350','original40'):
        expected=[j for j in jobs if j['group']==group]
        dimensions=('overall','mother_suite','category','difficulty_level') if group=='perturbed350' else ('overall','mother_suite')
        for dimension in dimensions:
            values=['all'] if dimension=='overall' else sorted({str(j['case'][dimension]) for j in expected})
            for value in values:
                inside=lambda j:dimension=='overall' or str(j['case'][dimension])==value
                common=[(j,r) for j,r in matched if j['group']==group and inside(j)];n=len(common)
                for mode in LABELS:
                    success=sum(r[mode]['success'] for _,r in common)
                    rows.append(dict(group=group,dimension=dimension,value=value,mode=mode,expected_cases=sum(inside(j) for j in expected),
                        common_complete_cases=n,successes=success,success_rate=success/n if n else None))
                for reference in ('full_d10','v14_d10','matching_draft'):
                    wins=sum(r['v14_d2']['success'] and not r[reference]['success'] for _,r in common)
                    losses=sum(not r['v14_d2']['success'] and r[reference]['success'] for _,r in common)
                    paired.append(dict(group=group,dimension=dimension,value=value,candidate='v14_d2',reference=reference,n=n,
                        wins=wins,losses=losses,both_success=sum(r['v14_d2']['success'] and r[reference]['success'] for _,r in common),
                        both_failure=sum(not r['v14_d2']['success'] and not r[reference]['success'] for _,r in common),
                        success_difference=(wins-losses)/n if n else None))
    return rows,paired


def trajectory(path):
    """Actual recorded step indices: no synthetic old step0, no interpolation."""
    data={}
    with (Path(path).parent/'telemetry.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line);step=row['step'];xyz=np.asarray(row['actual']['eef_xyz_world'],float)
            if step in data or xyz.shape!=(3,) or not np.isfinite(xyz).all(): raise ValueError('Invalid/duplicate actual XYZ')
            data[step]=xyz
    return data


def actual_distance(first,second):
    steps=sorted(set(first)&set(second))
    distances=np.array([np.linalg.norm(first[s]-second[s])*1000 for s in steps])
    return dict(matched_steps=len(steps),first_common_step=steps[0] if steps else None,last_common_step=steps[-1] if steps else None,
                mean_distance_mm=float(distances.mean()) if steps else None,max_distance_mm=float(distances.max()) if steps else None,
                final_common_distance_mm=float(distances[-1]) if steps else None,
                interpretation='Closed-loop actual world EEF path separation on exact shared native steps; no padding, no same-input action-error claim')


def costs(path):
    totals=defaultdict(float);values=defaultdict(list);requests=0;bad=[]
    with (Path(path).parent/'infer.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line);requests+=1;response=row['response']
            for key in ('completed_s',):
                if key in row: values['client_completed_ms'].append(1000*row[key])
            for key,value in response.get('timing',{}).items():
                if key.endswith('_ms') and isinstance(value,(int,float)): values[key].append(value)
            for key,value in response.get('observed_calls',{}).items(): totals[key]+=value
            if response.get('mode')=='v14_d2':
                call=response['observed_calls']
                if call['ae_layer_calls']!=36 or call['vision_encoder_calls']!=1 or call['vlm_prefix_calls']!=1 or call['verify_calls'] or call['constructor_full_record_calls']: bad.append(row['request'])
    return dict(requests=requests,observed_call_totals=dict(totals),D2_contract_violations=bad,
        descriptive_request_ms={key:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),p95=float(np.percentile(v,95))) for key,v in values.items()})


def csv_write(path,rows):
    if not rows:return
    with Path(path).open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def analyze(root,output,allow_partial=False,include_costs=True):
    root=Path(root).resolve();manifest=protocol_identity(root/'manifest.json')
    if manifest.get('kind')!='native_LIBERO_Plus_half_V14_D2_supplement': raise ValueError('Independent V14 D2 experiment required')
    ledger_binding=manifest['audit_ledger']
    if sha(ledger_binding['path'])!=ledger_binding['sha256']: raise ValueError('Old585 audit ledger changed')
    old=json.loads(Path(ledger_binding['path']).read_text())
    old_by={(r['group'],r['source_index'],r['mode']):r for r in old['selected_completed']}
    if len(old_by)!=585:raise ValueError('Exact585 unique historical results required')
    matched=[];technical=[];new_complete=[];missing=[]
    for job in manifest['jobs']:
        good,bad=completed(root,job,manifest['manifest_hash']);technical+=bad
        if not good:missing.append(dict(group=job['group'],source_index=job['source_index']));continue
        new_complete+=good;rs={'v14_d2':good[0]}
        for row in job['paired_results']:
            old_row=old_by[(job['group'],job['source_index'],row['mode'])]
            if old_row['path']!=row['path'] or old_row['sha256']!=row['sha256'] or sha(row['path'])!=row['sha256']:raise ValueError('Historical pairing evidence changed')
            value=json.loads(Path(row['path']).read_text())
            if value.get('status')!='complete' or value.get('case_hash')!=job['case_hash'] or value.get('mode')!=row['mode'] or type(value.get('success')) is not bool:raise ValueError('Historical identity mismatch')
            label='matching_draft' if row['mode'].startswith('draft_') else row['mode'];rs[label]=row
        if set(rs)!=set(LABELS):raise ValueError('Missing historical condition')
        matched.append((job,rs))
    final=len(matched)==195
    if not final and not allow_partial:raise RuntimeError('Four-way195 not complete; pass --allow-partial for common-case denominator')
    summary,paired=summarize(manifest['jobs'],matched);xyz=[];videos=[];timings=[];representatives=[]
    for job,rs in matched:
        common=dict(group=job['group'],source_index=job['source_index'],case_hash=job['case_hash'])
        d2=trajectory(rs['v14_d2']['path'])
        for reference in ('full_d10','v14_d10','matching_draft'):
            xyz.append(dict(**common,reference=reference,**actual_distance(d2,trajectory(rs[reference]['path']))))
        if job['group']=='perturbed350' and job['source_index'] in VIDEOS:
            representative=dict(**common,category=job['case']['category'],trajectories={})
            for mode,row in rs.items():
                tr=trajectory(row['path']);representative['trajectories'][mode]=dict(steps=sorted(tr),xyz_world_m=[tr[s].tolist() for s in sorted(tr)])
            representatives.append(representative)
        for mode,row in rs.items():
            if include_costs:timings.append(dict(**common,mode=mode,result_path=row['path'],**costs(row['path'])))
    # Index all preregistered representative videos, even when D2 is pending.
    for job in manifest['jobs']:
        if job['group']!='perturbed350' or job['source_index'] not in VIDEOS:continue
        rs={('matching_draft' if r['mode'].startswith('draft_') else r['mode']):r for r in job['paired_results']}
        good,_=completed(root,job,manifest['manifest_hash'])
        if good:rs['v14_d2']=good[0]
        for mode in LABELS:
            row=rs.get(mode);path=Path(row['path']).parent/'video.mp4' if row else None
            videos.append(dict(source_index=job['source_index'],category=job['case']['category'],mode=mode,
                status='available' if path and path.exists() else 'missing_preserved_no_rerun',path=str(path) if path and path.exists() else '',
                reused_historical=mode!='v14_d2'))
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    report=dict(status='final' if final else 'partial',manifest_hash=manifest['manifest_hash'],manifest_sha256=sha(root/'manifest.json'),
        source_sha256=sha(__file__),target_new_conditions=195,completed_new_conditions=len(new_complete),reused_historical_conditions=585,
        common_four_way_cases=len(matched),missing_new_conditions=missing,technical_attempts_excluded=technical,
        summary=summary,paired=paired,actual_xyz=xyz,representative_actual_xyz=representatives,videos=videos,request_costs=timings,
        cross_gpu_limitation=manifest['cross_gpu_limitation'],
        limitations=['Perturbed175 and original20 are separate populations; all displayed comparison rates use the same four-way completed cases within each stratum.',
                    'Descriptive runtime spans historical GPU3 and new assigned GPU and includes actual environment/inference progression; not a controlled hardware speedup.',
                    'Actual XYZ separation compares closed-loop trajectories at exact shared native steps; old traces have no step0; no interpolation or padded tail.',
                    'Missing historical video is indexed and preserved without rerunning baseline.'])
    write_json(out/'summary.json',report);csv_write(out/'success_four_way.csv',summary);csv_write(out/'paired_v14_d2.csv',paired)
    csv_write(out/'actual_xyz.csv',xyz);csv_write(out/'videos.csv',videos)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--allow-partial',action='store_true');p.add_argument('--skip-request-costs',action='store_true');a=p.parse_args()
    report=analyze(a.root,a.output,a.allow_partial,not a.skip_request_costs)
    print(json.dumps({k:report[k] for k in ('status','completed_new_conditions','common_four_way_cases','reused_historical_conditions')}))


if __name__=='__main__':main()
