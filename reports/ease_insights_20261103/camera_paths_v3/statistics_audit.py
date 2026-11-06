"""Independent raw-count/cost audit; original control records retain identities."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import time

ROOT=Path('/root/nfs/embodied/benchmark')
STUDY=ROOT/'results/ease_insights_20261103/camera_paths_v3/discovery'
REFERENCE=ROOT/'results/ease_insights_20261103/vlm_diversity_v2/discovery'



def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def require(value,message):
    if not value:raise ValueError(message)


def independent_decision(counts,ratios,suites):
    totals={c:sum(sum(counts[c][s]) for s in suites) for c in counts}
    candidates=['agent_fresh_d5','wrist_fresh_d5']
    qualified={c:totals[c]>=60 and totals['full_d10']-totals[c]<=12 and ratios[c]>=1.15 for c in candidates}
    crossings=[]
    for a,b in itertools.combinations(candidates,2):
        if not (qualified[a] and qualified[b]) or max(ratios[a],ratios[b])>1.10*min(ratios[a],ratios[b]):continue
        for x,y in itertools.permutations(suites,2):
            da=[v-w for v,w in zip(counts[a][x],counts[b][x])]
            db=[v-w for v,w in zip(counts[b][y],counts[a][y])]
            sa,sb=sum(da),sum(db)
            if (sa<2 or sb<2 or sum(v>0 for v in da)<2 or sum(v>0 for v in db)<2
                or any(sa-v<=0 for v in da) or any(sb-v<=0 for v in db)):continue
            crossings.append(dict(paths=[a,b],directions=[dict(suite=x,better=a,worse=b,discovery_difference=sa/20),
                dict(suite=y,better=b,worse=a,discovery_difference=sb/20)],ranking_counts=[min(sa,sb),sa+sb,min(totals[a],totals[b])],
                lexical_tie_key=[a,b,x,y]))
    crossings.sort(key=lambda c:tuple(-v for v in c['ranking_counts'])+tuple(c['lexical_tie_key']))
    return dict(status='selected' if crossings else 'no_candidate',selected=crossings[0] if crossings else None,
        qualified=qualified,totals=totals,eligible_crossings=len(crossings))


def audit(analysis_path,case_audit_path,output):
    prelaunch=read(ROOT/'reports/ease_insights_20261103/camera_paths_v3/prelaunch_discovery.json')
    require(sha(STUDY/'manifest.json')==prelaunch['manifest_file_sha256'],'Camera raw manifest changed')
    m=read(STUDY/'manifest.json');payload=dict(m);declared=payload.pop('manifest_hash')
    require(hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()==declared,'Module canonical manifest changed')
    for name,pin in m['source_files'].items():require(sha(ROOT/name)==pin,'Frozen source changed: '+name)
    accepted,reported=read(case_audit_path),read(analysis_path)
    require(accepted.get('status')=='pass' and accepted.get('new_cases')==240 and accepted.get('preserved_control_cells')==80
        and accepted.get('manifest_hash')==declared,'Need complete320-new/80-preserved case acceptance')
    require(reported.get('complete') is True and reported.get('valid_episodes')==320
        and reported.get('new_discovery_episodes')==240 and reported.get('preserved_reference_episodes')==80
        and reported.get('manifest_hash')==declared,'Need complete320-cell analysis')
    suites=m['suites'];ids=m['analytical_conditions'];counts={c:{s:[0]*10 for s in suites} for c in ids};receipts=[]
    for cid in ids:
        source_cid=m['reference_conditions'].get(cid,cid)
        origin=REFERENCE if cid in m['reference_conditions'] else STUDY
        identity=m['preserved_reference']['manifest_hash'] if origin==REFERENCE else declared
        for suite in suites:
            for task in range(10):
                for initial in (45,46):
                    directory=origin/source_cid/'episodes'/suite/f'task{task:02d}'/f'init{initial:02d}'
                    path=directory/'episode_receipt.json';r=read(path);result=read(directory/'result.json')
                    require(type(r['success']) is bool and r['success']==result['success'] and r['condition']==source_cid
                        and r['manifest_hash']==identity and r['case']==dict(suite=suite,task=task,init=initial),'Case origin/outcome identity differs')
                    for name,digest in r['artifacts'].items():require(sha(directory/name)==digest,'Case artifact changed')
                    counts[cid][suite][task]+=int(r['success'])
                    receipts.append(dict(path=str(path),sha256=sha(path),analysis_condition=cid,source_condition=source_cid,
                        source_manifest_hash=identity,new_case=origin==STUDY))
    costs=[]
    for path in sorted(STUDY.glob('*/runs/*/fixed_input_cost.json')):
        done,raw=read(path.parent/'result.json'),read(path)
        require(done.get('complete') is True and done['manifest_hash']==declared and raw['manifest_hash']==declared,'Incomplete cost service')
        require([s['suite'] for s in raw['suites']]==suites,'Cost suite coverage differs')
        sums={c:0. for c in ids};numbers={c:0 for c in ids}
        for suite in raw['suites']:
            require(sorted(b['condition'] for b in suite['conditions'])==sorted(ids),'Cost configuration set differs')
            for block in suite['conditions']:
                values=[r['policy_wall_ms'] for r in block['measured']];cid=block['condition']
                require(len(values)==20 and all(math.isfinite(v) and v>0 for v in values),'Invalid native cost')
                sums[cid]+=sum(values);numbers[cid]+=len(values)
        require(set(numbers.values())=={80},'Unequal native cost sample count')
        costs.append(dict(path=str(path),sha256=sha(path),ratios={c:sums['full_d10']/sums[c] for c in ids}))
    require(len(costs)==6,'Need all six completed new services')
    ratios={c:statistics.median(row['ratios'][c] for row in costs) for c in ids}
    actual=independent_decision(counts,ratios,suites);expected=reported['selection'];differences=[]
    def compare(field,a,b):
        if a!=b:differences.append(dict(field=field,reconstructed=a,reported=b))
    compare('status',actual['status'],expected['status'])
    compare('eligible_crossings',actual['eligible_crossings'],len(expected['eligible_crossings']))
    for cid in ids:
        if cid in ('agent_fresh_d5','wrist_fresh_d5'):
            gate=expected['condition_eligibility'][cid]
            compare(cid+'/qualified',actual['qualified'][cid],gate['qualified'])
            compare(cid+'/successes',actual['totals'][cid],gate['successes'])
            require(abs(ratios[cid]-gate['median_fixed_input_speed_ratio'])<1e-10,'Production native cost ratio differs')
        for suite in suites:
            rows=[r for r in reported['condition_summaries'] if r['level']=='suite' and r['condition']==cid and r['suite']==suite]
            require(len(rows)==1,'Missing suite summary');compare(cid+'/'+suite,sum(counts[cid][suite]),rows[0]['successes'])
    if actual['selected']:
        for key in ('paths','directions','ranking_counts','lexical_tie_key'):compare('selected/'+key,actual['selected'][key],expected['selected'][key])
    else:compare('selected',None,expected['selected'])
    value=dict(status='pass' if not differences else 'fail',differences=differences,manifest_hash=declared,
        analytical_cells=len(receipts),new_cases=sum(r['new_case'] for r in receipts),preserved_control_cells=sum(not r['new_case'] for r in receipts),
        preserved_controls_are_new_replications=False,independent_decision=actual,task_success_counts=counts,cost_median_ratios=ratios,
        case_evidence=receipts,cost_evidence=costs,analysis=dict(path=str(analysis_path),sha256=sha(analysis_path)),
        case_audit=dict(path=str(case_audit_path),sha256=sha(case_audit_path)),
        auditor_source=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),time=time.time(),
        production_selection_imported=False,scientific_records_modified=False,
        scope='Independent integer counts, original control identities and native-cost sums with10% pair gate; no new inferential claim')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(value,indent=2)+'\n')
    return {k:v for k,v in value.items() if k not in ('case_evidence','cost_evidence','task_success_counts')}


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--case-audit',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();value=audit(a.analysis,a.case_audit,a.output);print(json.dumps(value));raise SystemExit(0 if value['status']=='pass' else 1)
