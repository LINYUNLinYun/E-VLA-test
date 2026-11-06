"""Count-only waiting thencomplete800-case reconstruction,analysis and independent audit."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path('/root/nfs/embodied/benchmark');sys.path.insert(0,str(ROOT))
from preparations.ease_insights_20261103.camera_paths_v6 import common as c
REPORT=Path(__file__).resolve().parent;STUDY=c.FAMILY_ROOT/'confirmation'


def read(path):return json.loads(Path(path).read_text())


def progress():
    counts={cid:len(list((STUDY/cid/'episodes').glob('*/task*/init*/episode_receipt.json'))) for cid in c.CONDITIONS}
    markers=len(list(STUDY.glob('*/confirm_shard*of02_complete.json')))
    attention=any(read(p)['phase']=='attention_holding' for p in (ROOT/'results/ease_insights_20261103/resource_pool_v4/sessions').glob('*/status.json'))
    value=dict(time=time.time(),completed=counts,total_new=sum(counts.values()),completed_shard_markers=markers,expected_new=800,
        attention=attention,all_complete=all(n==200 for n in counts.values()) and markers==8,partial_success_values_read=False)
    c.atomic_json(REPORT/'confirmation_progress.json',value);return value


def finish():
    target=REPORT/'confirmation_pipeline_acceptance.json'
    if target.exists():
        old=read(target);c.require(old['status']=='pass' and all(c.sha(p)==h for p,h in old['artifacts'].items()),'Prior complete acceptance changed')
        return dict(status='already_complete',conclusion=old['conclusion'])
    state=progress();c.require(state['all_complete'] and not state['attention'],'Await800 new cases and eight completed services')
    m=c.read_manifest(STUDY/'manifest.json',verify_sources=True)
    for name in ('finish_confirmation.py','confirmation_audit.py','confirmation_analysis.py','confirmation_load.py','confirmation_math.py','confirmation_statistics_audit.py'):
        p=REPORT/name;c.require(c.sha(p)==m['source_files'][str(p.relative_to(ROOT))],'Frozen confirmation source changed: '+name)
    analysis=REPORT/'confirmation_analysis/analysis.json';case_audit=REPORT/'confirmation_acceptance.json';independent=REPORT/'confirmation_independent_statistics.json'
    commands=[
        [str(c.MODEL_PYTHON),'-B',str(REPORT/'confirmation_audit.py')],
        [str(c.MODEL_PYTHON),'-B','-m','reports.ease_insights_20261103.camera_paths_v6.confirmation_analysis','--root',str(STUDY),'--output',str(analysis.parent)],
        [str(c.MODEL_PYTHON),'-B',str(REPORT/'confirmation_statistics_audit.py'),'--analysis',str(analysis),'--case-audit',str(case_audit),'--output',str(independent)]]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1')
    with (REPORT/'confirmation_acceptance.log').open('a') as log:
        for cmd in commands:
            log.write(json.dumps(dict(time=time.time(),command=cmd))+'\n');log.flush()
            subprocess.run(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
    accepted,reported,stats=read(case_audit),read(analysis),read(independent)
    c.require(accepted['status']=='pass' and stats['status']=='pass' and stats['differences']==[] and reported['complete'],'Complete independent acceptance failed')
    c.require(all(v['manifest_hash']==m['manifest_hash'] for v in (accepted,reported,stats)) and stats['new_cases']==800 and stats['preserved_control_cells']==0,'Mixed or incomplete confirmation identity')
    files=(STUDY/'manifest.json',case_audit,analysis,independent,Path(m['selection_evidence']['path']))
    value=dict(status='pass',time=time.time(),manifest_hash=m['manifest_hash'],new_cases=800,preserved_control_cells=0,
        conclusion=reported['conclusion'],success_counts=stats['success_counts'],native_cost_ratios=stats['cost_ratios'],
        primary_directions=stats['directions'],DenseD2_dominates_both_at_no_higher_native_cost=reported['DenseD2_dominates_both_at_no_higher_native_cost'],
        adaptive_routing_benefit_established=False,artifacts={str(p):c.sha(p) for p in files},
        source=dict(path=str(Path(__file__).resolve()),sha256=c.sha(__file__)),scientific_records_modified=False,
        goal_complete=False,requires_overall_goal_completion_audit=True,no_reselection=True,no_added_significance_rollouts=True)
    c.atomic_json(target,value);return {k:v for k,v in value.items() if k not in ('artifacts','source')}


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--wait',action='store_true');a=p.parse_args()
    lock=(REPORT/'confirmation_completion.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    c.read_manifest(STUDY/'manifest.json',verify_sources=True)
    if a.wait:
        while True:
            state=progress()
            if state['attention']:raise RuntimeError('Technical attention;no retries or partial quality analysis')
            if state['all_complete']:break
            time.sleep(2)
    print(json.dumps(finish()),flush=True)
