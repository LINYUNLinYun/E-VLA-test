"""CPU count observer,complete-only acceptance,independent audit,and pair freezing."""
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
REPORT=Path(__file__).resolve().parent;STUDY=c.FAMILY_ROOT/'discovery'


def read(path):return json.loads(Path(path).read_text())


def progress():
    counts={cid:len(list((STUDY/cid/'episodes').glob('*/task*/init*/episode_receipt.json'))) for cid in c.NEW_CONDITIONS}
    markers=len(list(STUDY.glob('*/discovery_shard*of02_complete.json')))
    attention=any(read(p)['phase']=='attention_holding' for p in (ROOT/'results/ease_insights_20261103/resource_pool_v4/sessions').glob('*/status.json'))
    value=dict(time=time.time(),completed=counts,total_new=sum(counts.values()),completed_shard_markers=markers,attention=attention,
        all_complete=all(v==80 for v in counts.values()) and markers==4,partial_success_values_read=False)
    c.atomic_json(REPORT/'progress.json',value);return value


def finish():
    target=REPORT/'discovery_pipeline_acceptance.json'
    if target.exists():
        old=read(target);c.require(old['status']=='pass' and all(c.sha(p)==h for p,h in old['artifacts'].items()),'Prior acceptance changed')
        return dict(status='already_complete',selection_status=old['selection_status'])
    state=progress();c.require(state['all_complete'] and not state['attention'],'Await all160 cases and4completed services')
    m=c.read_manifest(STUDY/'manifest.json',verify_sources=True)
    for name in ('finish_discovery.py','discovery_analysis.py','discovery_audit.py','statistics_audit.py'):
        p=REPORT/name;c.require(c.sha(p)==m['source_files'][str(p.relative_to(ROOT))],'Discovery auditor changed: '+name)
    analysis=STUDY/'analysis/analysis.json'
    commands=[
        [str(c.MODEL_PYTHON),'-B',str(REPORT/'discovery_audit.py')],
        [str(c.MODEL_PYTHON),'-B','-m','reports.ease_insights_20261103.camera_paths_v6.discovery_analysis','--root',str(STUDY),'--output',str(analysis.parent)],
        [str(c.MODEL_PYTHON),'-B',str(REPORT/'statistics_audit.py'),'--analysis',str(analysis),'--case-audit',str(REPORT/'discovery_acceptance.json'),'--output',str(REPORT/'independent_statistics.json')]]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1')
    with (REPORT/'discovery_acceptance.log').open('a') as log:
        for cmd in commands:
            log.write(json.dumps(dict(time=time.time(),command=cmd))+'\n');log.flush()
            subprocess.run(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
    accepted,independent,report=read(REPORT/'discovery_acceptance.json'),read(REPORT/'independent_statistics.json'),read(analysis)
    c.require(accepted['status']=='pass' and independent['status']=='pass' and independent['differences']==[] and report['complete'],'Complete independent acceptance failed')
    c.require(all(v['manifest_hash']==m['manifest_hash'] for v in (accepted,independent,report)),'Mixed scientific identities')
    from reports.ease_insights_20261103.camera_paths_v6.discovery_analysis import freeze
    selection=freeze(analysis.parent,report)
    files=(STUDY/'manifest.json',REPORT/'discovery_acceptance.json',REPORT/'independent_statistics.json',analysis,selection)
    value=dict(status='pass',time=time.time(),manifest_hash=m['manifest_hash'],selection_status=report['selection']['status'],new_cases=160,preserved_control_cells=160,
        artifacts={str(p):c.sha(p) for p in files},condition_counts={r['condition']:r['successes'] for r in report['condition_summaries'] if r['level']=='all'},
        source=dict(path=str(Path(__file__).resolve()),sha256=c.sha(__file__)),scientific_records_modified=False,goal_complete=False,confirmation_launched=False)
    c.atomic_json(target,value);return {k:v for k,v in value.items() if k not in ('artifacts','source')}


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--wait',action='store_true');a=p.parse_args()
    handle=(REPORT/'discovery_completion.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.wait:
        while True:
            state=progress()
            if state['attention']:raise RuntimeError('Technical attention;no automatic retries or quality analysis')
            if state['all_complete']:break
            time.sleep(2)
    print(json.dumps(finish()),flush=True)
