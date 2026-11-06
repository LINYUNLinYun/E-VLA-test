"""Run fixed read-only acceptance and freeze only after all240 cases/six shards finish."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/root/nfs/embodied/benchmark');sys.path.insert(0,str(ROOT))
REPORT=Path(__file__).resolve().parent;STUDY=ROOT/'results/ease_insights_20261103/camera_paths_v3/discovery'
PYTHON=ROOT/'policies/realtime-vla-flash/.venv/bin/python'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ready():
    v=read(REPORT/'progress.json')
    return (v['all_complete'] is True and v['total_new']==240 and v['completed_shard_markers']==6 and not v['attention']),v


def run():
    from preparations.ease_insights_20261103.camera_paths_v3 import common
    target=REPORT/'discovery_pipeline_acceptance.json'
    if target.exists():
        prior=read(target)
        if prior['status']!='pass' or any(sha(p)!=h for p,h in prior['artifacts'].items()):raise ValueError('Prior acceptance changed')
        return dict(status='already_complete',selection_status=prior['selection_status'])
    ok,progress=ready()
    if not ok:return dict(status='pending',total_new=progress['total_new'])
    m=common.read_manifest(STUDY/'manifest.json',verify_sources=True);analysis=STUDY/'analysis/analysis.json'
    commands=[
        [str(PYTHON),str(REPORT/'discovery_audit.py')],
        [str(PYTHON),'-m','preparations.ease_insights_20261103.camera_paths_v3.analysis_entry','--root',str(STUDY),'--output',str(analysis.parent)],
        [str(PYTHON),str(REPORT/'statistics_audit.py'),'--analysis',str(analysis),'--case-audit',str(REPORT/'discovery_acceptance.json'),'--output',str(REPORT/'independent_statistics.json')]]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    with (REPORT/'acceptance_pipeline.log').open('a') as log:
        for command in commands:
            log.write(json.dumps(dict(time=time.time(),command=command))+'\n');log.flush()
            subprocess.run(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
    cases=read(REPORT/'discovery_acceptance.json');statistics=read(REPORT/'independent_statistics.json');value=read(analysis)
    if (cases['status']!='pass' or statistics['status']!='pass' or statistics['differences']!=[]
        or not value['complete'] or value['new_discovery_episodes']!=240 or value['preserved_reference_episodes']!=80):raise ValueError('Complete acceptance failed')
    if any(v['manifest_hash']!=m['manifest_hash'] for v in (cases,statistics,value)):raise ValueError('Mixed scientific identities')
    from preparations.ease_insights_20261103.camera_paths_v3.selection import freeze
    selection=freeze(analysis.parent,value)
    result=dict(status='pass',time=time.time(),manifest_hash=m['manifest_hash'],selection_status=value['selection']['status'],
        new_cases=240,preserved_control_cells=80,preserved_controls_are_new_replications=False,source_pins=len(m['source_files']),
        artifacts={str(p):sha(p) for p in (STUDY/'manifest.json',REPORT/'discovery_acceptance.json',REPORT/'independent_statistics.json',analysis,selection)},
        condition_counts={r['condition']:r['successes'] for r in value['condition_summaries'] if r['level']=='all'},
        source=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),scientific_records_modified=False,
        goal_complete=False,confirmation_launched=False,
        next='Prepare independent confirmation only if selected; otherwise preserve no_candidate and continue the overall goal')
    common.atomic_json(target,result)
    return {k:v for k,v in result.items() if k not in ('artifacts','source')}


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    lock=(REPORT/'discovery_completion.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.wait:
        while not ready()[0]:
            _,progress=ready()
            if progress['attention']:raise RuntimeError('Science needs attention; do not retry or select')
            time.sleep(2)
    print(json.dumps(run()),flush=True)
