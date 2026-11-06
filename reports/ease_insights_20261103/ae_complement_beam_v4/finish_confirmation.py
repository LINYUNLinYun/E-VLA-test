"""CPU-only complete720 acceptance; no allocation, scientific retry, or partial inference."""
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
REPORT=Path(__file__).resolve().parent
BUNDLE=REPORT/'confirmation_freeze.json'
FAMILY=ROOT/'results/ease_insights_20261103/ae_complement_beam_v4'
ROOTS=(FAMILY/'confirmation',FAMILY)
PYTHON=ROOT/'policies/realtime-vla-flash/.venv/bin/python'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(bundle_pin,source_pin):
    if sha(__file__)!=source_pin or sha(BUNDLE)!=bundle_pin:raise ValueError('Pinned completion driver/bundle changed')
    from preparations.ease_insights_20261103.ae_complement_beam_v4.prepare_confirmation import read_bundle
    return read_bundle(BUNDLE,verify_sources=True)


def progress(identity,bundle):
    streams=[]
    for repeat,root in enumerate(ROOTS):
        counts={cid:sum(p.parent.name.startswith('init') and p.parent.name[4:].isdigit()
            for p in (root/cid/'episodes').glob('*/task*/init*/episode_receipt.json')) for cid in ('full_d10','vlm_c','ae14_d3')}
        markers=len(list(root.glob('*/confirm_shard*of02_complete.json')))
        streams.append(dict(noise_repeat=repeat,completed=counts,new_cases=sum(counts.values()),completed_shards=markers,
            all_complete=all(n==120 for n in counts.values()) and markers==6))
    attention=any(read(p)['phase']=='attention_holding' for p in (ROOT/'results/ease_insights_20261103/resource_pool_v4/sessions').glob('*/status.json'))
    value=dict(time=time.time(),bundle_hash=bundle['bundle_hash'],observer=identity,streams=streams,
        new_cases=sum(s['new_cases'] for s in streams),expected_cases=720,attention=attention,
        all_complete=all(s['all_complete'] for s in streams),successes_read_for_partial_analysis=False)
    from preparations.idea1_v3_1.common import atomic_json
    atomic_json(REPORT/'confirmation_progress.json',value);return value


def run(bundle_pin,source_pin,wait):
    bundle=verify(bundle_pin,source_pin)
    with (REPORT/'confirmation_completion.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return run_locked(bundle,bundle_pin,source_pin,wait)


def run_locked(bundle,bundle_pin,source_pin,wait):
    target=REPORT/'confirmation_pipeline_acceptance.json'
    if target.exists():
        old=read(target)
        if old['status']!='pass' or old['bundle_hash']!=bundle['bundle_hash'] or any(sha(p)!=pin for p,pin in old['artifacts'].items()):
            raise ValueError('Prior confirmation acceptance changed')
        return dict(status='already_complete',conclusion=old['conclusion'])
    identity=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        source_sha256=source_pin,bundle_file_sha256=bundle_pin,time=time.time(),CPU_only=True,GPU_jobs_launched=False)
    from preparations.idea1_v3_1.common import atomic_json
    atomic_json(REPORT/'confirmation_driver.json',identity);last=None
    while True:
        value=progress(identity,bundle);now=(value['new_cases']//10,tuple(s['completed_shards'] for s in value['streams']),value['attention'])
        if now!=last:print(json.dumps({k:v for k,v in value.items() if k!='observer'}),flush=True);last=now
        if value['attention']:raise RuntimeError('Technical attention; retain GPUs and stop inference without retry/exclusion')
        if value['all_complete']:break
        if not wait:return dict(status='pending',new_cases=value['new_cases'])
        time.sleep(2)
    verify(bundle_pin,source_pin)
    commands=[[str(PYTHON),'-B',str(REPORT/'confirmation_audit.py')],
        [str(PYTHON),'-B','-m','preparations.ease_insights_20261103.ae_complement_beam_v4.analyze_confirmation'],
        [str(PYTHON),'-B',str(REPORT/'confirmation_statistics_audit.py')]]
    with (REPORT/'confirmation_acceptance_pipeline.log').open('a') as log:
        for command in commands:
            log.write(json.dumps(dict(time=time.time(),command=command))+'\n');log.flush()
            subprocess.run(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,check=True)
    cases=read(REPORT/'confirmation_acceptance.json');statistics=read(REPORT/'confirmation_independent_statistics.json');analysis=read(REPORT/'confirmation_analysis.json')
    if (cases['status']!='pass' or cases['new_cases']!=720 or statistics['status']!='pass' or statistics['differences']
        or not analysis['complete'] or analysis['valid_episodes']!=720 or statistics['conclusion']!=analysis['conclusion']
        or any(v['bundle_hash']!=bundle['bundle_hash'] for v in (cases,statistics,analysis))):
        raise ValueError('Complete720 independent confirmation acceptance failed')
    verify(bundle_pin,source_pin)
    result=dict(status='pass',time=time.time(),bundle_hash=bundle['bundle_hash'],conclusion=analysis['conclusion'],
        new_cases=720,preserved_control_cells=0,unique_physical_states_per_algorithm=120,rollouts_per_algorithm=240,
        source=dict(path=str(Path(__file__).resolve()),sha256=source_pin),
        artifacts={str(p):sha(p) for p in [BUNDLE,*[r/'manifest.json' for r in ROOTS],REPORT/'confirmation_acceptance.json',
            REPORT/'confirmation_independent_statistics.json',REPORT/'confirmation_analysis.json']},
        success_counts=statistics['success_counts'],directions=statistics['directions'],quality_eligibility=statistics['candidate_qualified'],
        native_cost_ratios=statistics['cost_ratios'],scientific_records_modified=False,goal_complete=False,
        scope='All720 fresh rollouts accepted with independent ten-task arithmetic; overall goal completion requires a separate evidence audit')
    atomic_json(target,result)
    return {k:v for k,v in result.items() if k not in ('artifacts','directions')}


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--bundle-sha256',required=True);p.add_argument('--source-sha256',required=True);p.add_argument('--wait',action='store_true');a=p.parse_args()
    print(json.dumps(run(a.bundle_sha256,a.source_sha256,a.wait)),flush=True)
