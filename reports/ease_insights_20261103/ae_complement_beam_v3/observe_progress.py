"""Read-only2s result-count observer; never reads success or queries GPUs."""
import json
import fcntl
import os
from pathlib import Path
import time

ROOT=Path('/root/nfs/embodied/benchmark')
STUDY=ROOT/'results/ease_insights_20261103/ae_complement_beam_v3/discovery'
REPORT=Path(__file__).resolve().parent
RESOURCE=ROOT/'results/ease_insights_20261103/resource_pool_v4'

if __name__=='__main__':
    singleton=(REPORT/'progress_observer.lock').open('a')
    fcntl.flock(singleton,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        started_at=time.time(),scope='Read-only module new-case progress; no success read/GPU query/science launch')
    (REPORT/'progress_observer.json').write_text(json.dumps(identity,indent=2)+'\n');last=None
    while True:
        counts={cid:sum(p.parent.name.startswith('init') and p.parent.name[4:].isdigit()
            for p in (STUDY/cid/'episodes').glob('**/episode_receipt.json')) for cid in ('path_a','path_b','path_c','path_d')}
        holders=[];attention=False
        for slot in sorted((RESOURCE/'sessions').iterdir()):
            status=json.loads((slot/'status.json').read_text());attention|=status['phase']=='attention_holding'
            holders.append(dict(gpu_uuid=slot.name,phase=status['phase'],condition=status.get('condition'),guardian_pid=status.get('guardian_pid')))
        markers=list(STUDY.glob('*/discovery_shard*of02_complete.json'))
        complete=sum(counts.values())==320 and len(markers)==8
        value=dict(time=time.time(),observer=identity,new_completed=counts,total_new=sum(counts.values()),expected_new=320,
            preserved_control_cells=160,new_replications_of_preserved_controls=0,completed_shard_markers=len(markers),
            all_complete=complete,attention=attention,holders=holders,scientific_successes_read=False,selection_run=False)
        path=REPORT/'progress.json';temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)
        key=(sum(counts.values())//10,tuple((h['phase'],h['condition']) for h in holders),complete,attention)
        if key!=last:
            print(json.dumps({k:v for k,v in value.items() if k!='observer'}),flush=True);last=key
        if complete or attention:break
        time.sleep(2)
