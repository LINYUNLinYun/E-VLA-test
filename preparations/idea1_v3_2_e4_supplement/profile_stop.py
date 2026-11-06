"""Stop an owned profiling sampler/redispatcher, drain its current warmup children.

No scientific controller, model server, or live renderer child is signalled.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import time
from .native_envs import write_json,sha
from .parallel_continue import process_identity,same_process


def main():
    p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);p.add_argument('--receipt',required=True);a=p.parse_args()
    identity=process_identity(a.pid)
    if not identity or 'supplement.profile_plus_parallel ' not in identity['argv'] or '--child-index ' in identity['argv']:
        raise ValueError('Not the owned profile parent; no signal sent')
    path=Path(a.receipt);row=dict(status='draining',parent=identity,source_sha256=sha(__file__),started_unix=time.time(),signals=[])
    os.kill(a.pid,signal.SIGSTOP);row['signals'].append(dict(pid=a.pid,signal='SIGSTOP',reason='Stop asymmetric highfrequencyGPUobserver and avoid secondserial redispatch'))
    children_file=Path('/proc')/str(a.pid)/'task'/str(a.pid)/'children'
    prior_path=path.parent/'controlled_stop.json'
    prior=json.loads(prior_path.read_text()) if prior_path.exists() else {}
    known={child['pid']:child for child in prior.get('children',[]) if child}
    children=[process_identity(int(pid)) for pid in children_file.read_text().split()]
    if any(child and child['state']!='Z' and ('supplement.profile_plus_parallel ' not in child['argv'] or '--child-index ' not in child['argv']) for child in children):
        os.kill(a.pid,signal.SIGCONT);raise ValueError('Unexpected child; parent resumed')
    children=[known.get(child['pid'],child) if child and child['state']=='Z' else child for child in children]
    row['children']=children;write_json(path.parent/'controlled_stop.json',row)
    while any(child and same_process(child) and process_identity(child['pid'])['state']!='Z' for child in children): time.sleep(.2)
    if same_process(identity): os.kill(a.pid,signal.SIGTERM);os.kill(a.pid,signal.SIGCONT)
    original=json.loads(path.read_text());write_json(path.parent/'receipt_before_controlled_stop.json',original)
    original.update(status='complete',deploy_parallel=False,controlled_early_stop=True,completed_unix=time.time(),
        decision_reason='First pair0.692x contains asymmetric GPUobserver overhead; not acceptable deployment evidence. Secondparallel children drained; unrun secondserial omitted. No productionhandoff.',
        observer_confound='Parallel-only repeated nvidia-smi; retained negative result is not isolated evidence batching/parallelism intrinsically unhelpful')
    write_json(path,original)
    row.update(status='complete',completed_unix=time.time(),child_signals_sent=False)
    write_json(path.parent/'controlled_stop.json',row);print(json.dumps(row))


if __name__=='__main__': main()
