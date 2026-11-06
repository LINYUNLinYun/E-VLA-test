"""Logical ledger plus reconstructed constructor work, not a latency profile."""
from collections import defaultdict
import json
from pathlib import Path
import sqlite3

import torch

from scripts.experiment11.protocol import RESULTS, CONVERTED
from scripts.experiment1.run_intermediate_action_probe import SUITE_ARTIFACT
from scripts.experiment10.common import dump_json, sha256


def count_length_transitions(lengths):
    previous=None;count=0
    for length in lengths:
        if previous!=length:
            count+=1;previous=length
    return count


def main():
    status=json.loads((RESULTS/'experiment11_task_type_probe_status_round2_20261011_warmup_retry1.json').read_text())
    if not status.get('complete') or status.get('validated_jobs')!=40 or status.get('active') or status.get('failures'):
        raise RuntimeError('Internal work accounting waits for complete 40/40 queue')
    directory=RESULTS/'task_type_layer_probe/round2_20261011'
    manifest=json.loads((directory/'manifest.json').read_text())
    samples={s['sample_id']:s for s in manifest['samples']}
    bytask={(s['suite'],s['task_id']):s['prompt'] for s in samples.values()}
    lengths={};source_hashes={str(directory/'manifest.json'):sha256(directory/'manifest.json')}
    for (suite,task),prompt in bytask.items():
        artifact=CONVERTED/SUITE_ARTIFACT[suite]
        path=artifact/'manifest.json';source_hashes[str(path)]=sha256(path)
        entry=json.loads(path.read_text())['prompts'][prompt]
        embedding=artifact/entry['embed_path'];source_hashes[str(embedding)]=sha256(embedding)
        lengths[suite,task]=len(torch.load(embedding,map_location='cpu',weights_only=True))
    database=directory/'call_budget.sqlite'
    connection=sqlite3.connect(f'file:{database}?mode=ro',uri=True)
    rows=connection.execute('select id,pid,identity,status,details from calls order by id').fetchall();connection.close()
    require_complete=not any(status=='reserved' for _,_,_,status,_ in rows)
    if not require_complete:
        raise RuntimeError('Unfinished calls; work accounting must wait for the queue')
    bypid=defaultdict(list)
    for call,pid,identity,status,details in rows:
        bypid[pid].append((call,json.loads(identity),status,json.loads(details)))
    processes=[];recorded_parity=recorded_graph=0
    for pid,calls in bypid.items():
        ids=[r['sample_id'] for _,r,_,_ in calls if 'sample_id' in r]
        suites={r['suite'] for _,r,_,_ in calls if 'suite' in r}
        suites.update(samples[sid]['suite'] for sid in ids if sid in samples)
        if len(suites)!=1:
            raise RuntimeError(f'Ambiguous worker suite: {pid}')
        suite=next(iter(suites));sequence=[]
        for _,identity,_,details in calls:
            kind=identity.get('kind','')
            sid=identity.get('sample_id')
            if sid in samples:
                task=samples[sid]['task_id']
            elif kind in ('original_suite_warmup','parity_history_original_warmup'):
                task=0
            elif kind in ('original_preceding_task_last_replan','parity_history_last_task_observation'):
                task=int(identity['task'])
            else:
                raise RuntimeError(f'Unmapped prepare context: {kind}')
            sequence.append(lengths[suite,task])
            recorded_parity+=details.get('internal_d10_parity_forwards',0)
            recorded_graph+=details.get('internal_warmup_capture_forwards',0)
        constructors=count_length_transitions(sequence)
        processes.append({'pid':pid,'suite':suite,'charged_calls':len(calls),
                          'backend_runtime_constructions_reconstructed':constructors,
                          'constructor_d10_internal_forwards_reconstructed':4*constructors})
    constructors=sum(r['backend_runtime_constructions_reconstructed'] for r in processes)
    code=Path(__file__).resolve().parents[1]/'spec/triton/pi0_infer.py';source_hashes[str(code)]=sha256(code)
    source_hashes[str(Path(__file__).resolve())]=sha256(Path(__file__).resolve())
    result={'complete':True,'charged_logical_calls':len(rows),'processes':processes,
            'runtime_constructions_reconstructed':constructors,'constructor_d10_internal_forwards_reconstructed':4*constructors,
            'ledger_recorded_internal_d10_parity_forwards':recorded_parity,
            'ledger_recorded_internal_graph_warmup_capture_forwards':recorded_graph,
            'source_hashes':source_hashes,'ledger_sha256':sha256(database),
            'method':'Backend creates D10 runtime on each prompt-length change; constructor warms 3 and captures 1 forwards',
            'limitations':['constructor counts reconstructed from prepare contexts,not direct telemetry',
                           'native diagnostic D3 constructor is already in ledger graph/capture work,not double counted',
                           'failed calls remain charged;internal partial-failure work may be unknown',
                           'forward counts are not measured time or new closed-loop evaluations'],
            'new_predictions':0,'environment_steps':0}
    dump_json(RESULTS/'analysis/next_round/round2_20261011/internal_work_accounting.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('processes','source_hashes')}))


if __name__=='__main__':
    main()
