"""Verify full coverage, scientific identity, disjoint groups, and completion before merging."""
import argparse,hashlib,json
from pathlib import Path
from experiments.pi05_hidden_probe.common import load_config,select_rows,config_hash,digest,read_jsonl,write_json
from experiments.pi05_hidden_probe.workflow import expected_result_keys,row_key
from experiments.pi05_hidden_probe.data import write_jsonl

p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--shards',type=int,default=2);a=p.parse_args()
config=load_config(a.config);out=Path(config['output_dir']);rows=select_rows(config)
identities=[];records=[];sample_ids=set();seen_groups=set()
for i in range(a.shards):
    shard=out/f'shard{i:02d}'
    identity=json.loads((shard/'run_identity.json').read_text())
    completed=json.loads((shard/'completion.json').read_text())
    raw=(shard/'probe_results.jsonl').read_bytes()
    if not completed.get('completed') or hashlib.sha256(raw).hexdigest()!=completed['results_sha256']:raise ValueError('Shard incomplete or changed')
    if identity['config_hash']!=config_hash(config):raise ValueError('Scientific config changed')
    current=set(identity['sample_ids'])
    if current&sample_ids:raise ValueError('Overlapping sample IDs')
    own_rows=[r for r in rows if r['sample_id'] in current]
    groups={r['group_id'] for r in own_rows}
    if groups&seen_groups:raise ValueError('Independent groups crossed shard boundary')
    if digest(own_rows)!=identity['manifest_digest']:raise ValueError('Shard manifest changed')
    sample_ids|=current;seen_groups|=groups
    identities.append(identity);records+=read_jsonl(shard/'probe_results.jsonl')
if sample_ids!={r['sample_id'] for r in rows}:raise ValueError('Missing global samples')
first=identities[0]
for identity in identities[1:]:
    for key in ['source_identity','target_identity']:
        if identity[key]!=first[key]:raise ValueError(f'{key} differs across shards')
layers=config['source_layers']
if layers=='all':layers=list(range(first['source_identity']['audit']['depth']+1))
keys=[row_key(r) for r in records]
if len(keys)!=len(set(keys)) or set(keys)!=expected_result_keys(config,rows,layers):raise ValueError('Global result matrix is incomplete')
merged={**first,'sample_ids':sorted(sample_ids),'manifest_digest':digest(rows),
        'source_input_digests':{k:v for identity in identities for k,v in identity['source_input_digests'].items()}}
records.sort(key=lambda r:row_key(r))
write_jsonl(out/'probe_results.jsonl',records)
write_json(out/'run_identity.json',merged)
write_json(out/'completion.json',{'completed':True,'result_rows':len(records),'run_identity_digest':digest(merged),
           'results_sha256':hashlib.sha256((out/'probe_results.jsonl').read_bytes()).hexdigest(),'shards':a.shards})
print(out/'probe_results.jsonl',len(records),flush=True)
