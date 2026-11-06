"""Run one disjoint group shard with the unchanged scientific configuration."""
import argparse,os
from pathlib import Path
from experiments.pi05_hidden_probe.common import load_config,select_rows,make_backends,prepare_run,write_json
from experiments.pi05_hidden_probe.workflow import integrity_check,run_probe

p=argparse.ArgumentParser()
p.add_argument('--config',required=True)
p.add_argument('--shard-index',type=int,required=True)
p.add_argument('--shards',type=int,default=2)
p.add_argument('--resume',action='store_true')
a=p.parse_args()
if not 0<=a.shard_index<a.shards:raise ValueError('Invalid shard')
config=load_config(a.config)
rows=select_rows(config)
groups=sorted({r['group_id'] for r in rows})
owned={g for i,g in enumerate(groups) if i%a.shards==a.shard_index}
rows=[r for r in rows if r['group_id'] in owned]
config['output_dir']=str(Path(config['output_dir'])/f'shard{a.shard_index:02d}')
source,target=make_backends(config)
out,identity=prepare_run(config,rows,source,target,command='probe')
write_json(out/'device.json',{'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'shard_index':a.shard_index,'shards':a.shards})
report=integrity_check(source,target,rows[0],config)
report['identity']=identity
write_json(out/'patch_integrity.json',report)
if not report['passed']:raise RuntimeError('Stop: integrity gate failed')
print(run_probe(config,rows,source,target,out,identity,resume=a.resume),flush=True)
