"""Offline Source features only: no Target, test answers, or configuration selection."""
import time
from pathlib import Path
from experiments.pi05_hidden_probe.backend import Backend
from experiments.pi05_hidden_probe.common import FeatureCache,load_config,model_identity,write_json
from experiments.pi05_hidden_probe.data import read_manifest,load_observation

base=Path(__file__).resolve().parent
config=load_config(base/'source_extraction_calibration.yaml')
rows=[r for r in read_manifest(config['manifest']) if r['split']=='test']
start=time.monotonic()
backend=Backend(**config['model'])
identity=model_identity(config['model'],backend)
layers=list(range(backend.depth+1))
cache=FeatureCache(config['cache_dir'],identity)
index=[]
for row in rows:
    obs=load_observation(row)
    for probe in config['probes']:
        if probe['task_type']!=row['task_type']:continue
        _,positions,key=cache.get(backend,obs,probe['source_selector'],layers,row.get('object_text'))
        index.append({'sample_id':row['sample_id'],'probe_id':probe['id'],'cache_key':key,'layers':layers,'positions':positions})
    print('extracted test source',row['sample_id'],flush=True)
out=base/'source_extraction_test'
write_json(out/'source_index.json',{'source_identity':identity,'samples':len(rows),'features':index,
    'scope':'frozen Source extraction only; no test-set Target decoding, scores, or target-layer selection'})
write_json(out/'complete.json',{'completed':True,'samples':len(rows),'feature_records':len(index),'seconds':time.monotonic()-start})
