"""Predeclared cache gate: exact prefill logits and identical greedy IDs on 32 calibration cases."""
import sys,time,json,hashlib
from pathlib import Path
import numpy as np
from cached_target import CachedHFBackend
from experiments.pi05_hidden_probe.hf_backend import HFBackend
from experiments.pi05_hidden_probe.backend import Patch,select_positions
from experiments.pi05_hidden_probe.common import control_observation,write_json
from experiments.pi05_hidden_probe.data import read_manifest,load_observation

base=Path(__file__).resolve().parent
rows=[r for r in read_manifest(base/'counterfactual_data_v2/manifest.jsonl') if r['split']=='calibration']
import yaml
config=yaml.safe_load((base/'external_calibration.yaml').read_text())
index=json.loads((base/'source_extraction_calibration/source_index.json').read_text())
features={(f['sample_id'],f['probe_id']):f for f in index['features']}
model=CachedHFBackend('/root/nfs/embodied/benchmark/shared/checkpoints/pi05_hidden_probe/paligemma-3b-mix-224')
rng=np.random.default_rng(42)
candidates=[(p,r,l,q,c) for p in config['probes'] for r in rows if r['task_type']==p['task_type']
            for l in config['source_layers'] for q in [3,6,9,12] for c in config['controls']]
selected=rng.choice(len(candidates),size=28,replace=False)
cases=[]
for i in selected:
    probe,row,layer,q,control=candidates[int(i)]
    prompt=probe['prompt'] if probe.get('prompt') is not None else row['question']
    prefix=model.prepare_target(control_observation(load_observation(row),control),prompt)
    feature=features[(row['sample_id'],probe['id'])]
    with np.load(base/'external_source_cache'/(feature['cache_key']+'.npz'),allow_pickle=False) as data:
        values=data[f'layer_{layer}'].copy()
    patch=Patch(q,select_positions(prefix,probe['target_selector'],'X'),values)
    cases.append((prefix,patch,{'sample_id':row['sample_id'],'probe_id':probe['id'],'source_layer':layer,'target_layer':q,'control_id':control['id']}))
for row in [r for r in rows if r['task_type']=='visual_attribute'][:4]:
    prefix=model.prepare_target(load_observation(row),row['question'])
    cases.append((prefix,None,{'sample_id':row['sample_id'],'condition':'direct_positive'}))
records=[]
for prefix,patch,meta in cases:
    model.torch.cuda.synchronize();start=time.monotonic()
    reference=HFBackend.generate(model,prefix,patch=patch,max_new_tokens=8)
    model.torch.cuda.synchronize();reference_seconds=time.monotonic()-start
    reference_logits=model.forward(prefix,patch=patch).logits
    prefill_logits,cache=model._cached_prefill(prefix,patch)
    del cache
    actual_logits=model._array(prefill_logits[0]);del prefill_logits
    model.torch.cuda.synchronize();start=time.monotonic()
    cached=model.generate(prefix,patch=patch,max_new_tokens=8)
    model.torch.cuda.synchronize();cached_seconds=time.monotonic()-start
    record={**meta,'reference':reference,'cached':cached,'token_ids_equal':reference['token_ids']==cached['token_ids'],
            'prefill_logits_exact':bool(np.array_equal(reference_logits,actual_logits)),
            'prefill_logits_max_abs':float(np.max(np.abs(reference_logits-actual_logits))),
            'reference_seconds':reference_seconds,'cached_seconds':cached_seconds}
    records.append(record)
    print(meta,'equal',record['token_ids_equal'],'prefill_exact',record['prefill_logits_exact'],flush=True)
passed=all(r['token_ids_equal'] and r['prefill_logits_exact'] for r in records)
write_json(base/'cached_generation_validation.json',{'passed':passed,'scope':'32 preselected calibration cases, mixed tasks, paths, layers and controls; no test-set tuning',
    'implementation_sha256':hashlib.sha256((base/'cached_target.py').read_bytes()).hexdigest(),
    'criterion':'exact first-token logits and identical full greedy token sequences; no relaxed numeric threshold',
    'reference_mean_seconds':float(np.mean([r['reference_seconds'] for r in records])),
    'cached_mean_seconds':float(np.mean([r['cached_seconds'] for r in records])),'records':records})
if not passed:raise SystemExit('Cache validation failed; keep the reference generator')
