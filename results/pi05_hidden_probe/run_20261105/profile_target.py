"""Measure actual patched-reader cost before choosing any inference optimization."""
import time,json
from pathlib import Path
import numpy as np
from experiments.pi05_hidden_probe.hf_backend import HFBackend
from experiments.pi05_hidden_probe.backend import Patch,select_positions
from experiments.pi05_hidden_probe.common import control_observation,write_json
from experiments.pi05_hidden_probe.data import read_manifest,load_observation

base=Path(__file__).resolve().parent
rows=[r for r in read_manifest(base/'counterfactual_data_v2/manifest.jsonl') if r['split']=='calibration' and r['task_type']=='visual_attribute']
row=rows[0]
backend=HFBackend('/root/nfs/embodied/benchmark/shared/checkpoints/pi05_hidden_probe/paligemma-3b-mix-224')
prefix=backend.prepare_target(control_observation(load_observation(row),{'kind':'constant','value':127}),row['question'])
index=json.loads((base/'source_extraction_calibration/source_index.json').read_text())
feature=next(f for f in index['features'] if f['sample_id']==row['sample_id'] and f['probe_id']=='visual_image')
with np.load(base/'external_source_cache'/(feature['cache_key']+'.npz'),allow_pickle=False) as data:
    patch=Patch(6,select_positions(prefix,'image:base_0_rgb'),data['layer_6'].copy())
records=[]
for iteration in range(7):
    backend.torch.cuda.synchronize()
    start=time.monotonic()
    generation=backend.generate(prefix,patch=patch,max_new_tokens=8)
    scores=backend.score_answers(prefix,row['labels'],patch=patch)
    backend.torch.cuda.synchronize()
    record={'iteration':iteration,'warmup':iteration<2,'seconds':time.monotonic()-start,'generation':generation,'scores':scores}
    records.append(record);print(record,flush=True)
warm=[r['seconds'] for r in records if not r['warmup']]
write_json(base/'target_profile.json',{'records':records,'mean_warm_seconds_per_record':float(np.mean(warm)),
    'estimated_calibration_two_gpu_seconds':38400*float(np.mean(warm))/2,
    'estimated_evaluation_two_gpu_seconds':54656*float(np.mean(warm))/2,
    'max_memory_allocated_bytes':backend.torch.cuda.max_memory_allocated(),'scope':'one deterministic calibration sample, actual pi05 layer-6 visual features, Target layer6; not a semantic accuracy estimate'})
