"""External Target task and injection readiness on calibration color pairs only."""
import time
from pathlib import Path
import numpy as np
from PIL import Image
from experiments.pi05_hidden_probe.hf_backend import HFBackend
from experiments.pi05_hidden_probe.backend import Patch,select_positions
from experiments.pi05_hidden_probe.common import append_jsonl,control_observation,write_json
from experiments.pi05_hidden_probe.data import read_manifest,load_observation
from experiments.pi05_hidden_probe.metrics import parse_answer,balanced_accuracy
from experiments.pi05_hidden_probe.workflow import difference

base=Path(__file__).resolve().parent
out=base/'external_text_smoke'
out.mkdir(exist_ok=True)
if (out/'generations.jsonl').exists():raise FileExistsError('Preserve previous measurements; use a new output directory')
rows=[r for r in read_manifest(base/'counterfactual_data_v2/manifest.jsonl') if r['split']=='calibration' and r['task_type']=='visual_attribute']
backend=HFBackend('/root/nfs/embodied/benchmark/shared/checkpoints/pi05_hidden_probe/paligemma-3b-mix-224')
write_json(out/'model_audit.json',backend.audit())
first=load_observation(rows[0]);question=rows[0]['question'];real=backend.prepare_target(first,question)
reference=backend.reference_forward(real)
original=backend.forward(real,capture_layers=[0,6])
positions=select_positions(real,'images')
self_result=backend.forward(real,patch=Patch(6,positions,original.hidden_states[6][positions]))
blank=backend.prepare_target(control_observation(first,{'kind':'constant','value':127}),question)
layer0=backend.forward(blank,patch=Patch(0,positions,real.embedded[positions]))
checks={'reference_normalized':difference(original.normalized,reference.normalized,1e-5,1e-5),
        'reference_logits':difference(original.logits,reference.logits,1e-5,1e-5),
        'self_patch':difference(self_result.normalized,original.normalized,1e-5,1e-5),
        'layer0':difference(layer0.normalized,original.normalized,1e-5,1e-5)}
write_json(out/'patch_integrity.json',checks)
if not all(c['passed'] for c in checks.values()):raise RuntimeError('External Target integrity failed')
records=[]
start=time.monotonic()
for row in rows:
    obs=load_observation(row)
    for condition in ['direct','gray','black','official_single_image']:
        if condition=='official_single_image':
            inputs=backend._processor(text=row['question'].rstrip('\n'),images=Image.fromarray(obs['observation/image']),return_tensors='pt').to(backend.device)
            inputs['pixel_values']=inputs['pixel_values'].to(backend.dtype)
            with backend.torch.inference_mode():
                generated=backend._model.generate(**inputs,max_new_tokens=16,do_sample=False)
                logits=backend._model(**inputs,use_cache=False,logits_to_keep=1).logits[0,-1].float().cpu().numpy().astype(np.float64)
            tokens=generated[0,inputs['input_ids'].shape[1]:].tolist()
            generation={'text':backend._tokenizer.decode(tokens,skip_special_tokens=True),'token_ids':tokens,'eos':bool(tokens and tokens[-1]==backend._tokenizer.eos_token_id)}
            answer_ids={a:backend._tokenizer.encode(a,add_special_tokens=False) for a in row['labels']}
            if any(len(ids)!=1 for ids in answer_ids.values()):raise RuntimeError('Color references must have one-token labels')
            log_z=logits.max()+np.log(np.exp(logits-logits.max()).sum())
            scores={a:float(logits[ids[0]]-log_z) for a,ids in answer_ids.items()}
        else:
            input_obs=obs if condition=='direct' else control_observation(obs,{'kind':'constant','value':127 if condition=='gray' else 0})
            prefix=backend.prepare_target(input_obs,row['question'])
            generation=backend.generate(prefix,max_new_tokens=16)
            scores=backend.score_answers(prefix,row['labels'])
        record={k:row[k] for k in ['sample_id','episode_id','scene_id','counterfactual_group_id','split','label','labels']}
        record.update(condition=condition,raw_generation=generation['text'],token_ids=generation['token_ids'],eos=generation['eos'],
                      prediction=parse_answer(generation['text'],row['labels']),candidate_scores=scores,score_prediction=max(scores,key=scores.get))
        append_jsonl(out/'generations.jsonl',record);records.append(record)
    print(row['sample_id'],records[-4]['raw_generation'],records[-1]['raw_generation'],flush=True)
summary={c:{'n':len(rs),'generation_bacc':balanced_accuracy(rs),'candidate_bacc':balanced_accuracy(rs,'score_prediction')}
         for c in ['direct','gray','black','official_single_image'] for rs in [[r for r in records if r['condition']==c]]}
write_json(out/'readiness_summary.json',summary)
write_json(out/'complete.json',{'completed':True,'seconds':time.monotonic()-start,'scope':'calibration color pairs only; own patch integrity and official single-image task reference'})
print(summary,flush=True)
