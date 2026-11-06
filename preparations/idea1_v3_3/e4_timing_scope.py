"""Separate cross-device wall observations from measured verifier module time."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from preparations.idea1_v3.io import atomic
from .analyze import OUT,records,summary


def run():
    groups=records();result={}
    for cid in ['E4_tau0.8','E4_periodic','E4_tau1.6']:
        rows=[r for r in groups[cid] if r['case']['init'] in (0,1)];assert len(rows)==80
        bindings=Counter();modules=0.;requests=0;sources=[]
        for r in rows:
            d=Path(r['_directory']);side=d/'engineering_revision.json';b=json.loads(side.read_text());bindings[(b['model_gpu'],b['render_gpu'])]+=1
            infer=d/'infer.jsonl'
            for q in map(json.loads,infer.read_text().splitlines()):
                modules+=q['timing'].get('verify_ae_ms',0);requests+='verify_ae_ms' in q['timing']
            sources.append(dict(case=r['case'],binding_path=str(side),binding_sha256=hashlib.sha256(side.read_bytes()).hexdigest(),infer_path=str(infer),infer_sha256=hashlib.sha256(infer.read_bytes()).hexdigest()))
        s=summary(rows);result[cid]=dict(n=80,model_render_physical_counts={str(k):v for k,v in bindings.items()},observed_policy_ms_per_action=s['ms_per_action'],actual_actions=s['actions'],
            verify_module_CUDA_event_ms_sum=modules,verify_module_CUDA_event_ms_per_actual_action=modules/s['actions'],requests_with_verify_module=requests,sources=sources)
    left,right=result['E4_tau0.8'],result['E4_periodic'];legacy=OUT/'E4_tau08_periodic_exact_receipt.json'
    data=dict(time=time.time(),conditions=result,prior_action_equivalence_receipt_sha256=hashlib.sha256(legacy.read_bytes()).hexdigest(),
        observed_cross_GPU_wall_difference_ms_per_action=left['observed_policy_ms_per_action']-right['observed_policy_ms_per_action'],
        observed_cross_GPU_wall_relative_difference=left['observed_policy_ms_per_action']/right['observed_policy_ms_per_action']-1,
        interpretation='Tau.8 ran physicalGPU2; periodic and tau1.6 ran physicalGPU3. The2.796ms/action /18.48% aggregate difference is unmatched cross-device observation, NOT identified same-GPU verifier net overhead. Action/outcome equivalence remains valid. verify_ae_ms is the actual CUDA-event module timing inside tau.8 itself, already included in policy wall; it excludes CPU/synchronization overhead and is not total net policy overhead.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic(OUT/'E4_timing_device_scope_receipt.json',data);return data


if __name__=='__main__':run()
