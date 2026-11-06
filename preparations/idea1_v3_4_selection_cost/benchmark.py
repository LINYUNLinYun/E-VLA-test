"""Same-input, alternating eager VJP groups; preparation and peaks separate."""
import gc
import json
import time
from pathlib import Path
import numpy as np
import torch
from preparations.idea1_v3.io import atomic
from preparations.kv_reuse_probe.probe import load_observation
from preparations.idea1_v3_4_E2_cycle3.common import coefficients
from preparations.idea1_v3_4_E2_comp_error_multivjp.model import calculate as original_calculate
from .common import ROOT, OUT, write_csv
from .model import Selector
from .scoring import score_and_choose


def run(backend,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((OUT/'manifest.json').read_text())
    samples=json.loads((ROOT/'results/idea1_v3_4_E2_comp_error_cost/manifest.json').read_text())['samples']
    records=[];checks=[];preparations=[];peaks=[];scoring=[];arrays={}
    start=time.perf_counter()
    for index,sample in enumerate(samples):
        obs,prompt=load_observation(sample['current']);seed=sample['current']['seed']
        tick=time.perf_counter();context,_=backend.prepare(obs,prompt,())
        _,native_raw,_=backend.sample(context,seed,10)
        # Include the verifier's live weights/buffers in deployment memory.
        backend.verify(context,context,native_raw,seed,sample['suite'])
        native_preparation_s=time.perf_counter()-tick
        selector=Selector(backend);selector.batch_size=1
        skip,reference,reference_score,cost=selector.run(context,seed,coefficients(),6)
        # Independent old calculate, without capturing its former CUDA Graph.
        raw,gradient=original_calculate(selector.model,selector.item,selector.W,selector.g)
        compatibility=dict(sample=index,kind='original_serial_formula',
            raw_equal=bool(torch.equal(raw,selector.raw)),
            gradient_max_abs=float((gradient-selector.grad).abs().max()),
            passed=bool(torch.equal(raw,selector.raw) and torch.allclose(gradient,selector.grad,atol=1e-4,rtol=1e-3)))
        checks.append(compatibility)
        assert compatibility['passed'],compatibility
        arrays[f'serial_gradient_{index}']=reference.copy()
        reference_raw=selector.raw.float().cpu().numpy().copy()
        arrays[f'serial_raw_{index}']=reference_raw
        active=[]
        for batch_size in (1,2,4):
            selector.clear();selector.batch_size=batch_size
            gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            before=torch.cuda.memory_allocated()
            tick=time.perf_counter()
            try:
                mask,gradient,layer_score,cold=selector.run(context,seed,coefficients(),6)
                preparation_s=time.perf_counter()-tick
                raw=selector.raw.float().cpu().numpy().copy()
                numeric=bool(np.allclose(gradient,reference,atol=1e-4,rtol=1e-3))
                layer_ok=bool(np.allclose(layer_score,reference_score,atol=1e-4,rtol=1e-3))
                masks={str(k):score_and_choose(gradient,k)[0] for k in (4,6)}
                expected={str(k):score_and_choose(reference,k)[0] for k in (4,6)}
                check=dict(sample=index,kind='batch_vs_serial',batch_size=batch_size,
                    raw_equal=bool(np.array_equal(raw,reference_raw)),
                    gradient_max_abs=float(np.max(np.abs(gradient-reference))),
                    per_direction_max_abs=np.max(np.abs(gradient-reference),axis=1).tolist(),
                    gradient_max_scaled_error=float(np.max(np.abs(gradient-reference)/(1e-4+1e-3*np.abs(reference)))),
                    gradient_passed=numeric,layer_score_passed=layer_ok,
                    layer_score_max_abs=float(np.max(np.abs(layer_score-reference_score))),
                    masks=masks,reference_masks=expected,masks_equal=masks==expected)
                check['passed']=check['raw_equal'] and numeric and layer_ok and check['masks_equal']
                checks.append(check);arrays[f'batch{batch_size}_gradient_{index}']=gradient
                preparations.append(dict(sample=index,batch_size=batch_size,
                    native_preparation_s=native_preparation_s,first_selector_call_s=preparation_s,
                    selector_setup_ms=cold['selector_setup_ms'],graph_capture_ms=0.))
                for _ in range(manifest['warmups_per_method_per_input']):
                    selector.run(context,seed,coefficients(),6)
                peaks.append(dict(sample=index,batch_size=batch_size,oom=False,
                    before_allocated_mib=before/2**20,
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                    incremental_peak_mib=(torch.cuda.max_memory_allocated()-before)/2**20))
                active.append(batch_size)
            except torch.cuda.OutOfMemoryError:
                peaks.append(dict(sample=index,batch_size=batch_size,oom=True,
                    before_allocated_mib=before/2**20,
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
                checks.append(dict(sample=index,kind='batch_vs_serial',batch_size=batch_size,
                    passed=False,reason='OOM on the requested single card'))
            finally:
                selector.clear();gc.collect();torch.cuda.empty_cache()
        selectors={batch:Selector(backend) for batch in active}
        for batch,method in selectors.items():
            method.batch_size=batch;method.run(context,seed,coefficients(),6)
        for repeat in range(manifest['measurements_per_method_per_input']):
            order=active if (repeat+index)%2==0 else list(reversed(active))
            for batch_size in order:
                method=selectors[batch_size]
                tick=time.perf_counter();_,_,_,cost=method.run(context,seed,coefficients(),6)
                records.append(dict(sample=index,suite=sample['suite'],task=sample['task'],repeat=repeat,
                    order='-'.join(map(str,order)),batch_size=batch_size,
                    measured_wall_ms=(time.perf_counter()-tick)*1000,**cost))
        for k in (4,6):
            for repeat in range(100):
                order=('layer_rms_sum','signed_sum_rms') if repeat%2==0 else ('signed_sum_rms','layer_rms_sum')
                for aggregation in order:
                    tick=time.perf_counter();chosen,_,scores=score_and_choose(reference,k,aggregation)
                    elapsed=(time.perf_counter()-tick)*1000
                    scoring.append(dict(sample=index,k=k,repeat=repeat,aggregation=aggregation,
                        scoring_ms=elapsed,chosen=json.dumps(chosen),minimum_score=float(scores.min())))
        for method in selectors.values():method.clear()
        del selectors,selector,context
        gc.collect();torch.cuda.empty_cache()
        atomic(out/'checks.json',dict(checks=checks,finished_samples=index+1))
        write_csv(out/'timings.csv',records);write_csv(out/'preparation.csv',preparations)
        write_csv(out/'memory.csv',peaks);write_csv(out/'scoring.csv',scoring)
        np.savez(out/'gradients.npz',**arrays)
        print(json.dumps(dict(benchmark_samples=index+1,total=len(samples))),flush=True)
    equivalent={str(batch):all(row['passed'] for row in checks if row.get('batch_size')==batch)
        for batch in (1,2,4)}
    result=dict(complete=True,wall_s=time.perf_counter()-start,samples=len(samples),
        equivalence=equivalent,graph_capture_ms=0.,frozen_tolerance=[1e-4,1e-3],
        scope='Same real observations, Full-D10 replay; vision/native inference excluded from selector timing')
    atomic(out/'complete.json',result)
    return result
