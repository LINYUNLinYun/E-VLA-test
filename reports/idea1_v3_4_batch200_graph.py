"""User-authorized graph-only Full-D10 / four-projection batched VJP probe."""
import gc,json,os,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import lock
from preparations.idea1_v3_4_selection_cost.common import ROOT,OUT,sha,write_csv
from preparations.idea1_v3_4_selection_cost.model import calculate
from preparations.idea1_v3_4_selection_cost.scoring import score_and_choose
from preparations.idea1_v3_4_E2_cycle3_full10_jacobian.model import borrow,current_item
from preparations.idea1_v3_4_E2_comp_error_multivjp.model import projection_weights
from preparations.kv_reuse_probe.probe import load_observation

DEST=OUT/'batch200_graph';DEST.mkdir(parents=True,exist_ok=True)
UUID='GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1'


def main():
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',SPEC_TRITON_DRAFT_GRAPH='0',
        SPEC_TRITON_VERIFY_GRAPH='0',SPEC_TRITON_VERIFY_SEMANTICS_GRAPH='0')
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    with lock(Path('/tmp')/f'embodied-benchmark-gpu-{UUID}.lock'):
        torch.cuda.init();alive=torch.zeros(1,device='cuda')
        atomic(DEST/'owner.json',dict(pid=os.getpid(),gpu=0,gpu_uuid=UUID,time=time.time()))
        from preparations.idea1_v3.safe_qk import install
        from preparations.idea1_v3.policy import TritonBackend
        install(str(DEST/'qk_receipt.json'));start=time.perf_counter();backend=TritonBackend()
        model_load_s=time.perf_counter()-start
        atomic(DEST/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),gpu_uuid=UUID,model_load_s=model_load_s,
            source_sha256=sha(__file__),frozen_model_source_sha256=sha(ROOT/'preparations/idea1_v3_4_selection_cost/model.py'),
            projections=projection_weights(4).tolist(),full_steps=10,batch_size=4,selector_graph=True,time=time.time()))
        samples=json.loads((ROOT/'results/idea1_v3_4_E2_comp_error_cost/manifest.json').read_text())['samples']
        rows=[];checks=[];preparation=[];saved={}
        with np.load(OUT/'benchmark/gradients.npz') as serial:
            for index,sample in enumerate(samples):
                atomic(DEST/'current.json',dict(sample=index,phase='preparing',time=time.time()))
                obs,prompt=load_observation(sample['current']);seed=sample['current']['seed']
                tick=time.perf_counter();context,_=backend.prepare(obs,prompt,())
                _,native_raw,_=backend.sample(context,seed,10)
                backend.verify(context,context,native_raw,seed,sample['suite'])
                native_prepare_s=time.perf_counter()-tick
                model=borrow(backend,context);item=current_item(context,seed)
                static={k:v.clone() for k,v in item.items()}
                weights=torch.as_tensor(projection_weights(4),device='cuda')
                gates=torch.ones(17,device='cuda',requires_grad=True)
                raw_ref,grad_ref,_=calculate(model,item,weights,gates,4)
                expected_raw=raw_ref.float().cpu().numpy().copy();expected_grad=grad_ref.float().cpu().numpy().copy()
                torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):calculate(model,static,weights,gates,4)
                torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):
                    raw,gradient,_=calculate(model,static,weights,gates,4)
                torch.cuda.synchronize();capture_s=time.perf_counter()-tick
                def call():
                    fresh=current_item(context,seed)
                    for key,value in fresh.items():static[key].copy_(value)
                    graph.replay();g=gradient.float().cpu().numpy().copy()
                    chosen,score,_=score_and_choose(g,6)
                    return g,chosen,score
                for _ in range(20):call()
                for repeat in range(10):
                    torch.cuda.synchronize();tick=time.perf_counter();g,chosen,score=call();torch.cuda.synchronize()
                    rows.append(dict(sample=index,suite=sample['suite'],task=sample['task'],repeat=repeat,
                        total_ms=(time.perf_counter()-tick)*1000))
                raw_cpu=raw.float().cpu().numpy().copy();reference=serial[f'serial_gradient_{index}']
                comparison=dict(sample=index,raw_vs_eager_batch=bool(np.array_equal(raw_cpu,expected_raw)),
                    gradient_vs_eager_batch=bool(np.allclose(g,expected_grad,atol=1e-4,rtol=1e-3)),
                    gradient_vs_eager_batch_max_abs=float(np.max(np.abs(g-expected_grad))),
                    raw_vs_serial=bool(np.allclose(raw_cpu,serial[f'serial_raw_{index}'],atol=1e-5,rtol=1e-5)),
                    gradient_vs_serial=bool(np.allclose(g,reference,atol=1e-4,rtol=1e-3)),
                    gradient_vs_serial_max_abs=float(np.max(np.abs(g-reference))),
                    per_direction_max_abs=np.max(np.abs(g-reference),axis=1).tolist(),
                    masks={str(k):score_and_choose(g,k)[0] for k in (4,6)},
                    serial_masks={str(k):score_and_choose(reference,k)[0] for k in (4,6)})
                comparison['masks_vs_serial']=comparison['masks']==comparison['serial_masks']
                comparison['graph_preserves_batched']=comparison['raw_vs_eager_batch'] and comparison['gradient_vs_eager_batch']
                comparison['equivalent_to_serial']=comparison['raw_vs_serial'] and comparison['gradient_vs_serial'] and comparison['masks_vs_serial']
                checks.append(comparison)
                preparation.append(dict(sample=index,native_prepare_s=native_prepare_s,capture_with_warmups_s=capture_s,
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
                saved[f'raw_{index}']=raw_cpu;saved[f'gradient_{index}']=g
                write_csv(DEST/'timings.csv',rows);write_csv(DEST/'preparation.csv',preparation)
                atomic(DEST/'checks.json',dict(checks=checks,finished_samples=index+1))
                np.savez(DEST/'gradients.npz',**saved)
                sample_times=[r['total_ms'] for r in rows if r['sample']==index]
                atomic(DEST/'progress.json',dict(finished_samples=index+1,total_samples=8,
                    mean_ms=float(np.mean([r['total_ms'] for r in rows])),p95_ms=float(np.percentile([r['total_ms'] for r in rows],95)),
                    last_mean_ms=float(np.mean(sample_times)),last_check=comparison,time=time.time()))
                print(json.dumps(dict(sample=index,mean_ms=float(np.mean(sample_times)),capture_s=capture_s,
                    graph_matches_batched=comparison['graph_preserves_batched'],equivalent_to_serial=comparison['equivalent_to_serial'])),flush=True)
                del graph,raw,gradient,static,model,item,raw_ref,grad_ref,gates,weights,context
                gc.collect();torch.cuda.empty_cache()
        values=[r['total_ms'] for r in rows]
        result=dict(samples=8,n=len(values),mean_ms=float(np.mean(values)),p95_ms=float(np.percentile(values,95)),
            max_ms=max(values),under_200_mean=float(np.mean(values))<200,under_200_p95=float(np.percentile(values,95))<200,
            all_samples_under_200=all(np.mean([r['total_ms'] for r in rows if r['sample']==i])<200 for i in range(8)),
            graph_preserves_batched=all(c['graph_preserves_batched'] for c in checks),
            equivalent_to_serial=all(c['equivalent_to_serial'] for c in checks),
            changed_mask_pairs=sum(c['masks'][k]!=c['serial_masks'][k] for c in checks for k in ('4','6')),
            model_load_s=model_load_s,total_probe_wall_s=time.perf_counter()-start,time=time.time())
        atomic(DEST/'complete.json',result);print(json.dumps(result),flush=True)
        # Avoid ineligible GPU0 service reloads while the other two GPUs finish.
        deadline=time.time()+1800
        while not (OUT/'complete.json').exists() and not (DEST/'release.flag').exists() and time.time()<deadline:time.sleep(5)
        atomic(DEST/'released.json',dict(time=time.time()))


if __name__=='__main__':
    try:main()
    except Exception:
        atomic(DEST/'failure.json',dict(error=traceback.format_exc(),time=time.time()));raise
