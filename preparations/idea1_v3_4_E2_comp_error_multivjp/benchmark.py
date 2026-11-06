"""Eight same-input nonzero-error cases, q1..4; balanced blocked timing."""
import argparse,json,os,time
import numpy as np
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4.analyze import write_csv
from .common import ROOT, OUT, COST, sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);args=p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4')
    import torch
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    from preparations.idea1_v3.safe_qk import install
    from preparations.idea1_v3.policy import TritonBackend
    from preparations.idea1_v3_4_E2_cycle3.selection import prepare_full
    from preparations.idea1_v3_4_E2_cycle3_extrapolate.backend import capture
    from preparations.idea1_v3_4_E2_comp_error_cost.model import Selector as Old,calculate as old_calculate
    from preparations.kv_reuse_probe.probe import load_observation
    from .model import Selector,calculate,forecast_error,projection_weights
    out=OUT/'cost';out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise FileExistsError('Cost probe complete')
    frozen=json.loads((OUT/'manifest.json').read_text())
    assert all(sha(ROOT/n)==h for n,h in frozen['source_files'].items())
    samples=json.loads((COST/'manifest.json').read_text())['samples']
    atomic(out/'manifest.json',dict(parent_manifest_sha256=sha(OUT/'manifest.json'),samples=samples,projections=[1,2,3,4],warmups=20,repeats=10,gpu=args.gpu,scope='balanced blocked same-input timing; not per-repeat AB/BA'))
    install(str(out/'qk_receipt.json'));tick=time.perf_counter();b=TritonBackend();setup_s=time.perf_counter()-tick
    rows=[];checks=[];arrays={};captures=[]
    for si,sample in enumerate(samples):
        banks=[]
        for side in ('older','last','current'):
            assert sha(sample[side]['path'])==sample[side]['sha256']
            obs,prompt=load_observation(sample[side]);c,_=prepare_full(b,obs,prompt,trace=True)
            banks.append(capture(c.pop('hidden_trace')))
        error=forecast_error(banks[2],banks[1],banks[0]);seed=sample['current']['seed']
        old=Old(b);_,old_grad,old_score,old_cost=old.run(c,seed,projection_weights(1)[0],error,6)
        old_raw=old.raw.clone();old.clear();del old
        order=[1,2,3,4];offset=si%4;order=order[offset:]+order[:offset]
        if si%2:order=order[::-1]
        for q in order:
            selector=Selector(b);W=projection_weights(q)
            skip,grad,score,cold=selector.run(c,seed,W,error,6)
            eager_raw,eager_grad=calculate(selector.model,selector.item,selector.W,selector.g)
            assert torch.equal(selector.raw,old_raw) and torch.equal(selector.raw,eager_raw)
            assert torch.allclose(selector.grad,eager_grad,atol=1e-4,rtol=1e-3)
            assert np.allclose(grad[0],old_grad,atol=1e-4,rtol=1e-3),(si,q,'first projection')
            if q==1:
                assert np.array_equal(score,old_score) and skip==__import__('preparations.idea1_v3_4_E2_cycle3.common',fromlist=['choose']).choose(old_score,6)
            checks.append(dict(sample=si,q=q,passed=True,neutral_full_bitwise=True,eager_raw_bitwise=True,graph_eager_grad_max=float((selector.grad-eager_grad).abs().max()),prefix_grad_max=float(np.abs(grad[0]-old_grad).max()),skip=skip))
            arrays[f'gradient_{si}_q{q}']=grad;arrays[f'score_{si}_q{q}']=score
            def call():
                forecast_error(banks[2],banks[1],banks[0],out=error)
                return selector.run(c,seed,W,error,6)
            for _ in range(20):call()
            for rep in range(10):
                torch.cuda.synchronize();start=time.perf_counter();call();torch.cuda.synchronize()
                rows.append(dict(sample=si,suite=sample['suite'],task=sample['task'],q=q,rep=rep,order=str(order),selector_with_error_ms=(time.perf_counter()-start)*1000))
            captures.append(dict(sample=si,q=q,capture_ms=cold['selector_capture_ms'],allocated_mib=torch.cuda.memory_allocated()/2**20,reserved_mib=torch.cuda.memory_reserved()/2**20))
            selector.clear();del selector,eager_raw,eager_grad
            print(json.dumps(dict(sample=si,q=q,mean_ms=float(np.mean([r['selector_with_error_ms'] for r in rows if r['sample']==si and r['q']==q])),passed=True)),flush=True)
        del banks,error,old_raw
        torch.cuda.empty_cache()
        write_csv(out/'timings.csv',rows);atomic(out/'checks.json',checks);write_csv(out/'capture.csv',captures)
    np.savez_compressed(out/'arrays.npz',**arrays)
    summary=dict(passed=True,samples=8,measurements=len(rows),model_setup_s=setup_s,gpu=args.gpu,checks=checks,
        timing={str(q):dict(mean_ms=float(np.mean([r['selector_with_error_ms'] for r in rows if r['q']==q])),median_ms=float(np.median([r['selector_with_error_ms'] for r in rows if r['q']==q])),p95_ms=float(np.percentile([r['selector_with_error_ms'] for r in rows if r['q']==q],95))) for q in (1,2,3,4)})
    atomic(out/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='checks'}),flush=True)

if __name__=='__main__':main()
