import csv,json,time,hashlib,subprocess,os
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from preparations.idea1_v3_4_E2_E4_single import torch_model as tm
from preparations.idea1_v3_4_E2_E4_single.common import ROOT,OUT as OLD,dump,sha
from preparations.idea1_v3_4_E2_E4_single.export import key
from preparations.idea1_v3_4_E2_E4_single.probe import setup,timing

OUT=ROOT/'results/idea1_v3_4_E2_200ms'
PLAN=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_200ms_plan.md'

def corr(a,b):return float(spearmanr(a,b).statistic)
def csvwrite(p,rs):
    if not rs:return
    keys=list(dict.fromkeys(k for r in rs for k in r))
    with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rs)

def subset_cpu(pairs):
    rs=[]
    for p in pairs:
        if not p['e2']:continue
        with np.load(OLD/'E2'/f"sample{p['id']:03d}"/'arrays.npz') as z:
            J=z['jacobian'];d=np.stack([z[f'L{l}_g0'][:5,:7]-z['full_raw'][:5,:7] for l in range(1,18)],-1)
            target=np.linalg.norm(d.reshape(35,17),axis=0)
            for name,sl,dim in [('all5',slice(5),slice(7)),('first1',slice(1),slice(7)),('first2',slice(2),slice(7)),('xyz5',slice(5),slice(3)),('xyz1',slice(1),slice(3)),('x1',slice(1),slice(1))]:
                score=np.linalg.norm(J[sl,dim,:].reshape(-1,17),axis=0)
                rs.append(dict(id=p['id'],**p['case'],method=name,rho_all7_R5=corr(score,target)))
    csvwrite(OUT/'subsets.csv',rs)

def projections():
    # Nested fixed projections, not selected against observed quality.
    return np.random.default_rng(340610).choice([-1.,1.],size=(4,35)).astype(np.float32)

def anchors(model,pairs):
    rows=[];R=projections()
    with torch.no_grad():pass
    for p in pairs:
        d=OUT/'anchors'/f"sample{p['id']:03d}";d.mkdir(parents=True,exist_ok=True)
        if (d/'complete.json').exists():
            rows+=json.loads((d/'metrics.json').read_text());continue
        item=model.item(OLD/'inputs'/f"{key(p['current'])}.pt")
        with torch.no_grad():full=model.encode(item,trace=True);ref=model.sample(item,full)
        with np.load(OLD/'E2'/f"sample{p['id']:03d}"/'arrays.npz') as z:
            J=z['jacobian'].reshape(35,17);actual=np.stack([z[f'L{l}_g0'][:5,:7].reshape(-1)-z['full_raw'][:5,:7].reshape(-1) for l in range(1,18)],1)
            raw0=[z[f'L{l}_g0'].copy() for l in range(1,18)]
            assert np.allclose(ref.cpu().numpy(),z['full_raw'],atol=1e-5,rtol=1e-5)
        derivatives={1.:R@J};stored={};start=time.perf_counter()
        for point in (0.,.5):
            values=[]
            for l in range(1,18):
                g=torch.tensor(point,device='cuda',requires_grad=True)
                prefix=dict(full);prefix['inputs']=list(full['inputs'])
                prefix['inputs'][l]=g*full['outputs'][l-1]+(1-g)*full['inputs'][l-1]
                raw=model.forward(item,prefix=prefix,start=l)
                if point==0:assert np.allclose(raw.detach().cpu().numpy(),raw0[l-1],atol=1e-5,rtol=1e-5)
                projected=torch.as_tensor(R,device='cuda')@raw[:5,:7].reshape(-1)
                values.append([float(torch.autograd.grad(projected[j],g,retain_graph=j<3)[0]) for j in range(4)])
                stored[f'L{l}_g{point}_raw']=raw.detach().cpu().numpy()
                del raw,projected,g,prefix
            derivatives[point]=np.asarray(values).T
        true=R@actual;preds={'at1':-derivatives[1.],'at0':-derivatives[0.],'mid':-derivatives[.5],
            'trapezoid':-(derivatives[0.]+derivatives[1.])/2,
            'simpson':-(derivatives[0.]+4*derivatives[.5]+derivatives[1.])/6}
        rr=[]
        for name,pred in preds.items():
            rr.append(dict(id=p['id'],**p['case'],method=name,rho_full_actual=corr(np.linalg.norm(pred,axis=0),np.linalg.norm(actual,axis=0)),
                projected_relative_error=float(np.linalg.norm(pred-true)/np.linalg.norm(true)),
                projected_direction_cosine=float(np.mean(np.sum(pred*true,axis=0)/(np.linalg.norm(pred,axis=0)*np.linalg.norm(true,axis=0)+1e-12))),
                anchor_work_s=time.perf_counter()-start))
        np.savez_compressed(d/'arrays.npz',R=R,J1=J,actual=actual,derivative0=derivatives[0.],derivative_mid=derivatives[.5],**stored)
        dump(d/'metrics.json',rr);dump(d/'complete.json',dict(id=p['id'],time=time.time()));rows+=rr
        print(json.dumps(dict(anchor_complete=p['id'],wall_s=time.perf_counter()-start)),flush=True)
    csvwrite(OUT/'anchors.csv',rows)

def bench(model,pairs,dtype):
    R=projections();rows=[];compat=[]
    for p in pairs:
        item=model.item(OLD/'inputs'/f"{key(p['current'])}.pt")
        if dtype=='bf16':
            for k in ('x','state','encoder_rope','decoder_rope','noise'):item[k]=item[k].to(torch.bfloat16)
        r=torch.as_tensor(R,device='cuda')
        for method,n in [('forward',0),('first1_exact',7),('xyz1_exact',3),('x1',1),('project1',1),('project2',2),('project4',4)]:
            def call():
                gates=torch.ones(17,device='cuda',requires_grad=n>0)
                t=time.perf_counter()
                with torch.set_grad_enabled(n>0):raw=model.forward(item,gates=gates if n else None)
                torch.cuda.synchronize();fwd=(time.perf_counter()-t)*1000
                if n:
                    target=raw[:5,:7].float().reshape(-1)
                    if method.startswith('project'):values=r[:n]@target
                    elif method=='first1_exact':values=target[:7]
                    elif method=='xyz1_exact':values=target[:3]
                    else:values=target[:1]
                    grads=[]
                    for j in range(n):grads.append(torch.autograd.grad(values[j],gates,retain_graph=j<n-1)[0].detach())
                    score=torch.stack(grads).float().square().sum(0).sqrt();order=torch.argsort(score)
                    # Include transfer of the small selected mask in the wall.
                    chosen=order[:4].cpu().tolist()
                else:chosen=[]
                torch.cuda.synchronize()
                return raw.detach(),dict(forward_ms=fwd,mask=chosen)
            for _ in range(2):call()
            for rep in range(7):
                (raw,details),total=timing(call)
                rows.append(dict(id=p['id'],**p['case'],dtype=dtype,method=method,vjp=n,rep=rep,total_ms=total,
                    forward_ms=details['forward_ms'],added_ms=total-details['forward_ms'],mask=str(details['mask'])))
            if method=='forward':
                with np.load(OLD/'E2'/f"sample{p['id']:03d}"/'arrays.npz') as z:
                    compat.append(dict(id=p['id'],dtype=dtype,raw_prefix7_rms=float(np.sqrt(np.mean((raw.float().cpu().numpy()[:5,:7]-z['full_raw'][:5,:7])**2)))))
        csvwrite(OUT/f'latency_{dtype}.csv',rows);dump(OUT/f'compatibility_{dtype}.json',compat)
        print(json.dumps(dict(bench_complete=p['id'],dtype=dtype)),flush=True)
    return rows

def main():
    setup();OUT.mkdir(parents=True,exist_ok=True)
    pairs=json.loads((OLD/'selection.json').read_text());selected=[p for p in pairs if p['e2'] and p['case']['task'] in (0,2)]
    assert len(selected)==8
    dump(OUT/'manifest.json',dict(plan_sha256=sha(PLAN),source_sha256=sha(__file__),reference_selection_sha256=sha(OLD/'selection.json'),
        selected_ids=[p['id'] for p in selected],projection_seed=340610,time=time.time()))
    subset_cpu(pairs)
    model=tm.Model(OLD/'model.pt');anchors(model,selected);rows=bench(model,selected,'fp32')
    for k,v in model.w.items():model.w[k]=v.to(torch.bfloat16)
    for name in ('biases','outw','outb'):setattr(model,name,[v.to(torch.bfloat16) for v in getattr(model,name)])
    torch.cuda.empty_cache()
    tm.rms=lambda x:(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)).to(x.dtype)
    rows+=bench(model,selected,'bf16')
    summary=[]
    for dtype in ('fp32','bf16'):
        for method in sorted({r['method'] for r in rows}):
            a=[r for r in rows if r['dtype']==dtype and r['method']==method];v=np.array([r['total_ms'] for r in a])
            summary.append(dict(dtype=dtype,method=method,n=len(a),mean_ms=float(v.mean()),p50_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95)),max_ms=float(v.max()),under200=float(np.mean(v<200)),added_mean_ms=float(np.mean([r['added_ms'] for r in a]))))
    csvwrite(OUT/'latency_summary.csv',summary)
    report(summary)
    dump(OUT/'complete.json',dict(passed=True,requests=8,time=time.time()))

def report(summary):
    with (OUT/'subsets.csv').open() as f:sub=list(csv.DictReader(f))
    with (OUT/'anchors.csv').open() as f:anc=list(csv.DictReader(f))
    lines=['# E2：200ms成本与展开点探测','',
        '8个固定真实输入，D2/H50/R5。计时含可微VLM、完整两步AE/FM、所列反传、17层排序及同步；不含固定视觉嵌入提取/加载/RPC。BF16为独立近似后端，FP32 RMS统计。以下为热态小批实测，非部署全链路P95保证。',
        '', '|精度|方法|均值ms|P95 ms|最大ms|<200ms比例|新增反传等ms|','|---|---|---|---|---|---|---|']
    for r in summary:lines.append(f"|{r['dtype']}|{r['method']}|{r['mean_ms']:.2f}|{r['p95_ms']:.2f}|{r['max_ms']:.2f}|{r['under200']:.1%}|{r['added_mean_ms']:.2f}|")
    lines+=['','## 原40请求的输出子集','', '|子集|对实际R5全7维动作扰动的排序ρ|','|---|---|']
    for method in sorted({r['method'] for r in sub}):lines.append(f"|{method}|{np.mean([float(r['rho_all7_R5']) for r in sub if r['method']==method]):.3f}|")
    lines+=['','## 8请求的单层展开点','', '|方法|排序ρ|投影向量相对误差|投影方向余弦|','|---|---|---|---|']
    for method in ('at1','at0','mid','trapezoid','simpson'):
        a=[r for r in anc if r['method']==method]
        lines.append('|'+ '|'.join([method,*[f"{np.mean([float(r[k]) for r in a]):.3f}" for k in ('rho_full_actual','projected_relative_error','projected_direction_cosine')]])+'|')
    lines+=['', '展开点比较使用4个固定相同投影方向和逐层独立基点，不是同时关闭所有层。g=0/1端点对照原始动作通过。多点积分比较已计入每个请求的anchor_work_s，不能将其离线效果当作200ms在线成本。完整first1_exact需要7次VJP；project1只用1次VJP但仍覆盖35个输出，随机投影排序质量与数值精度需分别解释。',
        '', '[方案](idea1_v3_4_E2_200ms_plan.md) · [计时CSV](../../../results/idea1_v3_4_E2_200ms/latency_summary.csv) · [展开点CSV](../../../results/idea1_v3_4_E2_200ms/anchors.csv)']
    (ROOT/'docs/experiments/idea1/idea1_v3_4_E2_200ms_report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
