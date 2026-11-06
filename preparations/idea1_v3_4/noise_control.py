"""Bounded E3 initialization/noise control, after the original GPU2 owner exits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import environment,lock
from preparations.idea1_v3.cli import verify_gpu
from preparations.kv_reuse_probe.probe import action_metrics
from .manifest import ROOT,OUT

DEST=OUT/'E3_noise_control'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze():
    if (DEST/'manifest.json').exists(): raise FileExistsError('Supplement already frozen')
    selected=[]
    for shadow in sorted((OUT/'E3_parent'/'episodes').glob('*/task*/init*/shadow_*.json')):
        d=shadow.parent
        if '.invalid.' in d.name or not (d/'complete.json').exists(): continue
        s=json.loads(shadow.read_text()); result=json.loads((d/'result.json').read_text())
        rows=[json.loads(line) for line in (d/'infer.jsonl').read_text().splitlines()]
        previous=None
        if s['round']>0:
            prev=rows[s['round']-1]
            previous=dict(index=prev['replan'],executed=prev['executed_length'],arrays=str(d/'arrays.npz'),arrays_sha256=sha(d/'arrays.npz'))
        selected.append(dict(id=len(selected),case=result['case'],group=s['group'],shadow=str(shadow),shadow_sha256=sha(shadow),
            shadow_arrays=str(shadow.with_suffix('.npz')),shadow_arrays_sha256=sha(shadow.with_suffix('.npz')),
            input=s['input_path'],input_sha256=s['input_file_sha256'],seed=s['seed'],refine_seed=s['refine_noise_seed'],previous=previous))
    if len(selected)!=207 or sum(s['group']=='zero' for s in selected)!=127: raise ValueError('Requires original complete fixed E3 sample')
    plan=ROOT/'docs/experiments/idea1/idea1_v3_4_E3_noise_control_plan.md'
    sources=[Path(__file__),Path(__file__).with_name('backend.py'),Path(__file__).with_name('server.py'),ROOT/'preparations/idea1_v3/policy.py',ROOT/'preparations/idea1_v3/safe_qk.py',ROOT/'preparations/idea1_v2_2/pi0_adapter.py']
    m=dict(selected=selected,q=.3,steps=2,atol=1e-5,rtol=1e-5,gpu=2,
        plan=str(plan),plan_sha256=sha(plan),sources={str(p):sha(p) for p in sources},original_manifest_sha256=sha(OUT/'manifest.json'))
    atomic(DEST/'manifest.json',m)


def owner_live(pid):
    p=subprocess.run(['ps','-p',str(pid),'-o','args='],text=True,capture_output=True)
    return p.returncode==0 and 'preparations.idea1_v3_4.dispatch' in p.stdout


def run():
    m=json.loads((DEST/'manifest.json').read_text())
    for path,expected in m['sources'].items():
        if sha(path)!=expected: raise ValueError('Frozen supplement source changed')
    for path,expected in ((m['plan'],m['plan_sha256']),(OUT/'manifest.json',m['original_manifest_sha256'])):
        if sha(path)!=expected: raise ValueError('Frozen protocol changed')
    identity=verify_gpu(2)
    os.environ['CUDA_VISIBLE_DEVICES']='2'
    tick=time.perf_counter()
    from preparations.idea1_v3.safe_qk import install
    install(str(DEST/'qk_receipt.json'))
    from .backend import Backend,shifted_tail
    from .server import gate
    b=Backend(); gate(b,DEST)
    atomic(DEST/'protocol.json',dict(identity,backend=b.protocol,pid=os.getpid(),load_gate_s=time.perf_counter()-tick))
    rows=[]
    for item in m['selected']:
        dest=DEST/'samples'/f"sample{item['id']:03d}"; dest.mkdir(parents=True,exist_ok=True)
        if (dest/'complete.json').exists(): continue
        for field,hashfield in (('input','input_sha256'),('shadow','shadow_sha256'),('shadow_arrays','shadow_arrays_sha256')):
            if sha(item[field])!=item[hashfield]: raise ValueError('Fixed E3 input changed')
        with np.load(item['input']) as saved:
            obs={k:saved[k] for k in saved.files if k not in ('instruction','seed')}; prompt=str(saved['instruction'])
            if int(saved['seed'])!=item['seed']: raise ValueError('Seed mismatch')
        with np.load(item['shadow_arrays']) as saved:
            actual_full=saved['full_d10_raw']; actual_decoded=saved['full_d10']; actual_draft=saved['v14_d2_raw']
        c,prep=b.prepare(obs,prompt,())
        outputs={}
        for which,seed in (('e0',item['seed']),('e1',item['refine_seed'])):
            outputs['full_d10_'+which]=b.sample(c,seed,10)
            outputs['full_d2_'+which]=b.sample(c,seed,2)
        a,r,_=outputs['full_d10_e0']; raw_np=r.float().cpu().numpy()
        check=dict(raw_max_abs=float(np.abs(raw_np-actual_full).max()),decoded_max_abs=float(np.abs(a-actual_decoded).max()),
            passed=bool(np.allclose(raw_np,actual_full,atol=m['atol'],rtol=m['rtol']) and np.allclose(a,actual_decoded,atol=m['atol'],rtol=m['rtol'])))
        atomic(dest/'reference_parity.json',check)
        if not check['passed']: raise RuntimeError('Same archived input Full reference differs; no pooling')
        previous=item['previous']; remaining=None
        if previous:
            if sha(previous['arrays'])!=previous['arrays_sha256']: raise ValueError('Historical execution arrays changed')
            with np.load(previous['arrays']) as saved: remaining=shifted_tail(saved[f"raw_{previous['index']}"],previous['executed'])
        for name,initial in (('draft',actual_draft),('remaining',remaining)):
            if initial is None: continue
            for which,seed in (('e0',item['seed']),('e1',item['refine_seed'])):
                outputs[name+'_q03_'+which]=b.warm_sample(c,initial,seed,.3,2)
        arrays={}; metrics=[]
        for name,(a,r,cost) in outputs.items():
            raw=r.float().cpu().numpy(); arrays[name]=a; arrays[name+'_raw']=raw
            for reference in ('e0','e1'):
                ref,ref_raw,_=outputs['full_d10_'+reference]
                metrics.append(dict(candidate=name,reference=reference,cost=cost,action_metrics=action_metrics(ref,a),
                    raw_first5_rmse=float(np.sqrt(np.mean((raw[:5,:7]-ref_raw.float().cpu().numpy()[:5,:7])**2)))))
        np.savez_compressed(dest/'actions.npz',**arrays)
        atomic(dest/'metrics.json',dict(selection=item,prepare=prep,metrics=metrics))
        atomic(dest/'complete.json',dict(id=item['id'],manifest_sha256=sha(DEST/'manifest.json')))
        print(json.dumps(dict(completed=item['id']+1,total=len(m['selected']))),flush=True)
    atomic(DEST/'finished.json',dict(time=time.time(),selected=len(m['selected'])))


def summarize():
    from .analyze import write_csv
    m=json.loads((DEST/'manifest.json').read_text()); rows=[]; errors=[]; maxdiff=0.
    for item in m['selected']:
        dest=DEST/'samples'/f"sample{item['id']:03d}"
        if not (dest/'complete.json').exists(): errors.append(dict(id=item['id'],error='missing')); continue
        data=json.loads((dest/'metrics.json').read_text())
        with np.load(dest/'actions.npz') as saved:
            arrays={k:saved[k] for k in saved.files}
        for name,value in arrays.items():
            if not np.isfinite(value).all() or value.shape!=((50,32) if name.endswith('_raw') else (50,7)):
                errors.append(dict(id=item['id'],error='invalid array '+name))
        for metric in data['metrics']:
            a=arrays[metric['candidate']]; reference=arrays['full_d10_'+metric['reference']]
            calculated=action_metrics(reference,a)
            for got,want in zip(metric['action_metrics'],calculated):
                for field,value in want.items():
                    if field in ('horizon','component') or value=='':
                        if got[field]!=value: errors.append(dict(id=item['id'],error='metric key mismatch'))
                    else:
                        difference=abs(float(got[field])-float(value)); maxdiff=max(maxdiff,difference)
                        if difference>1e-5: errors.append(dict(id=item['id'],error='action metric differs'))
            trans=next(r for r in calculated if r['horizon']==5 and r['component']=='translation')
            rot=next(r for r in calculated if r['horizon']==5 and r['component']=='rotation')
            grip=next(r for r in calculated if r['horizon']==5 and r['component']=='gripper')
            rows.append(dict(id=item['id'],**item['case'],group=item['group'],candidate=metric['candidate'],reference=metric['reference'],
                translation_mm=trans['clipped_controller_translation_mm_proxy'],rotation_l2=rot['mean_l2'],gripper_sign=grip['sign_mismatch_fraction'],ae_ms=metric['cost']['ae_ms']))
    summary=[]
    for group,name,ref in sorted({(r['group'],r['candidate'],r['reference']) for r in rows}):
        chosen=[r for r in rows if (r['group'],r['candidate'],r['reference'])==(group,name,ref)]
        summary.append(dict(group=group,candidate=name,reference=ref,n=len(chosen),**{k:float(np.mean([r[k] for r in chosen])) for k in ('translation_mm','rotation_l2','gripper_sign','ae_ms')}))
    write_csv(DEST/'requests.csv',rows); write_csv(DEST/'summary.csv',summary)
    atomic(DEST/'summary.json',summary)
    passed=not errors and len(list((DEST/'samples').glob('sample*/complete.json')))==len(m['selected'])
    atomic(DEST/'audit.json',dict(passed=passed,requests=len(m['selected']),errors=errors,max_abs_metric_difference=maxdiff,manifest_sha256=sha(DEST/'manifest.json'),source_sha256=sha(__file__)))
    print(json.dumps(dict(passed=passed,requests=len(m['selected']),summary_rows=len(summary))))
    if not passed: raise SystemExit(1)


def main():
    p=argparse.ArgumentParser(); p.add_argument('op',choices=['freeze','watch','run','summarize']); a=p.parse_args()
    DEST.mkdir(parents=True,exist_ok=True)
    if a.op=='freeze': freeze(); return
    if a.op=='summarize': summarize(); return
    with lock(DEST/('watch.lock' if a.op=='watch' else 'owner.lock'),blocking=False):
        if a.op=='watch':
            atomic(DEST/'watch_process.json',dict(pid=os.getpid(),owner_wait_pid=809593,started=time.time()))
            while owner_live(809593): time.sleep(30)
            env=environment(2); env['PYTHONPATH']=str(ROOT/'policies/realtime-vla-flash/src')+':'+env['PYTHONPATH']
            # A distinct process imports CUDA only after the original model exits.
            with (DEST/'run.log').open('a') as log:
                completed=subprocess.run([sys.executable,'-m','preparations.idea1_v3_4.noise_control','run'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            if completed.returncode: raise RuntimeError('Noise control failed; inspect run.log')
            summarize()
        else: run()

if __name__=='__main__': main()
