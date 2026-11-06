"""Uniform 20Hz free closed loops for baseline and selected finite candidates."""
import argparse,json,time
from pathlib import Path
import numpy as np
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import lock
from preparations.idea1_v3_4_E2_max2_repair.evaluate import Fresh
from preparations.idea1_v3_4_selection_async.rollout import episode
from preparations.idea1_v3_4_selection_async.audit import episode as audit
from preparations.idea1_v3_4_selection_async.common import OUT,ck
from reports.idea1_v3_4_selection_async_analysis import directory
from preparations.idea1_v2_1.runtime_client import rpc


def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True)
    p.add_argument('--worker',required=True);a=p.parse_args()
    m=json.loads((OUT/'followup_manifest.json').read_text());pool=Fresh()
    recovery_path=OUT/'paced_recovery_manifest.json'
    recovery=json.loads(recovery_path.read_text()) if recovery_path.exists() else None
    cases=m['cases']+([] if recovery is None else recovery['cases'])
    try:
        for ci,case in enumerate(cases):
            claim=lock(OUT/'paced_claims'/f'{ck(case)}.lock',blocking=False)
            try:claim.__enter__()
            except BlockingIOError:continue
            try:
                cfgs=m['conditions'];offset=ci%len(cfgs);cfgs=cfgs[offset:]+cfgs[:offset]
                for cfg in cfgs:
                    original=directory(cfg['id'],case)
                    d=OUT/'paced'/cfg['id']/'episodes'/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}"
                    if (d/'complete.json').exists():continue
                    if d.exists():raise RuntimeError(f'Interrupted paced episode: {d}')
                    atomic(OUT/'paced_workers'/a.worker/'current.json',dict(condition=cfg['id'],case=case,time=time.time()))
                    owner=recovery if recovery is not None and case in recovery['cases'] else m
                    result=episode(d,a.port,case,cfg,pool,owner['manifest_hash'],pace_hz=20)
                    if recovery is not None:
                        atomic(d/'execution_amendment.json',dict(amendment_hash=recovery['manifest_hash'],
                            original_manifest_hash=m['manifest_hash']))
                    checked=audit(cfg,d)
                    old=[json.loads(line) for line in (original/'infer.jsonl').read_text().splitlines()]
                    new=[json.loads(line) for line in (d/'infer.jsonl').read_text().splitlines()]
                    checks=[]
                    with np.load(original/'arrays.npz') as aa,np.load(d/'arrays.npz') as bb:
                        for i,(x,y) in enumerate(zip(old,new)):
                            checks.append(dict(request=i,input=x['input_hash']==y['input_hash'],
                                route=x['route']['path']==y['route']['path'],
                                mask=x['route']['draft_skip']==y['route']['draft_skip'],
                                raw=bool(np.allclose(aa[f'raw_{i}'],bb[f'raw_{i}'],atol=1e-5,rtol=1e-5)),
                                actions=bool(np.allclose(aa[f'actions_{i}'],bb[f'actions_{i}'],atol=1e-5,rtol=1e-5)),
                                executed=x['executed_length']==y['executed_length']))
                    passed=len(old)==len(new) and all(all(c[k] for k in ('input','route','mask','raw','actions','executed')) for c in checks)
                    atomic(d/'pacing_equivalence.json',dict(passed=passed,checks=checks,mechanism_audit=checked))
                    assert passed,'Pacing changed the policy trajectory'
                    print(json.dumps(dict(condition=cfg['id'],case=case,success=result['success'],
                        actions=result['actions'],wall_s=result['wall_s'],equivalence=passed)),flush=True)
                rpc(a.port,dict(op='reset_all'))
            finally:claim.__exit__(None,None,None)
    finally:pool.close()


if __name__=='__main__':main()
