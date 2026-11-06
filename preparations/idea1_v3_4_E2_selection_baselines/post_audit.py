"""Check matched random baselines share first proposals, then retain final evidence."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from .common import OUT,REFERENCE
from preparations.idea1_v3.io import atomic

def main():
    m=json.loads((OUT/'manifest.json').read_text());checks=[]
    ref=json.loads((REFERENCE/'manifest.json').read_text())
    for root,name,expected in [(OUT,c['id'],m['manifest_hash']) for c in m['conditions']]+[(REFERENCE,'exponential_fixed',ref['manifest_hash'])]:
        results=[p for p in (root/name/'episodes').glob('*/task*/init*/result.json') if '.invalid.' not in p.parent.name and (p.parent/'complete.json').exists()]
        assert len(results)==40
        assert all(json.loads(p.read_text())['manifest_hash']==expected for p in results)
    for case in m['cases']:
        suffix=Path('episodes')/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}"
        a,b=[OUT/name/suffix for name in ['random_online_s1','random_episode_s1']]
        ia,ib=[json.loads((p/'infer.jsonl').read_text().splitlines()[0]) for p in [a,b]]
        assert ia['input_hash']==ib['input_hash'] and ia['seed']==ib['seed'] and ia['route']['skip']==ib['route']['skip']
        with np.load(a/'arrays.npz') as x,np.load(b/'arrays.npz') as y:
            raw=float(abs(x['raw_0']-y['raw_0']).max());action=float(abs(x['actions_0']-y['actions_0']).max())
            passed=bool(np.allclose(x['raw_0'],y['raw_0'],atol=1e-5,rtol=1e-5) and np.allclose(x['actions_0'],y['actions_0'],atol=1e-5,rtol=1e-5))
        checks.append(dict(case=case,passed=passed,raw_max_abs=raw,action_max_abs=action));assert passed,checks[-1]
    atomic(OUT/'random_first_request_audit.json',dict(passed=True,checks=checks,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),time=time.time()))
    print(json.dumps(dict(passed=True,cases=len(checks),raw_max_abs=max(x['raw_max_abs'] for x in checks),action_max_abs=max(x['action_max_abs'] for x in checks))),flush=True)
if __name__=='__main__':main()
