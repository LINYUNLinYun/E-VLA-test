"""One real saved Xiaomi observation: native Full-D2 equivalence receipt."""
import argparse
from pathlib import Path
import time
import hashlib
import numpy as np
from preparations.idea1_v3.io import atomic
from .backends import XiaomiBackend


def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--output',required=True); a=p.parse_args()
    saved=np.load(a.input,allow_pickle=False); obs={k:saved[k] for k in saved.files if k not in ('instruction','seed')}; instruction=str(saved['instruction']); seed=int(saved['seed'])
    tick=time.perf_counter(); backend=XiaomiBackend(); load=time.perf_counter()-tick
    c,t=backend.prepare(obs,instruction,()); actions,raw,cost=backend.sample(c,seed,2)
    native=backend.native.native_reference(obs,instruction,seed=seed,skip=(),num_steps=2,return_details=True)
    diff=float(np.abs(actions-native['actions']).max()); passed=bool(np.allclose(actions,native['actions'],atol=1e-5,rtol=1e-5) and native['audit']['actual_NFE']==2)
    atomic(a.output,dict(passed=passed,max_abs=diff,atol=1e-5,rtol=1e-5,native_audit=native['audit'],input=str(Path(a.input).resolve()),input_sha256=hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),load_s=load,total_s=time.perf_counter()-tick,
        scope='Saved first official closed-loop observation, original native Full-D2 vs persistent adapter, same seed and mask. Not another scientific episode.'))
    if not passed: raise ValueError('Xiaomi Full native gate failed')


if __name__=='__main__': main()
