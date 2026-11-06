"""Compare a saved real first Full request across service revisions."""
from pathlib import Path
import hashlib
import numpy as np


def check(backend):
    root=Path(__file__).resolve().parents[2]
    directory=root/'results/idea1_v3_3/pi0_parent/episodes/libero_spatial/task00/init06'
    saved=np.load(directory/'input0.npz',allow_pickle=False)
    obs={k:saved[k] for k in saved.files if k not in ('instruction','seed')}
    instruction,seed=str(saved['instruction']),int(saved['seed'])
    arrays=np.load(directory/'arrays.npz',allow_pickle=False)
    c,t=backend.prepare(obs,instruction,()); actions,raw,cost=backend.sample(c,seed,10)
    raw=raw.float().cpu().numpy()
    same_raw=bool(np.array_equal(raw,arrays['raw_0'])); same_actions=bool(np.array_equal(actions,arrays['actions_0']))
    return dict(passed=same_raw and same_actions,raw_bitwise_equal=same_raw,decoded_bitwise_equal=same_actions,
        raw_max_abs=float(np.abs(raw-arrays['raw_0']).max()),decoded_max_abs=float(np.abs(actions-arrays['actions_0']).max()),
        reference=str(directory),input_sha256=hashlib.sha256((directory/'input0.npz').read_bytes()).hexdigest(),reference_arrays_sha256=hashlib.sha256((directory/'arrays.npz').read_bytes()).hexdigest(),
        scope='Same saved real initial obs/prompt/seed, Full-D10 across original/r2 services. V14 request1 RGB was not saved; no fabricated cross-restart V14 reference.')
