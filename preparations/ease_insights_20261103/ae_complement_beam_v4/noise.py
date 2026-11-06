"""Two prospectively fixed streams; stream0 preserves all known-control seeds."""
import hashlib
from preparations.idea1_v2_2.common import seed as original_seed


def model_seed(repeat,suite,task,initial,request):
    if type(repeat) is not int or repeat not in (0,1):raise ValueError('Exactly two registered policy-noise streams')
    base=original_seed(suite,task,initial,request)
    if repeat==0:return base
    return int.from_bytes(hashlib.sha256(f'ease-cross-module-v4/noise-repeat1/{base}'.encode()).digest()[:4],'little')
