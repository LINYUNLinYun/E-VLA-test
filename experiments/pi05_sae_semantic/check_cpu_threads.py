"""Compare a faster CPU threading setting against saved native tensors."""
import argparse
import json
from pathlib import Path
import time

import torch

from .native_probe import NativeProbe


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--preflight',required=True)
    p.add_argument('--threads',type=int,default=32)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    base=Path(args.preflight).resolve()
    original=json.loads((base/'native_preflight.json').read_text())
    reference=torch.load(base/'native_preflight_tensors.pt',map_location='cpu',weights_only=True)
    probe=NativeProbe(original['checkpoint'],device='cpu',threads=args.threads)
    start=time.perf_counter()
    embedded,attention,positions,layout=probe.prepare(original['input'])
    _,captured=probe.full(embedded,attention,positions,layout['selected_positions'])
    full_seconds=time.perf_counter()-start
    start=time.perf_counter()
    tail=probe.tail(reference['hidden5'],attention,positions,reference['selected'])
    tail_seconds=time.perf_counter()-start
    checks={}
    for name,a,b in [('L5_full_sequence',reference['hidden5'],captured['full_L5']),
                     ('L16_sample',reference['L16'],captured[16]),('L16_tail',reference['L16'],tail)]:
        checks[name]=dict(exact_equal=bool(torch.equal(a,b)),max_abs=float((a-b).abs().max()))
    result=dict(scope='One real observation, same checkpoint, unchanged BF16/FP32 policy; threading only',
                input=original['input'],input_sha256=original['input_sha256'],checkpoint=original['checkpoint'],
                device='cpu',threads=args.threads,checks=checks,passed=all(r['exact_equal'] for r in checks.values()),
                full_prefix_s=full_seconds,tail_s=tail_seconds,reference_8threads_s=original['timings'],load_s=probe.load_seconds)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
