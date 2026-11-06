"""Fixed-input batch order/noise diagnostics against the already loaded service."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from preparations.idea1_v3_1.server import rpc
from preparations.idea1_v3_1.benchmark import load_banks
from preparations.idea1_v3_1.common import atomic_json
from preparations.idea1_v2_1.runtime_client import input_digest, value_digest
from .batch_policy import noises, compare


def errors(a, b):
    delta = np.abs(a-b)
    return dict(compare(a,b), first5=compare(a[:5],b[:5]),
        per_channel_max=delta.max(axis=0).tolist(), max_index=list(map(int,np.unravel_index(delta.argmax(),delta.shape))))


def run(port, bank, output):
    records, identities = load_banks([bank])
    references = [rpc(port, dict(op='control', records=[r])) for r in records]
    rows, arrays = [], {}
    for offset in range(0,len(records)-3,4):
        group=records[offset:offset+4]
        batch=rpc(port,dict(op='control',records=group))
        repeated=rpc(port,dict(op='control',records=group))
        reverse=rpc(port,dict(op='control',records=group[::-1]))
        for j,record in enumerate(group):
            index=offset+j
            rows.append(dict(index=index,input_hash=input_digest(record['obs'],record['instruction']),
                noise_hash=value_digest(noises([record])[0]), noise_batch_invariant=bool(np.array_equal(noises(group)[j],noises([record])[0])),
                raw=errors(references[index]['raw'][0],batch['raw'][j]),
                decoded=errors(references[index]['actions'][0],batch['actions'][j]),
                repeat=compare(batch['raw'][j],repeated['raw'][j]),
                permutation_restored=compare(batch['raw'][j],reverse['raw'][3-j])))
        arrays[f'raw_b1_{offset}']=np.concatenate([references[offset+j]['raw'] for j in range(4)])
        arrays[f'raw_b4_{offset}']=batch['raw']
        arrays[f'decoded_b1_{offset}']=np.concatenate([references[offset+j]['actions'] for j in range(4)])
        arrays[f'decoded_b4_{offset}']=batch['actions']
    path=Path(output)
    np.savez_compressed(path/'batch_diagnostic_arrays.npz',**arrays)
    result=dict(service_pid=rpc(port,dict(op='health'))['protocol']['server_pid'],banks=identities,
        rows=rows, finite=all(np.isfinite(v).all() for v in arrays.values()),
        no_noise_mixup=all(r['noise_batch_invariant'] for r in rows),
        repeat_exact=all(r['repeat']['max_abs']==0 for r in rows),
        permutation_exact=all(r['permutation_restored']['max_abs']==0 for r in rows),
        scope='Original independently transformed inputs are stacked by tree.map; actual order invariance checked via reversed batch. No internal intermediate tensor capture claimed.')
    atomic_json(path/'batch_diagnostics.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=19332)
    parser.add_argument('--bank',default='results/idea1_v3_1/input_bank.pkl')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    run(args.port,args.bank,args.output)
