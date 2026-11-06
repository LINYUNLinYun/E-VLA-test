"""Four bounded same-device preflight calls; no model construction or quality."""
import argparse
import json
from pathlib import Path
import pickle
import time
import numpy as np
from .policy_native_watch import rpc
from preparations.idea1_v3_2.transfer_protocol import sha
from preparations.idea1_v3_1.common import atomic_json


def run(port,bank,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    record=pickle.loads(Path(bank).read_bytes())[0]
    health=rpc(port,dict(op='health'),timeout=10)['protocol']
    receipt=dict(status='running',port=port,source_sha256=sha(__file__),bank_path=str(bank),bank_sha256=sha(bank),
        server_pid=health['server_pid'],gpu_uuid=health['gpu_uuid'],manifest_hash=health['manifest_hash'],
        category='preflight',model_constructed=False,quality_requests=0,max_requests=4,calls=[],started_unix=time.time())
    atomic_json(output/'receipt.json',receipt);responses=[]
    for i,mode in enumerate(('full_d10','v14_d10','full_d10','v14_d10')):
        request=dict(op='predict',family='libero_transfer',environment='libero',mother_suite='libero_goal',mode=mode,
            observation=record['obs'],instruction=record['instruction'],seed=int(record['seed']),category='preflight')
        tick=time.perf_counter();response=rpc(port,request,timeout=120)
        if response['manifest_hash']!=health['manifest_hash']:raise RuntimeError('Source identity changed')
        raw=np.asarray(response['raw_actions'],np.float32);decoded=np.asarray(response['actions'],np.float32)
        if raw.shape!=(50,32) or decoded.shape!=(50,7) or not np.isfinite(raw).all() or not np.isfinite(decoded).all():
            raise RuntimeError('Invalid actual source outputs')
        prefix=output/('%02d_%s'%(i,mode));np.savez_compressed(str(prefix)+'.npz',raw_actions=raw,actions=decoded)
        atomic_json(str(prefix)+'.json',response)
        receipt['calls'].append(dict(index=i,mode=mode,input_hash=response['input_hash'],noise_hash=response['noise_hash'],
            observed_calls=response['observed_calls'],startup_events=response.get('startup_events'),
            response_path=str(prefix)+'.json',response_sha256=sha(str(prefix)+'.json'),array_path=str(prefix)+'.npz',
            array_sha256=sha(str(prefix)+'.npz'),wall_ms=(time.perf_counter()-tick)*1000))
        responses.append(response);atomic_json(output/'receipt.json',receipt)
    comparisons=[]
    for first,second in ((0,2),(1,3)):
        row=dict(mode=responses[first]['mode'],first=first,second=second,
            input_hash_equal=responses[first]['input_hash']==responses[second]['input_hash'],
            noise_hash_equal=responses[first]['noise_hash']==responses[second]['noise_hash'],outputs={})
        for key in ('raw_actions','actions'):
            a,b=np.asarray(responses[first][key],np.float32),np.asarray(responses[second][key],np.float32)
            row['outputs'][key]=dict(bitwise_equal=bool(np.array_equal(a,b)),max_abs=float(np.max(np.abs(a-b))))
        comparisons.append(row)
    receipt.update(status='complete',requests=4,comparisons=comparisons,completed_unix=time.time(),
        interpretation='Same-device repeated output test only; startup constructor suppression count1 means a new prompt runtime was constructed while suppressing monolithic graph recording, not a Full inference graph replay. Count0 means runtime already present. This difference is a runtime-history confound; neither proves cause nor cross-GPU-only arithmetic difference.')
    atomic_json(output/'receipt.json',receipt)
    print(json.dumps(dict(status='complete',comparisons=comparisons)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=19143)
    p.add_argument('--bank',default='results/idea1_v3_1/input_bank.pkl');p.add_argument('--output',required=True)
    a=p.parse_args();run(a.port,a.bank,a.output)


if __name__=='__main__':main()
