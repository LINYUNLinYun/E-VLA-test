"""Audit collection identities, real request payloads and control-step alignment."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from preparations.idea1_v2_1.runtime_client import input_digest,value_digest
from .common import SUITES,PROTOCOL,atomic_json,episode_path,read_jsonl,request_seed


def collection(root):
    started=time.perf_counter();cases=[];sample_keys=set();phase_counts=Counter();total_requests=0
    for suite in SUITES:
        for task in range(10):
            for init in range(3):
                directory=episode_path(root,suite,task,init)
                identity=json.loads((directory/'identity.json').read_text())
                result=json.loads((directory/'result.json').read_text())
                completion=json.loads((directory/'complete.json').read_text())
                selection=json.loads((directory/'selection.json').read_text())
                infer=read_jsonl(directory/'infer.jsonl')
                trace=read_jsonl(directory/'trace.jsonl')
                phase_trace=read_jsonl(directory/'phase_trace.jsonl')
                assert identity['protocol']==PROTOCOL and completion['protocol']==PROTOCOL
                assert completion['valid'] and result['valid']
                assert (identity['suite'],identity['task'],identity['init'])==(suite,task,init)
                expected_gpu='1' if task%2==0 else '2'
                assert identity['service']['cuda_visible_devices']==expected_gpu
                assert len(infer)==result['requests']
                assert len(trace)==len(phase_trace)==result['steps']+1
                assert [r['step'] for r in trace]==list(range(result['steps']+1))
                assert len(list(directory.glob('request_*.npz')))==len(infer)
                executed=0
                for index,row in enumerate(infer):
                    assert row['request']==index and row['step']==executed
                    assert 1<=row['executed_length']<=5
                    assert Path(row['path'])==directory/f'request_{index:04d}.npz'
                    executed+=row['executed_length']
                assert executed==result['steps']
                samples=selection['samples']
                assert len(samples)==min(20,len(infer))
                assert len({s['request'] for s in samples})==len(samples)
                assert {samples[0]['request'],samples[-1]['request']}=={0,len(infer)-1}
                assert sum('uniform' in s['sources'] for s in samples)==min(12,len(infer))
                total_requests+=len(infer)
                for sample in samples:
                    key=(suite,task,init,sample['request']);assert key not in sample_keys;sample_keys.add(key)
                    row=infer[sample['request']];step=row['step']
                    assert sample['step']==step and sample['phase']==phase_trace[step]['phase']
                    assert sample['path']==row['path']
                    with np.load(row['path'],allow_pickle=False) as payload:
                        obs={name[5:]:payload[name] for name in payload.files if name.startswith('obs__')}
                        instruction=str(payload['instruction'].item())
                        noise=payload['noise'];actions=payload['actions'];raw=payload['raw_actions']
                        seed=int(payload['seed'])
                        assert seed==row['seed']==request_seed(suite,task,init,sample['request'])
                        assert noise.shape==(10,32) and noise.dtype==np.float32 and np.isfinite(noise).all()
                        assert actions.shape==(10,7) and raw.shape==(10,32)
                        assert np.isfinite(actions).all() and np.isfinite(raw).all()
                        assert input_digest(obs,instruction)==row['input_hash']
                        assert value_digest(noise)==row['noise_hash']
                        np.testing.assert_array_equal(obs['robot0_eef_pos'],trace[step]['eef_pos'])
                        for offset in range(row['executed_length']):
                            np.testing.assert_array_equal(actions[offset],
                                np.asarray(trace[step+offset+1]['executed_action'],np.float32))
                    phase_counts[sample['phase']]+=1
                cases.append(dict(suite=suite,task=task,init=init,success=result['success'],
                                  steps=result['steps'],requests=len(infer),selected=len(samples),
                                  service=identity['service']))
    receipt=dict(passed=True,scope='All120 completed rollouts; every recorded request path/control span; every selected payload actual RGB/state/instruction/noise hash and executed prefix',
        episodes=len(cases),recorded_requests=total_requests,selected_requests=len(sample_keys),
        successes=sum(r['success'] for r in cases),phase_counts=dict(phase_counts),
        checks='Frozen protocol, GPU service ownership, 12+8 request identity, explicit noise, source input hash, EEF timing and action-prefix alignment',
        excluded_scope='Full activation tensors and final analysis are audited separately; no claim that inferred phase labels are ground truth',
        cases=cases,elapsed_s=time.perf_counter()-started,
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_json(root/'collection_audit.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='cases'}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();collection(args.root.resolve())


if __name__=='__main__':main()
