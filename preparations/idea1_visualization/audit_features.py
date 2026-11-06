"""Incremental full-archive schema and small-payload lineage verification.

Every NPY header is checked. Large hidden arrays are not re-read or rehashed;
the noise origin and raw/decoded final actions are read and checked explicitly.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np

from .common import SUITES,atomic_json,episode_path

VERSION='all-headers-lineage-v1'


def signature(directory):
    names=('activations.npz','pooled.npz','residual_stats.npz','metadata.json','sample.json','complete.json','parity.json')
    return {name:[(st:= (directory/name).stat()).st_size,st.st_mtime_ns] for name in names}


def header(archive,key):
    with archive.open(key+'.npy') as stream:
        version=np.lib.format.read_magic(stream)
        if version==(1,0):return np.lib.format.read_array_header_1_0(stream)
        if version==(2,0):return np.lib.format.read_array_header_2_0(stream)
        raise ValueError(f'Unexpected NPY version {version}: {key}')


def verify_context(directory,sample,identity):
    metadata=json.loads((directory/'metadata.json').read_text())
    completion=json.loads((directory/'complete.json').read_text())
    indexed=json.loads((directory/'sample.json').read_text())
    parity=json.loads((directory/'parity.json').read_text())
    assert completion['passed'] and parity['passed']
    assert parity['atol']==parity['rtol']==1e-5
    assert metadata['source']==completion['source']==sample['path']
    assert metadata['service']==identity['service']
    assert metadata['discrete_state_input'] is False
    assert indexed['request']==sample['request'] and indexed['step']==sample['step']
    assert indexed['phase']==sample['phase'] and indexed['sources']==sample['sources']
    assert metadata['parity']==completion['parity']
    for kind in ('raw','decoded'):assert metadata['parity'][kind]==parity[kind]
    return metadata,completion


def verify_one(directory,sample,identity,context=None):
    metadata,completion=context or verify_context(directory,sample,identity)
    full=directory/'activations.npz'
    with np.load(full,allow_pickle=False) as archive:
        pm=archive['prefix_mask'];tm=archive['prompt_mask'];tokens=archive['prompt_tokens']
        assert pm.ndim==2 and pm.shape[0]==1
        assert tm.ndim==2 and tm.shape[0]==1 and tm.shape==tokens.shape
        cameras=('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')
        flags=[archive[f'image_masks/{name}'] for name in cameras]
        assert all(f.shape==(1,) and f.dtype==bool for f in flags)
        assert [bool(f.item()) for f in flags]==[True,True,False]
        np.testing.assert_array_equal(pm,np.concatenate([*[np.repeat(f[:,None],256,axis=1) for f in flags],tm],axis=1))
        n=int(pm.sum());assert n==metadata['prefix_valid_tokens']
        times=archive['flow/time'];x=archive['flow/x_t']
        np.testing.assert_allclose(times,np.arange(10,dtype=np.float32)*np.float32(-.1)+1,atol=1e-6,rtol=0)
        np.testing.assert_array_equal(times,np.asarray(metadata['flow_times'],np.float32))
        with np.load(sample['path'],allow_pickle=False) as source:
            np.testing.assert_array_equal(x[0,0],source['noise'])
            actual_raw=archive['action/final_raw'];actual_decoded=archive['action/final_decoded']
            for name,a,b in (('raw',actual_raw,source['raw_actions']),('decoded',actual_decoded,source['actions'])):
                np.testing.assert_allclose(a,b,atol=1e-5,rtol=1e-5)
                measured=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))
                assert measured==metadata['parity'][name]['max_abs']
                assert bool(np.array_equal(a,b))==metadata['parity'][name]['equal']
    shapes={
        'vlm/hidden':(19,1,n,2048),
        'action_expert/hidden':(10,19,1,10,1024),
        'flow/x_t':(10,1,10,32),'flow/velocity':(10,1,10,32),'flow/time':(10,),
        'prefix_mask':tuple(pm.shape),'prompt_mask':tuple(tm.shape),'prompt_tokens':tuple(tokens.shape),
        'action/final_raw':(10,32),'action/final_decoded':(10,7),
    }
    for name in cameras:
        shapes[f'image_masks/{name}']=(1,)
    for kind in ('post_attention','attn_update','ffn_update'):
        shapes[f'vlm/{kind}']=(18,1,n,2048)
        shapes[f'action_expert/{kind}']=(10,18,1,10,1024)
    for name in cameras[:2]:
        shapes[f'vision/{name}/hidden']=(28,1,256,1152)
        shapes[f'vision/{name}/projected']=(1,256,2048)
        for kind in ('post_attention','attn_update','ffn_update'):
            shapes[f'vision/{name}/{kind}']=(27,1,256,1152)
    assert set(shapes)==set(metadata['dtypes'])|{'action/final_raw','action/final_decoded'}
    with zipfile.ZipFile(full) as archive:
        assert set(archive.namelist())=={key+'.npy' for key in shapes}
        for key,expected in shapes.items():
            shape,_,dtype=header(archive,key)
            assert shape==expected,(directory,key,shape,expected)
            semantic=metadata['dtypes'].get(key,'float32')
            expected_dtype=np.dtype('uint16' if semantic=='bfloat16' else semantic)
            assert dtype==expected_dtype,(directory,key,dtype,expected_dtype)
    actual_bytes=sum((directory/name).stat().st_size for name in ('activations.npz','pooled.npz','residual_stats.npz'))
    assert actual_bytes==metadata['bytes']==completion['bytes']
    return dict(bytes=actual_bytes,pooling=metadata.get('pooling_impl','cpu_v1'),
                raw_max_abs=metadata['parity']['raw']['max_abs'],
                decoded_max_abs=metadata['parity']['decoded']['max_abs'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--partial',action='store_true');args=parser.parse_args();root=args.root.resolve()
    cache_path=root/'feature_audit_cache.json'
    saved=json.loads(cache_path.read_text()) if cache_path.exists() else {}
    cache=saved.get('entries',{}) if saved.get('version')==VERSION else {}
    checked=[];pending=[];reused=0;started=time.perf_counter();expected=0
    for suite in SUITES:
        for task in range(10):
            for init in range(3):
                episode=episode_path(root,suite,task,init)
                identity=json.loads((episode/'identity.json').read_text())
                samples=json.loads((episode/'selection.json').read_text())['samples']
                for sample in samples:
                    key=f"{suite}/{task}/{init}/{sample['request']}";expected+=1
                    directory=root/'features'/suite/f'task{task:02d}'/f'init{init:02d}'/f"request_{sample['request']:04d}"
                    if not (directory/'complete.json').exists() or not (directory/'sample.json').exists():
                        pending.append(key);continue
                    stamp=signature(directory)
                    context=verify_context(directory,sample,identity)
                    previous=cache.get(key)
                    # Legacy entries did not include the parity file signature;
                    # recheck its full context above without re-reading large
                    # immutable archive headers when all prior stamps match.
                    if previous and all(stamp.get(name)==value for name,value in previous['signature'].items()):
                        result=previous['result'];reused+=1
                        previous['signature']=stamp
                    else:
                        result=verify_one(directory,sample,identity,context)
                        cache[key]=dict(signature=stamp,result=result)
                    checked.append(result)
                    if len(checked)%100==0:
                        atomic_json(cache_path,dict(version=VERSION,entries=cache))
                        atomic_json(root/'feature_audit_progress.json',dict(checked=len(checked),pending_seen=len(pending),reused=reused,elapsed_s=time.perf_counter()-started))
    atomic_json(cache_path,dict(version=VERSION,entries=cache))
    receipt=dict(passed=not pending,checked_subset_passed=True,status='partial' if pending else 'complete',version=VERSION,
        expected=expected,verified=len(checked),pending=len(pending),reused=reused,
        bytes_verified_structure=sum(r['bytes'] for r in checked),
        pooling_counts=dict(Counter(r['pooling'] for r in checked)),
        max_raw_abs=max((r['raw_max_abs'] for r in checked),default=0.),
        max_decoded_abs=max((r['decoded_max_abs'] for r in checked),default=0.),
        scope='Every NPY header, full component names/shapes/dtypes, mask layout, source/noise origin, final raw/decoded action payloads, service/case identity and byte counts',
        excluded_scope='Large hidden tensor payloads are not re-read/rehashed; inferred phase accuracy and final analysis checked separately',
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),elapsed_s=time.perf_counter()-started)
    destination=root/('feature_audit_partial.json' if args.partial else 'feature_audit.json')
    atomic_json(destination,receipt);print(json.dumps(receipt),flush=True)
    atomic_json(root/'feature_audit_progress.json',dict(status='partial_checked' if pending else 'complete',checked=len(checked),pending=len(pending),reused=reused,elapsed_s=time.perf_counter()-started))
    if pending and not args.partial:raise RuntimeError(f'{len(pending)} selected requests not yet complete')


if __name__=='__main__':main()
