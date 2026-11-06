"""Fixed-noise action-path checks, not action-error learning or rollout quality."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .native_probe import NativeProbe
from .runtime_router import SemanticSkipRuntime,select_mask


def prepare(probe,path):
    from preparations.idea2.experiments import native_raw
    with np.load(path,allow_pickle=False) as data:
        instruction=str(data['instruction'].item())
        obs={k:data[k].copy() for k in data.files if k not in ('instruction','seed')}
    return probe.backend.prepare(native_raw(dict(obs=obs,instruction=instruction))),instruction


def difference(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    return dict(exact_equal=bool(np.array_equal(a,b)),max_abs=float(np.abs(a-b).max()))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--candidate',required=True)
    p.add_argument('--collection',required=True)
    p.add_argument('--preflight',required=True)
    p.add_argument('--assets',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=32)
    args=p.parse_args()
    candidate=json.loads(Path(args.candidate).read_text());collection=Path(args.collection).resolve()
    cfg=json.loads((collection/'collection_config.json').read_text());output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    preflight=json.loads((Path(args.preflight)/'native_preflight.json').read_text())
    golden=torch.load(Path(args.preflight)/'native_preflight_tensors.pt',map_location='cpu',weights_only=True)
    print(json.dumps(dict(event='loading_original_policy_for_runtime_check',device=args.device)),flush=True)
    probe=NativeProbe(cfg['checkpoint'],args.device,args.threads)
    prepared,instruction=prepare(probe,preflight['input']);noise=probe.backend.noise(1137)
    start=time.perf_counter()
    reference_l5=[]
    handle=probe.lm.layers[5].register_forward_hook(
        lambda module,inputs,output: reference_l5.append(output[0][0,golden['selected']].detach().cpu().clone()))
    try:
        reference=probe.backend.original(prepared,noise)
    finally:
        handle.remove()
    reference_decoded=probe.backend.output(prepared,reference)
    print(json.dumps(dict(event='unmodified_action_path_complete',seconds=time.perf_counter()-start)),flush=True)
    runtime=SemanticSkipRuntime(probe,candidate,Path(args.assets),capture_debug=True)
    raw,decoded,trace=runtime.infer_prepared(prepared,instruction,noise,'noop','sae_only')
    noop=dict(raw=difference(reference.detach().float().cpu().numpy(),raw.detach().float().cpu().numpy()),
              decoded=difference(reference_decoded,decoded),trace=trace,
              selected_positions_equal=runtime.last_debug['layout']['selected_positions']==golden['selected'],
              L5_samples_equal=bool(len(reference_l5)==1 and torch.equal(runtime.last_debug['samples'],reference_l5[0])),
              saved_CPU_L5_samples_equal=bool(torch.equal(runtime.last_debug['samples'],golden['L5'][0])))
    print(json.dumps(dict(event='readout_only_noop_checked',result=noop)),flush=True)
    # The positive example is preselected by the frozen calibration readout,
    # never by an action error or a rollout outcome.
    selection=Path(candidate['selection'])
    observations=json.loads((selection/'score_observations.json').read_text());scores=np.load(selection/'semantic_scores.npz')
    positive=None
    for i,row in enumerate(observations):
        if row['split']!='calibration':continue
        confidence={name:float(scores[f"L5_{name.replace('/','_')}_{candidate['token_pool']}"][i]) for name in candidate['early_concepts']}
        mask,active,reason=select_mask(confidence,candidate)
        if mask:positive=(row,mask,confidence);break
    if positive is None:raise ValueError('No preselected positive calibration observation')
    row,mask,expected_confidence=positive;prepared,instruction=prepare(probe,row['observation_path']);noise=probe.backend.noise(1138)
    with probe.skip_ffn(mask):direct=probe.backend.original(prepared,noise)
    direct_decoded=probe.backend.output(prepared,direct)
    runtime.reset_episode('positive',start_calls=2)
    raw,decoded,trace=runtime.infer_prepared(prepared,instruction,noise,'positive','sae_route')
    route=dict(input=row['observation_path'],expected_mask=list(mask),trace=trace,
               raw=difference(direct.detach().float().cpu().numpy(),raw.detach().float().cpu().numpy()),
               decoded=difference(direct_decoded,decoded),expected_offline_confidence=expected_confidence)
    runtime.detach()
    passed=(noop['raw']['exact_equal'] and noop['decoded']['exact_equal'] and noop['selected_positions_equal']
            and noop['L5_samples_equal'] and route['raw']['exact_equal'] and route['decoded']['exact_equal'] and trace['mask']==list(mask))
    report=dict(scope='One fixed-noise no-op and one intended-mask action-path equality check; no SR or algorithmic speedup claim',
                checkpoint=cfg['checkpoint'],device=str(probe.backend.device),threads=args.threads,noise_seeds=[1137,1138],
                candidate_sha256=hashlib.sha256(Path(args.candidate).read_bytes()).hexdigest(),
                runtime_sources={str(Path(__file__).with_name(name).resolve()):hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                 for name in ('check_runtime.py','runtime_router.py','native_probe.py')},
                comparison='Readout and reference L5 samples are compared on the same device; saved CPU samples are diagnostic across devices',
                readout_only_noop=noop,intended_mask_path=route,passed=bool(passed))
    (output/'runtime_check.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(event='runtime_check_complete',passed=passed,mask=trace['mask'])),flush=True)
    if not passed:raise ValueError('Runtime sampling, no-op or intended mask path differs; investigate without widening tolerance')


if __name__=='__main__':
    main()
