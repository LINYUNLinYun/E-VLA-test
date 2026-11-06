"""Check whether old replay reconstructs the actual initial Full encoder input."""
import json
import torch
from .common import ROOT,OUT
from preparations.idea1_v3.io import atomic

def main():
    from preparations.idea1_v3.safe_qk import install
    OUT.mkdir(parents=True,exist_ok=True);install(str(OUT/'input_probe_qk_receipt.json'))
    from preparations.idea1_v3.policy import TritonBackend
    from preparations.kv_reuse_probe.probe import load_observation
    from preparations.idea1_v3_4_E2_online_time_weights.selector import current_item
    b=TritonBackend();pairs=json.loads((ROOT/'results/idea1_v3_4_E2_E4_single/selection.json').read_text())
    p=next(x for x in pairs if x['e2']);obs,prompt=load_observation(p['current']);seed=p['current']['seed']
    c,_=b.prepare(obs,prompt,(),encode=False);initial=current_item(c,seed)['x'].cpu()
    c,_=b.prepare(obs,prompt,());after=current_item(c,seed)['x'].cpu()
    c['runtime'].buffers['encoder_x'][512:].copy_(c['runtime'].weights['language_embeds'])
    restored=current_item(c,seed)['x'].cpu()
    r=dict(old_replay_full_input_equal=bool(torch.equal(initial,after)),old_vision_max_abs=float((initial[:512].float()-after[:512].float()).abs().max()),old_language_max_abs=float((initial[512:].float()-after[512:].float()).abs().max()),restored_input_equal=bool(torch.equal(initial,restored)),restored_max_abs=float((initial.float()-restored.float()).abs().max()),tokens=len(initial))
    atomic(OUT/'input_reconstruction_probe.json',r);print(json.dumps(r),flush=True)
if __name__=='__main__':main()
