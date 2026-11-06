"""Strict fixed NEW-history reference; retain OLD-history diagnostic.

Only used on recovered non-task0 initial shard. The probe checks its real
first prompt, then reconstructs task0; this is the frozen engineering NEW
history, not the original service's task0-first probe history.
"""
import hashlib
from pathlib import Path
import numpy as np
from .cross_restart import check as old_check


def check(backend):
    captured={};sample=backend.sample
    def capture(*args,**kwargs):
        result=sample(*args,**kwargs);captured['actions']=result[0].copy();captured['raw']=result[1].float().cpu().numpy().copy();return result
    backend.sample=capture
    try:legacy=old_check(backend)
    finally:backend.sample=sample
    root=Path(__file__).resolve().parents[2]/'results/idea1_v3_3'
    ref=root/'engineering_cost_rebaseline/pi0_parent_cost_gpu2/episodes/libero_spatial/task00/init06'
    with np.load(ref/'arrays.npz') as arrays:
        expected_raw=arrays['raw_0'];expected_actions=arrays['actions_0']
    raw_exact=np.array_equal(captured['raw'],expected_raw);actions_exact=np.array_equal(captured['actions'],expected_actions)
    return dict(passed=bool(raw_exact and actions_exact),history_revision='cross-restart-fixed-new-h1',reference=str(ref),
        fixed_history='Actual initial shard prompt Task1 probe+metadata, then Task0 reconstruction; frozen engineering NEW cohort, no reference selected from observed output.',
        strict_bitwise=True,raw_bitwise_equal=bool(raw_exact),decoded_bitwise_equal=bool(actions_exact),
        raw_max_abs=float(np.max(np.abs(captured['raw']-expected_raw))),decoded_max_abs=float(np.max(np.abs(captured['actions']-expected_actions))),
        reference_arrays_sha256=hashlib.sha256((ref/'arrays.npz').read_bytes()).hexdigest(),input_sha256=hashlib.sha256((ref/'input0.npz').read_bytes()).hexdigest(),
        output_sha256={name:hashlib.sha256(value.tobytes()).hexdigest() for name,value in captured.items()},
        legacy_original_parent_diagnostic=dict(legacy,nonfatal_reason='Original Task0-first probe cohort differs from this non-Task0-first shard history; known fixed numerical group, not a tolerance relaxation.'))
